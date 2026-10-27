# -*- coding: utf-8 -*-
"""見逃した正解は「見えていない」のか「半分に届かない」のか（`41`）。

なぜ要るか
----------
箇所評価は**正解のかたまりの 50% を覆って初めて検出**とみなす（`COV_GT`）。
だから「見逃し」には性質の違う二種類が混ざっている。

    覆い 0%     ── モデルはそこを危険だと思っていない（盲）
    覆い 45%    ── 危険だとは思っているが、範囲が足りない（ずれ）

`40` までの分析はこの二つを区別していない。区別しないと打ち手が決まらない。
**盲なら入力を足す話**になり、**ずれなら1段目のしきい値・損失の話**になる。
2段目は刈るだけなので再現率を作れない（`24.6`）。

測るもの（正解のかたまり1件ごと）
--------------------------------
    cov        予測のかたまり全部で何割覆われたか（これが `found` の元）
    cov_best   **いちばん大きく重なる予測ひとつ**で何割か
               cov は大きいのに cov_best が小さいなら、**分裂**している
    n_hit      重なっている予測のかたまりの数
    frac_pos   かたまりの中で**ロジットが正**の画素の割合
               cov より大きいなら、`min_size` と連結で落ちている
    lg_max     かたまりの中のロジットの最大値（0 が確率 0.5）
    lg_mean    同じく平均
    pdist      最寄りの予測画素までの距離 [m]。覆いが 0 のときだけ意味がある

出力は `cache/analysis/coverage/<条件>__<地域>__gtcov.csv`。
`run_building_miss.py --save` が残した gt.csv と (tile, gtblob) で突き合わせ、
**area と found が全件一致することを確かめてから**建物別の内訳を出す。

    python analysis/run_miss_coverage.py --region hiroshima --save
    python analysis/run_miss_coverage.py --region shimane --save
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_overlap_prune import infer, score                  # noqa: E402
from dc5lib import paths                                             # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import label_instances, overlap_matrix     # noqa: E402
from dc5lib.registry import get                                      # noqa: E402

COV_GT = 0.5        # `run_building_miss.py` と同じ。ここを変えると比較できない
MIN_SIZE = 10
# 1画素の大きさ [m]。**東西と南北で違う**（`38`）。
PX_M = (3.60, 4.50)
# 箇所F のゆらぎ幅（`docs/ゆらぎ幅.md`）。**島根は広島の2.8倍広い。**
BAND = {"hiroshima": 0.0071, "shimane": 0.0198}


def gt_coverage(GT, PB, LG, verbose=True):
    """正解のかたまりごとに、どれだけ覆われたかを測る。"""
    my, mx = PX_M
    rows = []
    for k in range(len(GT)):
        gl, gs = label_instances(GT[k], 8, MIN_SIZE)
        if not len(gs):
            continue
        lab, sizes = label_instances(PB[k], 8, MIN_SIZE)
        O = (overlap_matrix(gl, len(gs), lab, len(sizes))
             if len(sizes) else np.zeros((len(gs), 0), np.int64))
        # 最寄りの予測画素まで。**min_size で落ちたかたまりは含めない**
        # （`found` の判定に使われないものを「近い予測」と呼ばないため）
        pm = lab > 0
        dist = (ndimage.distance_transform_edt(~pm, sampling=(my, mx))
                if pm.any() else None)
        lg = LG[k]
        pos = lg > 0.0
        for i in range(len(gs)):
            sel = gl == (i + 1)
            o = O[i]
            cov = float(o.sum()) / gs[i]
            rows.append(dict(
                tile=k, gtblob=i + 1, area=int(gs[i]),
                found=bool(cov >= COV_GT),
                cov=cov,
                cov_best=float(o.max()) / gs[i] if len(o) else 0.0,
                n_hit=int((o > 0).sum()),
                frac_pos=float(pos[sel].mean()),
                lg_max=float(lg[sel].max()),
                lg_mean=float(lg[sel].mean()),
                pdist=float(dist[sel].min()) if dist is not None
                else float("nan")))
        if verbose and k % 500 == 0:
            print(f"    {k}/{len(GT)}", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    return pd.DataFrame(rows)


def hist(d, col, edges, labels):
    """edges で区切って (札, 件数, 中身) を返す。"""
    k = np.digitize(d[col].to_numpy(), edges, right=False) - 1
    k = np.clip(k, 0, len(labels) - 1)
    return [(lb, int((k == i).sum()), d[k == i]) for i, lb in enumerate(labels)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    cond = get(a.condition)
    device = torch.device(a.device) if a.device else pick_device()
    band = BAND[a.region]
    print(f"条件 {cond.name} / 地域 {a.region} / device {device}")

    GT, PB, LG = infer(cond, a.region, device)
    R0, P0, F0, n0 = score(GT, PB)
    print(f"\n刈り取り前  箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}"
          f" / 予測数 {n0:,}")

    print("\n正解のかたまりの被覆率を測ります …", flush=True)
    d = gt_coverage(GT, PB, LG)
    miss = d[~d.found]
    print(f"正解のかたまり {len(d):,}  見逃し {len(miss):,}"
          f"  箇所R（表から） {d.found.mean():.4f}")

    # --- 既存の建物表と突き合わせる -------------------------------
    bpath = (paths.cache_dir("analysis") / "building"
             / f"{cond.name}__{a.region}__gt.csv")
    bld = None
    if bpath.exists():
        g = pd.read_csv(bpath)
        m = d.merge(g[["tile", "gtblob", "tile_no", "area", "found",
                       "bld_in", "bld_50", "bld_dist"]],
                    on=["tile", "gtblob"], suffixes=("", "_b"))
        if len(m) != len(d):
            sys.exit(f"突き合わせで件数が変わりました {len(d)} -> {len(m)}")
        if not (m.area == m.area_b).all():
            sys.exit("面積が一致しません。かたまりの付番が違います")
        if not (m.found == m.found_b).all():
            sys.exit("検出の判定が一致しません")
        print(f"  {bpath.name} と全 {len(m):,} 件で面積・判定が一致")
        bld = m
    else:
        print(f"  （{bpath.name} が無いので建物別の内訳は出しません）")

    EDG = [0.0, 1e-9, 0.05, 0.15, 0.30, 0.45, 0.50]
    LAB = ["0%（まったく重ならない）", "0超〜5%", "5〜15%", "15〜30%",
           "30〜45%", "45〜50%（あと一歩）"]

    print("\n" + "=" * 74)
    print("【1】見逃したかたまりは何割覆われていたか")
    print("=" * 74)
    print(f"{'覆い':<26}{'件数':>8}{'割合':>8}{'中央面積':>10}"
          f"{'ロジット最大':>14}{'正の画素%':>11}")
    for lb, n, s in hist(miss, "cov", EDG, LAB):
        if not n:
            continue
        print(f"{lb:<26}{n:>8,}{100 * n / len(miss):>7.1f}%"
              f"{s.area.median():>10.0f}{s.lg_max.mean():>14.3f}"
              f"{100 * s.frac_pos.mean():>10.1f}%")
    print(f"{'（見逃し全体）':<26}{len(miss):>8,}{100.0:>7.1f}%"
          f"{miss.area.median():>10.0f}{miss.lg_max.mean():>14.3f}"
          f"{100 * miss.frac_pos.mean():>10.1f}%")
    f = d[d.found]
    print(f"{'（当たった正解）':<26}{len(f):>8,}{'':>8}"
          f"{f.area.median():>10.0f}{f.lg_max.mean():>14.3f}"
          f"{100 * f.frac_pos.mean():>10.1f}%")
    print("\n  **0% に山があれば「見えていない」。"
          "30〜50% に山があれば「範囲が足りない」。**")

    print("\n" + "=" * 74)
    print("【2】覆いが 0 の見逃しは、どれくらい離れたところを撃っているか")
    print("=" * 74)
    z = miss[miss['cov'] <= 0]
    print(f"  覆い 0 の見逃し {len(z):,} 件 / 見逃し {len(miss):,} 件"
          f"（{100 * len(z) / max(len(miss), 1):.1f}%）")
    none = z.pdist.isna()
    print(f"  そのうち**同じタイルに予測が1つも無い** {int(none.sum()):,} 件"
          f"（{100 * none.mean():.1f}%）")
    if (~none).any():
        q = z.pdist[~none].quantile([0.25, 0.5, 0.75]).to_numpy()
        print(f"  予測がある場合の最寄りまでの距離  "
              f"25% {q[0]:.0f}m / 中央 {q[1]:.0f}m / 75% {q[2]:.0f}m")

    print("\n" + "=" * 74)
    print("【3】分裂しているか（cov は足し算、cov_best はひとつぶん）")
    print("=" * 74)
    print(f"{'':<26}{'cov 平均':>11}{'cov_best 平均':>15}"
          f"{'重なる予測の数':>16}")
    for nm, s in (("見逃し", miss), ("当たった正解", f)):
        print(f"{nm:<26}{s['cov'].mean():>11.4f}{s.cov_best.mean():>15.4f}"
              f"{s.n_hit.mean():>16.2f}")
    split = miss[(miss['cov'] >= 0.30) & (miss.cov_best < 0.30)]
    print(f"\n  cov 30%以上なのに最大の予測ひとつでは 30%未満 "
          f"= {len(split):,} 件")
    print("  **ここが多いなら、つなげるだけで検出になる。**")

    print("\n" + "=" * 74)
    print("【4】`min_size` と連結で落ちている分（ロジットは正なのに）")
    print("=" * 74)
    print(f"{'':<26}{'正の画素の割合':>16}{'覆い':>10}{'差':>10}")
    for nm, s in (("見逃し", miss), ("当たった正解", f)):
        print(f"{nm:<26}{s.frac_pos.mean():>16.4f}{s['cov'].mean():>10.4f}"
              f"{s.frac_pos.mean() - s['cov'].mean():>+10.4f}")
    nearly = miss[(miss.frac_pos >= COV_GT) & (miss['cov'] < COV_GT)]
    print(f"\n  **ロジットが正の画素だけなら 50% を超えるのに見逃し "
          f"= {len(nearly):,} 件**")
    print(f"  （見逃しの {100 * len(nearly) / max(len(miss), 1):.1f}%、"
          f"箇所R にして {len(nearly) / len(d):+.4f}"
          f" = {len(nearly) / len(d) / band:.1f}帯）")

    if bld is not None:
        print("\n" + "=" * 74)
        print("【5】建物の位置別の内訳（`41` の段差はどちらの種類か）")
        print("=" * 74)
        mb = bld[~bld.found]
        grp = (("かたまりに建物あり", mb.bld_in > 0),
               ("建物なし・50m内に家", (mb.bld_in <= 0) & (mb.bld_50 > 0)),
               ("建物なし・50m外", (mb.bld_in <= 0) & (mb.bld_50 <= 0)))
        print(f"{'見逃しの種類':<24}{'件数':>8}{'覆い0%':>10}"
              f"{'30%以上':>10}{'平均の覆い':>12}{'ロジット最大':>14}")
        for nm, m in grp:
            s = mb[m]
            if not len(s):
                continue
            print(f"{nm:<24}{len(s):>8,}"
                  f"{100 * (s['cov'] <= 0).mean():>9.1f}%"
                  f"{100 * (s['cov'] >= 0.30).mean():>9.1f}%"
                  f"{s['cov'].mean():>12.4f}{s.lg_max.mean():>14.3f}")

    print("\n" + "=" * 74)
    print("【6】しきい値を下げたら（A で選び B で測る。`13.1` の作法）")
    print("=" * 74)
    print("  2段目は刈るだけなので再現率を作れない（`24.6`）。"
          "作れるのはここだけ。")
    A = np.arange(0, len(GT), 2)
    Bi = np.arange(1, len(GT), 2)
    _, _, FB0, _ = score(GT, PB, Bi)
    print(f"\n{'しきい値（確率）':>18}{'A 箇所F':>10}{'B 箇所R':>10}"
          f"{'B 箇所P':>10}{'B 箇所F':>10}{'正味':>9}{'帯':>7}")
    best, bestFA = None, -1.0
    for p in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10):
        t = float(np.log(p / (1 - p)))
        Q = LG > t
        FA = score(GT, Q, A)[2]
        RB, PBr, FB, _ = score(GT, Q, Bi)
        if FA > bestFA:
            best, bestFA = p, FA
        print(f"{p:>18.2f}{FA:>10.4f}{RB:>10.4f}{PBr:>10.4f}{FB:>10.4f}"
              f"{FB - FB0:>+9.4f}{(FB - FB0) / band:>7.1f}")
    print(f"\n  **A が選ぶのは {best:.2f}。** "
          f"0.50 のままなら、しきい値は既に最適である。")

    if a.save:
        od = paths.cache_dir("analysis") / "coverage"
        od.mkdir(parents=True, exist_ok=True)
        out = bld if bld is not None else d
        p = od / f"{cond.name}__{a.region}__gtcov.csv"
        out.to_csv(p, index=False, encoding="utf-8")
        print(f"\n保存: {p}")


if __name__ == "__main__":
    main()
