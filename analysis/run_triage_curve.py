# -*- coding: utf-8 -*-
"""**上位N件だけ見たときの再現率**。運用の曲線を描く（SCIS&ISIS 2026 PID-109）。

なぜ要るか
----------
発表資料の `Backup 7` にこう書いている ──
「適合率は目標ではなく**予算の制約**。適合率 p なら 1/p 倍の面積を見る」。
ところが**順番の最適化は何もしていない。**
いま「箇所F 0.7908」と1点で報告しているものは、
**「上位 N 件だけ現地を見れば、見逃しの何割を拾えるか」という曲線**にできる。

`13` の刈り取りは既にロジットの大小でかたまりを捨てている。
つまり**順位はすでに手元にある。** しきい値を1本選んで捨てる代わりに、
**順位をそのまま曲線にする**。学習は要らない。

何を出すか
----------
  1. 予測のかたまりをロジット最大値の降順に並べる
  2. 上位 n 件を「現地を見る」としたとき
       ・箇所Recall（被覆 cov_gt>=0.5 に達した正解の割合）
       ・箇所Precision（見た n 件のうち的中の割合）
       ・見る面積 [km2]（かたまりの画素数 x 16.2 m2/px）
  3. 同じ推論から**分割の上限**（`19.3`）を刈り取りの前後で出し直す
     ── 2分岐デコーダ（PID-106）をやる価値があるかの判定

**【2】の刈り取りのしきい値はテスト集合の分位で決めている。**
実装書 `7.3` の後処理は**絶対値**でしきい値を持つ（集合が変わるたびに動くため）。
ここは「分割で上限がどこまで動くか」の分析で、記録もしないので分位でよい。
**記録する数字を出すときは絶対値を使うこと。**

**判定は要らない（1 は新しい図）。2 は帯と比べる。**

出力
----
  cache/analysis/triage/<条件>__<地域>__blobs.csv   かたまりごと（傾斜量つき）
  cache/analysis/triage/<条件>__<地域>__pairs.csv   (正解, 予測) の重なり
  cache/analysis/triage/<条件>__<地域>__curve.csv   曲線そのもの

`blobs.csv` は `run_blob_cluster.py`（クラスタリングの検証）が読む。
**推論をやり直さないため**に保存する。

    python analysis/run_triage_curve.py
    python analysis/run_triage_curve.py --condition FinalFusion_SAM_APM_OverlapTrain
"""

from __future__ import annotations

import argparse
import os
import re
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

from analysis.data import load_pkl                                   # noqa: E402
from analysis.run_building_miss import pixel_metres                  # noqa: E402
from analysis.run_overlap_eval import REGION_DIR, target_tiles       # noqa: E402
from analysis.run_overlap_prune import infer, score                  # noqa: E402
from analysis.run_split_bound import f_value                         # noqa: E402
from dc5lib import paths                                             # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import (BAND, PRIMARY,                     # noqa: E402
                                  label_instances, overlap_matrix)
from dc5lib.registry import get                                      # noqa: E402

STRUCT4 = ndimage.generate_binary_structure(2, 1)
# **被覆のしきい値は `PRIMARY` の文字列から取る。** 直書きすると
# `dc5lib/instance_eval.py` を直したときに静かに食い違う（実装書 §5）。
_m = re.search(r"cov_gt=([\d.]+).*cov_pred=([\d.]+)", PRIMARY)
assert _m, f"PRIMARY の形が変わりました: {PRIMARY}"
COV_GT, COV_PRED = float(_m.group(1)), float(_m.group(2))
# 箇所F のゆらぎ幅は `dc5lib/instance_eval.py` の `BAND` が原本（実装書 `5`）
# 自前の被覆計算と `instance_eval` の差をどこまで許すか。
# **正解のかたまりの数え方で1〜2件ずれる**（8,977 件のうち 2 件で 0.0002）。
# 定義が食い違えばこれよりずっと大きく動くので、ここを超えたら止める。
TOL = 0.002


def tables(GT, PB, LG, slope, verbose=True):
    """かたまりの表と (正解, 予測) の重なりの表。

    **ラベリングは評価と同じ（8近傍・10px）。** ここを変えると数字が比べられない。
    """
    blobs, pairs, gts = [], [], []
    for k in range(len(GT)):
        pl, ps = label_instances(PB[k], 8, 10)
        gl, gs = label_instances(GT[k], 8, 10)
        O = overlap_matrix(gl, len(gs), pl, len(ps))
        for i in range(len(gs)):
            gts.append(dict(tile=k, gtblob=i + 1, gt_area=int(gs[i])))
        if len(ps) == 0:
            continue
        cov_p = (O.sum(axis=0) / ps) if len(gs) else np.zeros(len(ps))
        lg, sl = LG[k], slope[k]
        for j in range(len(ps)):
            sel = pl == (j + 1)
            inner = ndimage.binary_erosion(sel, STRUCT4)
            v, s = lg[sel], sl[sel]
            blobs.append(dict(
                tile=k, blob=j + 1, area=int(ps[j]),
                tp=bool(cov_p[j] >= COV_PRED), cov_pred=float(cov_p[j]),
                logit_max=float(v.max()), logit_mean=float(v.mean()),
                compactness=float(sel.sum() - inner.sum()) / max(int(ps[j]), 1),
                slope_mean=float(s.mean()), slope_p90=float(np.percentile(s, 90)),
                slope_max=float(s.max()), slope_std=float(s.std())))
        for i, j in zip(*np.nonzero(O)):
            pairs.append(dict(tile=k, gtblob=int(i) + 1, blob=int(j) + 1,
                              px=int(O[i, j])))
        if verbose and k % 500 == 0:
            print(f"    {k}/{len(GT)}", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    return (pd.DataFrame(blobs), pd.DataFrame(pairs), pd.DataFrame(gts))


def curve(blobs, pairs, gts, m2, order="logit_max", steps=200):
    """順位を上から辿って、再現率・適合率・見る面積を積む。

    **正解は「覆われた画素の合計が面積の 50% に達した時点」で検出とみなす。**
    1件の予測で足りず2件目で達することがあるので、**足し算で辿る。**
    """
    b = blobs.sort_values(order, ascending=False).reset_index(drop=True)
    gkey = {(int(t), int(g)): i for i, (t, g)
            in enumerate(zip(gts.tile, gts.gtblob))}
    garea = gts.gt_area.to_numpy(float)
    covered = np.zeros(len(gts))
    need = garea * COV_GT
    found = np.zeros(len(gts), bool)

    # 予測のかたまり -> そのかたまりが覆う (正解の添字, 画素数)
    hit = {}
    for t, g, j, px in zip(pairs.tile, pairs.gtblob, pairs.blob, pairs.px):
        hit.setdefault((int(t), int(j)), []).append((gkey[(int(t), int(g))], px))

    n_gt = len(gts)
    rows = []
    n_found = 0
    cum_area = 0
    cum_tp = 0
    mark = set(np.unique(np.linspace(1, len(b), steps).astype(int)).tolist())
    for n in range(1, len(b) + 1):
        r = b.iloc[n - 1]
        cum_area += int(r.area)
        cum_tp += int(bool(r.tp))
        for gi, px in hit.get((int(r.tile), int(r.blob)), ()):
            covered[gi] += px
            if not found[gi] and covered[gi] >= need[gi]:
                found[gi] = True
                n_found += 1
        if n in mark:
            rows.append(dict(n=n, frac=n / len(b),
                             recall=n_found / n_gt, precision=cum_tp / n,
                             km2=cum_area * m2 / 1e6,
                             logit=float(r[order])))
    return pd.DataFrame(rows)


def split_bound(GT, PB):
    """`19.3` と同じ分割上限。かたまりを完璧に分けられたときの箇所F。

    delta = 飲み込まれた正解 - 融合予測。**n_pred と tp_pred が同じだけ増える。**
    箇所Recall は分割で厳密に不変（cov_g は予測の和集合にしか依存しない）。
    """
    n_gt = n_pred = tp_gt = tp_pred = mp = gim = 0
    for k in range(len(GT)):
        pl, ps = label_instances(PB[k], 8, 10)
        gl, gs = label_instances(GT[k], 8, 10)
        n_gt += len(gs)
        n_pred += len(ps)
        if len(gs) == 0 or len(ps) == 0:
            continue
        O = overlap_matrix(gl, len(gs), pl, len(ps))
        tp_gt += int((O.sum(axis=1) / gs >= COV_GT).sum())
        tp_pred += int((O.sum(axis=0) / ps >= COV_PRED).sum())
        touch = O > 0
        merged = touch.sum(axis=0) >= 2
        mp += int(merged.sum())
        gim += int(touch[:, merged].any(axis=1).sum())
    R = tp_gt / n_gt if n_gt else 0.0
    P = tp_pred / n_pred if n_pred else 0.0
    d = gim - mp
    Ps = (tp_pred + d) / (n_pred + d) if (n_pred + d) else 0.0
    return dict(n_gt=n_gt, n_pred=n_pred, merged_pred=mp, gt_in_merge=gim,
                swallow=gim / n_gt if n_gt else 0.0,
                R=R, P=P, F=f_value(R, P),
                F_split=f_value(R, Ps), F_prune=f_value(R, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane"])
    ap.add_argument("--prune-q", type=float, default=0.30,
                    help="分割上限を出し直すときに捨てる割合（`39.5` は 30%%）")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cond = get(a.condition)
    device = torch.device(a.device) if a.device else pick_device()
    print(f"条件 {cond.name} / 地域 {a.region} / device {device}")
    print("オーバーラップ推論（9窓・cos窓）。二値化は確率側、順位はロジット")

    GT, PB, LG = infer(cond, a.region, device)

    # 傾斜量（ds.DEM は傾斜量図そのもの。analysis/features.py と同じ扱い）
    dirname = REGION_DIR[a.region]
    # `pkl_for` は絶対パスを返す。`dataset_path` に通すと pathlib が
    # 左側を捨てて**たまたま**同じ結果になるだけなので、通さない（レビュー 4）。
    pkl = cond.pkl_for(dirname)
    print(f"  傾斜量を読み直す: {os.path.basename(pkl)} ...", flush=True)
    ds = load_pkl(pkl)
    tiles = target_tiles(ds, a.region, float(getattr(cond, "bg_ratio", 0.0) or 0.0))
    if len(tiles) != len(GT):
        sys.exit(f"タイル数が合いません（推論 {len(GT)} / 読み直し {len(tiles)}）")
    slope = np.stack([np.asarray(ds.DEM[k], dtype=np.float32) for k in tiles])
    my, mx = pixel_metres(ds.GeoInfo[tiles[0]])
    m2 = my * mx
    tile_no = [int(ds.No[k]) for k in tiles]
    del ds
    print(f"  1画素 {my:.2f} x {mx:.2f} m = {m2:.1f} m2")

    blobs, pairs, gts = tables(GT, PB, LG, slope)
    blobs["tile_no"] = [tile_no[t] for t in blobs.tile]
    out = paths.cache_dir("analysis") / "triage"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{cond.name}__{a.region}"
    for name, d in (("blobs", blobs), ("pairs", pairs), ("gt", gts)):
        d.to_csv(out / f"{stem}__{name}.csv", index=False, encoding="utf-8-sig")

    n_tp = int(blobs.tp.sum())
    print(f"\n予測のかたまり {len(blobs):,}（的中 {n_tp:,} / "
          f"外れ {len(blobs) - n_tp:,}）/ 正解のかたまり {len(gts):,}")

    # ---------------------------------------------------------------- 1. 曲線
    print("\n" + "=" * 74)
    print("【1】上位 N 件だけ現地を見たときの再現率（ロジット最大値の降順）")
    print("=" * 74)
    cv = curve(blobs, pairs, gts, m2)
    cv.to_csv(out / f"{stem}__curve.csv", index=False, encoding="utf-8-sig")
    full = cv.iloc[-1]
    print(f"{'見る割合':>10}{'件数':>9}{'見る面積':>11}{'箇所R':>9}"
          f"{'箇所P':>9}{'箇所F':>9}{'ロジット':>10}")
    for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        r = cv.iloc[(cv.frac - q).abs().argmin()]
        print(f"{100 * r.frac:>9.0f}%{int(r.n):>9,}{r.km2:>10.1f}km2"
              f"{r.recall:>9.4f}{r.precision:>9.4f}"
              f"{f_value(r.recall, r.precision):>9.4f}{r.logit:>10.2f}")
    # ---- **自前の被覆計算を `PRIMARY` と突き合わせる**（実装書 §5）。
    # 曲線は「上位N件を足しながら被覆を見る」形なので instance_eval には無い。
    # だから全件のときだけ、同じ数字が出るかを機械で確かめる。
    # **表はもう CSV に書いてあるので、ここで止めても推論は無駄にならない。**
    pR, pP, pF, pn = score(GT, PB)
    dR, dP = abs(full.recall - pR), abs(full.precision - pP)
    print(f"\n  突き合わせ（instance_eval の {PRIMARY}）")
    print(f"    箇所R 自前 {full.recall:.4f} / PRIMARY {pR:.4f}（差 {dR:.4f}）")
    print(f"    箇所P 自前 {full.precision:.4f} / PRIMARY {pP:.4f}（差 {dP:.4f}）")
    print(f"    予測のかたまり 自前 {len(blobs):,} / PRIMARY {pn:,}")
    if max(dR, dP) > TOL:
        sys.exit(
            f"**自前の被覆計算が PRIMARY と {max(dR, dP):.4f} 違います**"
            f"（許容 {TOL}）。被覆の定義か連結性か最小サイズが食い違っています。"
            " 実装書 §5 を読んでから直してください。")
    print(f"    差は許容 {TOL} の中（正解のかたまりの数え方で1〜2件ずれる）")
    print(f"\n  全件見たときの箇所R {full.recall:.4f}"
          f"（記録と一致するか確かめる）/ 見る面積 {full.km2:.1f} km2")
    for frac in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
        tgt = frac * full.recall
        hitr = cv[cv.recall >= tgt]
        if len(hitr):
            r = hitr.iloc[0]
            # **「全件見たときに拾える正解」のうち何割か**である（見逃しではない。
            # 2026-11-12 に見出しを直した。計算は `frac * full.recall` のまま）
            print(f"  全件で拾える正解の {100 * frac:>3.0f}% を拾うには "
                  f"上位 {int(r.n):>6,} 件（{100 * r.frac:>5.1f}%・"
                  f"{r.km2:>6.1f} km2）")

    # ------------------------------------------------- 2. 分割上限（刈り取り後）
    print("\n" + "=" * 74)
    print("【2】分割の上限 — 2分岐デコーダ（PID-106）をやる価値があるか")
    print("=" * 74)
    thr = float(blobs.logit_max.quantile(a.prune_q))
    drop = blobs[blobs.logit_max <= thr]
    PP = PB.copy()
    for k, g in drop.groupby("tile"):
        pl, _ = label_instances(PB[k], 8, 10)
        for j in g.blob:
            PP[k][pl == j] = False
    rows = {"刈り取り前": split_bound(GT, PB),
            f"刈り取り後（下位{100 * a.prune_q:.0f}%・しきい値 {thr:.2f}）":
                split_bound(GT, PP)}
    band = BAND[a.region]
    print(f"{'':<34}{'飲み込み率':>11}{'箇所F':>9}{'分割上限':>10}"
          f"{'分割の伸び':>12}{'帯で':>8}{'刈り取り上限':>13}")
    for lb, s in rows.items():
        print(f"{lb:<34}{s['swallow']:>11.4f}{s['F']:>9.4f}"
              f"{s['F_split']:>10.4f}{s['F_split'] - s['F']:>+12.4f}"
              f"{(s['F_split'] - s['F']) / band:>7.1f}倍{s['F_prune']:>13.4f}")
    print("\n  箇所Recall は分割で厳密に不変（`19.4`）。動くのは適合率だけで、")
    print("  n_pred と tp_pred が同じだけ増えるので天井が低い。")
    print(f"  判定の帯は {band:.4f}。**伸びが帯を下回るなら作る価値は無い。**")
    print(f"\n  {out}")


if __name__ == "__main__":
    main()
