# -*- coding: utf-8 -*-
"""外れた予測に人家があるか。**正解データの見逃しを疑う**（D2 / 第6章 6.4）。

なぜ要るか
----------
`37` で 2段目の余地を測り、残りは「渡す量」しか無いと分かった。
`14.4` の候補3「**そもそも刈れない**」も消去法で残っている。
そこで**正解データそのものを疑う**。

急傾斜地崩壊危険区域の指定基準は、傾斜度30度以上・高さ5m以上に加えて
**人家5戸以上**（または公共施設）である。つまり

    斜面であっても人家が無ければ、そもそも区域に指定されない

**外れた予測に人家があるなら**、それは「危険だが指定されていない斜面」
＝正解データの見逃しであって、モデルの誤りとは言い切れない。
**人家が無いなら**、指定され得ない場所を予測しているだけで、
建物で刈れば適合率が上がる。

どちらなのかを測る。

測るもの
--------
予測のかたまり1件ごとに（3x3 のモザイクで、タイルの端も正しく扱う）

    bld_in      かたまりの内側の建物被覆率
    bld_dist    最近の建物までの距離 [m]
    bld_50/100  周囲 50m / 100m の建物被覆率
    house_50    周囲 50m / 100m の棟数（建物の連結成分の重心で数える）

正解のかたまり側も同じものを測る（検出できた／見逃した）。

    python analysis/run_building_miss.py
    python analysis/run_building_miss.py --condition FinalFusion_SAM_APM_OverlapTrain

**島根は建物データを取得していないので広島のみ。**
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

from analysis.data import load_pkl                                   # noqa: E402
from analysis.run_overlap_eval import (REGION_DIR, grid_index,       # noqa: E402
                                       target_tiles)
from analysis.run_overlap_prune import infer, prune, score           # noqa: E402
from dc5lib import building                                          # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import label_instances, overlap_matrix     # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402
from dc5lib.registry import get                                      # noqa: E402

T = 128
COV_PRED = 0.3      # 予測のかたまりが「的中」とみなされる被覆
COV_GT = 0.5        # 正解のかたまりが「検出された」とみなされる被覆
BAND_H = 0.0071     # 広島の箇所F のゆらぎ幅
RADII = (50.0, 100.0)

BLD_PKL = {"Hiroshima": "hiroshima_bld.pkl", "Shimane": "shimane_bld.pkl"}


def auc(label, scorev):
    y = np.asarray(label).astype(bool)
    s = np.asarray(scorev, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pixel_metres(geo):
    """1画素の大きさ [m]。**東西と南北で違う**（4.50 x 3.60 m）。"""
    lat = float(geo[3])
    mx = abs(float(geo[1])) * 111320 * np.cos(np.radians(lat))
    my = abs(float(geo[5])) * 110570
    return my, mx


def bld_mosaic(B, cell, gi, gj, k):
    """中央タイルと周囲8枚の建物被覆率を 3x3 に貼る。外は 0。

    **端を 0 で埋めるのは「建物が無い」という意味になる。**
    広島は県境の外も収録メッシュに入っているので、
    モザイクが欠けるのは格子の外（海側）だけである。
    """
    a, b = gi[k], gj[k]
    out = np.zeros((3 * T, 3 * T), np.float32)
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            n = cell.get((a + p, b + q))
            if n is None:
                continue
            r, c = (p + 1) * T, (q + 1) * T
            out[r:r + T, c:c + T] = B[n, 0] / 255.0
    return out


def tile_maps(bcov, my, mx):
    """タイル1枚ぶんの建物の地図を先に作る。**かたまりごとに作らない。**

    かたまりごとに膨張させると 3万個 x 3半径 になって終わらない。
    距離変換と箱フィルタはタイルに1回で済む。
    """
    dist = ndimage.distance_transform_edt(bcov <= 0, sampling=(my, mx))
    near = {}
    for r in RADII:
        ky = 2 * int(round(r / my)) + 1
        kx = 2 * int(round(r / mx)) + 1
        near[r] = ndimage.uniform_filter(bcov, size=(ky, kx), mode="constant")
    # 棟の重心。被覆率 > 0 の連結成分を1棟とみなす
    lab, n = ndimage.label(bcov > 0)
    if n:
        cen = np.array(ndimage.center_of_mass(bcov > 0, lab, range(1, n + 1)))
        cen = cen * np.array([my, mx])           # 画素 -> m
    else:
        cen = np.zeros((0, 2))
    return dist, near, cen


def features(mask, dist, near, cen, my, mx):
    """かたまり1件ぶんの建物の特徴。mask はモザイク座標の bool。"""
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean() * my, xs.mean() * mx
    out = {"bld_dist": float(dist[mask].min())}
    for r in RADII:
        out[f"bld_{int(r)}"] = float(near[r][mask].mean())
    if len(cen):
        d = np.hypot(cen[:, 0] - cy, cen[:, 1] - cx)
        for r in RADII:
            out[f"house_{int(r)}"] = int((d <= r).sum())
    else:
        for r in RADII:
            out[f"house_{int(r)}"] = 0
    return out


def tables(GT, PB, LG, B, cell, gi, gj, tiles, my, mx, no=None, verbose=True):
    """予測側と正解側のかたまり表を作る。

    `no` を渡すと**タイル番号**も残す。これが無いと `30` の撮影年や
    `31` の指定年と突き合わせられない（`tile` は並び順の位置でしかない）。
    """
    pred, gt = [], []
    for n, k in enumerate(tiles):
        bcov = bld_mosaic(B, cell, gi, gj, k)
        dist, near, cen = tile_maps(bcov, my, mx)
        bc = bcov[T:2 * T, T:2 * T]

        lab, sizes = label_instances(PB[n], 8, 10)
        gl, gs = label_instances(GT[n], 8, 10)
        O = (overlap_matrix(gl, len(gs), lab, len(sizes))
             if len(gs) and len(sizes) else None)

        if len(sizes):
            cov_p = (O.sum(axis=0) / sizes if O is not None
                     else np.zeros(len(sizes)))
            for j in range(len(sizes)):
                sel = lab == (j + 1)
                big = np.zeros((3 * T, 3 * T), bool)
                big[T:2 * T, T:2 * T] = sel
                row = dict(tile=n, tile_no=int(no[k]) if no is not None
                           else -1, blob=j + 1,
                           tp=bool(cov_p[j] >= COV_PRED),
                           area=int(sizes[j]),
                           logit_max=float(LG[n][sel].max()),
                           bld_in=float(bc[sel].mean()))
                row.update(features(big, dist, near, cen, my, mx))
                pred.append(row)

        if len(gs):
            cov_g = (O.sum(axis=1) / gs if O is not None
                     else np.zeros(len(gs)))
            for i in range(len(gs)):
                sel = gl == (i + 1)
                big = np.zeros((3 * T, 3 * T), bool)
                big[T:2 * T, T:2 * T] = sel
                row = dict(tile=n, tile_no=int(no[k]) if no is not None
                           else -1, gtblob=i + 1,
                           found=bool(cov_g[i] >= COV_GT),
                           area=int(gs[i]),
                           bld_in=float(bc[sel].mean()))
                row.update(features(big, dist, near, cen, my, mx))
                gt.append(row)

        if verbose and n % 200 == 0:
            print(f"    {n}/{len(tiles)}", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    return pd.DataFrame(pred), pd.DataFrame(gt)


COLS = ["bld_in", "bld_dist", "bld_50", "bld_100", "house_50", "house_100"]
LABEL = {"bld_in": "内側の建物被覆率", "bld_dist": "最近の建物まで[m]",
         "bld_50": "50m内の建物被覆率", "bld_100": "100m内の建物被覆率",
         "house_50": "50m内の棟数", "house_100": "100m内の棟数"}


def compare(df, flag, yes, no):
    print(f"{'量':<22}{yes:>12}{no:>12}{'差':>12}{'AUC':>9}")
    y = df[flag].to_numpy().astype(bool)
    for c in COLS:
        a, b = df.loc[y, c].mean(), df.loc[~y, c].mean()
        print(f"{LABEL[c]:<22}{a:>12.4f}{b:>12.4f}{a - b:>+12.4f}"
              f"{auc(y, df[c]):>9.4f}")
    print(f"  件数 {yes} {int(y.sum()):,} / {no} {int((~y).sum()):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", action="store_true",
                    help="かたまり表を cache/analysis/building/ に残す")
    a = ap.parse_args()

    cond = get(a.condition)
    device = torch.device(a.device) if a.device else pick_device()
    dirname = REGION_DIR[a.region]
    print(f"条件 {cond.name} / 地域 {a.region} / device {device}")

    # --- 建物と格子。**タイル番号の一致を必ず確かめる** ---
    bld_path = dataset_path(dirname, BLD_PKL[dirname])
    if not bld_path.exists():
        sys.exit(f"{bld_path} がありません。tools/build_building_pkl.py で作ります")
    print(f"  読み込み: {bld_path.name} …", flush=True)
    bl = load_pkl(bld_path)
    B = np.asarray(bl.Building)
    bno = np.asarray(bl.No)
    geo0 = bl.GeoInfo[0]
    my, mx = pixel_metres(geo0)
    print(f"  建物 {B.shape}  1画素 {mx:.2f} x {my:.2f} m"
          f"  倍率 {getattr(bl, 'BuildingSuper', '?')}")

    src = dataset_path(dirname, cond.pkl_for(dirname))
    print(f"  読み込み: {src.name}（格子と対象タイルのため）…", flush=True)
    ds = load_pkl(src)
    dno = np.asarray(ds.No)
    if not (len(dno) == len(bno) and (dno == bno).all()):
        sys.exit(f"{bld_path.name} と {src.name} でタイル番号が違います。"
                 f"建物を {src.name} から作り直してください")
    bad = sum(1 for g, h in zip(ds.GeoInfo, bl.GeoInfo)
              if not np.allclose(np.asarray(g, float), np.asarray(h, float)))
    if bad:
        sys.exit(f"GeoInfo が {bad} 枚で違います")
    print(f"  タイル番号と GeoInfo の一致を確認（{len(dno):,} 枚）")
    gi, gj, cell = grid_index(ds)
    tiles = target_tiles(ds, a.region,
                         float(getattr(cond, "bg_ratio", 0.0) or 0.0))
    print(f"  対象タイル {len(tiles):,}")
    del ds, bl
    import gc
    gc.collect()

    GT, PB, LG = infer(cond, a.region, device)
    assert len(GT) == len(tiles), (len(GT), len(tiles))

    print("\nかたまりの表を作ります …", flush=True)
    dp, dg = tables(GT, PB, LG, B, cell, gi, gj, tiles, my, mx, no=dno)
    R0, P0, F0, n0 = score(GT, PB)
    print(f"刈り取り前  箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f} / "
          f"予測数 {n0:,}")
    print(f"完璧な刈り取りの上限  箇所F {2 * R0 / (1 + R0):.4f}")

    print("\n" + "=" * 70)
    print("【1】予測のかたまり — 外れた予測に人家はあるか")
    print("=" * 70)
    compare(dp, "tp", "的中", "外れ")
    print("\n  **外れに人家があるなら**『危険だが未指定』の線が残る。")
    print("  **無いなら**指定され得ない場所を予測しているだけで、建物で刈れる。")

    print("\n" + "=" * 70)
    print("【2】正解のかたまり — 見逃した正解に人家はあるか")
    print("=" * 70)
    compare(dg, "found", "検出", "見逃し")

    print("\n" + "=" * 70)
    print("【3】建物で刈れるか（A で選び B で測る。`13.1` と同じ作法）")
    print("=" * 70)
    A = np.arange(0, len(GT), 2)
    Bi = np.arange(1, len(GT), 2)
    _, _, FB0, _ = score(GT, PB, Bi)
    QS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"{'量':<22}{'選ばれた割合':>13}{'しきい値':>10}"
          f"{'B 箇所R':>9}{'B 箇所P':>9}{'B 箇所F':>9}{'正味':>9}{'帯':>7}")
    for c in COLS:
        # bld_dist は**大きいほど人家が遠い**ので、刈る向きが逆
        v = -dp[c] if c == "bld_dist" else dp[c]
        bestq, bestF = 0.0, score(GT, PB, A)[2]
        for q in QS[1:]:
            thr = v[dp.tile.isin(A)].quantile(q)
            F = score(GT, prune(PB, dp[v <= thr]), A)[2]
            if F > bestF:
                bestq, bestF = q, F
        if bestq == 0:
            print(f"{LABEL[c]:<22}{'刈らないのが最良':>13}")
            continue
        thr = v[dp.tile.isin(A)].quantile(bestq)
        RB, PBr, FB, _ = score(GT, prune(PB, dp[v <= thr]), Bi)
        raw = -thr if c == "bld_dist" else thr
        print(f"{LABEL[c]:<22}{100 * bestq:>12.0f}%{raw:>10.3f}"
              f"{RB:>9.4f}{PBr:>9.4f}{FB:>9.4f}{FB - FB0:>+9.4f}"
              f"{(FB - FB0) / BAND_H:>7.1f}")
    print(f"\n  B 刈る前の箇所F {FB0:.4f}。ゆらぎ幅 {BAND_H} の何倍かが『帯』")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("【4】ロジット刈り取りに**足せる**のか（これが本題）")
    print("=" * 70)
    print("  建物は航空写真に写っているので、1段目のロジットに既に"
          "織り込まれている\n  可能性がある（`14.3` と同じ重複）。"
          "**足して伸びなければ重複である。**")
    print()

    inA = dp.tile.isin(A)
    QL = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    def thr_of(col, q, sub=None):
        v = dp[col][inA if sub is None else (inA & sub)]
        return -np.inf if (q == 0 or not len(v)) else v.quantile(q)

    def drop_mask(ql, qb):
        return ((dp.logit_max <= thr_of("logit_max", ql))
                | (dp.bld_in <= thr_of("bld_in", qb)))

    def best_on_A(pairs):
        """**しきい値は A だけで選ぶ。** B は測るためにしか使わない。"""
        best, bestF = (0.0, 0.0), -1.0
        for ql, qb in pairs:
            F = score(GT, prune(PB, dp[drop_mask(ql, qb)]), A)[2]
            if F > bestF:
                best, bestF = (ql, qb), F
        return best, bestF

    only_lg, _ = best_on_A([(q, 0.0) for q in QL])
    only_bd, _ = best_on_A([(0.0, q) for q in QL])
    both, _ = best_on_A([(a_, b_) for a_ in QL for b_ in QL])

    rows = [("刈らない", (0.0, 0.0)),
            (f"ロジット最大値だけ（{100 * only_lg[0]:.0f}%）", only_lg),
            (f"建物被覆率だけ（{100 * only_bd[1]:.0f}%）", only_bd),
            (f"**両方**（ロジット {100 * both[0]:.0f}% / "
             f"建物 {100 * both[1]:.0f}%）", both)]

    print(f"{'刈り方':<40}{'捨てた':>8}{'B 箇所R':>9}{'B 箇所P':>9}"
          f"{'B 箇所F':>9}{'正味':>9}{'帯':>7}")
    got = {}
    for name, (ql, qb) in rows:
        m = drop_mask(ql, qb)
        nd = 0 if (ql == 0 and qb == 0) else int(m.sum())
        R, P, F, _ = (score(GT, PB, Bi) if nd == 0
                      else score(GT, prune(PB, dp[m]), Bi))
        got[name] = F
        print(f"{name:<40}{nd:>8,}{R:>9.4f}{P:>9.4f}{F:>9.4f}"
              f"{F - FB0:>+9.4f}{(F - FB0) / BAND_H:>7.1f}")

    f_lg = got[rows[1][0]]
    f_bo = got[rows[3][0]]
    d = f_bo - f_lg
    print()
    print(f"  **ロジットだけ {f_lg:.4f} → 建物も足して {f_bo:.4f}"
          f"（{d:+.4f}、{d / BAND_H:.1f}帯）**")
    if abs(d) < BAND_H:
        print("  帯の中。**建物の情報はロジットと重複している。**")
        print("  `37.4` の原則どおりである。建物は航空写真に写っているので、")
        print("  1段目の内側に存在する量であり、新しい情報ではない。")
    else:
        print("  帯の外。**建物はロジットに無い情報を持っている。**")

    if a.save:
        from dc5lib import paths
        d = paths.cache_dir("analysis") / "building"
        d.mkdir(parents=True, exist_ok=True)
        dp.to_csv(d / f"{cond.name}__{a.region}__pred.csv",
                  index=False, encoding="utf-8")
        dg.to_csv(d / f"{cond.name}__{a.region}__gt.csv",
                  index=False, encoding="utf-8")
        print(f"\n書き出し: {d}")


if __name__ == "__main__":
    main()
