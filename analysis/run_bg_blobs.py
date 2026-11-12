# -*- coding: utf-8 -*-
"""**背景タイルに出た誤検出は、未指定の危険斜面か**（SCIS&ISIS 2026 PID-186）。

なぜ要るか
----------
PID-186（渦電流検査）は「**管1本につき最も深い欠陥だけが注記されている**」
状況を **部分ラベル教師**（partial-label supervision）として明示的に定式化する。

**こちらの正解も部分ラベルである。**
発表資料は「目的は**人の目視で見落とされうる斜面を出すこと**」と書いている ──
つまり**正解に入っていない危険斜面の存在を前提にしている**のに、
評価ではそれを全部「誤検出」として数えている。

| いまの扱い | 部分ラベルとしての扱い |
|---|---|
| 指定区域の外＝負例 | 指定区域の外＝**未知**（負例とは限らない）|

**`39` との違い** ── `39`（`run_building_miss.py`）は
**警戒区域を含むテストタイルの中**で「外れた予測に人家があるか」を測り、
「人家はあるが**指定基準の5戸には届かない**」と結論した（`39.7`）。
本節が見るのは**警戒区域を1画素も含まないタイル**である。
そこに出た予測は**定義上すべて誤検出**で、かつ
「近くの指定区域のついで」という説明が効かない。
**未指定の危険斜面という仮説が、いちばん素直に試せる場所である。**

急傾斜地崩壊危険区域の指定基準
------------------------------
    傾斜度 30 度以上 / 高さ 5m 以上 / **人家 5 戸以上**（または公共施設）

**3条件を満たす誤検出がどれだけあるか**を数える。満たすなら
「モデルの誤り」ではなく「**指定の漏れ**」の候補である。

**高さ 5m はタイル単位の起伏（Max_H - Min_H）しか無い**ので、
かたまり単位では確かめていない。満たさない方向の見落としはあり得る。

何を出すか
----------
  ・背景タイル全件にオーバーラップ推論をかけ、かたまりごとに
      ロジット最大値 / 面積 / 傾斜量（平均・90%点・最大）
      建物（内側被覆率・最近距離・50m/100m 被覆率・棟数）
      隣接タイルに指定区域があるか（**区域のすぐ外か、独立しているか**）
  ・ロジットの上位群と下位群で、上の量に差が出るか
  ・**指定基準の3条件を満たす件数**を順位ごとに

出力
----
  cache/analysis/triage/<条件>__hiroshima_bg__blobs.csv

    python analysis/run_bg_blobs.py
    python analysis/run_bg_blobs.py --sample 4000      # 下見
"""

from __future__ import annotations

import argparse
import gc
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
from analysis.run_building_miss import (BLD_PKL, COLS, LABEL,        # noqa: E402
                                        auc, bld_mosaic, features,
                                        pixel_metres, tile_maps)
from analysis.run_overlap_eval import (REGION_DIR,                   # noqa: E402
                                       grid_index)
from analysis.run_overlap_prune import infer                         # noqa: E402
from dc5lib import paths, splits as _splits                          # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import label_instances                     # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402
from dc5lib.registry import get                                      # noqa: E402
from dc5lib.traincfg import train_cfg                                # noqa: E402

T = 128
STRUCT4 = ndimage.generate_binary_structure(2, 1)
# 指定基準（急傾斜地崩壊危険区域）
SLOPE_DEG, HOUSES = 30.0, 5


def neighbour_has_gt(cell, gi, gj, k, has_gt):
    """8近傍のどれかが `has_gt` に当たるか。**区域のすぐ外かを分けるため。**

    `OFFSETS` は画素のずらし量なので格子には使えない。格子は ±1 で見る。
    `has_gt` はただの真偽の配列なので、**学習タイルかどうかにも使える**
    （`near_train`）。
    """
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            n = cell.get((gi[k] + a, gj[k] + b))
            if n is not None and has_gt[n]:
                return True
    return False


def blob_table(PB, LG, slope, B, cell, gi, gj, tiles, no, relief,
               near_gt, my, mx, verbose=True, near_train=None):
    rows = []
    for n, k in enumerate(tiles):
        lab, sizes = label_instances(PB[n], 8, 10)
        if len(sizes) == 0:
            continue
        bcov = bld_mosaic(B, cell, gi, gj, k)
        dist, near, cen = tile_maps(bcov, my, mx)
        bc = bcov[T:2 * T, T:2 * T]
        lg, sl = LG[n], slope[n]
        for j in range(len(sizes)):
            sel = lab == (j + 1)
            big = np.zeros((3 * T, 3 * T), bool)
            big[T:2 * T, T:2 * T] = sel
            inner = ndimage.binary_erosion(sel, STRUCT4)
            v, s = lg[sel], sl[sel]
            row = dict(tile=n, tile_no=int(no[k]), blob=j + 1,
                       tp=False, area=int(sizes[j]),
                       logit_max=float(v.max()), logit_mean=float(v.mean()),
                       compactness=float(sel.sum() - inner.sum())
                       / max(int(sizes[j]), 1),
                       slope_mean=float(s.mean()),
                       slope_p90=float(np.percentile(s, 90)),
                       slope_max=float(s.max()), slope_std=float(s.std()),
                       relief=float(relief[k]),
                       near_gt=bool(near_gt[n]),
                       near_train=(None if near_train is None
                                   else bool(near_train[n])),
                       bld_in=float(bc[sel].mean()))
            row.update(features(big, dist, near, cen, my, mx))
            rows.append(row)
        if verbose and n % 500 == 0:
            print(f"    {n}/{len(tiles)}", end="\r", flush=True)
    if verbose:
        print(" " * 40, end="\r")
    return pd.DataFrame(rows)


def groups(df, col, q=0.1):
    hi = df[df[col] >= df[col].quantile(1 - q)]
    lo = df[df[col] <= df[col].quantile(q)]
    return hi, lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--sample", type=int, default=0,
                    help="背景タイルを何枚に絞るか（0 なら全件）。下見用")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cond = get(a.condition)
    device = torch.device(a.device) if a.device else pick_device()
    dirname = REGION_DIR["hiroshima"]
    print(f"条件 {cond.name} / 広島の背景タイル / device {device}")

    # --- 建物。**タイル番号と GeoInfo の一致を必ず確かめる**（`39` と同じ作法）
    bld_path = dataset_path(dirname, BLD_PKL[dirname])
    if not bld_path.exists():
        sys.exit(f"{bld_path} がありません")
    print(f"  読み込み: {bld_path.name} …", flush=True)
    bl = load_pkl(bld_path)
    B = np.asarray(bl.Building)
    bno = np.asarray(bl.No)

    # `pkl_for` は絶対パスを返す。`dataset_path` に通すと pathlib が
    # 左側を捨てて**たまたま**同じ結果になるだけなので、通さない（レビュー 4）。
    src = cond.pkl_for(dirname)
    print(f"  読み込み: {src.name} …", flush=True)
    ds = load_pkl(src)
    dno = np.asarray(ds.No)
    if not (len(dno) == len(bno) and (dno == bno).all()):
        sys.exit(f"{bld_path.name} と {src.name} でタイル番号が違います")
    bad = sum(1 for g, h in zip(ds.GeoInfo, bl.GeoInfo)
              if not np.allclose(np.asarray(g, float), np.asarray(h, float)))
    if bad:
        sys.exit(f"GeoInfo が {bad} 枚で違います")
    print(f"  タイル番号と GeoInfo の一致を確認（{len(dno):,} 枚）")

    gi, gj, cell = grid_index(ds)
    has_gt = np.array([np.max(np.asarray(ds.Mask[k])) >= 1
                       for k in range(len(dno))])
    tiles = [k for k in range(len(dno)) if not has_gt[k]]
    print(f"  背景タイル {len(tiles):,} / 全 {len(dno):,}"
          f"（警戒区域あり {int(has_gt.sum()):,}）")
    if a.sample and a.sample < len(tiles):
        rs = np.random.RandomState(42)
        tiles = sorted(rs.choice(tiles, a.sample, replace=False).tolist())
        print(f"  **下見のため {len(tiles):,} 枚に絞った**")
    slope = np.stack([np.asarray(ds.DEM[k], dtype=np.float32) for k in tiles])
    relief = {k: float(ds.Max_H[k]) - float(ds.Min_H[k]) for k in tiles}
    near_gt = [neighbour_has_gt(cell, gi, gj, k, has_gt) for k in tiles]

    # --- **学習タイルに接しているか**（レビュー 3.1 の交絡）。
    # オーバーラップ学習は近傍に背景タイルを貼るので、**学習タイルに接した
    # 背景タイルは外側の最大 64px がマスク0（負例）として学習に入っている。**
    # 指定区域の隣ほどそうなりやすいので、`near_gt` の比較はこれと交絡する。
    # 分けて数えられるように列にしておく。
    keep = [k for k in range(len(dno)) if has_gt[k]]
    sp = _splits.resolve(cond, [int(dno[k]) for k in keep], train_cfg(cond))
    is_train = np.zeros(len(dno), bool)
    is_train[[keep[int(i)] for i in sp["idx"]["train"]]] = True
    near_train = [neighbour_has_gt(cell, gi, gj, k, is_train) for k in tiles]

    my, mx = pixel_metres(ds.GeoInfo[tiles[0]])
    m2 = my * mx
    print(f"  1画素 {my:.2f} x {mx:.2f} m = {m2:.1f} m2")
    print(f"  隣に区域がある背景タイル {int(np.sum(near_gt)):,}"
          f" / 隣が学習タイルの背景タイル {int(np.sum(near_train)):,}"
          f"（学習 {int(is_train.sum()):,} 枚）")
    del ds, bl
    gc.collect()

    GT, PB, LG = infer(cond, "hiroshima", device, tiles=tiles)
    if GT.any():
        sys.exit("背景タイルのはずなのに正解が入っています。タイルの選び方が違います")

    print("\nかたまりの表を作ります …", flush=True)
    df = blob_table(PB, LG, slope, B, cell, gi, gj, tiles, dno, relief,
                    near_gt, my, mx, near_train=near_train)
    out = paths.cache_dir("analysis") / "triage"
    out.mkdir(parents=True, exist_ok=True)
    # **下見は別名に書く**（レビュー 3.4 ── 全件の表を上書きしていた）。
    tag = f"__sample{a.sample}" if a.sample else ""
    f = out / f"{cond.name}__hiroshima_bg{tag}__blobs.csv"
    df.to_csv(f, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 74)
    print("【1】背景タイルに出た誤検出の量")
    print("=" * 74)
    print(f"  かたまり {len(df):,} 件 / 面積の合計 "
          f"{df.area.sum() * m2 / 1e6:.1f} km2 / "
          f"1枚あたり {len(df) / len(tiles):.2f} 件")
    print(f"  出たタイル {df.tile.nunique():,} / {len(tiles):,} "
          f"（{100 * df.tile.nunique() / len(tiles):.1f}%）")
    print(f"  隣に指定区域があるタイルに出たもの "
          f"{int(df.near_gt.sum()):,}（{100 * df.near_gt.mean():.1f}%）")
    if "near_train" in df and df.near_train.notna().all():
        print(f"  隣が**学習タイル**のものに出たもの "
              f"{int(df.near_train.sum()):,}（{100 * df.near_train.mean():.1f}%）")
        print("  **`near_gt` の比較はこれと交絡する。** オーバーラップ学習は")
        print("  近傍に背景タイルを貼るので、学習タイルに接した背景タイルは")
        print("  外側の最大 64px がマスク0（負例）として学習に入っている。")

    print("\n" + "=" * 74)
    print("【2】ロジットの上位10% と下位10% で素性が違うか")
    print("=" * 74)
    hi, lo = groups(df, "logit_max")
    cols = ["slope_mean", "slope_p90", "slope_max", "relief", "area"] + COLS
    lab = dict(LABEL, slope_mean="傾斜 平均[度]", slope_p90="傾斜 90%点[度]",
               slope_max="傾斜 最大[度]", relief="タイルの起伏[m]",
               area="面積[px]")
    print(f"{'量':<22}{'上位10%':>12}{'下位10%':>12}{'差':>12}{'AUC*':>9}")
    y = np.zeros(len(df), bool)
    y[df.index.get_indexer(hi.index)] = True
    for c in cols:
        print(f"{lab[c]:<22}{hi[c].mean():>12.4f}{lo[c].mean():>12.4f}"
              f"{hi[c].mean() - lo[c].mean():>+12.4f}"
              f"{auc(y, df[c]):>9.4f}")
    print(f"  件数 上位 {len(hi):,} / 下位 {len(lo):,} / 全 {len(df):,}")
    print("\n  **『上位10%』『下位10%』『差』は上下10%どうしの比較。**")
    print("  ***AUC だけは『上位10% 対 残りの90%』**（全件で順位を見る）。")
    print("  **集団が違うので、差と AUC を同じ比較として読まないこと**")
    print("  （2026-11-08。レビュー 3.2 で見出しだけが上下10%になっていた）。")
    print("  AUC は『ロジットが高いほどその量も大きいか』。0.5 なら無関係。")
    print("  上位ほど急で人家が多いなら、")
    print("  誤検出の上位は**地形的に危険斜面の性質を持っている。**")

    print("\n" + "=" * 74)
    print(f"【3】指定基準を満たす誤検出（傾斜90%点 >= {SLOPE_DEG:.0f}度 かつ "
          f"50m内の棟数 >= {HOUSES}）")
    print("=" * 74)
    ok = (df.slope_p90 >= SLOPE_DEG) & (df.house_50 >= HOUSES)
    df2 = df.assign(ok=ok).sort_values("logit_max", ascending=False)
    print(f"{'上位':>10}{'件数':>9}{'基準を満たす':>14}{'割合':>9}"
          f"{'うち隣に区域なし':>18}")
    for n in (100, 500, 1000, 2000, 5000, len(df2)):
        if n > len(df2):
            continue
        s = df2.head(n)
        indep = s[s.ok & ~s.near_gt]
        print(f"{n:>10,}{n:>9,}{int(s.ok.sum()):>14,}"
              f"{100 * s.ok.mean():>8.1f}%{len(indep):>18,}")
    print(f"\n  全体で基準を満たすもの {int(ok.sum()):,} 件"
          f"（{100 * ok.mean():.1f}%）/ "
          f"面積 {df[ok].area.sum() * m2 / 1e6:.2f} km2")
    print("  **高さ5m はタイルの起伏しか無いので、かたまり単位では確かめていない。**")
    print(f"\n  {f}")


if __name__ == "__main__":
    main()
