# -*- coding: utf-8 -*-
"""既存の pkl に土地利用のチャネルを足した pkl を作る。

なぜ pkl にするのか
-------------------
土地利用の格子そのものは 0.4MB の npz で足りるが、**学習の入力は pkl で
持つ**。理由は3つ。

  ・2台のPCの間で運ぶ単位が pkl。npz + その場生成だと「npz」と
    「landuse.py の版」の両方を揃える必要があり、片方が古いと黙ってずれる
  ・学習ノートブックは pkl を読んで属性を引くだけにしておきたい。
    その場生成はメッシュコードの計算をノートブックに持ち込む
  ・pkl は results/ssd_manifests/ に sha で載る凍った成果物。
    その場生成はコードの版に依存して過去の記録が再現できない

**航空写真と地形量は作り直さない。** 既存の pkl を1枚読んで、
タイルごとに GeoInfo からメッシュを引き、属性を2つ足して書き出すだけ。

足す属性
--------
    LandUse       uint8 [128,128]  区分番号（CHANNEL_ORDER の添字、255は不明）
    LandUseOrder  区分の並び。**pkl 自身に持たせる**

LandUseOrder を中に入れるのが要点。これが無いと、あとで CHANNEL_ORDER を
変えたときに古い pkl の番号が黙って別の区分として読まれる。
（今回コード表を取り違えたのと同じ事故になる）

使い方（mayo 環境。先に geopack で npz を作っておく）
----------------------------------------------------
    python tools/dump_landuse_grids.py --folder <shp> --out <SSD>/landuse_grids.npz

    python tools/build_landuse_pkl.py --grids <SSD>/cache/landuse/landuse_grids.npz
    python tools/build_landuse_pkl.py --grids ... --regions Hiroshima
    python tools/build_landuse_pkl.py --grids ... --verify-only
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data import load_pkl                                   # noqa: E402
from dc5lib import landuse                                           # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402

# 元にする pkl と、作る pkl
SOURCES = {
    "Hiroshima": ("hiroshima_sam_apm.pkl", "hiroshima_sam_apm_lu.pkl"),
    "Shimane":   ("shimane_sam_apm.pkl",   "shimane_sam_apm_lu.pkl"),
}

# 属性はそのまま引き継ぐ
CARRY = ("No", "DEM", "Mask", "GeoInfo", "EPSG", "Max_H", "Min_H", "AirPhoto")


def build(region, grids, overwrite=False):
    src_name, out_name = SOURCES[region]
    src = dataset_path(region, src_name)
    out = dataset_path(region, out_name)
    if out.exists() and not overwrite:
        print(f"  {out.name} はすでにあります（--overwrite で作り直す）")
        return out

    print(f"  読み込み: {src.name}（{src.stat().st_size / 2**30:.1f} GB）…",
          flush=True)
    ds = load_pkl(src)
    n = len(ds.No)
    print(f"  {n:,} タイル")

    lu = np.empty((n, landuse.TILE, landuse.TILE), np.uint8)
    t0 = time.time()
    for k in range(n):
        _, cls = landuse.tile_channels_grid(ds.GeoInfo[k], grids)
        lu[k] = cls
        if k % 5000 == 0:
            print(f"    {k:,}/{n:,}  {time.time() - t0:.0f}秒", end="\r", flush=True)
    print(" " * 40, end="\r")

    known = int((lu != landuse.UNKNOWN).sum())
    print(f"  区分あり {known:,} / {lu.size:,} 画素 "
          f"({100 * known / lu.size:.1f}%)")

    # 参照だけ移すので、ここでデータが二重になることはない
    newds = SimpleNamespace(**{k: getattr(ds, k) for k in CARRY if hasattr(ds, k)})
    newds.LandUse = lu
    newds.LandUseOrder = list(landuse.CHANNEL_ORDER)
    newds.LandUseCodelist = landuse.CODELIST
    del ds
    gc.collect()

    print(f"  書き出し: {out.name} …", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(newds, f, protocol=4)
    tmp.replace(out)
    print(f"  完了: {out}（{out.stat().st_size / 2**30:.1f} GB）")
    del newds
    gc.collect()
    return out


def verify(region, grids):
    """書いた pkl を読み直して、区分の分布と並びを確かめる。"""
    _, out_name = SOURCES[region]
    out = dataset_path(region, out_name)
    if not out.exists():
        print(f"  {out.name} がありません")
        return
    ds = load_pkl(out)
    order = list(getattr(ds, "LandUseOrder", []))
    print(f"  {out.name}: {len(ds.No):,} タイル")
    print(f"  LandUseOrder = {order}")
    if order != list(landuse.CHANNEL_ORDER):
        print("  ※ いまの CHANNEL_ORDER と違います。読むときに展開を合わせること")
    print(f"  LandUseCodelist = {getattr(ds, 'LandUseCodelist', '（無し）')}")

    lu = np.asarray(ds.LandUse)
    c = Counter(lu.ravel().tolist())
    tot = lu.size
    print("  %-18s %12s %8s" % ("区分", "画素", "割合"))
    for i, code in enumerate(order):
        name = landuse.LANDUSE_CLASSES.get(code, code)
        print("  %-18s %12s %7.2f%%"
              % (name, f"{c.get(i, 0):,}", 100 * c.get(i, 0) / tot))
    u = c.get(landuse.UNKNOWN, 0)
    print("  %-18s %12s %7.2f%%" % ("（不明）", f"{u:,}", 100 * u / tot))

    # 元の npz から作り直して一致するか（先頭200タイル）
    bad = 0
    for k in range(min(200, len(ds.No))):
        _, cls = landuse.tile_channels_grid(ds.GeoInfo[k], grids)
        if not np.array_equal(cls, lu[k]):
            bad += 1
    print(f"  npz から作り直して一致: {200 - bad}/200"
          + ("" if bad == 0 else f"  ※ {bad} 枚が不一致"))
    del ds
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True,
                    help="tools/dump_landuse_grids.py が書いた npz")
    ap.add_argument("--regions", nargs="*", default=list(SOURCES))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    grids = landuse.load_grids_npz(args.grids)
    print(f"土地利用の格子 {len(grids)} 枚: {sorted(grids)}")
    print(f"区分の並び {landuse.CHANNEL_ORDER}（{landuse.CODELIST}）")

    for region in args.regions:
        print(f"\n=== {region} ===")
        if not args.verify_only:
            build(region, grids, overwrite=args.overwrite)
        print("  --- 検証 ---")
        verify(region, grids)


if __name__ == "__main__":
    main()
