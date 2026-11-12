# -*- coding: utf-8 -*-
"""建物 GML（1.6 GB・117ファイル）を読んで、多角形の索引 npz を1つ作る。

なぜ分けるのか
--------------
`build_building_pkl.py` は 5 GB の pkl を読む。そこで毎回 GML も読むと
やり直しのたびに数分余計にかかるので、**GML の読み取りだけ先に1回で済ませる。**
`dump_landuse_grids.py` → `build_landuse_pkl.py` と同じ二段である。

使い方
------
    python tools/dump_building_index.py --gml C:/Users/hirok/dc5/Architec/Hiroshima
    python tools/dump_building_index.py --gml <dir> --region Hiroshima

出来るもの（<SSD>/cache/building/building_index__hiroshima.npz、約 300 MB）

    verts/starts/bbox   外環の多角形（float32。(lon, lat)）
    kind                KIND_ORDER の添字（ラスタの面）
    tcode               TYPE_ORDER の添字（棟数の内訳）
    lvl                 orgGILvl（2500 / 25000 ...）
    hole_*              穴。全体で 1,928 個（0.1%）
    meshes              GML があった2次メッシュの一覧
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dc5lib import building, paths                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gml", required=True, help="FG-GML-* が並ぶフォルダ")
    ap.add_argument("--region", default="Hiroshima")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    out = (paths.cache_dir("building")
           / f"building_index__{a.region.lower()}.npz")
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} はすでにあります（--overwrite で作り直す）")

    root = Path(a.gml)
    t0 = time.time()

    def prog(n, tot, got):
        print(f"  {n}/{tot} ファイル  {got:,} 棟  "
              f"{time.time() - t0:.0f}秒", end="\r", flush=True)

    print(f"読み取り: {root}")
    pack = building.parse_folder(root, progress=prog)
    print(" " * 60, end="\r")
    n = len(pack["kind"])
    print(f"  BldA {n:,} 棟 / {pack['n_file']} ファイル  "
          f"{time.time() - t0:.0f}秒")
    print(f"  頂点 {len(pack['verts']):,}（1棟あたり "
          f"{len(pack['verts']) / max(n, 1):.1f}）")
    print(f"  穴 {len(pack['hole_starts']) - 1:,}")
    for i, t in enumerate(building.TYPE_ORDER):
        c = int((pack["tcode"] == i).sum())
        print(f"  {t:<12}{c:>10,}  {100 * c / max(n, 1):5.1f}%")
    for lv in sorted(set(pack["lvl"].tolist())):
        c = int((pack["lvl"] == lv).sum())
        print(f"  1/{lv:<10}{c:>10,}  {100 * c / max(n, 1):5.1f}%")

    # GML があった2次メッシュ。**穴（データが無い場所）の判定に要る**
    meshes = sorted({d.name.split("-")[2] for d in root.glob("FG-GML-*")
                     if len(d.name.split("-")) > 2})
    print(f"  2次メッシュ {len(meshes)}")

    print(f"書き出し: {out} …", flush=True)
    np.savez_compressed(
        out, version=building.INDEX_VERSION,
        kind_order=np.array(building.KIND_ORDER),
        type_order=np.array(building.TYPE_ORDER),
        meshes=np.array(meshes),
        **{k: v for k, v in pack.items() if k != "n_file"})
    print(f"完了: {out}（{out.stat().st_size / 2**20:.0f} MB、"
          f"{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
