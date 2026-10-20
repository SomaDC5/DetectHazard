# -*- coding: utf-8 -*-
"""土地利用メッシュの .shp を読んで、1次メッシュごとの格子を npz に書き出す。

なぜ分けるのか
--------------
シェープファイルを読むのに pyshp が要るが、それが入っているのは `geopack`
環境で、そこには pyyaml が無い（`dc5lib.regions` が読めない）。学習側は
`mayo` 環境。**GIS は geopack、学習は mayo** という切り分けはハザードマップ
生成と同じなので、ここも同じにする。

このスクリプトは `dc5lib.landuse` だけを使い、yaml を要する部分に触らない。

    # geopack で読み出す
    "...\\envs\\geopack\\python.exe" tools/dump_landuse_grids.py ^
        --folder C:\\Users\\hirok\\dc5\\Tochiriyou --out <SSD>\\landuse_grids.npz

    # 以降は mayo で npz を読む
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dc5lib import landuse  # noqa: E402  （numpy と標準ライブラリだけに依存）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="L03-b の .shp があるフォルダ")
    ap.add_argument("--out", required=True, help="書き出す .npz")
    args = ap.parse_args()

    print("土地利用メッシュを読む（測地系の検査つき）")
    grids = landuse.load_all_grids(args.folder)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        order=np.array(landuse.CHANNEL_ORDER),
        keys=np.array(sorted(grids)),
        **{f"g{k}": grids[k] for k in sorted(grids)})
    size = out.stat().st_size / 2 ** 20
    print(f"\n書き出し: {out}（{size:.1f}MB / 1次メッシュ {len(grids)} 枚）")
    print(f"区分の並び: {landuse.CHANNEL_ORDER}")
    print("  " + " / ".join(landuse.LANDUSE_CLASSES.get(c, c)
                            for c in landuse.CHANNEL_ORDER))


if __name__ == "__main__":
    main()
