"""
GeoTIFFファイルを、GDAL/rasterioを使わずに読み込む（tifffileのみ使用）。

「基盤地図情報標高DEM変換ツール」等で作成された標準的なGeoTIFF
（ModelPixelScaleTag=33550, ModelTiepointTag=33922 を持つ、回転なしの
北が上のラスタ）を想定している。

読み込んだ結果は gml2npz.py が出力する形式と同じキーを持つdictで返すので、
build_shimane_dataset_v2.py 側の変更は最小限で済む。
"""

import glob
import os

import numpy as np
import tifffile


def read_geotiff(path, nodata=0.0, nodata_from_tag=True):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray().astype(np.float32)
        tags = page.tags

        pixel_scale = tags[33550].value  # (scaleX, scaleY, scaleZ)
        tiepoint = tags[33922].value     # (I, J, K, X, Y, Z)

        scale_x, scale_y = pixel_scale[0], pixel_scale[1]
        tie_i, tie_j = tiepoint[0], tiepoint[1]
        tie_x, tie_y = tiepoint[3], tiepoint[4]

        # タイポイントが(0,0)以外の場合にも対応
        lon_min = tie_x - tie_i * scale_x
        lat_max = tie_y + tie_j * scale_y

        n_rows, n_cols = arr.shape
        lon_max = lon_min + n_cols * scale_x
        lat_min = lat_max - n_rows * scale_y

        # NoData値（GDAL_NODATAタグ=42113があれば優先的に使用）
        if nodata_from_tag and 42113 in tags:
            try:
                nodata = float(tags[42113].value)
            except (ValueError, TypeError):
                pass

    return {
        "elevation": arr,
        "lat_min": lat_min,
        "lon_min": lon_min,
        "lat_max": lat_max,
        "lon_max": lon_max,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nodata": nodata,
        "epsg": 6668,  # JGD2011（変換ツールの出力がこの座標系である前提。異なる場合は要調整）
    }


def list_geotiffs(folder, pattern="*.tif"):
    return sorted(glob.glob(os.path.join(folder, "**", pattern), recursive=True))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使い方: python read_geotiff.py <GeoTIFFファイルパス>")
        sys.exit(1)

    d = read_geotiff(sys.argv[1])
    print("shape:", d["elevation"].shape)
    print("lat_min, lon_min, lat_max, lon_max:", d["lat_min"], d["lon_min"], d["lat_max"], d["lon_max"])
    print("nodata:", d["nodata"])
    print("標高の範囲(nodata除く):",
          d["elevation"][d["elevation"] != d["nodata"]].min(),
          d["elevation"][d["elevation"] != d["nodata"]].max())
