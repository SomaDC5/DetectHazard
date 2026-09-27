"""
島根県テストデータセット生成パイプライン（DEM + Mask 版）
GDAL/rasterio/fiona/geopandas 不使用。

処理の流れ:
  1. 「基盤地図情報標高DEM変換ツール」等で作成したGeoTIFF(.tif)を直接読み込む（tifffile使用、GDAL不使用）
  2. メッシュ全体を256x256に最近傍補間でリサイズする
  3. 256x256を4分割し、128x128のタイルを4枚作る（左上・右上・左下・右下）
     ※256/2=128で割り切れるため、境界は常に元メッシュのちょうど中央になる
  4. 各タイルの緯度経度範囲に対応する警戒区域マスクを、シェープファイルから
     128x128で直接ラスタライズする
  5. タイルごとの標高最大値・最小値（Max_H/Min_H）を記録する
     （256x256にリサイズした後の値を使用。実際に入力データとして使うのは
     このリサイズ後の値のため）
  6. タイルごとのGeoInfo（GDAL方式geotransform、128x128基準）を計算する
  7. すべてを dem_dataset クラスに詰めてpickle保存する

今回は傾斜量図（SAM）は作らず、DEM（標高そのもの）とMaskのみを生成する。

学習データと違い、テストデータとして「警戒区域を含まないタイル」も除外せず全て含める。

【欠損値について】
このデータの変換ツールは、欠損値をGDAL_NODATAタグなしで単純に0.0として埋めている
（実データで確認済み：河川などレーザーが届かない箇所が0.0になる）。
そのため read_geotiff.py 側でnodataの既定値を0.0としている。
メッシュ全体が海などでほぼ欠損の場合はメッシュごと除外し、タイル内に残る
少量の欠損（主に川筋）はタイル内の有効値平均で埋める2段階の処理を行う。

依存ライブラリ: numpy, pyshp (shapefile), matplotlib, tifffile のみ。GDAL系は一切不使用。

※ このスクリプトは同じフォルダにある read_geotiff.py を使用します。
   2つのファイルを同じフォルダに置いて実行してください。
"""

import argparse
import glob
import os
import pickle

import numpy as np
import shapefile
from matplotlib.path import Path

from read_geotiff import read_geotiff, list_geotiffs

NODATA = -9999.0


# ============================================================
# dem_dataset クラス（お伝えいただいた定義そのまま）
# ============================================================
class dem_dataset:
    def __init__(self):
        self.__KeyName = ["No", "DEM", "Mask", "GeoInfo", "EPSG", "Max_H", "Min_H"]
        self.No = []
        self.DEM = []
        self.Mask = []
        self.GeoInfo = []
        self.EPSG = []
        self.Max_H = []
        self.Min_H = []

    def key(self):
        return self.__KeyName


# ============================================================
# 最近傍リサイズ
# ============================================================
def nearest_resize(arr, out_h, out_w):
    in_h, in_w = arr.shape
    row_idx = np.clip((np.arange(out_h) * in_h / out_h).astype(int), 0, in_h - 1)
    col_idx = np.clip((np.arange(out_w) * in_w / out_w).astype(int), 0, in_w - 1)
    return arr[np.ix_(row_idx, col_idx)]


# ============================================================
# シェープファイル読み込み・マスクラスタライズ
# ============================================================
def load_polygons(shp_path, field_name=None, accept_values=None):
    sf = shapefile.Reader(shp_path, encoding="cp932")
    polygons = []
    for sr in sf.iterShapeRecords():
        shape = sr.shape
        rec = sr.record.as_dict()
        if field_name is not None and str(rec.get(field_name)) not in accept_values:
            continue
        points = np.array(shape.points)
        parts = list(shape.parts) + [len(points)]
        rings = [points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        polygons.append({"bbox": shape.bbox, "rings": rings})
    return polygons


def _bbox_overlap(bbox_a, bbox_b):
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def rasterize_tile(polygons, lat_min, lon_min, lat_max, lon_max, n_rows, n_cols):
    tile_bbox = (lon_min, lat_min, lon_max, lat_max)
    candidates = [p for p in polygons if _bbox_overlap(p["bbox"], tile_bbox)]
    mask = np.zeros((n_rows, n_cols), dtype=np.uint8)
    if not candidates:
        return mask

    col_idx = np.arange(n_cols)
    row_idx = np.arange(n_rows)
    lon_centers = lon_min + (col_idx + 0.5) * (lon_max - lon_min) / n_cols
    lat_centers = lat_max - (row_idx + 0.5) * (lat_max - lat_min) / n_rows
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    for poly in candidates:
        for ring in poly["rings"]:
            inside = Path(ring).contains_points(points)
            mask.ravel()[inside] = 1
    return mask


# ============================================================
# メイン処理
# ============================================================
def split_bounds_half(lat_min, lon_min, lat_max, lon_max, top, left):
    """
    256x256にリサイズ済みの前提で、4分割のうち1タイル分の緯度経度範囲を返す。
    top=Trueなら上半分（北側）、left=Trueなら左半分（西側）。
    256/2=128でちょうど半分に割れるため、常に元メッシュの中央で分割される。
    """
    lat_mid = (lat_min + lat_max) / 2.0
    lon_mid = (lon_min + lon_max) / 2.0

    sub_lat_min = lat_mid if top else lat_min
    sub_lat_max = lat_max if top else lat_mid
    sub_lon_min = lon_min if left else lon_mid
    sub_lon_max = lon_mid if left else lon_max
    return sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max


def build_dataset(tif_dir, shp_path, field_name, accept_values, out_pkl, tile_size=128,
                   resize_to=256, mesh_missing_threshold=0.5, tile_missing_threshold=0.3):
    """
    mesh_missing_threshold : メッシュ全体の欠損値割合がこれを超えたらメッシュごと除外する
        （海に大きくかかっているメッシュなどを弾くためのもの。既定0.5）
    tile_missing_threshold : 4分割した後の1タイルの欠損値割合がこれを超えたらそのタイルを除外する
        （既定0.3）
    どちらの閾値を通過したタイルでも、残った欠損ピクセル（主に川筋）は
    そのタイル内の有効値の平均で埋める（生の欠損値センチネルがモデル入力に
    残らないようにするため）。
    """
    polygons = load_polygons(shp_path, field_name, accept_values)
    print(f"警戒区域ポリゴン数（フィルタ後）: {len(polygons)}")

    ds = dem_dataset()
    tif_files = list_geotiffs(tif_dir)
    print(f"対象メッシュ数: {len(tif_files)}")

    mesh_missing_ratios = []
    n_excluded_mesh = 0
    n_excluded_tile = 0
    idx = 0
    for tif_path in tif_files:
        d = read_geotiff(tif_path)
        elevation = d["elevation"]
        lat_min, lon_min = d["lat_min"], d["lon_min"]
        lat_max, lon_max = d["lat_max"], d["lon_max"]
        epsg = d["epsg"]
        nodata = d["nodata"]

        missing_ratio = np.mean(elevation == nodata)
        mesh_missing_ratios.append(missing_ratio)
        if missing_ratio > mesh_missing_threshold:
            n_excluded_mesh += 1
            continue

        # 1. メッシュ全体を256x256にリサイズ
        elevation_256 = nearest_resize(elevation, resize_to, resize_to)

        # 2. 256x256を4分割（左上・右上・左下・右下）
        half = resize_to // 2  # =128
        quadrants = [
            (0, half, 0, half, True, True),      # 左上（北西）
            (0, half, half, resize_to, True, False),   # 右上（北東）
            (half, resize_to, 0, half, False, True),   # 左下（南西）
            (half, resize_to, half, resize_to, False, False),  # 右下（南東）
        ]

        for (r0, r1, c0, c1, top, left) in quadrants:
            dem_tile = elevation_256[r0:r1, c0:c1].copy()
            valid = dem_tile != nodata
            tile_missing_ratio = 1.0 - valid.mean()

            if not valid.any():
                continue
            if tile_missing_ratio > tile_missing_threshold:
                n_excluded_tile += 1
                continue

            # 残った欠損ピクセル（主に川筋）はタイル内の有効値の平均で埋める
            if not valid.all():
                fill_value = float(dem_tile[valid].mean())
                dem_tile[~valid] = fill_value

            sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max = split_bounds_half(
                lat_min, lon_min, lat_max, lon_max, top, left
            )

            mask_tile = rasterize_tile(
                polygons, sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max,
                tile_size, tile_size
            )

            max_h = float(dem_tile[valid].max())
            min_h = float(dem_tile[valid].min())

            geo_info = (
                sub_lon_min,
                (sub_lon_max - sub_lon_min) / tile_size,
                0.0,
                sub_lat_max,
                0.0,
                -(sub_lat_max - sub_lat_min) / tile_size,
            )

            ds.No.append(idx)
            ds.DEM.append(dem_tile.astype(np.float32))
            ds.Mask.append(mask_tile)
            ds.GeoInfo.append(geo_info)
            ds.EPSG.append(epsg)
            ds.Max_H.append(max_h)
            ds.Min_H.append(min_h)
            idx += 1

    ratios = np.array(mesh_missing_ratios)
    print(f"\nメッシュごとの欠損値割合の分布:")
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}パーセンタイル: {np.percentile(ratios, p)*100:.1f}%")
    print(f"欠損値によりメッシュごと除外した数: {n_excluded_mesh} / {len(tif_files)}")
    print(f"欠損値によりタイル単位で除外した数: {n_excluded_tile}")
    print(f"生成タイル数: {idx}")
    print(f"うち警戒区域を含むタイル数: {sum(1 for m in ds.Mask if m.sum() > 0)}")

    with open(out_pkl, "wb") as f:
        pickle.dump(ds, f)
    print(f"保存しました: {out_pkl}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="島根県テストデータセット生成（DEM+Mask版）")
    parser.add_argument("tif_dir", help="GeoTIFF(.tif)ファイルが入っているフォルダ")
    parser.add_argument("shp_path", help="警戒区域シェープファイル(.shp)のパス")
    parser.add_argument("out_pkl", help="出力するpklファイルパス")
    parser.add_argument("--field", default="A33_001", help="フィルタに使う属性フィールド名")
    parser.add_argument("--values", nargs="+", default=["1"], help="対象とする属性値（島根県データではA33_001=1が急傾斜地の崩壊）")
    parser.add_argument("--mesh-missing-threshold", type=float, default=0.5,
                         help="メッシュ全体の欠損割合がこれを超えたらメッシュごと除外（既定0.5）")
    parser.add_argument("--tile-missing-threshold", type=float, default=0.3,
                         help="タイル単位の欠損割合がこれを超えたらそのタイルを除外（既定0.3）")
    args = parser.parse_args()

    build_dataset(
        args.tif_dir, args.shp_path, args.field, set(args.values), args.out_pkl,
        mesh_missing_threshold=args.mesh_missing_threshold,
        tile_missing_threshold=args.tile_missing_threshold,
    )
