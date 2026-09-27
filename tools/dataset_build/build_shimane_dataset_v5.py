"""
島根県テストデータセット生成パイプライン（DEM/SAM + Mask 版）
GDAL/rasterio/fiona/geopandas 不使用。

【v3からの変更点】
  - 傾斜量図（SAM）モードを追加（--mode sam）
  - 欠損値（川筋など）の穴埋めを、ネイティブ解像度の時点で「最近傍の有効ピクセル値」
    で埋める方式に変更（scipy.ndimage.distance_transform_edtを使用）。
    256x256にリサイズしてから埋めると、最近傍補間による値の重複と埋め処理が
    干渉し、傾斜計算に偽の平坦地や偽の急斜面が生じるため、必ず
    「ネイティブ解像度で穴埋め→傾斜計算→リサイズ→分割」の順序で処理する。

処理の流れ:
  1. GeoTIFFを読み込む（tifffile使用、GDAL不使用）
  2. メッシュ全体の欠損割合をチェックし、閾値を超えたらメッシュごと除外
  3. 欠損ピクセル（主に川筋）を最近傍の有効ピクセル値で穴埋め（ネイティブ解像度）
  4. mode=sam の場合はここでHorn法により傾斜量図を計算（ネイティブ解像度、
     実距離[m]換算のピクセル間隔を使用）。mode=dem の場合は標高そのものを使う
  5. 出力対象データ・標高データ・元の欠損マスクをそれぞれ256x256に最近傍補間でリサイズ
  6. 256x256を4分割し、128x128のタイルを4枚作る
  7. 分割後のタイルで、元の欠損割合が閾値を超えるタイルは除外
  8. 各タイルの緯度経度範囲に対応する警戒区域マスクをシェープファイルから
     128x128で直接ラスタライズする
  9. タイルごとの標高最大値・最小値（Max_H/Min_H、元の欠損ピクセルは除外して計算）を記録
  10. タイルごとのGeoInfo（GDAL方式geotransform、128x128基準）を計算する
  11. すべてを dem_dataset クラスに詰めてpickle保存する

学習データと違い、テストデータとして「警戒区域を含まないタイル」も除外せず全て含める。

依存ライブラリ: numpy, pyshp (shapefile), matplotlib, tifffile, scipy のみ。GDAL系は一切不使用。

※ このスクリプトは同じフォルダにある read_geotiff.py を使用します。
"""

import argparse
import pickle
import time

import numpy as np
import requests
import shapefile
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt

from read_geotiff import read_geotiff, list_geotiffs
from fetch_aerial_v2 import get_airphoto_exact


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
# 欠損値の穴埋め（最近傍の有効ピクセル値で埋める）
# ============================================================
def fill_nodata_nearest(arr, invalid_mask):
    """invalid_maskがTrueの箇所を、最も近い有効ピクセルの値で埋める"""
    if not invalid_mask.any():
        return arr
    idx = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


# ============================================================
# Horn法による傾斜量図計算
# ============================================================
def compute_slope_map(elevation, dx_m, dy_m):
    """elevationは既に穴埋め済み（欠損なし）の前提"""
    z = elevation.astype(np.float64)
    zp = np.pad(z, pad_width=1, mode="edge")
    z1 = zp[0:-2, 0:-2]; z2 = zp[0:-2, 1:-1]; z3 = zp[0:-2, 2:]
    z4 = zp[1:-1, 0:-2];                      z6 = zp[1:-1, 2:]
    z7 = zp[2:, 0:-2];   z8 = zp[2:, 1:-1];    z9 = zp[2:, 2:]

    dzdx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * dx_m)
    dzdy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * dy_m)

    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    return slope_deg.astype(np.float32)


def pixel_spacing_from_bounds(lat_min, lon_min, lat_max, lon_max, n_rows, n_cols):
    mean_lat_rad = np.radians((lat_min + lat_max) / 2.0)
    deg_to_m_lat = 111320.0
    deg_to_m_lon = 111320.0 * np.cos(mean_lat_rad)
    dx_m = (lon_max - lon_min) / n_cols * deg_to_m_lon
    dy_m = (lat_max - lat_min) / n_rows * deg_to_m_lat
    return dx_m, dy_m


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
    lat_mid = (lat_min + lat_max) / 2.0
    lon_mid = (lon_min + lon_max) / 2.0
    sub_lat_min = lat_mid if top else lat_min
    sub_lat_max = lat_max if top else lat_mid
    sub_lon_min = lon_min if left else lon_mid
    sub_lon_max = lon_mid if left else lon_max
    return sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max


def build_dataset(tif_dir, shp_path, field_name, accept_values, out_pkl, mode="dem",
                   tile_size=128, resize_to=256,
                   mesh_missing_threshold=0.5, tile_missing_threshold=0.3,
                   with_apm=False, apm_zoom=18, apm_cache_dir="tile_cache",
                   checkpoint_every=200):
    assert mode in ("dem", "sam")

    polygons = load_polygons(shp_path, field_name, accept_values)
    print(f"警戒区域ポリゴン数（フィルタ後）: {len(polygons)}")

    ds = dem_dataset()
    if with_apm:
        ds.AirPhoto = []
    tif_files = list_geotiffs(tif_dir)
    print(f"対象メッシュ数: {len(tif_files)}")

    session = requests.Session() if with_apm else None
    start_time = time.time()

    mesh_missing_ratios = []
    n_excluded_mesh = 0
    n_excluded_tile = 0
    idx = 0
    for mesh_i, tif_path in enumerate(tif_files):
        d = read_geotiff(tif_path)
        elevation = d["elevation"]
        lat_min, lon_min = d["lat_min"], d["lon_min"]
        lat_max, lon_max = d["lat_max"], d["lon_max"]
        epsg = d["epsg"]
        nodata = d["nodata"]
        n_rows, n_cols = elevation.shape

        invalid_native = elevation == nodata
        missing_ratio = invalid_native.mean()
        mesh_missing_ratios.append(missing_ratio)
        if missing_ratio > mesh_missing_threshold:
            n_excluded_mesh += 1
            continue

        # ネイティブ解像度で欠損を穴埋め（最近傍の有効値）
        elevation_filled = fill_nodata_nearest(elevation, invalid_native)

        if mode == "sam":
            dx_m, dy_m = pixel_spacing_from_bounds(lat_min, lon_min, lat_max, lon_max, n_rows, n_cols)
            output_native = compute_slope_map(elevation_filled, dx_m, dy_m)
        else:
            output_native = elevation_filled

        # リサイズ（傾斜量図 or 標高、標高そのもの、元の欠損マスクの3つ）
        output_256 = nearest_resize(output_native, resize_to, resize_to)
        elevation_256 = nearest_resize(elevation_filled, resize_to, resize_to)
        invalid_256 = nearest_resize(invalid_native.astype(np.uint8), resize_to, resize_to).astype(bool)

        half = resize_to // 2
        quadrants = [
            (0, half, 0, half, True, True),
            (0, half, half, resize_to, True, False),
            (half, resize_to, 0, half, False, True),
            (half, resize_to, half, resize_to, False, False),
        ]

        for (r0, r1, c0, c1, top, left) in quadrants:
            out_tile = output_256[r0:r1, c0:c1]
            elev_tile = elevation_256[r0:r1, c0:c1]
            invalid_tile = invalid_256[r0:r1, c0:c1]

            tile_missing_ratio = invalid_tile.mean()
            if tile_missing_ratio > tile_missing_threshold:
                n_excluded_tile += 1
                continue

            valid_tile = ~invalid_tile
            if not valid_tile.any():
                continue

            sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max = split_bounds_half(
                lat_min, lon_min, lat_max, lon_max, top, left
            )

            mask_tile = rasterize_tile(
                polygons, sub_lat_min, sub_lon_min, sub_lat_max, sub_lon_max,
                tile_size, tile_size
            )

            max_h = float(elev_tile[valid_tile].max())
            min_h = float(elev_tile[valid_tile].min())

            geo_info = (
                sub_lon_min,
                (sub_lon_max - sub_lon_min) / tile_size,
                0.0,
                sub_lat_max,
                0.0,
                -(sub_lat_max - sub_lat_min) / tile_size,
            )

            ds.No.append(idx)
            ds.DEM.append(out_tile.astype(np.float32))
            ds.Mask.append(mask_tile)
            ds.GeoInfo.append(geo_info)
            ds.EPSG.append(epsg)
            ds.Max_H.append(max_h)
            ds.Min_H.append(min_h)

            if with_apm:
                air = get_airphoto_exact(
                    geo_info, apm_cache_dir, dem_size=tile_size, zoom=apm_zoom, session=session
                )
                ds.AirPhoto.append(air)

            idx += 1

        # 一定件数のメッシュを処理するごとに途中経過を保存（中断してもここまでは残る）
        if with_apm and checkpoint_every and (mesh_i + 1) % checkpoint_every == 0:
            elapsed = time.time() - start_time
            with open(out_pkl, "wb") as f:
                pickle.dump(ds, f)
            print(f"  [チェックポイント] メッシュ{mesh_i + 1}/{len(tif_files)}処理済み "
                  f"タイル数={idx} 経過時間={elapsed/60:.1f}分 途中保存: {out_pkl}")

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
    parser = argparse.ArgumentParser(description="島根県テストデータセット生成（DEM/SAM+Mask版）")
    parser.add_argument("tif_dir", help="GeoTIFF(.tif)ファイルが入っているフォルダ")
    parser.add_argument("shp_path", help="警戒区域シェープファイル(.shp)のパス")
    parser.add_argument("out_pkl", help="出力するpklファイルパス")
    parser.add_argument("--mode", choices=["dem", "sam"], default="dem",
                         help="DEMフィールドに標高そのもの(dem)か傾斜量図(sam)のどちらを入れるか")
    parser.add_argument("--field", default="A33_001", help="フィルタに使う属性フィールド名")
    parser.add_argument("--values", nargs="+", default=["1"], help="対象とする属性値（島根県データではA33_001=1が急傾斜地の崩壊）")
    parser.add_argument("--mesh-missing-threshold", type=float, default=0.5,
                         help="メッシュ全体の欠損割合がこれを超えたらメッシュごと除外（既定0.5）")
    parser.add_argument("--tile-missing-threshold", type=float, default=0.3,
                         help="タイル単位の欠損割合がこれを超えたらそのタイルを除外（既定0.3）")
    parser.add_argument("--apm", action="store_true", help="航空写真(AirPhoto)も取得してデータセットに含める")
    parser.add_argument("--apm-zoom", type=int, default=18, help="航空写真タイルのズームレベル（既定18、Hiroshimaデータと同じ）")
    parser.add_argument("--apm-cache-dir", default="tile_cache", help="航空写真タイルのローカルキャッシュ先フォルダ")
    parser.add_argument("--checkpoint-every", type=int, default=200,
                         help="航空写真取得時、何メッシュごとに途中経過を保存するか（既定200）")
    args = parser.parse_args()

    build_dataset(
        args.tif_dir, args.shp_path, args.field, set(args.values), args.out_pkl,
        mode=args.mode,
        mesh_missing_threshold=args.mesh_missing_threshold,
        tile_missing_threshold=args.tile_missing_threshold,
        with_apm=args.apm,
        apm_zoom=args.apm_zoom,
        apm_cache_dir=args.apm_cache_dir,
        checkpoint_every=args.checkpoint_every,
    )
