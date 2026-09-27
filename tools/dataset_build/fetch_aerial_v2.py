"""
国土地理院の地理院タイル（シームレス空中写真）から航空写真を取得するモジュール。
以前Hiroshimaデータ作成に使用した get_airphoto_exact と同じ手法
（該当範囲をカバーするタイルをモザイク結合し、矩形クロップしてからリサイズ）で実装。
GDAL不使用。

タイルURL: https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg

以前との違い:
  - ローカルキャッシュを追加（同じタイルの再ダウンロードを避ける。
    島根県全体では隣接タイル間でのタイル共有が多いため効果が大きい）
  - リサイズをcv2ではなくPillowのBILINEARで行う（cv2.INTER_LINEARと同等）

依存ライブラリ: requests, pillow, numpy
"""

import math
import os
import time

import numpy as np
import requests
from PIL import Image

TILE_URL_TEMPLATE = "https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg"
TILE_PX = 256


def lonlat_to_pixel(lon, lat, zoom):
    """緯度経度を、指定ズームレベルでのグローバルピクセル座標に変換する"""
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = TILE_PX * (2 ** zoom)
    x = scale * (0.5 + lon / 360.0)
    y = scale * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y


def pixel_to_tile(px, py):
    return int(px // TILE_PX), int(py // TILE_PX)


def download_tile(tx, ty, zoom, cache_dir, session=None, retries=3, timeout=15):
    """1枚のタイル画像を取得する（ローカルキャッシュ優先）。
    取得失敗時は黒画像を返す（以前のノートブックの挙動に合わせる）"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{zoom}_{tx}_{ty}.jpg")

    if os.path.exists(cache_path):
        try:
            return np.array(Image.open(cache_path).convert("RGB"))
        except Exception:
            os.remove(cache_path)

    url = TILE_URL_TEMPLATE.format(z=zoom, x=tx, y=ty)
    sess = session or requests
    for attempt in range(retries):
        try:
            r = sess.get(url, timeout=timeout, headers={"User-Agent": "shimane-dataset-builder/1.0"})
            if r.status_code == 200:
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return np.array(Image.open(cache_path).convert("RGB"))
            elif r.status_code == 404:
                return np.zeros((TILE_PX, TILE_PX, 3), dtype=np.uint8)
        except requests.RequestException:
            time.sleep(0.5 * (attempt + 1))
    return np.zeros((TILE_PX, TILE_PX, 3), dtype=np.uint8)


def get_airphoto_exact(geo, cache_dir, dem_size=128, zoom=18, session=None):
    """
    geo: GeoInfoタプル (lon_min, pixel_width, 0, lat_max, 0, -pixel_height)
    以前のHiroshimaデータ作成(get_airphoto_exact)と同じロジック。
    """
    left = geo[0]
    top = geo[3]
    right = left + dem_size * geo[1]
    bottom = top + dem_size * geo[5]

    px1, py1 = lonlat_to_pixel(left, top, zoom)
    px2, py2 = lonlat_to_pixel(right, bottom, zoom)

    tx_min, ty_min = pixel_to_tile(px1, py1)
    tx_max, ty_max = pixel_to_tile(px2, py2)

    mosaic = np.zeros(
        ((ty_max - ty_min + 1) * TILE_PX, (tx_max - tx_min + 1) * TILE_PX, 3),
        dtype=np.uint8,
    )

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = download_tile(tx, ty, zoom, cache_dir, session=session)
            y0 = (ty - ty_min) * TILE_PX
            x0 = (tx - tx_min) * TILE_PX
            mosaic[y0:y0 + TILE_PX, x0:x0 + TILE_PX] = tile

    local_x1 = int(px1 - tx_min * TILE_PX)
    local_y1 = int(py1 - ty_min * TILE_PX)
    local_x2 = int(px2 - tx_min * TILE_PX)
    local_y2 = int(py2 - ty_min * TILE_PX)

    crop = mosaic[local_y1:local_y2, local_x1:local_x2]

    if crop.size == 0:
        # クロップ範囲が不正（極端に小さい等）の場合はゼロ埋めで返す
        return np.zeros((dem_size, dem_size, 3), dtype=np.uint8)

    img = Image.fromarray(crop).resize((dem_size, dem_size), Image.BILINEAR)
    return np.array(img).astype(np.uint8)
