"""
産総研（地質調査総合センター）「20万分の1日本シームレス地質図V2」タイルを取得するモジュール。
以前のAirPhoto取得(fetch_aerial_v2.py, get_airphoto_exact)と同じ
「タイルをモザイク結合→矩形クロップ→バイリニアリサイズ」方式で実装。GDAL不使用。

【地理院タイル・地理院シームレス空中写真タイルとの違いに注意】
  - タイルURLのパス順序が {z}/{y}/{x} で、xとyの順序が逆になっている
    （通常のXYZタイル規約=地理院タイルや航空写真タイルは {z}/{x}/{y}）
  - ズームレベルは 0〜13 までしか提供されていない（航空写真は18まであった）。
    そのため、地質図は航空写真より解像度が粗い。128x128タイル（実質約500m四方）に
    対して、zoom=13でも1タイルあたり約4.9m/pxなので、詳細な模様は出ないが、
    地質区分（大まかな塗り分け）としては問題ない解像度。
  - タイルが存在しない場所は透明PNG（RGBA、alpha=0）を返す（エラーにはならない）。

利用規約: 政府標準利用規約(第2.0版)。出典表記のみで利用可能。
https://gbank.gsj.jp/seamless/v2/api/1.2/

依存ライブラリ: requests, pillow, numpy
"""

import math
import os
import time

import numpy as np
import requests
from PIL import Image

# {z}/{y}/{x}の順序に注意（AirPhotoの{z}/{x}/{y}と逆）
TILE_URL_TEMPLATE = "https://gbank.gsj.jp/seamless/v2/api/1.2/tiles/{z}/{y}/{x}.png"
TILE_PX = 256
MAX_ZOOM = 13  # このサービスが提供する最大ズームレベル


def lonlat_to_pixel(lon, lat, zoom):
    """緯度経度を、指定ズームレベルでのグローバルピクセル座標に変換する（地理院タイル/GoogleMaps方式）"""
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = TILE_PX * (2 ** zoom)
    x = scale * (0.5 + lon / 360.0)
    y = scale * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))
    return x, y


def pixel_to_tile(px, py):
    return int(px // TILE_PX), int(py // TILE_PX)


def download_tile(tx, ty, zoom, cache_dir, layer="g", session=None, retries=3, timeout=15):
    """1枚のタイル画像を取得する（ローカルキャッシュ優先）。
    取得失敗・タイル未提供時は透明(alpha=0)として扱い、白背景のRGBに変換して返す"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"geo_{layer}_{zoom}_{tx}_{ty}.png")

    if os.path.exists(cache_path):
        try:
            return _to_rgb_white_bg(Image.open(cache_path))
        except Exception:
            os.remove(cache_path)

    url = TILE_URL_TEMPLATE.format(z=zoom, y=ty, x=tx)  # x,yの順序に注意
    params = {"layer": layer} if layer else None
    sess = session or requests
    for attempt in range(retries):
        try:
            r = sess.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "shimane-dataset-builder/1.0"})
            if r.status_code == 200:
                with open(cache_path, "wb") as f:
                    f.write(r.content)
                return _to_rgb_white_bg(Image.open(cache_path))
        except requests.RequestException:
            time.sleep(0.5 * (attempt + 1))
    # 取得できなかった場合は白背景の空タイルを返す（データなし域として扱う）
    return np.full((TILE_PX, TILE_PX, 3), 255, dtype=np.uint8)


def _to_rgb_white_bg(img):
    """透明PNGを白背景のRGBに変換する（透明=データなしを白として扱う）"""
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composited = Image.alpha_composite(bg, img).convert("RGB")
    return np.array(composited)


def get_geology_exact(geo, cache_dir, dem_size=128, zoom=MAX_ZOOM, layer="g", session=None):
    """
    geo: GeoInfoタプル (lon_min, pixel_width, 0, lat_max, 0, -pixel_height)
    AirPhotoのget_airphoto_exactと同じロジック。zoomは既定で最大値(13)を使う。
    """
    zoom = min(zoom, MAX_ZOOM)

    left = geo[0]
    top = geo[3]
    right = left + dem_size * geo[1]
    bottom = top + dem_size * geo[5]

    px1, py1 = lonlat_to_pixel(left, top, zoom)
    px2, py2 = lonlat_to_pixel(right, bottom, zoom)

    tx_min, ty_min = pixel_to_tile(px1, py1)
    tx_max, ty_max = pixel_to_tile(px2, py2)

    mosaic = np.full(
        ((ty_max - ty_min + 1) * TILE_PX, (tx_max - tx_min + 1) * TILE_PX, 3),
        255, dtype=np.uint8,
    )

    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = download_tile(tx, ty, zoom, cache_dir, layer=layer, session=session)
            y0 = (ty - ty_min) * TILE_PX
            x0 = (tx - tx_min) * TILE_PX
            mosaic[y0:y0 + TILE_PX, x0:x0 + TILE_PX] = tile

    local_x1 = int(px1 - tx_min * TILE_PX)
    local_y1 = int(py1 - ty_min * TILE_PX)
    local_x2 = int(px2 - tx_min * TILE_PX)
    local_y2 = int(py2 - ty_min * TILE_PX)

    crop = mosaic[local_y1:local_y2, local_x1:local_x2]

    if crop.size == 0:
        return np.full((dem_size, dem_size, 3), 255, dtype=np.uint8)

    img = Image.fromarray(crop).resize((dem_size, dem_size), Image.BILINEAR)
    return np.array(img).astype(np.uint8)
