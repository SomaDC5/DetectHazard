"""
基盤地図情報 数値標高モデル（DEM5A等）のGMLファイルをGeoTIFFに変換するスクリプト。

【フォーマットの仕様（国土地理院 基盤地図情報 製品仕様書に基づく）】
- gml:lowerCorner / gml:upperCorner : メッシュがカバーする緯度経度範囲(JGD2011)
    lowerCorner = (南西端の緯度, 南西端の経度)
    upperCorner = (北東端の緯度, 北東端の経度)
- gml:GridEnvelope low="0 0" high="224 149" : グリッドは 225列 × 150行（5mメッシュ固定）
    low  = 北西端のグリッド番号（常に 0 0）
    high = 南東端のグリッド番号（5mメッシュは常に 224 149）
- gml:sequenceRule order="+x-y" : データの並び順
    +x : x方向（列）は西→東に正の向き
    -y : y方向（行）は北→南に正の向き（＝配列としては上から下）
    → つまりデータは「北西端から開始し、東へ向かって1行分埋めたら
       次の行（南側）へ進む」通常のラスタ画像と同じ並び（row-major, 北が上）
- gml:startPoint : 実データの開始グリッド番号。先頭が欠損している場合は (0,0) 以外になる。
    startPointより前のセルにはデータが存在しないので nodata で埋める。
- gml:tupleList : 1行1セルで "種別,標高値" の形式。種別が「データなし」等の場合や
    値が "-9999." の場合は欠損値。

このスクリプトは、フォルダ内の全GMLファイルを走査し、同名のGeoTIFF
（EPSG:6668 = JGD2011 経緯度）を出力する。
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import from_origin

NS = {
    "gml": "http://www.opengis.net/gml/3.2",
    "fgd": "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema",
}

NODATA = -9999.0


def _find_text(root, tag):
    """名前空間を気にせず最初にマッチした要素のテキストを返す"""
    for elem in root.iter():
        if elem.tag.endswith(tag):
            return elem.text
    return None


def parse_dem_gml(path):
    """1つのGMLファイルを解析し、(elevation_array, transform, crs, mesh_code) を返す"""
    tree = ET.parse(path)
    root = tree.getroot()

    # --- 緯度経度範囲 ---
    lower = _find_text(root, "lowerCorner").split()
    upper = _find_text(root, "upperCorner").split()
    lat_min, lon_min = float(lower[0]), float(lower[1])
    lat_max, lon_max = float(upper[0]), float(upper[1])

    # --- グリッドサイズ ---
    low = _find_text(root, "low").split()
    high = _find_text(root, "high").split()
    x_high, y_high = int(high[0]), int(high[1])
    n_cols = x_high + 1  # 通常225
    n_rows = y_high + 1  # 通常150

    # --- 開始点 ---
    start_text = _find_text(root, "startPoint")
    if start_text:
        start_x, start_y = map(int, start_text.split())
    else:
        start_x, start_y = 0, 0
    start_index = start_y * n_cols + start_x

    # --- 標高値の抽出 ---
    tuple_text = _find_text(root, "tupleList")
    lines = [ln.strip() for ln in tuple_text.strip().splitlines() if ln.strip()]

    values = np.full(n_cols * n_rows, NODATA, dtype=np.float32)
    for i, line in enumerate(lines):
        idx = start_index + i
        if idx >= values.size:
            break
        # "地表面,23.16" のような "種別,標高値" 形式
        parts = line.split(",")
        val_str = parts[-1]
        try:
            val = float(val_str)
        except ValueError:
            val = NODATA
        values[idx] = val

    elevation = values.reshape((n_rows, n_cols))  # row-major, 北が上（1行目=北端）

    # --- ジオリファレンス ---
    # ピクセルサイズ = 範囲 / セル数
    px_w = (lon_max - lon_min) / n_cols
    px_h = (lat_max - lat_min) / n_rows
    # 原点は北西端（upper-left） = (lon_min, lat_max)
    transform = from_origin(lon_min, lat_max, px_w, px_h)

    mesh_code_match = re.search(r"(\d{6,8})", os.path.basename(path))
    mesh_code = mesh_code_match.group(1) if mesh_code_match else os.path.splitext(os.path.basename(path))[0]

    return elevation, transform, mesh_code


def convert_file(gml_path, out_dir):
    elevation, transform, mesh_code = parse_dem_gml(gml_path)
    out_path = os.path.join(out_dir, f"{mesh_code}.tif")

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=elevation.shape[0],
        width=elevation.shape[1],
        count=1,
        dtype=elevation.dtype,
        crs="EPSG:6668",  # JGD2011 (経緯度)
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(elevation, 1)

    return out_path


def convert_folder(in_dir, out_dir, pattern="*.xml"):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "**", pattern), recursive=True))
    if not files:
        print(f"警告: {in_dir} 内に {pattern} に一致するファイルが見つかりません")
        return []

    outputs = []
    n_missing = 0
    for f in files:
        try:
            elevation, transform, mesh_code = parse_dem_gml(f)
        except Exception as e:
            print(f"スキップ（解析エラー）: {f} -> {e}")
            continue

        # 欠損値（-9999.）が全体の一定割合以上を占める場合は欠損メッシュとして記録
        missing_ratio = np.mean(elevation == NODATA)
        if missing_ratio > 0.99:
            n_missing += 1

        out_path = convert_file(f, out_dir)
        outputs.append(out_path)

    print(f"変換完了: {len(outputs)} 枚 (うち欠損の疑いが強いメッシュ: {n_missing} 枚)")
    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="基盤地図情報DEM(GML)→GeoTIFF変換")
    parser.add_argument("in_dir", help="ダウンロードしたGML(XML)ファイルが入っているフォルダ")
    parser.add_argument("out_dir", help="GeoTIFFの出力先フォルダ")
    parser.add_argument("--pattern", default="*.xml", help="入力ファイルのパターン（既定: *.xml）")
    args = parser.parse_args()

    convert_folder(args.in_dir, args.out_dir, args.pattern)
