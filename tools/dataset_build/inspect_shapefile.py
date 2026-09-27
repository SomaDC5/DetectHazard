"""
国土数値情報からダウンロードしたシェープファイル（A33_*.shp）の中身を確認するスクリプト。
「現象の種類」「区域区分」フィールドは実際には数値コードであり、テキスト属性は
Shift-JIS(CP932)でエンコードされているため、それに対応する。

GDAL不使用（pyshpのみ）。
"""

import sys
from collections import Counter

import shapefile  # pip install pyshp


def inspect(shp_path):
    # 国土数値情報のシェープファイルは通常Shift-JIS(CP932)でエンコードされている
    sf = shapefile.Reader(shp_path, encoding="cp932")

    print("=== フィールド一覧 ===")
    for f in sf.fields[1:]:  # 先頭はDeletionFlagなのでスキップ
        print(f"  名前: {f[0]:<12} 型: {f[1]}  長さ: {f[2]}")

    print(f"\n=== レコード件数: {len(sf)} ===")

    print("\n=== 先頭3件の属性値 ===")
    for i, rec in enumerate(sf.iterRecords()):
        if i >= 3:
            break
        print(f"--- レコード{i} ---")
        print(dict(rec.as_dict()))

    print("\n=== A33_001（現象の種類）の値ごとの件数 ===")
    counter_001 = Counter()
    counter_002 = Counter()
    for rec in sf.iterRecords():
        d = rec.as_dict()
        counter_001[d.get("A33_001")] += 1
        counter_002[d.get("A33_002")] += 1

    total = sum(counter_001.values())
    for val, cnt in counter_001.most_common():
        print(f"  値={val!r:>6}  件数={cnt:>7}  割合={cnt/total*100:5.1f}%")

    print("\n=== A33_002（区域区分）の値ごとの件数 ===")
    for val, cnt in counter_002.most_common():
        print(f"  値={val!r:>6}  件数={cnt:>7}  割合={cnt/total*100:5.1f}%")

    print("\n=== 図形（先頭1件）の範囲・座標数 ===")
    shape = sf.shape(0)
    print(f"  bbox: {shape.bbox}")
    print(f"  points数: {len(shape.points)}")
    print(f"  parts: {shape.parts}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python inspect_shapefile.py <シェープファイルの.shpパス（拡張子省略可）>")
        sys.exit(1)
    inspect(sys.argv[1])
