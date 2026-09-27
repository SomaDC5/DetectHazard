"""
実際のGeoTIFFファイルの欠損値（NoData）がどう表現されているかを調査する診断スクリプト。

確認すること:
  1. GDAL_NODATAタグ(42113)が存在するか、あればその値
  2. 標高値のヒストグラム（特定の値に不自然なスパイクがないか）
  3. 0.0や負の極端な値がどれくらいの割合を占めるか
"""

import sys
from collections import Counter

import numpy as np
import tifffile


def diagnose(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        arr = page.asarray().astype(np.float32)
        tags = page.tags

        print("=== 全タグ一覧 ===")
        for t in tags:
            # tupleListのような長い値は省略表示
            val = t.value
            if isinstance(val, (tuple, list)) and len(val) > 10:
                val = f"(長さ{len(val)}の配列)"
            print(f"  {t.code:6d} {t.name:<25} {val}")

        print(f"\n=== 配列情報 ===")
        print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        print(f"  最小値: {arr.min()}")
        print(f"  最大値: {arr.max()}")
        print(f"  平均値: {arr.mean():.3f}")

        print(f"\n=== 値の分布（頻出値トップ15） ===")
        # 浮動小数点なので丸めてカウント
        rounded = np.round(arr, 1)
        counter = Counter(rounded.ravel().tolist())
        total = arr.size
        for val, cnt in counter.most_common(15):
            print(f"  値={val:>10}  件数={cnt:>8}  割合={cnt/total*100:5.2f}%")

        print(f"\n=== 特定の値のチェック ===")
        for candidate in [0.0, -9999.0, -9999.0 * 100, np.nan]:
            if np.isnan(candidate):
                cnt = np.isnan(arr).sum()
                label = "NaN"
            else:
                cnt = (arr == candidate).sum()
                label = str(candidate)
            print(f"  値={label:>10} の件数: {cnt} ({cnt/total*100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python diagnose_tiff.py <怪しい線が写っていたタイルの元になったTIFFファイルパス>")
        sys.exit(1)
    diagnose(sys.argv[1])
