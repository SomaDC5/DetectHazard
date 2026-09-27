"""
生成したdem_dataset pklから、警戒区域を含むタイルをいくつかピックアップして
DEMとMaskを画像として保存し、目視確認できるようにするスクリプト。

依存ライブラリ: numpy, matplotlib のみ。
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np


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


def visualize(pkl_path, out_path, n_samples=6, only_with_hazard=True):
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)

    n = len(ds.No)
    print(f"タイル総数: {n}")

    if only_with_hazard:
        candidates = [i for i in range(n) if ds.Mask[i].sum() > 0]
        print(f"警戒区域を含むタイル数: {len(candidates)}")
    else:
        candidates = list(range(n))

    picks = candidates[:n_samples]
    if not picks:
        print("表示できるタイルがありません。")
        return

    fig, axes = plt.subplots(len(picks), 2, figsize=(8, 4 * len(picks)))
    if len(picks) == 1:
        axes = axes.reshape(1, 2)

    for row, idx in enumerate(picks):
        dem = ds.DEM[idx]
        mask = ds.Mask[idx]
        geo = ds.GeoInfo[idx]

        dem_disp = np.where(dem == -9999.0, np.nan, dem)

        ax1 = axes[row, 0]
        im1 = ax1.imshow(dem_disp, cmap="terrain")
        ax1.set_title(f"タイル{idx} DEM\nmin={ds.Min_H[idx]:.1f} max={ds.Max_H[idx]:.1f}")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = axes[row, 1]
        ax2.imshow(mask, cmap="gray")
        ax2.set_title(f"タイル{idx} Mask\nGeoInfo原点=({geo[0]:.4f}, {geo[3]:.4f})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dem_dataset pklの目視確認用可視化")
    parser.add_argument("pkl_path", help="生成したpklファイルのパス")
    parser.add_argument("--out", default="check_tiles.png", help="出力する画像ファイル名")
    parser.add_argument("--n", type=int, default=6, help="表示するタイル数")
    parser.add_argument("--all", action="store_true", help="警戒区域を含むタイルに限定せず先頭から表示")
    args = parser.parse_args()

    visualize(args.pkl_path, args.out, n_samples=args.n, only_with_hazard=not args.all)
