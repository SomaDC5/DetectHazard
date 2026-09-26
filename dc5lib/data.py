# -*- coding: utf-8 -*-
"""pkl の読み込みと、学習時と同じ前処理・同じ分割の再現。

もともと `C:\\Users\\hirok\\OneDrive\\...\\MakeDataSet\\load_any_dataset.py` に
置いてあったものを、リポジトリ内に取り込んだ。
研究室PC固有のパスに依存しなくなるので、Ubuntu でも Mac でも同じコードが動く。

    from dc5lib.data import load_dataset
    ds = load_dataset(dataset_path("Hiroshima", "hiroshima_sam_apm.pkl"))
"""

from __future__ import annotations

import pickle

import numpy as np

GAUSSIAN_SIGMA = 1.0
TILE_SIZE = 128
BORDER_CROP = 16
TEST_SIZE = 0.2
RANDOM_STATE = 42


class dem_dataset:
    """pkl に入っているデータセットの器（単一入力版）。

    pkl 内のクラス名が何であっても、この形で受け取れるようにしてある。
    """

    def __init__(self):
        self.No, self.DEM, self.Mask = [], [], []
        self.GeoInfo, self.EPSG, self.Max_H, self.Min_H = [], [], [], []


class photo_dataset(dem_dataset):
    """航空写真つき（デュアル入力版）。"""

    def __init__(self):
        super().__init__()
        self.AirPhoto = []


class _Bag:
    """クラス定義が手元に無い pkl を受け取るための器。属性は復元時に生える。"""


class _RedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            return _Bag
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # データセットを入れていたユーザ定義クラスだけがここに来る
            return _Bag


def load_dataset(path):
    """クラス名を問わず pkl を読む。"""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            return _RedirectUnpickler(f).load()


def preprocess_mask(mask):
    """学習時と同じマスク前処理。uint8*255 → ガウシアン(σ=1) → >0。"""
    from scipy.ndimage import gaussian_filter
    m = np.asarray(mask).astype(np.uint8) * 255
    m = gaussian_filter(m, sigma=GAUSSIAN_SIGMA)
    return (m > 0).astype(np.uint8)


def holdout_indices(n, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """学習時と同じ train_test_split のテスト側インデックスを再現する。"""
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(np.arange(n), test_size=test_size,
                                   random_state=random_state)
    return test_idx
