# -*- coding: utf-8 -*-
"""pkl の読み込みと、学習時と同じ前処理・同じテスト分割の再現。

学習ノートブックでの扱いをそのまま再現している。

- 広島：`max(Mask) >= 1` のタイルだけを元の順序で残し（14,816枚）、
  `train_test_split(..., test_size=0.2, random_state=42)` のテスト側を使う。
  デュアル入力条件も単一入力条件も同じ引数で分割しているので、
  **テストタイルは全条件で共通**。
- 島根：背景タイルも含めて全件（24,569枚）を使う。
- マスク：`uint8 * 255` → `gaussian_filter(sigma=1)` → `> 0`。
- 航空写真：uint8 のまま保持し、モデル入力の直前に `/255.0`。
- 地形量：pkl の値をそのまま使う（DEM条件は正規化済みpklを読む）。
"""

import gc
import os
import random
import pickle

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

from . import config
from dc5lib import regions as _regions


# ------------------------------------------------------------------ pkl 読み込み
class _Bag:
    """pkl 内のクラス名が何であっても同じ属性名で受け取るための器。"""

    def __init__(self):
        for k in ("No", "DEM", "Mask", "GeoInfo", "EPSG", "Max_H", "Min_H", "AirPhoto"):
            setattr(self, k, [])


class _RedirectUnpickler(pickle.Unpickler):
    """pklを作ったときのクラス定義（ノートブック内、load_any_dataset.py 等）が
    手元に無くても読めるようにする。numpy などの復元は素通しする。"""

    def find_class(self, module, name):
        if module == "__main__":
            return _Bag
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # データセットを入れていたユーザ定義クラスだけがここに来る
            return _Bag


def load_pkl(path):
    """NewDatasModel/DataSet 以下の pkl を読み込む（クラス名は問わない）。"""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            f.seek(0)
            return _RedirectUnpickler(f).load()


# ------------------------------------------------------------------ 前処理
def preprocess_mask(mask):
    """学習時と同じマスク前処理。"""
    m = np.asarray(mask).astype(np.uint8) * 255
    m = gaussian_filter(m, sigma=config.GAUSSIAN_SIGMA)
    return (m > 0).astype(np.uint8)


def hiroshima_test_indices(n_mask_tiles, bg_ratio=0.0, n_bg_tiles=0):
    """広島のホールドアウト（テスト側）のインデックス。

    学習ノートブックと同じ `train_test_split` を、インデックス配列に対して
    呼び直すことで再現する。

    **bg_ratio > 0 のときは、背景タイルを混ぜたあとで分割しないとずれる。**
    ノートブックは「警戒ありタイル + 選ばれた背景タイル」を1本の配列にしてから
    train_test_split を呼ぶ。train_test_split の結果は要素数に依存するので、
    警戒ありタイルだけで分割すると**まったく別の分割**になり、
    学習に使ったタイルがテスト側に混ざる。
    bg_ratio=0.1 では、警戒ありテスト2,964枚のうち1,746枚（58.9%）が
    ノートブックでは学習に使われていた。

    戻り値は (警戒ありタイル側のテスト位置, 背景タイル側のテスト位置)。
    bg_ratio=0 のときは後者が空になる。
    """
    if bg_ratio <= 0:
        _, test_idx = train_test_split(
            np.arange(n_mask_tiles), test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE)
        return test_idx, np.array([], dtype=int)

    # ノートブック（セル7）と同じ背景タイルの選び方を再現する
    #   評価用ホールドアウトの外側からだけ選び、seed 42 でシャッフルして先頭から取る
    pool, _ = train_test_split(np.arange(n_bg_tiles), test_size=config.TEST_SIZE,
                               random_state=config.RANDOM_STATE)
    order = list(pool)
    random.seed(config.RANDOM_STATE)
    random.shuffle(order)
    sel_bg = order[:int(n_bg_tiles * bg_ratio)]

    # 警戒あり + 選ばれた背景、の順に並べてから分割（ノートブックと同じ）
    n_comb = n_mask_tiles + len(sel_bg)
    _, test_pos = train_test_split(np.arange(n_comb), test_size=config.TEST_SIZE,
                                   random_state=config.RANDOM_STATE)
    hz_test = np.array(sorted(i for i in test_pos if i < n_mask_tiles), dtype=int)
    bg_test = np.array(sorted(sel_bg[i - n_mask_tiles] for i in test_pos
                              if i >= n_mask_tiles), dtype=int)
    return hz_test, bg_test


def background_holdout_indices(n_bg_tiles):
    """背景タイル（警戒区域を1画素も含まないタイル）側の「テスト相当」。

    背景タイルは学習に一切使われていない（bg_ratio=0）ので、本来はすべてが
    未学習データになる。ただしそのまま全部足すと、テスト集合の背景比率が
    地域の実際の構成比より高くなってしまう。

    そこで、警戒区域ありのタイルと**同じ 20 % のホールドアウト率**を背景側にも
    当てはめる。こうすると「最初から全38,148枚を 8:2 に分けていたら得られた
    テスト集合」と同じ構成になり、島根（地域内の全タイル）と比較できる。
    """
    idx = np.arange(n_bg_tiles)
    _, test_idx = train_test_split(
        idx, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    return test_idx


# ------------------------------------------------------------------ タイル集合
class TileSet:
    """評価に使うタイルの集合。

    terrain : float32 [N, 1, 128, 128]  地形量（pklの値そのまま）
    air     : uint8   [N, 3, 128, 128]  航空写真（なければ None）
    mask    : uint8   [N, 1, 128, 128]  前処理済みの正解
    no      : list    タイル番号
    geo     : list    GeoInfo（左上経度・画素サイズなど）
    max_h / min_h : list  標高の最大・最小（起伏量の算出に使う）
    """

    def __init__(self, terrain, air, mask, no, geo, max_h, min_h, region, source,
                 flags=None):
        self.terrain = terrain
        self.air = air
        self.mask = mask
        self.no = no
        self.geo = geo
        self.max_h = max_h
        self.min_h = min_h
        self.region = region
        self.source = source
        # タイルごとの付加情報（推論結果CSVにそのまま列として書き出す）
        self.flags = flags or {}

    def __len__(self):
        return len(self.no)


def build_tileset(pkl_path, region, need_airphoto, bg_ratio=0.0, verbose=True):
    """pkl を1回読んで、その地域の評価用タイル集合を作る。

    bg_ratio はノートブックの分割を再現するために要る（hiroshima_test_indices
    の説明を参照）。bg_ratio > 0 の条件では、テスト集合に背景タイルも含まれる。
    """
    if verbose:
        size_gb = os.path.getsize(pkl_path) / 1024 ** 3
        print(f"  読み込み: {os.path.basename(pkl_path)} ({size_gb:.1f} GB) …", flush=True)
    ds = load_pkl(pkl_path)
    n_all = len(ds.No)

    flags = {}
    subset = _regions.get(region).subset if region in _regions.keys() else "all"

    if subset == "holdout":
        # 警戒区域を含むタイルだけを元の順序で残し、学習時と同じ分割のテスト側を取る
        keep = [i for i in range(n_all) if np.max(np.asarray(ds.Mask[i])) >= 1]
        bgs = [i for i in range(n_all) if np.max(np.asarray(ds.Mask[i])) < 1]
        hz_pos, bg_pos = hiroshima_test_indices(len(keep), bg_ratio, len(bgs))
        sel = [keep[i] for i in hz_pos] + [bgs[i] for i in bg_pos]
        flags["has_gt"] = np.array([True] * len(hz_pos) + [False] * len(bg_pos), dtype=bool)
        if verbose:
            extra = f" + 背景 {len(bg_pos)} 枚" if len(bg_pos) else ""
            print(f"  全 {n_all} 枚 → 警戒区域あり {len(keep)} 枚 → "
                  f"テスト {len(hz_pos)} 枚{extra}（bg_ratio={bg_ratio}）", flush=True)
    elif subset == "background":
        # 警戒区域を1画素も含まないタイル。学習には一切使われていない。
        sel = [i for i in range(n_all) if np.max(np.asarray(ds.Mask[i])) < 1]
        hold = set(background_holdout_indices(len(sel)).tolist())
        flags["bg_holdout"] = np.array([k in hold for k in range(len(sel))], dtype=bool)
        if verbose:
            print(f"  全 {n_all} 枚 → 背景タイル {len(sel)} 枚 "
                  f"（うちホールドアウト相当 {int(flags['bg_holdout'].sum())} 枚）", flush=True)
    else:                                   # subset == "all"
        sel = list(range(n_all))
        if verbose:
            print(f"  全 {n_all} 枚をそのまま使用", flush=True)

    n = len(sel)
    terrain = np.empty((n, 1, config.TILE_SIZE, config.TILE_SIZE), dtype=np.float32)
    mask = np.empty((n, 1, config.TILE_SIZE, config.TILE_SIZE), dtype=np.uint8)
    air = (np.empty((n, 3, config.TILE_SIZE, config.TILE_SIZE), dtype=np.uint8)
           if need_airphoto else None)
    no, geo, max_h, min_h = [], [], [], []

    for k, i in enumerate(sel):
        terrain[k, 0] = np.asarray(ds.DEM[i], dtype=np.float32)
        mask[k, 0] = preprocess_mask(ds.Mask[i])
        if need_airphoto:
            air[k] = np.transpose(np.asarray(ds.AirPhoto[i], dtype=np.uint8), (2, 0, 1))
        no.append(ds.No[i])
        geo.append(ds.GeoInfo[i])
        max_h.append(ds.Max_H[i])
        min_h.append(ds.Min_H[i])

    del ds
    gc.collect()
    return TileSet(terrain, air, mask, no, geo, max_h, min_h, region,
                   os.path.basename(pkl_path), flags=flags)
