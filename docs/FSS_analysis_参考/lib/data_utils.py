# -*- coding: utf-8 -*-
"""
FSSデータセット（pickle形式）の読み込み・前処理の共通処理。

各実験ノートブックがそれぞれ独自にpickleを読み込んで前処理していたが、
中身は共通のパターン（マスクありタイルの抽出→ガウシアンフィルタ→
train_test_split(test_size=0.2, random_state=42)）なので、ここに集約した。

【正規化について・重要】
ノートブックのコードだけでは、実際に学習で使われた正規化方法が
一意に決まらない場合がある（同じ前処理コードが複数の実験フォルダに
コピーされ、一部だけ後から書き換えられているため）。
本ライブラリで確認済みの組み合わせは NORMALIZE_PRESETS の通りだが、
新しい実験を追加する際は、必ず lib/verify_checkpoint.py で
best_model.pth の best_f1_score と一致するか検証してから使うこと。
"""
import pickle
import sys
import types
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter


def _ensure_pickle_compat_classes():
    """
    データセットのpickleは、作成時に __main__.dem_dataset / __main__.photo_dataset
    というクラス名で保存されている。ライブラリ経由で読み込むと __main__ が
    run_analysis.py 等に変わってしまい unpickle できないため、
    読み込み前に同名のダミークラスを __main__ に登録しておく。
    （属性の中身は pickle 側の __dict__ でそのまま復元されるため、
    ここでは空のクラスで構わない）
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")
        sys.modules["__main__"] = main_mod
    for name in ("dem_dataset", "photo_dataset"):
        if not hasattr(main_mod, name):
            setattr(main_mod, name, type(name, (), {}))


class SingleInputDataset:
    """DEM単一 / SAM単一で使う dem_dataset 相当のコンテナ。"""

    def __init__(self):
        self.No = []
        self.DEM = []
        self.Mask = []
        self.GeoInfo = []
        self.EPSG = []
        self.Max_H = []
        self.Min_H = []


class DualInputDataset:
    """DEM＋APM / SAM＋APM で使う photo_dataset 相当のコンテナ。"""

    def __init__(self):
        self.No = []
        self.DEM = []
        self.Mask = []
        self.GeoInfo = []
        self.EPSG = []
        self.Max_H = []
        self.Min_H = []
        self.AirPhoto = []


def load_pickle(path):
    _ensure_pickle_compat_classes()
    with open(path, "rb") as f:
        return pickle.load(f)


def _apply_gaussian_mask(mask, sigma=1):
    m = np.array(mask).astype(np.uint8) * 255
    m = gaussian_filter(m, sigma=sigma)
    return (m > 0).astype(np.uint8)


def split_with_mask(raw_dataset, has_airphoto):
    """
    生のpickleオブジェクトを「マスクあり(dataset2)」「マスクなし(dataset3)」に分割し、
    マスクにガウシアンフィルタを適用する。has_airphoto=True なら DualInputDataset、
    False なら SingleInputDataset を使う。
    """
    cls = DualInputDataset if has_airphoto else SingleInputDataset
    with_mask, without_mask = cls(), cls()
    n = len(raw_dataset.No)
    for i in range(n):
        tgt = with_mask if np.max(np.array(raw_dataset.Mask[i])) >= 1 else without_mask
        tgt.DEM.append(raw_dataset.DEM[i])
        tgt.Mask.append(raw_dataset.Mask[i])
        tgt.No.append(raw_dataset.No[i])
        tgt.GeoInfo.append(raw_dataset.GeoInfo[i])
        tgt.EPSG.append(raw_dataset.EPSG[i])
        tgt.Max_H.append(raw_dataset.Max_H[i])
        tgt.Min_H.append(raw_dataset.Min_H[i])
        if has_airphoto:
            tgt.AirPhoto.append(raw_dataset.AirPhoto[i])

    for i in range(len(with_mask.No)):
        with_mask.Mask[i] = _apply_gaussian_mask(with_mask.Mask[i])

    return with_mask, without_mask


def train_test_indices(n, test_size=0.2, random_state=42):
    """
    学習時と全く同じ分割を再現するための添字を返す。
    元のノートブックは train_test_split に配列そのものを渡しているが、
    分割結果は「サンプル数nとrandom_state」だけで決まるため、
    np.arange(n) を分割すればどの実験でも同じ添字が得られる。
    """
    idx = np.arange(n)
    return train_test_split(idx, test_size=test_size, random_state=random_state)


NORMALIZE_PRESETS = {
    # 検証により、各条件の best_model.pth が実際に学習された入力スケールと
    # 判明したもの。理由は README.md の「前処理まとめ」を参照。
    "DemOnly": "raw",             # 生の標高値(m)
    "KeisyaOnly": "global_minmax",  # 学習集合全体でのmin-max正規化
    "DEM_AirPhoto": "raw",        # 生の標高値(m)
    "SAM_AirPhoto": "raw",        # 生の傾斜値(度)
}


def normalize_terrain(values, mode, max_h_train=None, min_h_train=None,
                       max_h_all=None, min_h_all=None):
    """
    values: (N, H, W) の地形量配列（DEM または SAM）
    mode:
      "raw"            そのまま使う（変換なし）
      "global_minmax"  学習集合全体の Max_H / Min_H で min-max 正規化
                        （テスト側がその範囲を超える場合は範囲を広げる。
                         元ノートブックの挙動をそのまま再現している）
      "per_tile_minmax" タイルごとの Max_H / Min_H で正規化（参考用。
                        今回確認された4条件のどれにも該当しなかった）
    """
    if mode == "raw":
        return values.copy(), None

    if mode == "global_minmax":
        xh = float(np.max(max_h_train))
        nh = float(np.min(min_h_train))
        if max_h_all is not None and xh < float(np.max(max_h_all)):
            xh = float(np.max(max_h_all))
        if min_h_all is not None and nh > float(np.min(min_h_all)):
            nh = float(np.min(min_h_all))
        out = (values - nh) / (xh - nh)
        return out, (xh, nh)

    if mode == "per_tile_minmax":
        out = np.empty_like(values, dtype=np.float32)
        for i in range(values.shape[0]):
            lo, hi = float(min_h_all[i]), float(max_h_all[i])
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            out[i] = (values[i] - lo) / denom
        return out, None

    raise ValueError(f"unknown normalize mode: {mode}")


def stack_test_subset(dataset, idx_test, has_airphoto):
    """
    テスト添字だけを numpy 配列にスタックする（train分もまとめて配列化すると
    メモリを食うため、推論・分析にはテスト分だけ使えば十分）。
    """
    dem = np.stack([np.asarray(dataset.DEM[i], dtype=np.float32) for i in idx_test])
    mask = np.stack([np.asarray(dataset.Mask[i], dtype=np.float32) for i in idx_test])
    no = np.array([dataset.No[i] for i in idx_test])
    air = None
    if has_airphoto:
        air = np.stack([np.asarray(dataset.AirPhoto[i], dtype=np.float32) for i in idx_test])
    return dem, mask, air, no


def to_input_tensor_single(dem_norm):
    """(N,H,W) -> (N,1,H,W) の contiguous float32 tensor。"""
    arr = np.ascontiguousarray(np.expand_dims(dem_norm, axis=1)).astype(np.float32)
    return torch.tensor(arr)


def to_input_tensor_dual(dem_norm, air_raw_0to255):
    """
    dem_norm: (N,H,W) 正規化済み地形量
    air_raw_0to255: (N,H,W,3) 0-255スケールのRGB
    戻り値: dem_tensor (N,1,H,W), air_tensor (N,3,H,W) いずれも contiguous
    """
    dem_t = torch.tensor(np.ascontiguousarray(np.expand_dims(dem_norm, axis=1)).astype(np.float32))
    air_t = torch.tensor(
        np.ascontiguousarray(np.transpose(air_raw_0to255, (0, 3, 1, 2)) / 255.0).astype(np.float32)
    )
    return dem_t, air_t


def to_mask_tensor(mask):
    return torch.tensor(np.expand_dims(mask, axis=1).astype(np.float32))
