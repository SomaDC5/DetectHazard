# -*- coding: utf-8 -*-
"""タイルそのものの性質（正解の形・地形の性質・位置）を1行1タイルで書き出す。

モデルとは無関係なので地域ごとに1回作ればよい。推論結果と `tile_no` で
突き合わせることで、「どういうタイルで当たり/外れが起きているか」を
条件横断で調べられるようにする。

傾斜角は SAM のpkl（傾斜角マップ、単位は度）から、起伏量は pkl に入っている
Max_H / Min_H から求める。
"""

import os

import numpy as np
import pandas as pd
from scipy import ndimage

from . import config
from .data import (load_pkl, preprocess_mask, hiroshima_test_indices,
                    background_holdout_indices)

FEATURE_PKL = {
    "hiroshima": "hiroshima_sam.pkl",
    "hiroshima_bg": "hiroshima_sam.pkl",
    "shimane": "shimane_sam.pkl",
}


def _gt_shape_stats(mask, crop):
    """正解マスク1枚の形の特徴。"""
    area = int(mask.sum())
    if area == 0:
        return dict(gt_components=0, gt_max_comp=0, gt_mean_comp=0.0,
                    gt_perimeter=0, gt_compactness=np.nan,
                    gt_border_pix=0, gt_border_ratio=np.nan, gt_edge_touch=0)

    lab, ncomp = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, ncomp + 1))
    eroded = ndimage.binary_erosion(mask.astype(bool), border_value=0)
    perimeter = int(mask.astype(bool).sum() - eroded.sum())

    ring = mask.copy()
    ring[crop:-crop, crop:-crop] = 0
    border_pix = int(ring.sum())

    edge_touch = int(mask[0, :].any() or mask[-1, :].any()
                     or mask[:, 0].any() or mask[:, -1].any())

    return dict(
        gt_components=int(ncomp),
        gt_max_comp=int(sizes.max()),
        gt_mean_comp=float(sizes.mean()),
        gt_perimeter=perimeter,
        gt_compactness=float(perimeter) / float(area),
        gt_border_pix=border_pix,
        gt_border_ratio=border_pix / area,
        gt_edge_touch=edge_touch,
    )


def build_features(region, verbose=True):
    path = os.path.join(config.DATASET_DIR, config.REGION_DIR[region], FEATURE_PKL[region])
    if verbose:
        print(f"[{region}] {os.path.basename(path)} "
              f"({os.path.getsize(path)/1024**3:.1f} GB) を読み込み …", flush=True)
    ds = load_pkl(path)
    n_all = len(ds.No)

    holdout = None
    if region == "hiroshima":
        keep = [i for i in range(n_all) if np.max(np.asarray(ds.Mask[i])) >= 1]
        sel = [keep[i] for i in sorted(hiroshima_test_indices(len(keep)))]
    elif region == "hiroshima_bg":
        sel = [i for i in range(n_all) if np.max(np.asarray(ds.Mask[i])) < 1]
        holdout = set(background_holdout_indices(len(sel)).tolist())
    else:
        sel = list(range(n_all))
    if verbose:
        print(f"[{region}] 対象 {len(sel)} 枚", flush=True)

    crop = config.BORDER_CROP
    rows = []
    for k, i in enumerate(sel):
        slope = np.asarray(ds.DEM[i], dtype=np.float32)
        mask = preprocess_mask(ds.Mask[i])
        geo = ds.GeoInfo[i]
        m = mask.astype(bool)

        row = dict(
            tile_no=ds.No[i],
            region=region,
            lon=float(geo[0]) if geo is not None else np.nan,
            lat=float(geo[3]) if geo is not None else np.nan,
            gt_pos=int(mask.sum()),
            slope_mean=float(slope.mean()),
            slope_std=float(slope.std()),
            slope_p90=float(np.percentile(slope, 90)),
            slope_max=float(slope.max()),
            slope_in_gt=float(slope[m].mean()) if m.any() else np.nan,
            slope_out_gt=float(slope[~m].mean()) if (~m).any() else np.nan,
            max_h=float(ds.Max_H[i]),
            min_h=float(ds.Min_H[i]),
        )
        row["relief"] = row["max_h"] - row["min_h"]
        row["gt_ratio"] = row["gt_pos"] / (config.TILE_SIZE ** 2)
        row["slope_contrast"] = row["slope_in_gt"] - row["slope_out_gt"]
        if holdout is not None:
            row["bg_holdout"] = k in holdout
        row.update(_gt_shape_stats(mask, crop))
        rows.append(row)

        if verbose and k % 2000 == 0:
            print(f"    {k}/{len(sel)}", end="\r", flush=True)

    del ds
    df = pd.DataFrame(rows)
    if verbose:
        print(f"[{region}] 完了 {len(df)} 行            ", flush=True)
    return df


def features_path(region):
    return os.path.join(config.FEATURES_DIR, f"tile_features__{region}.csv")


def load_features(region):
    return pd.read_csv(features_path(region))
