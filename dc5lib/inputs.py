# -*- coding: utf-8 -*-
"""モデルへの入力の組み立てを1か所にまとめる。

2入力（傾斜角＋航空写真）と3入力（＋土地利用）を、呼ぶ側が
`cond.n_inputs` を気にせず書けるようにする。推論の経路が
run_inference / run_ensemble_eval / run_overlap_eval / run_overlap_prune /
run_stage2_* と5か所あるので、分岐を各所に書くと必ずどこかが食い違う。
"""

from __future__ import annotations

import numpy as np
import torch

from . import landuse


def needs_landuse(cond):
    """この条件は土地利用（3入力目）を要るか。"""
    for i in getattr(cond, "inputs", []) or []:
        if getattr(i, "source", None) == "LU":
            return True
    return False


def model_args(cond, terrain, air=None, lu_cls=None, device=None):
    """model(*args) に渡す並びを作る。

    terrain  torch [B,1,H,W]（呼ぶ側で device に載せておく）
    air      numpy/torch [B,3,H,W] uint8 か float。0-255 なら /255 する
    lu_cls   numpy [B,H,W] uint8 の区分番号。3入力のときだけ要る
    """
    args = [terrain]
    if getattr(cond, "use_airphoto", False) or air is not None:
        if air is None:
            raise ValueError(f"{getattr(cond, 'name', cond)}: 航空写真が要ります")
        a = air if torch.is_tensor(air) else torch.from_numpy(np.ascontiguousarray(air))
        a = a.to(device) if device is not None else a
        a = a.float()
        if a.dtype == torch.float32 and a.max() > 1.5:
            a = a / 255.0
        args.append(a)
    if needs_landuse(cond):
        if lu_cls is None:
            raise ValueError(
                f"{getattr(cond, 'name', cond)}: 3入力の条件ですが土地利用が"
                f"渡されていません。build_tileset(..., need_landuse=True) か、"
                f"TileSet.landuse を確認してください")
        args.append(landuse.onehot_torch(lu_cls, device=device))
    return tuple(args)
