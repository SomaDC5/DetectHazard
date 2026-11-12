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


def third_input(cond):
    """3入力目のうち、**土地利用ではないもの**の `Input` を返す。

    土地利用は pkl に区分番号1chで入っていて読み込み時に one-hot に
    展開するという特別な事情があるので `needs_landuse` で別に扱う。
    建物（`BVB`）のように pkl の値をそのまま渡すものはこちらを通る。
    """
    ins = list(getattr(cond, "inputs", []) or [])
    for i in ins[2:]:
        if getattr(i, "source", None) != "LU":
            return i
    return None


def model_args(cond, terrain, air=None, lu_cls=None, device=None, third=None):
    """model(*args) に渡す並びを作る。

    terrain  torch [B,1,H,W]（呼ぶ側で device に載せておく）
    air      numpy/torch [B,3,H,W] uint8 か float。0-255 なら /255 する
    lu_cls   numpy [B,H,W] uint8 の区分番号。土地利用の3入力のときだけ要る
    third    numpy/torch [B,C,H,W] 土地利用以外の3入力目（建物など）。
             0-255 なら /255 する。**航空写真と同じ扱い**
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
    t = third_input(cond)
    if t is not None:
        if third is None:
            raise ValueError(
                f"{getattr(cond, 'name', cond)}: 3入力目（{t.source} / "
                f"pkl の {t.field}）が渡されていません。"
                f"build_tileset(..., third_field='{t.field}') か、"
                f"TileSet.third を確認してください")
        x = (third if torch.is_tensor(third)
             else torch.from_numpy(np.ascontiguousarray(third)))
        x = (x.to(device) if device is not None else x).float()
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[1] != t.channels:
            raise ValueError(
                f"{getattr(cond, 'name', cond)}: 3入力目の面が "
                f"{x.shape[1]} ですが config は {t.channels} です")
        args.append(x)
    return tuple(args)
