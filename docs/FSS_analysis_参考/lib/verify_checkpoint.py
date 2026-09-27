# -*- coding: utf-8 -*-
"""
新しい実験を追加したとき、「実際に学習で使われた正規化方法」を
自動的に絞り込むための診断ツール。

背景: ノートブックのコード上では正規化の有無・方法が複数通り考えられる
ことがあり、実際にどれが使われたかはコードを読むだけでは分からない
場合があった（DEM単一・SAM単一・DEM＋APM・SAM＋APMの4条件すべてで、
それぞれ異なる正規化が使われていたことが判明している）。

使い方:
    from lib.verify_checkpoint import try_normalizations
    try_normalizations(
        model, dem_test_raw, air_test, y_test,
        max_h_train=..., min_h_train=..., max_h_test=..., min_h_test=...,
        target_f1=ckpt_best_f1_score,
    )
候補をすべて試して Recall/Precision/F1 を表示するので、
チェックポイントの best_f1_score と一致するものを採用する。
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from .data_utils import normalize_terrain
from .inference import run_inference_single, run_inference_dual


def _eval_preds(preds, gt):
    p = preds.astype(int).flatten()
    t = gt.astype(int).flatten()
    return dict(
        recall=recall_score(t, p, zero_division=0),
        precision=precision_score(t, p, zero_division=0),
        f1=f1_score(t, p, zero_division=0),
    )


def try_normalizations(model, dem_raw, gt, device="cpu", air_raw_0to255=None,
                        max_h_train=None, min_h_train=None, max_h_test=None, min_h_test=None,
                        target_f1=None, batch_size=16):
    """
    dem_raw: (N,H,W) 生の地形量配列
    gt: (N,H,W) 0/1マスク
    air_raw_0to255: dual入力モデルの場合のみ (N,H,W,3) を渡す。Noneなら単一入力として扱う。
    target_f1: チェックポイントの best_f1_score。渡すと各候補との差分も表示する。

    候補: raw / global_minmax / per_tile_minmax（地形量のみ）に加え、
    dualの場合は air を 0-1正規化 / 生値(0-255) / ImageNet正規化 の3通りも試す。
    """
    candidates = [("raw", {})]
    if max_h_train is not None and min_h_train is not None:
        candidates.append(("global_minmax", dict(
            max_h_train=max_h_train, min_h_train=min_h_train,
            max_h_all=max_h_test, min_h_all=min_h_test)))
    if max_h_test is not None and min_h_test is not None:
        candidates.append(("per_tile_minmax", dict(max_h_all=max_h_test, min_h_all=min_h_test)))

    air_variants = [("air/255", None)]
    if air_raw_0to255 is not None:
        air_variants = [
            ("air/255", lambda a: torch.tensor(np.ascontiguousarray(
                np.transpose(a, (0, 3, 1, 2)) / 255.0).astype(np.float32))),
            ("air raw(0-255)", lambda a: torch.tensor(np.ascontiguousarray(
                np.transpose(a, (0, 3, 1, 2))).astype(np.float32))),
        ]

    results = []
    for norm_name, kwargs in candidates:
        dem_norm, _ = normalize_terrain(dem_raw, norm_name, **kwargs)
        dem_t = torch.tensor(np.ascontiguousarray(np.expand_dims(dem_norm, 1)).astype(np.float32))

        if air_raw_0to255 is None:
            preds, _ = run_inference_single(model, dem_t, batch_size=batch_size, device=device)
            m = _eval_preds(preds, gt)
            results.append((f"dem={norm_name}", m))
        else:
            for air_name, air_fn in air_variants:
                air_t = air_fn(air_raw_0to255)
                preds, _, _ = run_inference_dual(model, dem_t, air_t, batch_size=batch_size, device=device)
                m = _eval_preds(preds, gt)
                results.append((f"dem={norm_name}, {air_name}", m))

    print(f"{'candidate':40s} {'Recall':>8s} {'Precision':>10s} {'F1':>8s}" +
          ("  (target F1={:.4f})".format(target_f1) if target_f1 else ""))
    for name, m in sorted(results, key=lambda x: -x[1]["f1"]):
        mark = ""
        if target_f1 is not None and abs(m["f1"] - target_f1) < 0.001:
            mark = "  <-- MATCH"
        print(f"{name:40s} {m['recall']:8.4f} {m['precision']:10.4f} {m['f1']:8.4f}{mark}")
    return results
