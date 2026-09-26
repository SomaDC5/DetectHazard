# -*- coding: utf-8 -*-
"""保存済みの重みでテスト集合を推論し、**1行 = 1タイル**の集計を書き出す。

ModelComparison がノートブックの出力から条件全体のスコアを拾うのに対して、
こちらは実際に推論をやり直してタイル単位に分解する。全条件を合計すれば
ModelComparison の Recall / Precision / F1 と一致するはずで、
`scripts/run_inference.py --verify` でその照合ができる。
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from . import config
from dc5lib.models import build_model, load_weights


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


TILE_COLUMNS = [
    "tile_no", "tp", "fp", "fn", "tn",
    "tp_c", "fp_c", "fn_c", "tn_c",
    "gt_pos", "pred_pos", "gt_pos_c", "pred_pos_c",
    "prob_mean", "prob_pos_mean", "prob_max",
]


@torch.no_grad()
def run_condition(cond, tiles, device, batch_size=32, progress_every=20):
    """1条件ぶんの推論。タイルごとの混同行列を DataFrame で返す。"""
    model = build_model(cond.model_class).to(device)
    meta = load_weights(model, cond.weights, device=device)
    model.eval()

    crop = config.BORDER_CROP
    n = len(tiles)
    out = {c: np.zeros(n, dtype=np.float64) for c in TILE_COLUMNS}
    out["tile_no"] = np.asarray(tiles.no)

    t0 = time.time()
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        terrain = torch.from_numpy(tiles.terrain[s:e]).to(device)
        target = torch.from_numpy(tiles.mask[s:e]).to(device).float() > 0.5

        if cond.use_airphoto:
            air = torch.from_numpy(tiles.air[s:e]).to(device).float() / 255.0
            logits = model(terrain, air)
        else:
            logits = model(terrain)

        prob = torch.sigmoid(logits)
        pred = prob > config.THRESHOLD

        def counts(p, t, key_suffix=""):
            tp = (p & t).flatten(1).sum(1)
            fp = (p & ~t).flatten(1).sum(1)
            fn = (~p & t).flatten(1).sum(1)
            tn = (~p & ~t).flatten(1).sum(1)
            out["tp" + key_suffix][s:e] = tp.cpu().numpy()
            out["fp" + key_suffix][s:e] = fp.cpu().numpy()
            out["fn" + key_suffix][s:e] = fn.cpu().numpy()
            out["tn" + key_suffix][s:e] = tn.cpu().numpy()
            return tp, fp, fn

        counts(pred, target)
        counts(pred[:, :, crop:-crop, crop:-crop],
               target[:, :, crop:-crop, crop:-crop], "_c")

        out["gt_pos"][s:e] = target.flatten(1).sum(1).cpu().numpy()
        out["pred_pos"][s:e] = pred.flatten(1).sum(1).cpu().numpy()
        out["gt_pos_c"][s:e] = target[:, :, crop:-crop, crop:-crop].flatten(1).sum(1).cpu().numpy()
        out["pred_pos_c"][s:e] = pred[:, :, crop:-crop, crop:-crop].flatten(1).sum(1).cpu().numpy()

        out["prob_mean"][s:e] = prob.flatten(1).mean(1).cpu().numpy()
        out["prob_max"][s:e] = prob.flatten(1).max(1).values.cpu().numpy()
        gt_sum = target.flatten(1).sum(1).clamp(min=1)
        out["prob_pos_mean"][s:e] = ((prob * target).flatten(1).sum(1) / gt_sum).cpu().numpy()

        if progress_every and (s // batch_size) % progress_every == 0:
            done = e
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"    {done:6d}/{n}  {rate:5.1f} tiles/s  残り {eta/60:4.1f} 分",
                  end="\r", flush=True)

    _sync(device)
    el = time.time() - t0
    print(f"    {n:6d}/{n}  完了 {el/60:.1f} 分 ({n/el:.1f} tiles/s)        ", flush=True)

    df = pd.DataFrame(out)
    for c in TILE_COLUMNS:
        if c not in ("prob_mean", "prob_pos_mean", "prob_max", "tile_no"):
            df[c] = df[c].astype(np.int64)
    for name, values in tiles.flags.items():
        df[name] = values
    df.insert(0, "condition", cond.name)
    df.insert(1, "region", tiles.region)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return df, meta


def tile_path(cond_name, region):
    return os.path.join(config.TILES_DIR, f"{cond_name}__{region}.csv")


def totals(df):
    """タイル集計を足し上げて、条件全体の Recall / Precision / F1 を出す。"""
    res = {}
    for suf, tag in (("", "通常"), ("_c", "境界")):
        tp = df["tp" + suf].sum()
        fp = df["fp" + suf].sum()
        fn = df["fn" + suf].sum()
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * rec * pre / (rec + pre) if rec + pre else 0.0
        res[tag] = {"recall": rec, "precision": pre, "f1": f1,
                    "TP": int(tp), "FP": int(fp), "FN": int(fn)}
    return res
