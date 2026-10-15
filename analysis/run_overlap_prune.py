# -*- coding: utf-8 -*-
"""オーバーラップ推論にロジットを残して、かたまり単位の刈り取りを掛ける。

なぜ必要か（展望_再現率をどう上げるか.md 12.3 / 12.6）
------------------------------------------------------
12.3 で、ロジットの大小でかたまりを刈ると箇所F が +0.047 動くと分かった。
ただしあれは**従来推論**での測定である。1段目の最良構成は
「オーバーラップ学習 + オーバーラップ推論」（箇所R 0.7846 / 上限 0.8793）だが、
**オーバーラップ推論は確率を cos 窓で混ぜるのでロジットが残らない。**
最良構成に刈り取りが同じだけ効くかが未確認だった。

やること
--------
9枚の窓を貼るところまでは run_overlap_eval.py と同じにして、
**確率とロジットの両方**を cos 窓で混ぜる。

  ・二値化は**確率側**で行う（従来どおり）。
    こうすると出発点が記録済みの値と完全に一致するので、
    刈り取りの効果だけを切り出して測れる
  ・混ぜたロジットは、かたまりごとの特徴としてだけ使う

しきい値は**別のタイル**で選んで、残りのタイルで測る（12.3 と同じ作法）。

使い方
------
    python analysis/run_overlap_prune.py --condition FinalFusion_SAM_APM_OverlapTrain
    python analysis/run_overlap_prune.py --condition ... --region shimane
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import instances                                       # noqa: E402
from analysis.data import load_pkl                                   # noqa: E402
from analysis.run_instance_eval import SETTINGS                      # noqa: E402
from analysis.run_overlap_eval import (OFFSETS, REGION_DIR,          # noqa: E402
                                       cos_window, grid_index, mosaic,
                                       prep_mask, target_tiles)
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import (PRIMARY, label_instances,          # noqa: E402
                                  overlap_matrix)
from dc5lib.models import build_model_for, load_weights              # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402
from dc5lib.registry import get                                      # noqa: E402

STRUCT4 = ndimage.generate_binary_structure(2, 1)
COV_PRED = 0.3
QS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


@torch.no_grad()
def infer(cond, region, device, verbose=True):
    """タイルごとに (正解, 二値マスク, 混ぜたロジット) を返す。

    二値化は確率側で行う（run_overlap_eval.py と同じ）。
    """
    model = build_model_for(cond).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()

    dirname = REGION_DIR[region]
    pkl = dataset_path(dirname, cond.pkl_for(dirname))
    if verbose:
        print(f"  読み込み: {os.path.basename(pkl)} ...", flush=True)
    ds = load_pkl(pkl)
    gi, gj, cell = grid_index(ds)
    tiles = target_tiles(ds, region, float(getattr(cond, "bg_ratio", 0.0) or 0.0))
    W = cos_window()
    if verbose:
        print(f"  {len(tiles)} タイル", flush=True)

    GT = np.zeros((len(tiles), 128, 128), bool)
    PB = np.zeros_like(GT)                       # 確率で二値化したもの
    LG = np.zeros((len(tiles), 128, 128), np.float32)
    agree = 0
    for n, k in enumerate(tiles):
        D, A, _ = mosaic(ds, cell, gi, gj, k)
        dems, airs = [], []
        for dy, dx in OFFSETS:
            r, c = 128 + dy, 128 + dx
            dems.append(D[r:r + 128, c:c + 128])
            airs.append(A[r:r + 128, c:c + 128])
        dem = torch.from_numpy(np.stack(dems)[:, None]).float().to(device)
        air = (torch.from_numpy(np.stack(airs).transpose(0, 3, 1, 2))
               .float().to(device) / 255.0)
        logit = model(dem, air)[:, 0]
        pr = torch.sigmoid(logit).cpu().numpy()
        lg = logit.cpu().numpy()

        wp = np.zeros((128, 128), np.float32)
        wl = np.zeros((128, 128), np.float32)
        ws = np.zeros((128, 128), np.float32)
        for (dy, dx), p, l in zip(OFFSETS, pr, lg):
            ys, ye = max(0, dy), min(128, 128 + dy)
            xs, xe = max(0, dx), min(128, 128 + dx)
            w = W[ys - dy:ye - dy, xs - dx:xe - dx]
            wp[ys:ye, xs:xe] += p[ys - dy:ye - dy, xs - dx:xe - dx] * w
            wl[ys:ye, xs:xe] += l[ys - dy:ye - dy, xs - dx:xe - dx] * w
            ws[ys:ye, xs:xe] += w
        ws = np.maximum(ws, 1e-6)
        prob = wp / ws
        GT[n] = prep_mask(ds, k)
        PB[n] = prob > 0.5
        LG[n] = wl / ws
        # 参考：ロジットを混ぜてから二値化しても同じ判定になるか
        agree += int((PB[n] == (LG[n] > 0.0)).sum())
        if verbose and n % 500 == 0:
            print(f"    {n}/{len(tiles)}", end="\r", flush=True)
    if verbose:
        print(f"  確率で混ぜた判定とロジットで混ぜた判定の一致 "
              f"{100 * agree / GT.size:.3f}%            ")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GT, PB, LG


def blob_table(GT, PB, LG):
    """かたまりごとに 的中/外れ の札と、混ぜたロジットの統計量。"""
    rows = []
    for k in range(len(GT)):
        lab, sizes = label_instances(PB[k], 8, 10)
        n = len(sizes)
        if n == 0:
            continue
        gl, gs = label_instances(GT[k], 8, 10)
        if len(gs):
            O = overlap_matrix(gl, len(gs), lab, n)
            cov_p = O.sum(axis=0) / sizes
        else:
            cov_p = np.zeros(n)
        lg = LG[k]
        for j in range(n):
            sel = lab == (j + 1)
            inner = ndimage.binary_erosion(sel, STRUCT4)
            v = lg[sel]
            rows.append(dict(tile=k, blob=j + 1,
                             tp=bool(cov_p[j] >= COV_PRED),
                             area=int(sizes[j]),
                             logit_max=float(v.max()),
                             logit_mean=float(v.mean()),
                             compactness=float(sel.sum() - inner.sum()) /
                             max(int(sizes[j]), 1)))
    return pd.DataFrame(rows)


def score(GT, PB, sel=None):
    idx = range(len(GT)) if sel is None else sel
    rows = [instances.evaluate_tile_multi(GT[k], PB[k], SETTINGS,
                                          connectivity=8, min_size=10)
            for k in idx]
    df = pd.DataFrame(rows)
    s = instances.summarize_setting(df, PRIMARY)
    return (float(s["箇所Recall"]), float(s["箇所Precision"]),
            float(s["箇所F値"]), int(df.n_pred.sum()))


def prune(PB, drop):
    """drop に入っているかたまりを消したマスクを返す。"""
    out = PB.copy()
    for k, g in drop.groupby("tile"):
        lab, _ = label_instances(PB[k], 8, 10)
        for b in g.blob:
            out[k][lab == b] = False
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--feature", default="logit_max",
                    choices=["logit_max", "logit_mean", "area"])
    ap.add_argument("--thr", type=float, default=None,
                    help="他の地域で決めたしきい値をそのまま当てる（地域をまたぐ確認）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cond = get(args.condition)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"条件 {cond.name} / 地域 {args.region} / device {device}")
    print("オーバーラップ推論（9窓・cos窓）。二値化は確率側、刈り取りはロジット")

    GT, PB, LG = infer(cond, args.region, device)
    df = blob_table(GT, PB, LG)
    n_tp = int(df.tp.sum())
    print(f"\n予測したかたまり {len(df)}（的中 {n_tp} / 外れ {len(df) - n_tp}"
          f" = 外れが {100 * (len(df) - n_tp) / len(df):.1f}%）")

    R0, P0, F0, n0 = score(GT, PB)
    print(f"刈り取り前   箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f} / 予測数 {n0}")
    print(f"完璧な刈り取りの上限  箇所F {2 * R0 / (R0 + 1):.4f}")

    f = args.feature
    print(f"\n--- {f} で下位を刈る（全タイルでしきい値を決めた場合＝甘い）---")
    print("%-12s %9s %8s %8s %8s %9s" % ("捨てる割合", "しきい値", "箇所R",
                                         "箇所P", "箇所F", "予測数"))
    for q in QS:
        if q == 0:
            print("%-12s %9s %8.4f %8.4f %8.4f %9d"
                  % ("0%", "-", R0, P0, F0, n0))
            continue
        thr = df[f].quantile(q)
        p = prune(PB, df[df[f] <= thr])
        R, P, F, nn = score(GT, p)
        print("%-12s %9.3f %8.4f %8.4f %8.4f %9d"
              % (f"{100 * q:.0f}%", thr, R, P, F, nn))

    if args.thr is not None:
        drop = df[df[f] <= args.thr]
        R, P, F, nn = score(GT, prune(PB, drop))
        print(f"\n--- 他の地域で決めたしきい値 {args.thr:.3f} をそのまま当てる ---")
        print(f"  捨てたかたまり {len(drop)} / {len(df)}"
              f"（{100 * len(drop) / len(df):.1f}%）")
        print(f"  刈る前   箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}")
        print(f"  刈った後 箇所R {R:.4f} / P {P:.4f} / F {F:.4f} / 予測数 {nn}")
        print(f"  正味 {F - F0:+.4f}")
        return

    # しきい値は別のタイルで決める
    A = np.arange(0, len(GT), 2)
    B = np.arange(1, len(GT), 2)
    _, _, FB0, _ = score(GT, PB, B)
    bestq, bestF = 0.0, -1.0
    for q in QS:
        if q == 0:
            _, _, F, _ = score(GT, PB, A)
        else:
            thr = df[df.tile.isin(A)][f].quantile(q)
            F = score(GT, prune(PB, df[df[f] <= thr]), A)[2]
        if F > bestF:
            bestq, bestF = q, F
    if bestq == 0:
        print(f"\nA 側では刈らないのが最良だった（箇所F {bestF:.4f}）")
        return
    thr = df[df.tile.isin(A)][f].quantile(bestq)
    RB, PBr, FB, nB = score(GT, prune(PB, df[df[f] <= thr]), B)
    print(f"\n--- しきい値を別タイルで決めた場合（A {len(A)}枚で選び B {len(B)}枚で測る）---")
    print(f"  選ばれた割合 {100 * bestq:.0f}%（しきい値 {thr:.3f}）/ A の箇所F {bestF:.4f}")
    print(f"  B 刈る前   箇所F {FB0:.4f}")
    print(f"  B 刈った後 箇所R {RB:.4f} / P {PBr:.4f} / F {FB:.4f} / 予測数 {nB}")
    print(f"  正味 {FB - FB0:+.4f}")


if __name__ == "__main__":
    main()
