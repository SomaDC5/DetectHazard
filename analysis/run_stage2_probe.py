# -*- coding: utf-8 -*-
"""2段目（かたまり単位の「残す/捨てる」）が成立するかを、学習せずに調べる。

なぜ必要か（展望_再現率をどう上げるか.md 10.5 / 11.5）
------------------------------------------------------
1段目に完璧な刈り取り器を付けられたら箇所F は 0.8260〜0.8793 まで行く。
ただしこれは「かたまりを完璧に選別できたら」の値で、選別できる保証が無い。
しかも 11.5 で、偽陽性の画素は確率が 1.0 に張り付いている（中位数 1.0000）と
分かった。**確率は TP と FP を分けていない。**

ならば何なら分けるのか。それを学習せずに測る。

  ・ロジット（シグモイド前）… 確率は飽和しても、ロジットは飽和しない。
    σ(8) も σ(20) も 1.0 だが、中身の確信度は違う
  ・かたまりの大きさ、形、タイル外周への掛かり方
  ・かたまりの場所の地形（傾斜角の平均・ばらつき）

各特徴について、TP のかたまりと FP のかたまりを分ける AUC を出す。
AUC 0.5 なら情報が無い。0.7 以上あれば2段目は作れる。

そのうえで「その特徴1本で刈ったら箇所F が実際いくらになるか」を、
マスクを刈り直して箇所数評価をやり直して出す。上限との差が分かる。

使い方
------
    python analysis/run_stage2_probe.py
    python analysis/run_stage2_probe.py --condition FinalFusion_DEM_APM_OverlapTrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, data as adata, instances                 # noqa: E402
from analysis.run_ensemble_eval import _region_dir                    # noqa: E402
from analysis.run_instance_eval import SETTINGS                       # noqa: E402
from dc5lib import inputs as minputs  # noqa: E402
from dc5lib.device import pick_device                                 # noqa: E402
from dc5lib.instance_eval import (PRIMARY, label_instances,           # noqa: E402
                                  overlap_matrix)
from dc5lib.models import build_model_for, load_weights               # noqa: E402
from dc5lib.registry import get                                       # noqa: E402

STRUCT4 = ndimage.generate_binary_structure(2, 1)
COV_PRED = 0.3          # PRIMARY（被覆 cov_gt=0.5 / cov_pred=0.3）の予測側のしきい


def predict_logits(cond, tiles, device, batch_size=32):
    """シグモイドを掛けない生の出力。[N, H, W] float32。"""
    model = build_model_for(cond).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()
    out = np.empty((len(tiles.no), config.TILE_SIZE, config.TILE_SIZE),
                   dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(tiles.no), batch_size):
            e = min(s + batch_size, len(tiles.no))
            t = torch.from_numpy(tiles.terrain[s:e]).to(device)
            if cond.use_airphoto:
                logits = model(*minputs.model_args(
                    cond, t, tiles.air[s:e],
                    None if tiles.landuse is None else tiles.landuse[s:e],
                    device=device))
            else:
                logits = model(t)
            out[s:e] = logits.squeeze(1).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def blob_rows(gt, logit, terrain, thr_logit=0.0):
    """かたまりごとに1行つくる。TP/FP の札と特徴量。"""
    rows = []
    H, W = gt.shape[1:]
    # タイル内の各画素の「外周までの距離」
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    edge_dist = np.minimum(np.minimum(rr, cc), np.minimum(H - 1 - rr, W - 1 - cc))

    for k in range(len(gt)):
        pred = logit[k] > thr_logit
        pred_lab, pred_sizes = label_instances(pred, 8, 10)
        n = len(pred_sizes)
        if n == 0:
            continue
        gt_lab, gt_sizes = label_instances(gt[k], 8, 10)
        m = len(gt_sizes)
        if m > 0:
            O = overlap_matrix(gt_lab, m, pred_lab, n)
            cov_p = O.sum(axis=0) / pred_sizes
        else:
            cov_p = np.zeros(n)

        lg = logit[k]
        sl = terrain[k][0] if terrain[k].ndim == 3 else terrain[k]
        objs = ndimage.find_objects(pred_lab)
        for j in range(n):
            sel = pred_lab == (j + 1)
            inner = ndimage.binary_erosion(sel, STRUCT4)
            per = int(sel.sum() - inner.sum())       # 周囲長（境界画素数）
            v = lg[sel]
            s = sl[sel]
            sl_bg = sl[~sel]
            rows.append(dict(
                tile=k, blob=j + 1,
                tp=bool(cov_p[j] >= COV_PRED),
                area=int(pred_sizes[j]),
                logit_mean=float(v.mean()),
                logit_max=float(v.max()),
                logit_p10=float(np.percentile(v, 10)),
                slope_mean=float(s.mean()),
                slope_std=float(s.std()),
                slope_contrast=float(s.mean() - sl_bg.mean()) if sl_bg.size else 0.0,
                edge_min=float(edge_dist[sel].min()),
                compactness=per / max(int(pred_sizes[j]), 1),
                bbox_fill=float(pred_sizes[j] /
                                max((objs[j][0].stop - objs[j][0].start) *
                                    (objs[j][1].stop - objs[j][1].start), 1)),
            ))
    return pd.DataFrame(rows)


def auc(y, x):
    """ランクベースの AUC。y が bool、x が特徴量。"""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(x).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def inst_scores(gt, pred):
    rows = [instances.evaluate_tile_multi(gt[k], pred[k], SETTINGS,
                                          connectivity=8, min_size=10)
            for k in range(len(gt))]
    s = instances.summarize_setting(pd.DataFrame(rows), PRIMARY)
    return (float(s["箇所Recall"]), float(s["箇所Precision"]),
            float(s["箇所F値"]), int(pd.DataFrame(rows).n_pred.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="かたまり単位の CSV を書き出す")
    args = ap.parse_args()

    cond = get(args.condition)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"条件 {cond.name} / 地域 {args.region} / device {device}")

    tiles = adata.build_tileset(str(cond.pkl_for(_region_dir(args.region))),
                                args.region, cond.use_airphoto,
                                cond.bg_ratio, verbose=False,
                                need_landuse=minputs.needs_landuse(cond))
    gt = (tiles.mask[:, 0] > 0)
    print(f"タイル {len(gt)} 枚")
    logit = predict_logits(cond, tiles, device, args.batch_size)
    terrain = tiles.terrain
    print("推論おわり")

    df = blob_rows(gt, logit, terrain)
    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8")
        print("書き出し:", args.out)

    n_tp, n_fp = int(df.tp.sum()), int((~df.tp).sum())
    print(f"\n予測したかたまり {len(df)}（的中 {n_tp} / 外れ {n_fp}"
          f" = 外れが {100 * n_fp / len(df):.1f}%）")

    feats = ["area", "logit_mean", "logit_max", "logit_p10", "slope_mean",
             "slope_std", "slope_contrast", "edge_min", "compactness", "bbox_fill"]
    print("\n--- 的中と外れを分ける力（AUC。0.5 で情報なし）---")
    print("%-18s %7s   %14s %14s" % ("特徴量", "AUC", "的中の中位数", "外れの中位数"))
    scored = []
    for f in feats:
        a = auc(df.tp, df[f])
        scored.append((abs(a - 0.5), f, a))
        print("%-18s %7.4f   %14.4f %14.4f"
              % (f, a, df.loc[df.tp, f].median(), df.loc[~df.tp, f].median()))
    scored.sort(reverse=True)
    print("\n分離に効く順:", ", ".join(f"{f}({a:.3f})" for _, f, a in scored[:4]))

    # いまの値と上限
    R0, P0, F0, n0 = inst_scores(gt, logit > 0.0)
    print(f"\n刈り取り前       箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f} / 予測数 {n0}")
    print(f"完璧な刈り取りの上限  箇所F {2 * R0 / (R0 + 1):.4f}")

    def pruned_mask(rows, tiles_sel=None):
        """rows のかたまりを消したマスク。tiles_sel があればそのタイルだけ。"""
        pred = logit > 0.0
        for k, g in rows.groupby("tile"):
            lab, _ = label_instances(pred[k], 8, 10)
            for b in g.blob:
                pred[k][lab == b] = False
        if tiles_sel is None:
            return pred, gt
        return pred[tiles_sel], gt[tiles_sel]

    def drop_rows(feat, q, hi_is_tp, pool=None):
        """pool（既定は全部）の分位でしきい値を決め、捨てる行を返す。"""
        src = df if pool is None else df[df.tile.isin(pool)]
        thr = src[feat].quantile(q if hi_is_tp else 1 - q)
        mask = (df[feat] <= thr) if hi_is_tp else (df[feat] >= thr)
        return thr, df[mask]

    qs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    CAND = [(f, a > 0.5) for _, f, a in scored[:2]] + [("area", True)]
    seen, cands = set(), []
    for f, h in CAND:
        if f not in seen:
            seen.add(f); cands.append((f, h))

    for feat, hi_is_tp in cands:
        print(f"\n--- {feat} 1本で刈る（{'小さい' if hi_is_tp else '大きい'}側を捨てる）---")
        print("%-12s %9s %8s %8s %8s %9s" % ("捨てる割合", "しきい値", "箇所R",
                                             "箇所P", "箇所F", "予測数"))
        for q in qs:
            thr, drop = drop_rows(feat, q, hi_is_tp)
            p, g = pruned_mask(drop)
            R, P, F, nn = inst_scores(g, p)
            print("%-12s %9.3f %8.4f %8.4f %8.4f %9d"
                  % (f"{100 * q:.0f}%", thr, R, P, F, nn))

    # ---- しきい値を別のタイルで決める（上の表は同じデータで選んでいるので甘い）----
    n_tiles = len(gt)
    A = np.arange(0, n_tiles, 2)          # しきい値を決める側
    B = np.arange(1, n_tiles, 2)          # 測る側
    print(f"\n--- しきい値を別タイルで決めた場合（A {len(A)}枚で選び、B {len(B)}枚で測る）---")
    print("%-14s %9s %8s %8s %8s" % ("特徴量", "選んだ割合", "A の箇所F",
                                     "B の箇所F", "刈る前のB"))
    pB, gB = (logit > 0.0)[B], gt[B]
    _, _, F_before_B, _ = inst_scores(gB, pB)
    for feat, hi_is_tp in cands:
        bestq, bestF = None, -1.0
        for q in qs:
            _, drop = drop_rows(feat, q, hi_is_tp, pool=A)
            p, g = pruned_mask(drop, A)
            _, _, F, _ = inst_scores(g, p)
            if F > bestF:
                bestq, bestF = q, F
        _, drop = drop_rows(feat, bestq, hi_is_tp, pool=A)
        p, g = pruned_mask(drop, B)
        _, _, F_B, _ = inst_scores(g, p)
        print("%-14s %9s %8.4f %8.4f %8.4f"
              % (feat, f"{100 * bestq:.0f}%", bestF, F_B, F_before_B))


if __name__ == "__main__":
    main()
