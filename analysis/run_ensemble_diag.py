# -*- coding: utf-8 -*-
"""アンサンブルが「なぜ」効いた／効かなかったかを確率の段階で見る。

run_ensemble_eval.py は結果（F値）を出すが、理由は出ない。
`Ens_OverlapPair2` は箇所F が単体より下がったのに面積は減っていた。
平均後の確率がどこに来ているのかを見ないと説明がつかないので、これを作った。

測るもの
--------
  メンバーごと   抽出面積/正解面積（合計の比）、予測した箇所数
  平均後         確率の帯ごとの画素の割合（0.4〜0.6 に溜まると分断が起きる）
                 メンバーがしきい値 0.5 を挟んで食い違う画素の割合
                 その食い違い画素のうち、平均が 0.4〜0.6 に落ちた割合
                 かたまりの数・平均の大きさ・正解が割れた/予測がまとめた件数

使い方
------
    python analysis/run_ensemble_diag.py --ensemble Ens_OverlapPair2
    python analysis/run_ensemble_diag.py --conditions A B --region hiroshima

「2本だから」なのか「2本とも広く塗るから」なのかを分けるには、
広く塗らない2本でも同じことをして比べる。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, data as adata, instances            # noqa: E402
from analysis.run_ensemble_eval import _region_dir, predict      # noqa: E402
from analysis.run_instance_eval import SETTINGS                  # noqa: E402
from dc5lib import ensembles as ens                              # noqa: E402
from dc5lib.device import pick_device                            # noqa: E402
from dc5lib.instance_eval import PRIMARY                         # noqa: E402
from dc5lib.registry import get                                  # noqa: E402

BANDS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BAND_LABELS = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]


def blob_stats(gt, pred):
    """かたまりの数・大きさ・割れ方。run_instance_eval と同じ設定で数える。"""
    rows = [instances.evaluate_tile_multi(gt[k], pred[k], SETTINGS,
                                          connectivity=8, min_size=10)
            for k in range(len(gt))]
    df = pd.DataFrame(rows)
    s = instances.summarize_setting(df, PRIMARY)
    n_pred = int(df.n_pred.sum())
    return {
        "箇所F": float(s["箇所F値"]),
        "箇所R": float(s["箇所Recall"]),
        "箇所P": float(s["箇所Precision"]),
        "予測箇所数": n_pred,
        "正解箇所数": int(df.n_gt.sum()),
        "1箇所の平均画素": (int(df.pred_pixels.sum()) / n_pred) if n_pred else 0.0,
        "正解が割れたタイル": int((df.split_gt > 0).sum()),
        "予測がまとめたタイル": int((df.merged_pred > 0).sum()),
        "タイル数": len(df),
    }


def area_ratio(gt, pred):
    g = int(gt.sum())
    return (int(pred.sum()) / g) if g else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "hiroshima_bg", "shimane"])
    ap.add_argument("--ensemble", default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prob-thr", type=float, default=None)
    args = ap.parse_args()

    if args.ensemble:
        e = ens.get(args.ensemble)
        names, weights = list(e.members), e.weights_normalized()
        thr = args.prob_thr if args.prob_thr is not None else float(e.threshold)
        label = args.ensemble
    elif args.conditions:
        names = list(args.conditions)
        weights = [1.0 / len(names)] * len(names)
        thr = config.THRESHOLD if args.prob_thr is None else args.prob_thr
        label = "+".join(names)
    else:
        sys.exit("--ensemble か --conditions のどちらかを指定してください")

    device = torch.device(args.device) if args.device else pick_device()
    conds = [get(n) for n in names]
    bgs = {c.bg_ratio for c in conds}
    if len(bgs) > 1:
        sys.exit(f"members の bg_ratio が揃っていません {sorted(bgs)}")

    print(f"{label}")
    print(f"地域 {args.region} / しきい値 {thr} / device {device} / {len(names)}本")

    # pkl の種類ごとに1回だけ読む。タイル番号で突き合わせる
    by_pkl: dict[str, list] = {}
    for c in conds:
        by_pkl.setdefault(str(c.pkl_for(_region_dir(args.region))), []).append(c)

    ref_no = None
    gt = None
    probs: dict[str, np.ndarray] = {}
    for pkl, group in by_pkl.items():
        print(f"\n読み込み: {os.path.basename(pkl)}")
        tiles = adata.build_tileset(pkl, args.region,
                                    any(c.use_airphoto for c in group),
                                    group[0].bg_ratio, verbose=False)
        no = list(tiles.no)
        g = (tiles.mask[:, 0] > 0)
        if ref_no is None:
            ref_no, gt = no, g
            order = np.arange(len(no))
        else:
            if set(no) != set(ref_no):
                sys.exit("pkl 間でタイル集合が違います")
            pos = {v: i for i, v in enumerate(no)}
            order = np.array([pos[v] for v in ref_no])
        for c in group:
            print(f"  推論 {c.name}")
            p = predict(c, tiles, device, args.batch_size)
            probs[c.name] = p[order].astype(np.float32)
        del tiles

    print(f"\nタイル {len(gt)} 枚 / 正解画素 {int(gt.sum()):,}")

    # メンバー単体
    print("\n--- メンバー単体 ---")
    print("%-44s %8s %10s %12s" % ("", "面積比", "予測箇所数", "1箇所の平均画素"))
    for n in names:
        pr = probs[n] > thr
        b = blob_stats(gt, pr)
        print("%-44s %8.3f %10d %12.1f"
              % (n, area_ratio(gt, pr), b["予測箇所数"], b["1箇所の平均画素"]))

    # 平均後
    mix = np.zeros_like(gt, dtype=np.float32)
    for n, w in zip(names, weights):
        mix += w * probs[n]
    pred = mix > thr

    print("\n--- 平均後 ---")
    b = blob_stats(gt, pred)
    print("面積比 %.3f / 予測箇所数 %d（正解 %d）/ 1箇所の平均画素 %.1f"
          % (area_ratio(gt, pred), b["予測箇所数"], b["正解箇所数"], b["1箇所の平均画素"]))
    print("箇所F %.4f / R %.4f / P %.4f" % (b["箇所F"], b["箇所R"], b["箇所P"]))
    print("正解が複数の予測に割れたタイル %d / %d、予測が複数の正解をまとめたタイル %d / %d"
          % (b["正解が割れたタイル"], b["タイル数"],
             b["予測がまとめたタイル"], b["タイル数"]))

    # 確率の帯
    tot = mix.size
    idx = np.digitize(mix, BANDS) - 1
    cnt = np.bincount(idx.ravel(), minlength=len(BAND_LABELS))
    print("\n平均後の確率の分布（全画素 %,d のうち）" .replace(",d", "d") % tot)
    for lab, c in zip(BAND_LABELS, cnt):
        print("  %-8s %12d  %6.3f%%" % (lab, c, 100 * c / tot))

    # メンバーがしきい値を挟んで食い違う画素
    stack = np.stack([probs[n] > thr for n in names])
    k = stack.sum(axis=0)
    disagree = (k > 0) & (k < len(names))
    nd = int(disagree.sum())
    print("\nメンバーが 0.5 を挟んで食い違う画素 %d（全体の %.3f%%）"
          % (nd, 100 * nd / tot))
    if nd:
        mid = ((mix >= 0.4) & (mix <= 0.6) & disagree).sum()
        print("  そのうち平均が 0.4〜0.6 に落ちたもの %d（%.1f%%）"
              % (int(mid), 100 * mid / nd))
        print("  食い違い画素が最終的に陽性になった割合 %.1f%%"
              % (100 * float((pred & disagree).sum()) / nd))
    agree_pos = int(((k == len(names))).sum())
    print("  全員が陽性の画素 %d（全体の %.3f%%）" % (agree_pos, 100 * agree_pos / tot))


if __name__ == "__main__":
    main()
