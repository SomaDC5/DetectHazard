# -*- coding: utf-8 -*-
"""条件ごとに出力確率の分布を測る。学習は不要（推論のやり直しだけ）。

なぜ必要か（展望_再現率をどう上げるか.md 11 / 展望.md 8）
----------------------------------------------------------
損失の alpha を 0.7→0.85 と上げると面積比が +0.134 動くのに、
0.7→0.5 と下げても −0.017 しか動かない（約8倍の非対称）。
仮説は「出力が飽和しているので、消すべき偽陽性も既にほぼ1に振り切れており、
過検出を重く罰しても勾配がほとんど無い」というもの。

これを確かめるには**偽陽性の画素がどの確率帯にいるか**を見ればよい。
飽和していれば 0.98 超えに集まり、していなければ中間帯に散らばる。

出すもの
--------
  全画素・正解ありの画素・正解なしの画素のそれぞれで
    確率 <0.02 / (0.02,0.98) / >0.98 の割合
  偽陽性（予測は陽性・正解は陰性）の画素の確率分布
  見逃し（予測は陰性・正解は陽性）の画素の確率分布

使い方
------
    python analysis/run_prob_hist.py --conditions A B C
    python analysis/run_prob_hist.py --conditions A B C --region shimane
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, data as adata                       # noqa: E402
from analysis.run_ensemble_eval import _region_dir, predict      # noqa: E402
from dc5lib.device import pick_device                            # noqa: E402
from dc5lib.registry import get                                  # noqa: E402

LO, HI = 0.02, 0.98
BANDS = [0.0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98, 1.0001]


def shares(p):
    """<0.02 / 中間 / >0.98 の割合。"""
    n = p.size
    if n == 0:
        return 0.0, 0.0, 0.0
    lo = float((p < LO).sum()) / n
    hi = float((p > HI).sum()) / n
    return lo, 1.0 - lo - hi, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "hiroshima_bg", "shimane"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prob-thr", type=float, default=None)
    args = ap.parse_args()

    thr = config.THRESHOLD if args.prob_thr is None else args.prob_thr
    device = torch.device(args.device) if args.device else pick_device()
    conds = [get(n) for n in args.conditions]
    bgs = {c.bg_ratio for c in conds}
    if len(bgs) > 1:
        sys.exit(f"bg_ratio が揃っていません {sorted(bgs)}。分割がずれて比べられません")

    print(f"地域 {args.region} / しきい値 {thr} / device {device}")

    by_pkl: dict[str, list] = {}
    for c in conds:
        by_pkl.setdefault(str(c.pkl_for(_region_dir(args.region))), []).append(c)

    ref_no, gt = None, None
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
            probs[c.name] = predict(c, tiles, device, args.batch_size)[order].astype(np.float32)
        del tiles

    print(f"\nタイル {len(gt)} 枚 / 全画素 {gt.size:,} / 正解画素 {int(gt.sum()):,}")

    print("\n--- 確率の帯ごとの割合 ---")
    print("%-40s %10s %12s %10s" % ("", "<0.02", "(0.02,0.98)", ">0.98"))
    for n in args.conditions:
        lo, mid, hi = shares(probs[n])
        print("%-40s %9.3f%% %11.3f%% %9.3f%%" % (n, 100 * lo, 100 * mid, 100 * hi))

    print("\n--- 正解なしの画素だけ（偽陽性の素になる場所） ---")
    print("%-40s %10s %12s %10s" % ("", "<0.02", "(0.02,0.98)", ">0.98"))
    for n in args.conditions:
        lo, mid, hi = shares(probs[n][~gt])
        print("%-40s %9.3f%% %11.3f%% %9.3f%%" % (n, 100 * lo, 100 * mid, 100 * hi))

    print("\n--- 偽陽性の画素（予測は陽性・正解は陰性）がどの帯にいるか ---")
    print("%-40s %12s %10s %12s %10s"
          % ("", "偽陽性画素数", "うち>0.98", "(0.5,0.98)", "中位数"))
    for n in args.conditions:
        p = probs[n]
        fp = p[(p > thr) & (~gt)]
        if fp.size == 0:
            print("%-40s %12d" % (n, 0)); continue
        print("%-40s %12d %9.1f%% %11.1f%% %10.4f"
              % (n, fp.size, 100 * float((fp > HI).sum()) / fp.size,
                 100 * float(((fp > thr) & (fp <= HI)).sum()) / fp.size,
                 float(np.median(fp))))

    print("\n--- 見逃しの画素（予測は陰性・正解は陽性）がどの帯にいるか ---")
    print("%-40s %12s %10s %12s %10s"
          % ("", "見逃し画素数", "うち<0.02", "(0.02,0.5)", "中位数"))
    for n in args.conditions:
        p = probs[n]
        fn = p[(p <= thr) & gt]
        if fn.size == 0:
            print("%-40s %12d" % (n, 0)); continue
        print("%-40s %12d %9.1f%% %11.1f%% %10.4f"
              % (n, fn.size, 100 * float((fn < LO).sum()) / fn.size,
                 100 * float(((fn >= LO) & (fn <= thr)).sum()) / fn.size,
                 float(np.median(fn))))

    print("\n--- 細かい帯（全画素に対する割合） ---")
    head = "".join("%11s" % f"{BANDS[i]:g}-{min(BANDS[i+1], 1.0):g}"
                   for i in range(len(BANDS) - 1))
    print("%-40s%s" % ("", head))
    for n in args.conditions:
        idx = np.digitize(probs[n], BANDS) - 1
        cnt = np.bincount(idx.ravel(), minlength=len(BANDS) - 1)
        row = "".join("%10.3f%%" % (100 * c / probs[n].size) for c in cnt)
        print("%-40s%s" % (n, row))


if __name__ == "__main__":
    main()
