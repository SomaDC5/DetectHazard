# -*- coding: utf-8 -*-
"""和集合アンサンブル。何本足すとどこまで箇所Recall が上がるか。

なぜ和集合なのか（展望_再現率をどう上げるか.md 21・22）
------------------------------------------------------
    cov_g = |g ∩ (予測の和集合)| / |g|

は予測の画素の和集合にしか依存しないので、マスクを OR で重ねると
**箇所Recall は下がりえない。** 刈り取りの天井 2R/(1+R) もそのぶん上がる。
`9` の確率平均アンサンブル（AND 寄り。断片化して失敗）とは逆向きである。

**マスクの OR は max(ロジット) > 0 と同じ**なので、刈り取りに渡す
logit_max が自然に定まる。

やること
--------
指定した条件すべてのオーバーラップ推論を回し、

  ・各条件の 箇所R と 0画素の見逃し
  ・**貪欲に1本ずつ足して**、箇所R がどこまで伸びるか（折り返しを探す）
  ・最後の和集合で刈り取りまで通す（全タイル版と A/B 版）

使い方
------
    python analysis/run_miss_overlap.py
    python analysis/run_miss_overlap.py --conditions A B C
    python analysis/run_miss_overlap.py --region shimane
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, instances                          # noqa: E402
from analysis.run_overlap_prune import infer as ol_infer        # noqa: E402
from dc5lib.device import pick_device                           # noqa: E402
from dc5lib.registry import get                                 # noqa: E402

DEFAULT = ["FinalFusion_SAM_APM_OverlapTrain",
           "FinalFusion_DEM_APM_OverlapTrain",
           "FinalFusion_SAM_APM_LU_OverlapTrain",
           "FinalFusion_SAM_APM_OL_LogitHinge"]
COV_GT = 0.5


def coverage(GT, PD):
    """正解のかたまりごとの被覆率。順番はタイル順・かたまり順で固定。"""
    cov = []
    for g, p in zip(GT, PD):
        for r in instances.per_gt_instance(g.astype(np.uint8), p.astype(np.uint8)):
            cov.append(r["coverage"])
    return np.asarray(cov)


def f_value(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else 0.0


def prune_curve(gts, PD, LG, label, qs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """かたまりごとの logit_max で下位を捨てたときの箇所R/P/F。13.1 と同じ形。"""
    from analysis.run_overlap_prune import blob_table, prune, score
    tb = blob_table(gts, PD, LG)
    n_tp = int(tb.tp.sum())
    print(f"\n--- {label}")
    print(f"  予測したかたまり {len(tb):,}（的中 {n_tp:,} / 外れ "
          f"{len(tb)-n_tp:,} = {100*(len(tb)-n_tp)/len(tb):.1f}%）")
    R0, P0, F0, _ = score(gts, PD)
    print(f"  刈り取り前  箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}"
          f"   完璧な刈り取りの上限 {2*R0/(R0+1):.4f}")
    print(f"  {'捨てる割合':>10s} {'しきい値':>9s} {'箇所R':>8s} {'箇所P':>8s} {'箇所F':>8s}")
    best = (0.0, F0)
    for q in qs:
        if q == 0:
            print(f"  {'0%':>10s} {'-':>9s} {R0:>8.4f} {P0:>8.4f} {F0:>8.4f}")
            continue
        thr = tb["logit_max"].quantile(q)
        p = prune(PD, tb[tb["logit_max"] <= thr])
        R, P, F, _ = score(gts, p)
        print(f"  {f'{100*q:.0f}%':>10s} {thr:>9.3f} {R:>8.4f} {P:>8.4f} {F:>8.4f}")
        if F > best[1]:
            best = (q, F)
    print(f"  ★ 最良 {100*best[0]:.0f}% で 箇所F {best[1]:.4f}")
    return best


def prune_ab(gts, PD, LG, label, qs=(0.1, 0.2, 0.3, 0.4, 0.5), seed=0):
    """A 半分で**捨てる割合もしきい値も**選び、B 半分で測る。

    13.1 ではしきい値だけ A で選んで割合は固定していたが、
    和集合は本数が増えると最良の割合が動く（2本で30%、3本で40%）。
    **割合まで A で選ばないと、本数の比較が不公平になる。**
    """
    from analysis.run_overlap_prune import blob_table, prune, score
    n = len(gts)
    order = np.random.default_rng(seed).permutation(n)
    A = sorted(order[:n // 2].tolist())
    Bi = sorted(order[n // 2:].tolist())
    gA, pA, lA = gts[A], PD[A], LG[A]
    tbA = blob_table(gA, pA, lA)
    bq, bf = None, -1.0
    for q in qs:
        t = tbA["logit_max"].quantile(q)
        _, _, f, _ = score(gA, prune(pA, tbA[tbA["logit_max"] <= t]))
        if f > bf:
            bq, bf = q, f
    q = bq
    thr = tbA["logit_max"].quantile(q)
    gB, pB, lB = gts[Bi], PD[Bi], LG[Bi]
    R0, P0, F0, _ = score(gB, pB)
    tbB = blob_table(gB, pB, lB)
    pr = prune(pB, tbB[tbB["logit_max"] <= thr])
    R, P, F, _ = score(gB, pr)
    print(f"\n--- {label}  A({len(A)}枚)で選び B({len(Bi)}枚)で測る"
          f" / A で選んだ割合 {100*q:.0f}%（A の箇所F {bf:.4f}）")
    print(f"  しきい値 {thr:.3f}")
    print(f"  B 刈る前    箇所F {F0:.4f}")
    print(f"  B 刈った後  箇所R {R:.4f} / P {P:.4f} / F {F:.4f}   正味 {F-F0:+.4f}")
    return F0, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else pick_device()
    names = args.conditions or DEFAULT
    print(f"device = {dev} / 条件 {len(names)}本")

    masks, logits, labels, gts = {}, {}, {}, None
    for name in names:
        cond = get(name)
        labels[name] = cond.label
        print(f"\n■ {cond.label}")
        GT, PD, LG = ol_infer(cond, args.region, dev)
        if gts is None:
            gts = GT
        else:
            assert np.array_equal(gts, GT), f"{name}: 正解マスクが一致しない"
        masks[name], logits[name] = PD, LG

    covs = {n: coverage(gts, masks[n]) for n in names}
    n_gt = len(next(iter(covs.values())))
    print(f"\n正解のかたまり {n_gt:,}")
    print(f"{'条件':44s} {'箇所R':>8s} {'0画素':>8s} {'天井':>8s}")
    for n in names:
        c = covs[n]; R = (c >= COV_GT).mean()
        print(f"{labels[n][:42]:44s} {R:>8.4f} {int((c==0).sum()):>8,} "
              f"{f_value(R,1.0):>8.4f}")

    # 2本ずつの重なり（0画素の見逃し）
    print(f"\n=== 0画素の見逃しの重なり（Jaccard。1.0 なら完全に同じ箇所）===")
    for a, b in itertools.combinations(names, 2):
        za, zb = covs[a] == 0, covs[b] == 0
        print(f"  {a[12:28]:>18s} vs {b[12:28]:<18s} {(za&zb).sum()/max((za|zb).sum(),1):.4f}")

    # 貪欲に1本ずつ足す
    print(f"\n=== 貪欲に足す（毎回いちばん箇所R が上がる1本を選ぶ）===")
    chosen, rest = [names[0]], [n for n in names[1:]]
    U = masks[names[0]].copy(); LU = logits[names[0]].copy()
    R = (covs[names[0]] >= COV_GT).mean()
    print(f"  1本 {labels[names[0]][:36]:38s} 箇所R {R:.4f}  天井 {f_value(R,1.0):.4f}"
          f"  予測画素 {U.sum():,}")
    hist = [(1, names[0], R, f_value(R, 1.0), int(U.sum()))]
    while rest:
        best = None
        for n in rest:
            u = U | masks[n]
            r = (coverage(gts, u) >= COV_GT).mean()
            if best is None or r > best[1]:
                best = (n, r, u)
        n, r, u = best
        U = u; LU = np.maximum(LU, logits[n])
        chosen.append(n); rest.remove(n)
        print(f"  {len(chosen)}本 + {labels[n][:34]:36s} 箇所R {r:.4f}"
              f" ({r-R:+.4f})  天井 {f_value(r,1.0):.4f}  予測画素 {u.sum():,}")
        hist.append((len(chosen), n, r, f_value(r, 1.0), int(u.sum())))
        R = r

    # 和集合の本数ごとに刈り取りまで通す
    print(f"\n{'='*64}\n本数ごとに刈り取りまで通す")
    U2 = masks[chosen[0]].copy(); L2 = logits[chosen[0]].copy()
    for i, n in enumerate(chosen):
        if i > 0:
            U2 = U2 | masks[n]
            L2 = np.maximum(L2, logits[n])
        prune_curve(gts, U2, L2, f"{i+1}本の和集合")
        prune_ab(gts, U2, L2, f"{i+1}本の和集合")

    pd.DataFrame(hist, columns=["本数", "足した条件", "箇所R", "刈り取り天井",
                                "予測画素"]).to_csv(
        os.path.join(config.REPORT_DIR, f"miss_overlap_{args.region}.csv"),
        index=False, encoding="utf-8-sig")
    print(f"\n{os.path.join(config.REPORT_DIR, f'miss_overlap_{args.region}.csv')}")


if __name__ == "__main__":
    main()
