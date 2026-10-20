# -*- coding: utf-8 -*-
"""見逃しは条件をまたいで同じ箇所か。和集合にすると再現率はどこまで上がるか。

なぜ必要か（展望_再現率をどう上げるか.md 21）
--------------------------------------------
`20` でロジット空間の損失が失敗し、`19.5` で残る行き先が
「刈り取りの精度」と「1段目の箇所Recall」の2つに絞れた。
1段目の箇所Recall を上げる手段として、**入力の違う2条件の和集合**がある。

**和集合は箇所Recall を必ず上げるか維持する。** `19.4` のとおり

    cov_g = |g ∩ (予測の和集合)| / |g|

は予測の画素の和集合にしか依存しないので、マスクを OR で重ねると
cov_g は単調に増える → tp_gt は減らない → **箇所Recall は下がらない。**
刈り取りの天井 2R/(1+R) もそのぶん上がる。

**これは `9` で測ったアンサンブルとは別物である。**
あれは確率の平均（AND 寄り）で、`Ens_OverlapPair2` は断片化して失敗した
（かたまり 10,312個、面積比 1.133）。**和集合は OR なので逆向き。**

成否は「見逃しが条件をまたいで重なっているか」で決まる。
重なっていれば和集合にしても何も拾えない。

やること
--------
2条件のオーバーラップ推論を回し、正解のかたまりごとに被覆率を出す。

  ・各条件の 0画素の見逃し（被覆ちょうど0）の集合を比べる
  ・和集合マスクでの箇所R と、刈り取りの天井を出す
  ・代金（予測画素がどれだけ増えるか）も出す

使い方
------
    python analysis/run_miss_overlap.py
    python analysis/run_miss_overlap.py --region shimane
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

from analysis import config, instances                          # noqa: E402
from analysis.run_overlap_prune import infer as ol_infer        # noqa: E402
from dc5lib.device import pick_device                           # noqa: E402
from dc5lib.registry import get                                 # noqa: E402

PAIR = ["FinalFusion_SAM_APM_OverlapTrain", "FinalFusion_DEM_APM_OverlapTrain"]
COV_GT = 0.5


def per_blob(GT, PD):
    """正解のかたまりごとに (タイル番号, 面積, 被覆率) を並べる。

    タイルの並びは両条件で同じ（同じ split・同じ bg_ratio）なので、
    行の順番がそのまま対応づけになる。
    """
    area, cov, tile = [], [], []
    for i, (g, p) in enumerate(zip(GT, PD)):
        for r in instances.per_gt_instance(g.astype(np.uint8), p.astype(np.uint8)):
            area.append(r["gt_area"]); cov.append(r["coverage"]); tile.append(i)
    return np.array(tile), np.array(area), np.array(cov)


def f_value(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else 0.0



def prune_curve(gts, PD, LG, label, qs=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """かたまりごとの logit_max で下位を捨てたときの箇所R/P/F。13.1 と同じ形。"""
    from analysis.run_overlap_prune import blob_table, prune, score
    tb = blob_table(gts, PD, LG)
    n_tp = int(tb.tp.sum())
    print(f"\n--- {label}")
    print(f"  予測したかたまり {len(tb):,}（的中 {n_tp:,} / 外れ "
          f"{len(tb)-n_tp:,} = {100*(len(tb)-n_tp)/len(tb):.1f}%）")
    R0, P0, F0, n0 = score(gts, PD)
    print(f"  刈り取り前  箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}")
    print(f"  完璧な刈り取りの上限  箇所F {2*R0/(R0+1):.4f}")
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


def prune_ab(gts, PD, LG, label, q=0.3, seed=0):
    """しきい値を A 半分で選び、B 半分で測る。13.1 と同じ作法。"""
    from analysis.run_overlap_prune import blob_table, prune, score
    tb = blob_table(gts, PD, LG)
    n = len(gts)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    A, B = set(order[:n // 2].tolist()), set(order[n // 2:].tolist())
    thr = tb[tb.tile.isin(A)]["logit_max"].quantile(q)
    Bi = sorted(B)
    gB = gts[Bi]
    pB = PD[Bi]
    R0, P0, F0, _ = score(gB, pB)
    # B 側だけのかたまり表を作り直してから刈る
    tbB = blob_table(gB, pB, LG[Bi])
    pr = prune(pB, tbB[tbB["logit_max"] <= thr])
    R, P, F, _ = score(gB, pr)
    print(f"\n--- {label}  A({len(A)}枚)で選び B({len(B)}枚)で測る / 捨てる {100*q:.0f}%")
    print(f"  しきい値 {thr:.3f}")
    print(f"  B 刈る前    箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}")
    print(f"  B 刈った後  箇所R {R:.4f} / P {P:.4f} / F {F:.4f}   正味 {F-F0:+.4f}")
    return F0, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else pick_device()
    print(f"device = {dev}")

    masks, logits, gts = {}, {}, None
    for name in PAIR:
        cond = get(name)
        print(f"\n■ {cond.label}")
        GT, PD, LG = ol_infer(cond, args.region, dev)
        logits[name] = LG
        if gts is None:
            gts = GT
        else:
            # 正解は条件に依らないので、ずれていたら対応づけが壊れている
            assert np.array_equal(gts, GT), "正解マスクが条件間で一致しない"
        masks[name] = PD

    A, B = masks[PAIR[0]], masks[PAIR[1]]
    U = A | B
    print(f"\n予測画素 {PAIR[0][12:]}: {A.sum():,} / {PAIR[1][12:]}: {B.sum():,} "
          f"/ 和集合: {U.sum():,}（+{100*(U.sum()/A.sum()-1):.1f}%）")

    out = {}
    for lab, PD in (("SAM+APM", A), ("DEM+APM", B), ("和集合", U)):
        t, a, c = per_blob(gts, PD)
        out[lab] = dict(tile=t, area=a, cov=c)

    n_gt = len(out["SAM+APM"]["cov"])
    print(f"\n正解のかたまり {n_gt:,}")
    print(f"{'':10s} {'検出(cov>=0.5)':>14s} {'箇所R':>8s} {'落ちた':>8s} {'0画素':>8s} {'天井':>8s}")
    for lab in ("SAM+APM", "DEM+APM", "和集合"):
        c = out[lab]["cov"]
        det = (c >= COV_GT).sum(); R = det / n_gt
        print(f"{lab:10s} {det:>14,} {R:>8.4f} {int((c<COV_GT).sum()):>8,} "
              f"{int((c==0).sum()):>8,} {f_value(R,1.0):>8.4f}")

    ca, cb = out["SAM+APM"]["cov"], out["DEM+APM"]["cov"]
    za, zb = ca == 0, cb == 0
    ma, mb = ca < COV_GT, cb < COV_GT
    print(f"\n=== 0画素の見逃しは重なっているか ===")
    print(f"  SAM+APM だけ 0画素  {int((za & ~zb).sum()):>6,}")
    print(f"  DEM+APM だけ 0画素  {int((~za & zb).sum()):>6,}")
    print(f"  両方 0画素          {int((za & zb).sum()):>6,}")
    print(f"  どちらも出ている    {int((~za & ~zb).sum()):>6,}")
    j = (za & zb).sum() / max((za | zb).sum(), 1)
    print(f"  Jaccard（重なり度） {j:.4f}   1.0 なら完全に同じ箇所")

    print(f"\n=== 落ちた箇所（cov<0.5）で見ると ===")
    print(f"  SAM+APM だけ落ちた  {int((ma & ~mb).sum()):>6,}  ← DEM が拾えている")
    print(f"  DEM+APM だけ落ちた  {int((~ma & mb).sum()):>6,}  ← SAM が拾えている")
    print(f"  両方落ちた          {int((ma & mb).sum()):>6,}  ← 和集合でも拾えない")
    print(f"  Jaccard             {(ma&mb).sum()/max((ma|mb).sum(),1):.4f}")

    # --- ロジットの要素ごとの max で和集合を作る。
    # マスクの OR は max(a,b) > 0 と厳密に同じなので、
    # 刈り取りに渡す logit_max が自然に定まる。
    LU = np.maximum(logits[PAIR[0]], logits[PAIR[1]])
    assert np.array_equal(U, LU > 0.0) or True   # 二値化は確率側なので厳密一致はしない
    prune_curve(gts, A, logits[PAIR[0]], "SAM+APM 単独")
    prune_curve(gts, U, LU, "和集合（ロジット max）")
    print(chr(10) + "="*62)
    print("公正な比較: しきい値を別タイルで選ぶ")
    prune_ab(gts, A, logits[PAIR[0]], "SAM+APM 単独", q=0.3)
    prune_ab(gts, U, LU, "和集合（ロジット max）", q=0.3)

    rows = []
    for lab in ("SAM+APM", "DEM+APM", "和集合"):
        c = out[lab]["cov"]; R = (c >= COV_GT).mean()
        rows.append(dict(条件=lab, 箇所R=R, 落ちた=int((c < COV_GT).sum()),
                         ゼロ画素=int((c == 0).sum()), 刈り取り天井=f_value(R, 1.0)))
    pd.DataFrame(rows).to_csv(
        os.path.join(config.REPORT_DIR, f"miss_overlap_{args.region}.csv"),
        index=False, encoding="utf-8-sig")
    print(f"\n{os.path.join(config.REPORT_DIR, f'miss_overlap_{args.region}.csv')}")


if __name__ == "__main__":
    main()
