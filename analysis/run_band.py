# -*- coding: utf-8 -*-
"""ゆらぎ幅（帯）を出す。キャッシュ済みの結果から計算するので推論は不要。

なぜ必要か（ゆらぎ幅.md）
------------------------
条件どうしの差を「効いた」と言うには、**同じ構成を学び直したときに
どれだけ動くか**が要る。複数シードの条件を学習したあと、この一本で
面積評価・箇所数評価の帯をまとめて出す。

**帯は指標と地域で11倍違う**（広島 面積F 0.0018 / 島根 箇所F 0.0198）。
判定にはその指標・その地域の帯を使うこと。

前提
----
指定した条件の推論と箇所数評価が済んでいること。

    python analysis/run_inference.py     --region both --conditions <条件...>
    python analysis/run_instance_eval.py --region both --conditions <条件...>

使い方
------
    # 既定（出発点 + シード3本）で帯を出す
    python analysis/run_band.py

    # 任意の条件群で
    python analysis/run_band.py --conditions A B C --region hiroshima

    # 帯と比べたい差を一緒に出す
    python analysis/run_band.py --compare FinalFusion_SAM_APM_OverlapTrain
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config                                   # noqa: E402
from dc5lib.instance_eval import PRIMARY                      # noqa: E402
from dc5lib.registry import get                               # noqa: E402

TILES = os.path.join(config.OUTPUT_DIR, "tiles")
INSTS = os.path.join(config.OUTPUT_DIR, "instances")

# 既定: 同一構成でシードだけ変えた4本（ゆらぎ幅.md の測定対象）
DEFAULT = ["FinalFusion_SAM_APM",
           "FinalFusion_SAM_APM_seed1",
           "FinalFusion_SAM_APM_seed2",
           "FinalFusion_SAM_APM_seed3"]


def _f(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else float("nan")


def area_metrics(cond, region):
    """タイル単位の混同行列を micro で足して面積評価を出す。"""
    f = os.path.join(TILES, f"{cond}__{region}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)

    def mk(a, b, c):
        tp, fp, fn = d[a].sum(), d[b].sum(), d[c].sum()
        R, P = tp / (tp + fn), tp / (tp + fp)
        return R, P, _f(R, P)

    R, P, F = mk("tp", "fp", "fn")
    Rb, Pb, Fb = mk("tp_c", "fp_c", "fn_c")
    return {"面積R": R, "面積P": P, "面積F": F,
            "境界R": Rb, "境界P": Pb, "境界F": Fb}


def inst_metrics(cond, region):
    """箇所数評価（主設定）を micro で足す。"""
    f = os.path.join(INSTS, f"{cond}__{region}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    c = PRIMARY + "__"
    if c + "tp_gt" not in d.columns:
        return None
    n_gt, n_pred = d.n_gt.sum(), d.n_pred.sum()
    R = d[c + "tp_gt"].sum() / n_gt
    P = d[c + "tp_pred"].sum() / n_pred
    return {"箇所R": R, "箇所P": P, "箇所F": _f(R, P)}


def collect(conds, region):
    rows = {}
    for n in conds:
        m = {}
        for part in (area_metrics(n, region), inst_metrics(n, region)):
            if part:
                m.update(part)
        if m:
            rows[n] = m
        else:
            print(f"  ※ {n}（{region}）の結果がありません。先に "
                  "run_inference.py / run_instance_eval.py を回してください")
    return rows


def band_table(rows):
    keys = ["面積R", "面積P", "面積F", "境界F", "箇所R", "箇所P", "箇所F"]
    out = []
    for k in keys:
        v = np.array([m[k] for m in rows.values() if k in m])
        if len(v) < 2:
            continue
        sd = v.std(ddof=1)
        out.append(dict(指標=k, n=len(v), 平均=v.mean(), 最小=v.min(),
                        最大=v.max(), 幅=np.ptp(v), 標準偏差=sd,
                        **{"±2sd": 2 * sd}))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="帯を測る条件群（同一構成でシードだけ違うもの）")
    ap.add_argument("--region", default="both",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--compare", nargs="*", default=None,
                    help="帯と比べたい条件。帯の何倍かを出す")
    args = ap.parse_args()
    conds = args.conditions or DEFAULT
    regions = (["hiroshima", "shimane"] if args.region == "both"
               else [args.region])

    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    fm = {c: "{:.4f}".format for c in
          ("平均", "最小", "最大", "幅", "標準偏差", "±2sd")}

    for region in regions:
        print(f"\n{'='*66}\n{region}  帯を測る条件 {len(conds)}本")
        rows = collect(conds, region)
        if len(rows) < 2:
            print("  2本以上そろっていないので帯は出せません")
            continue
        for n, m in rows.items():
            try:
                lab = get(n).label
            except Exception:
                lab = n
            s = "  ".join(f"{k} {m[k]:.4f}" for k in
                          ("面積F", "箇所F") if k in m)
            print(f"  {lab[:40]:42s} {s}")
        bt = band_table(rows)
        print()
        print(bt.to_string(index=False, formatters=fm))

        if args.compare:
            print(f"\n--- 帯との比較 ---")
            comp = collect(args.compare, region)
            base = {r["指標"]: r for _, r in bt.iterrows()}
            print(f"{'条件':40s} {'指標':>6s} {'値':>9s} {'帯の平均との差':>14s}"
                  f" {'幅':>8s} {'帯の何倍':>9s} 判定")
            for n, m in comp.items():
                for k in ("面積F", "箇所F"):
                    if k not in m or k not in base:
                        continue
                    d = m[k] - base[k]["平均"]
                    w = base[k]["幅"]
                    r = abs(d) / w if w else float("inf")
                    judge = ("帯の外" if r > 1 else "帯の中")
                    print(f"{n[:38]:40s} {k:>6s} {m[k]:>9.4f} {d:>+14.4f}"
                          f" {w:>8.4f} {r:>8.1f}倍 {judge}")

    print("\n帯は同一構成でのみ有効。構造を変えると分散自体が変わりうる"
          "（ゆらぎ幅.md 7）。")


if __name__ == "__main__":
    main()
