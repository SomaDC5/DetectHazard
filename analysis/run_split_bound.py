# -*- coding: utf-8 -*-
"""「完璧に分割できたら」の箇所F の上限。キャッシュから計算、推論も学習も不要。

なぜ必要か（展望_再現率をどう上げるか.md 19）
--------------------------------------------
2段目は「残す / 捨てる」しか選べないので、融合したかたまりを分けられない
（`10.5` で意図的にそう制約した）。オーバーラップ学習は飲み込みが全条件で最大
（飲み込まれた正解 27.62%）なので、**2段目の設計を「分ける」に変える価値が
あるのでは**という問いが出た。`10.2` で刈り取りの上限を出したのと同じやり方で
計算できる。

指標の定義から分かること（analysis/instances.py evaluate_tile, mode="coverage"）
--------------------------------------------------------------------------------
    cov_g = O.sum(axis=1) / gt_sizes    = |g_i ∩ (予測の和集合)| / |g_i|
    tp_gt = (cov_g >= 0.5).sum()

**cov_g は予測の和集合にしか依存しない。** かたまりを分割しても画素の和集合は
変わらないので、tp_gt は厳密に不変 → **箇所Recall は分割で一切変わらない。**
これは測定ではなく算術の帰結であり、推論の仕方（従来 / オーバーラップ）にも
依存しない。

変わるのは適合率だけである。

    cov_p = O.sum(axis=0) / pred_sizes  （予測ごと）
    tp_pred = (cov_p >= 0.3).sum()

融合した予測1個を k 個に割ると n_pred も tp_pred も増える。
楽観的な上限として、割った破片は全部 TP になると仮定する。

    delta    = gt_in_merge - merged_pred
    n_pred  += delta
    tp_pred += delta

**n_pred と tp_pred が同じだけ増えるので、適合率は 1 に向かうだけ。**
だから上限が低い。

使い方
------
    python analysis/run_split_bound.py
    python analysis/run_split_bound.py --region shimane
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config                                   # noqa: E402
from dc5lib.instance_eval import PRIMARY                      # noqa: E402

INSTANCES_DIR = os.path.join(config.OUTPUT_DIR, "instances")


def f_value(r, p):
    return 2 * r * p / (r + p) if (r + p) > 0 else 0.0


def bounds(df):
    """1条件ぶん。現在 / 刈り取りの上限 / 分割の上限。"""
    n_gt, n_pred = df.n_gt.sum(), df.n_pred.sum()
    tp_gt, tp_pred = df[PRIMARY + "__tp_gt"].sum(), df[PRIMARY + "__tp_pred"].sum()
    mp, gim = df.merged_pred.sum(), df.gt_in_merge.sum()
    R, P = tp_gt / n_gt, tp_pred / n_pred
    delta = gim - mp                      # 分割で増える予測の数
    P_split = (tp_pred + delta) / (n_pred + delta)
    return dict(
        条件=df.condition.iloc[0], 正解=int(n_gt), 予測=int(n_pred),
        融合予測=int(mp), 飲み込まれた正解=int(gim), 飲み込み率=gim / n_gt,
        R=R, P=P, F=f_value(R, P),
        刈り取り上限=f_value(R, 1.0),          # FP を全除去（10.2 と同じ）
        分割上限=f_value(R, P_split),          # R は不変
        分割の伸び=f_value(R, P_split) - f_value(R, P),
        刈り取りの伸び=f_value(R, 1.0) - f_value(R, P),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(INSTANCES_DIR, f"*__{args.region}.csv"))):
        if f.endswith("__gt.csv"):
            continue
        d = pd.read_csv(f)
        if PRIMARY + "__tp_gt" not in d.columns:
            continue
        rows.append(bounds(d))
    if not rows:
        sys.exit(f"{INSTANCES_DIR} に {args.region} の結果がありません。"
                 "先に run_instance_eval.py を実行してください")

    x = pd.DataFrame(rows).sort_values("F", ascending=False)
    out = os.path.join(config.REPORT_DIR, f"split_bound_{args.region}.csv")
    x.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    cols = ["条件", "正解", "予測", "飲み込まれた正解", "飲み込み率",
            "R", "P", "F", "刈り取り上限", "分割上限", "分割の伸び"]
    fm = {c: "{:.4f}".format for c in
          ("飲み込み率", "R", "P", "F", "刈り取り上限", "分割上限", "分割の伸び")}
    print(f"=== {args.region}・従来推論（キャッシュ済み） ===")
    print(x[cols].to_string(index=False, formatters=fm))
    print()
    print("箇所Recall は分割で厳密に不変（cov_g は予測の和集合にしか依存しない）。")
    print("伸びるのは適合率だけで、n_pred と tp_pred が同じだけ増えるので天井が低い。")
    print("\n", out)


if __name__ == "__main__":
    main()
