# -*- coding: utf-8 -*-
"""データの鮮度と地形を統制しても、県の差が残るかを測る（F / 第6章）。

なぜ回帰なのか
--------------
`32` で層別したが、**写真・DEM・指定年・地形・正例率は互いに相関しており、
層別では交絡が落とせない**ところまで分かった。
一方を固定してもう一方を見る二元表でも、どちらも単調でなかった。

切り分けるには「**他を固定したときの各要因の寄与**」が要る。
正解のかたまり1件ずつを単位にして、

    当たったか（被覆 >= 0.5） ~ 県 + 写真年 + DEM年 + 指定年
                              + 大きさ + 正例率 + 傾斜

をロジスティック回帰で当てる。**知りたいのは県ダミーの係数**である。

  県の係数が小さくなる → 差はデータの鮮度や地形で説明できる
  県の係数が残る       → 鮮度では説明できない。県そのものに何かある

段階的に変数を足して、県の係数がどう動くかを見る。

使い方
------
    python analysis/run_vintage_regression.py
    python analysis/run_vintage_regression.py --condition FinalFusion_SAM_APM_LU_OverlapTrain

**単位はかたまりであってタイルではない。** 同じタイルのかたまりは
似た条件を共有するので、標準誤差は見かけより小さく出る。
係数の向きと大きさを読み、有意性の細かい判定はしない。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths

PROV = paths.REPO_ROOT / "provenance"
BAND = 0.0071


def load(cond, region, side="gt"):
    """かたまり1件1行に、そのタイルの鮮度と地形をくっつける。

    side="gt"   正解のかたまり。当たったか＝**再現率**を説明する
    side="pred" 予測のかたまり。的中したか＝**適合率**を説明する
    """
    cache = paths.cache_dir("analysis")
    g = pd.read_csv(cache / "instances" / f"{cond}__{region}__{side}.csv")
    v = pd.read_csv(PROV / f"tile_vintage__{region}.csv")
    g = g.merge(v[["tile_no", "gt_pos", "apm_photo_year", "dem_survey_year",
                   "mask_year_max"]], on="tile_no", how="left")
    f = cache / "features" / f"tile_features__{region}.csv"
    if f.exists():
        t = pd.read_csv(f)
        keep = [c for c in ("tile_no", "slope_mean", "slope_std", "slope_p90",
                            "relief", "slope_contrast") if c in t.columns]
        g = g.merge(t[keep], on="tile_no", how="left")
    g["region"] = region
    return g


def prep(df, side="gt"):
    """当たり外れと、単位をそろえた説明変数を作る。

    **再現率と適合率で判定のしきい値が違う。**
    正解側は被覆 0.5 以上で検出、予測側は被覆 0.3 以上で的中（`2.3`）。
    """
    df = df.copy()
    if side == "gt":
        df["hit"] = (df.coverage >= 0.5).astype(int)
        df["log_area"] = np.log(df.gt_area.clip(lower=1))
    else:
        df["hit"] = (df.cov_pred >= 0.3).astype(int)
        df["log_area"] = np.log(df.pred_area.clip(lower=1))
    df["gt_ratio"] = df.gt_pos / (128 * 128)
    df["is_shimane"] = (df.region == "shimane").astype(int)
    return df


def logit_irls(X, y, ridge=1e-6, iters=50, tol=1e-8):
    """ロジスティック回帰を IRLS で解く。切片は自分で足す。

    **sklearn を使わない。** この環境では LogisticRegression が
    無言でプロセスごと落ちる（exit 127。OpenMP/MKL の衝突と思われる）。
    IRLS は20行で書けて、標準誤差も副産物で出るので、むしろ都合がよい。
    """
    n, k = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    b = np.zeros(k + 1)
    for _ in range(iters):
        z = A @ b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        H = A.T @ (A * w[:, None]) + ridge * np.eye(k + 1)
        g = A.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < tol:
            break
    z = A @ b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    w = np.clip(p * (1 - p), 1e-9, None)
    H = A.T @ (A * w[:, None]) + ridge * np.eye(k + 1)
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    acc = float(((p >= 0.5).astype(int) == y).mean())
    return b[1:], se[1:], acc


def fit(df, cols, label):
    """係数は**標準化してから**出す（単位が違うので、そのままでは比べられない）。"""
    d = df.dropna(subset=cols + ["hit"])
    X = d[cols].astype(float).values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    coef, se, acc = logit_irls(Z, d.hit.values.astype(float))
    co = dict(zip(cols, coef))
    se_ = dict(zip(cols, se))
    i = cols.index("is_shimane") if "is_shimane" in cols else None
    tag = (f"県 {coef[i]:+.4f} (SE {se[i]:.4f})" if i is not None else "")
    print(f"  {label:<30} n={len(d):>6}  的中率 {acc:.4f}  {tag}")
    return co, se_, len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_LU_OverlapTrain")
    ap.add_argument("--side", default="gt", choices=["gt", "pred"],
                    help="gt=再現率を説明する / pred=適合率を説明する")
    args = ap.parse_args()

    parts = []
    for r in ("hiroshima", "shimane"):
        try:
            parts.append(load(args.condition, r, args.side))
        except FileNotFoundError as e:
            print(f"※ {r} が読めない: {e}")
    df = prep(pd.concat(parts, ignore_index=True), args.side)

    what = "箇所Recall" if args.side == "gt" else "箇所Precision"
    print(f"条件 {args.condition} / 説明するもの {what}")
    print(f"かたまり {len(df):,} 件"
          f"（広島 {int((df.region == 'hiroshima').sum()):,} / "
          f"島根 {int((df.region == 'shimane').sum()):,}）")
    print()
    print(f"【素の{what}】")
    for r, s in df.groupby("region"):
        print(f"  {r:<12} {s.hit.mean():.4f}（{len(s):,} 件）")
    d = (df[df.region == "hiroshima"].hit.mean()
         - df[df.region == "shimane"].hit.mean())
    print(f"  差 {d:+.4f}（ゆらぎ幅 {BAND} の {abs(d) / BAND:.1f} 倍）")
    print()

    has_slope = "slope_mean" in df.columns and df.slope_mean.notna().any()
    steps = [
        (["is_shimane"], "県だけ"),
        (["is_shimane", "log_area"], "+ かたまりの大きさ"),
        (["is_shimane", "log_area", "gt_ratio"], "+ 正例率"),
        (["is_shimane", "log_area", "gt_ratio", "apm_photo_year"], "+ 写真年"),
        (["is_shimane", "log_area", "gt_ratio", "apm_photo_year",
          "dem_survey_year"], "+ DEM年"),
        (["is_shimane", "log_area", "gt_ratio", "apm_photo_year",
          "dem_survey_year", "mask_year_max"], "+ 指定年"),
    ]
    if has_slope:
        steps.append((steps[-1][0] + ["slope_mean", "slope_std", "relief"],
                      "+ 傾斜と起伏"))

    print("【段階的に変数を足したときの県ダミーの係数】")
    print("  （正なら島根のほうが当たる。0 に近づくほど『県の差は説明できた』）")
    last = None
    for cols, lab in steps:
        last = fit(df, cols, lab)
    print()

    print("【最後のモデルの標準化係数】")
    co, se, n = last
    print(f"  {'変数':<18}{'係数':>10}{'SE':>9}{'係数/SE':>9}")
    for k, v in sorted(co.items(), key=lambda x: -abs(x[1])):
        e = se[k]
        print(f"  {k:<18}{v:>+10.4f}{e:>9.4f}{v / e:>9.1f}")
    print()
    print("※ 単位はかたまりであってタイルではない。同じタイルのかたまりは")
    print("   似た条件を共有するので、標準誤差は見かけより小さく出る。")
    print("   係数の向きと大きさを読み、有意性の細かい判定はしない。")


if __name__ == "__main__":
    main()
