# -*- coding: utf-8 -*-
"""背景タイルの偽陽性を減らせるかを測る（H / 第5章）。

なぜ要るか
----------
島根は `subset: all` で全 24,569 枚を評価する（実運用の模倣）。
うち 14,890 枚は**正解を1画素も含まない**ので、そこに出た予測は
定義上すべて偽陽性になる。実測で予測の 19.9 %、的中率 0.0003（`34`）。

**主系統（＝実運用の値）を上げる残り道はここしかない。**
`34.6` で測ったとおり、オーバーラップ学習も土地利用も、
評価範囲による適合率の差（6.7〜8.5帯）をまったく減らしていない。
いまの後処理はすべて「正解のあるタイルの中で」精度を調整するものだからである。

何を測るか
----------
1. **上限** — 背景タイルの予測を完全に消せたら箇所F はどこまで上がるか
2. **分離** — 既存のタイル単位の統計量だけで、背景タイルを当てられるか
   （当てられるなら**学習なし**で済む）
3. **代償** — 正解ありタイルを誤って消すと再現率が落ちる。その収支

    python analysis/run_bg_gate.py
    python analysis/run_bg_gate.py --condition FinalFusion_SAM_APM --region shimane
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

BAND = 0.0198          # 島根の箇所F のゆらぎ幅
COV_GT, COV_PRED = 0.5, 0.3


def auc(score, label):
    """順位相関にもとづく AUC。sklearn を使わない（環境で落ちるため）。"""
    s = np.asarray(score, dtype=float)
    y = np.asarray(label).astype(bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(cond, region):
    cache = paths.cache_dir("analysis")
    t = pd.read_csv(cache / "tiles" / f"{cond}__{region}.csv")
    g = pd.read_csv(cache / "instances" / f"{cond}__{region}__gt.csv")
    p = pd.read_csv(cache / "instances" / f"{cond}__{region}__pred.csv")
    v = pd.read_csv(paths.REPO_ROOT / "provenance" / f"tile_vintage__{region}.csv")
    t = t.merge(v[["tile_no", "gt_pos"]].rename(columns={"gt_pos": "gt_pix"}),
                on="tile_no", how="left")
    t["has_gt"] = t.gt_pix > 0
    p = p.merge(t[["tile_no", "has_gt"]], on="tile_no", how="left")
    p["hit"] = p.cov_pred >= COV_PRED
    g["hit"] = g.coverage >= COV_GT
    return t, g, p


def score(keep_tiles, g, p):
    """残すタイルを決めたときの箇所R・P・F。

    **ゲートは予測を消すだけで、正解は消さない。**
    正解ありタイルを誤って消すと、そこの正解は検出できなくなる
    （予測が無くなるので）。再現率はそのぶん下がる。
    """
    pk = p[p.tile_no.isin(keep_tiles)]
    gk = g.copy()
    gk.loc[~gk.tile_no.isin(keep_tiles), "hit"] = False
    r, pr = float(gk.hit.mean()), float(pk.hit.mean()) if len(pk) else 0.0
    return r, pr, (2 * r * pr / (r + pr) if (r + pr) else 0.0), len(pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM")
    ap.add_argument("--region", default="shimane")
    a = ap.parse_args()

    t, g, p = load(a.condition, a.region)
    n_bg = int((~t.has_gt).sum())
    print(f"条件 {a.condition} / 地域 {a.region}")
    print(f"タイル {len(t):,}（正解あり {int(t.has_gt.sum()):,} / "
          f"背景 {n_bg:,}）")
    print(f"予測 {len(p):,}（背景タイル内 {int((~p.has_gt).sum()):,} = "
          f"{(~p.has_gt).mean() * 100:.1f} %、的中率 "
          f"{p[~p.has_gt].hit.mean():.4f}）")
    print()

    base = score(set(t.tile_no), g, p)
    ceil = score(set(t[t.has_gt].tile_no), g, p)
    print("【上限】背景タイルの予測を完全に消せた場合")
    print(f"{'':<24}{'箇所R':>9}{'箇所P':>9}{'箇所F':>9}{'予測数':>9}")
    print(f"{'いま（主系統）':<24}{base[0]:>9.4f}{base[1]:>9.4f}"
          f"{base[2]:>9.4f}{base[3]:>9,}")
    print(f"{'完璧なタイル判定':<24}{ceil[0]:>9.4f}{ceil[1]:>9.4f}"
          f"{ceil[2]:>9.4f}{ceil[3]:>9,}")
    d = ceil[2] - base[2]
    print(f"  伸びしろ {d:+.4f}（ゆらぎ幅 {BAND} の {d / BAND:.1f} 倍）")
    print()

    # --- 既存の統計量だけで背景タイルを当てられるか ---
    t["n_pred"] = t.tile_no.map(p.groupby("tile_no").size()).fillna(0)
    # **prob_pos_mean は使えない。** (prob * 正解).sum / 正解画素数 で
    # 計算されており、**正解そのものを見ている**（infer.py:99）。
    # 正解が無いタイルでは 0 になるので AUC が 1.0 になるが、これはリークである。
    LEAK = {"prob_pos_mean"}
    cands = [c for c in ("prob_max", "prob_mean", "pred_pos", "n_pred")
             if c in t.columns and c not in LEAK]
    print("【分離】タイルが正解を含むかを、既存の統計量で当てられるか")
    print(f"{'統計量':<18}{'AUC':>8}")
    for c in sorted(cands, key=lambda c: -abs(auc(t[c], t.has_gt) - 0.5)):
        print(f"{c:<18}{auc(t[c], t.has_gt):>8.4f}")
    print("  0.5 = 情報なし。**学習なしで分けられるなら、これで足りる**")
    print()

    # --- しきい値を振って収支を見る ---
    best = max(cands, key=lambda c: abs(auc(t[c], t.has_gt) - 0.5))
    print(f"【収支】{best} でタイルを切る（この値が小さいタイルを捨てる）")
    print(f"{'しきい値':>12}{'残すタイル':>11}{'箇所R':>9}{'箇所P':>9}"
          f"{'箇所F':>9}{'対 主系統':>10}")
    v = t[best].astype(float)
    # 値の分位からしきい値の候補を作る。**同じ値が多いので重複を潰す。**
    thrs = sorted(set(np.quantile(v, np.linspace(0, 0.95, 40))))
    rows = []
    for thr in thrs:
        keep = set(t.loc[v >= thr, "tile_no"])
        if not keep:
            continue
        r, pr, f, n = score(keep, g, p)
        rows.append((thr, len(keep) / len(t), r, pr, f))
    shown = set()
    for thr, frac, r, pr, f in rows:
        k = round(frac, 2)
        if k in shown:
            continue
        shown.add(k)
        print(f"{thr:>12.1f}{frac * 100:>10.0f}%{r:>9.4f}{pr:>9.4f}"
              f"{f:>9.4f}{f - base[2]:>+10.4f}")
    if rows:
        b = max(rows, key=lambda x: x[4])
        print()
        print(f"  **最良** しきい値 {b[0]:.1f} / 残すタイル {b[1] * 100:.0f} % / "
              f"箇所F {b[4]:.4f}（主系統比 {b[4] - base[2]:+.4f}、"
              f"{(b[4] - base[2]) / BAND:.1f} 帯）")
        print(f"  上限 {ceil[2]:.4f} のうち "
              f"{(b[4] - base[2]) / (ceil[2] - base[2]) * 100:.0f} % を取った")
    print()
    print("  ※ ゲートは予測を消すだけで正解は消さない。")
    print("     正解ありタイルを誤って消すと、そこの正解は検出できなくなる。")
    print("  ※ しきい値は**この地域で選んでいる**ので、これは上限寄りの値である。")
    print("     実用では広島で選んで島根に当てる必要がある（`13` と同じ作法）。")


if __name__ == "__main__":
    main()
