# -*- coding: utf-8 -*-
"""ブロック分割の結果を、学習データからの距離で層別する（第6章 6.2）。

同じ1本のモデル・同じ重みを、距離の層ごとに集計する。
距離以外の条件が完全に揃うので、**成績の差は自己相関だけに帰せられる。**
層ごとの正例率は設計の段階で揃えてある（7.265 / 7.251 / 7.311 %）。

    python analysis/run_block_eval.py
    python analysis/run_block_eval.py --condition FinalFusion_SAM_APM_BlockSplit

面積評価はタイル単位の CSV（cache/analysis/tiles/）、
箇所数評価は正解1件ごとの CSV（cache/analysis/instances/*__gt.csv）を使う。
どちらも推論・評価のときに書き出してあるので、ここでは読んで分けるだけ。
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

from dc5lib import paths, splits

ORDER = ["1-3", "4-15", "16+"]
LABEL = {"1-3": "1-3 歩（〜1.7km）",
         "4-15": "4-15 歩（〜8.6km）",
         "16+": "16 歩以上（8.6km〜）"}


def _f1(r, p):
    return 2 * r * p / (r + p) if (r + p) else float("nan")


def area_by_bin(cond, region, bins):
    """面積評価を距離の層ごとに出す。タイル単位の tp/fp/fn を足し合わせる。"""
    f = paths.cache_dir("analysis") / "tiles" / f"{cond}__{region}.csv"
    if not f.exists():
        print(f"  ※ {f} が無い")
        return None
    t = pd.read_csv(f)
    t["bin"] = t.tile_no.astype(int).map(bins)
    rows = []
    for b in ORDER:
        s = t[t["bin"] == b]
        if not len(s):
            continue
        tp, fp, fn = s.tp.sum(), s.fp.sum(), s.fn.sum()
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        rows.append(dict(層=LABEL[b], タイル数=len(s), R=r, P=p, F=_f1(r, p)))
    return pd.DataFrame(rows)


def inst_by_bin(cond, region, bins, cov_gt=0.5):
    """箇所数評価の再現率を距離の層ごとに出す。

    **適合率は層ごとに出せない。** 予測のかたまりは「どの正解に当たったか」で
    層が決まるので、外れた予測はどの層にも属さない。
    層別できるのは正解側＝再現率である。
    """
    f = paths.cache_dir("analysis") / "instances" / f"{cond}__{region}__gt.csv"
    if not f.exists():
        print(f"  ※ {f} が無い（--save-instances を付けて評価し直す）")
        return None
    g = pd.read_csv(f)
    g["bin"] = g.tile_no.astype(int).map(bins)
    rows = []
    for b in ORDER:
        s = g[g["bin"] == b]
        if not len(s):
            continue
        rows.append(dict(層=LABEL[b], 正解数=len(s),
                         箇所R=float((s.coverage >= cov_gt).mean()),
                         最良IoU=float(s.best_iou.mean()),
                         中央の大きさ=float(s.gt_area.median())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_BlockSplit")
    ap.add_argument("--region", default="hiroshima_block")
    args = ap.parse_args()

    bins = splits.dist_bin()
    sp = splits.load()
    te = sp[sp.split == "test"]
    print(splits.describe())
    print()

    a = area_by_bin(args.condition, args.region, bins)
    if a is not None:
        print("【面積評価】")
        print(a.to_string(index=False,
                          formatters={"R": "{:.4f}".format, "P": "{:.4f}".format,
                                      "F": "{:.4f}".format}))
        d = a.F.max() - a.F.min()
        print(f"  層のあいだの幅 {d:.4f}"
              f"（面積F のゆらぎ幅 0.0018 の {d / 0.0018:.1f} 倍）")
        print()

    i = inst_by_bin(args.condition, args.region, bins)
    if i is not None:
        print("【箇所数評価（再現率）】")
        print(i.to_string(index=False,
                          formatters={"箇所R": "{:.4f}".format,
                                      "最良IoU": "{:.4f}".format,
                                      "中央の大きさ": "{:.0f}".format}))
        d = i.箇所R.max() - i.箇所R.min()
        print(f"  層のあいだの幅 {d:.4f}"
              f"（箇所F のゆらぎ幅 0.0071 の {d / 0.0071:.1f} 倍）")
        print()

    # 設計どおり正例率が揃っているかの念押し
    geo = pd.read_csv(paths.cache_dir("analysis") / "tiles_geo__hiroshima.csv")
    geo = geo[geo.gt_pos > 0]
    m = geo[geo.tile_no.isin(te.tile_no)].copy()
    m["bin"] = m.tile_no.astype(int).map(bins)
    r = m.groupby("bin").apply(
        lambda t: t.gt_pos.sum() / t.n_pix.sum() * 100, include_groups=False)
    print("【念のため：層ごとの正例率】" +
          "  ".join(f"{LABEL[b]} {r.get(b, float('nan')):.3f}%" for b in ORDER))


if __name__ == "__main__":
    main()
