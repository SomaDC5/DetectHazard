# -*- coding: utf-8 -*-
"""航空写真の撮影年度で成績を層別する（教授の指摘の検証）。

指摘
----
「精度が落ちているのは、航空写真が古く、宅地開発が進んでいるところを
見落としているのではないか」

これは検証できる。正解データは国土数値情報 A33 の**令和4年度（2022年）版**で、
2022年時点の指定を表す。急傾斜地崩壊危険区域の指定基準には**人家の有無**が
含まれるので、

  写真が 2022 年より古い → 指定の原因になった宅地が写真に写っていない
                         → モデルは「人家のない斜面」と見て落とす

という筋になる。**写真が古いタイルほど再現率が低い**なら指摘が支持される。

撮影年度は `run_photo_year.py --scan` が集める
（`seamlessphoto` と年度別レイヤが同一であることは `--verify` で確認済み）。

    python analysis/run_photo_year_eval.py
    python analysis/run_photo_year_eval.py --condition FinalFusion_SAM_APM

**交絡に注意。** 撮影年度は地域とひも付いている（市街地ほど新しく撮り直される）。
古い写真のタイルで再現率が低くても、それが「写真が古いから」なのか
「そういう地域だから」なのかは、この集計だけでは分けられない。
正例率とかたまりの大きさを併記するのはそのためである。
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

MASK_YEAR = 2022            # 国土数値情報 A33 令和4年度版


def _f1(r, p):
    return 2 * r * p / (r + p) if (r + p) else float("nan")


def load_year(region):
    f = paths.cache_dir("analysis") / f"photo_year__{region}.csv"
    if not f.exists():
        raise SystemExit(
            f"{f} が無い。先に\n"
            f"  python analysis/run_photo_year.py --scan --test-only\n"
            f"を走らせること。")
    t = pd.read_csv(f)
    return dict(zip(t.tile_no.astype(int), t.year))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM")
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--cov", type=float, default=0.5)
    args = ap.parse_args()

    yr = load_year(args.region)
    cache = paths.cache_dir("analysis")

    # --- 面積評価（タイル単位） ---
    f = cache / "tiles" / f"{args.condition}__{args.region}.csv"
    geo = pd.read_csv(cache / f"tiles_geo__{args.region}.csv")
    geo = geo.set_index(geo.tile_no.astype(int))
    rows = []
    if f.exists():
        t = pd.read_csv(f)
        t["year"] = t.tile_no.astype(int).map(yr)
        for y, s in t.groupby("year", dropna=True):
            tp, fp, fn = s.tp.sum(), s.fp.sum(), s.fn.sum()
            r = tp / (tp + fn) if (tp + fn) else np.nan
            p = tp / (tp + fp) if (tp + fp) else np.nan
            g = geo.reindex(s.tile_no.astype(int))
            rows.append(dict(撮影年度=int(y), タイル数=len(s),
                             正例率=g.gt_pos.sum() / g.n_pix.sum() * 100,
                             面積R=r, 面積P=p, 面積F=_f1(r, p)))
    area = pd.DataFrame(rows).sort_values("撮影年度", ascending=False)

    # --- 箇所数評価（正解1件ごと） ---
    f = cache / "instances" / f"{args.condition}__{args.region}__gt.csv"
    inst = None
    if f.exists():
        g = pd.read_csv(f)
        g["year"] = g.tile_no.astype(int).map(yr)
        inst = (g.dropna(subset=["year"]).groupby("year")
                .apply(lambda s: pd.Series({
                    "正解数": len(s),
                    "箇所R": float((s.coverage >= args.cov).mean()),
                    "最良IoU": float(s.best_iou.mean()),
                    "中央の大きさ": float(s.gt_area.median()),
                }), include_groups=False)
                .reset_index().sort_values("year", ascending=False))
        inst["year"] = inst.year.astype(int)

    print(f"条件 {args.condition} / 地域 {args.region}")
    print(f"正解データ: 国土数値情報 A33 令和4年度版（{MASK_YEAR} 年時点の指定）")
    print()
    if len(area):
        print("【撮影年度ごとの面積評価】")
        print(area.to_string(index=False, formatters={
            "正例率": "{:.3f}%".format, "面積R": "{:.4f}".format,
            "面積P": "{:.4f}".format, "面積F": "{:.4f}".format}))
        print()
    if inst is not None and len(inst):
        print("【撮影年度ごとの箇所数評価（再現率）】")
        print(inst.to_string(index=False, formatters={
            "正解数": "{:.0f}".format, "箇所R": "{:.4f}".format,
            "最良IoU": "{:.4f}".format, "中央の大きさ": "{:.0f}".format}))
        print()

        # --- 本題：写真が正解より古いか新しいか ---
        old = inst[inst.year < MASK_YEAR]
        new = inst[inst.year >= MASK_YEAR]
        if len(old) and len(new):
            def agg(d):
                w = d.正解数
                return float((d.箇所R * w).sum() / w.sum()), int(w.sum())
            ro, no = agg(old)
            rn, nn = agg(new)
            print(f"【指摘の検証】正解（{MASK_YEAR}年）より")
            print(f"  写真が古い（〜{MASK_YEAR - 1}年度）  正解 {no:>5} 件  箇所R {ro:.4f}")
            print(f"  写真が同時期以降（{MASK_YEAR}年度〜）正解 {nn:>5} 件  箇所R {rn:.4f}")
            d = rn - ro
            print(f"  差 {d:+.4f}（箇所F のゆらぎ幅 0.0071 の {abs(d) / 0.0071:.1f} 倍）")
            print()
            if d > 0.0071:
                print("  → **新しい写真のほうが当たる。指摘を支持する向き。**")
            elif d < -0.0071:
                print("  → **古い写真のほうが当たる。指摘とは逆向き。**")
            else:
                print("  → **帯の中。写真の新旧で再現率は変わらない。**")
            print()
            print("  ※ 撮影年度は地域とひも付く（市街地ほど新しく撮り直される）。")
            print("     上の表の正例率と中央の大きさが層で違うなら、")
            print("     差の一部はそちらに由来する。")


if __name__ == "__main__":
    main()
