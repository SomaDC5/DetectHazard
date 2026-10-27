# -*- coding: utf-8 -*-
"""撮影年 × 建物の有無 で検出率を見る。**教授の指摘を正面から測る。**

指摘（`31` / `32`）
-------------------
> 精度が落ちているのは、航空写真が古く、宅地開発がすすんでいるところを
> 見落としているのでは？

これが正しいなら、見逃した区域は

    **写真が古い（2018年）タイルで、いま（2025年）建物がある場所**

＝写真の後に開発された場所に偏るはずである。

なぜ測れるのか
--------------
**建物データは2025/2026年時点である**（`38.2`）。写真は2018〜2023年。
したがって「写真が古い × いま建物がある」は
**写真に写っていない可能性のある人家**を指す。

逆に「写真が新しい × いま建物がある」は写真にも写っている。
この2つで検出率が変わるなら、指摘は当たっている。

`32` は**指定年**と写真年を比べた（区域がいつ指定されたか）。
ここでは**建物**と写真年を比べる（人家がいつからあるか）。別の角度である。

    python analysis/run_building_photo.py

先に `run_building_miss.py --save` が要る。
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

from dc5lib import paths                                            # noqa: E402

BAND = {"hiroshima": 0.0071, "shimane": 0.0198}

# 「古い写真」の境目。**地域で変える。**
#   広島は 2018 / 2022 / 2023 の3段しか無い（2020年は1枚）。変化が乏しい
#   島根は 2009〜2024 に散っており、23% が 2017年以前（`30`）。
#   **こちらのほうが教授の指摘を強く試せる。**
CUT = {"hiroshima": 2018, "shimane": 2017}


def load(cond, region):
    d = paths.cache_dir("analysis") / "building"
    fp, fg = (d / f"{cond}__{region}__pred.csv",
              d / f"{cond}__{region}__gt.csv")
    for f in (fp, fg):
        if not f.exists():
            sys.exit(f"{f} がありません。先に\n"
                     f"  python analysis/run_building_miss.py --save")
    p, g = pd.read_csv(fp), pd.read_csv(fg)
    if "tile_no" not in g.columns:
        sys.exit("表に tile_no がありません。run_building_miss.py --save を"
                 "作り直した版で再実行してください")
    v = pd.read_csv(paths.REPO_ROOT / "provenance"
                    / f"tile_vintage__{region}.csv")
    keep = ["tile_no", "apm_photo_year", "dem_survey_year",
            "mask_year_min", "gt_pos"]
    for df in (p, g):
        df["has_bld"] = df.bld_in > 0
    p = p.merge(v[keep], on="tile_no", how="left")
    g = g.merge(v[keep], on="tile_no", how="left")
    return p, g


def rate_table(df, flag, title):
    print(title)
    print(f"{'写真年':>8}{'建物':>8}{'件数':>9}{'率':>9}")
    sub = df[df.apm_photo_year.notna()]
    for yr in sorted(sub.apm_photo_year.unique()):
        for hb in (True, False):
            m = (sub.apm_photo_year == yr) & (sub.has_bld == hb)
            if m.sum() < 20:
                continue
            print(f"{int(yr):>8}{'あり' if hb else 'なし':>8}"
                  f"{int(m.sum()):>9,}{sub.loc[m, flag].mean():>9.4f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane"])
    ap.add_argument("--cut", type=int, default=None,
                    help="これ以下の年を『古い写真』とする（既定は CUT）")
    a = ap.parse_args()

    p, g = load(a.condition, a.region)
    print(f"条件 {a.condition} / 地域 {a.region}")
    print(f"予測のかたまり {len(p):,}  正解のかたまり {len(g):,}")
    print(f"撮影年のあるタイルのかたまり: 予測 "
          f"{int(p.apm_photo_year.notna().sum()):,} / 正解 "
          f"{int(g.apm_photo_year.notna().sum()):,}")
    print()
    yrs = sub_years = p.apm_photo_year.dropna()
    print(f"※ 建物データは2025/2026年時点（`38.2`）。写真は "
          f"{int(yrs.min())}〜{int(yrs.max())}年。")
    print("   「写真が古い × いま建物がある」= 写真に写っていない"
          "可能性のある人家。")
    print()

    print("=" * 56)
    print("【1】正解のかたまりの検出率")
    print("=" * 56)
    rate_table(g, "found", "")

    sub = g[g.apm_photo_year.notna()]
    cut = a.cut if a.cut is not None else CUT[a.region]
    old = sub.apm_photo_year <= cut
    print(f"【まとめ】検出率（{cut}年以下を『古い』とする）")
    print(f"{'':<26}{f'写真 <={cut}':>11}{f'写真 >{cut}':>12}{'差':>10}")
    for hb, nm in ((True, "いま建物がある"), (False, "いま建物が無い")):
        a_ = sub.loc[old & (sub.has_bld == hb), "found"].mean()
        b_ = sub.loc[~old & (sub.has_bld == hb), "found"].mean()
        print(f"{nm:<26}{a_:>11.4f}{b_:>12.4f}{a_ - b_:>+10.4f}")
    print()
    print(f"  教授の指摘が正しいなら、**『写真 <={cut} × いま建物がある』の")
    print("  検出率が際立って低い**はずである（写真の後に開発された人家）。")
    print()

    print("=" * 56)
    print("【2】予測のかたまりの的中率")
    print("=" * 56)
    rate_table(p, "tp", "")

    # 建物の有無と写真年のどちらが効いているか
    print("=" * 56)
    print("【3】どちらが効いているか（正解側・検出率）")
    print("=" * 56)
    print(f"{'切り方':<34}{'低い側':>10}{'高い側':>10}{'差':>10}")
    for nm, m in (("建物の有無（1画素も無いか）", sub.has_bld),
                  (f"写真年（{cut}以下か それより新しいか）", ~old)):
        lo = sub.loc[~m, "found"].mean()
        hi = sub.loc[m, "found"].mean()
        print(f"{nm:<34}{lo:>10.4f}{hi:>10.4f}{hi - lo:>+10.4f}")
    print()
    print("  **建物の有無のほうが大きければ、効いているのは")
    print("  『開発されているか』であって『写真が古いか』ではない。**")


if __name__ == "__main__":
    main()
