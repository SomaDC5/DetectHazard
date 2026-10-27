# -*- coding: utf-8 -*-
"""`39`・`41` の**導出値**を、保存した CSV から出し直す。推論しない。

なぜ要るか
----------
`run_building_miss.py` が書く `*__gt.csv` / `*__pred.csv` から先は、
`41.2`・`41.3`・`41.4` のように**手で計算した数字**が docs に残っていた。
`43.5` の向きの直しで測り直すことになったとき、
**「どの列でどう出したのか」が記録されておらず復元に時間がかかった。**
`41.3` の面積層別の2行は結局再現できていない。

**だから導出をコードに残す。** 数字を docs に書くときは、
ここに関数を足してから書く。

    python analysis/run_building_derived.py
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\hirok\dc5\DetectHazard")

D = r"E:\dc5-data\cache\analysis\building"
C = "FinalFusion_SAM_APM_OverlapTrain"
BAND = {"hiroshima": 0.0071, "shimane": 0.0198}
# 前回（向きが違っていたとき）の数字。並べて出す
OLD = {
    "hiroshima": dict(d50=0.135, d100=0.071, near=0.5407, far=0.5019,
                      bins=[0.8270, 0.5163, 0.5867, 0.5219, 0.5238, 0.4425]),
    "shimane": dict(d50=0.192, d100=0.116, near=0.3401, far=0.2612,
                    bins=[0.6391, 0.3695, 0.3110, 0.2959, 0.2683, 0.2106]),
}
BINS = [(-1, 0), (0, 25), (25, 50), (50, 100), (100, 200), (200, 1e9)]
BINNM = ["中にある", "〜25m", "25-50m", "50-100m", "100-200m", "200m〜"]

for reg in ("hiroshima", "shimane"):
    g = pd.read_csv(f"{D}\\{C}__{reg}__gt.csv")
    band = BAND[reg]
    o = OLD[reg]
    n = len(g)
    miss = g[~g.found]
    print("=" * 72)
    print(f"{reg}  正解のかたまり {n:,} / 見逃し {len(miss):,}"
          f"（箇所R {g.found.mean():.4f}）")
    print("=" * 72)

    # ---- 41.2 「建物がない」の定義しだい
    print("\n【41.2】見逃しのうち『建物がない』割合")
    print(f"{'定義':<34}{'今回':>10}{'前回':>10}{'差':>10}")
    v = float((miss.bld_in <= 0).mean())
    print(f"{'かたまりの中に1画素もない（無事）':<34}{v:>10.1%}{0.310 if reg=='hiroshima' else 0.330:>10.1%}"
          f"{v - (0.310 if reg == 'hiroshima' else 0.330):>+10.1%}")
    for lab, key, th in (("50m 以内に1画素もない", "d50", 50.0),
                         ("100m 以内に1画素もない", "d100", 100.0)):
        v = float((miss.bld_dist > th).mean())
        print(f"{lab:<34}{v:>10.1%}{o[key]:>10.1%}{v - o[key]:>+10.1%}")

    # ---- 41.3 距離別の検出率
    print("\n【41.3】最寄り建物までの距離別の検出率")
    print(f"{'':<12}" + "".join(f"{b:>11}" for b in BINNM))
    row_new = []
    for lo, hi in BINS:
        s = g[(g.bld_dist > lo) & (g.bld_dist <= hi)]
        row_new.append(float(s.found.mean()) if len(s) else float("nan"))
    print(f"{'今回':<12}" + "".join(f"{x:>11.4f}" for x in row_new))
    print(f"{'前回':<12}" + "".join(f"{x:>11.4f}" for x in o["bins"]))
    print(f"{'件数':<12}" + "".join(
        f"{int(((g.bld_dist > lo) & (g.bld_dist <= hi)).sum()):>11,}"
        for lo, hi in BINS))

    # ---- 41.3 「中にある」対「外」。面積を揃えた版（無事な量。確認用）
    inside = g.bld_in > 0
    raw = float(g[inside].found.mean() - g[~inside].found.mean())
    g = g.copy()
    g["q"] = pd.qcut(g.area, 5, labels=False, duplicates="drop")
    adj, wsum = 0.0, 0
    for q in sorted(g.q.unique()):
        s = g[g.q == q]
        a, b = s[s.bld_in > 0], s[s.bld_in <= 0]
        if len(a) and len(b):
            adj += (a.found.mean() - b.found.mean()) * len(s)
            wsum += len(s)
    print(f"\n【41.3 無事な量・確認用】『中にある』対『外』 "
          f"揃えない {raw:+.4f} / 面積を揃えた {adj / wsum:+.4f}")

    # ---- 41.3 重なりの無いもの（bld_in<=0）の中で、近い／遠い
    sub = g[g.bld_in <= 0]
    near = sub[sub.bld_dist <= 50.0]
    far = sub[sub.bld_dist > 50.0]
    rn, rf = float(near.found.mean()), float(far.found.mean())
    print(f"\n【41.3】中に建物が無いかたまり {len(sub):,} の中で")
    print(f"{'':<22}{'今回':>10}{'前回':>10}")
    print(f"{'50m以内に家あり':<22}{rn:>10.4f}{o['near']:>10.4f}"
          f"   （{len(near):,} 件）")
    print(f"{'50m外':<22}{rf:>10.4f}{o['far']:>10.4f}"
          f"   （{len(far):,} 件）")
    print(f"{'差':<22}{rn - rf:>+10.4f}{o['near'] - o['far']:>+10.4f}")

    # ---- 41.4 麓に置く上限
    # 仮定：仮想建物を麓に置くと、その見逃しは「50m外」から
    #       「50m以内に家あり」の群に移る。移った群の検出率の差が
    #       そのまま効くとする（＝近接の効果だけ買える、という上限）。
    #   上限 = （50m外の見逃しの件数 / 全正解のかたまり数）× 近接の効果
    cand = int(((~g.found) & (g.bld_in <= 0) & (g.bld_dist > 50.0)).sum())
    ceil_new = cand / n * (rn - rf)
    old_ceil = 0.0023 if reg == "hiroshima" else 0.0088
    print(f"\n【41.4】麓に置く上限")
    print(f"  対象＝中に建物が無く 50m外の見逃し {cand:,} 件"
          f"（全かたまりの {cand / n:.1%}）")
    print(f"  上限 = {cand:,}/{n:,} × {rn - rf:+.4f} "
          f"= **{ceil_new:+.4f}（{ceil_new / band:.1f}帯）**")
    print(f"  前回 {old_ceil:+.4f}（{old_ceil / band:.1f}帯）")
    print()
