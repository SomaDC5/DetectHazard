# -*- coding: utf-8 -*-
"""`run_union_blind.py` が保存した CSV から、**部分集合ごとの床**を出す。推論しない。

なぜ要るか
----------
**床（どの構成員も1画素も光らない正解の割合）は、構成員ごとの覆いが
分かっていれば部分集合について正確に計算できる。** 1本でも覆い>0 なら
和集合の覆いも>0 だからである。だから「5本で回した CSV から
`旧3本+BLD` の床を出す」のに追加の推論は要らない。

**和集合の箇所R はこうは出せない。** マスクの論理和は、構成員ごとの
覆いの最大より大きくなりうる（`44.7`）。そこは推論し直すしかない。

`44.4`〜`44.6` の数字はここで出した。`run_building_derived.py` と同じ理由で
残してある ── **導出を記録しないと、測り直すときに復元できない**
（`41.3` の面積層別の2行は結局再現できていない）。

    python analysis/run_union_subsets.py --region hiroshima
    python analysis/run_union_subsets.py --region both --tag seedrun \
        --base SAM_APM DEM_APM SAM_APM_LU --candidates SAM_APM_seed1 SAM_APM_seed2

構成員の名前は CSV の列名そのまま（`cov_<略称>` の略称。
`run_union_blind.py` の `short()` が付ける）。
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import pandas as pd                                              # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths                                         # noqa: E402

# `run_union_blind.py` と同じ値。変えると比較できない
COV_GT = 0.5
BAND = {"hiroshima": 0.0071, "shimane": 0.0198}
BASE = ["SAM_APM", "DEM_APM", "SAM_APM_LU"]        # `UNION3`。`42` からの手札


def load(region, tag):
    name = f"union_blind__{region}.csv" if not tag else \
        f"union_blind__{region}__{tag}.csv"
    p = paths.cache_dir("analysis") / "coverage" / name
    if not p.exists():
        sys.exit(f"{p} がありません。先に run_union_blind.py を --save で回してください")
    return pd.read_csv(p), p


def floor_mask(df, members):
    """その手札で**どの構成員も1画素も光らない**かたまり。"""
    return (df[[f"cov_{m}" for m in members]] <= 0).all(axis=1)


def report(region, tag, base, cands):
    df, p = load(region, tag)
    need = [f"cov_{m}" for m in base + cands]
    miss = [c for c in need if c not in df.columns]
    if miss:
        have = sorted(c[4:] for c in df.columns if c.startswith("cov_")
                      and c != "cov_union")
        sys.exit(f"{region}: 列がありません {miss}\n  CSV にあるのは {have}")
    n, band = len(df), BAND[region]
    print("=" * 72)
    print(f"{region}  正解のかたまり {n:,}  ゆらぎ幅 {band}")
    print(f"  {p.name}")
    print("=" * 72)

    sets = [("土台", base)]
    sets += [(f"土台+{c}", base + [c]) for c in cands]
    if len(cands) > 1:
        sets.append((f"全{len(base) + len(cands)}本", base + cands))
    b0 = int(floor_mask(df, base).sum())

    print(f"\n{'手札':<22}{'床の件数':>10}{'床の割合':>10}"
          f"{'土台からの減り':>15}{'減った割合':>12}{'天井':>9}")
    for nm, ms in sets:
        k = int(floor_mask(df, ms).sum())
        print(f"{nm:<22}{k:>10,}{100 * k / n:>9.2f}%"
              f"{k - b0:>15,}{100 * (b0 - k) / max(b0, 1):>11.1f}%"
              f"{100 * (1 - k / n):>8.2f}%")

    if not cands:
        return
    blind = df[floor_mask(df, base)]
    lit = {c: blind[f"cov_{c}"] > 0 for c in cands}
    print(f"\n土台の盲の群 {len(blind):,} 件のうち")
    for c in cands:
        print(f"{'  ' + c + ' が光る':<30}{int(lit[c].sum()):>8,}"
              f"{100 * lit[c].mean():>8.1f}%")

    if len(cands) > 1:
        print(f"\n{'組':<30}{'両方':>8}{'片方だけ':>10}{'Jaccard':>10}")
        for a, b in itertools.combinations(cands, 2):
            both = int((lit[a] & lit[b]).sum())
            either = int((lit[a] | lit[b]).sum())
            j = both / either if either else float("nan")
            print(f"{a + ' / ' + b:<30}{both:>8,}{either - both:>10,}{j:>10.3f}")
        print("\n  **1 に近いほど同じかたまりを光らせている。**"
              " 低いほど手札として重ならない")

    print(f"\n{'構成員':<18}{'ロジット平均':>13}{'中央':>10}"
          f"{'負の割合':>11}{'覆い>0':>9}{'覆い50%以上':>13}")
    for m in base + cands:
        s, cv = blind[f"lg_{m}"], blind[f"cov_{m}"]
        print(f"{m:<18}{s.mean():>13.3f}{s.median():>10.3f}"
              f"{100 * (s <= 0).mean():>10.1f}%"
              f"{100 * (cv > 0).mean():>8.1f}%"
              f"{100 * (cv >= COV_GT).mean():>12.1f}%")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--tag", default="",
                    help="CSV の名前の後ろに付いている印（union_blind__<地域>__<印>.csv）")
    ap.add_argument("--base", nargs="+", default=BASE,
                    help="土台にする構成員の略称")
    ap.add_argument("--candidates", nargs="*",
                    default=["SAM_APM_BVB", "SAM_APM_BLD"],
                    help="土台に1本ずつ足して床を見る構成員の略称")
    a = ap.parse_args()
    regs = ["hiroshima", "shimane"] if a.region == "both" else [a.region]
    for r in regs:
        report(r, a.tag, list(a.base), list(a.candidates))


if __name__ == "__main__":
    main()
