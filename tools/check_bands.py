# -*- coding: utf-8 -*-
"""ゆらぎ幅（帯）の直書きを探す。**原本は `dc5lib/instance_eval.py`。**

なぜ必要か
----------
帯の写しが `analysis/` に **14本**あった（2026-11-10 の5回目のレビュー 4）。
値はどれも正しいので**今日は何も起きない。** 困るのは帯を測り直したときで、
写しは静かに古い値のまま残り、**「帯の中か外か」の判定が反転する。**
数字がずれるより、判定が壊れるほうが怖い。

実際に1本壊れていた ── `analysis/run_photo_year_eval.py` は `--region` を
取るのに帯が広島固定（`0.0071`）で、**島根で回すと帯の中の差を
「効いた」と誤読する**（2026-11-11 に原本読みへ直した）。

残っている写しは**すぐには直さないと決めた**ので（値は正しいので実害が無く、
読んでいないスクリプトを9本触るほうが危ない）、
**既知として許し、増えたときだけ落ちる**形にしてある。
**写しを1本減らしたら `KNOWN` からも1行減らすこと。**

見つけ方
--------
行に帯の値が現れたら1つと数える（コメントや docstring の中も数える。
**文章の中の古い帯も写しである**）。`0.00710` のように後ろに数字が続くものは
別の値なので数えない。

対象は `analysis/` `dc5lib/` `tools/` の `.py` だけ。
**`docs/` と `.ipynb` は文章なので対象外（未検査）。**
"""
from __future__ import annotations

import re
from pathlib import Path

# 帯の実測値（docs/ゆらぎ幅.md の 0 の表）。原本は dc5lib/instance_eval.py の
# BAND（箇所F）と BAND_AREA（面積F）。
LITERALS = ("0.0071", "0.0198", "0.0018", "0.0118")

# 直書きしてよい場所と、その理由（実装書 5 の表）
ALLOWED = {
    "dc5lib/instance_eval.py": "原本である",
    "analysis/run_band.py": "帯を測る側。原本から読むと循環する",
    "tools/verify_docs.py": "実装書の表と突き合わせる相手",
    "tools/check_bands.py": "探す値そのものを持っている",
    "tools/test_guards.py": "この検査が効くかを確かめる側",
}

# まだ写しのまま残っている本と、その行数（2026-11-11 時点）。
# **減らしたらここも減らす。** 増えたら検査が落ちる。
#
# **帯ではない同じ数字も混じる。** `dc5lib/models.py` の `+0.0118` は
# 島根 面積F の帯ではなく、ASPPStride8 の**伸び**である（偶然同じ）。
# 行で数えているだけなので、こういうものは既知に入れて黙らせる。
KNOWN = {
    "analysis/run_bg_gate.py": 1,
    "analysis/run_blob_cluster.py": 1,
    "analysis/run_building_derived.py": 1,
    "analysis/run_building_miss.py": 2,
    "analysis/run_designation_year.py": 1,
    "analysis/run_miss_coverage.py": 1,
    "analysis/run_union_blind.py": 1,
    "analysis/run_union_subsets.py": 1,
    "analysis/run_vintage_regression.py": 1,
    "dc5lib/models.py": 1,          # 帯ではない（ASPPStride8 の伸び +0.0118）
}

DIRS = ("analysis", "dc5lib", "tools")
_PAT = re.compile("|".join(rf"(?<![\d.]){re.escape(v)}(?!\d)" for v in LITERALS))


def counts(root) -> dict:
    """`リポジトリからの相対パス -> 帯の値が現れる行数`（0 の本は入れない）。"""
    root = Path(root)
    out = {}
    for d in DIRS:
        if not (root / d).is_dir():
            continue
        for p in sorted((root / d).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if rel in ALLOWED:
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except Exception:                                 # noqa: BLE001
                continue
            k = sum(1 for ln in src.splitlines() if _PAT.search(ln))
            if k:
                out[rel] = k
    return out


def scan(root) -> list:
    """既知より増えた直書きを文にして返す。問題が無ければ空。"""
    got = counts(root)
    problems = []
    for rel, k in sorted(got.items()):
        want = KNOWN.get(rel, 0)
        if k > want:
            problems.append(
                f"{rel} に帯の直書きが {k} 行あります"
                f"（既知は {want} 行）。"
                "`dc5lib/instance_eval.py` の `BAND` / `BAND_AREA` から"
                "読んでください（実装書 5）")
    for rel, want in sorted(KNOWN.items()):
        if rel not in got:
            problems.append(
                f"{rel} の帯の直書きが無くなりました（既知は {want} 行）。"
                "`tools/check_bands.py` の `KNOWN` から消してください")
        elif got[rel] < want:
            problems.append(
                f"{rel} の帯の直書きが {got[rel]} 行に減りました"
                f"（既知は {want} 行）。`KNOWN` も直してください")
    return problems


def main():
    import sys
    root = Path(__file__).resolve().parent.parent
    got = counts(root)
    print(f"帯の直書き（対象 {'/'.join(DIRS)} の .py）")
    for rel, k in sorted(got.items()):
        mark = "既知" if KNOWN.get(rel, 0) == k else "**増えた**"
        print(f"  {mark:<10}{rel:<44}{k} 行")
    if not got:
        print("  （1件も無い）")
    print(f"\n直書きしてよい場所 {len(ALLOWED)} 本（原本・測る側・照合する側）")
    problems = scan(root)
    if problems:
        print("\n問題:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\n帯の直書き: 既知どおり")


if __name__ == "__main__":
    main()
