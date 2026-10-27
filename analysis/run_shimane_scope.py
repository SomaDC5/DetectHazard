# -*- coding: utf-8 -*-
"""島根の値を、評価範囲を揃えた補助指標と**並べずに**出す（G / 第6章）。

⚠ 主系統と補助を混ぜないこと
----------------------------
| | 測り方 | 位置づけ |
|---|---|---|
| **主系統** | `subset: all`（全 24,569 枚）| **これが基準。** 島根は未知の県に掛ける実運用の模倣で、**汎化性能としてはこちらが正しい。** 精度を上げる対象もこちら |
| 補助 | `subset: gtonly`（正解あり 9,679 枚）| **考察用。** 「島根の落ち込みは再現率側の現象」を言うためだけに使う |

**主系統の値は書き換えない。** これまで報告してきた島根の値はすべて
`subset: all` で、そのまま基準であり続ける。
補助は**別の表**として出す。**成績として掲げない。**
背景タイルの偽陽性は実運用では本当に出るので、それを外した値は
モデルの性質の診断であって、達成した精度ではない。

なぜ差が出るのか
----------------
広島は `subset: holdout` で**正解ありタイルしか評価していない**が、
島根は `subset: all` で**背景タイル 14,890 枚を含む**。
そこに出た予測は定義上すべて偽陽性になる（実測で 19.9 %、的中率 0.0003）。
したがって**適合率だけ**が比較できない。再現率は正解ありタイルにしか
正解が無いので、土俵は元から同じ。

使い方
------
    python analysis/run_shimane_scope.py
    python analysis/run_shimane_scope.py --conditions FinalFusion_SAM_APM ...

`*__pred.csv` が要る。無ければ
    python analysis/run_instance_eval.py --region shimane \
        --conditions <条件> --save-instances
で作る。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths

BAND = 0.0198          # 島根の箇所F のゆらぎ幅
COV_GT, COV_PRED = 0.5, 0.3


def measure(cond, region="shimane"):
    """主系統と補助を**別の辞書**にして返す。同じ行に混ぜない。"""
    cache = paths.cache_dir("analysis")
    fg = cache / "instances" / f"{cond}__{region}__gt.csv"
    fp = cache / "instances" / f"{cond}__{region}__pred.csv"
    if not (fg.exists() and fp.exists()):
        return None, None
    g = pd.read_csv(fg)
    p = pd.read_csv(fp)
    v = pd.read_csv(paths.REPO_ROOT / "provenance" / f"tile_vintage__{region}.csv")
    p = p.merge(v[["tile_no", "gt_pos"]], on="tile_no", how="left")
    p["hit"] = p.cov_pred >= COV_PRED

    # 再現率は両方で同じ。正解は正解ありタイルにしか無いため。
    r = float((g.coverage >= COV_GT).mean())

    def pack(sel, label):
        pr = float(sel.hit.mean())
        return dict(条件=cond, 範囲=label, 箇所R=r, 箇所P=pr,
                    箇所F=2 * r * pr / (r + pr) if (r + pr) else float("nan"),
                    予測数=len(sel))

    main = pack(p, "全 24,569 枚")
    aux = pack(p[p.gt_pos > 0], "正解あり 9,679 枚")
    return main, aux


DEFAULT = [
    "FinalFusion_SAM_APM",
    "FinalFusion_SAM_APM_OverlapTrain",
    "FinalFusion_SAM_APM_LU_OverlapTrain",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=DEFAULT)
    args = ap.parse_args()

    mains, auxes, missing = [], [], []
    for c in args.conditions:
        m, a = measure(c)
        if m is None:
            missing.append(c)
            continue
        mains.append(m)
        auxes.append(a)

    fmt = {"箇所R": "{:.4f}".format, "箇所P": "{:.4f}".format,
           "箇所F": "{:.4f}".format}

    print("=" * 66)
    print("【主系統】島根 = 全 24,569 枚（実運用の模倣）")
    print("  **これが基準。報告する成績はこちら。**")
    print("=" * 66)
    if mains:
        print(pd.DataFrame(mains).drop(columns=["範囲"])
              .to_string(index=False, formatters=fmt))
    print()

    print("-" * 66)
    print("【補助・考察用】評価範囲を広島に揃えた場合（正解あり 9,679 枚）")
    print("  **成績として掲げない。** 背景タイルの偽陽性は実運用では本当に出る。")
    print("  言えるのは「落ち込みが再現率側か適合率側か」だけ。")
    print("-" * 66)
    if auxes:
        print(pd.DataFrame(auxes).drop(columns=["範囲"])
              .to_string(index=False, formatters=fmt))
    print()

    if mains:
        print("【差の内訳】")
        print(f"  {'条件':<38}{'箇所P の差':>11}{'帯':>7}")
        for m, a in zip(mains, auxes):
            d = a["箇所P"] - m["箇所P"]
            print(f"  {m['条件']:<38}{d:>+11.4f}{d / BAND:>7.1f}")
        print()
        print("  再現率は両方で同じ（正解は正解ありタイルにしか無い）。")
        print("  **差が出るのは適合率だけ**であり、その分は評価範囲の差である。")

    if missing:
        print()
        print("※ 予測側の記録が無い条件:")
        for c in missing:
            print(f"   {c}")
        print("   python analysis/run_instance_eval.py --region shimane \\")
        print("       --conditions <条件> --save-instances")


if __name__ == "__main__":
    main()
