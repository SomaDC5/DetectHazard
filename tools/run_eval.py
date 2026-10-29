# -*- coding: utf-8 -*-
"""学習済みの重みを**素のタイル**で測り、`results/evals/` に記録する。

なぜ要るか
----------
`dc5lib.evaluate.evaluate` は数字を出すだけで、長らく記録していなかった。
その結果 `44.7` に書いた素のタイルの箇所F（0.6935／0.6610）の出どころが
**使い捨てのプロセスの出力にしか無い**状態になった
（`CLAUDE.md` の「出した計算をコードに残す」に反する）。

    python tools/run_eval.py <条件名> [<条件名> ...]
    python tools/run_eval.py <条件名> --region Shimane
    python tools/run_eval.py <条件名> --no-record      # 記録せず見るだけ

**オーバーラップ推論ではない。** `42`〜`44` の数字はそちらで、
`analysis/run_overlap_eval.py --record` が担当する。ここは
学習ノートブックの最後に出る値（素のタイル）と同じものである。

同じ重みを測り直したときは `run_id` が同じになるので、
**`results/evals/` の行は増えず、古い記録が置き換わる**
（`dc5lib/evaluate.py` の `_run_id_for`）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import ckpt                                       # noqa: E402
from dc5lib.evaluate import evaluate                          # noqa: E402
from dc5lib.pipeline import build_loaders, seed_everything    # noqa: E402
from dc5lib.registry import get                               # noqa: E402
from dc5lib.traincfg import train_cfg                         # noqa: E402


def split_key(cond, region):
    """同じ pkl・同じ分割なら loaders を使い回せる。6GB の pkl を読み直さない。"""
    sp = cond.split or {}
    return (region, (cond.dataset or {}).get(region),
            sp.get("kind"), sp.get("test_size"), sp.get("val_size"),
            sp.get("random_state"), float(getattr(cond, "bg_ratio", 0) or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--region", default="Hiroshima")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--no-instance", action="store_true")
    a = ap.parse_args()

    ld = key = None
    for i, name in enumerate(a.conditions):
        cond = get(name)
        print("=" * 78)
        print(cond.display())
        print(f"  protocol {cond.protocol} / ensemble {cond.ensemble}"
              f" / seeded {cond.seeded}")
        print("=" * 78, flush=True)
        if not cond.weights.exists():
            print(f"  **重みがありません**: {cond.weights}")
            continue
        print(ckpt.describe(cond.weights), flush=True)

        k = split_key(cond, a.region)
        if ld is None or k != key:
            seed_everything(train_cfg(cond).seed)
            ld, key = build_loaders(cond, a.region), k
        else:
            print(f"  （{a.conditions[i - 1]} と同じ pkl・同じ分割なので"
                  f" loaders を使い回します）", flush=True)

        t0 = time.time()
        evaluate(cond, ld, a.split, record=not a.no_record,
                 instance=not a.no_instance)
        print(f"  {(time.time() - t0) / 60:.1f} 分", flush=True)


if __name__ == "__main__":
    main()
