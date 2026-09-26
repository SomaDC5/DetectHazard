# -*- coding: utf-8 -*-
"""results/evals/*.json を結合して results/metrics.csv を作り直す。

metrics.csv が git でコンフリクトしたら、手で直さずにこれを実行する。
書き出す順序は決定的なので、どのPCで作っても同じ内容になる。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import results  # noqa: E402

if __name__ == "__main__":
    out = results.build_metrics_csv()
    n = sum(1 for _ in open(out, encoding="utf-8")) - 1
    print(f"{out}  ({n} 行)")
