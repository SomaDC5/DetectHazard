#!/bin/bash
# 箇所数評価を全条件で。結果が早く見たい順（広島 → 島根 → 広島の背景タイル）に直列実行。
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
echo "===== 1/3 広島 ====="   ; $PY scripts/run_instance_eval.py --region hiroshima --save-instances
echo "===== 2/3 島根 ====="   ; $PY scripts/run_instance_eval.py --region shimane
echo "===== 3/3 広島 背景 =====" ; $PY scripts/run_instance_eval.py --region hiroshima_bg
echo "===== 完了 ====="
