#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
echo "===== 広島 背景タイル 推論 ====="; $PY scripts/run_inference.py --region hiroshima_bg
echo "===== 広島 背景タイル 特徴量 ====="; $PY scripts/run_features.py --region hiroshima_bg
echo "===== 完了 ====="
