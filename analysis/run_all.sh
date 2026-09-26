#!/bin/bash
# 推論とタイル特徴量の算出を順番に回す（メモリを食うpklを同時に開かないよう直列）。
set -e
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
echo "===== 1/4 広島 推論 ====="; $PY scripts/run_inference.py --region hiroshima --verify
echo "===== 2/4 広島 タイル特徴量 ====="; $PY scripts/run_features.py --region hiroshima
echo "===== 3/4 島根 推論 ====="; $PY scripts/run_inference.py --region shimane --verify
echo "===== 4/4 島根 タイル特徴量 ====="; $PY scripts/run_features.py --region shimane
echo "===== 完了 ====="
