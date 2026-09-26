# -*- coding: utf-8 -*-
"""コミット前の検査。異常があれば終了コード 1。

  - 実験フォルダ名が大文字小文字だけ違っていないか（Ubuntu と Windows で壊れる）
  - config.yaml があるか、フォルダ名と name が一致しているか
  - 重みがリポジトリ内に紛れ込んでいないか
  - metrics.csv が evals/ と食い違っていないか
"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import registry, results, paths  # noqa: E402

problems = registry.check()

# metrics.csv が最新か
csv_path = paths.results_dir() / "metrics.csv"
if csv_path.exists():
    before = csv_path.read_bytes()
    results.build_metrics_csv()
    if csv_path.read_bytes() != before:
        problems.append("metrics.csv が evals/ と食い違っています（作り直しました）")

# git に大きなファイルが載っていないか
try:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=paths.REPO_ROOT).stdout.split()
    for f in files:
        p = paths.REPO_ROOT / f
        if p.exists() and p.stat().st_size > 20 * 1024 * 1024:
            problems.append(f"20MBを超えるファイルが追跡されています: {f} "
                            f"({p.stat().st_size/2**20:.0f}MB)")
except Exception:
    pass

if problems:
    print("検査で問題が見つかりました:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("検査: 問題なし")
