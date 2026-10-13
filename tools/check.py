# -*- coding: utf-8 -*-
"""コミット前の検査。異常があれば終了コード 1。

  - 実験フォルダ名が大文字小文字だけ違っていないか（Ubuntu と Windows で壊れる）
  - config.yaml があるか、フォルダ名と name が一致しているか
  - 重みがリポジトリ内に紛れ込んでいないか
  - metrics.csv が evals/ と食い違っていないか
  - ensembles.yaml のメンバーが実在し、bg_ratio が揃っているか
  - 学習ノートブックが config.yaml を無視して直書きしていないか
"""
import json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import registry, results, paths, ensembles  # noqa: E402

problems = registry.check() + ensembles.check()

# 学習ノートブックの直書き。config を変えても効かないまま学習してしまうため
# （FinalFusion_SAM_APM_OverlapTrain で早期終了が効かず 1000 エポック回った）。
# すでに学習を終えた条件は、当時の記録として残すので対象外にする。
_HARDCODE = [
    (r"^num_epochs\s*=\s*\d+", "エポック数が直書き"),
    (r"^EARLY_STOP_PATIENCE\s*=\s*\d+", "早期終了が直書き"),
    (r"lr=1e-4", "学習率が直書き"),
]
_TRAINED = {p.parent.name for p in (paths.results_dir() / "curves").glob("*.csv")}
for d in sorted(Path(paths.experiments_dir()).iterdir()):
    nb = d / "Keisya_airPhoto_final6.ipynb"
    if not nb.exists() or (paths.results_dir() / "curves" / f"{d.name}.csv").exists():
        continue        # 未作成、または学習済み（当時の記録として触らない）
    try:
        cells = json.load(open(nb, encoding="utf-8"))["cells"]
    except Exception:
        continue
    src = "\n".join("".join(c["source"]) for c in cells if c.get("cell_type") == "code")
    hits = [msg for pat, msg in _HARDCODE if re.search(pat, src, re.M)]
    if "train_cfg(CONDITION)" not in src:
        hits.append("config を読んでいない（dc5lib.traincfg を使う）")
    if hits:
        problems.append(f"{d.name} のノートブック: {' / '.join(hits)}"
                        "（tools/wire_config.py で直せます）")

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
