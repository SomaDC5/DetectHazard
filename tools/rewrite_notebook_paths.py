# -*- coding: utf-8 -*-
"""ノートブック内のハードコードされたパスを dc5lib 経由に書き換える。

研究室PC固有の C:\\Users\\hirok\\... や D:\\... が埋まっていると、
Ubuntu でも Mac でも動かない。ここを一度きりで潰す。

  before  path = "D:\\NewDatasModel\\DataSet\\Hiroshima\\hiroshima_sam_apm.pkl"
  after   path = dataset_path("Hiroshima", "hiroshima_sam_apm.pkl")

  before  save_path = "best_model.pth"        # カレントディレクトリに保存
  after   save_path = weights_path(CONDITION) # 外部SSDの dc5-data/weights/<条件>/

各ノートブックの先頭に共通の読み込みセルを1つ挿入する。出力は書き換えない。

    python tools/rewrite_notebook_paths.py --dry-run
    python tools/rewrite_notebook_paths.py --run
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import paths  # noqa: E402

MARK = "dc5-bootstrap"

BOOTSTRAP = '''# === dc5 共通設定（自動挿入: {mark}）=======================
# 重み・チェックポイント・データセットの場所は dc5lib が自動で解決する。
# 外部SSDを挿すだけでよく、OS ごとにパスを書き分ける必要はない。
#   weights_path(COND)     → dc5-data/weights/<条件>/best_model.pth
#   checkpoint_path(COND)  → dc5-data/checkpoints/<条件>/model_checkpoint.pth
#   dataset_path(地域, ファイル名) → dc5-data/datasets/<地域>/<ファイル名>
#   curve_path(COND)       → results/curves/<条件>.csv
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    raise RuntimeError("dc5lib が見つかりません。リポジトリの中で起動してください。")

from dc5lib.paths import weights_path, checkpoint_path, dataset_path, curve_path
from dc5lib.data import load_dataset, dem_dataset, photo_dataset

CONDITION = "{cond}"
print("条件:", CONDITION)
print("重み:", weights_path(CONDITION))
'''

# (正規表現, 置換) の順に適用する
RULES = [
    # 研究室PC固有の sys.path.append は不要（load_any_dataset は dc5lib に取り込み済み）
    (re.compile(r'^\s*sys\.path\.append\(\s*r?["\'][^"\']*MakeDataSet[^"\']*["\']\s*\)[^\n]*$', re.M),
     '# sys.path.append(...MakeDataSet)  # dc5lib.data に取り込んだため不要'),
    (re.compile(r'^\s*from\s+load_any_dataset\s+import[^\n]*$', re.M),
     '# from load_any_dataset import ...  # 冒頭の dc5 共通設定セルで読み込み済み'),
    # D:\NewDatasModel\DataSet\<地域>\<ファイル>
    (re.compile(r'["\'][A-Za-z]:\\\\?NewDatasModel\\\\?DataSet\\\\?'
                r'(Hiroshima|Shimane)\\\\?([A-Za-z0-9_.\-]+\.pkl)["\']'),
     r'dataset_path("\1", "\2")'),
    # C:\Users\hirok\...\MakeDataSet\<ファイル>.pkl （広島データが置いてあった旧パス）
    (re.compile(r'["\'][A-Za-z]:\\\\?Users\\\\?hirok[^"\']*MakeDataSet\\\\?'
                r'([A-Za-z0-9_.\-]+\.pkl)["\']'),
     r'dataset_path("Hiroshima", "\1")'),
    # 重み・チェックポイント・学習曲線（カレントディレクトリ依存をやめる）
    (re.compile(r'(?<![\w.])"best_model\.pth"'), 'weights_path(CONDITION)'),
    (re.compile(r"(?<![\w.])'best_model\.pth'"), 'weights_path(CONDITION)'),
    (re.compile(r'(?<![\w.])"model_checkpoint\.pth"'), 'checkpoint_path(CONDITION)'),
    (re.compile(r"(?<![\w.])'model_checkpoint\.pth'"), 'checkpoint_path(CONDITION)'),
    (re.compile(r'(?<![\w.])"losses\.csv"'), 'curve_path(CONDITION)'),
    (re.compile(r"(?<![\w.])'losses\.csv'"), 'curve_path(CONDITION)'),
]


def rewrite(nb_path: Path, cond: str, dry=True):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed, counts = 0, {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if MARK in src:
            continue
        new = src
        for pat, rep in RULES:
            new, n = pat.subn(rep, new)
            if n:
                counts[pat.pattern[:34]] = counts.get(pat.pattern[:34], 0) + n
        if new != src:
            changed += 1
            if not dry:
                c["source"] = new.splitlines(keepends=True)

    has_boot = any(MARK in "".join(c["source"]) for c in nb["cells"])
    if not has_boot and not dry:
        nb["cells"].insert(0, {
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [],
            "source": BOOTSTRAP.format(cond=cond, mark=MARK).splitlines(keepends=True)})
    if not dry:
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    return changed, counts, (not has_boot)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()

    total = {}
    for cond_dir in sorted(paths.experiments_dir().iterdir()):
        if not cond_dir.is_dir():
            continue
        for nb in sorted(cond_dir.glob("*.ipynb")):
            ch, counts, boot = rewrite(nb, cond_dir.name, dry=a.dry_run)
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
            print(f"{cond_dir.name:<38} {nb.name:<40} セル{ch:3d} "
                  f"{'＋冒頭セル' if boot else ''}")
    print("\n書き換えの内訳")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"  {v:4d}  {k}")
    print("\n--dry-run（何も書いていません）" if a.dry_run else "\n書き換えました。")


if __name__ == "__main__":
    main()
