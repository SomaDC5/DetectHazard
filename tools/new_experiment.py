# -*- coding: utf-8 -*-
"""新しい実験条件の雛形を作る。**学習コードを書く前に実行する。**

2台のPCで並行して学習するので、条件名がかぶると後から直すのが面倒になる
（フォルダ名・config.yaml の name・ノートブックの CONDITION・SSD上の
weights/<条件名>/ の4か所）。名前を決めるこの瞬間に、相手と衝突しないかを
確かめてしまうのが確実。

このスクリプトは
  1. git fetch して、ローカルにも **他のブランチにも** 同名が無いか調べる
  2. 大文字小文字だけ違う名前も弾く（Ubuntu と Windows で壊れるため）
  3. config.yaml を作り、雛形のノートブックをコピーする
  4. 「先に config.yaml だけ push して名前を予約する」コマンドを表示する

    python tools/new_experiment.py MiddleFusion_DEM_APM_bg30 \\
        --from MIddleFusion_DEM_APM --bg-ratio 0.3 \\
        --note "背景タイルを3割混ぜた版"

    python tools/new_experiment.py --list        # いま使われている名前を一覧
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from dc5lib import paths, registry  # noqa: E402


def _git(*args, cwd=None):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=cwd or paths.REPO_ROOT, timeout=60)
        return r.stdout
    except Exception:
        return ""


def names_everywhere(fetch=True):
    """ローカルと、すべてのリモートブランチで使われている条件名を集める。

    相手のPCがまだマージしていないブランチで使っている名前も拾いたい。
    """
    local = {p.name for p in paths.experiments_dir().iterdir() if p.is_dir()}
    if fetch:
        print("  git fetch 中 …", flush=True)
        _git("fetch", "--all", "--quiet")
    remote = set()
    refs = _git("for-each-ref", "--format=%(refname)", "refs/remotes").split()
    for ref in refs:
        if ref.endswith("/HEAD"):
            continue
        out = _git("ls-tree", "-d", "--name-only", ref, "experiments/")
        for line in out.splitlines():
            n = line.strip().rstrip("/").split("/")[-1]
            if n:
                remote.add(n)
    return local, remote, refs


def check_name(name, local, remote):
    problems = []
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        problems.append("名前に使えるのは英数字と _ - だけです")
    if name in local:
        problems.append(f"ローカルに同名の条件があります: experiments/{name}")
    if name in remote:
        problems.append(f"リモートのブランチに同名の条件があります: {name}"
                        f"（相手のPCが使っている可能性）")
    # 大文字小文字だけ違うものは、Ubuntu と Windows/exFAT で扱いが変わり壊れる
    for other in local | remote:
        if other != name and other.lower() == name.lower():
            problems.append(f"大文字小文字だけが違う条件があります: {other}"
                            f"（Ubuntuでは別物、Windows/exFATでは同じ扱いになり壊れます）")
    return problems


TEMPLATE = {
    "name": None, "arch": "MultiEncoderUNet", "family": "Final",
    "inputs": [
        {"key": "terrain", "source": "SAM", "channels": 1, "field": "DEM",
         "note": "傾斜角マップ(度)"},
        {"key": "airphoto", "source": "APM", "channels": 3, "field": "AirPhoto",
         "note": "航空写真 RGB"},
    ],
    "augment": False, "bg_ratio": 0.0, "status": "active", "label": "", "note": "",
    "dataset": {"Hiroshima": "hiroshima_sam_apm.pkl", "Shimane": "shimane_sam_apm.pkl"},
    "train": {"epochs": 1000, "batch_size": 32, "lr": 1.0e-4, "seed": 42,
              "loss": {"type": "focal_tversky", "alpha": 0.7, "beta": 0.3, "gamma": 0.75},
              "augment_params": None},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", help="新しい条件名")
    ap.add_argument("--from", dest="base", help="この条件をひな形にする（config とノートブック）")
    ap.add_argument("--arch", help="モデルクラス名")
    ap.add_argument("--bg-ratio", type=float, help="学習に混ぜる背景タイルの割合")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true", help="使われている名前を一覧するだけ")
    ap.add_argument("--no-fetch", action="store_true", help="git fetch を省く（オフライン時）")
    a = ap.parse_args()

    local, remote, refs = names_everywhere(fetch=not a.no_fetch)

    if a.list or not a.name:
        print(f"\nローカル（experiments/）{len(local)} 件")
        for n in sorted(local):
            print("   ", n)
        only_remote = sorted(remote - local)
        if only_remote:
            print(f"\nリモートのブランチにだけある {len(only_remote)} 件"
                  f"（相手のPCが作業中の可能性）")
            for n in only_remote:
                print("   ", n)
        print(f"\n参照したリモートブランチ: {len(refs)} 本")
        if not a.name:
            print("\n条件名を指定すると雛形を作ります。")
        return

    problems = check_name(a.name, local, remote)
    if problems:
        print(f"\n「{a.name}」は使えません。")
        for p in problems:
            print("  -", p)
        print("\n別の名前にしてください。")
        sys.exit(1)

    print(f"\n「{a.name}」は使われていません。雛形を作ります。")

    # config を作る
    if a.base:
        base_cfg = paths.experiment_dir(a.base) / "config.yaml"
        if not base_cfg.exists():
            print(f"ひな形が見つかりません: {base_cfg}")
            sys.exit(1)
        cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    else:
        cfg = dict(TEMPLATE)
    cfg["name"] = a.name
    if a.arch:
        cfg["arch"] = a.arch
    if a.bg_ratio is not None:
        cfg["bg_ratio"] = a.bg_ratio
    if a.augment:
        cfg["augment"] = True
    cfg["note"] = a.note
    cfg["status"] = "active"

    d = paths.experiment_dir(a.name)
    d.mkdir(parents=True, exist_ok=False)
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        f.write("# 実験条件の定義。フォルダを増やせば条件が増える。\n"
                "# 重みは dc5-data/weights/<name>/ にあり、ここには置かない。\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"  作成: experiments/{a.name}/config.yaml")

    # ノートブックをコピー
    if a.base:
        for nb in sorted((paths.experiment_dir(a.base)).glob("*.ipynb")):
            dst = d / nb.name
            shutil.copyfile(nb, dst)
            # 冒頭セルの CONDITION を差し替える
            t = dst.read_text(encoding="utf-8")
            t = t.replace(f'CONDITION = \\"{a.base}\\"', f'CONDITION = \\"{a.name}\\"')
            dst.write_text(t, encoding="utf-8")
            print(f"  複製: experiments/{a.name}/{nb.name}（CONDITION を差し替え）")

    print(f"""
次にやること

  1. 名前を予約する（学習を始める前に。これで相手とぶつからない）
       git add experiments/{a.name}
       git commit -m "{a.name} の条件を追加"
       git push

  2. 検査
       python tools/check.py

  3. ノートブックを編集して学習
       重みは自動的に dc5-data/weights/{a.name}/ に保存されます
""")
    if not a.base:
        print("  ※ --from <既存の条件> を付けると、config とノートブックを"
              "複製して始められます。")


if __name__ == "__main__":
    main()
