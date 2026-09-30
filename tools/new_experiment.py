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
from dc5lib.losses import DEFAULTS as LOSS_DEFAULTS  # noqa: E402

SOURCE_CHANNELS = {"DEM": 1, "SAM": 1, "APM": 3, "GEO": 3}
SOURCE_KEY = {"DEM": "terrain", "SAM": "terrain", "APM": "airphoto", "GEO": "geology"}
SOURCE_FIELD = {"DEM": "DEM", "SAM": "DEM", "APM": "AirPhoto", "GEO": "Geo"}


def parse_inputs(spec):
    """'SAM+APM+GEO' を inputs のリストにする。"""
    out = []
    for token in spec.split("+"):
        t = token.strip().upper()
        if t not in SOURCE_CHANNELS:
            raise SystemExit(f"未知の入力: {t}（使えるのは {list(SOURCE_CHANNELS)}）")
        out.append({"key": SOURCE_KEY[t], "source": t,
                    "channels": SOURCE_CHANNELS[t], "field": SOURCE_FIELD[t]})
    return out


def _git(*args, cwd=None):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True,
                           cwd=cwd or paths.REPO_ROOT, timeout=60)
        return r.stdout
    except Exception:
        return ""


def _names_in(ref):
    out = _git("ls-tree", "-d", "--name-only", ref, "experiments/")
    return {line.strip().rstrip("/").split("/")[-1]
            for line in out.splitlines() if line.strip()}


def names_everywhere(fetch=True):
    """条件名がどこで使われているかを集める。

    「自分のブランチの履歴にある」のと「相手のPCのブランチにある」のは
    意味が違う（前者は自分で消せる）ので、分けて返す。
    """
    local = {p.name for p in paths.experiments_dir().iterdir() if p.is_dir()}
    if fetch:
        print("  git fetch 中 …", flush=True)
        _git("fetch", "--all", "--quiet")

    here = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    mine = _names_in("HEAD")                     # 自分のブランチの履歴
    others, refs = set(), []
    for ref in _git("for-each-ref", "--format=%(refname)", "refs/remotes").split():
        if ref.endswith("/HEAD"):
            continue
        refs.append(ref)
        names = _names_in(ref)
        if ref.endswith("/" + here):
            mine |= names                        # 自分のブランチのリモート側
        else:
            others |= names
    return local, mine, others, refs


def check_name(name, local, mine, others):
    problems, hints = [], []
    if not name or not all(c.isalnum() or c in "_-" for c in name):
        problems.append("名前に使えるのは英数字と _ - だけです")
    if name in local:
        problems.append(f"ローカルの作業ツリーに同名の条件があります: experiments/{name}")
    if name in mine and name not in local:
        problems.append(f"自分のブランチの履歴に同名の条件が残っています: {name}")
        hints.append("  （作業ツリーからは消えているが、コミットには残っている状態）")
        hints.append(f"  消してよければ: git rm -r --cached experiments/{name} && "
                     f"git commit -m 'remove {name}' && git push")
    elif name in mine:
        hints.append("  （自分のブランチにあるものなので、消すか名前を変えれば使えます）")
    if name in others:
        problems.append(f"別のブランチに同名の条件があります: {name}"
                        f"（他のPCが作業中の可能性）")
    for other in local | mine | others:
        if other != name and other.lower() == name.lower():
            problems.append(f"大文字小文字だけが違う条件があります: {other}"
                            f"（Ubuntuでは別物、Windows/exFATでは同じ扱いになり壊れます）")
    return problems, hints


def patch_notebook_loss(nb_path: Path, cfg: dict):
    """ノートブックの損失を config 駆動に差し替える。

    もとは `criterion = FocalTverskyLoss(alpha=0.7, ...)` と直書きされている。
    これを `criterion, NEEDS_WEIGHT = build_loss(CONDITION)` にすると、
    config.yaml を変えるだけで損失を切り替えられる。
    """
    import json
    import re

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    pat = re.compile(r"^criterion\s*=\s*FocalTverskyLoss\([^)]*\)\s*$", re.M)
    repl = ("criterion, NEEDS_WEIGHT = build_loss(CONDITION)  "
            "# 損失は config.yaml の train.loss で決まる\n"
            "print('損失:', describe(CONDITION))")
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        new, k = pat.subn(repl, src)
        if k:
            c["source"] = new.splitlines(keepends=True)
            n += k
    if n:
        # 冒頭セルに build_loss の読み込みを足す
        first = "".join(nb["cells"][0]["source"])
        if "build_loss" not in first:
            first = first.replace(
                "from dc5lib.data import load_dataset, dem_dataset, photo_dataset",
                "from dc5lib.data import load_dataset, dem_dataset, photo_dataset\n"
                "from dc5lib.losses import build_loss, describe, weight_params")
            nb["cells"][0]["source"] = first.splitlines(keepends=True)

    needs = (cfg.get("train", {}).get("loss", {}).get("type", "")
             == "instance_weighted_focal_tversky")
    if needs:
        single = len(cfg.get("inputs", [])) <= 1
        body = TODO_INSTANCE_LOSS.format(
            getitem=("return image, mask, w" if single
                     else "return (dem, air), mask, w"),
            loop=("for images, masks, weights in train_loader:\n"
                  "    loss = criterion(model(images), masks, weights)" if single
                  else "for (dem, air), masks, weights in train_loader:\n"
                       "    loss = criterion(model(dem, air), masks, weights)"),
            mask_src=("dataset2.Mask" if single else "dataset2.Mask"))
        nb["cells"].insert(1, {
            "cell_type": "markdown", "metadata": {},
            "source": body.splitlines(keepends=True)})
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    return n


TODO_INSTANCE_LOSS = """## ⚠ 箇所正規化損失を使うために、あと3箇所だけ手を入れる

`criterion` は config 駆動に差し替え済み。ただしこの損失は**重みマップ**を
必要とする（`NEEDS_WEIGHT == True`）。次の3つを入れてから学習を始めること。
詳しい理由と背景は `docs/使い方.md` と `docs/FSS_analysis_参考/` を参照。

**① 分割セルの直後で、重みマップを事前計算する**

```python
from dc5lib.instance_loss import precompute_weights
wp = weight_params(CONDITION)          # config の a0 / p / center_lam
W_ALL = precompute_weights({mask_src}.numpy(), **wp)   # float16, 約485MB
```

`train_test_split` に `torch.from_numpy(W_ALL)` も一緒に渡して、
`w_train / w_val / w_test` を作る。

**② JointTransform に weight を通す**（マスクと同じ変換、最近傍補間）

```python
mask   = TF.affine(mask,   ..., interpolation=TF.InterpolationMode.NEAREST)
weight = TF.affine(weight, ..., interpolation=TF.InterpolationMode.NEAREST)
```

**③ Dataset が重みも返し、学習ループで渡す**

```python
{getitem}          # Dataset.__getitem__
...
{loop}
```

検証・テストでは重みを使わない（`criterion(out, masks)`）。
モデル選択は従来どおり検証の画素F値のままにして、箇所F値は別に記録する。
"""


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
    ap.add_argument("--arch", help="モデルクラス名（dc5lib/models.py に登録されているもの）")
    ap.add_argument("--family", help="図表のグループ分け: Single/Early/Middle/Final/Attention/Transformer")
    ap.add_argument("--inputs", help="入力の構成。例 SAM / SAM+APM / SAM+APM+GEO")
    ap.add_argument("--bg-ratio", type=float, help="学習に混ぜる背景タイルの割合")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--loss", choices=sorted(LOSS_DEFAULTS),
                    help="損失の種類。既定は元の条件のまま")
    ap.add_argument("--alpha", type=float, help="損失: False Negative(見逃し)の重み")
    ap.add_argument("--beta", type=float, help="損失: False Positive(過検出)の重み")
    ap.add_argument("--gamma", type=float, help="損失: Focal の指数")
    ap.add_argument("--a0", type=int, help="箇所正規化: 面積の下限")
    ap.add_argument("--p", type=float, help="箇所正規化: 重みの強さ(0で無効, 1.0で完全)")
    ap.add_argument("--center-lam", type=float, help="箇所正規化: 中心重みの強さ(まず0)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--note", default="")
    ap.add_argument("--list", action="store_true", help="使われている名前を一覧するだけ")
    ap.add_argument("--no-fetch", action="store_true", help="git fetch を省く（オフライン時）")
    a = ap.parse_args()

    local, mine, others, refs = names_everywhere(fetch=not a.no_fetch)

    if a.list or not a.name:
        print(f"\nローカル（experiments/）{len(local)} 件")
        for n in sorted(local):
            print("   ", n)
        stale = sorted(mine - local)
        if stale:
            print(f"\n自分のブランチの履歴にだけ残っている {len(stale)} 件")
            for n in stale:
                print("   ", n)
        only_others = sorted(others - local - mine)
        if only_others:
            print(f"\n別のブランチにだけある {len(only_others)} 件"
                  f"（他のPCが作業中の可能性）")
            for n in only_others:
                print("   ", n)
        print(f"\n参照したリモートブランチ: {len(refs)} 本")
        if not a.name:
            print("\n条件名を指定すると雛形を作ります。")
        return

    problems, hints = check_name(a.name, local, mine, others)
    if problems:
        print(f"\n「{a.name}」は使えません。")
        for p in problems:
            print("  -", p)
        for h in hints:
            print(h)
        print("\n別の名前にするか、上の方法で消してください。")
        sys.exit(1)

    print(f"\n「{a.name}」は使われていません。雛形を作ります。")

    # arch の妥当性は、作ってから check.py で気づくより、ここで言うほうが早い
    if a.arch:
        from dc5lib.models import MODEL_CLASSES
        if a.arch not in MODEL_CLASSES:
            print(f"\n  ! モデルクラス「{a.arch}」は dc5lib/models.py に登録されていません。")
            print(f"    登録済み: {', '.join(MODEL_CLASSES)}")
            print(f"    このまま作れますが、学習の前に models.py にクラスを追加して")
            print(f"    MODEL_CLASSES に登録してください（python -m dc5lib.models で一覧）。")

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
    if a.family:
        cfg["family"] = a.family
    if a.inputs:
        cfg["inputs"] = parse_inputs(a.inputs)
    if a.bg_ratio is not None:
        cfg["bg_ratio"] = a.bg_ratio
    if a.augment:
        cfg["augment"] = True
    cfg["note"] = a.note
    cfg["status"] = "active"

    # ひな形が旧プロトコル（検証分割なし・テストF値でベスト選択）なら、
    # そのまま引き継ぐと同じ欠陥を繰り返すので、正しい分割に直す。
    tr = cfg.setdefault("train", {})
    if "検証分割なし" in str(tr.get("split", "")) or \
       "テストデータのF値" in str(tr.get("model_selection", "")):
        tr["split"] = ("6:2:2 (train_test_split test_size=0.2 -> 0.25, "
                       "random_state=42)")
        tr["model_selection"] = "検証データのF値が最良のエポック"
        print("  ! ひな形が旧プロトコル（検証分割なし）だったので、"
              "6:2:2 + 検証でのベスト選択に直しました")
        print("    → ノートブック側も train/val/test の3分割に直す必要があります")

    # 損失の指定
    train = cfg.setdefault("train", {})
    if a.epochs is not None:
        train["epochs"] = a.epochs
    if a.seed is not None:
        train["seed"] = a.seed
    loss = dict(train.get("loss") or {})
    if a.loss:
        loss = {"type": a.loss, **LOSS_DEFAULTS[a.loss]}
    for k in ("alpha", "beta", "gamma", "a0", "p"):
        v = getattr(a, k)
        if v is not None:
            loss[k] = v
    if a.center_lam is not None:
        loss["center_lam"] = a.center_lam
    if loss:
        train["loss"] = loss

    # 入力数とモデルの forward が食い違っていたら、作る前に言う
    try:
        from dc5lib.models import MODEL_CLASSES, forward_arity
        arch = cfg.get("arch")
        if arch in MODEL_CLASSES:
            want = forward_arity(arch)
            got = len(cfg.get("inputs", []))
            if got and want != got:
                srcs = "+".join(i["source"] for i in cfg["inputs"])
                print(f"\n  ! 入力数が合いません。{arch}.forward は {want} 入力ですが、"
                      f"inputs は {got} 個（{srcs}）です。")
                print(f"    --inputs か --arch を見直してください。")
                sys.exit(1)
    except SystemExit:
        raise
    except Exception:
        pass

    # inputs を変えたのに dataset がひな形のままだと、余計なものを読み込む。
    # 致命的ではないので警告にとどめる。
    if a.inputs:
        keys = {i["key"] for i in cfg.get("inputs", [])}
        for region, fn in (cfg.get("dataset") or {}).items():
            low = fn.lower()
            for key, token, label in (("airphoto", "_apm", "航空写真"),
                                      ("geology", "_geo", "地質図")):
                if token in low and key not in keys:
                    print(f"\n  ! {region} の pkl '{fn}' は{label}を含みますが、"
                          f"inputs には入っていません。")
                    print(f"    読み込みが無駄に重くなります。"
                          f"config.yaml の dataset を見直してください。")
                    break

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
            msg = "CONDITION を差し替え"
            n = patch_notebook_loss(dst, cfg)
            if n:
                msg += f" / 損失を config 駆動に差し替え（{n}箇所）"
            print(f"  複製: experiments/{a.name}/{nb.name}（{msg}）")

    # 何が出来たかをその場で見せる。指定し忘れ（行継続の切れなど）に気づけるように。
    try:
        from dc5lib.registry import get as _get
        from dc5lib.losses import describe as _describe
        c = _get(a.name)
        print("\n作成された条件")
        print(f"  arch     {c.arch}")
        print(f"  family   {c.family}")
        print(f"  inputs   {'+'.join(i.source for i in c.inputs)} "
              f"({'+'.join(str(i.channels) for i in c.inputs)} ch)")
        print(f"  augment  {c.augment}")
        print(f"  bg_ratio {c.bg_ratio}")
        print(f"  epochs   {c.train.get('epochs')}   seed {c.train.get('seed')}")
        print(f"  loss     {_describe(a.name)}")
        for region, fn in c.dataset.items():
            print(f"  dataset  {region}: {fn}")
        base_loss = (yaml.safe_load(
            (paths.experiment_dir(a.base) / "config.yaml").read_text(encoding="utf-8"))
            .get("train", {}).get("loss", {}).get("type")) if a.base else None
        if a.loss is None and base_loss:
            print(f"\n  ※ --loss を指定していないので、ひな形の {base_loss} を"
                  f"引き継いでいます。")
            print(f"    違うものにしたい場合は --loss を付け直すか、"
                  f"config.yaml の train.loss を直接編集してください。")
    except Exception as e:
        print(f"（確認表示に失敗: {e}）")

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
