# -*- coding: utf-8 -*-
"""実験条件の一覧を `experiments/*/config.yaml` から組み立てる。

中央に条件のリストを持たない。**フォルダを1つ足せば条件が1つ増える。**
モデルが増えていく前提なので、どこか1か所を編集し忘れて壊れる作りを避けている。

    from dc5lib.registry import load_all, get
    for c in load_all():
        print(c.name, c.arch, c.weights)

    python -m dc5lib.registry          # 一覧と、重み・データの有無を表示
    python -m dc5lib.registry --check  # 命名の衝突などを検査（CI用。異常なら終了コード1）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import paths


@dataclass
class Condition:
    name: str
    arch: str                       # 実装クラス名
    family: str                     # Single / Early / Middle / Final / Attention / Transformer / Triple
    terrain: str                    # DEM / SAM
    use_airphoto: bool
    augment: bool
    bg_ratio: float
    dataset: dict                   # {"Hiroshima": "xxx.pkl", "Shimane": "yyy.pkl"}
    train: dict
    status: str = "active"          # active / excluded
    note: str = ""
    label: str = ""
    dir: Path = field(default=None, repr=False)

    @property
    def weights(self) -> Path:
        return paths.weights_dir(self.name) / "best_model.pth"

    @property
    def checkpoint(self) -> Path:
        return paths.data_root() / "checkpoints" / self.name / "model_checkpoint.pth"

    def pkl(self, region: str) -> Path:
        return paths.dataset_path(region, self.dataset[region])

    @property
    def dual(self) -> bool:
        return self.use_airphoto

    def display(self) -> str:
        if self.label:
            return self.label
        fam = {"Single": "単一入力", "Early": "Early", "Middle": "Middle", "Final": "Final",
               "Attention": "Attention", "Transformer": "TransUNet",
               "Triple": "Triple"}.get(self.family, self.family)
        inputs = self.terrain + ("+APM" if self.use_airphoto else " のみ")
        return f"{'拡張+' if self.augment else ''}{fam} {inputs}"


def load_one(config_path: Path) -> Condition:
    d = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    d.setdefault("status", "active")
    d.setdefault("note", "")
    d.setdefault("label", "")
    d.setdefault("bg_ratio", 0.0)
    d.setdefault("train", {})
    c = Condition(dir=config_path.parent, **d)
    if c.name != config_path.parent.name:
        raise ValueError(f"config の name とフォルダ名が違います: "
                         f"{c.name} vs {config_path.parent.name}")
    return c


def load_all(include_excluded=False):
    out = []
    for cfg in sorted(paths.experiments_dir().glob("*/config.yaml")):
        c = load_one(cfg)
        if c.status == "active" or include_excluded:
            out.append(c)
    return out


def get(name: str) -> Condition:
    return load_one(paths.experiment_dir(name) / "config.yaml")


# ------------------------------------------------------------------ 検査
def check():
    """CI / コミット前に回す検査。異常があれば行を返す。"""
    problems = []
    names = [p.name for p in sorted(paths.experiments_dir().iterdir()) if p.is_dir()]

    # exFAT と Windows は大文字小文字を区別しないが、Ubuntu の ext4 は区別する。
    # 大小だけ違うフォルダを作ると、OSをまたいだ瞬間に壊れる。
    lowered = {}
    for n in names:
        lowered.setdefault(n.lower(), []).append(n)
    for low, group in lowered.items():
        if len(group) > 1:
            problems.append(f"大文字小文字だけが違うフォルダ: {group} "
                            f"（Ubuntuでは別物、Windows/exFATでは同じ扱いになり壊れます）")

    for n in names:
        cfg = paths.experiment_dir(n) / "config.yaml"
        if not cfg.exists():
            problems.append(f"{n}: config.yaml がありません")
            continue
        try:
            c = load_one(cfg)
        except Exception as e:
            problems.append(f"{n}: config.yaml を読めません — {e}")
            continue
        if not any(paths.experiment_dir(n).glob("*.ipynb")):
            problems.append(f"{n}: ノートブックがありません")
        for bad in paths.experiment_dir(n).glob("*.pth"):
            problems.append(f"{n}: 重みがリポジトリ内にあります — {bad.name} は dc5-data へ")
    return problems


if __name__ == "__main__":
    if "--check" in sys.argv:
        problems = check()
        if problems:
            print("検査で問題が見つかりました:")
            for p in problems:
                print("  -", p)
            sys.exit(1)
        print("検査: 問題なし")
        sys.exit(0)

    has_data = paths.data_root(required=False) is not None
    conds = load_all(include_excluded=True)
    print(f"{'条件':<38}{'系統':<12}{'入力':<10}{'拡張':<5}{'重み':<6}{'状態'}")
    for c in conds:
        w = ("あり" if c.weights.exists() else "無し") if has_data else "SSD未接続"
        inputs = c.terrain + ("+APM" if c.use_airphoto else "")
        print(f"{c.name:<38}{c.family:<12}{inputs:<10}"
              f"{'あり' if c.augment else '—':<5}{w:<6}{c.status}")
    print(f"\n計 {len(conds)} 条件（うち解析対象 {sum(1 for c in conds if c.status=='active')}）")
