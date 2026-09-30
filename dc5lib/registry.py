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
class Input:
    """モデルに入れる1つの入力。

    key      forward() に渡す順序を決める識別子（terrain / airphoto / geology ...）
    source   何のデータか（DEM / SAM / APM / GEO ...）
    channels チャネル数（DEM・SAMは1、航空写真は3）
    field    pkl の中の属性名。省略時は key から推定
    """
    key: str
    source: str
    channels: int = 1
    field: str = ""
    note: str = ""

    def __post_init__(self):
        if not self.field:
            self.field = {"terrain": "DEM", "airphoto": "AirPhoto"}.get(self.key, self.key)


@dataclass
class Condition:
    name: str
    arch: str                       # 実装クラス名
    family: str                     # Single / Early / Middle / Final / Attention / Transformer / Triple
    augment: bool
    dataset: dict                   # {"Hiroshima": "xxx.pkl", "Shimane": "yyy.pkl"}
    train: dict
    inputs: list = field(default_factory=list)   # [Input, ...]
    terrain: str = ""               # inputs から導出（互換のため残す）
    use_airphoto: bool = False      # 同上
    bg_ratio: float = 0.0
    status: str = "active"          # active / excluded
    note: str = ""
    label: str = ""
    dir: Path = field(default=None, repr=False)

    def __post_init__(self):
        # inputs が書かれていなければ、旧来の terrain / use_airphoto から組み立てる
        if not self.inputs:
            self.inputs = [Input(key="terrain", source=self.terrain or "DEM", channels=1)]
            if self.use_airphoto:
                self.inputs.append(Input(key="airphoto", source="APM", channels=3))
        else:
            self.inputs = [i if isinstance(i, Input) else Input(**i) for i in self.inputs]
        # 逆に inputs から terrain / use_airphoto を埋め直す（表示や層別で使う）
        for i in self.inputs:
            if i.key == "terrain":
                self.terrain = i.source
        self.use_airphoto = any(i.key == "airphoto" for i in self.inputs)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    @property
    def in_channels(self) -> list:
        return [i.channels for i in self.inputs]

    @property
    def weights(self) -> Path:
        return paths.weights_dir(self.name) / "best_model.pth"

    @property
    def checkpoint(self) -> Path:
        return paths.data_root() / "checkpoints" / self.name / "model_checkpoint.pth"

    def pkl_for(self, region_dir: str) -> Path:
        """region_dir は regions.yaml の dir（'Hiroshima' など）。"""
        if region_dir not in self.dataset:
            raise KeyError(
                f"{self.name} に {region_dir} のデータセットが定義されていません。"
                f"config.yaml の dataset に追加してください。"
                f"（定義済み: {list(self.dataset)}）")
        return paths.dataset_path(region_dir, self.dataset[region_dir])

    # 旧名
    def pkl(self, region_dir: str) -> Path:
        return self.pkl_for(region_dir)

    @property
    def dual(self) -> bool:
        return self.use_airphoto

    def display(self) -> str:
        if self.label:
            return self.label
        fam = {"Single": "単一入力", "Early": "Early", "Middle": "Middle", "Final": "Final",
               "Attention": "Attention", "Transformer": "TransUNet",
               "Triple": "Triple"}.get(self.family, self.family)
        if self.n_inputs == 1:
            inputs = self.inputs[0].source + " のみ"
        else:
            inputs = "+".join(i.source for i in self.inputs)
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
def _check_unicode_paths():
    """同じ名前が NFC と NFD の両方で追跡されていないか。

    macOS はファイル名を NFD で持つことがあり、git の core.precomposeunicode が
    NFC に変換する。両方が index に入ると、Linux では別ファイル、
    macOS / exFAT では同じファイルとして扱われ、pull が止まる。
    実際に一度これで SSD 側の pull が失敗した。
    """
    import collections
    import subprocess
    import unicodedata

    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=paths.REPO_ROOT, timeout=60).stdout.splitlines()
    except Exception:
        return []
    g = collections.defaultdict(list)
    for p in out:
        g[unicodedata.normalize("NFC", p)].append(p)
    problems = []
    for k, v in g.items():
        if len(v) > 1:
            problems.append(f"同じパスが Unicode 正規化違いで重複して追跡されています: {k}"
                            f"（Linux では別ファイル、macOS/exFAT では同じ扱いになり pull が止まります）")
    return problems


def check():
    """CI / コミット前に回す検査。異常があれば行を返す。"""
    problems = _check_unicode_paths()
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

        from . import regions as _regions
        known = set(r.dir for r in _regions.all_regions())
        for region in c.dataset:
            if region not in known:
                problems.append(f"{n}: dataset の地域 '{region}' が regions.yaml にありません")

        # inputs と pkl の食い違い。航空写真を入力に使うのに pkl に入っていなければ、
        # 学習中に AttributeError で落ちる。ファイル名から推定して先に止める。
        for region, fn in c.dataset.items():
            low = fn.lower()
            for key, token, label in (("airphoto", "_apm", "航空写真"),
                                      ("geology", "_geo", "地質図")):
                uses = any(i.key == key for i in c.inputs)
                if uses and token not in low:
                    problems.append(
                        f"{n}: inputs に{label}があるのに、{region} の pkl "
                        f"'{fn}' には含まれていないようです")

        if c.status == "active":
            try:
                from .models import MODEL_CLASSES, forward_arity
                if c.arch not in MODEL_CLASSES:
                    problems.append(f"{n}: モデルクラス {c.arch} が dc5lib/models.py に"
                                    f"登録されていません"
                                    f"（登録済み: {list(MODEL_CLASSES)}）")
                else:
                    want = forward_arity(c.arch)
                    if want != c.n_inputs:
                        problems.append(
                            f"{n}: 入力数が合いません。{c.arch}.forward は {want} 入力ですが、"
                            f"config の inputs は {c.n_inputs} 個です"
                            f"（{'+'.join(i.source for i in c.inputs)}）")
            except Exception:
                pass
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
    print(f"{'条件':<38}{'系統':<12}{'入力':<16}{'ch':<8}{'拡張':<5}{'重み':<6}{'状態'}")
    for c in conds:
        w = ("あり" if c.weights.exists() else "無し") if has_data else "SSD未接続"
        inputs = "+".join(i.source for i in c.inputs)
        ch = "+".join(str(i.channels) for i in c.inputs)
        print(f"{c.name:<38}{c.family:<12}{inputs:<16}{ch:<8}"
              f"{'あり' if c.augment else '—':<5}{w:<6}{c.status}")
    print(f"\n計 {len(conds)} 条件（うち解析対象 {sum(1 for c in conds if c.status=='active')}）")
