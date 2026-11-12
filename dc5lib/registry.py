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

import re
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



# 構造の短い呼び名。ここに無い構造はクラス名から機械的に作るので、
# 新しいモデルを足したときに書き忘れても表示名は壊れない。
# 値が "" のものは「その系統の標準の構造」で、表示名には出さない。
ARCH_JA = {
    "UNet": "", "EarlyFusionUNet": "", "MiddleFusionUNet": "",
    "MultiEncoderUNet": "", "AttentionMultiEncoderUNet": "", "TransUNetDual": "",
    "FullSkipMultiEncoderUNet": "全スケール結合",
    "AllSkipDeepSupMultiEncoderUNet": "全スキップ+深層監督",
    "DeepSupMultiEncoderUNet": "深層監督",
    "ASPPDeepSupMultiEncoderUNet": "ASPP+深層監督",
    "ASPPStride4MultiEncoderUNet": "ASPP+ストライド4",
    "ASPPStride8MultiEncoderUNet": "ASPP+ストライド8",
    "SupAttnMultiEncoderUNet": "監督Attention",
    "SupAttnPerBranchMultiEncoderUNet": "監督Attention(枝別)",
    # 族の比較用。符号化器は MultiEncoderUNet と同じで、頭だけが違う。
    "SegFormerDual": "SegFormer",
    "FCNDual": "FCN-8s",
    "PSPNetDual": "PSPNet",
    "DeepLabV3Dual": "DeepLabv3",
    "MLPDecoderDual": "全MLPデコーダ",
}


def arch_tag(arch: str) -> str:
    """構造の短い呼び名。未登録ならクラス名から接尾辞を落として使う。"""
    if arch in ARCH_JA:
        return ARCH_JA[arch]
    return re.sub(r"(MultiEncoder)?UNet(Dual)?$", "", arch) or arch


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
    # どの世代の手順で学習したか。**どう学習したかは重みから復元できない**
    # （`best_model.pth` に epoch と best_f1 しか入っていない）ので、
    # ここに書いておくしかない。`docs/設計図_修正案.md` の `6`。
    #   v1-notebook   旧世代。CFG を通らず値が直書き。分割は他と一致（実測）
    #   v1-blocksplit 空間ブロック分割。**別の土俵**
    #   v2-config     CFG 経由。seed は条件によって未固定
    #   v3-base       基盤ノートブック。全項目が config 経由
    protocol: str = "v1-notebook"
    # 和集合（アンサンブル）に入れてよいか。**分割か選択が他と違う条件は false。**
    # テストのF値で最良エポックを選んだ条件を混ぜると、和集合の意味が変わる。
    ensemble: bool = True
    # 乱数を固定して学習したか（`torch.manual_seed`）。`ゆらぎ幅.md` の `8`
    seeded: bool = False
    # 分割の指定。**空なら train の test_size / val_size / random_state を使う**
    # （旧条件はこれ）。`kind` で種類を選べるようにしてある。
    #   {"kind": "random", "test_size": 0.2, "val_size": 0.25, "random_state": 42}
    #   {"kind": "csv", "file": "hiroshima_block_20km"}   → dc5lib/splits.py
    split: dict = field(default_factory=dict)
    # オーバーラップ学習。**空なら素のタイルで学習する**。
    #   {"max_shift": 64, "seed": 42}
    # 既存の OverlapTrain 条件はノートブックに直書きしていた値で、
    # 2026-10-29 に実装を読んで config に写した（挙動は変わらない）。
    overlap: dict = field(default_factory=dict)
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
        base = f"{'拡張+' if self.augment else ''}{fam} {inputs}"
        # 損失が既定と違うなら、それも名前に出す。
        # そうしないと「損失だけ違う条件」が同じラベルになり、
        # 図や表で区別できなくなる。
        loss = (self.train or {}).get("loss") or {}
        kind = loss.get("type", "focal_tversky")
        if kind == "instance_weighted_focal_tversky":
            base += f" 箇所損失p{loss.get('p', '?')}"
        elif kind != "focal_tversky":
            base += f" {kind}"
        if self.bg_ratio:
            base += f" bg{self.bg_ratio}"
        # 構造が系統の標準と違うなら、それも名前に出す。
        # 同じクラスでも補助出力やASPPの有無で別条件になるので、そこも拾う。
        parts = []
        tag = arch_tag(self.arch)
        if tag:
            parts.append(tag)
        if loss.get("aux_stages") and "監督" not in tag:
            parts.append("監督")
        if loss.get("use_aspp") and "ASPP" not in tag:
            parts.append("ASPP")
        if parts:
            base += " " + "+".join(parts)
        return base


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


def _notebook_sources(name):
    """その条件のノートブックのコードセルを1本の文字列にして返す。"""
    import json
    out = []
    for nbp in sorted(paths.experiment_dir(name).glob("*.ipynb")):
        try:
            nb = json.loads(nbp.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append((nbp.name, "\n".join("".join(c.get("source", []))
                                        for c in nb.get("cells", [])
                                        if c.get("cell_type") == "code")))
    return out


def _model_class_names() -> set:
    """MODEL_CLASSES のキー（モデルクラス名）だけを集める。

    **torch を import せずに済ませている。** tools/check.py は
    「リポジトリの整合性を見るだけ」の道具なので、torch が入っていない
    環境（conda env を有効にしていない素の python など）でも動くべき。
    import できるなら import し、できなければ models.py の本文から
    キーを読む。どちらでも同じ集合になる。
    """
    try:
        from .models import MODEL_CLASSES
        return set(MODEL_CLASSES)
    except Exception:
        src = (Path(__file__).parent / "models.py").read_text(encoding="utf-8")
        m = re.search(r"^MODEL_CLASSES\s*=\s*\{(.*?)^\}", src, re.S | re.M)
        if not m:
            return set()
        return set(re.findall(r'"([A-Za-z_]\w*)"\s*:', m.group(1)))


def _check_notebook_matches_config(name, c):
    """config とノートブックの食い違いを止める。

    config を書き換えてもノートブックが直書きのままだと、**学習が最後まで
    走ってから初めて気づく。** 実際に FinalFusion_SAM_APM256 で 310 エポック
    回したあと、使われていたのが別の pkl と別のモデルだったことが分かった。

    古い条件のノートブックは pkl 名もモデルも直書きしている。そこを
    --from のひな形にすると、config だけ新しくなって中身が追従しない。
    """
    import re
    problems = []
    for fn, src in _notebook_sources(name):
        where = f"{name}/{fn}"

        # (1) 読み込む pkl が config の dataset と一致しているか
        for region, pkl in set(re.findall(
                r'dataset_path\(\s*"(\w+)"\s*,\s*"([\w\.\-]+)"\s*\)', src)):
            want = c.dataset.get(region)
            if want and pkl != want:
                problems.append(
                    f"{where}: 読み込む pkl が config と違います — "
                    f"ノートブック '{pkl}' / config '{want}'。"
                    f"config から引くには pkl_for() を使ってください")

        # (2) 作られるモデルが config の arch と一致しているか
        #     build_model を使っていれば config 駆動なので問題ない。
        #     直書きの古いノートブックは、**書いてあるクラス名が arch と
        #     一致していれば実害が無い**ので止めない。一致しないものだけ止める。
        if "build_model" not in src:
            hard = set(re.findall(r'\bmodel\s*=\s*([A-Za-z_]\w*)\s*\(', src))
            # `model = ModelClass(...)` のように変数経由で作っているものは、
            # ここでは解決できないので触らない。**実在のクラス名を直書き
            # している場合だけ**、arch と一致しているかを見る。
            named = hard & _model_class_names()
            if named and c.arch not in named:
                problems.append(
                    f"{where}: 作られるモデルが config の arch と違います — "
                    f"ノートブック {sorted(named)} / config '{c.arch}'。"
                    f"config を変えてもノートブックが追従していません")

        # (3) config に early_stop_patience があるのに、ノートブックに
        #     対応するコードが無いと、黙って epochs の上限まで走り続ける。
        #
        #     **ただし `dc5lib.train.fit` に任せているなら、ノートブックに
        #     書かないのが正しい**（基盤ノートブック＝v3）。`fit` が
        #     `cfg.early_stop_patience` を見て打ち切る。
        #     2026-10-29 にこの除外を足した。足さないと新しい雛形が
        #     必ず赤くなり、検査が信用されなくなる。
        if (c.train or {}).get("early_stop_patience"):
            delegated = ("from dc5lib.train import fit" in src
                         or "dc5lib.train" in src or "fit(" in src)
            own = ("EARLY_STOP_PATIENCE" in src and "epochs_since_best" in src)
            if not own and not delegated:
                problems.append(
                    f"{where}: config に early_stop_patience があるのに、"
                    f"ノートブックに早期終了のコードが無く、"
                    f"dc5lib.train.fit も使っていません。"
                    f"epochs の上限まで走り続けます")
    return problems


def _check_overlap_names():
    """名前が OverlapTrain / OL と名乗るなら config に overlap が要る。

    **config の overlap.max_shift が 0 か未設定なら素のタイルで学習する。**
    名前だけオーバーラップのつもりになっていると、条件の意味が変わる
    （dc5lib/overlap.py の __getitem__ は max_shift<=0 で中央固定）。
    """
    out = []
    for c in load_all(include_excluded=True):
        by_name = ("OverlapTrain" in c.name) or ("_OL_" in c.name) \
            or c.name.endswith("_OL")
        ov = int((getattr(c, 'overlap', None) or {}).get('max_shift', 0) or 0)
        if by_name and ov <= 0:
            out.append(
                f"{c.name}: 名前はオーバーラップ学習だが config の overlap.max_shift が "
                f"{ov} です。このままだと素のタイルで学習します")
        if ov > 0 and not by_name:
            out.append(
                f"{c.name}: config に overlap.max_shift={ov} があるのに名前に "
                f"OverlapTrain も OL も入っていません")
    return out


def split_declaration(cond):
    """config の `split:` が条件の実態に合っているか。**判定はここだけ。**

    返りは `(種別, 文)`。種別は

    | 種別 | 意味 |
    |---|---|
    | `None` | 問題なし |
    | `"undeclared"` | csv 分割なのに `split:` が無い。**直せる。検査で止める** |
    | `"cannot_declare"` | 宣言できない条件。**既知として黙らせる** |

    `tools/audit_splits.py` と `_check_declared_split` が**別々に判定していて
    食い違っていた**（2026-11-10 の再レビュー 4）── audit 側は
    `testselect` も対象にしていたので、`Train_Hiroshima_Test_Shimane_SAM_APM`
    に**消せない警告**が毎回出ていた。**消せない警告は読み飛ばされる。**
    """
    pr = str(getattr(cond, "protocol", "") or "")
    kind = str((getattr(cond, "split", None) or {}).get("kind") or "")
    if "BlockSplit" in cond.name or "blocksplit" in pr:
        if kind != "csv":
            return ("undeclared",
                    f"ブロック分割の条件ですが config の split.kind が "
                    f"{kind or '空'} です。このままだと splits.resolve が"
                    "乱数に落ちて、学習時と違う分割になります"
                    "（split: {kind: csv, file: …} を書いてください）")
        return (None, "")
    if "testselect" in pr:
        # 検証を取らない 8:2。`resolve` は val_size=0 を扱えないので宣言できない。
        # **テストは一致する**（test_size / random_state が同じなので
        # train_test_split の1段目が同じ）ので評価は正しく、架空なのは val だけ。
        return ("cannot_declare",
                "検証を取らない 8:2 なので split: に書けない（実装書 2）。"
                "テストは既定の分割と一致するので評価は正しい")
    if kind == "csv":
        # **宣言は済んでいる。問題は名前のほうである**（再々レビュー 6）。
        # `undeclared` と同じ札にすると、将来 csv 分割を足したときに
        # 「宣言し忘れた」と誤解される。
        return ("misnamed",
                "config に split.kind=csv があるのに名前も protocol も"
                "ブロック分割と名乗っていません（宣言は済んでいます）")
    return (None, "")


def _check_declared_split():
    """**csv 分割の条件は config に `split:` を書いているか。**

    `splits.resolve` は `split:` が空なら**黙って `random` に落ちる**。
    だから書き忘れると `build_loaders` が**学習時と違う分割**を作る。

    2026-11-08 に実際に踏んだ ── `FinalFusion_SAM_APM_BlockSplit` は
    protocol が `v1-blocksplit` なのに `split:` が無く、
    `audit_splits` に分割の指紋を出すまで**出発点と同じ分割に見えていた。**
    いまの乱数テスト（正解ありの 2,963枚）のうち **2,158枚（72.8%）**が
    この条件の学習か検証に入っており、**潜在的なデータリーク**だった。

    `_check_overlap_names` と同じたちの事故である（名乗っているものが
    config に届いていない）。**csv で表せるものだけを見る** ──
    `Train_Hiroshima_Test_Shimane_SAM_APM` は検証を取らない 8:2 で、
    `resolve` は `val_size=0` を扱えないので宣言できない（実装書 `2`）。
    """
    out = []
    for c in load_all(include_excluded=True):
        kind, msg = split_declaration(c)
        if kind in ("undeclared", "misnamed"):
            out.append(f"{c.name}: {msg}")
    return out


# `train:` に書いてよい鍵。**ここに無い鍵は誰も読まない。**
#   読まれるもの … `traincfg.DEFAULTS` の鍵と `loss`（`dc5lib/losses.py`）
#   記述だけのもの … 当時の手順を文章で残した3つ。**値を変えても何も起きない**
_TRAIN_NOTES = ("augment_params", "model_selection", "split")


def _check_train_keys():
    """**config の `train:` に、誰も読まない鍵が書かれていないか。**

    `traincfg.train_cfg` は `DEFAULTS` にある鍵だけを拾い、**それ以外を
    黙って捨てる。** だから `batchsize: 16` と打ち間違えると既定の 32 で
    学習が走り、config を読んだ人は 16 で回ったと思う
    （設計書 §1 の「書いてあるから効いている、が嘘になる」と同じ形）。
    2026-11-11 のレビューで、全条件の `train:` の鍵を数えて確かめた。
    """
    from .traincfg import DEFAULTS
    ok = set(DEFAULTS) | {"loss"} | set(_TRAIN_NOTES)
    # `train.loss` の中も同じ。`build_loss` は損失の種類ごとに決まった鍵だけを
    # 拾うので、`alhpa: 0.8` と打ち間違えると既定の 0.7 で学習する。
    # **torch が無い環境でも検査は回す**（`_model_class_names` と同じ理由）ので、
    # 損失の表が読めなければ `loss` の中は見ない。
    try:
        from . import losses as _losses
    except Exception:                                     # noqa: BLE001
        _losses = None
    out = []
    for c in load_all(include_excluded=True):
        extra = sorted(set(c.train or {}) - ok)
        if extra:
            out.append(
                f"{c.name}: config の train: に誰も読まない鍵があります "
                f"{extra}。打ち間違いなら直し、新しい項目なら先に "
                "dc5lib/traincfg.py の DEFAULTS に足してください")
        spec = dict((c.train or {}).get("loss") or {})
        if _losses is not None and spec:
            kind = spec.get("type", "focal_tversky")
            # `clamp` を読むのは FocalTversky の2種だけ（箇所正規化の損失は
            # ±10 で固定。`dc5lib/instance_loss.py`）
            known = (set(_losses.DEFAULTS.get(kind, {}))
                     | set(_losses.AUX_KEYS) | {"type"}
                     | ({"clamp"} if kind.startswith("focal_tversky") else set()))
            extra = sorted(set(spec) - known)
            if kind in _losses.DEFAULTS and extra:
                out.append(
                    f"{c.name}: config の train.loss に、損失 {kind} が"
                    f"読まない鍵があります {extra}"
                    "（dc5lib/losses.py の DEFAULTS を参照）")
    return out


def _check_labels():
    """図表に出す表示ラベルが重複していないか。

    重複すると、図の棒や表の行がどの条件のものか区別できなくなる。
    条件が増えるほど起きやすいので、検査で止める。
    """
    import collections
    seen = collections.defaultdict(list)
    for c in load_all(include_excluded=True):
        seen[c.display()].append(c.name)
    return [f"表示ラベル「{lab}」が {len(v)} 条件で重複しています: {v}"
            f"（config.yaml の label で区別してください）"
            for lab, v in seen.items() if len(v) > 1]


def check():
    """CI / コミット前に回す検査。異常があれば行を返す。"""
    problems = (_check_unicode_paths() + _check_labels()
                + _check_overlap_names() + _check_declared_split()
                + _check_train_keys())
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
            problems += _check_notebook_matches_config(n, c)
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
