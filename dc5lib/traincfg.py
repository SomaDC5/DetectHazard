# -*- coding: utf-8 -*-
"""学習のハイパーパラメータを config.yaml から1か所で読む。

ノートブックに直書きしないためのもの。直書きしていると、config を変えても
効かず、気づかないまま別の設定で学習してしまう。実際に
`FinalFusion_SAM_APM_OverlapTrain` では early_stop_patience が
config にも無く、ノートブックにも早期終了が無かったため、
検証F値が ep115 で頭打ちのまま 1000 エポック回り続けた。

    from dc5lib.traincfg import train_cfg
    CFG = train_cfg(CONDITION)
    print(CFG.describe())

既定値は「この研究でこれまで使ってきた値」。config に書いてあればそちらが勝つ。
split（test_size / val_size / random_state）は、以前は free text の
`split:` にしか書かれておらず機械から読めなかったので、個別の項目にした。
書かれていなければ従来と同じ 0.2 / 0.25 / 42 になる。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .registry import get

# これまで全条件で共通に使ってきた値。config に無いときだけ使う
DEFAULTS = {
    "epochs": 1000,
    "batch_size": 32,
    "lr": 1e-4,
    "seed": 42,
    "early_stop_patience": 100,
    "test_size": 0.2,       # 全体から切るテストの割合
    "val_size": 0.25,       # 残り(trainval)から切る検証の割合
    "random_state": 42,
}


@dataclass
class TrainCfg:
    name: str
    epochs: int
    batch_size: int
    lr: float
    seed: int
    early_stop_patience: int
    test_size: float
    val_size: float
    random_state: int
    bg_ratio: float
    arch: str
    source: dict = field(default_factory=dict)   # 各項目が config 由来か既定か

    def describe(self) -> str:
        rows = []
        for k in ("arch", "epochs", "batch_size", "lr", "seed",
                  "early_stop_patience", "test_size", "val_size",
                  "random_state", "bg_ratio"):
            v = getattr(self, k)
            src = self.source.get(k, "既定")
            rows.append(f"  {k:<20}{v!s:<12}({src})")
        head = f"[{self.name}] 学習設定"
        tail = ""
        if not self.early_stop_patience:
            tail = ("\n  ※ 早期終了が無効。検証F値が伸びなくても "
                    f"{self.epochs} エポック回り切る")
        return head + "\n" + "\n".join(rows) + tail


def train_cfg(condition) -> TrainCfg:
    """条件名から学習設定を読む。config に無い項目は DEFAULTS を使う。"""
    c = get(condition) if isinstance(condition, str) else condition
    t = dict(c.train or {})
    vals, src = {}, {}
    for k, d in DEFAULTS.items():
        if k in t and t[k] is not None:
            vals[k], src[k] = t[k], "config"
        else:
            vals[k], src[k] = d, "既定"
    src["arch"] = src["bg_ratio"] = "config"
    return TrainCfg(name=c.name, arch=c.arch,
                    bg_ratio=float(getattr(c, "bg_ratio", 0.0) or 0.0),
                    epochs=int(vals["epochs"]),
                    batch_size=int(vals["batch_size"]),
                    lr=float(vals["lr"]),
                    seed=int(vals["seed"]),
                    early_stop_patience=int(vals["early_stop_patience"]),
                    test_size=float(vals["test_size"]),
                    val_size=float(vals["val_size"]),
                    random_state=int(vals["random_state"]),
                    source=src)


if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["FinalFusion_SAM_APM_OverlapTrain"]):
        print(train_cfg(n).describe())
        print()
