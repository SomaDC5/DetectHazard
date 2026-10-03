# -*- coding: utf-8 -*-
"""config.yaml の `train.loss` から損失関数を組み立てる。

これまで config.yaml の `train.loss` は飾りで、実際の損失はノートブックに
直書きされていた。設定と実装がずれると、あとから「この条件はどの損失で
学習したのか」が追えなくなるので、config を読んで作るようにした。

    from dc5lib.losses import build_loss
    criterion, needs_weight = build_loss(CONDITION)
    loss = criterion(outputs, masks, weights) if needs_weight else criterion(outputs, masks)

対応している type

    focal_tversky                      現行。alpha(FN) / beta(FP) / gamma
    instance_weighted_focal_tversky    箇所正規化版。上に加えて a0 / p / center_lam
                                       正解の箇所ごとに重みの総和を揃える。
                                       重みマップを別に渡す必要がある（needs_weight=True）

    python -m dc5lib.losses                       対応している type と既定値
    python -m dc5lib.losses <条件名>              その条件の損失を表示
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from .instance_loss import InstanceWeightedFocalTverskyLoss, make_instance_weight  # noqa: F401


class FocalTverskyLoss(nn.Module):
    """現行の実装。alpha が False Negative、beta が False Positive の重み。

    論文本文では alpha と beta のラベルが逆に書かれているが、
    実装（と意図）はこちらが正しい。alpha > beta で見逃しを重く罰する。
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets, weight=None):
        preds = torch.sigmoid(logits.clamp(-10, 10)).reshape(logits.size(0), -1)
        t = targets.reshape(targets.size(0), -1).to(preds.dtype)
        TP = (preds * t).sum(1)
        FP = ((1 - t) * preds).sum(1)
        FN = (t * (1 - preds)).sum(1)
        tv = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        tv = torch.clamp(tv, min=self.smooth, max=1.0)
        return torch.pow(1 - tv, self.gamma).mean()


DEFAULTS = {
    "focal_tversky": {"alpha": 0.7, "beta": 0.3, "gamma": 0.75},
    "instance_weighted_focal_tversky": {
        "alpha": 0.7, "beta": 0.5, "gamma": 0.75,
        "a0": 300, "p": 0.5, "center_lam": 0.0,
    },
}

# 重みマップの作り方に関わる引数（損失そのものには渡さない）
WEIGHT_KEYS = ("a0", "p", "center_lam")

# 深層監督の補助損失の重み（損失そのものには渡さない）。
# DeepSupMultiEncoderUNet のように補助出力を持つモデルでだけ使う。
AUX_KEYS = ("aux_lambda", "aux_stages", "use_aspp")


def _spec(cond):
    """条件名 / Condition / dict のどれでも受ける。"""
    if isinstance(cond, dict):
        return cond
    if isinstance(cond, str):
        from . import registry
        cond = registry.get(cond)
    return (cond.train or {}).get("loss") or {}


def build_loss(cond):
    """(criterion, needs_weight) を返す。

    needs_weight が True の条件は、学習ループで重みマップを渡す必要がある。
    重みマップは dc5lib.instance_loss.precompute_weights で事前に作り、
    データ拡張と同じ変換を掛けて持ち回る（docs/使い方.md を参照）。
    """
    spec = dict(_spec(cond))
    kind = spec.pop("type", "focal_tversky")
    for k in AUX_KEYS:          # 補助損失の重みは criterion には渡さない
        spec.pop(k, None)
    if kind not in DEFAULTS:
        raise ValueError(f"未知の損失 type: {kind}。対応しているのは {list(DEFAULTS)}")
    p = dict(DEFAULTS[kind])
    p.update({k: v for k, v in spec.items() if v is not None})

    if kind == "focal_tversky":
        return FocalTverskyLoss(p["alpha"], p["beta"], p["gamma"]), False

    crit = InstanceWeightedFocalTverskyLoss(p["alpha"], p["beta"], p["gamma"])
    return crit, True


def weight_params(cond) -> dict:
    """重みマップを作るときの引数（a0 / p / center_lam）を取り出す。"""
    spec = _spec(cond)
    kind = spec.get("type", "focal_tversky")
    p = dict(DEFAULTS.get(kind, {}))
    p.update({k: v for k, v in spec.items() if v is not None})
    return {k: p[k] for k in WEIGHT_KEYS if k in p}


def aux_lambda(cond) -> float:
    """深層監督の補助損失の重み λ。

    L = FocalTversky(マスク) + λ * FocalTversky(存在マップ)

    補助出力を持たないモデルでは使われない。0 なら補助損失を無効にできる。
    """
    return float(_spec(cond).get("aux_lambda", 0.0) or 0.0)


def aux_stages(cond) -> tuple:
    """深層監督を入れるスキップ接続の段。

    AllSkipDeepSupMultiEncoderUNet で使う。既定は (1, 2, 3)。
    16x16 の段(4)は 50px の箇所がサブピクセルになるので既定から外している。
    """
    v = _spec(cond).get("aux_stages")
    return tuple(int(x) for x in v) if v else (1, 2, 3)


def use_aspp(cond) -> bool:
    """スキップに ASPP を入れるか。FullSkipMultiEncoderUNet で使う。"""
    return bool(_spec(cond).get("use_aspp", False))


def describe(cond) -> str:
    spec = _spec(cond)
    kind = spec.get("type", "focal_tversky")
    p = dict(DEFAULTS.get(kind, {}))
    p.update({k: v for k, v in spec.items() if k != "type" and v is not None})
    lam = p.pop("aux_lambda", None)
    stg = p.pop("aux_stages", None)
    asp = p.pop("use_aspp", None)      # モデル側の設定。criterion の引数ではない
    body = ", ".join(f"{k}={v}" for k, v in p.items())
    _, needs = build_loss(cond)
    out = f"{kind}({body})"
    if lam:
        out += f" + {lam}*深層監督"
        if stg:
            out += f"(段 {','.join(str(x) for x in stg)})"
    if asp:
        out += " / スキップに ASPP"
    return out + ("  ※重みマップが必要" if needs else "")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            print(f"{name:<40}{describe(name)}")
    else:
        print("対応している損失と既定値\n")
        for k, v in DEFAULTS.items():
            print(f"  {k}")
            for kk, vv in v.items():
                note = ""
                if kk == "alpha":
                    note = "  False Negative（見逃し）の重み"
                elif kk == "beta":
                    note = "  False Positive（過検出）の重み"
                elif kk == "a0":
                    note = "  面積の下限。小さい箇所の重みが発散するのを防ぐ"
                elif kk == "p":
                    note = "  重みの強さ。1.0 で完全な箇所正規化、0 で無効"
                elif kk == "center_lam":
                    note = "  中心重みの強さ。まず 0 で試す"
                print(f"      {kk:<12}{vv}{note}")
            print()
        print("条件名を渡すと、その条件の損失を表示します。")
