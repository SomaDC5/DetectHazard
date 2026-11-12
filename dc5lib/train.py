# -*- coding: utf-8 -*-
"""学習ループ。**早期終了と曲線の書き出しを取りこぼさない。**

なぜ要るか
----------
`docs/設計図_現状.md` の `5.1` のとおり、`early_stop_patience` は全条件の
config に書いてあるのに、**ループに打ち切りが入っているのは32本中**で
23本は `epochs` まで回り切っていた。`FinalFusion_SAM_APM_OverlapTrain` が
1000エポック回ったのはこれが原因である。

曲線も取りこぼしていた。`BVB` は159エポックまで回ったのに CSV は150行、
`BLD` は189まで回ったのに180行しか無い ── **9行ぶん書き遅れていた。**
ここでは**毎エポック flush する**ので、途中で落ちても行は残る。

    from dc5lib.train import fit
    hist = fit(cond, loaders)          # 重みと曲線は自動で正しい場所へ

差し込み（`hooks`）
-------------------
細工はノートブックから入れる。**ループを書き写さない。**

    fit(cond, ld, hooks={
        "loss_fn":      lambda out, y: ...,     # 損失の計算を置き換える
        "on_epoch_end": lambda ep, m: ...,      # 毎エポックの後
        "forward":      lambda model, xs: ...,  # 補助出力つきの前向き
    })

**気をつけること**

- モデル選択は**検証F値が最良のエポック**。テストは一度も見ない
  （旧プロトコルの3条件はテストで選んでいて、2026-10-29 に削除した）
- 重みは `dc5lib.ckpt.save` で保存するので、**素性が一緒に入る**
- 補助出力（深層監督など）を返すモデルは `forward` フックで畳む。
  既定では `out[0]` を使う
"""
from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch

from . import ckpt as _ckpt
from . import paths
from .losses import build_loss
from .traincfg import train_cfg

COLUMNS = ["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1-score",
           "Validation-F1-score"]


@dataclass
class History:
    rows: list = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    epochs_ran: int = 0
    stopped_early: bool = False
    curve: str = ""
    weights: str = ""

    def describe(self) -> str:
        tail = ("早期終了で止まった" if self.stopped_early
                else "epochs まで回り切った")
        return (f"{self.epochs_ran} エポック回って{tail}。"
                f"最良はエポック {self.best_epoch} の検証F {self.best_val_f1:.4f}\n"
                f"  曲線 {self.curve}\n  重み {self.weights}")


def _counts(pred, target):
    """画素の TP / FP / FN。**しきい値 0.5。**"""
    p = (pred > 0.5)
    t = (target > 0.5)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


@torch.no_grad()
def evaluate_pixel(model, loader, device, forward=None):
    """画素のF値。**学習中の検証に使う。**"""
    model.eval()
    TP = FP = FN = TN = 0
    for xs, y in loader:
        xs = [x.to(device) for x in (xs if isinstance(xs, (list, tuple))
                                     else [xs])]
        y = y.to(device)
        out = forward(model, xs) if forward else model(*xs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        p = torch.sigmoid(out)
        tp, fp, fn, tn = _counts(p, y)
        TP += tp; FP += fp; FN += fn; TN += tn          # noqa: E702
    pr, rc, f1 = _prf(TP, FP, FN)
    acc = (TP + TN) / max(TP + FP + FN + TN, 1)
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}


def _guard_before_training(cond, cfg, loaders, wpath, curve, overwrite,
                           hooks=None):
    """**データを読んだあと・学習を始める前**に止める（設計書 §7）。

    4つを見る。どれも「黙って違うことをする」たちの事故である。

    1. **既存の重みと曲線。** `fit` は曲線を `"w"` で開いた瞬間に中身を消し、
       `best = -1.0` から始まるので1エポック目で `best_model.pth` を
       上書きする。方針は「既存の重みと曲線は作り直さない」（CLAUDE.md）
       なので、既存があれば止めて `overwrite=True` で明示させる
    2. **`augment`。** `build_loaders` は `cond.augment` を見ない。
       `transform=` を渡さないかぎり拡張は掛からないので、
       `augment: true` と書いたのに素のまま学習する
    3. **`train.init_from`。** `TrainCfg.describe()` は「追加学習」と出すが、
       `fit` は読まずゼロから学習する

    4. **補助損失（深層監督）。** config の `train.loss.aux_lambda` が 0 より
       大きい条件は、モデルに補助ヘッドが付く（`build_model_for`）。だが `fit` は
       `model(*xs)` を呼ぶだけで `return_aux=True` を渡さず、損失も本体の出力に
       しか掛けない。**補助ヘッドは一度も学習されず、補助損失なしの学習になる**
       （2026-11-12 のレビュー。該当は12条件で、どれもノートブックの自前の
       ループで学習したもの）。`hooks` に `forward` と `loss_fn` の両方が
       あれば、呼ぶ側が補助損失を組んでいるとみなして通す

    2・3・4 は**実装しないと決めた**（設計書 §8）。止めるだけである。
    """
    if not overwrite:
        exist = [p for p in (wpath, curve) if Path(p).exists()]
        if exist:
            raise FileExistsError(
                f"{cond.name} の " + " と ".join(Path(p).name for p in exist)
                + " が既にあります。**学習を始めると曲線は開いた瞬間に消え、"
                "1エポック目で重みが上書きされます。**"
                " 作り直してよいなら fit(..., overwrite=True)"
                "（`tools/run_training.py --overwrite`）。"
                f"\n  {Path(wpath).parent}")

    if getattr(cond, "augment", False):
        tr = (getattr(loaders, "datasets", {}) or {}).get("train")
        if getattr(tr, "transform", None) is None:
            raise NotImplementedError(
                f"{cond.name} は augment: true ですが、"
                "`build_loaders` は `cond.augment` を見ません。"
                "**このまま回すと拡張なしで学習します**（設計書 §8）。"
                " 拡張を掛けるなら build_loaders(..., transform=…) を渡し、"
                "掛けないなら config の augment を false にしてください。")

    if getattr(cfg, "init_from", None):
        raise NotImplementedError(
            f"{cond.name} は train.init_from={cfg.init_from!r} ですが、"
            "`fit` は追加学習を実装していません。"
            "**describe() は「追加学習」と出るのにゼロから学習します**"
            "（設計書 §8）。ノートブック側で重みを読み込んでから"
            " hooks={'model': …} で渡してください。")

    from .losses import aux_lambda
    lam = aux_lambda(cond)
    h = hooks or {}
    if lam > 0 and not (h.get("forward") and h.get("loss_fn")):
        raise NotImplementedError(
            f"{cond.name} は train.loss.aux_lambda={lam} ですが、"
            "`fit` は補助損失（深層監督）を計算しません。"
            "**このまま回すと補助ヘッドは学習されず、補助損失なしの学習になります**"
            "（設計書 §8）。hooks に forward（`model(*xs, return_aux=True)`）と"
            " loss_fn（本体＋λ×補助）の両方を渡すか、補助損失を使わないなら"
            " config の aux_lambda を 0 にしてください。")


def fit(cond, loaders, device=None, hooks=None, epochs=None,
        weights=None, curve=None, verbose=True, overwrite=False) -> History:
    """`epochs` / `weights` / `curve` は試運転のための差し替え。

    **既定では本物の場所に書く。** 試運転で本物の重みを上書きしないよう、
    短く回すときは必ず `weights=` と `curve=` を別の場所にする。

    `overwrite=False` のまま既存の重みか曲線があると、
    **学習を始めずに `FileExistsError` で止まる**（`_guard_before_training`）。
    """
    cfg = train_cfg(cond)
    curve = Path(curve) if curve else paths.curve_path(cond.name)
    wpath = Path(weights) if weights else cond.weights
    hooks = dict(hooks or {})
    _guard_before_training(cond, cfg, loaders, wpath, curve, overwrite,
                           hooks=hooks)
    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "cpu"))
    from .models import build_model_for
    model = hooks.get("model") or build_model_for(cond)
    model = model.to(device)

    criterion, needs_w = build_loss(cond)
    if needs_w and "loss_fn" not in hooks:
        raise ValueError(
            f"{cond.name} の損失は重みマップを要求します"
            "（instance_weighted_*）。hooks['loss_fn'] を渡してください。"
            "docs/ノートブックの作り方.md の「重みマップが要る損失」を参照")
    opt = (hooks.get("optimizer")
           or torch.optim.Adam(model.parameters(), lr=cfg.lr))
    forward = hooks.get("forward")
    loss_fn = hooks.get("loss_fn") or (lambda out, y: criterion(
        out[0] if isinstance(out, (tuple, list)) else out, y))

    n_epochs = int(epochs or cfg.epochs)
    curve.parent.mkdir(parents=True, exist_ok=True)
    hist = History(curve=str(curve), weights=str(wpath))
    started = _dt.datetime.now().astimezone()
    best = -1.0
    since_best = 0

    # **毎エポック flush する。** 途中で落ちても行は残る
    f = open(curve, "w", encoding="utf-8", newline="")
    f.write(",".join(COLUMNS) + "\n")
    f.flush()
    try:
        for ep in range(1, n_epochs + 1):
            model.train()
            t0 = time.time()
            tot = n = 0
            TP = FP = FN = TN = 0
            for xs, y in loaders.train:
                xs = [x.to(device) for x in (xs if isinstance(xs, (list, tuple))
                                             else [xs])]
                y = y.to(device)
                opt.zero_grad()
                out = forward(model, xs) if forward else model(*xs)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                tot += float(loss.item()) * y.shape[0]
                n += y.shape[0]
                with torch.no_grad():
                    o = out[0] if isinstance(out, (tuple, list)) else out
                    tp, fp, fn_, tn = _counts(torch.sigmoid(o), y)
                    TP += tp; FP += fp; FN += fn_; TN += tn   # noqa: E702
            pr, rc, f1 = _prf(TP, FP, FN)
            acc = (TP + TN) / max(TP + FP + FN + TN, 1)
            vm = evaluate_pixel(model, loaders.val, device, forward)
            row = [ep, tot / max(n, 1), acc, pr, rc, f1, vm["f1"]]
            f.write(",".join(str(x) for x in row) + "\n")
            f.flush()
            hist.rows.append(row)
            hist.epochs_ran = ep

            if vm["f1"] > best:
                best, since_best = vm["f1"], 0
                hist.best_epoch, hist.best_val_f1 = ep, vm["f1"]
                meta = _ckpt.build_meta(
                    cond, cfg=cfg, split=loaders.meta_split(),
                    region=loaders.region, seeded=hooks.get("seeded"),
                    extra={"epochs_ran": ep, "best_epoch": ep,
                           "best_val_f1": vm["f1"],
                           "started_at": started.isoformat(timespec="seconds")})
                _ckpt.save(wpath, model, meta, epoch=ep,
                           best_f1=vm["f1"])
                mark = "  ← 保存"
            else:
                since_best += 1
                mark = ""
            if verbose:
                print(f"ep {ep:>4}  loss {row[1]:.4f}  学習F {f1:.4f}  "
                      f"検証F {vm['f1']:.4f}  {time.time() - t0:.0f}秒"
                      f"  更新なし {since_best}{mark}", flush=True)
            if hooks.get("on_epoch_end"):
                hooks["on_epoch_end"](ep, {"train": row, "val": vm})

            if cfg.early_stop_patience and since_best >= cfg.early_stop_patience:
                hist.stopped_early = True
                if verbose:
                    print(f"検証F が {since_best} エポック更新されていないので"
                          f"打ち切ります（最良 {best:.4f} / エポック "
                          f"{hist.best_epoch}）。{ep}/{n_epochs}")
                break
    finally:
        f.close()

    # 最後にもう一度 meta を書き直して、回り切った数を残す
    if hist.best_epoch:
        m = _ckpt.meta_of(wpath) or {}
        m.update({"epochs_ran": hist.epochs_ran,
                  "stopped_early": hist.stopped_early,
                  "finished_at": _dt.datetime.now().astimezone()
                  .isoformat(timespec="seconds")})
        obj = torch.load(wpath, map_location="cpu", weights_only=False)
        obj["meta"] = m
        torch.save(obj, wpath)

        # **最良エポックの重みを model に戻す。**
        # 戻さないと model は最終エポックのままで、fit のあとに
        # model を渡して評価すると**選んでいないモデルを測る**。
        # 実測で箇所F 0.6610（最終 ep184）対 0.6914（最良 ep84）だった。
        sd = obj.get("model_state_dict", obj.get("state_dict"))
        if sd is None:
            raise AssertionError(
                f"{wpath} に model_state_dict がありません。最良の重みを "
                "model に戻せません")
        model.load_state_dict(sd)
        if verbose:
            print(f"最良エポック {hist.best_epoch} の重みを model に戻しました"
                  "（これ以降 model を渡して評価してよい）")
    if verbose:
        print(hist.describe())
    return hist
