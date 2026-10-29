# -*- coding: utf-8 -*-
"""テストの画素評価と箇所評価。**評価ロジックは持たない。**

なぜ薄いか
----------
箇所評価は `analysis/instances.py` と `dc5lib/instance_eval.py` にあり、
`展望_再現率をどう上げるか.md` の `42`〜`44` の数字はそこで出している。
**ここで別に実装すると数字が動く。** だから設定を揃えて呼ぶだけにする。

    COV_GT = 0.5 / COV_PRED = 0.3 / MIN_SIZE = 10 / connectivity 8

この4つを変えると過去の数字と比べられない。`PRIMARY` を使い回す。

    from dc5lib.evaluate import evaluate
    r = evaluate(get(CONDITION), ld)            # テストで測る
    print(r.describe())

`record=True` を渡すと `results/evals/` に記録し、`metrics.csv` に並ぶ。
**記録しないと docs に書いた数字の出どころが残らない**（`CLAUDE.md` の「出した計算をコードに残す」）。
`run_id` は重みの素性（`machine` と `started_at`）から作るので、
**同じ重みを測り直すと同じ記録を置き換える** ── 行が増えない。

**オーバーラップ推論はここではしない。** 学習ノートブックで見るのは
素のタイルでの値で、オーバーラップ推論の箇所評価は
`analysis/run_overlap_eval.py` と `analysis/run_overlap_prune.py` が
担当する（そちらが `42`〜`44` の数字の出どころ）。
**同じ数字を2か所で出さない。**
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@dataclass
class Result:
    pixel: dict = field(default_factory=dict)
    instance: dict = field(default_factory=dict)
    n_tiles: int = 0
    split: str = "test"

    def describe(self) -> str:
        p, i = self.pixel, self.instance
        out = [f"{self.split} {self.n_tiles:,} 枚（素のタイル。"
               "オーバーラップ推論ではない）",
               f"  画素  R {p['recall']:.4f} / P {p['precision']:.4f}"
               f" / F {p['f1']:.4f} / 正解率 {p['accuracy']:.4f}"]
        if i:
            out.append(f"  箇所  R {i['recall']:.4f} / P {i['precision']:.4f}"
                       f" / F {i['f1']:.4f}"
                       f"  正解 {i['n_gt']:,} / 予測 {i['n_pred']:,}")
            out.append(f"        設定 {i['setting']}")
        return "\n".join(out)


@torch.no_grad()
def predict(model, loader, device=None, forward=None, thr=0.5):
    """`(正解, 予測の二値, ロジット)` を numpy で返す。**タイル順は loader 順。**"""
    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "cpu"))
    model = model.to(device).eval()
    GT, PB, LG = [], [], []
    for xs, y in loader:
        xs = [x.to(device) for x in (xs if isinstance(xs, (list, tuple))
                                     else [xs])]
        out = forward(model, xs) if forward else model(*xs)
        if isinstance(out, (tuple, list)):
            out = out[0]
        lg = out.detach().cpu().numpy()[:, 0]
        GT.append((y.numpy()[:, 0] > 0.5))
        PB.append(1 / (1 + np.exp(-lg)) > thr)
        LG.append(lg)
    return (np.concatenate(GT), np.concatenate(PB), np.concatenate(LG))


def pixel_scores(GT, PB) -> dict:
    tp = int((PB & GT).sum())
    fp = int((PB & ~GT).sum())
    fn = int((~PB & GT).sum())
    tn = int((~PB & ~GT).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return {"precision": pr, "recall": rc, "f1": f1,
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
            "tp": tp, "fp": fp, "fn": fn}


def instance_scores(GT, PB) -> dict:
    """箇所評価。**`analysis` の設定をそのまま使う。**"""
    import pandas as pd

    from analysis import instances
    from analysis.run_instance_eval import SETTINGS
    from dc5lib.instance_eval import PRIMARY

    rows = [instances.evaluate_tile_multi(GT[k], PB[k], SETTINGS,
                                          connectivity=8, min_size=10)
            for k in range(len(GT))]
    df = pd.DataFrame(rows)
    s = instances.summarize_setting(df, PRIMARY)
    return {"recall": float(s["箇所Recall"]), "precision": float(s["箇所Precision"]),
            "f1": float(s["箇所F値"]), "n_pred": int(df.n_pred.sum()),
            "n_gt": int(df.n_gt.sum()), "setting": PRIMARY}


def _run_id_for(cond, meta):
    """**重みの素性から run_id を作る。**

    こうすると同じ学習の評価が同じ run_id になり、`record_eval` が
    古い記録を置き換える。毎回新しい id にすると `metrics.csv` に
    同じ測定が何行も並ぶ。素性が無い古い重みでは今の時刻になる。
    """
    from . import results as R
    m = meta or {}
    when = str(m.get("started_at") or m.get("saved_at") or "")
    # 2026-10-29T16:49:53+09:00 -> 20261029-164953
    digits = "".join(ch for ch in when[:19] if ch.isdigit())
    stamp = (digits[:8] + "-" + digits[8:14]) if len(digits) >= 14 \
        else None
    return R.new_run_id(cond.name, m.get("machine"), stamp)


def _record(cond, loaders, r, split, meta, verbose):
    """画素と箇所を `results/evals/` に記録する。"""
    from . import results as R
    region = str(getattr(loaders, "region", "") or "").lower()
    if not region:
        print("  ※ 記録できません: loaders に region がありません")
        return []
    run_id = _run_id_for(cond, meta)
    bg = float(getattr(cond, "bg_ratio", 0.0) or 0.0)
    tileset = R.tileset_label(region, bg)
    note = ("dc5lib.evaluate.evaluate。**素のタイル**（オーバーラップ推論ではない）"
            f" / {split} {r.n_tiles:,} 枚")
    out = [R.record_eval(
        run_id=run_id, condition=cond.name, region=region,
        tileset=tileset, scope="通常", metric_kind="面積",
        recall=r.pixel["recall"], precision=r.pixel["precision"],
        f1=r.pixel["f1"], n_tiles=r.n_tiles, note=note)]
    if r.instance:
        out.append(R.record_eval(
            run_id=run_id, condition=cond.name, region=region,
            tileset=tileset, scope="通常", metric_kind="箇所",
            setting=r.instance["setting"],
            recall=r.instance["recall"],
            precision=r.instance["precision"], f1=r.instance["f1"],
            n_gt_instances=r.instance["n_gt"],
            n_pred_instances=r.instance["n_pred"],
            n_tiles=r.n_tiles, note=note))
    if verbose:
        print(f"  記録しました {len(out)} 件 / run_id {run_id}")
        print(f"    {out[0].parent}")
    return out


def evaluate(cond, loaders, split="test", model=None, device=None,
             forward=None, thr=0.5, instance=True, verbose=True,
             record=False, rebuild_csv=True) -> Result:
    from .ckpt import meta_of
    loader = getattr(loaders, split)
    meta = meta_of(cond.weights) if cond.weights.exists() else None
    if model is None:
        from .models import build_model_for, load_weights
        model = build_model_for(cond)
        load_weights(model, cond.weights, device="cpu")
        if verbose and meta is None:
            print("  ※ この重みには素性がありません（固定前の世代）")
    GT, PB, _ = predict(model, loader, device, forward, thr)
    r = Result(pixel=pixel_scores(GT, PB), n_tiles=len(GT), split=split)
    if instance:
        r.instance = instance_scores(GT, PB)
    if verbose:
        print(r.describe())
    if record:
        if _record(cond, loaders, r, split, meta, verbose) \
                and rebuild_csv:
            from . import results as R
            out = R.build_metrics_csv()
            if verbose:
                print(f"  metrics.csv を作り直しました: {out}")
    return r
