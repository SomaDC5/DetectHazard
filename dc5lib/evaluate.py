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


def _recordable(loaders, region, bg):
    """**{ラベル: 記録してよい枚数}**（設計書 §6 / 実装書 §8 の表そのもの）。

    **ラベルは枚数を指していない。同じ語が地域で違う集合を指す。**

    | 地域 | ラベル | 枚数 | `metrics.csv` の既存の行（条件の数）|
    |---|---|---:|---:|
    | 広島 | `警戒のみ` | 2,964（テスト分割）| 414（48）|
    | 島根 | `警戒のみ` | 9,679（正解ありの全件＝`gtonly`）| 17（17）|
    | 島根 | `全件` | 24,569（`subset: all`。主系統）| 420（48）|

    （行の数は 2026-11-11 に数え直した。以前の「47 / 45」は再現できなかった。
    下の文中の「17件」「47件」は「その地域・そのラベルの既存の行」と読むこと）

    島根は `subset: all` が主系統なので集合が3つあるのに、語は2つしか無い。
    **`build_loaders` が内部で割るテスト 1,936枚にはラベルが無い**ので、
    そこで記録すると「島根・警戒のみ」の17件を置き換える
    （2026-11-10 の再レビュー 1。2026-11-08 の直しは「全件」を守っただけで、
    衝突が「警戒のみ」に移っていた）。

    **島根の2つの評価はどちらも残る** ── 主系統（24,569＝`全件`。
    記録するのは `analysis/run_overlap_eval.py --record`）と
    補助 `gtonly`（9,679＝`警戒のみ`。記録するのは
    `templates/eval_shimane.ipynb` のセル3）である。
    止めるのは3つ目（`build_loaders` が内部で割るテスト 1,936枚）だけで、
    あれはどちらの評価でもない。
    （`analysis/run_shimane_scope.py` は**記録しない。** 推論もしない）

    **これは島根に限った話ではない。** 広島の正解ありタイル 14,816枚を
    `eval_shimane` と同じ手で測って記録すれば、既存47件と同じ鍵で衝突する。
    島根は `subset: all` のせいで先に踏んだだけである。

    **ラベルは枚数から選ぶ。** `tileset_label` は地域と `bg_ratio` だけから
    決めるので、島根では常に「全件」になってしまい、
    補助の `gtonly`（9,679枚）を測ったのか分からない。
    """
    if bg and bg > 0:
        # 「警戒+背景」。`build_loaders` が bg_ratio > 0 を拒むのでここには来ない
        return {}
    # **既定の乱数分割でなければ記録しない**（4回目のレビュー E）。
    # 広島の `警戒のみ` は「random/42 のテスト 2,964枚」という意味であって、
    # `split["n_test"]` が何枚でも受け入れるという意味ではない。
    # `FinalFusion_SAM_APM_BlockSplit` のテストは 3,673枚の**別の集合**で、
    # そのまま記録すると表の 2,964 と食い違う（ラベルが嘘になる）。
    sp = getattr(loaders, "split", None)
    if sp:
        from . import splits as _sp
        if not _sp.is_standard(sp):
            return {}
    n_all = int(getattr(loaders, "n_all", 0) or 0)
    n_used = int(getattr(loaders, "n_used", 0) or 0)
    n_test = int((getattr(loaders, "split", None) or {}).get("n_test", 0) or 0)
    if region == "hiroshima":
        # 広島は `subset: holdout`。**「警戒のみ」はテスト分割を指す**
        return {"警戒のみ": n_test} if n_test else {}
    if region == "shimane":
        out = {}
        if n_all:
            out["全件"] = n_all          # 主系統 `subset: all`
        if n_used:
            out["警戒のみ"] = n_used     # 補助 `subset: gtonly`
        return out
    return {}


def _tileset_for(loaders, region, bg, n_tiles):
    """**実際に測ったタイル集合**の呼び名。ラベルが無ければ `None`。"""
    for label, want in _recordable(loaders, region, bg).items():
        if int(n_tiles) == int(want):
            return label
    return None


def _trained_region(cond):
    """その重みが**どの地域で学習されたか**。素性が無ければ `None`。

    `ckpt.build_meta` が `region` を入れている。素性の無い古い重みでは
    **「分からない」ので None を返す**（設計書 §7 の作法。でっち上げない）。
    """
    try:
        from .ckpt import meta_of
        m = meta_of(cond.weights) if cond.weights.exists() else None
    except Exception:                                     # noqa: BLE001
        return None
    r = str((m or {}).get("region") or "").strip().lower()
    return r or None


def check_training_region(cond, region, n_tiles):
    """**学習に使った地域で、テスト分割以外を記録しようとしていないか。**

    島根の `全件`（24,569）と `警戒のみ`（9,679）は**学習タイルを含む。**
    島根で学習した重みでそこを測って記録すると、
    学習データでの値が成績として残る（設計書 §6）。

    素性の無い古い重みでは**「分からない」ので止めない**（§7 の作法）。

    `dc5lib/evaluate.py` と `analysis/run_overlap_eval.py --record` の
    **両方がこれを呼ぶ** ── 2026-11-10 の5回目のレビュー 2 で、
    島根の主系統を記録する側に入っていなかったことが分かった。
    """
    region = str(region).lower()
    from . import splits as _sp
    exp = _sp.expected(region) or {}
    if int(n_tiles) == int(exp.get("n_test", -1)):
        return                      # テスト分割なら学習タイルを含まない
    tr = _trained_region(cond)
    if tr and tr == region:
        raise ValueError(
            f"{cond.name} は {region} で学習した重みです。"
            f"{int(n_tiles):,} 枚は**学習タイルを含む集合**なので記録できません"
            "（学習データでの値が成績として残ります。設計書 §6）。"
            " 見るだけなら記録を切ってください。")


def check_training_region_names(names, region, n_tiles=None):
    """**構成員の全員**に `check_training_region` を掛ける。**推論の前に呼ぶ。**

    `analysis/` の記録の経路（オーバーラップ・刈り取り・2段目・和集合）が呼ぶ。
    2026-11-11 のレビューで、歯止めが `run_overlap_eval` にしか無く、
    しかも**推論のあと**で呼ばれていたことが分かった。

    `n_tiles` を省くと、**その地域で `analysis/` の道が測る集合**の枚数を
    `regions.yaml` の `subset` から決める（`all`＝全件・`gtonly`＝正解あり・
    `holdout`＝テスト分割）。それ以外の `subset`（`blocktest`・`background`）は
    **ここでは見ない** ── 分割と集合の組み合わせは `splits.fits_subset` が見る。
    """
    from . import regions as _rg
    from . import splits as _sp
    from .registry import get
    region = str(region).lower()
    if n_tiles is None:
        if region not in _rg.keys():
            return
        r = _rg.get(region)
        key = {"all": "n_all", "gtonly": "n_used",
               "holdout": "n_test"}.get(r.subset)
        exp = _sp.expected(r.dir)
        if key is None or exp is None:
            return
        n_tiles, region = exp[key], str(r.dir).lower()
    for n in names:
        check_training_region(get(n) if isinstance(n, str) else n, region, n_tiles)


def can_record_split(region, split="test"):
    """**pkl を読む前**に「この地域・この split を記録できるか」を返す。

    返りは `(記録してよいか, 理由)`。道具（`tools/run_eval.py` /
    `tools/run_training.py`）は**この規則を自分で持たない**
    （2026-11-10 の再々レビュー 5 ── `region != "shimane"` が
    2つの道具に直書きされていた。設計書 §5 と同じ形の事故である）。

    `build_loaders` が作る分割から静的に決まることだけを見る。

    | 地域 | split | |
    |---|---|---|
    | どこでも | `test` 以外 | **記録しない**（鍵に split が入らない）|
    | 広島 | `test` | 記録できる（2,964枚＝`警戒のみ`）|
    | 島根 | `test` | **記録しない**（1,936枚にラベルが無い）|

    枚数そのものの検査は `_check_recordable`（`loaders` が要る）。
    """
    if split != "test":
        return (False, f"split={split!r} は記録できません（鍵に split が"
                       "入らないので、テストの行を置き換えます。設計書 §6）")
    # **規則を書き写さない**（4回目のレビュー B）。`EXPECT` から偽の `loaders`
    # を作って `_recordable` に通す ── 判断は1か所だけに置く。
    from . import splits as _sp
    exp = _sp.expected(region)
    if exp is None:
        return (False, f"{region} は dc5lib/splits.py の EXPECT に無いので、"
                       "記録してよい枚数が分かりません（実装書 §4）")
    import types
    ld = types.SimpleNamespace(
        n_all=exp["n_all"], n_used=exp["n_used"],
        split={"n_test": exp["n_test"], "kind": "random",
               "test_size": _sp.STD["test_size"],
               "val_size": _sp.STD["val_size"],
               "random_state": _sp.STD["random_state"]})
    ok = _recordable(ld, str(region).lower(), 0.0)
    hit = [k for k, v in ok.items() if int(v) == int(exp["n_test"])]
    if hit:
        return (True, "")
    table = "、".join(f"{k} = {v:,} 枚" for k, v in ok.items()) or "なし"
    return (False, f"{region} のテスト分割（{exp['n_test']:,} 枚）には"
                   f"ラベルがありません（記録できるのは {table}）。"
                   "実装書 §8 を参照")


def _check_recordable(cond, loaders, split):
    """**推論の前に**、この測り方が記録できるかを見る（設計書 §6・§7）。

    `r.n_tiles` は推論のあとにしか分からないので、
    **DataLoader の中身の数**で先に確かめる。1時間回してから怒らない。
    """
    region = str(getattr(loaders, "region", "") or "").lower()
    if not region:
        return
    bg = float(getattr(cond, "bg_ratio", 0.0) or 0.0)
    ok = _recordable(loaders, region, bg)
    got = len(getattr(getattr(loaders, split, None), "dataset", None) or [])
    if any(got == int(v) for v in ok.values()):
        # **学習に使った地域では、テスト分割以外を記録しない**
        # （4回目のレビュー A）。島根の `全件`／`警戒のみ` は学習タイルを含む。
        check_training_region(cond, region, got)
        return
    table = "、".join(f"{k} = {v:,} 枚" for k, v in ok.items()) or "なし"
    raise ValueError(
        f"{cond.name} / {region} / {split} の {got:,} 枚は記録できません。"
        f"記録の鍵は tileset の語だけで「どのタイルを測ったか」を言うので、"
        f"**この枚数にはラベルがありません**（{region} で記録できるのは {table}）。"
        f"このまま記録すると、同じ鍵の別の集合の行を置き換えます（設計書 §6）。"
        f"\n  見るだけなら record=False。"
        f"\n  島根を記録する経路は2つ ── 主系統（全 24,569 枚）は"
        f" analysis/run_overlap_eval.py --record、背景を含まない補助"
        f"（正解あり 9,679 枚）は templates/eval_shimane.ipynb のセル3です。"
        f"\n  （analysis/run_shimane_scope.py は**記録しない** ──"
        f" キャッシュの CSV を読んで表示するだけです）")


def _shifted_parts(ds) -> int:
    """**ずらした窓を返すデータセット**が何本混じっているか。

    `OverlapDataset` は `deterministic=False` かつ `max_shift > 0` のとき、
    素のタイルではなく**乱数でずらした窓**を返す（学習用）。
    `ConcatDataset` でつないだものは中身を1本ずつ見る。
    """
    parts = getattr(ds, "datasets", None)
    if isinstance(parts, (list, tuple)):
        return sum(_shifted_parts(p) for p in parts)
    return int((not getattr(ds, "deterministic", True))
               and int(getattr(ds, "max_shift", 0) or 0) > 0)


_SHIFTED_MSG = (
    "測ろうとしている DataLoader に**ずらした窓を返す学習用データセット**が"
    "入っています（オーバーラップ条件の `datasets['train']`）。"
    "**素のタイルの値になりません**（乱数でずれるので測るたびに変わる）。"
    " 警戒区域ありの全件を素のタイルで測るなら `loaders.datasets['all']` を"
    "使ってください（設計書 §6）。")


def _guard_record(cond, loaders, split):
    """**記録の前に見るものを全部。推論の前に呼ぶ**（設計書 §7）。

    どれかに当たれば `ValueError`。`evaluate(record=True)` と
    `can_record` の両方がここを通る ── 規則を2か所に書かない。
    """
    # 記録の鍵に split が入らないので、検証や学習の値を記録すると
    # テストの行を置き換える（レビュー 1.1）。
    if split != "test":
        raise ValueError(
            f"split={split!r} の値は記録できません。記録の鍵は "
            "(condition, region, tileset, scope, metric_kind, setting) で "
            "**split が入らない**ため、テストの行を置き換えてしまいます"
            "（設計書 §6）。見るだけなら record=False を渡してください。")
    if _shifted_parts(getattr(getattr(loaders, split, None), "dataset", None)):
        raise ValueError(f"{cond.name}: {_SHIFTED_MSG}")
    _check_recordable(cond, loaders, split)


def can_record(cond, loaders, split="test"):
    """**`loaders` を作ったあと・学習の前**に「記録できるか」を返す。

    返りは `(記録してよいか, 理由)`。`can_record_split` は地域と split しか
    見ないので、**csv 分割の条件は通ってしまい、学習が終わったあとの
    `evaluate(record=True)` で初めて止まる**（2026-11-11 のレビューで実測）。
    こちらは `evaluate` と同じ検査（`_guard_record`）を例外にせず返す。
    """
    try:
        _guard_record(cond, loaders, split)
    except ValueError as e:
        return (False, str(e))
    return (True, "")


def _record(cond, loaders, r, split, meta, verbose):
    """画素と箇所を `results/evals/` に記録する。"""
    from . import results as R
    region = str(getattr(loaders, "region", "") or "").lower()
    if not region:
        print("  ※ 記録できません: loaders に region がありません")
        return []
    run_id = _run_id_for(cond, meta)
    bg = float(getattr(cond, "bg_ratio", 0.0) or 0.0)
    tileset = _tileset_for(loaders, region, bg, r.n_tiles)
    # `_check_recordable` が推論の前に見ているので、ここに来たら必ず決まる。
    # **決まらなければ書かない**（鍵が別の集合と衝突する）。
    if tileset is None:
        print(f"  ※ 記録できません: {r.n_tiles:,} 枚にラベルがありません")
        return []
    # **split の名前で枚数を語らない**（4回目のレビュー A3）。
    # `eval_shimane` のセル3は `loaders.test` を差し替えて
    # 学習・検証・テストを合わせた 9,679枚を渡すので、「test 9,679枚」は嘘になる。
    note = ("dc5lib.evaluate.evaluate。**素のタイル**（オーバーラップ推論ではない）"
            f" / {tileset} {r.n_tiles:,} 枚（loaders.{split} の中身）")
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
    # **推論の前に止める**（設計書 §7）。split・ずらした窓・枚数を見る。
    if record:
        _guard_record(cond, loaders, split)
    elif verbose and _shifted_parts(
            getattr(getattr(loaders, split, None), "dataset", None)):
        print(f"  ※ {_SHIFTED_MSG}")
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
