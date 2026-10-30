# -*- coding: utf-8 -*-
"""アンサンブル（和集合）の定義と、**混ぜてよいかの検査**。

なぜ要るか
----------
和集合は構成員が**同じテストタイルで測られていて**初めて意味を持つ。
構成員Aの学習タイルがBのテストタイルに入っていたら、和集合の再現率は
データリークで膨らむ。ところが構成員の名前は

    analysis/run_stage2_room.py   STAGE1 / UNION3
    analysis/run_miss_overlap.py  DEFAULT
    tools/audit_splits.py         DEFAULT

の3か所に別々に書かれていた。**どれかを足し忘れると、測った集合と
語っている集合が食い違う。** だから宣言を `unions/*.yaml` に集め、
コードはそこから読む。

    from dc5lib import union
    e = ensemble.get("union5")
    print(e.describe())
    ensemble.verify(e)            # 混ぜてよいかを確かめる。だめなら例外

検査するもの
------------
1. 構成員が実在し、重みがあるか
2. **`ensemble: false` の条件が入っていないか**
   （分割か選択が他と違う条件。混ぜると和集合の意味が変わる）
3. **全員の分割が同じか。** 重みの `meta.split.fingerprint` を見るので
   **pkl を読み直さない**（`dc5lib/ckpt.py`）。素性の無い古い重みは
   「分からない」として報告し、`tools/audit_splits.py` を促す
4. 入力が全員同じでないこと（同じなら本数を増やす意味が薄い）は
   **止めない。** 情報として出す
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import ckpt as _ckpt
from . import paths
from .registry import get as _get


def dir_() -> Path:
    return paths.REPO_ROOT / "unions"


def names() -> list:
    return sorted(p.stem for p in dir_().glob("*.yaml"))


@dataclass
class UnionSet:
    name: str
    base: str
    members: list
    label: str = ""
    note: str = ""
    recorded: str = ""
    path: Path = field(default=None, repr=False)

    @property
    def all(self) -> list:
        """**基準を先頭にした全構成員。** 順序は比較の基準を決める。"""
        return [self.base] + list(self.members)

    def describe(self) -> str:
        out = [f"{self.name}  {self.label or ''}".rstrip(),
               f"  構成員 {len(self.all)} 本（先頭が基準）"]
        for i, n in enumerate(self.all):
            try:
                c = _get(n)
                ch = "+".join(x.source for x in c.inputs)
                flag = "" if c.ensemble else "  **ensemble: false**"
                w = "重みあり" if c.weights.exists() else "**重みなし**"
                out.append(f"    {'基準' if i == 0 else '    '} {n:<44}"
                           f"{ch:<14}{c.protocol:<20}{w}{flag}")
            except Exception as e:                        # noqa: BLE001
                out.append(f"         {n:<44}**読めません: {e}**")
        if self.recorded:
            out.append(f"  記録済み: {self.recorded}（**この定義は動かさない**）")
        if self.note:
            out.append(f"  {self.note}")
        return "\n".join(out)


def get(name: str) -> UnionSet:
    p = dir_() / f"{name}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"和集合（unions/）の定義がありません: {p}\n"
            f"あるのは {names()}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return UnionSet(name=d.get("name", name), base=d["base"],
                    members=list(d.get("members") or []),
                    label=d.get("label", ""), note=d.get("note", ""),
                    recorded=str(d.get("recorded", "")), path=p)


def verify(e: UnionSet, strict=True) -> dict:
    """混ぜてよいかを確かめる。`strict` で例外にするか報告だけにするか。"""
    problems, warn = [], []
    conds = {}
    for n in e.all:
        try:
            conds[n] = _get(n)
        except Exception as ex:                           # noqa: BLE001
            problems.append(f"{n} の config が読めません: {ex}")
    for n, c in conds.items():
        if not c.weights.exists():
            problems.append(f"{n} の重みがありません（{c.weights}）")
        if not c.ensemble:
            problems.append(
                f"{n} は ensemble: false です。"
                f"分割か選択が他と違うので混ぜられません（protocol={c.protocol}）")

    # --- 分割。**重みの素性を見る。pkl は読まない**
    fps, unknown = {}, []
    for n, c in conds.items():
        m = _ckpt.meta_of(c.weights) if c.weights.exists() else None
        fp = ((m or {}).get("split") or {}).get("fingerprint")
        if fp:
            fps[n] = fp
        else:
            unknown.append(n)
    if len(set(fps.values())) > 1:
        problems.append(
            "構成員の分割が食い違っています: "
            + " / ".join(f"{k}={v}" for k, v in fps.items()))
    if unknown:
        warn.append(
            f"分割を重みから確かめられない構成員が {len(unknown)} 本あります"
            f"（素性の無い古い重み）: {', '.join(unknown)}\n"
            f"    → python tools/audit_splits.py --region both --conditions "
            + " ".join(e.all))

    # --- 入力の多様性（止めない。情報として出す）
    sig = {n: "+".join(x.source for x in c.inputs) for n, c in conds.items()}
    if len(set(sig.values())) == 1:
        warn.append(f"全員の入力が同じです（{next(iter(sig.values()))}）。"
                    "本数を増やす効き目は構造か学習の違いだけになります")

    if problems and strict:
        raise AssertionError("和集合の検査で止めました:\n  - "
                             + "\n  - ".join(problems))
    return {"problems": problems, "warn": warn,
            "fingerprints": fps, "unknown": unknown, "inputs": sig}


def record(name, members, region, scores, n_blobs, floor, csv_name,
           setting=None):
    """**和集合そのもの**（刈り取り前）を `results/evals/` に記録する。

    なぜ要るか
    ----------
    `42`〜`45` の和集合の箇所R/P/F と上限は **`metrics.csv` に無かった**
    （`docs/設計_アンサンブルの記録.md` の `2.5`）。刈り取り後だけが
    記録されていたので、「和集合で何本ぶん広がったか」を表から追えなかった。

    `scores` は `(箇所R, 箇所P, 箇所F, 予測数)`。

    **床と天井は `recall`/`precision` 列に入れない。** 箇所R/P/F の形を
    していないので、入れると列の意味が壊れる（同 `3.4`）。`note` に書く。
    """
    from . import results as _R
    from .instance_eval import PRIMARY

    RU, PU, FU, nU = scores
    cond = f"Union_{name}"
    tileset = {"hiroshima": "警戒のみ", "shimane": "全件"}[region]
    run_id = _R.new_run_id(cond)
    prov = {"members": "|".join(members), "n_members": len(members)}
    ceil_f = 2 * RU / (1 + RU) if RU else 0.0
    pct = 100 * floor / max(n_blobs, 1)
    note = (f"和集合そのもの（刈り取り前）。構成員 {len(members)}本。"
            f"床 {floor:,}件 / {pct:.2f}%、天井 {100 - pct:.2f}%。"
            f"正解のかたまり {n_blobs:,} / 予測 {nU:,}。"
            f"出どころ {csv_name} / analysis/run_union_blind.py")
    rows = (("和集合", RU, PU, FU),
            ("和集合上限", RU, 1.0, ceil_f))
    n = 0
    for scope, r, pr, f1 in rows:
        _R.record_eval(run_id=run_id, condition=cond, region=region,
                       tileset=tileset, scope=scope, metric_kind="箇所",
                       setting=setting or PRIMARY,
                       recall=r, precision=pr, f1=f1,
                       n_gt_instances=n_blobs, n_pred_instances=nU,
                       note=note, **prov)
        n += 1
    out = _R.build_metrics_csv()
    print(f"\n記録: evals {n} 件 / metrics.csv を更新（{out}）")
    print(f"  condition={cond} / scope=和集合・和集合上限"
          f" / 構成員 {len(members)}本")
    print("  **床と天井は note にだけ書いた**"
          "（箇所R/P/F の形をしていないので列に入れない）")
    return cond


def report(e: UnionSet) -> str:
    r = verify(e, strict=False)
    out = [e.describe(), ""]
    if r["problems"]:
        out.append("**混ぜられません**")
        out += [f"  - {p}" for p in r["problems"]]
    else:
        out.append("混ぜてよい条件は満たしています（ensemble / 重み）")
    if r["fingerprints"]:
        u = set(r["fingerprints"].values())
        out.append(f"分割の指紋 {', '.join(sorted(u))}"
                   f"{'（全員一致）' if len(u) == 1 else ' **食い違い**'}")
    for w in r["warn"]:
        out.append(f"  ※ {w}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:] or names()
    for a in args:
        print("=" * 90)
        print(report(get(a)))
