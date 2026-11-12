# -*- coding: utf-8 -*-
"""タイルの分割表を読む。

既存条件の分割は `train_test_split(random_state=42)` で再現できるので
表を持つ必要がなかったが、**空間ブロック分割は乱数だけでは再現できない。**
ブロックの割り当てを、距離別の正例率が揃うように探して決めているためである
（`analysis/run_block_split.py`）。そこで結果を CSV に固定し、
学習ノートブックと評価の両方がこれを読む。

    from dc5lib import splits
    s = splits.load()                 # DataFrame
    splits.tile_split()               # {タイル番号: train/val/test}
    splits.test_tiles()               # テスト側のタイル番号の集合
    splits.dist_bin()                 # {タイル番号: '1-3' / '4-15' / '16+' ...}

**学習と評価で同じ表を見ること。** 片方だけ別の分割を使うと、
学習に使ったタイルで評価してしまう。
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import pandas as pd

from . import paths

DEFAULT = "hiroshima_block_20km"


def path(name: str = DEFAULT):
    return paths.REPO_ROOT / "splits" / f"{name}.csv"


@lru_cache(maxsize=8)
def load(name: str = DEFAULT) -> pd.DataFrame:
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(
            f"分割表がありません: {p}\n"
            "python analysis/run_block_split.py --write --km 20 で作れます。")
    return pd.read_csv(p)


@lru_cache(maxsize=8)
def tile_split(name: str = DEFAULT) -> dict:
    d = load(name)
    return dict(zip(d.tile_no.astype(int), d.split))


@lru_cache(maxsize=8)
def test_tiles(name: str = DEFAULT) -> frozenset:
    d = load(name)
    return frozenset(d.loc[d.split == "test", "tile_no"].astype(int))


@lru_cache(maxsize=8)
def dist_bin(name: str = DEFAULT) -> dict:
    """テストタイルの距離の層。学習データから何歩離れているか。

    `unreachable` は島など、学習タイルから格子をたどって到達できないもの。
    **層別の集計から外すこと。** 距離が定義できない。
    """
    d = load(name)
    return dict(zip(d.tile_no.astype(int), d["bin"].fillna("")))


# 既定の乱数分割（kind=random / test_size 0.2 / val_size 0.25 /
# random_state 42）で合うべき値。**この辞書が唯一の原本である。**
#
# `tools/run_training.py` の EXPECT・`tools/audit_splits.py` の EXPECT・
# `templates/train.ipynb` の照合セル・実装書 `4` の表に、
# **同じ値が別々に並んでいた**（2026-11-10 の再レビュー 3）。
# 構成員の名前が3か所に並んでいたのと同じ形で（設計書 §5）、
# 実際に実装書の島根の指紋が間違っていて食い違いかけていた。
#
# **`fingerprint` は地域をまたいで比べられない** ── タイル番号そのものを
# 混ぜるので、同じ random_state でも地域が違えば必ず違う値になる。
# `n_all` も持つ ── これがあると `evaluate.can_record_split` が
# **pkl を読まずに**「記録してよい枚数」を導ける（4回目のレビュー B）。
EXPECT = {
    "Hiroshima": {"n_all": 38148, "n_used": 14816, "n_train": 8889,
                  "n_val": 2963, "n_test": 2964,
                  "tiles_fingerprint": "a75ad23bad3e",
                  "fingerprint": "653b8cbe3d28"},
    "Shimane": {"n_all": 24569, "n_used": 9679, "n_train": 5807,
                "n_val": 1936, "n_test": 1936,
                "tiles_fingerprint": "ff11694a9f3f",
                "fingerprint": "2dd13cd65a6c"},
}
# 既定の値（`traincfg.DEFAULTS` と同じ）。これと違う分割は EXPECT と比べない
STD = {"test_size": 0.2, "val_size": 0.25, "random_state": 42}


def expected(region) -> dict | None:
    """地域名から期待値を引く（`Hiroshima` / `hiroshima` どちらでもよい）。"""
    for k, v in EXPECT.items():
        if k.lower() == str(region).lower():
            return dict(v)
    return None


def compare_expected(region, n_used, split) -> dict:
    """`EXPECT` と照合して**違うものだけ** `{鍵: (実際, 期待)}` を返す。

    **`split` に無い鍵は見ない。** `n_all` は `resolve` の返りに無いので、
    素朴に `got[k]` を書くと **`KeyError`** になる ── 実際に
    `EXPECT` へ `n_all` を足したとき、`tools/run_training.py` と
    `templates/train.ipynb` が pkl を読んだ直後に死んだ
    （2026-11-10 の5回目のレビュー 1）。

    **照合はここだけに置く。** 呼ぶのは `tools/run_training.py`・
    `tools/audit_splits.py`・`templates/train.ipynb` の3つ。
    """
    want = expected(region)
    if not want:
        return {}
    got = {"n_used": int(n_used)}
    for k in ("n_train", "n_val", "n_test",
              "tiles_fingerprint", "fingerprint"):
        if k in split:
            got[k] = split[k]
    return {k: (got[k], v) for k, v in want.items()
            if k in got and got[k] != v}


def is_standard(sp: dict) -> bool:
    """`EXPECT` と比べてよい分割か（`kind=random` かつ既定の値）。"""
    return (sp.get("kind") == "random"
            and all(sp.get(k) == v for k, v in STD.items()))


def spec_is_standard(cond) -> bool:
    """**pkl を読む前**に、条件の分割の**宣言**が既定の乱数分割かを見る。

    `is_standard` は `resolve` の返り（pkl を読んだあと）を見る。
    こちらは config だけを見るので、推論の前・学習の前に使える。
    値の決まり方は `resolve` と同じ順（`split:` → `train:` → 既定）。
    """
    from .traincfg import DEFAULTS
    spec = dict(getattr(cond, "split", None) or {})
    if str(spec.get("kind") or "random") != "random":
        return False
    tr = dict(getattr(cond, "train", None) or {})
    for k, v in STD.items():
        got = spec.get(k, tr.get(k) if tr.get(k) is not None else DEFAULTS[k])
        if float(got) != float(v):
            return False
    return True


def fits_subset(cond, subset) -> tuple:
    """`analysis/` の道で、**その条件をそのタイル集合で測ってよいか。**

    返りは `(よいか, 理由)`。

    `analysis/data.py` と `analysis/run_overlap_eval.py` は条件の `split:` を
    見ない ── 広島の `holdout` は**いつも乱数分割（random/42）のテスト**を
    作る。だから**別の分割で学習した条件**をそこで測ると、テストに
    その条件の学習タイルが入る（2026-11-11 のレビュー。
    `FinalFusion_SAM_APM_BlockSplit` を乱数テストで測ると 2,964枚のうち
    2,158枚が学習か検証のタイルである。`tools/audit_splits.py` の実測）。
    逆向きも同じで、乱数分割の条件をブロック分割のテストで測ると漏れる。

    | subset | 測ってよい条件 |
    |---|---|
    | `holdout`（広島の乱数テスト）| 宣言が既定の乱数分割 |
    | `blocktest`（ブロック分割のテスト）| `split.kind == csv` |
    | それ以外（別の県・背景）| どれでも |
    """
    kind = str((getattr(cond, "split", None) or {}).get("kind") or "random")
    name = getattr(cond, "name", cond)
    if subset == "holdout" and not spec_is_standard(cond):
        return (False,
                f"{name} は既定の乱数分割で学習していません（split.kind={kind}）。"
                "広島の holdout は random/42 のテストなので、"
                "**この条件の学習タイルが入ります**。"
                "ブロック分割の条件は --region hiroshima_block で測ってください")
    if subset == "blocktest" and kind != "csv":
        return (False,
                f"{name} はブロック分割で学習していません（split.kind={kind}）。"
                "ブロック分割のテストには**この条件の学習タイルが入ります**")
    return (True, "")



def fingerprint(parts: dict) -> str:
    """分割の指紋。**同じ指紋なら同じタイルが同じ側にある。**

    `parts` は {"train": [タイル番号...], "val": [...], "test": [...]}。
    順序に依らないよう各集合を並べ替えてから混ぜる。

    > **ここが1か所にあることが要点である。** `pipeline` と
    > `tools/audit_splits.py` が別々に計算すると、いずれ食い違って
    > 「同じ分割か」の判定が信用できなくなる。
    """
    h = hashlib.sha256()
    for k in ("train", "val", "test"):
        h.update(k.encode())
        for i in sorted(int(x) for x in parts.get(k, ())):
            h.update(i.to_bytes(8, "little", signed=True))
    return h.hexdigest()[:12]


def tiles_fingerprint(no) -> str:
    """pkl のタイル番号の並びの指紋。**並びが同じかを見る。**

    `train_test_split` が返すのは添字なので、**並びが違えば同じ添字が
    別のタイルを指す。** だから分割の指紋とは別にこれも記録する。
    """
    h = hashlib.sha256()
    for x in no:
        h.update(repr(int(x)).encode())
    return h.hexdigest()[:12]


def resolve(cond, no, cfg) -> dict:
    """条件とタイル番号から、学習・検証・テストの**添字**を決める。

    `cond.split` の `kind` で種類を選ぶ。**空なら `random`** として
    `cfg`（`train:` 由来）の値を使う ── 既存条件はこれで、いまの
    ノートブックと1枚も違わない結果になる。

        {"kind": "random", "test_size": 0.2, "val_size": 0.25, "random_state": 42}
        {"kind": "csv", "file": "hiroshima_block_20km"}

    `csv` は `dc5lib/splits.py` の表を引く（**空間ブロック分割はここ**）。
    表に載っていないタイルは**どの集合にも入れない**。黙って学習に
    混ぜるより、落としたことを数で見せるほうが安全である。

    返す辞書は `idx` のほかに `kind` と指紋と枚数を持つ。
    そのまま `ckpt.build_meta(split=...)` に渡せる。
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    spec = dict(getattr(cond, "split", None) or {})
    kind = str(spec.get("kind") or "random")
    n = len(no)
    ids = [int(x) for x in no]

    if kind == "random":
        ts = float(spec.get("test_size", cfg.test_size))
        vs = float(spec.get("val_size", cfg.val_size))
        rs = int(spec.get("random_state", cfg.random_state))
        idx_all = np.arange(n)
        idx_tv, idx_te = train_test_split(idx_all, test_size=ts,
                                          random_state=rs)
        idx_tr, idx_va = train_test_split(idx_tv, test_size=vs,
                                          random_state=rs)
        detail = {"test_size": ts, "val_size": vs, "random_state": rs}
        dropped = 0
    elif kind == "csv":
        name = str(spec.get("file") or DEFAULT)
        table = tile_split(name)
        where = {"train": [], "val": [], "test": []}
        dropped = 0
        for i, t in enumerate(ids):
            s = table.get(t)
            if s in where:
                where[s].append(i)
            else:
                dropped += 1
        idx_tr = np.asarray(where["train"], dtype=int)
        idx_va = np.asarray(where["val"], dtype=int)
        idx_te = np.asarray(where["test"], dtype=int)
        detail = {"file": name, "rows": len(table)}
    else:
        raise ValueError(
            f"分割の kind が分かりません: {kind!r}。"
            "対応しているのは 'random' と 'csv' です")

    parts = {k: [ids[int(i)] for i in v] for k, v in
             (("train", idx_tr), ("val", idx_va), ("test", idx_te))}
    if set(parts["train"]) & set(parts["test"]):
        raise AssertionError("学習とテストが重なっています")
    if set(parts["train"]) & set(parts["val"]):
        raise AssertionError("学習と検証が重なっています")
    # 選択は検証F なので、検証とテストが重なるとテストで選んだことになる
    if set(parts["val"]) & set(parts["test"]):
        raise AssertionError("検証とテストが重なっています")
    return {
        "kind": kind,
        "idx": {"train": idx_tr, "val": idx_va, "test": idx_te},
        "tiles": parts,
        "fingerprint": fingerprint(parts),
        "tiles_fingerprint": tiles_fingerprint(ids),
        "n_tiles": n,
        "n_train": len(idx_tr), "n_val": len(idx_va), "n_test": len(idx_te),
        "n_dropped": dropped,
        **detail,
    }


def describe_split(sp: dict) -> str:
    d = (f"  file={sp['file']}" if sp["kind"] == "csv"
         else f"  test_size={sp['test_size']} val_size={sp['val_size']}"
              f" random_state={sp['random_state']}")
    out = [f"分割 {sp['kind']}{d}",
           f"  タイル {sp['n_tiles']:,} / 学習 {sp['n_train']:,}"
           f" / 検証 {sp['n_val']:,} / テスト {sp['n_test']:,}",
           f"  分割の指紋 {sp['fingerprint']} / 並びの指紋 "
           f"{sp['tiles_fingerprint']}"]
    if sp["n_dropped"]:
        out.append(f"  **表に無くて落としたタイル {sp['n_dropped']:,} 枚**")
    return "\n".join(out)


def describe(name: str = DEFAULT) -> str:
    d = load(name)
    c = d.split.value_counts()
    te = d[d.split == "test"]
    b = te["bin"].value_counts()
    lines = [f"{name}: 学習 {c.get('train', 0)} / 検証 {c.get('val', 0)} "
             f"/ テスト {c.get('test', 0)}"]
    for k in ("1-3", "4-15", "16+", "unreachable"):
        if k in b:
            lines.append(f"  距離 {k:<12} {b[k]:>5} 枚")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
