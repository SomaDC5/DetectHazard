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
