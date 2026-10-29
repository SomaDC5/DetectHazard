# -*- coding: utf-8 -*-
"""pkl から `DataLoader` までを1本で作る。**ノートブックに手順を書かない。**

なぜ要るか
----------
`docs/設計図_現状.md` の `2` のとおり、いまのノートブックは
`bg_ratio` も `batch_size` も `test_size` も**直書き**で、config の値とは
手で合わせているだけだった。1件すでにずれていた。
**手順を1か所に置けば、ずれようがない。**

ここが引き受けること
--------------------
1. pkl を読む
2. **学習・検証・テストに使うのは警戒区域を含むタイルだけ**（`max(Mask) >= 1`）。
   既存ノートブックの `dataset2` と同じ。背景タイルは `dataset3` 側で、
   `bg_ratio > 0` のときだけ**学習データに**混ぜる
3. **近傍を貼る材料は全タイル（背景を含む）。** 既存の
   `ctx_* = list(dataset2.X) + list(dataset3.X)` と同じ順で持つ。
   **並びがこの順であることが添字の前提**で、`dataset2` が先にあるから
   `idx_train` / `idx_val` / `idx_test` がそのまま `ctx` の添字になる
3. マスクの前処理（ガウシアン σ=1 → >0）
4. 分割（`dc5lib.splits.resolve`。`random` と `csv`＝ブロック分割）
5. `OverlapDataset` を学習・検証・テストぶん作る
   （**検証とテストは `deterministic=True`** で素のタイルになる）
6. 各入力の形・dtype・平均を数えて返す（**検算をノートブックで見せるため**）

使い方
------
    from dc5lib.pipeline import build_loaders
    ld = build_loaders(get(CONDITION), "Hiroshima")
    print(ld.describe())
    # ld.train / ld.val / ld.test は torch の DataLoader
    # ld.split は ckpt.build_meta(split=...) にそのまま渡せる

**気をつけること**

- **近傍は背景タイルからも貼る。** 警戒区域を含むタイルだけで貼ると
  8近傍が揃うのは 0.2% しかない（格子が疎なので）。背景も入れると上がる。
  **背景タイルは学習・検証・テストのどこにも属さない**（`bg_ratio=0`）ので、
  近傍として使っても漏れにはならない。
- **漏れるのは検証・テストのタイルを近傍にしたときだけ。** だから
  `allowed_except(len(ctx), 検証, テスト)` でその2つだけを外し、
  **外れていることを assert で確かめている。** `allowed=None` は使わない
  （全タイルを貼るので検証・テストが入る）。
- ずらす幅は最大64px＝タイルの半分なので、**窓は隣のタイルの半分までしか
  入らない。** その隣が背景なら問題なく、検証・テストなら漏れる。
  上の assert がそこを止めている。
- メモリ。航空写真と3入力目は **uint8 のまま持つ**。float32 で全件持つと
  島根で数GBになる。`/255` は `OverlapDataset.__getitem__` がやる。
- `bg_ratio > 0` は**まだ通していない**。`NotImplementedError` を投げる。
  既存の背景混ぜは `dataset3` から `random.seed(CFG.seed)` で選ぶ手順で、
  **再現を確かめずに書くと黙って違うものを学習する**ので、
  必要になったときに実データで確かめてから入れる。
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import inputs as _inputs
from . import splits as _splits
from .data import load_dataset, preprocess_mask
from .overlap import OverlapDataset, allowed_except
from .traincfg import train_cfg


@dataclass
class Loaders:
    """`build_loaders` の返り。**ノートブックはこれを表示するだけでよい。**"""
    train: DataLoader
    val: DataLoader
    test: DataLoader
    split: dict                      # splits.resolve の返り（idx 入り）
    fields: dict                     # 入力ごとの形・dtype・平均
    region: str
    n_all: int                       # pkl の全タイル
    n_used: int                      # 警戒区域を含むタイル（学習・検証・テストの母数）
    n_bg: int = 0                    # 背景タイル。**近傍の材料にはなる**
    n_allowed: int = 0               # 貼ってよい近傍の枚数
    overlap: dict = field(default_factory=dict)
    datasets: dict = field(default_factory=dict)

    def meta_split(self) -> dict:
        """`ckpt.build_meta(split=...)` に渡す形（配列を落とす）。"""
        return {k: v for k, v in self.split.items()
                if k not in ("idx", "tiles")}

    def describe(self) -> str:
        out = [f"地域 {self.region} / 全 {self.n_all:,} 枚 = "
               f"警戒区域あり {self.n_used:,} + 背景 {self.n_bg:,}",
               _splits.describe_split(self.split)]
        if self.overlap:
            out.append(f"オーバーラップ max_shift={self.overlap['max_shift']}"
                       f" seed={self.overlap['seed']} / "
                       f"8近傍が揃うタイル {100 * self.overlap['cov']:.1f}%"
                       f"（平均 {self.overlap['mean']:.2f}/8）")
            out.append(f"  貼ってよい近傍 {self.n_allowed:,} 枚"
                       f"（全 {self.n_all:,} から検証とテストを外した。"
                       f"**背景は近傍に使う**）")
        else:
            # **黙ってはいけない。** config の overlap: を書き忘れると
            # 素のタイルで学習が走り、名前だけ OverlapTrain になる
            out.append("**オーバーラップ学習なし（素のタイルで学習）**"
                       " ── config の overlap.max_shift が 0 か未設定")
        for k, v in self.fields.items():
            out.append(f"入力 {k:<10}{v['shape']} {v['dtype']}"
                       f"  平均 {v['mean']:.5f}"
                       f"  >0 の割合 {100 * v['pos']:.2f}%")
        return "\n".join(out)


def _stats(arr_list, where, n=256, seed=0):
    """**学習に使うタイルから** n 枚を抜いて平均と「>0 の割合」を数える。

    全件は読まない（島根で数GBになる）。`where` に学習の添字を渡すので、
    背景タイルや検証・テストの値は混ざらない。
    """
    where = np.asarray(where)
    if where.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    pick = where if where.size <= n else rng.choice(where, n, replace=False)
    a = np.stack([np.asarray(arr_list[int(i)], dtype=np.float32) for i in pick])
    return float(a.mean()), float((a > 0).mean())


def build_loaders(cond, region="Hiroshima", batch_size=None, transform=None,
                  num_workers=0, verbose=True) -> Loaders:
    cfg = train_cfg(cond)
    bs = int(batch_size or cfg.batch_size)
    bg = float(getattr(cond, "bg_ratio", 0.0) or 0.0)
    if bg > 0:
        raise NotImplementedError(
            f"{cond.name} は bg_ratio={bg} です。背景タイルの混ぜ方は"
            "まだ pipeline に入れていません（実データで再現を確かめてから入れる）。"
            "docs/ノートブックの作り方.md の「まだ通っていない道」を参照。")

    path = cond.pkl_for(region)
    if verbose:
        print(f"読み込み: {path.name} "
              f"({path.stat().st_size / 1024 ** 3:.1f} GB) …", flush=True)
    ds = load_dataset(path)
    n_all = len(ds.No)

    # --- 警戒区域を含むタイル（既存の dataset2）と背景（dataset3）に分ける。
    # **並びは keep が先、bg が後。** 既存の
    # `ctx = list(dataset2.X) + list(dataset3.X)` と同じで、
    # これが「dataset2 の添字がそのまま ctx の添字になる」前提を作る。
    keep, bgs = [], []
    for i in range(n_all):
        (keep if np.max(np.asarray(ds.Mask[i])) >= 1 else bgs).append(i)
    order = keep + bgs
    assert len(order) == n_all, "タイルの振り分けが全数と合いません"
    no = [int(ds.No[i]) for i in keep]

    terr_f = cond.inputs[0].field or "DEM"
    air_f = cond.inputs[1].field or "AirPhoto" if len(cond.inputs) > 1 else ""
    dem = [np.asarray(getattr(ds, terr_f)[i], dtype=np.float32) for i in order]
    air = ([np.asarray(getattr(ds, air_f)[i], dtype=np.uint8) for i in order]
           if air_f else None)
    mask = [preprocess_mask(ds.Mask[i]) for i in order]
    geo = [ds.GeoInfo[i] for i in order]

    third = lu = None
    t3 = _inputs.third_input(cond)
    if _inputs.needs_landuse(cond):
        lu = [np.asarray(ds.LandUse[i]) for i in order]
    elif t3 is not None:
        third = [np.asarray(getattr(ds, t3.field)[i], dtype=np.uint8)
                 for i in order]
    del ds
    gc.collect()

    sp = _splits.resolve(cond, no, cfg)
    idx = sp["idx"]

    # --- 近傍は**背景も含む全タイル**から貼る。外すのは検証とテストだけ。
    # 背景タイルは学習・検証・テストのどこにも属さないので漏れにならない。
    allowed = allowed_except(n_all, idx["val"], idx["test"])
    hold = {int(i) for i in idx["val"]} | {int(i) for i in idx["test"]}
    bad = set(int(a) for a in allowed) & hold
    if bad:
        raise AssertionError(
            f"検証かテストのタイルが近傍に混入しています（{len(bad)} 枚）")
    # 学習の添字が ctx の前半（警戒区域ありの範囲）に収まっているか
    if idx["train"].size and int(idx["train"].max()) >= len(keep):
        raise AssertionError("学習の添字が背景タイルの範囲に入っています")

    ov = dict(getattr(cond, "overlap", None) or {})
    max_shift = int(ov.get("max_shift", 0))
    ov_seed = int(ov.get("seed", cfg.seed))

    def make(which, deterministic):
        return OverlapDataset(
            dem, air, mask, geo, indices=idx[which], allowed=allowed,
            max_shift=(0 if deterministic else max_shift),
            transform=(None if deterministic else transform),
            deterministic=deterministic, seed=ov_seed,
            landuse=lu, third=third)

    dtr = make("train", False)
    dva = make("val", True)
    dte = make("test", True)

    tr = idx["train"]
    fields = {}
    for nm, arr in (("terrain", dem), ("airphoto", air),
                    (t3.field if t3 is not None else "third", third),
                    ("LandUse", lu), ("Mask", mask)):
        if arr is None:
            continue
        fields[nm] = {"shape": tuple(np.shape(arr[0])),
                      "dtype": str(np.asarray(arr[0]).dtype)}
        fields[nm]["mean"], fields[nm]["pos"] = _stats(arr, tr)

    cov, mean = (dtr.coverage() if max_shift > 0 else (0.0, 0.0))
    ld = Loaders(
        train=DataLoader(dtr, batch_size=bs, shuffle=True,
                         num_workers=num_workers),
        val=DataLoader(dva, batch_size=bs, shuffle=False,
                       num_workers=num_workers),
        test=DataLoader(dte, batch_size=bs, shuffle=False,
                        num_workers=num_workers),
        split=sp, fields=fields, region=region, n_all=n_all, n_used=len(no),
        n_bg=len(bgs), n_allowed=len(allowed),
        overlap=({"max_shift": max_shift, "seed": ov_seed,
                  "cov": cov, "mean": mean} if max_shift > 0 else {}),
        datasets={"train": dtr, "val": dva, "test": dte},
    )
    if verbose:
        print(ld.describe())
    return ld


def seed_everything(seed: int):
    """乱数を固定する。**これが無いと `train.seed` は何も変えない。**

    `ゆらぎ幅.md` の `8` のとおり、既存56本のうち12本しか呼んでいなかった。
    cuDNN の非決定性は残るのでビット単位では一致しない
    （`deterministic=True` は学習が遅くなるので入れない）。
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed
