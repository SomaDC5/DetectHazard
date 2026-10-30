# -*- coding: utf-8 -*-
"""**条件どうしで学習・検証・テストに使ったタイルが本当に同じか**を確かめる。

なぜ要るか
----------
和集合（アンサンブル）は、構成員が**同じテストタイル**で測られていて
初めて意味を持つ。**構成員Aの学習タイルが構成員Bのテストタイルに
入っていたら、和集合の再現率はデータリークで膨らむ。**

`random_state` が全条件で 42 でも、それだけでは足りない。

    train_test_split(np.arange(N), test_size=0.2, random_state=42)

が返すのは**添字**である。`N` が違えば分かれ方が変わるし、`N` が同じでも
**pkl の中のタイルの並びが違えば、同じ添字が別のタイルを指す。**
だから確かめるのは添字ではなく**タイル番号（`No`）の集合**である。

何をするか
----------
1. pkl を1つずつ読み、`No` だけ取り出してすぐ捨てる（峰を抑える）
2. そのノートブックと同じ手順で分割し、**`No` の集合**に直す
3. 条件どうしで集合を比べる。違っていたら
   **「Aのテストのうち何枚がBの学習に入っているか」= 漏れの枚数**を出す

    python tools/audit_splits.py --region hiroshima
    python tools/audit_splits.py --region both --conditions <条件...>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np                                      # noqa: E402
from sklearn.model_selection import train_test_split    # noqa: E402

from dc5lib import paths                                # noqa: E402
from dc5lib import splits as _splits
from dc5lib.data import load_dataset                    # noqa: E402
from dc5lib.registry import get                         # noqa: E402
from dc5lib.traincfg import train_cfg                   # noqa: E402

REGION_DIR = {"hiroshima": "Hiroshima", "shimane": "Shimane"}
# 既定は和集合の構成員。**ここが揃っていなければ `42` 以降の数字が崩れる。**
# **構成員は unions/union5.yaml に書いてある。** ここには書かない
from dc5lib import union as _uni                             # noqa: E402

DEFAULT = _uni.get("union5").all


def read_no(path: Path):
    """pkl を1つ読んで**学習に使うタイルの `No`** を返す。すぐ捨てる。

    > **2026-10-29 の直し。** 最初の版は pkl の全タイルで分割していたが、
    > **ノートブックは `max(Mask) >= 1` のタイルだけを `dataset2` に集めてから
    > 分割している**（警戒区域を含まないタイルは `dataset3` に回る）。
    > 全タイルで計算すると、条件どうしの一致／不一致の判定は変わらないが
    > **枚数が実際の分割と違う**。22,888/7,630/7,630 と報告したのは誤りだった。
    """
    gb = path.stat().st_size / 1024 ** 3
    print(f"  読み込み {path.name} ({gb:.1f} GB) … ", end="", flush=True)
    t = time.time()
    ds = load_dataset(path)
    n_all = len(ds.No)
    keep = [i for i in range(n_all)
            if np.max(np.asarray(ds.Mask[i])) >= 1]
    no = [int(ds.No[i]) for i in keep]
    del ds
    gc.collect()
    print(f"{time.time() - t:.0f}秒 / 全 {n_all:,} 枚 → "
          f"警戒区域あり {len(no):,} 枚", flush=True)
    return no, n_all


def split_of(no, cond, cfg):
    """`dc5lib.splits.resolve` で分割し、**タイル番号の集合**で返す。

    **指紋と分割の計算は `dc5lib/splits.py` に1か所だけ置いてある。**
    ここで別に実装すると、いずれ食い違って
    「同じ分割か」の判定が信用できなくなる。
    """
    sp = _splits.resolve(cond, no, cfg)
    return ({k: frozenset(sp["tiles"][e]) for k, e in
             (("学習", "train"), ("検証", "val"), ("テスト", "test"))}, sp)


def digest(no):
    return _splits.tiles_fingerprint(no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--conditions", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    regions = ["hiroshima", "shimane"] if a.region == "both" else [a.region]

    for region in regions:
        print("=" * 100)
        print(f"{region} — {len(a.conditions)} 条件")
        print("=" * 100)

        # 同じ pkl を指す条件は1回だけ読む
        by_pkl = {}
        for n in a.conditions:
            cond = get(n)
            p = cond.pkl_for(REGION_DIR[region])
            by_pkl.setdefault(p, []).append(n)

        nos, splits, cfgs, specs = {}, {}, {}, {}
        for p, names in by_pkl.items():
            if not p.exists():
                print(f"  {p.name} がありません（{', '.join(names)}）")
                continue
            no, n_mask = read_no(p)
            for n in names:
                nos[n] = no
                cfgs[n] = train_cfg(n)
                splits[n], specs[n] = split_of(no, get(n), cfgs[n])
            print(f"      全 {n_mask:,} 枚 / 使うタイルの指紋 {digest(no)}")

        print(f"\n{'条件':<40}{'pkl':<30}{'タイル':>8}"
              f"{'指紋':>14}{'学習':>8}{'検証':>8}{'テスト':>8}")
        for n in a.conditions:
            if n not in nos:
                continue
            s = splits[n]
            print(f"{n[:39]:<40}{get(n).pkl_for(REGION_DIR[region]).name[:29]:<30}"
                  f"{len(nos[n]):>8,}{digest(nos[n]):>14}"
                  f"{len(s['学習']):>8,}{len(s['検証']):>8,}{len(s['テスト']):>8,}")

        names = [n for n in a.conditions if n in splits]
        print("\n" + "-" * 100)
        print("条件どうしで、分かれたタイルは同じか")
        print("-" * 100)
        bad = 0
        for x, y in itertools.combinations(names, 2):
            sx, sy = splits[x], splits[y]
            same = all(sx[k] == sy[k] for k in ("学習", "検証", "テスト"))
            if same:
                print(f"  一致      {x[:34]:<36} = {y[:34]}")
                continue
            bad += 1
            leak_xy = len(sx["テスト"] & (sy["学習"] | sy["検証"]))
            leak_yx = len(sy["テスト"] & (sx["学習"] | sx["検証"]))
            print(f"  **違う**  {x[:34]:<36} ≠ {y[:34]}")
            for k in ("学習", "検証", "テスト"):
                if sx[k] != sy[k]:
                    print(f"      {k}: {len(sx[k]):,} 対 {len(sy[k]):,} / "
                          f"共通 {len(sx[k] & sy[k]):,}")
            print(f"      **漏れ**: {x[:28]} のテストのうち {leak_xy:,} 枚が "
                  f"{y[:28]} の学習か検証に入っている")
            print(f"      **漏れ**: {y[:28]} のテストのうち {leak_yx:,} 枚が "
                  f"{x[:28]} の学習か検証に入っている")
        print(f"\n  食い違った組 {bad} / "
              f"{len(names) * (len(names) - 1) // 2} 組")
        if bad == 0 and names:
            print("  **全組で学習・検証・テストのタイルが完全に一致した。**"
                  " 和集合にデータリークは無い")


if __name__ == "__main__":
    main()
