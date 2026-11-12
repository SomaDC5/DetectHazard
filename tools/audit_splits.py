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

from dc5lib import splits as _splits                    # noqa: E402
from dc5lib.data import load_dataset                    # noqa: E402
from dc5lib.registry import get                         # noqa: E402
from dc5lib.traincfg import train_cfg                   # noqa: E402

REGION_DIR = {"hiroshima": "Hiroshima", "shimane": "Shimane"}
# 既定は和集合の構成員。**ここが揃っていなければ `42` 以降の数字が崩れる。**
# **構成員は unions/*.yaml に書いてある。** ここには書かない
from dc5lib import union as _uni                             # noqa: E402


def _all_members():
    """**`unions/*.yaml` の全定義の構成員**（重複なし・現れた順）。

    以前は `union5` だけを既定にしていたので、`union6` で足した構成員と
    `greedy4` の1本が**既定の検査から外れていた**（2026-11-11 のレビュー。
    しかも構成員の多くは素性の無い古い重みで、`union.verify` は
    「分からない」としか言えない ── 確かめられるのはこの道具だけである）。
    yaml の名前をここに書かないので、定義を足しても書き直さなくてよい。
    """
    out = []
    for name in _uni.names():
        for m in _uni.get(name).all:
            if m not in out:
                out.append(m)
    return out


DEFAULT = _all_members()


# 期待値（`EXPECT`）と「比べてよい分割か」（`is_standard`）の原本は
# **`dc5lib/splits.py`** である（2026-11-10 の再レビュー 3）。
# 「特別な分割の宣言」の判定は **`dc5lib/registry.py`** に1つだけ置いた
# （同 4 ── ここと registry で判定が食い違い、宣言できない条件に
# **消せない警告**が毎回出ていた。消せない警告は読み飛ばされる）。
from dc5lib.registry import split_declaration                # noqa: E402
from dc5lib.splits import (compare_expected, expected,       # noqa: E402
                           is_standard)


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


def names_in(conditions, nos):
    return [n for n in conditions if n in nos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--conditions", nargs="*", default=DEFAULT)
    a = ap.parse_args()
    regions = ["hiroshima", "shimane"] if a.region == "both" else [a.region]

    bad_record = False
    bad_pairs = 0           # 分割が食い違った組（全地域の合計）
    not_checked = []        # pkl が無くて確かめられなかった条件
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
                not_checked += [f"{region}/{n}" for n in names]
                continue
            no, n_mask = read_no(p)
            for n in names:
                nos[n] = no
                cfgs[n] = train_cfg(n)
                splits[n], specs[n] = split_of(no, get(n), cfgs[n])
            print(f"      全 {n_mask:,} 枚 / 使うタイルの指紋 {digest(no)}")

        print(f"\n{'条件':<34}{'pkl':<26}{'タイル':>8}"
              f"{'並びの指紋':>14}{'分割の指紋':>14}"
              f"{'学習':>8}{'検証':>8}{'テスト':>8}")
        for n in a.conditions:
            if n not in nos:
                continue
            s, sp = splits[n], specs[n]
            print(f"{n[:33]:<34}{get(n).pkl_for(REGION_DIR[region]).name[:25]:<26}"
                  f"{len(nos[n]):>8,}{digest(nos[n]):>14}"
                  f"{sp['fingerprint']:>14}"
                  f"{len(s['学習']):>8,}{len(s['検証']):>8,}{len(s['テスト']):>8,}")

        # ---- 記録（実装書 `4`）との照合。**2つの指紋を両方見る。**
        # それまでは並びの指紋しか画面に出しておらず、
        # **分割の指紋は計算して捨てていた**（2026-11-08 のレビューの続き）。
        want, miss = expected(region), 0
        if want and names_in(a.conditions, nos):
            print("\n" + "-" * 100)
            print(f"記録との照合（実装書 4 の {region}）")
            print("-" * 100)
            # `n_all` は `resolve` の返りに無いので照合には使わない
            print(f"  記録: 並び {want['tiles_fingerprint']}"
                  f" / 分割 {want['fingerprint']}"
                  f" / 警戒区域あり {want['n_used']:,} 枚")
            for n in a.conditions:
                if n not in nos:
                    continue
                sp = specs[n]
                kind, msg = split_declaration(get(n))
                if kind in ("undeclared", "misnamed"):
                    print(f"  **注意**  {n[:34]:<36} {msg}")
                elif kind == "cannot_declare":
                    print(f"  既知      {n[:34]:<36} {msg}")
                if not is_standard(sp):
                    d = (f"kind={sp.get('kind')}"
                         if sp.get("kind") != "random"
                         else f"test_size={sp.get('test_size')}"
                              f" val_size={sp.get('val_size')}"
                              f" random_state={sp.get('random_state')}")
                    print(f"  飛ばす    {n[:34]:<36} 別の分割指定（{d}）")
                    continue
                # **照合は `splits.compare_expected` が1か所で持つ**
                # （5回目のレビュー 1）。`n_all` のように `resolve` の
                # 返りに無い鍵は見ない。
                bad = compare_expected(region, len(nos[n]), sp)
                if not bad:
                    print(f"  一致      {n[:34]:<36} 並び・分割・枚数すべて")
                else:
                    miss, bad_record = miss + 1, True
                    print(f"  **記録と違う** {n[:34]}")
                    for k, (g, e) in bad.items():
                        print(f"      {k}: {g} （記録は {e}）")
            if miss:
                print(f"\n  **記録と違う条件 {miss} 本。** pkl か分割の指定か"
                      "前処理が変わっている。**どれも他の条件と比べられなくする。**")
                print("  意図した変更なら **`dc5lib/splits.py` の EXPECT**"
                      "（期待値の原本）と実装書 `4` を直すこと。")
            else:
                print("\n  **記録どおり。** 2つの指紋と枚数が実装書 `4` と一致した")

        names = [n for n in a.conditions if n in splits]
        print("\n" + "-" * 100)
        print("条件どうしで、分かれたタイルは同じか")
        print("-" * 100)
        bad = 0
        for x, y in itertools.combinations(names, 2):
            sx, sy = splits[x], splits[y]
            same = all(sx[k] == sy[k] for k in ("学習", "検証", "テスト"))
            # **集合の比較は指紋より強い。** だから指紋は「集合と食い違って
            # いないか」の検査に使う ── 集合が同じなのに指紋が違えば
            # `splits.fingerprint()` 自体が壊れている。
            fx, fy = specs[x]["fingerprint"], specs[y]["fingerprint"]
            if same != (fx == fy):
                print(f"  **指紋と集合が食い違う** {x[:30]} / {y[:30]}"
                      f" ── 集合 {'同じ' if same else '違う'} / "
                      f"指紋 {fx} 対 {fy}")
                print("      **`dc5lib/splits.py` の fingerprint() を疑うこと。**")
                bad += 1
                continue
            if same:
                print(f"  一致      {x[:34]:<36} = {y[:34]}（指紋 {fx}）")
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
        bad_pairs += bad
        if bad == 0 and names:
            print("  **全組で学習・検証・テストのタイルが完全に一致した。**"
                  " 和集合にデータリークは無い")

    # **終了コードで知らせる**（2026-11-11 のレビュー）。食い違いがあっても
    # 0 で終わっていたので、出力を読まないかぎり気づけなかった。
    # 「確かめられなかった」も成功にしない ── 分からないことを分かったと言わない。
    if bad_record:
        sys.exit("**記録（実装書 4）と違う分割がある。** 上を読むこと")
    if bad_pairs:
        sys.exit(f"**分割が食い違った組が {bad_pairs} 組ある。混ぜないこと。**"
                 " 上の「漏れ」の枚数を読むこと")
    if not_checked:
        sys.exit("**pkl が無くて確かめられなかった条件がある（未検査）**: "
                 + ", ".join(not_checked))


if __name__ == "__main__":
    main()
