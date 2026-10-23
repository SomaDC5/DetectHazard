# -*- coding: utf-8 -*-
"""空間ブロック分割を設計する（第6章 6.2 / リサーチマップ C）。

なぜ要るか
----------
広島のテスト分割はタイル単位の無作為抽出なので、**学習タイルとテストタイルが
隣り合う。** 実測すると、テスト 2,964 枚のうち **2,838 枚（95.7 %）が
学習タイルと接している。** 隣接タイルは地形も土地利用も似ているので、
報告している広島の精度は「未知の地域への汎化」ではなく
「同一地域内での精度」である。前報でもそう断ってきた。

何を測るか — **距離で層別する**
-------------------------------
当初は「ブロック分割で学習した1本」と「無作為分割で学習した1本」を
比べる案を立てたが、**これは測れない。**
テスト集合を連続したブロックに固定した時点で、残りをどう割っても
テスト集合は孤立する（隣接する学習タイルは 5.5 % 対 6.2 % でほぼ同じ）。
自己相関の正体は「学習側の割り方」ではなく
**「テスト集合が散らばっているか連続しているか」**だった。

そこで**1本のモデルを、学習データからの距離で層別して評価する。**
同じモデル・同じ重みなので、距離以外の条件は完全に揃う。

採用した設計（`--write` が出すもの）
  ・20 km 角のブロック
  ・学習ブロック 75 % / テストブロック 25 %
  ・検証は**学習ブロックの内側から**無作為に 2 割
  ・距離は学習タイルからの歩数（4近傍の幅優先）

この設計を選んだ理由は 2 つある。

| | |
|---|---|
| 学習枚数が揃う | 8,914 枚。既存条件の 8,890 枚とほぼ同じなので、データ量の差が混ざらない |
| 距離別の正例率が揃う | 1-3 歩 7.265 % / 4-15 歩 7.251 % / 16 歩以上 7.311 %（差 0.060 点）。**揃っていないと、成績の差が自己相関のせいかテストが難しいせいか区別できない** |

10 km 角だと距離とともに正例率が 7.24 % から 4.62 % まで単調に下がり、交絡する。

使い方
------
    python analysis/run_block_split.py --report        # ブロックの大きさを振って比べる
    python analysis/run_block_split.py --write         # 採用する分割を書き出す
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd

# Windows のコンソールは cp932 なので、日本語の記号で落ちることがある。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths, registry
from dc5lib.data import load_dataset

GEO_CSV = "tiles_geo__{region}.csv"
BINS = [(1, 3, "1-3"), (4, 15, "4-15"), (16, 10 ** 9, "16+")]


# ------------------------------------------------------------------ 座標
def tile_geo(region="hiroshima", condition="FinalFusion_SAM_APM", force=False,
             gt_only=True):
    """全タイルの経緯度と正例画素数を取り出して小さな CSV に残す。

    pkl は数GBあるので毎回読むのは重い。1回読んだら使い回す。
    **学習に使っているのは正例を含むタイルだけ**（広島 14,816 枚）なので、
    既定ではそれだけを返す。全 38,148 枚で設計すると実際の学習と別の集合を割る。
    """
    out = paths.cache_dir("analysis") / GEO_CSV.format(region=region)
    if not out.exists() or force:
        cond = registry.get(condition)
        pkl = cond.pkl_for(region.capitalize())
        print(f"読み込み: {pkl} …（数GBあるので時間がかかる）", flush=True)
        ds = load_dataset(pkl)
        rows = []
        for i in range(len(ds.No)):
            geo = ds.GeoInfo[i]
            m = np.asarray(ds.Mask[i])
            rows.append(dict(tile_no=ds.No[i],
                             lon=float(geo[0]) if geo is not None else np.nan,
                             lat=float(geo[3]) if geo is not None else np.nan,
                             gt_pos=int((m > 0).sum()), n_pix=int(m.size)))
        df = pd.DataFrame(rows)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"書き出し: {out}（{len(df)} タイル）")
    df = pd.read_csv(out)
    if gt_only:
        df = df[df.gt_pos > 0].reset_index(drop=True)
    return df


# ------------------------------------------------------------------ 格子
def grid(df):
    """タイルを格子の整数座標に直し、1歩が何kmかを返す。

    タイルは等間隔に並ぶので、経緯度の差の最頻値を1歩とみなす。
    **経度1度と緯度1度は距離が違う**（北緯34度では経度1度≒92km、緯度1度≒111km）。
    """
    lon = np.sort(df.lon.unique())
    lat = np.sort(df.lat.unique())
    dlon = np.median(np.diff(lon)) if len(lon) > 1 else 1.0
    dlat = np.median(np.diff(lat)) if len(lat) > 1 else 1.0
    gx = np.rint((df.lon.values - lon[0]) / dlon).astype(int)
    gy = np.rint((df.lat.values - lat[0]) / dlat).astype(int)
    lat0 = float(np.mean(df.lat))
    return gx, gy, dlon * 111.32 * np.cos(np.radians(lat0)), dlat * 110.57


def assign_blocks(df, km=20.0):
    gx, gy, km_x, km_y = grid(df)
    return ((gx * km_x // km).astype(int) * 100000
            + (gy * km_y // km).astype(int)), km_x, km_y


def dist_to_train(df, labels, gx=None, gy=None):
    """学習タイルからの歩数（4近傍の幅優先）。到達できないタイルは -1。

    広島には島があり、本土の学習タイルから格子をたどって到達できない。
    そこは距離が定義できないので層別から外す。
    """
    if gx is None:
        gx, gy, _, _ = grid(df)
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(gx, gy))}
    d = np.full(len(df), -1)
    q = deque()
    for i, l in enumerate(labels):
        if l == "train":
            d[i] = 0
            q.append(i)
    while q:
        i = q.popleft()
        a, b = int(gx[i]), int(gy[i])
        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos.get((a + da, b + db))
            if j is not None and d[j] < 0:
                d[j] = d[i] + 1
                q.append(j)
    return d


def bin_name(d):
    if d < 0:
        return "unreachable"
    for lo, hi, nm in BINS:
        if lo <= d <= hi:
            return nm
    return "train"


# ------------------------------------------------------------------ 割り当て
def split_blocks(df, blocks, tr_ratio=0.75, seed=42, tries=4000, min_bin=250):
    """ブロックを学習側とテスト側に割る。**距離別の正例率が揃うものを選ぶ。**

    ブロックは大きさも正例率もばらばらなので、無作為に割ると
    距離の層ごとに正例率が揃わない。そうなると成績の差が
    「自己相関が消えたから」なのか「その層が難しいから」なのか区別できない。
    無作為な割り当てを何通りも作り、**距離別の正例率の幅が最小**のものを採る。
    """
    g = (pd.DataFrame({"block": blocks, "pos": df.gt_pos.values,
                       "pix": df.n_pix.values})
         .groupby("block").agg(n=("pos", "size"), pos=("pos", "sum"),
                               pix=("pix", "sum")).reset_index())
    gx, gy, _, _ = grid(df)
    rng = np.random.default_rng(seed)
    target = np.array([tr_ratio, 1 - tr_ratio]) * g.n.sum()
    best = None
    for _ in range(tries):
        lab = np.empty(len(g), dtype=object)
        cum = np.zeros(2)
        for k in rng.permutation(len(g)):
            s = int(np.argmax(target - cum))
            lab[k] = ("train", "test")[s]
            cum[s] += g.n.values[k]
        gg = g.assign(split=lab)
        side = (pd.DataFrame({"block": blocks})
                .merge(gg[["block", "split"]], on="block", how="left").split.values)
        d = dist_to_train(df, side, gx, gy)
        te = np.where(side == "test")[0]
        rates, sizes = [], []
        for lo, hi, _nm in BINS:
            m = (d[te] >= lo) & (d[te] <= hi)
            sub = df.iloc[te[m]]
            sizes.append(int(m.sum()))
            rates.append(sub.gt_pos.sum() / sub.n_pix.sum() * 100 if m.sum() else np.nan)
        if min(sizes) < min_bin:
            continue
        spread = float(np.nanmax(rates) - np.nanmin(rates))
        if best is None or spread < best[0]:
            best = (spread, side, d, rates, sizes)
    if best is None:
        raise RuntimeError("条件を満たす割り当てが見つからない")
    return best


# ------------------------------------------------------------------ 出力
def report(df, kms, tr_ratio=0.75):
    gx, gy, km_x, km_y = grid(df)
    print(f"タイル {len(df)} 枚 / 1 歩 = {km_x:.2f} km（東西）"
          f"× {km_y:.2f} km（南北）")
    print(f"全体の正例率 {df.gt_pos.sum() / df.n_pix.sum() * 100:.3f} %")

    from dc5lib.data import holdout_indices
    ti = set(holdout_indices(len(df)).tolist())
    lab = np.array(["test" if i in ti else "train" for i in range(len(df))])
    pos = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(gx, gy))}
    te = np.where(lab == "test")[0]
    near = sum(1 for i in te
               if any(pos.get((int(gx[i]) + da, int(gy[i]) + db)) is not None
                      and lab[pos[(int(gx[i]) + da, int(gy[i]) + db)]] == "train"
                      for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))))
    print(f"\n【いまのランダム分割】テスト {len(te)} 枚のうち "
          f"**学習タイルと隣接 {near} 枚（{near / len(te) * 100:.1f} %）**\n")

    print(f"{'ブロック':<10}{'数':>5}{'学習':>7}{'テスト':>8}"
          f"{'1-3歩':>9}{'4-15歩':>9}{'16歩+':>9}{'正例率の幅':>11}")
    for km in kms:
        blocks, _, _ = assign_blocks(df, km)
        try:
            spread, side, d, rates, sizes = split_blocks(df, blocks, tr_ratio)
        except RuntimeError:
            print(f"{km:>5.0f} km{'':<4}{len(set(blocks)):>5}  （層が小さすぎる）")
            continue
        print(f"{km:>5.0f} km{'':<4}{len(set(blocks)):>5}"
              f"{int((side == 'train').sum()):>7}{int((side == 'test').sum()):>8}"
              + "".join(f"{r:>9.3f}" for r in rates)
              + f"{spread:>11.3f}")
    print("\n正例率の幅が小さいほどよい（揃っていないと、成績の差が"
          "自己相関のせいかその層が難しいせいか区別できない）。")


def write(df, km=20.0, tr_ratio=0.75, val_ratio=0.2, seed=42, out=None):
    """採用する分割を書き出す。検証は**学習ブロックの内側から**取る。

    検証をテスト側のブロックから取ると、距離で層別する対象が減るうえ、
    モデル選択に使ったタイルが評価に混ざる。
    学習ブロックの内側から取れば、テストブロック全体を評価に回せる。
    """
    blocks, _, _ = assign_blocks(df, km)
    spread, side, d, rates, sizes = split_blocks(df, blocks, tr_ratio, seed)

    rng = np.random.default_rng(seed)
    split = np.array(side, dtype=object)
    tr = np.where(side == "train")[0]
    rng.shuffle(tr)
    split[tr[:int(round(len(tr) * val_ratio))]] = "val"

    m = pd.DataFrame({
        "tile_no": df.tile_no.values,
        "block": blocks,
        "split": split,
        "dist": d,
        "bin": [bin_name(x) if s == "test" else "" for x, s in zip(d, side)],
    })
    out = Path(out) if out else (paths.REPO_ROOT / "splits"
                                 / f"hiroshima_block_{int(km)}km.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False, encoding="utf-8")

    c = m.split.value_counts()
    print(f"書き出し: {out}")
    print(f"  学習 {c.get('train', 0)} / 検証 {c.get('val', 0)} "
          f"/ テスト {c.get('test', 0)}")
    print("  （既存条件は 学習 8890 / 検証 2963 / テスト 2963）")
    print()
    print(f"{'距離の層':<12}{'枚数':>7}{'正例率':>9}")
    for (lo, hi, nm), r, s in zip(BINS, rates, sizes):
        print(f"{nm:<12}{s:>7}{r:>9.3f}%")
    un = int((m[m.split == "test"].dist < 0).sum())
    print(f"{'到達不能':<12}{un:>7}  （島。層別から外す）")
    print(f"\n正例率の幅 {spread:.3f} 点")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--km", nargs="*", type=float, default=[10, 15, 20, 25, 30])
    ap.add_argument("--train-ratio", type=float, default=0.75)
    ap.add_argument("--force", action="store_true", help="座標を取り直す")
    args = ap.parse_args()

    df = tile_geo(args.region, force=args.force)
    print(f"正例を含むタイルだけを使う: {len(df)} 枚")
    if args.report or not args.write:
        report(df, args.km, args.train_ratio)
    if args.write:
        km = args.km[0] if len(args.km) == 1 else 20.0
        write(df, km, args.train_ratio)


if __name__ == "__main__":
    main()
