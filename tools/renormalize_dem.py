# -*- coding: utf-8 -*-
"""既存の pkl の DEM を「滑らかな基準面を引く」方式に作り替える。

なぜ必要か（展望_再現率をどう上げるか.md 5.6 / 修士論文 2.1.2）
--------------------------------------------------------------
現行の正規化は **タイルごとの最低標高を引く**（normalize_dem.py）。

    DEM = (raw - Min_H[k]) / scale          ← Min_H がタイルごとに違う

これはタイル境界で値の不連続を作る。実測で、境界をまたぐ画素値の跳びは
タイル内部の変化の **24.5〜27.3倍**あり、ΔMin_H/scale との相関は **0.9996**。
傾斜角（SAM）では 1.3〜2.0倍にとどまる（差分なので定数のずれが消えるため）。

**割る定数（500）は関係ない。** 一様な定数倍は最初の BatchNorm に吸収される
ので性能に影響しない。効いているのは引き算の側である。

やること
--------
引く量をタイルごとの定数ではなく、**地理的に連続な面**にする。

  1. 各タイルの Min_H をタイル格子（整数座標）に並べる
  2. 穴（タイルが無い格子点）を最近傍で埋める
  3. ガウシアンで平滑化する（σ はタイル単位）
  4. 画素の位置に**双線形補間**で戻して、画素ごとの基準面 B を作る
  5. DEM = (raw - B) / scale

**面は連続なので、境界の段差は「隣接画素ぶんの勾配」に落ちる。**
一方でタイル内では面がほぼ定数なので、**斜面そのものは残る。**

> **素朴な「平滑化した DEM を引く」をやってはいけない。**
> 一様な30°斜面を平滑化したものはその斜面自身なので、
> 引くと 0 になって**傾斜の情報が消える。**
> 引く面は「タイル格子のスケールで」滑らかでなければならない。

σ の決め方
----------
σ が大きいほど面はタイル内で平らになる（斜面が残る）が、
地域の標高差を追えなくなって残差の値域が広がる。
`report` で σ を振って

  ・境界の段差（これを小さくしたい）
  ・タイル内での面の変化（斜面をどれだけ食うか。小さくしたい）
  ・残差の値域

の3つを出すので、それを見て決める。

使い方
------
    # σ を振って様子を見る（書き換えない）
    python tools/renormalize_dem.py report Hiroshima hiroshima_dem_apm_normalized.pkl

    # 適用して別名で保存する（元のファイルは触らない）
    python tools/renormalize_dem.py apply Hiroshima \\
        hiroshima_dem_apm_normalized.pkl hiroshima_dem_apm_basenorm.pkl --sigma 2.0

入力 pkl が既に正規化済みのとき（*_normalized.pkl）は、
生の標高を raw = DEM * scale + Min_H で復元してから作り直す。
正規化していない pkl を渡すときは --raw を付ける。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dc5lib.paths import dataset_path                      # noqa: E402

TILE = 128
DLON, DLAT = 0.00625, 0.00416667      # タイルの格子間隔（広島・島根で共通）
SCALE = 500.0                         # 現行の割る定数


def load_pkl(path):
    """既存の読み込み関数を使う。

    pkl はデータセットのクラスごと漬けてあり、そのクラスは
    もう無いモジュール（load_any_dataset など）に定義されている。
    analysis.data.load_pkl がクラス名を問わず読めるようにしてあるので流用する。
    """
    from analysis.data import load_pkl as _load
    return _load(path)


def grid_coords(ds):
    """GeoInfo から整数の格子座標 (gi, gj) を作る。

    gi は経度、gj は緯度から作る。**g[3] が緯度**で、g[1] は画素の幅である
    （ここを間違えると無関係なタイルを隣と見なす。実際に一度間違えた）。
    """
    geo = np.array([[g[0], g[3]] for g in ds.GeoInfo], dtype=np.float64)
    gi = np.round((geo[:, 0] - geo[:, 0].min()) / DLON).astype(int)
    gj = np.round((geo[:, 1] - geo[:, 1].min()) / DLAT).astype(int)
    return gi, gj


def base_grid(ds, gi, gj, sigma, stat="min"):
    """タイルごとの代表標高を格子に並べ、穴を埋めて平滑化する。

    戻り値 S は shape (nj, ni) の float64。S[j, i] が格子点 (i, j) の値。
    """
    from scipy import ndimage

    ni, nj = gi.max() + 1, gj.max() + 1
    vals = (np.asarray(ds.Min_H, np.float64) if stat == "min"
            else np.asarray([np.mean(d) for d in ds.DEM], np.float64))
    S = np.full((nj, ni), np.nan)
    S[gj, gi] = vals

    # 穴を最近傍で埋める。埋めないと平滑化で縁が引っ張られる
    hole = np.isnan(S)
    if hole.any():
        idx = ndimage.distance_transform_edt(
            hole, return_distances=False, return_indices=True)
        S = S[tuple(idx)]
    if sigma > 0:
        S = ndimage.gaussian_filter(S, sigma=sigma, mode="nearest")
    return S


def sample_base(S, i, j, sigma_note=""):
    """格子 S を、タイル (i, j) の 128x128 画素の位置で双線形補間する。

    画素 (row, col) の連続格子座標は

        x = i + (col + 0.5)/TILE - 0.5
        y = j + 0.5 - (row + 0.5)/TILE

    **行は北から南へ増える**（mosaic の r = (1-q)*TILE と同じ向き）。
    列は西から東へ増える。
    """
    nj, ni = S.shape
    u = (np.arange(TILE) + 0.5) / TILE - 0.5          # col 方向のずれ
    x = i + u                                          # (TILE,)
    y = j + 0.5 - (np.arange(TILE) + 0.5) / TILE       # (TILE,) row 方向

    x = np.clip(x, 0, ni - 1)
    y = np.clip(y, 0, nj - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, ni - 1)
    y1 = np.minimum(y0 + 1, nj - 1)
    tx = (x - x0)[None, :]
    ty = (y - y0)[:, None]

    s00 = S[np.ix_(y0, x0)]
    s01 = S[np.ix_(y0, x1)]
    s10 = S[np.ix_(y1, x0)]
    s11 = S[np.ix_(y1, x1)]
    top = s00 * (1 - tx) + s01 * tx
    bot = s10 * (1 - tx) + s11 * tx
    return (top * (1 - ty) + bot * ty).astype(np.float32)


def raw_dem(ds, k, already_normalized, scale):
    """生の標高を取り出す（必要なら逆算する）。"""
    d = np.asarray(ds.DEM[k], np.float32)
    if already_normalized:
        return d * scale + float(ds.Min_H[k])
    return d


def report(ds, scale, already_normalized, sigmas, n_sample=400, seed=0):
    """σ を振って、段差・斜面の食われ方・値域を出す。"""
    gi, gj = grid_coords(ds)
    ni, nj = gi.max() + 1, gj.max() + 1
    cell = {(a, b): k for k, a, b in zip(range(len(gi)), gi, gj)}
    print(f"タイル {len(gi):,} / 格子 {ni} x {nj}"
          f"（格子点のうちタイルがあるのは {100*len(gi)/(ni*nj):.1f}%）")

    minh = np.asarray(ds.Min_H, np.float64)
    maxh = np.asarray(ds.Max_H, np.float64)
    relief = maxh - minh
    print(f"Min_H   {minh.min():.1f} 〜 {minh.max():.1f} m（中位 {np.median(minh):.1f}）")
    print(f"起伏量  中位 {np.median(relief):.1f} m / 99% {np.percentile(relief,99):.1f} m")

    # 現行方式での境界の段差 = 隣接タイルの Min_H の差
    jumps = []
    for k, (a, b) in enumerate(zip(gi, gj)):
        for d in ((1, 0), (0, 1)):
            n = cell.get((a + d[0], b + d[1]))
            if n is not None:
                jumps.append(abs(minh[k] - minh[n]))
    jumps = np.asarray(jumps)
    print(f"\n現行方式（タイルごとに Min_H を引く）")
    print(f"  境界の段差  中位 {np.median(jumps):.2f} m / 平均 {jumps.mean():.2f} m"
          f" / 99% {np.percentile(jumps,99):.1f} m")
    print(f"  正規化後に直すと 中位 {np.median(jumps)/scale:.4f}"
          f"（タイル内の起伏は中位 {np.median(relief)/scale:.4f}）")
    print(f"  段差 / タイル内の起伏 = {np.median(jumps)/np.median(relief):.3f}")
    # **文書（展望 5.6）の「24.5倍」はこちらの指標。**
    # 隣接画素どうしの変化と比べると、段差の異常さがはっきりする
    rng0 = np.random.default_rng(1)
    pk = rng0.choice(len(gi), size=min(200, len(gi)), replace=False)
    g_in = []
    for k in pk:
        raw = raw_dem(ds, k, already_normalized, scale)
        g_in.append(float(np.median(np.abs(np.diff(raw, axis=1)))))
    g_in = float(np.median(g_in))
    print(f"  タイル内の隣接画素間の変化 中位 {g_in:.3f} m")
    print(f"  **境界の段差 / 内部の勾配 = {np.median(jumps)/max(g_in,1e-9):.1f} 倍**"
          f"（展望 5.6 の 24.5〜27.3倍に対応）")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(gi), size=min(n_sample, len(gi)), replace=False)

    print(f"\n新方式（滑らかな基準面を引く）。{len(pick)} タイルを抜き取って測る")
    # 現行方式での「タイルごとの平均残差のばらつき」を基準として出す。
    # これが大きいほど、絶対標高がモデルに漏れていることになる
    cur = np.asarray([float(np.mean(raw_dem(ds, k, already_normalized, scale)
                                    - float(ds.Min_H[k]))) / scale
                      for k in pick])
    print(f"現行方式のタイル平均残差 ばらつき(標準偏差) {cur.std():.4f}"
          f" / 範囲 [{cur.min():.3f}, {cur.max():.3f}]")
    print()
    print(f"{'σ(タイル)':>9s} {'境界の段差':>11s} {'面のタイル内変化':>15s}"
          f" {'/起伏':>7s} {'平均残差のばらつき':>18s} {'残差の値域':>18s}")
    for s in sigmas:
        S = base_grid(ds, gi, gj, s)
        inner, lo, hi, mean_r = [], [], [], []
        edge = []
        for k in pick:
            B = sample_base(S, gi[k], gj[k])
            inner.append(float(B.max() - B.min()))
            raw = raw_dem(ds, k, already_normalized, scale)
            r = (raw - B) / scale
            lo.append(float(r.min())); hi.append(float(r.max()))
            mean_r.append(float(r.mean()))
            # 東隣との境界で、面がどれだけ跳ぶか（= 新方式の段差）
            n = cell.get((gi[k] + 1, gj[k]))
            if n is not None:
                Bn = sample_base(S, gi[n], gj[n])
                edge.append(float(np.abs(B[:, -1] - Bn[:, 0]).max()))
        inner = np.asarray(inner)
        mr = np.asarray(mean_r)
        print(f"{s:>9.1f} {np.median(edge):>10.3f}m {np.median(inner):>14.2f}m"
              f" {np.median(inner)/np.median(relief):>7.3f}"
              f" {mr.std():>18.4f}"
              f"   [{np.median(lo):+.3f}, {np.median(hi):+.3f}]")
    print(f"\n段差は小さいほどよい（現行は中位 {np.median(jumps):.2f} m）。")
    print("面のタイル内変化は小さいほど斜面が残る。/起伏 が 0.1 を超えると食いすぎ。")
    print("平均残差のばらつきは**現行方式と同程度に収まっていること**を確かめる。")
    print("大きくなっていたら、基準面が地域の標高を追えておらず")
    print("絶対標高がモデルに漏れている（それを消すために引いているのに）。")


def apply(ds, out_path, scale, already_normalized, sigma, stat="min"):
    gi, gj = grid_coords(ds)
    S = base_grid(ds, gi, gj, sigma, stat=stat)
    n = len(ds.No)
    lo = hi = None
    for k in range(n):
        B = sample_base(S, gi[k], gj[k])
        raw = raw_dem(ds, k, already_normalized, scale)
        d = ((raw - B) / scale).astype(np.float32)
        ds.DEM[k] = d
        lo = d.min() if lo is None else min(lo, d.min())
        hi = d.max() if hi is None else max(hi, d.max())
        if k % 2000 == 0:
            print(f"  {k}/{n}", end="\r", flush=True)
    # **何をしたかを pkl に残す。** あとから追えないと再現できない
    ds.DemNorm = dict(method="smooth_base", sigma_tiles=float(sigma),
                      scale=float(scale), stat=stat,
                      note="(raw - 平滑化した基準面) / scale。"
                           "基準面はタイル格子の Min_H をガウシアンで平滑化して"
                           "双線形補間したもの。tools/renormalize_dem.py")
    print(f"  書き出し: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(ds, f, protocol=4)
    print(f"  タイル {n:,} / 値域 [{lo:.4f}, {hi:.4f}]")
    print(f"  記録: {ds.DemNorm['method']} sigma={sigma} scale={scale}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("report", help="σ を振って様子を見る（書き換えない）")
    pr.add_argument("region", choices=["Hiroshima", "Shimane"])
    pr.add_argument("pkl")
    pr.add_argument("--sigmas", nargs="*", type=float,
                    default=[0.5, 1.0, 2.0, 4.0, 8.0])
    pr.add_argument("--raw", action="store_true",
                    help="入力が正規化前の pkl のとき")
    pr.add_argument("--scale", type=float, default=SCALE)
    pr.add_argument("--samples", type=int, default=400)

    ap_ = sub.add_parser("apply", help="適用して別名で保存する")
    ap_.add_argument("region", choices=["Hiroshima", "Shimane"])
    ap_.add_argument("pkl")
    ap_.add_argument("out")
    ap_.add_argument("--sigma", type=float, required=True)
    ap_.add_argument("--raw", action="store_true")
    ap_.add_argument("--scale", type=float, default=SCALE)
    ap_.add_argument("--stat", default="min", choices=["min", "mean"])

    a = ap.parse_args()
    src = dataset_path(a.region, a.pkl)
    print(f"読み込み: {src}")
    ds = load_pkl(src)
    print(f"  フィールド: {[k for k in vars(ds) if not k.startswith('_')]}")

    if a.cmd == "report":
        report(ds, a.scale, not a.raw, a.sigmas, n_sample=a.samples)
    else:
        out = dataset_path(a.region, a.out)
        if Path(out).exists():
            sys.exit(f"{out} が既にあります。別の名前にしてください")
        apply(ds, out, a.scale, not a.raw, a.sigma, stat=a.stat)


if __name__ == "__main__":
    main()
