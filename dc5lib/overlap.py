# -*- coding: utf-8 -*-
"""オーバーラップ学習 — 固定格子ではなく、ずらした窓で学習する。

推論では 3x3 の近傍を貼って 64px ずつずらした9窓で見ると箇所F が
+0.047〜+0.060 上がる（docs/実験のまとめ.md 2節）。これは学習時と推論時で
入力の切り方が違っているということなので、学習側でもずらす。

    実質はランダムクロップ拡張にあたる。データ拡張は学習側で唯一
    既存4構造の帯を抜けた施策（面積F +0.0247）なので、方向は一致している。
    狙いは再現率。境界で切れた断片を、学習時に窓の中心で見ることになる。

**分割をまたぐ漏れに注意すること。**
学習タイルの周囲を貼ると、窓は隣のタイルの半分まで入る。その隣が検証や
テストのタイルだと、評価用の画素が学習に入る。この研究は一度テスト集合の
汚染で結論が2つ覆っている（docs/実験のまとめ.md 8節）。
そのため **allowed に学習集合のタイル番号だけを渡すこと**。
allowed に無い近傍は、評価時と同じく中央タイルの鏡像で埋める。

使い方（ノートブックの cell 13 を差し替える）は
docs/練習_オーバーラップ学習.md を参照。
"""
from __future__ import annotations

import numpy as np

DLON, DLAT = 0.00625, 0.00416667      # タイルの格子間隔（広島・島根で共通）


def allowed_except(n_tiles, *holdouts):
    """貼ってよい近傍の集合を「検証とテスト以外すべて」として作る。

    この規則にしておくと取り違えにくい。

      - 背景タイル（警戒区域を含まない）は bg_ratio=0 の条件では
        学習にもテストにも使われない。**文脈として貼っても漏れにならない**ので
        含めてよい。実際これを含めると使える近傍が平均 3.28 → 5.64 枚に増える
      - bg_ratio>0 の条件では背景の一部がテストに入るが、
        その分は holdout に渡されるので自動的に除かれる

        allowed = allowed_except(len(dataset2.No), No_val_idx, No_test_idx)
    """
    out = set(range(int(n_tiles)))
    for h in holdouts:
        out -= set(int(x) for x in h)
    return out


def grid_index(geoinfo):
    """GeoInfo から格子座標を作り、(i, j) -> 並び順の位置 の辞書を返す。

    analysis/run_overlap_eval.py の同名関数と同じ規則。あちらは評価用に
    データセット全体を見るが、こちらは学習側から使うので配列だけを受ける。
    """
    geo = np.asarray([[g[0], g[3]] for g in geoinfo], dtype=np.float64)
    gi = np.round((geo[:, 0] - geo[:, 0].min()) / DLON).astype(int)
    gj = np.round((geo[:, 1] - geo[:, 1].min()) / DLAT).astype(int)
    cell = {(int(a), int(b)): k for k, a, b in zip(range(len(gi)), gi, gj)}
    return gi, gj, cell


class NeighborMosaic:
    """中央タイルの周囲8枚を貼って 3x3 を作る。

    allowed に無い近傍（存在しない／別の分割に属する）は中央の鏡像で埋める。
    貼り方は analysis/run_overlap_eval.py の mosaic() と同じ向きにしてある
    （緯度は上が大きいので行は反転）。
    """

    def __init__(self, geoinfo, allowed=None, tile=128):
        self.gi, self.gj, self.cell = grid_index(geoinfo)
        self.tile = int(tile)
        self.allowed = None if allowed is None else set(int(x) for x in allowed)

    def neighbors_available(self, k):
        """中央 k の8近傍のうち、貼れるものの数（0〜8）。"""
        a, b = int(self.gi[k]), int(self.gj[k])
        n = 0
        for p in (-1, 0, 1):
            for q in (-1, 0, 1):
                if p == 0 and q == 0:
                    continue
                m = self.cell.get((a + p, b + q))
                if m is not None and (self.allowed is None or m in self.allowed):
                    n += 1
        return n

    def build(self, k, planes):
        """3x3 を貼る。planes は {名前: 添字でひける配列}。

        戻り値は {名前: (3T, 3T, ...) の配列}。中央は必ず k 自身。
        """
        T = self.tile
        a, b = int(self.gi[k]), int(self.gj[k])
        out, ok = {}, np.zeros((3, 3), bool)
        for name, src in planes.items():
            c, _ = _to_hwc(src[k])
            big = np.zeros((3 * T, 3 * T, c.shape[2]), dtype=c.dtype)
            out[name] = big

        for p in (-1, 0, 1):
            for q in (-1, 0, 1):
                m = self.cell.get((a + p, b + q))
                if m is None or (self.allowed is not None and m not in self.allowed):
                    continue
                ok[p + 1, q + 1] = True
                r, c0 = (1 - q) * T, (p + 1) * T
                for name, src in planes.items():
                    tile, _ = _to_hwc(src[m])
                    out[name][r:r + T, c0:c0 + T] = tile

        # 貼れなかったところは中央の鏡像で埋める（評価側と同じ規則）
        for p in (-1, 0, 1):
            for q in (-1, 0, 1):
                if ok[p + 1, q + 1]:
                    continue
                r, c0 = (1 - q) * T, (p + 1) * T
                for name in out:
                    cen = out[name][T:2 * T, T:2 * T]
                    out[name][r:r + T, c0:c0 + T] = \
                        cen[::(-1 if q else 1), ::(-1 if p else 1)]
        return out


def _to_hwc(arr):
    """どの形で来ても (H, W, C) に揃える。元の形は kind で覚えておく。

    dataset2 は (1,H,W) のテンソル、dataset3 は (H,W) や (H,W,3) の配列と
    形が混ざることがあるので、貼り合わせる前に必ずここを通す。
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[:, :, None], "hw"
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[0] != a.shape[-1]:
        return np.transpose(a, (1, 2, 0)), "chw"     # CHW -> HWC
    return a, "hwc"


def _from_hwc(a, kind):
    if kind == "hw":
        return a[:, :, 0]
    if kind == "chw":
        return np.transpose(a, (2, 0, 1))
    return a


class OverlapDataset:
    """ずらした窓で切り出す学習用データセット（torch の Dataset として使う）。

    __getitem__ は ((dem, air), mask) を返す。既存の DualInputDataset と
    同じ形なので、学習ループは変えなくてよい。

    引数
      dem/air/mask  データセット全体の配列（分割前）。添字は dataset2 の並び
      geoinfo       同上
      indices       この分割に属するタイルの添字（学習なら学習集合）
      allowed       貼ってよい近傍の添字。**学習では indices と同じものを渡す**
                    （None にすると全タイルを貼るので分割をまたぐ漏れが起きる）
      max_shift     ずらす幅の上限 [画素]。0 で従来どおり（ずらさない）
      transform     既存の JointTransform。ずらしたあとに掛かる
      deterministic True で中央固定。検証・テスト用
    """

    def __init__(self, dem, air, mask, geoinfo, indices, allowed,
                 max_shift=64, tile=128, transform=None, deterministic=False,
                 seed=42):
        self.dem, self.air, self.mask = dem, air, mask
        self.idx = [int(i) for i in indices]
        self.tile = int(tile)
        self.max_shift = int(max_shift)
        self.transform = transform
        self.deterministic = bool(deterministic)
        self.mos = NeighborMosaic(geoinfo, allowed=allowed, tile=tile)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def coverage(self):
        """8近傍がすべて貼れるタイルの割合と、近傍数の平均。"""
        ns = [self.mos.neighbors_available(k) for k in self.idx]
        ns = np.asarray(ns)
        return float((ns == 8).mean()), float(ns.mean())

    def __getitem__(self, i):
        import torch

        k = self.idx[i]
        T = self.tile
        if self.deterministic or self.max_shift <= 0:
            dem = np.asarray(self.dem[k])
            air = np.asarray(self.air[k])
            msk = np.asarray(self.mask[k])
        else:
            # 元が CHW だったかを覚えておき、切り出したあとで戻す
            _, dem_kind = _to_hwc(self.dem[k])
            _, air_kind = _to_hwc(self.air[k])
            _, msk_kind = _to_hwc(self.mask[k])
            big = self.mos.build(k, {"dem": self._plane(self.dem),
                                     "air": self._plane(self.air),
                                     "mask": self._plane(self.mask)})
            dy, dx = self._sample_shift(k)
            r, c = T + dy, T + dx
            dem = _from_hwc(big["dem"][r:r + T, c:c + T], dem_kind)
            air = _from_hwc(big["air"][r:r + T, c:c + T], air_kind)
            msk = _from_hwc(big["mask"][r:r + T, c:c + T], msk_kind)

        dem = torch.as_tensor(np.ascontiguousarray(dem)).float()
        air = torch.as_tensor(np.ascontiguousarray(air)).float() / 255.0
        msk = torch.as_tensor(np.ascontiguousarray(msk)).float()
        if dem.ndim == 2:
            dem = dem.unsqueeze(0)
        if msk.ndim == 2:
            msk = msk.unsqueeze(0)
        if air.ndim == 3 and air.shape[-1] == 3:
            air = air.permute(2, 0, 1)
        if self.transform:
            dem, air, msk = self.transform(dem, air, msk)
        return (dem, air), msk

    def _sample_shift(self, k):
        """使える近傍の方向にだけずらす。

        貼れなかった近傍は中央の鏡像で埋まる＝偽の地形なので、
        そちらに窓を伸ばすと存在しない地形を学ぶ。方向ごとに
        近傍があるかを見て、ある側にだけずらす。
        """
        s = self.max_shift
        a, b = int(self.mos.gi[k]), int(self.mos.gj[k])

        def have(p, q):
            m = self.mos.cell.get((a + p, b + q))
            return m is not None and (self.mos.allowed is None
                                      or m in self.mos.allowed)

        # 行は緯度の反転（dy<0 で上＝北 q=+1 側へ伸びる）
        lo_y = -s if have(0, +1) else 0
        hi_y = +s if have(0, -1) else 0
        lo_x = -s if have(-1, 0) else 0
        hi_x = +s if have(+1, 0) else 0
        dy = int(self.rng.integers(lo_y, hi_y + 1))
        dx = int(self.rng.integers(lo_x, hi_x + 1))
        # 斜めに伸びるなら角の近傍も要る。無ければ片方を諦める
        if dy and dx and not have(1 if dx > 0 else -1, -1 if dy > 0 else 1):
            if self.rng.random() < 0.5:
                dy = 0
            else:
                dx = 0
        return dy, dx

    # --- 内部 ---
    class _Plane:
        """添字でひくと HWC / HW を返す薄い包み。貼り合わせの都合。"""

        def __init__(self, src):
            self.src = src

        def __getitem__(self, k):
            return _to_hwc(self.src[k])[0]

        def __len__(self):
            return len(self.src)

    def _plane(self, src):
        return OverlapDataset._Plane(src)
