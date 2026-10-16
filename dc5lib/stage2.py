# -*- coding: utf-8 -*-
"""WNet の2段目。かたまりごとに「残す / 捨てる」を決める小さな CNN。

なぜこれを作るのか（展望_再現率をどう上げるか.md 12.5 / 13.4）
--------------------------------------------------------------
1段目（オーバーラップ学習 + オーバーラップ推論）は箇所F 0.7071 で、
**完璧に刈り取れたときの上限は 0.8793。** ロジット最大値のしきい値1本で
0.7384 まで取れたが、これは余っていた分の 22.5% にすぎない。

残りを取るには「ロジットの大小」以外の手がかりが要る。12.2 の実測では

    ロジット最大値  AUC 0.8006   ← しきい値1本が使っているもの
    形（周囲長/面積） 反転で 0.77  ← 別系統。未使用
    傾斜角の平均    AUC 0.5017   ← 手作りの地形統計では分けられない

傾斜角の平均に情報が無いので、**必要なのは「かたまりの周囲をどう切り出すか」**
という文脈である。それを学習させるのがこの2段目。

設計
----
**入力は 64x64 の切り出し 6ch + スカラー5つ。**
切り出しは 3x3 モザイクから取るので、タイル外周のかたまりにも本物の文脈が来る。

    ch1     SAM 傾斜角（/45）
    ch2-4   航空写真 RGB（/255）
    ch5     混ぜたロジット（/40）。**中央タイルの外は0**
    ch6     判定対象のかたまりだけのマスク。どれを見ているかを示す

    スカラー log1p(面積) / 細長さ / ロジット最大 / ロジット平均 / 外周までの距離

**スカラーにロジット最大値を入れているのは意図的。**
しきい値1本が使っている情報を必ず持たせることで、
原理的に 0.7384 を下回らないようにしている。

**1段目は凍結。** この2段目だけを学習する（10.6b の順序）。

限界
----
ch5 のロジットは中央タイルの分だけで、隣のタイルのロジットは持っていない
（9倍の推論が要るため）。地形と写真は本物の近傍が来ている。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CROP = 64                 # 切り出しの一辺（周囲 287m）
WIDE = 256                # 広域の切り出しの一辺（モザイク上。周囲 1,147m）
TILE = 128
SLOPE_SCALE = 45.0        # 傾斜角[度]の正規化
LOGIT_SCALE = 40.0        # ロジットの正規化。的中の中位数が約40
N_SCALAR = 5
N_CH = 6
N_LU = 9                  # 土地利用の区分数（landuse.CHANNEL_ORDER と同じ）
# 広域の枝: 傾斜角1 + 写真3 + 対象のかたまり1 (+ 土地利用9)
N_WIDE_BASE = 5
N_WIDE_CH = N_WIDE_BASE

SCALAR_NAMES = ("log1p面積", "細長さ", "ロジット最大", "ロジット平均", "外周までの距離")



def _onehot_small(cls, n):
    """区分番号 [H,W] を one-hot [n,H,W] に。landuse.onehot の軽量版
    （dc5lib.landuse を import すると循環するのでここに置く）。"""
    out = np.zeros((n,) + cls.shape, np.float32)
    for i in range(n):
        out[i] = (cls == i)
    return out


def set_wide_channels(with_landuse: bool):
    """広域の枝のチャネル数を切り替える。

    土地利用を2段目にだけ入れる構成のために要る。**1段目は触らない。**
    """
    global N_WIDE_CH
    N_WIDE_CH = N_WIDE_BASE + (N_LU if with_landuse else 0)
    return N_WIDE_CH


def _canvas(small, size=3 * TILE, channels=None):
    """中央タイルの配列を 3x3 の大きさに置く。外は0。"""
    if channels is None:
        out = np.zeros((size, size), dtype=np.float32)
        out[TILE:2 * TILE, TILE:2 * TILE] = small
    else:
        out = np.zeros((size, size, channels), dtype=np.float32)
        out[TILE:2 * TILE, TILE:2 * TILE] = small
    return out


def extract_tile(dem_mos, air_mos, logit, pred, labels, sizes, crop=CROP,
                 wide=False, lu_mos=None):
    """1タイル分のかたまりを切り出す。

    dem_mos  3x3 に貼った傾斜角 (384, 384)
    air_mos  3x3 に貼った航空写真 (384, 384, 3)
    logit    中央タイルの混ぜたロジット (128, 128)
    pred     中央タイルの二値マスク (128, 128)
    labels   label_instances の結果 (128, 128) int
    sizes    かたまりごとの画素数

    戻り値 (crops [n, 6, 64, 64] float32, scalars [n, 5] float32,
            blob_ids [n] int)
    """
    from scipy import ndimage

    n = len(sizes)
    if n == 0:
        return (np.zeros((0, N_CH, crop, crop), np.float32),
                np.zeros((0, N_SCALAR), np.float32),
                np.zeros(0, np.int32))

    lg_c = _canvas(logit.astype(np.float32))
    half = crop // 2
    hi = 3 * TILE - crop
    st4 = ndimage.generate_binary_structure(2, 1)

    # タイル内の各画素の外周までの距離（スカラー用）
    rr, cc = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    edge = np.minimum(np.minimum(rr, cc), np.minimum(TILE - 1 - rr, TILE - 1 - cc))

    crops = np.zeros((n, N_CH, crop, crop), np.float32)
    wides = np.zeros((n, N_WIDE_CH, crop, crop), np.float32) if wide else None
    scal = np.zeros((n, N_SCALAR), np.float32)
    ids = np.zeros(n, np.int32)
    whalf = WIDE // 2
    whi = 3 * TILE - WIDE

    for j in range(n):
        sel = labels == (j + 1)
        # 重心をモザイク座標に移して窓を切る
        rs, cs = np.nonzero(sel)
        cy = int(round(rs.mean())) + TILE
        cx = int(round(cs.mean())) + TILE
        y0 = min(max(cy - half, 0), hi)
        x0 = min(max(cx - half, 0), hi)
        ys, xs = slice(y0, y0 + crop), slice(x0, x0 + crop)

        m_c = _canvas(sel.astype(np.float32))
        crops[j, 0] = dem_mos[ys, xs] / SLOPE_SCALE
        crops[j, 1:4] = air_mos[ys, xs].transpose(2, 0, 1) / 255.0
        crops[j, 4] = lg_c[ys, xs] / LOGIT_SCALE
        crops[j, 5] = m_c[ys, xs]

        v = logit[sel]
        per = int(sel.sum() - ndimage.binary_erosion(sel, st4).sum())
        scal[j] = (np.log1p(sizes[j]),
                   per / max(int(sizes[j]), 1),
                   float(v.max()) / LOGIT_SCALE,
                   float(v.mean()) / LOGIT_SCALE,
                   float(edge[sel].min()) / crop)
        ids[j] = j + 1

        if wide:
            # かたまりを中心に 256x256 を切って 64x64 に縮小する。
            # 周囲 1,147m を 17.9m/画素で見ることになる（近傍の 287m に対して4倍）。
            # **14.4 の「残りは周囲 64x64 には無い」への手当て。**
            wy = min(max(cy - whalf, 0), whi)
            wx = min(max(cx - whalf, 0), whi)
            ws, wsx = slice(wy, wy + WIDE), slice(wx, wx + WIDE)
            blk = np.empty((N_WIDE_CH, WIDE, WIDE), np.float32)
            blk[0] = dem_mos[ws, wsx] / SLOPE_SCALE
            blk[1:4] = air_mos[ws, wsx].transpose(2, 0, 1) / 255.0
            blk[4] = _canvas(sel.astype(np.float32))[ws, wsx]
            if lu_mos is not None:
                blk[5:] = _onehot_small(lu_mos[ws, wsx], N_LU)
            # 平均プーリングで 256 -> 64（4x4 ずつまとめる）
            f = WIDE // crop
            wides[j] = blk.reshape(N_WIDE_CH, crop, f, crop, f).mean(axis=(2, 4))
    return (crops, scal, ids) if not wide else (crops, wides, scal, ids)


class Stage2Net(nn.Module):
    """6ch の切り出し + スカラー5つ から「残す確率」のロジットを1つ出す。

    1段目が 43.58M なのに対してこちらは 0.4M 程度。
    学習に使えるかたまりが 9,880個しかないので、小さく保つ。
    """

    def __init__(self, in_ch=N_CH, n_scalar=N_SCALAR, width=32, p_drop=0.3,
                 wide_ch=0):
        super().__init__()
        self.wide_ch = int(wide_ch)

        def blk(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(
            blk(in_ch, width),            # 64 -> 32
            blk(width, width * 2),        # 32 -> 16
            blk(width * 2, width * 4),    # 16 -> 8
            nn.AdaptiveAvgPool2d(1))
        # 広域の枝。近傍 287m とは**別の解像度**なので、同じ画像のチャネルとして
        # 重ねずに枝を分ける（画素の対応が取れないため）
        self.wide = None
        extra = 0
        if self.wide_ch:
            self.wide = nn.Sequential(
                blk(self.wide_ch, width),
                blk(width, width * 2),
                blk(width * 2, width * 4),
                nn.AdaptiveAvgPool2d(1))
            extra = width * 4

        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(width * 4 + extra + n_scalar, 64), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(64, 1))

    def forward(self, x, s, w=None):
        f = [self.features(x).flatten(1)]
        if self.wide is not None:
            if w is None:
                raise ValueError("広域の枝があるのに広域の切り出しが渡されていません")
            f.append(self.wide(w).flatten(1))
        f.append(s)
        return self.head(torch.cat(f, dim=1)).squeeze(1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    x = torch.randn(4, N_CH, CROP, CROP)
    s = torch.randn(4, N_SCALAR)
    for label, wch in (("近傍のみ", 0),
                       ("+ 広域", set_wide_channels(False)),
                       ("+ 広域 + 土地利用", set_wide_channels(True))):
        m = Stage2Net(wide_ch=wch)
        w = torch.randn(4, wch, CROP, CROP) if wch else None
        print("%-18s wide_ch=%-2d パラメータ %9s  出力 %s"
              % (label, wch, f"{n_params(m):,}", tuple(m(x, s, w).shape)))
    set_wide_channels(False)
    print("広域の範囲: %d 画素 -> %d 画素（%.0f m 四方 / %.1f m per px）"
          % (WIDE, CROP, WIDE * 4.48, WIDE * 4.48 / CROP))
    print("スカラー:", SCALAR_NAMES)
