# -*- coding: utf-8 -*-
"""仮想の建物を、斜面の麓に置く。

なぜ要るか
----------
`リサーチマップ 2.3 の K3`（`展望_再現率をどう上げるか.md 43`）。
`42` で、正解のかたまりの 8.0%（広島）／18.4%（島根）は
`UNION3` のどの構成員も1画素すら光らせないと分かった。
**この割合がそのまま再現率の天井を決める。** 床を下げるには、
いまの3本に無い情報を入れた構成員が要る。

建物は候補である。急傾斜地崩壊危険区域の指定基準は
**保全対象（人家5戸以上など）の有無を含む**ので、
地形と写真だけでは原理的に解けない部分がある（`失敗の所在.md 197行`）。

ただし**実建物だけを渡すと危ない。** モデルは
「建物が無い → 危険区域でない」という拒否を覚えうる。
その拒否は盲の群を深くする方向に働く。
**仮想建物はその拒否を無効化するために置く。**

どこに置くか
------------
指定基準をそのまま写す。

  1. 傾斜 30度以上の画素を8連結でまとめる（`SLOPE_DEG`）
  2. 標高差 5m 以上の塊だけ残す（`MIN_HEIGHT_M`）。
     **タイルの縁に接する塊は高さを測れないので免除する**
  3. 塊の**低い側**に接する緩斜面を「麓」とする。
     最寄りの塊の平均標高より低い画素が麓である
  4. 麓から 5〜30m の帯（`FOOT_MIN_M`〜`FOOT_MAX_M`）
  5. そのうち**家はあるが5戸に届かない**所だけ（`SPARSE_RADIUS_M` で
     平滑化した実建物の被覆率が `SPARSE_LO` 以上 `SPARSE_HI` 未満）
  6. 20m 格子（`SPACING_M`）に 10m 角（`HOUSE_M`）を並べる。
     人家が5戸ほど並んだ形になる

**5 で下限を置くのが要点である。** 「建物が無い所に一律に置く」設計は
**使うとモデルを悪くする。** 実データで測った（`43.2`）。

    正解の平均 / 背景の平均            広島      島根
      傾斜（いまの主入力）             1.065     1.001
      実建物                           2.967     5.346
      仮想建物（建物が無い所に一律）   0.340     0.254  ← 逆向き
      仮想建物（5戸に足りない麓だけ）  3.549     4.146

広島はタイルの画素の 39.2% が 30度以上で、傾斜の中央値そのものが 26.6度、
高さ5m の条件は塊の 6.9% しか落とさない。
**つまり無人の山地がまるごと「家が建ちうる麓」になる。** そこには正解が無いので、
一律に置くと「仮想建物 → 背景」を学ばせることになる。

`1C2-4`（FSS2026）が仮想の建物データを使っているが、あちらは
インスタンスセグメンテーションで、しかも表2の 98.15% は仮想建物の
効きではない（`41.3`）。**ここは発想だけを借りている。**

値の持ち方
----------
実建物（`dc5lib/building.py`）と同じ **被覆率 × 255 の uint8** にする。
ただし作り方が違う。

    実建物   PIL の多角形塗りを 16倍スーパーサンプル。輪郭を外側に丸めるので
             面積が **+2.3% 過大**（`building.py` の `BuildingNote`）
    仮想建物 軸に平行な矩形なので**解析的に重なり面積を出す**。偏りは無い

**同じ面に混ぜない理由がこれでもある。** 別チャネルで渡す。

タイル単位であることの限界
--------------------------
タイルは 128px ≈ 576 × 460m で、斜面はタイルの縁で切れる。
切れた塊は高さを測れないので免除しているが、
**麓の判定もタイルの中だけで見ている。** 縁をまたぐ斜面では
麓が隣のタイルにあって、こちらには置かれないことがある。
実建物は経緯度の大域索引で焼いているのでこの問題が無い。
**ここは仮想建物だけが背負う誤差である。**
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

TILE = 128
# 1画素の大きさ。(行＝南北, 列＝東西)。`run_miss_coverage.py` の PX_M と同じ
PX_M = (3.60, 4.50)

SLOPE_DEG = 30.0        # 急傾斜地の基準
MIN_HEIGHT_M = 5.0      # 同じく基準の高さ
MIN_STEEP_PX = 10       # これ未満の塊は数えない（`instance_eval` の MIN_SIZE と同じ）
FOOT_MIN_M = 5.0        # 麓の帯の内側
FOOT_MAX_M = 30.0       # 同じく外側
SPARSE_RADIUS_M = 50.0  # 「建物が少ない」を見る範囲
SPACING_M = 20.0        # 戸の間隔
HOUSE_M = 10.0          # 1戸の1辺（普通建物の実寸にあわせた）

# 「家はあるが5戸に届かない」の範囲。実データの分布から決めた（`43.2`）。
# 10m角の家5戸 = 500 m2、100m 四方 = 10,000 m2 なので 5戸相当が 0.05。
# 下限 0.01 は「1戸はある」に相当する。**下限を外すと逆向きになる。**
SPARSE_LO = 0.01
SPARSE_HI = 0.05

NOTE = (
    "仮想建物。傾斜30度以上・標高差5m以上の塊の麓 5〜30m のうち、"
    "実建物の50m平滑被覆率が 0.01以上0.05未満（家はあるが5戸に届かない）"
    "の所に、20m格子で10m角を置いたもの。値は被覆率×255。"
    "下限を外して『建物が無い所に一律』にすると、正解/背景の平均の比が"
    "広島 0.340・島根 0.254 と逆向きになるので置かない。"
    "矩形の重なり面積は解析的に出しているので、実建物の +2.3% の偏りは無い。"
    "タイル単位なので、縁をまたぐ斜面の麓は取りこぼす。"
)


def _sparse_size(px_m=PX_M, radius_m=SPARSE_RADIUS_M):
    """平滑化の窓の大きさ（画素）。奇数にする。"""
    return tuple(max(1, int(2 * radius_m / m) | 1) for m in px_m)


def steep_blobs(slope, elev, px_m=PX_M, slope_deg=SLOPE_DEG,
                min_height_m=MIN_HEIGHT_M, min_px=MIN_STEEP_PX):
    """基準を満たす急斜面の塊。(ラベル, 塊数, 平均標高) を返す。

    **タイルの縁に接する塊は高さの条件を免除する。** 切れているので
    標高差が実際より小さく出て、本当は基準を満たす斜面を落としてしまう。
    """
    lab, n = ndimage.label(slope >= slope_deg, structure=np.ones((3, 3), int))
    if n == 0:
        return lab, 0, np.zeros(0, np.float64)
    idx = np.arange(1, n + 1)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    hi = ndimage.maximum(elev, lab, idx)
    lo = ndimage.minimum(elev, lab, idx)
    # 縁に接するか。塊のラベルが上下左右の端に出ているかで見る
    edge = np.zeros(n + 1, bool)
    for e in (lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]):
        edge[np.unique(e)] = True
    keep = (size >= min_px) & ((np.asarray(hi) - np.asarray(lo)
                                >= min_height_m) | edge[1:])
    if not keep.any():
        return np.zeros_like(lab), 0, np.zeros(0, np.float64)
    # 残した塊だけに付番し直す（1..m）
    remap = np.zeros(n + 1, np.int32)
    remap[idx[keep]] = np.arange(1, int(keep.sum()) + 1)
    lab2 = remap[lab]
    m = int(keep.sum())
    mean = ndimage.mean(elev, lab2, np.arange(1, m + 1))
    return lab2, m, np.atleast_1d(np.asarray(mean, np.float64))


def foot_band(slope, elev, px_m=PX_M, **kw):
    """麓の帯（bool）と、急斜面のラベルを返す。

    帯の画素は「最寄りの急斜面の塊の平均標高より低い」ものだけである。
    これで斜面の**頂ではなく麓**を選ぶ。
    """
    lab, m, mean = steep_blobs(slope, elev, px_m=px_m, **kw)
    if m == 0:
        return np.zeros(lab.shape, bool), lab
    S = lab > 0
    dist, ind = ndimage.distance_transform_edt(
        ~S, sampling=px_m, return_distances=True, return_indices=True)
    near = lab[tuple(ind)]                      # 最寄りの急斜面画素の塊番号
    lower = elev < mean[np.clip(near - 1, 0, m - 1)]
    band = ((~S) & (slope < SLOPE_DEG) & (dist >= FOOT_MIN_M)
            & (dist <= FOOT_MAX_M) & lower)
    return band, lab


def lattice_centers(band, px_m=PX_M, spacing_m=SPACING_M):
    """帯の中から、20m 格子に1点ずつ中心を選ぶ。

    格子の升ごとに、升の中心に最も近い帯の画素を1つ採る。
    **帯がどんな形でも、升に帯があれば必ず1点取れる。**
    斜めに細い帯でも落ちない。
    """
    ii, jj = np.nonzero(band)
    if not len(ii):
        return ii, jj
    y = (ii + 0.5) * px_m[0]
    x = (jj + 0.5) * px_m[1]
    ly = np.floor(y / spacing_m).astype(np.int64)
    lx = np.floor(x / spacing_m).astype(np.int64)
    cy = (ly + 0.5) * spacing_m
    cx = (lx + 0.5) * spacing_m
    d = (y - cy) ** 2 + (x - cx) ** 2
    cell = ly * 100000 + lx
    order = np.lexsort((d, cell))                # 升ごとに距離の小さい順
    c = cell[order]
    first = np.ones(len(order), bool)
    first[1:] = c[1:] != c[:-1]
    sel = order[first]
    return ii[sel], jj[sel]


def draw_houses(ii, jj, shape, px_m=PX_M, house_m=HOUSE_M):
    """中心の並びから、被覆率 float32 [0,1] を作る。

    軸に平行な矩形なので**解析的に重なり面積を出す**。
    スーパーサンプルの丸めが無いので、実建物の +2.3% の偏りは乗らない。
    重なりは和ではなく **max** で入れる（20m 間隔なので普通は重ならないが、
    念のため 1 を超えないようにする）。
    """
    out = np.zeros(shape, np.float32)
    if not len(ii):
        return out
    hy = house_m / 2.0 / px_m[0]                 # 画素単位の半径
    hx = house_m / 2.0 / px_m[1]
    ry, rx = int(np.ceil(hy)) + 1, int(np.ceil(hx)) + 1
    for i0, j0 in zip(ii.tolist(), jj.tolist()):
        y0, y1 = i0 + 0.5 - hy, i0 + 0.5 + hy
        x0, x1 = j0 + 0.5 - hx, j0 + 0.5 + hx
        rs, re = max(0, i0 - ry), min(shape[0], i0 + ry + 1)
        cs, ce = max(0, j0 - rx), min(shape[1], j0 + rx + 1)
        r = np.arange(rs, re)[:, None]
        c = np.arange(cs, ce)[None, :]
        fy = np.clip(np.minimum(y1, r + 1) - np.maximum(y0, r), 0, None)
        fx = np.clip(np.minimum(x1, c + 1) - np.maximum(x0, c), 0, None)
        np.maximum(out[rs:re, cs:ce], fy * fx, out=out[rs:re, cs:ce])
    return out


def tile_virtual(slope, elev, real, px_m=PX_M, lo=SPARSE_LO, hi=SPARSE_HI,
                 return_parts=False):
    """1タイル分の仮想建物の被覆率 uint8 [tile, tile] を作る。

    slope  float32 [H,W] 傾斜角（度）。SAM 系 pkl の `DEM` そのもの
    elev   float32 [H,W] 標高（m）。`*_dem.pkl` の `DEM`。**同じ格子**
    real   [H,W] か [2,H,W] の実建物。uint8（被覆率×255）でも float でもよい

    `return_parts=True` で (面, 内訳の dict) を返す。確認用。
    """
    slope = np.asarray(slope, np.float32)
    elev = np.asarray(elev, np.float32)
    r = np.asarray(real, np.float32)
    if r.ndim == 3:
        r = r.max(axis=0)
    if r.max() > 1.5:
        r = r / 255.0

    band, lab = foot_band(slope, elev, px_m=px_m)
    local = ndimage.uniform_filter(r, size=_sparse_size(px_m), mode="nearest")
    # 実建物がすでに載っている画素の上には置かない
    free = band & (local >= lo) & (local < hi) & (r <= 0)
    ii, jj = lattice_centers(free, px_m=px_m)
    cov = draw_houses(ii, jj, slope.shape, px_m=px_m)
    plane = np.rint(cov * 255).astype(np.uint8)
    if not return_parts:
        return plane
    return plane, dict(n_steep=int(lab.max()), band_px=int(band.sum()),
                       free_px=int(free.sum()), n_house=int(len(ii)),
                       local_med=float(np.median(local)))


def _selftest():
    """合成の斜面で、麓にだけ置かれることを確かめる。"""
    H = W = 128
    # 北（上）が高く、中央に 30度超の崖、南（下）は平ら
    elev = np.zeros((H, W), np.float32)
    elev[:40] = 100.0
    for k, i in enumerate(range(40, 50)):
        elev[i] = 100.0 - (k + 1) * 10.0        # 10m/画素 ≈ 70度
    elev[50:] = 0.0
    dy = PX_M[0]
    slope = np.degrees(np.arctan(np.abs(np.gradient(elev, dy, axis=0))))
    slope = slope.astype(np.float32)
    real = np.zeros((H, W), np.uint8)

    # (A) 実建物がまったく無い → 置かない（下限 SPARSE_LO があるので）
    _, pa = tile_virtual(slope, elev, real, return_parts=True)
    print("  急斜面の塊 %d / 麓の帯 %d 画素" % (pa["n_steep"], pa["band_px"]))
    print("  (A) 実建物が無い場合の戸 %d（0 であるべき）" % pa["n_house"])
    assert pa["n_house"] == 0, "建物が無い所に置かれている（下限が効いていない）"

    # (B) まばらに実建物がある（5戸に届かない）→ 麓に置く
    sparse = np.zeros((H, W), np.uint8)
    sparse[50::7, ::6] = 255
    plane, p = tile_virtual(slope, elev, sparse, return_parts=True)
    print("  (B) まばらな実建物（局所の中央 %.4f）→ 対象 %d 画素 / 戸 %d"
          % (p["local_med"], p["free_px"], p["n_house"]))
    assert p["n_house"] > 0, "5戸に足りない麓に1戸も置かれていない"
    rows = np.nonzero(plane.any(axis=1))[0]
    print("      置かれた行 %d〜%d（崖は 39〜50行）" % (rows.min(), rows.max()))
    assert rows.min() > 50, "崖より上（＝頂）に置かれている"
    assert plane.max() <= 255

    # 被覆率の確かめ。10m 角 = 100 m2、1画素 = 16.2 m2
    area = plane.astype(np.float64).sum() / 255.0 * (PX_M[0] * PX_M[1])
    want = p["n_house"] * HOUSE_M ** 2
    print("      面積 %.0f m2 / 期待 %.0f m2（ずれ %+.2f%%）"
          % (area, want, 100 * (area / want - 1)))
    assert abs(area / want - 1) < 0.03, "矩形の面積が合っていない"

    # (C) 実建物で埋まっている（すでに5戸以上）→ 置かない
    dense = np.zeros((H, W), np.uint8)
    dense[50:, :] = 255
    _, pc = tile_virtual(slope, elev, dense, return_parts=True)
    print("  (C) 実建物で埋めた場合の戸 %d（0 であるべき）" % pc["n_house"])
    assert pc["n_house"] == 0, "すでに5戸以上あるのに置かれている"

    # (D) 上下を逆にすると、麓も逆（北側）になること
    plane3 = tile_virtual(slope[::-1].copy(), elev[::-1].copy(),
                          sparse[::-1].copy())
    r3 = np.nonzero(plane3.any(axis=1))[0]
    print("  (D) 上下を逆にしたとき %d〜%d（崖は 77〜88行）"
          % (r3.min(), r3.max()))
    assert r3.max() < 77, "逆にしたら頂に置かれている"
    print("  vbuilding の自己診断: 通った")


if __name__ == "__main__":
    _selftest()
