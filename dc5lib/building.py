# -*- coding: utf-8 -*-
"""基盤地図情報の建築物（BldA）を、既存タイルの格子に焼き込む。

なぜ要るか
----------
`リサーチマップ 2.3 の D`。正解データ（急傾斜地崩壊危険区域）の見逃しと、
「航空写真が古く宅地開発が進んだところを見落としているのでは」という
教授の指摘を確かめるのに、**建物の位置**が要る。

タイル番号の対応
----------------
**ここを間違えると全部無駄になる。**
既存の pkl は1タイルごとに `GeoInfo` を持っており、中身は GDAL の
アフィン変換 `(lon0, dlon, 0, lat0, 0, dlat)` である（`dlat` は負）。

    画素 (i, j) の中心 → lat = lat0 + (i + 0.5) * dlat
                          lon = lon0 + (j + 0.5) * dlon

`dc5lib.landuse.tile_channels_grid` と同じ読み方である。
**タイルの番号も並びも元の pkl の `No` をそのまま使う**ので、
SAM・DEM・マスクと画素単位でずれない。

BldA だけを使う
---------------
GML には BldA（面）と BldL（線）がある。広島県で数えると

    BldA 1,749,730 棟 / BldL 1,873,716 本

で種類の比率がほぼ同一（普通建物 87.3%）なので、**BldL は同じ建物の
外周線**である。両方使うと二重に数えることになるので BldA だけを使う。

建築年は取れない
----------------
`lfSpanFr` と `devDate` は広島県の全建物で**100% 一致**し、年は 2025 と
2026 しかない。これは**データの供給日**であって建築年ではない
（基盤地図情報 DEM の `devDate` と同じ事情）。
**建物がいつ建ったかはこのデータからは分からない。**

代わりに `orgGILvl`（元の地図の縮尺）が使える。1/2500 が 88.0%、
1/25000 が 10.7% で、**1/25000 由来は位置精度が粗い**。
タイルごとにその割合を残しておく。

被覆率で持つ
------------
タイルの画素は 3.6〜4.5 m で、普通建物は 10 m 程度しかない。
二値にすると小さい家が落ちるので、**16倍スーパーサンプル（1画素を256点で
判定）して面積の割合 × 255** を uint8 で持つ。

倍率を 16 にした理由は2つある。
  ・値の段階が 4倍だと 17 段しか無く、建物の少ない地域で潰れる。16倍なら 257 段
  ・PIL の多角形塗りは輪郭を外側に丸めるので、面積が周長に比例して多く出る。
    中心サンプリングと比べた相対の偏りは 4倍 +10.4% / 8倍 +5.1% / **16倍 +2.3%**
広島の 38,148 枚で 16 分。**残る +2.3% は過大評価である**（`BuildingNote` に書く）。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TILE = 128
SUPER = 16                     # スーパーサンプルの倍率。1画素を 16x16 で判定

# PIL は整数座標を**画素の中心**として扱うので、格子の中心を整数に合わせる。
# これを 0 にすると建物が半格子ぶん南東にずれる。
_SHIFT = -0.5

# ラスタの面。**この並びを pkl に持たせる**（LandUseOrder と同じ理由）
KIND_ORDER = ("建物", "無壁舎")
# BldA の <type> から面への割り当て
TYPE_TO_KIND = {
    "普通建物": 0,
    "堅ろう建物": 0,
    "普通無壁舎": 1,
    "堅ろう無壁舎": 1,
}
# 棟数はこの4種類で別に数える（情報を落とさないため）
TYPE_ORDER = ("普通建物", "堅ろう建物", "普通無壁舎", "堅ろう無壁舎")
TYPE_INDEX = {t: i for i, t in enumerate(TYPE_ORDER)}

# 索引 npz の版。中身の意味を変えたら上げる
#   1: 最初の版。**座標を float32 で持っていた。使ってはいけない**
#   2: 座標を float64 に。枡の計算を丸めに直し、3x3 の枡を見るようにした
INDEX_VERSION = 2

# **座標は float64 で持つ。** 経度 132.456589 は float32 では約 1.3 m 刻みに
# なり、スーパーサンプルの格子（約 1.1 m）と同じ桁の誤差になる。
# 13.15M 頂点でも float64 で 210 MB なので、精度を落とす理由が無い。
COORD = np.float64

_RE_TYPE = re.compile(r"<type>(.*?)</type>")
_RE_LVL = re.compile(r"<orgGILvl>(\d+)</orgGILvl>")
_RE_POS = re.compile(r"<gml:posList>(.*?)</gml:posList>", re.S)
_RE_INTERIOR = re.compile(r"<gml:interior>")


# ------------------------------------------------------------------ 読み取り
def parse_bld_a(path):
    """BldA の XML 1枚を読んで、外環と穴の多角形を返す。

    posList は **緯度 経度の順**（`srsName="fguuid:jgd2024.bl"`）。
    返す座標は (lon, lat) に直してある。

    戻り値は辞書で、
        ring   : list[(n,2) float64]  外環。1棟に1つ
        kind   : list[int]            KIND_ORDER の添字（ラスタの面）
        tcode  : list[int]            TYPE_ORDER の添字（棟数の内訳）
        lvl    : list[int]            orgGILvl（2500 / 25000 ...）
        hole   : list[(n,2) float64]  穴。棟とは対応づけない（下記）
    穴は全体で 1,928 個（0.1%）しかないので、**どの棟の穴かは持たず、
    最後にまとめて引く。** 建物の中に別の建物がある例は無いので、
    「穴を引く」と「その棟の穴を引く」は同じ結果になる。
    """
    txt = Path(path).read_text(encoding="utf-8")
    ring, kind, tcode, lvl, hole = [], [], [], [], []
    for chunk in txt.split("<BldA ")[1:]:
        m = _RE_TYPE.search(chunk)
        name = m.group(1) if m else ""
        k = TYPE_TO_KIND.get(name)
        if k is None:
            continue
        ml = _RE_LVL.search(chunk)
        # 穴の始まりより前の posList が外環
        cut = _RE_INTERIOR.search(chunk)
        cut = cut.start() if cut else len(chunk)
        first = True
        for pm in _RE_POS.finditer(chunk):
            v = np.array(pm.group(1).split(), dtype=np.float64)
            if len(v) < 6:                      # 3点未満は多角形にならない
                continue
            pts = v.reshape(-1, 2)[:, ::-1]     # (lat, lon) -> (lon, lat)
            if pm.start() < cut and first:
                ring.append(pts)
                kind.append(k)
                tcode.append(TYPE_INDEX[name])
                lvl.append(int(ml.group(1)) if ml else 0)
                first = False
            elif pm.start() >= cut:
                hole.append(pts)
    return dict(ring=ring, kind=kind, tcode=tcode, lvl=lvl, hole=hole)


def _flatten(polys):
    """多角形のリストを (verts, starts, bbox) の平らな配列にする。"""
    if not polys:
        return (np.zeros((0, 2), np.float64), np.zeros(1, np.int64),
                np.zeros((0, 4), np.float64))
    lens = np.array([len(p) for p in polys], np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)])
    verts = np.concatenate(polys, axis=0)
    bbox = np.empty((len(polys), 4), np.float64)
    for i, p in enumerate(polys):
        bbox[i] = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
    return verts, starts, bbox


def parse_folder(root, progress=None):
    """`FG-GML-*/` を全部読んで、平らな配列にまとめる。"""
    files = sorted(Path(root).glob("FG-GML-*/*-BldA-*.xml"))
    if not files:
        raise FileNotFoundError(f"{root} に BldA の XML がありません")
    ring, kind, tcode, lvl, hole = [], [], [], [], []
    for n, f in enumerate(files, 1):
        d = parse_bld_a(f)
        ring += d["ring"]
        kind += d["kind"]
        tcode += d["tcode"]
        lvl += d["lvl"]
        hole += d["hole"]
        if progress:
            progress(n, len(files), len(ring))
    rv, rs, rb = _flatten(ring)
    hv, hs, hb = _flatten(hole)
    return dict(n_file=len(files),
                verts=rv.astype(COORD), starts=rs, bbox=rb.astype(COORD),
                kind=np.array(kind, np.uint8), tcode=np.array(tcode, np.uint8),
                lvl=np.array(lvl, np.int32),
                hole_verts=hv.astype(COORD), hole_starts=hs,
                hole_bbox=hb.astype(COORD))


# ------------------------------------------------------------------ 索引
def build_index(pack, lon0, lat1, dlon_t, dlat_t, nx, ny):
    """多角形をタイル格子の枡に振り分ける。

    1棟が複数のタイルに跨がることがあるので、**跨がる枡すべてに入れる。**
    枡ごとのリストを Python で持つと 1.8M 個のリストになるので、
    `(枡番号, 多角形番号)` の配列を作って枡番号で並べ、
    `searchsorted` で引く形にする。

    lon0 / lat1 は格子の左上（最小経度・最大緯度）、
    dlon_t / dlat_t は1タイルの幅（正の値）。
    """
    out = {}
    for tag, bbox in (("", pack["bbox"]), ("hole_", pack["hole_bbox"])):
        if len(bbox) == 0:
            out[tag + "cell"] = np.zeros(0, np.int64)
            out[tag + "poly"] = np.zeros(0, np.int64)
            continue
        ix0 = np.floor((bbox[:, 0] - lon0) / dlon_t).astype(np.int64)
        ix1 = np.floor((bbox[:, 2] - lon0) / dlon_t).astype(np.int64)
        iy0 = np.floor((lat1 - bbox[:, 3]) / dlat_t).astype(np.int64)
        iy1 = np.floor((lat1 - bbox[:, 1]) / dlat_t).astype(np.int64)
        np.clip(ix0, 0, nx - 1, out=ix0); np.clip(ix1, 0, nx - 1, out=ix1)
        np.clip(iy0, 0, ny - 1, out=iy0); np.clip(iy1, 0, ny - 1, out=iy1)
        cells, polys = [], []
        span = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
        # ほとんどは1枡に収まる。そこは一気に、跨がる分だけ展開する
        one = span == 1
        cells.append(iy0[one] * nx + ix0[one])
        polys.append(np.flatnonzero(one))
        for i in np.flatnonzero(~one):
            yy, xx = np.meshgrid(np.arange(iy0[i], iy1[i] + 1),
                                 np.arange(ix0[i], ix1[i] + 1), indexing="ij")
            cells.append((yy * nx + xx).ravel())
            polys.append(np.full(yy.size, i, np.int64))
        cell = np.concatenate(cells)
        poly = np.concatenate(polys)
        o = np.argsort(cell, kind="stable")
        out[tag + "cell"] = cell[o]
        out[tag + "poly"] = poly[o]
    out.update(lon0=lon0, lat1=lat1, dlon_t=dlon_t, dlat_t=dlat_t, nx=nx, ny=ny)
    return out


def _lookup(cell_sorted, poly_sorted, cell_id):
    lo = np.searchsorted(cell_sorted, cell_id, "left")
    hi = np.searchsorted(cell_sorted, cell_id, "right")
    return poly_sorted[lo:hi]


def _tile_cell(geoinfo, idx):
    """タイルの左上がどの枡かを返す。

    **切り捨てを使ってはいけない。** タイルは格子の線にぴったり乗るので
    `(lat1 - lat0) / dlat_t` は整数になるはずだが、実際には
    185.99999999999966 のような値が出る。`int()` で切り捨てると 185 になり、
    建物は 186 の枡に入っているので**ほとんど見つからなくなる**
    （実測で 554 棟のうち 29 棟しか拾えなかった）。
    タイル側は**四捨五入**し、さらに `cells_around` で隣の枡も見る。
    """
    iy = int(round((idx["lat1"] - float(geoinfo[3])) / idx["dlat_t"]))
    ix = int(round((float(geoinfo[0]) - idx["lon0"]) / idx["dlon_t"]))
    return iy, ix


def _cells_around(iy, ix, idx):
    """自分と周囲8枡の番号。**丸めが1つずれても落とさないため。**

    多角形は bbox が触る枡すべてに入れてあるので、原理的には自分の枡だけで
    足りる。それでも隣を見るのは、枡の計算が1つずれても静かに間違わないため。
    実際の重なりは呼び出し側が bbox で確かめるので、余分に拾っても害は無い。
    """
    nx, ny = idx["nx"], idx["ny"]
    out = []
    for dy in (-1, 0, 1):
        y = iy + dy
        if y < 0 or y >= ny:
            continue
        for dx in (-1, 0, 1):
            x = ix + dx
            if 0 <= x < nx:
                out.append(y * nx + x)
    return out


def _candidates(geoinfo, idx, cell_key="cell", poly_key="poly"):
    iy, ix = _tile_cell(geoinfo, idx)
    ids = [_lookup(idx[cell_key], idx[poly_key], c)
           for c in _cells_around(iy, ix, idx)]
    ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
    return np.unique(ids)


def _tile_bounds(geoinfo, tile=TILE):
    """タイルの経緯度の範囲 (経度min, 経度max, 緯度min, 緯度max)。"""
    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    lo_a, lo_b = lon0, lon0 + dlon * tile
    la_a, la_b = lat0, lat0 + dlat * tile
    return (min(lo_a, lo_b), max(lo_a, lo_b),
            min(la_a, la_b), max(la_a, la_b))


def _overlapping(ids, bbox, b):
    """bbox がタイルに重なる多角形だけに絞る。**まとめて判定する。**"""
    if len(ids) == 0:
        return ids
    bb = bbox[ids]
    keep = ((bb[:, 2] >= b[0]) & (bb[:, 0] <= b[1])
            & (bb[:, 3] >= b[2]) & (bb[:, 1] <= b[3]))
    return ids[keep]


# ------------------------------------------------------------------ 焼き込み
def tile_planes(geoinfo, pack, idx, tile=TILE, sup=SUPER):
    """1タイル分の被覆率 uint8 [len(KIND_ORDER), tile, tile] を作る。

    `geoinfo` は元の pkl の GeoInfo。**そこから座標を引くので、
    タイル番号の対応はこの関数の外で崩れない。**
    """
    from PIL import Image, ImageDraw

    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    n = tile * sup
    b = _tile_bounds(geoinfo, tile)

    def to_px(pts):
        """経緯度 → スーパーサンプル後の画素座標。

        `_SHIFT = -0.5` で**格子の中心を整数に合わせる。**
        PIL は整数座標を画素の中心として扱うので、
        ずらさないと半格子ぶん南東にずれる。
        """
        x = (pts[:, 0] - lon0) / dlon * sup + _SHIFT
        y = (pts[:, 1] - lat0) / dlat * sup + _SHIFT
        return list(zip(x.tolist(), y.tolist()))

    imgs = [Image.new("1", (n, n), 0) for _ in KIND_ORDER]
    drws = [ImageDraw.Draw(im) for im in imgs]
    st, vt, kd = pack["starts"], pack["verts"], pack["kind"]
    ids = _overlapping(_candidates(geoinfo, idx), pack["bbox"], b)
    for i in ids:
        drws[kd[i]].polygon(to_px(vt[st[i]:st[i + 1]]), fill=1)

    # 穴を引く。面を分けずに両方から引く（穴は建物の中にしか無い）
    hids = _overlapping(
        _candidates(geoinfo, idx, "hole_cell", "hole_poly"),
        pack["hole_bbox"], b)
    if len(hids):
        hst, hvt = pack["hole_starts"], pack["hole_verts"]
        hole = Image.new("1", (n, n), 0)
        hd = ImageDraw.Draw(hole)
        for i in hids:
            hd.polygon(to_px(hvt[hst[i]:hst[i + 1]]), fill=1)
        hm = np.asarray(hole, bool)
        imgs = [Image.fromarray(np.asarray(im, bool) & ~hm) for im in imgs]

    out = np.empty((len(KIND_ORDER), tile, tile), np.uint8)
    for c, im in enumerate(imgs):
        # PIL の mode "1" は numpy では bool になる
        a = np.asarray(im, np.uint8).reshape(tile, sup, tile, sup)
        # 16点のうち何点が建物か → 0〜255
        out[c] = (a.sum(axis=(1, 3)) * 255 + (sup * sup) // 2) // (sup * sup)
    return out


def tile_stats(geoinfo, pack, idx, tile=TILE):
    """そのタイルに重心がある棟の数（4種類別）と、1/25000 由来の数。

    **重心で数える**ので、隣のタイルと二重に数えない。
    """
    lo_lo, lo_hi, la_lo, la_hi = _tile_bounds(geoinfo, tile)
    ids = _candidates(geoinfo, idx)
    n_kind = np.zeros(len(TYPE_ORDER), np.int32)
    n_coarse = 0
    if len(ids):
        bb, lvl = pack["bbox"], pack["lvl"]
        cx = (bb[ids, 0] + bb[ids, 2]) * 0.5
        cy = (bb[ids, 1] + bb[ids, 3]) * 0.5
        sel = ids[(cx >= lo_lo) & (cx < lo_hi) & (cy >= la_lo) & (cy < la_hi)]
        if len(sel):
            np.add.at(n_kind, pack["tcode"][sel].astype(np.int64), 1)
            n_coarse = int((lvl[sel] >= 25000).sum())
    return n_kind, n_coarse


def check_index(geoinfo, pack, idx, tile=TILE):
    """索引で拾えた棟数と、全棟を総なめして拾えた棟数を返す。

    **この2つは必ず一致する。** 一致しなければ枡の計算が狂っている。

    最初の版はここが狂っていて、重なる 554 棟のうち 29 棟しか拾えていなかった。
    それでも例外は出ず、被覆率が小さく出るだけだったので、
    **山間部と見分けがつかなかった。** 等式で見るのが確実である。
    """
    b = _tile_bounds(geoinfo, tile)
    bb = pack["bbox"]
    brute = np.flatnonzero((bb[:, 2] >= b[0]) & (bb[:, 0] <= b[1])
                           & (bb[:, 3] >= b[2]) & (bb[:, 1] <= b[3]))
    fast = _overlapping(_candidates(geoinfo, idx), bb, b)
    return len(fast), len(brute)


def check_tile(geoinfo, pack, idx, tile=TILE, sup=SUPER):
    """**索引を使わずに**同じタイルを焼き、索引版と比べる。

    なぜ要るか
    ----------
    最初の版は枡の計算が1行ずれていて、重なる 554 棟のうち 29 棟しか
    拾えていなかった。**それでも例外は出ず、値は「少なめ」に見えるだけだった。**
    被覆率が小さいのは山間部なら当たり前なので、目で見て気づけない。

    ここでは全多角形の bbox を総なめして候補を作り（索引を一切使わない）、
    `matplotlib.path` で点が中に入るかを判定する。
    索引版と一致すれば、枡の計算も焼き込みも正しい。

    戻り値 (索引版の平均被覆率, 総なめ版の平均被覆率, 総なめ版が見た棟数)
    """
    from matplotlib.path import Path as MPath

    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    b = _tile_bounds(geoinfo, tile)
    bb = pack["bbox"]
    ids = np.flatnonzero((bb[:, 2] >= b[0]) & (bb[:, 0] <= b[1])
                         & (bb[:, 3] >= b[2]) & (bb[:, 1] <= b[3]))

    n = tile * sup
    gx = lon0 + (np.arange(n) + 0.5) * dlon / sup
    gy = lat0 + (np.arange(n) + 0.5) * dlat / sup
    LA, LO = np.meshgrid(gy, gx, indexing="ij")
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    inside = np.zeros(len(pts), bool)
    st, vt = pack["starts"], pack["verts"]
    for i in ids:
        p = vt[st[i]:st[i + 1]]
        # その多角形の bbox に入る点だけ判定する
        m = ((pts[:, 0] >= p[:, 0].min()) & (pts[:, 0] <= p[:, 0].max())
             & (pts[:, 1] >= p[:, 1].min()) & (pts[:, 1] <= p[:, 1].max()))
        if not m.any():
            continue
        inside[m] |= MPath(p).contains_points(pts[m])
    brute = inside.reshape(tile, sup, tile, sup).sum(axis=(1, 3)) / (sup * sup)

    pl = tile_planes(geoinfo, pack, idx, tile, sup)
    fast = pl.astype(np.float64).sum(axis=0).mean() / 255
    return fast, float(brute.mean()), len(ids)


def mesh2_of(lon, lat):
    """2次メッシュコード（6桁）。GML が揃っているかの判定に使う。"""
    p = int(np.floor(lat * 1.5))
    u = int(np.floor(lon - 100.0))
    r = int(np.floor((lat - p / 1.5) / (1 / 12.0)))
    c = int(np.floor((lon - (u + 100.0)) / (1 / 8.0)))
    return f"{p:02d}{u:02d}{r}{c}"


def mesh_ok(geoinfo, meshes, tile=TILE):
    """タイルが収録メッシュに収まっているか。

    **タイルの端ではなく、隅の画素の中心で引く。**
    端で引くとメッシュ境界と一致したときに隣のメッシュを見てしまい、
    `dlat` の最後の桁が少し違うだけで答えが変わる
    （実測で同じ地域が 38,148 枚 OK と 38,114 枚 OK に振れた）。
    画素の中心は必ずタイトルの内側なので、曖昧さが無い。

    画素ごとに全部引いて数えたところ、広島の 38,148 枚は
    **収録外の画素が 0 枚**だった。この関数もそれと一致する。
    """
    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    for j in (0.5, tile - 0.5):
        for i in (0.5, tile - 0.5):
            if mesh2_of(lon0 + j * dlon, lat0 + i * dlat) not in meshes:
                return 0
    return 1
