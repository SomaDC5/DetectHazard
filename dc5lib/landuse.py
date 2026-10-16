# -*- coding: utf-8 -*-
"""国土数値情報「土地利用細分メッシュ」(L03-b) をタイルの1チャネルに変換する。

なぜ必要か（失敗の所在.md 197行 / 展望_再現率をどう上げるか.md 14.4）
----------------------------------------------------------------------
TP=0 のタイルの **60% は「正解領域のほうが周囲より緩斜面」**（中央値 −1.410度）。
急傾斜地崩壊危険区域の指定基準は**保全対象（人家）の有無**を含むので、
建物が見えないモデルは**見えない条件で決まるラベル**を当てさせられている。
失敗の所在.md はこれを「解けない。入力の追加が要る」と判定している。

なぜ rasterio が要らないのか
----------------------------
土地利用細分メッシュは**標準地域メッシュの規則的な格子**なので、
ポリゴンのラスタ化ではなく**メッシュコードの計算**で画素に割り当てられる。
緯度経度からコードを求めて辞書を引くだけ。

    1次メッシュ(80km)  4桁  緯度 2/3度       経度 1度
    2次メッシュ(10km)  6桁  緯度 1/12度      経度 1/8度
    3次メッシュ(1km)   8桁  緯度 1/120度     経度 1/80度
    1/10 細分(100m)   10桁  緯度 1/1200度    経度 1/800度

解像度の注意
------------
100m メッシュは **1セルが約 25.6 画素**（タイルは 4.48m/画素）。
128×128 のタイルに **5×5 セルしか入らない**ので、この入力は非常に粗い。
「この斜面の下に人家があるか」という粗い情報としては足りる見込みだが、
細かい建物の形は表現できない。

    python -m dc5lib.landuse          # 自己検査
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 1画素あたりの度（タイル 128px / 0.00625度 x 0.00416667度）
DLON, DLAT = 0.00625, 0.00416667
TILE = 128

# メッシュ1セルの大きさ[度]
MESH_DLAT = 1.0 / 1200.0
MESH_DLON = 1.0 / 800.0

# L03-b の土地利用区分（コード -> 名前）。原典の仕様書より
LANDUSE_CLASSES = {
    "0100": "田",
    "0200": "その他の農用地",
    "0300": "森林",
    "0400": "荒地",
    "0500": "建物用地",
    "0600": "道路",
    "0700": "鉄道",
    "0801": "その他の用地",
    "0901": "河川地及び湖沼",
    "1000": "海浜",
    "1100": "海水域",
    "1400": "ゴルフ場",
    "1500": "その他",
    "0500_old": "建物用地",
}

# モデルに渡すときの並び。**建物用地を先頭に置く**（指定基準に直結するため）
CHANNEL_ORDER = ("0500", "0600", "0700", "0100", "0200", "0300", "0400")


def mesh_code(lat: float, lon: float) -> str:
    """緯度経度から 1/10 細分（100m）メッシュコード10桁を作る。"""
    p, lat_r = divmod(lat * 1.5, 1.0)
    u, lon_r = divmod(lon - 100.0, 1.0)
    q, lat_r = divmod(lat_r * 8.0, 1.0)
    v, lon_r = divmod(lon_r * 8.0, 1.0)
    r, lat_r = divmod(lat_r * 10.0, 1.0)
    w, lon_r = divmod(lon_r * 10.0, 1.0)
    m = int(lat_r * 10.0)
    n = int(lon_r * 10.0)
    return (f"{int(p):02d}{int(u):02d}{int(q)}{int(v)}"
            f"{int(r)}{int(w)}{m}{n}")


def mesh_bounds(code: str):
    """メッシュコードから (南端緯度, 西端経度, 北端緯度, 東端経度)。

    mesh_code の逆。自己検査に使う。
    """
    if len(code) not in (4, 6, 8, 10):
        raise ValueError(f"メッシュコードの桁数が不正: {code!r}")
    lat = int(code[0:2]) / 1.5
    lon = int(code[2:4]) + 100.0
    dlat, dlon = 2.0 / 3.0, 1.0
    if len(code) >= 6:
        dlat, dlon = dlat / 8.0, dlon / 8.0
        lat += int(code[4]) * dlat
        lon += int(code[5]) * dlon
    if len(code) >= 8:
        dlat, dlon = dlat / 10.0, dlon / 10.0
        lat += int(code[6]) * dlat
        lon += int(code[7]) * dlon
    if len(code) >= 10:
        dlat, dlon = dlat / 10.0, dlon / 10.0
        lat += int(code[8]) * dlat
        lon += int(code[9]) * dlon
    return lat, lon, lat + dlat, lon + dlon


def tile_mesh_codes(geoinfo, tile=TILE):
    """1タイル分の各画素のメッシュコード (tile, tile) の文字列配列。

    geoinfo は GDAL 形式の6要素 (原点経度, 画素幅, 0, 原点緯度, 0, -画素高)。
    **緯度は上が大きい**（画素高が負）ことに注意。
    """
    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    # 画素の中心をとる
    lons = lon0 + (np.arange(tile) + 0.5) * dlon
    lats = lat0 + (np.arange(tile) + 0.5) * dlat
    out = np.empty((tile, tile), dtype="<U10")
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            out[i, j] = mesh_code(la, lo)
    return out


def load_l03b(path):
    """L03-b の GeoJSON を読んで {メッシュコード: 土地利用コード} を返す。

    国土数値情報の GeoJSON は 1 フィーチャが 1 メッシュで、属性に
    メッシュコード（`メッシュ` / `MESH_CODE` 等）と土地利用コード
    （`土地利用種別` / `LANDUSE` 等）が入っている。
    版によって属性名が違うので候補を順に試す。
    """
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    feats = d.get("features", d if isinstance(d, list) else [])
    mesh_keys = ("メッシュ", "メッシュコード", "MESH_CODE", "mesh", "L03b_001")
    use_keys = ("土地利用種別", "土地利用コード", "LANDUSE", "L03b_002", "landuse")
    out = {}
    mk = uk = None
    for f in feats:
        pr = f.get("properties", f)
        if mk is None:
            mk = next((k for k in mesh_keys if k in pr), None)
            uk = next((k for k in use_keys if k in pr), None)
            if mk is None or uk is None:
                raise KeyError(f"属性名が分かりません: {sorted(pr)[:12]}")
        out[str(pr[mk])] = str(pr[uk]).zfill(4)
    return out


def tile_channels(geoinfo, table, order=CHANNEL_ORDER, tile=TILE):
    """1タイル分を one-hot のチャネルにする (len(order), tile, tile) float32。

    table に無いメッシュは全チャネル0（= 不明）。
    """
    codes = tile_mesh_codes(geoinfo, tile)
    idx = {c: i for i, c in enumerate(order)}
    out = np.zeros((len(order), tile, tile), np.float32)
    for code, use in ((c, table.get(c)) for c in np.unique(codes)):
        if use is None:
            continue
        i = idx.get(use)
        if i is None:
            continue
        out[i][codes == code] = 1.0
    return out


def _selftest():
    ok = True

    # (1) コード -> 範囲 -> 中心 -> コード の往復
    rng = np.random.default_rng(0)
    for _ in range(2000):
        lat = float(rng.uniform(31.0, 45.0))
        lon = float(rng.uniform(129.0, 145.0))
        c = mesh_code(lat, lon)
        s, w, n, e = mesh_bounds(c)
        if not (s <= lat < n and w <= lon < e):
            print(f"  NG 範囲に入らない lat={lat} lon={lon} code={c} "
                  f"bounds=({s},{w},{n},{e})")
            ok = False
            break
        if mesh_code((s + n) / 2, (w + e) / 2) != c:
            print(f"  NG 往復しない code={c}")
            ok = False
            break
    print("  往復検査 2000点:", "OK" if ok else "NG")

    # (2) セルの大きさ
    s, w, n, e = mesh_bounds("53394611" + "00")
    print(f"  1セルの大きさ: 緯度 {n - s:.8f}度 / 経度 {e - w:.8f}度"
          f"（期待 {MESH_DLAT:.8f} / {MESH_DLON:.8f}）")
    assert abs((n - s) - MESH_DLAT) < 1e-12 and abs((e - w) - MESH_DLON) < 1e-12

    # (3) 既知の位置。東京駅は1次メッシュ 5339、広島市付近は 5132
    tokyo = mesh_code(35.681, 139.767)
    hiro = mesh_code(34.40, 132.46)
    print(f"  東京駅 -> {tokyo}（1次メッシュ {tokyo[:4]}、期待 5339）")
    print(f"  広島市 -> {hiro}（1次メッシュ {hiro[:4]}、期待 5132）")
    assert tokyo[:4] == "5339", tokyo
    assert hiro[:4] == "5132", hiro
    # 本人の DEM ファイル名 FG-GML-513213-DEM5A... が 2次メッシュ 513213 なので、
    # 広島のタイルがその中に来るかも見る
    print(f"  広島市 の2次メッシュ -> {hiro[:6]}"
          f"（手元の DEM は 513213〜513240 の範囲）")

    # (4) タイル1枚が何セルにまたがるか
    gi = (132.40625, DLON / TILE, 0.0, 34.025, 0.0, -DLAT / TILE)
    codes = tile_mesh_codes(gi)
    uniq = len(np.unique(codes))
    print(f"  タイル1枚にかかるメッシュ: {uniq} セル"
          f"（1セル ≒ {MESH_DLAT / (DLAT / TILE):.1f} x "
          f"{MESH_DLON / (DLON / TILE):.1f} 画素）")

    # (5) one-hot。建物用地だけの作り表で形と中身を見る
    table = {c: "0500" for c in np.unique(codes)}
    ch = tile_channels(gi, table)
    print(f"  one-hot の形 {ch.shape} / 建物用地チャネルの合計 "
          f"{ch[0].sum():.0f}（期待 {TILE * TILE}）")
    assert ch.shape == (len(CHANNEL_ORDER), TILE, TILE)
    assert ch[0].sum() == TILE * TILE
    assert ch[1:].sum() == 0
    print("  自己検査: 問題なし" if ok else "  自己検査: NG")
    return ok


if __name__ == "__main__":
    print("土地利用細分メッシュ (L03-b) の変換モジュール")
    print(f"チャネルの並び: {CHANNEL_ORDER}")
    print(f"  {' / '.join(LANDUSE_CLASSES.get(c, c) for c in CHANNEL_ORDER)}")
    print()
    _selftest()
