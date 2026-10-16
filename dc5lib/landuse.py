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

# 土地利用区分（コード -> 名前）。
#
# **コード表は年次ごとに違う。** GML の rangeParameters に
#   <ksj:landuseCode gml:CodeSpace="LandUseCd-09.html" />
# と書いてあるものが正。下は平成28年版（LandUseCd-09）。
#   https://nlftp.mlit.go.jp/ksj/gml/codelist/LandUseCd-09.html
#
# 別の年次を使うときは必ず GML の CodeSpace を見て確認すること。
# 以前ここに別の版の表を書いていて、森林(0500)を建物用地と読み違えた。
# 分布を見れば気づける（広島・島根なら森林が約6割、海水域が約2割）。
CODELIST = "LandUseCd-09"
LANDUSE_CLASSES = {
    "0100": "田",
    "0200": "その他の農用地",
    "0500": "森林",
    "0600": "荒地",
    "0700": "建物用地",
    "0901": "道路",
    "0902": "鉄道",
    "1000": "その他の用地",
    "1100": "河川地及び湖沼",
    "1400": "海浜",
    "1500": "海水域",
    "1600": "ゴルフ場",
}

# 平成28年の広島・島根8メッシュでの実測の出現割合（確認用）
OBSERVED_SHARE = {
    "0500": 59.22, "1500": 21.50, "0100": 7.32, "0700": 4.96, "0200": 2.80,
    "1100": 1.89, "1000": 0.95, "0600": 0.71, "0901": 0.26, "1600": 0.20,
    "0902": 0.16, "1400": 0.03,
}

# モデルに渡すときの並び。**建物用地を先頭に置く**（指定基準に直結するため）。
# この9区分で実測の 99.6% を覆う。残り（鉄道・海浜・ゴルフ場）は UNKNOWN。
CHANNEL_ORDER = ("0700", "0901", "1000", "0100", "0200", "0500", "0600",
                 "1100", "1500")


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


MESH_KEYS = ("メッシュ", "メッシュコード", "MESH_CODE", "mesh", "L03b_001",
             "L03b_c01", "MESHCODE")
USE_KEYS = ("土地利用種別", "土地利用種", "土地利用コード", "LANDUSE", "L03b_002",
            "landuse", "L03b_c02", "LU_CODE")
# 「土地利用種」は DBF のフィールド名が **10バイトで打ち切られる**ため。
# cp932 では漢字1字が2バイトなので「土地利用種別」(12B) が「土地利用種」(10B) になる。

# 測地系。**ここを間違えると約430m ずれる**（1タイル 573m、1メッシュ 100m）。
# 本プロジェクトのタイルは全件 EPSG:6668（JGD2011、世界測地系）。
DATUM_WORLD = {4612, 6668, 4326, 6697}      # JGD2000 / JGD2011 / WGS84
DATUM_TOKYO = {4301, 4300}                  # 日本測地系（旧測地系）


def _check_datum(epsg, path):
    """日本測地系のファイルを黙って受け入れない。

    国土数値情報の土地利用細分メッシュは、平成18年以降の年次で
    **日本測地系と世界測地系の両方**が配布されている。日本測地系を
    読むと、広島・島根の緯度で **約430m（約96画素、メッシュ4セル超）**
    ずれる。タイル1枚が 573m なので、ほぼ隣のタイルを見ることになる。
    """
    if epsg is None:
        raise ValueError(
            f"{Path(path).name}: 測地系が判定できません。"
            f"シェープファイルの .prj があるか確認してください。"
            f"**世界測地系（JGD2000/JGD2011）のファイルを使ってください**")
    e = int(epsg)
    if e in DATUM_TOKYO:
        raise ValueError(
            f"{Path(path).name}: EPSG:{e} は**日本測地系**です。"
            f"本プロジェクトのタイルは EPSG:6668（世界測地系）なので、"
            f"約430m ずれます。ダウンロードページで**世界測地系**の方を"
            f"選び直してください")
    if e not in DATUM_WORLD:
        raise ValueError(
            f"{Path(path).name}: EPSG:{e} は想定外です。"
            f"世界測地系（{sorted(DATUM_WORLD)}）のファイルを使ってください")
    return e


def _pick_keys(fields):
    mk = next((k for k in MESH_KEYS if k in fields), None)
    uk = next((k for k in USE_KEYS if k in fields), None)
    if mk is None or uk is None:
        raise KeyError(f"属性名が分かりません: {sorted(fields)[:12]}")
    return mk, uk


def load_l03b_geojson(path):
    """GeoJSON 版。配布には無いが、自分で変換した場合に使う。

    **測地系の検査ができない**ので、変換元が世界測地系だったことを
    自分で保証すること。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    feats = d.get("features", d if isinstance(d, list) else [])
    out, mk, uk = {}, None, None
    for f in feats:
        pr = f.get("properties", f)
        if mk is None:
            mk, uk = _pick_keys(pr)
        out[str(pr[mk])] = str(pr[uk]).zfill(4)
    return out


def load_l03b_shapefile(path):
    """シェープファイル版（配布形式）。`geopack` 環境の python で動かす。

    .prj から測地系を読んで、日本測地系なら止める。
    """
    import shapefile                      # pyshp（純 Python。DLL 無し）

    path = Path(path)
    epsg = None
    prj = path.with_suffix(".prj")
    if prj.exists():
        txt = prj.read_text(encoding="utf-8", errors="ignore")
        for code, keys in ((6668, ("JGD2011", "JGD_2011")),
                           (4612, ("JGD2000", "JGD_2000", "Japan_2000")),
                           (4301, ("Tokyo",)),
                           (4326, ("WGS_1984", "WGS84"))):
            if any(k in txt for k in keys):
                epsg = code
                break
    _check_datum(epsg, path)

    r = shapefile.Reader(str(path), encoding="cp932")
    fields = [f[0] for f in r.fields[1:]]
    mk, uk = _pick_keys(fields)
    im, iu = fields.index(mk), fields.index(uk)
    out = {}
    for rec in r.iterRecords():
        out[str(rec[im])] = str(rec[iu]).zfill(4)
    return out


def load_l03b(path):
    """拡張子で振り分ける。"""
    p = Path(path)
    if p.suffix.lower() in (".shp",):
        return load_l03b_shapefile(p)
    if p.suffix.lower() in (".json", ".geojson"):
        return load_l03b_geojson(p)
    raise ValueError(f"未対応の形式: {p.suffix}（.shp か .geojson）")


# ---- 格子で持つ版 ----------------------------------------------------------
# 1次メッシュは 1/10細分で 800x800 セル（8 x 10 x 10）。
# コード -> 辞書は 8ファイルで 512万件になって重いので、
# 1次メッシュごとに 800x800 の uint8 配列に詰める（1枚 640KB）。

SUBDIV = 800            # 1次メッシュあたりのセル数（各軸）
UNKNOWN = 255           # 未整備・海域など


def code_to_rowcol(code: str):
    """10桁コード -> 1次メッシュ内の (南からの行, 西からの列)。"""
    q, v, r, w, m, n = (int(code[4]), int(code[5]), int(code[6]),
                        int(code[7]), int(code[8]), int(code[9]))
    return q * 100 + r * 10 + m, v * 100 + w * 10 + n


def load_l03b_grid(path, order=CHANNEL_ORDER):
    """1ファイルを (1次メッシュコード, 800x800 の区分番号) にする。

    区分番号は order の添字。order に無い区分と未整備は UNKNOWN。
    """
    table = load_l03b(path)
    check_codes(table, Path(path).name)
    idx = {c: i for i, c in enumerate(order)}
    grid = np.full((SUBDIV, SUBDIV), UNKNOWN, np.uint8)
    primary = None
    for code, use in table.items():
        if len(code) != 10:
            continue
        if primary is None:
            primary = code[:4]
        r, c = code_to_rowcol(code)
        grid[r, c] = idx.get(use, UNKNOWN)
    return primary, grid


def check_codes(table, path=""):
    """知らないコードが混ざっていないかを見る。年次を取り違えると全部ずれる。"""
    from collections import Counter
    c = Counter(table.values())
    unknown = {k: v for k, v in c.items() if k not in LANDUSE_CLASSES}
    if unknown:
        raise ValueError(
            f"{path}: {CODELIST} に無いコードがあります {dict(unknown)}。"
            f"**年次が違うかもしれません。** GML の rangeParameters の "
            f"CodeSpace を確認してください")
    return c


def load_all_grids(folder, order=CHANNEL_ORDER, verbose=True):
    """フォルダの *.shp を全部読んで {1次メッシュコード: 800x800} にする。"""
    folder = Path(folder)
    shps = sorted(folder.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"{folder} に .shp がありません")
    out = {}
    for p in shps:
        primary, grid = load_l03b_grid(p, order)
        out[primary] = grid
        if verbose:
            known = int((grid != UNKNOWN).sum())
            print(f"  {p.name}: 1次メッシュ {primary} / "
                  f"区分あり {known:,} / {SUBDIV * SUBDIV:,} セル "
                  f"({100 * known / SUBDIV ** 2:.1f}%)")
    return out


def load_grids_npz(path):
    """tools/dump_landuse_grids.py が書いた npz を読む。

    シェープファイルを読むには pyshp が要り、それがあるのは geopack 環境。
    学習側は mayo 環境なので、間に npz を置いて渡す。
    """
    z = np.load(path, allow_pickle=False)
    order = tuple(str(x) for x in z["order"])
    if order != tuple(CHANNEL_ORDER):
        raise ValueError(
            f"{Path(path).name}: チャネルの並びが違います。"
            f"npz は {order} / いまのコードは {tuple(CHANNEL_ORDER)}。"
            f"dump_landuse_grids.py で作り直してください")
    return {str(k): z[f"g{k}"] for k in z["keys"]}


def tile_channels_grid(geoinfo, grids, order=CHANNEL_ORDER, tile=TILE):
    """格子版。(len(order), tile, tile) float32 の one-hot。

    tile_channels と同じ結果だが、文字列を作らないので速い。
    """
    lon0, dlon = float(geoinfo[0]), float(geoinfo[1])
    lat0, dlat = float(geoinfo[3]), float(geoinfo[5])
    lons = lon0 + (np.arange(tile) + 0.5) * dlon
    lats = lat0 + (np.arange(tile) + 0.5) * dlat
    LA, LO = np.meshgrid(lats, lons, indexing="ij")

    p = np.floor(LA * 1.5).astype(int)
    u = np.floor(LO - 100.0).astype(int)
    row = np.floor((LA - p / 1.5) / MESH_DLAT).astype(int)
    col = np.floor((LO - (u + 100.0)) / MESH_DLON).astype(int)
    np.clip(row, 0, SUBDIV - 1, out=row)
    np.clip(col, 0, SUBDIV - 1, out=col)

    cls = np.full((tile, tile), UNKNOWN, np.uint8)
    for pk in np.unique(p * 100 + u):
        key = f"{pk // 100:02d}{pk % 100:02d}"
        g = grids.get(key)
        if g is None:
            continue
        sel = (p * 100 + u) == pk
        cls[sel] = g[row[sel], col[sel]]

    out = np.zeros((len(order), tile, tile), np.float32)
    for i in range(len(order)):
        out[i][cls == i] = 1.0
    return out, cls


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
    table = {c: CHANNEL_ORDER[0] for c in np.unique(codes)}   # 直書きしない
    ch = tile_channels(gi, table)
    print(f"  one-hot の形 {ch.shape} / 建物用地チャネルの合計 "
          f"{ch[0].sum():.0f}（期待 {TILE * TILE}）")
    assert ch.shape == (len(CHANNEL_ORDER), TILE, TILE)
    assert ch[0].sum() == TILE * TILE
    assert ch[1:].sum() == 0
    # (6) 測地系の検査が日本測地系を弾くか
    import tempfile
    for epsg, want_ng in ((6668, False), (4612, False), (4301, True), (2444, True),
                          (None, True)):
        try:
            _check_datum(epsg, "dummy.shp")
            got_ng = False
        except ValueError:
            got_ng = True
        mark = "OK " if got_ng == want_ng else "NG "
        if got_ng != want_ng:
            ok = False
        print(f"  {mark}測地系 EPSG:{epsg} -> "
              f"{'弾く' if got_ng else '通す'}（期待 {'弾く' if want_ng else '通す'}）")

    print("  自己検査: 問題なし" if ok else "  自己検査: NG")
    return ok


if __name__ == "__main__":
    print("土地利用細分メッシュ (L03-b) の変換モジュール")
    print(f"チャネルの並び: {CHANNEL_ORDER}")
    print(f"  {' / '.join(LANDUSE_CLASSES.get(c, c) for c in CHANNEL_ORDER)}")
    print()
    _selftest()
