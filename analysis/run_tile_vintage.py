# -*- coding: utf-8 -*-
"""タイルごとに3種類のデータの「いつのものか」を1枚の表にする。

なぜ要るか
----------
学習に使っている3つのデータは、**どれも単一の年のものではない。**

  DEM   基盤地図情報 5mメッシュ。区画ごとに測量年度が違う
        （広島は2022年度が6割、島根は2009年度が2割）
  APM   地理院のシームレス空中写真。複数時期の合成で、地点ごとに撮影年が違う
  マスク 国土数値情報 A33。版は1つだが、区域ごとに**指定年月日**が違う

「古いデータのところで精度が落ちているのではないか」という問いは
**タイル単位でこの3つが分からないと答えられない。** その表を作る。

作り方
------
  DEM   元の GML（FG-GML-...-DEM5A-....xml）の先頭を読む。
        orgMDId が元の測量（H24 なら平成24年度）、Envelope が範囲。
        **devDate は基盤地図情報としての提供日で、測量年とは別。**
  APM   年度別レイヤ nendophoto{年} を新しい順に当て、画像が返る最初の年度。
        run_photo_year.py が集めたものを読む。
  マスク A33 のポリゴンの外接矩形が掛かるタイルに、その指定年を配る。

使い方
------
    python analysis/run_tile_vintage.py --dem            # GML を読む（時間がかかる・1回でよい）
    python analysis/run_tile_vintage.py --mask
    python analysis/run_tile_vintage.py --build          # 表にまとめる
    python analysis/run_tile_vintage.py --region shimane --dem --mask --build

APM は別途 `run_photo_year.py --scan` で集める（地理院に取りに行くため遅い）。
無ければその列は空になる。
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths

# 元データの置き場所。**リポジトリの外**なので、無ければその列は作らない。
SRC = {
    "hiroshima": dict(
        dem=r"C:\Users\hirok\OneDrive\Desktop\CrZ\newHirosima\Ori",
        shp=r"C:\Users\hirok\OneDrive\Desktop\CrZ\newHirosima\Shape\A33-22_34Polygon",
        mask_edition="A33 令和4年度版（2022年）",
    ),
    "shimane": dict(
        dem=r"C:\Users\hirok\OneDrive\Desktop\CrZ\simane\simaneOri",
        shp=r"C:\Users\hirok\OneDrive\Desktop\CrZ\simane\32島根県\Shape\A33-20_32Polygon",
        mask_edition="A33 令和2年度版（2020年）",
    ),
}
KIND = 1                      # A33_001 = 1 は「急傾斜地の崩壊」。学習もこれだけ
OUT_DIR = paths.REPO_ROOT / "provenance"

_pat = {k: re.compile(p) for k, p in {
    "org": r"<orgMDId>([^<]*)</orgMDId>",
    "dev": r"<devDate[^>]*>\s*<gml:timePosition>([^<]*)<",
    "mesh": r"<mesh>([^<]*)</mesh>",
    "low": r"<gml:lowerCorner>\s*([-\d.]+)\s+([-\d.]+)\s*</gml:lowerCorner>",
    "high": r"<gml:upperCorner>\s*([-\d.]+)\s+([-\d.]+)\s*</gml:upperCorner>",
}.items()}


def wareki(mid):
    """H24GC008 → 2012（平成24年度）。R05... → 2023（令和5年度）。"""
    m = re.match(r"\s*([HRS])(\d{2})", mid or "")
    if not m:
        return None
    base = {"H": 1988, "R": 2018, "S": 1925}[m.group(1)]
    return base + int(m.group(2))


# ------------------------------------------------------------------ DEM
def scan_dem(region, force=False):
    """GML の先頭を読んで、区画ごとの測量年度と範囲を表にする。"""
    out = paths.cache_dir("analysis") / f"dem_mesh__{region}.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    d = Path(SRC[region]["dem"])
    if not d.exists():
        print(f"  ※ DEM の元データが無い: {d}")
        return None
    files = sorted(d.glob("*DEM5A*.xml"))
    print(f"DEM の GML を読む: {len(files)} ファイル（OneDrive なので時間がかかる）")
    rows = []
    for i, f in enumerate(files, 1):
        try:
            head = f.open("rb").read(3500).decode("utf-8", "replace")
        except Exception:
            continue
        g = {k: p.search(head) for k, p in _pat.items()}
        if not (g["low"] and g["high"]):
            continue
        rows.append(dict(
            mesh=g["mesh"].group(1) if g["mesh"] else f.stem,
            survey_year=wareki(g["org"].group(1) if g["org"] else ""),
            supply_date=g["dev"].group(1) if g["dev"] else "",
            lat0=float(g["low"].group(1)), lon0=float(g["low"].group(2)),
            lat1=float(g["high"].group(1)), lon1=float(g["high"].group(2)),
        ))
        if i % 2000 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False, encoding="utf-8")
    print(f"  書き出し: {out}（{len(t)} 区画）")
    return t


def dem_by_tile(geo, mesh):
    """タイルの中心が入る DEM 区画の測量年度を割り当てる。

    DEM の区画は 0.0125 度 x 0.00833 度で、タイル（約 0.0062 x 0.0052 度）より
    大きいが、**またぐタイルもある。** 中心で決め、またぐ場合は
    年度が割れているかを別の列で持たせる。
    """
    if mesh is None or not len(mesh):
        return {}, {}
    # 粗い格子に区画を入れておく（総当たりを避ける）
    cell = 0.05
    idx = defaultdict(list)
    for r in mesh.itertuples():
        for gx in range(int(r.lon0 // cell), int(r.lon1 // cell) + 1):
            for gy in range(int(r.lat0 // cell), int(r.lat1 // cell) + 1):
                idx[(gx, gy)].append(r)
    dlon = float(np.median(np.diff(np.sort(geo.lon.unique()))))
    dlat = float(np.median(np.diff(np.sort(geo.lat.unique()))))
    year, spread = {}, {}
    for t, lo, la in zip(geo.tile_no.astype(int), geo.lon, geo.lat):
        cx, cy = lo + dlon / 2, la - dlat / 2          # タイルの中心
        hit, over = None, set()
        for r in idx.get((int(cx // cell), int(cy // cell)), []):
            if r.lon0 <= cx <= r.lon1 and r.lat0 <= cy <= r.lat1:
                hit = r.survey_year
            # タイルの矩形と重なるか（年度が割れているかを見る）
            if not (r.lon1 < lo or r.lon0 > lo + dlon
                    or r.lat1 < la - dlat or r.lat0 > la):
                if r.survey_year is not None and not pd.isna(r.survey_year):
                    over.add(int(r.survey_year))
        if hit is not None and not pd.isna(hit):
            year[t] = int(hit)
        spread[t] = len(over)
    return year, spread


# ------------------------------------------------------------------ マスク
def mask_by_tile(region, geo):
    """A33 の指定年月日を、ポリゴンが掛かるタイルに配る。

    **かたまり単位ではなくタイル単位である。** ラスタ化後の連結成分と
    A33 のポリゴンは1対1に対応しないので、ここで分かるのは
    「そのタイルに掛かる区域の指定年の範囲」まで。
    """
    try:
        import shapefile
    except ImportError:
        print("  ※ pyshp が無い（pip install pyshp）")
        return {}
    p = SRC[region]["shp"]
    if not Path(p + ".shp").exists():
        print(f"  ※ シェープファイルが無い: {p}")
        return {}
    dlon = float(np.median(np.diff(np.sort(geo.lon.unique()))))
    dlat = float(np.median(np.diff(np.sort(geo.lat.unique()))))
    lon0, lat0 = geo.lon.min(), geo.lat.max()
    idx = {}
    for t, lo, la in zip(geo.tile_no.astype(int), geo.lon, geo.lat):
        idx[(int(round((lo - lon0) / dlon)), int(round((lat0 - la) / dlat)))] = t

    r = shapefile.Reader(p, encoding="cp932")
    names = [f[0] for f in r.fields[1:]]
    ik, id_ = names.index("A33_001"), names.index("A33_007")
    acc = defaultdict(list)
    n = 0
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        if rec[ik] != KIND:
            continue
        y = getattr(rec[id_], "year", None)
        if y is None or y > 2100:
            continue
        n += 1
        x0, y0, x1, y1 = sh.bbox
        for gx in range(int((x0 - lon0) // dlon), int((x1 - lon0) // dlon) + 1):
            for gy in range(int((lat0 - y1) // dlat), int((lat0 - y0) // dlat) + 1):
                t = idx.get((gx, gy))
                if t is not None:
                    acc[t].append(y)
    print(f"  指定年のあるポリゴン {n} 件 → {len(acc)} タイルに割り当て")
    return {t: (min(v), max(v), len(v)) for t, v in acc.items()}


# ------------------------------------------------------------------ 組み立て
def build(region):
    from analysis.run_block_split import tile_geo
    geo = tile_geo(region, gt_only=False)
    cache = paths.cache_dir("analysis")

    mesh = None
    f = cache / f"dem_mesh__{region}.csv"
    if f.exists():
        mesh = pd.read_csv(f)
    dem_year, dem_spread = dem_by_tile(geo, mesh)

    f = cache / f"photo_year__{region}.csv"
    apm = {}
    if f.exists():
        t = pd.read_csv(f)
        apm = dict(zip(t.tile_no.astype(int), t.year))
    else:
        print(f"  ※ {f} が無い（run_photo_year.py --scan）。APM の列は空になる")

    mk = mask_by_tile(region, geo[geo.gt_pos > 0])

    out = pd.DataFrame({
        "tile_no": geo.tile_no.astype(int),
        "lon": geo.lon, "lat": geo.lat,
        "gt_pos": geo.gt_pos,
        "dem_survey_year": [dem_year.get(t) for t in geo.tile_no.astype(int)],
        "dem_year_n": [dem_spread.get(t) for t in geo.tile_no.astype(int)],
        "apm_photo_year": [apm.get(t) for t in geo.tile_no.astype(int)],
        "mask_year_min": [mk.get(t, (None,) * 3)[0] for t in geo.tile_no.astype(int)],
        "mask_year_max": [mk.get(t, (None,) * 3)[1] for t in geo.tile_no.astype(int)],
        "mask_n_polygon": [mk.get(t, (None,) * 3)[2] for t in geo.tile_no.astype(int)],
    })
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"tile_vintage__{region}.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"\n書き出し: {p}（{len(out)} タイル）")
    print(f"マスクの版: {SRC[region]['mask_edition']}")
    for c, lab in [("dem_survey_year", "DEM 測量年度"),
                   ("apm_photo_year", "APM 撮影年度"),
                   ("mask_year_max", "指定年（最新）")]:
        s = out[c].dropna()
        if not len(s):
            print(f"  {lab:<14} なし")
            continue
        print(f"  {lab:<14} {len(s):>6} タイル  "
              f"{int(s.min())}〜{int(s.max())}  中央 {int(s.median())}")
    n = int((out.dem_year_n.fillna(0) > 1).sum())
    if n:
        print(f"  ※ DEM の年度が割れているタイル {n} 枚"
              f"（複数の測量区画にまたがる）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=list(SRC))
    ap.add_argument("--dem", action="store_true", help="GML を読み直す")
    ap.add_argument("--mask", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.dem:
        scan_dem(a.region, a.force)
    if a.build or not (a.dem or a.mask):
        build(a.region)


if __name__ == "__main__":
    main()
