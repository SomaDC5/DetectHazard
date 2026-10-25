# -*- coding: utf-8 -*-
"""指定年月日と航空写真の撮影年度を突き合わせる（教授の指摘の検証・本命）。

指摘
----
「精度が落ちているのは、航空写真が古く、宅地開発が進んでいるところを
見落としているのではないか」

なぜこれが効くのか
------------------
急傾斜地崩壊危険区域の指定基準には**人家の有無**が含まれる。
だから「指定が写真より後」の区域では、

  指定の原因になった宅地が、写真にはまだ写っていない
  → モデルは「人家のない斜面」と見て落とす

という筋が立つ。国土数値情報 A33 は**指定年月日（A33_007）を持っている**ので、
タイルごとの撮影年度（`run_photo_year.py`）と突き合わせれば直接測れる。

タイル単位で撮影年度ごとに切るだけでは弱い。それだと
「写真が古い地域はもともと当てにくい」と区別できないが、
**同じタイルの中で「写真より後に指定された区域」と「前から指定されていた区域」**を
比べれば、地域の性質が揃うので交絡がだいぶ減る。

    python analysis/run_designation_year.py
    python analysis/run_designation_year.py --condition FinalFusion_SAM_APM_OverlapTrain
"""

from __future__ import annotations

import argparse
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

SHP = r"C:\Users\hirok\OneDrive\Desktop\CrZ\newHirosima\Shape\A33-22_34Polygon"
KIND = 1              # A33_001 = 1 は「急傾斜地の崩壊」。学習もこれだけを使う
BAND = 0.0071         # 箇所F のゆらぎ幅


def tile_grid(geo):
    """タイルの1歩（経度・緯度）を求める。GeoInfo の左上座標から復元する。"""
    lon = np.sort(geo.lon.unique())
    lat = np.sort(geo.lat.unique())
    return (float(np.median(np.diff(lon))), float(np.median(np.diff(lat))))


def designation_by_tile(geo, verbose=True):
    """タイルごとの『いちばん新しい指定年』を返す。

    区域のポリゴンはタイル（573 m 四方）より小さいが、
    **またぐものもある**ので、外接矩形が重なるタイルすべてに配る。
    """
    import shapefile
    dlon, dlat = tile_grid(geo)
    lon0, lat0 = geo.lon.min(), geo.lat.max()
    # タイルの左上座標 → タイル番号
    idx = {}
    for t, lo, la in zip(geo.tile_no.astype(int), geo.lon, geo.lat):
        idx[(int(round((lo - lon0) / dlon)), int(round((lat0 - la) / dlat)))] = t

    r = shapefile.Reader(SHP, encoding="cp932")
    names = [f[0] for f in r.fields[1:]]
    i_kind, i_date = names.index("A33_001"), names.index("A33_007")
    latest, n_poly, n_hit = defaultdict(int), 0, 0
    for sh, rec in zip(r.iterShapes(), r.iterRecords()):
        if rec[i_kind] != KIND:
            continue
        d = rec[i_date]
        y = getattr(d, "year", None)
        if y is None or y > 2100:            # 9999-01-01 は未設定
            continue
        n_poly += 1
        x0, y0, x1, y1 = sh.bbox            # (lon_min, lat_min, lon_max, lat_max)
        cx0 = int((x0 - lon0) // dlon)
        cx1 = int((x1 - lon0) // dlon)
        cy0 = int((lat0 - y1) // dlat)
        cy1 = int((lat0 - y0) // dlat)
        hit = False
        for gx in range(cx0, cx1 + 1):
            for gy in range(cy0, cy1 + 1):
                t = idx.get((gx, gy))
                if t is not None:
                    latest[t] = max(latest[t], y)
                    hit = True
        n_hit += hit
    if verbose:
        print(f"ポリゴン {n_poly} 件（指定年あり）／タイルに当たった {n_hit} 件")
        print(f"指定年が付いたタイル {len(latest)} / 全 {len(geo)}")
    return dict(latest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM")
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--cov", type=float, default=0.5)
    args = ap.parse_args()

    cache = paths.cache_dir("analysis")
    geo = pd.read_csv(cache / f"tiles_geo__{args.region}.csv")
    geo = geo[geo.gt_pos > 0].reset_index(drop=True)

    py = pd.read_csv(cache / f"photo_year__{args.region}.csv")
    photo = dict(zip(py.tile_no.astype(int), py.year))

    desig = designation_by_tile(geo)
    print()

    f = cache / "instances" / f"{args.condition}__{args.region}__gt.csv"
    if not f.exists():
        raise SystemExit(f"{f} が無い（--save-instances を付けて評価する）")
    g = pd.read_csv(f)
    g["photo"] = g.tile_no.astype(int).map(photo)
    g["desig"] = g.tile_no.astype(int).map(desig)
    g = g.dropna(subset=["photo", "desig"])
    g["hit"] = (g.coverage >= args.cov)
    g["after"] = g.desig > g.photo          # 指定が写真より後

    print(f"条件 {args.condition} / 正解 {len(g)} 件（写真年と指定年が揃ったもの）")
    print()
    print("【指定年と撮影年度の関係】")
    t = (g.groupby("after")
         .agg(正解数=("hit", "size"), 箇所R=("hit", "mean"),
              最良IoU=("best_iou", "mean"), 中央の大きさ=("gt_area", "median"))
         .reset_index())
    t["after"] = t.after.map({False: "指定が写真より前（写っている）",
                              True: "**指定が写真より後（写っていない）**"})
    print(t.to_string(index=False, formatters={
        "箇所R": "{:.4f}".format, "最良IoU": "{:.4f}".format,
        "中央の大きさ": "{:.0f}".format}))
    if len(t) == 2:
        d = float(t.箇所R.iloc[1] - t.箇所R.iloc[0])
        print(f"  差 {d:+.4f}（ゆらぎ幅 {BAND} の {abs(d) / BAND:.1f} 倍）")
        if d < -BAND:
            print("  → **写真より後に指定された区域のほうが当てられない。指摘を支持する。**")
        elif d > BAND:
            print("  → **逆向き。写真より後に指定された区域のほうが当たる。**")
        else:
            print("  → **帯の中。指定と写真の前後では変わらない。**")
    print()

    print("【同じ撮影年度の中で比べる（地域の性質を揃える）】")
    rows = []
    for yr, s in g.groupby("photo"):
        if len(s) < 200:
            continue
        a = s[s.after]
        b = s[~s.after]
        if len(a) < 50 or len(b) < 50:
            rows.append(dict(撮影年度=int(yr), 写真より後=len(a), 写真より前=len(b),
                             R_後=np.nan, R_前=np.nan, 差=np.nan))
            continue
        rows.append(dict(撮影年度=int(yr), 写真より後=len(a), 写真より前=len(b),
                         R_後=a.hit.mean(), R_前=b.hit.mean(),
                         差=a.hit.mean() - b.hit.mean()))
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, formatters={
            "R_後": "{:.4f}".format, "R_前": "{:.4f}".format, "差": "{:+.4f}".format}))
    print()
    print("【指定年ごと】")
    g["desig5"] = (g.desig // 5 * 5).astype(int)
    print(g.groupby("desig5").agg(正解数=("hit", "size"), 箇所R=("hit", "mean"))
          .reset_index().to_string(index=False,
                                   formatters={"箇所R": "{:.4f}".format}))


if __name__ == "__main__":
    main()
