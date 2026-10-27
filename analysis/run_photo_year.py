# -*- coding: utf-8 -*-
"""タイルごとの航空写真の撮影年度を調べる（第2章 使用データ / 第6章 汎化）。

なぜ要るか
----------
学習に使っている航空写真は地理院の `seamlessphoto`（全国最新写真・シームレス）で、
**複数時期の写真の合成**である。タイル自体に撮影年の情報が無いので、
「写真が古くて新しい宅地開発を見落としているのではないか」という問いに
そのままでは答えられない。

どうやるか
----------
地理院は年度別のレイヤ `nendophoto{年}` も配信している。
ある地点で**画像が返る最も新しい年度**を探せば、その地点の写真の鮮度が分かる。

    https://cyberjapandata.gsi.go.jp/xyz/nendophoto2022/18/230100/103470.png

**ただし `seamlessphoto` と `nendophoto` が同じ写真とは限らない。**
シームレス写真には地方公共団体の写真や簡易空中写真も入っており、
それらは年度別レイヤに無いことがある。そこで `--verify` で
同じ座標の2枚を実際に突き合わせ、**どれだけ一致するかを測ってから**使う。

使い方
------
    python analysis/run_photo_year.py --verify            # まず一致率を測る
    python analysis/run_photo_year.py --scan --limit 300  # 試しに300タイル
    python analysis/run_photo_year.py --scan              # テスト集合ぜんぶ

出力は cache/analysis/photo_year__<地域>.csv（タイル番号, 撮影年度）。
**取得は地理院のサーバに負荷をかけるので、間隔を空ける。**
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import requests
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/{lyr}/{z}/{x}/{y}.{ext}"
YEARS = list(range(2024, 2006, -1))      # 新しい年度から探す
PROBE_Z = 16                             # 約600m。学習タイル（573m）に近い
SLEEP = 0.05                             # 地理院への間隔


def lonlat_to_tile(lon, lat, z):
    siny = min(max(math.sin(math.radians(lat)), -0.9999), 0.9999)
    n = 2 ** z
    x = int(n * (0.5 + lon / 360.0))
    y = int(n * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)))
    return x, y


def _get(sess, lyr, z, x, y, ext):
    try:
        r = sess.get(TILE_URL.format(lyr=lyr, z=z, x=x, y=y, ext=ext), timeout=12)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    ex = im.convert("L").getextrema()
    return None if ex[0] == ex[1] else im      # 単色は中身なし


def newest_year(sess, x, y, z=PROBE_Z, cache=None):
    """その座標で画像が返るいちばん新しい年度。無ければ None。"""
    key = (z, x, y)
    if cache is not None and key in cache:
        return cache[key]
    out = None
    for yr in YEARS:
        if _get(sess, f"nendophoto{yr}", z, x, y, "png") is not None:
            out = yr
            break
        time.sleep(SLEEP)
    if cache is not None:
        cache[key] = out
    return out


def verify(df, n=40, seed=0):
    """seamlessphoto と nendophoto が同じ写真かを実測する。

    **ここを飛ばしてはいけない。** 違う写真なら、年度別レイヤから出した
    「撮影年度」は学習に使った写真の年度ではない。
    """
    sess = requests.Session()
    sess.headers["User-Agent"] = "dc5-dataset-provenance/1.0"
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    rows = []
    for k in idx:
        r = df.iloc[k]
        x, y = lonlat_to_tile(r.lon, r.lat, PROBE_Z)
        sp = _get(sess, "seamlessphoto", PROBE_Z, x, y, "jpg")
        if sp is None:
            continue
        yr = newest_year(sess, x, y)
        if yr is None:
            rows.append(dict(tile_no=r.tile_no, year=None, diff=np.nan))
            continue
        np_im = _get(sess, f"nendophoto{yr}", PROBE_Z, x, y, "png")
        if np_im is None:
            continue
        a = np.asarray(sp.resize((128, 128)), dtype=np.float32)
        b = np.asarray(np_im.resize((128, 128)), dtype=np.float32)
        rows.append(dict(tile_no=r.tile_no, year=yr,
                         diff=float(np.abs(a - b).mean())))
    t = pd.DataFrame(rows)
    ok = t.dropna(subset=["diff"])
    print(f"照合 {len(t)} タイル / 年度が取れた {len(ok)} タイル")
    if len(ok):
        print(f"  画素差の平均 {ok['diff'].mean():.1f}（0〜255）")
        print(f"  5 未満（ほぼ同一） {int((ok['diff'] < 5).sum())} / {len(ok)}")
        print(f"  20 以上（別物）   {int((ok['diff'] >= 20).sum())} / {len(ok)}")
        print()
        print(ok.year.value_counts().sort_index(ascending=False).to_string())
    miss = int(t.year.isna().sum())
    if miss:
        print(f"  ※ 年度別レイヤに無い {miss} タイル"
              f"（地方公共団体の写真などで、シームレスにしか入っていない）")
    return t


def scan(df, out, limit=None, resume=True):
    sess = requests.Session()
    sess.headers["User-Agent"] = "dc5-dataset-provenance/1.0"
    done = {}
    if resume and out.exists():
        old = pd.read_csv(out)
        done = dict(zip(old.tile_no.astype(int), old.year))
        print(f"続きから: 済み {len(done)} タイル")
    todo = df if limit is None else df.head(limit)
    cache, rows, t0 = {}, [], time.time()
    for i, r in enumerate(todo.itertuples(), 1):
        if int(r.tile_no) in done:
            rows.append(dict(tile_no=int(r.tile_no), year=done[int(r.tile_no)]))
            continue
        x, y = lonlat_to_tile(r.lon, r.lat, PROBE_Z)
        rows.append(dict(tile_no=int(r.tile_no),
                         year=newest_year(sess, x, y, cache=cache)))
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(todo)}  {el:.0f}秒  "
                  f"残り {el / i * (len(todo) - i) / 60:.0f}分", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False, encoding="utf-8")
    print(f"書き出し: {out}（{len(t)} タイル）")
    print(t.year.value_counts(dropna=False).sort_index(ascending=False).to_string())
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--verify", action="store_true", help="シームレスと年度別を突き合わせる")
    ap.add_argument("--scan", action="store_true", help="撮影年度を集める")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--test-only", action="store_true",
                    help="テスト集合のタイルだけを対象にする（広島のホールドアウト）")
    ap.add_argument("--gt-only", action="store_true",
                    help="正解を含むタイルだけにする（既定は regions.yaml に従う）")
    args = ap.parse_args()

    # **対象は regions.yaml の subset に合わせる。**
    # 島根は subset: all で全 24,569 タイルを評価に使うのに、
    # 既定の gt_only=True で正解ありの 9,679 枚しか集めていなかった。
    # 評価に使うタイルの year が空になるので、既定を地域から決める。
    from dc5lib import regions
    from analysis.run_block_split import tile_geo
    subset = regions.get(args.region).subset if args.region in regions.keys() else "all"
    gt_only = args.gt_only or (subset != "all" and not args.test_only)
    df = tile_geo(args.region, gt_only=gt_only)
    if args.test_only:
        from dc5lib.data import holdout_indices
        keep = set(holdout_indices(len(df)).tolist())
        df = df.iloc[sorted(keep)].reset_index(drop=True)
    print(f"対象 {len(df)} タイル"
          f"（subset={subset} / 正解ありのみ={gt_only}）")

    if args.verify or not args.scan:
        verify(df)
    if args.scan:
        out = paths.cache_dir("analysis") / f"photo_year__{args.region}.csv"
        scan(df, out, args.limit)


if __name__ == "__main__":
    main()
