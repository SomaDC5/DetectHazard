# -*- coding: utf-8 -*-
"""既存の pkl から、航空写真だけを高解像度で作り直す。

なぜこれが必要か（docs/展望.md 6節）
------------------------------------
航空写真はズーム18（約0.49 m/画素）で取得しているのに、DEM に合わせて
128x128（約4.48 m/画素）へ縮めてから保存している。

    fetch_aerial_v2.py:110
        img = Image.fromarray(crop).resize((dem_size, dem_size), Image.BILINEAR)

573m のタイトルなら本来 約1164x1164 画素ぶんの情報があるので、
**線形で約9倍、面積で約83倍を捨てている。** 捨てているのは、まさに
小さい箇所を見分けるための情報。32m 四方の箇所は、いま 7x7 画素しかない。

なぜ GeoTIFF から作り直さないのか
----------------------------------
既存の pkl に GeoInfo（GDAL 方式の geotransform）が入っているので、
**航空写真だけを差し替えられる。** DEM・マスク・タイル番号・分割に使う
並び順は一切変わらないので、既存条件と直接比較できる。

GeoTIFF から作り直すと元データ（SSD B）が必要になり、タイルの選別や
欠損の穴埋めの再現性も担保しなければならない。ここを避ける。

タイルキャッシュを使うので**ダウンロードは発生しない**
--------------------------------------------------------
    ~/デスクトップ/LandSlides/MakeDataset/tile_cache   ズーム18が 975,106枚 / 17GB

30タイトルを無作為に選んでキャッシュだけで引けるか試し、ミス0件だった。
国土地理院のタイルサーバには触らない。

地理的に同じ場所を指すことの確認
--------------------------------
`get_airphoto_exact` は **dem_size から地理範囲を計算する**。

    right = left + dem_size * geo[1]

そのため dem_size だけ増やすと範囲が広がり、別の場所を切り出してしまう。
画素ステップを倍率で割った geotransform を渡すことで範囲を保つ。

256 で作って 128 に落とし、既存の APM と比べた結果（8タイル）

    平均絶対差 0.73〜2.33（255階調）/ 相関 0.9980〜0.9994

使い方
------
    python tools/dataset_build/rebuild_apm_highres.py \
        --region Hiroshima --src hiroshima_sam_apm.pkl --factor 2

    # 出力は dc5-data/datasets/<region>/<元の名前>_apm<N>.pkl

メモリの注意
------------
**APM を 512 にすると広島で 30GB になり、31GB のマシンでは読み込めない。**
いまの学習・解析は pkl を丸ごと RAM に載せる作りなので、512 にするには
memmap 方式へ切り替える別の作業が必要。

    解像度   広島の APM   島根の APM
    128      1.9 GB       1.2 GB     （現在）
    256      7.5 GB       4.8 GB     ← 31GB マシンで扱える上限
    512     30.0 GB      19.3 GB     要 memmap 化
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import fetch_aerial_v2 as fa                      # noqa: E402
from dc5lib.data import load_dataset, photo_dataset  # noqa: E402
from dc5lib.paths import dataset_path              # noqa: E402

DEFAULT_CACHE = os.path.expanduser("~/デスクトップ/LandSlides/MakeDataset/tile_cache")
TILE_PX = 128          # 既存 pkl の1タイルの画素数


def geo_scaled(geo, factor):
    """同じ地理範囲を factor 倍の画素数で切り出すための geotransform。

    画素ステップ（geo[1] と geo[5]）を factor で割る。これをしないと
    get_airphoto_exact が範囲を factor 倍に広げてしまう。
    """
    lon0, sx, r1, lat0, r2, sy = geo
    return (lon0, sx / factor, r1, lat0, r2, sy / factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, help="Hiroshima / Shimane")
    ap.add_argument("--src", required=True, help="元の pkl のファイル名")
    ap.add_argument("--factor", type=int, default=2, help="解像度の倍率（2 なら 256x256）")
    ap.add_argument("--zoom", type=int, default=18, help="航空写真タイルのズーム")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="タイルキャッシュの場所")
    ap.add_argument("--out", default=None, help="出力ファイル名（既定は <元>_apm<N>.pkl）")
    ap.add_argument("--allow-download", action="store_true",
                    help="キャッシュに無いタイルをダウンロードする（既定は失敗させる）")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N タイルだけ処理（試運転用）")
    args = ap.parse_args()

    size = TILE_PX * args.factor
    out_name = args.out or f"{Path(args.src).stem}_apm{size}.pkl"
    out_path = dataset_path(args.region, out_name)

    if not os.path.isdir(args.cache):
        sys.exit(f"タイルキャッシュがありません: {args.cache}")

    # ダウンロードを封じる。キャッシュに無ければ気づけるようにする
    misses = []
    if not args.allow_download:
        _orig = fa.download_tile

        def cache_only(tx, ty, zoom, cache_dir, **kw):
            p = os.path.join(cache_dir, f"{zoom}_{tx}_{ty}.jpg")
            if not os.path.exists(p):
                misses.append((zoom, tx, ty))
                raise RuntimeError("cache miss")
            return _orig(tx, ty, zoom, cache_dir, **kw)

        fa.download_tile = cache_only

    print(f"元データ   : {dataset_path(args.region, args.src)}")
    print(f"出力       : {out_path}")
    print(f"解像度     : {TILE_PX}x{TILE_PX} → {size}x{size}  (factor={args.factor})")
    print(f"キャッシュ : {args.cache}")

    t0 = time.time()
    ds = load_dataset(dataset_path(args.region, args.src))
    n = len(ds.No) if not args.limit else min(args.limit, len(ds.No))
    print(f"読み込み {time.time() - t0:.0f}s / タイル {len(ds.No)}（処理 {n}）")

    # 元の航空写真は使わないので、GeoInfo を確保したら先に捨てる（約2GB）
    geoinfo = list(ds.GeoInfo[:n])
    if hasattr(ds, "AirPhoto"):
        ds.AirPhoto = None
        gc.collect()

    print(f"APM の確保: {n * size * size * 3 / 1e9:.1f} GB")
    apm = np.empty((n, size, size, 3), dtype=np.uint8)

    t0, failed = time.time(), []
    for i in range(n):
        try:
            img = fa.get_airphoto_exact(geo_scaled(geoinfo[i], args.factor),
                                        args.cache, dem_size=size, zoom=args.zoom)
            a = np.asarray(img, dtype=np.uint8)
            if a.shape != (size, size, 3):
                a = np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR),
                               dtype=np.uint8)
            apm[i] = a
        except Exception:
            apm[i] = 0
            failed.append(i)
        if i % 500 == 0 or i == n - 1:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-9)
            print(f"  {i+1}/{n}  {rate:.0f} tiles/s  残り {(n-i-1)/max(rate,1e-9)/60:.1f} 分"
                  f"  失敗 {len(failed)}", end="\r", flush=True)
    print()
    print(f"生成 {time.time()-t0:.0f}s / 失敗 {len(failed)} 件 / キャッシュミス {len(misses)} 件")
    if failed[:5]:
        print(f"  失敗したタイルの番号（先頭5件）: {[ds.No[i] for i in failed[:5]]}")
    if misses[:3]:
        print(f"  キャッシュミスの例: {misses[:3]}")

    out = photo_dataset()
    out.No = list(ds.No[:n])
    out.DEM = list(ds.DEM[:n])
    out.Mask = list(ds.Mask[:n])
    out.GeoInfo = geoinfo
    out.EPSG = list(ds.EPSG[:n])
    out.Max_H = list(ds.Max_H[:n])
    out.Min_H = list(ds.Min_H[:n])
    out.AirPhoto = apm

    del ds
    gc.collect()
    t0 = time.time()
    print("書き出し中 …")
    with open(out_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"書き出し {time.time()-t0:.0f}s / {os.path.getsize(out_path)/1e9:.1f} GB")
    print(f"\n完了: {out_path}")
    if failed:
        print("失敗したタイルは真っ黒（0）で埋めてある。件数が多いときは原因を確かめること。")


if __name__ == "__main__":
    main()
