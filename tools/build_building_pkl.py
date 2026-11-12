# -*- coding: utf-8 -*-
"""建物の被覆率をタイルに焼き込んで pkl にする。

**タイル番号は元の pkl の `No` をそのまま使う。**
座標も元の pkl の `GeoInfo` から引くので、SAM・DEM・マスクと画素単位でずれない。
（`build_landuse_pkl.py` と同じ作法。詳しくは `dc5lib/building.py` の冒頭）

作るもの
--------
    hiroshima_bld.pkl           約 0.3 GB。建物だけ。解析・突き合わせ用
    hiroshima_sam_apm_bld.pkl   約 6 GB。SAM+APM に建物を足したもの。学習用

足す属性（両方に入る）
----------------------
    Building        uint8 [n, 2, 128, 128]  被覆率 × 255
                      [0] 建物（普通建物 + 堅ろう建物）
                      [1] 無壁舎（普通無壁舎 + 堅ろう無壁舎）
    BuildingOrder   面の並び。**pkl 自身に持たせる**（LandUseOrder と同じ理由）
    BuildingCount   int32 [n, 4]  棟数。TYPE_ORDER の順。**重心で数える**
    BuildingCoarse  int32 [n]     うち 1/25000 の地図由来（位置精度が粗い）
    BuildingMeshOK  uint8 [n]     1 = タイルの4隅すべてに GML がある
    BuildingTypeOrder / BuildingSource / BuildingNote

使い方
------
    python tools/dump_building_index.py --gml <GMLのフォルダ>
    python tools/build_building_pkl.py --region Hiroshima --light-only
    python tools/build_building_pkl.py --region Hiroshima
    python tools/build_building_pkl.py --region Hiroshima --verify-only

`--light-only` で軽い pkl だけ先に作れる。**6 GB を書く前に中身を確かめられる。**
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data import load_pkl                                  # noqa: E402
from dc5lib import building, paths                                  # noqa: E402
from dc5lib.paths import dataset_path                               # noqa: E402

SOURCES = {
    "Hiroshima": ("hiroshima_sam_apm.pkl", "hiroshima_bld.pkl",
                  "hiroshima_sam_apm_bld.pkl"),
    "Shimane":   ("shimane_sam_apm.pkl", "shimane_bld.pkl",
                  "shimane_sam_apm_bld.pkl"),
}

# 重い pkl にそのまま引き継ぐ属性
CARRY = ("No", "DEM", "Mask", "GeoInfo", "EPSG", "Max_H", "Min_H", "AirPhoto")
# 軽い pkl に持たせる属性。**GeoInfo は必ず入れる**（あとで照合できるように）
LIGHT = ("No", "GeoInfo", "EPSG")

NOTE = ("基盤地図情報 建築物(BldA)。値は画素に占める面積の割合×255"
        "（16倍スーパーサンプル）。BldL は同じ建物の外周線なので使っていない。"
        "lfSpanFr と devDate はデータの供給日で、建築年ではない。"
        "PIL の多角形塗りが輪郭を外側に丸めるため、中心サンプリングと比べて"
        "被覆率が約 +2.3% 過大に出る。")


def load_index(region):
    f = paths.cache_dir("building") / f"building_index__{region.lower()}.npz"
    if not f.exists():
        sys.exit(f"{f} がありません。先に tools/dump_building_index.py を実行")
    z = np.load(f, allow_pickle=False)
    ver = int(z["version"])
    if ver != building.INDEX_VERSION:
        sys.exit(f"索引の版が違います（npz {ver} / コード "
                 f"{building.INDEX_VERSION}）。作り直してください")
    order = [str(s) for s in z["kind_order"]]
    if order != list(building.KIND_ORDER):
        sys.exit(f"面の並びが違います。npz {order} / "
                 f"コード {list(building.KIND_ORDER)}")
    pack = {k: z[k] for k in ("verts", "starts", "bbox", "kind", "tcode", "lvl",
                              "hole_verts", "hole_starts", "hole_bbox")}
    meshes = {str(s) for s in z["meshes"]}
    return pack, meshes, f


def tile_grid(geo):
    """全タイルの GeoInfo から格子の左上と1タイルの大きさを求める。

    タイルは等間隔に並ぶので、座標の差の最頻値を1歩とみなす
    （`run_block_split.grid` と同じ考え方）。
    """
    lon = np.array([float(g[0]) for g in geo])
    lat = np.array([float(g[3]) for g in geo])
    ul = np.unique(lon)
    ua = np.unique(lat)
    dlon_t = float(np.median(np.diff(ul))) if len(ul) > 1 else 1.0
    dlat_t = float(np.median(np.diff(ua))) if len(ua) > 1 else 1.0
    lon0, lat1 = float(ul[0]), float(ua[-1])
    nx = int(round((float(ul[-1]) - lon0) / dlon_t)) + 2
    ny = int(round((lat1 - float(ua[0])) / dlat_t)) + 2
    return lon0, lat1, dlon_t, dlat_t, nx, ny


def rasterize(ds, pack, meshes):
    n = len(ds.No)
    geo = ds.GeoInfo
    lon0, lat1, dlon_t, dlat_t, nx, ny = tile_grid(geo)
    print(f"  格子 左上 ({lon0:.5f}, {lat1:.5f})  1タイル "
          f"{dlon_t:.6f}×{dlat_t:.6f}度  枡 {nx}×{ny}")
    idx = building.build_index(pack, lon0, lat1, dlon_t, dlat_t, nx, ny)
    print(f"  索引 {len(idx['cell']):,} 組（1棟が複数タイルに跨がる分を含む）")

    # 画素の大きさを一応 m で出す。**非正方**なので両方書く
    g0 = geo[0]
    lat_c = float(g0[3])
    mx = abs(float(g0[1])) * 111320 * np.cos(np.radians(lat_c))
    my = abs(float(g0[5])) * 110570
    print(f"  1画素 {mx:.2f} m（東西）× {my:.2f} m（南北）"
          f"  → {building.SUPER}倍で {mx / building.SUPER:.2f}×"
          f"{my / building.SUPER:.2f} m")

    # --- 焼き込む前に索引を検査する。**ここが狂うと静かに間違う** ---
    rng = np.random.default_rng(0)
    probe = rng.choice(n, min(60, n), replace=False)
    bad = []
    for k in probe:
        a, b_ = building.check_index(geo[k], pack, idx)
        if a != b_:
            bad.append((int(ds.No[k]), a, b_))
    if bad:
        print(f"  **索引が狂っています。** {len(bad)}/{len(probe)} 枚で不一致")
        for no, a, b_ in bad[:5]:
            print(f"    タイル {no}: 索引 {a} 棟 / 総なめ {b_} 棟")
        sys.exit("中止します。dc5lib/building.py の枡の計算を直してください")
    print(f"  索引の検査 {len(probe)}/{len(probe)} 枚で総なめと一致")

    B = np.zeros((n, len(building.KIND_ORDER), building.TILE, building.TILE),
                 np.uint8)
    C = np.zeros((n, len(building.TYPE_ORDER)), np.int32)
    Co = np.zeros(n, np.int32)
    OK = np.zeros(n, np.uint8)
    t0 = time.time()
    for k in range(n):
        g = geo[k]
        B[k] = building.tile_planes(g, pack, idx)
        C[k], c = building.tile_stats(g, pack, idx)
        Co[k] = c
        OK[k] = building.mesh_ok(g, meshes)
        if k % 2000 == 0:
            el = time.time() - t0
            eta = el / max(k, 1) * (n - k)
            print(f"    {k:,}/{n:,}  {el:.0f}秒  残り {eta / 60:.1f}分",
                  end="\r", flush=True)
    print(" " * 60, end="\r")
    print(f"  焼き込み完了 {time.time() - t0:.0f}秒")
    print(f"  棟数 合計 {int(C.sum()):,}（GML の BldA "
          f"{len(pack['kind']):,} に対して "
          f"{100 * C.sum() / max(len(pack['kind']), 1):.1f}%）")
    for i, t in enumerate(building.TYPE_ORDER):
        print(f"    {t:<12}{int(C[:, i].sum()):>10,}")
    print(f"  1/25000 由来 {int(Co.sum()):,}"
          f"（{100 * Co.sum() / max(C.sum(), 1):.1f}%）")
    print(f"  GML が隅の画素まで覆うタイル {int(OK.sum()):,}/{n:,}"
          f"（{100 * OK.mean():.2f}%）")
    for i, nm in enumerate(building.KIND_ORDER):
        nz = int((B[:, i] > 0).sum())
        print(f"  面[{i}] {nm:<8}建物のある画素 {nz:,}"
              f"（{100 * nz / B[:, i].size:.2f}%）"
              f"  平均被覆率 {B[:, i].mean() / 255:.4f}")
    return B, C, Co, OK


def attach(obj, B, C, Co, OK, source, src_name):
    obj.Building = B
    obj.BuildingOrder = list(building.KIND_ORDER)
    obj.BuildingTypeOrder = list(building.TYPE_ORDER)
    obj.BuildingCount = C
    obj.BuildingCoarse = Co
    obj.BuildingMeshOK = OK
    obj.BuildingSource = source
    obj.BuildingNote = NOTE
    obj.BuildingFrom = src_name
    obj.BuildingSuper = building.SUPER
    return obj


def dump(obj, out):
    print(f"  書き出し: {out.name} …", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(out)
    print(f"  完了: {out}（{out.stat().st_size / 2**30:.2f} GB）")


def build(region, light_only=False, overwrite=False):
    src_name, light_name, full_name = SOURCES[region]
    src = dataset_path(region, src_name)
    light = dataset_path(region, light_name)
    full = dataset_path(region, full_name)
    if not src.exists():
        sys.exit(f"{src} がありません")
    if light.exists() and not overwrite and (light_only or full.exists()):
        print(f"  {light.name} はすでにあります（--overwrite で作り直す）")
        return

    # 軽い方が既にあるなら、焼き込みをやり直さずそこから読む。
    # **16分の節約になるうえ、2つの pkl の中身が必ず同一になる。**
    reuse = light.exists() and not light_only
    if reuse:
        print(f"  {light.name} から Building を読み直します"
              f"（焼き込みをやり直さない）", flush=True)
        lp = load_pkl(light)
        B = np.asarray(lp.Building)
        C = np.asarray(lp.BuildingCount)
        Co = np.asarray(lp.BuildingCoarse)
        OK = np.asarray(lp.BuildingMeshOK)
        source = str(getattr(lp, "BuildingSource", "?"))
        sup = int(getattr(lp, "BuildingSuper", building.SUPER))
        if sup != building.SUPER:
            sys.exit(f"{light.name} は {sup}倍で焼いてあり、いまのコードは "
                     f"{building.SUPER}倍です。--overwrite で作り直してください")
        lno = np.asarray(lp.No)
        del lp
        gc.collect()
        print(f"  読み込み: {src.name}（{src.stat().st_size / 2**30:.1f} GB）…",
              flush=True)
        ds = load_pkl(src)
        if not (len(lno) == len(ds.No) and (lno == np.asarray(ds.No)).all()):
            sys.exit(f"{light.name} と {src.name} でタイル番号が違います")
        print(f"  {len(ds.No):,} タイル  タイル番号の一致を確認")
    else:
        pack, meshes, idxf = load_index(region)
        print(f"  索引: {idxf.name}  BldA {len(pack['kind']):,} 棟")
        print(f"  読み込み: {src.name}（{src.stat().st_size / 2**30:.1f} GB）…",
              flush=True)
        ds = load_pkl(src)
        print(f"  {len(ds.No):,} タイル")

        B, C, Co, OK = rasterize(ds, pack, meshes)
        source = f"基盤地図情報 BldA / 2次メッシュ {len(meshes)} 枚"
        del pack
        gc.collect()

    # 軽い方を先に書く。**6 GB を書く前に中身を確かめられる**
    if not reuse:
        lo = SimpleNamespace(**{k: getattr(ds, k)
                                for k in LIGHT if hasattr(ds, k)})
        dump(attach(lo, B, C, Co, OK, source, src_name), light)
        del lo
        gc.collect()

    if light_only:
        print("  --light-only なので重い pkl は作りません")
        del ds, B
        gc.collect()
        return

    hi = SimpleNamespace(**{k: getattr(ds, k) for k in CARRY if hasattr(ds, k)})
    dump(attach(hi, B, C, Co, OK, source, src_name), full)
    del hi, ds, B
    gc.collect()


def verify(region):
    """書いた pkl を読み直して、元の pkl とタイル番号が一致するかを確かめる。

    **ここが本題。** 番号がずれていたら他は全部無意味になる。
    """
    src_name, light_name, full_name = SOURCES[region]
    src = dataset_path(region, src_name)
    for name in (light_name, full_name):
        out = dataset_path(region, name)
        if not out.exists():
            print(f"  {name}: ありません")
            continue
        ds = load_pkl(out)
        B = np.asarray(ds.Building)
        print(f"\n  {name}: {len(ds.No):,} タイル  Building {B.shape} "
              f"{B.dtype}")
        print(f"    BuildingOrder = {list(getattr(ds, 'BuildingOrder', []))}")
        print(f"    BuildingTypeOrder = "
              f"{list(getattr(ds, 'BuildingTypeOrder', []))}")
        print(f"    BuildingSource = {getattr(ds, 'BuildingSource', '?')}")
        no = np.asarray(ds.No)
        geo = ds.GeoInfo
        del ds
        gc.collect()

        print(f"    読み込み（照合用）: {src.name} …", flush=True)
        s = load_pkl(src)
        sno = np.asarray(s.No)
        sgeo = s.GeoInfo
        same_no = bool(len(sno) == len(no) and (sno == no).all())
        bad = sum(1 for a, b in zip(geo, sgeo)
                  if not np.allclose(np.asarray(a, float),
                                     np.asarray(b, float)))
        print(f"    No が {src.name} と完全一致: {same_no}")
        print(f"    GeoInfo が一致しないタイル: {bad}")
        if not same_no or bad:
            print("    ※ **ずれています。使ってはいけません**")
        else:
            print("    → SAM・DEM・マスクと画素単位で対応しています")
        del s
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Hiroshima", choices=list(SOURCES))
    ap.add_argument("--light-only", action="store_true",
                    help="軽い pkl だけ作る（6 GB を書く前の確認用）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    print(f"地域 {a.region}")
    if a.verify_only:
        verify(a.region)
        return
    build(a.region, light_only=a.light_only, overwrite=a.overwrite)
    verify(a.region)


if __name__ == "__main__":
    main()
