# -*- coding: utf-8 -*-
"""仮想建物をタイルに焼き込んで pkl にする。

**タイル番号は元の pkl の `No` をそのまま使う。**
`build_building_pkl.py` と同じ作法。置き方は `dc5lib/vbuilding.py` の冒頭。

作るもの
--------
    hiroshima_vbld.pkl            約 0.6 GB。仮想建物だけ。解析・突き合わせ用
    hiroshima_sam_apm_bldvb.pkl   約 7 GB。SAM+APM に実建物＋仮想建物。学習用

足す属性
--------
    VBuilding        uint8 [n, 1, 128, 128]  仮想建物の被覆率 × 255
    VBuildingCount   int32 [n]               そのタイルに置いた戸数
    VBuildingParams  dict                    しきい値（あとで再現できるように）
    VBuildingNote / VBuildingFrom

重い pkl には**3面をまとめた**ものを入れる（入力は1つの属性で渡すので）。

    BuildingVB       uint8 [n, 3, 128, 128]
                       [0] 建物（実）
                       [1] 無壁舎（実）
                       [2] 仮想建物
    BuildingVBOrder  ("建物", "無壁舎", "仮想建物")

読むもの
--------
    <pre>_sam.pkl           DEM = **傾斜角（度）**
    <pre>_dem.pkl           DEM = **標高（m）**。同じ格子であることを確かめる
    <pre>_bld.pkl           Building（実建物）
    <pre>_sam_apm_bld.pkl    重い pkl の素。2段目で読む

使い方
------
    python tools/build_vbuilding_pkl.py --region Hiroshima --sample 2000
    python tools/build_vbuilding_pkl.py --region Hiroshima --light-only
    python tools/build_vbuilding_pkl.py --region Hiroshima
    python tools/build_vbuilding_pkl.py --region Hiroshima --verify-only

`--sample N` は**何も書かずに密度だけ出す。** 7 GB を書く前に確かめられる。
`--light-only` で軽い pkl だけ作る。
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
from dc5lib import vbuilding as vb                                  # noqa: E402
from dc5lib.paths import dataset_path                               # noqa: E402

# (傾斜, 標高, 実建物, 重い素, 軽い出力, 重い出力)
SOURCES = {
    "Hiroshima": ("hiroshima_sam.pkl", "hiroshima_dem.pkl",
                  "hiroshima_bld.pkl", "hiroshima_sam_apm_bld.pkl",
                  "hiroshima_vbld.pkl", "hiroshima_sam_apm_bldvb.pkl"),
    "Shimane":   ("shimane_sam.pkl", "shimane_dem.pkl",
                  "shimane_bld.pkl", "shimane_sam_apm_bld.pkl",
                  "shimane_vbld.pkl", "shimane_sam_apm_bldvb.pkl"),
}

# 重い pkl にそのまま引き継ぐ属性
CARRY = ("No", "DEM", "Mask", "GeoInfo", "EPSG", "Max_H", "Min_H", "AirPhoto",
         "Building", "BuildingOrder", "BuildingTypeOrder", "BuildingCount",
         "BuildingCoarse", "BuildingMeshOK", "BuildingSource", "BuildingNote",
         "BuildingFrom", "BuildingSuper")

VB_ORDER = ("建物", "無壁舎", "仮想建物")


def _params():
    return dict(slope_deg=vb.SLOPE_DEG, min_height_m=vb.MIN_HEIGHT_M,
                min_steep_px=vb.MIN_STEEP_PX, foot_min_m=vb.FOOT_MIN_M,
                foot_max_m=vb.FOOT_MAX_M, sparse_radius_m=vb.SPARSE_RADIUS_M,
                sparse_lo=vb.SPARSE_LO, sparse_hi=vb.SPARSE_HI,
                spacing_m=vb.SPACING_M, house_m=vb.HOUSE_M, px_m=vb.PX_M)


def _read_field(path, field, dtype, label):
    """pkl を1つ読んで1つの属性だけ取り出し、すぐ捨てる。

    **峰を抑えるためにこうしている。** 3つ同時に抱えると 10 GB を超える。
    `No` も一緒に返して、呼ぶ側で並びを突き合わせる。
    """
    gb = Path(path).stat().st_size / 1024 ** 3
    print(f"  読み込み: {Path(path).name} ({gb:.1f} GB) … ", end="", flush=True)
    t = time.time()
    ds = load_pkl(path)
    no = np.asarray(ds.No).copy()
    src = getattr(ds, field)
    out = [np.asarray(src[i], dtype) for i in range(len(no))]
    extra = {k: getattr(ds, k) for k in ("BuildingOrder",) if hasattr(ds, k)}
    del ds, src
    gc.collect()
    print(f"{time.time() - t:.0f}秒 / {label} {len(out):,} 枚", flush=True)
    return no, out, extra


def make_light(region, sample=0):
    """仮想建物の面と戸数を作る。`sample>0` なら書かずに密度だけ出す。"""
    sam, dem, bld, _, out_light, _ = SOURCES[region]
    no_s, slope, _ = _read_field(dataset_path(region, sam), "DEM",
                                 np.float32, "傾斜")
    no_d, elev, _ = _read_field(dataset_path(region, dem), "DEM",
                                np.float32, "標高")
    if not (len(no_d) == len(no_s) and (no_d == no_s).all()):
        sys.exit(f"{dem} のタイルの並びが {sam} と違います。"
                 f"仮想建物は格子が揃っていないと意味がありません")
    no_b, real, ex = _read_field(dataset_path(region, bld), "Building",
                                 np.uint8, "実建物")
    if not (len(no_b) == len(no_s) and (no_b == no_s).all()):
        sys.exit(f"{bld} のタイルの並びが {sam} と違います")
    order = list(ex.get("BuildingOrder", []))
    if order and order != list(vb_kind_order()):
        sys.exit(f"{bld} の BuildingOrder が {order} です。"
                 f"コードは {list(vb_kind_order())} を前提にしています")

    n = len(no_s)
    idx = (np.arange(0, n, max(1, n // sample)) if sample else np.arange(n))
    print(f"\n  仮想建物を置きます（{len(idx):,} 枚）…", flush=True)
    V = np.zeros((len(idx), 1, vb.TILE, vb.TILE), np.uint8)
    C = np.zeros(len(idx), np.int32)
    band = np.zeros(len(idx), np.int32)
    t = time.time()
    for k, i in enumerate(idx.tolist()):
        plane, p = vb.tile_virtual(slope[i], elev[i], real[i],
                                   return_parts=True)
        V[k, 0] = plane
        C[k] = p["n_house"]
        band[k] = p["band_px"]
        if (k + 1) % 5000 == 0:
            el = time.time() - t
            print(f"    {k + 1:,}/{len(idx):,}  {el:.0f}秒  "
                  f"（残り約 {el / (k + 1) * (len(idx) - k - 1):.0f}秒）",
                  flush=True)
    print(f"  置き終わり {time.time() - t:.0f}秒", flush=True)

    print(f"\n  戸数 合計 {C.sum():,} / 1枚あたり {C.mean():.1f}")
    print(f"  1戸以上あるタイル {100 * (C > 0).mean():.1f}%"
          f"（麓の帯があるタイル {100 * (band > 0).mean():.1f}%）")
    print(f"  戸数の分布 50%点 {np.percentile(C, 50):.0f} / "
          f"90%点 {np.percentile(C, 90):.0f} / 最大 {C.max()}")
    cov = V.astype(np.float64).mean() / 255.0
    print(f"  面の平均被覆率 {cov:.5f}（画素の {100 * (V > 0).mean():.2f}% が >0）")

    if sample:
        print("\n  --sample なので何も書きません。")
        return None

    obj = SimpleNamespace()
    obj.No = no_s
    obj.VBuilding = V
    obj.VBuildingCount = C
    obj.VBuildingBandPx = band
    obj.VBuildingParams = _params()
    obj.VBuildingNote = vb.NOTE
    obj.VBuildingFrom = f"{sam} + {dem} + {bld}"
    p = dataset_path(region, out_light)
    print(f"\n  書き出し: {p} …", flush=True)
    t = time.time()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  {p.stat().st_size / 1024 ** 3:.2f} GB / {time.time() - t:.0f}秒")
    return p


def vb_kind_order():
    from dc5lib import building
    return building.KIND_ORDER


def make_heavy(region):
    """軽い pkl の面を読み直して、学習用の重い pkl を書く。"""
    _, _, _, src, out_light, out_heavy = SOURCES[region]
    lp = load_pkl(dataset_path(region, out_light))
    V = np.asarray(lp.VBuilding)
    C = np.asarray(lp.VBuildingCount)
    no_l = np.asarray(lp.No)
    note, prm = str(lp.VBuildingNote), dict(lp.VBuildingParams)
    frm = str(lp.VBuildingFrom)
    del lp
    gc.collect()

    sp = dataset_path(region, src)
    print(f"  読み込み: {sp.name} "
          f"({sp.stat().st_size / 1024 ** 3:.1f} GB) …", flush=True)
    t = time.time()
    ds = load_pkl(sp)
    print(f"  {time.time() - t:.0f}秒", flush=True)
    no = np.asarray(ds.No)
    if not (len(no) == len(no_l) and (no == no_l).all()):
        sys.exit(f"{src} のタイルの並びが {out_light} と違います")
    if not hasattr(ds, "Building"):
        sys.exit(f"{src} に Building がありません。"
                 f"tools/build_building_pkl.py で作ったものを指定してください")

    obj = SimpleNamespace()
    miss = [k for k in CARRY if not hasattr(ds, k)]
    for k in CARRY:
        if hasattr(ds, k):
            setattr(obj, k, getattr(ds, k))
    if miss:
        print(f"  （{src} に無くて引き継がなかった属性: {miss}）")

    B = np.asarray(ds.Building)
    if B.shape[1] != 2:
        sys.exit(f"Building の面が {B.shape[1]} です。2 を前提にしています")
    print(f"  3面にまとめます {B.shape} + {V.shape} …", flush=True)
    obj.BuildingVB = np.concatenate([B, V], axis=1)
    obj.BuildingVBOrder = list(VB_ORDER)
    obj.VBuilding = V
    obj.VBuildingCount = C
    obj.VBuildingParams = prm
    obj.VBuildingNote = note
    obj.VBuildingFrom = frm
    del ds, B
    gc.collect()

    p = dataset_path(region, out_heavy)
    print(f"  書き出し: {p} …", flush=True)
    t = time.time()
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  {p.stat().st_size / 1024 ** 3:.2f} GB / {time.time() - t:.0f}秒")
    return p


def verify(region):
    _, _, _, _, out_light, out_heavy = SOURCES[region]
    for name in (out_light, out_heavy):
        p = dataset_path(region, name)
        if not p.exists():
            print(f"  {name}: ありません")
            continue
        ds = load_pkl(p)
        n = len(np.asarray(ds.No))
        V = np.asarray(ds.VBuilding)
        C = np.asarray(ds.VBuildingCount)
        print(f"\n  {name}: {n:,} タイル  VBuilding {V.shape} {V.dtype}")
        print(f"    戸数 合計 {C.sum():,} / 1枚あたり {C.mean():.1f} / "
              f"1戸以上 {100 * (C > 0).mean():.1f}%")
        print(f"    しきい値 {ds.VBuildingParams}")
        if hasattr(ds, "BuildingVB"):
            A = np.asarray(ds.BuildingVB)
            print(f"    BuildingVB {A.shape} 並び {list(ds.BuildingVBOrder)}")
            same = np.array_equal(A[:, 2:3], V)
            print(f"    3面目が VBuilding と一致: {same}")
            if not same:
                sys.exit("    BuildingVB の3面目が VBuilding と違います")
            for c, nm in enumerate(ds.BuildingVBOrder):
                print(f"      [{c}] {nm}  平均被覆率 "
                      f"{A[:, c].astype(np.float64).mean() / 255:.5f} / "
                      f">0 の画素 {100 * (A[:, c] > 0).mean():.2f}%")
        del ds
        gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Hiroshima",
                    choices=list(SOURCES))
    ap.add_argument("--sample", type=int, default=0,
                    help="この枚数だけ試して密度を出す。何も書かない")
    ap.add_argument("--light-only", action="store_true")
    ap.add_argument("--heavy-only", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"地域 {a.region}")
    print(f"しきい値 {_params()}\n")
    if a.verify_only:
        verify(a.region)
        return
    if not a.heavy_only:
        if make_light(a.region, a.sample) is None:
            return
    if not a.light_only:
        print()
        make_heavy(a.region)
    print()
    verify(a.region)


if __name__ == "__main__":
    main()
