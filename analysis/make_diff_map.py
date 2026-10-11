# -*- coding: utf-8 -*-
"""ハザードマップと正解の差分を1枚のラスタにする（QGIS で色分けして見る用）。

run_hazard_map.py が出した <名前>__<地域>.tif（バンド2が正解）と
<名前>__<地域>__mask.tif（二値化した予測）を突き合わせ、画素ごとに

    0  両方なし（TN）     … 透明にする
    1  見逃し（FN）       … 正解にあるのに出していない
    2  誤検出（FP）       … 出しているが正解にない
    3  一致（TP）
    255 範囲外（nodata）

を書き出す。あわせて QGIS の配色ファイル（.qml）を同じ名前で置くので、
.tif をドラッグするだけで色が付く。

    python analysis/make_diff_map.py Ens_InputDiverse4__shimane

Windows では rasterio が mayo に入らない（Smart App Control が pip 版の DLL を
弾く。docs/ハザードマップの作り方.md 9.2）。rasterio のある環境で実行すること。

    "C:/Users/hirok/anaconda3/envs/geopack/python.exe" analysis/make_diff_map.py ...

このスクリプトは torch を使わないので、推論用の環境でなくてよい。
"""
import argparse
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODATA = 255
LABELS = [(0, "両方なし", "#000000", 0),
          (1, "見逃し FN", "#d7191c", 255),
          (2, "誤検出 FP", "#fdae61", 255),
          (3, "一致 TP", "#1a9641", 255)]

QML = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <rasterTransparency/>
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""


def write_qml(path):
    rows = "\n".join(
        f'        <paletteEntry value="{v}" color="{c}" alpha="{a}" label="{lab}"/>'
        for v, lab, c, a in LABELS)
    with open(path, "w", encoding="utf-8") as f:
        f.write(QML.format(entries=rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="<名前>__<地域>（拡張子なし）")
    ap.add_argument("--dir", default=None, help="既定は dc5-data/hazard_map")
    args = ap.parse_args()

    d = args.dir
    if d is None:
        from dc5lib.paths import data_root
        d = os.path.join(str(data_root()), "hazard_map")
    base = os.path.join(d, args.stem)
    f_main, f_mask = base + ".tif", base + "__mask.tif"
    for p in (f_main, f_mask):
        if not os.path.exists(p):
            sys.exit(f"{p} がありません。先に run_hazard_map.py を実行してください。")

    out = base + "__diff.tif"
    with rasterio.open(f_main) as src, rasterio.open(f_mask) as msk:
        prof = src.profile.copy()
        prof.update(count=1, dtype="uint8", nodata=NODATA,
                    compress="deflate", tiled=True, predictor=1)
        n = dict(tp=0, fp=0, fn=0, tn=0, out=0)
        with rasterio.open(out, "w", **prof) as dst:
            for _, win in src.block_windows(1):
                gt = src.read(2, window=win)
                pr = msk.read(1, window=win)
                valid = (gt != NODATA) & (pr != NODATA)
                g = (gt == 1) & valid
                p = (pr == 1) & valid
                a = np.full(gt.shape, NODATA, np.uint8)
                a[valid] = 0
                a[valid & ~g & p] = 2
                a[valid & g & ~p] = 1
                a[valid & g & p] = 3
                dst.write(a, 1, window=win)
                n["tp"] += int((g & p).sum())
                n["fp"] += int((~g & p & valid).sum())
                n["fn"] += int((g & ~p & valid).sum())
                n["tn"] += int((~g & ~p & valid).sum())
                n["out"] += int((~valid).sum())
        with rasterio.open(out, "r+") as dst:
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.nearest)

    write_qml(base + "__diff.qml")

    R = n["tp"] / max(n["tp"] + n["fn"], 1)
    P = n["tp"] / max(n["tp"] + n["fp"], 1)
    F = 2 * R * P / (R + P) if R + P else 0.0
    print(f"書き出し: {out}")
    print(f"配色    : {base}__diff.qml（.tif と同じ名前なので自動で当たる）")
    print(f"\n  有効画素 {n['tp']+n['fp']+n['fn']+n['tn']:,} / 範囲外 {n['out']:,}")
    print(f"  一致 TP   {n['tp']:>12,}")
    print(f"  見逃し FN {n['fn']:>12,}")
    print(f"  誤検出 FP {n['fp']:>12,}")
    print(f"\n  面積F {F:.4f} / R {R:.4f} / P {P:.4f}")
    print("  ※ この値が metrics.csv と一致するのは --no-overlap で作った地図のみ")


if __name__ == "__main__":
    main()
