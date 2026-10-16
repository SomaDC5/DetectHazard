# -*- coding: utf-8 -*-
"""土地利用メッシュがタイルと正しく重なっているかを確かめ、同時に効くかを見る。

位置合わせの検査と仮説の検査を同じ計算で済ませる
----------------------------------------------------
急傾斜地崩壊危険区域の指定基準は**保全対象（人家）の有無**を含む。
したがって、位置合わせが正しければ

    危険区域のタイルは、背景タイルより「建物用地」が多い

はずである。逆に測地系や格子の取り違えがあれば、この差は消える。
**だから片方の検査がもう片方の検査になっている。**

失敗の所在.md は TP=0 タイルの 60% を「正解の方が緩斜面 = 入力とラベルの矛盾」
として「解けない。入力の追加が要る」と判定した。その追加入力がこれ。

使い方
------
    # まず geopack で npz を作る
    tools/dump_landuse_grids.py --folder <shpのフォルダ> --out <SSD>/landuse_grids.npz
    # こちらは mayo で動かす
    python analysis/run_landuse_check.py --grids <SSD>/landuse_grids.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data import load_pkl                                  # noqa: E402
from dc5lib import landuse                                          # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402
from dc5lib.registry import get                                     # noqa: E402

NAMES = {c: landuse.LANDUSE_CLASSES.get(c, c) for c in landuse.CHANNEL_ORDER}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", required=True,
                    help="tools/dump_landuse_grids.py が書いた npz")
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--regions", nargs="*", default=["Hiroshima", "Shimane"])
    ap.add_argument("--sample", type=int, default=4000,
                    help="各群から何タイル見るか（0で全件）")
    args = ap.parse_args()

    print("--- 土地利用メッシュを読む ---")
    grids = landuse.load_grids_npz(args.grids)
    print(f"  1次メッシュ {len(grids)} 枚: {sorted(grids)}")

    cond = get(args.condition)
    for region in args.regions:
        ds = load_pkl(dataset_path(region, cond.pkl_for(region)))
        n = len(ds.No)
        has_gt = np.array([np.max(np.asarray(ds.Mask[k])) >= 1 for k in range(n)])
        print(f"\n=== {region}: {n:,} タイル"
              f"（危険区域あり {int(has_gt.sum()):,} / 背景 {int((~has_gt).sum()):,}）===")

        rng = np.random.default_rng(42)
        groups = {}
        for label, mask in (("危険区域あり", has_gt), ("背景", ~has_gt)):
            idx = np.nonzero(mask)[0]
            if args.sample and len(idx) > args.sample:
                idx = rng.choice(idx, args.sample, replace=False)
            groups[label] = idx

        # 区分ごとの画素割合と、未整備の割合
        res = {}
        for label, idx in groups.items():
            acc = np.zeros(len(landuse.CHANNEL_ORDER))
            unknown = 0.0
            for k in idx:
                ch, cls = landuse.tile_channels_grid(ds.GeoInfo[k], grids)
                acc += ch.sum(axis=(1, 2))
                unknown += float((cls == landuse.UNKNOWN).sum())
            tot = len(idx) * landuse.TILE ** 2
            res[label] = (acc / tot, unknown / tot)

        print("%-14s %12s %12s %10s" % ("区分", "危険区域あり", "背景", "差"))
        a, ua = res["危険区域あり"]
        b, ub = res["背景"]
        for i, c in enumerate(landuse.CHANNEL_ORDER):
            print("%-14s %11.2f%% %11.2f%% %+9.2f%%"
                  % (NAMES[c], 100 * a[i], 100 * b[i], 100 * (a[i] - b[i])))
        print("%-14s %11.2f%% %11.2f%% %+9.2f%%"
              % ("（未整備）", 100 * ua, 100 * ub, 100 * (ua - ub)))

        # 位置合わせの判定。建物用地の差が出ていなければ疑う
        d = a[0] - b[0]
        print()
        if ua > 0.5:
            print(f"  ※ 未整備が {100 * ua:.1f}% と多い。"
                  f"メッシュが足りているか確認すること")
        if d > 0.02:
            print(f"  位置合わせ: **妥当**。建物用地が危険区域タイルで "
                  f"{100 * d:+.2f}% 多い")
        elif d < -0.02:
            print(f"  位置合わせ: 逆向き（{100 * d:+.2f}%）。"
                  f"測地系や行列の向きを疑うこと")
        else:
            print(f"  位置合わせ: 差が小さい（{100 * d:+.2f}%）。"
                  f"取り違えか、そもそも効かないかの切り分けが必要")


if __name__ == "__main__":
    main()
