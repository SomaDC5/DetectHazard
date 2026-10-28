# -*- coding: utf-8 -*-
"""OverlapDataset の third が面の数に依らず通るか。

**2面で落ちていた**（_to_hwc が shape[0] in (1,3) 決め打ちだった）。
面の数を 1〜9 で回して、形と「地形と同じ窓から来ているか」を確かめる。
"""
import sys
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\hirok\dc5\DetectHazard")
from dc5lib.overlap import OverlapDataset, allowed_except, _to_hwc
from analysis.run_overlap_eval import DLON, DLAT

T = 128
print("_to_hwc の判定")
for shp, want in (((T, T), "hw"), ((1, T, T), "chw"), ((2, T, T), "chw"),
                  ((3, T, T), "chw"), ((9, T, T), "chw"),
                  ((T, T, 3), "hwc"), ((T, T, 1), "hwc"),
                  ((256, 256, 3), "hwc"), ((3, 256, 256), "chw")):
    _, kind = _to_hwc(np.zeros(shp, np.uint8))
    ok = "OK " if kind == want else "NG "
    print("  %s %-16s -> %-4s（%s のはず）" % (ok, shp, kind, want))
    assert kind == want, shp

geo, dem, air, msk = [], [], [], []
for j in range(3):
    for i in range(3):
        geo.append((130.0 + i * DLON, DLON / T, 0.0,
                    34.0 + j * DLAT, 0.0, -DLAT / T))
        v = 10 * j + i
        dem.append(np.full((1, T, T), float(v), np.float32))
        air.append(np.full((3, T, T), v, np.uint8))
        msk.append(np.zeros((1, T, T), np.uint8))
allowed = allowed_except(9)

print("\n面の数ごとに OverlapDataset を回す")
for C in (1, 2, 3, 4, 9):
    third = [np.full((C, T, T), (10 * (k // 3) + k % 3) * 10, np.uint8)
             for k in range(9)]
    bad = 0
    for s in range(12):
        d = OverlapDataset(dem, air, msk, geo, indices=[4], allowed=allowed,
                           max_shift=64, seed=s, third=third)
        (a, _, t), _ = d[0]
        assert tuple(t.shape) == (C, T, T), (C, tuple(t.shape))
        # dem の値 v に対し third は v*10/255
        if not torch.allclose(t[0], a[0] * 10.0 / 255.0, atol=2e-3):
            bad += 1
    print("  %d 面  形 %s  窓のずれ %d/12" % (C, (C, T, T), bad))
    assert bad == 0, C
print("\n  面の数に依らず通った")
