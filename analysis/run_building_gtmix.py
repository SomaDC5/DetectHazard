# -*- coding: utf-8 -*-
"""島根の落ち込みは「人家の薄い区域が多いから」で説明できるか。**推論なし。**

なぜ推論が要らないか
--------------------
`39.3` で、広島の正解のかたまりの検出率は

    建物が1画素も無い  0.5246（1,262件）
    建物がある        0.8270（7,715件）

と測れた。もし島根の落ち込みがこの偏りで説明できるなら、
**島根は「建物が無い正解」の割合が高い**はずで、そこに広島の検出率を
当てれば島根の再現率が再現できる。

正解のかたまりと建物はどちらもデータ側にあるので、
**モデルを動かさずに割合を数えられる。**

    python analysis/run_building_gtmix.py

出てくる数が島根の実測（箇所R 0.4714、`34.4`）に届かなければ、
**建物だけでは説明しきれない**ということになる。
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.data import load_pkl                                  # noqa: E402
from analysis.run_overlap_eval import prep_mask                     # noqa: E402
from dc5lib.instance_eval import label_instances                    # noqa: E402
from dc5lib.paths import dataset_path                               # noqa: E402

SRC = {"Hiroshima": ("hiroshima_sam_apm.pkl", "hiroshima_bld.pkl"),
       "Shimane": ("shimane_sam_apm.pkl", "shimane_bld.pkl")}

# `39.3` で広島で測った、正解のかたまりの検出率
RATE_NOBLD, RATE_BLD = 0.5246, 0.8270
# 実測の箇所R。**RATE_* と同じ条件・同じ推論の値でなければならない。**
#   どちらも FinalFusion_SAM_APM_OverlapTrain のオーバーラップ推論
#   （広島 holdout 2,964枚 / 島根 全 24,569枚）。
# 最初は島根に `34.4` の 0.4714 を入れていたが、あれは別の条件・従来推論の値で、
# **説明できる割合を 5% と過小に見せてしまった**（正しくは 8.6%）。
ACTUAL_R = {"Hiroshima": 0.7844, "Shimane": 0.5699}

# 地域ごとの実測の層別検出率。**島根でも測ったので、分解できる。**
RATE = {"Hiroshima": {"bld": 0.8270, "no": 0.5246},
        "Shimane": {"bld": 0.6391, "no": 0.2963}}


def gt_blob_mix(region, verbose=True):
    """正解のかたまりごとに、建物があるかを数える。"""
    s, b = SRC[region]
    bl = load_pkl(dataset_path(region, b))
    B = np.asarray(bl.Building)
    bno = np.asarray(bl.No)
    del bl
    gc.collect()
    ds = load_pkl(dataset_path(region, s))
    dno = np.asarray(ds.No)
    if not (len(dno) == len(bno) and (dno == bno).all()):
        sys.exit(f"{region}: タイル番号が違います")

    n_bld, n_no, areas_b, areas_n = 0, 0, [], []
    cov_all = []
    for k in range(len(dno)):
        m = np.asarray(ds.Mask[k])
        if m.max() < 1:
            continue
        gt = prep_mask(ds, k)
        gl, gs = label_instances(gt, 8, 10)
        if not len(gs):
            continue
        bc = B[k, 0] / 255.0
        for i in range(len(gs)):
            sel = gl == (i + 1)
            c = float(bc[sel].mean())
            cov_all.append(c)
            if c > 0:
                n_bld += 1
                areas_b.append(int(gs[i]))
            else:
                n_no += 1
                areas_n.append(int(gs[i]))
        if verbose and k % 3000 == 0:
            print(f"    {k}/{len(dno)}", end="\r", flush=True)
    if verbose:
        print(" " * 30, end="\r")
    del ds, B
    gc.collect()
    return dict(n_bld=n_bld, n_no=n_no,
                area_b=float(np.mean(areas_b)) if areas_b else 0.0,
                area_n=float(np.mean(areas_n)) if areas_n else 0.0,
                cov=np.array(cov_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="*", default=["Hiroshima", "Shimane"])
    a = ap.parse_args()

    res = {}
    for r in a.regions:
        print(f"{r} …", flush=True)
        res[r] = gt_blob_mix(r)

    print()
    print("【正解のかたまりの内訳】")
    print(f"{'':<20}{'件数':>10}{'建物あり':>11}{'建物なし':>11}"
          f"{'建物なしの割合':>15}")
    for r, v in res.items():
        tot = v["n_bld"] + v["n_no"]
        print(f"{r:<20}{tot:>10,}{v['n_bld']:>11,}{v['n_no']:>11,}"
              f"{100 * v['n_no'] / max(tot, 1):>14.1f}%")
    print()
    print(f"{'':<20}{'建物ありの面積':>16}{'建物なしの面積':>16}")
    for r, v in res.items():
        print(f"{r:<20}{v['area_b']:>16.0f}{v['area_n']:>16.0f}")
    print("  （画素数。1画素 約16 m2）")

    print()
    print("【広島の検出率を当てて島根の再現率を予想する】")
    print(f"  使う率: 建物あり {RATE_BLD:.4f} / 建物なし {RATE_NOBLD:.4f}"
          f"（`39.3`）")
    print(f"{'':<20}{'予想 箇所R':>13}{'実測 箇所R':>13}{'差':>11}")
    for r, v in res.items():
        tot = v["n_bld"] + v["n_no"]
        p = (v["n_bld"] * RATE_BLD + v["n_no"] * RATE_NOBLD) / max(tot, 1)
        act = ACTUAL_R.get(r, float("nan"))
        print(f"{r:<20}{p:>13.4f}{act:>13.4f}{p - act:>+11.4f}")
    print()
    print("  **差が小さければ建物の偏りで説明できる。**")
    print("  大きければ、島根の落ち込みには建物以外の原因がある。")

    # --- 構成の差と、層の中の差に分ける ---
    if len(res) >= 2 and all(r in RATE for r in res):
        rs = list(res)
        base, other = rs[0], rs[-1]
        vb, vo = res[base], res[other]
        pb = vb["n_no"] / max(vb["n_bld"] + vb["n_no"], 1)
        po = vo["n_no"] / max(vo["n_bld"] + vo["n_no"], 1)

        def mix(p, rate):
            return p * rate["no"] + (1 - p) * rate["bld"]

        r_base = mix(pb, RATE[base])
        r_mixonly = mix(po, RATE[base])     # 構成だけ入れ替える
        r_other = mix(po, RATE[other])
        print()
        print(f"【差の分解】{base} {r_base:.4f} → {other} {r_other:.4f}"
              f"（{r_other - r_base:+.4f}）")
        tot = r_other - r_base
        nm1 = (f"構成の差（建物なしの割合 "
               f"{100 * pb:.1f}% → {100 * po:.1f}%）")
        nm2 = "層の中の差（同じ種類の区域を見つけられない）"
        print(f"{nm1:<44}{r_mixonly - r_base:>+9.4f}"
              f"{100 * (r_mixonly - r_base) / tot:>7.1f}%")
        print(f"{nm2:<44}{r_other - r_mixonly:>+9.4f}"
              f"{100 * (r_other - r_mixonly) / tot:>7.1f}%")
        print()
        print(f"  層別の検出率  建物あり {RATE[base]['bld']:.4f} → "
              f"{RATE[other]['bld']:.4f}（{RATE[other]['bld'] - RATE[base]['bld']:+.4f}）")
        print(f"               建物なし {RATE[base]['no']:.4f} → "
              f"{RATE[other]['no']:.4f}（{RATE[other]['no'] - RATE[base]['no']:+.4f}）")
        print("  **両方の層でほぼ同じだけ落ちているなら、"
              "構成の問題ではなく県の違いである。**")

    print()
    print("【建物被覆率の分布（正解のかたまり）】")
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"{'':<20}" + "".join(f"{str(int(100 * q)) + '%':>10}" for q in qs))
    for r, v in res.items():
        c = v["cov"]
        print(f"{r:<20}" + "".join(f"{np.quantile(c, q):>10.4f}" for q in qs))


if __name__ == "__main__":
    main()
