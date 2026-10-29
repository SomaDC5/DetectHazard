# -*- coding: utf-8 -*-
"""既存の `union_blind__<地域>.csv` に**構成員の列だけを足す**。

なぜ要るか
----------
`union_blind` の CSV は「正解のかたまり1件ごと」1行で、**正解マスクが
全構成員で同一であることを前提に付番を使い回す**つくりである
（`run_union_blind.py` の `blob_cov`）。だから構成員を1本足すのに
全員を推論し直す必要はない ── 足す1本だけ推論して列を継ぎ足せばよい。

5本で回すと広島+島根で1時間以上かかる。1本なら十数分である。

**行の並びが合っている証明を取る**
--------------------------------
行数が合うだけでは並びの保証にならない。だから**記録済みの構成員を1本
照合用に推論し直し、保存済みの列と一致するかを確かめる**。
一致しなければ**止める**（タイルの並びか正解の作り方が変わっている）。

この方法で足せないもの（**正直に書く**）
--------------------------------------
足せるのは `cov_<略称>` と `lg_<略称>` だけである。

    **`cov_union` は足せない。** 二値マスクの論理和の覆いは、構成員ごとの
    覆いの最大より大きくなりうる（`44.7`）。本当の和集合の箇所R/P/F が
    要るなら `run_union_blind.py` を全構成員で回すしかない。

だから出力の `cov_union` / `n_light` は**元の構成員のままの値**であり、
足した構成員は入っていない。`run_union_subsets.py` は `cov_*` しか
見ないので、**床と Jaccard とロジットはこの CSV で正しく出る。**

    python analysis/run_union_addmember.py --region both --tag seed2 \
        --new FinalFusion_SAM_APM_BLD_seed2_OverlapTrain
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

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
import torch                                                      # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_overlap_prune import infer, score               # noqa: E402
from analysis.run_union_blind import blob_cov, short              # noqa: E402
from dc5lib import paths                                         # noqa: E402
from dc5lib.device import pick_device                             # noqa: E402
from dc5lib.registry import get                                   # noqa: E402

COV_GT = 0.5                      # `run_union_blind.py` と同じ
# 照合に使う記録済みの構成員。**この1本を出し直して列が一致するかを見る**
VERIFY = "FinalFusion_SAM_APM_BLD_OverlapTrain"
TOL = 1e-6                        # 覆い率の許容差（GPU の演算順で下位桁は動く）


def load_src(region, tag):
    name = f"union_blind__{region}.csv" if not tag else \
        f"union_blind__{region}__{tag}.csv"
    p = paths.cache_dir("analysis") / "coverage" / name
    if not p.exists():
        sys.exit(f"{p} がありません。先に run_union_blind.py を --save で"
                 f"回してください")
    return pd.read_csv(p), p


def check_alignment(df, name, cov, lgm):
    """照合用の構成員を出し直した結果が、保存済みの列と一致するか。"""
    s = short(name)
    col = f"cov_{s}"
    if col not in df.columns:
        sys.exit(f"照合に使う {col} が CSV にありません。"
                 f"--verify で CSV にある構成員を指定してください")
    if len(cov) != len(df):
        sys.exit(f"行数が違います: 出し直し {len(cov):,} / CSV {len(df):,}。"
                 f"タイルの並びか正解の作り方が変わっています")
    a, b = np.asarray(df[col], float), np.asarray(cov, float)
    dmax = float(np.nanmax(np.abs(a - b)))
    lit_a, lit_b = a > 0, b > 0
    det_a, det_b = a >= COV_GT, b >= COV_GT
    n_lit = int((lit_a != lit_b).sum())
    n_det = int((det_a != det_b).sum())
    lcol = f"lg_{s}"
    if lcol in df.columns:
        la, lb = np.asarray(df[lcol], float), np.asarray(lgm, float)
        ok = ~(np.isnan(la) | np.isnan(lb))
        print(f"  ロジット最大の最大差      "
              f"{float(np.max(np.abs(la[ok] - lb[ok]))):.3e}")
    print(f"  覆い率の最大差            {dmax:.3e}")
    print(f"  『1画素でも光る』の不一致 {n_lit:,} 件")
    print(f"  『覆い50%以上』の不一致   {n_det:,} 件")
    if dmax > TOL or n_lit or n_det:
        sys.exit(
            "**照合に失敗しました。** 記録済みの構成員を出し直したのに" + chr(10)
            + "保存済みの列と一致しません。行の並びが合っていないので" + chr(10)
            + "列を継ぎ足せません。run_union_blind.py を全構成員で" + chr(10)
            + "回し直してください（--tag を付けて）")
    print("  **一致しました。行の並びは同じです**")


def run(region, src_tag, out_tag, verify, news, device):
    df, srcp = load_src(region, src_tag)
    print("=" * 74)
    print(f"{region}  かたまり {len(df):,}  もとの CSV {srcp.name}")
    print(f"  もとの構成員 "
          f"{[c[4:] for c in df.columns if c.startswith('cov_') and c != 'cov_union']}")
    print("=" * 74, flush=True)

    dup = [short(n) for n in news if f"cov_{short(n)}" in df.columns]
    if dup:
        sys.exit(f"もう CSV に入っている構成員です: {dup}")

    # --- 照合用の1本。**これが一致しなければ先に進まない**
    print(f"--- 照合 {short(verify)} を出し直します …", flush=True)
    GT, PB, LG = infer(get(verify), region, device)
    cov, lgm, labels = blob_cov(GT, PB, LG, labels=None)
    areas = np.array([int(s) for _, gs in labels for s in gs])
    if not np.array_equal(areas, np.asarray(df["area"], int)):
        sys.exit("area の列が合いません。かたまりの付番が変わっています")
    base = GT
    check_alignment(df, verify, cov, lgm)
    del PB, LG
    gc.collect()

    # --- 足す構成員。**正解マスクの同一性を確かめ、付番を使い回す**
    out = df.copy()
    for n in news:
        print(f"{chr(10)}--- 足す {short(n)} を推論します …", flush=True)
        G2, PB, LG = infer(get(n), region, device)
        if not (G2.shape == base.shape and (G2 == base).all()):
            sys.exit(f"{short(n)} の正解マスクが {short(verify)} と違います。"
                     f"タイルの並びか正解の作り方がずれています")
        cov2, lg2, _ = blob_cov(G2, PB, LG, labels=labels)
        s = short(n)
        out[f"cov_{s}"] = cov2
        out[f"lg_{s}"] = lg2
        R, P, F, npred = score(base, PB)
        print(f"  箇所 R {R:.4f} / P {P:.4f} / F {F:.4f} / 予測 {npred:,}")
        print(f"  1画素でも光る {100 * (cov2 > 0).mean():.1f}%"
              f" / 覆い50%以上 {100 * (cov2 >= COV_GT).mean():.1f}%")
        del PB, LG, G2
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    od = paths.cache_dir("analysis") / "coverage"
    od.mkdir(parents=True, exist_ok=True)
    p = od / f"union_blind__{region}__{out_tag}.csv"
    out.to_csv(p, index=False, encoding="utf-8")
    print(f"{chr(10)}保存: {p}")
    print("  **cov_union と n_light はもとの構成員のままです。**"
          " 足した構成員は入っていません（この方法では出せない）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--tag", required=True,
                    help="書き先の印（union_blind__<地域>__<印>.csv）")
    ap.add_argument("--src-tag", default="",
                    help="もとにする CSV の印。既定は無印（記録済みの5本）")
    ap.add_argument("--verify", default=VERIFY,
                    help="行の並びを照合するために出し直す、記録済みの構成員")
    ap.add_argument("--new", nargs="+", required=True,
                    help="足す構成員（条件名）")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    device = torch.device(a.device) if a.device else pick_device()
    print(f"device {device}")
    regs = ["hiroshima", "shimane"] if a.region == "both" else [a.region]
    for r in regs:
        run(r, a.src_tag, a.tag, a.verify, list(a.new), device)


if __name__ == "__main__":
    main()
