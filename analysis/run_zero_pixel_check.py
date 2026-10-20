# -*- coding: utf-8 -*-
"""落ちた箇所のうち「1画素も出ていない」ものの割合を測る。

なぜ必要か（展望_再現率をどう上げるか.md 2(b) / 16）
----------------------------------------------------
箇所正規化損失（InstLoss）が効かなかった理由は

    「被覆0.5を落ちた箇所の 69.4 % は1画素も出ていない」
    → その画素のロジットが大きな負の値に張り付いている
    → シグモイドの微分がほぼ0なので、重みを8倍しても勾配は 0 × 8 = 0

だった（`dc5lib/instance_loss.py` の設計メモ）。

オーバーラップ学習は、境界の断片だったかたまりを学習時に中心で見せる。
**それによって「1画素も出ていない」が減っているなら、重みに掛ける相手が
できたことになり、InstLoss を足す理由が立つ。** 69.4 % のままなら、
同じ理由で同じように失敗する。

この1つの数字で、OL学習 + InstLoss をやるかやらないかが決まる。

やること
--------
正解のかたまりごとに被覆率（`analysis.instances.per_gt_instance` の coverage）を
出して、被覆 < 0.5（= 主設定で落ちた箇所）のうち **coverage == 0 ちょうど**の
割合を数える。学習はしない。推論のみ。

従来推論とオーバーラップ推論の両方で測る。オーバーラップ推論の経路は
`run_overlap_prune.py` の infer() をそのまま使うので、`13` の測定と
同じ二値マスクになる。

使い方
------
    python analysis/run_zero_pixel_check.py
    python analysis/run_zero_pixel_check.py --region shimane
    python analysis/run_zero_pixel_check.py --conditions FinalFusion_SAM_APM
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, instances                               # noqa: E402
from analysis.data import build_tileset                              # noqa: E402
from dc5lib import inputs as minputs                                 # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.models import build_model_for, load_weights              # noqa: E402
from dc5lib.paths import dataset_path                                # noqa: E402
from dc5lib.registry import get                                      # noqa: E402

REGION_DIR = {"hiroshima": "Hiroshima", "shimane": "Shimane"}

# (条件名, 推論の種類) の組。推論の種類は "normal" か "overlap"。
DEFAULT_PAIRS = [
    ("FinalFusion_SAM_APM", "normal"),              # 69.4% を再現できるか
    ("FinalFusion_SAM_APM_InstLoss", "normal"),     # 参考（p=0.5）
    ("FinalFusion_SAM_APM_OverlapTrain", "normal"),
    ("FinalFusion_SAM_APM_OverlapTrain", "overlap"),  # いまの1段目
]

# 箇所の大きさの区切り（画素）。instance_loss.py の設計メモと揃えた。
BINS = [0, 50, 100, 300, 1000, 3000, 10 ** 9]
BIN_NAMES = ["〜50", "50-100", "100-300", "300-1000", "1000-3000", "3000〜"]


@torch.no_grad()
def masks_normal(cond, region, device, batch_size=32):
    """従来推論。(正解, 予測) をタイルごとに積んで返す。"""
    dirname = REGION_DIR[region]
    pkl = dataset_path(dirname, cond.pkl_for(dirname))
    tiles = build_tileset(pkl, region, cond.use_airphoto,
                          bg_ratio=float(getattr(cond, "bg_ratio", 0.0) or 0.0),
                          need_landuse=minputs.needs_landuse(cond))
    model = build_model_for(cond).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()

    n = len(tiles.no)
    GT = np.zeros((n, 128, 128), bool)
    PD = np.zeros_like(GT)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        terrain = torch.from_numpy(tiles.terrain[s:e]).to(device)
        if cond.use_airphoto:
            logits = model(*minputs.model_args(
                cond, terrain, tiles.air[s:e],
                None if tiles.landuse is None else tiles.landuse[s:e],
                device=device))
        else:
            logits = model(terrain)
        PD[s:e] = (torch.sigmoid(logits)[:, 0] > config.THRESHOLD).cpu().numpy()
        GT[s:e] = tiles.mask[s:e][:, 0] > 0.5
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GT, PD


def coverage_rows(GT, PD, connectivity=8, min_size=10):
    """正解のかたまりごとに (面積, 被覆率) を集める。"""
    area, cov = [], []
    for g, p in zip(GT, PD):
        for r in instances.per_gt_instance(g.astype(np.uint8), p.astype(np.uint8),
                                           connectivity, min_size):
            area.append(r["gt_area"])
            cov.append(r["coverage"])
    return np.asarray(area), np.asarray(cov)


def summarize(area, cov, label):
    """被覆 < 0.5 の箇所のうち、被覆ちょうど 0 の割合。"""
    missed = cov < 0.5
    zero = missed & (cov == 0.0)
    out = dict(条件=label, 正解の箇所数=int(len(cov)),
               落ちた箇所=int(missed.sum()),
               うち0画素=int(zero.sum()),
               **{"0画素の割合": float(zero.sum() / max(missed.sum(), 1))})
    # 大きさ別
    idx = np.digitize(area, BINS[1:-1], right=True)
    for b, name in enumerate(BIN_NAMES):
        sel = idx == b
        m = missed & sel
        out[name] = (float((zero & sel).sum() / m.sum()) if m.sum() else np.nan)
        # 重み付けが届きうる箇所（落ちたが1画素以上は出ている）の実数。
        # InstLoss の伸びしろはこの数が上限になる。
        out[name + "_届く数"] = int((m & (cov > 0.0)).sum())
    out["届く数_合計"] = int((missed & (cov > 0.0)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="省略時は DEFAULT_PAIRS。指定すると従来推論のみ")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    config.ensure_dirs()
    device = pick_device(args.device)
    print(f"device = {device}")

    pairs = ([(c, "normal") for c in args.conditions] if args.conditions
             else DEFAULT_PAIRS)

    rows = []
    for name, how in pairs:
        cond = get(name)
        label = f"{cond.label}" + ("（OL推論）" if how == "overlap" else "")
        print(f"\n■ {label}  [{how}]", flush=True)
        if how == "overlap":
            # 13 と同じ経路。確率側で二値化するので記録済みの値と一致する。
            from analysis.run_overlap_prune import infer as ol_infer
            GT, PD, _ = ol_infer(cond, args.region, device)
        else:
            GT, PD = masks_normal(cond, args.region, device, args.batch_size)
        area, cov = coverage_rows(GT, PD)
        r = summarize(area, cov, label)
        rows.append(r)
        print(f"   落ちた箇所 {r['落ちた箇所']} / うち0画素 {r['うち0画素']} "
              f"= {r['0画素の割合']:.1%}")
        del GT, PD

    df = pd.DataFrame(rows)
    out = os.path.join(config.REPORT_DIR, f"zero_pixel_{args.region}.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"\n=== 落ちた箇所のうち「1画素も出ていない」割合（{args.region}） ===")
    show = ["条件", "正解の箇所数", "落ちた箇所", "うち0画素", "0画素の割合"]
    print(df[show].to_string(index=False,
                             formatters={"0画素の割合": "{:.1%}".format}))
    print("\n=== 大きさ別（同じ割合） ===")
    print(df[["条件"] + BIN_NAMES].to_string(
        index=False, formatters={n: (lambda v: "" if pd.isna(v) else f"{v:.1%}")
                                 for n in BIN_NAMES}))
    print("\n=== 重み付けが届きうる箇所の実数（落ちたが1画素以上は出ている） ===")
    print(df[["条件", "届く数_合計"] + [n + "_届く数" for n in BIN_NAMES]]
          .to_string(index=False))
    print("\n", out)


if __name__ == "__main__":
    main()
