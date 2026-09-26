# -*- coding: utf-8 -*-
"""Attention Gate が実際にどこを見ているかを測る。

Attention U-Net のスキップ接続に掛かる注意係数（0〜1）を取り出し、
各段・各モダリティについて

  - 係数の平均（そのブランチをどれだけ通しているか）
  - 正解領域の内側と外側での平均の差（警戒区域に絞り込めているか）

をタイルごとに記録する。「Attention を入れたのに精度が変わらない」のが
「注意が効いていない」からなのか「注意は効いているが精度に繋がっていない」
のかを切り分けるための材料。

    python scripts/run_attention.py --region hiroshima
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from analysis import config, infer  # noqa: E402
from analysis.data import build_tileset  # noqa: E402
from dc5lib.models import build_model, load_weights  # noqa: E402

ATTENTION_CONDITIONS = [
    "AttentionUNet_SAM_APM",
    "AttentionUNet_DEM_APM",
    "DataOgument_AttentionUNet_SAM_APM",
]


@torch.no_grad()
def run(cond, tiles, device, batch_size=16):
    model = build_model(cond.model_class).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()

    n = len(tiles)
    keys = [f"l{l}_{b}" for l in (1, 2, 3, 4) for b in ("slope", "curv")]
    cols = {"tile_no": np.asarray(tiles.no)}
    for k in keys:
        cols[f"{k}_mean"] = np.zeros(n)
        cols[f"{k}_in"] = np.full(n, np.nan)
        cols[f"{k}_out"] = np.full(n, np.nan)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        terrain = torch.from_numpy(tiles.terrain[s:e]).to(device)
        air = torch.from_numpy(tiles.air[s:e]).to(device).float() / 255.0
        gt = torch.from_numpy(tiles.mask[s:e]).to(device).float()

        _, att = model(terrain, air, return_attention=True)

        for k, a in att.items():
            a = a.float()                                   # [B, 1, H, W]
            size = a.shape[-1]
            g = F.adaptive_avg_pool2d(gt, size) > 0.5       # その解像度での正解
            flat_a = a.flatten(1)
            flat_g = g.flatten(1)
            cols[f"{k}_mean"][s:e] = flat_a.mean(1).cpu().numpy()

            n_in = flat_g.sum(1)
            n_out = (~flat_g).sum(1)
            sum_in = (flat_a * flat_g).sum(1)
            sum_out = (flat_a * ~flat_g).sum(1)
            mean_in = torch.where(n_in > 0, sum_in / n_in.clamp(min=1),
                                  torch.full_like(sum_in, float("nan")))
            mean_out = torch.where(n_out > 0, sum_out / n_out.clamp(min=1),
                                   torch.full_like(sum_out, float("nan")))
            cols[f"{k}_in"][s:e] = mean_in.cpu().numpy()
            cols[f"{k}_out"][s:e] = mean_out.cpu().numpy()

        if (s // batch_size) % 20 == 0:
            print(f"    {e}/{n}", end="\r", flush=True)

    print(f"    {n}/{n} 完了          ", flush=True)
    df = pd.DataFrame(cols)
    for k in keys:
        df[f"{k}_sel"] = df[f"{k}_in"] - df[f"{k}_out"]
    df.insert(0, "condition", cond.name)
    df.insert(1, "region", tiles.region)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config.ensure_dirs()
    device = infer.pick_device(args.device)
    print(f"device = {device}")

    conds = [config.BY_NAME[n] for n in ATTENTION_CONDITIONS]
    frames = []
    for pkl_path, group in config.group_by_pkl(conds, args.region).items():
        print(f"\n[{args.region}] {os.path.basename(pkl_path)}")
        tiles = build_tileset(pkl_path, args.region, True)
        for c in group:
            print(f"  → {c.name}")
            frames.append(run(c, tiles, device, args.batch_size))
        del tiles

    df = pd.concat(frames, ignore_index=True)
    out = os.path.join(config.REPORT_DIR, f"attention_coefficients__{args.region}.csv")
    df.to_csv(out, index=False)

    # 条件×段×モダリティの要約
    rows = []
    for (cond,), g in df.groupby(["condition"]):
        for level in (1, 2, 3, 4):
            for branch, ja in (("slope", "地形量"), ("curv", "航空写真")):
                k = f"l{level}_{branch}"
                rows.append({
                    "condition": cond, "region": args.region,
                    "段": f"level {level}（{128 // 2 ** (level - 1)}px）", "ブランチ": ja,
                    "係数の平均": round(g[f"{k}_mean"].mean(), 4),
                    "正解の内側": round(g[f"{k}_in"].mean(), 4),
                    "正解の外側": round(g[f"{k}_out"].mean(), 4),
                    "内外差(選択性)": round(g[f"{k}_sel"].mean(), 4),
                })
    summary = pd.DataFrame(rows)
    sp = os.path.join(config.REPORT_DIR, f"attention_summary__{args.region}.csv")
    summary.to_csv(sp, index=False)
    print("\n", summary.to_string(index=False))
    print(f"\n{out}\n{sp}")


if __name__ == "__main__":
    main()
