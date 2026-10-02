# -*- coding: utf-8 -*-
"""箇所数評価。推論をやり直して、タイルごとに「かたまり」単位で数える。

対応づけの方式としきい値を複数まとめて評価するので、あとからしきい値を
振り直すために推論をやり直す必要はない。

    # まず1条件・広島だけで様子を見る（45秒ほど）
    python scripts/run_instance_eval.py --region hiroshima \
        --conditions FinalFusion_SAM_APM

    # 代表条件を比較
    python scripts/run_instance_eval.py --region hiroshima \
        --conditions Train_Hiroshima_Test_Shimane_OnlySAM FinalFusion_SAM_APM \
                     DataOgument_AttentionUNet_SAM_APM

    # 全条件・両地域
    python scripts/run_instance_eval.py --region both

出力
    output/instances/<条件>__<地域>.csv   1行=1タイル（設定ごとの TP/FN/FP と診断値）
    output/report/instance_summary.csv    条件 × 設定 の箇所 Recall / Precision / F値
    output/report/instance_diagnostics.csv 飲み込み・分裂・面積比の集計
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis import config, infer, instances  # noqa: E402
from analysis.data import build_tileset  # noqa: E402
from dc5lib.models import build_model, load_weights  # noqa: E402

INSTANCES_DIR = os.path.join(config.OUTPUT_DIR, "instances")

# 比較する設定。推論1回でこれら全部を評価する。
SETTINGS = {
    "被覆 (cov_gt=0.5, cov_pred=0.3)": dict(mode="coverage", cover_gt=0.5, cover_pred=0.3),
    "被覆 (cov_gt=0.3, cov_pred=0.3)": dict(mode="coverage", cover_gt=0.3, cover_pred=0.3),
    "被覆 (cov_gt=0.5, cov_pred=0.1)": dict(mode="coverage", cover_gt=0.5, cover_pred=0.1),
    "IoU多対1 (0.3)": dict(mode="iou_many", iou_thr=0.3),
    "IoU多対1 (0.1)": dict(mode="iou_many", iou_thr=0.1),
    "IoU厳密1対1 (0.3)": dict(mode="iou_1to1", iou_thr=0.3),
    "IoU厳密1対1 (0.1)": dict(mode="iou_1to1", iou_thr=0.1),
}


@torch.no_grad()
def run_condition(cond, tiles, device, batch_size=32, connectivity=8, min_size=10,
                  prob_thr=None, save_instances=False):
    model = build_model(cond.model_class).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()
    thr = config.THRESHOLD if prob_thr is None else prob_thr

    rows, inst_rows = [], []
    n = len(tiles)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        terrain = torch.from_numpy(tiles.terrain[s:e]).to(device)
        if cond.use_airphoto:
            air = torch.from_numpy(tiles.air[s:e]).to(device).float() / 255.0
            logits = model(terrain, air)
        else:
            logits = model(terrain)
        pred = (torch.sigmoid(logits) > thr).squeeze(1).cpu().numpy()
        gt = tiles.mask[s:e, 0] > 0

        for k in range(e - s):
            r = instances.evaluate_tile_multi(
                gt[k], pred[k], SETTINGS,
                connectivity=connectivity, min_size=min_size)
            r["tile_no"] = tiles.no[s + k]
            rows.append(r)
            if save_instances:
                for ir in instances.per_gt_instance(
                        gt[k], pred[k], connectivity=connectivity, min_size=min_size):
                    ir["tile_no"] = tiles.no[s + k]
                    inst_rows.append(ir)

        if (s // batch_size) % 20 == 0:
            print(f"    {e}/{n}", end="\r", flush=True)

    print(f"    {n}/{n} 完了            ", flush=True)
    df = pd.DataFrame(rows)
    df.insert(0, "condition", cond.name)
    df.insert(1, "region", tiles.region)
    inst = pd.DataFrame(inst_rows) if save_instances else None
    if inst is not None and len(inst):
        inst.insert(0, "condition", cond.name)
        inst.insert(1, "region", tiles.region)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return df, inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "hiroshima_bg", "shimane", "both", "all"])
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8],
                    help="かたまりの連結性（斜めをつなぐか）")
    ap.add_argument("--min-size", type=int, default=10,
                    help="この画素数未満のかたまりは数えない")
    ap.add_argument("--prob-thr", type=float, default=None,
                    help="二値化しきい値（既定は学習時と同じ 0.5）")
    ap.add_argument("--save-instances", action="store_true",
                    help="正解の箇所を1行ずつ output/instances/*__gt.csv に書き出す")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    os.makedirs(INSTANCES_DIR, exist_ok=True)
    device = infer.pick_device(args.device)
    print(f"device = {device}　連結性 {args.connectivity} 近傍 / "
          f"最小サイズ {args.min_size} px / 二値化 "
          f"{args.prob_thr if args.prob_thr is not None else config.THRESHOLD}")

    conds = config.CONDITIONS
    if args.conditions:
        conds = [config.BY_NAME[n] for n in args.conditions]
    if args.region == "both":
        regions = ["hiroshima", "shimane"]
    elif args.region == "all":
        regions = config.REGIONS
    else:
        regions = [args.region]

    def path(c, r):
        return os.path.join(INSTANCES_DIR, f"{c}__{r}.csv")

    for region in regions:
        todo = [c for c in conds if args.force or not os.path.exists(path(c.name, region))]
        if not todo:
            print(f"[{region}] 済み。--force で再計算")
            continue
        for (pkl_path, bg_ratio), group in config.group_by_pkl(todo, region).items():
            need_air = any(c.use_airphoto for c in group)
            print(f"\n[{region}] {os.path.basename(pkl_path)}")
            tiles = build_tileset(pkl_path, region, need_air, bg_ratio=bg_ratio)
            for c in group:
                print(f"  → {c.name}")
                df, inst = run_condition(c, tiles, device, args.batch_size,
                                         args.connectivity, args.min_size,
                                         args.prob_thr, args.save_instances)
                df.to_csv(path(c.name, region), index=False)
                if inst is not None and len(inst):
                    inst.to_csv(os.path.join(
                        INSTANCES_DIR, f"{c.name}__{region}__gt.csv"), index=False)
                for name in SETTINGS:
                    r = instances.summarize_setting(df, name)
                    print(f"     {name:<30} R={r['箇所Recall']:.4f} "
                          f"P={r['箇所Precision']:.4f} F={r['箇所F値']:.4f}")
            del tiles

    # ---------------------------------------------------------- 集計
    frames = []
    for c in config.CONDITIONS:
        for region in config.REGIONS:
            p = path(c.name, region)
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
    if not frames:
        return
    all_df = pd.concat(frames, ignore_index=True)

    labels = {c.name: c.label for c in config.CONDITIONS}
    summ, diag = [], []
    for (cond, region), g in all_df.groupby(["condition", "region"], sort=False):
        for name in SETTINGS:
            r = instances.summarize_setting(g, name)
            r.update(condition=cond, label=labels[cond], region=region)
            summ.append(r)
        diag.append({
            "condition": cond, "label": labels[cond], "region": region,
            "タイル数": len(g),
            "正解の箇所数": int(g["n_gt"].sum()), "予測の箇所数": int(g["n_pred"].sum()),
            "予測/正解 箇所数比": round(g["n_pred"].sum() / max(g["n_gt"].sum(), 1), 3),
            "飲み込んだ予測": int(g["merged_pred"].sum()),
            "飲み込まれた正解": int(g["gt_in_merge"].sum()),
            "飲み込まれた正解の割合": round(g["gt_in_merge"].sum() / max(g["n_gt"].sum(), 1), 4),
            "1予測が覆う正解の最大": int(g["max_gt_per_pred"].max()),
            "分裂した正解": int(g["split_gt"].sum()),
            "分裂した正解の割合": round(g["split_gt"].sum() / max(g["n_gt"].sum(), 1), 4),
            "予測面積/正解面積": round(float(g["area_ratio"].replace(
                [np.inf, -np.inf], np.nan).dropna().mean()), 3),
            "正解ごとの最良IoU平均": round(float(g["mean_best_iou"].dropna().mean()), 4),
        })

    cols = ["region", "label", "設定", "箇所_正解数", "箇所_予測数", "検出できた正解",
            "的中した予測", "箇所Recall", "箇所Precision", "箇所F値", "condition"]
    s = pd.DataFrame(summ)[cols]
    d = pd.DataFrame(diag)
    s.to_csv(os.path.join(config.REPORT_DIR, "instance_summary.csv"), index=False)
    d.to_csv(os.path.join(config.REPORT_DIR, "instance_diagnostics.csv"), index=False)
    print("\n=== 箇所数評価 ===")
    pd.set_option("display.width", 250)
    print(s.drop(columns=["condition"]).to_string(index=False))
    print("\n=== 診断（飲み込み・分裂） ===")
    print(d.drop(columns=["condition"]).to_string(index=False))
    print(f"\n{config.REPORT_DIR}/instance_summary.csv, instance_diagnostics.csv")


if __name__ == "__main__":
    main()
