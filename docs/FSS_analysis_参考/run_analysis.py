# -*- coding: utf-8 -*-
"""
急傾斜地崩壊危険区域抽出モデルの誤差分析メインスクリプト。

使い方:
    conda activate mayo
    cd FSS_analysis
    python run_analysis.py                 # lib/conditions.py の全条件を解析
    python run_analysis.py DemOnly          # 指定した条件だけ解析
    python run_analysis.py DemOnly SAM_AirPhoto

各条件について、
  1. 保存済みモデルで学習時と同じテスト分割に対して推論を再実行
  2. スライド/ノートブック記載のRecall/Precision/F1と一致するか検証（sanity check）
  3. タイルごとのTP/FP/FN統計・地形特徴量・判定区分(bucket)をCSVに出力
  4. 代表的な見逃し／過検出／良好タイルの比較画像を出力
を行う。新しい条件を追加したい場合は lib/conditions.py に1エントリ追加すればよい。

前処理の詳細・つまずきやすい点は README.md を参照。
"""
import sys
import os
import gc
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.conditions import CONDITIONS
from lib.models import UNet, MultiEncoderUNet, load_model
from lib import data_utils as du
from lib import inference as inf
from lib import error_analysis as ea

DEVICE = "cpu"  # メモリが逼迫している環境ではCUDAより安定するためCPU推論を既定とする
torch.set_num_threads(4)


def analyze_single(name, cfg):
    print(f"=== [{name}] loading {cfg['pkl_path']} ===")
    raw = du.load_pickle(cfg["pkl_path"])
    with_mask, _ = du.split_with_mask(raw, has_airphoto=False)
    del raw
    gc.collect()
    n = len(with_mask.No)
    print(f"with-mask N = {n}")

    idx_train, idx_test = du.train_test_indices(n)
    dem_test, mask_test, _, no_test = du.stack_test_subset(with_mask, idx_test, has_airphoto=False)
    max_h = np.array(with_mask.Max_H, dtype=np.float32)
    min_h = np.array(with_mask.Min_H, dtype=np.float32)
    del with_mask
    gc.collect()

    dem_norm, norm_params = du.normalize_terrain(
        dem_test, cfg["normalize"],
        max_h_train=max_h[idx_train], min_h_train=min_h[idx_train],
        max_h_all=max_h[idx_test], min_h_all=min_h[idx_test],
    )
    x_tensor = du.to_input_tensor_single(dem_norm)

    model = UNet()
    model, best_f1, epoch = load_model(model, cfg["ckpt_path"], DEVICE)
    print(f"checkpoint best_f1_score={best_f1} (epoch {epoch})")

    preds, dem_arr = inf.run_inference_single(model, x_tensor, device=DEVICE)
    gt = mask_test

    TPm, FPm, FNm, TNm = ea.confusion_masks(preds, gt)
    metrics = ea.overall_metrics(TPm, FPm, FNm)
    _report_sanity(name, cfg, metrics)

    # 特徴量として報告する際は、正規化前の物理量（標高m・傾斜度など）に戻す。
    # (dem_arr自体はモデルに入力した値＝表示用画像としてはそのまま使う)
    if norm_params is not None:
        xh, nh = norm_params
        terrain_physical = dem_arr * (xh - nh) + nh
    else:
        terrain_physical = dem_arr
    grad = np.stack([np.linalg.norm(np.gradient(terrain_physical[i]), axis=0) for i in range(terrain_physical.shape[0])])
    feature_maps = {"terrain_value": terrain_physical, "local_gradient": grad}

    df = ea.build_tile_table(no_test, preds, gt, TPm, FPm, FNm, feature_maps=feature_maps)
    _save_outputs(cfg, df, metrics, extra_summary={
        "terrain_value_TP": float(np.nanmean(df["terrain_value_tp"])),
        "terrain_value_FN": float(np.nanmean(df["terrain_value_fn"])),
        "terrain_value_FP": float(np.nanmean(df["terrain_value_fp"])),
    })

    groups = ea.pick_representative_tiles(df)

    def export(folder, tag, idx):
        ea.save_composite_single(
            os.path.join(cfg["out_dir"], folder), tag, idx, df,
            dem_arr, gt, preds, TPm, FPm, FNm, input_label=cfg["terrain_label"],
        )
    ea.clear_and_export(cfg["out_dir"], groups, export)
    print(f"[{name}] done -> {cfg['out_dir']}\n")


def analyze_dual(name, cfg):
    print(f"=== [{name}] loading {cfg['pkl_path']} ===")
    raw = du.load_pickle(cfg["pkl_path"])
    with_mask, _ = du.split_with_mask(raw, has_airphoto=True)
    del raw
    gc.collect()
    n = len(with_mask.No)
    print(f"with-mask N = {n}")

    idx_train, idx_test = du.train_test_indices(n)
    dem_test, mask_test, air_test_raw, no_test = du.stack_test_subset(with_mask, idx_test, has_airphoto=True)
    del with_mask
    gc.collect()

    dem_norm, _ = du.normalize_terrain(dem_test, cfg["normalize"])
    dem_t, air_t = du.to_input_tensor_dual(dem_norm, air_test_raw)

    model = MultiEncoderUNet()
    model, best_f1, epoch = load_model(model, cfg["ckpt_path"], DEVICE)
    print(f"checkpoint best_f1_score={best_f1} (epoch {epoch})")

    preds, dem_arr, air_arr = inf.run_inference_dual(model, dem_t, air_t, device=DEVICE)
    gt = mask_test

    TPm, FPm, FNm, TNm = ea.confusion_masks(preds, gt)
    metrics = ea.overall_metrics(TPm, FPm, FNm)
    _report_sanity(name, cfg, metrics)

    r_ch, g_ch = air_arr[:, 0], air_arr[:, 1]
    green_idx = (g_ch - r_ch) / (g_ch + r_ch + 1e-6)  # 簡易植生指標
    brightness = air_arr.mean(axis=1)
    feature_maps = {"terrain_value": dem_arr, "green_idx": green_idx, "brightness": brightness}

    df = ea.build_tile_table(no_test, preds, gt, TPm, FPm, FNm, feature_maps=feature_maps)
    _save_outputs(cfg, df, metrics, extra_summary={
        "terrain_value_TP": float(np.nanmean(df["terrain_value_tp"])),
        "terrain_value_FN": float(np.nanmean(df["terrain_value_fn"])),
        "terrain_value_FP": float(np.nanmean(df["terrain_value_fp"])),
        "green_idx_TP": float(np.nanmean(df["green_idx_tp"])),
        "green_idx_FN": float(np.nanmean(df["green_idx_fn"])),
        "green_idx_FP": float(np.nanmean(df["green_idx_fp"])),
    })

    groups = ea.pick_representative_tiles(df)

    def export(folder, tag, idx):
        ea.save_composite_dual(
            os.path.join(cfg["out_dir"], folder), tag, idx, df,
            dem_arr, air_arr, gt, preds, TPm, FPm, FNm, terrain_label=cfg["terrain_label"],
        )
    ea.clear_and_export(cfg["out_dir"], groups, export)
    print(f"[{name}] done -> {cfg['out_dir']}\n")


def _report_sanity(name, cfg, metrics):
    target = cfg.get("pptx_metrics")
    msg = f"[sanity check:{name}] Recall={metrics['recall']:.4f} Precision={metrics['precision']:.4f} F1={metrics['f1']:.4f}"
    if target:
        msg += f"  (target: {target['recall']:.4f}/{target['precision']:.4f}/{target['f1']:.4f})"
        if abs(metrics["f1"] - target["f1"]) > 0.01:
            msg += "  !! F1が想定と大きくずれています。正規化方法を lib/verify_checkpoint.py で再確認してください。"
    print(msg)


def _save_outputs(cfg, df, metrics, extra_summary=None):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    df.to_csv(os.path.join(cfg["out_dir"], "error_stats.csv"), index=False, encoding="utf-8-sig")
    summary = dict(metrics)
    summary["n_test_tiles"] = int(len(df))
    summary["bucket_counts"] = df["bucket"].value_counts().to_dict()
    if extra_summary:
        summary.update(extra_summary)
    with open(os.path.join(cfg["out_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    names = sys.argv[1:] or list(CONDITIONS.keys())
    for name in names:
        if name not in CONDITIONS:
            print(f"unknown condition: {name} (available: {list(CONDITIONS.keys())})")
            continue
        cfg = CONDITIONS[name]
        if cfg["kind"] == "single":
            analyze_single(name, cfg)
        else:
            analyze_dual(name, cfg)


if __name__ == "__main__":
    main()
