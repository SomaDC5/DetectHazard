# -*- coding: utf-8 -*-
"""
予測結果からTP/FP/FN/TNマスクを作り、タイルごとの統計・分類・
比較画像の書き出しまで行う共通処理。
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def confusion_masks(preds, gt):
    """preds, gt: (N,H,W) の0/1配列。各 (N,H,W) のbool配列を返す。"""
    TPm = (preds == 1) & (gt == 1)
    FPm = (preds == 1) & (gt == 0)
    FNm = (preds == 0) & (gt == 1)
    TNm = (preds == 0) & (gt == 0)
    return TPm, FPm, FNm, TNm


def overall_metrics(TPm, FPm, FNm):
    tp, fp, fn = TPm.sum(), FPm.sum(), FNm.sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return dict(recall=float(recall), precision=float(precision), f1=float(f1),
                TP=int(tp), FP=int(fp), FN=int(fn))


def _safe_mean(arr, mask):
    return float(arr[mask].mean()) if mask.sum() > 0 else np.nan


def build_tile_table(No_test, preds, gt, TPm, FPm, FNm, feature_maps=None):
    """
    タイルごとのTP/FP/FN画素数・Recall/Precision/F1、および任意の特徴量
    （標高・傾斜・明度・緑色度など）のFN/FP/TP領域内平均値をまとめたDataFrameを作る。

    feature_maps: {"mean_elev": (N,H,W)配列, "green_idx": (N,H,W)配列, ...}
                  のように渡すと、各特徴について "<name>_fn" / "<name>_fp" / "<name>_tp"
                  列が自動的に追加される。
    """
    feature_maps = feature_maps or {}
    n = preds.shape[0]
    rows = []
    for i in range(n):
        tpx, fpx, fnx = int(TPm[i].sum()), int(FPm[i].sum()), int(FNm[i].sum())
        rec_i = tpx / (tpx + fnx) if (tpx + fnx) > 0 else np.nan
        prec_i = tpx / (tpx + fpx) if (tpx + fpx) > 0 else np.nan
        f1_i = (2 * prec_i * rec_i / (prec_i + rec_i)
                if (not np.isnan(prec_i) and not np.isnan(rec_i) and (prec_i + rec_i) > 0)
                else np.nan)
        row = dict(
            idx=i, No=No_test[i],
            gt_area=int(gt[i].sum()), pred_area=int(preds[i].sum()),
            TP=tpx, FP=fpx, FN=fnx,
            recall=rec_i, precision=prec_i, f1=f1_i,
        )
        for name, arr in feature_maps.items():
            row[f"{name}_fn"] = _safe_mean(arr[i], FNm[i])
            row[f"{name}_fp"] = _safe_mean(arr[i], FPm[i])
            row[f"{name}_tp"] = _safe_mean(arr[i], TPm[i])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["bucket"] = df.apply(classify_bucket, axis=1)
    return df


def classify_bucket(row, good_f1=0.75, bad_recall=0.35, bad_precision=0.35):
    """タイルを 良好(good) / 見逃し優勢(high_FN) / 過検出優勢(high_FP) / mixed に分類する。"""
    if not np.isnan(row.f1) and row.f1 >= good_f1:
        return "good"
    if not np.isnan(row.recall) and row.recall < bad_recall:
        return "high_FN"
    if not np.isnan(row.precision) and row.precision < bad_precision:
        return "high_FP"
    return "mixed"


def pick_representative_tiles(df, n_worst_fn=20, n_worst_fp=20, n_good=15):
    """
    可視化・報告用に代表的なタイルの行インデックスを選ぶ。
    - worst_fn: FN画素数（実面積）が多い順（Recall比だけで選ぶとGT面積0の
      タイルが紛れ込むため、画素数の絶対値で選ぶ）
    - worst_fp: FPが実在する中でFP画素数が多い順（Precision比だとFP=0の
      タイル＝そもそも何も予測していないタイルが誤って混ざるため除外）
    - good: F1値が高い順
    """
    worst_fn = df.sort_values("FN", ascending=False).head(n_worst_fn).index.tolist()
    fp_candidates = df[df["FP"] > 0].sort_values("FP", ascending=False)
    worst_fp = fp_candidates.head(n_worst_fp).index.tolist()
    good = df.sort_values("f1", ascending=False).head(n_good).index.tolist()
    return dict(worst_FN=worst_fn, worst_FP=worst_fp, good=good)


def save_composite_single(out_dir, tag, idx, df, input_img, gt, preds, TPm, FPm, FNm,
                           input_label="Input"):
    """DEM単一・SAM単一など、入力が1枚の場合の4パネル比較画像を保存する。"""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(input_img[idx], cmap="gray"); axes[0].set_title(input_label); axes[0].axis("off")
    axes[1].imshow(gt[idx], cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow(preds[idx], cmap="gray"); axes[2].set_title("Pred"); axes[2].axis("off")
    err = _error_rgb(TPm[idx], FPm[idx], FNm[idx])
    axes[3].imshow(err); axes[3].set_title("TP green/FN red/FP blue"); axes[3].axis("off")
    r = df.iloc[idx]
    fig.suptitle(f"No={r.No} F1={r.f1:.2f} Rec={r.recall:.2f} Prec={r.precision:.2f}")
    plt.tight_layout()
    _save_and_close(fig, out_dir, tag, idx, r.No)


def save_composite_dual(out_dir, tag, idx, df, terrain_img, air_img, gt, preds, TPm, FPm, FNm,
                         terrain_label="Terrain"):
    """DEM＋APM・SAM＋APMなど、入力が2枚（地形＋航空写真）の場合の5パネル比較画像を保存する。"""
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    axes[0].imshow(terrain_img[idx], cmap="gray"); axes[0].set_title(terrain_label); axes[0].axis("off")
    axes[1].imshow(np.transpose(air_img[idx], (1, 2, 0))); axes[1].set_title("AirPhoto"); axes[1].axis("off")
    axes[2].imshow(gt[idx], cmap="gray"); axes[2].set_title("GT"); axes[2].axis("off")
    axes[3].imshow(preds[idx], cmap="gray"); axes[3].set_title("Pred"); axes[3].axis("off")
    err = _error_rgb(TPm[idx], FPm[idx], FNm[idx])
    axes[4].imshow(err); axes[4].set_title("TP green/FN red/FP blue"); axes[4].axis("off")
    r = df.iloc[idx]
    fig.suptitle(f"No={r.No} F1={r.f1:.2f} Rec={r.recall:.2f} Prec={r.precision:.2f}")
    plt.tight_layout()
    _save_and_close(fig, out_dir, tag, idx, r.No)


def _error_rgb(tp_mask, fp_mask, fn_mask):
    err = np.zeros((*tp_mask.shape, 3), dtype=np.uint8)
    err[tp_mask] = [0, 200, 0]
    err[fn_mask] = [255, 0, 0]
    err[fp_mask] = [0, 120, 255]
    return err


def _save_and_close(fig, out_dir, tag, idx, no):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{tag}_{idx:05d}_No{no}.png"), dpi=100)
    plt.close(fig)


def clear_and_export(base_out_dir, groups, export_fn):
    """
    worst_FN / worst_FP / good の各フォルダを作り直してから画像を書き出す。
    groups: pick_representative_tiles() の戻り値
    export_fn(folder_name, tag, idx) -> None  を呼び出し側で用意する。
    """
    for folder in ("worst_FN", "worst_FP", "good"):
        d = os.path.join(base_out_dir, folder)
        if os.path.isdir(d):
            shutil.rmtree(d)
    for idx in groups["worst_FN"]:
        export_fn("worst_FN", "fn", idx)
    for idx in groups["worst_FP"]:
        export_fn("worst_FP", "fp", idx)
    for idx in groups["good"]:
        export_fn("good", "good", idx)
