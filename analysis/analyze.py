# -*- coding: utf-8 -*-
"""タイル単位の推論結果を条件横断で分析する。

output/tiles/*.csv（1行=1タイル）と output/features/*.csv（タイルの性質）を
突き合わせて、

  - 条件ごとの集計（画素単位のmicro と タイル単位のmacro）
  - 島根を「警戒区域を含むタイルだけ」に揃えた再集計
  - 同じタイル上での条件どうしの対比較（Attention vs Final など）
  - タイルの性質（正解の面積・分割数・傾斜・起伏・境界への掛かり方）で層別した精度
  - 誤差の内訳（見逃し/過検出、境界16pxのリング vs 中心）
  - 条件間の一致度と「どの条件でも外すタイル」の抽出

を出す。
"""

import itertools
import os

import numpy as np
import pandas as pd

from . import config
from .features import features_path
from .infer import tile_path

CROP = config.BORDER_CROP

# 箇所数評価でレポートの本表に出す設定（run_instance_eval.py の SETTINGS のキー）
PRIMARY_INSTANCE_SETTING = "被覆 (cov_gt=0.5, cov_pred=0.3)"
CENTER = config.TILE_SIZE - 2 * CROP


# ------------------------------------------------------------------ 読み込み
def load_tiles(region, conditions=None):
    conds = conditions or [c.name for c in config.CONDITIONS]
    frames = []
    for name in conds:
        p = tile_path(name, region)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"{region} のタイル結果が output/tiles にありません")
    return pd.concat(frames, ignore_index=True)


def add_metrics(df):
    """タイルごとの指標を足す。"""
    d = df.copy()
    for suf in ("", "_c"):
        tp, fp, fn = d["tp" + suf], d["fp" + suf], d["fn" + suf]
        denom = 2 * tp + fp + fn
        d["f1" + suf] = np.where(denom > 0, 2 * tp / denom.replace(0, np.nan), np.nan)
        d["iou" + suf] = np.where((tp + fp + fn) > 0,
                                  tp / (tp + fp + fn).replace(0, np.nan), np.nan)
        d["recall" + suf] = np.where(d["gt_pos" + suf] > 0,
                                     tp / d["gt_pos" + suf].replace(0, np.nan), np.nan)
        d["precision" + suf] = np.where(d["pred_pos" + suf] > 0,
                                        tp / d["pred_pos" + suf].replace(0, np.nan), np.nan)
    # 境界16pxのリング部分（全体 − 中心）
    for k in ("tp", "fp", "fn", "gt_pos", "pred_pos"):
        d[k + "_ring"] = d[k] - d[k + "_c"]
    d["has_gt"] = d["gt_pos"] > 0
    d["err"] = d["fp"] + d["fn"]
    d["err_ring"] = d["fp_ring"] + d["fn_ring"]
    return d


def attach_meta(df):
    meta = pd.DataFrame([{
        "condition": c.name, "label": c.label, "arch": c.arch, "terrain": c.terrain,
        "augmented": c.augmented, "use_airphoto": c.use_airphoto,
        "input_mode": "デュアル入力" if c.use_airphoto else "単一入力",
    } for c in config.CONDITIONS])
    return df.merge(meta, on="condition", how="left")


def load_all(region, with_features=True):
    df = add_metrics(load_tiles(region))
    df = attach_meta(df)
    if with_features:
        fp = features_path(region)
        if os.path.exists(fp):
            feats = pd.read_csv(fp).drop(columns=["region", "gt_pos"], errors="ignore")
            # 推論結果側にも入っている列（bg_holdout など）は落として重複を避ける
            dup = [c for c in feats.columns if c != "tile_no" and c in df.columns]
            feats = feats.drop(columns=dup)
            df = df.merge(feats, on="tile_no", how="left")
    return df


# ------------------------------------------------------------------ 集計
def micro(df, suf=""):
    """画素単位の Recall / Precision / F値 と、その元になった画素数。

    以前は割合だけ返して tp/fp/fn を捨てていた。それだと results/metrics.csv に
    絶対数が残らず、あとから陽性画素率などを再計算できなくなるので、
    カウントも返すようにした。
    """
    tp = int(df["tp" + suf].sum())
    fp = int(df["fp" + suf].sum())
    fn = int(df["fn" + suf].sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    pre = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * rec * pre / (rec + pre) if rec and pre else np.nan
    return rec, pre, f1, tp, fp, fn


def summary_by_condition(df, region):
    """条件ごとの集計。画素単位(micro)とタイル単位(macro)の両方。"""
    rows = []
    for (cond, label), g in df.groupby(["condition", "label"], sort=False):
        gt = g[g["has_gt"]]
        row = {"region": region, "condition": cond, "label": label,
               "n_tiles": len(g), "n_tiles_with_gt": len(gt)}
        for suf, tag in (("", "通常"), ("_c", "境界")):
            r, p, f, tp, fp, fn = micro(g, suf)
            row[f"{tag}_recall"] = round(r, 4)
            row[f"{tag}_precision"] = round(p, 4)
            row[f"{tag}_f1"] = round(f, 4)
            row[f"{tag}_tp"] = tp
            row[f"{tag}_fp"] = fp
            row[f"{tag}_fn"] = fn
            row[f"{tag}_f1_macro"] = round(gt["f1" + suf].mean(), 4)
            row[f"{tag}_f1_median"] = round(gt["f1" + suf].median(), 4)
            row[f"{tag}_iou_macro"] = round(gt["iou" + suf].mean(), 4)
        # 正解の無いタイルでの過検出（島根のみ存在する）
        bg = g[~g["has_gt"]]
        row["n_tiles_bg"] = len(bg)
        row["bg_fp_pixels"] = int(bg["fp"].sum())
        row["bg_fp_share"] = round(bg["fp"].sum() / g["fp"].sum(), 4) if g["fp"].sum() else 0.0
        row["bg_tiles_with_fp"] = int((bg["pred_pos"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("通常_f1", ascending=False)


def load_hiroshima_all(holdout_only=True, with_features=True):
    """広島を「背景タイル込み」にした集合。

    広島は警戒区域ありのタイルだけを 8:2 に分けてテストを作っていたので、
    そのままでは島根（地域内の全タイル）と構成が違う。背景タイルにも同じ
    20 % のホールドアウト率を当てはめて足すと、「最初から全 38,148 枚を
    8:2 に分けていたら得られたテスト集合」と同じ構成になる。

    holdout_only=False なら背景タイル 23,332 枚すべてを足す（背景の比率は
    地域の実際より高くなるので、過検出の見積もりの上限として使う）。
    """
    pos = load_all("hiroshima", with_features=with_features)
    bg = load_all("hiroshima_bg", with_features=with_features)
    if holdout_only:
        if "bg_holdout" not in bg.columns:
            raise KeyError("bg_holdout 列がありません。run_inference.py を再実行してください")
        bg = bg[bg["bg_holdout"].astype(bool)]
    out = pd.concat([pos, bg], ignore_index=True)
    out["region"] = "hiroshima_all"
    return out


def region_matrix(frames):
    """地域 × テスト集合の作り方 の一覧。

    frames: {"広島 警戒区域ありのみ": df, "広島 全件相当": df, ...}
    """
    rows = []
    for cond in [c.name for c in config.CONDITIONS]:
        row = {"condition": cond}
        for name, df in frames.items():
            g = df[df["condition"] == cond]
            if g.empty:
                continue
            row["label"] = g["label"].iloc[0]
            r, p, f, tp, fp, fn = micro(g)
            row[f"{name}_n"] = len(g)
            row[f"{name}_recall"] = round(r, 4)
            row[f"{name}_precision"] = round(p, 4)
            row[f"{name}_f1"] = round(f, 4)
            row[f"{name}_tp"] = tp
            row[f"{name}_fp"] = fp
            row[f"{name}_fn"] = fn
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = ["condition", "label"] + [c for c in out.columns if c not in ("condition", "label")]
    return out[cols]


def matched_subset_summary(df, region):
    """「警戒区域を含むタイルだけ」に揃えた再集計（広島と同じ選び方）。"""
    rows = []
    for (cond, label), g in df.groupby(["condition", "label"], sort=False):
        gt = g[g["has_gt"]]
        r_all, p_all, f_all, tp_all, fp_all, fn_all = micro(g)
        r_gt, p_gt, f_gt, tp_gt, fp_gt, fn_gt = micro(gt)
        rows.append({
            "region": region, "condition": cond, "label": label,
            "全タイル_n": len(g), "全タイル_recall": round(r_all, 4),
            "全タイル_precision": round(p_all, 4), "全タイル_f1": round(f_all, 4),
            "警戒区域ありのみ_n": len(gt), "警戒区域ありのみ_recall": round(r_gt, 4),
            "警戒区域ありのみ_precision": round(p_gt, 4), "警戒区域ありのみ_f1": round(f_gt, 4),
            "f1差": round(f_gt - f_all, 4), "precision差": round(p_gt - p_all, 4),
        })
    return pd.DataFrame(rows).sort_values("警戒区域ありのみ_f1", ascending=False)


# ------------------------------------------------------------------ 対比較
PAIRS = [
    # (基準, 比較対象, 何を見たいか)
    ("FinalFusion_SAM_APM", "AttentionUNet_SAM_APM", "Attention機構の効果 (SAM)"),
    ("FinalFusion_DEM_APM", "AttentionUNet_DEM_APM", "Attention機構の効果 (DEM)"),
    ("FinalFusion_SAM_APM", "TransUNet_SAM_APM", "Transformerの効果 (SAM)"),
    ("FinalFusion_SAM_APM", "MiddleFusion_SAM_APM", "Middle vs Final (SAM)"),
    ("FinalFusion_DEM_APM", "MIddleFusion_DEM_APM", "Middle vs Final (DEM)"),
    ("FinalFusion_SAM_APM", "EarlyFusionUNet_SAM_APM", "Early vs Final (SAM)"),
    ("FinalFusion_DEM_APM", "EarlyFusionUNet_DEM_APM", "Early vs Final (DEM)"),
    ("FinalFusion_SAM_APM", "DataOgument_FinalFusion_SAM_APM", "データ拡張の効果 (Final SAM)"),
    ("FinalFusion_DEM_APM", "DataOgument_FinalFusion_DEM_APM", "データ拡張の効果 (Final DEM)"),
    ("AttentionUNet_SAM_APM", "DataOgument_AttentionUNet_SAM_APM", "データ拡張の効果 (Attention SAM)"),
    ("Train_Hiroshima_Test_Shimane_OnlySAM", "FinalFusion_SAM_APM", "航空写真の追加 (SAM)"),
    ("Train_Hiroshima_Test_Shimane_OnlyDEM", "FinalFusion_DEM_APM", "航空写真の追加 (DEM)"),
    ("FinalFusion_SAM_APM", "FinalFusion_DEM_APM", "SAM vs DEM (Final)"),
]


def paired_comparison(df, base, other, metric="f1", region=None):
    """同じタイルの上での差を見る。正解の無いタイルは除く。"""
    cols = ["tile_no", metric]
    a = df[(df["condition"] == base) & df["has_gt"]][cols].rename(columns={metric: "base"})
    b = df[(df["condition"] == other) & df["has_gt"]][cols].rename(columns={metric: "other"})
    m = a.merge(b, on="tile_no", how="inner").dropna()
    if m.empty:
        return None
    d = m["other"] - m["base"]
    try:
        from scipy.stats import wilcoxon
        p = wilcoxon(m["other"], m["base"]).pvalue if len(m) > 10 else np.nan
    except Exception:
        p = np.nan
    return {
        "region": region, "基準": base, "比較": other, "n": len(m),
        "基準_平均": round(m["base"].mean(), 4), "比較_平均": round(m["other"].mean(), 4),
        "平均差": round(d.mean(), 4), "中央値差": round(d.median(), 4),
        "改善タイル数": int((d > 0.01).sum()), "悪化タイル数": int((d < -0.01).sum()),
        "ほぼ同じ": int((d.abs() <= 0.01).sum()),
        "改善率": round((d > 0.01).mean(), 3),
        "大きく改善(>0.1)": int((d > 0.1).sum()), "大きく悪化(<-0.1)": int((d < -0.1).sum()),
        "Wilcoxon_p": p,
    }


def all_pairs(df, region):
    rows = []
    have = set(df["condition"].unique())
    for base, other, why in PAIRS:
        if base in have and other in have:
            r = paired_comparison(df, base, other, region=region)
            if r:
                r["観点"] = why
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 層別
STRATA = [
    ("gt_ratio", [0, 0.02, 0.05, 0.10, 0.20, 1.01],
     ["〜2%", "2-5%", "5-10%", "10-20%", "20%〜"], "正解の面積割合"),
    ("slope_mean", [0, 10, 15, 20, 25, 90],
     ["〜10°", "10-15°", "15-20°", "20-25°", "25°〜"], "タイル平均傾斜角"),
    ("relief", [0, 50, 100, 200, 400, 10000],
     ["〜50m", "50-100m", "100-200m", "200-400m", "400m〜"], "起伏量(最大-最小標高)"),
    ("gt_components", [0, 1, 2, 3, 5, 1000],
     ["1個", "2個", "3個", "4-5個", "6個〜"], "正解の連結成分数"),
    ("gt_border_ratio", [-0.001, 0.0001, 0.1, 0.3, 0.5, 1.01],
     ["境界に無し", "〜10%", "10-30%", "30-50%", "50%〜"], "正解が境界16pxに掛かる割合"),
    ("gt_compactness", [0, 0.15, 0.25, 0.35, 0.5, 10],
     ["〜0.15", "0.15-0.25", "0.25-0.35", "0.35-0.5", "0.5〜"], "形の複雑さ(周長/面積)"),
]


def stratified(df, region, metric="f1"):
    rows = []
    gt = df[df["has_gt"]].copy()
    for col, bins, labels, why in STRATA:
        if col not in gt.columns:
            continue
        gt["_bin"] = pd.cut(gt[col], bins=bins, labels=labels, include_lowest=True)
        g = gt.groupby(["condition", "label", "_bin"], observed=True)[metric]
        agg = g.agg(["mean", "median", "count"]).reset_index()
        agg["変数"] = why
        agg["変数名"] = col
        agg = agg.rename(columns={"_bin": "区分", "mean": "平均F1",
                                  "median": "中央値F1", "count": "タイル数"})
        agg["平均F1"] = agg["平均F1"].round(4)
        agg["中央値F1"] = agg["中央値F1"].round(4)
        agg["region"] = region
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ 誤差の内訳
def error_decomposition(df, region):
    rows = []
    for (cond, label), g in df.groupby(["condition", "label"], sort=False):
        fp, fn = g["fp"].sum(), g["fn"].sum()
        err = fp + fn
        ring_px = config.TILE_SIZE ** 2 - CENTER ** 2
        ring_share_of_area = ring_px / config.TILE_SIZE ** 2
        rows.append({
            "region": region, "condition": cond, "label": label,
            "見逃しFN": int(fn), "過検出FP": int(fp),
            "FN割合": round(fn / err, 4) if err else np.nan,
            "境界リングの誤差割合": round(g["err_ring"].sum() / err, 4) if err else np.nan,
            "リングの面積割合": round(ring_share_of_area, 4),
            "リング誤差の集中度": round(
                (g["err_ring"].sum() / err) / ring_share_of_area, 3) if err else np.nan,
            "抽出面積/正解面積": round(g["pred_pos"].sum() / g["gt_pos"].sum(), 4)
            if g["gt_pos"].sum() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("condition")


# ------------------------------------------------------------------ 条件間の一致
def difficulty_and_agreement(df, region):
    """タイル×条件の F1 行列から、条件間の相関と「全条件が外すタイル」を出す。"""
    gt = df[df["has_gt"]]
    mat = gt.pivot_table(index="tile_no", columns="label", values="f1")
    mat = mat.dropna()
    corr = mat.corr(method="spearman").round(3)
    diff = pd.DataFrame({
        "tile_no": mat.index,
        "平均F1": mat.mean(axis=1).round(4).values,
        "最良F1": mat.max(axis=1).round(4).values,
        "最悪F1": mat.min(axis=1).round(4).values,
        "条件間のばらつき": mat.std(axis=1).round(4).values,
    })
    diff["region"] = region
    return corr, diff.sort_values("平均F1")
