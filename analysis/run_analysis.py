# -*- coding: utf-8 -*-
"""タイル単位の結果をまとめ、CSV・図・レポートを output/ に書き出す。

    python scripts/run_analysis.py
    python scripts/run_analysis.py --region hiroshima
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import analyze, config  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "BIZ UDGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

def _disp(path):
    """表示用のパス。Windows では出力先(SSD)とリポジトリのドライブが違い、
    os.path.relpath が ValueError を投げるので、その場合は絶対パスを返す。"""
    try:
        return os.path.relpath(path, config.PROJECT_DIR)
    except ValueError:
        return str(path)


BLUE = "#2D6E8E"
ORANGE = "#C0662A"
GREY = "#8A9AA4"
REGION_JA = {"hiroshima": "広島", "shimane": "島根"}
REGION_COLOR = {"hiroshima": BLUE, "shimane": ORANGE}

# 図の並び順（融合方式→機構、DEM→SAM、拡張は最後）
ORDER = [
    "Train_Hiroshima_Test_Shimane_OnlyDEM", "Train_Hiroshima_Test_Shimane_OnlySAM",
    "EarlyFusionUNet_DEM_APM", "EarlyFusionUNet_SAM_APM",
    "MIddleFusion_DEM_APM", "MiddleFusion_SAM_APM",
    "FinalFusion_DEM_APM", "FinalFusion_SAM_APM",
    "AttentionUNet_DEM_APM", "AttentionUNet_SAM_APM",
    "TransUNet_SAM_APM",
    "DataOgument_FinalFusion_DEM_APM", "DataOgument_FinalFusion_SAM_APM",
    "DataOgument_AttentionUNet_SAM_APM",
]


def _save(fig, name):
    path = os.path.join(config.FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  図:", _disp(path))
    return path


def _ordered(df, col="condition"):
    """図の並び順。ORDER に載っていない条件は末尾に回す。

    以前は ORDER に無い条件を黙って捨てていたため、
    新しく学習した条件が図から消えていた。条件は増えていくので、
    知らない名前が来ても落とさない。
    """
    have = list(dict.fromkeys(df[col]))
    present = [c for c in ORDER if c in have]
    return present + [c for c in have if c not in ORDER]


# ------------------------------------------------------------------ 図
def fig_tile_f1_distribution(data):
    regions = list(data)
    fig, axes = plt.subplots(1, len(regions), figsize=(7.2 * len(regions), 7.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        df = data[region]
        gt = df[df["has_gt"]]
        conds = _ordered(gt)
        vals = [gt[gt["condition"] == c]["f1"].dropna().values for c in conds]
        labels = [gt[gt["condition"] == c]["label"].iloc[0] for c in conds]
        bp = ax.boxplot(vals, vert=False, showfliers=False, widths=0.6, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(REGION_COLOR[region])
            patch.set_alpha(0.45)
            patch.set_edgecolor(REGION_COLOR[region])
        for med in bp["medians"]:
            med.set_color("#16232E")
            med.set_linewidth(1.6)
        means = [v.mean() for v in vals]
        ax.scatter(means, range(1, len(vals) + 1), color="#16232E", s=18, zorder=3,
                   label="平均")
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("タイルごとの F値")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{REGION_JA[region]}（警戒区域を含むタイルのみ, "
                     f"n={gt['tile_no'].nunique()}）")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("タイル単位の F値の分布", fontsize=14)
    return _save(fig, "fig_tile_f1_distribution.png")


def fig_paired(data, base, other, name, title):
    regions = list(data)
    fig, axes = plt.subplots(1, len(regions), figsize=(5.6 * len(regions), 5.4))
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        df = data[region]
        gt = df[df["has_gt"]]
        a = gt[gt["condition"] == base][["tile_no", "f1"]].rename(columns={"f1": "a"})
        b = gt[gt["condition"] == other][["tile_no", "f1"]].rename(columns={"f1": "b"})
        m = a.merge(b, on="tile_no").dropna()
        if m.empty:
            ax.axis("off")
            continue
        ax.hexbin(m["a"], m["b"], gridsize=42, cmap="Blues", bins="log", mincnt=1)
        ax.plot([0, 1], [0, 1], color="#16232E", lw=1, ls="--")
        d = m["b"] - m["a"]
        win, lose = int((d > 0.01).sum()), int((d < -0.01).sum())
        ax.set_xlabel(f"{base} の F値")
        ax.set_ylabel(f"{other} の F値")
        ax.set_title(f"{REGION_JA[region]}　平均差 {d.mean():+.4f}\n"
                     f"改善 {win} / 悪化 {lose} / 同等 {len(d)-win-lose} タイル", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(title, fontsize=13)
    return _save(fig, name)


def fig_strata(data, strat_df, varname, title, name):
    regions = list(data)
    fig, axes = plt.subplots(1, len(regions), figsize=(7.0 * len(regions), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        s = strat_df[(strat_df["region"] == region) & (strat_df["変数名"] == varname)]
        if s.empty:
            ax.axis("off")
            continue
        conds = _ordered(s)
        cmap = plt.get_cmap("tab20")
        for i, c in enumerate(conds):
            sub = s[s["condition"] == c]
            ax.plot(sub["区分"].astype(str), sub["平均F1"], marker="o", ms=4, lw=1.4,
                    color=cmap(i % 20), label=sub["label"].iloc[0])
        counts = s.groupby("区分", observed=True)["タイル数"].max()
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([f"{k}\n(n={int(v)})" for k, v in counts.items()], fontsize=9)
        ax.set_ylabel("タイル平均 F値")
        ax.grid(alpha=0.3)
        ax.set_title(REGION_JA[region])
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(title, fontsize=13)
    return _save(fig, name)


def fig_shimane_background(matched, summary_s):
    conds = _ordered(matched)
    m = matched.set_index("condition").reindex(conds)
    s = summary_s.set_index("condition").reindex(conds)
    y = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    ax = axes[0]
    ax.barh(y - 0.2, m["全タイル_f1"], height=0.38, color=GREY,
            label="全24,569枚（今回の報告値）")
    ax.barh(y + 0.2, m["警戒区域ありのみ_f1"], height=0.38, color=ORANGE,
            label="警戒区域を含むタイルのみ（広島と同じ選び方）")
    ax.set_yticks(y)
    ax.set_yticklabels(m["label"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("F値")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("島根：テストタイルの選び方を揃えるとF値はどう変わるか")

    ax = axes[1]
    ax.barh(y, s["bg_fp_share"] * 100, color=GREY)
    for i, v in enumerate(s["bg_fp_share"] * 100):
        ax.text(v + 1, y[i], f"{v:.0f}%", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("全過検出画素のうち、警戒区域を含まないタイルで出たものの割合 [%]")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("島根：過検出はどこで出ているか")
    return _save(fig, "fig_shimane_background.png")


def fig_error_ring(errdf):
    regions = sorted(errdf["region"].unique())
    conds = _ordered(errdf)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(conds))
    for i, region in enumerate(regions):
        e = errdf[errdf["region"] == region].set_index("condition").reindex(conds)
        ax.barh(y + (i - 0.5) * 0.38, e["リング誤差の集中度"], height=0.36,
                color=REGION_COLOR[region], label=REGION_JA[region])
        labels = e["label"]
    ax.axvline(1.0, color="#16232E", ls="--", lw=1)
    ax.text(1.02, -0.8, "1.0 = 面積どおり（境界に偏りなし）", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("境界16pxリングへの誤差の集中度（誤差の割合 ÷ 面積の割合 43.75%）")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    ax.set_title("誤差は本当に画像の縁に偏っているか")
    return _save(fig, "fig_error_ring.png")


def fig_fn_share(errdf):
    regions = sorted(errdf["region"].unique())
    conds = _ordered(errdf)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(conds))
    for i, region in enumerate(regions):
        e = errdf[errdf["region"] == region].set_index("condition").reindex(conds)
        ax.barh(y + (i - 0.5) * 0.38, e["FN割合"] * 100, height=0.36,
                color=REGION_COLOR[region], label=REGION_JA[region])
        labels = e["label"]
    ax.axvline(50, color="#16232E", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("誤差に占める見逃し(FN)の割合 [%]　← 過検出寄り / 見逃し寄り →")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    ax.set_title("誤りの向き：見逃しと過検出のバランス")
    return _save(fig, "fig_fn_share.png")


def fig_difficulty(diffs):
    regions = list(diffs)
    fig, axes = plt.subplots(1, len(regions), figsize=(6.4 * len(regions), 4.8))
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        d = diffs[region]
        ax.hist(d["平均F1"], bins=40, color=REGION_COLOR[region], alpha=0.75)
        hard = (d["最良F1"] < 0.3).mean() * 100
        easy = (d["最悪F1"] > 0.7).mean() * 100
        ax.set_xlabel("そのタイルでの全条件平均 F値")
        ax.set_ylabel("タイル数")
        ax.set_title(f"{REGION_JA[region]}　どの条件でもF<0.3: {hard:.1f}%　"
                     f"どの条件でもF>0.7: {easy:.1f}%", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("タイルの難しさの分布（モデルを変えても当たらないタイルがどれだけあるか）",
                 fontsize=13)
    return _save(fig, "fig_tile_difficulty.png")


def fig_agreement(corrs):
    regions = list(corrs)
    fig, axes = plt.subplots(1, len(regions), figsize=(8.2 * len(regions), 7.4))
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        c = corrs[region]
        im = ax.imshow(c.values, vmin=0.4, vmax=1.0, cmap="YlGnBu")
        ax.set_xticks(range(len(c)))
        ax.set_xticklabels(c.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(c)))
        ax.set_yticklabels(c.index, fontsize=8)
        ax.set_title(f"{REGION_JA[region]}　タイル別F値のSpearman相関")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("条件どうしは同じタイルで当たり外れしているか", fontsize=13)
    return _save(fig, "fig_agreement.png")



def fig_attention(regions):
    """Attention Gate の注意係数（あれば）。"""
    frames = []
    for region in regions:
        p = os.path.join(config.REPORT_DIR, f"attention_summary__{region}.csv")
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        return None
    a = pd.concat(frames, ignore_index=True)
    regs = [r for r in regions if r in set(a["region"])]
    fig, axes = plt.subplots(1, len(regs), figsize=(6.8 * len(regs), 5.0), sharey=True)
    axes = np.atleast_1d(axes)
    levels = ["level 1（128px）", "level 2（64px）", "level 3（32px）", "level 4（16px）"]
    styles = {"地形量": "-o", "航空写真": "--s"}
    colors = {"AttentionUNet_SAM_APM": BLUE, "AttentionUNet_DEM_APM": "#6B8E23",
              "DataOgument_AttentionUNet_SAM_APM": ORANGE}
    for ax, region in zip(axes, regs):
        d = a[a["region"] == region]
        for cond, g in d.groupby("condition"):
            for branch, st in styles.items():
                gg = g[g["ブランチ"] == branch].set_index("段").reindex(levels)
                ax.plot(range(4), gg["内外差(選択性)"], st, lw=1.5, ms=5,
                        color=colors.get(cond, GREY),
                        label=f"{cond.replace('DataOgument_', '拡張+')} / {branch}")
        ax.axhline(0, color="#16232E", lw=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["最終段\n128px", "64px", "32px", "最深部\n16px"])
        ax.set_ylabel("注意係数：正解の内側 − 外側")
        ax.grid(alpha=0.3)
        ax.set_title(REGION_JA[region])
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Attention Gate は警戒区域を絞り込めているか（正が絞り込み）", fontsize=13)
    return _save(fig, "fig_attention_selectivity.png")



def fig_region_matrix(rm):
    """地域 × テスト集合の作り方 で F値を並べる。"""
    conds = _ordered(rm)
    d = rm.set_index("condition").reindex(conds)
    y = np.arange(len(conds))
    series = [
        ("広島 警戒区域ありのみ (2,964)", "広島_警戒のみ_f1", BLUE, 0.85),
        ("広島 背景込み・全件相当 (7,631)", "広島_全件相当_f1", BLUE, 0.40),
        ("島根 警戒区域ありのみ (9,679)", "島根_警戒のみ_f1", ORANGE, 0.85),
        ("島根 全件 (24,569)", "島根_全件_f1", ORANGE, 0.40),
    ]
    series = [s for s in series if s[1] in d.columns]
    h = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(13, 7.6))
    for i, (lab, col, color, alpha) in enumerate(series):
        ax.barh(y + (i - (len(series) - 1) / 2) * h, d[col], height=h * 0.92,
                color=color, alpha=alpha, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("F値（画素単位）")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("テスト集合の作り方を揃えて比べる")
    return _save(fig, "fig_region_matrix.png")



def fig_area_vs_instance(iv):
    """面積評価と箇所数評価（被覆方式）を並べる。"""
    regions = [r for r in ["hiroshima", "shimane"] if r in set(iv["region"])]
    if not regions:
        return None
    conds = _ordered(iv)
    fig, axes = plt.subplots(1, len(regions), figsize=(7.6 * len(regions), 7.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, region in zip(axes, regions):
        d = iv[iv["region"] == region].set_index("condition").reindex(conds)
        y = np.arange(len(conds))
        ax.barh(y - 0.2, d["面積F値"], height=0.38, color=BLUE, label="面積評価 F値")
        ax.barh(y + 0.2, d["箇所F値"], height=0.38, color=ORANGE, label="箇所数評価 F値（被覆）")
        ax.set_yticks(y)
        ax.set_yticklabels(d["label"], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("F値")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(REGION_JA[region])
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("面積評価と箇所数評価（cover_gt=0.5, cover_pred=0.3）", fontsize=13)
    return _save(fig, "fig_area_vs_instance.png")


# ------------------------------------------------------------------ レポート
def md_table(df, cols=None, floatfmt=4):
    d = df[cols] if cols else df
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.values]
    return "\n".join([head, sep] + rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both", choices=["hiroshima", "shimane", "both"])
    args = ap.parse_args()
    config.ensure_dirs()

    regions = ["hiroshima", "shimane"] if args.region == "both" else [args.region]
    data, summaries, matched, pairs, strata, errors, corrs, diffs = {}, [], [], [], [], [], {}, {}

    for region in regions:
        print(f"[{region}] 読み込み …")
        df = analyze.load_all(region)
        data[region] = df
        summaries.append(analyze.summary_by_condition(df, region))
        matched.append(analyze.matched_subset_summary(df, region))
        pairs.append(analyze.all_pairs(df, region))
        strata.append(analyze.stratified(df, region))
        errors.append(analyze.error_decomposition(df, region))
        c, d = analyze.difficulty_and_agreement(df, region)
        corrs[region], diffs[region] = c, d
        print(f"[{region}] {df['condition'].nunique()} 条件 × "
              f"{df['tile_no'].nunique()} タイル")

    summary = pd.concat(summaries, ignore_index=True)
    matched_df = pd.concat(matched, ignore_index=True)
    pairs_df = pd.concat(pairs, ignore_index=True)
    strat_df = pd.concat(strata, ignore_index=True)
    err_df = pd.concat(errors, ignore_index=True)

    R = config.REPORT_DIR
    summary.to_csv(os.path.join(R, "summary_by_condition.csv"), index=False)
    matched_df.to_csv(os.path.join(R, "matched_tileset_summary.csv"), index=False)
    pairs_df.to_csv(os.path.join(R, "paired_comparisons.csv"), index=False)
    strat_df.to_csv(os.path.join(R, "stratified_f1.csv"), index=False)
    err_df.to_csv(os.path.join(R, "error_decomposition.csv"), index=False)
    for region in regions:
        corrs[region].to_csv(os.path.join(R, f"agreement_{region}.csv"))
        diffs[region].to_csv(os.path.join(R, f"tile_difficulty_{region}.csv"), index=False)
    print("  CSV:", _disp(R))


    # ---------------------------------------------------------- 地域 × 集合
    rm = None
    if os.path.exists(os.path.join(config.TILES_DIR,
                                   "FinalFusion_SAM_APM__hiroshima_bg.csv")):
        print("広島（背景タイル込み）を集計 …")
        h_pos = data.get("hiroshima")
        if h_pos is None:
            h_pos = analyze.load_all("hiroshima")
        h_all = analyze.load_hiroshima_all(holdout_only=True)
        h_full = analyze.load_hiroshima_all(holdout_only=False)
        s_all = data.get("shimane")
        frames = {"広島_警戒のみ": h_pos, "広島_全件相当": h_all, "広島_背景全件": h_full}
        if s_all is not None:
            frames["島根_警戒のみ"] = s_all[s_all["has_gt"]]
            frames["島根_全件"] = s_all
        rm = analyze.region_matrix(frames)
        for a, b, name in (("広島_警戒のみ", "島根_警戒のみ", "汎化低下_警戒のみ"),
                           ("広島_全件相当", "島根_全件", "汎化低下_全件相当")):
            if f"{a}_f1" in rm.columns and f"{b}_f1" in rm.columns:
                rm[name] = (rm[f"{a}_f1"] - rm[f"{b}_f1"]).round(4)
        rm.to_csv(os.path.join(R, "region_matrix.csv"), index=False)
        fig_region_matrix(rm)


    # ---------------------------------------------------------- 箇所数評価
    inst_view = None
    ipath = os.path.join(R, "instance_summary.csv")
    if os.path.exists(ipath):
        isum = pd.read_csv(ipath)
        primary = analyze.PRIMARY_INSTANCE_SETTING
        cur = isum[isum["設定"] == primary].copy()
        if not cur.empty:
            area = summary[["region", "condition", "label", "通常_f1",
                            "通常_recall", "通常_precision"]].rename(
                columns={"通常_f1": "面積F値", "通常_recall": "面積Recall",
                         "通常_precision": "面積Precision"})
            inst_view = cur.merge(area, on=["region", "condition", "label"], how="left")
            inst_view = inst_view.rename(columns={"箇所_正解数": "正解の箇所数",
                                                  "箇所_予測数": "予測の箇所数"})
            inst_view["面積F−箇所F"] = (inst_view["面積F値"] - inst_view["箇所F値"]).round(4)
            inst_view.to_csv(os.path.join(R, "area_vs_instance.csv"), index=False)
            fig_area_vs_instance(inst_view)
            print("  箇所数評価: 反映済み")

    print("図を作成 …")
    fig_tile_f1_distribution(data)
    fig_paired(data, "FinalFusion_SAM_APM", "AttentionUNet_SAM_APM",
               "fig_paired_attention_sam.png",
               "同じタイルでの比較：Final SAM+APM → Attention SAM+APM")
    fig_paired(data, "FinalFusion_DEM_APM", "AttentionUNet_DEM_APM",
               "fig_paired_attention_dem.png",
               "同じタイルでの比較：Final DEM+APM → Attention DEM+APM")
    fig_paired(data, "FinalFusion_SAM_APM", "DataOgument_FinalFusion_SAM_APM",
               "fig_paired_augment_sam.png",
               "同じタイルでの比較：データ拡張の有無（Final SAM+APM）")
    fig_paired(data, "Train_Hiroshima_Test_Shimane_OnlySAM", "FinalFusion_SAM_APM",
               "fig_paired_airphoto_sam.png",
               "同じタイルでの比較：SAM単一入力 → SAM+APM（航空写真の追加）")
    for var, title, name in [
        ("gt_ratio", "正解の面積割合で層別したタイル平均F値", "fig_strata_gtratio.png"),
        ("slope_mean", "タイル平均傾斜角で層別したタイル平均F値", "fig_strata_slope.png"),
        ("gt_border_ratio", "正解が境界16pxに掛かる割合で層別したタイル平均F値",
         "fig_strata_border.png"),
        ("gt_components", "正解の連結成分数で層別したタイル平均F値", "fig_strata_components.png"),
        ("relief", "起伏量で層別したタイル平均F値", "fig_strata_relief.png"),
    ]:
        fig_strata(data, strat_df, var, title, name)
    if "shimane" in data:
        fig_shimane_background(matched_df[matched_df["region"] == "shimane"],
                               summary[summary["region"] == "shimane"])
    fig_error_ring(err_df)
    fig_fn_share(err_df)
    fig_difficulty(diffs)
    fig_agreement(corrs)
    fig_attention(regions)

    # ---------------------------------------------------------- report.md
    lines = ["# タイル単位の再解析レポート", "",
             "`scripts/run_inference.py` で保存済みの重みを使って推論をやり直し、",
             "1タイルずつ混同行列を取り直した結果をまとめたもの。",
             "条件全体の値は dc5/ModelComparison（ノートブック出力の抽出）と一致する。", ""]

    lines += ["## 1. 条件ごとの集計", "",
              "micro = 画素単位（従来どおり）、macro = タイルごとにF値を出して平均。", ""]
    for region in (regions if not summary.empty and "region" in summary else []):
        s = summary[summary["region"] == region]
        lines += [f"### {REGION_JA[region]}", "",
                  md_table(s, ["label", "n_tiles", "n_tiles_with_gt", "通常_recall",
                               "通常_precision", "通常_f1", "通常_f1_macro",
                               "通常_f1_median", "境界_f1"]), ""]

    if "shimane" in regions:
        lines += ["## 2. 島根のテスト集合を広島と同じ選び方に揃えると", "",
                  "島根は背景タイルも含む全24,569枚で評価していたが、",
                  "広島と同じく「警戒区域を含むタイルだけ」に絞ると次のようになる。", "",
                  md_table(matched_df[matched_df["region"] == "shimane"],
                           ["label", "全タイル_f1", "警戒区域ありのみ_n",
                            "警戒区域ありのみ_recall", "警戒区域ありのみ_precision",
                            "警戒区域ありのみ_f1", "f1差"]), "",
                  "![](../figures/fig_shimane_background.png)", ""]


    if rm is not None:
        cols = [c for c in ["label", "広島_警戒のみ_f1", "広島_全件相当_f1",
                            "島根_警戒のみ_f1", "島根_全件_f1",
                            "汎化低下_警戒のみ", "汎化低下_全件相当"] if c in rm.columns]
        lines += ["## 2b. 広島も背景タイル込みで評価し直す", "",
                  "広島は警戒区域ありのタイルだけを 8:2 に分けていたので、背景タイルにも",
                  "同じ 20 % のホールドアウト率を当てはめて足した（2,964 + 4,667 = 7,631枚、",
                  "陽性タイル率 38.8 %）。島根の 24,569枚（陽性タイル率 39.4 %）とほぼ同じ構成になる。", "",
                  md_table(rm, cols), "",
                  "![](../figures/fig_region_matrix.png)", ""]


    if inst_view is not None:
        lines += ["## 2c. 箇所数評価（被覆方式）", "",
                  "警戒区域を「かたまり」単位で数えたもの。対応づけは被覆方式",
                  f"（{analyze.PRIMARY_INSTANCE_SETTING}）。設計と方式の比較は",
                  "`docs/箇所数評価_設計メモ.md` を参照。", "",
                  md_table(inst_view[[c for c in ["region", "label", "正解の箇所数",
                                                  "予測の箇所数", "箇所Recall",
                                                  "箇所Precision", "箇所F値", "面積F値",
                                                  "面積F−箇所F"] if c in inst_view.columns]]),
                  "",
                  "![](../figures/fig_area_vs_instance.png)", "",
                  "方式ごとの比較は `instance_summary.csv`、飲み込み・分裂の診断は",
                  "`instance_diagnostics.csv`。", ""]

    lines += ["## 3. 同じタイル上での対比較", "",
              "正解を含むタイルだけを対象に、タイルごとのF値の差を取ったもの。",
              "「改善」「悪化」は差が ±0.01 を超えたタイル数。", ""]
    if pairs_df.empty or "region" not in pairs_df.columns:
        # analyze.PAIRS に載っている組のタイルが両方そろっていないとき。
        # PC を移ると旧条件のタイルがキャッシュに無く、ここが空になる。
        lines += ["対比較できる組がありません（`analysis/analyze.py` の `PAIRS` の",
                  "基準・比較の両方のタイルが必要です）。`run_inference.py` で",
                  "両方の条件を推論してから再実行してください。", ""]
    else:
        for region in regions:
            p = pairs_df[pairs_df["region"] == region]
            lines += [f"### {REGION_JA[region]}", "",
                      md_table(p, ["観点", "n", "基準_平均", "比較_平均", "平均差", "中央値差",
                                   "改善タイル数", "悪化タイル数", "大きく改善(>0.1)",
                                   "大きく悪化(<-0.1)", "Wilcoxon_p"]), ""]

    lines += ["## 4. タイルの性質で層別した精度", "", "図を参照。CSV は `stratified_f1.csv`。", "",
              "![](../figures/fig_strata_gtratio.png)", "",
              "![](../figures/fig_strata_border.png)", "",
              "![](../figures/fig_strata_slope.png)", ""]

    lines += ["## 5. 誤差の内訳", "",
              "「リング誤差の集中度」は、境界16pxのリングに落ちた誤差の割合を",
              "リングの面積割合（43.75%）で割ったもの。1.0 なら偏りなし。", ""]
    for region in (regions if not err_df.empty and "region" in err_df else []):
        e = err_df[err_df["region"] == region]
        lines += [f"### {REGION_JA[region]}", "",
                  md_table(e, ["label", "見逃しFN", "過検出FP", "FN割合",
                               "境界リングの誤差割合", "リング誤差の集中度",
                               "抽出面積/正解面積"]), ""]
    lines += ["![](../figures/fig_error_ring.png)", "",
              "![](../figures/fig_fn_share.png)", ""]

    if os.path.exists(os.path.join(config.FIGURES_DIR, "fig_attention_selectivity.png")):
        lines += ["## 6. Attention Gate は何を見ているか", "",
                  "スキップ接続に掛かる注意係数(0-1)の、正解領域の内側と外側での平均の差。",
                  "正なら警戒区域側を強く通している。CSV は `attention_summary__<地域>.csv`。", "",
                  "![](../figures/fig_attention_selectivity.png)", ""]

    lines += ["## 7. 条件間の一致度とタイルの難しさ", "",
              "![](../figures/fig_agreement.png)", "",
              "![](../figures/fig_tile_difficulty.png)", ""]
    for region in regions:
        d = diffs[region]
        lines += [f"- {REGION_JA[region]}：どの条件でも F<0.3 のタイル "
                  f"{(d['最良F1'] < 0.3).mean()*100:.1f}%、"
                  f"どの条件でも F>0.7 のタイル {(d['最悪F1'] > 0.7).mean()*100:.1f}%、"
                  f"条件間のばらつき（標準偏差）の中央値 {d['条件間のばらつき'].median():.4f}"]
    lines += [""]

    path = os.path.join(R, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  レポート:", _disp(path))


if __name__ == "__main__":
    main()
