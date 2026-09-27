# -*- coding: utf-8 -*-
"""
複数条件の error_stats.csv を横断比較し、
「入力データの種類によらず問題が残るタイル」を自動的に洗い出すツール。

本レポート5章「4条件に共通する傾向」の分析（FN/FP画素数の相関、
過検出・見逃しの残存率など）を、run_analysis.py 実行後にいつでも
再実行・別条件の組み合わせで再現できるようにしたもの。

使い方:
    python compare_conditions.py DemOnly KeisyaOnly
    python compare_conditions.py DemOnly KeisyaOnly DEM_AirPhoto SAM_AirPhoto
    python compare_conditions.py --all             # lib/conditions.py の全条件

出力:
    - 各条件ペアのFN/FP画素数の相関係数
    - 条件Aで問題（high_FN/high_FP）だったタイルが条件Bでも問題のままである割合
    - 全指定条件で共通して問題が残るタイル（"chronic" タイル）のNo一覧を
      compare_output/chronic_high_FN.csv, chronic_high_FP.csv に保存
      （現地確認・正解データ側の妥当性検証の対象候補として使える）
"""
import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.conditions import CONDITIONS

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_output")


def load_condition_df(name):
    cfg = CONDITIONS[name]
    path = os.path.join(cfg["out_dir"], "error_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} が見つかりません。先に `python run_analysis.py {name}` を実行してください。"
        )
    df = pd.read_csv(path)[["No", "FN", "FP", "recall", "precision", "f1", "bucket"]]
    return df.rename(columns={c: f"{c}_{name}" for c in ["FN", "FP", "recall", "precision", "f1", "bucket"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="*", help="比較する条件名（lib/conditions.py のキー）")
    ap.add_argument("--all", action="store_true", help="定義済みの全条件を比較する")
    args = ap.parse_args()

    names = list(CONDITIONS.keys()) if args.all else args.conditions
    if len(names) < 2:
        print("2つ以上の条件を指定してください。例: python compare_conditions.py DemOnly KeisyaOnly")
        print(f"利用可能な条件: {list(CONDITIONS.keys())}")
        return

    dfs = [load_condition_df(n) for n in names]
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="No", how="inner")
    print(f"共通するテストタイル数: {len(merged)} 件\n")

    print("=== ペアごとのFN/FP画素数の相関係数 ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            fn_corr = merged[f"FN_{a}"].corr(merged[f"FN_{b}"])
            fp_corr = merged[f"FP_{a}"].corr(merged[f"FP_{b}"])
            print(f"  {a} vs {b}:  FN相関={fn_corr:.3f}  FP相関={fp_corr:.3f}")

    print("\n=== 「問題あり」タイルの残存率（1番目に挙げた条件を基準） ===")
    base = names[0]
    for other in names[1:]:
        for bucket in ("high_FN", "high_FP"):
            was_bad = merged[merged[f"bucket_{base}"] == bucket]
            if len(was_bad) == 0:
                continue
            still_bad = (was_bad[f"bucket_{other}"] == bucket).sum()
            print(f"  {base}で{bucket}だった{len(was_bad)}件のうち、"
                  f"{other}でも{bucket}のまま: {still_bad}件 ({100*still_bad/len(was_bad):.1f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    for bucket, fname in (("high_FN", "chronic_high_FN.csv"), ("high_FP", "chronic_high_FP.csv"),
                          ("good", "chronic_good.csv")):
        mask = pd.Series(True, index=merged.index)
        for n in names:
            mask &= (merged[f"bucket_{n}"] == bucket)
        chronic = merged[mask].copy()
        if bucket == "good" and len(chronic) > 0:
            f1_cols = [f"f1_{n}" for n in names]
            chronic["f1_mean"] = chronic[f1_cols].mean(axis=1)
            chronic = chronic.sort_values("f1_mean", ascending=False)
        chronic.to_csv(os.path.join(OUT_DIR, fname), index=False, encoding="utf-8-sig")
        label = "良好" if bucket == "good" else bucket
        print(f"\n全{len(names)}条件で共通して{label}なタイル: {len(chronic)}件 "
              f"({100*len(chronic)/len(merged):.1f}%) -> {fname} に保存")


if __name__ == "__main__":
    main()
