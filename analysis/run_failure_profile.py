# -*- coding: utf-8 -*-
"""まったく当てられないタイル（TP=0）の正体を数える。

`docs/失敗の所在.md` 2節の表と、発表スライドの「失敗の61%は外周16px帯」を
出しているのがこれ。初版は docs の付録に貼った一時スニペットだったので、
再現と再計算ができるようプログラムにした。

推論もモデルの読み込みもしない。既にある2つのCSVを突き合わせるだけなので速い。

    analysis/run_features.py --region hiroshima     # タイルの性質（1回でよい）
    analysis/run_inference.py --region hiroshima    # タイルごとの tp/fp/fn

が先に要る。そのうえで

    python analysis/run_failure_profile.py --conditions FinalFusion_SAM_APM_bg10
    python analysis/run_failure_profile.py --region both --conditions A B --csv

判定の中身は3つ。

  正解が全部 外周16px帯にある   gt_border_ratio >= 0.999
      gt_border_ratio は analysis/features.py の _gt_shape_stats が作る。
      正解画素のうち、四辺から 16px（config.BORDER_CROP）の帯に入る割合。
  傾斜コントラストが負          slope_contrast < 0
      正解の中の平均傾斜 − 外の平均傾斜。負なら正解のほうが緩斜面。
  小さい                        gt_mean_comp < 300（タイル内の箇所の平均画素数）

「対照」列は同じ条件で TP>0 だったタイルの値。外周帯だけは対照との差が
6倍あり、これが「失敗の主因はタイルの切り方」という読みの根拠になっている。
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import config                       # noqa: E402

BORDER_FULL = 0.999       # 正解画素が「全部」外周帯にあるとみなすしきい値
SMALL_PX = 300            # 「小さい」箇所の平均画素数

TESTS = [
    ("正解が全部 外周16px帯にある", lambda d: d["gt_border_ratio"] >= BORDER_FULL),
    ("傾斜コントラストが負（正解の方が緩斜面）", lambda d: d["slope_contrast"] < 0),
    (f"小さい（平均 {SMALL_PX}px 未満）", lambda d: d["gt_mean_comp"] < SMALL_PX),
]


def _need(path, how):
    if not os.path.exists(path):
        sys.exit(f"{path} がありません。先に {how} を実行してください。")
    return path


def profile(condition, region):
    """1条件ぶんの内訳を返す。(表の行, 要約) のタプル。"""
    fpath = _need(os.path.join(config.FEATURES_DIR, f"tile_features__{region}.csv"),
                  f"analysis/run_features.py --region {region}")
    tpath = _need(os.path.join(config.TILES_DIR, f"{condition}__{region}.csv"),
                  f"analysis/run_inference.py --region {region} "
                  f"--conditions {condition}")

    f = pd.read_csv(fpath)
    t = pd.read_csv(tpath)
    m = t.merge(f, on="tile_no", how="left")
    m = m[(m["tp"] + m["fn"]) > 0]          # 正解のあるタイルだけを母数にする
    zero, det = m[m["tp"] == 0], m[m["tp"] > 0]
    if zero.empty:
        return [], dict(condition=condition, region=region, n_tiles=len(m),
                        n_tp0=0, ratio_tp0=0.0)

    rows, any_hit = [], None
    for label, fn in TESTS:
        hit = fn(zero)
        any_hit = hit if any_hit is None else (any_hit | hit)
        rows.append({
            "condition": condition, "region": region, "条件": label,
            "枚数": int(hit.sum()),
            "割合": round(float(hit.mean()), 4),
            "検出できたタイルでは": round(float(fn(det).mean()), 4),
        })
    rows.append({
        "condition": condition, "region": region, "条件": "いずれかに該当",
        "枚数": int(any_hit.sum()), "割合": round(float(any_hit.mean()), 4),
        "検出できたタイルでは": None,
    })
    summary = dict(condition=condition, region=region, n_tiles=len(m),
                   n_tp0=len(zero), ratio_tp0=round(len(zero) / len(m), 4))
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="省略すると active な条件すべて（タイルCSVがあるものだけ）")
    ap.add_argument("--csv", action="store_true",
                    help="report/failure_profile.csv にも書き出す")
    args = ap.parse_args()

    regions = ["hiroshima", "shimane"] if args.region == "both" else [args.region]
    names = args.conditions or sorted(c.name for c in config.CONDITIONS)

    all_rows = []
    for region in regions:
        for name in names:
            tpath = os.path.join(config.TILES_DIR, f"{name}__{region}.csv")
            if args.conditions is None and not os.path.exists(tpath):
                continue        # 一括実行のときは推論していない条件を黙って飛ばす
            rows, sm = profile(name, region)
            if not rows:
                print(f"\n[{region}] {name}: TP=0 のタイルなし")
                continue
            all_rows += rows
            print(f"\n[{region}] {name}")
            print(f"  正解のあるタイル {sm['n_tiles']} / "
                  f"TP=0 {sm['n_tp0']} 枚（{100*sm['ratio_tp0']:.1f}%）")
            print(f"  {'条件':<42}{'枚数':>6}{'割合':>8}{'検出できたタイルでは':>22}")
            for r in rows:
                ref = "—" if r["検出できたタイルでは"] is None \
                    else f"{100*r['検出できたタイルでは']:.1f}%"
                print(f"  {r['条件']:<42}{r['枚数']:>6}"
                      f"{100*r['割合']:>7.1f}%{ref:>22}")

    if args.csv and all_rows:
        os.makedirs(config.REPORT_DIR, exist_ok=True)
        out = os.path.join(config.REPORT_DIR, "failure_profile.csv")
        pd.DataFrame(all_rows).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n書き出し: {out}")


if __name__ == "__main__":
    main()
