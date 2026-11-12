# -*- coding: utf-8 -*-
"""背景タイルの誤検出を**指定区域と並べる**（`展望 47.3`・`47.4`）。**推論しない。**

なぜ要るか
----------
`47.3`（上位群は指定区域と見分けがつかない）と `47.4`（隣のはみ出しではない）の
表は、`run_bg_blobs.py` と `run_triage_curve.py` と `run_building_miss.py` が
保存した CSV を**手で突き合わせて**作っていた。
CLAUDE.md の 6「数字を docs に書くときは、出した計算をコードに残す」を
満たしていなかったので、ここに起こす。

2026-11-08 のレビュー 3.1 で見つかった**交絡も一緒に出す** ──
オーバーラップ学習は近傍に背景タイルを貼るので、**学習タイルに接した背景タイルは
外側の最大 64px がマスク0（負例）として学習に入っている。**
`near_gt`（隣に指定区域がある）とこれは相関するので、分けて数える。

読むもの（すべて `cache/analysis/`。**無ければ作り方を言って止まる**）
------------------------------------------------------------------
    triage/<条件>__hiroshima__blobs.csv       予測のかたまり（傾斜つき）
    triage/<条件>__hiroshima_bg__blobs.csv    背景タイルの誤検出
    building/<条件>__hiroshima__pred.csv      予測のかたまり（建物つき）
    building/<条件>__hiroshima__gt.csv        正解のかたまり（建物つき）

統計の取り方（**published の表と同じにしてある**）
    傾斜は**平均**、面積とロジットは**中位**、「30度以上」は割合

    python analysis/run_bg_compare.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths                                             # noqa: E402

# 指定基準（急傾斜地崩壊危険区域）
SLOPE_DEG, HOUSES = 30.0, 5
TOPS = (100, 500, 1000)


def need(path, how):
    if not path.exists():
        sys.exit(f"{path} がありません。先に:\n    {how}")
    return pd.read_csv(path)


def slope_row(label, d):
    return dict(群=label, 件数=len(d),
                傾斜90=d.slope_p90.mean(),
                度30以上=(d.slope_p90 >= SLOPE_DEG).mean(),
                傾斜最大=d.slope_max.mean(),
                中位面積=d.area.median(),
                ロジット中位=d.logit_max.median())


def house_row(label, d):
    return dict(群=label, 件数=len(d),
                戸5以上=(d.house_50 >= HOUSES).mean(),
                棟数平均=d.house_50.mean(), 棟数中位=d.house_50.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    a = ap.parse_args()
    c = a.condition
    tri = paths.cache_dir("analysis") / "triage"
    bld = paths.cache_dir("analysis") / "building"

    pb = need(tri / f"{c}__hiroshima__blobs.csv",
              "python analysis/run_triage_curve.py")
    bg = need(tri / f"{c}__hiroshima_bg__blobs.csv",
              "python analysis/run_bg_blobs.py")
    hp = need(bld / f"{c}__hiroshima__pred.csv",
              "python analysis/run_building_miss.py --record")
    hg = need(bld / f"{c}__hiroshima__gt.csv",
              "python analysis/run_building_miss.py --record")
    print(f"条件 {c}")
    print(f"  予測 {len(pb):,}（的中 {int(pb.tp.sum()):,}）/ 背景の誤検出 "
          f"{len(bg):,} / 建物つき 予測 {len(hp):,}・正解 {len(hg):,}")

    # 突き合わせ ── 同じかたまりを指しているか
    m = pb.merge(hp, on=("tile", "blob"), suffixes=("", "_b"))
    assert len(m) == len(pb), f"triage と building で件数が違う（{len(m)} / {len(pb)}）"
    assert (m.area == m.area_b).all() and (m.tp == m.tp_b).all(), \
        "triage と building で面積か的中が食い違う"
    print("  面積と的中が全件一致（同じかたまりを見ている）")

    bgs = bg.sort_values("logit_max", ascending=False)

    # ---------------------------------------------------------------- 47.3
    print("\n" + "=" * 86)
    print("【47.3】地形 ── 背景タイルの上位は、指定区域と見分けがつくか")
    print("=" * 86)
    rows = [slope_row("警戒タイル・的中（＝指定区域の中）", pb[pb.tp == 1]),
            slope_row("警戒タイル・外れ", pb[pb.tp == 0]),
            slope_row("背景タイル・誤検出 全件", bg)]
    for n in sorted(TOPS, reverse=True):
        rows.append(slope_row(f"背景タイル・誤検出 上位{n:,}", bgs.head(n)))
    t = pd.DataFrame(rows)
    print(f"{'群':<34}{'件数':>8}{'傾斜90%点':>11}{'30度以上':>10}"
          f"{'傾斜最大':>10}{'中位面積':>10}{'ロジット中位':>13}")
    for _, r in t.iterrows():
        print(f"{r.群:<34}{int(r.件数):>8,}{r.傾斜90:>11.2f}"
              f"{100 * r.度30以上:>9.1f}%{r.傾斜最大:>10.2f}"
              f"{r.中位面積:>10.0f}{r.ロジット中位:>13.2f}")

    print("\n  人家（**「5戸以上」は建物ラスタの連結成分。棟数ではない。**")
    print("  指定区域そのものに当てても満たすのは一部なので、比でだけ読む）")
    hrows = [house_row("正解の塊（指定区域）", hg),
             house_row("うち検出できた", hg[hg.found == 1]),
             house_row("うち見逃した", hg[hg.found == 0]),
             house_row("予測の塊 的中", hp[hp.tp == 1]),
             house_row("予測の塊 外れ（警戒タイル内）", hp[hp.tp == 0]),
             house_row("背景タイルの誤検出 全件", bg)]
    for n in sorted(TOPS, reverse=True):
        hrows.append(house_row(f"背景タイルの誤検出 上位{n:,}", bgs.head(n)))
    print(f"{'群':<34}{'件数':>8}{'50m内に5戸以上':>16}{'棟数 平均':>11}{'中位':>7}")
    for _, r in pd.DataFrame(hrows).iterrows():
        print(f"{r.群:<34}{int(r.件数):>8,}{100 * r.戸5以上:>15.2f}%"
              f"{r.棟数平均:>11.2f}{r.棟数中位:>7.0f}")

    # ---------------------------------------------------------------- 47.4
    print("\n" + "=" * 86)
    print("【47.4】隣の区域のはみ出しか ── **学習タイルとの接触も一緒に見る**")
    print("=" * 86)
    has_tr = "near_train" in bg.columns and bg.near_train.notna().all()
    if not has_tr:
        print("  ※ `near_train` が CSV にありません（2026-11-08 より前の run）。")
        print("     python analysis/run_bg_blobs.py で作り直すと出ます。")
    print(f"{'群':<24}{'件数':>8}{'隣に区域あり':>14}"
          f"{'隣が学習タイル':>16}{'どちらも無い':>14}")
    for label, d in ([(f"上位{n:,}", bgs.head(n)) for n in sorted(TOPS)]
                     + [("全件", bg)]):
        ng = d.near_gt.mean()
        if has_tr:
            nt = d.near_train.mean()
            neither = (~d.near_gt.astype(bool)
                       & ~d.near_train.astype(bool)).mean()
            print(f"{label:<24}{len(d):>8,}{100 * ng:>13.1f}%"
                  f"{100 * nt:>15.1f}%{100 * neither:>13.1f}%")
        else:
            print(f"{label:<24}{len(d):>8,}{100 * ng:>13.1f}%"
                  f"{'—':>16}{'—':>14}")

    print("\n  **`near_gt` が下がることだけを「はみ出しではない」の根拠にしない。**")
    print("  オーバーラップ学習は近傍に背景タイルを貼るので、学習タイルに")
    print("  接した背景タイルは外側の最大 64px がマスク0（負例）として")
    print("  学習に入っている。**指定区域の隣ほどそうなりやすい**ので、")
    print("  誤検出が少なく出る向きに偏る（2026-11-08 のレビュー 3.1）。")

    if has_tr:
        print("\n  **学習タイルに接していないものだけで取り直す**（交絡を外す）")
        ind = bgs[~bgs.near_train.astype(bool)]
        print(f"  対象 {len(ind):,} 件（全 {len(bg):,} 件のうち "
              f"{100 * len(ind) / len(bg):.1f}%）")
        rows = [slope_row(f"上位{n:,}（学習タイルに接しない）", ind.head(n))
                for n in sorted(TOPS, reverse=True)]
        rows.append(slope_row("的中（＝指定区域の中）", pb[pb.tp == 1]))
        print(f"{'群':<34}{'件数':>8}{'傾斜90%点':>11}{'30度以上':>10}"
              f"{'傾斜最大':>10}{'中位面積':>10}{'ロジット中位':>13}")
        for _, r in pd.DataFrame(rows).iterrows():
            print(f"{r.群:<34}{int(r.件数):>8,}{r.傾斜90:>11.2f}"
                  f"{100 * r.度30以上:>9.1f}%{r.傾斜最大:>10.2f}"
                  f"{r.中位面積:>10.0f}{r.ロジット中位:>13.2f}")
        h = [house_row(f"上位{n:,}（学習タイルに接しない）", ind.head(n))
             for n in sorted(TOPS, reverse=True)]
        h.append(house_row("正解の塊（指定区域）", hg))
        print(f"{'群':<34}{'件数':>8}{'50m内に5戸以上':>16}{'棟数 平均':>11}")
        for _, r in pd.DataFrame(h).iterrows():
            print(f"{r.群:<34}{int(r.件数):>8,}{100 * r.戸5以上:>15.2f}%"
                  f"{r.棟数平均:>11.2f}")

    # ---------------------------------------------------------------- 基準
    print("\n" + "=" * 86)
    print(f"【参考】指定基準（傾斜90%点 >= {SLOPE_DEG:.0f}度 かつ "
          f"50m内の棟数 >= {HOUSES}）を満たす割合")
    print("=" * 86)
    for label, d in ([("正解の塊（指定区域）", hg)]
                     + [(f"背景 上位{n:,}", bgs.head(n)) for n in sorted(TOPS)]
                     + [("背景 全件", bg)]):
        if "slope_p90" in d.columns:
            ok = (d.slope_p90 >= SLOPE_DEG) & (d.house_50 >= HOUSES)
            print(f"  {label:<24}{int(ok.sum()):>8,} / {len(d):>8,}"
                  f"（{100 * ok.mean():>5.2f}%）")
        else:
            ok = d.house_50 >= HOUSES
            print(f"  {label:<24}{int(ok.sum()):>8,} / {len(d):>8,}"
                  f"（{100 * ok.mean():>5.2f}%・**傾斜は列に無いので人家だけ**）")


if __name__ == "__main__":
    main()
