# -*- coding: utf-8 -*-
"""合意数は「盲」の群に届くのか（`K2` / `42`）。

なぜ要るか
----------
`41` で、見逃しの半分以上は**予測と1画素も重ならず**、しかも
かたまり全体でロジットが負（確率 10⁻⁵ 未満）だと分かった。
後処理（しきい値・連結・刈り取り）で動かせる余地はすべて帯の中である。

残っている唯一の方向が**合意数**である（`37` で唯一帯を超えた +0.0105）。
だが合意数の効きが**刈り取り側だけ**なら、盲の群には届かない。
`J2`（合意数を3本→7本）に進む前に、ここを確かめる。

    和集合が盲の群を拾う  → `J2` は再現率に触れる。進む価値がある
    どの構成員も光らない  → 合意数は適合率だけの話。`J2` は再現率を上げない

測り方
------
`UNION3` の3本をそれぞれ推論し、**正解のかたまり1件ごと**に
各構成員がどれだけ覆ったか（`cov_<略称>`）を出す。和集合は
二値マスクの論理和で作る（`21` の1段目の和集合と同じ）。

    **正解マスクが3本で一致することを必ず確かめる。**
    pkl が違う（sam_apm / dem_apm_normalized / sam_apm_lu）ので、
    タイルの並びか正解の作り方がずれていれば、かたまりの付番も狂う。

    python analysis/run_union_blind.py --region hiroshima --save
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_overlap_prune import infer, score                  # noqa: E402
from analysis.run_stage2_room import STAGE1, UNION3                  # noqa: E402
from dc5lib import paths                                             # noqa: E402
from dc5lib.device import pick_device                                # noqa: E402
from dc5lib.instance_eval import label_instances, overlap_matrix     # noqa: E402
from dc5lib.registry import get                                      # noqa: E402

COV_GT = 0.5        # `run_miss_coverage.py` と同じ。変えると比較できない
MIN_SIZE = 10
BAND = {"hiroshima": 0.0071, "shimane": 0.0198}

# 無印の union_blind__<地域>.csv に入っている構成員（略称・この順）。
# **44.4/44.6 の数字の出どころなので、構成員が違うまま無印で上書きさせない。**
RECORDED5 = ["SAM_APM", "DEM_APM", "SAM_APM_LU", "SAM_APM_BVB",
             "SAM_APM_BLD"]


def guard_save(save, tag, shorts):
    """無印の CSV を別の構成員で上書きしようとしていたら止める。

    **推論の前に呼ぶ。** あとで呼ぶと1時間回してから止まる。
    """
    if not save or tag or shorts == RECORDED5:
        return
    sys.exit(
        "無印の union_blind__<地域>.csv は **44.4/44.6 の数字の出どころ**です。" + chr(10)
        + f"  記録済みの構成員 {RECORDED5}" + chr(10)
        + f"  いまの構成員     {shorts}" + chr(10)
        + "構成員が違うまま無印で保存すると過去の記述が復元できなくなります。"
          "--tag を付けてください（**推論を始める前に止めました**）")


def short(name):
    """`FinalFusion_SAM_APM_LU_OverlapTrain` -> `SAM_APM_LU`。"""
    return name.replace("FinalFusion_", "").replace("_OverlapTrain", "")


def blob_cov(GT, PB, LG=None, labels=None):
    """正解のかたまりごとの (被覆率, 中のロジット最大) を返す。

    `labels` を渡すと**その付番を使い回す**。正解は全員で同じなので、
    毎回ラベル付けし直すと遅いうえ、万一ずれたときに気づけない。
    `LG` を省くとロジットは NaN になる（和集合にロジットは無い）。
    """
    cov, lgm, lab_out = [], [], []
    for k in range(len(GT)):
        if labels is None:
            gl, gs = label_instances(GT[k], 8, MIN_SIZE)
            lab_out.append((gl, gs))
        else:
            gl, gs = labels[k]
        if not len(gs):
            continue
        plab, psz = label_instances(PB[k], 8, MIN_SIZE)
        O = (overlap_matrix(gl, len(gs), plab, len(psz))
             if len(psz) else np.zeros((len(gs), 0), np.int64))
        lg = None if LG is None else LG[k]
        for i in range(len(gs)):
            cov.append(float(O[i].sum()) / gs[i])
            lgm.append(float("nan") if lg is None
                       else float(lg[gl == (i + 1)].max()))
    return np.array(cov), np.array(lgm), lab_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane"])
    ap.add_argument("--members", nargs="*", default=[STAGE1] + UNION3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--record", action="store_true",
                    help="和集合そのものを results/evals/ に記録する"
                         "（--name が要る。docs/設計_アンサンブルの記録.md の 3.4）")
    ap.add_argument("--name", default="",
                    help="記録に使うアンサンブルの名前（unions/<これ>.yaml の名前）。"
                         "condition は Union_<これ> になる")
    ap.add_argument("--tag", default="",
                    help="保存名の後ろに付ける印（union_blind__<地域>__<印>.csv）。"
                         "**構成員が記録済みの5本と違うときは必須**")
    a = ap.parse_args()

    device = torch.device(a.device) if a.device else pick_device()
    band = BAND[a.region]
    names = list(dict.fromkeys(a.members))          # 重複を落として順を保つ
    print(f"地域 {a.region} / device {device}")
    print(f"構成員 {len(names)} 本")
    for n in names:
        print(f"  {short(n)}")

    guard_save(a.save, a.tag, [short(n) for n in names])
    # **推論の前に止める。** あとで止めると1時間回してから怒られる
    if a.record and not a.name:
        sys.exit("--record には --name が要ります（condition を Union_<名前> にするため）。"
                 "unions/*.yaml の名前を渡してください")
    if a.record and not a.save:
        sys.exit("--record には --save が要ります（出どころの CSV 名を記録に残すため）")

    base = None
    labels = None
    U = None                                        # 和集合の二値マスク
    d = {}
    areas = None
    rows_single = []

    for idx, n in enumerate(names):
        print(f"\n--- {short(n)} を推論します …", flush=True)
        GT, PB, LG = infer(get(n), a.region, device)
        if base is None:
            base = GT
            labels = None
        elif not (GT.shape == base.shape and (GT == base).all()):
            sys.exit(f"{short(n)} の正解マスクが {short(names[0])} と違います。"
                     f"タイルの並びか正解の作り方がずれています")
        cov, lgm, lab = blob_cov(GT, PB, LG, labels=labels)
        if labels is None:
            labels = lab
            areas = np.array([int(s) for _, gs in lab for s in gs])
        d[f"cov_{short(n)}"] = cov
        d[f"lg_{short(n)}"] = lgm
        R, P, F, npred = score(GT, PB)
        rows_single.append((short(n), R, P, F, npred))
        U = PB.copy() if U is None else (U | PB)
        del PB, LG, GT
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n--- 和集合を測ります …", flush=True)
    covU, _, _ = blob_cov(base, U, labels=labels)
    RU, PU, FU, nU = score(base, U)

    df = pd.DataFrame(d)
    df["area"] = areas
    df["cov_union"] = covU
    cc = [f"cov_{short(n)}" for n in names]
    df["n_light"] = (df[cc] > 0).sum(axis=1)
    b = cc[0]
    print(f"\n正解のかたまり {len(df):,}")

    print("\n" + "=" * 74)
    print("【1】単体と和集合")
    print("=" * 74)
    print(f"{'構成員':<22}{'箇所R':>9}{'箇所P':>9}{'箇所F':>9}{'予測数':>10}")
    for nm, R, P, F, npd in rows_single:
        print(f"{nm:<22}{R:>9.4f}{P:>9.4f}{F:>9.4f}{npd:>10,}")
    _, R0, P0, F0, _ = rows_single[0]
    print(f"{'**和集合**':<22}{RU:>9.4f}{PU:>9.4f}{FU:>9.4f}{nU:>10,}")
    print(f"\n  和集合の 箇所R は基準より {RU - R0:+.4f}"
          f"（{(RU - R0) / band:.1f}帯）、箇所P は {PU - P0:+.4f}")
    print(f"  **完璧な刈り取りの上限** 箇所F {2 * RU / (1 + RU):.4f}"
          f"（基準の上限 {2 * R0 / (1 + R0):.4f}）")

    print("\n" + "=" * 74)
    print("【2】基準が『盲』だったかたまりを、他の構成員は拾うか")
    print("=" * 74)
    blind = df[df[b] <= 0]
    print(f"  基準（{short(names[0])}）で覆い 0 のかたまり {len(blind):,} 件")
    other = [c for c in cc[1:]]
    any_light = (blind[other] > 0).any(axis=1)
    print(f"{'':<34}{'件数':>9}{'割合':>9}")
    print(f"{'他のどれかが1画素でも光る':<34}"
          f"{int(any_light.sum()):>9,}{100 * any_light.mean():>8.1f}%")
    print(f"{'**全員まったく光らない**':<34}"
          f"{int((~any_light).sum()):>9,}{100 * (~any_light).mean():>8.1f}%")
    rec = blind.cov_union >= COV_GT
    print(f"{'和集合で検出になる（覆い50%以上）':<34}"
          f"{int(rec.sum()):>9,}{100 * rec.mean():>8.1f}%")
    print(f"\n  **箇所R にして {int(rec.sum()) / len(df):+.4f}"
          f"（{int(rec.sum()) / len(df) / band:.1f}帯）**")

    print("\n" + "=" * 74)
    print("【3】その群での各構成員のロジット最大（0 が確率 0.5）")
    print("=" * 74)
    print(f"{'構成員':<22}{'平均':>10}{'中央':>10}{'負の割合':>11}"
          f"{'覆い>0 の%':>12}")
    for n in names:
        s = short(n)
        v = blind[f"lg_{s}"]
        print(f"{s:<22}{v.mean():>10.3f}{v.median():>10.3f}"
              f"{100 * (v <= 0).mean():>10.1f}%"
              f"{100 * (blind[f'cov_{s}'] > 0).mean():>11.1f}%")
    print("\n  **全員が負なら、どの構成員も見ていない。**"
          " 構成員を増やしても届かない。")

    print("\n" + "=" * 74)
    print("【4】面積層別 — 小さい正解は和集合でも拾えないか")
    print("=" * 74)
    df["q"] = pd.qcut(df.area, 5, labels=False, duplicates="drop")
    print(f"{'面積層':>6}{'中央面積':>10}{'件数':>8}{'基準の箇所R':>13}"
          f"{'和集合の箇所R':>15}{'伸び':>9}{'帯':>7}")
    for q in sorted(df.q.unique()):
        s = df[df.q == q]
        r0 = float((s[b] >= COV_GT).mean())
        ru = float((s.cov_union >= COV_GT).mean())
        g = (ru - r0) * len(s) / len(df)
        print(f"{int(q):>6}{s.area.median():>10.0f}{len(s):>8,}"
              f"{r0:>13.4f}{ru:>15.4f}{g:>+9.4f}{g / band:>7.1f}")

    print("\n" + "=" * 74)
    print("【5】何本が光っているか（合意数そのもの）")
    print("=" * 74)
    print(f"{'光った本数':>12}{'件数':>10}{'割合':>9}{'和集合での検出率':>18}")
    for k in range(len(names) + 1):
        s = df[df.n_light == k]
        if not len(s):
            continue
        print(f"{k:>12}{len(s):>10,}{100 * len(s) / len(df):>8.1f}%"
              f"{float((s.cov_union >= COV_GT).mean()):>18.4f}")

    if a.save:
        od = paths.cache_dir("analysis") / "coverage"
        od.mkdir(parents=True, exist_ok=True)
        p = od / (f"union_blind__{a.region}.csv" if not a.tag
                  else f"union_blind__{a.region}__{a.tag}.csv")
        df.to_csv(p, index=False, encoding="utf-8")
        print(f"\n保存: {p}")
        if a.record:
            from dc5lib import union as _uni
            _uni.record(a.name, names, a.region, (RU, PU, FU, nU),
                        len(df), int((df[cc] <= 0).all(axis=1).sum()),
                        p.name)


if __name__ == "__main__":
    main()
