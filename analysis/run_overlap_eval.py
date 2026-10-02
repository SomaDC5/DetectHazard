# -*- coding: utf-8 -*-
"""オーバーラップ推論で評価し、従来（タイル1枚）の値と並べて出す。

タイルの境界で切れた箇所が検出できない問題への対処。
docs/失敗の所在.md 2節のとおり、1画素も当てられないタイルの61%は
「正解が全部タイル外周16px帯にある」。3x3 の近傍を貼り合わせ、
64px ずつずらした9窓で推論して統合すると、その境界断片が窓の中心に来る。

統合は中心ほど重い cos 窓での平均。単純平均は境界の予測を薄めて Recall を
落とし、最大は誤検出を拾って Precision を落とす（失敗の所在.md 4.5節の実測）。

    python analysis/run_overlap_eval.py --region hiroshima --conditions <条件名>
    python analysis/run_overlap_eval.py --region both --conditions <条件名> --record

従来の評価（run_inference.py）は変更していない。こちらは別軸の評価として
results/evals に scope="オーバーラップ" で記録する（--record を付けたとき）。
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dc5lib import results as dc5results                      # noqa: E402
from dc5lib.instance_eval import PRIMARY, InstanceEvaluator    # noqa: E402
from dc5lib.models import build_model, load_weights            # noqa: E402
from dc5lib.paths import dataset_path                          # noqa: E402
from dc5lib.registry import get                                # noqa: E402
from analysis.data import load_pkl                             # noqa: E402

DLON, DLAT = 0.00625, 0.00416667      # タイルの格子間隔（広島・島根で共通）
OFFSETS = [(dy, dx) for dy in (-64, 0, 64) for dx in (-64, 0, 64)]
REGION_DIR = {"hiroshima": "Hiroshima", "shimane": "Shimane"}


def cos_window(n=128):
    """中心ほど重い重み。sliding-window 推論の定石。"""
    yy, xx = np.mgrid[0:n, 0:n]
    w = np.cos((yy - (n - 1) / 2) / n * np.pi) * np.cos((xx - (n - 1) / 2) / n * np.pi)
    return np.clip(w, 1e-3, None).astype(np.float32)


def grid_index(ds):
    """GeoInfo から格子座標を作り、(i, j) -> タイル番号 の辞書を返す。"""
    geo = np.array([[g[0], g[3]] for g in ds.GeoInfo])
    gi = np.round((geo[:, 0] - geo[:, 0].min()) / DLON).astype(int)
    gj = np.round((geo[:, 1] - geo[:, 1].min()) / DLAT).astype(int)
    return gi, gj, {(a, b): k for k, a, b in zip(range(len(gi)), gi, gj)}


def mosaic(ds, cell, gi, gj, k):
    """3x3 の近傍を貼り合わせて 384x384 に。欠けた近傍は中央の鏡像で埋める。"""
    a, b = gi[k], gj[k]
    D = np.zeros((384, 384), np.float32)
    A = np.zeros((384, 384, 3), np.uint8)
    ok = np.zeros((3, 3), bool)
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            n = cell.get((a + p, b + q))
            if n is None:
                continue
            ok[p + 1, q + 1] = True
            r, c = (1 - q) * 128, (p + 1) * 128     # 緯度は上が大きいので行は反転
            D[r:r + 128, c:c + 128] = np.asarray(ds.DEM[n], np.float32)
            A[r:r + 128, c:c + 128] = np.asarray(ds.AirPhoto[n], np.uint8)
    for p in (-1, 0, 1):
        for q in (-1, 0, 1):
            if ok[p + 1, q + 1]:
                continue
            r, c = (1 - q) * 128, (p + 1) * 128
            cen_d = D[128:256, 128:256]
            cen_a = A[128:256, 128:256]
            D[r:r + 128, c:c + 128] = cen_d[::(-1 if q else 1), ::(-1 if p else 1)]
            A[r:r + 128, c:c + 128] = cen_a[::(-1 if q else 1), ::(-1 if p else 1)]
    return D, A, bool(ok.all())


def target_tiles(ds, region, bg_ratio=0.0):
    """評価対象のタイル。広島は学習時と同じホールドアウト、島根は全件。

    bg_ratio > 0 のときは背景タイルを混ぜてから分割しないとずれる
    （analysis/data.py の hiroshima_test_indices を参照）。
    """
    if region == "shimane":
        return list(range(len(ds.No)))
    from analysis.data import hiroshima_test_indices
    hz = [k for k in range(len(ds.No)) if np.max(np.asarray(ds.Mask[k])) >= 1]
    bgs = [k for k in range(len(ds.No)) if np.max(np.asarray(ds.Mask[k])) < 1]
    hp, bp = hiroshima_test_indices(len(hz), bg_ratio, len(bgs))
    return [hz[i] for i in hp] + [bgs[i] for i in bp]


def prep_mask(ds, k):
    """学習時と同じ前処理（uint8*255 -> ガウシアン -> 0より大）。"""
    m = np.asarray(ds.Mask[k]).astype(np.uint8) * 255
    return gaussian_filter(m, sigma=1) > 0


@torch.no_grad()
def evaluate(cond, region, device, verbose=True):
    model = build_model(cond.arch).to(device)
    meta = load_weights(model, cond.weights, device=device)
    model.eval()

    dirname = REGION_DIR[region]
    pkl = dataset_path(dirname, cond.pkl_for(dirname))
    if verbose:
        print(f"  読み込み: {os.path.basename(pkl)} ...", flush=True)
    ds = load_pkl(pkl)
    gi, gj, cell = grid_index(ds)
    tiles = target_tiles(ds, region, float(getattr(cond, "bg_ratio", 0.0) or 0.0))
    W = cos_window()
    if verbose:
        print(f"  {len(tiles)} タイル", flush=True)

    acc = {k: dict(tp=0, fp=0, fn=0) for k in ("従来", "オーバーラップ")}
    evs = {k: InstanceEvaluator() for k in acc}
    n_full = 0
    for n, k in enumerate(tiles):
        D, A, full = mosaic(ds, cell, gi, gj, k)
        n_full += full
        dems, airs = [], []
        for dy, dx in OFFSETS:
            r, c = 128 + dy, 128 + dx
            dems.append(D[r:r + 128, c:c + 128])
            airs.append(A[r:r + 128, c:c + 128])
        dem = torch.from_numpy(np.stack(dems)[:, None]).float().to(device)
        air = torch.from_numpy(np.stack(airs).transpose(0, 3, 1, 2)).float().to(device) / 255.0
        pr = torch.sigmoid(model(dem, air)).cpu().numpy()[:, 0]

        wa = np.zeros((128, 128), np.float32)
        ws = np.zeros_like(wa)
        for (dy, dx), p in zip(OFFSETS, pr):
            ys, ye = max(0, dy), min(128, 128 + dy)
            xs, xe = max(0, dx), min(128, 128 + dx)
            wa[ys:ye, xs:xe] += p[ys - dy:ye - dy, xs - dx:xe - dx] * W[ys - dy:ye - dy, xs - dx:xe - dx]
            ws[ys:ye, xs:xe] += W[ys - dy:ye - dy, xs - dx:xe - dx]

        gt = prep_mask(ds, k)
        out = {"従来": pr[OFFSETS.index((0, 0))] > 0.5,
               "オーバーラップ": (wa / np.maximum(ws, 1e-6)) > 0.5}
        for name, b in out.items():
            acc[name]["tp"] += int((b & gt).sum())
            acc[name]["fp"] += int((b & ~gt).sum())
            acc[name]["fn"] += int((~b & gt).sum())
            evs[name].add(gt[None], b[None])
        if verbose and n % 500 == 0:
            print(f"    {n}/{len(tiles)}", end="\r", flush=True)

    if verbose:
        print(f"  8近傍が揃うタイル {100 * n_full / len(tiles):.1f}%            ")
    return acc, evs, len(tiles), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane", "both"])
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--record", action="store_true",
                    help="results/evals に記録する（既定は表示のみ）")
    args = ap.parse_args()

    regions = ["hiroshima", "shimane"] if args.region == "both" else [args.region]
    recorded = 0
    for name in args.conditions:
        cond = get(name)
        for region in regions:
            print(f"\n[{region}] {name}", flush=True)
            acc, evs, n_tiles, meta = evaluate(cond, region, args.device)
            print(f"  {'規則':<14}{'面積R':>8}{'面積P':>8}{'面積F':>8}  "
                  f"{'箇所R':>8}{'箇所P':>8}{'箇所F':>8}")
            run_id = dc5results.new_run_id(name)
            for rule in ("従来", "オーバーラップ"):
                a = acc[rule]
                R = a["tp"] / max(a["tp"] + a["fn"], 1)
                P = a["tp"] / max(a["tp"] + a["fp"], 1)
                F = 2 * R * P / (R + P) if R + P else 0.0
                s = evs[rule].summary(PRIMARY)
                print(f"  {rule:<14}{R:>8.4f}{P:>8.4f}{F:>8.4f}  "
                      f"{s['箇所Recall']:>8.4f}{s['箇所Precision']:>8.4f}{s['箇所F値']:>8.4f}")
                if not args.record or rule == "従来":
                    continue      # 従来の値は run_inference.py 側が記録済み
                tileset = dc5results.tileset_label(
                    region, float(getattr(cond, "bg_ratio", 0.0) or 0.0))
                dc5results.record_eval(
                    run_id=run_id, condition=name, region=region, tileset=tileset,
                    scope="オーバーラップ", metric_kind="面積",
                    recall=round(R, 4), precision=round(P, 4), f1=round(F, 4),
                    n_tiles=n_tiles,
                    note="run_overlap_eval.py。3x3近傍の9窓・中心重みつき平均")
                dc5results.record_eval(
                    run_id=run_id, condition=name, region=region, tileset=tileset,
                    scope="オーバーラップ", metric_kind="箇所", setting=PRIMARY,
                    recall=s["箇所Recall"], precision=s["箇所Precision"], f1=s["箇所F値"],
                    n_gt_instances=s["箇所_正解数"], n_pred_instances=s["箇所_予測数"],
                    note="run_overlap_eval.py。連結性8近傍 / 最小サイズ10px")
                recorded += 2
    if args.record:
        out = dc5results.build_metrics_csv()
        print(f"\n{recorded} 件を記録し、metrics.csv を作り直しました: {out}")


if __name__ == "__main__":
    main()
