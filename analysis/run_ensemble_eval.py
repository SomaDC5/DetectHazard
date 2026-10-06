# -*- coding: utf-8 -*-
"""複数条件の予測確率を平均して、アンサンブルの実効値を測る。

なぜ必要か（docs/アンサンブルで解けるか.md 8.2 / 8.5）
------------------------------------------------------
タイルごとに最良の条件を選べたと仮定した天井は +0.0744〜+0.1769 あった。
ただしこれは「選べたら」の値で、選び方は別の問題。

**平均なら選び方が要らない。** 天井のうちどれだけが選択なしで取れるかを測る。

既存のキャッシュ（cache/analysis/tiles/*.csv）には混同行列しか無く、
確率マップが無いので平均できない。そのため推論をやり直す。

使い方
------
    # 単純平均
    python analysis/run_ensemble_eval.py --region hiroshima \
        --conditions FinalFusion_SAM_APM AttentionUNet_SAM_APM

    # 重みつき（--conditions と同じ順で）
    python analysis/run_ensemble_eval.py --region hiroshima \
        --conditions A B --weights 0.7 0.3

    # 単体も一緒に出す（比較用）
    python analysis/run_ensemble_eval.py --region hiroshima \
        --conditions A B --each

入力の pkl が条件ごとに違ってよい（DEM系と SAM系を混ぜる場合）。
**タイル番号で突き合わせる**ので、共通するタイルだけが対象になる。

メモリ
------
確率は float16 で持つ。広島2,964枚なら1条件あたり約97MB。
pkl は種類ごとに1回だけ読み、読み終えたら解放する。
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, data as adata, instances            # noqa: E402
from analysis.run_instance_eval import SETTINGS                   # noqa: E402
from dc5lib.instance_eval import PRIMARY                          # noqa: E402
from dc5lib.models import build_model_for, load_weights           # noqa: E402
from dc5lib.registry import get                                   # noqa: E402
from dc5lib.device import pick_device                             # noqa: E402

def predict(cond, tiles, device, batch_size=32):
    """1条件ぶんの sigmoid 確率を返す。float16 [N, H, W]。"""
    model = build_model_for(cond).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()
    out = np.empty((len(tiles.no), config.TILE_SIZE, config.TILE_SIZE), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, len(tiles.no), batch_size):
            e = min(s + batch_size, len(tiles.no))
            t = torch.from_numpy(tiles.terrain[s:e]).to(device)
            if cond.use_airphoto:
                a = torch.from_numpy(np.ascontiguousarray(tiles.air[s:e])).to(device).float() / 255.0
                logits = model(t, a)
            else:
                logits = model(t)
            out[s:e] = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float16)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def evaluate(gt, prob, thr, connectivity=8, min_size=10):
    """面積と箇所の両方を返す。run_instance_eval と同じ設定で測る。"""
    pred = prob.astype(np.float32) > thr
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    area_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    rows = [instances.evaluate_tile_multi(gt[k], pred[k], SETTINGS,
                                          connectivity=connectivity, min_size=min_size)
            for k in range(len(gt))]
    # run_instance_eval と同じ集計（設定ごとに足し上げる）
    r = instances.summarize_setting(pd.DataFrame(rows), PRIMARY)
    return {
        "面積F": round(area_f1, 4),
        "面積R": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "面積P": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "箇所F": round(float(r["箇所F値"]), 4),
        "箇所R": round(float(r["箇所Recall"]), 4),
        "箇所P": round(float(r["箇所Precision"]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "hiroshima_bg", "shimane"])
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="--conditions と同じ順。省略すると単純平均")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prob-thr", type=float, default=None)
    ap.add_argument("--each", action="store_true", help="単体の値も出す")
    ap.add_argument("--out", default=None, help="結果を書き出す CSV")
    args = ap.parse_args()

    thr = config.THRESHOLD if args.prob_thr is None else args.prob_thr
    device = torch.device(args.device) if args.device else pick_device()
    conds = [get(n) for n in args.conditions]
    w = np.array(args.weights, dtype=np.float64) if args.weights else np.ones(len(conds))
    if len(w) != len(conds):
        sys.exit(f"--weights の数 {len(w)} が --conditions の数 {len(conds)} と合いません")
    w = w / w.sum()

    print(f"地域 {args.region} / しきい値 {thr} / device {device}")
    for c, wi in zip(conds, w):
        print(f"  {c.name:<40} 重み {wi:.3f}  pkl {c.dataset.get(_region_dir(args.region))}")

    # pkl の種類ごとに1回だけ読む
    by_pkl: dict[str, list] = {}
    for c in conds:
        by_pkl.setdefault(str(c.pkl_for(_region_dir(args.region))), []).append(c)

    probs, gts, nos = {}, {}, {}
    for pkl, group in by_pkl.items():
        print(f"\n読み込み: {os.path.basename(pkl)}")
        tiles = adata.build_tileset(pkl, args.region, any(c.use_airphoto for c in group),
                                    group[0].bg_ratio, verbose=True)
        gts[pkl] = (tiles.mask[:, 0] > 0)
        nos[pkl] = list(tiles.no)
        for c in group:
            print(f"  推論 {c.name} …", flush=True)
            probs[c.name] = predict(c, tiles, device, args.batch_size)
        del tiles
        gc.collect()

    # タイル番号で突き合わせる（pkl が違っても並びが違うだけ）
    common = None
    for pkl in by_pkl:
        s = set(nos[pkl])
        common = s if common is None else (common & s)
    common = sorted(common)
    print(f"\n共通タイル {len(common)} 枚")
    idx = {pkl: [nos[pkl].index(t) for t in common] for pkl in by_pkl}
    pkl0 = next(iter(by_pkl))
    gt = gts[pkl0][idx[pkl0]]
    for pkl in by_pkl:                      # 正解が一致しているか確かめる
        if not np.array_equal(gts[pkl][idx[pkl]], gt):
            sys.exit("pkl 間で正解マスクが一致しません。突き合わせを確認してください")

    aligned = {}
    for c in conds:
        pkl = str(c.pkl_for(_region_dir(args.region)))
        aligned[c.name] = probs[c.name][idx[pkl]]

    rows = []
    if args.each:
        for c in conds:
            r = evaluate(gt, aligned[c.name], thr); r["対象"] = c.name; rows.append(r)
    mix = np.zeros(gt.shape, dtype=np.float32)
    for c, wi in zip(conds, w):
        mix += wi * aligned[c.name].astype(np.float32)
    r = evaluate(gt, mix, thr)
    r["対象"] = "アンサンブル（" + "+".join(c.name for c in conds) + "）"
    rows.append(r)

    df = pd.DataFrame(rows)[["対象", "面積F", "面積R", "面積P", "箇所F", "箇所R", "箇所P"]]
    print("\n" + df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"\n書き出し: {args.out}")


def _region_dir(region):
    from dc5lib import regions as _r
    return _r.get(region).dir


if __name__ == "__main__":
    main()
