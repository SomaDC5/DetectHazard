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
from dc5lib import ensembles as ens                               # noqa: E402
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


def evaluate(gt, prob, thr, connectivity=8, min_size=10, full=False):
    """面積（通常・境界）と箇所（全設定）を返す。

    full=False なら画面表示用に主要値だけ。True なら記録用に全設定。
    境界は四辺から config.BORDER_CROP を落とした中心のみ（既存の評価と同じ）。
    """
    pred = prob.astype(np.float32) > thr

    def area(p, g):
        tp = int(np.logical_and(p, g).sum())
        fp = int(np.logical_and(p, ~g).sum())
        fn = int(np.logical_and(~p, g).sum())
        d = 2 * tp + fp + fn
        return {"f1": 2 * tp / d if d else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "precision": tp / (tp + fp) if (tp + fp) else 0.0}

    a_full = area(pred, gt)
    cr = config.BORDER_CROP
    a_cent = area(pred[:, cr:-cr, cr:-cr], gt[:, cr:-cr, cr:-cr])

    rows = [instances.evaluate_tile_multi(gt[k], pred[k], SETTINGS,
                                          connectivity=connectivity, min_size=min_size)
            for k in range(len(gt))]
    df = pd.DataFrame(rows)
    inst = {name: instances.summarize_setting(df, name) for name in SETTINGS}
    pr = inst[PRIMARY]

    out = {
        "面積F": round(a_full["f1"], 4),
        "面積R": round(a_full["recall"], 4),
        "面積P": round(a_full["precision"], 4),
        "箇所F": round(float(pr["箇所F値"]), 4),
        "箇所R": round(float(pr["箇所Recall"]), 4),
        "箇所P": round(float(pr["箇所Precision"]), 4),
    }
    if full:
        out["_area_full"], out["_area_center"], out["_inst"] = a_full, a_cent, inst
        out["_n_tiles"] = len(gt)
    return out


def record(name, label, region, cond_bg, res, when=None):
    """results/evals/ に記録する。metrics.csv は evals だけから作られるので、
    runs/*.json が無くても単一条件と並んで載る。

    run_id は「学習」ではないので、アンサンブル名 + 機械名 + 日時 で合成する。
    """
    from dc5lib import results as R
    from dc5lib.regions import get as region_get

    tileset = {"hiroshima": "警戒+背景" if cond_bg else "警戒のみ",
               "hiroshima_bg": "背景のみ", "shimane": "全件"}[region]
    run_id = R.new_run_id(name, when=when)
    n = 0
    for scope, a in (("通常", res["_area_full"]), ("境界", res["_area_center"])):
        R.record_eval(run_id=run_id, condition=name, region=region, tileset=tileset,
                      scope=scope, metric_kind="面積",
                      recall=a["recall"], precision=a["precision"], f1=a["f1"],
                      n_tiles=res["_n_tiles"],
                      note=f"アンサンブル（{label}）。ensembles.yaml / run_ensemble_eval.py")
        n += 1
    for setting, r in res["_inst"].items():
        R.record_eval(run_id=run_id, condition=name, region=region, tileset=tileset,
                      scope="通常", metric_kind="箇所", setting=setting,
                      recall=r["箇所Recall"], precision=r["箇所Precision"], f1=r["箇所F値"],
                      n_gt_instances=r["箇所_正解数"], n_pred_instances=r["箇所_予測数"],
                      note="連結性8近傍 / 最小サイズ10px / アンサンブル")
        n += 1
    out = R.build_metrics_csv()
    print(f"\n記録: evals {n} 件 / metrics.csv を更新（{out}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "hiroshima_bg", "shimane"])
    ap.add_argument("--ensemble", default=None,
                    help="ensembles.yaml の名前。--conditions の代わりに使う")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="--conditions と同じ順。省略すると単純平均")
    ap.add_argument("--rule", default=None, choices=list(ens.RULES),
                    help="統合規則。既定は mean（ensembles.yaml の rule が優先）")
    ap.add_argument("--record", action="store_true",
                    help="results/evals/ に記録して metrics.csv に載せる")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--prob-thr", type=float, default=None)
    ap.add_argument("--each", action="store_true", help="単体の値も出す")
    ap.add_argument("--out", default=None, help="結果を書き出す CSV")
    args = ap.parse_args()

    # ensembles.yaml から引くか、コマンドラインで直に指定するか
    if args.ensemble:
        if args.conditions:
            sys.exit("--ensemble と --conditions は同時に指定できません")
        e = ens.get(args.ensemble)
        names, weights = list(e.members), e.weights_normalized()
        rule = args.rule or e.rule
        thr = args.prob_thr if args.prob_thr is not None else float(e.threshold)
        label = e.display()
    elif args.conditions:
        names = list(args.conditions)
        weights = ([x / sum(args.weights) for x in args.weights] if args.weights
                   else [1.0 / len(names)] * len(names))
        if len(weights) != len(names):
            sys.exit(f"--weights の数 {len(weights)} が --conditions の数 {len(names)} と合いません")
        rule = args.rule or "mean"
        thr = config.THRESHOLD if args.prob_thr is None else args.prob_thr
        label = "アンサンブル（" + "+".join(names) + "）"
    else:
        sys.exit("--ensemble か --conditions のどちらかを指定してください")

    device = torch.device(args.device) if args.device else pick_device()
    conds = [get(n) for n in names]
    w = np.array(weights, dtype=np.float64)

    bgs = {c.bg_ratio for c in conds}
    if len(bgs) > 1:
        sys.exit(f"members の bg_ratio が揃っていません {sorted(bgs)}。"
                 f"背景タイルの混ぜ方が変わって分割がずれ、タイルが突き合いません")

    print(f"{label}")
    print(f"地域 {args.region} / 規則 {rule} / しきい値 {thr} / device {device}")
    for c, wi in zip(conds, w):
        print(f"  {c.name:<40} 重み {wi:.3f}  pkl {c.dataset.get(_region_dir(args.region))}")

    # pkl の種類ごとに1回だけ読む
    by_pkl: dict[str, list] = {}
    for c in conds:
        by_pkl.setdefault(str(c.pkl_for(_region_dir(args.region))), []).append(c)

    # 逐次に足し込む。全条件の確率を同時に持つと、島根24,569枚 x 14条件で
    # 11GB を超えて 31GB のマシンで OOM になる。
    #   1条件の確率 float16 = 805MB（島根）/ 97MB（広島）
    rows = []
    ref_no = None          # タイルの基準順。最初の pkl のもの
    gt = None
    mix = None             # mean なら重みつき和、max なら最大値（float32）
    stack = None           # median のときだけ全メンバーを持つ（メモリを食う）
    if rule == "median":
        print("  ※ median は全メンバーの確率を同時に持つのでメモリを食います")

    for pi, (pkl, group) in enumerate(by_pkl.items()):
        print(f"\n読み込み: {os.path.basename(pkl)}")
        tiles = adata.build_tileset(pkl, args.region, any(c.use_airphoto for c in group),
                                    group[0].bg_ratio, verbose=True)
        no = list(tiles.no)
        g = (tiles.mask[:, 0] > 0)
        if ref_no is None:
            ref_no, gt = no, g
            mix = (np.zeros(gt.shape, dtype=np.float32) if rule != "max"
                   else np.zeros(gt.shape, dtype=np.float32))
            stack = [] if rule == "median" else None
            order = np.arange(len(no))
        else:
            if set(no) != set(ref_no):
                sys.exit(f"pkl 間でタイル集合が違います（{os.path.basename(pkl)}）。"
                         f"同じ分割の条件だけを指定してください")
            pos = {t: k for k, t in enumerate(no)}
            order = np.array([pos[t] for t in ref_no])
            if not np.array_equal(g[order], gt):
                sys.exit("pkl 間で正解マスクが一致しません。突き合わせを確認してください")
        for c in group:
            print(f"  推論 {c.name} …", flush=True)
            pr = predict(c, tiles, device, args.batch_size)[order]
            if args.each:
                r = evaluate(gt, pr, thr); r["対象"] = c.name; rows.append(r)
            wi = float(w[names.index(c.name)])
            if rule == "mean":
                mix += wi * pr.astype(np.float32)
            elif rule == "max":
                np.maximum(mix, pr.astype(np.float32), out=mix)
            else:                                   # median
                stack.append(pr.copy())
            del pr
            gc.collect()
        del tiles, g
        gc.collect()

    print(f"\n共通タイル {len(ref_no)} 枚")
    if rule == "median":
        mix = np.median(np.stack(stack).astype(np.float32), axis=0)
        del stack
        gc.collect()
    res = evaluate(gt, mix, thr, full=True)
    r = {k: v for k, v in res.items() if not k.startswith("_")}
    r["対象"] = label
    rows.append(r)

    df = pd.DataFrame(rows)[["対象", "面積F", "面積R", "面積P", "箇所F", "箇所R", "箇所P"]]
    print("\n" + df.to_string(index=False))

    if args.record:
        if not args.ensemble:
            sys.exit("--record は --ensemble で名前が決まっているときだけ使えます")
        record(args.ensemble, label, args.region, conds[0].bg_ratio, res)
    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"\n書き出し: {args.out}")


def _region_dir(region):
    from dc5lib import regions as _r
    return _r.get(region).dir


if __name__ == "__main__":
    main()
