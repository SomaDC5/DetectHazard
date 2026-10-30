# -*- coding: utf-8 -*-
"""学習した2段目で刈って、箇所数評価を出す。

しきい値の決め方は `展望_再現率をどう上げるか.md 13.1` と同じ作法にする。

    1. --split val  で残す確率のしきい値を掃いて、箇所F が最良のものを選ぶ
    2. --split test --thr <その値>  で測る

val は1段目が重みの更新に使っていないタイルで、2段目の学習に使った split。
したがって 1 で選んだしきい値は 2 のタイルを見ていない。

使い方
------
    python analysis/run_stage2_eval.py --model Stage2_SAM_OL_v1 --split val
    python analysis/run_stage2_eval.py --model Stage2_SAM_OL_v1 --split test --thr 0.42
    python analysis/run_stage2_eval.py --postprocess Post_SAM_OL_Stage2 --split test --record
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config, instances                                  # noqa: E402
from analysis.data import load_pkl                                      # noqa: E402
from analysis.run_instance_eval import SETTINGS                         # noqa: E402
from analysis.run_overlap_eval import (OFFSETS, cos_window, grid_index,  # noqa: E402
                                       mosaic, prep_mask)
from analysis.run_overlap_prune import full_scores, score               # noqa: E402
from analysis.run_stage2_train import split_indices, union_logits       # noqa: E402
from dc5lib import inputs as minputs  # noqa: E402
from dc5lib import landuse as _landuse  # noqa: E402
from dc5lib import paths, stage2                                        # noqa: E402
from dc5lib.device import pick_device                                   # noqa: E402
from dc5lib.instance_eval import label_instances                        # noqa: E402
from dc5lib.models import build_model_for, load_weights                 # noqa: E402
from dc5lib.paths import dataset_path                                   # noqa: E402
from dc5lib.registry import get                                         # noqa: E402

THRS = [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]


@torch.no_grad()
def run(stage1, model2, split, device, region="hiroshima", wide_ch=0,
        use_lu=False, union_cond=None, agree=False):
    """1段目を走らせ、かたまりごとに2段目の「残す確率」を付ける。

    戻り値 (GT, PB, rows) で rows は tile / blob / keep を持つ DataFrame。

    union_cond を渡すと1段目を**和集合**にする（展望 21）。
    ロジットは要素ごとの max、予測は OR。学習側（run_stage2_train.build）と
    同じ作り方にしないと、2段目が見たものと違うものを渡すことになる。
    """
    dirname = {"hiroshima": "Hiroshima", "shimane": "Shimane"}[region]
    # 和集合の相方は、こちらの pkl を読む前に済ませて解放する（メモリ）
    u_list = ([] if union_cond is None else
              ([union_cond] if not isinstance(union_cond, (list, tuple))
               else list(union_cond)))
    # **1段目の3入力目（建物など）はこの経路では未対応。**
    # 相方（union_logits）は third_planes で通るようにしたが、
    # 1段目はまだ土地利用だけである。黙って落ちるより先に止める。
    if minputs.third_input(stage1) is not None:
        sys.exit(
            f"1段目 {stage1.name} は3入力目"
            f"（{minputs.third_input(stage1).field}）を要りますが、"
            "この経路は土地利用以外の3入力目に未対応です。"
            "和集合の相方としてなら --union-with で使えます")
    u_packs = [union_logits(c2, split, device, region=region) for c2 in u_list]
    pkl = dataset_path(dirname, stage1.pkl_for(dirname))
    print(f"  読み込み: {pkl.name}")
    ds = load_pkl(pkl)
    if region == "shimane":
        tiles = list(range(len(ds.No)))
    else:
        hz = [k for k in range(len(ds.No)) if np.max(np.asarray(ds.Mask[k])) >= 1]
        tr, va, te = split_indices(len(hz))
        tiles = [hz[i] for i in {"train": tr, "val": va, "test": te}[split]]
    print(f"  {split}: {len(tiles)} タイル")

    m1 = build_model_for(stage1).to(device)
    load_weights(m1, stage1.weights, device=device)
    m1.eval()
    gi, gj, cell = grid_index(ds)
    W = cos_window()

    GT = np.zeros((len(tiles), 128, 128), bool)
    PB = np.zeros_like(GT)
    rows = []
    for n, k in enumerate(tiles):
        D, A, _, LU = mosaic(ds, cell, gi, gj, k)
        dems, airs, lus = [], [], []
        for dy, dx in OFFSETS:
            r, c = 128 + dy, 128 + dx
            dems.append(D[r:r + 128, c:c + 128])
            airs.append(A[r:r + 128, c:c + 128])
            if LU is not None:
                # 地形と同じ窓で切る
                lus.append(LU[r:r + 128, c:c + 128])
        dem = torch.from_numpy(np.stack(dems)[:, None]).float().to(device)
        air = (torch.from_numpy(np.stack(airs).transpose(0, 3, 1, 2))
               .float().to(device) / 255.0)
        # **pkl に土地利用があるかではなく、モデルが3入力かで決める。**
        # _lu.pkl を2入力の条件で読むと、余計な引数を渡して落ちる。
        extra = ([_landuse.onehot_torch(np.stack(lus), device=device)]
                 if (LU is not None and minputs.needs_landuse(stage1)) else [])
        lo = m1(dem, air, *extra)[:, 0]
        pr = torch.sigmoid(lo).cpu().numpy()
        lg = lo.cpu().numpy()

        wp = np.zeros((128, 128), np.float32)
        wl = np.zeros((128, 128), np.float32)
        ws = np.zeros((128, 128), np.float32)
        for (dy, dx), p, l in zip(OFFSETS, pr, lg):
            ys, ye = max(0, dy), min(128, 128 + dy)
            xs, xe = max(0, dx), min(128, 128 + dx)
            w = W[ys - dy:ye - dy, xs - dx:xe - dx]
            wp[ys:ye, xs:xe] += p[ys - dy:ye - dy, xs - dx:xe - dx] * w
            wl[ys:ye, xs:xe] += l[ys - dy:ye - dy, xs - dx:xe - dx] * w
            ws[ys:ye, xs:xe] += w
        ws = np.maximum(ws, 1e-6)
        logit = wl / ws
        pred = (wp / ws) > 0.5
        n_pos = pred.astype(np.float32)        # 合意数（まず自分の分）
        for u_logit, u_nos in u_packs:
            assert k in u_logit, f"タイル {k} が相方に無い"
            assert u_nos[k] == int(ds.No[k]), (
                f"タイル番号が一致しない: {u_nos[k]} != {int(ds.No[k])}")
            lg2, pd2 = u_logit[k]
            logit = np.maximum(logit, lg2)   # OR は max(ロジット) と同じ
            pred = pred | pd2
            n_pos = n_pos + pd2.astype(np.float32)
        ag = (n_pos / (1 + len(u_packs))) if agree else None
        GT[n] = prep_mask(ds, k)
        PB[n] = pred

        lab, sizes = label_instances(pred, 8, 10)
        if len(sizes) == 0:
            continue
        if wide_ch:
            crops, wd, scal, ids = stage2.extract_tile(
                D, A, logit, pred, lab, sizes, wide=True,
                lu_mos=LU if use_lu else None, agree=ag)
            wt = torch.from_numpy(wd).to(device)
        else:
            crops, scal, ids = stage2.extract_tile(D, A, logit, pred, lab, sizes,
                                                   agree=ag)
            wt = None
        keep = torch.sigmoid(model2(
            torch.from_numpy(crops).to(device),
            torch.from_numpy(scal).to(device), wt)).cpu().numpy()
        for j in range(len(ids)):
            rows.append(dict(tile=n, blob=int(ids[j]), keep=float(keep[j]),
                             logit_max=float(scal[j, 2]) * stage2.LOGIT_SCALE))
        if n % 500 == 0:
            print(f"    {n}/{len(tiles)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    del m1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return GT, PB, pd.DataFrame(rows)


def prune(PB, drop):
    out = PB.copy()
    for k, g in drop.groupby("tile"):
        lab, _ = label_instances(PB[k], 8, 10)
        for b in g.blob:
            out[k][lab == b] = False
    return out


def u_list_names_differ(pp, given):
    """定義と指定が食い違っていないか。指定が無ければ食い違いではない。

    **記録する数字と実際に測ったものがずれるのを防ぐ。**
    """
    if not given:
        return False
    return list(pp.union_list) != list(given)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--model", default=None, help="2段目の重みの名前")
    ap.add_argument("--postprocess", default=None,
                    help="ensembles.yaml の postprocess 名。model と thr を定義から読む")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--region", default="hiroshima", choices=["hiroshima", "shimane"])
    ap.add_argument("--thr", type=float, default=None,
                    help="しきい値。省略すると掃いて表にする")
    ap.add_argument("--union-with", nargs="*", default=None,
                    help="和集合にする相方の条件名（展望 21）。"
                         "学習時と同じものを指定すること")
    ap.add_argument("--by", default="keep", choices=["keep", "logit_max"],
                    help="何で刈るか。keep=2段目の出力 / "
                         "logit_max=しきい値1本（同じ split で比べるための対照）")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--thr-split", default="検証split",
                    help="しきい値をどの split で選んだか。**記録に残る**")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    pp = None
    if args.postprocess:
        from dc5lib import postprocess as pp_mod
        pp = pp_mod.get(args.postprocess)
        args.stage1 = pp.base
        args.model = pp.model          # dc5lib/postprocess.py の既知の項目
        # **定義の union_with を使う。** これが無いと --postprocess
        # だけ指定したときに単体を測ってしまい、記録だけ
        # 「和集合」になる（docs/設計_アンサンブルの記録.md の 3.3）
        if not args.union_with:
            args.union_with = list(pp.union_list)
        if args.thr is None:
            args.thr = pp.threshold
        print(f"後処理 {pp.name}: {pp.display()}")
    if args.record and not pp:
        sys.exit("--record には --postprocess が必要です"
                 "（ensembles.yaml の postprocess に定義する。"
                 "docs/設計_アンサンブルの記録.md の 3.1）")
    if pp and u_list_names_differ(pp, args.union_with):
        sys.exit(
            "--postprocess の union_with と --union-with が違います。" + chr(10)
            + f"  定義 {list(pp.union_list)}" + chr(10)
            + f"  指定 {list(args.union_with or [])}" + chr(10)
            + "記録と実際に測ったものが食い違うので止めました。"
              "--union-with を省けば定義のほうを使います")
    if not args.model:
        sys.exit("--model か --postprocess を指定してください")

    device = torch.device(args.device) if args.device else pick_device()
    stage1 = get(args.stage1)
    u_list_names = args.union_with or []
    w = paths.weights_path(args.model)
    if not w.exists():
        sys.exit(f"{w} がありません。run_stage2_train.py で学習してください")
    ck = torch.load(w, map_location=device)
    wide_ch = int(ck.get("wide_ch", 0))
    use_lu = bool(ck.get("landuse", False))
    if wide_ch:
        stage2.set_wide_channels(use_lu)
        if stage2.N_WIDE_CH != wide_ch:
            sys.exit(f"重みの広域チャネル {wide_ch} と設定 {stage2.N_WIDE_CH} が違います")
    # 合意数チャネルの有無は重みに記録してある（古い重みには無いので既定 6ch）
    in_ch = int(ck.get("in_ch", stage2.N_CH_BASE))
    use_agree = bool(ck.get("agree", False))
    stage2.set_agree_channel(use_agree)
    if stage2.N_CH != in_ch:
        sys.exit(f"重みの入力チャネル {in_ch} と設定 {stage2.N_CH} が違います")
    if use_agree and not u_list_names:
        sys.exit("この重みは合意数チャネルを使っています。"
                 "--union-with を学習時と同じように指定してください")
    if use_agree:
        print(f"  合意数チャネルあり → 入力 {in_ch}ch")
    m2 = stage2.Stage2Net(in_ch=in_ch, wide_ch=wide_ch).to(device)
    m2.load_state_dict(ck["model_state_dict"])
    m2.eval()
    if wide_ch:
        print(f"  広域の枝 {wide_ch}ch{'・土地利用こみ' if use_lu else ''}")
    print(f"1段目 {stage1.name}（凍結）/ 2段目 {args.model} "
          f"(ep{ck.get('epoch')} 検証AUC {ck.get('val_auc', float('nan')):.4f})")
    print(f"device {device}")

    GT, PB, rows = run(stage1, m2, args.split, device, args.region,
                       union_cond=[get(u) for u in args.union_with]
                       if args.union_with else None, agree=use_agree,
                       wide_ch=wide_ch, use_lu=use_lu)
    R0, P0, F0, n0 = score(GT, PB)
    print(f"\nかたまり {len(rows)}")
    print(f"刈り取り前   箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f} / 予測数 {n0}")
    print(f"完璧な刈り取りの上限  箇所F {2 * R0 / (R0 + 1):.4f}")

    if args.thr is None:
        sweep = (THRS if args.by == "keep"
                 else [0.0, 10.0, 15.0, 20.0, 24.0, 27.0, 29.0, 31.0, 34.0, 37.0])
        print(f"\n--- {args.by} のしきい値を掃く ---")
        print("%-10s %9s %8s %8s %8s %9s"
              % ("しきい値", "捨てる割合", "箇所R", "箇所P", "箇所F", "予測数"))
        best = (-1.0, None)
        for t in sweep:
            drop = rows[rows[args.by] < t]
            p = prune(PB, drop) if len(drop) else PB
            R, P, F, nn = score(GT, p)
            if F > best[0]:
                best = (F, t)
            print("%-10s %9s %8.4f %8.4f %8.4f %9d"
                  % (f"{t:.2f}", f"{100 * len(drop) / len(rows):.1f}%", R, P, F, nn))
        print(f"\n最良 しきい値 {best[1]:.2f} で箇所F {best[0]:.4f}")
        print("この値を --thr に渡して、別の split で測ること")
        return

    drop = rows[rows[args.by] < args.thr]
    pruned = prune(PB, drop) if len(drop) else PB
    R, P, F, nn = score(GT, pruned)
    print(f"\n--- {args.by} のしきい値 {args.thr:.2f} を当てる ---")
    print(f"  捨てたかたまり {len(drop)} / {len(rows)}"
          f"（{100 * len(drop) / len(rows):.1f}%）")
    print(f"  刈った後 箇所R {R:.4f} / P {P:.4f} / F {F:.4f} / 予測数 {nn}")
    print(f"  正味 {F - F0:+.4f}")
    if args.record:
        from analysis.run_overlap_prune import record
        res = full_scores(GT, pruned)
        print(f"  面積F {res[0]['f1']:.4f}（境界 {res[1]['f1']:.4f}）")
        record(pp, args.region, res, thr=args.thr,
               thr_split=args.thr_split)


if __name__ == "__main__":
    main()
