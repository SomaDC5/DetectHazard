# -*- coding: utf-8 -*-
"""アンサンブル + オーバーラップ推論で、県全体のハザードマップを GeoTIFF に書く。

QGIS でそのまま開ける形で出す。タイルを貼り合わせて1枚にする。

    python analysis/run_hazard_map.py --ensemble Ens_InputDiverse4 --region shimane

出力（dc5-data/hazard_map/ に出る。2台のSSDで同期される。リポジトリには入れない）

    <アンサンブル名>__<地域>.tif        確率（uint8 0〜100、255 が範囲外）
    <アンサンブル名>__<地域>__mask.tif  二値化（0/1、255 が範囲外）
    <アンサンブル名>__<地域>__meta.json 条件・しきい値・画素数などの記録

バンド構成（確率側）
    1  アンサンブルの確率 0〜100（%）。255 = タイルが無い範囲
       **これを 50 でしきい値処理しても mask.tif と一致しない。**
       uint8 に落とすと、意見が割れた画素（確率ちょうど 0.5）の扱いが変わる。
       評価値と比べるときは mask.tif を使うこと
    2  正解（警戒区域）0/1。255 = タイルが無い範囲
    3  学習に使ったタイルか 0/1。**広島は半分以上が学習に使われている**ので、
       地図を見せるときはこのバンドで除外するか、島根の地図を使う

なぜバンド3が要るか
-------------------
広島は学習に使った県なので、県全体の地図には**学習済みタイルが混ざる**。
そこだけ成績が良く見えるため、地図をそのまま「性能」として見せられない。
島根は学習に一切使っていないので、こちらが正味の地図になる。

大きさ（実測）
    広島  格子 240x276 → 30,720 x 35,328 画素（1,085M）uint8 で 1.1GB
    島根  格子 280x324 → 35,840 x 41,472 画素（1,486M）uint8 で 1.5GB
タイルがあるのは広島 57.6% / 島根 27.1% で、残りは 255（範囲外）。
圧縮（DEFLATE）が効くので実ファイルはこれより小さくなる。

オーバーラップ推論
------------------
run_overlap_eval.py と同じ方式。3x3 の近傍を貼り合わせ、64px ずつずらした
9窓で推論して、中心ほど重い cos 窓で統合する。タイル境界で切れた箇所が
窓の中心に来るので拾える（箇所F +0.047〜+0.060）。

--no-overlap を付けるとタイル1枚ずつの推論になる（速いが境界に弱い）。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data import load_pkl                                  # noqa: E402
from analysis.run_overlap_eval import (DLON, DLAT, OFFSETS, REGION_DIR,  # noqa: E402
                                       cos_window, grid_index, mosaic, prep_mask)
from dc5lib import ensembles as ens                                 # noqa: E402
from dc5lib.device import pick_device                               # noqa: E402
from dc5lib.models import build_model_for, load_weights             # noqa: E402
from dc5lib.paths import cache_dir, dataset_path, hazard_map_dir    # noqa: E402
from dc5lib.registry import get                                     # noqa: E402

TILE = 128
NODATA = 255


@torch.no_grad()
def predict_region(cond, region, overlap, device, verbose=True):
    """1条件ぶんを格子に並べて返す。

    戻り値 (prob, gi, gj, cell, ds_meta)
      prob  float32 [タイル数, 128, 128] そのpklのタイル順
    """
    dirname = REGION_DIR[region]
    pkl = dataset_path(dirname, cond.pkl_for(dirname))
    if verbose:
        print(f"    読み込み: {os.path.basename(str(pkl))} …", flush=True)
    ds = load_pkl(pkl)
    gi, gj, cell = grid_index(ds)
    n = len(ds.No)

    model = build_model_for(cond).to(device)
    meta = load_weights(model, cond.weights, device=device)
    model.eval()
    W = cos_window()
    out = np.empty((n, TILE, TILE), dtype=np.float32)
    t0 = time.time()

    for k in range(n):
        if overlap:
            D, A, _ = mosaic(ds, cell, gi, gj, k)
            dems = np.stack([D[128 + dy:256 + dy, 128 + dx:256 + dx] for dy, dx in OFFSETS])
            airs = np.stack([A[128 + dy:256 + dy, 128 + dx:256 + dx] for dy, dx in OFFSETS])
            t = torch.from_numpy(dems[:, None]).to(device)
            if cond.use_airphoto:
                a = torch.from_numpy(np.ascontiguousarray(
                    airs.transpose(0, 3, 1, 2))).to(device).float() / 255.0
                logit = model(t, a)
            else:
                logit = model(t)
            p = torch.sigmoid(logit).squeeze(1).cpu().numpy()        # [9,128,128]
            # 統合式は run_overlap_eval.py の evaluate() と同一にしてある。
            # 自分で導出すると符号を間違える（実際に間違えた）
            num = np.zeros((TILE, TILE), np.float32)
            den = np.zeros((TILE, TILE), np.float32)
            for (dy, dx), pi in zip(OFFSETS, p):
                ys, ye = max(0, dy), min(TILE, TILE + dy)
                xs, xe = max(0, dx), min(TILE, TILE + dx)
                num[ys:ye, xs:xe] += (pi[ys - dy:ye - dy, xs - dx:xe - dx]
                                      * W[ys - dy:ye - dy, xs - dx:xe - dx])
                den[ys:ye, xs:xe] += W[ys - dy:ye - dy, xs - dx:xe - dx]
            out[k] = num / np.maximum(den, 1e-6)
        else:
            d = np.asarray(ds.DEM[k], np.float32)[None, None]
            t = torch.from_numpy(d).to(device)
            if cond.use_airphoto:
                a = np.asarray(ds.AirPhoto[k], np.uint8).transpose(2, 0, 1)[None]
                a = torch.from_numpy(np.ascontiguousarray(a)).to(device).float() / 255.0
                logit = model(t, a)
            else:
                logit = model(t)
            out[k] = torch.sigmoid(logit).squeeze().cpu().numpy()
        if verbose and (k % 2000 == 0 or k == n - 1):
            el = time.time() - t0
            rate = (k + 1) / max(el, 1e-9)
            print(f"      {k+1}/{n}  {rate:.0f} tiles/s  残り {(n-k-1)/max(rate,1e-9)/60:.1f} 分",
                  end="\r", flush=True)
    if verbose:
        print(f"      {n}/{n} 完了 {(time.time()-t0)/60:.1f} 分   ", flush=True)

    no = list(ds.No)
    masks = np.stack([prep_mask(ds, k) for k in range(n)])
    del model, ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, gi, gj, no, masks, meta


def write_geotiff(path, bands, lon0, lat1, dlon, dlat, epsg, nodata=NODATA):
    """QGIS で開ける GeoTIFF。圧縮・タイル化・概観つき。"""
    import rasterio
    from rasterio.transform import from_origin

    h, w = bands[0].shape
    tr = from_origin(lon0, lat1, dlon, dlat)
    with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=len(bands),
            dtype="uint8", crs=f"EPSG:{epsg}", transform=tr, nodata=nodata,
            compress="deflate", predictor=1, tiled=True,
            blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(b, i)
        dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        dst.update_tags(1, DESCRIPTION="ensemble probability 0-100 (%)")
        if len(bands) > 1:
            dst.update_tags(2, DESCRIPTION="ground truth 0/1")
        if len(bands) > 2:
            dst.update_tags(3, DESCRIPTION="used for training 0/1")
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default=None,
                    help="ensembles.yaml の名前。--conditions の代わりに使う")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="学習済み条件を直に指定。1つだけ書けば単体モデルの地図になる")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="--conditions と同じ順。省略すると単純平均")
    ap.add_argument("--name", default=None,
                    help="出力ファイル名に使う名前。--conditions のときの既定は条件名")
    ap.add_argument("--region", required=True, choices=["hiroshima", "shimane"])
    ap.add_argument("--no-overlap", action="store_true", help="タイル1枚ずつの推論にする")
    ap.add_argument("--prob-thr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="条件ごとの確率を保存しない（再開できなくなる）")
    ap.add_argument("--fresh", action="store_true",
                    help="保存済みを無視して推論し直す")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="GPU が無くても CPU で回す（数十倍遅い）")
    args = ap.parse_args()

    # ensembles.yaml から引くか、条件を直に指定するか。
    # **1条件だけ指定すれば単体モデルの地図になる**（アンサンブルでなくてよい）。
    if args.ensemble:
        if args.conditions:
            sys.exit("--ensemble と --conditions は同時に指定できません")
        e = ens.get(args.ensemble)
        if e.rule != "mean":
            sys.exit(f"いまは rule=mean だけ対応しています（この定義は {e.rule}）")
        members, w = list(e.members), e.weights_normalized()
        thr = args.prob_thr if args.prob_thr is not None else float(e.threshold)
        label, name = e.display(), args.ensemble
    elif args.conditions:
        members = list(args.conditions)
        w = ([x / sum(args.weights) for x in args.weights] if args.weights
             else [1.0 / len(members)] * len(members))
        if len(w) != len(members):
            sys.exit(f"--weights の数 {len(w)} が --conditions の数 {len(members)} と合いません")
        thr = 0.5 if args.prob_thr is None else args.prob_thr
        name = args.name or (members[0] if len(members) == 1 else "mix_" + str(len(members)))
        label = members[0] if len(members) == 1 else "アンサンブル（" + "+".join(members) + "）"
    else:
        sys.exit("--ensemble か --conditions のどちらかを指定してください")

    overlap = not args.no_overlap
    device = torch.device(args.device) if args.device else pick_device()
    # **CPU に黙って落ちると、気づかないまま何十分も走る。**
    # 実際に GPU ドライバが入っていない状態で CPU 推論になり、
    # 1条件（通常1分）に29分かかった。明示しない限り止める。
    if device.type == "cpu" and not args.allow_cpu:
        sys.exit("GPU が使えません（CPU になります）。地図作成は CPU だと数十倍遅いので止めました。\n"
                 "  確認: nvidia-smi / python -m dc5lib.device\n"
                 "  それでも CPU で回すなら --allow-cpu を付けてください")
    conds = [get(m) for m in members]

    bgs = {c.bg_ratio for c in conds}
    if len(bgs) > 1:
        sys.exit(f"bg_ratio が揃っていません {sorted(bgs)}。"
                 f"背景タイルの混ぜ方が変わって分割がずれ、タイルが突き合いません")
    missing = [c.name for c in conds if not c.weights.exists()]
    if missing:
        sys.exit(f"重みがありません: {missing}。SSD を確認してください"
                 f"（python -m dc5lib.sync status）")

    out_dir = Path(args.out_dir) if args.out_dir else hazard_map_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__{args.region}" + ("" if overlap else "__nooverlap")

    print(f"{label} / {args.region} / "
          f"{'オーバーラップ9窓' if overlap else 'タイル1枚'} / しきい値 {thr}")
    print(f"条件 {len(conds)} 件 / 出力 {out_dir}")

    # pkl ごとにまとめる（同じ pkl のメンバーは1回の読み込みで済む）
    by_pkl = {}
    for c in conds:
        by_pkl.setdefault(c.pkl_for(REGION_DIR[args.region]).name, []).append(c)

    # ---- 条件ごとに推論して足し込む
    #
    # **1条件ぶんの確率を中間ファイルに保存する。** 14条件で65分かかるので、
    # 途中で落ちたときに全部やり直すのは割に合わない。実際に GPU が
    # ハングして（Xid 8）2条件目で落ち、1条件ぶん（4.4分）が無駄になった。
    # 2回目以降は保存済みを読むので、落ちた条件から再開できる。
    #
    # **float32 で保存する。** float16 にすると容量は半分になるが、
    # 再開したときの結果が「推論し直したとき」と一致しない。メンバーが
    # 飽和していて確率がちょうど 0.5 に乗るため、丸めの差で判定が反転する
    # （実測で 118,329 画素 = 0.008% が反転した）。再開は同じ結果になるべき。
    #
    # 1ファイル 約2.0GB（島根）/ 約0.3GB（広島）。dc5-data/cache/ に置くので
    # **同期対象には入らない**（hazard_map/ は成果物だけ）。
    # --no-cache で保存しない。--fresh で保存を無視して取り直す。
    # 中間確率は **cache 側**に置く。hazard_map/ は同期対象なので、
    # 11GB の中間物を入れると同期が重くなる。cache は同期されない。
    pdir = cache_dir("hazard_map_probs", f"{args.region}{'' if overlap else '_nooverlap'}")
    ref_no = None
    acc = None          # 重みつき和。タイル順は ref_no に揃える
    gi = gj = masks = None
    metas = {}
    for pkl_name, group in by_pkl.items():
        print(f"  [{pkl_name}] {len(group)} 条件")
        for c in group:
            f = pdir / f"{c.name}.npz"
            use_cache = f.exists() and not args.fresh
            if use_cache:
                print(f"    {c.name}  ← 保存済みを読む（{f.stat().st_size/1e6:.0f} MB）",
                      flush=True)
                z = np.load(f, allow_pickle=True)
                pr, g_i, g_j = z["prob"], z["gi"], z["gj"]
                no, m = list(z["no"]), z["mask"]
                meta = dict(z["meta"].item()) if "meta" in z else {}
                z.close()
            else:
                print(f"    {c.name}", flush=True)
                pr, g_i, g_j, no, m, meta = predict_region(c, args.region, overlap, device)
                if not args.no_cache:
                    # **np.savez は拡張子が .npz でないと勝手に付け足す。**
                    # f.with_suffix(".npz.tmp") にすると X.npz.tmp.npz ができ、
                    # replace() が存在しないファイルを探して失敗する（実際にやった）
                    tmp = f.with_name(f.stem + ".tmp.npz")
                    np.savez(tmp, prob=pr.astype(np.float32), gi=g_i, gj=g_j,
                             no=np.array(no), mask=m, meta=np.array(meta or {}, dtype=object))
                    tmp.replace(f)       # 書き終わってから置き換える（中途半端を残さない）
                    print(f"      保存: {f.name} ({f.stat().st_size/1e6:.0f} MB)", flush=True)
            pr = pr.astype(np.float32)
            if ref_no is None:
                ref_no, gi, gj, masks, acc = no, g_i, g_j, m, np.zeros_like(pr)
                order = np.arange(len(no))
            else:
                if set(no) != set(ref_no):
                    sys.exit(f"pkl 間でタイル集合が違います（{pkl_name}）")
                pos = {t: k for k, t in enumerate(no)}
                order = np.array([pos[t] for t in ref_no])
            acc += float(w[members.index(c.name)]) * pr[order]
            metas[c.name] = {k: (float(v) if isinstance(v, (int, float)) else v)
                             for k, v in (meta or {}).items()}
            del pr
            gc.collect()

    # ---- 格子に貼り合わせる
    gi, gj = np.asarray(gi), np.asarray(gj)
    W = int(gi.max() - gi.min() + 1)
    H = int(gj.max() - gj.min() + 1)
    print(f"\n貼り合わせ: 格子 {W} x {H} → 画素 {W*TILE} x {H*TILE} "
          f"({W*TILE*H*TILE/1e6:.0f} M画素)")

    prob = np.full((H * TILE, W * TILE), NODATA, np.uint8)
    gtm = np.full((H * TILE, W * TILE), NODATA, np.uint8)
    trn = np.full((H * TILE, W * TILE), NODATA, np.uint8)
    # **二値化は float の確率から作る。uint8 に落としたあとで比べてはいけない。**
    # メンバーが飽和しているので、意見が割れた画素の確率はちょうど 0.5 になる。
    # 2条件・等重みの島根では [0.49,0.51] に 8,611,940 画素（有効の2.1%）あり、
    # 「> 0.5」と「>= 50（切り捨て後）」で 1,192,518 画素の判定が変わった。
    # 同点の扱いだけで Recall +0.025 / Precision -0.021 動く。
    binm = np.full((H * TILE, W * TILE), NODATA, np.uint8)

    # 学習に使ったタイルの判定。広島は「警戒ありのうちホールドアウト以外」が学習側
    train_flag = np.zeros(len(ref_no), bool)
    if args.region == "hiroshima":
        from analysis.data import hiroshima_test_indices
        hz = [k for k in range(len(ref_no)) if masks[k].any()]
        bgs = [k for k in range(len(ref_no)) if not masks[k].any()]
        hp, bp = hiroshima_test_indices(len(hz), float(conds[0].bg_ratio or 0.0), len(bgs))
        test = {hz[i] for i in hp} | {bgs[i] for i in bp}
        for k in range(len(ref_no)):
            train_flag[k] = masks[k].any() and k not in test     # 警戒ありで非テスト
    for k in range(len(ref_no)):
        r = (int(gj.max()) - int(gj[k])) * TILE      # 緯度は上が大きい
        c = (int(gi[k]) - int(gi.min())) * TILE
        prob[r:r + TILE, c:c + TILE] = np.clip(acc[k] * 100, 0, 100).astype(np.uint8)
        binm[r:r + TILE, c:c + TILE] = (acc[k] > thr).astype(np.uint8)
        gtm[r:r + TILE, c:c + TILE] = masks[k].astype(np.uint8)
        trn[r:r + TILE, c:c + TILE] = np.uint8(train_flag[k])

    covered = int((prob != NODATA).sum())
    print(f"  タイルがある画素 {covered/1e6:.0f} M ({covered/prob.size:.1%})")
    if args.region == "hiroshima":
        print(f"  **学習に使ったタイル {int(train_flag.sum())} / {len(ref_no)} "
              f"({train_flag.mean():.1%})** — 地図を見せるときはバンド3で除く")

    # ---- 書き出し
    # 左上の座標は GeoInfo から取る（格子の最小経度・最大緯度）
    pkl = dataset_path(REGION_DIR[args.region], conds[0].pkl_for(REGION_DIR[args.region]).name)
    ds0 = load_pkl(pkl)
    lon0 = min(g[0] for g in ds0.GeoInfo)
    lat1 = max(g[3] for g in ds0.GeoInfo)
    dlon = abs(ds0.GeoInfo[0][1])
    dlat = abs(ds0.GeoInfo[0][5])
    epsg = int(ds0.EPSG[0]) if getattr(ds0, "EPSG", None) else 6668
    del ds0
    gc.collect()

    t0 = time.time()
    f1 = out_dir / f"{stem}.tif"
    n1 = write_geotiff(f1, [prob, gtm, trn], lon0, lat1, dlon, dlat, epsg)
    f2 = out_dir / f"{stem}__mask.tif"
    n2 = write_geotiff(f2, [binm], lon0, lat1, dlon, dlat, epsg)
    print(f"\n書き出し {time.time()-t0:.0f}秒")
    print(f"  {f1.name}  {n1/1e6:.0f} MB  （3バンド: 確率 / 正解 / 学習フラグ）")
    print(f"  {f2.name}  {n2/1e6:.0f} MB  （二値化 しきい値 {thr}）")

    meta = {
        "name": name, "ensemble": args.ensemble, "label": label, "region": args.region,
        "members": members, "weights": w, "threshold": thr,
        "overlap": overlap, "windows": len(OFFSETS) if overlap else 1,
        "grid": [W, H], "pixels": [W * TILE, H * TILE],
        "covered_pixels": covered, "coverage": round(covered / prob.size, 4),
        "epsg": epsg, "lon0": lon0, "lat1": lat1, "dlon": dlon, "dlat": dlat,
        "train_tiles": int(train_flag.sum()), "tiles": len(ref_no),
        "member_meta": metas,
        "note": "band1 確率0-100 / band2 正解0-1 / band3 学習に使ったタイル0-1 / 255=範囲外",
    }
    f3 = out_dir / f"{stem}__meta.json"
    f3.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {f3.name}")


if __name__ == "__main__":
    main()
