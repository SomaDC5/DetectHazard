# -*- coding: utf-8 -*-
"""WNet の2段目を学習する。1段目は凍結したまま触らない。

なぜこの作り方なのか（展望_再現率をどう上げるか.md 10.6c）
----------------------------------------------------------
2段目を「1段目が学習に使ったタイル」のかたまりで学習させてはいけない。
1段目は学習タイルでは出来すぎている（学習F − 検証F が +0.065）ので、
きれいすぎる入力で学習した2段目は、本番の汚い入力に対応できない。

そこで **検証split**（1段目が重みの更新に使っていない。早期終了の判定に
だけ使った）のかたまりで学習する。実測で、検証splitとテストでの1段目の
出来はほぼ同じだった。

    検証   かたまり 9,880（外れ 35.7%）箇所F 0.7003
    テスト かたまり 9,670（外れ 35.6%）箇所F 0.7071

2段目の学習/検証は**タイル単位**で分ける。同じタイルのかたまりが
両側に跨がると、隣り合うかたまりで答えを覗くことになる。

使い方
------
    # かたまりデータを作る（1段目の推論。数分。SSDにキャッシュ）
    python analysis/run_stage2_train.py --build

    # 学習（キャッシュから。軽い）
    python analysis/run_stage2_train.py --train --name Stage2_SAM_OL_v1

    # まとめて
    python analysis/run_stage2_train.py --build --train --name Stage2_SAM_OL_v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import config
from analysis.data import load_pkl
from analysis.run_overlap_eval import (OFFSETS, cos_window, grid_index,  # noqa: E402
                                       mosaic, prep_mask)
from dc5lib import inputs as minputs  # noqa: E402
from dc5lib import landuse as _landuse  # noqa: E402
from dc5lib import paths, stage2                                        # noqa: E402
from dc5lib.device import pick_device                                   # noqa: E402
from dc5lib.instance_eval import label_instances, overlap_matrix        # noqa: E402
from dc5lib.models import build_model_for, load_weights                 # noqa: E402
from dc5lib.paths import dataset_path                                   # noqa: E402
from dc5lib.registry import get                                         # noqa: E402

COV_PRED = 0.3            # 被覆 (cov_gt=0.5, cov_pred=0.3) の予測側


def split_indices(n_hz):
    """学習ノートブックと同じ3分割。戻り値は hz 内の位置。"""
    idx_all = np.arange(n_hz)
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.25, random_state=config.RANDOM_STATE)
    return idx_train, idx_val, idx_test


def cache_path(cond_name, split, wide=False, lu=False, union=None):
    """構成ごとに別ファイルにする。中身が違うのに同じ名前だと混ざる。"""
    d = paths.cache_dir("stage2")
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if not wide else ("__wide_lu" if lu else "__wide")
    if union:
        # 和集合は中身が別物なので必ず別ファイルにする
        tag += "__union_" + union.replace("FinalFusion_", "")
    return d / f"{cond_name}__hiroshima__{split}{tag}.npz"


@torch.no_grad()
def union_logits(cond2, split, device, batch_hint=500, region="hiroshima"):
    """和集合の相方（2つめの条件）のロジットを、タイルごとに先に作る。

    なぜ別パスにするか（展望_再現率をどう上げるか.md 21.5）
    ------------------------------------------------------
    2条件は **別の pkl** を使う（SAM+APM と DEM+APM_normalized）。
    同時に読むと 10GB 近くになるので、相方のロジットだけ先に作って
    pkl を解放してから本体の build を回す。
    ロジットは タイル数 x 128 x 128 x 4byte なので、val なら 120MB 程度。

    戻り値は {タイル番号: (ロジット, 確率マスク)}。
    タイル番号は ds.No の添字で、2つの pkl で並びが同じであることを前提にする
    （assert で確かめる）。
    """
    dirname = {"hiroshima": "Hiroshima", "shimane": "Shimane"}[region]
    pkl = dataset_path(dirname, cond2.pkl_for(dirname))
    print(f"  [和集合の相方] 読み込み: {pkl.name}")
    ds = load_pkl(pkl)
    if region == "shimane":
        # 島根は学習に使っていないので全タイル（run_stage2_eval.run と同じ）
        tiles = list(range(len(ds.No)))
    else:
        hz = [k for k in range(len(ds.No)) if np.max(np.asarray(ds.Mask[k])) >= 1]
        tr, va, te = split_indices(len(hz))
        tiles = [hz[i] for i in {"train": tr, "val": va, "test": te}[split]]
    print(f"  [和集合の相方] {cond2.name}: {len(tiles)} タイル")

    model = build_model_for(cond2).to(device)
    load_weights(model, cond2.weights, device=device)
    model.eval()
    gi, gj, cell = grid_index(ds)
    W = cos_window()

    out, nos = {}, {}
    t0 = time.time()
    for n, k in enumerate(tiles):
        D, A, _, _ = mosaic(ds, cell, gi, gj, k)
        dems, airs = [], []
        for dy, dx in OFFSETS:
            r, c = 128 + dy, 128 + dx
            dems.append(D[r:r + 128, c:c + 128])
            airs.append(A[r:r + 128, c:c + 128])
        dem = torch.from_numpy(np.stack(dems)[:, None]).float().to(device)
        air = (torch.from_numpy(np.stack(airs).transpose(0, 3, 1, 2))
               .float().to(device) / 255.0)
        lo = model(dem, air)[:, 0]
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
        out[k] = ((wl / ws).astype(np.float32), (wp / ws) > 0.5)
        nos[k] = int(ds.No[k])
        if n % batch_hint == 0:
            print(f"    [相方] {n}/{len(tiles)}  {time.time()-t0:.0f}秒",
                  end="\r", flush=True)
    print(f"  [和集合の相方] 完了 {len(out)} タイル            ")
    del model, ds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, nos


@torch.no_grad()
def build(cond, split, device, batch_hint=500, wide=False, lu=False,
          pkl_override=None, union_cond=None):
    """1段目を走らせて、かたまりごとの切り出し・スカラー・ラベルを作る。

    union_cond を渡すと**和集合**を1段目にする（展望 21）。
    マスクの OR は max(ロジット) > 0 と同じなので、ロジットは要素ごとの max、
    予測は2つの確率マスクの OR にする。切り出しの地形と写真は
    **cond 側（1つめ）のもの**を使うので、2段目の入力は 6ch のまま変わらない。
    """
    # 土地利用を2段目に入れるときは、1段目の pkl に LandUse が無いことがある。
    # **1段目のモデルは触らない**（SAM+APM しか使わない）ので、
    # 同じタイル並びで LandUse を持つ pkl に差し替えてよい。
    # 和集合の相方は、こちらの pkl を読む前に済ませて解放する（メモリ）
    u_logit, u_nos = (union_logits(union_cond, split, device)
                      if union_cond is not None else (None, None))

    pkl = dataset_path("Hiroshima",
                       pkl_override or cond.pkl_for("Hiroshima"))
    print(f"  読み込み: {pkl.name}")
    ds = load_pkl(pkl)
    hz = [k for k in range(len(ds.No)) if np.max(np.asarray(ds.Mask[k])) >= 1]
    tr, va, te = split_indices(len(hz))
    pos = {"train": tr, "val": va, "test": te}[split]
    tiles = [hz[i] for i in pos]
    print(f"  {split}: {len(tiles)} タイル")

    model = build_model_for(cond).to(device)
    load_weights(model, cond.weights, device=device)
    model.eval()
    gi, gj, cell = grid_index(ds)
    W = cos_window()

    if wide:
        stage2.set_wide_channels(lu)
        print(f"  広域の枝あり（{stage2.N_WIDE_CH}ch"
              f"{'・土地利用こみ' if lu else ''}）")
    C, WD, S, Y, T, B = [], [], [], [], [], []
    t0 = time.time()
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
                 if (LU is not None and minputs.needs_landuse(cond)) else [])
        lo = model(dem, air, *extra)[:, 0]
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
        prob, logit = wp / ws, wl / ws
        pred = prob > 0.5
        if u_logit is not None:
            # タイルの並びが2つの pkl で同じことを確かめる。
            # ずれていたら無関係なタイルを重ねることになる
            assert k in u_logit, f"タイル {k} が相方に無い"
            assert u_nos[k] == int(ds.No[k]), (
                f"タイル番号が一致しない: {u_nos[k]} != {int(ds.No[k])}")
            lg2, pd2 = u_logit[k]
            logit = np.maximum(logit, lg2)   # OR は max(ロジット) と同じ
            pred = pred | pd2
        gt = prep_mask(ds, k)

        lab, sizes = label_instances(pred, 8, 10)
        if len(sizes) == 0:
            continue
        if wide:
            lu_mos = LU if lu else None
            if lu and LU is None:
                raise RuntimeError("土地利用こみを指定しましたが pkl に LandUse が"
                                   "ありません。*_lu.pkl の条件を指定してください")
            crops, wd, scal, ids = stage2.extract_tile(
                D, A, logit, pred, lab, sizes, wide=True, lu_mos=lu_mos)
            WD.append(wd.astype(np.float16))
        else:
            crops, scal, ids = stage2.extract_tile(D, A, logit, pred, lab, sizes)

        gl, gs = label_instances(gt, 8, 10)
        if len(gs):
            O = overlap_matrix(gl, len(gs), lab, len(sizes))
            cov_p = O.sum(axis=0) / sizes
        else:
            cov_p = np.zeros(len(sizes))

        C.append(crops.astype(np.float16))
        S.append(scal)
        Y.append((cov_p >= COV_PRED).astype(np.uint8))
        T.append(np.full(len(sizes), k, np.int32))
        B.append(ids)
        if n % batch_hint == 0:
            el = time.time() - t0
            print(f"    {n}/{len(tiles)}  {el:.0f}秒", end="\r", flush=True)

    out = cache_path(cond.name, split, wide, lu,
                     union=union_cond.name if union_cond else None)
    C = np.concatenate(C); S = np.concatenate(S)
    Y = np.concatenate(Y); T = np.concatenate(T); B = np.concatenate(B)
    kw = {"wide": np.concatenate(WD)} if wide else {}
    np.savez_compressed(out, crops=C, scalars=S, label=Y, tile=T, blob=B, **kw)
    print(f"  書き出し: {out.name}  かたまり {len(Y):,}（的中 {int(Y.sum()):,} / "
          f"外れ {int((1 - Y).sum()):,} = {100 * (1 - Y.mean()):.1f}%）"
          f"  {out.stat().st_size / 2**20:.0f}MB")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def auc(y, s):
    import pandas as pd
    y = np.asarray(y, bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def augment(x, jitter=0):
    """左右上下反転と90度回転。かたまりの向きに意味は無いので全部使える。

    学習に使えるかたまりが 7,889個しかないので、これが無いと
    297k パラメータの CNN はすぐ過学習する（実測で ep12 が最良）。

    jitter>0 で**窓をずらす**。オーバーラップ学習（8節）の2段目版にあたる。
    推論では窓をかたまりの重心に合わせるが、タイル外周のかたまりは
    モザイクの端で切り詰められて中心から外れるので、学習でずらしておくと
    その変動に強くなる、という狙い。
    """
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    if jitter > 0:
        n = x.shape[-1]
        x = torch.nn.functional.pad(x, (jitter,) * 4, mode="replicate")
        dy = int(torch.randint(0, 2 * jitter + 1, (1,)).item())
        dx = int(torch.randint(0, 2 * jitter + 1, (1,)).item())
        x = x[..., dy:dy + n, dx:dx + n]
    return x


def train(cond, name, device, epochs=60, batch_size=128, lr=1e-3,
          patience=12, seed=42, split_seed=42, val_frac=0.2, use_augment=False,
          no_scalar=False, no_logit=False, weight_decay=0.0, jitter=0,
          wide=False, lu=False, no_narrow=False, union=None):
    f = cache_path(cond.name, "val", wide, lu, union=union)
    if not f.exists():
        sys.exit(f"{f} がありません。先に --build を実行してください")
    z = np.load(f)
    X, S, Y, T = z["crops"], z["scalars"], z["label"], z["tile"]
    Wd = z["wide"] if wide else None
    if wide:
        stage2.set_wide_channels(lu)
        assert Wd.shape[1] == stage2.N_WIDE_CH, (Wd.shape, stage2.N_WIDE_CH)
        print(f"  広域の枝 {Wd.shape[1]}ch"
              f"{'・土地利用こみ' if lu else ''}")
    print(f"かたまり {len(Y):,}（的中率 {100 * Y.mean():.1f}%）")

    # タイル単位で分ける。同じタイルのかたまりが両側に跨がらないように
    tiles = np.unique(T)
    # タイル分割は split_seed、モデル初期化とシャッフルは seed。
    # 分けておくと「分割のゆらぎ」と「初期化のゆらぎ」を切り分けられる。
    rng = np.random.default_rng(split_seed)
    rng.shuffle(tiles)
    n_val = int(len(tiles) * val_frac)
    val_tiles = set(tiles[:n_val].tolist())
    is_val = np.array([t in val_tiles for t in T])
    print(f"タイル {len(tiles)} → 学習 {len(tiles) - n_val} / 検証 {n_val}"
          f"（分割 seed={split_seed} / 初期化 seed={seed}）")
    print(f"かたまり 学習 {int((~is_val).sum()):,} / 検証 {int(is_val.sum()):,}")
    assert not (set(T[is_val].tolist()) & set(T[~is_val].tolist())), "タイルが跨がっている"

    if no_scalar:
        # 画像だけに判別情報があるかを見る診断。スカラーを全部0にする
        S = np.zeros_like(S)
        print("※ スカラーを全部0にした（画像だけで分けられるかの診断）")

    if no_narrow:
        # 近傍の切り出しを全部0にする。広域だけで分けられるかの診断。
        # 14.3 で近傍 64x64 の情報はロジットと重複と分かったので、
        # **広域 1,147m に新しい情報があるか**を単独で見る。
        X = np.zeros_like(X)
        print("※ 近傍の切り出しを全部0にした（広域だけで分けられるかの診断）")

    if no_logit:
        # ロジットを画像からもスカラーからも消す。
        # ch5 がロジットマップなので、スカラーだけ0にしても情報は残っている。
        # 「地形と写真の文脈だけで分けられるか」を見るにはこちらが必要。
        X = X.copy()
        X[:, 4] = 0
        S = S.copy()
        S[:, 2] = 0
        S[:, 3] = 0
        print("※ ロジットを画像(ch5)とスカラーの両方から消した"
              "（地形と写真だけで分けられるかの診断）")

    torch.manual_seed(seed)
    model = stage2.Stage2Net(
        wide_ch=stage2.N_WIDE_CH if wide else 0).to(device)
    print(f"Stage2Net パラメータ {stage2.n_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    def tensors(mask):
        return (torch.from_numpy(X[mask].astype(np.float32)),
                torch.from_numpy(S[mask].astype(np.float32)),
                torch.from_numpy(Y[mask].astype(np.float32)),
                torch.from_numpy(Wd[mask].astype(np.float32)) if wide else None)

    Xtr, Str, Ytr, Wtr = tensors(~is_val)
    Xva, Sva, Yva, Wva = tensors(is_val)
    n = len(Ytr)

    best, best_ep, since, best_state = -1.0, 0, 0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            xb = Xtr[b].to(device)
            wb = Wtr[b].to(device) if wide else None
            if use_augment or jitter:
                # 近傍と広域は**同じ変換**を当てる（別々にすると向きが食い違う）
                g = torch.get_rng_state()
                xb = augment(xb, jitter=jitter)
                if wide:
                    torch.set_rng_state(g)
                    wb = augment(wb, jitter=jitter)
            out = model(xb, Str[b].to(device), wb)
            loss = lossf(out, Ytr[b].to(device))
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            sc = []
            for s in range(0, len(Yva), 512):
                sc.append(model(Xva[s:s + 512].to(device),
                                Sva[s:s + 512].to(device),
                                Wva[s:s + 512].to(device) if wide else None
                                ).cpu().numpy())
            sc = np.concatenate(sc)
        a = auc(Yva.numpy() > 0.5, sc)
        if a > best:
            best, best_ep, since = a, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:3d}  学習loss {tot / n:.4f}  検証AUC {a:.4f}"
                  f"{'  ←最良' if since == 0 else ''}")
        if since >= patience:
            print(f"  検証AUC が {since} エポック更新されないので打ち切り")
            break

    print(f"最良 ep{best_ep}  検証AUC {best:.4f}")
    # 比較用：スカラーのロジット最大値だけで引いた AUC
    base = auc(Y[is_val] > 0.5, z["scalars"][is_val, 2])
    print(f"参考  ロジット最大値だけの AUC {base:.4f}（2段目が {best:.4f}、"
          f"差 {best - base:+.4f}）")

    model.load_state_dict(best_state)
    out = paths.weights_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": best_ep,
                "val_auc": float(best), "base_auc": float(base),
                "stage1": cond.name, "crop": stage2.CROP,
                "wide_ch": stage2.N_WIDE_CH if wide else 0,
                "wide": bool(wide), "landuse": bool(lu)}, out)
    print("重みを保存:", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--name", default="Stage2_SAM_OL_v1")
    ap.add_argument("--build", action="store_true", help="かたまりデータを作る")
    ap.add_argument("--splits", nargs="*", default=["val", "test"],
                    help="--build の対象。既定は val（学習用）と test（評価用）")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--augment", action="store_true",
                    help="反転と90度回転で増やす（かたまりが少ないので効く）")
    ap.add_argument("--no-scalar", action="store_true",
                    help="スカラーを0にする。画像だけで分けられるかの診断")
    ap.add_argument("--no-narrow", action="store_true",
                    help="近傍の切り出しを0にする。広域だけで分けられるかの診断")
    ap.add_argument("--no-logit", action="store_true",
                    help="ロジットを画像(ch5)とスカラーの両方から消す。"
                         "地形と写真の文脈だけで分けられるかの診断")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--wide", action="store_true",
                    help="広域の枝を足す（モザイクから 256px を 64px に縮小）")
    ap.add_argument("--landuse", action="store_true",
                    help="広域の枝に土地利用も入れる（--wide が要る）")
    ap.add_argument("--pkl", default=None,
                    help="データセットを差し替える。--landuse のとき "
                         "hiroshima_sam_apm_lu.pkl を指定する")
    ap.add_argument("--jitter", type=int, default=0,
                    help="窓をこの画素数までずらす（オーバーラップ学習の2段目版）")
    ap.add_argument("--union-with", default=None,
                    help="和集合にする相方の条件名（展望 21）。"
                         "ロジットは要素ごとの max、予測は OR")
    ap.add_argument("--seed", type=int, default=42,
                    help="モデル初期化とバッチ順。ゆらぎ幅を測るとき振る")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="学習/検証のタイル分割。既定で固定しておくと "
                         "分割のゆらぎを混ぜずに初期化のゆらぎだけ測れる")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not (args.build or args.train):
        sys.exit("--build か --train のどちらかを指定してください")

    cond = get(args.stage1)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"1段目 {cond.name}（凍結）/ device {device}")

    if args.build:
        for sp in args.splits:
            print(f"\n--- かたまりデータを作る: {sp} ---")
            build(cond, sp, device, wide=args.wide, lu=args.landuse,
                  pkl_override=args.pkl,
                  union_cond=get(args.union_with) if args.union_with else None)
    if args.train:
        print(f"\n--- 2段目を学習: {args.name} ---")
        train(cond, args.name, device, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr, patience=args.patience,
              use_augment=args.augment, no_scalar=args.no_scalar,
              no_logit=args.no_logit, weight_decay=args.weight_decay,
              jitter=args.jitter, wide=args.wide, lu=args.landuse,
              no_narrow=args.no_narrow,
              seed=args.seed, split_seed=args.split_seed,
              union=args.union_with)


if __name__ == "__main__":
    main()
