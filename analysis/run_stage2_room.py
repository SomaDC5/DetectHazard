# -*- coding: utf-8 -*-
"""2段目の余地が**データ量・容量・情報**のどれで止まっているかを切り分ける。

なぜ要るか
----------
`24.6` の天井 0.9199 は「完璧なかたまり判定ができたら」の値で、
**2段目の上限そのもの**である（箇所P=1 のとき F = 2R/(1+R)、R=0.8517）。
いま 0.7784 なので残りは 0.1415。検証AUC は 0.8350 でオラクルは 1.0。

**ここで手を動かす前に、何が縛っているのかを決めないといけない。**

| もし縛っているのが | 次に作るもの |
|---|---|
| **データ量** | 1段目のK分割交差で、全タイルに「学習に使っていないロジット」を作る |
| **容量** | もっと大きい2段目、または事前学習した符号化器 |
| **情報** | 新しい量を足す（`24.7`：1段目の内側に無い量でなければならない）|

学習に使えるかたまりは 7,889個しかない（1段目が重みの更新に使っていない
val split のタイルだけ）。0.4M の CNN に対してこれは少なく、
実測で ep12 が最良になる＝過学習している。**データ量が本命の疑い。**

何を測るか
----------
1. **データ量の曲線** 学習タイルを 25/50/75/100 % に間引いて検証AUC
   → 100 % でまだ上がり続けていれば**データ量が縛っている**
2. **容量の曲線** width 16/32/64
   → 32 が最良（大きくして落ちる）なら**容量は足りている**
3. **単純な基準** ロジット最大値だけ・合意数だけ・オラクル

いずれも3シードで帯を出す。差は小さいので1本では判定できない。

    python analysis/run_stage2_room.py
    python analysis/run_stage2_room.py --seeds 3 --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_stage2_train import auc, augment, cache_path  # noqa: E402
from dc5lib import stage2                                       # noqa: E402
from dc5lib.device import pick_device                           # noqa: E402

STAGE1 = "FinalFusion_SAM_APM_OverlapTrain"
UNION3 = ["FinalFusion_DEM_APM_OverlapTrain",
          "FinalFusion_SAM_APM_LU_OverlapTrain"]


def load(stage1, union, agree):
    f = cache_path(stage1, "val", union=union, agree=agree)
    if not f.exists():
        sys.exit(f"{f} がありません。run_stage2_train.py --build で作ります")
    z = np.load(f)
    return z["crops"], z["scalars"], z["label"], z["tile"], f


def tile_split(T, split_seed=42, val_frac=0.2):
    """**run_stage2_train.train と同じ分け方。** 比較できるようにする。

    タイル単位で分ける。同じタイルのかたまりが両側に跨がると
    検証AUC が上振れする（隣り合うかたまりは似ているため）。
    """
    tiles = np.unique(T)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(tiles)
    n_val = int(len(tiles) * val_frac)
    val_tiles = set(tiles[:n_val].tolist())
    is_val = np.array([t in val_tiles for t in T])
    return is_val, tiles[n_val:]


def fit(X, S, Y, is_val, train_tiles, T, device, width=32, seed=42,
        epochs=60, patience=12, batch_size=128, lr=1e-3, frac=1.0):
    """2段目を1本学習して、最良の検証AUC を返す。

    **間引きはタイル単位で行う。** かたまり単位で間引くと同じタイルの
    残りのかたまりから情報が漏れて、「データ量を減らした」ことにならない。
    """
    keep = train_tiles.copy()
    if frac < 1.0:
        np.random.default_rng(seed).shuffle(keep)
        keep = keep[:max(1, int(len(keep) * frac))]
    sel = np.isin(T, keep) & ~is_val

    torch.manual_seed(seed)
    model = stage2.Stage2Net(in_ch=X.shape[1], width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    def ten(m):
        return (torch.from_numpy(X[m].astype(np.float32)),
                torch.from_numpy(S[m].astype(np.float32)),
                torch.from_numpy(Y[m].astype(np.float32)))

    Xtr, Str, Ytr = ten(sel)
    Xva, Sva, Yva = ten(is_val)
    n = len(Ytr)

    best, best_ep, since = -1.0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            loss = lossf(model(augment(Xtr[b].to(device)), Str[b].to(device)),
                         Ytr[b].to(device))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sc = np.concatenate([
                model(Xva[s:s + 512].to(device),
                      Sva[s:s + 512].to(device)).cpu().numpy()
                for s in range(0, len(Yva), 512)])
        a = auc(Yva.numpy() > 0.5, sc)
        if a > best:
            best, best_ep, since = a, ep, 0
        else:
            since += 1
        if since >= patience:
            break
    return best, best_ep, n, len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=STAGE1)
    ap.add_argument("--union", nargs="*", default=UNION3)
    ap.add_argument("--agree", action="store_true", default=True)
    ap.add_argument("--no-agree", dest="agree", action="store_false")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--fracs", nargs="*", type=float,
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--widths", nargs="*", type=int, default=[16, 32, 64])
    a = ap.parse_args()

    # **1段目の名前は入れない。** cache_path の union タグは相方だけを並べる
    union = list(a.union) or None
    X, S, Y, T, f = load(a.stage1, union, a.agree)
    is_val, train_tiles = tile_split(T)
    device = pick_device()

    print(f"データ {f.name}")
    print(f"かたまり {len(Y):,}（的中率 {100 * Y.mean():.1f}%）"
          f"  チャネル {X.shape[1]}  装置 {device}")
    print(f"学習タイル {len(train_tiles)} / 検証タイル "
          f"{len(np.unique(T[is_val]))}")
    print(f"学習かたまり {int((~is_val).sum()):,} / 検証かたまり "
          f"{int(is_val.sum()):,}")
    print(flush=True)

    # --- 単純な基準 ---
    yv = Y[is_val] > 0.5
    print("【基準】検証split での単体AUC")
    print(f"{'量':<26}{'AUC':>8}")
    for i, nm in enumerate(stage2.SCALAR_NAMES):
        print(f"{nm:<26}{auc(yv, S[is_val, i]):>8.4f}")
    if X.shape[1] >= 7:
        # 合意数（ch7）をかたまりの画素だけで平均する。ch6 が対象マスク
        m = X[is_val, 5].astype(np.float32).reshape(int(is_val.sum()), -1)
        g = X[is_val, 6].astype(np.float32).reshape(int(is_val.sum()), -1)
        print(f"{'合意数（かたまり平均）':<26}"
              f"{auc(yv, (g * m).sum(1) / np.maximum(m.sum(1), 1e-6)):>8.4f}")
    print(f"{'オラクル':<26}{1.0:>8.4f}")
    print(flush=True)

    # --- データ量の曲線 ---
    print("【データ量】学習タイルを間引く（width 32 固定）")
    print(f"{'割合':>7}{'タイル':>8}{'かたまり':>10}"
          + "".join(f"{'s' + str(i + 1):>9}" for i in range(a.seeds))
          + f"{'平均':>9}{'幅':>8}{'最良ep':>8}")
    curve = {}
    for fr in a.fracs:
        res, eps, nb, nt = [], [], 0, 0
        for s in range(a.seeds):
            b, be, nb, nt = fit(X, S, Y, is_val, train_tiles, T, device,
                                width=32, seed=42 + s, epochs=a.epochs,
                                frac=fr)
            res.append(b)
            eps.append(be)
        curve[fr] = res
        print(f"{fr * 100:>6.0f}%{nt:>8}{nb:>10,}"
              + "".join(f"{r:>9.4f}" for r in res)
              + f"{np.mean(res):>9.4f}{max(res) - min(res):>8.4f}"
              f"{np.mean(eps):>8.1f}", flush=True)
    print(flush=True)

    # --- 容量の曲線 ---
    print("【容量】width を振る（データ 100 %）")
    print(f"{'width':>7}{'パラメータ':>12}"
          + "".join(f"{'s' + str(i + 1):>9}" for i in range(a.seeds))
          + f"{'平均':>9}{'幅':>8}{'最良ep':>8}")
    for w in a.widths:
        npar = stage2.n_params(stage2.Stage2Net(in_ch=X.shape[1], width=w))
        res, eps = [], []
        for s in range(a.seeds):
            b, be, _, _ = fit(X, S, Y, is_val, train_tiles, T, device,
                              width=w, seed=42 + s, epochs=a.epochs, frac=1.0)
            res.append(b)
            eps.append(be)
        print(f"{w:>7}{npar:>12,}"
              + "".join(f"{r:>9.4f}" for r in res)
              + f"{np.mean(res):>9.4f}{max(res) - min(res):>8.4f}"
              f"{np.mean(eps):>8.1f}", flush=True)
    print(flush=True)

    # --- 判定の目安 ---
    if len(a.fracs) >= 2:
        lo, hi = sorted(a.fracs)[-2], sorted(a.fracs)[-1]
        d = np.mean(curve[hi]) - np.mean(curve[lo])
        band = max(max(curve[hi]) - min(curve[hi]),
                   max(curve[lo]) - min(curve[lo]))
        print(f"【判定】{lo * 100:.0f}% → {hi * 100:.0f}% で AUC {d:+.4f}"
              f"（シードの幅 {band:.4f}）")
        print("  幅より大きく上がっていれば**データ量が縛っている**"
              "（1段目のK分割交差に進む価値がある）")
        print("  幅の中なら**データ量は飽和**。容量か情報の側を見る")
        print()
        print("  ※ AUC で判定している。`23.6` で AUC と箇所F の逆転が出て"
              "いるので、\n     ここで上がった構成は必ず箇所F で測り直す。")


if __name__ == "__main__":
    main()
