# -*- coding: utf-8 -*-
"""2段目の画像は本当に効いているのか。**スカラーだけ**で同じ AUC が出ないか。

なぜ要るか
----------
`14.3` は「画像だけ（スカラーを0に）」と「ロジット抜き」を測ったが、
**その逆（スカラーだけ・画像を見ない）を測っていない。**

いま 7ch の 2段目は検証AUC 0.8350 で、単体の最良は
ロジット平均 0.8224・合意数 0.8215 である。**差は +0.0126 しかない。**
もし 6個のスカラーを並べただけで 0.835 に届くなら、
**297k パラメータの CNN と 64×64 の切り出しは何もしていない**ことになり、
余地は「かたまりの周囲の画像」には無いと決まる。

> `14.4` の3候補のうち「3. そもそも刈れない」を、
> 別の角度から確かめることにあたる。

スカラー6個（`stage2.SCALAR_NAMES` の5つ + 合意数のかたまり平均）で、
**学習の仕方だけを変えて**どこまで出るか。

| 当てるもの | 何を見るか |
|---|---|
| 単体それぞれ | `run_stage2_room.py` と同じ値（検算）|
| ロジスティック回帰 | **線形で足りるか** |
| 勾配ブースティング | **非線形・交互作用で足りるか** |

0.835 に届けば画像は不要。届かなければ差分が画像の寄与である。

    python analysis/run_stage2_scalar_only.py
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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_stage2_room import load, tile_split, STAGE1, UNION3  # noqa: E402
from analysis.run_stage2_train import auc                              # noqa: E402
from dc5lib import stage2                                              # noqa: E402

CNN_AUC = 0.8350      # `24.4` の 7ch 2段目（3シードの最良）


def _solve(A, b):
    """Gauss-Jordan（部分ピボット）。**`np.linalg.solve` を使わない。**

    この環境の numpy は BLAS/LAPACK 経路に入ると無言でプロセスごと落ちる
    （exit 127。`33` と同じ事情）。行列は 8×8 なので手で解いて十分。
    """
    n = len(b)
    M = np.concatenate([np.asarray(A, float), np.asarray(b, float).reshape(n, 1)],
                       axis=1)
    for i in range(n):
        k = i + int(np.argmax(np.abs(M[i:, i])))
        if k != i:
            M[[i, k]] = M[[k, i]]
        M[i] = M[i] / M[i, i]
        for j in range(n):
            if j != i and M[j, i] != 0.0:
                M[j] = M[j] - M[j, i] * M[i]
    return M[:, n].copy()


def _mv(X, b):
    """X @ b を**要素積の和**で書く。gemm を呼ばせないため。"""
    return (X * np.asarray(b)).sum(axis=1)


def logit_irls(X, y, ridge=1e-4, iters=60, tol=1e-9):
    """**sklearn を使わない。** この環境では LogisticRegression が
    無言でプロセスごと落ちる（`33` と同じ事情）。IRLS を自分で書く。

    行列積も使わない（上の `_mv` / 明示の二重ループ）。
    列が 8 本しかないので速度は問題にならない。
    """
    X = np.c_[np.ones(len(X)), X]
    m = X.shape[1]
    b = np.zeros(m)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(_mv(X, b), -30, 30)))
        w = np.maximum(p * (1 - p), 1e-8)
        H = np.empty((m, m))
        for i in range(m):
            xi = X[:, i] * w
            for j in range(i, m):
                H[i, j] = H[j, i] = float((xi * X[:, j]).sum())
        H[np.diag_indices(m)] += ridge
        r = y - p
        grad = np.array([float((X[:, i] * r).sum()) for i in range(m)]) - ridge * b
        step = _solve(H, grad)
        b = b + step
        if np.abs(step).max() < tol:
            break
    return b


def stumps(Xtr, ytr, Xva, n_round=300, lr=0.06, n_bin=32):
    """勾配ブースティング（深さ1の決定株）。外部ライブラリを使わない。

    **非線形と単調でない関係を拾えるか**を見るためだけのもの。
    深さ1なので交互作用は取れないが、`外周までの距離`（AUC 0.4584）のような
    単調でない量が効くかは分かる。
    """
    # 分位で区切る。同じ値が多い列があるので重複を潰す
    edges = [np.unique(np.quantile(Xtr[:, j], np.linspace(0, 1, n_bin + 1)))[1:-1]
             for j in range(Xtr.shape[1])]
    Btr = np.stack([np.searchsorted(edges[j], Xtr[:, j])
                    for j in range(Xtr.shape[1])], axis=1)
    Bva = np.stack([np.searchsorted(edges[j], Xva[:, j])
                    for j in range(Xva.shape[1])], axis=1)
    f_tr = np.full(len(ytr), np.log(ytr.mean() / (1 - ytr.mean())))
    f_va = np.full(len(Bva), f_tr[0])
    for _ in range(n_round):
        p = 1.0 / (1.0 + np.exp(-np.clip(f_tr, -30, 30)))
        g, h = ytr - p, np.maximum(p * (1 - p), 1e-8)
        best = None
        for j in range(Btr.shape[1]):
            nb = int(Btr[:, j].max()) + 1
            G = np.bincount(Btr[:, j], g, nb)
            H = np.bincount(Btr[:, j], h, nb) + 1.0
            gain = float((G * G / H).sum())
            if best is None or gain > best[0]:
                best = (gain, j, G / H)
        _, j, val = best
        f_tr = f_tr + lr * val[Btr[:, j]]
        f_va = f_va + lr * val[np.clip(Bva[:, j], 0, len(val) - 1)]
    return f_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=STAGE1)
    ap.add_argument("--union", nargs="*", default=UNION3)
    ap.add_argument("--agree", action="store_true", default=True)
    a = ap.parse_args()

    X, S, Y, T, f = load(a.stage1, list(a.union) or None, a.agree)
    is_val, _ = tile_split(T)

    names = list(stage2.SCALAR_NAMES)
    cols = [S[:, i].astype(np.float64) for i in range(S.shape[1])]
    if X.shape[1] >= 7:
        # 合意数（ch7）をかたまりの画素だけで平均。ch6 が対象マスク
        m = X[:, 5].astype(np.float32).reshape(len(Y), -1)
        g = X[:, 6].astype(np.float32).reshape(len(Y), -1)
        cols.append((g * m).sum(1) / np.maximum(m.sum(1), 1e-6))
        names.append("合意数")
    del X
    M = np.stack(cols, axis=1)
    # 標準化。IRLS の条件数のため（AUC は単調変換で変わらない）
    M = (M - M[~is_val].mean(0)) / np.maximum(M[~is_val].std(0), 1e-9)
    y = (Y > 0.5).astype(np.float64)

    print(f"データ {f.name}")
    print(f"かたまり {len(Y):,}  スカラー {M.shape[1]}個"
          f"（{'・'.join(names)}）")
    print(f"学習 {int((~is_val).sum()):,} / 検証 {int(is_val.sum()):,}")
    print()

    yv = y[is_val] > 0.5
    print("【単体】検証split")
    print(f"{'量':<22}{'AUC':>8}")
    for i, nm in enumerate(names):
        print(f"{nm:<22}{auc(yv, M[is_val, i]):>8.4f}")
    print()

    print("【組み合わせ】スカラーだけ・画像を一切見ない")
    print(f"{'当て方':<30}{'AUC':>8}")
    b = logit_irls(M[~is_val], y[~is_val])
    a_lin = auc(yv, _mv(M[is_val], b[1:]) + b[0])
    print(f"{'ロジスティック回帰（線形）':<30}{a_lin:>8.4f}")
    a_gb = auc(yv, stumps(M[~is_val], y[~is_val], M[is_val]))
    print(f"{'勾配ブースティング（決定株300本）':<30}{a_gb:>8.4f}")
    print(f"{'2段目CNN（7ch・297k・`24.4`）':<30}{CNN_AUC:>8.4f}")
    print(f"{'オラクル':<30}{1.0:>8.4f}")
    print()

    best = max(a_lin, a_gb)
    d = CNN_AUC - best
    print(f"【判定】スカラーだけの最良 {best:.4f} / CNN {CNN_AUC:.4f}"
          f"  差 {d:+.4f}")
    if d <= 0.005:
        print("  **画像はほとんど効いていない。** 64×64 の切り出しに")
        print("  スカラーを超える情報は無く、余地は画像の側には無い")
    else:
        print(f"  画像の寄与は {d:+.4f}。切り出しは効いている")
    print()
    print("  ※ 係数（標準化後。正=残す側）")
    for nm, c in sorted(zip(names, b[1:]), key=lambda x: -abs(x[1])):
        print(f"    {nm:<22}{c:>+9.4f}")


if __name__ == "__main__":
    main()
