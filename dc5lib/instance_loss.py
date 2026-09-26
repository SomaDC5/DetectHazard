# -*- coding: utf-8 -*-
"""箇所正規化した Focal Tversky Loss（研究用PCへの持ち込み用・単体で動く）

狙い
----
画素を合計する損失では、小さい警戒区域が構造的に軽く扱われる。
広島テストの正解 8,977 箇所を実測すると、

    100px 未満の箇所   箇所数の 22.1 %   しかし画素数では 2.6 %   → 8.45 倍の過小評価
    3000px 以上の箇所  箇所数の  2.2 %   しかし画素数では 18.8 %  → 9 倍の過大評価

そのうえ、被覆0.5を落ちた箇所の 69.4 % は「1画素も出ていない」状態で、
その大半が小さい箇所だった（100px未満では落ちた箇所の 89.0 %）。
つまり小さい箇所は、境界がずれているのではなく検出そのものができていない。

そこで、**正解の箇所ごとに重みの総和が面積によらず揃うように**重みを掛ける。
これは被覆方式の箇所Recall（= 箇所ごとに 1/面積 で正規化した Recall）を
微分可能にしたものに相当するので、損失項を別に足す必要はない。

設計上の決めごと
----------------
* 重みを掛けるのは TP と FN だけ。**FP には掛けない。**
  こうすると過検出の罰は画素どおりに保たれるので、「箇所を取りにいって
  塗り広げる」方向への暴走を面積側が抑える。
* 前景の重みの総和は前景画素数に一致させる。前景と背景のバランスは
  現行と変わらないので、alpha / beta / gamma をそのまま引き継げる。
* 素の 1/面積 は重み比が 900 倍を超えて発散するため、a0 で下限を切る。
  a0=300 で 34 倍、a0=200 で 52 倍。p=0.5 にするとさらに緩くなる。

使い方
------
    python instance_loss.py          # 自己テスト（元の損失との一致確認つき）

学習ノートブックへの組み込みは同ディレクトリの README.md を参照。
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# ----------------------------------------------------------------------
# 重みマップ
# ----------------------------------------------------------------------
def make_instance_weight(mask, a0=300, p=1.0, connectivity=8, min_size=0,
                         center_lam=0.0, dtype=np.float32):
    """正解マスク 1 枚から重みマップを作る。

    mask        : (H, W) の 0/1。学習時に使うのと同じ、ガウシアン処理済みのもの
    a0          : 面積の下限。これ以下の箇所は a0 とみなす（重みの発散を防ぐ）
    p           : 重みの強さ。1.0 で完全な箇所正規化、0.5 で緩め、0 で無効
    connectivity: 8 なら斜めもつないで 1 つの箇所とみなす
    min_size    : これ未満の画素数の箇所は重み 1（ごみを持ち上げない）
    center_lam  : 0 より大きいと、箇所の中心ほど重くする中心重みを上乗せする。
                  **まず 0 で試すこと。** 箇所正規化の効果を確かめてから足す。

    戻り値: (H, W) の float32。背景は 1.0。前景は総和が前景画素数に一致する。
    """
    mask = np.asarray(mask).astype(bool)
    w = np.ones(mask.shape, dtype=dtype)
    if p == 0 or not mask.any():
        return w

    st = STRUCT8 if connectivity == 8 else STRUCT4
    lab, n = ndimage.label(mask, structure=st)
    if n == 0:
        return w
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(np.float64)

    # 箇所ごとの基本重み: (1 / max(面積, a0)) ** p
    raw = (1.0 / np.maximum(sizes, a0)) ** p
    if min_size > 0:
        raw[sizes < min_size] = raw[sizes >= min_size].mean() if (sizes >= min_size).any() else 1.0

    wmap = np.zeros(mask.shape, dtype=np.float64)
    wmap[mask] = raw[lab[mask] - 1]

    # 中心重み（任意）: 箇所ごとに縁からの距離を 0-1 に正規化して乗せる
    if center_lam > 0:
        dist = np.zeros(mask.shape, dtype=np.float64)
        for i in range(1, n + 1):
            inst = lab == i
            d = ndimage.distance_transform_edt(inst)
            m = d.max()
            if m > 0:
                dist[inst] = d[inst] / m
        wmap[mask] *= (1.0 + center_lam * dist[mask])

    # 前景の重みの総和 = 前景画素数 になるよう正規化（前景/背景バランスを保存）
    s = wmap[mask].sum()
    if s > 0:
        wmap[mask] *= mask.sum() / s
    w[mask] = wmap[mask].astype(dtype)
    return w


def precompute_weights(masks, a0=300, p=1.0, connectivity=8, min_size=0,
                       center_lam=0.0, dtype=np.float16, verbose=True):
    """マスクの配列 [N, 1, H, W] または [N, H, W] から重みマップをまとめて作る。

    学習のたびに scipy を呼ぶと遅いので、事前に 1 回だけ作って持ち回る。
    float16 で保持すれば 14,816 枚 x 128x128 で約 485 MB。
    """
    arr = np.asarray(masks)
    squeeze = arr.ndim == 4
    if squeeze:
        arr = arr[:, 0]
    out = np.empty(arr.shape, dtype=dtype)
    for i in range(len(arr)):
        out[i] = make_instance_weight(arr[i], a0, p, connectivity, min_size,
                                      center_lam, dtype=np.float32).astype(dtype)
        if verbose and i % 2000 == 0:
            print(f"  重みマップ {i}/{len(arr)}", end="\r", flush=True)
    if verbose:
        print(f"  重みマップ {len(arr)}/{len(arr)} 完了      ", flush=True)
    return out[:, None] if squeeze else out


# ----------------------------------------------------------------------
# 損失
# ----------------------------------------------------------------------
class InstanceWeightedFocalTverskyLoss(nn.Module):
    """箇所正規化した Focal Tversky Loss。

    weight を渡さない（None）場合は、現行の FocalTverskyLoss と完全に一致する。
    まず一致を確認してから重みを入れると、切り分けが確実になる。

        alpha : False Negative（見逃し）の重み。現行 0.7
        beta  : False Positive（過検出）の重み。現行 0.3
                ※ 箇所重みは見逃し側を強めるので、塗り広がりが気になる場合は
                  beta を 0.5 まで上げて様子を見ること。
        gamma : Focal の指数。現行 0.75
        weight_fp : True にすると FP にも重みを掛ける（非推奨。面積の罰が緩む）
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6, weight_fp=False):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.smooth = smooth
        self.weight_fp = weight_fp

    def forward(self, logits, targets, weight=None):
        preds = torch.sigmoid(logits.clamp(-10, 10)).reshape(logits.size(0), -1)
        t = targets.reshape(targets.size(0), -1).to(preds.dtype)
        if weight is None:
            w = torch.ones_like(t)
        else:
            w = weight.reshape(weight.size(0), -1).to(preds.dtype)

        TP = (w * preds * t).sum(1)
        FN = (w * t * (1 - preds)).sum(1)
        if self.weight_fp:
            FP = (w * (1 - t) * preds).sum(1)
        else:
            FP = ((1 - t) * preds).sum(1)     # 面積の罰はそのまま

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        tversky = torch.clamp(tversky, min=self.smooth, max=1.0)
        return torch.pow(1 - tversky, self.gamma).mean()


# ----------------------------------------------------------------------
# 自己テスト
# ----------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)

    # --- 1. 重みマップの性質 ---------------------------------------
    mask = np.zeros((128, 128), np.uint8)
    mask[20:24, 20:24] = 1          # 16 px の小さい箇所
    mask[60:100, 40:100] = 1        # 2400 px の大きい箇所
    w = make_instance_weight(mask, a0=300, p=1.0)
    small, large = mask.copy(), mask.copy()
    small[60:100, 40:100] = 0
    large[20:24, 20:24] = 0
    print("■ 重みマップ（a0=300, p=1.0）")
    print(f"  小さい箇所   面積 {small.sum():5d} px  1画素の重み {w[small > 0].mean():7.3f}  "
          f"重みの総和 {w[small > 0].sum():9.1f}")
    print(f"  大きい箇所   面積 {large.sum():5d} px  1画素の重み {w[large > 0].mean():7.3f}  "
          f"重みの総和 {w[large > 0].sum():9.1f}")
    print(f"  前景の重みの総和 {w[mask > 0].sum():.1f} = 前景画素数 {mask.sum()} "
          f"（{'一致' if abs(w[mask > 0].sum() - mask.sum()) < 1 else '不一致'}）")
    print(f"  背景の重み {np.unique(w[mask == 0])}")
    print()

    # --- 2. a0 / p を振ったときの重み比 ------------------------------
    print("■ a0 と p を振ったときの「小さい箇所 / 大きい箇所」の1画素あたり重み比")
    for p in (0.5, 1.0):
        row = []
        for a0 in (100, 200, 300, 600):
            ww = make_instance_weight(mask, a0=a0, p=p)
            row.append(f"a0={a0}: {ww[small > 0].mean() / ww[large > 0].mean():5.1f}x")
        print(f"  p={p}  " + "   ".join(row))
    print()

    # --- 3. 重み無しのとき、現行の損失と一致するか --------------------
    class OriginalFocalTversky(nn.Module):
        """学習ノートブックに書かれている現行の実装（比較用にそのまま写したもの）"""
        def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
            super().__init__()
            self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

        def forward(self, preds, targets):
            preds = torch.sigmoid(preds.clamp(-10, 10))
            preds = preds.view(preds.size(0), -1)
            targets = targets.view(targets.size(0), -1)
            TP = (preds * targets).sum(1)
            FP = ((1 - targets) * preds).sum(1)
            FN = (targets * (1 - preds)).sum(1)
            tv = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
            tv = torch.clamp(tv, min=self.smooth, max=1.0)
            return torch.pow((1 - tv), self.gamma).mean()

    logits = torch.tensor(rng.normal(0, 2, (4, 1, 128, 128)), dtype=torch.float32)
    tgt = torch.tensor(rng.integers(0, 2, (4, 1, 128, 128)), dtype=torch.float32)
    a = OriginalFocalTversky()(logits, tgt).item()
    b = InstanceWeightedFocalTverskyLoss()(logits, tgt, None).item()
    print("■ 重み無しのとき現行実装と一致するか")
    print(f"  現行 {a:.8f} / 本実装 {b:.8f} / 差 {abs(a - b):.2e} "
          f"→ {'一致' if abs(a - b) < 1e-6 else '不一致（要確認）'}")

    # a0 を十分大きくすると全箇所の重みが等しくなり、やはり一致するはず
    wt = torch.tensor(np.stack([make_instance_weight(tgt[i, 0].numpy(), a0=10 ** 9)
                                for i in range(4)])[:, None], dtype=torch.float32)
    c = InstanceWeightedFocalTverskyLoss()(logits, tgt, wt).item()
    print(f"  a0=1e9（全箇所が同じ重み）だと {c:.8f} / 差 {abs(a - c):.2e} "
          f"→ {'一致' if abs(a - c) < 1e-6 else '不一致（要確認）'}")
    print()

    # --- 4. 小さい箇所を落としたときの損失の差 -------------------------
    print("■ 「小さい箇所だけ取りこぼした予測」を、重みがどれだけ罰するか")
    tgt1 = torch.tensor(mask[None, None], dtype=torch.float32)
    pred_miss_small = np.zeros_like(mask, np.float32)
    pred_miss_small[60:100, 40:100] = 1.0        # 大きい箇所だけ当てる
    pred_miss_large = np.zeros_like(mask, np.float32)
    pred_miss_large[20:24, 20:24] = 1.0          # 小さい箇所だけ当てる
    wt1 = torch.tensor(make_instance_weight(mask, a0=300, p=1.0)[None, None])
    lo = lambda a: torch.tensor((a * 20 - 10)[None, None], dtype=torch.float32)
    crit = InstanceWeightedFocalTverskyLoss()
    for name, pr in (("小さい箇所を落とす", pred_miss_small), ("大きい箇所を落とす", pred_miss_large)):
        u = crit(lo(pr), tgt1, None).item()
        v = crit(lo(pr), tgt1, wt1).item()
        print(f"  {name:<18} 重み無し {u:.4f} → 箇所正規化 {v:.4f}  （{v / u:5.2f}倍）")
    print("  小さい箇所を落としたときの罰が重くなっていれば意図どおり。")
    print()

    # --- 5. 勾配が流れるか -------------------------------------------
    x = torch.zeros(2, 1, 64, 64, requires_grad=True)
    t2 = torch.zeros(2, 1, 64, 64)
    t2[:, :, 10:20, 10:20] = 1
    w2 = torch.ones_like(t2)
    loss = InstanceWeightedFocalTverskyLoss()(x, t2, w2)
    loss.backward()
    print("■ 勾配")
    print(f"  loss {loss.item():.6f} / grad norm {x.grad.norm().item():.6f} "
          f"→ {'OK' if torch.isfinite(x.grad).all() and x.grad.norm() > 0 else 'NG'}")


if __name__ == "__main__":
    _selftest()
