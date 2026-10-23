# -*- coding: utf-8 -*-
"""比較用モデルの健全性を学習前に確かめる。

SegFormer では、分類ヘッドの初期化が悪くて初期の出力が損失の
clamp(-10, 10) の外に 62 % も出ており、**勾配が来ない状態**で
学習を始めかけた（docs/展望_再現率をどう上げるか.md）。
同じことを繰り返さないよう、学習の前に次を測る。

  ・出力の形が入力と同じか（128x128 に戻っているか）
  ・パラメータ数（族の違いと規模の違いを混ぜないため）
  ・初期の出力の標準偏差と、clamp の外に出る画素の割合
  ・1 ステップ逆伝播して、勾配が全層に届くか

    python analysis/run_arch_check.py
    python analysis/run_arch_check.py --archs FCNDual PSPNetDual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "models.py").exists():
        sys.path.insert(0, str(_p))
        break

import torch

from dc5lib.models import MODEL_CLASSES, build_model

DEFAULT = ["MultiEncoderUNet", "SegFormerDual",
           "FCNDual", "PSPNetDual", "DeepLabV3Dual"]
CLAMP = 10.0


def check(arch, n=4, size=128, seed=0):
    torch.manual_seed(seed)
    model = build_model(arch)
    dem = torch.randn(n, 1, size, size)
    air = torch.randn(n, 3, size, size)

    model.eval()
    with torch.no_grad():
        y = model(dem, air)
    if isinstance(y, (tuple, list)):
        y = y[0]

    n_par = sum(p.numel() for p in model.parameters())
    out_of = (y.abs() > CLAMP).float().mean().item() * 100

    # 勾配が全層に届くか。1 ステップだけ逆伝播する。
    model.train()
    z = model(dem, air)
    if isinstance(z, (tuple, list)):
        z = z[0]
        z.sum().backward()
    else:
        z.sum().backward()
    dead = [name for name, p in model.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    # FCN のスキップスコアは重みを 0 で始めるが、勾配は来る（入力が非零）。

    return {
        "arch": arch,
        "shape": tuple(y.shape),
        "params_M": n_par / 1e6,
        "std": y.std().item(),
        "out_of_clamp_%": out_of,
        "dead": dead,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=DEFAULT)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    unknown = [a for a in args.archs if a not in MODEL_CLASSES]
    if unknown:
        raise SystemExit(f"未知のモデル: {unknown}")

    print(f"{'モデル':<22}{'出力の形':<22}{'百万param':>10}"
          f"{'初期std':>10}{'clamp外%':>10}  勾配なしの層")
    rows = []
    for a in args.archs:
        r = check(a, size=args.size)
        rows.append(r)
        dead = "なし" if not r["dead"] else f"{len(r['dead'])} 件 {r['dead'][:3]}"
        print(f"{r['arch']:<22}{str(r['shape']):<22}{r['params_M']:>10.2f}"
              f"{r['std']:>10.3f}{r['out_of_clamp_%']:>10.2f}  {dead}")

    bad = [r for r in rows if r["out_of_clamp_%"] > 1.0 or r["dead"]]
    print()
    if bad:
        print("※ 学習の前に直すべき条件がある:")
        for r in bad:
            if r["out_of_clamp_%"] > 1.0:
                print(f"   {r['arch']}: 初期の出力が clamp(±{CLAMP:.0f}) の外に "
                      f"{r['out_of_clamp_%']:.1f} %。分類ヘッドを小さく初期化する。")
            if r["dead"]:
                print(f"   {r['arch']}: 勾配が来ない層が {len(r['dead'])} 件。")
        return 1
    print("すべて問題なし。出力は入力と同じ大きさ、"
          "初期の出力は clamp の内側、勾配は全層に届いている。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
