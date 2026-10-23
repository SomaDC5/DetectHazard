# -*- coding: utf-8 -*-
"""ロジットの飽和まわりの診断。展望_再現率をどう上げるか.md 18・19・24 の数字を出す。

4つの測定を1本にまとめてある。欲しいものだけフラグで選ぶ。

  --clamp   1段目のロジットが clamp(-10,10) の外にどれだけいるか（18.1）
            「勾配が厳密に0の画素が 97.3%」を出す
  --grad    重みに届く勾配ノルム。clamp の幅とヒンジの λ を振る（18.2）
            「clamp を外しても 1.08倍」「λ を同等にすると 0.019/0.106」を出す
  --stage2  2段目が飽和しているか（19.1）
            「勾配が実質0のかたまりは 0.00%」を出す
  --agree   合意数チャネルに情報があるか（24.3）
            「合意数単体の AUC 0.8220 > ロジット最大値 0.8191」を出す

使い方
------
    python analysis/run_logit_diag.py --clamp
    python analysis/run_logit_diag.py --grad
    python analysis/run_logit_diag.py --stage2 --model Stage2_SAM_OL_v1
    python analysis/run_logit_diag.py --agree
    python analysis/run_logit_diag.py --all

--clamp と --grad は1段目の推論をするので GPU がある方が速い。
--stage2 と --agree はキャッシュ済みのかたまりデータを読むだけ（CPUで足りる）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.data import build_tileset                        # noqa: E402
from dc5lib import inputs as minputs                           # noqa: E402
from dc5lib import paths, stage2                               # noqa: E402
from dc5lib.device import pick_device                          # noqa: E402
from dc5lib.losses import (FocalTverskyLogitMarginLoss,        # noqa: E402
                           FocalTverskyLoss)
from dc5lib.models import build_model_for, load_weights        # noqa: E402
from dc5lib.paths import cache_dir, dataset_path               # noqa: E402
from dc5lib.registry import get                                # noqa: E402

PAIR = ["FinalFusion_SAM_APM", "FinalFusion_SAM_APM_OverlapTrain"]
DIRNAME = {"hiroshima": "Hiroshima", "shimane": "Shimane"}


def _tiles(cond, region="hiroshima"):
    d = DIRNAME[region]
    return build_tileset(dataset_path(d, cond.pkl_for(d)), region,
                         cond.use_airphoto, bg_ratio=0.0, verbose=False,
                         need_landuse=minputs.needs_landuse(cond))


@torch.no_grad()
def clamp_outside(names, dev, clamp=10.0, region="hiroshima"):
    """clamp の外にいる画素の割合。外は勾配が厳密に0（18.1）。"""
    print()
    print(f"=== clamp(±{clamp:g}) の外にいる画素の割合（{region}）===")
    print("torch.clamp は範囲外の勾配が厳密に0。中間の列だけが学習できている。")
    print()
    head = (f"{'条件':40s} {'<-c':>8s} {'中間':>8s} {'>c':>8s} {'勾配0':>8s}"
            f" | {'正解<-c':>9s} {'正解中間':>9s} {'正解>c':>8s}")
    print(head)
    for name in names:
        cond = get(name)
        t = _tiles(cond, region)
        m = build_model_for(cond).to(dev)
        load_weights(m, cond.weights, device=dev)
        m.eval()
        a = dict.fromkeys(("n", "lo", "in", "hi", "g", "glo", "gin", "ghi"), 0)
        for s in range(0, len(t.no), 32):
            e = min(s + 32, len(t.no))
            z = m(*minputs.model_args(
                cond, torch.from_numpy(t.terrain[s:e]).to(dev), t.air[s:e],
                None if t.landuse is None else t.landuse[s:e],
                device=dev))[:, 0]
            g = torch.from_numpy(t.mask[s:e][:, 0]).to(dev) > 0.5
            lo, hi = z < -clamp, z > clamp
            insd = ~lo & ~hi
            a["n"] += z.numel()
            a["lo"] += int(lo.sum())
            a["in"] += int(insd.sum())
            a["hi"] += int(hi.sum())
            a["g"] += int(g.sum())
            a["glo"] += int((lo & g).sum())
            a["gin"] += int((insd & g).sum())
            a["ghi"] += int((hi & g).sum())
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        n, gq = a["n"], max(a["g"], 1)
        print(f"{cond.label[:38]:40s} {a['lo']/n:7.2%} {a['in']/n:7.2%} "
              f"{a['hi']/n:7.2%} {(a['lo']+a['hi'])/n:7.2%} | "
              f"{a['glo']/gq:8.2%} {a['gin']/gq:8.2%} {a['ghi']/gq:7.2%}")


def grad_norms(name, dev, region="hiroshima"):
    """重みに届く勾配ノルム。clamp の幅とヒンジの λ を振る（18.2）。"""
    cond = get(name)
    t = _tiles(cond, region)
    m = build_model_for(cond).to(dev)
    load_weights(m, cond.weights, device=dev)
    m.train()
    dem = torch.from_numpy(t.terrain[:32]).to(dev)
    air = t.air[:32]
    lu = None if t.landuse is None else t.landuse[:32]
    y = (torch.from_numpy(t.mask[:32][:, 0]).to(dev) > 0.5).float()[:, None]

    def wnorm(crit):
        m.zero_grad(set_to_none=True)
        z = m(*minputs.model_args(cond, dem, air, lu, device=dev))
        crit(z, y).backward()
        return sum(float(p.grad.pow(2).sum()) for p in m.parameters()
                   if p.grad is not None) ** 0.5

    print()
    print(f"=== 重みに届く勾配ノルム（{cond.label} / バッチ32）===")
    print("画素単位では「厳密に0」と「σ'(10)=4.5e-5」の違いがあるが、")
    print("重みに届く合計では埋もれる。感度を掛けると |z|=12.5 でも 7.7e-12。")
    print()
    base = wnorm(FocalTverskyLoss(clamp=10.0))
    print(f"{'損失':40s} {'勾配ノルム':>12s} {'従来比':>8s}")
    print(f"{'FocalTversky clamp=10（従来）':40s} {base:>12.4e} {1.0:>8.2f}")
    for c in (15, 20, 30, 1e4):
        v = wnorm(FocalTverskyLoss(clamp=float(c)))
        lab = f"clamp={c:g}" if c < 1e3 else "clamp 無しに相当"
        print(f"{'FocalTversky ' + lab:40s} {v:>12.4e} {v/base:>8.2f}")
    print()
    n_pos = wnorm(FocalTverskyLogitMarginLoss(lam_pos=1.0, lam_neg=0.0,
                                              clamp=30.0))
    n_neg = wnorm(FocalTverskyLogitMarginLoss(lam_pos=0.0, lam_neg=1.0,
                                              clamp=30.0))
    print(f"{'ヒンジ 正解側のみ λ=1':40s} {n_pos:>12.4e} {n_pos/base:>8.0f}")
    print(f"{'ヒンジ 背景側のみ λ=1':40s} {n_neg:>12.4e} {n_neg/base:>8.0f}")
    for lp, ln in ((0.006, 0.03), (0.02, 0.1), (0.06, 0.3)):
        v = wnorm(FocalTverskyLogitMarginLoss(lam_pos=lp, lam_neg=ln,
                                              clamp=30.0))
        print(f"{f'+ ヒンジ λ {lp}/{ln}':40s} {v:>12.4e} {v/base:>8.2f}")
    print()
    print(f"FT と同等にする λ: 正解側 {base/n_pos:.3f} / 背景側 {base/n_neg:.3f}")
    print("既定の 0.02/0.1 はこれを丸めた値。**合計では押し下げる力が5倍**で、")
    print("これが 20.6 の失敗（正解側の凍結が増えた）の原因だった。")
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def _load_blobs(npz, pattern):
    d = Path(cache_dir("stage2"))
    if npz:
        f = Path(npz)
    else:
        cands = sorted(d.glob(pattern))
        f = cands[0] if cands else None
    if f is None or not f.exists():
        sys.exit(f"{d} にかたまりデータがありません（探した形: {pattern}）。"
                 " run_stage2_train.py --build を回してください")
    return f, np.load(f)


def stage2_saturation(model_name, npz=None):
    """2段目が飽和しているか（19.1）。CPUで足りる。"""
    from sklearn.metrics import roc_auc_score
    f, z = _load_blobs(npz, "*__hiroshima__val.npz")
    X, S, Y = z["crops"], z["scalars"], z["label"]
    ck = torch.load(paths.weights_path(model_name), map_location="cpu",
                    weights_only=False)
    stage2.set_agree_channel(bool(ck.get("agree", False)))
    m = stage2.Stage2Net(in_ch=int(ck.get("in_ch", stage2.N_CH_BASE)),
                         wide_ch=int(ck.get("wide_ch", 0)))
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Y), 512):
            out.append(m(torch.from_numpy(X[s:s + 512].astype(np.float32)),
                         torch.from_numpy(S[s:s + 512].astype(np.float32))
                         ).reshape(-1))
    zz = torch.cat(out).numpy()
    y = Y.astype(bool)
    print()
    print(f"=== 2段目は飽和しているか（{model_name} / {f.name}）===")
    print(f"かたまり {len(Y):,}。1段目は clamp の外に 97.3% で勾配が厳密に0だった。")
    print()
    for lab, v in (("的中(残すべき)", zz[y]), ("外れ(捨てるべき)", zz[~y])):
        q = np.percentile(v, [1, 50, 99])
        print(f"  {lab:18s} 最小{v.min():7.2f} 1%{q[0]:7.2f} 中位{q[1]:7.2f}"
              f" 99%{q[2]:7.2f} 最大{v.max():7.2f}")
    g = 1 / (1 + np.exp(-zz)) - Y          # BCE の dL/dz
    print()
    print(f"  BCE の |dL/dz| が 1e-7 未満（実質0）: "
          f"{100*(np.abs(g) < 1e-7).mean():.2f}%"
          f"   中位 {np.median(np.abs(g)):.4e}")
    print(f"  |z| > 10 のかたまり {100*(np.abs(zz) > 10).mean():.2f}%"
          f" / |z| < 2 のかたまり {100*(np.abs(zz) < 2).mean():.2f}%"
          "  ← 判別できていない帯")
    print(f"  AUC {roc_auc_score(Y, zz):.4f}")
    print()
    print("**2段目は飽和していない。ヒンジには直す対象がない**（19.1）。")


def agree_info(npz=None):
    """合意数チャネルに情報があるか（24.3）。CPUで足りる。"""
    from sklearn.metrics import roc_auc_score
    f, z = _load_blobs(npz, "*__agree.npz")
    X, S, Y = z["crops"], z["scalars"], z["label"]
    if X.shape[1] < 7:
        sys.exit(f"{f.name} は {X.shape[1]}ch で合意数が入っていません。"
                 " --agree を付けて --build してください")
    ag = X[:, 6].astype(np.float32)
    msk = X[:, 5] > 0.5
    pos = ag[msk]
    n_mem = int(round(1 / pos[pos > 0].min())) if (pos > 0).any() else 0
    print()
    print(f"=== 合意数チャネル（{f.name}）===")
    print(f"かたまり {len(Y):,}（的中率 {Y.mean():.1%}）/ {n_mem}本の和集合")
    print()
    print("ch7 の値の分布（対象のかたまりの画素のみ）")
    for k in range(1, n_mem + 1):
        v = k / n_mem
        print(f"  {k}本 ({v:.3f})  {100*np.isclose(pos, v, atol=0.01).mean():6.2f}%")
    per = np.array([ag[i][msk[i]].mean() if msk[i].any() else np.nan
                    for i in range(len(Y))])
    ok = ~np.isnan(per)
    print()
    print("かたまりごとの平均合意数")
    print(f"  的中  中位 {np.median(per[ok & (Y == 1)]):.4f}"
          f"   平均 {per[ok & (Y == 1)].mean():.4f}")
    print(f"  外れ  中位 {np.median(per[ok & (Y == 0)]):.4f}"
          f"   平均 {per[ok & (Y == 0)].mean():.4f}")
    print()
    print(f"  合意数 単体の AUC         {roc_auc_score(Y[ok], per[ok]):.4f}")
    print(f"  ロジット最大値 単体の AUC  {roc_auc_score(Y, S[:, 2]):.4f}")
    print(f"  両者の相関                "
          f"{np.corrcoef(per[ok], S[ok, 2])[0, 1]:+.4f}  ← 1に近いと重複")
    print()
    print("**合意数は単一モデルの出力から計算できない量である**（24.7）。")
    print("広域（AUC +0.000）と土地利用（+0.004）が効かなかったのに対して、")
    print("合意数は +0.0105 動いた。14.3 の重複の唯一の例外。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clamp", action="store_true", help="18.1")
    ap.add_argument("--grad", action="store_true", help="18.2")
    ap.add_argument("--stage2", action="store_true", help="19.1")
    ap.add_argument("--agree", action="store_true", help="24.3")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--model", default="Stage2_SAM_OL_v1")
    ap.add_argument("--npz", default=None, help="かたまりデータを直接指定")
    ap.add_argument("--region", default="hiroshima",
                    choices=["hiroshima", "shimane"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not any((args.clamp, args.grad, args.stage2, args.agree, args.all)):
        ap.error("--clamp / --grad / --stage2 / --agree / --all のどれかを指定")

    names = args.conditions or PAIR
    need_gpu = args.clamp or args.grad or args.all
    dev = ((torch.device(args.device) if args.device else pick_device())
           if need_gpu else None)

    if args.clamp or args.all:
        clamp_outside(names, dev, region=args.region)
    if args.grad or args.all:
        grad_norms(names[-1], dev, region=args.region)
    if args.stage2 or args.all:
        stage2_saturation(args.model, args.npz)
    if args.agree or args.all:
        agree_info(args.npz)


if __name__ == "__main__":
    main()
