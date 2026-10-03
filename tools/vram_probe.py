# -*- coding: utf-8 -*-
"""条件ごとに、学習1ステップのVRAMピークを測って載るかどうかを見る。

FullSkip 系（全スケール結合）は 52〜53M パラメータで、中間特徴が
デコーダ4段ぶん重なるため batch 32 で 15GB 級を食う。16GB のカードでは
他に何か走っていると落ちる。長時間の学習を始める前にこれで確かめる。

    python tools/vram_probe.py FinalFusion_SAM_APM_FullSkip
    python tools/vram_probe.py 条件名A 条件名B --batch 32,16,8

順伝播だけでなく逆伝播まで回す。ピークは optimizer の状態を持つ前の値なので、
Adam を使うなら実際はパラメータ量 x 8 バイトぶん（FullSkip で約0.4GB）上乗せになる。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dc5lib.losses import aux_lambda, aux_stages, build_loss, use_aspp  # noqa: E402
from dc5lib.models import aux_target, build_model                       # noqa: E402
from dc5lib.registry import get                                         # noqa: E402


def probe(name, batch, device):
    """1ステップ回して (ピークGB, パラメータ数M) を返す。落ちたら例外。"""
    cond = get(name)
    lam = aux_lambda(name)
    stages = aux_stages(name) if lam > 0 else ()
    model = build_model(cond.arch, aux_stages=stages, use_aspp=use_aspp(name)).to(device)
    criterion, _ = build_loss(name)
    opt = torch.optim.Adam(model.parameters(), lr=cond.train.get("lr", 1e-4))
    npar = sum(p.numel() for p in model.parameters()) / 1e6

    torch.cuda.reset_peak_memory_stats(device)
    dem = torch.randn(batch, 1, 128, 128, device=device)
    air = torch.randn(batch, 3, 128, 128, device=device)
    mask = (torch.rand(batch, 1, 128, 128, device=device) > 0.9).float()

    model.train()
    opt.zero_grad(set_to_none=True)
    out, aux = model(dem, air, return_aux=True)
    loss = criterion(out, mask)
    if stages:
        # 学習ノートブックと同じく、補助損失は段数で割って総圧力を揃える
        w = lam / max(len(aux), 1)
        for key, a in aux.items():
            loss = loss + w * criterion(a, aux_target(mask, key))
    loss.backward()
    opt.step()

    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    del model, opt, dem, air, mask, out, aux, loss
    torch.cuda.empty_cache()
    return peak, npar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="+")
    ap.add_argument("--batch", default=None,
                    help="カンマ区切り（例 32,16,8）。既定は config の batch_size")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA が使えません。GPUのあるPCで実行してください。")
    device = torch.device("cuda")
    p = torch.cuda.get_device_properties(device)
    total = p.total_memory / 1024 ** 3
    used = torch.cuda.mem_get_info(device)
    free = used[0] / 1024 ** 3
    print(f"{p.name}  全体 {total:.2f} GB / 空き {free:.2f} GB")
    if total - free > 0.5:
        print(f"  ほかに {total - free:.2f} GB 使われています。"
              f"専有して測りたいなら他の学習を止めてください。")
    print()
    print(f"{'条件':<40}{'batch':>6}{'パラメータ':>11}{'ピーク':>9}  判定")
    for name in args.conditions:
        batches = ([int(x) for x in args.batch.split(",")] if args.batch
                   else [get(name).train.get("batch_size", 32)])
        for b in batches:
            try:
                peak, npar = probe(name, b, device)
            except RuntimeError as e:
                msg = "VRAM不足" if "out of memory" in str(e).lower() else str(e)[:40]
                print(f"{name:<40}{b:>6}{'':>11}{'':>9}  × {msg}")
                torch.cuda.empty_cache()
                continue
            mark = "○" if peak < free * 0.9 else "△ ぎりぎり"
            print(f"{name:<40}{b:>6}{npar:>10.2f}M{peak:>8.2f}G  {mark}")


if __name__ == "__main__":
    main()
