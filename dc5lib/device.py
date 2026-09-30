# -*- coding: utf-8 -*-
"""計算に使うデバイスを選ぶ。

ノートブックには `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
と書かれていた。研究室PC（CUDA）では問題ないが、Mac では MPS が使えるのに
CPU に落ちてしまい、手元で動作確認するのが現実的でなくなる。

    from dc5lib.device import pick_device
    device = pick_device()        # cuda → mps → cpu の順で選ぶ
"""

import os

import torch


def pick_device(prefer=None):
    """cuda → mps → cpu の順に、使えるものを選ぶ。

    環境変数 DC5_DEVICE か引数 prefer で明示指定もできる。
    """
    want = prefer or os.environ.get("DC5_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(device=None):
    d = device or pick_device()
    if d.type == "cuda":
        return f"{d} ({torch.cuda.get_device_name(0)})"
    return str(d)


if __name__ == "__main__":
    print("選ばれたデバイス:", describe())
    print("  cuda:", torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    print("  mps :", bool(mps and mps.is_available()))
