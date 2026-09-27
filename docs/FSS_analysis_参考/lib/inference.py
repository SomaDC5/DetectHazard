# -*- coding: utf-8 -*-
"""
テストデータへのバッチ推論の共通処理。

【Windows/CPU版PyTorchでの注意】
np.transpose や tensorのスライス直後の非連続（non-contiguous）テンソルを
そのままConv2dに渡すと、まれにネイティブ層でアクセス違反（Segmentation Fault /
Windows fatal exception: access violation）を起こすことが確認されている
（DEM＋APM条件の再解析時に発生・特定）。
そのため、モデルに渡す直前に必ず .contiguous() を呼ぶこと。
本モジュールの関数はすべてこの対策込みで実装している。
"""
import numpy as np
import torch


def run_inference_single(model, x_tensor, batch_size=32, device="cpu"):
    """
    単一入力モデル用推論。
    x_tensor: (N,1,H,W)
    戻り値: preds (N,H,W) の 0/1 float配列, inputs (N,H,W)
    """
    model.eval()
    preds_list, inputs_list = [], []
    with torch.no_grad():
        for s in range(0, x_tensor.shape[0], batch_size):
            xb = x_tensor[s:s + batch_size].contiguous().to(device)
            out = model(xb)
            pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()[:, 0]
            preds_list.append(pred)
            inputs_list.append(xb.cpu().numpy()[:, 0])
    return np.concatenate(preds_list), np.concatenate(inputs_list)


def run_inference_dual(model, dem_tensor, air_tensor, batch_size=16, device="cpu"):
    """
    デュアルエンコーダモデル用推論。
    dem_tensor: (N,1,H,W)  air_tensor: (N,3,H,W)
    戻り値: preds (N,H,W), dem_arr (N,H,W), air_arr (N,3,H,W)
    """
    model.eval()
    preds_list, dem_list, air_list = [], [], []
    with torch.no_grad():
        for s in range(0, dem_tensor.shape[0], batch_size):
            dem_b = dem_tensor[s:s + batch_size].contiguous().to(device)
            air_b = air_tensor[s:s + batch_size].contiguous().to(device)
            out = model(dem_b, air_b)
            pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()[:, 0]
            preds_list.append(pred)
            dem_list.append(dem_b.cpu().numpy()[:, 0])
            air_list.append(air_b.cpu().numpy())
    return np.concatenate(preds_list), np.concatenate(dem_list), np.concatenate(air_list)
