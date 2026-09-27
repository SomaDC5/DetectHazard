# -*- coding: utf-8 -*-
"""
急傾斜地崩壊危険区域抽出で使用しているU-Net系モデルの定義。
FSS_NoOgu_DemOnly / FSS_NoOgu_KeisyaOnly / FSS_KeisyaAndAirPhoto /
FSS_DEMAndAirPhoto_0per_gausian の各ノートブックに書かれていたモデルクラスを
そのまま切り出したもの。学習コード側でこのクラス定義を変更した場合は、
必ずこちらにも反映すること（state_dictのキー名が変わると load_state_dict が失敗する）。
"""
import torch
import torch.nn as nn


class UNet(nn.Module):
    """単一入力（DEM単一 / SAM単一）で使用するシンプルなU-Net。"""

    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        self.bottleneck = self.conv_block(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc4 = self.dropout2(enc4)
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.dropout(bottleneck)
        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], 1))
        dec4 = self.dropout2(dec4)
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], 1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], 1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], 1))
        return self.out_conv(dec1)


class MultiEncoderUNet(nn.Module):
    """
    2入力（地形量＋航空写真）用のデュアルエンコーダU-Net（Late Concat）。
    第1エンコーダ（slope_*）は1chの地形量（DEM or SAM）、
    第2エンコーダ（curv_*）は3chの航空写真（RGB）を受け取る。
    forward(dem, air) の引数名は歴史的経緯（元は傾斜量+航空写真用に書かれた
    コードを流用している）でこうなっているだけで、DEM＋APM条件でも
    第1引数には標高図（DEM）をそのまま渡してよい。
    """

    def __init__(self, out_channels=1):
        super(MultiEncoderUNet, self).__init__()
        self.slope_enc1 = self.conv_block(1, 64); self.slope_pool1 = nn.MaxPool2d(2)
        self.slope_enc2 = self.conv_block(64, 128); self.slope_pool2 = nn.MaxPool2d(2)
        self.slope_enc3 = self.conv_block(128, 256); self.slope_pool3 = nn.MaxPool2d(2)
        self.slope_enc4 = self.conv_block(256, 512); self.slope_pool4 = nn.MaxPool2d(2)
        self.curv_enc1 = self.conv_block(3, 64); self.curv_pool1 = nn.MaxPool2d(2)
        self.curv_enc2 = self.conv_block(64, 128); self.curv_pool2 = nn.MaxPool2d(2)
        self.curv_enc3 = self.conv_block(128, 256); self.curv_pool3 = nn.MaxPool2d(2)
        self.curv_enc4 = self.conv_block(256, 512); self.curv_pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        self.bottleneck = self.conv_block(1024, 1024)
        self.dropout = nn.Dropout(0.5)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512 + 512 * 2, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256 + 256 * 2, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128 + 128 * 2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64 + 64 * 2, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, dem, air):
        # dem, air は呼び出し側で必ず .contiguous() したテンソルを渡すこと。
        # (Windows CPU版PyTorchでは、転置(transpose)直後の非連続テンソルを
        #  直接Conv2dに渡すとネイティブ層でアクセス違反を起こすことがある。
        #  lib/inference.py の run_inference() は対策済み。)
        slope1 = self.slope_enc1(dem)
        slope2 = self.slope_enc2(self.slope_pool1(slope1))
        slope3 = self.slope_enc3(self.slope_pool2(slope2))
        slope4 = self.slope_enc4(self.slope_pool3(slope3))
        slope4 = self.dropout2(slope4)

        curv1 = self.curv_enc1(air)
        curv2 = self.curv_enc2(self.curv_pool1(curv1))
        curv3 = self.curv_enc3(self.curv_pool2(curv2))
        curv4 = self.curv_enc4(self.curv_pool3(curv3))
        curv4 = self.dropout2(curv4)

        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.bottleneck(fused)
        bottleneck = self.dropout(bottleneck)

        up4 = self.up4(bottleneck)
        skip4 = torch.cat([slope4, curv4], dim=1)
        dec4 = self.dec4(torch.cat([up4, skip4], dim=1))

        up3 = self.up3(dec4)
        skip3 = torch.cat([slope3, curv3], dim=1)
        dec3 = self.dec3(torch.cat([up3, skip3], dim=1))

        up2 = self.up2(dec3)
        skip2 = torch.cat([slope2, curv2], dim=1)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))

        up1 = self.up1(dec2)
        skip1 = torch.cat([slope1, curv1], dim=1)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))

        return self.out_conv(dec1)


def load_model(model, checkpoint_path, device):
    """best_model.pth を読み込んで eval() 状態で返す。best_f1_score も併せて返す。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt.get("best_f1_score"), ckpt.get("epoch")
