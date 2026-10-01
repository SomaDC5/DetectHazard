# -*- coding: utf-8 -*-
"""学習時に使ったモデル定義。

dc5/DetectHazard 以下のノートブックに書かれているクラス定義をそのまま写したもの。
重み（best_model.pth）をそのまま読み込めるように、層の名前と順序は変更していない。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shared_conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ============================================================
# 単一入力 U-Net（OnlyDEM / OnlySAM）
# ============================================================
class UNet(nn.Module):
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
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
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


# ============================================================
# Final Fusion（デュアルエンコーダ・デコーダ側で結合）
# ============================================================
class MultiEncoderUNet(nn.Module):
    def __init__(self, out_channels=1):
        super(MultiEncoderUNet, self).__init__()

        self.slope_enc1 = self.conv_block(1, 64)
        self.slope_pool1 = nn.MaxPool2d(2)
        self.slope_enc2 = self.conv_block(64, 128)
        self.slope_pool2 = nn.MaxPool2d(2)
        self.slope_enc3 = self.conv_block(128, 256)
        self.slope_pool3 = nn.MaxPool2d(2)
        self.slope_enc4 = self.conv_block(256, 512)
        self.slope_pool4 = nn.MaxPool2d(2)

        self.curv_enc1 = self.conv_block(3, 64)
        self.curv_pool1 = nn.MaxPool2d(2)
        self.curv_enc2 = self.conv_block(64, 128)
        self.curv_pool2 = nn.MaxPool2d(2)
        self.curv_enc3 = self.conv_block(128, 256)
        self.curv_pool3 = nn.MaxPool2d(2)
        self.curv_enc4 = self.conv_block(256, 512)
        self.curv_pool4 = nn.MaxPool2d(2)
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
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, dem, air):
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
        dec4 = self.dec4(torch.cat([up4, torch.cat([slope4, curv4], dim=1)], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, torch.cat([slope3, curv3], dim=1)], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, torch.cat([slope2, curv2], dim=1)], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, torch.cat([slope1, curv1], dim=1)], dim=1))

        return self.out_conv(dec1)


# ============================================================
# Early Fusion（入力段階でチャネル結合）
# ============================================================
class EarlyFusionUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        in_ch = 1 + 3

        self.enc1 = _shared_conv_block(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _shared_conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _shared_conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _shared_conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.bottleneck = _shared_conv_block(512, 1024)
        self.dropout = nn.Dropout(0.5)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _shared_conv_block(512 + 512, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _shared_conv_block(256 + 256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _shared_conv_block(128 + 128, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _shared_conv_block(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, dem, air):
        x = torch.cat([dem, air], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e4 = self.dropout2(e4)

        b = self.bottleneck(self.pool4(e4))
        b = self.dropout(b)

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# ============================================================
# Middle Fusion（各ステージで融合し、残差的に足し戻す）
# ============================================================
class FusionGate(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, kernel_size=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, a, b):
        return self.fuse(torch.cat([a, b], dim=1))


class MiddleFusionUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.slope_enc1 = _shared_conv_block(1, 64)
        self.curv_enc1 = _shared_conv_block(3, 64)
        self.fuse1 = FusionGate(64)
        self.pool1 = nn.MaxPool2d(2)

        self.slope_enc2 = _shared_conv_block(64, 128)
        self.curv_enc2 = _shared_conv_block(64, 128)
        self.fuse2 = FusionGate(128)
        self.pool2 = nn.MaxPool2d(2)

        self.slope_enc3 = _shared_conv_block(128, 256)
        self.curv_enc3 = _shared_conv_block(128, 256)
        self.fuse3 = FusionGate(256)
        self.pool3 = nn.MaxPool2d(2)

        self.slope_enc4 = _shared_conv_block(256, 512)
        self.curv_enc4 = _shared_conv_block(256, 512)
        self.fuse4 = FusionGate(512)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.bottleneck = _shared_conv_block(1024, 1024)
        self.dropout = nn.Dropout(0.5)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _shared_conv_block(512 + 512 * 2, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _shared_conv_block(256 + 256 * 2, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _shared_conv_block(128 + 128 * 2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _shared_conv_block(64 + 64 * 2, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _stage(self, slope_in, curv_in, slope_enc, curv_enc, fuse_gate):
        s = slope_enc(slope_in)
        c = curv_enc(curv_in)
        fused = fuse_gate(s, c)
        return s + fused, c + fused

    def forward(self, dem, air):
        slope1, curv1 = self._stage(dem, air, self.slope_enc1, self.curv_enc1, self.fuse1)
        slope2, curv2 = self._stage(self.pool1(slope1), self.pool1(curv1),
                                    self.slope_enc2, self.curv_enc2, self.fuse2)
        slope3, curv3 = self._stage(self.pool2(slope2), self.pool2(curv2),
                                    self.slope_enc3, self.curv_enc3, self.fuse3)
        slope4, curv4 = self._stage(self.pool3(slope3), self.pool3(curv3),
                                    self.slope_enc4, self.curv_enc4, self.fuse4)
        slope4 = self.dropout2(slope4)
        curv4 = self.dropout2(curv4)

        fused_bottom = torch.cat([self.pool4(slope4), self.pool4(curv4)], dim=1)
        b = self.bottleneck(fused_bottom)
        b = self.dropout(b)

        d4 = self.dec4(torch.cat([self.up4(b), slope4, curv4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), slope3, curv3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), slope2, curv2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), slope1, curv1], dim=1))

        return self.out_conv(d1)


# ============================================================
# Attention U-Net（Final Fusion のスキップ接続に Attention Gate）
# ============================================================
class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

    def coefficients(self, g, x):
        """注意係数そのもの（0-1）を返す。可視化・解析用。"""
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return self.psi(psi)


class AttentionMultiEncoderUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()

        self.slope_enc1 = _shared_conv_block(1, 64)
        self.slope_pool1 = nn.MaxPool2d(2)
        self.slope_enc2 = _shared_conv_block(64, 128)
        self.slope_pool2 = nn.MaxPool2d(2)
        self.slope_enc3 = _shared_conv_block(128, 256)
        self.slope_pool3 = nn.MaxPool2d(2)
        self.slope_enc4 = _shared_conv_block(256, 512)
        self.slope_pool4 = nn.MaxPool2d(2)

        self.curv_enc1 = _shared_conv_block(3, 64)
        self.curv_pool1 = nn.MaxPool2d(2)
        self.curv_enc2 = _shared_conv_block(64, 128)
        self.curv_pool2 = nn.MaxPool2d(2)
        self.curv_enc3 = _shared_conv_block(128, 256)
        self.curv_pool3 = nn.MaxPool2d(2)
        self.curv_enc4 = _shared_conv_block(256, 512)
        self.curv_pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.bottleneck = _shared_conv_block(1024, 1024)
        self.dropout = nn.Dropout(0.5)

        self.ag4_slope = AttentionGate(gate_ch=512, skip_ch=512, inter_ch=256)
        self.ag4_curv = AttentionGate(gate_ch=512, skip_ch=512, inter_ch=256)
        self.ag3_slope = AttentionGate(gate_ch=256, skip_ch=256, inter_ch=128)
        self.ag3_curv = AttentionGate(gate_ch=256, skip_ch=256, inter_ch=128)
        self.ag2_slope = AttentionGate(gate_ch=128, skip_ch=128, inter_ch=64)
        self.ag2_curv = AttentionGate(gate_ch=128, skip_ch=128, inter_ch=64)
        self.ag1_slope = AttentionGate(gate_ch=64, skip_ch=64, inter_ch=32)
        self.ag1_curv = AttentionGate(gate_ch=64, skip_ch=64, inter_ch=32)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _shared_conv_block(512 + 512 * 2, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _shared_conv_block(256 + 256 * 2, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _shared_conv_block(128 + 128 * 2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _shared_conv_block(64 + 64 * 2, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, dem, air, return_attention=False):
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
        b = self.bottleneck(fused)
        b = self.dropout(b)

        att = {}
        up4 = self.up4(b)
        if return_attention:
            att["l4_slope"] = self.ag4_slope.coefficients(up4, slope4)
            att["l4_curv"] = self.ag4_curv.coefficients(up4, curv4)
        slope4_att = self.ag4_slope(up4, slope4)
        curv4_att = self.ag4_curv(up4, curv4)
        dec4 = self.dec4(torch.cat([up4, slope4_att, curv4_att], dim=1))

        up3 = self.up3(dec4)
        if return_attention:
            att["l3_slope"] = self.ag3_slope.coefficients(up3, slope3)
            att["l3_curv"] = self.ag3_curv.coefficients(up3, curv3)
        slope3_att = self.ag3_slope(up3, slope3)
        curv3_att = self.ag3_curv(up3, curv3)
        dec3 = self.dec3(torch.cat([up3, slope3_att, curv3_att], dim=1))

        up2 = self.up2(dec3)
        if return_attention:
            att["l2_slope"] = self.ag2_slope.coefficients(up2, slope2)
            att["l2_curv"] = self.ag2_curv.coefficients(up2, curv2)
        slope2_att = self.ag2_slope(up2, slope2)
        curv2_att = self.ag2_curv(up2, curv2)
        dec2 = self.dec2(torch.cat([up2, slope2_att, curv2_att], dim=1))

        up1 = self.up1(dec2)
        if return_attention:
            att["l1_slope"] = self.ag1_slope.coefficients(up1, slope1)
            att["l1_curv"] = self.ag1_curv.coefficients(up1, curv1)
        slope1_att = self.ag1_slope(up1, slope1)
        curv1_att = self.ag1_curv(up1, curv1)
        dec1 = self.dec1(torch.cat([up1, slope1_att, curv1_att], dim=1))

        out = self.out_conv(dec1)
        if return_attention:
            return out, att
        return out


# ============================================================
# TransUNet（Final Fusion のボトルネックを Transformer に）
# ============================================================
class TransformerBottleneck(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=4, mlp_ratio=4, spatial_size=8):
        super().__init__()
        num_tokens = spatial_size * spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * mlp_ratio,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class TransUNetDual(nn.Module):
    def __init__(self, out_channels=1, num_heads=8, num_transformer_layers=4, input_size=128):
        super().__init__()

        self.slope_enc1 = _shared_conv_block(1, 64)
        self.slope_pool1 = nn.MaxPool2d(2)
        self.slope_enc2 = _shared_conv_block(64, 128)
        self.slope_pool2 = nn.MaxPool2d(2)
        self.slope_enc3 = _shared_conv_block(128, 256)
        self.slope_pool3 = nn.MaxPool2d(2)
        self.slope_enc4 = _shared_conv_block(256, 512)
        self.slope_pool4 = nn.MaxPool2d(2)

        self.curv_enc1 = _shared_conv_block(3, 64)
        self.curv_pool1 = nn.MaxPool2d(2)
        self.curv_enc2 = _shared_conv_block(64, 128)
        self.curv_pool2 = nn.MaxPool2d(2)
        self.curv_enc3 = _shared_conv_block(128, 256)
        self.curv_pool3 = nn.MaxPool2d(2)
        self.curv_enc4 = _shared_conv_block(256, 512)
        self.curv_pool4 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.bottleneck_conv = _shared_conv_block(1024, 1024)
        spatial_size = input_size // 16
        self.transformer = TransformerBottleneck(
            dim=1024, num_heads=num_heads,
            num_layers=num_transformer_layers, spatial_size=spatial_size,
        )
        self.dropout = nn.Dropout(0.5)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = _shared_conv_block(512 + 512 * 2, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = _shared_conv_block(256 + 256 * 2, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = _shared_conv_block(128 + 128 * 2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = _shared_conv_block(64 + 64 * 2, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, dem, air):
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
        b = self.bottleneck_conv(fused)
        b = self.transformer(b)
        b = self.dropout(b)

        up4 = self.up4(b)
        dec4 = self.dec4(torch.cat([up4, slope4, curv4], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, slope3, curv3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, slope2, curv2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, slope1, curv1], dim=1))

        return self.out_conv(dec1)


# ============================================================
# 深層監督つき MultiEncoder（スキップ接続に「箇所がある」を教える）
# ============================================================
class DeepSupMultiEncoderUNet(MultiEncoderUNet):
    """MultiEncoderUNet の 64x64 のスキップ接続に、存在マップの監督を足したもの。

    なぜここか（docs/展望.md 2節(c), 5節）
      ボトルネックは 8x8 = 71.7 m/画素で、300px 以下の箇所はサブピクセルになる。
      見逃し率はボトルネックでの画素数に対応しており、小さい箇所がデコーダに
      届く経路はスキップ接続しかない。Attention でゲートを足しても効かなかったのは
      「ここに箇所がある」という情報を新しく作る圧力が無いためなので、
      ゲートではなく監督信号を入れる。

    中心点ヒートマップ（CenterNet 風）ではなく存在マップにした理由
      正解の円形度（4pi*面積/周囲長^2）の中央値は 0.198 で、85.5% が 0.3 未満。
      警戒区域は斜面の裾を這う細長い帯で、湾曲した帯の重心はしばしば領域の外に
      落ちる。中心点1ピークでは形も決まらないので、ダウンサンプルした
      正解マスクそのものを教師にする。

    出力
      return_aux=False（既定） マスクだけ。推論・評価のコードは変更不要
      return_aux=True          (マスク, 存在マップ) の2つ。学習ループで使う
    """

    def __init__(self, out_channels=1, aux_channels=64):
        super().__init__(out_channels=out_channels)
        # slope2 / curv2 はどちらも 64x64 x 128ch。結合して 256ch
        self.aux_head = nn.Sequential(
            nn.Conv2d(128 * 2, aux_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(aux_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_channels, out_channels, kernel_size=1),
        )

    def forward(self, dem, air, return_aux=False):
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

        # 64x64 のスキップ接続を、そのまま補助ヘッドにも流す
        skip2 = torch.cat([slope2, curv2], dim=1)

        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.bottleneck(fused)
        bottleneck = self.dropout(bottleneck)

        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, torch.cat([slope4, curv4], dim=1)], dim=1))
        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, torch.cat([slope3, curv3], dim=1)], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, torch.cat([slope1, curv1], dim=1)], dim=1))

        out = self.out_conv(dec1)
        if return_aux:
            return out, self.aux_head(skip2)
        return out


def downsample_mask(mask, factor=2):
    """正解マスクを補助ヘッドの解像度に落とす。

    平均ではなく max。2x2 のどれか1画素でも正解なら正例にする。
    平均にすると小さい箇所の信号が薄まり、「小さい箇所を拾わせる」という
    狙いと逆に働く。50px(≈7x7) の箇所は 64x64 で約 4x4 画素として残る。
    """
    return torch.nn.functional.max_pool2d(mask, kernel_size=factor, stride=factor)


class AllSkipDeepSupMultiEncoderUNet(MultiEncoderUNet):
    """スキップ接続の複数の段に、存在マップの監督を入れたもの。

    DeepSupMultiEncoderUNet は 64x64 の1段だけを監督する。こちらは段を選べる。

      段 | 解像度   | ch   | 50px の箇所
      ---+---------+------+-------------
      1  | 128x128 |  128 | 7x7
      2  |  64x64  |  256 | 4x4
      3  |  32x32  |  512 | 1.8x1.8
      4  |  16x16  | 1024 | 0.9x0.9（サブピクセル。既定では使わない）

    細長い構造を全スケールで監督するのは HED（Holistically-nested Edge
    Detection）と同じ考え方。正解の円形度中央値は 0.198（85.5% が 0.3 未満）
    で細長いため、相性が良いと考えている。

    補助損失の重みは「合計が aux_lambda になる」ように段数で割る。
    そうすると 1段だけの条件と補助損失の総圧力が揃い、
    スケールを分散させた効果だけを比べられる。

    出力
      return_aux=False（既定） マスクだけ。推論・評価のコードは変更不要
      return_aux=True          (マスク, {段: 存在マップ}) 。学習ループで使う
    """

    SKIP_CH = {1: 64 * 2, 2: 128 * 2, 3: 256 * 2, 4: 512 * 2}

    def __init__(self, out_channels=1, aux_stages=(1, 2, 3), aux_channels=64):
        super().__init__(out_channels=out_channels)
        self.aux_stages = tuple(int(s) for s in aux_stages)
        assert all(s in self.SKIP_CH for s in self.aux_stages),             f"aux_stages は {sorted(self.SKIP_CH)} から選ぶ: {self.aux_stages}"
        self.aux_heads = nn.ModuleDict({
            str(s): nn.Sequential(
                nn.Conv2d(self.SKIP_CH[s], aux_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(aux_channels, out_channels, kernel_size=1),
            ) for s in self.aux_stages
        })

    def forward(self, dem, air, return_aux=False):
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

        skips = {1: torch.cat([slope1, curv1], dim=1),
                 2: torch.cat([slope2, curv2], dim=1),
                 3: torch.cat([slope3, curv3], dim=1),
                 4: torch.cat([slope4, curv4], dim=1)}

        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.dropout(self.bottleneck(fused))

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), skips[4]], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), skips[3]], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), skips[2]], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), skips[1]], dim=1))

        out = self.out_conv(dec1)
        if return_aux:
            return out, {s: self.aux_heads[str(s)](skips[s]) for s in self.aux_stages}
        return out


class SupAttnMultiEncoderUNet(MultiEncoderUNet):
    """監督つき Attention。存在マップを監督し、それでスキップ特徴を変調する。

    なぜこの形か（docs/展望.md 5節「発展」が予告している案）
      Attention U-Net は「効いていない」のではなく「冗長」だった。
      注意係数は正解領域内0.705 / 外0.453 とはっきり絞り込めているのに
      F値が変わらない。ゲートは既にある特徴を重み付けし直すだけで、
      何を強調すべきかを教わっていないため。
      一方、深層監督（DeepSupMultiEncoderUNet）は箇所F +0.003 とわずかに効いた。
      そこで、ゲートの中身を損失で縛る。

        skip ──→ aux_head ──→ 存在マップ ──→ 監督（FocalTversky）
          │                        │
          │                     sigmoid
          │                        ↓
          └────── × (1 + attn) ────────→ デコーダへ

      既存手法では MPRNet (Zamir et al., CVPR 2021) の Supervised Attention
      Module が同じ形。中間出力を監督し、それを attention に変えて特徴を変調する。

    変調を x * (1 + attn) の残差型にした理由
      x * attn にすると、学習初期に attn が約0.5 のとき信号が半減して
      学習が不安定になる。残差型なら attention が役に立たなければ
      モデルが無視できる。MPRNet も同じ形。

    出力
      return_aux=False（既定） マスクだけ。推論・評価のコードは変更不要
      return_aux=True          (マスク, {段: 存在マップ}) 。学習ループで使う
    """

    SKIP_CH = {1: 64 * 2, 2: 128 * 2, 3: 256 * 2, 4: 512 * 2}

    def __init__(self, out_channels=1, aux_stages=(2,), aux_channels=64):
        super().__init__(out_channels=out_channels)
        self.aux_stages = tuple(int(s) for s in aux_stages)
        assert all(s in self.SKIP_CH for s in self.aux_stages),             f"aux_stages は {sorted(self.SKIP_CH)} から選ぶ: {self.aux_stages}"
        self.aux_heads = nn.ModuleDict({
            str(s): nn.Sequential(
                nn.Conv2d(self.SKIP_CH[s], aux_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(aux_channels, out_channels, kernel_size=1),
            ) for s in self.aux_stages
        })

    def forward(self, dem, air, return_aux=False):
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

        skips = {1: torch.cat([slope1, curv1], dim=1),
                 2: torch.cat([slope2, curv2], dim=1),
                 3: torch.cat([slope3, curv3], dim=1),
                 4: torch.cat([slope4, curv4], dim=1)}

        # 監督つき Attention: 存在マップを作り、同じものでスキップを変調する
        aux = {}
        for st in self.aux_stages:
            logits = self.aux_heads[str(st)](skips[st])
            aux[st] = logits
            skips[st] = skips[st] * (1.0 + torch.sigmoid(logits))

        # ボトルネックは変調前の特徴から作る（デコーダ経路だけを変える）
        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.dropout(self.bottleneck(fused))

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), skips[4]], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), skips[3]], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), skips[2]], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), skips[1]], dim=1))

        out = self.out_conv(dec1)
        return (out, aux) if return_aux else out


def aux_target(mask, key):
    """補助出力のキーから、その段の教師（ダウンサンプルした正解マスク）を作る。

    キーは段の番号（2）でも、ブランチ付きの文字列（"2_slope"）でもよい。
    段1は等倍、段2は1/2、段3は1/4。
    """
    stage = int(str(key).split("_")[0])
    return mask if stage == 1 else downsample_mask(mask, 2 ** (stage - 1))


class SupAttnPerBranchMultiEncoderUNet(MultiEncoderUNet):
    """監督つき Attention を、ブランチ別に持つ版。

    SupAttnMultiEncoderUNet との違いは、監督と変調を concat の前に行うこと。

      結合後1ヘッド（SupAttn）      cat(slope, curv) -> 1ヘッド -> 両方に同じ変調
      ブランチ別2ヘッド（この版）     slope -> ヘッドA -> slope を変調
                                    curv  -> ヘッドB -> curv  を変調  -> cat

    なぜ分けるか
      1) 既存の AttentionMultiEncoderUNet はブランチ別にゲートを持っており
         （ag4_slope / ag4_curv ...）、設計が揃う。
      2) 注意係数の解析では、32px の段で地形量ブランチが正解領域内0.705 /
         外0.453 とはっきり絞り込む一方、航空写真側は +0.068 でほぼ素通しだった。
         2つのブランチは有効な注意パターンが違うので、同じ変調を強制するのは
         無理がある。
      3) 結合後に1ヘッドだと、ヘッドは易しい方（地形量）の特徴だけで損失を
         満たせてしまい、航空写真ブランチには監督の圧力がほとんど掛からない。
         ブランチ別なら、各ブランチが独立に存在マップを予測する必要がある。

    教師は両ヘッドとも同じ存在マップなので、新しいラベルは要らない。
    補助損失の重みはヘッド数で割るので、合計は aux_lambda のまま
    （結合後1ヘッド版と総圧力を揃えて比べるため）。

    出力
      return_aux=False（既定） マスクだけ。推論・評価のコードは変更不要
      return_aux=True          (マスク, {"<段>_slope": .., "<段>_curv": ..})
    """

    BRANCH_CH = {1: 64, 2: 128, 3: 256, 4: 512}

    def __init__(self, out_channels=1, aux_stages=(2,), aux_channels=64):
        super().__init__(out_channels=out_channels)
        self.aux_stages = tuple(int(x) for x in aux_stages)
        assert all(x in self.BRANCH_CH for x in self.aux_stages),             f"aux_stages は {sorted(self.BRANCH_CH)} から選ぶ: {self.aux_stages}"

        def head(ch):
            return nn.Sequential(
                nn.Conv2d(ch, aux_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(aux_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(aux_channels, out_channels, kernel_size=1),
            )

        self.aux_heads = nn.ModuleDict({
            f"{st}_{br}": head(self.BRANCH_CH[st])
            for st in self.aux_stages for br in ("slope", "curv")
        })

    def forward(self, dem, air, return_aux=False):
        slope1 = self.slope_enc1(dem)
        slope2 = self.slope_enc2(self.slope_pool1(slope1))
        slope3 = self.slope_enc3(self.slope_pool2(slope2))
        slope4 = self.dropout2(self.slope_enc4(self.slope_pool3(slope3)))

        curv1 = self.curv_enc1(air)
        curv2 = self.curv_enc2(self.curv_pool1(curv1))
        curv3 = self.curv_enc3(self.curv_pool2(curv2))
        curv4 = self.dropout2(self.curv_enc4(self.curv_pool3(curv3)))

        # ボトルネックは変調前の特徴から作る（デコーダ経路だけを変える）
        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.dropout(self.bottleneck(fused))

        sl = {1: slope1, 2: slope2, 3: slope3, 4: slope4}
        cv = {1: curv1, 2: curv2, 3: curv3, 4: curv4}

        # concat の前に、ブランチごとに監督して変調する
        aux = {}
        for st in self.aux_stages:
            s_log = self.aux_heads[f"{st}_slope"](sl[st])
            c_log = self.aux_heads[f"{st}_curv"](cv[st])
            aux[f"{st}_slope"], aux[f"{st}_curv"] = s_log, c_log
            sl[st] = sl[st] * (1.0 + torch.sigmoid(s_log))
            cv[st] = cv[st] * (1.0 + torch.sigmoid(c_log))

        skips = {k: torch.cat([sl[k], cv[k]], dim=1) for k in (1, 2, 3, 4)}

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), skips[4]], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), skips[3]], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), skips[2]], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), skips[1]], dim=1))

        out = self.out_conv(dec1)
        return (out, aux) if return_aux else out


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling。解像度を保ったまま複数の文脈を取る。

    64x64 の段（1画素 = 8.96 m）に置いたときの、各枝が見る範囲。

      枝            受容野      実寸
      1x1              1px      9 m
      3x3 d=2          5px     45 m
      3x3 d=4          9px     81 m   ← 箇所の典型サイズ（中央値328px ≒ 81m四方）
      3x3 d=8         17px    152 m
      大域プーリング      全体    573 m

    既存の aux_head は Conv3x3 のみで受容野 26.9 m しかなく、
    斜面全体と保全対象の有無で決まる指定を判定するには足りなかった
    （docs/次の手_文脈と不均衡.md 1節）。
    """

    def __init__(self, in_ch, out_ch=None, branch_ch=64, dilations=(2, 4, 8)):
        super().__init__()
        out_ch = out_ch or in_ch

        def cbr(k, d):
            return nn.Sequential(
                nn.Conv2d(in_ch, branch_ch, kernel_size=k,
                          padding=(0 if k == 1 else d), dilation=d, bias=False),
                nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))

        self.branches = nn.ModuleList([cbr(1, 1)] + [cbr(3, d) for d in dilations])
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(branch_ch * (len(self.branches) + 1), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        hw = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        feats.append(F.interpolate(self.pool(x), size=hw, mode="bilinear", align_corners=False))
        return self.project(torch.cat(feats, dim=1))


class ASPPDeepSupMultiEncoderUNet(MultiEncoderUNet):
    """スキップ接続に ASPP で文脈を足し、そこを深層監督する。

    docs/次の手_文脈と不均衡.md の案A（ASPP）と案C（補助ヘッドを厚くする）を
    1本で満たす構成。ASPP の出力に補助ヘッドを載せるので、
    補助ヘッドの Precision が上がれば「文脈不足が原因」という診断が確定し、
    同時にデコーダへ渡るスキップも文脈つきになる。

        skip2 (64x64, 256ch)
           └─ + ASPP(skip2) ──┬─→ aux_head ──→ 存在マップ（監督）
                              └─→ デコーダへ

    残差（skip + ASPP(skip)）にしてあるので、ASPP が役に立たなければ
    モデルは元のスキップをそのまま使える。

    なぜ「ボトルネックではなく 64x64 に置くのか」
      TransUNet(94M) は大域文脈をボトルネック(8x8)に入れたが Final(43.6M)より
      有意に悪かった（-0.0069, p=0.002）。8x8 は 71.7 m/画素で、300px 以下の
      箇所はサブピクセルになっており、解像度を失ったあとに文脈を足しても遅い。
      ASPP は解像度を保ったまま文脈を取るので別物。

    出力
      return_aux=False（既定） マスクだけ。推論・評価のコードは変更不要
      return_aux=True          (マスク, {段: 存在マップ})
    """

    SKIP_CH = {1: 64 * 2, 2: 128 * 2, 3: 256 * 2, 4: 512 * 2}

    def __init__(self, out_channels=1, aux_stages=(2,), aux_channels=64,
                 aspp_branch_ch=64, dilations=(2, 4, 8)):
        super().__init__(out_channels=out_channels)
        self.aux_stages = tuple(int(x) for x in aux_stages)
        assert all(x in self.SKIP_CH for x in self.aux_stages),             f"aux_stages は {sorted(self.SKIP_CH)} から選ぶ: {self.aux_stages}"
        self.aspp = nn.ModuleDict({
            str(st): _ASPP(self.SKIP_CH[st], branch_ch=aspp_branch_ch, dilations=dilations)
            for st in self.aux_stages})
        self.aux_heads = nn.ModuleDict({
            str(st): nn.Sequential(
                nn.Conv2d(self.SKIP_CH[st], aux_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(aux_channels), nn.ReLU(inplace=True),
                nn.Conv2d(aux_channels, out_channels, kernel_size=1),
            ) for st in self.aux_stages})

    def forward(self, dem, air, return_aux=False):
        slope1 = self.slope_enc1(dem)
        slope2 = self.slope_enc2(self.slope_pool1(slope1))
        slope3 = self.slope_enc3(self.slope_pool2(slope2))
        slope4 = self.dropout2(self.slope_enc4(self.slope_pool3(slope3)))

        curv1 = self.curv_enc1(air)
        curv2 = self.curv_enc2(self.curv_pool1(curv1))
        curv3 = self.curv_enc3(self.curv_pool2(curv2))
        curv4 = self.dropout2(self.curv_enc4(self.curv_pool3(curv3)))

        skips = {1: torch.cat([slope1, curv1], dim=1),
                 2: torch.cat([slope2, curv2], dim=1),
                 3: torch.cat([slope3, curv3], dim=1),
                 4: torch.cat([slope4, curv4], dim=1)}

        # ボトルネックは文脈を足す前の特徴から作る（デコーダ経路だけを変える）
        fused = torch.cat([self.slope_pool4(slope4), self.curv_pool4(curv4)], dim=1)
        bottleneck = self.dropout(self.bottleneck(fused))

        aux = {}
        for st in self.aux_stages:
            skips[st] = skips[st] + self.aspp[str(st)](skips[st])   # 残差で文脈を足す
            aux[st] = self.aux_heads[str(st)](skips[st])

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), skips[4]], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), skips[3]], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), skips[2]], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), skips[1]], dim=1))

        out = self.out_conv(dec1)
        return (out, aux) if return_aux else out


MODEL_CLASSES = {
    "UNet": UNet,
    "MultiEncoderUNet": MultiEncoderUNet,
    "EarlyFusionUNet": EarlyFusionUNet,
    "MiddleFusionUNet": MiddleFusionUNet,
    "AttentionMultiEncoderUNet": AttentionMultiEncoderUNet,
    "TransUNetDual": TransUNetDual,
    "DeepSupMultiEncoderUNet": DeepSupMultiEncoderUNet,
    "AllSkipDeepSupMultiEncoderUNet": AllSkipDeepSupMultiEncoderUNet,
    "SupAttnMultiEncoderUNet": SupAttnMultiEncoderUNet,
    "SupAttnPerBranchMultiEncoderUNet": SupAttnPerBranchMultiEncoderUNet,
    "ASPPDeepSupMultiEncoderUNet": ASPPDeepSupMultiEncoderUNet,
}


def build_model(class_name):
    if class_name not in MODEL_CLASSES:
        raise KeyError(f"未知のモデルクラス: {class_name}")
    return MODEL_CLASSES[class_name]()


def load_weights(model, checkpoint_path, device="cpu"):
    """best_model.pth を読み込む。

    ノートブックは {"model_state_dict": ..., "epoch": ..., ...} という辞書で
    保存しているが、state_dict がそのまま入っている場合にも対応する。
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True), None
    meta = {}
    if isinstance(ckpt, dict):
        for k in ("epoch", "best_f1_score"):
            if k in ckpt:
                meta[k] = ckpt[k]
    return meta


def forward_arity(class_name: str) -> int:
    """そのモデルの forward が受け取る入力の数。config の inputs と突き合わせる。"""
    import inspect
    cls = MODEL_CLASSES[class_name]
    return len([p for p in inspect.signature(cls.forward).parameters
                if p not in ("self", "return_attention", "return_aux")])


if __name__ == "__main__":
    import inspect

    print(f"{'クラス名':<28}{'forward の引数':<26}{'入力数'}")
    for name, cls in MODEL_CLASSES.items():
        sig = [p for p in inspect.signature(cls.forward).parameters
               if p not in ("self", "return_attention")]
        print(f"{name:<28}{'(' + ', '.join(sig) + ')':<26}{len(sig)}")
    print("\n新しいモデルを足すときは、このファイルにクラスを書いて "
          "MODEL_CLASSES に登録する。\n"
          "登録を忘れると tools/check.py が止めてくれる。")
