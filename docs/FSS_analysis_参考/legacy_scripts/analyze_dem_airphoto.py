# -*- coding: utf-8 -*-
import os, sys, pickle, json, gc, shutil, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_analysis\DEM_AirPhoto"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
torch.set_num_threads(4)
print("device:", device)

class photo_dataset:
    def __init__(self):
        self.No = []; self.DEM = []; self.Mask = []; self.GeoInfo = []
        self.EPSG = []; self.Max_H = []; self.Min_H = []; self.AirPhoto = []

path = r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\DEM_photo_dataset2.pkl"
with open(path, "rb") as f:
    dataset = pickle.load(f)
print("raw N:", len(dataset.No))

dataset2 = photo_dataset()
dataset3 = photo_dataset()
for i in range(len(dataset.No)):
    if np.max(np.array(dataset.Mask[i])) >= 1:
        dataset2.DEM.append(dataset.DEM[i]); dataset2.Mask.append(dataset.Mask[i]); dataset2.No.append(dataset.No[i])
        dataset2.GeoInfo.append(dataset.GeoInfo[i]); dataset2.EPSG.append(dataset.EPSG[i])
        dataset2.Max_H.append(dataset.Max_H[i]); dataset2.Min_H.append(dataset.Min_H[i])
        dataset2.AirPhoto.append(dataset.AirPhoto[i])
    else:
        dataset3.DEM.append(dataset.DEM[i]); dataset3.Mask.append(dataset.Mask[i]); dataset3.No.append(dataset.No[i])
        dataset3.GeoInfo.append(dataset.GeoInfo[i]); dataset3.EPSG.append(dataset.EPSG[i])
        dataset3.Max_H.append(dataset.Max_H[i]); dataset3.Min_H.append(dataset.Min_H[i])
        dataset3.AirPhoto.append(dataset.AirPhoto[i])
del dataset
print("with-mask N:", len(dataset2.DEM), "no-mask N:", len(dataset3.DEM))

for i in range(len(dataset2.No)):
    m = np.array(dataset2.Mask[i]).astype(np.uint8) * 255
    m = gaussian_filter(m, sigma=1)
    dataset2.Mask[i] = (m > 0).astype(np.uint8)

# フォルダ名 "0per_gausian" = マスク無しタイルを0%追加(=追加なし)
random.seed(42)
n_add = int(len(dataset3.No) * 0)
del dataset3
gc.collect()
print("final N:", len(dataset2.No))

No_arr = np.array(dataset2.No)
N = len(No_arr)
idx_all = np.arange(N)
idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=42)
No_test = No_arr[idx_test]
print("test N:", len(idx_test))

dem_test = np.stack([np.asarray(dataset2.DEM[i], dtype=np.float32) for i in idx_test])
air_test_raw = np.stack([np.asarray(dataset2.AirPhoto[i], dtype=np.float32) for i in idx_test])
mask_test = np.stack([np.asarray(dataset2.Mask[i], dtype=np.float32) for i in idx_test])
del dataset2
gc.collect()

dem_test_t = torch.tensor(np.ascontiguousarray(np.expand_dims(dem_test, 1)))
air_test_t = torch.tensor(np.ascontiguousarray(np.transpose(air_test_raw, (0, 3, 1, 2)) / 255.0), dtype=torch.float32)
y_test = torch.tensor(np.expand_dims(mask_test, 1))

class MultiEncoderUNet(nn.Module):
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
        self.dec4 = self.conv_block(512 + 512*2, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256 + 256*2, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128 + 128*2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64 + 64*2, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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

model = MultiEncoderUNet().to(device)
ckpt = torch.load(r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_DEMAndAirPhoto_0per_gausian\best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("checkpoint best_f1_score:", ckpt.get("best_f1_score"))

BATCH = 16
n_total = dem_test_t.shape[0]
preds_list, dem_list, air_list, gt_list = [], [], [], []
with torch.no_grad():
    for s in range(0, n_total, BATCH):
        dem_b = dem_test_t[s:s+BATCH].contiguous()
        air_b = air_test_t[s:s+BATCH].contiguous()
        outputs = model(dem_b, air_b)
        pred_b = (torch.sigmoid(outputs) > 0.5).float().numpy()[:, 0]
        preds_list.append(pred_b)
        dem_list.append(dem_b.numpy()[:, 0])
        air_list.append(air_b.numpy())
        gt_list.append(y_test[s:s+BATCH].numpy()[:, 0])
        if s % 320 == 0:
            print("processed", s, "/", n_total, flush=True)

preds = np.concatenate(preds_list)
dem_arr = np.concatenate(dem_list)
air_arr = np.concatenate(air_list)
gt = np.concatenate(gt_list)
del preds_list, dem_list, air_list, gt_list
gc.collect()

TPm = (preds == 1) & (gt == 1)
FPm = (preds == 1) & (gt == 0)
FNm = (preds == 0) & (gt == 1)
TNm = (preds == 0) & (gt == 0)

tp, fp, fn, tn = TPm.sum(), FPm.sum(), FNm.sum(), TNm.sum()
recall = tp / (tp + fn) if (tp+fn) > 0 else 0
precision = tp / (tp + fp) if (tp+fp) > 0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
print(f"[sanity check] Recall={recall:.4f} Precision={precision:.4f} F1={f1:.4f}  (pptx: 0.6968/0.6687/0.6825)")

rows = []
N = preds.shape[0]
for i in range(N):
    gt_area = int(gt[i].sum())
    pred_area = int(preds[i].sum())
    tpx, fpx, fnx = int(TPm[i].sum()), int(FPm[i].sum()), int(FNm[i].sum())
    rec_i = tpx/(tpx+fnx) if (tpx+fnx) > 0 else np.nan
    prec_i = tpx/(tpx+fpx) if (tpx+fpx) > 0 else np.nan
    f1_i = 2*prec_i*rec_i/(prec_i+rec_i) if (not np.isnan(prec_i) and not np.isnan(rec_i) and (prec_i+rec_i) > 0) else np.nan

    mean_elev_fn = float(dem_arr[i][FNm[i]].mean()) if fnx > 0 else np.nan
    mean_elev_fp = float(dem_arr[i][FPm[i]].mean()) if fpx > 0 else np.nan
    mean_elev_tp = float(dem_arr[i][TPm[i]].mean()) if tpx > 0 else np.nan

    r_ch, g_ch, b_ch = air_arr[i,0], air_arr[i,1], air_arr[i,2]
    green_idx = (g_ch - r_ch) / (g_ch + r_ch + 1e-6)
    brightness = (r_ch + g_ch + b_ch) / 3.0

    def m(arr, mask):
        return float(arr[mask].mean()) if mask.sum() > 0 else np.nan

    rows.append(dict(
        idx=i, No=No_test[i], gt_area=gt_area, pred_area=pred_area,
        TP=tpx, FP=fpx, FN=fnx, recall=rec_i, precision=prec_i, f1=f1_i,
        mean_elev_fn=mean_elev_fn, mean_elev_fp=mean_elev_fp, mean_elev_tp=mean_elev_tp,
        green_fn=m(green_idx, FNm[i]), green_fp=m(green_idx, FPm[i]), green_tp=m(green_idx, TPm[i]),
        bright_fn=m(brightness, FNm[i]), bright_fp=m(brightness, FPm[i]), bright_tp=m(brightness, TPm[i]),
    ))

df = pd.DataFrame(rows)

def bucket(r):
    if not np.isnan(r.f1) and r.f1 >= 0.75:
        return "good"
    if not np.isnan(r.recall) and r.recall < 0.35:
        return "high_FN"
    if not np.isnan(r.precision) and r.precision < 0.35:
        return "high_FP"
    return "mixed"

df["bucket"] = df.apply(bucket, axis=1)
df.to_csv(os.path.join(OUT_DIR, "error_stats.csv"), index=False, encoding="utf-8-sig")

summary = {
    "condition": "DEM(標高図)+APM(航空写真)",
    "n_test_tiles": int(N),
    "recall": recall, "precision": precision, "f1": f1,
    "mean_elev_TP": float(np.nanmean(df["mean_elev_tp"])),
    "mean_elev_FN": float(np.nanmean(df["mean_elev_fn"])),
    "mean_elev_FP": float(np.nanmean(df["mean_elev_fp"])),
    "green_idx_TP": float(np.nanmean(df["green_tp"])),
    "green_idx_FN": float(np.nanmean(df["green_fn"])),
    "green_idx_FP": float(np.nanmean(df["green_fp"])),
    "brightness_TP": float(np.nanmean(df["bright_tp"])),
    "brightness_FN": float(np.nanmean(df["bright_fn"])),
    "brightness_FP": float(np.nanmean(df["bright_fp"])),
    "bucket_counts": df["bucket"].value_counts().to_dict(),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))

for _sub in ("worst_FN", "worst_FP", "good"):
    _d = os.path.join(OUT_DIR, _sub)
    if os.path.isdir(_d):
        shutil.rmtree(_d)

def save_composite(i, folder, tag):
    d = os.path.join(OUT_DIR, folder)
    os.makedirs(d, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    axes[0].imshow(dem_arr[i], cmap="gray"); axes[0].set_title("DEM"); axes[0].axis("off")
    axes[1].imshow(np.transpose(air_arr[i], (1,2,0))); axes[1].set_title("AirPhoto"); axes[1].axis("off")
    axes[2].imshow(gt[i], cmap="gray"); axes[2].set_title("GT"); axes[2].axis("off")
    axes[3].imshow(preds[i], cmap="gray"); axes[3].set_title("Pred"); axes[3].axis("off")
    err = np.zeros((*gt[i].shape, 3), dtype=np.uint8)
    err[TPm[i]] = [0, 200, 0]
    err[FNm[i]] = [255, 0, 0]
    err[FPm[i]] = [0, 120, 255]
    axes[4].imshow(err); axes[4].set_title("TP green/FN red/FP blue"); axes[4].axis("off")
    r = df.iloc[i]
    fig.suptitle(f"No={r.No} F1={r.f1:.2f} Rec={r.recall:.2f} Prec={r.precision:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"{tag}_{i:05d}_No{r.No}.png"), dpi=100)
    plt.close(fig)

worst_fn = df.sort_values("FN", ascending=False).head(20).index.tolist()
fp_candidates = df[df["FP"] > 0].sort_values(["FP"], ascending=False)
worst_fp = fp_candidates.head(20).index.tolist()
best = df.sort_values("f1", ascending=False).head(15).index.tolist()

for i in worst_fn:
    save_composite(i, "worst_FN", "fn")
for i in worst_fp:
    save_composite(i, "worst_FP", "fp")
for i in best:
    save_composite(i, "good", "good")

print("done. saved to", OUT_DIR)
