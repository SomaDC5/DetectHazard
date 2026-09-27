# -*- coding: utf-8 -*-
import os, sys, pickle, json, gc, shutil
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

OUT_DIR = r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_analysis\KeisyaOnly"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cpu")
print("device:", device)

class dem_dataset:
    def __init__(self):
        self.No = []; self.DEM = []; self.Mask = []; self.GeoInfo = []
        self.EPSG = []; self.Max_H = []; self.Min_H = []

path = r"C:\Users\hirok\OneDrive\Desktop\CrZ\mayo_dataset\processed_nkeisya_dataset_ver2.pkl"
with open(path, "rb") as f:
    dataset = pickle.load(f)
print("raw N:", len(dataset.No))

dataset2 = dem_dataset()
for i in range(len(dataset.No)):
    if np.max(np.array(dataset.Mask[i])) >= 1:
        dataset2.DEM.append(dataset.DEM[i])
        dataset2.Mask.append(dataset.Mask[i])
        dataset2.No.append(dataset.No[i])
        dataset2.GeoInfo.append(dataset.GeoInfo[i])
        dataset2.EPSG.append(dataset.EPSG[i])
        dataset2.Max_H.append(dataset.Max_H[i])
        dataset2.Min_H.append(dataset.Min_H[i])
del dataset
print("filtered N:", len(dataset2.DEM))

for i in range(len(dataset2.No)):
    dataset2.Mask[i] = np.array(dataset2.Mask[i])
    dataset2.Mask[i] = dataset2.Mask[i].astype(np.uint8) * 255
    dataset2.Mask[i] = gaussian_filter(dataset2.Mask[i], sigma=1)
    dataset2.Mask[i] = (dataset2.Mask[i] > 0).astype(np.uint8)

dataset2.DEM = np.array(dataset2.DEM, dtype=np.float32)
dataset2.Mask = np.array(dataset2.Mask, dtype=np.float32)
dataset2.DEM = np.expand_dims(dataset2.DEM, axis=1)
dataset2.Mask = np.expand_dims(dataset2.Mask, axis=1)
dataset2.DEM = torch.tensor(dataset2.DEM, dtype=torch.float32)
dataset2.Mask = torch.tensor(dataset2.Mask, dtype=torch.float32)

X_train, X_test, y_train, y_test, No_train, No_test, xh_train, xh_test, nh_train, nh_test = train_test_split(
    dataset2.DEM, dataset2.Mask, dataset2.No, dataset2.Max_H, dataset2.Min_H,
    test_size=0.2, random_state=42
)
del dataset2, X_train, y_train, No_train
gc.collect()

# 検証の結果、SAM(傾斜量図)単一条件はノートブック記載どおりの
# グローバルmin-max正規化 (訓練集合全体のMax_H/Min_Hで正規化) が使われていた
# (この設定でRecall/Precision/F1がスライドの値と完全一致)。
xh = max(xh_train)
nh = min(nh_train)
if xh < max(xh_test):
    xh = max(xh_test)
if nh > min(nh_test):
    nh = min(nh_test)
for i in range(X_test.shape[0]):
    X_test[i][0] = (X_test[i][0] - nh) / (xh - nh)
del xh_train, nh_train
gc.collect()

print("test N:", X_test.shape[0], "xh:", xh, "nh:", nh)

class ImageDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return torch.as_tensor(self.images[idx], dtype=torch.float32), torch.as_tensor(self.masks[idx], dtype=torch.float32)

test_dataset = ImageDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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

model = UNet().to(device)
ckpt = torch.load(r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_NoOgu_KeisyaOnly\best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

all_preds, all_dem, all_gt = [], [], []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_dem.append(images.cpu().numpy())
        all_gt.append(masks.numpy())

preds = np.concatenate(all_preds)[:, 0]
dem_norm = np.concatenate(all_dem)[:, 0]
gt = np.concatenate(all_gt)[:, 0]

TPm = (preds == 1) & (gt == 1)
FPm = (preds == 1) & (gt == 0)
FNm = (preds == 0) & (gt == 1)
TNm = (preds == 0) & (gt == 0)

tp, fp, fn, tn = TPm.sum(), FPm.sum(), FNm.sum(), TNm.sum()
recall = tp / (tp + fn) if (tp+fn) > 0 else 0
precision = tp / (tp + fp) if (tp+fp) > 0 else 0
f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
print(f"[sanity check] Recall={recall:.4f} Precision={precision:.4f} F1={f1:.4f}  (pptx: 0.6744/0.5430/0.6016)")

# 正規化を逆変換して傾斜角相当(度)を復元
slope = dem_norm * (xh - nh) + nh

rows = []
N = preds.shape[0]
for i in range(N):
    gt_area = int(gt[i].sum())
    pred_area = int(preds[i].sum())
    tpx, fpx, fnx = int(TPm[i].sum()), int(FPm[i].sum()), int(FNm[i].sum())
    rec_i = tpx/(tpx+fnx) if (tpx+fnx) > 0 else np.nan
    prec_i = tpx/(tpx+fpx) if (tpx+fpx) > 0 else np.nan
    f1_i = 2*prec_i*rec_i/(prec_i+rec_i) if (not np.isnan(prec_i) and not np.isnan(rec_i) and (prec_i+rec_i) > 0) else np.nan

    mean_slope_all = float(slope[i].mean())
    mean_slope_fn = float(slope[i][FNm[i]].mean()) if fnx > 0 else np.nan
    mean_slope_fp = float(slope[i][FPm[i]].mean()) if fpx > 0 else np.nan
    mean_slope_tp = float(slope[i][TPm[i]].mean()) if tpx > 0 else np.nan

    n_fn_cc, fn_cc_sizes = 0, []
    if fnx > 0:
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(FNm[i].astype(np.uint8))
        n_fn_cc = retval - 1
        fn_cc_sizes = stats[1:, cv2.CC_STAT_AREA].tolist()
    n_fp_cc, fp_cc_sizes = 0, []
    if fpx > 0:
        retval, labels, stats, _ = cv2.connectedComponentsWithStats(FPm[i].astype(np.uint8))
        n_fp_cc = retval - 1
        fp_cc_sizes = stats[1:, cv2.CC_STAT_AREA].tolist()

    rows.append(dict(
        idx=i, No=No_test[i], gt_area=gt_area, pred_area=pred_area,
        TP=tpx, FP=fpx, FN=fnx, recall=rec_i, precision=prec_i, f1=f1_i,
        mean_slope_all=mean_slope_all, mean_slope_fn=mean_slope_fn, mean_slope_fp=mean_slope_fp, mean_slope_tp=mean_slope_tp,
        n_fn_cc=n_fn_cc, n_fp_cc=n_fp_cc,
        max_fn_cc=max(fn_cc_sizes) if fn_cc_sizes else 0,
        max_fp_cc=max(fp_cc_sizes) if fp_cc_sizes else 0,
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
    "condition": "SAM(傾斜量図)単一",
    "n_test_tiles": int(N),
    "recall": recall, "precision": precision, "f1": f1,
    "mean_slope_TP": float(np.nanmean(df["mean_slope_tp"])),
    "mean_slope_FN": float(np.nanmean(df["mean_slope_fn"])),
    "mean_slope_FP": float(np.nanmean(df["mean_slope_fp"])),
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
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(dem_norm[i], cmap="gray"); axes[0].set_title("SAM(slope, norm)"); axes[0].axis("off")
    axes[1].imshow(gt[i], cmap="gray"); axes[1].set_title("GT"); axes[1].axis("off")
    axes[2].imshow(preds[i], cmap="gray"); axes[2].set_title("Pred"); axes[2].axis("off")
    err = np.zeros((*gt[i].shape, 3), dtype=np.uint8)
    err[TPm[i]] = [0, 200, 0]
    err[FNm[i]] = [255, 0, 0]
    err[FPm[i]] = [0, 120, 255]
    axes[3].imshow(err); axes[3].set_title("TP green/FN red/FP blue"); axes[3].axis("off")
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
