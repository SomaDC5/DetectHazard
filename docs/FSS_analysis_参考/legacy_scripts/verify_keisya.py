# -*- coding: utf-8 -*-
import pickle, numpy as np, torch, gc
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.ndimage import gaussian_filter

device = torch.device("cpu")

class dem_dataset:
    def __init__(self):
        self.No=[];self.DEM=[];self.Mask=[];self.GeoInfo=[];self.EPSG=[];self.Max_H=[];self.Min_H=[]

path = r"C:\Users\hirok\OneDrive\Desktop\CrZ\mayo_dataset\processed_nkeisya_dataset_ver2.pkl"
with open(path,'rb') as f:
    dataset = pickle.load(f)
print("raw N", len(dataset.No))
print("DEM[0] shape/dtype", np.array(dataset.DEM[0]).shape, np.array(dataset.DEM[0]).dtype)
print("DEM[0] min/max", np.array(dataset.DEM[0]).min(), np.array(dataset.DEM[0]).max())
print("Max_H[0]/Min_H[0]", dataset.Max_H[0], dataset.Min_H[0])

dataset2 = dem_dataset()
for i in range(len(dataset.No)):
    if np.max(np.array(dataset.Mask[i])) >= 1:
        dataset2.DEM.append(dataset.DEM[i]); dataset2.Mask.append(dataset.Mask[i]); dataset2.No.append(dataset.No[i])
        dataset2.GeoInfo.append(dataset.GeoInfo[i]); dataset2.EPSG.append(dataset.EPSG[i])
        dataset2.Max_H.append(dataset.Max_H[i]); dataset2.Min_H.append(dataset.Min_H[i])
del dataset
print("filtered N", len(dataset2.No))

for i in range(len(dataset2.No)):
    m = np.array(dataset2.Mask[i]).astype(np.uint8)*255
    m = gaussian_filter(m, sigma=1)
    dataset2.Mask[i] = (m>0).astype(np.uint8)

DEM_raw = np.array(dataset2.DEM, dtype=np.float32)
Mask_raw = np.array(dataset2.Mask, dtype=np.float32)
MaxH = np.array(dataset2.Max_H, dtype=np.float32)
MinH = np.array(dataset2.Min_H, dtype=np.float32)
No_arr = np.array(dataset2.No)
del dataset2; gc.collect()

DEM_t = torch.tensor(np.expand_dims(DEM_raw,1))
Mask_t = torch.tensor(np.expand_dims(Mask_raw,1))

idx_train, idx_test = train_test_split(np.arange(len(No_arr)), test_size=0.2, random_state=42)
print("n_test", len(idx_test))
y_test = Mask_t[idx_test]
print("gt pos frac", y_test.mean().item())

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        def cb(i,o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                  nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.enc1=cb(in_channels,64); self.pool1=nn.MaxPool2d(2)
        self.enc2=cb(64,128); self.pool2=nn.MaxPool2d(2)
        self.enc3=cb(128,256); self.pool3=nn.MaxPool2d(2)
        self.enc4=cb(256,512); self.pool4=nn.MaxPool2d(2)
        self.dropout2=nn.Dropout(0.25)
        self.bottleneck=cb(512,1024); self.dropout=nn.Dropout(0.5)
        self.up4=nn.ConvTranspose2d(1024,512,2,2); self.dec4=cb(1024,512)
        self.up3=nn.ConvTranspose2d(512,256,2,2); self.dec3=cb(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,2); self.dec2=cb(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,2); self.dec1=cb(128,64)
        self.out_conv=nn.Conv2d(64,out_channels,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1)); e3=self.enc3(self.pool2(e2)); e4=self.enc4(self.pool3(e3))
        e4=self.dropout2(e4)
        b=self.dropout(self.bottleneck(self.pool4(e4)))
        d4=self.dropout2(self.dec4(torch.cat([self.up4(b),e4],1)))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.out_conv(d1)

model = UNet().to(device)
ckpt = torch.load(r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_NoOgu_KeisyaOnly\best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print("checkpoint best_f1_score:", ckpt.get("best_f1_score"))

def eval_full(name, X_full):
    all_preds=[]
    with torch.no_grad():
        for s in range(0, X_full.shape[0], 64):
            out = model(X_full[s:s+64])
            preds = (torch.sigmoid(out)>0.5).float()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds).numpy().astype(int).flatten()
    all_targets = y_test.numpy().astype(int).flatten()
    r = recall_score(all_targets, all_preds, zero_division=0)
    p = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    print(f"[{name}] Recall={r:.4f} Precision={p:.4f} F1={f1:.4f}")

X_raw = DEM_t[idx_test].clone()
eval_full("raw", X_raw)

xh = MaxH[idx_train].max(); nh = MinH[idx_train].min()
if xh < MaxH[idx_test].max(): xh = MaxH[idx_test].max()
if nh > MinH[idx_test].min(): nh = MinH[idx_test].min()
X_g = DEM_t[idx_test].clone()
for i in range(X_g.shape[0]):
    X_g[i][0] = (X_g[i][0]-nh)/(xh-nh)
eval_full("global minmax", X_g)

print("pptx target: Recall=0.6744 Precision=0.5430 F1=0.6016")
