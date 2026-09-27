# -*- coding: utf-8 -*-
import pickle, numpy as np, torch, gc, random
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.ndimage import gaussian_filter

device = torch.device("cpu")

class photo_dataset:
    def __init__(self):
        self.No=[];self.DEM=[];self.Mask=[];self.GeoInfo=[];self.EPSG=[];self.Max_H=[];self.Min_H=[];self.AirPhoto=[]

path = r"D:\CrZ\FSS\data\photo_dataset.pkl"
with open(path,'rb') as f:
    dataset = pickle.load(f)
print("raw N", len(dataset.No))
print("DEM[0] shape/dtype", np.array(dataset.DEM[0]).shape, np.array(dataset.DEM[0]).dtype)
print("DEM[0] min/max", np.array(dataset.DEM[0]).min(), np.array(dataset.DEM[0]).max())

dataset2 = photo_dataset(); dataset3 = photo_dataset()
for i in range(len(dataset.No)):
    tgt = dataset2 if np.max(np.array(dataset.Mask[i])) >= 1 else dataset3
    tgt.DEM.append(dataset.DEM[i]); tgt.Mask.append(dataset.Mask[i]); tgt.No.append(dataset.No[i])
    tgt.GeoInfo.append(dataset.GeoInfo[i]); tgt.EPSG.append(dataset.EPSG[i])
    tgt.Max_H.append(dataset.Max_H[i]); tgt.Min_H.append(dataset.Min_H[i]); tgt.AirPhoto.append(dataset.AirPhoto[i])
del dataset
print("with-mask", len(dataset2.No), "no-mask", len(dataset3.No))

for i in range(len(dataset2.No)):
    m = np.array(dataset2.Mask[i]).astype(np.uint8)*255
    m = gaussian_filter(m, sigma=1)
    dataset2.Mask[i] = (m>0).astype(np.uint8)

N = len(dataset2.No)
No_arr = np.array(dataset2.No)
idx_train, idx_test = train_test_split(np.arange(N), test_size=0.2, random_state=42)
print("n_test", len(idx_test))

DEM_test_raw = np.stack([np.asarray(dataset2.DEM[i], dtype=np.float32) for i in idx_test])
Air_test_raw = np.stack([np.asarray(dataset2.AirPhoto[i], dtype=np.float32) for i in idx_test])
Mask_test_raw = np.stack([np.asarray(dataset2.Mask[i], dtype=np.float32) for i in idx_test])
del dataset2, dataset3
gc.collect()

dem_test_raw_t = torch.tensor(np.expand_dims(DEM_test_raw, 1))
air_test = torch.tensor(np.transpose(Air_test_raw, (0,3,1,2)) / 255.0)
y_test = torch.tensor(np.expand_dims(Mask_test_raw, 1))
print("gt pos frac", y_test.mean().item())

class MultiEncoderUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        def cb(i,o):
            return nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                  nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.slope_enc1=cb(1,64); self.slope_pool1=nn.MaxPool2d(2)
        self.slope_enc2=cb(64,128); self.slope_pool2=nn.MaxPool2d(2)
        self.slope_enc3=cb(128,256); self.slope_pool3=nn.MaxPool2d(2)
        self.slope_enc4=cb(256,512); self.slope_pool4=nn.MaxPool2d(2)
        self.curv_enc1=cb(3,64); self.curv_pool1=nn.MaxPool2d(2)
        self.curv_enc2=cb(64,128); self.curv_pool2=nn.MaxPool2d(2)
        self.curv_enc3=cb(128,256); self.curv_pool3=nn.MaxPool2d(2)
        self.curv_enc4=cb(256,512); self.curv_pool4=nn.MaxPool2d(2)
        self.dropout2=nn.Dropout(0.25)
        self.bottleneck=cb(1024,1024); self.dropout=nn.Dropout(0.5)
        self.up4=nn.ConvTranspose2d(1024,512,2,2); self.dec4=cb(512+512*2,512)
        self.up3=nn.ConvTranspose2d(512,256,2,2); self.dec3=cb(256+256*2,256)
        self.up2=nn.ConvTranspose2d(256,128,2,2); self.dec2=cb(128+128*2,128)
        self.up1=nn.ConvTranspose2d(128,64,2,2); self.dec1=cb(64+64*2,64)
        self.out_conv=nn.Conv2d(64,out_channels,1)
    def forward(self, dem, air):
        s1=self.slope_enc1(dem); s2=self.slope_enc2(self.slope_pool1(s1)); s3=self.slope_enc3(self.slope_pool2(s2)); s4=self.slope_enc4(self.slope_pool3(s3)); s4=self.dropout2(s4)
        c1=self.curv_enc1(air); c2=self.curv_enc2(self.curv_pool1(c1)); c3=self.curv_enc3(self.curv_pool2(c2)); c4=self.curv_enc4(self.curv_pool3(c3)); c4=self.dropout2(c4)
        fused=torch.cat([self.slope_pool4(s4), self.curv_pool4(c4)],1)
        b=self.dropout(self.bottleneck(fused))
        u4=self.up4(b); d4=self.dec4(torch.cat([u4, torch.cat([s4,c4],1)],1))
        u3=self.up3(d4); d3=self.dec3(torch.cat([u3, torch.cat([s3,c3],1)],1))
        u2=self.up2(d3); d2=self.dec2(torch.cat([u2, torch.cat([s2,c2],1)],1))
        u1=self.up1(d2); d1=self.dec1(torch.cat([u1, torch.cat([s1,c1],1)],1))
        return self.out_conv(d1)

model = MultiEncoderUNet().to(device)
ckpt = torch.load(r"D:\FSS_KeisyaAndAP\best_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()
print("checkpoint best_f1_score:", ckpt.get("best_f1_score"))

def eval_full(name, dem_full, air_full):
    all_preds=[]
    with torch.no_grad():
        for s in range(0, dem_full.shape[0], 32):
            out = model(dem_full[s:s+32], air_full[s:s+32])
            preds = (torch.sigmoid(out)>0.5).float()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds).numpy().astype(int).flatten()
    all_targets = y_test.numpy().astype(int).flatten()
    r = recall_score(all_targets, all_preds, zero_division=0)
    p = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    print(f"[{name}] Recall={r:.4f} Precision={p:.4f} F1={f1:.4f}")

eval_full("raw DEM(=slope), air/255", dem_test_raw_t.clone(), air_test)
print("pptx target (SAM+APM): Recall=0.7500 Precision=0.6413 F1=0.6914")
