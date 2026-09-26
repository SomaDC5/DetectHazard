# 研究用PCへの持ち込み一式

このフォルダの 2 ファイルは numpy / scipy / torch だけで動きます。
`tileanalysis` パッケージには依存しないので、`DetectHazard` の各条件フォルダに
コピーするか、ノートブックから `sys.path` を通して使ってください。

| ファイル | 内容 |
|---|---|
| `instance_loss.py` | 箇所正規化した Focal Tversky Loss と、その重みマップ生成 |
| `instance_eval.py` | 箇所数評価（被覆方式・IoU 方式の比較つき） |

どちらも単体で自己テストが走ります。まずこれを通してから組み込んでください。

```bash
python instance_loss.py     # 重みの性質／現行損失との一致／勾配の確認
python instance_eval.py     # 合成例で 3 方式の挙動を表示
```

`instance_loss.py` の自己テストには、**重みを渡さないときに現行の
`FocalTverskyLoss` と数値が完全一致する**ことの確認が入っています
（差 0.00e+00）。ここが合わなければ、それ以降の比較は意味を持ちません。

---

## 1. 箇所数評価の組み込み

評価セルの推論ループにそのまま足せます。学習側は変えません。

```python
import sys; sys.path.append(r"C:\Users\hirok\dc5\handoff")
from instance_eval import InstanceEvaluator

ev = InstanceEvaluator(save_gt_instances=True)

model.eval()
with torch.no_grad():
    for (dem, air), masks in test_loader:
        out  = model(dem.to(device), air.to(device))
        pred = (torch.sigmoid(out) > 0.5).cpu().numpy()[:, 0]
        gt   = masks.numpy()[:, 0] > 0.5
        ev.add(gt, pred)

print(ev.summary_table())     # 方式ごとの 箇所 Recall / Precision / F値
print(ev.diagnostics())       # 飲み込み・分裂・面積比・「まったく出ていない」率
ev.to_csv("instance_eval")    # instance_eval_tiles.csv / _gt_instances.csv
```

単一入力の条件では `model(dem.to(device))` に変えるだけです。

### 見るべき数字

- **箇所F値（被覆 0.5 / 0.3）** … 本表に出す値
- **面積F値** … 必ず並記する。被覆方式だけを見ていると、予測を広げる方向に
  少し引っ張られる（予測を 2 px 膨張させると箇所F +0.009 / 面積F −0.020）
- **予測面積/正解面積** … 1.4〜1.6 が運用帯。1.9 を超えたら広げすぎ
- **まったく予測が出ていない箇所の割合** … 箇所Recall の天井は `1 − この値`。
  広島の Final SAM+APM では 26.5 % で、天井は 0.735

### しきい値

| パラメータ | 既定 | 意味 |
|---|---|---|
| `cover_gt` | 0.5 | 正解のうち何割が予測に覆われていれば「検出」とするか |
| `cover_pred` | 0.3 | 予測のうち何割が正解に当たっていれば「的中」とするか |
| `min_size` | 10 px | これ未満のかたまりは箇所として数えない |
| `connectivity` | 8 | 斜めに接した画素を同じかたまりとみなすか |

`connectivity` は結果をほとんど変えません（4 近傍でも箇所F 0.6504 対 0.6506）。
`min_size` は効きます（10 → 100 px で箇所F 0.6506 → 0.6996）ので、
根拠を決めて固定してください。

---

## 2. 箇所正規化した損失の組み込み

学習ノートブックへの変更は 4 か所です。
**重みマップは事前に 1 回だけ作り、拡張と一緒に変換して持ち回ります。**
毎回 scipy を呼ぶと遅いためです。

### (a) 重みマップを事前計算する（分割セルの直後）

```python
import sys; sys.path.append(r"C:\Users\hirok\dc5\handoff")
from instance_loss import precompute_weights, InstanceWeightedFocalTverskyLoss

# dataset2.Mask は [N, 1, 128, 128] の uint8（ガウシアン処理済み）
W_ALL = precompute_weights(dataset2.Mask.numpy(), a0=300, p=1.0)   # float16
w_train, w_val, w_test = W_ALL[idx_train], W_ALL[idx_val], W_ALL[idx_test]
```

`train_test_split` に重みも一緒に渡してしまうほうが確実です。

```python
dem_trainval, dem_test, air_trainval, air_test, y_trainval, y_test, \
    No_trainval, No_test, w_trainval, w_test = train_test_split(
        dataset2.DEM, dataset2.AirPhoto, dataset2.Mask, dataset2.No,
        torch.from_numpy(W_ALL),
        test_size=0.2, random_state=42)
```

### (b) JointTransform に重みマップを通す

マスクと同じ変換を、**最近傍補間で**掛けます。

```python
def __call__(self, dem, air, mask, weight=None):
    ...
    if self.hflip and random.random() < 0.5:
        dem, air, mask = TF.hflip(dem), TF.hflip(air), TF.hflip(mask)
        if weight is not None: weight = TF.hflip(weight)
    # vflip も同様
    dem  = TF.affine(dem,  ..., interpolation=TF.InterpolationMode.BILINEAR)
    air  = TF.affine(air,  ..., interpolation=TF.InterpolationMode.BILINEAR)
    mask = TF.affine(mask, ..., interpolation=TF.InterpolationMode.NEAREST)
    if weight is not None:
        weight = TF.affine(weight, ..., interpolation=TF.InterpolationMode.NEAREST)
        return dem, air, mask, weight
    return dem, air, mask
```

### (c) Dataset が重みも返すようにする

```python
class DualInputDataset(Dataset):
    def __init__(self, dem_images, air_images, masks, weights=None, transform=None):
        self.dem_images, self.air_images = dem_images, air_images
        self.masks, self.weights, self.transform = masks, weights, transform

    def __getitem__(self, idx):
        dem  = self.dem_images[idx].float()
        air  = self.air_images[idx].float() / 255.0
        mask = self.masks[idx].float()
        w = (self.weights[idx].float() if self.weights is not None
             else torch.ones_like(mask))
        if self.transform:
            dem, air, mask, w = self.transform(dem, air, mask, w)
        return (dem, air), mask, w
```

### (d) 学習ループと損失

```python
criterion = InstanceWeightedFocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)

for (dem, air), masks, weights in train_loader:
    dem, air = dem.to(device), air.to(device)
    masks, weights = masks.to(device), weights.to(device)
    outputs = model(dem, air)
    loss = criterion(outputs, masks, weights)
```

**検証・テストでは重みを使いません**（`criterion(outputs, masks, None)`）。
モデル選択の指標は従来どおり検証の画素F値のままにしておき、
箇所F値は別に記録して後から比べるのが安全です。

---

## 3. 進め方（この順で）

1. **まず現行のまま** `instance_eval.py` を回して、箇所数評価のベースラインを取る。
   このとき「まったく予測が出ていない箇所の割合」を必ず記録する
2. `instance_loss.py` の自己テストを通し、重み無しで現行損失と一致することを確認
3. `p=0.5, a0=300` で学習 → 面積F値・箇所F値・予測面積比の 3 つを見る
4. `p=1.0, a0=300` / `p=1.0, a0=200` と強めていく
5. 予測面積比が 1.9 を超えるようなら `beta` を 0.3 → 0.5 に上げる
6. ここまでで「出ていない箇所」が減ってから、必要なら `center_lam=0.5` を足す

### なぜこの順か

被覆 0.5 を落ちた箇所の **69.4 % は「1 画素も出ていない」**状態で、その大半が
小さい箇所です（100 px 未満では落ちた箇所の 89.0 %）。中心重みは、モデルが
既に見つけている箇所の形を整える手法なので、ここには手が届きません。
中心重みが効きうる「被覆 0.3〜0.5 のあと少しの箇所」は全体の 5.9 % で、
箇所正規化が狙う 26.5 % に対して母数が 4.5 倍違います。

### パラメータの目安

`a0` と `p` を振ったときの「小さい箇所 / 大きい箇所」の 1 画素あたり重み比。

| | a0=100 | a0=200 | a0=300 | a0=600 |
|---|---|---|---|---|
| p=0.5 | 4.9x | 3.5x | 2.8x | 2.0x |
| p=1.0 | 24.0x | 12.0x | 8.0x | 4.0x |

素の 1/面積（下限なし）だと重み比が 938 倍になって学習が壊れるので、
`a0` は必ず指定してください。現状の過小評価が 8.45 倍なので、
**`p=1.0, a0=300`（8.0 倍）がちょうど打ち消す設定**にあたります。
