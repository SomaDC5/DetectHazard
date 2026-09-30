# 手順：SAM+APM / MultiEncoderUNet / 箇所正規化損失

作るもの。

| 項目 | 値 |
|---|---|
| 入力 | SAM（傾斜角 1ch）＋ APM（航空写真 3ch） |
| モデル | `MultiEncoderUNet`（Final Fusion。エンコーダ2本をデコーダ側で結合） |
| 損失 | `instance_weighted_focal_tversky`（箇所正規化） |
| ひな形 | `FinalFusion_SAM_APM` |

`FinalFusion_SAM_APM` を土台にするのは、**arch・inputs・dataset・3分割が
最初から全部合っている**から。違うのは損失だけになるので、
`FinalFusion_SAM_APM` 自体がそのまま対照条件になる。

---

## ステップ1　名前を決める

作業は **PCごとのブランチ**で行い、main へは PR 経由でマージする。
main で直接作業しない。2台の変更が混ざる。

```bash
cd ~/Desktop/Master/DetectHazard
git checkout main
git pull
git checkout -b lab-windows        # 自分のPCのブランチ。既にあれば -b を外す
./.venv/bin/python tools/new_experiment.py --list
```

ブランチ名は `DC5_MACHINE` と揃えておくと、どのPCの作業か一目で分かる
（`lab-ubuntu` / `lab-windows` / `home`）。

一覧に無い名前を選ぶ。ここでは `FinalFusion_SAM_APM_InstLoss` とする。

---

## ステップ2　指定できるものを全部指定して作る

**1行で貼ること。** 行継続（`\`）で複数行にすると、貼り付け時に途中で切れて
後半の引数が渡らないことがある。実際それで `--loss` が効かず、
ひな形の `focal_tversky` のまま作られたことがあった。

```bash
./.venv/bin/python tools/new_experiment.py FinalFusion_SAM_APM_InstLoss --from FinalFusion_SAM_APM --arch MultiEncoderUNet --family Final --inputs SAM+APM --loss instance_weighted_focal_tversky --a0 300 --p 0.5 --beta 0.5 --epochs 1000 --seed 42 --note "SAM+APM・MultiEncoder に箇所正規化損失。FinalFusion_SAM_APM の損失違い"
```

実行すると、**作られた内容がその場で表示される**。ここで意図どおりか確かめる。

```
作成された条件
  arch     MultiEncoderUNet
  family   Final
  inputs   SAM+APM (1+3 ch)
  augment  False
  bg_ratio 0.0
  epochs   1000   seed 42
  loss     instance_weighted_focal_tversky(alpha=0.7, beta=0.5, gamma=0.75, a0=300, p=0.5, center_lam=0.0)  ※重みマップが必要
  dataset  Hiroshima: hiroshima_sam_apm.pkl
  dataset  Shimane: shimane_sam_apm.pkl
```

`loss` が `focal_tversky(...)` になっていたら `--loss` が渡っていない。
その場合はこう出る。

```
  ※ --loss を指定していないので、ひな形の focal_tversky を引き継いでいます。
    違うものにしたい場合は --loss を付け直すか、config.yaml の train.loss を直接編集してください。
```

引数の意味。

| 引数 | 意味 | なぜこの値か |
|---|---|---|
| `--from` | ひな形 | arch・inputs・dataset・3分割が最初から合う |
| `--arch` | モデルクラス | `python -m dc5lib.models` の一覧から |
| `--family` | 図表のグループ分け | 精度には影響しない |
| `--inputs` | 入力の構成 | `SAM+APM` → 1ch + 3ch |
| `--loss` | 損失 | 箇所正規化版 |
| `--a0 300` | 面積の下限 | 素の 1/面積 は重み比が938倍になり発散する。300で34倍に収まる |
| `--p 0.5` | 重みの強さ | 過小評価は8.45倍だが、まず緩めから。効けば 1.0 |
| `--beta 0.5` | 過検出の重み | 既定0.3のままだと、箇所重みと合わさって塗り広げ方向に押しすぎる |
| `--epochs` `--seed` | | ひな形と同じだが明示しておく |

`--arch` `--family` `--inputs` は**ひな形と同じ値**なので省略してもよいが、
明示しておくと config を見ただけで意図が分かる。

### 確認

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from dc5lib.registry import get
from dc5lib.losses import describe
c = get('FinalFusion_SAM_APM_InstLoss')
print(f'arch={c.arch} family={c.family} inputs={[i.source for i in c.inputs]} ch={c.in_channels}')
print(f'dataset={c.dataset}')
print(f'loss={describe(c.name)}')
"
```

期待する出力。

```
arch=MultiEncoderUNet family=Final inputs=['SAM', 'APM'] ch=[1, 3]
dataset={'Hiroshima': 'hiroshima_sam_apm.pkl', 'Shimane': 'shimane_sam_apm.pkl'}
loss=instance_weighted_focal_tversky(alpha=0.7, beta=0.5, gamma=0.75, a0=300, p=0.5, center_lam=0.0)  ※重みマップが必要
```

`※重みマップが必要` が出ていること。これが次の作業の前提になる。

---

## ステップ3　名前を予約する

```bash
./.venv/bin/python tools/check.py
git add experiments/FinalFusion_SAM_APM_InstLoss
git commit -m "FinalFusion_SAM_APM_InstLoss の条件を追加"
git push -u origin lab-windows
```

**名前の予約だけなら、この時点で PR を出して main にマージしてよい。**
config.yaml が main に載れば、相手が `git pull` した時点で名前が見える。
学習の完了を待つ必要はない。

---

## ステップ4　ノートブックに重みマップを配線する（手作業）

**ここは自動化できていない。** ノートブックの冒頭に手順を書いた注意セルが
挿入されているので、それを見ながら5箇所を直す。

`experiments/FinalFusion_SAM_APM_InstLoss/Keisya_airPhoto_final6.ipynb`

### ① セル0（dc5 共通設定）に import を足す

```python
from dc5lib.instance_loss import precompute_weights
from dc5lib.registry import get
```

### ② セル10（分割）の冒頭に重みマップの生成を足し、分割に流す

`dem_trainval, dem_test, ... = train_test_split(...)` の**前**に置く。

```python
_, NEEDS_WEIGHT = build_loss(CONDITION)
print("損失:", describe(CONDITION))

if NEEDS_WEIGHT:
    wp = weight_params(CONDITION)            # config の a0 / p / center_lam
    print("重みマップの設定:", wp)
    W_ALL = precompute_weights(dataset2.Mask.numpy(), **wp)   # [N,1,H,W] float16
    W_ALL = torch.from_numpy(W_ALL)
else:
    W_ALL = torch.ones_like(dataset2.Mask, dtype=torch.float16)
```

そのうえで、既存の `train_test_split` の**両方**に `W_ALL` を足す。

```python
dem_trainval, dem_test, air_trainval, air_test, y_trainval, y_test, \
    No_trainval, No_test, bg_idx_trainval, bg_idx_test, w_trainval, w_test = \
    train_test_split(dataset2.DEM, dataset2.AirPhoto, dataset2.Mask,
                     dataset2.No, dataset2.BgSourceIndex, W_ALL,
                     test_size=0.2, random_state=42)

dem_train, dem_val, air_train, air_val, y_train, y_val, \
    No_train, No_val, bg_idx_train, bg_idx_val, w_train, w_val = \
    train_test_split(dem_trainval, air_trainval, y_trainval,
                     No_trainval, bg_idx_trainval, w_trainval,
                     test_size=0.25, random_state=42)
```

> **`test_size` と `random_state` は絶対に変えない。**
> 変えるとテスト集合が他の条件とずれて、比較できなくなる。

### ③ セル11（JointTransform）に weight を通す

`__call__` の引数に `weight=None` を足し、**マスクと同じ最近傍補間**を掛ける。

```python
def __call__(self, dem, air, mask, weight=None):
    ...
    if self.hflip and random.random() < 0.5:
        dem, air, mask = TF.hflip(dem), TF.hflip(air), TF.hflip(mask)
        if weight is not None:
            weight = TF.hflip(weight)
    # vflip も同様
    ...
    mask = TF.affine(mask, ..., interpolation=TF.InterpolationMode.NEAREST)
    if weight is not None:
        weight = TF.affine(weight, angle=angle, translate=translations,
                           scale=scale, shear=0,
                           interpolation=TF.InterpolationMode.NEAREST)
        return dem, air, mask, weight
    return dem, air, mask
```

重みマップはマスクと同じ「位置に意味がある」データなので、双線形だと箇所の
境界が濁る。必ず `NEAREST`。

### ④ セル12（DualInputDataset）で重みを返す

```python
class DualInputDataset(Dataset):
    def __init__(self, dem_images, air_images, masks, weights=None, transform=None):
        self.dem_images, self.air_images = dem_images, air_images
        self.masks, self.weights, self.transform = masks, weights, transform

    def __getitem__(self, idx):
        dem = self.dem_images[idx].float()
        air = self.air_images[idx].float() / 255.0
        mask = self.masks[idx].float()

        if self.weights is None:
            if self.transform:
                dem, air, mask = self.transform(dem, air, mask)
            return (dem, air), mask

        w = self.weights[idx].float()
        if self.transform:
            dem, air, mask, w = self.transform(dem, air, mask, w)
        return (dem, air), mask, w
```

**`weights` を渡したときだけ3つ返す**のが要点。こうすると
検証・テスト・島根の評価ループは従来どおり2つ受けのままでよく、
セル23・24・25 を触らずに済む。

### ⑤ セル13（データセット作成）で学習側にだけ重みを渡す

```python
train_dataset = DualInputDataset(dem_train, air_train, y_train,
                                 w_train if NEEDS_WEIGHT else None,
                                 transform=joint_transform if get(CONDITION).augment else None)
val_dataset  = DualInputDataset(dem_val,  air_val,  y_val)
test_dataset = DualInputDataset(dem_test, air_test, y_test)
```

### ⑥ セル20（学習ループ）で重みを損失に渡す

```python
for batch in train_loader_tqdm:
    if NEEDS_WEIGHT:
        (dem, air), masks, weights = batch
        weights = weights.to(device)
    else:
        (dem, air), masks = batch
        weights = None
    dem, air, masks = dem.to(device), air.to(device), masks.to(device)

    optimizer.zero_grad()
    outputs = model(dem, air)
    loss = criterion(outputs, masks, weights) if NEEDS_WEIGHT else criterion(outputs, masks)
```

検証ループ（同じセル内の `val_loader`）は**触らない**。重みは学習にだけ使う。

---

## ステップ5　配線できたか確かめてから学習する

ノートブックを上から実行し、次の3つを確認する。

| セル | 出るはずの内容 |
|---|---|
| 0 | `重み: .../dc5-data/weights/FinalFusion_SAM_APM_InstLoss/best_model.pth` |
| 10 | `損失: instance_weighted_focal_tversky(...)` と `重みマップの設定: {'a0': 300, 'p': 0.5, ...}` |
| 13 | 学習バッチが **3つ組**、検証バッチが **2つ組** |

セル13のあとに、この確認を1行入れておくとよい。

```python
b = next(iter(train_loader)); print("train のバッチ:", len(b), "個")
b = next(iter(val_loader));   print("val のバッチ  :", len(b), "個")
```

`train のバッチ: 3 個` `val のバッチ: 2 個` なら配線できている。

**動作確認は `num_epochs` を一時的に 2 にして通す。** 確認できたら戻す。
Mac(MPS) だと1エポック数分かかるので、本番は研究室PC(CUDA)で。

---

## ステップ6　評価して記録する

```bash
./.venv/bin/python analysis/run_inference.py     --region all  --conditions FinalFusion_SAM_APM_InstLoss
./.venv/bin/python analysis/run_instance_eval.py --region both --conditions FinalFusion_SAM_APM_InstLoss
./.venv/bin/python analysis/run_analysis.py
./.venv/bin/python tools/make_metrics.py

./.venv/bin/python -m dc5lib.sync verify
./.venv/bin/python tools/check.py
git pull && git add -A
git commit -m "FinalFusion_SAM_APM_InstLoss を学習・評価"
git push
```

### main に入れる

```bash
git checkout main && git pull
git checkout lab-windows && git merge main    # 相手の変更を先に取り込む
./.venv/bin/python tools/check.py
git push
```

そのうえで GitHub で PR を作って main にマージする。
`results/metrics.csv` が衝突したら、手で直さず
`./.venv/bin/python tools/make_metrics.py` で作り直してから PR を更新する。

### 何と比べるか

`FinalFusion_SAM_APM` と並べる。**損失以外が完全に同じ**なので、差は損失の効果。

```bash
./.venv/bin/python -c "
import pandas as pd
d = pd.read_csv('results/metrics.csv')
d = d[d.condition.isin(['FinalFusion_SAM_APM','FinalFusion_SAM_APM_InstLoss'])]
d = d[(d.region=='hiroshima') & (d.scope=='通常')]
print(d.pivot_table(index=['metric_kind','setting'], columns='condition', values='f1').to_string())
"
```

**面積評価と箇所数評価の両方を見ること。** 箇所数評価だけ見ると、
予測を広げる方向に判断を誤る。狙いどおりなら、

- 箇所F値が上がる（小さい箇所が拾えるようになる）
- 面積F値はあまり落ちない
- 予測面積/正解面積が 1.4〜1.6 の範囲に収まっている（`instance_diagnostics.csv`）

面積比が 1.9 を超えていたら塗り広げすぎ。`--beta` を上げて作り直す。

---

## つまずきやすいところ

| 症状 | 原因 |
|---|---|
| `この損失は重みマップが必要です` | ④⑤⑥ のどれかが抜けている |
| `too many values to unpack` | ⑤で重みを渡したのに⑥を直していない（またはその逆） |
| 検証ループで落ちる | 検証側にも重みを渡してしまっている。検証は2つ受けのまま |
| 箇所の境界がぼやける | ③で `BILINEAR` を使っている。`NEAREST` にする |
| メモリが足りない | `precompute_weights` の `dtype` は float16 のまま使う |
| テスト集合が他と違う | ②で `test_size` か `random_state` を変えてしまった |

---

## 参考：実装済みの例

単一入力版だが、同じ配線が入ったものが
`experiments/UNet_SAM_InstLoss/` にある（セル7・8・9・10・14）。
書き方に迷ったらそちらを見るとよい。
