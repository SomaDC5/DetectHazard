# データセット構築ツール

GeoTIFF（数値標高モデル）と警戒区域のシェープファイルから、
学習に使う `.pkl`（タイル化済みデータセット）を作る一式。

もともと外付けSSD `Soma_SSD` の `simane/MakeDataSet/` にだけ置いてあり、
**版管理されておらず、片方のSSDにしか無い状態だった**ので、ここに取り込んだ。
これが失われるとデータセットを作り直せない。

実データ（GeoTIFF・航空写真・シェープファイル）はSSDに残してある。
ここに入っているのはスクリプトだけ。

---

## ファイル

| ファイル | 役割 |
|---|---|
| `build_shimane_dataset_v6.py` | **最新版。** タイル化の本体。地質図（3チャネル目）に対応 |
| `build_shimane_dataset_v5.py` | 航空写真（APM）まで対応した版。いまのデータセットの多くはこれで作られた |
| `build_shimane_dataset_v4.py` | 傾斜量図（SAM）モードを追加した版 |
| `build_shimane_dataset_v3.py` | DEMのみの初期版 |
| `read_geotiff.py` | GeoTIFF の読み込み。build_* から呼ばれる |
| `gml2geotiff.py` | 基盤地図情報の GML を GeoTIFF に変換する |
| `fetch_aerial_v2.py` | 国土地理院のタイルから航空写真を取得 |
| `fetch_geology.py` | 産総研シームレス地質図からタイルを取得 |
| `inspect_shapefile.py` | シェープファイルの属性を確認する |
| `diagnose_tiff.py` | GeoTIFF の欠損値などを点検する |
| `visualize_tiles.py` / `visualize_tiles2.py` | 生成したタイルを目視確認する |
| `データセット構築方法.txt` | 元の手順書（取得元URLを含む。原文のまま） |

`build_*` は名前が「shimane」だが、**広島でも同じスクリプトを使っている**。
地域ごとに分かれているわけではない。

---

## 使い方

`データセット構築方法.txt` が原典。要点だけ再掲する。

### 1. 標高データを用意する

国土地理院の基盤地図情報から 5mメッシュ航空レーザー測量データを取得し
（https://service.gsi.go.jp/kiban/ ）、zip を全部展開して1フォルダにまとめる。
基盤地図情報標高DEM変換ツール、または `gml2geotiff.py` で `.tif` に変換する。

### 2. 正解データを用意する

国土数値情報の土砂災害警戒区域データ（A33）。
https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-A33-v2_0.html

`--field A33_001 --values 1` が急傾斜地の崩壊にあたる。
属性の確認は `python inspect_shapefile.py <shp>`。

### 3. タイル化する

**SAM + マスク（2チャネル目なし）**

```bash
python build_shimane_dataset_v4.py "<tifフォルダ>" "<shpのパス>" shimane_sam.pkl \
    --mode sam --field A33_001 --values 1
```

`--mode sam` を `--mode dem` にすると DEM 版になる。

**SAM + 航空写真 + マスク（いまの主力）**

```bash
python build_shimane_dataset_v5.py "<tifフォルダ>" "<shpのパス>" shimane_sam_apm.pkl \
    --mode sam --field A33_001 --values 1 --apm
```

**SAM + 航空写真 + 地質図 + マスク（3チャネル）**

```bash
python build_shimane_dataset_v6.py "<tifフォルダ>" "<shpのパス>" shimane_sam_apm_geo.pkl \
    --mode sam --field A33_001 --values 1 --apm --geology
```

`--geology-zoom`（既定13）と `--geology-cache-dir` で地質図タイルの取得を調整できる。

### 4. DEM を使う場合は正規化する

```bash
python normalize_dem.py apply shimane_dem_apm_geo.pkl \
    shimane_dem_apm_geo_normalized.pkl --scale 500
```

`config.yaml` で DEM 系の条件が `*_normalized.pkl` を指しているのはこのため。
SAM は角度（度）なので正規化しない。

### 5. 置き場所

できた `.pkl` は外付けSSDの `dc5-data/datasets/<地域>/` に置く。
そのあと `regions.yaml` と各 `config.yaml` の `dataset:` に追記する
（手順は `docs/使い方.md` の「地域を1つ追加する」）。

---

## 3チャネル入力について

**データ側はすでに作れる。** `build_shimane_dataset_v6.py --geology` で
地質図を含む pkl を生成でき、広島・島根とも
`*_sam_apm_geo_final_v2.pkl` が両SSDにある。

足りないのはモデル側で、次の3か所に手を入れる必要がある。

- `dc5lib/models.py` に3エンコーダのクラスを追加して `MODEL_CLASSES` に登録
- `analysis/data.py` の `build_tileset` を `cond.inputs` 駆動にする
- `analysis/infer.py` の `run_condition` を `model(*tensors)` にする

`experiments/Triple_UNet_SAM_APM_GEO/` にノートブックとモデル定義があるので、
そこから移植するのが最短。

---

## 動かすための環境

GDAL / rasterio / geopandas / shapely が必要で、`environment.yml` には含めていない
（学習・解析だけなら不要なため）。データセットを作るときだけ別に用意する。

```bash
conda create -n dc5-build python=3.11 gdal rasterio geopandas shapely pillow requests tqdm -c conda-forge
```
