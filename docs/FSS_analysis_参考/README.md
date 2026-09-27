# FSS_analysis 解析プログラム利用ガイド

急傾斜地崩壊危険区域抽出モデル（FSS2026_V5.pptx で発表）の、テストデータに
おける見逃し（FN）・過検出（FP）を地形的・地表面的特徴の観点から分析する
ためのプログラム一式。今後の研究で新しい実験条件を追加しても、同じ枠組みで
そのまま解析できるように整理してある。

## 1. できること

- 保存済みモデル（`best_model.pth`）に対して、学習時と同一のテスト分割で
  推論を再実行し、Recall/Precision/F1 がスライド・ノートブック記載の値と
  一致することを検証する
- テストタイル1枚ごとに TP/FP/FN 画素数・地形量（標高・傾斜）・航空写真の
  明度／緑色度と、Recall/Precision/F1 をCSVにまとめる
- 各タイルを「良好」「見逃し優勢」「過検出優勢」「その他」に自動分類する
- 見逃し・過検出が大きいタイル、良好なタイルの比較画像（入力／GT／予測／
  誤差マップ）を自動出力する
- 複数条件を横断して、FN/FPの相関や「どの入力でも解消しない問題タイル」を
  自動的に洗い出す

## 2. 環境

- Anaconda の `mayo` 環境を使用する（PyTorch 2.7 + CUDA、OpenCV、scikit-learn 等
  が入っている）。他の環境（base 等）には torch が入っていないため動かない。

```bash
"C:\Users\hirok\anaconda3\envs\mayo\python.exe" run_analysis.py
```

PowerShellで環境をアクティベートして使う場合は次の通り。

```powershell
conda activate mayo
cd C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_analysis
python run_analysis.py
```

CPU推論を既定にしている（`run_analysis.py` 内 `DEVICE = "cpu"`）。理由は
5章「つまずいたポイント」を参照。GPUを使いたい場合は `DEVICE = "cuda"` に
変更できるが、システムメモリが逼迫している状況ではCPUの方が安定する。

## 3. ディレクトリ構成

```
FSS_analysis/
├── lib/
│   ├── models.py            U-Net / MultiEncoderUNet の定義
│   ├── data_utils.py        pickle読み込み・分割・正規化の共通処理
│   ├── inference.py         バッチ推論（.contiguous()対策込み）
│   ├── error_analysis.py    TP/FP/FN集計・分類・比較画像の出力
│   ├── verify_checkpoint.py 正規化方法の自動検証ツール
│   └── conditions.py        実験条件（データセット・モデルの組み合わせ）の一覧
├── run_analysis.py          メインスクリプト（条件を指定して解析）
├── compare_conditions.py    複数条件の横断比較ツール
├── DemOnly/ KeisyaOnly/ DEM_AirPhoto/ SAM_AirPhoto/
│                           条件ごとの出力（CSV・JSON・比較画像）
├── compare_output/           compare_conditions.py の出力
└── legacy_scripts/           リファクタリング前の個別スクリプト（参考保存用）
```

## 4. 使い方

### 4.1 既存条件を再解析する

```bash
python run_analysis.py                      # 全条件
python run_analysis.py DemOnly              # 1条件だけ
python run_analysis.py DemOnly SAM_AirPhoto # 複数条件
```

実行すると各条件フォルダに以下が出力される。

| ファイル | 内容 |
|---|---|
| `error_stats.csv` | タイル毎（No, TP/FP/FN画素数, Recall/Precision/F1, 地形量・写真特徴量の平均, bucket区分） |
| `summary.json` | 条件全体の集計値（本レポートの数値の元データ） |
| `worst_FN/` | FN画素数が多い上位20タイルの比較画像 |
| `worst_FP/` | FP画素数が多い上位20タイルの比較画像 |
| `good/` | F1値が高い上位15タイルの比較画像 |

`error_stats.csv` の `bucket` 列（`good`/`high_FN`/`high_FP`/`mixed`）で
全タイルが分類済みなので、Excel等でこの列によりフィルタすれば、
条件ごとの「見逃しが多い場所一覧」「過検出が多い場所一覧」がすぐに得られる。

### 4.2 複数条件を横断比較する

```bash
python compare_conditions.py DemOnly KeisyaOnly
python compare_conditions.py --all
```

FN/FP画素数の条件間相関、"問題あり"タイルの残存率を表示し、
すべての指定条件で共通して問題が残る／うまくいくタイルの一覧を
`compare_output/` 以下に保存する。

| ファイル | 内容 |
|---|---|
| `chronic_high_FN.csv` | 指定した全条件で見逃し優勢(high_FN)のままのタイル |
| `chronic_high_FP.csv` | 指定した全条件で過検出優勢(high_FP)のままのタイル |
| `chronic_good.csv` | 指定した全条件で良好(good)なタイル（F1平均値で降順ソート済み） |

`chronic_high_FN` / `chronic_high_FP` は、モデル改良で解決しない可能性が高い
「正解データ側の妥当性を確認すべき地点」の候補として使える。
`chronic_good` は逆に「入力データの種類によらず常にうまくいく地点」であり、
共通する成功パターン（本レポート5.1節では、地形量・航空写真のどちらでも
同じ位置に明瞭な境界が現れる河岸段丘状の急斜面、という特徴が確認された）を
分析するのに使える。4条件全部で実行した場合、209件（全体の8.8%）が
`chronic_good` に該当した。

### 4.3 新しい実験条件を追加する

1. `lib/conditions.py` の `CONDITIONS` 辞書に1エントリ追加する。
   - `kind`: 地形量のみなら `"single"`、地形量＋航空写真なら `"dual"`
   - `pkl_path` / `ckpt_path`: データセットとモデル重みのパス
   - `normalize`: いったん `"raw"` を仮置きする
   - `terrain_label`, `out_dir`, `pptx_metrics`（分かっていれば）
2. `lib/verify_checkpoint.py` の `try_normalizations()` で、
   `best_model.pth` に記録された `best_f1_score` と一致する正規化方法を確認し、
   `normalize` の値を確定させる（詳細は次章）。
3. `python run_analysis.py <新条件名>` を実行する。

## 5. 前処理まとめ（正規化は条件ごとに異なる・要注意）

同じような前処理コードが複数の実験ノートブックにコピーされ、
一部だけ書き換えられていたため、**ノートブックのコードを読むだけでは
実際に学習で使われた正規化方法が分からない**ことが今回の解析で判明した。
そこで、各チェックポイントの `best_f1_score`（学習時に記録された最良F1値）
と実際に一致するまで正規化方法を変えて推論し直す、という方法で確定させた。
確認済みの組み合わせは以下の通り。

| 条件 | 地形量チャネルの正規化 | 航空写真チャネル |
|---|---|---|
| DEM単一 | 正規化なし（生の標高値, m） | - |
| SAM単一 | 学習集合全体でのmin-max正規化 | - |
| DEM＋APM | 正規化なし（生の標高値, m） | 0〜1（÷255） |
| SAM＋APM | 正規化なし（生の傾斜値, 度） | 0〜1（÷255） |

新しい条件を追加する際は、**必ず `verify_checkpoint.try_normalizations()` で
再確認すること**。目安として、正規化が間違っている場合はRecallが
0.001未満などの明らかに異常な値になる（BatchNormの統計量が学習時の
入力スケールとズレるため、モデルがほぼ何も予測しなくなる）。

```python
from lib.conditions import CONDITIONS
from lib.models import UNet, load_model
from lib.verify_checkpoint import try_normalizations
from lib import data_utils as du

cfg = CONDITIONS["DemOnly"]
raw = du.load_pickle(cfg["pkl_path"])
with_mask, _ = du.split_with_mask(raw, has_airphoto=False)
idx_train, idx_test = du.train_test_indices(len(with_mask.No))
dem_test, mask_test, _, no_test = du.stack_test_subset(with_mask, idx_test, has_airphoto=False)
import numpy as np
max_h = np.array(with_mask.Max_H, dtype=np.float32)
min_h = np.array(with_mask.Min_H, dtype=np.float32)

model, best_f1, _ = load_model(UNet(), cfg["ckpt_path"], "cpu")
try_normalizations(
    model, dem_test, mask_test,
    max_h_train=max_h[idx_train], min_h_train=min_h[idx_train],
    max_h_test=max_h[idx_test], min_h_test=min_h[idx_test],
    target_f1=best_f1,
)
```

## 6. つまずいたポイントと対処法（今後同じ問題に当たった時のために）

- **`AttributeError: Can't get attribute 'dem_dataset' on <module '__main__'>`**
  データセットのpickleは `__main__.dem_dataset` / `__main__.photo_dataset` という
  クラス名で保存されている。`lib/data_utils.load_pickle()` が自動的に
  ダミークラスを登録するので、通常は意識しなくてよい。
- **Windows上のCPU版PyTorchで `Segmentation Fault` / `Windows fatal exception: access violation`**
  `np.transpose` やスライス直後の非連続（non-contiguous）テンソルを
  そのまま `Conv2d` に渡すと、ネイティブ層で異常終了することがある
  （DEM＋APM条件の再解析時に発見）。`lib/inference.py` はモデルに渡す
  直前に必ず `.contiguous()` を呼ぶ実装にしてあるので、新しい推論コードを
  書くときも同様にすること。
- **`numpy._core._exceptions._ArrayMemoryError`（メモリ不足）**
  このPCはOneDriveの同期プロセスがメモリを大量消費しており、空きメモリが
  数GBまで落ち込むことがある。対策として、①学習用データ（train側）は
  推論に不要なので変数を作らずに捨てる、②推論結果はバッチごとにnumpy化して
  リストに貯め、最後に `np.concatenate` する（テンソルのまま貯めない）、
  という実装にしてある。それでも失敗する場合はOneDriveの同期を一時停止するか、
  タスクマネージャで空きメモリを確認してから再実行する。
- **OneDriveの「オンデマンドファイル」（クラウドのみ）状態のファイル**
  `Get-Item` の `Attributes` に `ReparsePoint` が含まれる場合、実体が
  まだローカルにダウンロードされていない可能性がある。読み込み前に
  一度ファイル全体を読んで実体化（ハイドレート）しておくと安全。
- **正規化方法が条件ごとに違う**（5章参照）。

## 7. さらに踏み込んだ分析に向けた今後の展望

現状の `run_analysis.py` は「タイル単位のピクセル評価＋簡易な地形・写真特徴量」
までをカバーしている。次のステップとして、以下のような拡張が考えられる。

### 7.1 評価指標の拡張

- **箇所数ベースの評価をパイプラインに統合する**
  元のノートブックには、連結成分（危険区域のかたまり）単位でTP/FP/FNを
  数える `calc_eva` 関数があった（ピクセル単位のRecall/Precisionとは
  見え方が変わる）。`lib/error_analysis.py` に
  `connected_component_metrics()` を追加し、`error_stats.csv` に
  「タイル内の危険区域を何個中何個検出できたか」を出力できるようにすると、
  「大きな区域の一部だけ検出できた」ケースと「区域を丸ごと見逃した」ケースを
  区別できるようになる。
- **ボーダー評価の自動化**
  スライドの評価方法②（周囲16px除去）を `run_analysis.py` に
  オプション（`--crop 16`）として組み込み、画像端部を除いた場合の
  指標変化を全条件・全タイルで一括算出できるようにする。
- **オーバーラップ推論の効果測定**
  タイルを重ね合わせて推論し平均化する（パディング改善策として
  考察スライドで挙げた案）手法を実装し、境界部の精度がどれだけ
  改善するかを現状のパイプラインと同じ枠組みで定量比較する。

### 7.2 入力データの拡張

- `lib/conditions.py` の `kind` を `"single"`/`"dual"` の2種類から
  `n_channels` を指定できる汎用形式に変えれば、地質図・土地利用図・
  保全対象距離などを3つ目・4つ目のチャネル（あるいは3つ目のエンコーダ枝）
  として追加した新モデルもそのまま同じ枠組みで解析できるようになる。
- 森林被覆下での見逃しが4条件共通で確認されたことを踏まえ、
  NDVI相当の植生指標や土地被覆分類結果を追加チャネルとして
  組み込んだ実験を優先的に行い、`compare_conditions.py` で
  「森林被覆による見逃しが本当に解消したか」を chronic タイルのNoベースで
  ピンポイントに検証する。

### 7.3 モデル・学習の改良と分析の対応付け

- Attention U-Net 等、受容野やスキップ接続を変えたモデルを追加する場合も、
  `lib/models.py` に新しいクラスを足し、`lib/conditions.py` の
  エントリに `model_class` を指定できるようにしておくと、
  既存の評価パイプラインをそのまま流用できる。
- Grad-CAM等の説明可能AI手法を導入する際は、`lib/inference.py` の
  推論ループにフックを追加し、`worst_FN`/`worst_FP` として選ばれた
  タイルに対してだけ可視化を生成すると、「なぜ見逃したか／なぜ過検出したか」の
  仮説（本レポートで挙げた「濃い緑＝低リスクと誤学習している可能性」等）を
  直接検証できる。

### 7.4 運用・再現性まわり

- `run_analysis.py` の出力（`summary.json`）から、条件比較表・判定内訳グラフ・
  代表画像を自動でWord/PowerPointに差し込むレポート生成スクリプトを作れば、
  新しい条件を追加するたびに本レポートのようなドキュメントを手作業なしで
  更新できる。
- 他県データへの適用（今後の展望としてスライドに挙げられている汎化性能検証）を
  行う際は、`lib/conditions.py` に県ごとのエントリを追加するだけで
  同一の解析フローに乗せられる設計になっている。

## 8. 参考：本レポート作成時に使ったコマンド例

```bash
# 4条件すべてを解析
python run_analysis.py

# 単一入力どうし・APMありどうしをそれぞれ比較
python compare_conditions.py DemOnly KeisyaOnly
python compare_conditions.py DEM_AirPhoto SAM_AirPhoto
python compare_conditions.py --all
```
