# DetectHazard — 急傾斜地警戒区域の自動抽出

深層学習で急傾斜地の（特別）警戒区域を画素単位に抽出する。
広島県のデータで学習し、島根県で汎化性能を検証している。

このリポジトリは **3台のマシンで共有する**。

| マシン | OS | 役割 |
|---|---|---|
| 研究室PC 1 | Ubuntu | 学習 |
| 研究室PC 2 | Windows | 学習 |
| 自宅 | macOS | 解析・資料作成 |

---

## 1. 何をどこで管理しているか

**大きさと寿命で3つに分けている。** ここを混ぜると GitHub に上げられなくなる。

| 種類 | 置き場所 | 理由 |
|---|---|---|
| コード・設定・**精度** | このリポジトリ（GitHub） | 小さい。履歴が要る |
| **重み・チェックポイント・データセット** | 外部SSD `dc5-data/` | 大きい（65GB）。履歴は要らない |
| 推論結果などの中間物 | 外部SSD `dc5-data/cache/` | 再生成できる |

**重みをこのリポジトリの中に置かないこと。** `.gitignore` で弾いているが、
`python -m dc5lib.registry --check` でも検査している。

---

## 2. 最初にやること（新しいPCで）

```bash
git clone https://github.com/SomaDC5/DetectHazard.git
cd DetectHazard
conda env create -f environment.yml && conda activate dc5
# PyTorch は CUDA の有無で入れ分ける（environment.yml のコメント参照）

python -m dc5lib.paths        # SSDが見つかるか確認
python -m dc5lib.registry     # 条件の一覧と重みの有無
```

**外部SSDを挿すだけで動く。** パスの設定は要らない。

`dc5lib.paths` が SSD 内の目印 `dc5-data/.dc5-root.json` を探す。
Windows のドライブレター、Ubuntu の `/media/$USER/...`、macOS の `/Volumes/...` を
自動で走査するので、OS ごとの書き分けが不要。

見つからないときだけ環境変数で指定する。

```bash
export DC5_DATA=/media/$USER/SSD-PHPU3A/dc5-data     # Ubuntu
set DC5_DATA=D:\dc5-data                              # Windows
```

SSDは2台ある。両方挿しているときは `DC5_SSD_ID=A` のように指定する。

> **リポジトリは各PCのローカルディスクに clone すること。**
> 外部SSD（exFAT）上の clone はシンボリックリンクとパーミッションが扱えず、
> Ubuntu と Windows で差分が出る。SSDはデータ専用にする。

---

## 3. フォルダ構成

```
DetectHazard/
├── dc5lib/                 全PC共通のコード
│   ├── paths.py            ★ SSDの場所を自動検出。場所を知っているのはここだけ
│   ├── registry.py         experiments/*/config.yaml から条件一覧を組み立てる
│   ├── models.py           モデル定義（6クラス）
│   ├── results.py          精度の記録と metrics.csv の生成
│   ├── sync.py             2台のSSDの突き合わせ
│   ├── instance_eval.py    箇所数評価（被覆方式）
│   └── instance_loss.py    箇所正規化した損失
├── experiments/<条件名>/
│   ├── config.yaml         ★ 条件の定義。フォルダを足せば条件が増える
│   └── *.ipynb             学習・評価ノートブック
├── analysis/               タイル単位の再解析ツール
├── results/                ★ 精度の一括管理
├── docs/                   設計メモ・発表資料・引き継ぎ
├── workspace/              Claude Code の作業場所（マシンごとに分けてある）
└── tools/                  移行・集計スクリプト
```

---

## 4. 実験を1つ増やすには

**フォルダを1つ作るだけ。** 中央の一覧を編集する必要はない。

```bash
mkdir experiments/MyNewModel_SAM_APM
cp experiments/FinalFusion_SAM_APM/config.yaml experiments/MyNewModel_SAM_APM/
# config.yaml の name と arch を直す。ノートブックを置く。
python -m dc5lib.registry --check
```

新しいモデルクラスを使うなら `dc5lib/models.py` に追加して `MODEL_CLASSES` に登録する。

> **フォルダ名を大文字小文字だけ変えて作らないこと。**
> Ubuntu の ext4 は区別するが、Windows と exFAT は区別しない。
> `MiddleFusion_DEM_APM` と `MIddleFusion_DEM_APM` が共存すると、
> OS をまたいだ瞬間に壊れる。`registry --check` がこれを検査している。
> 既存の `MIddleFusion_DEM_APM`（Iが大文字）と `DataOgument`（正しくは Augment）は、
> 過去のログとの突き合わせのためあえて直していない。

ノートブックから重みを保存・読み込みするときは、パスを直接書かない。

```python
from dc5lib.paths import weights_path, checkpoint_path, dataset_path
torch.save(state, weights_path("MyNewModel_SAM_APM"))
ds = load_dataset(dataset_path("Hiroshima", "hiroshima_sam_apm.pkl"))
```

---

## 5. 精度の管理

**2台のPCで同時に回すので、単一のCSVを両方から編集すると必ずマージ衝突する。**
実体は1評価1ファイルにして、`metrics.csv` はその生成物にしてある。

```
results/
├── runs/<run_id>.json        1学習。run_id = <条件>__<機械名>__<日時>
├── evals/<eval_id>.json      1評価
├── curves/<条件>.csv         学習曲線（旧 losses.csv）
├── ssd_manifests/<id>.csv    どのSSDにどの重みがあるか
└── metrics.csv               ★ 一括管理。これを見ればよい
```

ファイル名に機械名と日時が入るので衝突しない。

```bash
python tools/make_metrics.py      # metrics.csv を作り直す
```

`metrics.csv` が衝突したら、**手で直さずに作り直す**。決定的な順序で書くので
どのPCで作っても同じ内容になる。

`metrics.csv` の1行 = 1評価。

| 列 | 値 |
|---|---|
| `region` | hiroshima / hiroshima_bg / shimane |
| `tileset` | 警戒のみ / 背景込み / 全件 / 背景のみ |
| `scope` | 通常 / 境界（中心96×96） |
| `metric_kind` | 面積 / 箇所 |
| `setting` | 箇所数評価の対応づけ方式 |

**面積評価と箇所数評価は必ず並記する。** 片方だけ見ると、予測を広げる方向か
細切れにする方向に判断を誤る。理由は `docs/箇所数評価_設計メモ.md`。

---

## 6. SSD 2台の管理

```bash
python -m dc5lib.sync verify          # 手元のSSDを検証し、在庫表を更新
python -m dc5lib.sync status          # git に載っている全SSDの在庫を表示（SSD不要）
python -m dc5lib.sync diff  <他方>    # 両方挿して突き合わせ
python -m dc5lib.sync pull  <他方> --run   # 足りないものだけコピー
```

`verify` は在庫表を `results/ssd_manifests/<ssd_id>.csv` にも書き出して git に載せる。
おかげで **SSDを挿さなくても「どの重みがどちらのSSDにあるか」が分かる。**

2台目のSSDを用意するときは、`dc5-data/.dc5-root.json` の `ssd_id` を `"B"` にする。

---

## 7. 家と研究室で作業を共有する

`workspace/` をマシンごとに分けてある。同じファイルを両方から触らなければ衝突しない。

```
workspace/home/          自宅（macOS）
workspace/lab-ubuntu/    研究室PC 1
workspace/lab-windows/   研究室PC 2
```

まとまった成果は `docs/` か `analysis/` に移す。`workspace/` は下書き置き場。

Claude Code を研究室PCで動かすときは、このリポジトリを clone した場所で起動すれば
`README.md` と `docs/` を読んで文脈を把握できる。
