# -*- coding: utf-8 -*-
"""
実験条件（入力データ・モデル重みの組み合わせ）の一覧。

新しい実験を追加するときは、このファイルに1エントリ追加するだけで
run_analysis.py からそのまま解析できるようにするための設定レジストリ。

各エントリの意味:
  kind        : "single"（地形量のみ）or "dual"（地形量＋航空写真）
  pkl_path    : 学習に使ったデータセットのpickleファイルパス
  ckpt_path   : best_model.pth のパス
  normalize   : 地形量チャネルの正規化方法。lib.data_utils.normalize_terrain の mode。
                新しい実験では、まず "raw" を仮置きし、
                lib/verify_checkpoint.py で best_f1_score と照合して確定させること。
  terrain_label: 比較画像に出す地形量パネルのタイトル（"DEM" or "SAM(slope)" など）
  out_dir     : 結果の出力先フォルダ名（FSS_analysis/ 以下）
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # FSS_analysis/

CONDITIONS = {
    "DemOnly": dict(
        kind="single",
        pkl_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\make_dataset\dem_divi_masuya.pkl",
        ckpt_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_NoOgu_DemOnly\best_model.pth",
        normalize="raw",
        terrain_label="DEM",
        out_dir=os.path.join(BASE_DIR, "DemOnly"),
        pptx_metrics=dict(recall=0.5918, precision=0.5514, f1=0.5709),
    ),
    "KeisyaOnly": dict(
        kind="single",
        pkl_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\mayo_dataset\processed_nkeisya_dataset_ver2.pkl",
        ckpt_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_NoOgu_KeisyaOnly\best_model.pth",
        normalize="global_minmax",
        terrain_label="SAM(slope)",
        out_dir=os.path.join(BASE_DIR, "KeisyaOnly"),
        pptx_metrics=dict(recall=0.6744, precision=0.5430, f1=0.6016),
    ),
    "DEM_AirPhoto": dict(
        kind="dual",
        pkl_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\DEM_photo_dataset2.pkl",
        ckpt_path=r"C:\Users\hirok\OneDrive\Desktop\CrZ\model\FSS_DEMAndAirPhoto_0per_gausian\best_model.pth",
        normalize="raw",
        terrain_label="DEM",
        out_dir=os.path.join(BASE_DIR, "DEM_AirPhoto"),
        pptx_metrics=dict(recall=0.6968, precision=0.6687, f1=0.6825),
    ),
    "SAM_AirPhoto": dict(
        kind="dual",
        pkl_path=r"D:\CrZ\FSS\data\photo_dataset.pkl",
        ckpt_path=r"D:\FSS_KeisyaAndAP\best_model.pth",
        normalize="raw",
        terrain_label="SAM(slope)",
        out_dir=os.path.join(BASE_DIR, "SAM_AirPhoto"),
        pptx_metrics=dict(recall=0.7500, precision=0.6413, f1=0.6914),
    ),
}
