# -*- coding: utf-8 -*-
"""analysis パッケージ用の設定。中身は dc5lib から引く。

以前はここに14条件のリストを手で書いていたが、モデルが増えるたびに
2か所（experiments/ と ここ）を直す必要があり、片方を忘れると壊れる。
いまは `experiments/*/config.yaml` が唯一の定義で、ここはその読み替えだけをする。

地域も `regions.yaml` から引くので、県を増やしてもこのファイルは触らなくてよい。
"""

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dc5lib import paths as _paths       # noqa: E402
from dc5lib import regions as _regions   # noqa: E402
from dc5lib import registry as _registry  # noqa: E402

# ------------------------------------------------------------------ パス
PROJECT_DIR = str(_REPO)
DATASET_DIR = str(_paths.data_root() / "datasets")
DETECT_HAZARD_DIR = str(_paths.experiments_dir())
MODELCOMPARISON_DIR = os.environ.get(
    "MODELCOMPARISON_DIR", str(Path(DATASET_DIR).parent.parent / "dc5" / "ModelComparison"))

# 解析の中間生成物は再生成できるので、リポジトリではなく SSD の cache に置く。
# （タイル単位のCSVだけで数GBになるため）
OUTPUT_DIR = str(_paths.cache_dir("analysis"))
TILES_DIR = os.path.join(OUTPUT_DIR, "tiles")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")

# ------------------------------------------------------------------ 地域
REGIONS = _regions.keys()
REGION_DIR = {r.key: r.dir for r in _regions.all_regions()}
REGION_JA = {r.key: r.label for r in _regions.all_regions()}
REGION_SUBSET = {r.key: r.subset for r in _regions.all_regions()}

# ------------------------------------------------------------------ 前処理・評価の定数
TILE_SIZE = 128
BORDER_CROP = 16          # 境界評価で四辺から落とす画素数（中心96x96）
GAUSSIAN_SIGMA = 1.0      # 学習時にマスクへ掛けたガウシアンフィルタ
THRESHOLD = 0.5           # sigmoid 出力の二値化しきい値
TEST_SIZE = 0.2           # 広島のホールドアウト
RANDOM_STATE = 42


class Condition:
    """dc5lib.registry.Condition を analysis 側の呼び名に合わせる薄い包み。"""

    def __init__(self, inner):
        self._c = inner

    def __getattr__(self, k):
        return getattr(self._c, k)

    # --- 呼び名の違いを吸収 ---
    @property
    def model_class(self):
        return self._c.arch

    @property
    def augmented(self):
        return self._c.augment

    @property
    def label(self):
        return self._c.display()

    @property
    def dual(self):
        return self._c.n_inputs >= 2

    @property
    def input_mode(self):
        return "デュアル入力" if self._c.n_inputs >= 2 else "単一入力"

    def pkl(self, region):
        """region は 'hiroshima' などの小文字キー。"""
        r = _regions.get(region)
        name = self._c.dataset.get(r.dir)
        if name is None:
            raise KeyError(f"{self._c.name} に {r.dir} のデータセットが定義されていません")
        return str(_paths.dataset_path(r.dir, name))

    @property
    def weights(self):
        return str(self._c.weights)


def _load():
    return [Condition(c) for c in _registry.load_all()]


CONDITIONS = _load()
BY_NAME = {c.name: c for c in CONDITIONS}
EXCLUDED = {c.name: c.note for c in _registry.load_all(include_excluded=True)
            if c.status == "excluded"}


def group_by_pkl(conditions, region):
    """同じpklを使う条件をまとめる（大きなpklを何度も読み直さないため）。"""
    groups = {}
    for c in conditions:
        groups.setdefault(c.pkl(region), []).append(c)
    return groups


def ensure_dirs():
    for d in (OUTPUT_DIR, TILES_DIR, FEATURES_DIR, FIGURES_DIR, REPORT_DIR):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    print(f"repo     : {PROJECT_DIR}")
    print(f"datasets : {DATASET_DIR}")
    print(f"出力     : {OUTPUT_DIR}")
    print(f"地域     : {REGIONS}")
    print()
    print(f"{'条件':<38}{'クラス':<26}{'入力':<8}{'重み'}")
    for c in CONDITIONS:
        print(f"{c.name:<38}{c.model_class:<26}{c.n_inputs:<8}"
              f"{'あり' if Path(c.weights).exists() else '無し'}")
    print(f"\n解析対象 {len(CONDITIONS)} 条件 / 対象外 {len(EXCLUDED)} 条件")
