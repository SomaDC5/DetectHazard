# -*- coding: utf-8 -*-
"""アンサンブルの一覧を ensembles.yaml から読む。

なぜ experiments/ ではないのか
------------------------------
`experiments/<条件>/` は「**1回の学習**」を表す単位になっている。
アンサンブルには学習が無いので、そこに置くと嘘を書くことになる。

    ノートブック   学習ループが無いので中身が空になる（check.py は存在を要求する）
    arch           単一のモデルクラスが存在しない
    inputs         メンバーごとに入力が違う（DEM系と SAM系を混ぜる）

registry では永久に「重み無し」と表示され、results/runs/ も作れない。
これらの検査を緩めると、実験条件側の安全性が下がる。

そこで regions.yaml と同じ発想で、リポジトリ直下の1ファイルに集める。

    from dc5lib.ensembles import all_ensembles, get
    e = get("Ens_InputDiverse4")
    e.members, e.weights_normalized(), e.rule, e.threshold

評価は analysis/run_ensemble_eval.py --ensemble <名前>。
結果は results/evals/ に入るので metrics.csv に単一条件と並んで載る
（build_metrics_csv は evals だけを読み、runs とは結合していない）。

    python -m dc5lib.ensembles          # 一覧と検査結果
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import yaml

from . import paths

CONFIG = paths.REPO_ROOT / "ensembles.yaml"

# 統合規則。docs/アンサンブルで解けるか.md 2.1 の実測では
#   単純平均 0.7102 / 最大 0.6974 / 中心重みつき平均 0.7446
# で、**最大（= OR の連続版）が最下位**だった。既定は mean。
RULES = {
    "mean":   "重みつき平均（既定）",
    "max":    "最大。OR の連続版。実測では最下位",
    "median": "中央値。外れたメンバー1つに引きずられにくい",
}

# weights を使わない規則。指定されていたら check() で止める（黙って無視しない）
WEIGHTLESS_RULES = {"max", "median"}


@dataclass
class Ensemble:
    name: str
    members: list
    label: str = ""
    note: str = ""
    weights: list | None = None      # None なら均等
    rule: str = "mean"
    threshold: float = 0.5
    status: str = "active"           # active / excluded
    extra: dict = field(default_factory=dict)

    def weights_normalized(self):
        """合計1に正規化した重み。weights が None なら均等。"""
        n = len(self.members)
        w = [1.0] * n if self.weights is None else [float(x) for x in self.weights]
        s = sum(w)
        return [x / s for x in w]

    def display(self) -> str:
        return self.label or f"アンサンブル {len(self.members)}条件"


_cache = None


def all_ensembles(include_excluded=False):
    global _cache
    if _cache is None:
        if not CONFIG.exists():
            _cache = []
        else:
            d = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
            _cache = []
            for name, v in (d.get("ensembles") or {}).items():
                v = dict(v or {})
                known = {"members", "label", "note", "weights", "rule", "threshold", "status"}
                extra = {k: v.pop(k) for k in list(v) if k not in known}
                _cache.append(Ensemble(name=name, extra=extra, **v))
    return [e for e in _cache if include_excluded or e.status == "active"]


def get(name: str) -> Ensemble:
    for e in all_ensembles(include_excluded=True):
        if e.name == name:
            return e
    raise KeyError(f"未知のアンサンブル: {name}。ensembles.yaml に定義してください。"
                   f"（定義済み: {[e.name for e in all_ensembles(include_excluded=True)]}）")


def names(include_excluded=False):
    return [e.name for e in all_ensembles(include_excluded)]


def check():
    """コミット前の検査。異常があれば行を返す（tools/check.py から呼ぶ）。"""
    from . import registry

    problems = []
    conds = {c.name: c for c in registry.load_all(include_excluded=True)}
    seen_labels = {}

    for e in all_ensembles(include_excluded=True):
        if e.name in conds:
            problems.append(f"アンサンブル '{e.name}': 同名の実験条件があります"
                            f"（metrics.csv で区別できなくなります）")
        if not e.members:
            problems.append(f"アンサンブル '{e.name}': members が空です")
            continue
        if len(set(e.members)) != len(e.members):
            problems.append(f"アンサンブル '{e.name}': members に重複があります")

        missing = [m for m in e.members if m not in conds]
        if missing:
            problems.append(f"アンサンブル '{e.name}': members に無い条件があります: {missing}")

        if e.weights is not None and len(e.weights) != len(e.members):
            problems.append(f"アンサンブル '{e.name}': weights の数 {len(e.weights)} が "
                            f"members の数 {len(e.members)} と合いません")
        if e.weights is not None and any(float(x) < 0 for x in e.weights):
            problems.append(f"アンサンブル '{e.name}': weights に負の値があります")

        if e.rule not in RULES:
            problems.append(f"アンサンブル '{e.name}': 未知の rule '{e.rule}'"
                            f"（使えるのは {list(RULES)}）")
        # weights は mean でしか使われない。median / max に指定しても黙って
        # 無視されるので、気づけるように止める
        if e.weights is not None and e.rule in WEIGHTLESS_RULES:
            problems.append(f"アンサンブル '{e.name}': rule '{e.rule}' は weights を"
                            f"使いません。weights を消すか rule を mean にしてください")
        if not 0 < float(e.threshold) < 1:
            problems.append(f"アンサンブル '{e.name}': threshold {e.threshold} が 0〜1 の外です")

        # **bg_ratio が揃っていないとテスト集合が別物になる。**
        # ノートブックは背景タイルを混ぜたあとで分割するので、bg_ratio が違うと
        # train_test_split の結果が変わり、タイルが突き合わない（評価が止まる）。
        present = [m for m in e.members if m in conds]
        bgs = {conds[m].bg_ratio for m in present}
        if len(bgs) > 1:
            problems.append(
                f"アンサンブル '{e.name}': members の bg_ratio が揃っていません {sorted(bgs)}。"
                f"分割が変わるのでタイルが突き合いません")

        # 弱いメンバーを混ぜると平均に引きずられる。止めはしないが気づけるように
        if e.status == "active" and len(present) < 2:
            problems.append(f"アンサンブル '{e.name}': 使えるメンバーが {len(present)} 件しかありません")

        lab = e.display()
        seen_labels.setdefault(lab, []).append(e.name)

    for lab, v in seen_labels.items():
        if len(v) > 1:
            problems.append(f"アンサンブルの表示ラベル「{lab}」が {len(v)} 件で重複しています: {v}")
    return problems


if __name__ == "__main__":
    es = all_ensembles(include_excluded=True)
    if not es:
        print(f"定義がありません（{CONFIG}）")
        sys.exit(0)
    print(f"{'名前':<26}{'規則':<8}{'しきい値':>8}  {'状態':<10}メンバー")
    for e in es:
        print(f"{e.name:<26}{e.rule:<8}{e.threshold:>8}  {e.status:<10}{len(e.members)}条件")
        for m, w in zip(e.members, e.weights_normalized()):
            print(f"{'':<26}  {w:.3f}  {m}")
    problems = check()
    print()
    if problems:
        print("検査で問題が見つかりました:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("検査: 問題なし")
