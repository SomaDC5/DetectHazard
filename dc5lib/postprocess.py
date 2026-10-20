# -*- coding: utf-8 -*-
"""学習済みの重みに後処理を掛けた「構成」の定義を読む。

なぜ ensembles.yaml と同じ場所なのか
------------------------------------
`展望_再現率をどう上げるか.md 13` の刈り取り（ロジットの小さいかたまりを捨てる）は
**実験条件ではない。** 新しい重みを作らないので experiments/ に置く筋のものではなく、
一方で metrics.csv に載らないと論文の表に出てこない。

アンサンブルがちょうど同じ立場にあり（重みを作らず、評価だけされる）、
`ensembles.yaml` に定義して `results/evals/` に記録する方式ができている。
後処理も同じ扱いにする。**定義は1か所、コードは変えない。**

項目
----
    base         後処理を掛ける条件名（experiments/ にあること）
    union_with   1段目を和集合にする相方の条件名（展望 21・23）。
                 1本でも、リストで複数本でもよい。
                 ロジットは要素ごとの max、予測は OR。省略時は単独。
                 **3本で折り返す**（4本目は箇所F が下がる。23.3）
    inference    通常 / オーバーラップ
    feature      かたまりを並べる特徴量。logit_max / logit_mean / area
    threshold    この値**以下**のかたまりを捨てる（絶対値）
    drop_quantile  threshold をどう選んだかの記録。評価には使わない
    label / note / status

**threshold は絶対値で持つ。** 分位で持つと評価する集合が変わるたびに
意味が変わってしまう。実測では広島で決めた絶対値がそのまま島根でも効いた
（`13.2`：+0.0259 対 島根内で選び直した +0.0263）ので、絶対値で固定できる。

    一覧   python -m dc5lib.postprocess
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml

from . import paths

CONFIG = paths.REPO_ROOT / "ensembles.yaml"

FEATURES = {
    "logit_max":  "かたまり内のロジットの最大値。実測で一番分離が良い（AUC 0.8006）",
    "logit_mean": "かたまり内のロジットの平均（AUC 0.7919）",
    "area":       "かたまりの画素数（AUC 0.7692）。ロジットより弱い",
    "stage2":     "学習した2段目の出す「残す確率」。threshold は 0〜1 で、"
                  "model に重みの名前が要る",
}
INFERENCES = {
    "通常": "タイルをそのまま1枚ずつ推論する",
    "オーバーラップ": "3x3 に貼って 64px ずつずらした9窓を cos 窓で混ぜる",
}


@dataclass
class PostProcess:
    name: str
    base: str
    threshold: float
    feature: str = "logit_max"
    inference: str = "オーバーラップ"
    drop_quantile: float | None = None
    model: str | None = None          # feature="stage2" のときの重みの名前
    # 1段目を和集合にする相方の条件名。1本でも複数本でもよい（展望 21・23）
    union_with: str | list | None = None

    @property
    def union_list(self) -> list:
        """union_with を常にリストで返す。str 1本でも複数でも同じ扱いにする。"""
        if not self.union_with:
            return []
        return ([self.union_with] if isinstance(self.union_with, str)
                else list(self.union_with))
    label: str = ""
    note: str = ""
    status: str = "active"
    extra: dict = field(default_factory=dict)

    def display(self) -> str:
        return self.label or f"{self.base} + 刈り取り({self.feature}<={self.threshold:g})"


_cache = None


def all_postprocess(include_excluded=False):
    global _cache
    if _cache is None:
        if not CONFIG.exists():
            _cache = []
        else:
            d = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
            _cache = []
            known = {"base", "threshold", "feature", "inference",
                     "drop_quantile", "label", "note", "status", "model",
                     "union_with"}
            for name, v in (d.get("postprocess") or {}).items():
                v = dict(v or {})
                extra = {k: v.pop(k) for k in list(v) if k not in known}
                _cache.append(PostProcess(name=name, extra=extra, **v))
    return [p for p in _cache if include_excluded or p.status == "active"]


def get(name: str) -> PostProcess:
    for p in all_postprocess(include_excluded=True):
        if p.name == name:
            return p
    raise KeyError(f"未知の後処理: {name}。ensembles.yaml の postprocess に"
                   f"定義してください。（定義済み: "
                   f"{[p.name for p in all_postprocess(include_excluded=True)]}）")


def names(include_excluded=False):
    return [p.name for p in all_postprocess(include_excluded)]


def check():
    """コミット前の検査。異常があれば行を返す（tools/check.py から呼ぶ）。"""
    from . import ensembles, registry

    problems = []
    conds = {c.name for c in registry.load_all(include_excluded=True)}
    ens = set(ensembles.names(include_excluded=True))
    seen_labels = {}

    for p in all_postprocess(include_excluded=True):
        # metrics.csv は condition 列ひとつで区別するので、名前がかぶると混ざる
        if p.name in conds:
            problems.append(f"後処理 '{p.name}': 同名の実験条件があります"
                            f"（metrics.csv で区別できなくなります）")
        if p.name in ens:
            problems.append(f"後処理 '{p.name}': 同名のアンサンブルがあります")
        if p.base not in conds:
            problems.append(f"後処理 '{p.name}': base '{p.base}' が experiments/ にありません")
        for u in p.union_list:
            if u not in conds:
                problems.append(f"後処理 '{p.name}': union_with '{u}' が "
                                "experiments/ にありません")
        if p.feature not in FEATURES:
            problems.append(f"後処理 '{p.name}': feature '{p.feature}' は未知です"
                            f"（使えるのは {list(FEATURES)}）")
        if p.inference not in INFERENCES:
            problems.append(f"後処理 '{p.name}': inference '{p.inference}' は未知です"
                            f"（使えるのは {list(INFERENCES)}）")
        if p.threshold is None:
            problems.append(f"後処理 '{p.name}': threshold がありません")
        # ロジットのしきい値を分位で持つと集合が変わるたびに意味が変わる
        if p.feature.startswith("logit") and p.threshold is not None and p.threshold <= 0:
            problems.append(f"後処理 '{p.name}': threshold {p.threshold} はロジットとして"
                            f"小さすぎます（0以下だと二値化の境目より下で、何も刈りません）")
        if p.feature == "stage2":
            if not p.model:
                problems.append(f"後処理 '{p.name}': feature=stage2 には model（2段目の"
                                f"重みの名前）が必要です")
            elif not paths.weights_path(p.model).exists():
                problems.append(f"後処理 '{p.name}': 2段目の重みがありません "
                                f"({paths.weights_path(p.model)})")
            if p.threshold is not None and not (0.0 < p.threshold < 1.0):
                problems.append(f"後処理 '{p.name}': feature=stage2 の threshold は"
                                f"「残す確率」なので 0〜1 の間にしてください"
                                f"（いま {p.threshold}）")
        lab = p.display()
        if lab in seen_labels:
            problems.append(f"後処理 '{p.name}': 表示ラベル '{lab}' が "
                            f"'{seen_labels[lab]}' と重複しています")
        seen_labels[lab] = p.name
    return problems


if __name__ == "__main__":
    rows = all_postprocess(include_excluded=True)
    if not rows:
        print("ensembles.yaml に postprocess の定義がありません")
    for p in rows:
        mark = "" if p.status == "active" else f"  [{p.status}]"
        print(f"{p.name}{mark}")
        print(f"  {p.display()}")
        print(f"  base       {p.base}")
        if p.union_list:
            print(f"  union_with {', '.join(p.union_list)}"
                  f"（計 {1 + len(p.union_list)}本の和集合。"
                  "ロジットは要素ごとの max、予測は OR）")
        print(f"  推論       {p.inference}（{INFERENCES[p.inference]}）")
        print(f"  特徴量     {p.feature}（{FEATURES.get(p.feature, '?')}）")
        if p.model:
            print(f"  2段目      {p.model}")
        print(f"  しきい値   {p.threshold:g} 以下のかたまりを捨てる", end="")
        print(f"（選んだときは下位 {100 * p.drop_quantile:.0f}%）"
              if p.drop_quantile else "")
        if p.note:
            print(f"  メモ       {p.note.strip()}")
        print()
