# -*- coding: utf-8 -*-
"""評価に使う地域の一覧を regions.yaml から読む。

地域名をコードに直書きすると、県を増やすたびに各所を直すことになる。
定義は regions.yaml の1か所に集める。

    from dc5lib.regions import all_regions, get
    for r in all_regions():
        print(r.key, r.label, r.dir, r.subset)
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from . import paths

CONFIG = paths.REPO_ROOT / "regions.yaml"


@dataclass
class Region:
    key: str
    dir: str
    label: str
    role: str = "eval"       # train / eval
    subset: str = "all"      # all / holdout / background
    note: str = ""

    def dataset(self, filename: str):
        return paths.dataset_path(self.dir, filename)


_cache = None


def all_regions(role=None):
    global _cache
    if _cache is None:
        d = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        _cache = [Region(key=k, **v) for k, v in d["regions"].items()]
    return [r for r in _cache if role is None or r.role == role]


def get(key: str) -> Region:
    for r in all_regions():
        if r.key == key:
            return r
    raise KeyError(f"未知の地域: {key}。regions.yaml に定義してください。"
                   f"（定義済み: {[r.key for r in all_regions()]}）")


def keys():
    return [r.key for r in all_regions()]


def label(key: str) -> str:
    try:
        return get(key).label
    except KeyError:
        return key


if __name__ == "__main__":
    print(f"{'key':<16}{'フォルダ':<12}{'表示':<18}{'役割':<8}{'対象タイル'}")
    for r in all_regions():
        print(f"{r.key:<16}{r.dir:<12}{r.label:<18}{r.role:<8}{r.subset}")
