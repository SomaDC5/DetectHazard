# -*- coding: utf-8 -*-
"""精度の記録と集約。

PCを2台同時に動かすので、**単一のCSVを両方から編集するとマージ衝突が必ず起きる。**
そこで実体は「1実験1ファイル / 1評価1ファイル」の追記専用にして、
`metrics.csv` はそれを結合した生成物として扱う。

    results/
    ├── runs/<run_id>.json      1学習。run_id = <条件>__<機械名>__<日時>
    ├── evals/<eval_id>.json    1評価
    ├── curves/<条件>.csv       学習曲線（旧 losses.csv）
    ├── ssd_manifests/<id>.csv  どのSSDにどの重みがあるか
    └── metrics.csv             ★ 一括管理。tools/make_metrics.py で再生成

ファイル名に機械名と日時が入るので、2台が同時に書いても衝突しない。
`metrics.csv` が衝突したら、直さずに作り直せばよい。

    from dc5lib.results import record_eval, new_run_id
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

from . import paths

# metrics.csv の列。順番はここが正。
COLUMNS = [
    "eval_id", "run_id", "condition", "region", "tileset", "scope",
    "metric_kind", "setting",
    "recall", "precision", "f1",
    "tp", "fp", "fn", "n_tiles", "n_gt_instances", "n_pred_instances",
    "evaluated_at", "machine", "weights_sha256", "note",
]

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")

# ファイル名に使う ASCII 表記。日本語をそのまま使うと OS 間で扱いが揺れるため、
# 記録そのもの（JSON の中身）は日本語のまま、**ファイル名だけ** ASCII にする。
_TOKENS = {
    # tileset
    "警戒のみ": "gtonly", "背景込み": "withbg", "全件": "all", "背景のみ": "bgonly",
    # scope
    "通常": "full", "境界": "center",
    # metric_kind
    "面積": "area", "箇所": "inst",
}


def _token(s) -> str:
    """ファイル名の一部にする。既知の日本語は決め打ち、未知はハッシュに落とす。"""
    s = str(s)
    if not s:
        return ""
    if s in _TOKENS:
        return _TOKENS[s]
    out = _SAFE.sub("-", s).strip("-")
    # 日本語などで中身が消えてしまう場合は、元の文字列のハッシュを混ぜて衝突を防ぐ
    if out.replace("-", "") == "" or any(ord(ch) > 127 for ch in s):
        import hashlib
        h = hashlib.md5(s.encode("utf-8")).hexdigest()[:6]
        out = f"{out.strip('-') or 'x'}-{h}"
    return out


def _slug(s: str) -> str:
    return _SAFE.sub("-", str(s))


def _num(v, nd=6):
    """空文字や None は None に。数えられない評価（正解0箇所など）があるため。"""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if f == f else None


def new_run_id(condition: str, machine: str | None = None, when: str | None = None) -> str:
    machine = machine or paths.machine_name()
    when = when or time.strftime("%Y%m%d-%H%M%S")
    return f"{_slug(condition)}__{_slug(machine)}__{when}"


def new_eval_id(run_id: str, region: str, tileset: str, scope: str,
                metric_kind: str, setting: str = "") -> str:
    parts = [run_id, region, tileset, scope, metric_kind] + ([setting] if setting else [])
    return "__".join(_token(p) for p in parts)


def record_run(condition: str, *, run_id: str | None = None, **fields) -> Path:
    """1回の学習を記録する。"""
    run_id = run_id or new_run_id(condition)
    rec = {"run_id": run_id, "condition": condition,
           "machine": fields.pop("machine", paths.machine_name()),
           "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rec.update(fields)
    p = paths.results_dir("runs") / f"{run_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True),
                 encoding="utf-8")
    return p


def record_eval(*, run_id, condition, region, tileset, scope, metric_kind,
                recall, precision, f1, setting="", **fields) -> Path:
    """1つの評価を記録する。

    region      hiroshima / hiroshima_bg / shimane
    tileset     警戒のみ / 背景込み / 全件
    scope       通常 / 境界
    metric_kind 面積 / 箇所
    setting     箇所数評価の対応づけ方式など（面積評価なら空）
    """
    eval_id = new_eval_id(run_id, region, tileset, scope, metric_kind, setting)
    rec = {"eval_id": eval_id, "run_id": run_id, "condition": condition,
           "region": region, "tileset": tileset, "scope": scope,
           "metric_kind": metric_kind, "setting": setting,
           "recall": _num(recall), "precision": _num(precision), "f1": _num(f1),
           "evaluated_at": fields.pop("evaluated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
           "machine": fields.pop("machine", paths.machine_name())}
    rec.update(fields)
    p = paths.results_dir("evals") / f"{eval_id}.json"
    p.write_text(json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True),
                 encoding="utf-8")
    return p


def load_evals():
    out = []
    for p in sorted(paths.results_dir("evals").glob("*.json")):
        if p.name.startswith("._"):
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[results] 読めません {p.name}: {e}")
    return out


def load_runs():
    out = []
    for p in sorted(paths.results_dir("runs").glob("*.json")):
        if p.name.startswith("._"):
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[results] 読めません {p.name}: {e}")
    return out


def build_metrics_csv() -> Path:
    """evals/*.json を結合して metrics.csv を作り直す。

    決定的な順序で書き出すので、どのPCで作っても同じ内容になる。
    衝突したらこれを実行して上書きすればよい。
    """
    rows = load_evals()
    rows.sort(key=lambda r: (r.get("condition", ""), r.get("region", ""),
                             r.get("tileset", ""), r.get("metric_kind", ""),
                             r.get("scope", ""), r.get("setting", ""),
                             r.get("eval_id", "")))
    out = paths.results_dir() / "metrics.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore",
                            lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
    return out
