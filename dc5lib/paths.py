# -*- coding: utf-8 -*-
"""外部SSD（dc5-data）の場所を自動で見つける。

このプロジェクトは3つのOSで動く。

    Windows   D:\\dc5-data  など（ドライブレターは挿す順で変わる）
    Ubuntu    /media/<user>/<label>/dc5-data, /mnt/<label>/dc5-data
    macOS     /Volumes/<label>/dc5-data

ドライブレターやラベルを決め打ちすると、どれかのOSで必ず壊れる。
そこで **SSDの中に置いた目印 `.dc5-root.json` を探す** 方式にしてある。
ラベルが何であっても、2台のSSDのどちらが挿さっていても見つかる。

探索の順序
    1. 環境変数 DC5_DATA が指すフォルダ
    2. リポジトリと同じ階層（<repo>/../dc5-data）
    3. OSごとのマウント先を総当たり

SSDが2台あるので、`.dc5-root.json` の `ssd_id` で区別する。
複数見つかったときは、環境変数 DC5_SSD_ID が一致するものを選ぶ。
指定がなければ最初に見つかったものを使い、警告を出す。

使い方
    from dc5lib.paths import weights_path, dataset_path, data_root
    torch.save(state, weights_path("FinalFusion_SAM_APM"))
    ds = load_dataset(dataset_path("Hiroshima", "hiroshima_sam_apm.pkl"))

    python -m dc5lib.paths        # いまどこを見ているかを表示
"""

from __future__ import annotations

import json
import os
import string
import sys
from pathlib import Path

MARKER = ".dc5-root.json"
DATA_DIRNAME = "dc5-data"

# リポジトリのルート（このファイルは <repo>/dc5lib/paths.py）
REPO_ROOT = Path(__file__).resolve().parent.parent

_cached_root: Path | None = None


def _candidate_mounts():
    """OSごとに、SSDがマウントされうる場所を並べる。"""
    out = []
    if sys.platform.startswith("win"):
        for letter in string.ascii_uppercase[3:]:      # D: 以降
            out.append(Path(f"{letter}:/"))
    elif sys.platform == "darwin":
        out.extend(sorted(Path("/Volumes").glob("*")))
    else:                                              # Linux (Ubuntu)
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        for base in (Path("/media") / user, Path("/media"), Path("/mnt"),
                     Path("/run/media") / user):
            if base.is_dir():
                out.extend(sorted(base.glob("*")))
    return out


def _read_marker(data_dir: Path):
    try:
        return json.loads((data_dir / MARKER).read_text(encoding="utf-8"))
    except Exception:
        return None


def find_data_roots():
    """見つかった dc5-data を全部返す。[(パス, マーカーの中身), ...]"""
    found, seen = [], set()

    def consider(p: Path):
        try:
            p = p.resolve()
        except OSError:
            return
        if p in seen or not p.is_dir():
            return
        info = _read_marker(p)
        if info is not None:
            seen.add(p)
            found.append((p, info))

    env = os.environ.get("DC5_DATA")
    if env:
        consider(Path(env))
    consider(REPO_ROOT.parent / DATA_DIRNAME)          # <repo>/../dc5-data
    consider(REPO_ROOT.parent.parent / DATA_DIRNAME)
    for m in _candidate_mounts():
        consider(m / DATA_DIRNAME)
    return found


def data_root(required=True) -> Path | None:
    """使う dc5-data を1つ決めて返す。"""
    global _cached_root
    if _cached_root is not None:
        return _cached_root

    found = find_data_roots()
    if not found:
        if not required:
            return None
        raise FileNotFoundError(
            "dc5-data が見つかりません。外部SSDが挿さっているか確認してください。\n"
            "  ・SSD内に dc5-data/.dc5-root.json があること\n"
            "  ・見つからない場合は環境変数 DC5_DATA にフルパスを指定\n"
            f"  探した場所: {[str(m / DATA_DIRNAME) for m in _candidate_mounts()][:8]} など"
        )

    want = os.environ.get("DC5_SSD_ID")
    if want:
        for p, info in found:
            if str(info.get("ssd_id")) == want:
                _cached_root = p
                return p
        raise FileNotFoundError(
            f"DC5_SSD_ID={want} のSSDが見つかりません。"
            f"見つかったのは {[info.get('ssd_id') for _, info in found]}")

    if len(found) > 1:
        ids = [f"{info.get('ssd_id')}({p})" for p, info in found]
        print(f"[dc5lib] dc5-data が複数見つかりました: {ids}\n"
              f"         {found[0][0]} を使います。"
              f"切り替えるには環境変数 DC5_SSD_ID を設定してください。", file=sys.stderr)
    _cached_root = found[0][0]
    return _cached_root


def ssd_id() -> str:
    info = _read_marker(data_root()) or {}
    return str(info.get("ssd_id", "?"))


# ------------------------------------------------------------------ 各領域
def weights_dir(condition: str) -> Path:
    return data_root() / "weights" / condition


def weights_path(condition: str, filename: str = "best_model.pth") -> Path:
    p = weights_dir(condition)
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def checkpoint_path(condition: str, filename: str = "model_checkpoint.pth") -> Path:
    p = data_root() / "checkpoints" / condition
    p.mkdir(parents=True, exist_ok=True)
    return p / filename


def dataset_path(region: str, filename: str) -> Path:
    """region は 'Hiroshima' または 'Shimane'。"""
    return data_root() / "datasets" / region / filename


def cache_dir(*parts) -> Path:
    p = data_root().joinpath("cache", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ リポジトリ側
def experiments_dir() -> Path:
    return REPO_ROOT / "experiments"


def experiment_dir(condition: str) -> Path:
    return experiments_dir() / condition


def results_dir(*parts) -> Path:
    p = REPO_ROOT.joinpath("results", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def machine_name() -> str:
    """結果に記録する機械名。環境変数 DC5_MACHINE があればそれを使う。"""
    import platform
    return os.environ.get("DC5_MACHINE") or platform.node().split(".")[0]


if __name__ == "__main__":
    print(f"repo         : {REPO_ROOT}")
    print(f"machine      : {machine_name()}  ({sys.platform})")
    found = find_data_roots()
    if not found:
        print("dc5-data     : 見つかりません（SSDが挿さっていますか）")
        sys.exit(1)
    for p, info in found:
        mark = "←使用" if p == data_root() else ""
        print(f"dc5-data     : {p}  ssd_id={info.get('ssd_id')} {mark}")
    root = data_root()
    for name in ("weights", "checkpoints", "datasets", "cache"):
        d = root / name
        n = sum(1 for _ in d.rglob("*")) if d.is_dir() else 0
        print(f"  {name:<12} {'あり' if d.is_dir() else '無し':<4} 項目数 {n}")
