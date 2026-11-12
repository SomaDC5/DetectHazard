# -*- coding: utf-8 -*-
"""重みの保存と読み出し。**重みに自分の素性を持たせる。**

なぜ要るか
----------
いまの `best_model.pth` には `model_state_dict` と `epoch` と
`best_f1_score` しか入っていない。だから

- **その重みがどの設定で学習されたか分からない。** config は後から
  書き換えられるし、`設計図_現状.md` の `2` のとおり config の項目の
  多くはそもそもノートブックに届いていなかった
- **アンサンブルの構成員が同じ分割で学習されたか分からない。**
  今日それを確かめるために pkl を 28GB 読み直した
  （`tools/audit_splits.py`）

**素性を重みに焼き込めば、重みを読むだけで両方が分かる。**

互換
----
`model_state_dict` の鍵はそのまま残すので、既存の
`dc5lib.models.load_weights` は**変更なしで読める**。`meta` は増えた鍵。
逆に、`meta` の無い古い重みを `meta_of()` に渡すと `None` が返る
（**無いことを「分からない」として返す。でっち上げない**）。

    from dc5lib import ckpt
    ckpt.save(cond.weights, model, ckpt.build_meta(cond, split=sp, ...))
    ckpt.meta_of(cond.weights)          # dict か None
    print(ckpt.describe(cond.weights))
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

from . import paths

SCHEMA = 1          # meta の形を変えたら上げる


def git_state(repo: Path | None = None) -> dict:
    """いまの commit と、作業ツリーが汚れているか。

    **取れなければ空で返す。** 嘘の hash を入れない。
    """
    repo = Path(repo or paths.REPO_ROOT)
    out = {}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            stderr=subprocess.DEVNULL, text=True).strip()
        out["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo,
            stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:                                   # noqa: BLE001
        pass
    return out


def versions() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
    }


def dataset_facts(cond, regions=("Hiroshima",)) -> dict:
    """pkl のファイル名・大きさ・更新時刻。**開かない。**"""
    out = {}
    for r in regions:
        try:
            p = cond.pkl_for(r)
        except Exception:                               # noqa: BLE001
            continue
        if p.exists():
            st = p.stat()
            out[r] = {"file": p.name, "bytes": st.st_size,
                      "mtime": int(st.st_mtime)}
        else:
            out[r] = {"file": p.name, "bytes": None, "mtime": None}
    return out


def build_meta(cond, cfg=None, split=None, region="Hiroshima",
               seeded=None, extra=None) -> dict:
    """重みに添える素性をまとめる。

    `split` は `dc5lib.pipeline.build_loaders` が返す辞書をそのまま渡す。
    **`fingerprint` が入っていることが肝**で、これで別の学習と
    同じ分割かを重みだけで照合できる。
    """
    from .traincfg import train_cfg
    cfg = cfg or train_cfg(cond)
    meta = {
        "schema": SCHEMA,
        "condition": cond.name,
        "protocol": getattr(cond, "protocol", ""),
        "ensemble": bool(getattr(cond, "ensemble", True)),
        "arch": cond.arch,
        "inputs": [{"key": i.key, "source": i.source,
                    "channels": i.channels, "field": i.field}
                   for i in cond.inputs],
        "bg_ratio": float(getattr(cond, "bg_ratio", 0.0) or 0.0),
        "loss": dict((cond.train or {}).get("loss") or {}),
        "train_cfg": (asdict(cfg) if is_dataclass(cfg) else dict(cfg)),
        "region": region,
        "dataset": dataset_facts(cond, (region,)),
        "split": dict(split or {}),
        "seeded": (bool(seeded) if seeded is not None
                   else bool(getattr(cond, "seeded", False))),
        "git": git_state(),
        "versions": versions(),
        "machine": paths.machine_name(),
        "saved_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    meta["train_cfg"].pop("source", None)       # 由来の表は重みに要らない
    meta.update(extra or {})
    return meta


def save(path, model, meta: dict, epoch=None, best_f1=None, **more):
    """`best_model.pth` を書く。**既存の鍵の名前を変えない。**"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"model_state_dict": model.state_dict(), "meta": meta, **more}
    if epoch is not None:
        obj["epoch"] = epoch
    if best_f1 is not None:
        obj["best_f1_score"] = best_f1
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                 # 途中で落ちても壊れた重みを残さない
    return path


def meta_of(path):
    """重みの素性。**無ければ None**（古い重み）。"""
    path = Path(path)
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return None
    return obj.get("meta")


def describe(path) -> str:
    m = meta_of(path)
    if m is None:
        return (f"{Path(path).name}: **素性がありません**（meta 無し）。"
                "固定前の世代で学習した重みです")
    g, v, s = m.get("git", {}), m.get("versions", {}), m.get("split", {})
    tc = m.get("train_cfg", {})
    rows = [
        f"条件        {m.get('condition')}  （{m.get('protocol')} / "
        f"ensemble={m.get('ensemble')} / seeded={m.get('seeded')}）",
        f"構造        {m.get('arch')}  入力 "
        f"{[i['channels'] for i in m.get('inputs', [])]}",
        f"損失        {m.get('loss')}",
        f"学習設定    epochs={tc.get('epochs')} batch={tc.get('batch_size')} "
        f"lr={tc.get('lr')} seed={tc.get('seed')} "
        f"patience={tc.get('early_stop_patience')}",
        f"分割        {s.get('kind', '?')} / 指紋 {s.get('fingerprint', '?')} "
        f"/ 学習 {s.get('n_train')} 検証 {s.get('n_val')} "
        f"テスト {s.get('n_test')}",
        f"データ      {m.get('dataset')}",
        f"git         {g.get('commit', '?')[:12]}"
        f"{'（作業ツリーが汚れている）' if g.get('dirty') else ''}",
        f"版          torch {v.get('torch')} / cuda {v.get('cuda')} "
        f"/ {v.get('device')}",
        f"機械・時刻  {m.get('machine')} / {m.get('saved_at')}",
    ]
    for k in ("epochs_ran", "best_epoch", "best_val_f1"):
        if k in m:
            rows.append(f"{k:<12}{m[k]}")
    return "\n".join(rows)


def same_split(a, b) -> bool | None:
    """2つの重みが同じ分割で学習されたか。**どちらかに素性が無ければ None。**"""
    ma, mb = meta_of(a), meta_of(b)
    if ma is None or mb is None:
        return None
    fa = (ma.get("split") or {}).get("fingerprint")
    fb = (mb.get("split") or {}).get("fingerprint")
    if not fa or not fb:
        return None
    return fa == fb


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="重みの素性を出す")
    ap.add_argument("target", nargs="+", help="条件名か .pth のパス")
    a = ap.parse_args()
    from .registry import get
    for t in a.target:
        p = Path(t)
        if not p.exists():
            p = get(t).weights
        print("=" * 78)
        print(describe(p))
    if len(a.target) == 2:
        ps = []
        for t in a.target:
            p = Path(t)
            ps.append(p if p.exists() else get(t).weights)
        r = same_split(*ps)
        print("=" * 78)
        print("同じ分割か:",
              {True: "はい", False: "**いいえ**", None: "分からない（素性が無い）"}[r])
