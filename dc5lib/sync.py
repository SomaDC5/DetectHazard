# -*- coding: utf-8 -*-
"""2台のSSDを突き合わせる。

SSDが2台あり、2台のPCで同時に学習を回す。放っておくと
「どっちのSSDにどの重みがあるか分からない」状態になるので、
各SSDに MANIFEST.csv（sha256つき）を持たせて照合する。

    python -m dc5lib.sync verify              手元のSSDを検証し MANIFEST.csv を作り直す
    python -m dc5lib.sync export              MANIFEST を results/ssd_manifests/ に写す
    python -m dc5lib.sync diff  <他方のパス>   2台を突き合わせる（両方挿したとき）
    python -m dc5lib.sync pull  <他方のパス>   足りないものだけコピーする
    python -m dc5lib.sync status              git に載っている全SSDの在庫を表示

MANIFEST を results/ssd_manifests/<ssd_id>.csv に写して git に載せておけば、
**SSDを挿さなくても「どの重みがどちらにあるか」が分かる。**
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from . import paths

KINDS = ("weights", "checkpoints", "datasets")
FIELDS = ["path", "size", "sha256", "kind"]


def sha256(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def _targets(root: Path):
    for kind in KINDS:
        d = root / kind
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.name.startswith("._"):
                yield kind, p


def manifest_path(root: Path) -> Path:
    return root / "MANIFEST.csv"


def read_manifest(root: Path):
    p = manifest_path(root)
    if not p.exists():
        return {}
    return {r["path"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def write_manifest(root: Path, rows):
    with open(manifest_path(root), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


def verify(root: Path, rehash=True):
    """実ファイルを走査して MANIFEST.csv を作り直し、旧版との差を報告する。"""
    old = read_manifest(root)
    rows, changed, missing = [], [], []
    t0, total = time.time(), 0
    items = list(_targets(root))
    for i, (kind, p) in enumerate(items, 1):
        rel = str(p.relative_to(root)).replace("\\", "/")
        size = p.stat().st_size
        digest = sha256(p) if rehash else old.get(rel, {}).get("sha256", "")
        rows.append({"path": rel, "size": size, "sha256": digest, "kind": kind})
        if rel in old and old[rel]["sha256"] and digest and old[rel]["sha256"] != digest:
            changed.append(rel)
        total += size
        print(f"  {i}/{len(items)}  {human(total)}", end="\r", flush=True)
    seen = {r["path"] for r in rows}
    missing = [k for k in old if k not in seen]
    write_manifest(root, rows)
    print(f"\n{len(rows)} 件 / {human(total)} / {time.time()-t0:.0f} 秒")
    if changed:
        print(f"  ! 中身が変わったファイル {len(changed)} 件: {changed[:5]}")
    if missing:
        print(f"  ! 消えたファイル {len(missing)} 件: {missing[:5]}")
    return rows


def export_manifest(root: Path):
    info = json.loads((root / paths.MARKER).read_text(encoding="utf-8"))
    sid = str(info.get("ssd_id", "unknown"))
    dst = paths.results_dir("ssd_manifests") / f"{sid}.csv"
    shutil.copyfile(manifest_path(root), dst)
    print(f"ssd_id={sid} の在庫を {dst} に書き出しました（git に載ります）")
    return dst


def diff(a: Path, b: Path):
    ma, mb = read_manifest(a), read_manifest(b)
    if not ma or not mb:
        print("どちらかの MANIFEST.csv がありません。先に verify を実行してください。")
        return
    only_a = sorted(set(ma) - set(mb))
    only_b = sorted(set(mb) - set(ma))
    diff_hash = sorted(k for k in set(ma) & set(mb)
                       if ma[k]["sha256"] != mb[k]["sha256"])
    print(f"A = {a}\nB = {b}\n")
    print(f"A にだけある : {len(only_a)} 件 "
          f"({human(sum(int(ma[k]['size']) for k in only_a))})")
    for k in only_a[:10]:
        print("   ", k)
    print(f"B にだけある : {len(only_b)} 件 "
          f"({human(sum(int(mb[k]['size']) for k in only_b))})")
    for k in only_b[:10]:
        print("   ", k)
    print(f"中身が違う   : {len(diff_hash)} 件")
    for k in diff_hash[:10]:
        print("   ", k)
    return only_a, only_b, diff_hash


def pull(dst: Path, src: Path, dry_run=True):
    """src にあって dst に無い（または中身が違う）ものをコピーする。"""
    msrc, mdst = read_manifest(src), read_manifest(dst)
    todo = [k for k in msrc
            if k not in mdst or mdst[k]["sha256"] != msrc[k]["sha256"]]
    size = sum(int(msrc[k]["size"]) for k in todo)
    print(f"コピー対象 {len(todo)} 件 / {human(size)}")
    for k in todo[:20]:
        print("   ", k)
    if dry_run:
        print("--run を付けると実行します。")
        return
    for i, k in enumerate(todo, 1):
        s, d = src / k, dst / k
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        print(f"  {i}/{len(todo)} {k}", flush=True)
    print("完了。dst 側で verify を実行してください。")


def status():
    """git に載っている在庫表を全部読んで、どのSSDに何があるかを表示。"""
    d = paths.results_dir("ssd_manifests")
    files = [p for p in sorted(d.glob("*.csv")) if not p.name.startswith("._")]
    if not files:
        print("在庫表がありません。各SSDで verify → export を実行してください。")
        return
    inv = {}
    for p in files:
        sid = p.stem
        for r in csv.DictReader(open(p, encoding="utf-8")):
            inv.setdefault(r["path"], set()).add(sid)
    ids = sorted({p.stem for p in files})
    print(f"SSD: {', '.join(ids)}\n")
    print(f"{'ファイル':<62}" + "".join(f"{i:>5}" for i in ids))
    for k in sorted(inv):
        marks = "".join(f"{'o' if i in inv[k] else '-':>5}" for i in ids)
        print(f"{k:<62}{marks}")
    only = {i: sum(1 for v in inv.values() if v == {i}) for i in ids}
    print(f"\n片方にしかないファイル: {only}")


def main(argv):
    if not argv:
        print(__doc__)
        return
    cmd = argv[0]
    root = paths.data_root()
    if cmd == "verify":
        verify(root, rehash="--fast" not in argv)
        export_manifest(root)
    elif cmd == "export":
        export_manifest(root)
    elif cmd == "diff":
        diff(root, Path(argv[1]))
    elif cmd == "pull":
        pull(root, Path(argv[1]), dry_run="--run" not in argv)
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
