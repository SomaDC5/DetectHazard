# -*- coding: utf-8 -*-
"""2台のSSDを突き合わせる。

SSDが2台あり、2台のPCで同時に学習を回す。放っておくと
「どっちのSSDにどの重みがあるか分からない」状態になるので、
各SSDに MANIFEST.csv（sha256つき）を持たせて照合する。

    python -m dc5lib.sync verify              在庫表を更新（差分のみ再ハッシュ。--full で全件）
    python -m dc5lib.sync check               在庫表が実ファイルとずれていないか
    python -m dc5lib.sync export              在庫表を results/ssd_manifests/ に写す
    python -m dc5lib.sync diff   <他方のパス>  2台を突き合わせる
    python -m dc5lib.sync pull   <他方のパス>  他方にしか無いものを取り込む
    python -m dc5lib.sync unify  <他方のパス>  ★ 2台を双方向に揃える（--run で実行）
    python -m dc5lib.sync status              git上の在庫（SSDを挿さなくてよい）

  共通オプション
    --run            実際に実行する（既定は確認のみ）
    --weights-only   チェックポイント(8.7GB)を除き、best_model.pth だけ揃える
    --force          両方にあって中身が違うファイルも上書きする（既定は中止）

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
FIELDS = ["path", "size", "mtime", "sha256", "kind"]


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


def scan(root: Path):
    """実ファイルを走査して {相対パス: (kind, size, mtime)} を返す。"""
    out = {}
    for kind, p in _targets(root):
        st = p.stat()
        out[str(p.relative_to(root)).replace("\\", "/")] = (kind, st.st_size, int(st.st_mtime))
    return out


def manifest_is_stale(root: Path):
    """MANIFEST.csv と実ファイルがずれていないか。ずれているものを返す。

    学習して重みが増えたのに verify を忘れると、同期で取りこぼす。
    それを検出するための確認。
    """
    man, disk = read_manifest(root), scan(root)
    added = [k for k in disk if k not in man]
    removed = [k for k in man if k not in disk]
    modified = [k for k in disk if k in man and
                (int(man[k]["size"]) != disk[k][1] or
                 str(man[k].get("mtime", "")) != str(disk[k][2]))]
    return added, removed, modified


def verify(root: Path, full=False):
    """MANIFEST.csv を作り直す。

    既定は差分のみ再ハッシュする（サイズか更新時刻が変わったファイルだけ）。
    full=True で全件を計算し直す。
    """
    old = read_manifest(root)
    disk = scan(root)
    rows, changed, rehashed = [], [], 0
    t0, total = time.time(), 0
    for i, (rel, (kind, size, mtime)) in enumerate(sorted(disk.items()), 1):
        prev = old.get(rel)
        unchanged = (prev and not full
                     and int(prev["size"]) == size
                     and str(prev.get("mtime", "")) == str(mtime)
                     and prev.get("sha256"))
        if unchanged:
            digest = prev["sha256"]
        else:
            digest = sha256(root / rel)
            rehashed += 1
            total += size
            if prev and prev.get("sha256") and prev["sha256"] != digest:
                changed.append(rel)
        rows.append({"path": rel, "size": size, "mtime": mtime,
                     "sha256": digest, "kind": kind})
        print(f"  {i}/{len(disk)}  再計算 {rehashed} 件 {human(total)}", end="\r", flush=True)
    missing = [k for k in old if k not in disk]
    write_manifest(root, rows)
    print(f"\n{len(rows)} 件（うち再計算 {rehashed} 件 / {human(total)}）"
          f" / {time.time()-t0:.0f} 秒")
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


def _check_fresh(root: Path, name: str):
    """MANIFEST.csv が実ファイルとずれていたら警告して True を返す。"""
    added, removed, modified = manifest_is_stale(root)
    if added or removed or modified:
        print(f"  ! {name} の在庫表が古いです"
              f"（未登録 {len(added)} / 消失 {len(removed)} / 変更 {len(modified)}）")
        for k in (added + modified)[:5]:
            print(f"      {k}")
        print(f"    先に  DC5_SSD_ID=<id> python -m dc5lib.sync verify  を実行してください。")
        return True
    return False


def pull(dst: Path, src: Path, dry_run=True, kinds=None, force=False):
    """src にあって dst に無いものをコピーする。

    **両方にあって中身が違うファイルは、既定ではコピーしない。**
    2台のPCで同じ名前の条件を学習してしまった場合、黙って上書きすると
    片方の学習結果が消える。そういうファイルは一覧を出して止める。
    """
    stale = _check_fresh(src, "コピー元") | _check_fresh(dst, "コピー先")
    msrc, mdst = read_manifest(src), read_manifest(dst)
    if kinds:
        msrc = {k: v for k, v in msrc.items() if v["kind"] in kinds}

    new_files = [k for k in msrc if k not in mdst]
    conflicts = [k for k in msrc
                 if k in mdst and mdst[k]["sha256"] != msrc[k]["sha256"]]

    size = sum(int(msrc[k]["size"]) for k in new_files)
    print(f"コピー対象（コピー先に無いもの）: {len(new_files)} 件 / {human(size)}")
    for k in new_files[:20]:
        print("   ", k)
    if len(new_files) > 20:
        print(f"    … 他 {len(new_files)-20} 件")

    if conflicts:
        print(f"\n  !! 両方にあって中身が違うファイル: {len(conflicts)} 件")
        for k in conflicts[:20]:
            print(f"      {k}")
        print("     同じ名前の条件を2台で別々に学習した可能性があります。")
        print("     どちらを残すか決めてから、--force を付けるか、"
              "片方の条件名を変えてください。")

    todo = new_files + (conflicts if force else [])
    if stale:
        print("\n在庫表が古いので中止しました。")
        return
    if dry_run:
        print("\n--run を付けると実行します。")
        return
    if not todo:
        print("\nコピーするものはありません。")
        return
    for i, k in enumerate(todo, 1):
        s_, d = src / k, dst / k
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s_, d)
        print(f"  {i}/{len(todo)} {k}", flush=True)
    print("完了。コピー先で verify を実行してください。")
    return todo


def unify(a: Path, b: Path, dry_run=True, kinds=None):
    """2台のSSDの中身を揃える。両方向にコピーする。

    2台のPCで別々のモデルを学習したあと、これ1回で双方向に揃う。
    同じ名前の条件を両方で学習していた場合は、コピーせずに一覧を出す。
    """
    print("=== 1/3  両方の在庫表を更新 ===")
    for root, name in ((a, "A側"), (b, "B側")):
        print(f"  [{name}] {root}")
        verify(root)
    print("\n=== 2/3  A → B ===")
    copied = pull(b, a, dry_run=dry_run, kinds=kinds)
    if copied and not dry_run:
        # コピーした直後は B の在庫表が古くなるので、次の向きの前に更新する
        verify(b)
    print("\n=== 3/3  B → A ===")
    pull(a, b, dry_run=dry_run, kinds=kinds)
    if not dry_run:
        print("\n=== 仕上げ：在庫表を更新して git に載せる ===")
        for root in (a, b):
            verify(root)
            export_manifest(root)
        print("\nこのあと git add results/ssd_manifests && git commit && git push")


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
    kinds = None
    if "--weights-only" in argv:
        kinds = {"weights"}
    if cmd == "verify":
        verify(root, full="--full" in argv)
        export_manifest(root)
    elif cmd == "export":
        export_manifest(root)
    elif cmd == "diff":
        diff(root, Path(argv[1]))
    elif cmd == "pull":
        pull(root, Path(argv[1]), dry_run="--run" not in argv,
             kinds=kinds, force="--force" in argv)
    elif cmd == "unify":
        unify(root, Path(argv[1]), dry_run="--run" not in argv, kinds=kinds)
    elif cmd == "check":
        added, removed, modified = manifest_is_stale(root)
        if added or removed or modified:
            print(f"在庫表が古いです: 未登録 {len(added)} / 消失 {len(removed)} "
                  f"/ 変更 {len(modified)}")
            for k in (added + modified + removed)[:20]:
                print("   ", k)
            sys.exit(1)
        print("在庫表は最新です。")
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
