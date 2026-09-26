# -*- coding: utf-8 -*-
"""重みとデータセットを dc5-data/ へ移す（同一ボリューム内の rename なので瞬時）。

移動前に移動表を JSON で残すので、--undo でいつでも元に戻せる。
    python migrate.py --dry-run
    python migrate.py --run
    python migrate.py --undo
"""
import argparse, json, os, sys, csv, hashlib, time

SSD = "/Volumes/SSD-PHPU3A"
SRC_DH = os.path.join(SSD, "dc5", "DetectHazard")
DATA = os.path.join(SSD, "dc5-data")
PLAN = os.path.join(DATA, "_migration_plan.json")

def build_plan():
    moves = []
    for cond in sorted(os.listdir(SRC_DH)):
        d = os.path.join(SRC_DH, cond)
        if not os.path.isdir(d) or cond.startswith('.'):
            continue
        for fn, dest in (("best_model.pth", "weights"),
                         ("model_checkpoint.pth", "checkpoints")):
            src = os.path.join(d, fn)
            if os.path.exists(src):
                moves.append({"src": src,
                              "dst": os.path.join(DATA, dest, cond, fn),
                              "size": os.path.getsize(src), "kind": dest})
    for region in ("Hiroshima", "Shimane"):
        src = os.path.join(SSD, "NewDatasModel", "DataSet", region)
        if os.path.isdir(src):
            for fn in sorted(os.listdir(src)):
                if fn.endswith(".pkl"):
                    s = os.path.join(src, fn)
                    moves.append({"src": s,
                                  "dst": os.path.join(DATA, "datasets", region, fn),
                                  "size": os.path.getsize(s), "kind": "datasets"})
    return moves

def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024: return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"

def sha256(path, buf=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--undo", action="store_true")
    ap.add_argument("--hash", action="store_true", help="移動後に sha256 を計算して MANIFEST.csv を作る")
    a = ap.parse_args()

    if a.undo:
        if not os.path.exists(PLAN):
            print("移動表がありません。何も戻せません。"); return
        plan = json.load(open(PLAN, encoding="utf-8"))
        n = 0
        for m in reversed(plan["moves"]):
            if os.path.exists(m["dst"]):
                os.makedirs(os.path.dirname(m["src"]), exist_ok=True)
                os.rename(m["dst"], m["src"]); n += 1
        print(f"{n} ファイルを元に戻しました。")
        os.rename(PLAN, PLAN + ".undone")
        return

    moves = build_plan()
    by_kind = {}
    for m in moves:
        k = by_kind.setdefault(m["kind"], [0, 0])
        k[0] += 1; k[1] += m["size"]
    print(f"{'種別':<12}{'件数':>6}{'容量':>12}")
    for k, (c, s) in sorted(by_kind.items()):
        print(f"{k:<12}{c:>6}{human(s):>12}")
    print(f"{'合計':<12}{len(moves):>6}{human(sum(m['size'] for m in moves)):>12}")
    print()
    for m in moves[:4] + (["..."] if len(moves) > 8 else []) + moves[-4:]:
        if m == "...": print("  ..."); continue
        print(f"  {m['src'].replace(SSD,'')}\n    → {m['dst'].replace(SSD,'')}")

    if a.dry_run:
        print("\n--dry-run のため何も移動していません。")
        return

    t0 = time.time()
    os.makedirs(DATA, exist_ok=True)
    json.dump({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "moves": moves},
              open(PLAN, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    done = 0
    for m in moves:
        os.makedirs(os.path.dirname(m["dst"]), exist_ok=True)
        os.rename(m["src"], m["dst"])
        done += 1
        print(f"  {done}/{len(moves)}", end="\r", flush=True)
    print(f"\n{done} ファイルを移動しました（{time.time()-t0:.1f} 秒）。移動表: {PLAN}")

    if a.hash:
        print("sha256 を計算中 …")
        rows = []
        for i, m in enumerate(moves, 1):
            rows.append({"path": os.path.relpath(m["dst"], DATA).replace(os.sep, "/"),
                         "size": m["size"], "sha256": sha256(m["dst"]), "kind": m["kind"]})
            print(f"  {i}/{len(moves)}", end="\r", flush=True)
        with open(os.path.join(DATA, "MANIFEST.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "size", "sha256", "kind"])
            w.writeheader(); w.writerows(rows)
        print(f"\nMANIFEST.csv を作成（{len(rows)} 件）")

if __name__ == "__main__":
    main()
