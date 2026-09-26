# -*- coding: utf-8 -*-
"""保存済みの重みで推論をやり直し、タイル単位の結果を output/tiles に書き出す。

同じpklを使う条件はまとめて処理するので、5GBのpklは1地域あたり最大4回しか
読み込まない。

    python scripts/run_inference.py --region hiroshima
    python scripts/run_inference.py --region shimane
    python scripts/run_inference.py --region both --verify

--verify を付けると、タイル集計の合計が dc5/ModelComparison の results.json
（ノートブックの出力から抽出した値）と一致するかを照合する。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from analysis import config, infer  # noqa: E402
from analysis.data import build_tileset  # noqa: E402


def load_reference():
    path = os.path.join(config.MODELCOMPARISON_DIR, "output", "results.json")
    if not os.path.exists(path):
        return {}
    return {r["condition"]: r for r in json.load(open(path, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both", choices=["hiroshima", "hiroshima_bg", "shimane", "both", "all"])
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="条件名を指定（省略時は全14条件）")
    ap.add_argument("--device", default=None, help="cuda / mps / cpu（省略時は自動）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="既存のCSVがあっても再計算する")
    ap.add_argument("--verify", action="store_true",
                    help="ModelComparison の results.json と突き合わせる")
    args = ap.parse_args()

    config.ensure_dirs()
    device = infer.pick_device(args.device)
    print(f"device = {device}")

    conds = config.CONDITIONS
    if args.conditions:
        conds = [config.BY_NAME[n] for n in args.conditions]

    if args.region == "both":
        regions = ["hiroshima", "shimane"]
    elif args.region == "all":
        regions = config.REGIONS
    else:
        regions = [args.region]
    ref = load_reference() if args.verify else {}
    verify_rows = []

    for region in regions:
        todo = [c for c in conds
                if args.force or not os.path.exists(infer.tile_path(c.name, region))]
        skipped = [c for c in conds if c not in todo]
        if skipped:
            print(f"[{region}] 済み {len(skipped)} 条件はスキップ（--force で再計算）")
        if not todo:
            continue

        for pkl_path, group in config.group_by_pkl(todo, region).items():
            need_air = any(c.use_airphoto for c in group)
            print(f"\n[{region}] {os.path.basename(pkl_path)} を使う条件: "
                  f"{', '.join(c.name for c in group)}")
            tiles = build_tileset(pkl_path, region, need_air)

            for c in group:
                print(f"  → {c.name}")
                df, meta = infer.run_condition(c, tiles, device, batch_size=args.batch_size)
                df.to_csv(infer.tile_path(c.name, region), index=False)

                t = infer.totals(df)
                print(f"     通常 R={t['通常']['recall']:.4f} P={t['通常']['precision']:.4f} "
                      f"F1={t['通常']['f1']:.4f}   "
                      f"境界 F1={t['境界']['f1']:.4f}   "
                      f"(best epoch {meta.get('epoch', '?')})")

                if args.verify and c.name in ref:
                    ev = ref[c.name]["evaluations"].get(f"{region}_通常")
                    evb = ref[c.name]["evaluations"].get(f"{region}_境界")
                    if ev:
                        verify_rows.append(dict(
                            condition=c.name, region=region,
                            f1_notebook=ev["f1"], f1_rerun=round(t["通常"]["f1"], 4),
                            f1b_notebook=(evb or {}).get("f1"),
                            f1b_rerun=round(t["境界"]["f1"], 4),
                            tp_notebook=ev["TP"], tp_rerun=t["通常"]["TP"],
                        ))
            del tiles

    if verify_rows:
        v = pd.DataFrame(verify_rows)
        v["差(通常F1)"] = (v["f1_rerun"] - v["f1_notebook"]).round(4)
        v["差(境界F1)"] = (v["f1b_rerun"] - v["f1b_notebook"]).round(4)
        out = os.path.join(config.REPORT_DIR, "verify_against_notebooks.csv")
        v.to_csv(out, index=False)
        print("\n=== ノートブック出力との照合 ===")
        print(v[["condition", "region", "f1_notebook", "f1_rerun",
                 "差(通常F1)", "差(境界F1)"]].to_string(index=False))
        worst = v["差(通常F1)"].abs().max()
        print(f"\n通常F1の最大ずれ: {worst:.4f}  → {out}")


if __name__ == "__main__":
    main()
