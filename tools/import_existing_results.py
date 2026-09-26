# -*- coding: utf-8 -*-
"""これまでの解析結果を results/ に取り込む（初回だけ実行する移行スクリプト）。

取り込み元は Mac 側の tileanalysis の出力。
  summary_by_condition.csv   面積評価（広島・島根）
  region_matrix.csv          地域 x テスト集合を揃えた面積評価
  instance_summary.csv       箇所数評価
  curves/<条件>.csv          学習曲線 → 最良エポックなどの run 情報

    python tools/import_existing_results.py --src ~/Desktop/Master/Research/output
"""
import argparse, csv, os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import paths, results, registry   # noqa: E402

WHEN = "2026-09-26T00:00:00"


def manifest_hashes():
    m = paths.data_root() / "MANIFEST.csv"
    out = {}
    if m.exists():
        for r in csv.DictReader(open(m, encoding="utf-8")):
            if r["kind"] == "weights":
                out[Path(r["path"]).parent.name] = r["sha256"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="tileanalysis の output フォルダ")
    args = ap.parse_args()
    src = Path(os.path.expanduser(args.src))
    rep = src / "report"
    sha = manifest_hashes()
    conds = {c.name: c for c in registry.load_all(include_excluded=True)}

    # ---- run（学習1回ぶん）
    run_id = {}
    for name, c in conds.items():
        curve = paths.results_dir("curves") / f"{name}.csv"
        best_ep = best_val = None
        n_ep = 0
        if curve.exists():
            rows = list(csv.DictReader(open(curve, encoding="utf-8-sig")))
            n_ep = len(rows)
            col = next((k for k in (rows[0].keys() if rows else [])
                        if "alidation" in k or "Test" in k), None)
            if col:
                b = max(rows, key=lambda r: float(r[col] or 0))
                best_ep, best_val = int(float(b["Epoch"])), round(float(b[col]), 4)
        rid = results.new_run_id(name, machine="lab-pc", when="20260925-000000")
        run_id[name] = rid
        results.record_run(name, run_id=rid, machine="lab-pc",
                           trained_at="2026-09（正確な日時は未記録）",
                           epochs_run=n_ep, best_epoch=best_ep,
                           best_val_f1=best_val,
                           arch=c.arch, augment=c.augment, bg_ratio=c.bg_ratio,
                           dataset=c.dataset, train=c.train,
                           weights_sha256=sha.get(name),
                           note="既存の学習結果を移行時に記録したもの。"
                                "学習日時と乱数状態は当時の記録が無い")
    print(f"runs: {len(run_id)} 件")

    n = 0

    def area(row, cond, region, tileset, prefix):
        nonlocal n
        for scope in ("通常", "境界"):
            k = f"{prefix}{scope}_f1"
            if k not in row or row[k] in ("", None):
                continue
            results.record_eval(
                run_id=run_id[cond], condition=cond, region=region, tileset=tileset,
                scope=scope, metric_kind="面積",
                recall=row.get(f"{prefix}{scope}_recall"),
                precision=row.get(f"{prefix}{scope}_precision"),
                f1=row[k], n_tiles=row.get("n_tiles"),
                evaluated_at=WHEN, machine="SomanoMacBook-Air",
                weights_sha256=sha.get(cond),
                note="tileanalysis による再推論。ノートブック出力と F値 0.0001 以内で一致")
            n += 1

    # 面積評価（警戒のみ / 全件）
    f = rep / "summary_by_condition.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            r = {k: (float(v) if k not in ("region", "condition", "label") and v not in ("", None)
                     else v) for k, v in r.items()}
            region = r["region"]
            tileset = {"hiroshima": "警戒のみ", "shimane": "全件",
                       "hiroshima_bg": "背景のみ"}.get(region, region)
            area(r, r["condition"], region, tileset, "")

    # 地域 x 集合を揃えた面積評価
    f = rep / "region_matrix.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            cond = r["condition"]
            for col, (region, tileset) in {
                    "広島_全件相当": ("hiroshima", "背景込み"),
                    "島根_警戒のみ": ("shimane", "警戒のみ")}.items():
                if r.get(f"{col}_f1"):
                    results.record_eval(
                        run_id=run_id[cond], condition=cond, region=region,
                        tileset=tileset, scope="通常", metric_kind="面積",
                        recall=r[f"{col}_recall"], precision=r[f"{col}_precision"],
                        f1=r[f"{col}_f1"], n_tiles=r.get(f"{col}_n"),
                        evaluated_at=WHEN, machine="SomanoMacBook-Air",
                        weights_sha256=sha.get(cond),
                        note="テスト集合の作り方を地域間で揃えた集計")
                    n += 1

    # 箇所数評価
    f = rep / "instance_summary.csv"
    if f.exists():
        for r in csv.DictReader(open(f, encoding="utf-8")):
            cond = r["condition"]
            region = r["region"]
            tileset = {"hiroshima": "警戒のみ", "shimane": "全件",
                       "hiroshima_bg": "背景のみ"}.get(region, region)
            results.record_eval(
                run_id=run_id[cond], condition=cond, region=region, tileset=tileset,
                scope="通常", metric_kind="箇所", setting=r["設定"],
                recall=r["箇所Recall"], precision=r["箇所Precision"], f1=r["箇所F値"],
                n_gt_instances=r["箇所_正解数"], n_pred_instances=r["箇所_予測数"],
                evaluated_at=WHEN, machine="SomanoMacBook-Air",
                weights_sha256=sha.get(cond),
                note="連結性8近傍 / 最小サイズ10px / 二値化0.5")
            n += 1

    print(f"evals: {n} 件")
    out = results.build_metrics_csv()
    print(f"metrics.csv を作成: {out}")


if __name__ == "__main__":
    main()
