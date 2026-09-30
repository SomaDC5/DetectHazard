# -*- coding: utf-8 -*-
"""解析結果（report/*.csv）を results/ の run・eval に取り込む。

初版は Mac からの一度きりの移行用だったが、新しい条件を学習したあとにも
使えるようにしてある。既定では **取り込み元の report に出てくる条件だけ**を
対象にするので、過去に記録済みの条件を上書きしない。

取り込み元は analysis/run_analysis.py などの出力フォルダ
（既定では dc5-data/cache/analysis）。
  summary_by_condition.csv   面積評価（広島・島根）
  region_matrix.csv          地域 x テスト集合を揃えた面積評価
  instance_summary.csv       箇所数評価
  curves/<条件>.csv          学習曲線 → 最良エポックなどの run 情報

    # 新しい条件を学習したあと（機械名・日時は自動で今のものが入る）
    python tools/import_existing_results.py --src <dc5-data>/cache/analysis

    # 当時の移行を再現したいとき
    python tools/import_existing_results.py --src <path>         --machine SomanoMacBook-Air --train-machine lab-pc         --when 2026-09-26T00:00:00 --trained-when 20260925-000000
"""
import argparse, csv, os, sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import paths, results, registry   # noqa: E402

def _conditions_in_report(rep):
    """report/*.csv に出てくる条件名を集める。取り込み対象の既定になる。"""
    found = set()
    for fn in ("summary_by_condition.csv", "region_matrix.csv", "instance_summary.csv"):
        f = rep / fn
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r.get("condition"):
                found.add(r["condition"])
    return found


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
    ap.add_argument("--src", required=True, help="解析結果の出力フォルダ（report/ を含む）")
    ap.add_argument("--machine", default=None,
                    help="評価を回した機械名。既定は今の機械（DC5_MACHINE）")
    ap.add_argument("--when", default=None,
                    help="評価日時 ISO8601。既定は現在時刻")
    ap.add_argument("--train-machine", default=None,
                    help="学習した機械名。既定は --machine と同じ")
    ap.add_argument("--trained-when", default=None,
                    help="run_id に入れる学習日時 YYYYmmdd-HHMMSS。既定は現在時刻")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="取り込む条件名。既定は report に出てくる条件だけ")
    args = ap.parse_args()
    src = Path(os.path.expanduser(args.src))
    rep = src / "report"
    sha = manifest_hashes()

    machine = args.machine or paths.machine_name()
    train_machine = args.train_machine or machine
    when = args.when or time.strftime("%Y-%m-%dT%H:%M:%S")
    trained_when = args.trained_when or time.strftime("%Y%m%d-%H%M%S")

    conds = {c.name: c for c in registry.load_all(include_excluded=True)}
    target = set(args.conditions) if args.conditions else _conditions_in_report(rep)
    if not target:
        print("取り込む条件が report にありません。--conditions で指定してください。")
        return
    unknown = target - set(conds)
    if unknown:
        print("registry に無い条件は飛ばします:", ", ".join(sorted(unknown)))
    conds = {k: v for k, v in conds.items() if k in target}
    print(f"対象 {len(conds)} 条件: {', '.join(sorted(conds))}")
    print(f"評価: machine={machine} when={when}")
    print(f"学習: machine={train_machine} when={trained_when}")

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
        rid = results.new_run_id(name, machine=train_machine, when=trained_when)
        run_id[name] = rid
        results.record_run(name, run_id=rid, machine=train_machine,
                           trained_at=trained_when,
                           epochs_run=n_ep, best_epoch=best_ep,
                           best_val_f1=best_val,
                           arch=c.arch, augment=c.augment, bg_ratio=c.bg_ratio,
                           dataset=c.dataset, train=c.train,
                           weights_sha256=sha.get(name),
                           note="results/curves の学習曲線から復元した run 情報")
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
                evaluated_at=when, machine=machine,
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
                        evaluated_at=when, machine=machine,
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
                evaluated_at=when, machine=machine,
                weights_sha256=sha.get(cond),
                note="連結性8近傍 / 最小サイズ10px / 二値化0.5")
            n += 1

    print(f"evals: {n} 件")
    out = results.build_metrics_csv()
    print(f"metrics.csv を作成: {out}")


if __name__ == "__main__":
    main()
