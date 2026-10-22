# -*- coding: utf-8 -*-
"""条件の成果物をまとめて消す（学習をやり直す前に使う）。

なぜ必要か
----------
学習をやり直すとき、消すものが**5か所に散っている。**

    dc5-data/weights/<条件>/              重み
    dc5-data/checkpoints/<条件>/           **チェックポイント**
    results/curves/<条件>.csv              学習曲線
    cache/analysis/tiles/<条件>__*.csv     推論の結果
    cache/analysis/instances/<条件>__*.csv 箇所数評価の結果
    cache/stage2/<条件>__*.npz             2段目のかたまりデータ

**どれか1つを忘れると、黙って混ざる。**

実際に SegFormer_SAM_APM で、重みと推論キャッシュだけ消して
**チェックポイントを残したまま**回し直したことがある。
ノートブックはチェックポイントがあれば自動で再開するので、
古い重み（直したかった初期化がそのまま入っている）から学習が続き、
曲線の 1〜120 行が前回と同一、121 行目から継続という状態になった。
**新しい初期化は一度も効いていなかった。**

使い方
------
    # 何が消えるかを見る（消さない）
    python tools/reset_condition.py SegFormer_SAM_APM

    # 実際に消す
    python tools/reset_condition.py SegFormer_SAM_APM --yes

    # 消さずに退避する（記録を残したいとき）
    python tools/reset_condition.py SegFormer_SAM_APM --archive --yes

**ノートブックを止めてから実行すること。** 動いている間は
毎エポック書き直されるので、消しても戻る（Windows では消せないこともある）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dc5lib import paths                                      # noqa: E402
from dc5lib.registry import get, load_all                     # noqa: E402


def targets(cond_name):
    """消す対象を (種類, パス) で返す。存在するものだけ。"""
    out = []
    data = Path(paths.data_root())
    for sub, label in (("weights", "重み"), ("checkpoints", "チェックポイント")):
        p = data / sub / cond_name
        if p.exists():
            out.append((label, p))
    cv = Path(paths.results_dir()) / "curves" / f"{cond_name}.csv"
    if cv.exists():
        out.append(("学習曲線", cv))
    cache = Path(paths.cache_dir("analysis"))
    for sub, label in (("tiles", "推論の結果"), ("instances", "箇所数評価"),
                       ("features", "タイルの性質")):
        d = cache / sub
        if d.is_dir():
            for p in sorted(d.glob(f"{cond_name}__*")):
                out.append((label, p))
    s2 = Path(paths.cache_dir("stage2"))
    if s2.is_dir():
        for p in sorted(s2.glob(f"{cond_name}__*")):
            out.append(("2段目のかたまり", p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--yes", action="store_true", help="実際に消す／退避する")
    ap.add_argument("--archive", action="store_true",
                    help="消さずに _archive 以下に退避する")
    args = ap.parse_args()

    try:
        cond = get(args.condition)
    except KeyError:
        sys.exit(f"未知の条件: {args.condition}\n"
                 f"（定義済み: {', '.join(names()[:8])} …）")

    items = targets(cond.name)
    if not items:
        print(f"{cond.name}: 消すものはありません")
        return

    print(f"{cond.name}（{cond.arch}）の成果物")
    total = 0
    for label, p in items:
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        else:
            size = p.stat().st_size
        total += size
        print(f"  [{label:14s}] {p}  {size / 2**20:.1f} MB")
    print(f"  合計 {total / 2**20:.1f} MB")

    if not args.yes:
        print()
        print("消すなら --yes を付けてください（--archive で退避に変わります）。")
        print("**ノートブックを止めてから実行すること。**")
        return

    if args.archive:
        dest = (Path(paths.data_root()) / "_archive"
                / f"{cond.name}_{date.today():%Y-%m-%d}")
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n退避先: {dest}")
    for label, p in items:
        try:
            if args.archive:
                shutil.move(str(p), str(dest / p.name))
                print(f"  退避 [{label}] {p.name}")
            elif p.is_dir():
                shutil.rmtree(p)
                print(f"  削除 [{label}] {p}")
            else:
                p.unlink()
                print(f"  削除 [{label}] {p}")
        except OSError as e:
            print(f"  ※ 失敗 [{label}] {p}: {e}")
            print("     ノートブックが動いていると消せません。止めてから再実行を")
    rest = targets(cond.name)
    print(f"\n残り {len(rest)} 件" + ("（完了）" if not rest else "（上の※を確認）"))


if __name__ == "__main__":
    main()
