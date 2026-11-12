# -*- coding: utf-8 -*-
"""条件の成果物をまとめて消す（学習をやり直す前に使う）。

なぜ必要か
----------
学習をやり直すとき、消すものが**5か所に散っている。**

    dc5-data/weights/<条件>/              重み
    dc5-data/checkpoints/<条件>/           **チェックポイント**
    results/curves/<条件>.csv              学習曲線
    cache/analysis/*/<条件>__*             推論・箇所評価・解析の表（**全フォルダ**）
    cache/stage2/<条件>__*.npz             2段目のかたまりデータ
    cache/dry/<条件>/                      試運転の重みと曲線

**どれか1つを忘れると、黙って混ざる。**

**`results/evals/` の記録は消さない**（git に載る成果物なので）。
残っている件数だけを出す ── 重みを作り直すと、測り直していない行は
古い重みの値のまま残る。**2段目の重み（別の名前で保存される）も消さない。**

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
    # **cache/analysis の下は、フォルダを数え上げずに全部見る。**
    # 以前は tiles / instances / features の3つを名指ししていたので、
    # あとから増えた triage / building / coverage が残った
    # （2026-11-12 のレビュー。残ると、作り直した重みの解析が
    # 古い重みの表を読む ── 表どうしは一致するので検査を通ってしまう）。
    # `<条件>__` で始まるものだけを拾うので、名前の前半が同じ別の条件は拾わない。
    known = {"tiles": "推論の結果", "instances": "箇所数評価",
             "features": "タイルの性質"}
    cache = Path(paths.cache_dir("analysis"))
    if cache.is_dir():
        for d in sorted(x for x in cache.iterdir() if x.is_dir()):
            for p in sorted(d.glob(f"{cond_name}__*")):
                out.append((known.get(d.name, f"解析の表（{d.name}）"), p))
    s2 = Path(paths.cache_dir("stage2"))
    if s2.is_dir():
        for p in sorted(s2.glob(f"{cond_name}__*")):
            out.append(("2段目のかたまり", p))
    dry = Path(paths.cache_dir("dry")) / cond_name
    if dry.exists():
        out.append(("試運転の重みと曲線", dry))
    return out


def stale_records(cond_name):
    """`results/evals/` に残っている、この条件の記録。**消さない。数えるだけ。**

    記録は git に載る成果物なので、この道具では触らない。ただし重みを
    作り直すと、**測り直していない行は古い重みの値のまま残る**
    （同じ鍵で測り直した行だけが置き換わる）。黙って混ざらないよう、
    残っていることを見せる。
    """
    import json
    out = []
    for p in sorted(Path(paths.results_dir("evals")).glob("*.json")):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get(
                    "condition") == cond_name:
                out.append(p)
        except Exception:                                 # noqa: BLE001
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--yes", action="store_true", help="実際に消す／退避する")
    ap.add_argument("--archive", action="store_true",
                    help="消さずに _archive 以下に退避する")
    args = ap.parse_args()

    # `get` は config.yaml を開くので、無い条件では FileNotFoundError になる
    # （KeyError だけを受けていて、しかも未定義の names() を呼んでいた）。
    try:
        cond = get(args.condition)
    except (KeyError, FileNotFoundError):
        known = [c.name for c in load_all(include_excluded=True)]
        sys.exit(f"未知の条件: {args.condition}\n"
                 f"（定義済み: {', '.join(known[:8])} …）")

    items = targets(cond.name)
    recs = stale_records(cond.name)
    if recs:
        print(f"※ results/evals/ にこの条件の記録が {len(recs)} 件あります。"
              "**この道具は記録を消しません。**")
        print("   重みを作り直すと、測り直していない行は古い重みの値のまま残ります。")
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
