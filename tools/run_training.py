# -*- coding: utf-8 -*-
"""config だけを見て学習する。**ノートブックを使わずに回すための入口。**

なぜ要るか
----------
長い学習（2時間級）をノートブックで回すと、途中でカーネルが落ちたときに
何も残らない。実際 `nbconvert --execute` で**カーネルが何も出力せずに
消える**ことがあった（メモリは10GB以上空き、GPU も空き。同じセル構成で
通ることもあるので非決定的）。原因は特定できていない。

**だから長い学習はここから回す。** 中身はノートブックと同じで、
`dc5lib` の同じ関数を同じ順に呼ぶだけである ── 手順を二重に持たない。

    python tools/run_training.py <条件名>
    python tools/run_training.py <条件名> --epochs 1 --dry   # 試運転
    python tools/run_training.py <条件名> --no-eval

`--dry` は重みと曲線を**別の場所**に書くので、本物を上書きしない。

終わったら `config.yaml` の `protocol` を `v3-base` に直すこと
（`docs/ノートブックの作り方.md` の 2.1）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import ckpt, paths                               # noqa: E402
from dc5lib import splits as _splits                         # noqa: E402
from dc5lib.evaluate import can_record, evaluate             # noqa: E402
from dc5lib.models import build_model_for                    # noqa: E402
from dc5lib.pipeline import build_loaders, seed_everything   # noqa: E402
from dc5lib.registry import get                              # noqa: E402
from dc5lib.train import fit                                 # noqa: E402
from dc5lib.traincfg import train_cfg                        # noqa: E402

# 期待値の原本は **`dc5lib/splits.py` の EXPECT** である
# （2026-11-10 の再レビュー 3。同じ値が4か所に並んでいた）。
# **違ったら止める。** 違うのは pkl か分割の指定か前処理で、
# どれも他の条件と比べられなくする（docs/ノートブックの作り方.md の 2.3）。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition")
    ap.add_argument("--region", default="Hiroshima")
    ap.add_argument("--epochs", type=int, default=None,
                    help="config の epochs を上書きする（試運転用）")
    ap.add_argument("--dry", action="store_true",
                    help="重みと曲線を別の場所に書く（本物を上書きしない）")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-expect", action="store_true",
                    help="分割の期待値の照合を飛ばす（別の pkl の条件など）")
    ap.add_argument("--overwrite", action="store_true",
                    help="既存の重みと曲線を作り直す（既定は止まる）")
    a = ap.parse_args()

    cond = get(a.condition)
    cfg = train_cfg(cond)
    print("=" * 78)
    print(cond.display())
    print(f"  protocol {cond.protocol} / ensemble {cond.ensemble}"
          f" / seeded {cond.seeded}")
    print(cfg.describe())
    print("=" * 78, flush=True)

    # **pkl を読む前に止める**（設計書 §7「止めるなら推論の前に止める」）。
    # `fit` も同じ歯止めを呼ぶが、そこまで行くと 6GB の pkl を読み終わっている。
    # **規則をここに書き写さない** ── `fit` と同じ関数を先に呼ぶ
    # （2026-11-11 のレビュー。既存の重みの検査だけを写していたので、
    # `augment` と `train.init_from` は pkl を読んだあとで止まっていた）。
    # この道具は `transform` を渡さないので、`augment: true` はここで必ず止まる。
    # --dry は書き先が cache/dry なので、既存の検査は対象外。
    from dc5lib.train import _guard_before_training
    try:
        _guard_before_training(cond, cfg, None, cond.weights,
                               paths.curve_path(cond.name),
                               overwrite=(a.overwrite or a.dry))
    except (FileExistsError, NotImplementedError) as e:
        sys.exit(f"学習を始めません: {e}"
                 + ("\n  作り直してよいなら --overwrite を付けてください。"
                    if isinstance(e, FileExistsError) else ""))

    seed_everything(cfg.seed)
    print(f"乱数を固定: {cfg.seed}", flush=True)

    ld = build_loaders(cond, a.region)

    want = _splits.expected(a.region) if not a.no_expect else None
    if want:
        # **照合は `splits.compare_expected` が1か所で持つ**（5回目のレビュー 1）。
        # ここに書き写していたせいで、`EXPECT` に `n_all` が増えた途端に
        # `KeyError` で学習経路が死んだ。
        bad = _splits.compare_expected(a.region, ld.n_used, ld.split)
        if bad:
            for k, (g, w) in bad.items():
                print(f"  **違う** {k}: {g}（既存条件は {w}）")
            sys.exit("分割が既存条件と違います。pkl か分割の指定か前処理が違う。"
                     "意図した違いなら --no-expect を付けてください")
        print("分割の照合: 全項目一致", flush=True)

    # **記録できるかを学習の前に言う。** `can_record_split` は地域しか
    # 見ないので、csv 分割の条件は通ってしまい、学習が終わったあとの
    # `evaluate(record=True)` で初めて止まっていた（2026-11-11 のレビュー）。
    # `can_record` は `evaluate` と同じ検査を、例外にせず返す。
    rec_ok, rec_why = can_record(cond, ld, "test")
    if not rec_ok and not (a.dry or a.no_eval):
        print(f"  ※ テストの値は**記録しません**（測るだけ）: "
              f"{rec_why.splitlines()[0]}", flush=True)

    model = build_model_for(cond)
    n = sum(p.numel() for p in model.parameters())
    print(f"入力 {cond.in_channels} / パラメータ {n / 1e6:.2f}M", flush=True)
    if len(cond.in_channels) >= 3:
        assert model.geo_enc1[0].in_channels == cond.in_channels[2], \
            "3本目の符号化器の入口が config と違う"

    kw = {}
    if a.dry:
        d = paths.cache_dir("dry") / cond.name
        d.mkdir(parents=True, exist_ok=True)
        kw = {"weights": d / "best_model.pth", "curve": d / "curve.csv"}
        print(f"【試運転】書き先を {d} にする（本物は上書きしない）", flush=True)

    t0 = time.time()
    # `--dry` は書き先が cache/dry なので、本物の有無に関わらず回せる。
    hist = fit(cond, ld, hooks={"model": model, "seeded": True},
               epochs=a.epochs, overwrite=(a.overwrite or a.dry), **kw)
    print(f"\n学習に {(time.time() - t0) / 60:.1f} 分", flush=True)

    wpath = kw.get("weights", cond.weights)
    print("\n" + "=" * 78)
    print(ckpt.describe(wpath))
    print("=" * 78, flush=True)

    if a.dry and not a.no_eval:
        print("【試運転】評価は飛ばします（本物の重みを読んでしまうため）")
    elif not a.no_eval:
        # **model= を渡さない。** ディスクに保存した最良の重みから読ませる。
        # fit は最良を model に戻すようになったが、二重に守る
        # （以前ここで最終エポックのモデルを測っていた）。
        #
        # **記録してよいかは `dc5lib/evaluate.py` が1か所で持つ**
        # （再々レビュー 5 ── ここに規則を書き写していた）。
        # record=True のまま呼ぶと**1000エポック回した直後に**落ちるので、
        # 学習の前に `can_record` で確かめた結果（`rec_ok`）を使う。
        if not rec_ok:
            print("  ※ 記録できないので測るだけにします（理由は学習の前に"
                  "出したとおり）", flush=True)
        evaluate(cond, ld, "test", record=rec_ok)
    print(f"\n曲線: {hist.curve}")
    print(f"重み: {hist.weights}")
    if not a.dry:
        print("\n**config の protocol を v3-base に直すこと**"
              "（docs/ノートブックの作り方.md の 2.1）")


if __name__ == "__main__":
    main()
