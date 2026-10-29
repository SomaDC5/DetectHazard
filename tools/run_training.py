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
from dc5lib.evaluate import evaluate                         # noqa: E402
from dc5lib.models import build_model_for                    # noqa: E402
from dc5lib.pipeline import build_loaders, seed_everything   # noqa: E402
from dc5lib.registry import get                              # noqa: E402
from dc5lib.train import fit                                 # noqa: E402
from dc5lib.traincfg import train_cfg                        # noqa: E402

# 広島・random・random_state=42 のときに合うべき値。
# **違ったら止める。** 違うのは pkl か分割の指定か前処理で、
# どれも他の条件と比べられなくする（docs/ノートブックの作り方.md の 2.3）。
EXPECT = {
    "Hiroshima": {"n_used": 14816, "n_train": 8889, "n_val": 2963,
                  "n_test": 2964, "tiles_fingerprint": "a75ad23bad3e",
                  "fingerprint": "653b8cbe3d28"},
}


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
    a = ap.parse_args()

    cond = get(a.condition)
    cfg = train_cfg(cond)
    print("=" * 78)
    print(cond.display())
    print(f"  protocol {cond.protocol} / ensemble {cond.ensemble}"
          f" / seeded {cond.seeded}")
    print(cfg.describe())
    print("=" * 78, flush=True)

    seed_everything(cfg.seed)
    print(f"乱数を固定: {cfg.seed}", flush=True)

    ld = build_loaders(cond, a.region)

    want = EXPECT.get(a.region) if not a.no_expect else None
    if want:
        got = {"n_used": ld.n_used,
               **{k: ld.split[k] for k in
                  ("n_train", "n_val", "n_test", "tiles_fingerprint",
                   "fingerprint")}}
        bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
        if bad:
            for k, (g, w) in bad.items():
                print(f"  **違う** {k}: {g}（既存条件は {w}）")
            sys.exit("分割が既存条件と違います。pkl か分割の指定か前処理が違う。"
                     "意図した違いなら --no-expect を付けてください")
        print("分割の照合: 全項目一致", flush=True)

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
    hist = fit(cond, ld, hooks={"model": model, "seeded": True},
               epochs=a.epochs, **kw)
    print(f"\n学習に {(time.time() - t0) / 60:.1f} 分", flush=True)

    wpath = kw.get("weights", cond.weights)
    print("\n" + "=" * 78)
    print(ckpt.describe(wpath))
    print("=" * 78, flush=True)

    if not a.no_eval:
        evaluate(cond, ld, "test", model=model)
    print(f"\n曲線: {hist.curve}")
    print(f"重み: {hist.weights}")
    if not a.dry:
        print("\n**config の protocol を v3-base に直すこと**"
              "（docs/ノートブックの作り方.md の 2.1）")


if __name__ == "__main__":
    main()
