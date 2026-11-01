# -*- coding: utf-8 -*-
"""`docs/実装書.md` に書いた署名と定数が、いまのコードと合っているかを確かめる。

なぜ要るか
----------
実装書は「**これだけ読めば実装できる**」という約束で書いてある。
だが手で写した署名はコードを直すと古くなり、**古い実装書は読まない方がまし**
という状態になる。だからここで機械的に照合する。

    python tools/verify_docs.py

食い違ったら**実装書を直す**（コードではなく文書が古い）。
署名を一覧で出し直すなら `python tools/api_map.py`。

ここに並べるのは**実装書が本文で名前を挙げているもの**だけである。
全関数を照合しない ── 実装書に書いていないものは照合する意味がない。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                       # noqa: BLE001
    pass

DOC = Path(__file__).resolve().parent.parent / "docs" / "実装書.md"


def claims():
    """(見出し, 実際の値を返す関数, 実装書に書いてある値) の並び。"""
    from dc5lib import postprocess, results, splits, union
    from dc5lib.evaluate import evaluate
    from dc5lib.instance_eval import PRIMARY
    from dc5lib.pipeline import build_loaders, seed_everything
    from dc5lib.registry import Condition
    from dc5lib.sync import KINDS
    from dc5lib.train import fit
    from dc5lib.traincfg import DEFAULTS, train_cfg

    def sg(fn):
        """**注釈を落とした署名。** 実装書は型注釈を写していない。

        返り値は別の列に書いてあり、引数の型は読む側に要らない。
        注釈を含めて比べると、注釈を足しただけで「古い」と出てしまう。
        """
        s = inspect.signature(fn)
        ps = [q.replace(annotation=inspect.Parameter.empty)
              for q in s.parameters.values()]
        return str(s.replace(parameters=ps,
                             return_annotation=inspect.Signature.empty))

    return [
        ("build_loaders の署名", sg(build_loaders),
         "(cond, region='Hiroshima', batch_size=None, transform=None, "
         "num_workers=0, verbose=True)"),
        ("seed_everything の署名", sg(seed_everything), "(seed)"),
        ("fit の署名", sg(fit),
         "(cond, loaders, device=None, hooks=None, epochs=None, "
         "weights=None, curve=None, verbose=True)"),
        ("evaluate の署名", sg(evaluate),
         "(cond, loaders, split='test', model=None, device=None, "
         "forward=None, thr=0.5, instance=True, verbose=True, "
         "record=False, rebuild_csv=True)"),
        ("train_cfg の署名", sg(train_cfg), "(condition)"),
        ("union.record の署名", sg(union.record),
         "(name, members, region, scores, n_blobs, floor, csv_name, "
         "setting=None)"),
        ("union.verify の署名", sg(union.verify), "(e, strict=True)"),
        ("results.record_eval の鍵になる引数",
         [p for p in inspect.signature(results.record_eval).parameters
          if p in ("condition", "region", "tileset", "scope",
                   "metric_kind", "setting")],
         ['condition', 'region', 'tileset', 'scope', 'metric_kind',
          'setting']),
        ("PRIMARY", PRIMARY, "被覆 (cov_gt=0.5, cov_pred=0.3)"),
        ("metrics.csv の末尾5列", results.COLUMNS[-5:],
         ['members', 'n_members', 'stage2_model', 'thr', 'thr_split']),
        ("traincfg の既定値",
         {k: DEFAULTS[k] for k in
          ("epochs", "batch_size", "lr", "seed", "early_stop_patience",
           "test_size", "val_size", "random_state")},
         {'epochs': 1000, 'batch_size': 32, 'lr': 0.0001, 'seed': 42,
          'early_stop_patience': 100, 'test_size': 0.2, 'val_size': 0.25,
          'random_state': 42}),
        ("sync が同期する領域", KINDS,
         ('weights', 'checkpoints', 'datasets', 'hazard_map')),
        ("Condition のフィールドに protocol/ensemble/seeded/split/overlap",
         all(f in Condition.__dataclass_fields__ for f in
             ("protocol", "ensemble", "seeded", "split", "overlap")), True),
        ("PostProcess.union_list は property",
         isinstance(postprocess.PostProcess.union_list, property), True),
        ("splits に fingerprint と tiles_fingerprint",
         all(hasattr(splits, f) for f in
             ("fingerprint", "tiles_fingerprint", "resolve")), True),
        ("tileset_label(hiroshima, 0)", results.tileset_label("hiroshima", 0.0),
         "警戒のみ"),
        ("tileset_label(shimane, 0)", results.tileset_label("shimane", 0.0),
         "全件"),
        ("tileset_label(hiroshima, 0.1)",
         results.tileset_label("hiroshima", 0.1), "警戒+背景"),
    ]


def union_claims():
    """`.all` は基準込み・`members` は基準なし という実装書の記述。"""
    from dc5lib import union
    out = []
    for name in union.names():
        e = union.get(name)
        out.append((f"{name}: members に基準が入っていない",
                    e.base in e.members, False))
        out.append((f"{name}: .all = base + members",
                    list(e.all), [e.base] + list(e.members)))
    return out


def main():
    if not DOC.exists():
        sys.exit(f"{DOC} がありません")
    rows = claims() + union_claims()
    ok = bad = 0
    for label, got, want in rows:
        if str(got) == str(want):
            ok += 1
        else:
            bad += 1
            print(f"  **食い違い** {label}")
            print(f"      実際     {got}")
            print(f"      実装書   {want}")
    print(f"実装書の照合: 一致 {ok} / 食い違い {bad} / 全 {len(rows)}")
    if bad:
        sys.exit("**docs/実装書.md が古くなっています。** 文書を直してください"
                 "（署名の一覧は python tools/api_map.py）")
    print("docs/実装書.md はコードと合っています")


if __name__ == "__main__":
    main()
