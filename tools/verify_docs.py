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


def _fake_loaders(n_all, n_used, n_test):
    """`_recordable` に渡す偽の loaders。

    **`split` は既定の乱数分割の形でなければならない** ──
    `_recordable` は `splits.is_standard` が偽なら空を返す
    （4回目のレビュー E。csv 分割の別の集合を記録させないため）。
    """
    import types

    from dc5lib.splits import STD
    return types.SimpleNamespace(
        n_all=n_all, n_used=n_used,
        split=dict(STD, kind="random", n_test=n_test))


def _recordable_hiroshima():
    from dc5lib.evaluate import _recordable
    return _recordable(_fake_loaders(38148, 14816, 2964), "hiroshima", 0.0)


def _recordable_shimane():
    from dc5lib.evaluate import _recordable
    return _recordable(_fake_loaders(24569, 9679, 1936), "shimane", 0.0)


def claims():
    """(見出し, 実際の値を返す関数, 実装書に書いてある値) の並び。"""
    from dc5lib import postprocess, registry as _registry
    from dc5lib import results, splits, union
    from dc5lib.evaluate import (can_record_split, check_training_region,
                                 evaluate)
    from dc5lib.instance_eval import BAND, BAND_AREA, band
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
         "weights=None, curve=None, verbose=True, overwrite=False)"),
        ("evaluate の署名", sg(evaluate),
         "(cond, loaders, split='test', model=None, device=None, "
         "forward=None, thr=0.5, instance=True, verbose=True, "
         "record=False, rebuild_csv=True)"),
        ("train_cfg の署名", sg(train_cfg), "(condition)"),
        ("splits.expected の署名", sg(splits.expected), "(region)"),
        ("splits.is_standard の署名", sg(splits.is_standard), "(sp)"),
        ("registry.split_declaration の署名",
         sg(_registry.split_declaration), "(cond)"),
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
        # 実装書 `8` / 設計書 `6` の「記録してよい枚数」の表。
        # **文書の表が機械で守られていなかった**（2026-11-10 の再々レビュー 1
        # ── 広島にも「全件」があると書いてあった）。
        ("記録してよい枚数（広島）", _recordable_hiroshima(),
         {"警戒のみ": 2964}),
        ("記録してよい枚数（島根）", _recordable_shimane(),
         {"全件": 24569, "警戒のみ": 9679}),
        ("splits.EXPECT の広島（実装書 4 の表）", splits.EXPECT["Hiroshima"],
         {"n_all": 38148, "n_used": 14816, "n_train": 8889, "n_val": 2963,
          "n_test": 2964, "tiles_fingerprint": "a75ad23bad3e",
          "fingerprint": "653b8cbe3d28"}),
        ("splits.EXPECT の島根（実装書 4 の表）", splits.EXPECT["Shimane"],
         {"n_all": 24569, "n_used": 9679, "n_train": 5807, "n_val": 1936,
          "n_test": 1936, "tiles_fingerprint": "ff11694a9f3f",
          "fingerprint": "2dd13cd65a6c"}),
        # **「claims() に足した」と書いて足していなかった**（4回目のレビュー C）
        ("can_record_split の署名", sg(can_record_split),
         "(region, split='test')"),
        ("compare_expected の署名", sg(splits.compare_expected),
         "(region, n_used, split)"),
        ("check_training_region の署名", sg(check_training_region),
         "(cond, region, n_tiles)"),
        ("instance_eval.BAND（実装書 4・5 の箇所F の帯）", BAND,
         {"hiroshima": 0.0071, "shimane": 0.0198}),
        ("instance_eval.BAND_AREA（同じ表の面積F の帯）", BAND_AREA,
         {"hiroshima": 0.0018, "shimane": 0.0118}),
        ("band の署名", sg(band), "(region, kind='instance')"),
        ("band（広島・面積F）", band("広島", "area"), 0.0018),
        ("can_record_split（広島・テスト）", can_record_split("Hiroshima")[0],
         True),
        ("can_record_split（島根・テスト）", can_record_split("Shimane")[0],
         False),
        ("can_record_split（EXPECT に無い地域）",
         can_record_split("Yamaguchi")[0], False),
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
        # 2026-11-11 のレビューで足した入口（実装書 3.2）
        ("can_record の署名", sg(_can_record()), "(cond, loaders, split='test')"),
        ("check_training_region_names の署名", sg(_ctrn()),
         "(names, region, n_tiles=None)"),
        ("splits.spec_is_standard の署名", sg(splits.spec_is_standard),
         "(cond)"),
        ("splits.fits_subset の署名", sg(splits.fits_subset),
         "(cond, subset)"),
        ("union.verify_names の署名", sg(union.verify_names),
         "(names, strict=True)"),
        ("主系統のしきい値と数え方（COV_GT, COV_PRED, MIN_SIZE, CONNECTIVITY）",
         _thresholds(), (0.5, 0.3, 10, 8)),
        ("build_loaders の datasets に all がある（素のタイルの全件）",
         "\"all\": dall" in inspect.getsource(build_loaders), True),
    ]


def _thresholds():
    from dc5lib import instance_eval as ie
    return (ie.COV_GT, ie.COV_PRED, ie.MIN_SIZE, ie.CONNECTIVITY)


def _can_record():
    from dc5lib.evaluate import can_record
    return can_record


def _ctrn():
    from dc5lib.evaluate import check_training_region_names
    return check_training_region_names


def doc_claims():
    """**実装書の本文に、そのまま書いてあるはずの文字列。** コードから作る。

    `claims()` が比べているのは「コード」と「**このファイルに写した値**」で、
    **実装書の本文は一度も読んでいなかった**（2026-11-11 のレビュー。
    `DOC` は在るかどうかを見ていただけ）。だから実装書だけが古くなっても
    通っていた ── 実際に署名が4つ、本文に無いか違う形で書かれていた。

    ここでは**コードの値から文字列を作り、本文にあるかを見る。**
    コードを変えて実装書を直し忘れても、実装書だけを書き換えても落ちる。

    返りは `(見出し, 探す文字列, 空白を無視するか)`。
    """
    from dc5lib import results, splits
    from dc5lib.instance_eval import BAND, BAND_AREA, PRIMARY

    out = []
    # 署名。実装書は文字列を `"` で書いている
    for label, got, _want in claims():
        if isinstance(got, str) and got.startswith("(") and "署名" in label:
            name = label.replace(" の署名", "").split(".")[-1]
            out.append((f"{label}（本文）", name + got.replace("'", '"'), False))
    # 分割の期待値（実装書 4 の表）。枚数は3桁区切り、指紋はそのまま
    for region, e in splits.EXPECT.items():
        for k, v in e.items():
            out.append((f"EXPECT[{region}][{k}]（本文）",
                        f"{v:,}" if isinstance(v, int) else str(v), False))
    for nm, tbl in (("BAND", BAND), ("BAND_AREA", BAND_AREA)):
        for region, v in tbl.items():
            out.append((f"{nm}[{region}]（本文）", f"{v:.4f}", False))
    out.append(("PRIMARY（本文）", PRIMARY, False))
    # metrics.csv の列（実装書 8）。本文は途中で改行しているので空白を無視する
    out.append(("metrics.csv の列の並び（本文）",
                ",".join(results.COLUMNS), True))
    return out


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
    # **実装書の本文そのものを読む**（`doc_claims`）
    import re
    doc = DOC.read_text(encoding="utf-8")
    doc_ns = re.sub(r"\s+", "", doc)
    drows = doc_claims()
    for label, token, nospace in drows:
        if (re.sub(r"\s+", "", token) in doc_ns) if nospace else (token in doc):
            ok += 1
        else:
            bad += 1
            print(f"  **本文に無い** {label}")
            print(f"      探した文字列   {token}")
    print(f"実装書の照合: 一致 {ok} / 食い違い {bad} / 全 {len(rows) + len(drows)}"
          f"（うち本文を読んで確かめたもの {len(drows)}）")
    if bad:
        sys.exit("**docs/実装書.md が古くなっています。** 文書を直してください"
                 "（署名の一覧は python tools/api_map.py）")
    print("docs/実装書.md はコードと合っています")


if __name__ == "__main__":
    main()
