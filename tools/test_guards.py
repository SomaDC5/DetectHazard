# -*- coding: utf-8 -*-
"""**歯止めが本当に止まるか**を確かめる。`tools/check.py` から自動で回る。

なぜ要るか
----------
設計書 `7` は歯止めの一覧を持っているが、**歯止めが効いているかを
機械で確かめたものは1つも無かった。** 2026-11-08 のレビューで、
文書が「止める」と書いているのに通ってしまう箇所が2つ見つかった
（`build_loaders` の assert は必ず通る・`run_training` の `EXPECT` が広島だけ）。

**歯止めを足したら、ここに1行足す。** 書いただけで効いていない歯止めは、
無いより悪い（あると思って油断する）。

**pkl も重みも読まない。** データを読む前に止まる歯止めだけを試すので、
SSD が無い機械でも回る（重みの有無を見る検査と `analysis/` の入口の検査は、
SSD が見えなければ飛ばす）。重みの素性は**読まずに差し替える** ──
本物の `_trained_region` は重みを丸ごと読むので、検査の最初に外してある。

    python tools/test_guards.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _tmpdir():
    """**検査の書き先。** 本物の重みや曲線に触れないための逃がし場所。"""
    import tempfile
    d = Path(tempfile.gettempdir()) / "dc5_test_guards"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _expect(problems, label, exc, fn, verbose):
    try:
        fn()
    except exc as e:
        if verbose:
            print(f"  止まった  {label}: {type(e).__name__}")
        return
    except Exception as e:                                # noqa: BLE001
        problems.append(f"{label}: {exc.__name__} ではなく "
                        f"{type(e).__name__}({e}) が出ました")
        return
    problems.append(f"{label}: **止まりませんでした**")


def _equal(problems, label, got, want, verbose):
    if got == want:
        if verbose:
            print(f"  一致      {label}: {got}")
    else:
        problems.append(f"{label}: {got}（期待 {want}）")


def run(verbose=False) -> list:
    """問題の一覧を返す。空なら全部通った。"""
    import dc5lib.evaluate as _ev
    from dc5lib.evaluate import (_check_recordable, _tileset_for,
                                 can_record_split, check_training_region,
                                 evaluate)
    from dc5lib.splits import compare_expected as _cmp
    from dc5lib.splits import expected as _ex
    _orig_trained = _ev._trained_region                   # noqa: SLF001
    from dc5lib.overlap import OverlapDataset
    from dc5lib import paths
    from dc5lib.registry import get, load_all, split_declaration
    from dc5lib.train import _guard_before_training, fit

    problems: list = []
    cond = get("FinalFusion_SAM_APM")

    # **重みを読まない。** `_trained_region` は本物だと重みを丸ごと読む
    # （`ckpt.meta_of` が `torch.load` する）。ここでは「素性が無い」ことに
    # しておく ── 重みの有無や世代で検査の結果が変わらないようにする
    # （2026-11-11 のレビュー。「pkl も重みも読まない」と書いて読んでいた）。
    def _untrained():
        _ev._trained_region = lambda _c: None             # noqa: SLF001
    _untrained()

    def _trained(region):
        """その地域で学習した重み**のふり**をする条件（素性だけを差し替える）。"""
        import dc5lib.evaluate as _ev
        c = types.SimpleNamespace(name=f"Fake_{region}", weights=Path("x.pth"))
        _ev._trained_region = lambda _c, _r=region: _r   # noqa: SLF001
        return c

    def _std(n_test):
        """既定の乱数分割の `split`（これでないと `_recordable` は空を返す）。"""
        from dc5lib.splits import STD
        return dict(STD, kind="random", n_test=n_test)


    # --- 記録はテストだけ（設計書 §6）
    _expect(problems, "evaluate(split='val', record=True)", ValueError,
            lambda: evaluate(cond, None, "val", record=True), verbose)

    # --- ラベルは枚数から選ぶ。表に無い枚数は記録しない（設計書 §6）
    #
    # **島根の2つの評価はどちらも記録できる**（主系統 24,569＝全件 /
    # 補助 gtonly 9,679＝警戒のみ）。記録できないのは `build_loaders` が
    # 内部で割るテスト 1,936枚で、あれはどちらの評価でもない。
    # **既定の乱数分割の split を渡すこと** ── `_recordable` は
    # `is_standard` でない分割に空を返す（4回目のレビュー E）
    ld = types.SimpleNamespace(n_all=24569, n_used=9679,
                               split=_std(1936))
    hi = types.SimpleNamespace(n_all=38148, n_used=14816,
                               split=_std(2964))
    for region, lo, cases in (
            ("shimane", ld, ((24569, "全件"), (9679, "警戒のみ"),
                             (1936, None), (5807, None))),
            ("hiroshima", hi, ((2964, "警戒のみ"), (14816, None),
                               (38148, None)))):
        for n_tiles, want in cases:
            _equal(problems, f"{region} {n_tiles:,}枚 の tileset",
                   _tileset_for(lo, region, 0.0, n_tiles), want, verbose)

    # --- **歯止めそのものを呼ぶ**（部品ではなく）。
    # `_tileset_for` だけ試していたので、`evaluate` から
    # `if record: _check_recordable(...)` の1行を消しても検査が通っていた
    # （2026-11-10 の再々レビュー 3）。偽の `loaders` で `evaluate` を呼ぶ
    # ── **pkl も重みも読まない**（推論の前に止まるので）。
    def fake(region, n_tiles, **kw):
        ds = types.SimpleNamespace(dataset=list(range(n_tiles)))
        return types.SimpleNamespace(region=region, test=ds, **kw)

    _expect(problems, "島根のテスト1,936枚で記録", ValueError,
            lambda: evaluate(cond, fake("Shimane", 1936, n_all=24569,
                                        n_used=9679, split=_std(1936)),
                             "test", record=True), verbose)
    _expect(problems, "広島の正解あり14,816枚で記録", ValueError,
            lambda: evaluate(cond, fake("Hiroshima", 14816, n_all=38148,
                                        n_used=14816, split=_std(2964)),
                             "test", record=True), verbose)
    # **既定の乱数分割でなければ記録しない**（4回目のレビュー E）
    _expect(problems, "csv 分割のテストで記録", ValueError,
            lambda: evaluate(cond, fake("Hiroshima", 3673, n_all=38148,
                                        n_used=14816,
                                        split={"kind": "csv",
                                               "n_test": 3673}),
                             "test", record=True), verbose)
    # **2つの入口の答えが一致するか**（4回目のレビュー B）
    for region, n_test in (("Hiroshima", 2964), ("Shimane", 1936)):
        lo = fake(region, n_test, n_all=_ex(region)["n_all"],
                  n_used=_ex(region)["n_used"], split=_std(n_test))
        by_table = _tileset_for(lo, region.lower(), 0.0, n_test) is not None
        _equal(problems, f"can_record_split と _recordable が一致（{region}）",
               can_record_split(region)[0], by_table, verbose)

    # **通る側も試す**（4回目のレビュー D）。歯止めが厳しすぎる向きに壊れると、
    # 気づくのは学習が終わったあとである。`_check_recordable` を直接呼ぶので
    # 推論には入らない。
    for label, lo, got in (
            ("広島 2,964枚は通る",
             fake("Hiroshima", 2964, n_all=38148, n_used=14816,
                  split=_std(2964)), 2964),
            ("島根 9,679枚は通る",
             fake("Shimane", 9679, n_all=24569, n_used=9679,
                  split=_std(1936)), 9679)):
        try:
            _check_recordable(cond, lo, "test")
            if verbose:
                print(f"  通った    {label}")
        except Exception as e:                            # noqa: BLE001
            problems.append(f"{label}: 止まってしまいました（{e}）")

    # --- 記録してよいかの規則は1か所（再々レビュー 5）
    for region, split, want in (("Hiroshima", "test", True),
                                ("Shimane", "test", False),
                                ("Hiroshima", "val", False)):
        _equal(problems, f"can_record_split({region}, {split})",
               can_record_split(region, split)[0], want, verbose)

    # --- 分割の照合（設計書 §7）。**`EXPECT` に鍵が増えても落ちない**
    # （5回目のレビュー 1 ── `n_all` を足したら `KeyError` で学習経路が死んだ。
    #  `resolve` の返りに無い鍵は見ないこと）。
    _sp_full = dict(_std(2964), n_train=8889, n_val=2963,
                    tiles_fingerprint="a75ad23bad3e",
                    fingerprint="653b8cbe3d28")
    _equal(problems, "compare_expected（広島・記録どおり）",
           _cmp("Hiroshima", 14816, _sp_full), {}, verbose)
    _equal(problems, "compare_expected（テストが違う）",
           _cmp("Hiroshima", 14816, dict(_sp_full, n_test=3673)),
           {"n_test": (3673, 2964)}, verbose)
    _equal(problems, "compare_expected（EXPECT に無い地域）",
           _cmp("Yamaguchi", 1, _sp_full), {}, verbose)
    # **`n_all` は `split` に無いので見ない**（ここが落ちたら回帰）
    _equal(problems, "compare_expected（n_all を見ない）",
           "n_all" in _cmp("Hiroshima", 14816, _sp_full), False, verbose)

    # --- 学習に使った地域で学習タイルを記録しない（設計書 §7）
    # 素性の無い重みでは**止めない**（「分からない」と言う作法）。
    try:
        check_training_region(cond, "hiroshima", 14816)
        if verbose:
            print("  通った    素性の無い重みでは止めない（分からないので）")
    except Exception as e:                                # noqa: BLE001
        problems.append(f"素性の無い重みで止まってしまいました: {e}")
    # 偽の条件で、学習した地域と測る地域が同じとき止まるか
    _expect(problems, "島根で学習した重みで 9,679枚を記録", ValueError,
            lambda: check_training_region(
                _trained("shimane"), "shimane", 9679), verbose)
    # テスト分割なら学習タイルを含まないので通る
    try:
        check_training_region(_trained("shimane"), "shimane", 1936)
        if verbose:
            print("  通った    テスト分割なら学習した地域でも止めない")
    except Exception as e:                                # noqa: BLE001
        problems.append(f"テスト分割で止まってしまいました: {e}")

    # --- 同じ歯止めを **構成員の全員**に、推論の前に掛ける入口
    # （2026-11-11 のレビュー。`analysis/` の記録の経路が呼ぶ）。
    # 枚数は `regions.yaml` の subset から決まる（島根＝全件・広島＝テスト）。
    from dc5lib.evaluate import check_training_region_names as _ctrn
    _expect(problems, "島根で学習した重みを島根の全件で記録（構成員）",
            ValueError, lambda: _ctrn([_trained("shimane")], "shimane"),
            verbose)
    try:
        _ctrn([_trained("hiroshima")], "hiroshima")
        if verbose:
            print("  通った    広島で学習した重みを広島のテストで記録（構成員）")
    except Exception as e:                                # noqa: BLE001
        problems.append(f"広島のテストで止まってしまいました: {e}")
    _untrained()

    # --- ずらした窓を「素のタイル」として記録しない（設計書 §6）。
    # オーバーラップ条件の `datasets["train"]` は乱数でずらした窓を返す。
    # `eval_shimane` のセル3がそれをつないで測っていた。
    from dc5lib.evaluate import _shifted_parts, can_record
    _moving = types.SimpleNamespace(deterministic=False, max_shift=64)
    _still = types.SimpleNamespace(deterministic=True, max_shift=0)
    _concat = types.SimpleNamespace(datasets=[_moving, _still, _still])
    _equal(problems, "ずらした窓を数える（つないだ中の1本）",
           _shifted_parts(_concat), 1, verbose)
    _equal(problems, "素のタイルは数えない（検証・テスト用）",
           _shifted_parts(_still), 0, verbose)
    _equal(problems, "素のタイルは数えない（オーバーラップ無しの学習用）",
           _shifted_parts(types.SimpleNamespace(deterministic=False,
                                                max_shift=0)), 0, verbose)
    _expect(problems, "ずらした窓の入った DataLoader で記録", ValueError,
            lambda: evaluate(cond, types.SimpleNamespace(
                region="Shimane", n_all=24569, n_used=9679, split=_std(1936),
                test=types.SimpleNamespace(dataset=_concat)),
                "test", record=True), verbose)

    # --- 記録できるかを**学習の前に**答える入口（`can_record`）。
    # `can_record_split` は地域しか見ないので、csv 分割は通ってしまう。
    _equal(problems, "can_record（広島 2,964枚）",
           can_record(cond, fake("Hiroshima", 2964, n_all=38148,
                                 n_used=14816, split=_std(2964)))[0],
           True, verbose)
    _equal(problems, "can_record（csv 分割のテスト）",
           can_record(cond, fake("Hiroshima", 3673, n_all=38148, n_used=14816,
                                 split={"kind": "csv", "n_test": 3673}))[0],
           False, verbose)
    _equal(problems, "can_record（split='val'）",
           can_record(cond, None, "val")[0], False, verbose)

    # --- **別の分割で学習した条件を、乱数分割のテストで測らない**（設計書 §3）。
    # `analysis/` の道は条件の `split:` を見ずに random/42 のテストを作る。
    from dc5lib.splits import fits_subset, spec_is_standard
    blk = get("FinalFusion_SAM_APM_BlockSplit")
    for c, subset, want in ((cond, "holdout", True), (blk, "holdout", False),
                            (cond, "blocktest", False), (blk, "blocktest", True),
                            (blk, "all", True), (cond, "gtonly", True)):
        _equal(problems, f"fits_subset({c.name[:30]}, {subset})",
               fits_subset(c, subset)[0], want, verbose)
    # 検証を取らない 8:2 の条件は、テストが既定と同じなので測ってよい
    _equal(problems, "spec_is_standard（testselect の条件）",
           spec_is_standard(get("Train_Hiroshima_Test_Shimane_SAM_APM")),
           True, verbose)
    _equal(problems, "spec_is_standard（train: の値が既定と違う）",
           spec_is_standard(types.SimpleNamespace(
               split={}, train={"test_size": 0.3})), False, verbose)
    # **歯止めそのものを呼ぶ。** 推論の入口2つが、pkl も重みも読む前に止まるか
    try:
        from analysis.run_overlap_eval import evaluate as _ov_eval
        from analysis.run_overlap_prune import infer as _ov_infer
    except Exception as e:                                # noqa: BLE001
        _ov_eval = _ov_infer = None
        if verbose:
            print(f"  飛ばす    analysis の入口の検査（import できない: "
                  f"{type(e).__name__}）")
    if _ov_eval is not None:
        _expect(problems, "run_overlap_eval がブロック分割の条件を広島で測る",
                ValueError, lambda: _ov_eval(blk, "hiroshima", None), verbose)
        _expect(problems, "run_overlap_prune.infer が同じものを測る",
                ValueError, lambda: _ov_infer(blk, "hiroshima", None), verbose)
        import contextlib
        import io

        from analysis import config as _acfg
        with contextlib.redirect_stdout(io.StringIO()):   # 「測らない」の表示
            _kept = sum(len(v) for v in _acfg.group_by_pkl(
                [_acfg.Condition(blk), _acfg.Condition(cond)],
                "hiroshima").values())
        _equal(problems, "group_by_pkl がブロック分割の条件を外す",
               _kept, 1, verbose)

    # --- 1入力の条件は pkl を読む前に止める（`OverlapDataset` は写真を必ず読む）
    one = [c for c in load_all(include_excluded=True) if len(c.inputs) < 2]
    if one:
        from dc5lib.pipeline import build_loaders
        _expect(problems, "1入力の条件で build_loaders", NotImplementedError,
                lambda: build_loaders(one[0], verbose=False), verbose)

    # --- `union.verify` を**名前の並びで**呼ぶ入口。スクリプトはここを通る。
    # 重みは読まない（素性を「無い」ことにする）。重みの有無は SSD が要る。
    from dc5lib import union as _un
    if paths.data_root(required=False) is not None:
        _real_ckpt = _un._ckpt                            # noqa: SLF001
        _un._ckpt = types.SimpleNamespace(meta_of=lambda _p: None)  # noqa: SLF001
        try:
            _expect(problems, "ensemble: false の条件を混ぜる", AssertionError,
                    lambda: _un.verify_names(
                        ["FinalFusion_SAM_APM_OverlapTrain", blk.name]),
                    verbose)
        finally:
            _un._ckpt = _real_ckpt                        # noqa: SLF001
    elif verbose:
        print("  飛ばす    verify_names の検査（SSD が見えない）")

    # --- config の `train:` に、誰も読まない鍵を書かない（設計書 §7）
    import dc5lib.registry as _reg
    _equal(problems, "train: の鍵（いまの config）",
           _reg._check_train_keys(), [], verbose)          # noqa: SLF001
    _real_load = _reg.load_all
    _reg.load_all = lambda include_excluded=False: [types.SimpleNamespace(
        name="X", train={"batchsize": 16, "epochs": 1})]
    try:
        _equal(problems, "train: の打ち間違いを見つける",
               len(_reg._check_train_keys()), 1, verbose)  # noqa: SLF001
    finally:
        _reg.load_all = _real_load

    # --- `run_eval` が loaders を使い回してよい条件（同じ pkl でも欄が違えば別）
    import dataclasses

    from dc5lib.registry import Input
    from tools.run_eval import split_key as _sk
    _other = dataclasses.replace(cond, inputs=[
        Input(key="terrain", source="SAM", channels=1, field="DEM_other"),
        Input(key="airphoto", source="APM", channels=3)])
    _equal(problems, "split_key（読む欄が違えば別の鍵）",
           _sk(cond, "Hiroshima") != _sk(_other, "Hiroshima"), True, verbose)
    _equal(problems, "split_key（同じ条件は同じ鍵）",
           _sk(cond, "Hiroshima") == _sk(get(cond.name), "Hiroshima"),
           True, verbose)

    # --- `audit_splits` の既定は**全定義の構成員**（union5 だけではない）
    from tools import audit_splits as _as
    _missing = [m for n in _un.names() for m in _un.get(n).all
                if m not in _as.DEFAULT]
    _equal(problems, "audit_splits の既定から漏れた構成員", _missing, [],
           verbose)

    # --- 主系統のしきい値の写しが、原本と同じ値か（2026-11-12 のレビュー）。
    # `COV_GT = 0.5` などの写しが `analysis/` に9本ある。値は同じだが、
    # `PRIMARY` を変えたときに写しだけ古いまま残ると判定が静かに食い違う。
    # **写しを消して回るより、食い違ったら落ちるようにする**（帯と同じ作法）。
    import re as _re

    from dc5lib import instance_eval as _ie
    _orig = {"COV_GT": _ie.COV_GT, "COV_PRED": _ie.COV_PRED,
             "MIN_SIZE": _ie.MIN_SIZE}
    _pat = _re.compile(r"^(COV_GT|COV_PRED|MIN_SIZE)\s*=\s*([0-9.]+)\s*(#.*)?$",
                       _re.M)
    _off = []
    for _f in sorted((Path(__file__).resolve().parent.parent
                      / "analysis").glob("*.py")):
        for _k, _v, _c in _pat.findall(_f.read_text(encoding="utf-8")):
            if float(_v) != float(_orig[_k]):
                _off.append(f"{_f.name}: {_k} = {_v}")
    _equal(problems, "主系統のしきい値の写しで、原本と違うもの", _off, [],
           verbose)

    # --- 2段目と和集合の入口（2026-11-12 のレビュー）。import に SSD が要る
    try:
        from analysis import run_stage2_eval as _s2e
        from analysis import run_stage2_train as _s2t
        from analysis import run_union_blind as _ub
    except Exception as e:                                # noqa: BLE001
        _s2e = None
        if verbose:
            print(f"  飛ばす    2段目と和集合の入口の検査（import できない: "
                  f"{type(e).__name__}）")
    if _s2e is not None:
        # 記録するのはテストだけ。`--split val --record` を通さない
        _expect(problems, "run_stage2_eval --split val --record", ValueError,
                lambda: _s2e.guard_record_split(True, "val"), verbose)
        for _rec, _sp in ((True, "test"), (False, "val")):
            try:
                _s2e.guard_record_split(_rec, _sp)
            except Exception as e:                        # noqa: BLE001
                problems.append(f"guard_record_split({_rec}, {_sp}) が"
                                f"止まってしまいました: {e}")
        # 別の1段目・別の相方で学習した2段目を当てない
        _expect(problems, "2段目の重みの1段目が違う", ValueError,
                lambda: _s2e.guard_stage2_weights({"stage1": "A"}, "B", []),
                verbose)
        _expect(problems, "2段目の重みの相方が違う", ValueError,
                lambda: _s2e.guard_stage2_weights(
                    {"stage1": "A", "union": ["X", "Y"]}, "A", ["X"]), verbose)
        for _ck, _u, _lb in (({"stage1": "A"}, ["X"], "相方の記載が無い古い重み"),
                             ({"stage1": "A", "union": ["Y", "X"]}, ["X", "Y"],
                              "相方の順だけ違う"),
                             ({}, [], "何も書いていない重み")):
            try:
                _s2e.guard_stage2_weights(_ck, "A", _u)
                if verbose:
                    print(f"  通った    2段目の重み（{_lb}）")
            except Exception as e:                        # noqa: BLE001
                problems.append(f"2段目の重み（{_lb}）で止まってしまいました: {e}")
        # 既存の2段目の重みを黙って作り直さない
        from dc5lib import postprocess as _pp
        _have = [p.model for p in _pp.all_postprocess(include_excluded=True)
                 if p.model and (paths.data_root() / "weights" / p.model
                                 / "best_model.pth").exists()]
        if _have:
            _expect(problems, "既存の2段目の重みがあるのに学習",
                    FileExistsError,
                    lambda: _s2t.guard_existing_weights(_have[0]), verbose)
            try:
                _s2t.guard_existing_weights(_have[0], overwrite=True)
                _s2t.guard_existing_weights("__no_such_stage2_weights__")
            except Exception as e:                        # noqa: BLE001
                problems.append(f"2段目の重みの歯止めが広すぎます: {e}")
        elif verbose:
            print("  飛ばす    既存の2段目の重みの検査（重みがこの機械に無い）")
        # 印つきの和集合の表を、別の構成員で上書きしない
        _u3 = [_ub.short(n) for n in _un.get("union3").all]
        if _ub.csv_path("hiroshima", "union3").exists():
            _expect(problems, "印つきの表を別の構成員で上書き", SystemExit,
                    lambda: _ub.guard_overwrite(True, "hiroshima", "union3",
                                                _u3[:2]), verbose)
            try:
                _ub.guard_overwrite(True, "hiroshima", "union3", _u3)
                _ub.guard_overwrite(False, "hiroshima", "union3", _u3[:2])
            except BaseException as e:                    # noqa: BLE001
                problems.append(f"和集合の表の歯止めが広すぎます: {e}")
        elif verbose:
            print("  飛ばす    和集合の表の検査（表がこの機械に無い）")
        _equal(problems, "RECORDED5 は unions/union5.yaml から来る",
               _ub.RECORDED5, [_ub.short(n) for n in _un.get("union5").all],
               verbose)

    # --- `reset_condition` が、あとから増えた解析の表も拾うか。
    # **本物の場所は見ない** ── 偽のフォルダを作って `paths` を差し替える。
    import tempfile as _tf2

    from tools import reset_condition as _rc
    with _tf2.TemporaryDirectory() as _td:
        _d = Path(_td)
        for _sub, _fn in (("cache/analysis/tiles", "Cond__hiroshima.csv"),
                          ("cache/analysis/new_table", "Cond__hiroshima.csv"),
                          ("cache/analysis/new_table", "CondOther__x.csv"),
                          ("cache/dry/Cond", "best_model.pth"),
                          ("results/evals", "x.json")):
            (_d / _sub).mkdir(parents=True, exist_ok=True)
            (_d / _sub / _fn).write_text(
                '{"condition": "Cond"}', encoding="utf-8")
        _real_paths = _rc.paths
        _rc.paths = types.SimpleNamespace(
            data_root=lambda: _d,
            results_dir=lambda *a: _d.joinpath("results", *a),
            cache_dir=lambda *a: _d.joinpath("cache", *a))
        try:
            _got = sorted(str(p.relative_to(_d)).replace("\\", "/")
                          for _lb, p in _rc.targets("Cond"))
            _equal(problems, "reset_condition が拾うもの", _got,
                   ["cache/analysis/new_table/Cond__hiroshima.csv",
                    "cache/analysis/tiles/Cond__hiroshima.csv",
                    "cache/dry/Cond"], verbose)
            _equal(problems, "reset_condition が数える記録",
                   len(_rc.stale_records("Cond")), 1, verbose)
        finally:
            _rc.paths = _real_paths

    # --- 新しい条件に複製する雛形は、条件ごとのものだけ（`union.ipynb` を混ぜない）
    from tools import new_experiment as _ne
    _tpl = Path(__file__).resolve().parent.parent / "templates"
    _equal(problems, "条件に複製する雛形",
           [p.name for p in _ne.condition_templates(_tpl)],
           ["eval_shimane.ipynb", "train.ipynb"], verbose)

    # --- 素のタイルの集計は、土地利用以外の3入力目の条件を先に外す
    # （渡す経路が無いので、pkl を読んだあとで落ちて残りの条件も止まる）
    if _ov_eval is not None:
        _third = [c for c in load_all(include_excluded=True)
                  if len(c.inputs) >= 3 and c.inputs[2].source != "LU"]
        if _third:
            with contextlib.redirect_stdout(io.StringIO()):
                _kept = sum(len(v) for v in _acfg.group_by_pkl(
                    [_acfg.Condition(_third[0]), _acfg.Condition(cond)],
                    "shimane").values())
            _equal(problems, "group_by_pkl が建物の3入力目の条件を外す",
                   _kept, 1, verbose)
        # 部分集合の表の土台と候補は unions/*.yaml から来る（直書きしない）
        from analysis import run_union_subsets as _us
        _equal(problems, "run_union_subsets の土台は union3.yaml",
               _us.BASE, [_ub.short(n) for n in _un.get("union3").all]
               if _s2e is not None else _us.BASE, verbose)

    # --- config の `split:` の宣言の判定（設計書 §7）
    for name, want in (("FinalFusion_SAM_APM_BlockSplit", None),
                       ("Train_Hiroshima_Test_Shimane_SAM_APM",
                        "cannot_declare"),
                       ("FinalFusion_SAM_APM", None)):
        _equal(problems, f"split_declaration({name[:28]})",
               split_declaration(get(name))[0], want, verbose)
    # 宣言を外した偽の条件で `undeclared` が出るか
    _equal(problems, "split_declaration（BlockSplit で split: 空）",
           split_declaration(types.SimpleNamespace(
               name="X_BlockSplit", protocol="v1-blocksplit",
               split={}))[0], "undeclared", verbose)
    _equal(problems, "split_declaration（csv なのに名前が違う）",
           split_declaration(types.SimpleNamespace(
               name="X", protocol="v3-base",
               split={"kind": "csv"}))[0], "misnamed", verbose)

    # --- 既存の重み・曲線を黙って作り直さない
    #
    # **本物の書き先で `fit` を呼ばない。** `_guard_before_training` は
    # ファイルを開かないので、歯止めが（並べ替えなどで）効かなくなっても
    # 何も壊れない。`fit` は曲線を `"w"` で開いた瞬間に中身を消すので、
    # ここで呼ぶと `check.py` を回しただけで曲線が消える
    # （実際に 2026-11-08 に消した。docs/レビュー_2026-11-08.md の 6）。
    ol = get("FinalFusion_SAM_APM_OverlapTrain")
    cfg0 = types.SimpleNamespace(init_from=None)
    if ol.weights.exists():
        _expect(problems, "既存の重みがあるのに学習", FileExistsError,
                lambda: _guard_before_training(
                    ol, cfg0, None, ol.weights,
                    paths.curve_path(ol.name), False), verbose)
    elif verbose:
        print("  飛ばす    既存の重みの検査（重みがこの機械に無い）")

    # `fit` が本当にその歯止めを呼んでいるか。**書き先は一時フォルダ**なので、
    # 歯止めが効かなくても消えるのは空のダミーだけである。
    tmp = _tmpdir()
    for p in (tmp / "w.pth", tmp / "c.csv"):
        p.write_text("", encoding="utf-8")
    _expect(problems, "fit が既存の検査を呼んでいるか", FileExistsError,
            lambda: fit(cond, None, weights=tmp / "w.pth",
                        curve=tmp / "c.csv"), verbose)

    # --- config に書いても届かない項目（設計書 §8）
    _expect(problems, "train.init_from を書いて学習", NotImplementedError,
            lambda: _guard_before_training(
                cond, types.SimpleNamespace(init_from=cond.name),
                None, "x.pth", "x.csv", True), verbose)

    # 補助損失（深層監督）。`fit` は `return_aux=True` を渡さないので、
    # hooks が無ければ補助ヘッドは学習されない（2026-11-12 のレビュー）
    from dc5lib.losses import aux_lambda as _auxl
    _aux = [c for c in load_all(include_excluded=True) if _auxl(c) > 0]
    if _aux:
        _expect(problems, "aux_lambda > 0 で hooks なし", NotImplementedError,
                lambda: _guard_before_training(
                    _aux[0], types.SimpleNamespace(init_from=None),
                    None, "x.pth", "x.csv", True), verbose)
        try:
            _guard_before_training(
                _aux[0], types.SimpleNamespace(init_from=None), None,
                "x.pth", "x.csv", True,
                hooks={"forward": lambda m, xs: None,
                       "loss_fn": lambda o, y: None})
            # 補助損失の無い条件は、hooks が無くても通る
            _guard_before_training(
                cond, types.SimpleNamespace(init_from=None), None,
                "x.pth", "x.csv", True)
            if verbose:
                print("  通った    aux_lambda > 0 で forward と loss_fn あり"
                      " / 補助損失の無い条件")
        except Exception as e:                            # noqa: BLE001
            problems.append(f"補助損失の歯止めが広すぎます: {e}")

    aug = [c for c in load_all() if getattr(c, "augment", False)]
    if aug:
        # **`fit` を呼ぶときは必ず書き先を逃がす**（`_tmpdir`）。
        safe = {"weights": tmp / "x.pth", "curve": tmp / "x.csv"}
        fake = types.SimpleNamespace(
            datasets={"train": types.SimpleNamespace(transform=None)})
        _expect(problems, "augment: true で transform なし",
                NotImplementedError,
                lambda: fit(aug[0], fake, overwrite=True, **safe), verbose)
        # transform を渡してあれば歯止めは掛からない
        fake2 = types.SimpleNamespace(
            datasets={"train": types.SimpleNamespace(transform=object())})
        try:
            _guard_before_training(aug[0],
                                   types.SimpleNamespace(init_from=None),
                                   fake2, "x.pth", "x.csv", True)
            if verbose:
                print("  通った    augment: true で transform あり")
        except Exception as e:                            # noqa: BLE001
            problems.append(f"augment の歯止めが広すぎます: {e}")

    # --- 近傍に検証・テストを貼らない（設計書 §3）
    _expect(problems, "OverlapDataset(allowed=None)", ValueError,
            lambda: OverlapDataset([], None, [], [], indices=[],
                                   allowed=None), verbose)

    # --- 分割の期待値が両地域にある（レビュー 2.3）。
    # 原本は `dc5lib/splits.py` の `EXPECT` だけである（再レビュー 3）。
    # **`run_training.py` を import しない** ── `check.py` のプロセスの
    # `sys.argv` を書き換える必要があり、戻し忘れると後ろに効く（同 6）。
    from dc5lib import splits as rt
    for region in ("Hiroshima", "Shimane"):
        if region not in rt.EXPECT:
            problems.append(f"dc5lib/splits.py の EXPECT に {region} が無い")
        elif verbose:
            print(f"  ある      EXPECT[{region}]")

    # --- 帯の直書きが増えたら気づくか（設計書 §7）。
    # **検査そのものが効くことを確かめる** ── 偽のリポジトリを作って、
    # 見つける側と見逃す側の両方を見る。**本物のファイルは触らない。**
    import tempfile as _tf

    from tools import check_bands as _cb
    _equal(problems, "帯の直書き（いまのリポジトリは既知どおり）",
           _cb.scan(Path(__file__).resolve().parent.parent), [], verbose)
    with _tf.TemporaryDirectory() as _td:
        _d = Path(_td)
        for _sub in ("analysis", "dc5lib", "tools"):
            (_d / _sub).mkdir()
        _fake = _d / "analysis" / "run_fake.py"
        _fake.write_text("BAND = 0.0071\nB2 = 0.0118\n", encoding="utf-8")
        _equal(problems, "帯の直書きを数える", _cb.counts(_d),
               {"analysis/run_fake.py": 2}, verbose)
        # 既知に無い本が増えたら `scan` が落とすか（KNOWN を一時的に空にする）
        _known = dict(_cb.KNOWN)
        try:
            _cb.KNOWN.clear()
            _equal(problems, "帯の直書きが増えたら落ちる",
                   len(_cb.scan(_d)), 1, verbose)
        finally:
            _cb.KNOWN.clear()
            _cb.KNOWN.update(_known)
        # 後ろに数字が続くものは別の値なので数えない
        _fake.write_text("THR = 0.00710\nX = 10.0071\n", encoding="utf-8")
        _equal(problems, "帯に似た別の値は数えない", _cb.counts(_d), {}, verbose)
        # 原本と「測る側」は対象外（原本から読むと循環する）
        (_d / "dc5lib" / "instance_eval.py").write_text(
            "BAND = 0.0071\n", encoding="utf-8")
        (_d / "analysis" / "run_band.py").write_text(
            "BAND = 0.0198\n", encoding="utf-8")
        _equal(problems, "原本と run_band.py は対象外", _cb.counts(_d), {}, verbose)

    _ev._trained_region = _orig_trained               # noqa: SLF001
    return problems


def main():
    problems = run(verbose=True)
    print("")
    if problems:
        print("歯止めの検査で問題が見つかりました:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("歯止めの検査: 問題なし")


if __name__ == "__main__":
    main()
