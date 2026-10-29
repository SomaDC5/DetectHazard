# -*- coding: utf-8 -*-
"""コミット前の検査。異常があれば終了コード 1。

  - 実験フォルダ名が大文字小文字だけ違っていないか（Ubuntu と Windows で壊れる）
  - config.yaml があるか、フォルダ名と name が一致しているか
  - 重みがリポジトリ内に紛れ込んでいないか
  - metrics.csv が evals/ と食い違っていないか
  - ensembles.yaml のメンバーが実在し、bg_ratio が揃っているか
  - 学習ノートブックが config.yaml を無視して直書きしていないか
"""
import json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import registry, results, paths, ensembles, postprocess  # noqa: E402

problems = registry.check() + ensembles.check() + postprocess.check()

# 学習ノートブックの直書き。config を変えても効かないまま学習してしまうため
# （FinalFusion_SAM_APM_OverlapTrain で早期終了が効かず 1000 エポック回った）。
# すでに学習を終えた条件は、当時の記録として残すので対象外にする。
_HARDCODE = [
    (r"^num_epochs\s*=\s*\d+", "エポック数が直書き"),
    (r"^EARLY_STOP_PATIENCE\s*=\s*\d+", "早期終了が直書き"),
    (r"lr=1e-4", "学習率が直書き"),
    (r"^model\s*=\s*\w+\(\s*out_channels=1", "モデルのクラス名が直書き"),
    # 実損失は build_loss から来るが、使われないクラス定義が残っていると
    # 既定値（alpha=0.8/beta=0.2）が config（0.7/0.3）と違うので直書きに見える
    (r"class FocalTverskyLoss", "未使用の損失クラスが残っている（読むと直書きに見える）"),
]
def _dc5lib_exports():
    """dc5lib が公開している 名前 -> それを持つモジュールの集合。

    describe や get のように複数のモジュールにある名前があるので、
    どれか1つから import していれば良しとする。
    """
    import importlib, inspect, pkgutil
    import dc5lib
    owner = {}
    for m in pkgutil.iter_modules(dc5lib.__path__):
        try:
            mod = importlib.import_module(f"dc5lib.{m.name}")
        except Exception:
            continue
        for n, v in vars(mod).items():
            if n.startswith("_") or not (inspect.isfunction(v) or inspect.isclass(v)):
                continue
            if getattr(v, "__module__", "").startswith("dc5lib"):
                owner.setdefault(n, set()).add(f"dc5lib.{m.name}")
    return owner


_EXPORTS = _dc5lib_exports()


def _bound_names(src):
    """import で**実際に束縛される名前**の集合。

    正規表現で「その行に名前の文字列があるか」を見てはいけない。
    `from dc5lib.registry import get as _get_init` は `get` を定義しないのに
    文字列としては含むので、`get` が import 済みだと誤認する。
    **実際にこれで NameError を見逃した**（SegFormer_SAM_APM）。
    """
    names = set()
    for ln in src.splitlines():
        t = ln.strip()
        if t.startswith("from ") and " import " in t:
            tail = t.split(" import ", 1)[1]
        elif t.startswith("import "):
            tail = t[len("import "):]
        else:
            continue
        tail = tail.split("#", 1)[0].strip().strip("()")
        for part in tail.split(","):
            part = part.strip()
            if not part:
                continue
            if " as " in part:
                names.add(part.split(" as ", 1)[1].strip())   # 別名だけ束縛
            else:
                names.add(part.split(".", 1)[0].strip())
    return names


def _missing_imports(src):
    """呼び出しているのに import も定義もしていない dc5lib の名前。"""
    bound = _bound_names(src)
    out = []
    for name, mods in _EXPORTS.items():
        if not re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", src):
            continue
        if re.search(rf"^\s*(def|class)\s+{re.escape(name)}\b", src, re.M):
            continue
        if name in bound:
            continue
        # モジュールごと import して `registry.get(...)` と呼ぶ形も許す
        if any(m.split(".")[-1] in bound or m in bound for m in mods):
            continue
        out.append(name)
    return out


_TRAINED = {p.parent.name for p in (paths.results_dir() / "curves").glob("*.csv")}
for d in sorted(Path(paths.experiments_dir()).iterdir()):
    nb = d / "Keisya_airPhoto_final6.ipynb"
    if not nb.exists() or (paths.results_dir() / "curves" / f"{d.name}.csv").exists():
        continue        # 未作成、または学習済み（当時の記録として触らない）
    try:
        cells = json.load(open(nb, encoding="utf-8"))["cells"]
    except Exception:
        continue
    src = "\n".join("".join(c["source"]) for c in cells if c.get("cell_type") == "code")
    hits = [msg for pat, msg in _HARDCODE if re.search(pat, src, re.M)]
    if "train_cfg(CONDITION)" not in src:
        hits.append("config を読んでいない（dc5lib.traincfg を使う）")
    # 使っているのに import していない dc5lib の名前（順に実行すると NameError）
    for name in sorted(_missing_imports(src)):
        hits.append(f"{name} を import していない")
    # build_model を使っているのに、使われないモデルクラスが直書きで残っている。
    # **読んだ人が「このクラスを学習している」と誤解する。**
    # 元のノートブックを雛形にすると必ず残るので、気づけるようにする。
    if "build_model(get(CONDITION).arch)" in src:
        for cname in re.findall(r"^class\s+(\w+)\s*\(\s*nn\.Module", src, re.M):
            # super(クラス名, self) は定義の一部なので、使用とは数えない
            uses = len(re.findall(rf"(?<![\w.])(?<!class ){re.escape(cname)}\s*\(", src))
            supers = len(re.findall(rf"super\(\s*{re.escape(cname)}", src))
            if uses - supers <= 0:
                hits.append(f"使っていないモデルクラス {cname} が直書きで残っている")
    # NEEDS_WEIGHT を受け取るだけで使っていない。config を箇所正規化損失に
    # 変えたとき、重みマップを渡さないまま黙って別の損失で学習してしまう
    if "NEEDS_WEIGHT" in src and src.count("NEEDS_WEIGHT") < 2:
        hits.append("NEEDS_WEIGHT を受け取るだけで使っていない")
    if hits:
        problems.append(f"{d.name} のノートブック: {' / '.join(hits)}"
                        "（tools/wire_config.py で直せます）")

# metrics.csv が最新か
csv_path = paths.results_dir() / "metrics.csv"
if csv_path.exists():
    before = csv_path.read_bytes()
    results.build_metrics_csv()
    if csv_path.read_bytes() != before:
        problems.append("metrics.csv が evals/ と食い違っています（作り直しました）")

# 3x3 を貼る向きが、評価側と学習側とで揃っているか。
# **揃っていなかった。** run_building_miss.py の bld_mosaic が行に経度・
# 列に緯度を使っていたので、周囲8枚が90度回って貼られていた（2026-10-27 に直した）。
# 中央タイルだけは正しいので、`41` の `bld_in` は無事だが近接の数字はずれていた。
# 向きは3か所にある（run_overlap_eval.mosaic / plane_mosaic /
# dc5lib.overlap.NeighborMosaic.build）ので、ここで突き合わせる。
try:
    import numpy as _np
    from types import SimpleNamespace as _NS
    from analysis.run_overlap_eval import (mosaic as _mos, plane_mosaic as _pm,
                                           grid_index as _gidx, DLON, DLAT)
    from dc5lib.overlap import NeighborMosaic as _NM

    _T = 128
    _geo, _dem, _pl = [], [], []
    for _j in range(3):
        for _i in range(3):
            _geo.append((130.0 + _i * DLON, DLON / _T, 0.0,
                         34.0 + _j * DLAT, 0.0, -DLAT / _T))
            _dem.append(_np.full((_T, _T), 10 * _j + _i, _np.float32))
            _pl.append(_np.full((1, _T, _T), 10 * _j + _i, _np.uint8))
    _ds = _NS(GeoInfo=_geo, DEM=_dem,
              AirPhoto=[_np.zeros((_T, _T, 3), _np.uint8)] * 9)
    _gi, _gj, _cell = _gidx(_ds)
    _k = [t for t in range(9) if _gi[t] == 1 and _gj[t] == 1][0]

    def _grid(a):
        return [[int(round(float(a[r * _T + 64, c * _T + 64]))) for c in range(3)]
                for r in range(3)]

    _a = _grid(_mos(_ds, _cell, _gi, _gj, _k)[0])
    _b = _grid(_pm(_np.asarray(_pl), _cell, _gi, _gj, _k)[0])
    _nm = _NM(_geo, allowed=list(range(9)), tile=_T)
    _c = _grid(_nm.build(_k, {"x": _np.asarray(_pl)})["x"][:, :, 0])
    for _name, _got in (("plane_mosaic", _b), ("overlap.NeighborMosaic", _c)):
        if _got != _a:
            problems.append(
                f"3x3 を貼る向きが mosaic() と {_name} で違います。"
                f"mosaic={_a} / {_name}={_got}")
except Exception as _e:
    problems.append(f"3x3 の向きの検査ができませんでした: {_e}")

# git に大きなファイルが載っていないか
try:
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                           cwd=paths.REPO_ROOT).stdout.split()
    for f in files:
        p = paths.REPO_ROOT / f
        if p.exists() and p.stat().st_size > 20 * 1024 * 1024:
            problems.append(f"20MBを超えるファイルが追跡されています: {f} "
                            f"({p.stat().st_size/2**20:.0f}MB)")
except Exception:
    pass

# ----------------------------------------------------------------------
# config が実装に届いているか（`tools/audit_wiring.py` を呼ぶ）
#
# **「直書きの値が config と違う」は0件でなければならない。**
# config を読んで条件を理解した人が、実際に回った値を間違えるからである。
# 2026-10-29 にこれを0件にした（1件あったのは削除した条件のもの）。
# 新世代（protocol: v3-base）には、さらに
#   ・CFG = train_cfg(...) がある
#   ・値の直書きが無い（どの項目も「経由」）
# を要求する。**旧世代には掛けない。** 掛けると検査が常に赤くなり、
# 誰も見なくなる（`docs/設計図_修正案.md` の `5`）。
# ----------------------------------------------------------------------
try:
    sys.path.insert(0, str(paths.REPO_ROOT / "tools"))
    import audit_wiring as _aw

    _rows = [_aw.audit_one(d.name, d)
             for d in sorted(x for x in paths.experiments_dir().iterdir()
                             if (x / "config.yaml").exists())]
    for _r in _rows:
        for _k, _where, _what in _r.get("diff", []):
            problems.append(
                f"{_r['name']}: 直書きの値が config と違います"
                f"（{_k} {_where} {_what}）")
        _pr = getattr(_r.get("cond"), "protocol", "")
        if _pr.startswith("v3"):
            if any("CFG = train_cfg" in p for p in _r.get("problems", [])):
                problems.append(
                    f"{_r['name']}: protocol が v3-base なのに "
                    "CFG = train_cfg(...) がありません")
            _hard = [k for k, m in _r.get("marks", {}).items()
                     if m in ("直書", "両方")]
            if _hard:
                problems.append(
                    f"{_r['name']}: protocol が v3-base なのに値の直書きが"
                    f"あります（{', '.join(_hard)}）。"
                    "docs/ノートブックの作り方.md の 1 を参照")
    # アンサンブルに ensemble: false の条件が混ざっていないか
    _no = {_r["name"] for _r in _rows
           if getattr(_r.get("cond"), "ensemble", True) is False}
    if _no:
        from analysis.run_stage2_room import STAGE1, UNION3
        _used = {STAGE1, *UNION3}
        _bad = _used & _no
        if _bad:
            problems.append(
                "アンサンブルの既定の構成員に ensemble: false の条件が"
                f"入っています: {sorted(_bad)}")
except Exception as _e:                                   # noqa: BLE001
    problems.append(f"配線の検査ができませんでした: {_e}")

if problems:
    print("検査で問題が見つかりました:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("検査: 問題なし")
