# -*- coding: utf-8 -*-
"""**config.yaml の各項目が、ノートブックのどこに実際に届いているか**を洗う。

なぜ要るか
----------
`tools/check.py` は「import が足りているか」「3x3 の向き」「巨大ファイル」など
**壊れていたら落ちるもの**を見る。`tools/wire_config.py --check` は
**自分が直せるもの**だけを見る。どちらも

    config.yaml に書いてある値が、学習で本当に使われているか

を通しで見ていない。`ゆらぎ幅.md` の `8` で分かったとおり、`train.seed` は
56本のうち44本で**どこにも届いていなかった**のに、config には書いてあった。
**config に項目があることは、それが効いていることを意味しない。**

項目1つずつについて次を区別する。

    経由   CFG.<項目> か get(CONDITION) を通って使われている
    両方   **いちばん危ない。** 経由もあるが古い直書きも残っている
    直書   値が直接書かれている（config を変えても効かない）
    無い   どちらも見つからない

偽陽性を避けるためにしていること
--------------------------------
- **コメントを落としてから照合する。** 文字列の中の `#` は残す
- **クラス定義を直書きに数えない。** `class FocalTverskyLoss(nn.Module)` は
  定義であって使用ではない。直書きと見るのは `criterion = XxxLoss(...)` の形だけ
- **字下げした行を本体と区別する。** `if __name__ == "__main__":` の中の
  `    model = MultiEncoderUNet(...)` は実行されない見本である。
  本体の判定は**行頭（字下げなし）の代入**だけで行い、字下げのものは
  `--show` で見えるようにしておく
- `--show` で**当たった行をそのまま出す。** 数字は必ず行で裏を取る

確かめられないこと（**正直に「未検査」と出す**）
------------------------------------------------
- **pkl の中に面が本当にあるか。** pkl は 5〜8GB で、開かないと属性名が
  分からない。ここは作る側（`tools/build_*_pkl.py`）と読む側（config の
  `field`）の**静的な突き合わせ**に留める
- **過去の学習がその値で回ったか。** `best_model.pth` には epoch と
  best_f1 しか入っていない。**記録が無いものは「分からない」と書く**

    python tools/audit_wiring.py                         # 配線（速い）
    python tools/audit_wiring.py --show seed arch        # 当たった行を出す
    python tools/audit_wiring.py --forward               # + 実際に1回 forward
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from dc5lib import paths                      # noqa: E402
from dc5lib import registry                   # noqa: E402

# (項目, 経由の印, 直書きの印)。**どちらも行単位で当てる。**
CHECKS = [
    ("arch",
     r"^model\s*=\s*build_model(?:_for)?\s*\(",
     r"^model\s*=\s*(?!build_model)[A-Za-z_]\w*\s*\("),
    ("dataset",
     r"\.pkl_for\s*\(|dataset_path\s*\(",
     r"""['"](?:hiroshima|shimane)[a-z0-9_]*\.pkl['"]"""),
    ("weights",
     r"get\s*\(\s*CONDITION\s*\)\s*\.weights|weights_path\s*\(|weights_dir\s*\(",
     r"""['"][^'"]*best_model\.pth['"]"""),
    ("curve",
     r"curve_path\s*\(",
     r"""['"][^'"]*curves[/\\][^'"]*['"]"""),
    ("loss",
     r"build_loss\s*\(\s*CONDITION",
     r"^\s*criterion\s*=\s*[A-Z]\w*(?:Loss|Entropy)\s*\("),
    ("epochs", r"num_epochs\s*=\s*CFG\.epochs", r"num_epochs\s*=\s*\d+"),
    ("batch_size", r"batch_size\s*=\s*CFG\.batch_size",
     r"batch_size\s*=\s*\d+"),
    ("lr", r"lr\s*=\s*CFG\.lr", r"lr\s*=\s*[0-9][0-9.e+-]*"),
    ("seed", r"torch\.manual_seed\s*\(\s*CFG\.seed",
     r"torch\.manual_seed\s*\(\s*\d"),
    ("patience", r"CFG\.early_stop_patience", r"patience\s*=\s*\d+"),
    ("test_size", r"test_size\s*=\s*CFG\.test_size",
     r"test_size\s*=\s*0\.2(?![0-9])"),
    ("val_size", r"test_size\s*=\s*CFG\.val_size", r"test_size\s*=\s*0\.25"),
    ("random_state", r"random_state\s*=\s*CFG\.random_state",
     r"random_state\s*=\s*\d+"),
    ("bg_ratio", r"CFG\.bg_ratio",
     r"^\s*(?:bg_ratio|BG_RATIO|ratio)\s*=\s*0?\.\d+"),
    # オーバーラップ学習。config の overlap.max_shift が効いているか。
    # **0 か overlap: 自体が無ければ素のタイルで学習する**ので、
    # 書き忘れに気づけないと別の学習をしてしまう
    ("overlap", r"max_shift\s*=\s*CFG\.",
     r"max_shift\s*=\s*\d+"),
]
MARK = {(True, False): "経由", (True, True): "両方", (False, True): "直書",
        (False, False): "無い"}
# **どの項目を dc5lib の誰が持っているか。**
# 基盤ノートブック（v3）は値をノートブックに書かないので、`経由` でも `直書` でも
# なく「無い」になる。だが**それが正しい状態**である。呼んでいる入口が
# その項目を持っているなら `委譲` と出す。
OWNER = {
    "arch": ("build_model_for",),
    "dataset": ("build_loaders",),
    "weights": ("fit", "ckpt."),
    "curve": ("fit", "curve_path"),
    "loss": ("fit", "build_loss"),
    "epochs": ("fit",), "lr": ("fit",), "patience": ("fit",),
    "batch_size": ("build_loaders", "fit"),
    "seed": ("seed_everything",),
    "test_size": ("build_loaders",), "val_size": ("build_loaders",),
    "random_state": ("build_loaders",), "bg_ratio": ("build_loaders",),
    "overlap": ("build_loaders",),
}
# 作る側が書き込む面の名前。**読む側（config の field）と突き合わせる。**
PRODUCED = {
    "DEM": "dataset_build（傾斜量図 or 標高）",
    "AirPhoto": "dataset_build（航空写真 RGB）",
    "Mask": "dataset_build（A33 の指定区域）",
    "LandUse": "dataset_build（L03-b の区分番号）",
    "Geo": "dataset_build（地質）",
    "Building": "build_building_pkl.py（建物・無壁舎の2面）",
    "BuildingVB": "build_vbuilding_pkl.py（+ 仮想建物で3面）",
}


def strip_comments(src: str) -> str:
    """`#` から行末を落とす。**文字列の中の `#` は残す。**"""
    out = []
    for line in src.split("\n"):
        q, i = None, 0
        while i < len(line):
            c = line[i]
            if q:
                if c == "\\":
                    i += 1
                elif c == q:
                    q = None
            elif c in "'\"":
                q = c
            elif c == "#":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def notebooks(d: Path):
    return sorted(p for p in d.glob("*.ipynb") if not p.name.startswith("."))


def code_of(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if c.get("cell_type") == "code")


def hits(pat: str, lines) -> list:
    rx = re.compile(pat)
    return [(i + 1, l) for i, l in enumerate(lines) if rx.search(l)]


def audit_one(name: str, d: Path):
    r = {"name": name, "problems": [], "marks": {}, "ev": {}, "nb": None}
    try:
        cond = registry.get(name)
    except Exception as e:                    # noqa: BLE001
        r["problems"].append(f"config が読めません: {e}")
        return r
    r["cond"] = cond
    if cond.name != name:
        r["problems"].append(f"config の name が {cond.name}（フォルダは {name}）")

    nbs = notebooks(d)
    if not nbs:
        r["problems"].append("ノートブックがありません")
        return r
    names = {x.name for x in nbs}
    if len(nbs) > 1 and names != {"train.ipynb", "eval_shimane.ipynb"}:
        r["problems"].append(f"ノートブックが {len(nbs)} 個: "
                             + " / ".join(p.name for p in nbs))
    # **train.ipynb があればそれを見る。** 名前順だと eval_shimane が先に来る
    nb = next((x for x in nbs if x.name == "train.ipynb"), nbs[0])
    r["nb"] = nb.name
    lines = strip_comments(code_of(nb)).split("\n")

    m = re.search(r"""CONDITION\s*=\s*['"]([^'"]+)['"]""", "\n".join(lines))
    if not m:
        r["problems"].append("CONDITION がノートブックにありません")
    elif m.group(1) != name:
        r["problems"].append(
            f"CONDITION が {m.group(1)}（フォルダは {name}）"
            "**別条件の重みを上書きします**")

    if not re.search(r"CFG\s*=\s*train_cfg\s*\(", "\n".join(lines)):
        r["problems"].append("CFG = train_cfg(...) が無い（train: の項目はどれも届かない）")

    whole = "\n".join(lines)
    r["uses"] = sorted(u for u in
                       ("build_loaders", "fit", "seed_everything",
                        "build_model_for", "build_loss", "evaluate",
                        "curve_path", "ckpt.")
                       if u in whole)
    for key, wp, hp in CHECKS:
        w, h = hits(wp, lines), hits(hp, lines)
        m = MARK[(bool(w), bool(h))]
        if m == "無い" and any(o in r["uses"] for o in OWNER.get(key, ())):
            m = "委譲"
        r["marks"][key] = m
        r["ev"][key] = {"経由": w, "直書": h}

    # 字下げされた model = Xxx(...) は見本の可能性。本体と分けて控えておく
    r["ev"]["arch_indented"] = {
        "見本？": hits(r"^\s+model\s*=\s*(?!build_model)[A-Za-z_]\w*\s*\(", lines)}

    # 直書きの値が config の値と一致しているか
    r["same"], r["diff"] = value_check(cond, lines)
    for key, where, what in r["diff"]:
        r["problems"].append(f"直書きの値が config と違う: {key} {where} {what}")

    # 早期終了。**`dc5lib.train.fit` に任せているならノートブックに無くてよい。**
    if not re.search(r"epochs_since_best|early_stop|打ち切", whole) \
            and "fit" not in r["uses"]:
        r["problems"].append(
            "学習ループに早期終了が無く、dc5lib.train.fit も使っていない")

    for reg, fn in (cond.dataset or {}).items():
        try:
            p = paths.dataset_path(reg, fn)
        except Exception as e:                # noqa: BLE001
            r["problems"].append(f"dataset {reg} のパスが引けません: {e}")
            continue
        if not p.exists():
            r["problems"].append(f"dataset {reg}: {fn} がありません")

    for i in cond.inputs:
        if i.field and i.field not in PRODUCED:
            r["problems"].append(
                f"inputs の field「{i.field}」を作っている所が分かりません（未検査）")

    try:
        from dc5lib.models import MODEL_CLASSES, forward_arity
        if cond.arch not in MODEL_CLASSES:
            r["problems"].append(f"arch「{cond.arch}」が models.py に無い")
        else:
            want, got = forward_arity(cond.arch), len(cond.inputs)
            if got and want != got:
                r["problems"].append(
                    f"forward は {want} 入力だが inputs は {got} 個")
    except Exception as e:                    # noqa: BLE001
        r["problems"].append(f"forward の引数の数を調べられません: {e}")
    return r


# 直書きの値を取り出して config の値と比べる。
# **「配線されていない」より「配線されていない上に値が違う」のほうが重い。**
# config を読んで条件を理解した人が、実際に回った値を間違える。
# 許す値が複数あるものは tuple で持つ（test_size は 0.2 と 0.25 の両方が正しい）。
VALUES = [
    ("bg_ratio", r"^\s*bg_ratio\s*=\s*([0-9.]+)\s*$", "bg_ratio"),
    ("epochs", r"num_epochs\s*=\s*([0-9]+)", "epochs"),
    ("batch_size", r"batch_size\s*=\s*([0-9]+)", "batch_size"),
    ("lr", r"\blr\s*=\s*([0-9][0-9.eE+-]*)", "lr"),
    ("seed", r"torch\.manual_seed\s*\(\s*([0-9]+)", "seed"),
    ("patience", r"patience\s*=\s*([0-9]+)", "early_stop_patience"),
    ("split", r"test_size\s*=\s*(0\.[0-9]+)", ("test_size", "val_size")),
    ("random_state", r"random_state\s*=\s*([0-9]+)", "random_state"),
    ("overlap", r"max_shift\s*=\s*([0-9]+)", "overlap.max_shift"),
]


def value_check(cond, lines):
    """直書きの値と config の値を突き合わせる。(一致, 食い違い) を返す。"""
    from dc5lib.traincfg import train_cfg
    try:
        cfg = train_cfg(cond)
    except Exception as e:                    # noqa: BLE001
        return [], [("train_cfg", "—", f"読めません: {e}")]
    same, diff = [], []
    for key, pat, attr in VALUES:
        attrs = attr if isinstance(attr, tuple) else (attr,)
        want = [_resolve(cfg, cond, an) for an in attrs]
        rx = re.compile(pat)
        for ln, txt in [(i + 1, l) for i, l in enumerate(lines)]:
            m = rx.search(txt)
            if not m:
                continue
            got = m.group(1)
            ok = any(_num_eq(got, w) for w in want)
            rec = (key, f"L{ln}", f"{got} / config {'|'.join(map(str, want))}")
            (same if ok else diff).append(rec)
    return same, diff


def _resolve(cfg, cond, an):
    """`epochs` は cfg から、`overlap.max_shift` は cond の入れ子から。"""
    if "." in an:
        head, tail = an.split(".", 1)
        d = getattr(cond, head, None) or {}
        return d.get(tail) if isinstance(d, dict) else None
    v = getattr(cfg, an, None)
    return getattr(cond, an, None) if v is None else v


def _num_eq(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-12 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def forward_check(r):
    """宣言したチャネル数で**実際に1回 forward する**。"""
    import torch
    from dc5lib.models import build_model_for
    cond = r.get("cond")
    if cond is None:
        return "—"
    try:
        model = build_model_for(cond).eval()
        xs = [torch.zeros(1, c, 128, 128) for c in cond.in_channels]
        with torch.no_grad():
            out = model(*xs)
        o = out[0] if isinstance(out, (tuple, list)) else out
        shape = tuple(o.shape)
        del model, xs, out
        return "通る" if shape == (1, 1, 128, 128) else f"形が違う {shape}"
    except Exception as e:                    # noqa: BLE001
        return f"落ちる: {type(e).__name__}: {str(e)[:70]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", nargs="*")
    ap.add_argument("--show", nargs="*", metavar="項目",
                    help="当たった行をそのまま出す（項目名。arch_indented も可）")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--only-problems", action="store_true")
    a = ap.parse_args()

    ex = paths.experiments_dir()
    dirs = sorted(d for d in ex.iterdir()
                  if d.is_dir() and (d / "config.yaml").exists())
    if a.condition:
        dirs = [d for d in dirs if d.name in set(a.condition)]
    rows = [audit_one(d.name, d) for d in dirs]
    keys = [k for k, _, _ in CHECKS]

    if a.show is not None:
        want = a.show or keys
        for r in rows:
            print("=" * 100)
            print(f"{r['name']}（{r['nb']}）")
            for k in want:
                ev = r["ev"].get(k, {})
                print(f"  [{k}] {r['marks'].get(k, '—')}")
                for kind, hs in ev.items():
                    for ln, txt in hs:
                        print(f"      {kind}  L{ln:<5} {txt.strip()[:92]}")
        return

    print("=" * 118)
    print(f"配線の監査 — {len(rows)} 条件")
    print("  経由=config を通る / 委譲=dc5lib に任せている（v3 の正解）"
          " / 両方=経由も直書きも在る（危険） / 直書=値が直接書かれている"
          " / 無い=どこにも無い")
    print("=" * 118)
    print(f"{'条件':<42}" + "".join(f"{k[:9]:>10}" for k in keys))
    for r in rows:
        if a.only_problems and not r["problems"]:
            continue
        cells = "".join(f"{r['marks'].get(k, '—'):>10}" for k in keys)
        print(f"{r['name'][:41]:<42}{cells}")

    print("\n" + "=" * 118)
    print("項目ごとの集計")
    print("=" * 118)
    print(f"{'項目':<16}{'経由':>8}{'委譲':>8}{'両方':>8}{'直書':>8}{'無い':>8}"
          "   config を変えて効くのは")
    for k in keys:
        c = {m: 0 for m in ("経由", "委譲", "両方", "直書", "無い")}
        for r in rows:
            c[r["marks"].get(k, "無い")] += 1
        ok = c["経由"] + c["委譲"]
        print(f"{k:<16}{c['経由']:>8}{c['委譲']:>8}{c['両方']:>8}{c['直書']:>8}"
              f"{c['無い']:>8}   {ok}/{len(rows)} 本")

    print("\n" + "=" * 118)
    print("見つかった食い違い")
    print("=" * 118)
    n = 0
    for r in rows:
        if not r["problems"]:
            continue
        n += len(r["problems"])
        print(f"\n{r['name']}（{r['nb']}）")
        for p in r["problems"]:
            print(f"    - {p}")
    print(f"\n食い違い {n} 件 / 問題のあった条件 "
          f"{sum(1 for r in rows if r['problems'])} 本 / 全 {len(rows)} 本")

    if a.forward:
        print("\n" + "=" * 118)
        print("宣言したチャネル数で実際に forward できるか（1 x C x 128 x 128）")
        print("=" * 118)
        for r in rows:
            print(f"{r['name'][:52]:<54}{forward_check(r)}")


if __name__ == "__main__":
    main()
