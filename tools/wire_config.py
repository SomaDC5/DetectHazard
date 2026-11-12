# -*- coding: utf-8 -*-
"""ノートブックの直書きを config.yaml 参照に置き換える。

直書きしていると config を変えても効かない。実際
`FinalFusion_SAM_APM_OverlapTrain` は early_stop_patience がノートブックに
無く、検証F値が ep115 で頭打ちのまま 1000 エポック回り切った。

    python tools/wire_config.py <条件名> [<条件名> ...]
    python tools/wire_config.py --all --check     # 直書きが残っていないか見るだけ

置き換えるもの（いずれも dc5lib.traincfg から読む）

    num_epochs = 1000          -> CFG.epochs
    batch_size=32              -> batch_size=CFG.batch_size
    lr=1e-4                    -> lr=CFG.lr
    random.seed(42)            -> random.seed(CFG.seed)
    test_size=0.2 / 0.25       -> CFG.test_size / CFG.val_size
    random_state=42            -> CFG.random_state
    （早期終了が無ければ学習ループに足す）

評価セル（島根・テスト）の batch_size は学習設定と関係ないので触らない。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import paths  # noqa: E402

NB = "Keisya_airPhoto_final6.ipynb"

CFG_CELL = '''# ============================================================
# 学習の設定は config.yaml から読む（直書きしない）。
# 直書きしていると config を変えても効かず、気づかないまま
# 別の設定で学習してしまう。
# ============================================================
from dc5lib.traincfg import train_cfg

CFG = train_cfg(CONDITION)
print(CFG.describe())
'''

# (説明, 正規表現, 置換後)
RULES = [
    ("エポック数", r"^num_epochs\s*=\s*\d+\s*$", "num_epochs = CFG.epochs"),
    ("エポック数", r"^num_epochs\s*=\s*get\(CONDITION\)\.train\.get\([^)]*\)\s*$",
     "num_epochs = CFG.epochs"),
    ("早期終了", r"^EARLY_STOP_PATIENCE\s*=\s*get\(CONDITION\).*$",
     "EARLY_STOP_PATIENCE = CFG.early_stop_patience"),
    ("学習率", r"lr=1e-4", "lr=CFG.lr"),
    ("乱数種", r"random\.seed\(\d+\)", "random.seed(CFG.seed)"),
    # モデルのクラス名の直書き。config の arch を変えても効かなくなる。
    # 「サンプルの使用例」ブロックの中（インデントあり）は触らない
    ("モデル", r"^model\s*=\s*\w+\(\s*\n?\s*out_channels=1,?\s*\n?\s*\)",
     "model = build_model(get(CONDITION).arch)"),
    ("モデル", r"^model\s*=\s*\w+\(out_channels=1\)$",
     "model = build_model(get(CONDITION).arch)"),
]


def _fix_loss(src):
    """損失まわりを直す。戻り値は (新しい中身, 直した内容)。

    実損失は build_loss(CONDITION) で config から来ているが、使われない
    FocalTverskyLoss のクラス定義が残っており、しかもその既定値（alpha=0.8,
    beta=0.2）が config（0.7 / 0.3）と違う。読むと直書きに見えるので消す。

    あわせて NEEDS_WEIGHT を受け取るだけで使っていないのを塞ぐ。
    config を箇所正規化損失に変えると重みマップが要るが、このノートブックは
    渡さないので、黙って違う損失で学習してしまう。
    """
    done = []
    m = re.search(r"\nclass FocalTverskyLoss\(nn\.Module\):.*?\n(?=criterion,)",
                  src, re.S)
    if m and "= FocalTverskyLoss(" not in src:
        src = src[:m.start()] + "\n" + src[m.end():]
        done.append("未使用の FocalTverskyLoss を削除")

    old = "criterion, NEEDS_WEIGHT = build_loss(CONDITION)  # 損失は config.yaml の train.loss で決まる"
    if old not in src:
        old = "criterion, NEEDS_WEIGHT = build_loss(CONDITION)"
    if old in src and "NEEDS_WEIGHT を渡していません" not in src:
        src = src.replace(old, "criterion, NEEDS_WEIGHT = build_loss(CONDITION)"
                          "  # 損失は config.yaml の train.loss で決まる" + """
# このノートブックは重みマップを渡さない。config を箇所正規化損失に変えた場合、
# 黙って違う損失で学習してしまうので止める（オーバーラップ学習では窓をずらすため
# 重みマップも一緒に切り出す必要があり、まだ対応していない）
assert not NEEDS_WEIGHT, (
    "この損失は重みマップが必要ですが、このノートブックは NEEDS_WEIGHT を渡していません")""")
        done.append("NEEDS_WEIGHT の未使用を塞ぐ")
    return src, done


def _add_early_stop(src):
    """学習ループに早期終了が無ければ足す。戻り値は (新しい中身, 直した内容)。

    config に early_stop_patience があってもノートブック側にコードが無いと
    epochs の上限まで回り切る。`FinalFusion_SAM_APM_OverlapTrain` がこれで
    検証F値 ep115 頭打ちのまま1000エポック回った。
    ひな形は `FinalFusion_SAM_APM_FullSkipSup`（本人が指定した参考実装）。
    """
    if "EARLY_STOP_PATIENCE" in src:
        return src, []

    init_old = "best_f1_score = 0.0\n\nif os.path.exists(save_path):"
    save_old = ('        print(f"Best model saved at epoch {epoch+1} '
                'with Validation F1-score: {val_f1:.4f}")\n')
    if init_old not in src or save_old not in src:
        return src, []

    # break を置く場所。ループ末尾の print より後、del より前
    tail_anchor = '        print(f"Epoch [{epoch+1}/{num_epochs}] val f1: {val_f1:.4f}")\n'
    if tail_anchor not in src:
        return src, []

    init_new = '''best_f1_score = 0.0
epochs_since_best = 0                        # 早期終了用

# 早期終了（検証F値が最良のエポックからこれだけ更新されなければ打ち切り）。
# 0 なら無効。直書きせず config.yaml の train.early_stop_patience から読む
EARLY_STOP_PATIENCE = CFG.early_stop_patience
print("早期終了:", f"{EARLY_STOP_PATIENCE} エポック" if EARLY_STOP_PATIENCE else "なし")

if os.path.exists(save_path):'''
    src = src.replace(init_old, init_new)
    src = src.replace(save_old, save_old + "        epochs_since_best = 0\n"
                                "    else:\n"
                                "        epochs_since_best += 1\n")
    tail_new = tail_anchor + '''
    # 早期終了: 検証F値が一定期間更新されなければ打ち切る
    if EARLY_STOP_PATIENCE and epochs_since_best >= EARLY_STOP_PATIENCE:
        print(f"\\n検証F値が {epochs_since_best} エポック更新されていないので打ち切ります"
              f"（最良 {best_f1_score:.4f}）。epoch {epoch+1}/{num_epochs}")
        break
'''
    src = src.replace(tail_anchor, tail_new)
    return src, ["早期終了を学習ループに追加"]



# モデルを作る行。**閉じ括弧を含めない。**
#
# > もとは `"model = build_model(get(CONDITION).arch)"` という
# > **閉じ括弧まで含む1行の文字列**を錨にしていた。そのため
# >
# >     model = build_model(get(CONDITION).arch,      # 引数が続いて改行する
# >     model = build_model_for(get(CONDITION))        # 3入力の条件
# >
# > のどちらにも当たらず、**乱数の固定が黙って飛ばされていた**
# > （未学習の4条件と、BLD / BVB / LU のノートブックが該当）。
# > 戻り値が「直した内容なし」になるだけなので、気づけなかった。
MODEL_LINE = re.compile(r"^model\s*=\s*build_model(?:_for)?\(", re.M)


def _add_seed(src):
    """乱数を固定する。戻り値は (新しい中身, 直した内容)。

    **どのノートブックも torch.manual_seed を呼んでいなかった**（29件すべて）。
    CFG.seed は random.seed にしか渡っておらず、それは背景タイルの選択に
    使うだけなので、bg_ratio=0 の条件では**何も変えていなかった**。
    つまり

      ・モデルの初期値は毎回ちがう（PyTorch の既定は OS のエントロピー）
      ・DataLoader の shuffle も毎回ちがう

    状態で、既存の全条件は再現できない。複数シードの実験をするには、
    まずここを固定しないと「シードを変えた」ことにならない。

    なお cuDNN の非決定性は残るので、ビット単位で同じにはならない
    （deterministic=True は学習が遅くなるので入れない）。
    """
    if "torch.manual_seed" in src:
        return src, []
    m = MODEL_LINE.search(src)
    if m is None:
        return src, []
    block = """# ============================================================
# 乱数の固定。**これが無いと CFG.seed は何も変えない。**
# CFG.seed は random.seed にしか渡っておらず、それは背景タイルの選択に
# 使うだけなので、bg_ratio=0 ではモデルの初期値も shuffle も毎回ちがった。
# cuDNN の非決定性は残るのでビット単位では一致しない。
# ============================================================
import random as _random

torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)
np.random.seed(CFG.seed)
_random.seed(CFG.seed)
print("乱数を固定:", CFG.seed)

"""
    return (src[:m.start()] + block + src[m.start():],
            ["乱数の固定を追加"])


# `model = build_model(get(CONDITION).arch)` を入れるときに要る import。
# **モジュールごとに1行ずつ確かめる。** build_model だけ見て get を忘れ、
# 学習の直前に NameError: name 'get' is not defined で落ちたことがある。
REQUIRED_IMPORTS = (
    ("from dc5lib.models import build_model", "build_model"),
    ("from dc5lib.registry import get", "get"),
)


def _class_uses(name, whole):
    """クラスが**実際に使われている**回数。

    `Name(` の形だけを数える。`class Name(nn.Module)` は定義なので除き、
    `super(Name, self)` は `Name(` の形にならないので最初から当たらない。

    > **ここを間違えて自作のノートブックを壊した。**
    > 当初 `uses - supers` としたが、supers は uses に数えられていないので
    > 二重に引いていた。使われているクラスを「未使用」と誤判定して削除し、
    > 残った参照が NameError になった。
    """
    pat = rf"(?<![\w.])(?<!class ){re.escape(name)}\s*\("
    return len(re.findall(pat, whole))


def _drop_main_guard(src):
    """`if __name__ == "__main__":` のブロックを消す。

    **ノートブックでは __name__ は "__main__" なので、このブロックは実行される。**
    元のスクリプトの作者は「import したときは動かさない」つもりで書いているので、
    ノートブックに貼ると意図と逆に動く。
    実際に SegFormer_SAM_APM では、ここで雛形の MultiEncoderUNet が
    instance 化されていた（24行あとの build_model で上書きされるので
    学習するモデルは正しかったが、消し忘れた参照が残る原因になっていた）。
    """
    import ast as _ast

    if '__name__' not in src:
        return src, []
    lines = src.splitlines(keepends=True)
    out, i, done = [], 0, []
    while i < len(lines):
        if re.match(r'^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', lines[i]):
            j = i + 1
            while j < len(lines) and (not lines[j].strip()
                                      or lines[j][:1].isspace()):
                j += 1
            done.append('if __name__ == "__main__" のブロックを削除')
            i = j
            continue
        out.append(lines[i])
        i += 1
    if not done:
        return src, []
    cut = "".join(out)
    try:
        _ast.parse(cut)
    except SyntaxError:
        return src, []
    return cut, done


def _drop_unused_model_class(src, whole):
    """build_model を使うのに残っている、使われないモデルクラスを消す。

    元のノートブックを雛形にすると、モデルクラスの直書きがそのまま残る。
    実際に学習されるのは build_model が返すクラスなので、
    **残っていると「このクラスを学習している」と誤解する。**

    **判定はノート全体（whole）で行う。** クラスの定義と使用は別のセルに
    あるので、編集するセル（src）だけを見ると
      ・build_model の行が別のセルにあるので発火しない
      ・別のセルでの使用を見落として消してしまう
    のどちらかになる。

    安全のため、使用が1つでもあれば消さない。
    消したあとに ast で解釈できなければ元に戻す。
    """
    import ast as _ast

    if "build_model(get(CONDITION).arch)" not in whole:
        return src, []
    done = []
    for cname in re.findall(r"^class\s+(\w+)\s*\(\s*nn\.Module", src, re.M):
        if _class_uses(cname, whole) > 0:
            continue
        lines = src.splitlines(keepends=True)
        start = next((i for i, l in enumerate(lines)
                      if re.match(rf"^class\s+{re.escape(cname)}\s*\(", l)), None)
        if start is None:
            continue
        end = len(lines)
        for i in range(start + 1, len(lines)):
            if lines[i].strip() and not lines[i][:1].isspace():
                end = i
                break
        cut = "".join(lines[:start]) + "".join(lines[end:])
        try:
            _ast.parse(cut)
        except SyntaxError:
            continue
        src = cut
        done.append(f"未使用のモデルクラス {cname} を削除")
    return src, done


def _add_model_import(src):
    """build_model を使うセルに、足りない import を足す。

    `("モデル", ...)` の置き換えで `model = build_model(get(CONDITION).arch)`
    を入れるが、**import は別のセルにあることが多く、取りこぼす。**
    元のノートブックがモデルクラスを直書きしていた条件では
    import が無いまま残り、学習の直前に NameError で落ちる。

    **名前ごとに確かめる。** `from dc5lib.registry import get as _get_init`
    のような別名つきの局所 import は `get` を定義しないので、
    行の有無ではなく「その名前が import されているか」で判定する。
    """
    if "from dc5lib.losses import" not in src:
        return src, []
    need = []
    for line, name in REQUIRED_IMPORTS:
        # `import get` / `import a, get` / `import get as x` を見分ける
        got = False
        for ln in src.splitlines():
            t = ln.strip()
            if not t.startswith("from ") or " import " not in t:
                continue
            names = t.split(" import ", 1)[1]
            for part in names.split(","):
                part = part.strip()
                if part == name or part.startswith(name + " as "):
                    # `as` つきは別名を定義するので、素の名前は入らない
                    if part == name:
                        got = True
            if got:
                break
        if not got:
            need.append((line, name))
    if not need:
        return src, []

    out, done = [], False
    for ln in src.splitlines(keepends=True):
        if not done and ln.lstrip().startswith("from dc5lib.losses import"):
            nl = "\n" if ln.endswith("\n") else ""
            for line, _ in need:
                out.append(line + nl)
            done = True
        out.append(ln)
    if not done:
        return src, []
    return "".join(out), [f"{n} の import を追加" for _, n in need]


def _add_init_from(src):
    """CFG.init_from があれば、その条件の重みを読んで続きから学習する。

    なぜ必要か（展望_再現率をどう上げるか.md 18.3）
    ----------------------------------------------
    ロジット空間のヒンジは**すでに飽和したモデルに効く**性質の損失である。
    確率空間の損失が勾配を送れない領域（正解画素の 20.3% が z < -10）に、
    大きさに依存しない定数の勾配を送る。

    だから「ゼロから学習」より「飽和した重みから追加学習」のほうが
    機構を直接試せるし、エポック数が少なくて済む。
    ゼロからの条件は、これで効くと分かってから作る。

    **ゼロから学習する条件には影響しない**（init_from が None なら素通り）。
    挿す場所は `model.to(device)` の直後。それより前だと CPU 上の
    モデルに GPU の state_dict を読むことになる。
    """
    if "CFG.init_from" in src:
        return src, []
    anchor = "model.to(device)"
    if anchor not in src:
        return src, []
    block = [
        "",
        "# ============================================================",
        "# 追加学習。CFG.init_from があれば、その条件の重みから続ける。",
        "# None（既定）なら何もしないので、ゼロから学習する条件には影響しない。",
        "# ロジット空間の損失は飽和した重みに効くので、こちらで安く試せる。",
        "# ============================================================",
        "if CFG.init_from:",
        "    from dc5lib.models import load_weights as _load_init",
        "    from dc5lib.registry import get as _get_init",
        "    _src_cond = _get_init(CFG.init_from)",
        "    _meta = _load_init(model, _src_cond.weights, device=device)",
        '    print(f"追加学習: {CFG.init_from} の重みから続ける")',
        '    print("  重み:", _src_cond.weights)',
        "    if _meta:",
        '        print("  元の記録:", _meta)',
        "else:",
        '    print("ゼロから学習（init_from なし）")',
        "",
    ]
    out, done = [], False
    for ln in src.splitlines(keepends=True):
        out.append(ln)
        if not done and ln.strip() == anchor:
            nl = "\n" if ln.endswith("\n") else ""
            out.append("\n".join(block) + nl)
            done = True
    if not done:
        return src, []
    return "".join(out), ["追加学習(init_from)の読み込みを追加"]


def _is_train_split(line):
    """学習の分割セルか（評価セルの train_test_split は触らない）。"""
    return "dataset2" in line or "idx_all" in line or "idx_trainval" in line


def wire(cond_dir, check_only=False):
    p = os.path.join(cond_dir, NB)
    if not os.path.exists(p):
        return None
    nb = json.load(open(p, encoding="utf-8"))
    cells = nb["cells"]
    changes, remaining = [], []
    # ノート全体。セルをまたぐ判定（クラスの使用など）に使う
    whole_src = "\n".join("".join(c.get("source", [])) for c in cells
                           if c.get("cell_type") == "code")

    # 1) CFG セルを cell0 の直後に置く（無ければ）
    has_cfg = any("train_cfg(CONDITION)" in "".join(c["source"]) for c in cells)
    if not has_cfg and not check_only:
        cells.insert(1, {"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": CFG_CELL.splitlines(keepends=True)})
        changes.append("CFG セルを追加")

    for ci, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        new = src
        for label, pat, rep in RULES:
            if re.search(pat, new, re.M):
                if check_only:
                    remaining.append(f"c{ci} {label}")
                else:
                    new = re.sub(pat, rep, new, flags=re.M)
                    changes.append(f"c{ci} {label}")
        # 学習の分割だけ置き換える
        if _is_train_split(src):
            for pat, rep, label in (
                    (r"test_size=0\.2(?![0-9])", "test_size=CFG.test_size", "test_size"),
                    (r"test_size=0\.25", "test_size=CFG.val_size", "val_size"),
                    (r"random_state=42", "random_state=CFG.random_state", "random_state")):
                if re.search(pat, new):
                    if check_only:
                        remaining.append(f"c{ci} {label}")
                    else:
                        new = re.sub(pat, rep, new)
                        changes.append(f"c{ci} {label}")
        # 学習用 DataLoader の batch_size（train_loader の直前にあるもの）
        if "train_loader" in src and "DataLoader" in src:
            if re.search(r"batch_size=32", new):
                if check_only:
                    remaining.append(f"c{ci} batch_size")
                else:
                    new = re.sub(r"batch_size=32", "batch_size=CFG.batch_size", new)
                    changes.append(f"c{ci} batch_size")
        # 損失まわり（未使用クラスの削除と NEEDS_WEIGHT の塞ぎ）
        if "build_loss(CONDITION)" in new or "class FocalTverskyLoss" in new:
            fixed, done = _fix_loss(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # 乱数の固定（モデルの初期値と shuffle）。
        # **ここも錨と同じ規則で引く。** 文字列の決め打ちだと
        # 改行して続く書き方と build_model_for を取りこぼす
        if MODEL_LINE.search(new):
            fixed, done = _add_seed(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # ノートブックでは __name__ == "__main__" が真なので、
        # 元スクリプトの「お試し実行」ブロックが動いてしまう。先に消す
        if "__name__" in new:
            fixed, done = _drop_main_guard(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # 使われないモデルクラスの直書きを消す（紛らわしいだけで害がある）。
        # **判定はノート全体で行う。** 定義と使用は別のセルにある
        if "class" in new and "nn.Module" in new:
            fixed, done = _drop_unused_model_class(new, whole_src)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # build_model の import（元がモデル直書きだった条件で落ちる）
        if "from dc5lib.losses import" in new:
            fixed, done = _add_model_import(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # 追加学習（init_from）の重み読み込み
        if "build_model(get(CONDITION).arch)" in new:
            fixed, done = _add_init_from(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        # 学習ループに早期終了が無ければ足す
        if "num_epochs" in new and "best_f1_score = 0.0" in new:
            fixed, done = _add_early_stop(new)
            if done:
                if check_only:
                    remaining += [f"c{ci} {d}" for d in done]
                else:
                    new = fixed
                    changes += [f"c{ci} {d}" for d in done]
        if new != src and not check_only:
            c["source"] = new.splitlines(keepends=True)

    if not check_only and changes:
        json.dump(nb, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return remaining if check_only else changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true", help="直すのではなく残りを表示")
    ap.add_argument("--include-trained", action="store_true",
                    help="学習済みの条件も対象にする（既定は飛ばす）")
    args = ap.parse_args()

    root = paths.experiments_dir()
    names = args.conditions
    if args.all or not names:
        names = sorted(d.name for d in Path(root).iterdir()
                       if (d / NB).exists())

    # 学習済みは当時の記録なので既定では触らない
    trained = {p.stem for p in (paths.results_dir() / "curves").glob("*.csv")}
    for n in names:
        if n in trained and not args.include_trained:
            print(f"  {n}: 学習済みのため飛ばす（--include-trained で対象にできます）")
            continue
        # **変化が止まるまで繰り返す。** whole_src は wire() の入口で一度だけ
        # 作るので、あるセルを書き換えた影響（お試し実行ブロックを消した結果
        # クラスが未使用になる、など）は次の周で初めて見える。
        out = wire(os.path.join(str(root), n), check_only=args.check)
        if out and not args.check:
            for _ in range(2):
                more = wire(os.path.join(str(root), n), check_only=False)
                if not more:
                    break
                out = list(out) + list(more)
        if out is None:
            print(f"  {n}: ノートブック無し")
        elif out:
            head = "直書きが残る" if args.check else "置き換え"
            # 省略すると見落とす（学習済みを誤って書き換えたことがある）ので全部出す
            print(f"  {n}: {head} {len(out)} 件")
            for o in out:
                print(f"      - {o}")
        else:
            print(f"  {n}: OK")


if __name__ == "__main__":
    main()
