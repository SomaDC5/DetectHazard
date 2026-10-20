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
    anchor = "model = build_model(get(CONDITION).arch)"
    if anchor not in src:
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
    return src.replace(anchor, block + anchor, 1), ["乱数の固定を追加"]


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
        # 乱数の固定（モデルの初期値と shuffle）
        if "build_model(get(CONDITION).arch)" in new:
            fixed, done = _add_seed(new)
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
        out = wire(os.path.join(str(root), n), check_only=args.check)
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
