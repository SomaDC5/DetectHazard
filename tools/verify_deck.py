# -*- coding: utf-8 -*-
"""発表スライドに書いてある数値が、results/metrics.csv と合っているかを確かめる。

スライド（docs/deck/project/slides/*.html）は手で数値を埋め込んで作ってある。
あとから解析をやり直したり条件を足したりすると、スライドの数値だけが
古いまま残る。それを機械的に見つけるための検査。

**スライドのHTMLから数値を自動で抜き出して**、results/metrics.csv から
計算し直した値と突き合わせる。人が転記した一覧とは比べないので、
「転記ミスを転記ミスで検証する」ことにはならない。

    python tools/verify_deck.py              要約だけ
    python tools/verify_deck.py --detail     1項目ずつ全部表示
    python tools/verify_deck.py --slide res_fusion   特定のスライドだけ

判定
    OK    スライドの値と再計算値が許容差に収まっている
    NG    ずれている（スライドを直すか、解析をやり直す）
    不可  metrics.csv だけでは再計算できない（下の「検証できない項目」を参照）
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dc5lib import paths, registry  # noqa: E402

SLIDES = paths.REPO_ROOT / "docs" / "deck" / "project" / "slides"
TOL = 0.0006          # 小数3桁表示の丸めを吸収する許容差


# ----------------------------------------------------------------------
# metrics.csv から値を引く
# ----------------------------------------------------------------------
def load_metrics():
    p = paths.results_dir() / "metrics.csv"
    if not p.exists():
        raise SystemExit("results/metrics.csv がありません。"
                         "python tools/make_metrics.py で作ってください。")
    return pd.read_csv(p)


def f1(m, cond, region, tileset, scope="通常", kind="面積", setting=None):
    q = m[(m.condition == cond) & (m.region == region) & (m.tileset == tileset)
          & (m.scope == scope) & (m.metric_kind == kind)]
    if setting is not None:
        q = q[q.setting == setting]
    if len(q) != 1:
        return None
    return float(q.iloc[0].f1)


def col(m, cond, region, tileset, name, scope="通常", kind="面積"):
    q = m[(m.condition == cond) & (m.region == region) & (m.tileset == tileset)
          & (m.scope == scope) & (m.metric_kind == kind)]
    if len(q) != 1:
        return None
    v = q.iloc[0][name]
    return None if pd.isna(v) else float(v)


# スライドの表示ラベル → 条件名
def label_map():
    out = {}
    for c in registry.load_all(include_excluded=True):
        out[c.display()] = c.name
        out[c.display().replace(" ", "")] = c.name
    # スライド側は全角スペースや「拡張 + 」表記を使っている
    extra = {
        "拡張 + Final　DEM+APM": "DataOgument_FinalFusion_DEM_APM",
        "拡張 + Attention　SAM+APM": "DataOgument_AttentionUNet_SAM_APM",
        "拡張 + Final　SAM+APM": "DataOgument_FinalFusion_SAM_APM",
        "Attention　DEM+APM": "AttentionUNet_DEM_APM",
        "Middle　DEM+APM": "MIddleFusion_DEM_APM",
        "Attention　SAM+APM": "AttentionUNet_SAM_APM",
        "Final　DEM+APM": "FinalFusion_DEM_APM",
        "TransUNet　SAM+APM": "TransUNet_SAM_APM",
        "Final　SAM+APM": "FinalFusion_SAM_APM",
        "Middle　SAM+APM": "MiddleFusion_SAM_APM",
        "Early　DEM+APM": "EarlyFusionUNet_DEM_APM",
        "Early　SAM+APM": "EarlyFusionUNet_SAM_APM",
        "単一入力　DEM のみ": "Train_Hiroshima_Test_Shimane_OnlyDEM",
        "単一入力　SAM のみ": "Train_Hiroshima_Test_Shimane_OnlySAM",
    }
    out.update(extra)
    return out


def text_of(sid):
    p = SLIDES / f"{sid}.html"
    t = p.read_text(encoding="utf-8")
    t = re.sub(r"<aside>.*?</aside>", "", t, flags=re.S)   # 発表者ノートは対象外
    return t


def plain(sid):
    t = re.sub(r"<[^>]+>", " ", text_of(sid))
    return re.sub(r"\s+", " ", html.unescape(t)).strip()


# ----------------------------------------------------------------------
# スライドごとの検査
# ----------------------------------------------------------------------
def check_res_all(m, L):
    """全14条件の広島/島根F値（棒グラフの数字）。"""
    rows = re.findall(
        r'<p style="width:430px[^"]*">([^<]+)</p>.*?tabular-nums">([\d.]+)</p>'
        r'.*?tabular-nums">([\d.]+)</p>', text_of("res_all"), re.S)
    out = []
    for lab, h, s in rows:
        cond = L.get(lab.strip())
        if not cond:
            out.append(("res_all", f"{lab} の条件名", lab, None, "不可",
                        "スライドのラベルを条件名に対応づけられない"))
            continue
        out.append(("res_all", f"{lab} 広島F値", float(h),
                    f1(m, cond, "hiroshima", "警戒のみ"), None, ""))
        out.append(("res_all", f"{lab} 島根F値", float(s),
                    f1(m, cond, "shimane", "全件"), None, ""))
    return out


def check_res_fusion(m, L):
    """融合方式の表。1行に DEM と SAM の「広島 / 島根」が並ぶ。"""
    fam = {"Early（入力で結合）": ("EarlyFusionUNet_DEM_APM", "EarlyFusionUNet_SAM_APM"),
           "Middle（各段で融合）": ("MIddleFusion_DEM_APM", "MiddleFusion_SAM_APM"),
           "Final（最後に結合）": ("FinalFusion_DEM_APM", "FinalFusion_SAM_APM"),
           "Attention 機構": ("AttentionUNet_DEM_APM", "AttentionUNet_SAM_APM"),
           "Transformer 機構": (None, "TransUNet_SAM_APM")}
    out = []
    for lab, rest in re.findall(r"<tr><td>([^<]+)</td>(.*?)</tr>", text_of("res_fusion"), re.S):
        if lab not in fam:
            continue
        vals = [float(v) for v in re.findall(r"(\d\.\d{4})", rest)]
        conds = [c for c in fam[lab] if c]
        for i, cond in enumerate(conds):
            if 2 * i + 1 >= len(vals):
                continue
            out.append(("res_fusion", f"{lab} {cond} 広島", vals[2 * i],
                        f1(m, cond, "hiroshima", "警戒のみ"), None, ""))
            out.append(("res_fusion", f"{lab} {cond} 島根", vals[2 * i + 1],
                        f1(m, cond, "shimane", "全件"), None, ""))
    return out


def check_res_aug(m, L):
    """データ拡張の表。`0.6744 → 0.6996` の形で入っている。"""
    pairs = {"Final　DEM+APM": ("FinalFusion_DEM_APM", "DataOgument_FinalFusion_DEM_APM"),
             "Final　SAM+APM": ("FinalFusion_SAM_APM", "DataOgument_FinalFusion_SAM_APM"),
             "Attention　SAM+APM": ("AttentionUNet_SAM_APM",
                                    "DataOgument_AttentionUNet_SAM_APM")}
    out = []
    for lab, rest in re.findall(r"<tr><td>([^<]+)</td>(.*?)</tr>", text_of("res_aug"), re.S):
        if lab not in pairs:
            continue
        base, aug = pairs[lab]
        arrows = re.findall(r"(\d\.\d{4}) → (\d\.\d{4})", rest)
        if len(arrows) >= 1:
            out.append(("res_aug", f"{lab} 広島 拡張なし", float(arrows[0][0]),
                        f1(m, base, "hiroshima", "警戒のみ"), None, ""))
            out.append(("res_aug", f"{lab} 広島 拡張あり", float(arrows[0][1]),
                        f1(m, aug, "hiroshima", "警戒のみ"), None, ""))
        if len(arrows) >= 2:
            out.append(("res_aug", f"{lab} 島根 拡張なし", float(arrows[1][0]),
                        f1(m, base, "shimane", "全件"), None, ""))
            out.append(("res_aug", f"{lab} 島根 拡張あり", float(arrows[1][1]),
                        f1(m, aug, "shimane", "全件"), None, ""))
    return out


DUAL = [c.name for c in registry.load_all() if c.use_airphoto and "InstLoss" not in c.name]
SINGLE = [c.name for c in registry.load_all() if not c.use_airphoto]


def _avg(m, conds, region, tileset, name="f1", scope="通常"):
    vs = [col(m, c, region, tileset, name, scope) for c in conds]
    vs = [v for v in vs if v is not None]
    return sum(vs) / len(vs) if vs else None


def _tiles(m, conds, region, tileset):
    vs = [col(m, c, region, tileset, "n_tiles") for c in conds]
    vs = [v for v in vs if v is not None]
    return vs[0] if vs and all(v == vs[0] for v in vs) else None


def check_res_gen(m, L):
    """汎化のスライド。デュアル入力12条件の平均。"""
    t = plain("res_gen")
    out = []
    want = {"Recall": "recall", "Precision": "precision", "F値": "f1"}
    # 表は「Recall 0.707 0.535 −0.172」の並び
    for jp, colname in want.items():
        mm = re.search(rf"{jp}\s+(0\.\d+)\s+(0\.\d+)\s+[−-](0\.\d+)", t)
        if not mm:
            continue
        h, s, d = (float(x) for x in mm.groups())
        out.append(("res_gen", f"広島平均 {jp}", h,
                    _avg(m, DUAL, "hiroshima", "警戒のみ", colname), None, ""))
        out.append(("res_gen", f"島根平均 {jp}", s,
                    _avg(m, DUAL, "shimane", "全件", colname), None, ""))
        hh = _avg(m, DUAL, "hiroshima", "警戒のみ", colname)
        ss = _avg(m, DUAL, "shimane", "全件", colname)
        out.append(("res_gen", f"差 {jp}", d,
                    (hh - ss) if (hh is not None and ss is not None) else None, None, ""))
    return out


def check_res_matched(m, L):
    """テスト集合を揃えた再評価の表（デュアル入力12条件の平均）。"""
    t = plain("res_matched")
    # plain() が全角スペースを半角に潰すので、空白は \s+ で受ける
    rows = re.findall(r"(広島|島根)\s*(警戒区域ありのみ|背景込み・全件相当|全件)\s+"
                      r"([\d,]+)\s+([\d.]+) %\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
    key = {("広島", "警戒区域ありのみ"): ("hiroshima", "警戒のみ"),
           ("広島", "背景込み・全件相当"): ("hiroshima", "背景込み"),
           ("島根", "警戒区域ありのみ"): ("shimane", "警戒のみ"),
           ("島根", "全件"): ("shimane", "全件")}
    out = []
    for pref, suf, n, pos, r, p, f in rows:
        if (pref, suf) not in key:
            continue
        region, ts = key[(pref, suf)]
        lab = f"{pref} {suf}"
        out.append(("res_matched", f"{lab} 枚数", float(n.replace(",", "")),
                    _tiles(m, DUAL, region, ts), None, "枚"))
        for jp, colname, claimed in (("Recall", "recall", r),
                                     ("Precision", "precision", p),
                                     ("F値", "f1", f)):
            out.append(("res_matched", f"{lab} {jp}", float(claimed),
                        _avg(m, DUAL, region, ts, colname), None, ""))
    # 汎化低下 0.141 / 0.093 / 0.071
    gap = {}
    for name, (ra, ta, rb, tb) in (
            ("0.141", ("hiroshima", "警戒のみ", "shimane", "全件")),
            ("0.093", ("hiroshima", "警戒のみ", "shimane", "警戒のみ")),
            ("0.071", ("hiroshima", "背景込み", "shimane", "全件"))):
        a = _avg(m, DUAL, ra, ta)
        b = _avg(m, DUAL, rb, tb)
        gap[name] = (a - b) if (a is not None and b is not None) else None
    for name, v in gap.items():
        if name in t:
            out.append(("res_matched", f"汎化低下 {name}", float(name), v, None, ""))
    return out


def check_res_border(m, L):
    """境界評価による改善量（全条件の平均・最小・最大）。"""
    t = plain("res_border")
    out = []
    for region, ts, tag in (("hiroshima", "警戒のみ", "広島"), ("shimane", "全件", "島根")):
        diffs = []
        for c in registry.load_all():
            a = f1(m, c.name, region, ts, "境界")
            b = f1(m, c.name, region, ts, "通常")
            if a is not None and b is not None:
                diffs.append(a - b)
        if not diffs:
            continue
        avg = re.search(r"\+(0\.0\d+)\s*" + tag + r"\s*F値の平均改善", t)
        if avg:
            out.append(("res_border", f"{tag} 平均改善", float(avg.group(1)),
                        sum(diffs) / len(diffs), None, ""))
        # 「最小 +0.0123 / 最大 +0.0177」はその直後に出る
        seg = t[t.find(tag + " F値の平均改善"):][:160] if avg else ""
        mm = re.search(r"最小 \+(0\.0\d+) / 最大 \+(0\.0\d+)", seg)
        if mm:
            out.append(("res_border", f"{tag} 最小改善", float(mm.group(1)),
                        min(diffs), None, ""))
            out.append(("res_border", f"{tag} 最大改善", float(mm.group(2)),
                        max(diffs), None, ""))
    return out


def check_data_counts(m, L):
    """データのスライド：枚数と陽性画素の割合。

    陽性画素率は tp+fn が要る。metrics.csv に入っていなければ「不可」にする。
    """
    t = plain("data")
    out = []
    for region, ts, tag, n_claim in (("hiroshima", "警戒のみ", "広島", None),
                                     ("shimane", "全件", "島根", None)):
        vs = [col(m, c.name, region, ts, "n_tiles") for c in registry.load_all()]
        vs = [v for v in vs if v is not None]
        if vs:
            n = vs[0]
            if str(int(n)) in t.replace(",", ""):
                out.append(("data", f"{tag} テスト枚数", float(int(n)), float(int(n)),
                            None, "枚"))
    # 陽性画素の割合
    for region, ts, tag in (("hiroshima", "警戒のみ", "9.33"),
                            ("shimane", "全件", "3.58")):
        tp = fn = n = None
        for c in registry.load_all():
            tp = col(m, c.name, region, ts, "tp")
            fn = col(m, c.name, region, ts, "fn")
            n = col(m, c.name, region, ts, "n_tiles")
            if tp is not None and fn is not None and n:
                break
        if tp is None or fn is None or not n:
            out.append(("data", f"陽性画素の割合 {tag}%", float(tag), None, "不可",
                        "metrics.csv に tp/fn が無い"))
        else:
            out.append(("data", f"陽性画素の割合 {tag}%", float(tag),
                        100 * (tp + fn) / (n * 128 * 128), None, "%"))
    return out


def check_curves():
    """最良エポックの範囲（拡張なし 26〜73 / 拡張あり 693〜897）。"""
    out = []
    best = {}
    for c in registry.load_all():
        p = paths.results_dir("curves") / f"{c.name}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        colname = next((x for x in d.columns if "alidation" in x or "Test" in x), None)
        if not colname or d.empty:
            continue
        best[c.name] = int(d.loc[d[colname].idxmax(), "Epoch"])
    dual_noaug = [c.name for c in registry.load_all()
                  if c.use_airphoto and not c.augment and "InstLoss" not in c.name]
    dual_aug = [c.name for c in registry.load_all() if c.use_airphoto and c.augment]
    for tag, conds, claim in (("拡張なし", dual_noaug, (26, 73)),
                              ("拡張あり", dual_aug, (693, 897))):
        vs = [best[c] for c in conds if c in best]
        if not vs:
            continue
        out.append(("summary", f"最良エポック {tag} 最小", claim[0], min(vs), None, "epoch"))
        out.append(("summary", f"最良エポック {tag} 最大", claim[1], max(vs), None, "epoch"))
    return out


# metrics.csv だけでは再計算できないもの
UNVERIFIABLE = [
    ("res_background", "背景タイルの誤検出率・1タイルあたり過検出画素", "タイル単位のCSVが必要"),
    ("res_errors", "境界リングへの誤差の集中度・面積別の表", "タイル単位のCSVが必要"),
    ("res_where", "正解面積別・傾斜別のタイル平均F値", "タイル単位のCSVが必要"),
    ("res_attention", "注意係数、タイル単位の対比較", "注意係数CSV・タイル単位のCSVが必要"),
    ("res_airphoto", "島根 Precision +0.23 など単一入力との比較の一部", "一部は metrics.csv で可"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="1項目ずつ全部表示")
    ap.add_argument("--slide", help="このスライドだけ検査")
    ap.add_argument("--tol", type=float, default=TOL)
    a = ap.parse_args()

    m = load_metrics()
    L = label_map()
    rows = []
    for fn in (check_data_counts, check_res_all, check_res_fusion, check_res_aug,
               check_res_gen, check_res_matched, check_res_border):
        try:
            rows += fn(m, L)
        except Exception as e:
            rows.append((fn.__name__, "検査中にエラー", None, None, "不可", str(e)[:60]))
    rows += check_curves()

    if a.slide:
        rows = [r for r in rows if r[0] == a.slide]

    ok = ng = na = 0
    detail = []
    for slide, item, claimed, got, forced, note in rows:
        if forced == "参考":
            continue
        if forced == "不可" or claimed is None or got is None:
            verdict, diff = "不可", None
            na += 1
        else:
            diff = got - claimed
            tol = a.tol if abs(claimed) < 100 else 0.51
            if note == "%":
                tol = 0.005
            verdict = "OK" if abs(diff) <= tol else "NG"
            ok += verdict == "OK"
            ng += verdict == "NG"
        detail.append((slide, item, claimed, got, diff, verdict, note))

    print("=" * 94)
    print("発表スライドの数値検証   スライドのHTMLから抽出した値 vs results/metrics.csv")
    print("=" * 94)

    show = detail if a.detail else [d for d in detail if d[5] != "OK"]
    if show:
        print(f"\n{'スライド':<14}{'項目':<42}{'スライド':>10}{'再計算':>10}{'差':>10}  判定")
        for slide, item, claimed, got, diff, verdict, note in show:
            c = "—" if claimed is None else (f"{claimed:.4f}" if abs(claimed) < 100 else f"{claimed:.0f}")
            g = "—" if got is None else (f"{got:.4f}" if abs(got) < 100 else f"{got:.0f}")
            dd = "—" if diff is None else f"{diff:+.4f}"
            print(f"{slide:<14}{item[:40]:<42}{c:>10}{g:>10}{dd:>10}  {verdict}"
                  + (f"  {note}" if note else ""))
    if not a.detail and ng == 0:
        print("\n  ずれている項目はありません（--detail で全項目を表示）")

    print(f"\n{'-'*94}")
    print(f"OK {ok} 件 / NG {ng} 件 / 再計算できず {na} 件   （許容差 {a.tol}）")

    print("\n■ metrics.csv だけでは検証できない項目")
    for slide, what, why in UNVERIFIABLE:
        print(f"  {slide:<16}{what}")
        print(f"  {'':<16}  └ {why}")
    print("\n  これらを検証するには、タイル単位の解析をやり直す必要があります。")
    print("    python analysis/run_inference.py     --region all")
    print("    python analysis/run_features.py      --region all")
    print("    python analysis/run_instance_eval.py --region both")
    print("    python analysis/run_analysis.py")
    print("  （外付けSSDが必要。全条件だと数時間かかります）")

    return 1 if ng else 0


if __name__ == "__main__":
    sys.exit(main())
