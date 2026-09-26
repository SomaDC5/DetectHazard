# -*- coding: utf-8 -*-
import csv, io, json, os

import os as _os
ROOT = _os.path.dirname(_os.path.abspath(__file__))
SL = os.path.join(ROOT, "project", "slides")
os.makedirs(SL, exist_ok=True)

DARK   = "#16232E"
INK    = "#1E2C37"
MUTED  = "#4F5E69"
BG1    = "#F6F5F1"
BG2    = "#EAEBE5"
BLUE   = "#2D6E8E"
ORANGE = "#C0662A"
LINE   = "#D6D6CE"
CARD   = "#FFFFFF"

HEAD = "'Zen Kaku Gothic New', 'Noto Sans JP', sans-serif"
BODY = "'Noto Sans JP', sans-serif"

def section(sid, inner, bg=BG1, notes=None, pad="128px 128px 160px", extra="", gap=40):
    a = f"\n  <aside>{notes}</aside>" if notes else ""
    return (f'<section id="{sid}" data-transition="fade" style="background:{bg}; color:{INK}; '
            f'font-family:{BODY}; padding:{pad}; display:flex; flex-direction:column; gap:{gap}px{extra}">\n'
            f'{inner}{a}\n</section>\n')

def title(t, eyebrow=None):
    s = ""
    if eyebrow:
        s += (f'  <p style="font-family:{HEAD}; font-size:24px; font-weight:700; letter-spacing:2px; '
              f'color:{ORANGE}">{eyebrow}</p>\n')
    s += (f'  <h2 style="font-family:{HEAD}; font-size:52px; font-weight:700; line-height:1.15; '
          f'color:{DARK}">{t}</h2>\n')
    return s

ORDER = ["cover","background","previous","aims","data","setup","arch1","arch2","eval",
         "res_all","res_airphoto","res_fusion","res_aug","res_gen","res_matched","res_background","res_border",
         "res_errors","res_where","res_attention","qualitative","caveats","summary","next",
         "appendix","appendix2"]


def foot(sid, note=""):
    n = ORDER.index(sid) + 1 if sid in ORDER else sid
    left = f'<span style="color:{MUTED}">{note}</span>' if note else ""
    return (f'  <p style="position:absolute; left:128px; bottom:64px; width:1664px; font-size:24px; '
            f'color:{MUTED}">{left}</p>\n'
            f'  <p style="position:absolute; right:128px; bottom:64px; width:120px; text-align:right; '
            f'font-size:24px; color:{MUTED}">{n}</p>\n')

def card(inner, bg=CARD, pad=32, border=LINE, radius=14, flex="1"):
    return (f'<div style="flex:{flex}; display:flex; flex-direction:column; gap:12px; background:{bg}; '
            f'padding:{pad}px; border:1px solid {border}; border-radius:{radius}px">{inner}</div>')

def h3(t, color=None):
    c = color or DARK
    return f'<h3 style="font-family:{HEAD}; font-size:30px; font-weight:700; color:{c}">{t}</h3>'

def p(t, size=26, color=None, weight=400, lh=1.55):
    c = color or MUTED
    return (f'<p style="font-size:{size}px; font-weight:{weight}; line-height:{lh}; color:{c}">{t}</p>')

# ---------------------------------------------------------------- data
_WIDE = _os.environ.get(
    "MODELCOMPARISON_WIDE",
    "/Volumes/SSD-PHPU3A/dc5/ModelComparison/output/results_wide.csv")
rows = list(csv.DictReader(io.open(_WIDE, encoding="utf-8-sig")))
by = {r["condition"]: r for r in rows}
F = lambda c, k: float(by[c][k])

LABEL = {
 "DataOgument_FinalFusion_DEM_APM":  "拡張 + Final　DEM+APM",
 "DataOgument_AttentionUNet_SAM_APM":"拡張 + Attention　SAM+APM",
 "DataOgument_FinalFusion_SAM_APM":  "拡張 + Final　SAM+APM",
 "AttentionUNet_DEM_APM":            "Attention　DEM+APM",
 "MIddleFusion_DEM_APM":             "Middle　DEM+APM",
 "AttentionUNet_SAM_APM":            "Attention　SAM+APM",
 "FinalFusion_DEM_APM":              "Final　DEM+APM",
 "TransUNet_SAM_APM":                "TransUNet　SAM+APM",
 "FinalFusion_SAM_APM":              "Final　SAM+APM",
 "MiddleFusion_SAM_APM":             "Middle　SAM+APM",
 "EarlyFusionUNet_DEM_APM":          "Early　DEM+APM",
 "EarlyFusionUNet_SAM_APM":          "Early　SAM+APM",
 "Train_Hiroshima_Test_Shimane_OnlyDEM": "単一入力　DEM のみ",
 "Train_Hiroshima_Test_Shimane_OnlySAM": "単一入力　SAM のみ",
}

def bar_rows(conds, track=780, vmax=0.80, label_w=430, bar_h=14, gap=7, row_gap=12):
    out = []
    for c in conds:
        h = F(c, "広島_通常_f1"); s = F(c, "島根_通常_f1")
        wh = int(track * h / vmax); ws = int(track * s / vmax)
        out.append(
          f'<div style="display:flex; align-items:center; gap:20px">'
          f'<p style="width:{label_w}px; font-size:24px; color:{INK}">{LABEL[c]}</p>'
          f'<div style="flex:1; display:flex; flex-direction:column; gap:{gap}px">'
          f'<div style="display:flex; align-items:center; gap:14px">'
          f'<div style="width:{wh}px; height:{bar_h}px; background:{BLUE}; border-radius:7px"></div>'
          f'<p style="font-size:24px; color:{BLUE}; font-variant-numeric:tabular-nums">{h:.3f}</p></div>'
          f'<div style="display:flex; align-items:center; gap:14px">'
          f'<div style="width:{ws}px; height:{bar_h}px; background:{ORANGE}; border-radius:7px"></div>'
          f'<p style="font-size:24px; color:{ORANGE}; font-variant-numeric:tabular-nums">{s:.3f}</p></div>'
          f'</div></div>')
    return f'<div style="display:flex; flex-direction:column; gap:{row_gap}px">' + "".join(out) + "</div>"

def legend():
    return (f'<div style="display:flex; gap:40px; align-items:center">'
            f'<div style="display:flex; gap:12px; align-items:center">'
            f'<div style="width:36px; height:14px; background:{BLUE}; border-radius:7px"></div>'
            f'<p style="font-size:24px; color:{MUTED}">広島テスト（学習側）</p></div>'
            f'<div style="display:flex; gap:12px; align-items:center">'
            f'<div style="width:36px; height:14px; background:{ORANGE}; border-radius:7px"></div>'
            f'<p style="font-size:24px; color:{MUTED}">島根テスト（他県へ汎化）</p></div></div>')

slides = {}

# ---------------------------------------------------------------- 1 cover
slides["cover"] = (
f'<section id="cover" data-transition="fade" style="background:{DARK}; color:#F2F1EC; '
f'font-family:{BODY}; padding:128px; display:flex; flex-direction:column; justify-content:center; gap:40px">\n'
f'  <p style="font-family:{HEAD}; font-size:26px; font-weight:700; letter-spacing:3px; color:#E3A067">'
f'修士研究　進捗報告</p>\n'
f'  <h1 style="font-family:{HEAD}; font-size:92px; font-weight:700; line-height:1.18; color:#F7F6F2">'
f'マルチ入力 U-Net による<br>急傾斜地警戒区域の自動抽出</h1>\n'
f'  <p style="font-size:34px; line-height:1.5; color:#BFCBD3">'
f'融合方式・データ拡張・他県への汎化性能の比較（全14条件）</p>\n'
f'  <hr style="width:200px; border-top:3px solid #E3A067">\n'
f'  <p style="font-size:28px; line-height:1.7; color:#A9B7C0">'
f'関西大学大学院 総合情報学研究科　賀屋 颯真<br>2026年9月</p>\n'
f'  <aside>今回は、前回の学会論文（SCIS&amp;ISIS 2026）のあとに回した14条件の実験をまとめて報告します。'
f'大きな柱は3つ、「入力の融合方式」「データ拡張」「他県（島根）への汎化」です。</aside>\n'
f'</section>\n')

# ---------------------------------------------------------------- 2 background
inner = title("研究背景と目的", "BACKGROUND")
inner += (
 '  <div style="display:flex; gap:28px">'
 + card(h3("課題") + p("土砂災害警戒区域の指定には、都道府県による<b>基礎調査</b>が必要。"
          "航空写真の目視判読と現地調査に、多大な時間・費用・人手がかかる。<br>"
          "技術者の確保も年々難しくなる一方で、豪雨・地震による土砂災害は増加している。"))
 + card(h3("目的") + p("数値標高モデル（DEM）と航空写真から、"
          "<b>急傾斜地の警戒区域を自動抽出</b>する手法を確立し、基礎調査の効率化につなげる。<br>"
          "セマンティックセグメンテーション（U-Net）で、画素単位に警戒区域を推定する。"))
 + '</div>\n'
 + '  <div style="display:flex; gap:28px">'
 + card(f'{h3("2024年の土砂災害", ORANGE)}'
        f'<p style="font-family:{HEAD}; font-size:64px; font-weight:700; color:{DARK}; '
        f'font-variant-numeric:tabular-nums">1,433 件</p>'
        f'{p("うち<b>がけ崩れが1,074件</b>と最多。急傾斜地の警戒区域抽出の優先度は高い。", 24)}')
 + card(f'{h3("見逃しを出さないことが最優先", BLUE)}'
        f'{p("警戒区域の見落としは人的被害に直結する。過検出よりも<b>再現率（Recall）</b>を重視し、"
            "抽出結果は現地調査の当たりをつける材料として使うことを想定する。", 24)}', flex="1.35")
 + '</div>\n')
inner += foot("background")
slides["background"] = section("background", inner, notes=
 "背景は前回の論文と同じです。基礎調査の負荷が大きく、自動化の需要がある。"
 "評価では見逃しを避けたいのでRecallを重視する、という立場も前回から変えていません。")

# ---------------------------------------------------------------- 3 previous
inner = title("前回までの到達点と、残っていた課題", "PREVIOUS WORK")
inner += (
 f'  <p style="font-size:28px; color:{MUTED}; line-height:1.6">'
 f'SCIS&amp;ISIS 2026 投稿論文：広島県のデータで、入力4条件（DEM / SAM / DEM+APM / SAM+APM）を比較した。</p>\n')
inner += (
 '  <div style="display:flex; gap:28px">'
 + card(f'{h3("分かったこと", BLUE)}'
        '<ul style="font-size:26px; line-height:1.65; color:' + MUTED + '">'
        '<li>航空写真（APM）の併用で全条件が改善</li>'
        '<li>SAM+APM が最良（F値 0.6497）</li>'
        '<li>境界16pxを除くと全条件で F値が向上 → U-Net の周縁部劣化を確認</li></ul>')
 + card(f'{h3("残った課題", ORANGE)}'
        '<ul style="font-size:26px; line-height:1.65; color:' + MUTED + '">'
        '<li>デュアル入力の<b>結合位置</b>を検討していない</li>'
        '<li><b>データ拡張</b>を導入していない</li>'
        '<li>広島県内のみ。<b>他県で使えるか未検証</b></li>'
        '<li>Attention / Transformer などの構造が未検討</li></ul>')
 + '</div>\n')
inner += foot("previous", "前回：Kaya et al., SCIS&amp;ISIS 2026（投稿）")
slides["previous"] = section("previous", inner, bg=BG2, notes=
 "前回の論文で残した「今後の課題」が、そのまま今回の実験項目になっています。"
 "論文の結論部で挙げた3点、拡張・構造改良・汎化のうち、今回は全部に手をつけました。")

# ---------------------------------------------------------------- 4 aims
inner = title("今回の検証：3つの問いを14条件で", "THIS WORK")
q = lambda n, t, d: card(
    f'<p style="font-family:{HEAD}; font-size:60px; font-weight:700; color:{ORANGE}">{n}</p>'
    f'{h3(t)}{p(d, 25)}')
inner += ('  <div style="display:flex; gap:28px">'
 + q("Q1", "結合位置と構造で<br>精度は変わるか", "Early / Middle / Final の3つの融合位置に、Attention 機構と Transformer 機構を加えた5種類を比較する。")
 + q("Q2", "データ拡張は<br>効くか", "回転・平行移動・拡大縮小・反転を学習時に適用した条件を、同一構造の非拡張条件と対にして比較する。")
 + q("Q3", "他県でも<br>使えるか", "広島県のみで学習したモデルを、一度も学習に使っていない島根県の全タイルにそのまま適用する。")
 + '</div>\n')
inner += (f'  <div style="display:flex; gap:28px; align-items:center; background:{CARD}; padding:28px 32px; '
 f'border:1px solid {LINE}; border-radius:14px">'
 f'<p style="font-size:26px; color:{INK}"><b>実験条件</b></p>'
 f'<p style="font-size:26px; color:{MUTED}">地形量 2種（DEM / SAM）× 融合方式 5種 × 拡張の有無、'
 f'および航空写真を使わない単一入力 2種　＝　<b style="color:{DARK}">計 14 条件</b>'
 f'（各条件について 広島 / 島根 × 通常 / 境界 の 4 通りの評価）</p></div>\n')
inner += foot("aims")
slides["aims"] = section("aims", inner, notes=
 "今回回した実験は全部で14条件です。指示書で対象外とした Final_SAM_APM と Triple_UNet_SAM_APM_GEO、"
 "それと FinalFusion_SAM_APM と中身が同じ Train_Hiroshima_Test_Shimane_SAM_APM は除いています。")

# ---------------------------------------------------------------- 5 data
inner = title("データ", "DATA")
inner += ('  <div style="display:flex; gap:24px">'
 + card(f'{h3("入力① DEM", BLUE)}{p("国土地理院 基盤地図情報の5mメッシュ標高。標高値をグレースケール化。", 24)}')
 + card(f'{h3("入力② SAM", BLUE)}{p("DEMから Horn 法（3×3窓）で算出した傾斜角マップ。急傾斜地の形状を直接表す。", 24)}')
 + card(f'{h3("入力③ APM", BLUE)}{p("同一地点・同一解像度の航空写真（RGB 3ch）。植生・土地利用・人工物の手がかり。", 24)}')
 + card(f'{h3("正解マスク", ORANGE)}{p("県公開の（特別）警戒区域シェープファイルを、タイルの座標情報で画素単位にラスタ化。", 24)}')
 + '</div>\n')
tbl = (f'<table style="font-family:{BODY}; font-size:26px; color:{INK}; border:1px solid {LINE}; '
 f'border-radius:12px">'
 f'<tr style="background:{BG2}"><th style="width:26%; padding:14px 18px; text-align:left">&nbsp;</th>'
 f'<th style="width:26%; text-align:left">広島県（学習・テスト）</th>'
 f'<th style="width:26%; text-align:left">島根県（汎化テストのみ）</th>'
 f'<th style="width:22%; text-align:left">備考</th></tr>'
 f'<tr><td>総タイル数</td><td>38,148</td><td>24,569</td><td>128×128 px に統一</td></tr>'
 f'<tr style="background:{CARD}"><td>使用タイル</td><td>14,816（警戒区域を含むもの）</td>'
 f'<td>24,569（背景タイルも全件）</td><td>選び方が異なる</td></tr>'
 f'<tr><td>学習 / 検証 / テスト</td><td>8,889 / 2,963 / 2,964</td><td>− / − / 24,569</td>'
 f'<td>広島は 6 : 2 : 2</td></tr>'
 f'<tr style="background:{CARD}"><td>陽性画素の割合</td><td>9.33 %</td><td>3.58 %</td>'
 f'<td>島根のほうが不均衡</td></tr></table>')
inner += "  " + tbl + "\n"
inner += foot("data", "マスクには σ=1 のガウシアンフィルタを適用（境界のあいまいさへの対処）")
slides["data"] = section("data", inner, bg=BG2, notes=
 "前回の論文から、データセット自体を作り直しています。前回は 24,324 タイル中 11,857 枚でしたが、"
 "今回は 38,148 タイル中 14,816 枚です。島根は学習には一切使わず、背景タイルも含めた全件を評価に使いました。"
 "この「使用タイルの選び方が広島と島根で違う」点は、あとの注意点スライドでもう一度触れます。")

# ---------------------------------------------------------------- 6 setup
inner = title("学習・評価の設定", "SETUP")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("学習設定")}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{MUTED}">'
   f'<tr><td style="width:46%">最適化</td><td>Adam（lr = 1×10⁻⁴）</td></tr>'
   f'<tr><td>損失関数</td><td>Focal Tversky（α=0.7, β=0.3, γ=0.75）</td></tr>'
   f'<tr><td>バッチサイズ</td><td>32</td></tr>'
   f'<tr><td>エポック数</td><td>1,000（デュアル入力条件）</td></tr>'
   f'<tr><td>モデル選択</td><td>検証データの F値が最良のエポック</td></tr>'
   f'<tr><td>Dropout</td><td>0.25（ボトルネック前） / 0.5（後）</td></tr></table>')
 + card(f'{h3("データ拡張（該当条件のみ）")}'
   f'{p("学習データにのみ、地形量・航空写真・マスクへ<b>同一のパラメータで</b>適用する。", 25)}'
   f'<ul style="font-size:25px; line-height:1.6; color:{MUTED}">'
   f'<li>ランダム回転　±15°</li><li>平行移動　±10 %</li>'
   f'<li>拡大縮小　0.8 〜 1.2 倍</li><li>左右・上下反転　各 50 %</li></ul>'
   f'{p("マスクは最近傍補間、画像は双線形補間。", 24)}')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:26px 32px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("損失は α &gt; β とすることで<b>見逃し（False Negative）の罰を相対的に軽くせず</b>、"
     "再現率を優先する設定。クラス不均衡（陽性 約9 %）への対処も兼ねる。", 25)}</div>\n')
inner += foot("setup")
slides["setup"] = section("setup", inner, notes=
 "ハイパーパラメータは前回論文とほぼ同じですが、学習率だけ 1e-2 から 1e-4 に下げています。"
 "拡張は地形量・航空写真・マスクの3つに同じ変換をかけるのがポイントです。")

# ---------------------------------------------------------------- 7 arch1
def chip(t, bg, c="#FFFFFF", w=None):
    ww = f"width:{w}px; " if w else ""
    return (f'<p style="{ww}font-size:24px; font-weight:700; color:{c}; background:{bg}; '
            f'padding:10px 16px; border-radius:8px; text-align:center">{t}</p>')
def arrow_down():
    return ('<div style="display:flex; justify-content:center">'
            f'<x-shape kind="arrow-down" style="width:26px; height:34px; background:{LINE}"></x-shape></div>')
def fcard(name, desc, blocks):
    return card(f'{h3(name)}' + blocks + p(desc, 24), pad=28)

b_early = ('<div style="display:flex; flex-direction:column; gap:8px">'
  '<div style="display:flex; gap:10px">' + chip("地形量 1ch", BLUE) + chip("航空写真 3ch", "#6F8FA3") + '</div>'
  + arrow_down() + chip("チャネル結合 → エンコーダ（1本）", DARK) + arrow_down() + chip("デコーダ", "#8A9AA4") + '</div>')
b_mid = ('<div style="display:flex; flex-direction:column; gap:8px">'
  '<div style="display:flex; gap:10px">' + chip("地形量", BLUE) + chip("航空写真", "#6F8FA3") + '</div>'
  + arrow_down() + chip("エンコーダ2本 ＋ 各段で Fusion Gate", DARK) + arrow_down() + chip("デコーダ", "#8A9AA4") + '</div>')
b_fin = ('<div style="display:flex; flex-direction:column; gap:8px">'
  '<div style="display:flex; gap:10px">' + chip("地形量", BLUE) + chip("航空写真", "#6F8FA3") + '</div>'
  + arrow_down() + chip("エンコーダ2本（独立）", DARK) + arrow_down()
  + chip("ボトルネックで結合 → デコーダ", "#8A9AA4") + '</div>')

inner = title("比較したモデル構造 ①　どこで2つの入力を結合するか", "MODELS 1 / 2")
inner += ('  <div style="display:flex; gap:28px">'
 + fcard("Early Fusion", "入力段階で 1ch + 3ch = 4ch に結合し、共有エンコーダ1本で処理する。もっとも単純。", b_early)
 + fcard("Middle Fusion", "各ステージの後で両ストリームの特徴を 1×1 conv で融合し、残差的に足し戻す。浅い段階から情報が行き来する。", b_mid)
 + fcard("Final Fusion", "2つのエンコーダを独立に走らせ、同じ深さの特徴をデコーダ側で連結する。前回の論文で使った構造。", b_fin)
 + '</div>\n')
inner += foot("arch1")
slides["arch1"] = section("arch1", inner, notes=
 "結合位置を3段階に分けて比較しました。Final が前回論文で使った構造で、これが基準になります。"
 "Middle の Fusion Gate は concat して 1x1 conv で元のチャネル数に戻し、両ストリームに残差加算する形です。")

# ---------------------------------------------------------------- 8 arch2
b_att = ('<div style="display:flex; flex-direction:column; gap:8px">'
  + chip("エンコーダ2本（Final と同じ）", DARK)
  + arrow_down()
  + chip("スキップ接続に Attention Gate（各モダリティ別）", ORANGE)
  + arrow_down() + chip("デコーダ", "#8A9AA4") + '</div>')
b_trs = ('<div style="display:flex; flex-direction:column; gap:8px">'
  + chip("エンコーダ2本（Final と同じ）", DARK)
  + arrow_down()
  + chip("Transformer ボトルネック（8ヘッド × 4層, 8×8トークン）", ORANGE)
  + arrow_down() + chip("デコーダ", "#8A9AA4") + '</div>')
inner = title("比較したモデル構造 ②　注意機構と Transformer", "MODELS 2 / 2")
inner += ('  <div style="display:flex; gap:28px">'
 + fcard("Attention U-Net", "デコーダ側のゲート信号から、スキップ接続に空間的な注意係数（0〜1）を掛ける（Oktay et al. 2018）。"
         "地形量・航空写真それぞれに独立した Attention Gate を置いた。", b_att)
 + fcard("TransUNet（デュアル入力版）", "CNN エンコーダで 8×8 まで落とした特徴を 64 トークンとして Transformer Encoder に通し、"
         "元の形に戻してデコーダへ渡す。大域的な文脈を捉えることを狙う。", b_trs)
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:26px 32px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("いずれも <b>Final Fusion を土台</b>にして、スキップ接続またはボトルネックだけを差し替えた構成。"
     "エンコーダ・デコーダの構造は共通なので、機構そのものの効果を見られる。", 25)}</div>\n')
inner += foot("arch2")
slides["arch2"] = section("arch2", inner, bg=BG2, notes=
 "AttentionとTransUNetは、どちらもFinal Fusionをベースに一箇所だけ差し替えた構成にしてあります。"
 "だから比較すれば機構そのものの効果が見えるはずだ、という設計です。")

# ---------------------------------------------------------------- 9 eval
inner = title("評価方法", "EVALUATION")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("通常評価（面積評価）", BLUE)}'
        f'{p("出力マスク 128×128 の全画素を正解と比較し、TP / FP / FN / TN から Recall・Precision・F値を求める。", 25)}')
 + card(f'{h3("境界評価", ORANGE)}'
        f'{p("四辺から 16 px（各辺 12.5 %）を除いた<b>中心 96×96</b> のみで同じ指標を計算する。"
            "畳み込みでは画像端の画素が周囲情報を参照できず精度が落ちる、という仮説を検証するための指標。", 25)}')
 + '</div>\n')
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("Recall（再現率）")}{p("実際の警戒区域のうち、正しく抽出できた割合。<b>見逃しの少なさ</b>。本研究で最重視する。", 24)}')
 + card(f'{h3("Precision（適合率）")}{p("抽出した領域のうち、実際に警戒区域だった割合。<b>過検出の少なさ</b>。", 24)}')
 + card(f'{h3("F値")}{p("Recall と Precision の調和平均。クラス不均衡下での総合指標として用いる。", 24)}')
 + '</div>\n')
inner += (f'  <p style="font-size:25px; color:{MUTED}">'
 f'各条件について <b style="color:{DARK}">広島 / 島根 × 通常 / 境界 の 4 通り</b>を算出し、'
 f'14条件 × 4 = 56 件の結果を横断比較した。</p>\n')
inner += foot("eval")
slides["eval"] = section("eval", inner, notes=
 "評価軸は前回と同じ、通常評価と境界評価の2つです。今回はこれに広島・島根の2地域が掛かるので、"
 "1条件あたり4つの結果が出ます。")

# ---------------------------------------------------------------- 10 all results
order = ["DataOgument_FinalFusion_DEM_APM","DataOgument_AttentionUNet_SAM_APM","DataOgument_FinalFusion_SAM_APM",
         "AttentionUNet_DEM_APM","MIddleFusion_DEM_APM","AttentionUNet_SAM_APM","FinalFusion_DEM_APM",
         "TransUNet_SAM_APM","FinalFusion_SAM_APM","MiddleFusion_SAM_APM","EarlyFusionUNet_DEM_APM",
         "EarlyFusionUNet_SAM_APM","Train_Hiroshima_Test_Shimane_OnlyDEM","Train_Hiroshima_Test_Shimane_OnlySAM"]
inner = title("全14条件の F値（通常評価）", "RESULTS")
inner += "  " + legend() + "\n"
inner += "  " + bar_rows(order) + "\n"
inner += foot("res_all", "広島の F値が高い順。バーは F値 0〜0.80 のスケール")
slides["res_all"] = section("res_all", inner, notes=
 "全14条件を広島のF値が高い順に並べたものです。青が広島、オレンジが島根。"
 "ここで見てほしいのは2点。下2つの単一入力だけが明確に低いこと、"
 "そして青とオレンジの差、つまり汎化のギャップがどの条件でも大きいことです。")

# ---------------------------------------------------------------- 11 airphoto
inner = title("結果 ①　航空写真の併用が、いちばん効いた", "Q1 — INPUT")
sa = 0.6758; ss = 0.5351; ta = 0.5808; ts = 0.3553
def bignum(v, lab, color, sub):
    return card(f'<p style="font-family:{HEAD}; font-size:72px; font-weight:700; color:{color}; '
                f'font-variant-numeric:tabular-nums">{v}</p>{h3(lab)}{p(sub, 24)}')
inner += ('  <div style="display:flex; gap:28px">'
 + bignum("+0.095", "広島テスト F値", BLUE, "単一入力 平均 0.581 → デュアル入力 平均 0.676")
 + bignum("+0.180", "島根テスト F値", ORANGE, "単一入力 平均 0.355 → デュアル入力 平均 0.535")
 + bignum("+0.23", "島根 Precision", ORANGE, "単一 SAM 0.287 → デュアル入力 平均 0.537。過検出が大きく減る")
 + '</div>\n')
inner += (f'  <div style="display:flex; gap:28px">'
 + card(f'{h3("読み取れること")}'
   f'<ul style="font-size:26px; line-height:1.6; color:{MUTED}">'
   f'<li>地形量だけでは「急だが崩れない斜面」を切り分けられず、過検出が多い</li>'
   f'<li>航空写真の植生・土地利用・人工物の情報がその判別を補っている</li>'
   f'<li>効果は<b>他県のほうが大きい</b>（+0.180）。地形だけに頼るモデルほど地域差に弱い</li></ul>')
 + card(f'{h3("注意", ORANGE)}'
   f'{p("単一入力の2条件は 100 エポック・検証分割なし（テストF値で最良を選択）で学習しており、"
       "デュアル入力条件と学習条件が揃っていない。<b>単一入力に有利な選び方でなお下回っている</b>ため、"
       "結論の向きは変わらないが、差の大きさはそのままの値として扱えない。", 24)}', flex="0.9")
 + '</div>\n')
inner += foot("res_airphoto")
slides["res_airphoto"] = section("res_airphoto", inner, bg=BG2, notes=
 "一番はっきり出たのが航空写真の効果です。前回の論文と同じ傾向ですが、"
 "他県に持っていくとその差がさらに開く、というのが今回の新しい発見です。"
 "ただし単一入力の学習条件が揃っていないので、数字そのものは割り引いて見る必要があります。")

# ---------------------------------------------------------------- 12 fusion
inner = title("結果 ②　融合方式の差は小さい。Early だけが明確に劣る", "Q2 — FUSION")
def fus_row(arch, dem, sam):
    def cell(c):
        if c is None: return '<td style="color:#98A3AB">—</td>'
        return (f'<td style="font-variant-numeric:tabular-nums">{F(c,"広島_通常_f1"):.4f}　'
                f'<span style="color:{ORANGE}">{F(c,"島根_通常_f1"):.4f}</span></td>')
    return f'<tr><td>{arch}</td>{cell(dem)}{cell(sam)}</tr>'
tbl = (f'<table style="font-family:{BODY}; font-size:28px; color:{INK}; border:1px solid {LINE}; border-radius:12px">'
 f'<tr style="background:{BG2}"><th style="width:34%; padding:14px 18px; text-align:left">融合方式 / 機構</th>'
 f'<th style="width:33%; text-align:left">DEM + APM　<span style="font-size:24px">広島 / 島根</span></th>'
 f'<th style="width:33%; text-align:left">SAM + APM　<span style="font-size:24px">広島 / 島根</span></th></tr>'
 + fus_row("Early（入力で結合）", "EarlyFusionUNet_DEM_APM", "EarlyFusionUNet_SAM_APM")
 + fus_row("Middle（各段で融合）", "MIddleFusion_DEM_APM", "MiddleFusion_SAM_APM")
 + fus_row("Final（最後に結合）", "FinalFusion_DEM_APM", "FinalFusion_SAM_APM")
 + fus_row("Attention 機構", "AttentionUNet_DEM_APM", "AttentionUNet_SAM_APM")
 + fus_row("Transformer 機構", None, "TransUNet_SAM_APM")
 + '</table>')
inner += ('  <div style="display:flex; gap:28px; align-items:flex-start">'
 f'<div style="flex:1.25">{tbl}</div>'
 + card(f'{h3("読み取れること")}'
   f'<ul style="font-size:25px; line-height:1.6; color:{MUTED}">'
   f'<li>Middle / Final / Attention / Transformer は広島 0.672〜0.681 の範囲に収まり、<b>実質的に横並び</b></li>'
   f'<li>Early だけが 0.636〜0.649 と低い。入力段でチャネルを混ぜると、'
   f'地形と写真それぞれの特徴を別々に取り出せなくなると考えられる</li>'
   f'<li>島根でも順位はほぼ同じ。<b>Early SAM+APM（0.464）だけが大きく崩れる</b></li>'
   f'<li>Attention・Transformer を足しても、Final から明確な上積みはなかった</li></ul>', flex="1")
 + '</div>\n')
inner += foot("res_fusion", "いずれもデータ拡張なしの条件。黒＝広島、オレンジ＝島根（通常評価 F値）")
slides["res_fusion"] = section("res_fusion", inner, notes=
 "構造を凝ってもほとんど変わらない、というのが正直な結果です。"
 "Early だけが落ちるのは、入力段で混ぜると2つのモダリティを分けて表現できなくなるからだと解釈しています。"
 "AttentionもTransformerも、Final比で上積みはありませんでした。")

# ---------------------------------------------------------------- 13 augmentation
inner = title("結果 ③　データ拡張は広島で +0.025、島根では効かない", "Q3 — AUGMENTATION")
def aug_row(name, base, augc):
    hb, ha = F(base,"広島_通常_f1"), F(augc,"広島_通常_f1")
    sb, sa_ = F(base,"島根_通常_f1"), F(augc,"島根_通常_f1")
    d1, d2 = ha-hb, sa_-sb
    col = lambda d: BLUE if d > 0.002 else (ORANGE if d < -0.002 else MUTED)
    sgn = lambda d: f"{d:+.4f}"
    return (f'<tr><td>{name}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{hb:.4f} → {ha:.4f}</td>'
            f'<td style="font-variant-numeric:tabular-nums; color:{col(d1)}">{sgn(d1)}</td>'
            f'<td style="font-variant-numeric:tabular-nums">{sb:.4f} → {sa_:.4f}</td>'
            f'<td style="font-variant-numeric:tabular-nums; color:{col(d2)}">{sgn(d2)}</td></tr>')
tbl = (f'<table style="font-family:{BODY}; font-size:26px; color:{INK}; border:1px solid {LINE}; border-radius:12px">'
 f'<tr style="background:{BG2}"><th style="width:30%; padding:14px 18px; text-align:left">条件</th>'
 f'<th style="width:20%; text-align:left">広島 F値</th><th style="width:12%; text-align:left">差</th>'
 f'<th style="width:20%; text-align:left">島根 F値</th><th style="width:18%; text-align:left">差</th></tr>'
 + aug_row("Final　DEM+APM", "FinalFusion_DEM_APM", "DataOgument_FinalFusion_DEM_APM")
 + aug_row("Final　SAM+APM", "FinalFusion_SAM_APM", "DataOgument_FinalFusion_SAM_APM")
 + aug_row("Attention　SAM+APM", "AttentionUNet_SAM_APM", "DataOgument_AttentionUNet_SAM_APM")
 + '</table>')
inner += "  " + tbl + "\n"
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("過学習は大きく抑えられた", BLUE)}'
   f'<div style="display:flex; gap:32px; align-items:center">'
   f'<div style="display:flex; flex-direction:column; gap:6px">'
   f'<p style="font-size:24px; color:{MUTED}">拡張なし　最良エポック</p>'
   f'<p style="font-family:{HEAD}; font-size:52px; font-weight:700; color:{MUTED}; '
   f'font-variant-numeric:tabular-nums">26 〜 73</p></div>'
   f'<x-shape kind="arrow-right" style="width:56px; height:28px; background:{LINE}"></x-shape>'
   f'<div style="display:flex; flex-direction:column; gap:6px">'
   f'<p style="font-size:24px; color:{MUTED}">拡張あり　最良エポック</p>'
   f'<p style="font-family:{HEAD}; font-size:52px; font-weight:700; color:{BLUE}; '
   f'font-variant-numeric:tabular-nums">693 〜 897</p></div></div>'
   f'{p("拡張なしでは数十エポックで検証F値が頭打ちになるのに対し、拡張ありでは 1,000 エポック近くまで伸び続けた。", 24)}')
 + card(f'{h3("ただし汎化は改善しなかった", ORANGE)}'
   f'{p("島根の通常評価では +0.003 / −0.010 / +0.004 と、実質的に変化なし。"
       "回転・反転・スケールといった<b>幾何的な拡張は、地域が変わることで生じるずれ</b>"
       "（地形の成り立ち、植生、撮影条件の違い）には対応できていないと考えられる。", 24)}'
   f'{p("境界評価では島根も +0.003 〜 +0.011 と、わずかに改善する傾向はあった。", 24)}')
 + '</div>\n')
inner += foot("res_aug")
slides["res_aug"] = section("res_aug", inner, bg=BG2, notes=
 "拡張は広島では素直に効きました。特に注目したいのは最良エポックで、"
 "拡張なしだと30〜70エポックで頭打ちになるのが、拡張ありだと700〜900まで伸びます。過学習の抑制は明確です。"
 "ただし島根での性能は上がりませんでした。幾何変換だけでは地域差は埋まらない、ということだと思います。")

# ---------------------------------------------------------------- 14 generalization
inner = title("結果 ④　他県への汎化：F値は 0.11 〜 0.17 下がる", "Q4 — GENERALIZATION")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("指標ごとの平均（デュアル入力12条件）")}'
   f'<table style="font-family:{BODY}; font-size:27px; color:{INK}">'
   f'<tr style="background:{BG2}"><th style="width:34%; padding:12px 16px; text-align:left">&nbsp;</th>'
   f'<th style="width:22%; text-align:left">広島</th><th style="width:22%; text-align:left">島根</th>'
   f'<th style="width:22%; text-align:left">差</th></tr>'
   f'<tr><td>Recall</td><td>0.707</td><td>0.535</td>'
   f'<td style="color:{ORANGE}">−0.172</td></tr>'
   f'<tr><td>Precision</td><td>0.648</td><td>0.537</td>'
   f'<td style="color:{ORANGE}">−0.111</td></tr>'
   f'<tr><td>F値</td><td>0.676</td><td>0.535</td>'
   f'<td style="color:{ORANGE}">−0.141</td></tr></table>'
   f'{p("<b>Recall の低下が主因</b>。島根では警戒区域を見逃す側に外している。", 25)}')
 + card(f'{h3("条件別の汎化低下（F値の差）")}'
   f'<ul style="font-size:25px; line-height:1.6; color:{MUTED}">'
   f'<li>最小　Early DEM+APM　−0.115（ただし広島の絶対値が低い）</li>'
   f'<li>Final DEM+APM　−0.126（島根 F値 0.549 で全条件中<b>最高</b>）</li>'
   f'<li>最大　Early SAM+APM　−0.172</li>'
   f'<li>単一入力　−0.207 / −0.245</li></ul>'
   f'{p("デュアル入力では、DEM+APM 平均 0.542 が SAM+APM 平均 0.530 をわずかに上回った。"
       "広島では両者ほぼ同じ（0.677 と 0.675）なので、<b>他県ではDEMのほうがやや頑健</b>という傾向。", 24)}')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:26px 32px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("島根の最良は <b>境界評価で 0.578</b>（拡張 + Attention, SAM+APM）。"
     "学習に一度も使っていない県で F値 0.55 前後を保つことは、現地調査の当たりをつける用途としては意味がある水準だが、"
     "広島と同じ精度は出ていない。", 25)}</div>\n')
inner += foot("res_gen")
slides["res_gen"] = section("res_gen", inner, notes=
 "今回いちばん知りたかった汎化性能です。F値で0.11から0.17落ちます。"
 "内訳を見るとRecallの低下が大きく、島根では見逃す方向に外しています。"
 "見逃しを避けたいという研究の目的からすると、ここは重い結果です。")

# ---------------------------------------------------------------- 15 border
inner = title("結果 ⑤　境界評価：全条件で改善し、周縁部の劣化を再確認", "Q5 — BORDER")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'<p style="font-family:{HEAD}; font-size:72px; font-weight:700; color:{BLUE}; '
        f'font-variant-numeric:tabular-nums">+0.0154</p>{h3("広島　F値の平均改善")}'
        f'{p("最小 +0.0123 / 最大 +0.0177。14条件すべてで改善。", 24)}')
 + card(f'<p style="font-family:{HEAD}; font-size:72px; font-weight:700; color:{ORANGE}; '
        f'font-variant-numeric:tabular-nums">+0.0255</p>{h3("島根　F値の平均改善")}'
        f'{p("最小 +0.0178 / 最大 +0.0358。広島より改善幅が大きい。", 24)}')
 + card(f'{h3("解釈")}'
   f'{p("四辺 16 px を除くだけで全条件・両地域が改善する。畳み込みで画像端の画素が"
       "周囲の情報を参照できないことによる劣化が、実際に起きていることを裏づける。", 24)}'
   f'{p("汎化が難しい島根ほど改善幅が大きく、<b>境界の不確かさは条件が厳しいほど効いてくる</b>。", 24)}', flex="1.2")
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:28px 32px; border:1px solid {LINE}; border-radius:14px; '
 f'display:flex; flex-direction:column; gap:10px">'
 f'{h3("対策の方向", ORANGE)}'
 f'{p("① 隣接タイルを重ねて推論し（overlap inference）、中心部の予測だけを採用して貼り合わせる　"
     "② パディング方式の見直し（reflect / replicate）　"
     "③ 推論時はタイルを大きく取り、評価は中心のみとする", 26)}</div>\n')
inner += foot("res_border")
slides["res_border"] = section("res_border", inner, bg=BG2, notes=
 "境界評価は前回論文と同じ結論で、全条件で改善します。今回追加で分かったのは、"
 "島根のほうが改善幅が大きいことです。対策としてはオーバーラップ推論が一番素直だと思っています。")

# ---------------------------------------------------------------- 16 qualitative
inner = (f'  <h2 style="font-family:{HEAD}; font-size:52px; font-weight:700; line-height:1.15; color:{DARK}">'
 f'抽出結果の例（拡張 + Attention, SAM+APM / 広島テスト）</h2>\n')
hdr = ('<div style="display:flex; gap:20px">'
 + "".join(f'<p style="flex:1; font-size:26px; font-weight:700; color:{INK}; text-align:center">{t}</p>'
           for t in ["入力 ① 傾斜角マップ（SAM）", "入力 ② 航空写真（APM）", "正解マスク", "抽出結果"])
 + '</div>')
img = lambda src: (f'<div style="display:flex; gap:20px">'
 + f'<img src="{src}" alt="SAM・航空写真・正解マスク・抽出結果の比較" '
   f'style="width:1664px; height:330px; object-fit:contain">' + '</div>')
inner += "  " + hdr + "\n"
inner += "  __IMG1__\n  __IMG2__\n"
inner += (f'  <p style="font-size:26px; color:{MUTED}; line-height:1.55">'
 f'位置と形はおおむね一致する。一方で、正解より広めに取る傾向と、'
 f'タイル端に現れる小さな取りこぼし・張り出しが残る。</p>\n')
inner += foot("qualitative")
slides["qualitative"] = section("qualitative", inner, notes=
 "定性的に見ると、位置はよく当たっています。ただ正解より広めに出る傾向があり、"
 "これは Recall 重視の損失設定の影響でもあります。"
 "現地調査の当たりをつける用途なら、広めに出るのはむしろ許容できる方向だと考えています。")

# ---------------------------------------------------------------- 17 caveats
inner = title("比較にあたっての注意点", "CAVEATS")
def cav(n, t, d):
    return card(f'<div style="display:flex; gap:16px; align-items:center">'
                f'<p style="font-family:{HEAD}; font-size:30px; font-weight:700; color:{ORANGE}">{n}</p>'
                f'{h3(t)}</div>{p(d, 25)}')
inner += ('  <div style="display:flex; flex-direction:column; gap:22px">'
 + cav("1", "広島と島根でテスト集合の作り方が違う",
   "広島テストは<b>警戒区域を含むタイルのみ</b> 2,964 枚、島根は<b>背景タイルを含む全 24,569 枚</b>。"
   "陽性画素の割合が 9.33 % と 3.58 % で 2.6 倍違うため、島根の Precision は不利に出る。"
   "汎化低下の数値には、地域差だけでなくこの構成差が混ざっている。")
 + cav("2", "単一入力条件だけ学習条件が揃っていない",
   "単一入力（OnlyDEM / OnlySAM）は 100 エポック・検証分割なしで、テストF値が最良のエポックを採用している。"
   "デュアル入力は 1,000 エポック・検証分割あり。単一入力に有利な選択をしてなお下回っているため結論は変わらないが、"
   "差の大きさをそのまま論文の数値にはできない。")
 + cav("3", "各条件 1 回のみの学習",
   "乱数シードは全条件で共通だが、条件ごとに複数回は回していない。"
   "F値 0.01 程度の差は再現性の範囲内の可能性があり、「Middle / Final / Attention / Transformer が横並び」"
   "という読み方はそれを踏まえたもの。")
 + '</div>\n')
inner += foot("caveats")
slides["caveats"] = section("caveats", inner, bg=BG2, notes=
 "自分で気づいた範囲での注意点を挙げておきます。特に1番目は、島根の数字をそのまま「汎化性能」と呼ぶのが"
 "少し乱暴だという話なので、次の実験で揃えたいところです。")

# ---------------------------------------------------------------- 18 summary
inner = title("まとめ", "SUMMARY")
def sm(n, t, d):
    return (f'<div style="display:flex; gap:24px; align-items:flex-start; background:{CARD}; '
            f'padding:24px 30px; border:1px solid {LINE}; border-radius:14px">'
            f'<p style="width:44px; font-family:{HEAD}; font-size:30px; font-weight:700; color:{ORANGE}">{n}</p>'
            f'<div style="flex:1; display:flex; flex-direction:column; gap:6px">'
            f'<p style="font-size:29px; font-weight:700; color:{DARK}">{t}</p>'
            f'{p(d, 25)}</div></div>')
inner += ('  <div style="display:flex; flex-direction:column; gap:18px">'
 + sm("1", "航空写真の併用がもっとも効く",
      "F値で広島 +0.095、島根 +0.180。地形量だけのモデルほど地域が変わると崩れる。")
 + sm("2", "融合方式・機構の差は小さい",
      "Middle / Final / Attention / Transformer は広島 0.672〜0.681 で横並び。Early のみ明確に劣る。")
 + sm("3", "データ拡張は過学習を抑えるが、汎化は改善しない",
      "広島 +0.025、最良エポックは 26〜73 から 693〜897 へ。一方、島根の F値はほぼ変化なし。")
 + sm("4", "他県への汎化は F値 −0.11 〜 −0.17、Recall の低下が主因",
      "島根の最良は境界評価で 0.578（拡張 + Attention, SAM+APM）。")
 + sm("5", "境界 16 px の除去で全条件が改善",
      "広島 +0.015、島根 +0.026。U-Net の周縁部劣化を両地域で再確認した。")
 + '</div>\n')
inner += foot("summary")
slides["summary"] = section("summary", inner, gap=28, notes=
 "5点にまとめました。一言でいうと「構造を凝るより、入力に何を入れるかのほうが効く」"
 "「拡張は広島の中では効くが県をまたぐと効かない」という結果です。")

# ---------------------------------------------------------------- 19 next
inner = title("今後の課題", "NEXT")
def nx(t, d, tag):
    return card(f'<p style="font-size:24px; font-weight:700; color:{ORANGE}">{tag}</p>{h3(t)}{p(d, 24)}')
inner += ('  <div style="display:flex; gap:24px">'
 + nx("テスト条件を揃えた再評価", "島根と同じく、広島も背景タイルを含む全件で評価し直す。"
      "地域差と集合の構成差を分けて議論できるようにする。", "最優先")
 + nx("島根データを含めた学習", "他県のデータを学習に混ぜたとき、どこまで汎化が回復するかを確認する。"
      "県をまたいだクロス検証も行う。", "汎化")
 + nx("複数シードでの再現性確認", "横並びに見えている条件が本当に横並びなのかを、条件ごとに複数回の学習で確かめる。", "信頼性")
 + '</div>\n')
inner += ('  <div style="display:flex; gap:24px">'
 + nx("特徴量の追加", "曲率・陰影起伏に加え、地質情報・土地利用データを入力に加える。"
      "タイルごとの位置情報を持たせてあるので、同一地点の別データを突き合わせられる。", "入力")
 + nx("境界劣化への対策", "隣接タイルのオーバーラップ推論と、パディング方式の見直しを実装して効果を測る。", "推論")
 + nx("説明可能性と実運用", "Attention の重みなどから根拠を可視化し、GIS 上に抽出結果を戻して"
      "基礎調査の候補地提示として使えるかを検討する。", "応用")
 + '</div>\n')
inner += foot("next")
slides["next"] = section("next", inner, bg=BG2, notes=
 "次にやることの優先順位です。まずテスト条件を揃えないと、汎化の議論が正確にできません。"
 "そのうえで島根を学習に混ぜたときにどこまで戻るかを見たいと考えています。")

# ---------------------------------------------------------------- 20 appendix
inner = title("付録：結果集約ツール ModelComparison", "APPENDIX")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("やっていること")}'
   f'<ul style="font-size:25px; line-height:1.6; color:{MUTED}">'
   f'<li>各モデルフォルダのノートブックの<b>保存済み出力</b>から、Recall / Precision / F値と混同行列を抽出</li>'
   f'<li>フォルダ名から実験条件（拡張の有無・入力・融合方式）を解釈し、'
   f'ノートブック内の実装クラスと食い違っていないか照合</li>'
   f'<li>条件を横断した比較表（CSV / JSON）とグラフを出力</li></ul>')
 + card(f'{h3("設計上のポイント")}'
   f'<ul style="font-size:25px; line-height:1.6; color:{MUTED}">'
   f'<li>評価セルの判別は print のラベルではなく<b>ソースコード</b>で行う'
   f'（ラベルが実態と食い違っている箇所があったため）</li>'
   f'<li>再推論しないので、重みやデータセットがなくても動く</li>'
   f'<li>条件フォルダを足して再実行するだけで表とグラフに反映される</li></ul>')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:26px 32px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("既存の FSS_analysis が<b>1条件の中でタイル単位に誤差要因を分析する</b>のに対し、"
     "本ツールは<b>条件どうしを横断して最終スコアを比較する</b>。用途が違うので併用できる。", 25)}</div>\n')
inner += foot("appendix", "dc5/ModelComparison/　collect_results.py → make_report.py")
slides["appendix"] = section("appendix", inner, notes=
 "今回の結果をまとめるために作った集約ツールの説明です。条件を足しても再実行するだけで表が更新されるので、"
 "この先の実験でもそのまま使えます。")


# ================================================================
#  タイル単位の再解析（Research/tileanalysis）を受けた追加・更新
# ================================================================

# ---------------------------------------------------------------- res_matched
inner = title("結果 ⑤　両地域とも背景タイル込みで評価し直す", "RE-ANALYSIS")
inner += (f'  <p style="font-size:25px; color:{MUTED}; line-height:1.45">'
 f'広島の背景タイル（23,332枚、学習に未使用）にも同じ 20 % のホールドアウト率を当てはめて足し、'
 f'島根と構成を揃えた。</p>\n')
tblm = (f'<table style="font-family:{BODY}; font-size:24px; color:{INK}; border:1px solid {LINE}; '
 f'border-radius:12px">'
 f'<tr style="background:{BG2}">'
 f'<th style="width:30%; padding:12px 16px; text-align:left">テスト集合</th>'
 f'<th style="width:14%; text-align:left">枚数</th>'
 f'<th style="width:14%; text-align:left">陽性タイル率</th>'
 f'<th style="width:14%; text-align:left">Recall</th>'
 f'<th style="width:14%; text-align:left">Precision</th>'
 f'<th style="width:14%; text-align:left">F値</th></tr>'
 f'<tr><td>広島　警戒区域ありのみ</td><td>2,964</td><td>100 %</td>'
 f'<td>0.707</td><td>0.648</td><td>0.676</td></tr>'
 f'<tr style="background:{CARD}"><td><b>広島　背景込み・全件相当</b></td><td>7,631</td>'
 f'<td>38.8 %</td><td>0.707</td><td>0.531</td><td><b>0.606</b></td></tr>'
 f'<tr><td>島根　警戒区域ありのみ</td><td>9,679</td><td>100 %</td>'
 f'<td>0.535</td><td>0.642</td><td>0.583</td></tr>'
 f'<tr style="background:{CARD}"><td><b>島根　全件</b></td><td>24,569</td>'
 f'<td>39.4 %</td><td>0.535</td><td>0.537</td><td><b>0.535</b></td></tr></table>')
inner += "  " + tblm + "\n"
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("汎化低下の見積もりは半分になった", ORANGE)}'
   f'<div style="display:flex; gap:16px; align-items:center">'
   f'<div style="display:flex; flex-direction:column; gap:2px">'
   f'<p style="font-size:23px; color:{MUTED}">当初の報告</p>'
   f'<p style="font-family:{HEAD}; font-size:44px; font-weight:700; color:{MUTED}; '
   f'font-variant-numeric:tabular-nums">0.141</p></div>'
   f'<x-shape kind="arrow-right" style="width:36px; height:18px; background:{LINE}"></x-shape>'
   f'<div style="display:flex; flex-direction:column; gap:2px">'
   f'<p style="font-size:23px; color:{MUTED}">両方 警戒のみ</p>'
   f'<p style="font-family:{HEAD}; font-size:44px; font-weight:700; color:{MUTED}; '
   f'font-variant-numeric:tabular-nums">0.093</p></div>'
   f'<x-shape kind="arrow-right" style="width:36px; height:18px; background:{LINE}"></x-shape>'
   f'<div style="display:flex; flex-direction:column; gap:2px">'
   f'<p style="font-size:23px; color:{MUTED}">両方 全件相当</p>'
   f'<p style="font-family:{HEAD}; font-size:44px; font-weight:700; color:{ORANGE}; '
   f'font-variant-numeric:tabular-nums">0.071</p></div></div>'
   f'{p("デュアル入力12条件の平均。<b>当初の 0.141 のうち半分はテスト集合の作り方の違い</b>で、"
       "地域差そのものは 0.071。条件どうしの順位はどの集合でもほぼ変わらない。", 24)}')
 + card(f'{h3("ただし広島の水準も下がる", BLUE)}'
   f'{p("背景タイルを入れると広島の F値も <b>0.676 → 0.606</b> に落ちる。Recall は不変で、"
       "Precision が 0.648 → 0.531。これまで報告してきた広島の数字は、"
       "<b>警戒区域を含むタイルだけを選んで流した場合の値</b>だった。", 24)}'
   f'{p("背景を全 23,332枚入れた場合（26,296枚）は F値 0.428・Precision 0.308。", 24)}')
 + '</div>\n')
inner += foot("res_matched", "デュアル入力12条件の平均。条件別は region_matrix.csv")
slides["res_matched"] = section("res_matched", inner, bg=BG2, gap=26, notes=
 "前回の発表で注意点として挙げていた「広島と島根でテスト集合の作り方が違う」件を、"
 "実際に両方揃えて計算しました。汎化低下は 0.141 から 0.071 まで下がります。半分は集合の違いでした。"
 "ただし同時に、広島の数字自体も 0.676 から 0.606 に下がることが分かりました。"
 "これまでの報告値は、警戒区域を含むタイルだけを選んで流した場合の値だった、ということです。")

# ---------------------------------------------------------------- res_background
inner = title("結果 ⑥　背景タイルでの誤検出：学習に入れていないツケ", "BACKGROUND")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("警戒区域が1画素も無いタイルでの挙動", ORANGE)}'
   f'<table style="font-family:{BODY}; font-size:24px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:40%; padding:10px 14px; text-align:left">&nbsp;</th>'
   f'<th style="width:30%; text-align:left">広島</th>'
   f'<th style="width:30%; text-align:left">島根</th></tr>'
   f'<tr><td>何か出してしまうタイル</td>'
   f'<td style="color:{ORANGE}">47.5 %</td><td style="color:{ORANGE}">41.5 %</td></tr>'
   f'<tr><td>1タイルあたり過検出画素</td><td>235 px（1.4 %）</td><td>160 px（1.0 %）</td></tr>'
   f'<tr><td>単一入力 SAM の場合</td><td>441 px</td><td>439 px</td></tr></table>'
   f'{p("デュアル入力12条件の平均。<b>航空写真で過検出は半分近くに減る</b>が、"
       "それでも4割強のタイルで何かを出す。", 23)}')
 + card(f'{h3("誤検出が出るのは「警戒区域と似た傾斜帯」", BLUE)}'
   f'<table style="font-family:{BODY}; font-size:24px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:34%; padding:10px 14px; text-align:left">背景タイルの平均傾斜</th>'
   f'<th style="width:22%; text-align:left">枚数</th>'
   f'<th style="width:44%; text-align:left">1タイルあたり過検出画素</th></tr>'
   f'<tr><td>10〜15°</td><td>177</td><td style="color:{ORANGE}">421 px</td></tr>'
   f'<tr><td>15〜20°</td><td>457</td><td>331 px</td></tr>'
   f'<tr><td>20〜25°</td><td>805</td><td>268 px</td></tr>'
   f'<tr><td>30°〜</td><td>1,750</td><td>109 px</td></tr></table>'
   f'{p("Final SAM+APM・広島。<b>急斜面ほど誤検出が多いわけではない</b>。"
       "警戒区域ありタイルの傾斜帯（中央値 20.5°）と重なる所で出している。", 23)}')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:22px 30px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("学習データは警戒区域を含むタイルだけ（14,816枚）で、背景 23,332枚は1枚も使っていない"
     "（ノートブックの <b>bg_ratio = 0.0</b>）。「指定されていない斜面を棄却する」ことを学ぶ機会が無かった。"
     "背景タイルの平均傾斜の中央値は 27.9° と、警戒区域ありタイルの 20.5° より<b>むしろ急</b>で、"
     "急峻さは指定の決め手ではない。<b>→ bg_ratio を 0 から上げるのが最優先の対策</b>"
     "（コードは割合指定に対応済み）。", 24)}</div>\n')
inner += foot("res_background", "広島の背景タイル 23,332枚すべてに推論。表はホールドアウト相当 4,667枚")
slides["res_background"] = section("res_background", inner, gap=26, notes=
 "背景タイルで何が起きているかを見ました。4割強のタイルで何かを出してしまいます。"
 "面白いのは、急斜面ほど誤検出が多いわけではないことです。警戒区域と似た傾斜帯で出している。"
 "背景タイルの方がむしろ急で、つまり急かどうかは指定の決め手ではなく、"
 "家や道路といった保全対象があるかどうかが効いている、ということだと思います。"
 "学習に背景タイルを1枚も入れていないので、棄却の仕方を学ぶ機会がありませんでした。"
 "コードにはbg_ratioという割合指定がすでにあるので、そこを上げるのが次の一手です。")

# ---------------------------------------------------------------- res_errors
inner = title("結果 ⑥　誤差の内訳：境界劣化は実在するが小さい", "ERROR ANALYSIS")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("境界16pxリングへの誤差の集中度", BLUE)}'
   f'<p style="font-family:{HEAD}; font-size:60px; font-weight:700; color:{BLUE}; '
   f'font-variant-numeric:tabular-nums">1.04 〜 1.13</p>'
   f'{p("リングはタイル面積の 43.75 %。そこに落ちる誤差の割合は 45.5〜49.2 % だった。"
       "つまり<b>面積比より 4〜13 % 多いだけ</b>。境界劣化は確かにあるが、"
       "境界評価でF値が 0.015〜0.026 上がるのはこの程度の偏りによるもの。", 24)}')
 + card(f'{h3("面積を揃えると、効くのは小さい区域だけ", ORANGE)}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:34%; padding:12px 16px; text-align:left">正解の面積</th>'
   f'<th style="width:33%; text-align:left">境界への掛かり 〜30 %</th>'
   f'<th style="width:33%; text-align:left">50 %以上</th></tr>'
   f'<tr><td>小（下位25 %）</td><td>0.419</td>'
   f'<td style="color:{ORANGE}">0.286</td></tr>'
   f'<tr><td>中</td><td>0.572 / 0.612</td><td>0.499 / 0.612</td></tr>'
   f'<tr><td>大（上位25 %）</td><td>0.704</td><td>0.713</td></tr></table>'
   f'{p("Final SAM+APM・広島のタイル平均F値。<b>小さい区域がタイル端に掛かったときだけ</b>"
       "大きく落ちる。オーバーラップ推論が効くのはここ。", 24)}')
 + '</div>\n')
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("誤りの向き（誤差に占める見逃しの割合）")}'
   f'{p("広島 29〜49 %　→　島根 45〜56 %。<b>島根では見逃し側に寄る</b>。"
       "抽出面積/正解面積で見ても、広島 1.01〜1.16 に対し島根は 0.89〜1.12 と控えめになる。", 24)}')
 + card(f'{h3("単一入力は過剰に塗る")}'
   f'{p("SAM単一入力の抽出面積は正解の <b>1.44倍（広島）/ 1.71倍（島根）</b>。"
       "急斜面をすべて警戒区域として塗ってしまう。航空写真の併用でこれが 1.0 前後まで下がる。", 24)}')
 + '</div>\n')
inner += foot("res_errors")
slides["res_errors"] = section("res_errors", inner, notes=
 "誤差をタイル単位で分解した結果です。境界評価でF値が上がるのは事実ですが、"
 "誤差の集中度で見ると面積比より1割多い程度で、効果としては小さい。"
 "面積を揃えて見ると、境界の影響は小さい警戒区域に限られていました。"
 "オーバーラップ推論を入れるなら、狙うのはそこです。")

# ---------------------------------------------------------------- res_where
inner = title("結果 ⑦　どういうタイルで外しているか", "WHERE IT FAILS")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("正解の面積が小さいほど当たらない", ORANGE)}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:34%; padding:12px 16px; text-align:left">正解の面積割合</th>'
   f'<th style="width:33%; text-align:left">広島</th>'
   f'<th style="width:33%; text-align:left">島根</th></tr>'
   f'<tr><td>〜2 %</td><td style="color:{ORANGE}">0.30</td>'
   f'<td style="color:{ORANGE}">0.17</td></tr>'
   f'<tr><td>2〜5 %</td><td>0.52</td><td>0.39</td></tr>'
   f'<tr><td>5〜10 %</td><td>0.59</td><td>0.46</td></tr>'
   f'<tr><td>10〜20 %</td><td>0.67</td><td>0.56</td></tr>'
   f'<tr><td>20 %〜</td><td>0.73</td><td>0.65</td></tr></table>'
   f'{p("Final SAM+APM のタイル平均F値。面積の小さい区域が最大の弱点。", 24)}')
 + card(f'{h3("急斜面ほど当たらない ＝ 航空写真が効く理由", BLUE)}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:34%; padding:12px 16px; text-align:left">タイル平均傾斜角</th>'
   f'<th style="width:33%; text-align:left">SAM単一入力</th>'
   f'<th style="width:33%; text-align:left">SAM+APM</th></tr>'
   f'<tr><td>〜10°</td><td>0.550</td><td>0.586</td></tr>'
   f'<tr><td>15〜20°</td><td>0.429</td><td>0.575</td></tr>'
   f'<tr><td>25°〜</td><td style="color:{ORANGE}">0.356</td><td>0.469</td></tr>'
   f'<tr><td>落差</td><td style="color:{ORANGE}">−0.194</td>'
   f'<td style="color:{BLUE}">−0.117</td></tr></table>'
   f'{p("<b>一面が急斜面の場所ほど、傾斜角だけでは切り分けられない</b>。"
       "航空写真はそこで効いている。", 24)}')
 + '</div>\n')
inner += (f'  <div style="display:flex; gap:28px">'
 + card(f'{h3("どの条件でも当たらないタイル")}'
   f'{p("14条件すべてで F値 &lt; 0.3 のタイルが <b>広島 6.3 % / 島根 12.0 %</b>。"
       "逆に全条件で F値 &gt; 0.7 は 広島 9.1 % / 島根 1.7 %。"
       "モデルを変えても届かない領域が一定量ある。", 24)}')
 + card(f'{h3("条件どうしはよく似ている")}'
   f'{p("タイル別F値の順位相関（Spearman）の中央値は <b>0.78</b>。"
       "Final / Attention / TransUNet の間は 0.87 で、<b>同じタイルで同じように外している</b>。"
       "最も似ていないのは単一入力との組み合わせ（0.60）。", 24)}')
 + '</div>\n')
inner += foot("res_where")
slides["res_where"] = section("res_where", inner, bg=BG2, notes=
 "タイルの性質で層別すると、弱点がはっきりします。面積の小さい警戒区域と、一面が急斜面の場所です。"
 "急斜面のところは、単一入力だと大きく落ちるのにデュアル入力だと持ちこたえる。"
 "航空写真が何を補っているかの具体的な答えになっています。")

# ---------------------------------------------------------------- res_attention
inner = title("結果 ⑧　Attention は働いている。ただし精度には繋がっていない", "ATTENTION")
inner += ('  <div style="display:flex; gap:28px">'
 + card(f'{h3("同じタイル上での Final との差", ORANGE)}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:32%; padding:12px 16px; text-align:left">条件</th>'
   f'<th style="width:20%; text-align:left">平均差</th>'
   f'<th style="width:26%; text-align:left">改善/悪化タイル</th>'
   f'<th style="width:22%; text-align:left">有意差</th></tr>'
   f'<tr><td>広島 SAM</td><td>+0.0038</td><td>1169 / 1156</td>'
   f'<td style="color:{MUTED}">なし (p=0.42)</td></tr>'
   f'<tr><td>広島 DEM</td><td>+0.0087</td><td>1271 / 1079</td><td>あり</td></tr>'
   f'<tr><td>島根 SAM</td><td>+0.0096</td><td>3863 / 3293</td><td>あり</td></tr>'
   f'<tr><td>島根 DEM</td><td style="color:{ORANGE}">−0.0144</td><td>3301 / 3958</td>'
   f'<td style="color:{ORANGE}">あり（悪化）</td></tr></table>'
   f'{p("勝ち負けがほぼ拮抗し、<b>符号が条件で反転する</b>。"
       "TransUNet も同様に 広島 −0.0069 / 島根 −0.0050 で、Final を上回らない。", 24)}')
 + card(f'{h3("注意係数は警戒区域を絞り込めているか", BLUE)}'
   f'<table style="font-family:{BODY}; font-size:25px; color:{INK}">'
   f'<tr style="background:{BG2}">'
   f'<th style="width:34%; padding:12px 16px; text-align:left">段・ブランチ</th>'
   f'<th style="width:22%; text-align:left">広島</th>'
   f'<th style="width:22%; text-align:left">島根</th>'
   f'<th style="width:22%; text-align:left">鈍り</th></tr>'
   f'<tr><td>32px・地形量</td><td style="color:{BLUE}">+0.252</td>'
   f'<td style="color:{BLUE}">+0.197</td><td>−0.055</td></tr>'
   f'<tr><td>16px・地形量</td><td style="color:{BLUE}">+0.205</td>'
   f'<td style="color:{BLUE}">+0.188</td><td>−0.017</td></tr>'
   f'<tr><td>32px・航空写真</td><td>+0.068</td><td>+0.067</td><td>−0.001</td></tr>'
   f'<tr><td>128px・地形量</td><td style="color:{ORANGE}">−0.178</td>'
   f'<td style="color:{ORANGE}">−0.103</td><td>+0.075</td></tr></table>'
   f'{p("Attention SAM+APM。正解の内側と外側での注意係数の差。"
       "<b>32pxの段で地形量を警戒区域へ強く絞り込む</b>一方、航空写真側はほぼ素通し、"
       "最終段では逆に抑制している。", 24)}')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:26px 32px; border:1px solid {LINE}; border-radius:14px; '
 f'display:flex; flex-direction:column; gap:8px">'
 f'{h3("解釈")}'
 f'{p("Attention Gate は<b>学習に失敗しているわけではない</b>。32px の段で地形量の特徴を"
     "警戒区域に明確に絞り込んでいる。それでも F値が変わらないということは、"
     "<b>デコーダが既に同じ情報を取り出せており、ゲートが冗長</b>だということになる。"
     "島根でも同じ形は保たれるが、32pxの絞り込みは 3条件とも 0.055〜0.090（2〜3割）鈍る。"
     "構造を足す方向より、入力に新しい情報（地質・土地利用）を足す方向のほうが期待できる。", 26)}</div>\n')
inner += foot("res_attention", "注意係数はスキップ接続に掛かる 0〜1 の係数。Research/tileanalysis で抽出")
slides["res_attention"] = section("res_attention", inner, notes=
 "Attentionが効かない理由を調べました。タイル単位で見ると勝ち負けが拮抗していて、"
 "しかも符号が条件で反転します。では注意機構がまったく学習できていないのかというとそうではなく、"
 "32pxの段では地形量を警戒区域にはっきり絞り込んでいました。"
 "絞り込めているのに精度が変わらない、つまりデコーダが既に同じ情報を持っていて冗長だ、という解釈になります。")

# ---------------------------------------------------------------- caveats（更新）
inner = title("比較にあたっての注意点", "CAVEATS")
def cav2(n, t, d):
    return card(f'<div style="display:flex; gap:16px; align-items:center">'
                f'<p style="font-family:{HEAD}; font-size:30px; font-weight:700; color:{ORANGE}">{n}</p>'
                f'{h3(t)}</div>{p(d, 25)}')
inner += ('  <div style="display:flex; flex-direction:column; gap:20px">'
 + cav2("1", "単一入力条件だけ学習条件が揃っていない",
   "単一入力（OnlyDEM / OnlySAM）は 100 エポック・検証分割なしで、テストF値が最良のエポックを採用している。"
   "デュアル入力は 1,000 エポック・検証分割あり。単一入力に有利な選択をしてなお下回っているため"
   "結論は変わらないが、差の大きさをそのまま論文の数値にはできない。")
 + cav2("2", "各条件 1 回のみの学習",
   "乱数シードは全条件で共通だが、条件ごとに複数回は回していない。"
   "F値 0.01 程度の差は再現性の範囲内の可能性がある。実際、Attention の効果はタイル単位で見ると"
   "改善と悪化がほぼ同数で、条件によって符号が反転していた。")
 + cav2("3", "画素単位（micro）とタイル単位（macro）で見え方が違う",
   "従来報告している F値は全画素を足し上げた micro。タイルごとに F値を出して平均する macro では、"
   "広島の最良でも 0.578（micro 0.699）まで下がる。micro は面積の大きいタイルに引っ張られるため。"
   "「1タイルあたりどれくらい当たるか」を言うときは macro で見る必要がある。")
 + cav2("4", "テスト集合の構成差は解消した",
   "「広島と島根でタイルの選び方が違う」問題は、広島側にも背景タイルを同じ 20 % で足して揃えた（結果 ⑤）。"
   "当初報告の汎化低下 0.141 は、揃えると 0.071 になる。"
   "ただし<b>これまで報告してきた広島の F値 0.676 は、警戒区域を含むタイルだけを流した場合の値</b>で、"
   "背景込みでは 0.606。論文に書く数字はどちらの集合かを明示する必要がある。")
 + '</div>\n')
inner += foot("caveats")
slides["caveats"] = section("caveats", inner, bg=BG2, notes=
 "注意点は4つ。3番目のmicroとmacroの違いは今回の再解析で気づいた点で、"
 "論文に数字を書くときにどちらを使うかは意識しておく必要があります。")

# ---------------------------------------------------------------- summary（更新）
inner = title("まとめ", "SUMMARY")
SUM_ITEMS = []
inner += ('  <div style="display:flex; gap:24px"><div style="flex:1; display:flex; '
          'flex-direction:column; gap:14px">'
 + sm("1", "航空写真の併用がもっとも効く",
      "F値で広島 +0.095、島根 +0.180。平均傾斜25°以上のタイルで単一入力は −0.194 落ちるが、"
      "デュアル入力は −0.117 に留まる。背景タイルでの過検出も半減する。")
 + sm("2", "融合方式・機構の差は小さい。Early のみ明確に劣る",
      "Middle / Final / Attention / Transformer は広島 0.672〜0.681。"
      "タイル単位でも Final との差は ±0.01 程度で、符号が条件で反転する。")
 + sm("3", "Attention は絞り込めているが、精度には繋がっていない",
      "32px の段で地形量を警戒区域へ +0.25 絞り込んでいる。それでもF値が変わらない＝ゲートが冗長。")
 + '</div><div style="flex:1; display:flex; flex-direction:column; gap:14px">'
 + sm("4", "データ拡張は過学習を抑える",
      "広島 +0.025（最良エポック 26〜73 → 693〜897）。島根もタイル平均では +0.016〜+0.026 で有意。"
      "背景込みの集合では拡張ありが上位を占める。")
 + sm("5", "汎化低下 0.141 の半分はテスト集合の作り方の違い。揃えると 0.071",
      "両地域とも背景タイル込みで評価し直した結果。広島 0.606 / 島根 0.535。Recall は不変で Precision だけが動く。")
 + sm("6", "残る弱点は「小さい警戒区域」と「一面が急斜面の場所」",
      "正解の面積が2%未満のタイルは広島 0.30 / 島根 0.17。全14条件で F<0.3 のタイルが広島6.3% / 島根12.0%。")
 + sm("7", "背景タイルを学習に入れていないため、4割強のタイルで誤検出",
      "広島 47.5 % / 島根 41.5 %。bg_ratio = 0 のまま学習していたのが原因。")
 + '</div></div>\n')
inner += foot("summary")
slides["summary"] = section("summary", inner, gap=28, notes=
 "6点にまとめました。構造を凝るより入力に何を入れるかのほうが効く、という前回の結論は変わりませんが、"
 "今回の再解析で「どこで効いているか」「汎化低下の中身は何か」まで踏み込めました。")

# ---------------------------------------------------------------- next（更新）
inner = title("今後の課題", "NEXT")
inner += ('  <div style="display:flex; gap:24px">'
 + nx("学習に背景タイルを混ぜる", "背景タイル 23,332枚を1枚も使っていないため、"
      "「指定されていない斜面を棄却する」ことを学べていない。bg_ratio を 0 から上げて、"
      "誤検出率と F値がどう動くかを測る。コードは割合指定に対応済み。", "最優先")
 + nx("小面積の警戒区域への対策", "最大の弱点は面積2%未満のタイル（広島 0.30 / 島根 0.17）。"
      "しきい値の見直し、損失の面積重み付け、高解像度での推論を試す。", "精度")
 + nx("オーバーラップ推論の実装", "境界の影響は小面積の区域に集中していることが分かった。"
      "隣接タイルを重ねて推論し中心だけを採用する方式で、そこがどれだけ救えるかを測る。", "推論")
 + '</div>\n')
inner += ('  <div style="display:flex; gap:24px">'
 + nx("島根データを含めた学習", "他県のデータを学習に混ぜたとき、残っている汎化低下 0.071 が"
      "どこまで戻るかを確認する。県をまたいだクロス検証も行う。", "汎化")
 + nx("特徴量の追加", "Attention を足しても上積みが無かったことから、構造より入力。"
      "地質情報・土地利用データ・曲率・陰影起伏を加える。位置情報を持たせてあるので突き合わせは可能。", "入力")
 + nx("複数シードでの再現性確認", "横並びに見えている条件が本当に横並びなのかを、"
      "条件ごとに複数回の学習で確かめる。特に Attention と Middle の差。", "信頼性")
 + '</div>\n')
inner += foot("next")
slides["next"] = section("next", inner, bg=BG2, notes=
 "優先順位を組み直しました。テスト集合の切り分けは済んだので、次は島根を学習に混ぜることと、"
 "最大の弱点である小面積の警戒区域への対策です。")

# ---------------------------------------------------------------- appendix2
inner = title("付録：タイル単位の再解析ツール tileanalysis", "APPENDIX")
inner += (f'  <p style="font-size:26px; color:{MUTED}; line-height:1.55">'
 f'保存済みの重みでテスト集合を推論し直し、<b>1タイルずつ混同行列を取り直す</b>ためのツール。'
 f'足し上げた値はノートブックの出力と F値 0.0001 以内で一致する。</p>\n')
inner += ('  <div style="display:flex; gap:24px">'
 + card(f'{h3("できること")}'
   f'<ul style="font-size:24px; line-height:1.6; color:{MUTED}">'
   f'<li>1行=1タイルの混同行列（通常評価・境界評価）</li>'
   f'<li>タイルの性質（正解の面積・分割数・形の複雑さ・境界への掛かり方・傾斜・起伏・位置）</li>'
   f'<li>同じタイル上での条件どうしの対比較（Wilcoxon 検定つき）</li>'
   f'<li>性質で層別した精度、誤差の内訳、条件間の一致度</li>'
   f'<li>Attention Gate の注意係数の抽出</li></ul>')
 + card(f'{h3("再現性の担保")}'
   f'<ul style="font-size:24px; line-height:1.6; color:{MUTED}">'
   f'<li>広島のテスト分割を学習時と同じ引数で再現（全条件で同一の2,964枚であることを確認済み）</li>'
   f'<li>マスクのガウシアン処理・航空写真の正規化・二値化しきい値も学習時と同じ</li>'
   f'<li>CUDA / MPS / CPU を自動選択。MacBook (M4) で全14条件×2地域が約2時間</li>'
   f'<li>条件を足すときは config.py に1行</li></ul>')
 + '</div>\n')
inner += (f'  <div style="background:{CARD}; padding:24px 32px; border:1px solid {LINE}; border-radius:14px">'
 f'{p("ModelComparison が<b>ノートブックの出力から条件全体のスコアを拾う</b>のに対し、"
     "tileanalysis は<b>推論をやり直してタイル単位に分解する</b>。前者は速く、後者は細かい。併用する。", 25)}</div>\n')
inner += foot("appendix2", "~/Desktop/Master/Research/　run_inference.py → run_features.py → run_analysis.py")
slides["appendix2"] = section("appendix2", inner, bg=BG2, notes=
 "今回の再解析に使ったツールです。Researchフォルダに置いてあり、条件を足しても再実行するだけで済みます。")

# res_gen に次ページへの導線を足す
slides["res_gen"] = slides["res_gen"].replace(
    "広島と同じ精度は出ていない。",
    "広島と同じ精度は出ていない。<b>ただしこの 0.141 には、"
    "広島と島根でテストタイルの選び方が違うことの影響が混ざっている（次ページ）。</b>")


# ---------------------------------------------------------------- write
# 定性評価スライドの画像は、アーティファクトにアップロード済みの asset を参照する
_BLOBS = {
    "__IMG1__": ("/_blob/e0d76417ff3d9457b4778bb409109649",
                 "例1：傾斜角マップ、航空写真、正解マスク、抽出結果の比較"),
    "__IMG2__": ("/_blob/458a2223dde4c3bffe626a85a01b7950",
                 "例2：傾斜角マップ、航空写真、正解マスク、抽出結果の比較"),
}
for sid in ORDER:
    html = slides[sid]
    for key, (url, alt) in _BLOBS.items():
        html = html.replace(
            key,
            f'<img src="{url}" alt="{alt}" '
            f'style="width:1664px; height:300px; object-fit:contain">')
    with io.open(os.path.join(SL, f"{sid}.html"), "w", encoding="utf-8") as f:
        f.write(html)

deck = {
  "v": 4,
  "createdOnFiles": {"v": 1, "at": "2026-09-25T04:40:00Z"},
  "title": "急傾斜地警戒区域抽出 モデル比較",
  "order": ORDER,
  "sections": {
    "intro":   {"description": "研究の背景と、今回検証した3つの問い", "start": "cover"},
    "method":  {"description": "データ・学習設定・比較したモデル構造・評価方法", "start": "data"},
    "results": {"description": "14条件の結果：入力・融合方式・データ拡張・汎化・境界評価", "start": "res_all"},
    "tile":    {"description": "タイル単位の再解析：汎化低下の分離、誤差の内訳、弱点、Attentionの中身",
                "start": "res_matched"},
    "wrapup":  {"description": "注意点、まとめ、今後の課題", "start": "caveats"}
  },
  "faces": {
    "zen-kaku-gothic-new": {"family": "Zen Kaku Gothic New",
      "href": "https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&display=swap"},
    "noto-sans-jp": {"family": "Noto Sans JP",
      "href": "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap"}
  },
  "designSystems": []
}
with io.open(os.path.join(ROOT, "project", "deck.json"), "w", encoding="utf-8") as f:
    json.dump(deck, f, ensure_ascii=False, indent=2)
print("written", len(ORDER), "slides")
