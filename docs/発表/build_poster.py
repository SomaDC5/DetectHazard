# -*- coding: utf-8 -*-
"""A4一枚の研究ポスターを作る。GTposter.pptx の体裁に合わせてある。

  キャンバス 540 x 780 pt（A4縦）
  全幅のタイトル帯 → 2段組 → 全幅の図と結論

図は poster_figs/ に置いてある3枚を使う（analysis から実測して作ったもの）。

    python build_poster.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Pt

HERE = Path(__file__).resolve().parent
FIG = HERE / "poster_figs"
OUT = HERE / "研究ポスター_A4_急傾斜地警戒区域の自動抽出.pptx"

W, H = 540.0, 780.0
DARK = RGBColor(0x23, 0x31, 0x3C)
MUTED = RGBColor(0x4A, 0x5A, 0x68)
RED = RGBColor(0xB8, 0x50, 0x42)
CREAM = RGBColor(0xF1, 0xF0, 0xE8)
SLATE = RGBColor(0x5B, 0x7B, 0x8C)
GREEN = RGBColor(0x3D, 0x6B, 0x4A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD7, 0xDC, 0xDF)
FONT = "Meiryo"

ML, MR = 8.0, 8.0                    # 左右の余白
COL_W = (W - ML - MR - 10.0) / 2     # 2段組の1列ぶん
COL_L, COL_R = ML, ML + COL_W + 10.0
FULL_W = W - ML - MR


def _runs(tf, blocks, size, color, bold=False, align=PP_ALIGN.LEFT, spacing=1.35):
    tf.word_wrap = True
    for i, blk in enumerate(blocks):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        parts = blk if isinstance(blk, list) else [(blk, bold, color)]
        for txt, b, c in parts:
            r = p.add_run()
            r.text = txt
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), b, c


def tb(sl, x, y, w, h, blocks, size, color, bold=False, align=PP_ALIGN.LEFT,
       spacing=1.35, anchor=None):
    s = sl.shapes.add_textbox(Pt(x), Pt(y), Pt(w), Pt(h))
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if anchor:
        tf.vertical_anchor = anchor
    _runs(tf, blocks, size, color, bold, align, spacing)
    return s


def panel(sl, x, y, w, h, fill=CREAM, border=True):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(x), Pt(y), Pt(w), Pt(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if border:
        s.line.color.rgb = LINE
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    s.adjustments[0] = 0.035
    return s


def section(sl, x, y, w, h, no, title, blocks, size=10.0):
    """番号つきの節。見出しは濃紺の丸、題は太字。"""
    panel(sl, x, y, w, h)
    d = sl.shapes.add_shape(MSO_SHAPE.OVAL, Pt(x + 11), Pt(y + 10), Pt(17), Pt(17))
    d.fill.solid(); d.fill.fore_color.rgb = DARK
    d.line.fill.background(); d.shadow.inherit = False
    tf = d.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    _runs(tf, [str(no)], 10.0, WHITE, True, PP_ALIGN.CENTER, 1.0)
    tb(sl, x + 34, y + 10, w - 46, 18, [title], 12.5, DARK, bold=True)
    if blocks:
        tb(sl, x + 12, y + 34, w - 24, h - 44, blocks, size, MUTED, spacing=1.45)
    return y + 34


def caption(sl, x, y, w, text):
    tb(sl, x, y, w, 11, [text], 7.5, MUTED, align=PP_ALIGN.CENTER, spacing=1.0)


def stat(sl, x, y, w, value, label, color=RED):
    tb(sl, x, y, w, 24, [value], 19.0, color, bold=True, align=PP_ALIGN.CENTER, spacing=1.0)
    tb(sl, x, y + 25, w, 12, [label], 7.5, MUTED, align=PP_ALIGN.CENTER, spacing=1.0)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Pt(W), Pt(H)
    sl = prs.slides.add_slide(prs.slide_layouts[6])

    # ---------------------------------------------------------- タイトル帯
    band = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Pt(0), Pt(0), Pt(W), Pt(92))
    band.fill.solid(); band.fill.fore_color.rgb = DARK
    band.line.fill.background(); band.shadow.inherit = False
    tb(sl, 22, 16, W - 44, 30,
       ["深層学習による急傾斜地の土砂災害警戒区域の自動抽出"],
       19.0, WHITE, bold=True, spacing=1.1)
    tb(sl, 22, 46, W - 44, 16,
       ["—— 地形情報と航空写真の2つを入力に使い、広島県で学習して島根県で確かめる"],
       10.5, RGBColor(0xC4, 0xCE, 0xD4), spacing=1.1)
    tb(sl, 22, 68, W - 44, 14,
       ["関西大学 総合情報学研究科　広兼ゼミ　　賀屋 颯真"],
       10.0, RGBColor(0xC4, 0xCE, 0xD4), spacing=1.1)

    # ---------------------------------------------------------- 1 / 2
    y0 = 96.0
    section(sl, COL_L, y0, COL_W, 120, 1, "はじめに",
            ["近年、土砂災害が頻発している。土砂災害防止法により、"
             "おそれのある箇所は5年ごとに基礎調査を行う必要がある。",
             "いまは職員が航空写真と地形図を見ながら現地調査を行っており、"
             "時間・費用・労力の負担が大きい。"])
    section(sl, COL_R, y0, COL_W, 120, 2, "目的",
            ["地形の情報と航空写真から、警戒区域になりうる範囲を"
             "深層学習で塗り分け、調査の下書きを作る。",
             "広島県で学習したモデルが、学習に使っていない"
             "島根県でも使えるか（汎化）を確かめる。"])

    # ---------------------------------------------------------- 図1（全幅）
    y = y0 + 120 + 8
    fw = 408.0
    fx = (W - fw) / 2
    fh = fw / 3.221
    sl.shapes.add_picture(str(FIG / "fig_example.png"), Pt(fx), Pt(y), Pt(fw), Pt(fh))
    caption(sl, ML, y + fh + 1, FULL_W,
            "図1. 入力（a, b）と、行政が指定した正解（c）、モデルの出力（d）")

    # ---------------------------------------------------------- 3 / 4
    y = y + fh + 12
    h34 = 282.0
    # --- 3. 方法 ---
    body_y = section(sl, COL_L, y, COL_W, h34, 3, "方法", None)
    bx, bw = COL_L + 12, COL_W - 24
    tb(sl, bx, body_y, bw, 42,
       ["県全体を 128×128 画素のタイルに切り、1枚ずつ塗り分ける。"
        "広島県 38,148枚のうち警戒区域を含む 14,816枚で学習し、"
        "島根県 24,569枚で確かめた。"], 9.5, MUTED, spacing=1.42)
    tb(sl, bx, body_y + 48, bw, 13, ["入力は3種類"], 9.5, DARK, bold=True, spacing=1.2)
    ry = body_y + 64
    for tag, name, desc in (("SAM", "傾斜量", "標高から計算した斜面の傾き"),
                            ("DEM", "標高", "生の高さ。傾きはモデルに学ばせる"),
                            ("APM", "航空写真", "RGB 3色。植生・宅地の手がかり")):
        c = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Pt(bx), Pt(ry), Pt(32), Pt(14))
        c.fill.solid(); c.fill.fore_color.rgb = SLATE
        c.line.fill.background(); c.shadow.inherit = False
        c.adjustments[0] = 0.25
        tf = c.text_frame; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        _runs(tf, [tag], 7.5, WHITE, True, PP_ALIGN.CENTER, 1.0)
        tb(sl, bx + 38, ry + 1, bw - 38, 12,
           [[(name + "　", True, DARK), (desc, False, MUTED)]], 8.0, MUTED, spacing=1.2)
        ry += 16
    tb(sl, bx, ry + 3, bw, 13, ["モデル"], 9.5, DARK, bold=True, spacing=1.2)
    mh = bw / 2.526
    sl.shapes.add_picture(str(FIG / "fig_model.png"), Pt(bx), Pt(ry + 18), Pt(bw), Pt(mh))
    caption(sl, bx, ry + 18 + mh + 1, bw,
            "図2. 2つの入力を別々に符号化し、最後に連結する U-Net")

    # --- 4. 結果 ---
    body_y = section(sl, COL_R, y, COL_W, h34, 4, "結果", None)
    rx, rw = COL_R + 12, COL_W - 24
    tb(sl, rx, body_y, rw, 30,
       ["航空写真を足すと大きく上がる。学習していない島根県では、"
        "その差がさらに開く。"], 9.5, MUTED, spacing=1.42)
    ih = rw / 2.498
    sl.shapes.add_picture(str(FIG / "fig_inputs.png"), Pt(rx), Pt(body_y + 32), Pt(rw), Pt(ih))
    caption(sl, rx, body_y + 32 + ih + 1, rw, "図3. 入力の組み合わせごとの精度（面積F値）")

    sy = body_y + 32 + ih + 16
    for i, (v, l, c) in enumerate([("0.673", "広島 面積F値", RED),
                                   ("0.651", "広島 箇所F値", RED),
                                   ("0.538", "島根 面積F値", SLATE)]):
        stat(sl, rx + i * (rw / 3), sy, rw / 3, v, l, c)
    tb(sl, rx, sy + 46, rw, 54,
       ["モデルの中身をどう変えても精度は ±0.01 しか動かなかった。",
        [("効いたのはタイルの切り方。", True, DARK),
         ("窓をずらして9回推論し重ねると、箇所F値が +0.05 上がった。",
          False, MUTED)]], 9.0, MUTED, spacing=1.42)

    # --- 5. 考察と今後（全幅） ---
    y5 = y + h34 + 8
    h5 = H - y5 - 8
    section(sl, ML, y5, FULL_W, h5, 5, "考察と今後", None)
    col = (FULL_W - 40) / 3
    for i, (hd, bd) in enumerate(
            [("精度を決めているのはモデルではなかった",
              "構造をどう変えても ±0.01。タイルの切り方を変えるだけで +0.05 動いた。"
              "「モデルが悪い」のではなく「問題の与え方」の側に律速がある。"),
             ("失敗の6割はタイルの境界で起きていた",
              "1画素も当てられなかったタイル 282枚のうち 61% は、"
              "正解がタイルのふちにしかなく、隣のタイルへ切れていた。"),
             ("次は入力を増やす",
              "航空写真はいま解像度を落として使っている。また、ふもとに守るべき"
              "建物があるかは地形からは分からない。建物・土地利用の情報を足したい。")]):
        x = ML + 14 + i * (col + 6)
        tb(sl, x, y5 + 36, col, 26, [hd], 9.5, DARK, bold=True, spacing=1.3)
        tb(sl, x, y5 + 64, col, h5 - 72, [bd], 8.3, MUTED, spacing=1.42)

    prs.save(str(OUT))
    print(f"保存: {OUT.name}")
    return OUT


if __name__ == "__main__":
    build()
