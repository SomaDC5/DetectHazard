# -*- coding: utf-8 -*-
"""総括スライドの表を、学習の出発点ごとに2系統へ分ける。

元の表は判定列を「bg=0 の出発点 0.6727 / 0.6506」1本で書いていたが、
ASPP・深層監督・監督Attention は bg10 の上に足した条件なので、
bg10 自体の目減り（面積F −0.009）を差し引かれていた。

系統ごとに出発点とテスト集合を揃え直す。値はすべて results/metrics.csv から。

    python patch_summary.py [ファイル名]
"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Pt

DARK, MUTED = RGBColor(0x23, 0x31, 0x3C), RGBColor(0x4A, 0x5A, 0x68)
RED, CREAM, WHITE = RGBColor(0xB8, 0x50, 0x42), RGBColor(0xF1, 0xF0, 0xE8), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Meiryo"

HEAD = ("施策", "触った場所", "広島 面積F", "広島 箇所F", "出発点との差　面積F ／ 箇所F")
COLW = (250.0, 150.0, 130.0, 130.0, 210.5)   # 合計 870.5

# (表示名, 触った場所, 面積F, 箇所F)   ※ results/metrics.csv の実測値
BAND0 = ("bg = 0 の系統　　出発点 Final SAM+APM  0.6727 / 0.6506　（警戒のみ 2,964枚）",
         0.6727, 0.6506,
         [("データ拡張", "入力（前処理）", 0.6974, 0.6642),
          ("監督Attention 枝別", "スキップ接続", 0.6776, 0.6523),
          ("箇所正規化損失 p=0.5", "損失関数", 0.6763, 0.6596),
          ("箇所正規化損失 p=1.0", "損失関数", 0.6692, 0.6378)])

BAND1 = ("bg = 0.1 の系統　　出発点 bg10  0.6543 / 0.6136　（警戒+背景 3,430枚）",
         0.6543, 0.6136,
         [("＋ ASPP", "スキップ接続", 0.6634, 0.6171),
          ("＋ ASPP ＋ 出力ストライド8", "ボトルネック", 0.6610, 0.6250),
          ("＋ 監督Attention 枝別", "スキップ接続", 0.6588, 0.6411),
          ("＋ 監督Attention", "スキップ接続", 0.6578, 0.6340),
          ("＋ 深層監督 3段", "スキップ接続", 0.6569, 0.6319),
          ("＋ 深層監督 1段", "スキップ接続", 0.6560, 0.6294)])

# bg=0 → bg=0.1 への乗り換えそのもの。系統をまたぐので別扱いにする。
STEP = ("背景タイル混入 bg=0.1", "入力データ", "—", "—", "× 系統の出発点が −0.009 / −0.025")


def cell_text(cell, text, size=10.0, color=MUTED, bold=False, align=PP_ALIGN.LEFT,
              fill=None):
    cell.margin_left = cell.margin_right = Pt(7)
    cell.margin_top = cell.margin_bottom = Pt(1)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if fill is None:
        cell.fill.background()
    else:
        cell.fill.solid()
        cell.fill.fore_color.rgb = fill
    tf = cell.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = align
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    r = p.add_run()
    r.text = text
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), bold, color


def diff(v, base):
    d = v - base
    return f"{'+' if d >= 0 else '−'}{abs(d):.3f}"


def judge(area, inst, base_a, base_i):
    """ゆらぎ幅 0.005（面積F・検証上位10エポック）を基準に、指標ごとに記号をつける。

    面積F だけで判定すると、bg10 系の監督系が稼いでいる箇所F の伸び
    （+0.016〜+0.028）が記号に出ない。2つ並べる。
    """
    def mark(d):
        return "○" if d > 0.015 else ("△" if d > 0.005 else ("±" if d >= -0.005 else "×"))
    da, di = area - base_a, inst - base_i
    return f"{mark(da)} {diff(area, base_a)}　／　{mark(di)} {diff(inst, base_i)}"


def build_rows():
    rows = [HEAD]
    for band, base_a, base_i, items in (BAND0, BAND1):
        rows.append((band,))
        for name, place, a, i in items:
            rows.append((name, place, f"{a:.4f}", f"{i:.4f}", judge(a, i, base_a, base_i)))
    rows.append(STEP)
    return rows


def main(path):
    prs = Presentation(path)
    # 本文にも同じ語が出るスライドがあるので、見出しの位置（y≈47.5）で判定する
    def is_summary(sl):
        for sh in sl.shapes:
            if (sh.has_text_frame and abs(sh.top / 12700 - 47.5) < 4
                    and sh.left / 12700 < 60
                    and "学習側の施策" in sh.text_frame.text):
                return True
        return False

    target = next((sl for sl in prs.slides if is_summary(sl)), None)
    if target is None:
        sys.exit("総括スライドが見つかりません")

    tbl = next((sh for sh in target.shapes if sh.has_table), None)
    if tbl is None:
        sys.exit("総括スライドに表がありません")
    shape, t = tbl, tbl.table
    rows = build_rows()

    # 行数を合わせる（python-pptx に add_row が無いので XML を複製する）
    import copy
    while len(t.rows) < len(rows):
        t._tbl.append(copy.deepcopy(t._tbl.tr_lst[-1]))
    while len(t.rows) > len(rows):
        t._tbl.remove(t._tbl.tr_lst[-1])

    for ci, w in enumerate(COLW):
        t.columns[ci].width = Pt(w)

    total = 270.7
    h_head, h_band, n_band = 24.0, 22.0, 2
    h_data = (total - h_head - h_band * n_band) / (len(rows) - 1 - n_band)
    bands = {1, 1 + len(BAND0[3]) + 1}

    for ri, row in enumerate(rows):
        tr = t.rows[ri]
        tr.height = Pt(h_head if ri == 0 else (h_band if ri in bands else h_data))
        if ri in bands:
            # 帯見出し。つなぐと文字が連結されるので、先に 2〜5列目を空にする
            for ci in range(5):
                cell_text(t.cell(ri, ci), "", 10.0, WHITE, fill=DARK)
            t.cell(ri, 0).merge(t.cell(ri, 4))
            cell_text(t.cell(ri, 0), row[0], 10.0, WHITE, bold=True, fill=DARK)
            continue
        for ci in range(5):
            txt = row[ci] if ci < len(row) else ""
            is_head = ri == 0
            is_step = row is STEP
            cell_text(
                t.cell(ri, ci), txt,
                size=10.5 if is_head else 10.0,
                color=DARK if is_head else (RED if is_step and ci == 4 else MUTED),
                bold=is_head or (is_step and ci == 0),
                align=PP_ALIGN.LEFT if ci in (0, 1, 4) else PP_ALIGN.RIGHT,
                fill=CREAM if is_head else (CREAM if is_step else None))
    shape.height = Pt(total)

    # 見出しと、下の注記を書き換える
    for sh in target.shapes:
        if not sh.has_text_frame:
            continue
        tx = sh.text_frame.text
        if "学習側の施策" in tx and abs(sh.top / 12700 - 47.5) < 4:
            sh.text_frame.paragraphs[0].runs[0].text = "学習側の施策 — それぞれの出発点を超えたか"
        elif "既存4構造が収まる帯" in tx:
            p = sh.text_frame.paragraphs[0]
            runs = list(p.runs)
            runs[0].text = ("面積F のゆらぎ幅は 0.003〜0.005（検証F1 上位10エポック）。"
                            "これを明確に超えたのは、どちらの系統でもデータ拡張だけ。"
                            "一方 箇所F は bg10 系の監督系が +0.016〜+0.028 で、ゆらぎより一桁大きい。")
            for r in runs[1:]:
                r.text = ""

    prs.save(path)
    print(f"総括スライドの表を {len(rows)} 行に差し替えました: {Path(path).name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "夏期進捗報告_急傾斜地崩壊危険区域の自動抽出_改訂.pptx")
