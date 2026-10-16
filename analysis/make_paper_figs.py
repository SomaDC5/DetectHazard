# -*- coding: utf-8 -*-
"""論文用の図を作る。

なぜ必要か
----------
論文（docs/論文/）の図は3つとも結果のグラフで、**入力データも手法も
出力例も図が無い**。読者は何を入力にしているのか分からない。

学習中でも作れるものと、そうでないものを分けてある。

    --pipeline   手法の全体図。データを読まないので常に安全
    --output     予測と正解の差分。ハザードマップの GeoTIFF から**窓読み**
                 するだけなので数MBしか使わない（30720x35328 の 512 タイル化）
    --inputs     入力データの例（傾斜量図・航空写真・正解）。
                 **pkl を 5.2GB 読むので、学習中は避ける**

使い方（geopack 環境。rasterio が要る）
---------------------------------------
    python analysis/make_paper_figs.py --pipeline --output --out <保存先>
    python analysis/make_paper_figs.py --inputs   --out <保存先>   # 学習が終わってから
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches          # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

plt.rcParams["font.family"] = ["BIZ UDGothic", "Yu Gothic", "Meiryo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 差分の色（docs/ハザードマップの作り方.md の .qml と同じ）
C_TP = "#2c7fb8"       # 当たり
C_FP = "#e34a33"       # 誤検出
C_FN = "#fdae61"       # 見逃し


def fig_pipeline(out: Path):
    """手法の全体図。入力2種 → 2本のエンコーダ → 融合 → 出力、と
    推論時に近傍を貼って9窓を混ぜること、を1枚で示す。"""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

    def box(x, y, w, h, text, sub="", fc="#eef2f7", ec="#5b6b7f", fs=9):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6", linewidth=0.8,
            facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), text,
                ha="center", va="center", fontsize=fs)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 2.6, sub, ha="center", va="center",
                    fontsize=fs - 1.5, color="#5b6b7f")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.9,
                                    color="#5b6b7f", shrinkA=0, shrinkB=0))

    box(1, 31, 20, 11, "傾斜量図 (SAM)", "1ch", fc="#e8eef5")
    box(1, 10, 20, 11, "航空写真 (APM)", "3ch", fc="#efe9e4")
    box(28, 31, 20, 11, "エンコーダ", "64→512")
    box(28, 10, 20, 11, "エンコーダ", "64→512")
    box(54, 20, 17, 13, "融合", "連結 1024", fc="#e4eee7")
    box(77, 20, 20, 13, "デコーダ", "→ 確率マップ", fc="#e4eee7")

    arrow(21, 36.5, 28, 36.5)
    arrow(21, 15.5, 28, 15.5)
    arrow(48, 36.5, 54, 29)
    arrow(48, 15.5, 54, 24)
    arrow(71, 26.5, 77, 26.5)

    ax.text(50, 47, "タイル 128×128 画素（573 m 四方、4.48 m/画素）",
            ha="center", fontsize=9, color="#33404f")
    ax.text(50, 3.5,
            "推論時は近傍8枚を貼った 3×3 から 64 画素ずつずらした 9 窓を推論し、"
            "中心重みの余弦窓で混ぜる",
            ha="center", fontsize=8.5, color="#33404f")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  書き出し:", out.name)


def _pick_window(pred_path, size=1400, step=8, band=2):
    """正解が多い場所を間引き読みで探す。全面を読まない。

    **バンドの意味に注意。** 本体の GeoTIFF は
      バンド1 = 抽出確率 0-100 / バンド2 = 正解 0/1 / バンド3 = 学習フラグ
    であり（build_paper.py の表VIII と GeoTIFF の DESCRIPTION が一致）、
    別ファイルの `__mask.tif` は**二値化した予測**である
    （DESCRIPTION は確率と書かれているが中身は 0/1 で、バンド1>50 と一致する）。
    最初これを取り違えて、正解と予測を入れ替えた図を作った。

    **nodata の扱いにも注意。** 0=区域外 / 1=区域 / 255=nodata なので、
    `> 0` で判定すると nodata を区域として数えてしまう。
    """
    import rasterio
    with rasterio.open(pred_path) as src:
        h, w = src.height, src.width
        small = src.read(band, out_shape=(h // step, w // step))
        nod = src.nodatavals[band - 1]
    nod = 255 if nod is None else int(nod)
    k = size // step
    best, pos = -1, (0, 0)
    for r in range(0, small.shape[0] - k, max(k // 4, 1)):
        for c in range(0, small.shape[1] - k, max(k // 4, 1)):
            blk = small[r:r + k, c:c + k]
            # nodata が多い窓は捨てる（海や県外）
            if float((blk == nod).mean()) > 0.05:
                continue
            v = int((blk == 1).sum())
            if v > best:
                best, pos = v, (r * step, c * step)
    if best < 0:
        raise RuntimeError("nodata の少ない窓が見つかりません")
    return pos, best


def fig_output(out: Path, name="Ens_AllStrong14", region="shimane", size=1400):
    """予測と正解の差分を実際の地図上で見せる。GeoTIFF の窓読みだけ。"""
    import rasterio
    from rasterio.windows import Window
    from dc5lib.paths import data_root

    hm = data_root() / "hazard_map"
    tif = hm / f"{name}__{region}.tif"
    if not tif.exists():
        sys.exit(f"{tif} がありません")

    (r0, c0), npix = _pick_window(tif, size)
    print(f"  窓: 行 {r0} 列 {c0} / {size}x{size}（間引きで見つけた正解 {npix} 画素）")
    win = Window(c0, r0, size, size)
    # 本体の1ファイルだけで済ませる。バンド1=確率 / バンド2=正解 / 255=nodata
    with rasterio.open(tif) as src:
        prob = src.read(1, window=win)
        truth = src.read(2, window=win)
        # バンド3 = 学習に使ったタイルか。**論文の表VIIIの警告のとおり、
        # 広島の地図には学習済みタイルが混ざるので、そのまま性能として出せない。**
        used = src.read(3, window=win) if src.count >= 3 else None
    valid = (truth != 255) & (prob != 255)
    if used is not None:
        frac = float((used[valid] == 1).mean())
        print(f"  学習に使ったタイルの画素 {100 * frac:.1f}%")
        if frac > 0.01:
            print("  ※ 学習済みタイルが混ざっています。"
                  "論文に載せるなら島根（--region shimane）を使うこと")
    gt = (truth == 1) & valid
    pr = (prob > 50) & valid          # しきい値0.5
    print(f"  有効画素 {100 * valid.mean():.1f}% / 正解 {int(gt.sum()):,} / "
          f"予測 {int(pr.sum()):,}")
    print(f"  当たり {int((gt & pr).sum()):,} / 誤検出 {int((pr & ~gt).sum()):,} / "
          f"見逃し {int((gt & ~pr).sum()):,}")

    diff = np.zeros(gt.shape + (3,), np.float32) + 1.0
    for sel, col in ((gt & pr, C_TP), (pr & ~gt, C_FP), (gt & ~pr, C_FN)):
        rgb = np.array([int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        diff[sel] = rgb

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, img, ttl in ((axes[0], gt, "正解（指定区域）"),
                         (axes[1], pr, "予測"),
                         (axes[2], diff, "差分")):
        ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "Greys",
                  interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[2].legend(handles=[
        mpatches.Patch(color=C_TP, label="当たり"),
        mpatches.Patch(color=C_FP, label="誤検出"),
        mpatches.Patch(color=C_FN, label="見逃し")],
        fontsize=7, loc="lower right", framealpha=0.9)
    km = size * 4.48 / 1000
    fig.suptitle(f"{km:.1f} km 四方の抜き出し（{region}・{name}）", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  書き出し:", out.name)


def fig_inputs(out: Path, condition="FinalFusion_SAM_APM", n=3):
    """入力データの例。**pkl を読むので学習中は避ける。**"""
    from analysis.data import load_pkl, preprocess_mask
    from dc5lib.paths import dataset_path
    from dc5lib.registry import get

    cond = get(condition)
    p = dataset_path("Hiroshima", cond.pkl_for("Hiroshima"))
    print(f"  読み込み: {p.name}（{p.stat().st_size / 2**30:.1f} GB）…", flush=True)
    ds = load_pkl(p)
    # 正解が程よく入っているタイルを選ぶ
    cand = []
    for k in range(len(ds.No)):
        m = np.asarray(ds.Mask[k])
        r = float((m > 0).mean())
        if 0.08 < r < 0.35:
            cand.append((abs(r - 0.18), k))
        if len(cand) > 4000:
            break
    cand.sort()
    picks = [k for _, k in cand[:n]]
    print("  選んだタイル:", [int(ds.No[k]) for k in picks])

    # 傾斜角の色は全タイルで共通にする（タイルごとに伸縮すると比べられない）
    vmax = float(np.percentile(
        np.concatenate([np.asarray(ds.DEM[k], np.float32).ravel() for k in picks]), 99))
    vmax = max(10.0, round(vmax / 5) * 5)
    print(f"  傾斜角の表示範囲: 0〜{vmax:.0f} 度")

    fig, axes = plt.subplots(n, 3, figsize=(6.4, 2.2 * n))
    axes = np.atleast_2d(axes)
    im0 = None
    for i, k in enumerate(picks):
        sam = np.asarray(ds.DEM[k], np.float32)
        air = np.asarray(ds.AirPhoto[k], np.uint8)
        gt = preprocess_mask(ds.Mask[k]) > 0
        im0 = axes[i, 0].imshow(sam, cmap="YlOrRd", vmin=0, vmax=vmax)
        axes[i, 1].imshow(air)
        axes[i, 2].imshow(air)
        axes[i, 2].imshow(np.dstack([gt, np.zeros_like(gt), np.zeros_like(gt),
                                     gt * 0.45]).astype(float))
        for j, ttl in enumerate(("傾斜量図 (SAM)", "航空写真 (APM)", "正解マスク")):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(ttl, fontsize=9)
        axes[i, 0].set_ylabel(f"タイル {int(ds.No[k])}", fontsize=8)
    fig.suptitle("入力データの例（1タイル = 128×128 画素 = 573 m 四方）", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # 傾斜角の凡例。単位が分からないと図の意味が伝わらない
    cax = fig.add_axes([0.13, 0.035, 0.22, 0.014])
    cb = fig.colorbar(im0, cax=cax, orientation="horizontal")
    cb.set_label("傾斜角［度］", fontsize=7.5, labelpad=1)
    cb.ax.tick_params(labelsize=6.5, length=2)
    fig.text(0.68, 0.035, "赤＝急傾斜地崩壊危険区域（正解）",
             fontsize=7.5, ha="center")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  書き出し:", out.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="保存先フォルダ")
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--output", action="store_true")
    ap.add_argument("--inputs", action="store_true",
                    help="pkl を 5.2GB 読む。学習中は避ける")
    ap.add_argument("--name", default="Ens_AllStrong14")
    ap.add_argument("--region", default="shimane",
                    help="論文用は島根。広島は学習済みタイルが混ざる")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path("docs/論文/figs")
    out.mkdir(parents=True, exist_ok=True)
    if not (args.pipeline or args.output or args.inputs):
        sys.exit("--pipeline / --output / --inputs のいずれかを指定してください")
    if args.pipeline:
        print("手法の全体図")
        fig_pipeline(out / "fig_pipeline.png")
    if args.output:
        print("出力例（GeoTIFF の窓読み）")
        fig_output(out / "fig_output_example.png", args.name, args.region)
    if args.inputs:
        print("入力データの例（pkl を読みます）")
        fig_inputs(out / "fig_inputs.png")


if __name__ == "__main__":
    main()
