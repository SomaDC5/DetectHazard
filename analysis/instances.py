# -*- coding: utf-8 -*-
"""箇所数評価（instance-level evaluation）。

面積評価が「画素をいくつ当てたか」を測るのに対して、箇所数評価は
「警戒区域という“かたまり”をいくつ見つけられたか」を測る。

-------------------------------------------------------------------------
なぜ素直に作ると面積評価より悪くなるのか
-------------------------------------------------------------------------
物体検出でよく使う「IoU でしきい値判定して 1 対 1 で対応づける」方式を
そのまま持ってくると、この課題では二重に損をする。

  (1) 飲み込み（merge）
      予測が広めに出るため、近接した正解 3〜4 個を 1 つの予測領域が
      まとめて覆ってしまう。1 対 1 対応では、そのうち 1 個しか TP に
      できず、残りは FN になる。

  (2) IoU がそもそも上がらない
      予測が正解より大きいと、1 個だけ対応づけた相手との IoU も
      |g| / |p| 程度まで落ちる。4 個ぶんの広さの予測なら IoU は 0.25 前後で、
      しきい値 0.3 を割る。結果として「4 個とも FN、さらに FP が 1 個」
      という、実態とかけ離れた集計になる。

現場の用途（抽出結果を見て現地調査の当たりをつける）で考えると、
1 つの予測領域が本物の警戒区域 4 個を覆っているのは**見逃しではない**。
そこへ行けば 4 個とも見つかる。逆に、広く塗りすぎて中身がほとんど
背景であれば、それは無駄足なので precision 側で罰すべきである。

そこで本モジュールでは、対応づけの決め方を 3 通り実装して切り替えられる
ようにした。どれが妥当かを数字で見比べてから選べるようにするのが狙い。

-------------------------------------------------------------------------
3 つの対応づけ方式
-------------------------------------------------------------------------
記号：正解の箇所を g_1..g_m、予測の箇所を p_1..p_n、重なり画素数を O[i,j]。

  mode="coverage"  （被覆方式・推奨）
      正解 g_i は、予測全体 P との重なりが自身の cover_gt 以上なら検出とみなす。
          |g_i ∩ P| / |g_i| >= cover_gt
      予測 p_j は、自身のうち正解全体 G と重なる割合が cover_pred 以上なら的中。
          |p_j ∩ G| / |p_j| >= cover_pred
      「どの予測が覆ったか」を問わないので飲み込みで損をしない。
      広く塗りすぎた予測は cover_pred を割って FP になる。

  mode="iou_many"  （IoU・多対1を許す）
      正解 g_i は max_j IoU(g_i, p_j) >= iou_thr なら検出。
      予測 p_j は max_i IoU(g_i, p_j) >= iou_thr なら的中。
      同じ予測が複数の正解に対応してよい（飲み込みの (1) は解消する）。
      ただし (2) の「大きすぎると IoU が上がらない」問題は残る。

  mode="iou_1to1"  （IoU・厳密1対1／COCO 流）
      IoU の大きい組から貪欲に 1 対 1 で確定し、iou_thr 以上を TP とする。
      前任者の実装がこれに近いと思われる。比較用に残してある。

いずれの方式でも
      箇所Recall    = 検出できた正解の箇所数 / 正解の箇所数
      箇所Precision = 的中した予測の箇所数   / 予測の箇所数
      箇所F値       = その調和平均
とする。coverage と iou_many では、TP の数え方が正解側と予測側で別々に
なる（1 つの予測が 4 個の正解を覆えば、正解側 TP は 4、予測側 TP は 1）。
これは「多対 1 を許す」以上は避けられないので、そのまま扱う。

-------------------------------------------------------------------------
そのほか結果を左右する決めごと
-------------------------------------------------------------------------
* 連結性 connectivity … 4 近傍か 8 近傍か。斜めに接した画素を同じかたまりと
  みなすかどうかで箇所数が変わる。既定は 8。
* 最小サイズ min_size … 数画素のごみを箇所として数えない。予測側のごみは
  FP を、正解側のごみ（ガウシアン処理の副産物）は FN を水増しするので、
  両方に同じしきい値を掛ける。既定は 10 画素。
* タイル境界 … 本モジュールはタイル単位で数える。タイルの縁で切れた
  警戒区域は、隣り合うタイルで別の箇所として数えられる。地域全体で
  モザイクしてから数えるのが本来だが、まずはタイル単位で揃える。

torch に依存しないので、そのまま単体で検証できる。
`python -m tileanalysis.instances` で、飲み込みを含む合成例に対して
3 方式がそれぞれどう数えるかを表示する。
"""

import numpy as np
from scipy import ndimage

# 8 近傍（斜めもつなぐ）
STRUCT8 = np.ones((3, 3), dtype=bool)
# 4 近傍（上下左右のみ）
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

DEFAULTS = dict(
    connectivity=8,
    min_size=10,
    iou_thr=0.3,
    cover_gt=0.5,
    cover_pred=0.3,
)


def label_instances(mask, connectivity=8, min_size=0):
    """二値マスクを「かたまり」に分ける。

    戻り値: (ラベル画像, 各かたまりの画素数の配列)
    ラベルは 1..n。min_size 未満のかたまりは 0（背景）に落とし、
    残ったものを 1 から振り直す。
    """
    st = STRUCT8 if connectivity == 8 else STRUCT4
    lab, n = ndimage.label(np.asarray(mask, dtype=bool), structure=st)
    if n == 0:
        return lab, np.zeros(0, dtype=np.int64)

    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    if min_size > 0 and (sizes < min_size).any():
        keep = np.where(sizes >= min_size)[0] + 1          # 残すラベル番号
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        lab = remap[lab]
        sizes = sizes[keep - 1]
    return lab, sizes.astype(np.int64)


def overlap_matrix(gt_lab, n_gt, pred_lab, n_pred):
    """O[i, j] = 正解 i と予測 j が重なる画素数（i, j は 0 始まり）。"""
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.int64)
    both = (gt_lab > 0) & (pred_lab > 0)
    idx = (gt_lab[both].astype(np.int64) - 1) * n_pred + (pred_lab[both].astype(np.int64) - 1)
    counts = np.bincount(idx, minlength=n_gt * n_pred)
    return counts.reshape(n_gt, n_pred)


def _greedy_one_to_one(iou, thr):
    """IoU の大きい組から貪欲に 1 対 1 で確定する（COCO 流）。"""
    m, n = iou.shape
    if m == 0 or n == 0:
        return 0
    order = np.dstack(np.unravel_index(np.argsort(iou, axis=None)[::-1], iou.shape))[0]
    used_g, used_p, matched = set(), set(), 0
    for i, j in order:
        if iou[i, j] < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(int(i))
        used_p.add(int(j))
        matched += 1
    return matched


def evaluate_tile(gt_mask, pred_mask, mode="coverage", connectivity=8, min_size=10,
                  iou_thr=0.3, cover_gt=0.5, cover_pred=0.3, with_diagnostics=True):
    """1 タイルぶんの箇所数評価。

    戻り値は dict。主な項目

        n_gt / n_pred      正解・予測の箇所数
        tp_gt              検出できた正解の箇所数（Recall の分子）
        tp_pred            的中した予測の箇所数（Precision の分子）
        fn                 n_gt - tp_gt
        fp                 n_pred - tp_pred

    with_diagnostics=True のとき、飲み込み・分裂の診断値も返す。

        merged_pred        2 個以上の正解に重なった予測の数
        gt_in_merge        そうした予測に飲み込まれた正解の数
        max_gt_per_pred    1 つの予測が重なった正解の最大数
        split_gt           2 個以上の予測に分かれた正解の数
        max_pred_per_gt    1 つの正解に重なった予測の最大数
        area_ratio         予測の総画素数 / 正解の総画素数
        mean_best_iou      正解ごとの最良 IoU の平均
    """
    gt_lab, gt_sizes = label_instances(gt_mask, connectivity, min_size)
    pred_lab, pred_sizes = label_instances(pred_mask, connectivity, min_size)
    m, n = len(gt_sizes), len(pred_sizes)

    out = dict(n_gt=m, n_pred=n, tp_gt=0, tp_pred=0, fn=m, fp=n)
    if with_diagnostics:
        out.update(merged_pred=0, gt_in_merge=0, max_gt_per_pred=0,
                   split_gt=0, max_pred_per_gt=0,
                   area_ratio=np.nan, mean_best_iou=np.nan,
                   gt_pixels=int(gt_sizes.sum()), pred_pixels=int(pred_sizes.sum()))
        if m > 0:
            out["area_ratio"] = float(pred_sizes.sum()) / float(gt_sizes.sum())
    if m == 0 or n == 0:
        return out

    O = overlap_matrix(gt_lab, m, pred_lab, n)
    union = gt_sizes[:, None] + pred_sizes[None, :] - O
    iou = np.where(union > 0, O / np.maximum(union, 1), 0.0)

    if mode == "coverage":
        cov_g = O.sum(axis=1) / gt_sizes                 # |g_i ∩ P| / |g_i|
        cov_p = O.sum(axis=0) / pred_sizes               # |p_j ∩ G| / |p_j|
        out["tp_gt"] = int((cov_g >= cover_gt).sum())
        out["tp_pred"] = int((cov_p >= cover_pred).sum())
    elif mode == "iou_many":
        out["tp_gt"] = int((iou.max(axis=1) >= iou_thr).sum())
        out["tp_pred"] = int((iou.max(axis=0) >= iou_thr).sum())
    elif mode == "iou_1to1":
        matched = _greedy_one_to_one(iou, iou_thr)
        out["tp_gt"] = out["tp_pred"] = matched
    else:
        raise ValueError(f"未知の mode: {mode}")

    out["fn"] = m - out["tp_gt"]
    out["fp"] = n - out["tp_pred"]

    if with_diagnostics:
        touch = O > 0
        gt_per_pred = touch.sum(axis=0)                  # 各予測が重なる正解の数
        pred_per_gt = touch.sum(axis=1)                  # 各正解が重なる予測の数
        merged = gt_per_pred >= 2
        out["merged_pred"] = int(merged.sum())
        out["gt_in_merge"] = int(touch[:, merged].any(axis=1).sum())
        out["max_gt_per_pred"] = int(gt_per_pred.max())
        out["split_gt"] = int((pred_per_gt >= 2).sum())
        out["max_pred_per_gt"] = int(pred_per_gt.max())
        out["mean_best_iou"] = float(iou.max(axis=1).mean())
    return out


def evaluate_tile_multi(gt_mask, pred_mask, settings, connectivity=8, min_size=10):
    """同じタイルに対して複数の設定をまとめて評価する。

    ラベリングと重なり行列は 1 回だけ計算するので、しきい値を振っても
    ほとんど追加コストがかからない。推論をやり直さずに条件を比較したいので
    こちらを使う。

    settings: {設定名: {"mode": ..., "iou_thr": ..., "cover_gt": ..., "cover_pred": ...}}
    戻り値  : 診断値 + 設定名ごとの tp_gt / tp_pred / fn / fp
    """
    gt_lab, gt_sizes = label_instances(gt_mask, connectivity, min_size)
    pred_lab, pred_sizes = label_instances(pred_mask, connectivity, min_size)
    m, n = len(gt_sizes), len(pred_sizes)

    out = dict(n_gt=m, n_pred=n,
               gt_pixels=int(gt_sizes.sum()), pred_pixels=int(pred_sizes.sum()),
               merged_pred=0, gt_in_merge=0, max_gt_per_pred=0,
               split_gt=0, max_pred_per_gt=0,
               area_ratio=np.nan, mean_best_iou=np.nan)
    if m > 0 and gt_sizes.sum() > 0:
        out["area_ratio"] = float(pred_sizes.sum()) / float(gt_sizes.sum())

    if m == 0 or n == 0:
        for name in settings:
            out[f"{name}__tp_gt"] = 0
            out[f"{name}__tp_pred"] = 0
            out[f"{name}__fn"] = m
            out[f"{name}__fp"] = n
        return out

    O = overlap_matrix(gt_lab, m, pred_lab, n)
    union = gt_sizes[:, None] + pred_sizes[None, :] - O
    iou = np.where(union > 0, O / np.maximum(union, 1), 0.0)
    cov_g = O.sum(axis=1) / gt_sizes
    cov_p = O.sum(axis=0) / pred_sizes
    best_iou_g = iou.max(axis=1)
    best_iou_p = iou.max(axis=0)

    touch = O > 0
    gt_per_pred = touch.sum(axis=0)
    pred_per_gt = touch.sum(axis=1)
    merged = gt_per_pred >= 2
    out.update(merged_pred=int(merged.sum()),
               gt_in_merge=int(touch[:, merged].any(axis=1).sum()),
               max_gt_per_pred=int(gt_per_pred.max()),
               split_gt=int((pred_per_gt >= 2).sum()),
               max_pred_per_gt=int(pred_per_gt.max()),
               mean_best_iou=float(best_iou_g.mean()))

    for name, cfg in settings.items():
        mode = cfg["mode"]
        if mode == "coverage":
            tp_gt = int((cov_g >= cfg.get("cover_gt", 0.5)).sum())
            tp_pred = int((cov_p >= cfg.get("cover_pred", 0.3)).sum())
        elif mode == "iou_many":
            thr = cfg.get("iou_thr", 0.3)
            tp_gt = int((best_iou_g >= thr).sum())
            tp_pred = int((best_iou_p >= thr).sum())
        elif mode == "iou_1to1":
            tp_gt = tp_pred = _greedy_one_to_one(iou, cfg.get("iou_thr", 0.3))
        else:
            raise ValueError(f"未知の mode: {mode}")
        out[f"{name}__tp_gt"] = tp_gt
        out[f"{name}__tp_pred"] = tp_pred
        out[f"{name}__fn"] = m - tp_gt
        out[f"{name}__fp"] = n - tp_pred
    return out


def per_gt_instance(gt_mask, pred_mask, connectivity=8, min_size=10):
    """正解の箇所を 1 行ずつ書き出す（レビュー・しきい値検討用）。

    列
        gt_area        その箇所の画素数
        coverage       |g ∩ 予測全体| / |g|　… 被覆方式の Recall 判定に使う値
        best_iou       いちばん重なった予測との IoU　… IoU 方式の判定に使う値
        best_pred_area その予測の画素数
        n_pred_touch   重なった予測の数（2 以上なら分裂）
        merged         いちばん重なった予測が、他の正解にも重なっているか
        n_gt_in_pred   その予測が覆っている正解の数（飲み込みの程度）
    """
    gt_lab, gt_sizes = label_instances(gt_mask, connectivity, min_size)
    pred_lab, pred_sizes = label_instances(pred_mask, connectivity, min_size)
    m, n = len(gt_sizes), len(pred_sizes)
    if m == 0:
        return []
    if n == 0:
        return [dict(gt_area=int(a), coverage=0.0, best_iou=0.0, best_pred_area=0,
                     n_pred_touch=0, merged=False, n_gt_in_pred=0) for a in gt_sizes]

    O = overlap_matrix(gt_lab, m, pred_lab, n)
    union = gt_sizes[:, None] + pred_sizes[None, :] - O
    iou = np.where(union > 0, O / np.maximum(union, 1), 0.0)
    gt_per_pred = (O > 0).sum(axis=0)

    rows = []
    for i in range(m):
        j = int(iou[i].argmax())
        has_overlap = O[i].sum() > 0
        rows.append(dict(
            gt_area=int(gt_sizes[i]),
            coverage=float(O[i].sum() / gt_sizes[i]),
            best_iou=float(iou[i, j]),
            best_pred_area=int(pred_sizes[j]) if has_overlap else 0,
            n_pred_touch=int((O[i] > 0).sum()),
            merged=bool(has_overlap and gt_per_pred[j] >= 2),
            n_gt_in_pred=int(gt_per_pred[j]) if has_overlap else 0,
        ))
    return rows


def summarize_setting(df, name):
    """evaluate_tile_multi の結果から、1 設定ぶんの箇所 Recall / Precision / F値。"""
    n_gt, n_pred = df["n_gt"].sum(), df["n_pred"].sum()
    tp_gt, tp_pred = df[f"{name}__tp_gt"].sum(), df[f"{name}__tp_pred"].sum()
    rec = tp_gt / n_gt if n_gt else np.nan
    pre = tp_pred / n_pred if n_pred else np.nan
    f1 = (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else (np.nan if not (n_gt or n_pred) else 0.0)
    return {"設定": name, "箇所_正解数": int(n_gt), "箇所_予測数": int(n_pred),
            "検出できた正解": int(tp_gt), "的中した予測": int(tp_pred),
            "箇所Recall": round(float(rec), 4), "箇所Precision": round(float(pre), 4),
            "箇所F値": round(float(f1), 4)}


def summarize(rows):
    """タイルごとの dict を足し上げて、箇所 Recall / Precision / F値 にする。"""
    import pandas as pd

    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    n_gt, n_pred = df["n_gt"].sum(), df["n_pred"].sum()
    tp_gt, tp_pred = df["tp_gt"].sum(), df["tp_pred"].sum()
    rec = tp_gt / n_gt if n_gt else np.nan
    pre = tp_pred / n_pred if n_pred else np.nan
    f1 = (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else (np.nan if not (n_gt or n_pred) else 0.0)
    res = {
        "箇所_正解数": int(n_gt), "箇所_予測数": int(n_pred),
        "検出できた正解": int(tp_gt), "的中した予測": int(tp_pred),
        "見逃しFN": int(n_gt - tp_gt), "過検出FP": int(n_pred - tp_pred),
        "箇所Recall": round(float(rec), 4), "箇所Precision": round(float(pre), 4),
        "箇所F値": round(float(f1), 4),
    }
    for k, ja in (("merged_pred", "飲み込んだ予測の数"), ("gt_in_merge", "飲み込まれた正解の数"),
                  ("split_gt", "分裂した正解の数")):
        if k in df:
            res[ja] = int(df[k].sum())
    if "area_ratio" in df:
        res["予測面積/正解面積"] = round(float(df["area_ratio"].replace(
            [np.inf, -np.inf], np.nan).dropna().mean()), 4)
    if "mean_best_iou" in df:
        res["正解ごとの最良IoU平均"] = round(float(df["mean_best_iou"].dropna().mean()), 4)
    return res


# ----------------------------------------------------------------------
# 合成例による動作確認（python -m tileanalysis.instances）
# ----------------------------------------------------------------------
def _demo():
    def blank():
        return np.zeros((128, 128), dtype=np.uint8)

    cases = {}

    # ① ちょうど当たっている：正解3個・予測3個がほぼ一致
    gt, pr = blank(), blank()
    for cx in (24, 64, 104):
        gt[50:70, cx - 10:cx + 10] = 1
        pr[48:72, cx - 12:cx + 12] = 1
    cases["① 3個をそれぞれ当てる"] = (gt, pr, "理想。どの方式でも Recall=Precision=1 に近いはず")

    # ② 飲み込み：正解4個を1つの大きな予測が覆う（今回の仮説そのもの）
    gt, pr = blank(), blank()
    for cx in (30, 55, 80, 105):
        gt[50:70, cx - 8:cx + 8] = 1
        pr[45:75, 20:115] = 1
    cases["② 4個を1つの予測が飲み込む"] = (
        gt, pr, "1対1だと3個がFNになる。多対1・被覆なら4個とも検出扱い")

    # ③ 飲み込み＋塗りすぎ：②よりさらに広く塗る
    gt, pr = blank(), blank()
    for cx in (30, 55, 80, 105):
        gt[50:70, cx - 8:cx + 8] = 1
    pr[20:110, 10:120] = 1
    cases["③ 飲み込み＋大幅に塗りすぎ"] = (
        gt, pr, "被覆方式なら Recall は高いが Precision で罰せられるべき")

    # ④ 分裂：1個の正解を予測が3つに割る
    gt, pr = blank(), blank()
    gt[40:90, 40:90] = 1
    for y in (45, 60, 75):
        pr[y:y + 10, 45:85] = 1
    cases["④ 1個の正解を3つに割る"] = (gt, pr, "予測側の FP が増えるのが妥当")

    # ⑤ 完全な外し：重なりゼロ
    gt, pr = blank(), blank()
    gt[20:40, 20:40] = 1
    pr[80:100, 80:100] = 1
    cases["⑤ まったく重ならない"] = (gt, pr, "Recall=0, Precision=0")

    modes = ["coverage", "iou_many", "iou_1to1"]
    print("=" * 96)
    print("合成例での挙動（min_size=10, connectivity=8, iou_thr=0.3, cover_gt=0.5, cover_pred=0.3）")
    print("=" * 96)
    for name, (gt, pr, note) in cases.items():
        print(f"\n{name}   — {note}")
        head = f"  {'方式':<10}{'正解':>5}{'予測':>5}{'検出':>5}{'的中':>5}" \
               f"{'Recall':>9}{'Prec':>8}{'F値':>8}"
        print(head)
        for mode in modes:
            r = evaluate_tile(gt, pr, mode=mode)
            rec = r["tp_gt"] / r["n_gt"] if r["n_gt"] else float("nan")
            pre = r["tp_pred"] / r["n_pred"] if r["n_pred"] else float("nan")
            f1 = 2 * rec * pre / (rec + pre) if rec and pre else 0.0
            print(f"  {mode:<10}{r['n_gt']:>5}{r['n_pred']:>5}{r['tp_gt']:>5}{r['tp_pred']:>5}"
                  f"{rec:>9.3f}{pre:>8.3f}{f1:>8.3f}")
        d = evaluate_tile(gt, pr, mode="coverage")
        print(f"  診断: 飲み込んだ予測 {d['merged_pred']} / 飲み込まれた正解 {d['gt_in_merge']} "
              f"/ 1予測が覆う正解の最大 {d['max_gt_per_pred']} / 分裂した正解 {d['split_gt']} "
              f"/ 面積比 {d['area_ratio']:.2f} / 最良IoU平均 {d['mean_best_iou']:.3f}")


if __name__ == "__main__":
    _demo()
