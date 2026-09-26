# -*- coding: utf-8 -*-
"""箇所数評価（研究用PCへの持ち込み用・単体で動く）

numpy と scipy だけで動く。torch には依存しないので、ノートブックの
評価セルから予測を numpy 配列で渡すだけで使える。

-------------------------------------------------------------------------
なぜ素直に作ると面積評価より悪くなるのか
-------------------------------------------------------------------------
物体検出でよく使う「IoU でしきい値判定して 1 対 1 で対応づける」方式は、
この課題では二重に損をする。

  (1) 飲み込み … 予測が広めに出るため、近接した正解を 1 つの予測が
      まとめて覆う。1 対 1 ではそのうち 1 個しか TP にできない。
  (2) IoU が上がらない … 正解が小さいと少しのずれで IoU が 0.3 を割る。

広島テスト（Final SAM+APM）で実測した結果。

  正解 8,977 箇所のうち、飲み込まれた正解 1,803 個（20.1 %）
  1 つの予測が覆う正解の最大 8 個
  飲み込まれた正解が IoU>=0.3 を通る率 37.7 %  一方 被覆>=0.5 を通る率 89.9 %

  箇所F値  IoU厳密1対1(0.3)  0.5432   ← 面積評価 0.6727 より 0.13 低い
           IoU多対1(0.3)     0.5477   ← 多対1を許すだけではほぼ改善しない
           被覆(0.5/0.3)     0.6506
           被覆(0.3/0.3)     0.6819

現場の用途（抽出結果を見て現地調査の当たりをつける）では、1 つの予測領域が
本物の警戒区域 4 個を覆っているのは見逃しではない。そこへ行けば 4 個とも
見つかる。逆に広く塗りすぎて中身がほとんど背景なら無駄足なので、
precision 側で罰するのが筋。これを実装したのが被覆方式。

-------------------------------------------------------------------------
被覆方式の定義
-------------------------------------------------------------------------
正解の箇所 g_1..g_m、予測の箇所 p_1..p_n、予測全体を P、正解全体を G とする。

    正解 g_i を検出した  <=>  |g_i ∩ P| / |g_i| >= cover_gt      （既定 0.5）
    予測 p_j が的中した  <=>  |p_j ∩ G| / |p_j| >= cover_pred    （既定 0.3）

    箇所Recall    = 検出できた正解の箇所数 / 正解の箇所数
    箇所Precision = 的中した予測の箇所数   / 予測の箇所数
    箇所F値       = その調和平均

1 つの予測が 4 個の正解を覆うと、正解側 TP は 4、予測側 TP は 1 になる。
多対 1 を許す以上これは避けられないので、そのまま扱う。

比較用に IoU 多対1 と IoU 厳密1対1 も実装してある。

-------------------------------------------------------------------------
被覆方式は「塗り広げ」に弱くないか
-------------------------------------------------------------------------
予測を膨張させて実測した（Final SAM+APM・広島）。

    膨張     面積F     箇所F(cover_gt=0.5)  箇所F(cover_gt=0.3)  予測/正解面積
     0 px   0.6727    0.6506               0.6819               1.39
     2 px   0.6531    0.6596  ← 最大        0.6791               1.87
     4 px   0.6160    0.6556               0.6695               2.39
     8 px   0.5415    0.5996               0.6069               3.51
    16 px   0.4329    0.4260               0.4286               6.00

cover_gt=0.5 だと 2 px 膨張のところに浅い山がある（+0.009）。ただし同時に
面積F は 0.020 下がるので、**面積評価と必ず並記すること**。片方だけを見て
調整すると、予測を広げる方向に少し引っ張られる。
cover_gt=0.3 なら単調減少で、この穴はない。

-------------------------------------------------------------------------
使い方
-------------------------------------------------------------------------
    python instance_eval.py        # 合成例で 3 方式の挙動を表示

ノートブックの評価セルから:

    from instance_eval import InstanceEvaluator

    ev = InstanceEvaluator(save_gt_instances=True)
    model.eval()
    with torch.no_grad():
        for (dem, air), masks in test_loader:
            out = model(dem.to(device), air.to(device))
            pred = (torch.sigmoid(out) > 0.5).cpu().numpy()[:, 0]
            gt   = (masks.numpy()[:, 0] > 0.5)
            ev.add(gt, pred)

    print(ev.summary_table())          # 方式ごとの箇所 R / P / F
    print(ev.diagnostics())            # 飲み込み・分裂・面積比
    ev.to_csv("instance_eval")         # タイル単位と箇所単位の CSV
"""

import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

# 既定で比較する設定
DEFAULT_SETTINGS = {
    "被覆 (cov_gt=0.5, cov_pred=0.3)": dict(mode="coverage", cover_gt=0.5, cover_pred=0.3),
    "被覆 (cov_gt=0.3, cov_pred=0.3)": dict(mode="coverage", cover_gt=0.3, cover_pred=0.3),
    "被覆 (cov_gt=0.5, cov_pred=0.1)": dict(mode="coverage", cover_gt=0.5, cover_pred=0.1),
    "IoU多対1 (0.3)": dict(mode="iou_many", iou_thr=0.3),
    "IoU多対1 (0.1)": dict(mode="iou_many", iou_thr=0.1),
    "IoU厳密1対1 (0.3)": dict(mode="iou_1to1", iou_thr=0.3),
    "IoU厳密1対1 (0.1)": dict(mode="iou_1to1", iou_thr=0.1),
}
PRIMARY = "被覆 (cov_gt=0.5, cov_pred=0.3)"


# ----------------------------------------------------------------------
# 基本部品
# ----------------------------------------------------------------------
def label_instances(mask, connectivity=8, min_size=10):
    """二値マスクを「かたまり」に分ける。min_size 未満は数えない。"""
    st = STRUCT8 if connectivity == 8 else STRUCT4
    lab, n = ndimage.label(np.asarray(mask, dtype=bool), structure=st)
    if n == 0:
        return lab, np.zeros(0, dtype=np.int64)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    if min_size > 0 and (sizes < min_size).any():
        keep = np.where(sizes >= min_size)[0] + 1
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        lab = remap[lab]
        sizes = sizes[keep - 1]
    return lab, sizes.astype(np.int64)


def overlap_matrix(gt_lab, n_gt, pred_lab, n_pred):
    """O[i, j] = 正解 i と予測 j が重なる画素数。"""
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.int64)
    both = (gt_lab > 0) & (pred_lab > 0)
    idx = (gt_lab[both].astype(np.int64) - 1) * n_pred + (pred_lab[both].astype(np.int64) - 1)
    return np.bincount(idx, minlength=n_gt * n_pred).reshape(n_gt, n_pred)


def _greedy_one_to_one(iou, thr):
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


def evaluate_tile(gt_mask, pred_mask, settings=None, connectivity=8, min_size=10):
    """1 タイルぶん。ラベリングと重なり行列は 1 回だけ計算して全設定に使い回す。"""
    settings = settings or DEFAULT_SETTINGS
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
        return out, []

    O = overlap_matrix(gt_lab, m, pred_lab, n)
    union = gt_sizes[:, None] + pred_sizes[None, :] - O
    iou = np.where(union > 0, O / np.maximum(union, 1), 0.0)
    cov_g = O.sum(axis=1) / gt_sizes
    cov_p = O.sum(axis=0) / pred_sizes
    best_g, best_p = iou.max(axis=1), iou.max(axis=0)

    touch = O > 0
    gt_per_pred = touch.sum(axis=0)
    pred_per_gt = touch.sum(axis=1)
    merged = gt_per_pred >= 2
    out.update(merged_pred=int(merged.sum()),
               gt_in_merge=int(touch[:, merged].any(axis=1).sum()),
               max_gt_per_pred=int(gt_per_pred.max()),
               split_gt=int((pred_per_gt >= 2).sum()),
               max_pred_per_gt=int(pred_per_gt.max()),
               mean_best_iou=float(best_g.mean()))

    for name, cfg in settings.items():
        mode = cfg["mode"]
        if mode == "coverage":
            tp_gt = int((cov_g >= cfg.get("cover_gt", 0.5)).sum())
            tp_pred = int((cov_p >= cfg.get("cover_pred", 0.3)).sum())
        elif mode == "iou_many":
            thr = cfg.get("iou_thr", 0.3)
            tp_gt = int((best_g >= thr).sum())
            tp_pred = int((best_p >= thr).sum())
        elif mode == "iou_1to1":
            tp_gt = tp_pred = _greedy_one_to_one(iou, cfg.get("iou_thr", 0.3))
        else:
            raise ValueError(f"未知の mode: {mode}")
        out[f"{name}__tp_gt"] = tp_gt
        out[f"{name}__tp_pred"] = tp_pred

    # 正解の箇所を 1 行ずつ（しきい値の検討・弱点の把握に使う）
    inst = []
    for i in range(m):
        j = int(iou[i].argmax())
        has = O[i].sum() > 0
        inst.append(dict(gt_area=int(gt_sizes[i]), coverage=float(cov_g[i]),
                         best_iou=float(iou[i, j]),
                         best_pred_area=int(pred_sizes[j]) if has else 0,
                         n_pred_touch=int(touch[i].sum()),
                         merged=bool(has and gt_per_pred[j] >= 2),
                         n_gt_in_pred=int(gt_per_pred[j]) if has else 0))
    return out, inst


# ----------------------------------------------------------------------
# 評価器
# ----------------------------------------------------------------------
class InstanceEvaluator:
    """バッチを add していき、最後に summary_table() で結果を見る。"""

    def __init__(self, settings=None, connectivity=8, min_size=10,
                 save_gt_instances=False):
        self.settings = settings or DEFAULT_SETTINGS
        self.connectivity = connectivity
        self.min_size = min_size
        self.save_gt_instances = save_gt_instances
        self.tiles = []
        self.instances = []

    def add(self, gt_batch, pred_batch, tile_nos=None):
        """gt_batch / pred_batch: (B, H, W) の bool もしくは 0/1。"""
        gt_batch = np.asarray(gt_batch)
        pred_batch = np.asarray(pred_batch)
        if gt_batch.ndim == 4:
            gt_batch = gt_batch[:, 0]
        if pred_batch.ndim == 4:
            pred_batch = pred_batch[:, 0]
        for k in range(len(gt_batch)):
            row, inst = evaluate_tile(gt_batch[k] > 0, pred_batch[k] > 0,
                                      self.settings, self.connectivity, self.min_size)
            if tile_nos is not None:
                row["tile_no"] = tile_nos[k]
            self.tiles.append(row)
            if self.save_gt_instances:
                for ir in inst:
                    if tile_nos is not None:
                        ir["tile_no"] = tile_nos[k]
                    self.instances.append(ir)

    # ---- 集計 --------------------------------------------------------
    def _frame(self):
        import pandas as pd
        return pd.DataFrame(self.tiles)

    def summary(self, name=PRIMARY):
        df = self._frame()
        n_gt, n_pred = df["n_gt"].sum(), df["n_pred"].sum()
        tp_gt, tp_pred = df[f"{name}__tp_gt"].sum(), df[f"{name}__tp_pred"].sum()
        rec = tp_gt / n_gt if n_gt else np.nan
        pre = tp_pred / n_pred if n_pred else np.nan
        f1 = (2 * rec * pre / (rec + pre)) if (rec + pre) > 0 else (np.nan if not (n_gt or n_pred) else 0.0)
        return {"設定": name, "箇所_正解数": int(n_gt), "箇所_予測数": int(n_pred),
                "検出できた正解": int(tp_gt), "的中した予測": int(tp_pred),
                "見逃しFN": int(n_gt - tp_gt), "過検出FP": int(n_pred - tp_pred),
                "箇所Recall": round(float(rec), 4),
                "箇所Precision": round(float(pre), 4), "箇所F値": round(float(f1), 4)}

    def summary_table(self):
        import pandas as pd
        return pd.DataFrame([self.summary(n) for n in self.settings])

    def diagnostics(self):
        df = self._frame()
        n_gt = max(int(df["n_gt"].sum()), 1)
        d = {
            "タイル数": len(df),
            "正解の箇所数": int(df["n_gt"].sum()),
            "予測の箇所数": int(df["n_pred"].sum()),
            "予測/正解 箇所数比": round(df["n_pred"].sum() / n_gt, 3),
            "飲み込んだ予測": int(df["merged_pred"].sum()),
            "飲み込まれた正解": int(df["gt_in_merge"].sum()),
            "飲み込まれた正解の割合": round(df["gt_in_merge"].sum() / n_gt, 4),
            "1予測が覆う正解の最大": int(df["max_gt_per_pred"].max()),
            "分裂した正解": int(df["split_gt"].sum()),
            "予測面積/正解面積": round(float(df["area_ratio"].replace(
                [np.inf, -np.inf], np.nan).dropna().mean()), 3),
            "正解ごとの最良IoU平均": round(float(df["mean_best_iou"].dropna().mean()), 4),
        }
        if self.instances:
            import pandas as pd
            g = pd.DataFrame(self.instances)
            d["まったく予測が出ていない箇所"] = int((g.coverage == 0).sum())
            d["同上の割合"] = round(float((g.coverage == 0).mean()), 4)
            d["箇所Recallの天井"] = round(float((g.coverage > 0).mean()), 4)
        return d

    def to_csv(self, prefix="instance_eval"):
        import pandas as pd
        self._frame().to_csv(f"{prefix}_tiles.csv", index=False)
        if self.instances:
            pd.DataFrame(self.instances).to_csv(f"{prefix}_gt_instances.csv", index=False)


# ----------------------------------------------------------------------
# 合成例
# ----------------------------------------------------------------------
def _demo():
    def blank():
        return np.zeros((128, 128), dtype=np.uint8)

    cases = {}
    gt, pr = blank(), blank()
    for cx in (24, 64, 104):
        gt[50:70, cx - 10:cx + 10] = 1
        pr[48:72, cx - 12:cx + 12] = 1
    cases["(1) 3個をそれぞれ当てる"] = (gt, pr, "理想。どの方式でも 1.0 付近")

    gt, pr = blank(), blank()
    for cx in (30, 55, 80, 105):
        gt[50:70, cx - 8:cx + 8] = 1
    pr[45:75, 20:115] = 1
    cases["(2) 4個を1つの予測が飲み込む"] = (gt, pr, "1対1だと3個がFN。被覆なら4個とも検出扱い")

    gt, pr = blank(), blank()
    for cx in (30, 55, 80, 105):
        gt[50:70, cx - 8:cx + 8] = 1
    pr[20:110, 10:120] = 1
    cases["(3) 飲み込み＋大幅に塗りすぎ"] = (gt, pr, "被覆でも precision で罰せられる")

    gt, pr = blank(), blank()
    gt[40:90, 40:90] = 1
    for y in (45, 60, 75):
        pr[y:y + 10, 45:85] = 1
    cases["(4) 1個の正解を3つに割る"] = (gt, pr, "予測側の FP が増えるのが妥当")

    gt, pr = blank(), blank()
    gt[20:40, 20:40] = 1
    pr[80:100, 80:100] = 1
    cases["(5) まったく重ならない"] = (gt, pr, "Recall=0, Precision=0")

    modes = [("coverage", dict(mode="coverage", cover_gt=0.5, cover_pred=0.3)),
             ("iou_many", dict(mode="iou_many", iou_thr=0.3)),
             ("iou_1to1", dict(mode="iou_1to1", iou_thr=0.3))]
    print("=" * 92)
    print("合成例での挙動（min_size=10, connectivity=8, iou_thr=0.3, cover_gt=0.5, cover_pred=0.3）")
    print("=" * 92)
    for name, (gt, pr, note) in cases.items():
        print(f"\n{name}   — {note}")
        print(f"  {'方式':<10}{'正解':>5}{'予測':>5}{'検出':>5}{'的中':>5}"
              f"{'Recall':>9}{'Prec':>8}{'F値':>8}")
        for mname, cfg in modes:
            r, _ = evaluate_tile(gt, pr, {mname: cfg})
            rec = r[f"{mname}__tp_gt"] / r["n_gt"] if r["n_gt"] else float("nan")
            pre = r[f"{mname}__tp_pred"] / r["n_pred"] if r["n_pred"] else float("nan")
            f1 = 2 * rec * pre / (rec + pre) if rec and pre else 0.0
            print(f"  {mname:<10}{r['n_gt']:>5}{r['n_pred']:>5}"
                  f"{r[f'{mname}__tp_gt']:>5}{r[f'{mname}__tp_pred']:>5}"
                  f"{rec:>9.3f}{pre:>8.3f}{f1:>8.3f}")
        d, _ = evaluate_tile(gt, pr)
        print(f"  診断: 飲み込んだ予測 {d['merged_pred']} / 飲み込まれた正解 {d['gt_in_merge']} "
              f"/ 1予測が覆う正解の最大 {d['max_gt_per_pred']} / 分裂 {d['split_gt']} "
              f"/ 面積比 {d['area_ratio']:.2f} / 最良IoU平均 {d['mean_best_iou']:.3f}")


if __name__ == "__main__":
    _demo()
