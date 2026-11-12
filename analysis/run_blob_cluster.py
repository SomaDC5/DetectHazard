# -*- coding: utf-8 -*-
"""**似た地形の塊をまとめて、1件の実地調査で塊ごと判定できるか**。

出どころ
--------
SCIS & ISIS 2026 の質疑でもらった案 ──

> 検出した地域それぞれについてクラスタリングを行えば、
> 同じような地形だった場所は、**その塊のうち一つを実地調査することで
> 塊全体を危険か否か判別できて**、適合率の問題である実世界の適用が
> 可能になるのでは。

**この案が成り立つ条件は1つだけである。**

    クラスタの中で、的中と外れが**混ざっていない**こと（純度）

混ざっていれば、1件調べた結果を塊全体に当てるのは間違いになる。
だから測るのは純度と、**それが素の的中率 0.6437 を超えるか**である。
超えなければ、クラスタリングは何も足していない。

**この案の値打ちは精度ではなく費用である。** いまは
「適合率 p なら 1/p 倍の面積を見る」（発表資料 `Backup 7`）と書いているだけで、
**9,669 件すべてを見る前提になっている。** クラスタリングが効けば、
**k 件の調査**に置き換わる。だから横軸は **k（調査の件数）**にする。

危ないところ（先に書いておく）
------------------------------
`14.3` と `39.6` で、外から持ってきた情報は**3回続けてロジットと重複した**
（広域・土地利用・建物。いずれも1段目の内側にある）。
**地形もロジットの内側にある。** だから対照を2つ置く。

| 対照 | これに勝てなければ |
|---|---|
| **でたらめなクラスタ**（同じ k）| クラスタが意味を持っていない |
| **ロジットを k 分位で切るだけ** | 地形を使う意味が無い（`39.5` と同じ重複）|

**判定は箇所F とゆらぎ幅（広島 0.0071）。** 純度が高くても、
捨てた塊に正解が隠れていれば箇所R が落ちる。**最後は箇所F で見る。**

    python analysis/run_blob_cluster.py
    python analysis/run_blob_cluster.py --with-logit     # ロジットも素性に入れる
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

for _p in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent.parent]:
    if (_p / "dc5lib" / "paths.py").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from analysis.run_overlap_eval import DLAT, DLON                     # noqa: E402
from analysis.run_split_bound import f_value                         # noqa: E402
from dc5lib import paths                                             # noqa: E402
from dc5lib.instance_eval import BAND, PRIMARY                       # noqa: E402

# 帯は `dc5lib/instance_eval.py` の `BAND` が原本（実装書 `5`）
# **被覆のしきい値は `PRIMARY` から取る。** 直書きしない（実装書 §5）。
_m = re.search(r"cov_gt=([\d.]+)", PRIMARY)
assert _m, f"PRIMARY の形が変わりました: {PRIMARY}"
COV_GT = float(_m.group(1))
KS = (5, 10, 20, 50, 100, 200, 500, 1000)
SEEDS = (1, 2, 3, 4, 5)
# 地形・形・建物。**ロジットは入れない**（入れるとモデルの順位の焼き直しになる）
TERRAIN = ["slope_mean", "slope_p90", "slope_max", "slope_std",
           "area", "compactness"]
BUILDING = ["bld_in", "bld_dist", "bld_50", "bld_100", "house_50", "house_100"]


def kmeans(X, k, seed=42, iters=60):
    """k-means（k-means++ 初期化）を **torch で**書く。

    **この環境では numpy の行列積そのものが落ちる**（`0xc06d007f`）。
    `np.float32 (9669,12) @ (12,500)` だけで Windows の致命的例外になる。
    だから sklearn の `KMeans` も scipy の `kmeans2` も使えない
    （どちらも内部で BLAS の gemm を呼ぶ）。
    `run_bg_gate.py` が「sklearn を使わない（環境で落ちるため）」と
    書いているのと同じ原因である。**torch は落ちない**ので torch で書く。
    """
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=g))
    idx = [first]
    d2 = ((Xt - Xt[first]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2.clamp_min(0)
        nxt = (int(torch.multinomial(p, 1, generator=g)) if float(p.sum()) > 0
               else int(torch.randint(n, (1,), generator=g)))
        idx.append(nxt)
        d2 = torch.minimum(d2, ((Xt - Xt[nxt]) ** 2).sum(1))
    C = Xt[idx].clone()
    lab = torch.full((n,), -1, dtype=torch.long)
    ones = torch.ones(n)
    for _ in range(iters):
        new = torch.cdist(Xt, C).argmin(1)
        if torch.equal(new, lab):
            break
        lab = new
        S = torch.zeros_like(C)
        cnt = torch.zeros(k)
        S.index_add_(0, lab, Xt)
        cnt.index_add_(0, lab, ones)
        keep = cnt > 0
        C[keep] = S[keep] / cnt[keep, None]      # 空のクラスタは動かさない
    return lab.numpy()


def load(cond, region):
    """①の表に、`39` の建物の列を (tile, blob) で突き合わせて足す。

    **面積と的中が全件一致することを確かめてから足す**（`run_miss_coverage.py`
    と同じ作法）。一致しなければ別の推論の表なので、混ぜてはいけない。
    """
    tri = paths.cache_dir("analysis") / "triage"
    b = pd.read_csv(tri / f"{cond}__{region}__blobs.csv")
    g = pd.read_csv(tri / f"{cond}__{region}__gt.csv")
    p = pd.read_csv(tri / f"{cond}__{region}__pairs.csv")

    bld = paths.cache_dir("analysis") / "building" / f"{cond}__{region}__pred.csv"
    if bld.exists():
        d = pd.read_csv(bld)[["tile", "blob", "tp", "area"] + BUILDING]
        m = b.merge(d, on=["tile", "blob"], how="left", suffixes=("", "_b"))
        if len(m) != len(b):
            sys.exit(f"突き合わせで行が増えました（{len(b)} -> {len(m)}）")
        bad_a = int((m.area != m.area_b).sum())
        bad_t = int((m.tp.astype(bool) != m.tp_b.astype(bool)).sum())
        miss = int(m.area_b.isna().sum())
        print(f"  建物の表と突き合わせ: 欠け {miss} / 面積の食い違い {bad_a}"
              f" / 的中の食い違い {bad_t}")
        if miss or bad_a or bad_t:
            sys.exit("別の推論の表です。混ぜません")
        b = m.drop(columns=["area_b", "tp_b"])
        cols = TERRAIN + BUILDING
    else:
        print(f"  **{bld.name} がありません。地形と形だけで組みます**")
        cols = TERRAIN
    return b, g, p, cols


def scorer(g, p, n_blob_index):
    """残す塊の集合から (箇所R, 箇所P, 箇所F) を出す関数を作る。

    箇所R は**被覆の足し算**で決める（1件で足りず2件目で 50% に達する正解がある）。
    **しきい値は `PRIMARY` から取る**（直書きしない。実装書 §5）。
    """
    gidx = {(int(t), int(b)): i for i, (t, b) in enumerate(zip(g.tile, g.gtblob))}
    need = g.gt_area.to_numpy(float) * COV_GT
    rows = [[] for _ in range(len(n_blob_index))]
    for t, gb, j, px in zip(p.tile, p.gtblob, p.blob, p.px):
        k = n_blob_index.get((int(t), int(j)))
        if k is not None:
            rows[k].append((gidx[(int(t), int(gb))], px))
    n_gt = len(g)

    def run(keep, tp):
        cov = np.zeros(n_gt)
        for k in np.nonzero(keep)[0]:
            for gi, px in rows[k]:
                cov[gi] += px
        R = float((cov >= need).mean())
        n = int(keep.sum())
        P = float(tp[keep].mean()) if n else 0.0
        return R, P, f_value(R, P), n
    return run


def purity(lab, y):
    """クラスタごとに多数派を採ったときの正解率（**上限**。神様が札を貼る場合）。"""
    tot = 0
    for c in np.unique(lab):
        s = y[lab == c]
        tot += max(int(s.sum()), int((~s).sum()))
    return tot / len(y)


def one_survey(lab, y, seeds=SEEDS):
    """**1件だけ調べて塊全体にその札を当てる**ときの正解率。

    調べた1件そのものは当たり前に正しいので**集計から外す**。
    クラスタが1件しかないときは、その1件を調べたことになる（除くと空）。
    """
    out = []
    for sd in seeds:
        rs = np.random.RandomState(sd)
        ok = n = 0
        for c in np.unique(lab):
            idx = np.nonzero(lab == c)[0]
            pick = rs.choice(idx)
            rest = idx[idx != pick]
            if len(rest) == 0:
                continue
            ok += int((y[rest] == y[pick]).sum())
            n += len(rest)
        out.append(ok / n if n else float("nan"))
    return float(np.mean(out)), float(np.min(out)), float(np.max(out))


def cluster_prune(lab, y, run, seeds=SEEDS):
    """調べた1件が外れだったクラスタを**丸ごと捨てる**。箇所R/P/F で見る。"""
    res = []
    for sd in seeds:
        rs = np.random.RandomState(sd)
        keep = np.zeros(len(y), bool)
        for c in np.unique(lab):
            idx = np.nonzero(lab == c)[0]
            if y[rs.choice(idx)]:
                keep[idx] = True
        res.append(run(keep, y))
    a = np.array(res, float)
    return a.mean(axis=0), a[:, 2].min(), a[:, 2].max()


def two_stage(lab, y, logit, run, seeds=SEEDS):
    """**捨てずに、順番だけ変える**版。これなら箇所R は落ちない。

    1. 各クラスタから1件ずつ調べる（費用 k 件）
    2. **当たったクラスタを先に**、中はロジットの降順で並べ替える
    3. 上位 n 件を見たときの箇所R を、ロジット順の曲線と同じ目盛で比べる

    クラスタの札がロジットの外側の情報を持っていれば曲線が上がる。
    `39.6` の重複（広域・土地利用・建物）と同じ形の問いである。
    """
    out = []
    for sd in seeds:
        rs = np.random.RandomState(sd)
        hit = {}
        for c in np.unique(lab):
            idx = np.nonzero(lab == c)[0]
            hit[c] = bool(y[rs.choice(idx)])
        key = np.array([(0 if hit[c] else 1) for c in lab], float)
        order = np.lexsort((-logit, key))        # 当たったクラスタ優先、中はロジット順
        out.append(order)
    return out


def recall_at(order, run, y, fracs=(0.1, 0.2, 0.3, 0.5)):
    """順番 order の上位 n 件だけ残したときの箇所R。"""
    res = {}
    for f in fracs:
        n = max(1, int(round(f * len(order))))
        keep = np.zeros(len(order), bool)
        keep[order[:n]] = True
        res[f] = run(keep, y)[0]
    return res


def spatial_groups(b, region):
    """タイル番号 -> 格子座標。`features/tile_features__<地域>.csv` の経緯度から。

    **「同じような地形だった場所」を素性の近さではなく土地の近さで取る。**
    隣り合う塊は同じ斜面系に属するので、案の意図にいちばん近いのはこちら。
    """
    f = (paths.cache_dir("analysis") / "features"
         / f"tile_features__{region}.csv")
    if not f.exists():
        return None
    d = pd.read_csv(f)[["tile_no", "lon", "lat"]]
    gi = np.round((d.lon - d.lon.min()) / DLON).astype(int)
    gj = np.round((d.lat - d.lat.min()) / DLAT).astype(int)
    pos = {int(t): (int(a), int(c)) for t, a, c in zip(d.tile_no, gi, gj)}
    miss = sum(1 for t in b.tile_no.unique() if int(t) not in pos)
    if miss:
        print(f"  **経緯度の分からないタイルが {miss} 枚。空間の群は出しません**")
        return None
    return np.array([pos[int(t)] for t in b.tile_no])


def agree_rate(keys, y):
    """同じ群に入った塊のペアで、札が一致する割合。

    群の大きさ n、的中 a のとき 一致ペア = C(a,2) + C(n-a,2)。
    **1件しかない群は数えない**（比べる相手がいない）。
    """
    df = pd.DataFrame({"k": keys, "y": y.astype(int)})
    g = df.groupby("k").agg(n=("y", "size"), a=("y", "sum"))
    g = g[g.n >= 2]
    same = (g.a * (g.a - 1) + (g.n - g.a) * (g.n - g.a - 1)) / 2
    allp = g.n * (g.n - 1) / 2
    return (float(same.sum() / allp.sum()) if allp.sum() else float("nan"),
            int(g.n.sum()), int(len(g)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="FinalFusion_SAM_APM_OverlapTrain")
    ap.add_argument("--region", default="hiroshima")
    ap.add_argument("--with-logit", action="store_true",
                    help="ロジットも素性に入れる（**重複を見るため**）")
    a = ap.parse_args()

    print(f"条件 {a.condition} / 地域 {a.region}")
    b, g, p, cols = load(a.condition, a.region)
    if a.with_logit:
        cols = cols + ["logit_max", "logit_mean"]
    y = b.tp.to_numpy().astype(bool)
    base = max(y.mean(), 1 - y.mean())
    band = BAND[a.region]
    index = {(int(t), int(j)): i for i, (t, j) in enumerate(zip(b.tile, b.blob))}
    run = scorer(g, p, index)

    R0, P0, F0, n0 = run(np.ones(len(b), bool), y)
    # **自前の箇所R/P を、突き合わせ済みの値と比べる**（実装書 §5）。
    # このスクリプトは推論しないので `PRIMARY` を直接は呼べない。代わりに
    # `run_triage_curve.py` が書いた曲線の全件の行と比べる ── あちらは
    # `PRIMARY` と `assert` で突き合わせてある。同じ表から同じ足し算をするので、
    # **差は 0 になるはず**（2026-11-12 に実測で 0）。目で見比べるだけにしない。
    cvp = (paths.cache_dir("analysis") / "triage"
           / f"{a.condition}__{a.region}__curve.csv")
    if cvp.exists():
        full = pd.read_csv(cvp).iloc[-1]
        dR, dP = abs(R0 - float(full.recall)), abs(P0 - float(full.precision))
        if max(dR, dP) > 1e-9:
            sys.exit(f"**自前の箇所R/P が曲線の全件の値と違います**"
                     f"（R の差 {dR:.2e} / P の差 {dP:.2e}）。"
                     "表が別の推論のものか、足し算の定義が食い違っています。")
        print("  自前の箇所R/P は曲線の全件の値と一致（差 0）")
    else:
        print(f"  ※ {cvp.name} が無いので、自前の箇所R/P は突き合わせていません"
              "（未検査）")
    print(f"\n塊 {len(b):,}（的中 {int(y.sum()):,} / 外れ {int((~y).sum()):,}）"
          f"/ 正解 {len(g):,}")
    print(f"素の的中率 {y.mean():.4f} → **多数派に全部合わせるだけで "
          f"{base:.4f} 取れる。純度はこれを超えなければ意味が無い**")
    print(f"刈り取り前  箇所R {R0:.4f} / P {P0:.4f} / F {F0:.4f}")
    print(f"素性 {len(cols)} 個: {'・'.join(cols)}")

    # 標準化。**面積と距離は裾が長いので log を取る**
    X = b[cols].to_numpy(float).copy()
    for i, c in enumerate(cols):
        if c in ("area", "bld_dist", "house_50", "house_100"):
            X[:, i] = np.log1p(X[:, i])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)

    print("\n" + "=" * 78)
    print("【1】純度 — クラスタの中で的中と外れは混ざっていないか")
    print("=" * 78)
    print(f"{'k':>6}{'純度(上限)':>12}{'1件調査の正解率':>18}"
          f"{'範囲':>17}{'でたらめ':>11}{'ロジットk分位':>14}")
    labs = {}
    for k in KS:
        if k >= len(b):
            continue
        lab = kmeans(X, k, seed=42)
        labs[k] = lab
        pu = purity(lab, y)
        m, lo, hi = one_survey(lab, y)
        rs = np.random.RandomState(42)
        rnd = purity(rs.randint(0, k, len(y)), y)
        qb = pd.qcut(b.logit_max.rank(method="first"), k,
                     labels=False).to_numpy()
        ql = purity(qb, y)
        print(f"{k:>6}{pu:>12.4f}{m:>18.4f}"
              f"{f'{lo:.4f}-{hi:.4f}':>17}{rnd:>11.4f}{ql:>14.4f}")
    print(f"\n  素の的中率（対照の下限）{base:.4f}")
    print("  **「1件調査の正解率」が案そのものの成績である。**")
    print("  調べた1件は集計から外している（当たり前に正しいので）。")

    print("\n" + "=" * 78)
    print("【2】外れだったクラスタを丸ごと捨てる — 箇所F で見る")
    print("=" * 78)
    print(f"{'k（調査件数）':>14}{'残った塊':>10}{'箇所R':>9}{'箇所P':>9}"
          f"{'箇所F':>9}{'正味':>10}{'帯で':>9}{'F の範囲':>19}")
    for k, lab in labs.items():
        (R, P, F, n), fl, fh = cluster_prune(lab, y, run)
        print(f"{k:>14,}{n:>10,.0f}{R:>9.4f}{P:>9.4f}{F:>9.4f}"
              f"{F - F0:>+10.4f}{(F - F0) / band:>8.1f}倍"
              f"{f'{fl:.4f}-{fh:.4f}':>19}")
    # 0.7471 は `46.5` の実測（刈り取り後の箇所F）。**この表では計算しない**
    print(f"\n  比べる相手 ── **ロジットを下位30%で刈るだけで 0.7471**（`46.5`）"
          f"（調査 0 件・`19`/`39` で測定済み）")
    print(f"  判定の帯 {band:.4f}。**調査 k 件を払ってこれを超えるかどうか。**")

    print("\n" + "=" * 78)
    print("【3】捨てずに順番だけ変える — これなら箇所R は落ちない")
    print("=" * 78)
    lg = b.logit_max.to_numpy(float)
    base_order = np.argsort(-lg)
    rb = recall_at(base_order, run, y)
    fr = sorted(rb)
    print(f"{'k（調査件数）':>14}" + "".join(f"{'上位' + f'{100 * f:.0f}%':>12}"
                                            for f in fr))
    print(f"{'ロジット順だけ':>14}" + "".join(f"{rb[f]:>12.4f}" for f in fr))
    for k, lab in labs.items():
        got = {f: [] for f in fr}
        for order in two_stage(lab, y, lg, run):
            r = recall_at(order, run, y, fr)
            for f in fr:
                got[f].append(r[f])
        print(f"{k:>14,}" + "".join(
            f"{np.mean(got[f]):>12.4f}" for f in fr))
    print("\n  **ロジット順だけの行を上回らなければ、クラスタはロジットの"
          "内側にある。**")
    print("  上回っても、k 件の調査を先払いしていることを忘れないこと。")

    print("\n" + "=" * 78)
    print("【4】素性ではなく**土地の近さ**でまとめる — 案の意図にいちばん近い形")
    print("=" * 78)
    pq = float(y.mean())
    chance = pq ** 2 + (1 - pq) ** 2
    print(f"  でたらめな一致率（的中率 {pq:.4f} から） {chance:.4f}")
    print(f"\n{'まとめ方':<30}{'一致率':>10}{'対象の塊':>11}"
          f"{'群の数':>9}{'1群あたり':>11}")
    rows = [("同じタイルの中", b.tile_no.to_numpy())]
    gxy = spatial_groups(b, a.region)
    if gxy is not None:
        for w in (2, 4, 8):
            key = (gxy[:, 0] // w) * 100000 + (gxy[:, 1] // w)
            km = w * 0.576                      # 1タイル 0.576km（東西）
            rows.append((f"{w}x{w} タイルの枠（約{km:.1f}km角）", key))
    for lb, key in rows:
        r, n, ng = agree_rate(key, y)
        print(f"{lb:<30}{r:>10.4f}{n:>11,}{ng:>9,}{n / max(ng, 1):>11.2f}")
    # 比べるため、素性のクラスタと同じ数の群での一致率も出す
    print()
    for k in (100, 1000):
        if k in labs:
            r, n, ng = agree_rate(labs[k], y)
            print(f"{f'素性のクラスタ k={k}':<30}{r:>10.4f}{n:>11,}"
                  f"{ng:>9,}{n / max(ng, 1):>11.2f}")
    print("\n  **でたらめな一致率を超えた分だけが「1件調べれば分かる」の根拠である。**")


if __name__ == "__main__":
    main()
