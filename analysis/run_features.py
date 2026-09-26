# -*- coding: utf-8 -*-
"""タイルそのものの性質（正解の形・傾斜・起伏・位置）を output/features に書き出す。

モデルに依存しないので地域ごとに1回でよい。

    python scripts/run_features.py --region both
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tileanalysis import config, features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="both", choices=["hiroshima", "hiroshima_bg", "shimane", "both", "all"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    if args.region == "both":
        regions = ["hiroshima", "shimane"]
    elif args.region == "all":
        regions = config.REGIONS
    else:
        regions = [args.region]

    for region in regions:
        out = features.features_path(region)
        if os.path.exists(out) and not args.force:
            print(f"[{region}] 済み: {out}（--force で再計算）")
            continue
        df = features.build_features(region)
        df.to_csv(out, index=False)
        print(f"[{region}] {out} に {len(df)} 行")


if __name__ == "__main__":
    main()
