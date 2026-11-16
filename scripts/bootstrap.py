"""Paired per-frame bootstrap CIs (percentile + BCa) for two strategies.

Resampling unit: the valid frame (frames with a zero denominator for the
requested metric are excluded BEFORE resampling). 1000 resamples, seed 42,
same-frame pairing (difference = A - B).

Usage:
    python scripts/bootstrap.py --config configs/tumtraf.yaml \
        --preds-a preds_A2 --preds-b preds_B2 --metric or40 --out boot.json

Metrics:
    or40 : per-frame 40m+ O-R = #(O & dist>=40 & pred_fg) / #(O & dist>=40)
    oar  : per-frame OAR      = #(O & w>0) / #O
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

from oar import paired_boot  # noqa: E402


def per_frame_values(preds_dir, masks_dir, metric, far_dist=40.0):
    vals, used, skipped = [], [], []
    for p in sorted(Path(preds_dir).glob("frame_*.npz"),
                    key=lambda x: int(x.stem.split("_")[1])):
        fid = int(p.stem.split("_")[1])
        pred = np.load(p)
        m = np.load(Path(masks_dir) / f"frame_{fid}.npz")
        status, dist = m["status"], m["dist"]
        pred_fg = (pred["pred_fg"].astype(bool) if "pred_fg" in pred
                   else pred["label"] == 1)
        is_O = status == "O"
        if metric == "or40":
            sel = is_O & (dist >= far_dist)
            den = int(sel.sum())
            num = int((sel & pred_fg).sum())
        else:  # oar
            w = pred["q_v"] if "q_v" in pred else pred["w"]
            sel = is_O
            den = int(sel.sum())
            num = int((sel & (w > 0)).sum())
        if den == 0:
            skipped.append(fid)
            continue
        used.append(fid)
        vals.append(num / den)
    return np.array(vals), used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preds-a", required=True)
    ap.add_argument("--preds-b", required=True)
    ap.add_argument("--masks", default=None)
    ap.add_argument("--metric", default="or40", choices=["or40", "oar"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tumtraf_io import paths_from_config
    masks_dir = Path(args.masks) if args.masks else \
        paths_from_config(args.config).masks_dir

    va, used, skip_a = per_frame_values(args.preds_a, masks_dir, args.metric)
    vb, used_b, skip_b = per_frame_values(args.preds_b, masks_dir, args.metric)
    assert used == used_b, "frame sets differ between the two pred dirs"
    res = paired_boot(va, vb, n_boot=args.n_boot, seed=args.seed)
    out = {
        "metric": args.metric,
        "unit": "frame",
        "n_frames_used": len(used),
        "n_frames_skipped_zero_denominator": len(skip_a),
        "skipped_fids": skip_a,
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "results": res,
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
