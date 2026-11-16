"""Run Ours (A2) or a baseline on TUMTraf-I, stage-2 protocol.

Usage:
    python scripts/run_tumtraf.py --config configs/tumtraf.yaml --strategy A2
    python scripts/run_tumtraf.py --config configs/tumtraf.yaml --strategy B2 --split val

The run reloads the initial background state, so val and test runs are
independent. Metrics are printed and (with --out) per-frame predictions are
saved as frame_<id>.npz.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

from baselines import STRATEGIES_MAIN  # noqa: E402
from posterior import DEFAULT_PARAMS, load_bg_state  # noqa: E402
from tumtraf_io import (load_config, make_dataset, paths_from_config,  # noqa: E402
                        read_pcd)
from update import run_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PACKAGE_ROOT / "configs/tumtraf.yaml"))
    ap.add_argument("--strategy", default="A2", choices=STRATEGIES_MAIN)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default=None, help="per-frame prediction output dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = paths_from_config(args.config)
    grid = cfg["grid"]
    params = dict(DEFAULT_PARAMS)
    params.update(cfg["params"])

    ds = make_dataset(paths, "test")  # both slices live in the repo's test list
    lo, hi = cfg["split"][args.split]
    fids = [f for f in sorted(ds.frame_ids()) if lo <= f <= hi]
    print(f"{args.strategy} on {args.split}: {len(fids)} frames ({fids[0]}-{fids[-1]})")

    frames = []
    for fid in fids:
        frame = ds.get_frame_by_id(fid)
        xyz, ring = read_pcd(paths, frame.path)
        m = np.load(paths.masks_dir / f"frame_{fid}.npz")
        status = m["status"]
        assert len(status) == len(xyz), f"frame {fid}: mask/point-count mismatch"
        frames.append((fid, xyz, ring, status))

    root = Path(args.config).resolve().parents[1]
    gmm = Path(cfg["paths"]["initial_gmm"])
    init = load_bg_state(gmm if gmm.is_absolute() else root / gmm,
                         grid["vert"], grid["horiz"],
                         cfg["model_selection"]["min_weight_bg"],
                         cfg["model_selection"]["sigma_max_bg"],
                         grid.get("sigma_floor", 1e-3))

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_sequence(args.strategy, params, frames, init, grid,
                       save_dir=args.out)
    res.update({"strategy": args.strategy, "split": args.split,
                "n_frames": len(fids)})
    print(json.dumps(res, indent=1, default=float))
    print(f"({time.time() - t0:.0f}s)")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "metrics.json").write_text(
            json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
