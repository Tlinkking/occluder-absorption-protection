"""Run the stage-4 ablation / counterfactual strategies (label-decoding
protocol): A2 (Ours), A6 (No-O), A7 (No-M), OMP (Allow-O), MMP (Allow-M).

Usage:
    python scripts/ablation.py --config configs/tumtraf.yaml --dataset tumtraf \
        --variants A2,A6,A7,OMP,MMP
    python scripts/ablation.py --config configs/highwayscene.yaml \
        --dataset highwayscene --variants A2,OMP --log-every 10

Per-point logs (frame_id, beam_id, q_V, q_O, q_M, label, update_eligible,
update_weight) are written to --logdir. For HighwayScene the default
--log-every 10 keeps frames whose id is divisible by 10 (99 of 998 frames,
fixed stride, deterministic) to control the log size.
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

from baselines import STRATEGIES_ABLATION  # noqa: E402
from posterior import DEFAULT_PARAMS, load_bg_state  # noqa: E402
from update import run_strategy  # noqa: E402


def load_frames_tumtraf(paths, ds, fids):
    from tumtraf_io import read_pcd
    frames = []
    for fid in fids:
        frame = ds.get_frame_by_id(fid)
        xyz, ring = read_pcd(paths, frame.path)
        m = np.load(paths.masks_dir / f"frame_{fid}.npz")
        frames.append((fid, xyz, ring, m["status"], m["dist"]))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["tumtraf", "highwayscene"])
    ap.add_argument("--variants", default="A2,A6,A7,OMP,MMP")
    ap.add_argument("--split", default="test")
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--log-every", type=int, default=None,
                    help="frame-id stride for logs (default 1 tumtraf, 10 HS)")
    ap.add_argument("--out", default=None, help="metrics JSON output path")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    for v in variants:
        assert v in STRATEGIES_ABLATION, v

    if args.dataset == "tumtraf":
        from tumtraf_io import load_config, make_dataset, paths_from_config
        cfg = load_config(args.config)
        paths = paths_from_config(args.config)
        grid = cfg["grid"]
        ds = make_dataset(paths, "test")
        lo, hi = cfg["split"][args.split]
        fids = [f for f in sorted(ds.frame_ids()) if lo <= f <= hi]
        frames = load_frames_tumtraf(paths, ds, fids)
        root = Path(args.config).resolve().parents[1]
        gmm = Path(cfg["paths"]["initial_gmm"])
        gmm = gmm if gmm.is_absolute() else root / gmm
        log_every = args.log_every if args.log_every is not None else 1
    else:
        import yaml
        from highwayscene_io import HSContext
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        grid = cfg["grid"]
        ctx = HSContext(args.config)
        lo, hi = cfg["split"]["test"]
        fids = [f for f in ctx.frame_ids() if lo <= f <= hi]
        frames = ctx.load_frames(fids)
        gmm = ctx.initial_gmm
        log_every = args.log_every if args.log_every is not None else 10

    params = dict(DEFAULT_PARAMS)
    params.update(cfg["params"])
    init = load_bg_state(gmm, grid["vert"], grid["horiz"],
                         cfg["model_selection"]["min_weight_bg"],
                         cfg["model_selection"]["sigma_max_bg"],
                         grid.get("sigma_floor", 1e-3))

    all_metrics = {}
    for v in variants:
        log_path = (Path(args.logdir) / f"logs_{v}_{args.dataset}.npz"
                    if args.logdir else None)
        t0 = time.time()
        res = run_strategy(v, params, frames, init, grid,
                           log_path=log_path, log_every=log_every)
        all_metrics[v] = res
        print(f"{v}: O-F1={res['O']['F1']:.4f} OAR={res['OAR']:.4f} "
              f"update_rate={res['update_rate']:.4f} cov={res['coverage']:.4f} "
              f"({time.time() - t0:.0f}s)")
    if args.out:
        Path(args.out).write_text(json.dumps(all_metrics, indent=1,
                                             default=float))


if __name__ == "__main__":
    main()
