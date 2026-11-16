"""Run Ours (A2) or a baseline on HighwayScene, stage-2 protocol.

Usage:
    python scripts/run_highwayscene.py --config configs/highwayscene.yaml --strategy A2

The run reloads the initial background state. Labels come from the lane-OBB
proxy protocol (inside OBB = foreground class 'O', outside = 'V'; no ignore).
"""

import argparse
import json
import sys
import time
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

from baselines import STRATEGIES_MAIN  # noqa: E402
from highwayscene_io import HSContext  # noqa: E402
from posterior import DEFAULT_PARAMS, load_bg_state  # noqa: E402
from update import run_sequence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=str(PACKAGE_ROOT / "configs/highwayscene.yaml"))
    ap.add_argument("--strategy", default="A2", choices=STRATEGIES_MAIN)
    ap.add_argument("--out", default=None, help="per-frame prediction output dir")
    args = ap.parse_args()

    ctx = HSContext(args.config)
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    grid = cfg["grid"]
    params = dict(DEFAULT_PARAMS)
    params.update(cfg["params"])

    lo, hi = cfg["split"]["test"]
    fids = [f for f in ctx.frame_ids() if lo <= f <= hi]
    print(f"{args.strategy} on test: {len(fids)} frames ({fids[0]}-{fids[-1]})")

    frames = []
    for fid, xyz, ch, status, _dist in ctx.load_frames(fids):
        frames.append((fid, xyz, ch, status))

    init = load_bg_state(ctx.initial_gmm, grid["vert"], grid["horiz"],
                         cfg["model_selection"]["min_weight_bg"],
                         cfg["model_selection"]["sigma_max_bg"],
                         grid.get("sigma_floor", 1e-3))

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = run_sequence(args.strategy, params, frames, init, grid,
                       save_dir=args.out)
    res.update({"strategy": args.strategy, "split": "test",
                "n_frames": len(fids)})
    print(json.dumps(res, indent=1, default=float))
    print(f"({time.time() - t0:.0f}s)")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "metrics.json").write_text(
            json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
