"""Validation-set grid search with constraints (TUMTraf-I, stage-2 protocol).

Objective:  J = F1_O + 0.2 * V_Recall - 0.5 * OAR
Hard constraints: V_Recall >= 0.90 and Coverage >= 0.95
                  (infeasible parameter sets are excluded from selection)

Grids (full factorial):
  B2: c in [1.0, 6.0] step 0.5 (11 points, full response curve in tune_log.json)
  A1: eta {0.3,0.5,0.7} x wV {0.5,1,2} x wO {0.5,1,2} (beta fixed, no S_B)
  A2: eta x wV x wO x beta {0.1,1,2,5,10} (135 sets)
  A3: eta x beta (g = 0, wV/wO inactive, 15 sets)
  A4: eta (g = 0 and no S_B, 3 sets)
  B0/B1: parameter-free, one reference run each

Selection happens on the validation slice only; the test slice is run
exactly once afterwards with the selected parameters (see docs/PROTOCOL.md).

Usage:
    python scripts/tune.py --config configs/tumtraf.yaml \
        --out-best best_params.json --out-log tune_log.json
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

from posterior import DEFAULT_PARAMS, load_bg_state  # noqa: E402
from tumtraf_io import (load_config, make_dataset, paths_from_config,  # noqa: E402
                        read_pcd)
from update import run_sequence  # noqa: E402

V_RECALL_MIN = 0.90
COVERAGE_MIN = 0.95
W_VR, W_OAR = 0.2, 0.5

ETA_GRID = [0.3, 0.5, 0.7]
W_GRID = [0.5, 1.0, 2.0]
BETA_GRID = [0.1, 1.0, 2.0, 5.0, 10.0]
C_GRID = [1.0 + 0.5 * i for i in range(11)]  # 1.0 .. 6.0


def objective(res):
    return res["f1_O"] + W_VR * res["V"]["R"] - W_OAR * res["absorb_rate"]


def feasible(res):
    return res["V"]["R"] >= V_RECALL_MIN and res["coverage"] >= COVERAGE_MIN


def record(log, strat, tag, params, res):
    entry = {
        "strategy": strat, "tag": tag,
        "params": {k: params[k] for k in ("c", "eta", "wV", "wO", "beta")
                   if k in params},
        "f1_O": res["f1_O"], "O_P": res["O"]["P"], "O_R": res["O"]["R"],
        "V_R": res["V"]["R"], "OAR": res["absorb_rate"],
        "FUR": res["false_update_rate"],
        "coverage": res["coverage"], "unknown": res["unknown_rate"],
        "J": objective(res), "feasible": feasible(res),
    }
    log.append(entry)
    flag = "" if entry["feasible"] else " [constraint violated]"
    print(f"{tag}: F1_O={res['f1_O']:.4f} V_R={res['V']['R']:.4f} "
          f"OAR={res['absorb_rate']:.4f} cov={res['coverage']:.4f} "
          f"J={entry['J']:.4f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PACKAGE_ROOT / "configs/tumtraf.yaml"))
    ap.add_argument("--out-best", default="best_params.json")
    ap.add_argument("--out-log", default="tune_log.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = paths_from_config(args.config)
    grid = cfg["grid"]

    ds = make_dataset(paths, "test")
    lo, hi = cfg["split"]["val"]
    fids = [f for f in sorted(ds.frame_ids()) if lo <= f <= hi]
    print(f"preloading val frames {fids[0]}-{fids[-1]} ({len(fids)} frames)...")
    frames = []
    for fid in fids:
        frame = ds.get_frame_by_id(fid)
        xyz, ring = read_pcd(paths, frame.path)
        m = np.load(paths.masks_dir / f"frame_{fid}.npz")
        frames.append((fid, xyz, ring, m["status"]))
    print("loaded\n")

    root = Path(args.config).resolve().parents[1]
    gmm = Path(cfg["paths"]["initial_gmm"])
    init = load_bg_state(gmm if gmm.is_absolute() else root / gmm,
                         grid["vert"], grid["horiz"],
                         cfg["model_selection"]["min_weight_bg"],
                         cfg["model_selection"]["sigma_max_bg"],
                         grid.get("sigma_floor", 1e-3))

    log = []
    t0 = time.time()

    def run(strat, p):
        return run_sequence(strat, p, frames, init, grid)

    for c in C_GRID:
        p = dict(DEFAULT_PARAMS, c=c)
        record(log, "B2", f"B2 c={c:.1f}", p, run("B2", p))

    for eta, wv, wo in itertools.product(ETA_GRID, W_GRID, W_GRID):
        p = dict(DEFAULT_PARAMS, eta=eta, wV=wv, wO=wo)
        record(log, "A1", f"A1 eta={eta} wV={wv} wO={wo}", p, run("A1", p))

    for eta, wv, wo, beta in itertools.product(ETA_GRID, W_GRID, W_GRID,
                                               BETA_GRID):
        p = dict(DEFAULT_PARAMS, eta=eta, wV=wv, wO=wo, beta=beta)
        record(log, "A2", f"A2 eta={eta} wV={wv} wO={wo} beta={beta}", p,
               run("A2", p))

    for eta, beta in itertools.product(ETA_GRID, BETA_GRID):
        p = dict(DEFAULT_PARAMS, eta=eta, beta=beta)
        record(log, "A3", f"A3 eta={eta} beta={beta}", p, run("A3", p))

    for eta in ETA_GRID:
        p = dict(DEFAULT_PARAMS, eta=eta)
        record(log, "A4", f"A4 eta={eta}", p, run("A4", p))

    for strat in ["B0", "B1"]:
        record(log, strat, strat, dict(DEFAULT_PARAMS),
               run(strat, dict(DEFAULT_PARAMS)))

    best = {}
    for strat in ["B0", "B1", "B2", "A1", "A2", "A3", "A4"]:
        entries = [e for e in log if e["strategy"] == strat]
        feas = [e for e in entries if e["feasible"]]
        pool = feas if feas else entries
        win = max(pool, key=lambda e: e["J"])
        best[strat] = {
            **win["params"],
            "val": {k: win[k] for k in ("f1_O", "O_P", "O_R", "V_R", "OAR",
                                        "FUR", "coverage", "unknown", "J")},
            "constraint_violated": not feas,
        }
        if not feas:
            print(f"!! {strat}: no feasible params on val, fell back to max J")

    Path(args.out_log).write_text(json.dumps(log, indent=2))
    Path(args.out_best).write_text(json.dumps(best, indent=2))
    print(f"\n=== best params ({time.time() - t0:.0f}s) ===")
    for s, b in best.items():
        pshow = {k: b[k] for k in ("c", "eta", "wV", "wO", "beta") if k in b}
        print(f"  {s}: {pshow} val={ {k: round(v, 4) for k, v in b['val'].items()} }"
              f"{' [constraint violated]' if b['constraint_violated'] else ''}")
    print(f"wrote {args.out_best} and {args.out_log}")


if __name__ == "__main__":
    main()
