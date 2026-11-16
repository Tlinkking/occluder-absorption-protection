"""Evaluate per-frame predictions against the proxy masks: O-P/R/F1, V-R,
OAR, coverage, and distance-bucketed O recall (incl. the 40 m+ bucket).

Usage:
    python scripts/evaluate.py --config configs/tumtraf.yaml \
        --preds <dir-with-frame_*.npz> [--table1] [--out metrics.json]

Predictions must contain pred_fg (stage-2 protocol, from run_*.py) or label
(stage-4 protocol, from ablation.py --out per-frame dirs); both are handled.
OAR requires q_v (stage-2) or w (stage-4) in the prediction files.

Note on the protocol: pooled metrics sum numerators and denominators over
all frames before dividing; per-frame means average per-frame values over
frames with a non-zero denominator. The two need not coincide.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT / "scripts/prepare"))

from oar import bucketed_recall, prf  # noqa: E402


def iter_frames(preds_dir, masks_dir):
    for p in sorted(Path(preds_dir).glob("frame_*.npz"),
                    key=lambda x: int(x.stem.split("_")[1])):
        fid = int(p.stem.split("_")[1])
        pred = np.load(p)
        m = np.load(Path(masks_dir) / f"frame_{fid}.npz")
        yield fid, pred, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--masks", default=None,
                    help="override mask dir (default: config masks_path; "
                         "HighwayScene recomputes lane-OBB masks)")
    ap.add_argument("--table1", action="store_true",
                    help="print a Table-1 style summary line")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tumtraf_io import load_config, paths_from_config
    cfg = load_config(args.config)
    if args.masks:
        masks_dir = Path(args.masks)
    elif cfg["dataset"] == "tumtraf":
        masks_dir = paths_from_config(args.config).masks_dir
    else:
        masks_dir = None  # HighwayScene: masks recomputed per frame below

    tpO = fpO = fnO = tpV = fnV = 0
    n_O = n_O_abs = n_eval = n_unk = 0
    o_sel_all, dist_all, fg_all = [], [], []

    if masks_dir is not None:
        frame_iter = iter_frames(args.preds, masks_dir)
    else:
        from highwayscene_io import HSContext
        ctx = HSContext(args.config)
        status_by_fid = {fid: (status, dist)
                         for fid, _x, _c, status, dist in
                         ctx.load_frames(ctx.frame_ids())}

        def frame_iter():
            for p in sorted(Path(args.preds).glob("frame_*.npz"),
                            key=lambda x: int(x.stem.split("_")[1])):
                fid = int(p.stem.split("_")[1])
                status, dist = status_by_fid[fid]
                yield fid, np.load(p), {"status": status, "dist": dist}
        frame_iter = frame_iter()

    for fid, pred, m in frame_iter:
        status, dist = m["status"], m["dist"]
        if "pred_fg" in pred:
            pred_fg = pred["pred_fg"].astype(bool)
        else:
            pred_fg = pred["label"] == 1
        w = pred["q_v"] if "q_v" in pred else pred["w"]

        is_O, is_V = status == "O", status == "V"
        is_ev = is_O | is_V
        tpO += int((is_O & pred_fg).sum()); fpO += int((is_V & pred_fg).sum())
        fnO += int((is_O & ~pred_fg).sum())
        tpV += int((is_V & ~pred_fg).sum()); fnV += int((is_V & pred_fg).sum())
        n_O += int(is_O.sum()); n_O_abs += int((is_O & (w > 0)).sum())
        n_eval += int(is_ev.sum())
        if "label" in pred:
            n_unk += int((is_ev & (pred["label"] == 2)).sum())
        elif "q_m" in pred:
            qv_post = pred["q_v_post"] if "q_v_post" in pred else pred["q_v"]
            n_unk += int((is_ev & (pred["q_m"] > qv_post)
                         & (pred["q_m"] > pred["q_o"])).sum())
        o_sel_all.append(is_O); dist_all.append(dist); fg_all.append(pred_fg)

    PO, RO, F1O = prf(tpO, fpO, fnO)
    PV, RV, F1V = prf(tpV, fnO, fnV)
    is_O = np.concatenate(o_sel_all); dist = np.concatenate(dist_all)
    pred_fg = np.concatenate(fg_all)
    res = {
        "O": {"P": PO, "R": RO, "F1": F1O},
        "V": {"P": PV, "R": RV, "F1": F1V},
        "OAR": n_O_abs / max(n_O, 1),
        "coverage": 1.0 - n_unk / max(n_eval, 1),
        "buckets": bucketed_recall(is_O, dist, pred_fg),
        "n_O": n_O, "n_eval": n_eval,
    }
    print(json.dumps(res, indent=1, default=float))
    if args.table1:
        or40 = res["buckets"]["40-infm"]["O_recall"]
        print(f"\nTable-1 row: O-F1={F1O:.4f} OAR={res['OAR']:.4f} "
              f"40m+ O-R={or40:.4f} V-R={RV:.4f} cov={res['coverage']:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
