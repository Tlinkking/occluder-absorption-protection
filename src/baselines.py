"""Strategy registry and semantics.

Two strategy families share the same scoring backend:

Main strategies (stage-2 protocol, point-wise pred_fg decoding):
  B0  Static      : frozen background, never updates (q_V == 0).
  B1  Online      : conventional online beam-wise GMM; every point updates.
  B2  Hard-gate   : pred_fg = ~has_model | (diff < -c * sigma); updates only on
                    points classified as background.
  A1              : three-state soft arbitration without the censoring term
                    (ell_O = -|z| + wO * g).
  A2  Ours        : full method, ell_O = beta * log S_B + wO * g,
                    update weight w = q_V * 1[q_V >= eta].
  A3              : censoring term but no g (g == 0).
  A4              : neither censoring term nor g (ell_O = -|z|).
  A5              : same scoring as A2 but binary update weight
                    w = 1[q_V >= eta].

Ablation / counterfactual strategies (stage-4 protocol, label decoding):
  A6  No-O        : output-semantics ablation; every non-V point is reported
                    as unknown (no foreground output).
  A7  No-M        : output-semantics ablation; every non-V point is reported
                    as foreground (no abstention).
  OMP Allow-O     : O-mass-permitted update, w = (q_V + q_O) * 1[q_V + q_O >= eta].
  MMP Allow-M     : M-mass-permitted update, w = (q_V + q_M) * 1[q_V + q_M >= eta].

Label encoding for the stage-4 protocol: 0 = bg, 1 = fg, 2 = unknown.
Beams without a model decode to fg (same convention as pred_fg = ~has_model | ...).

Sources: scripts/stage2_clean/runner.py (B/A1-A5),
         scripts/stage4_ablation/runner_ablation.py (A6/A7/OMP/MMP).
"""

import numpy as np

STRATEGIES_MAIN = ["B0", "B1", "B2", "A1", "A2", "A3", "A4", "A5"]
STRATEGIES_ABLATION = ["A2", "A6", "A7", "OMP", "MMP"]
LABEL = {"bg": 0, "fg": 1, "unknown": 2}


def baseline_scores(strategy, params, diff, sig_p, has_model):
    """B-family scoring rules.

    Returns (qv, qo, qm, pred_fg) with zero masses, or None if `strategy`
    is not a B-family baseline (A-family strategies use the softmax
    posterior in posterior.compute_scores instead).
    """
    n = len(diff)
    if strategy == "B0":
        pred_fg = ~(has_model & (np.abs(diff) < 2 * sig_p))
    elif strategy == "B1":
        pred_fg = ~(has_model & (np.abs(diff) < 2.5 * sig_p))
    elif strategy == "B2":
        pred_fg = ~has_model | (diff < -params["c"] * sig_p)
    else:
        return None
    z = np.zeros(n)
    return z, z.copy(), z.copy(), pred_fg


def decode_labels(strategy, qv, qo, qm, has_model):
    """Pre-update posterior decoding (no side effects), stage-4 protocol."""
    if strategy in ("A2", "OMP", "MMP"):
        lab = np.stack([qv, qo, qm], axis=1).argmax(axis=1).astype(np.int8)
        lab[~has_model] = LABEL["fg"]
        return lab
    q_nonv = qo + qm
    is_bg = qv > q_nonv
    if strategy == "A6":
        return np.where(is_bg, LABEL["bg"], LABEL["unknown"]).astype(np.int8)
    if strategy == "A7":
        return np.where(is_bg, LABEL["bg"], LABEL["fg"]).astype(np.int8)
    raise ValueError(strategy)


def update_weights(strategy, qv, qo, qm, has_model, eta):
    """Strategy-specific update weights (stage-4 protocol)."""
    if strategy in ("A2", "A6", "A7"):
        return qv * (qv >= eta) * has_model
    if strategy == "OMP":
        wq = qv + qo
        return wq * (wq >= eta) * has_model
    if strategy == "MMP":
        wq = qv + qm
        return wq * (wq >= eta) * has_model
    raise ValueError(strategy)
