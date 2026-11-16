"""Three-state posterior scoring: ell_V / ell_O / ell_M, softmax, S_B, g.

Per-beam background model: each beam keeps a single Gaussian (mu, sigma)
selected from a cached 3-component beam-wise GMM. The background component is
the qualifying component (weight >= min_weight_bg and sigma <= sigma_max_bg)
with the largest weight; beams without any qualifying component fall back to
the largest-weight component; beams absent from the cache have no model.

Evidence terms (A-family, softmax over three states V/O/M):
  ell_V = log N(diff; 0, sig_eff^2) + wV * g
  ell_O = beta * log S_B(z) + wO * g        (A2/A3/A5; S_B = norm.sf, z = diff/sig_eff)
  ell_O = -|z| + wO * g                     (A1/A4, no censoring term)
  ell_M = bM                                (constant bias)
  sig_eff^2 = sig_p^2 + sigma_r^2
  g       : spatial support -- fraction of the +/-2 horizontal neighbors
            (same ring, wrap-around) whose current return is nearer than
            mu - 2*sigma.

Unmodeled beams get a uniform posterior and are forced to not update.
B-family baselines (B0/B1/B2) do not use the softmax; their rules live in
baselines.baseline_scores.

Sources: scripts/stage2_clean/runner.py,
         scripts/stage3_highwayscene/runner_hs.py (identical math, frozen).
"""

import numpy as np
from scipy.stats import norm

from baselines import baseline_scores

EPS = 1e-9

DEFAULT_PARAMS = {
    "alpha": 0.05,    # learning rate
    "c": 2.0,         # Hard-gate (B2) threshold
    "eta": 0.5,       # soft update gate threshold (q_v >= eta)
    "wV": 1.0,        # weight of g in ell_V
    "wO": 1.0,        # weight of g in ell_O
    "beta": 1.0,      # censoring-term weight
    "bM": -2.0,       # constant bias of the M state
    "sigma_r": 0.05,  # sensor range noise (m)
}


def load_bg_state(npz_path, vert, horiz, min_weight_bg, sigma_max_bg,
                  sigma_floor=1e-3):
    """Load the initial per-beam (mu, sigma) grids from a cached GMM npz.

    Returns dict with mu/sig (vert, horiz) arrays (NaN where a beam has no
    model) and the `valid` mask.
    """
    d = np.load(npz_path, allow_pickle=True)
    keys, mu, sig, w = d["keys"], d["mu"], d["sigma"], d["weights"]
    qualifies = (w >= min_weight_bg) & (sig <= sigma_max_bg)
    score = np.where(qualifies, w, -1.0)
    pick = score.argmax(axis=1)
    no_qual = (score < 0).all(axis=1)
    pick[no_qual] = w[no_qual].argmax(axis=1)
    mu_g = np.full((vert, horiz), np.nan)
    sig_g = np.full((vert, horiz), np.nan)
    rows = np.arange(len(keys))
    mu_g[keys[:, 0], keys[:, 1]] = mu[rows, pick]
    sig_g[keys[:, 0], keys[:, 1]] = sig[rows, pick]
    return {"mu": mu_g, "sig": np.maximum(sig_g, sigma_floor),
            "valid": ~np.isnan(mu_g)}


def horiz_idx(x, y, horiz):
    """Horizontal beam index from the azimuth (repo indexing.py convention)."""
    az = np.degrees(np.arctan2(y, x)) + 180.0
    return np.clip(np.floor(az / (360.0 / horiz)).astype(np.int32), 0, horiz - 1)


def spatial_support(r, ver, hor, mu, sig, vert, horiz):
    """g: fraction of same-ring horizontal +/-2 beams (incl. self, wrap-around)
    whose current return is nearer than mu - 2*sigma."""
    near = np.zeros((vert, horiz), dtype=bool)
    valid_pts = ~np.isnan(mu[ver, hor])
    flag = valid_pts & (r < mu[ver, hor] - 2 * sig[ver, hor])
    near[ver[flag], hor[flag]] = True
    nf = near.astype(np.float64)
    gsum = sum(np.roll(nf, d, axis=1) for d in range(-2, 3))
    return (gsum / 5.0)[ver, hor]


def compute_scores(strategy, params, r, diff, sig_p, has_model, ver, hor,
                   mu, sig, vert, horiz):
    """Return (qv_raw, qo, qm, pred_fg).

    A-family: softmax posterior masses. B-family: zero masses and the
    rule-based pred_fg from baselines.baseline_scores.
    """
    b = baseline_scores(strategy, params, diff, sig_p, has_model)
    if b is not None:
        return b

    sig_eff2 = sig_p ** 2 + params["sigma_r"] ** 2
    use_g = strategy in ("A1", "A2", "A5")  # A5 shares the A2 scoring
    g = (spatial_support(r, ver, hor, mu, sig, vert, horiz) if use_g
         else np.zeros(len(r)))
    with np.errstate(invalid="ignore", divide="ignore"):
        lV = -0.5 * (np.log(2 * np.pi * sig_eff2) + diff ** 2 / sig_eff2) \
             + params["wV"] * g
        z = diff / np.sqrt(sig_eff2)
        if strategy in ("A2", "A3", "A5"):
            lO = params["beta"] * np.log(norm.sf(z) + EPS) + params["wO"] * g
        else:  # A1, A4: no censoring term, distance penalty
            lO = -np.abs(diff) / np.sqrt(sig_eff2) + params["wO"] * g
    lM = np.full(len(r), params["bM"])
    L = np.stack([lV, lO, lM], axis=1)
    L[~has_model] = [0.0, 0.0, 0.0]  # unmodeled -> uniform; update forced off
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    qv_raw, qo, qm = P[:, 0], P[:, 1], P[:, 2]
    pred_fg = ~has_model | (qo > qv_raw)
    return qv_raw, qo, qm, pred_fg
