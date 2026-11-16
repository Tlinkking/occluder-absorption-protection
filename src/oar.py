"""Metrics: P/R/F1, occluder absorption rate (OAR), distance-bucketed recall,
and the paired per-frame bootstrap (percentile + BCa).

OAR (occluder absorption rate): fraction of O-proxy points whose beam is
updated (update weight > 0). Pooled OAR sums numerator and denominator over
all frames before dividing; the per-frame variant averages per-frame values
over frames with a non-zero denominator.

Distance-bucketed O-R: recall computed on O-proxy points with
dist >= bucket_low (e.g. the 40 m+ bucket used in the paper).

Sources: scripts/stage2_clean/evaluate.py,
         scripts/stage6_revision/s6_common.py (paired_boot).
"""

from statistics import NormalDist

import numpy as np

_ND = NormalDist()

BUCKETS = [(0, 20), (20, 30), (30, 40), (40, np.inf)]


def prf(tp, fp, fn):
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def bucketed_recall(is_o, dist, pred_fg, buckets=BUCKETS):
    """Per-bucket O recall. is_o/dist/pred_fg are aligned 1-D arrays."""
    out = {}
    for lo, hi in buckets:
        sel = is_o & (dist >= lo) & (dist < hi)
        n = int(sel.sum())
        key = f"{lo:g}-{('inf' if np.isinf(hi) else format(hi, 'g'))}m"
        out[key] = {
            "n": n,
            "O_recall": float((sel & pred_fg).sum() / n) if n else None,
        }
    return out


def _bca_ci(obs, boot, per_frame_vals):
    """BCa 95% CI with the jackknife acceleration (closed form for the mean)."""
    n = len(per_frame_vals)
    jk = (n * obs - per_frame_vals) / (n - 1)  # leave-one-out means
    jm = jk.mean()
    d = jm - jk
    den = 6.0 * float((d ** 2).sum()) ** 1.5
    a = float((d ** 3).sum()) / den if den > 0 else 0.0
    frac = float((boot < obs).mean())
    frac = min(max(frac, 1e-10), 1 - 1e-10)
    z0 = _ND.inv_cdf(frac)
    ci = []
    for alpha in (0.025, 0.975):
        za = _ND.inv_cdf(alpha)
        p = _ND.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
        ci.append(float(np.percentile(boot, 100 * p)))
    return ci


def paired_boot(vals_a, vals_b, n_boot=1000, seed=42):
    """Paired per-frame bootstrap. vals_a/vals_b: per-frame values, same frame
    order. Frames with a zero denominator must be removed BEFORE calling this
    function (the resampling unit is the valid frame).

    Returns {A: ..., B: ..., difference: ...} with difference = A - B, each
    {mean, ci_percentile, ci_bca}.
    """
    va = np.asarray(vals_a, dtype=np.float64)
    vb = np.asarray(vals_b, dtype=np.float64)
    assert va.shape == vb.shape and va.ndim == 1
    n = len(va)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = va[idx].mean(axis=1)
    boot_b = vb[idx].mean(axis=1)
    boot_d = boot_a - boot_b

    def summ(obs, boot, vals):
        return {
            "mean": float(obs),
            "ci_percentile": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "ci_bca": _bca_ci(obs, boot, vals),
        }

    return {
        "A": summ(va.mean(), boot_a, va),
        "B": summ(vb.mean(), boot_b, vb),
        "difference": {
            "direction": "A - B",
            **summ(va.mean() - vb.mean(), boot_d, va - vb),
        },
    }
