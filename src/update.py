"""Update gating, beam-wise moment update, and the sequence runner.

Update rule (per beam, at most once per frame, using the point with the
largest update weight w in that beam):
    mu    <- mu + a * (r - mu)
    sigma^2 <- (1 - a) * sigma^2 + a * (r - mu)^2,   a = alpha * w

Two loop entry points mirror the two evaluation protocols:
  run_sequence : stage-2 protocol (point-wise pred_fg decoding; strategies
                 B0/B1/B2/A1-A5). Metrics use pred_fg.
  run_strategy : stage-4 protocol (label decoding 0/1/2; strategies
                 A2/A6/A7/OMP/MMP). Metrics use the decoded labels and add
                 update-side diagnostics (update rate, O-conditioned mass).

Per-frame order (both loops): load pre-frame state -> compute scores on the
pre-update posterior -> decode outputs (no side effects) -> accumulate
metrics -> apply the update -> next frame. Each run reloads the initial
state, so runs are mutually independent.

Sources: scripts/stage2_clean/runner.py (run_sequence),
         scripts/stage4_ablation/runner_ablation.py (run_strategy).
"""

from pathlib import Path

import numpy as np

from baselines import LABEL, decode_labels, update_weights
from posterior import EPS, compute_scores


def apply_update(mu, sig, ver, hor, r, w, alpha, horiz, sigma_floor=1e-3):
    """Vectorized background update: per beam per frame, update once with the
    point of maximum w. mu += a*(r-mu); sig^2 <- (1-a)*sig^2 + a*(r-mu)^2."""
    upd = w > 0
    if not upd.any():
        return
    order = np.argsort(-w, kind="stable")
    beam_key = ver * horiz + hor
    bk_sorted = beam_key[order]
    keep_sorted = np.ones(len(r), dtype=bool)
    keep_sorted[1:] = bk_sorted[1:] != bk_sorted[:-1]
    chosen = order[keep_sorted & (w[order] > 0)]
    vv, hh = ver[chosen], hor[chosen]
    a = alpha * w[chosen]
    d_ch = r[chosen] - mu[vv, hh]
    mu[vv, hh] += a * d_ch
    sig2 = (1 - a) * sig[vv, hh] ** 2 + a * d_ch ** 2
    sig[vv, hh] = np.maximum(np.sqrt(sig2), sigma_floor)


def _gated_weights(strategy, params, qv_raw, pred_fg, has_model):
    """Stage-2 protocol update weights."""
    if strategy == "B0":
        return np.zeros(len(qv_raw))
    if strategy == "B1":
        return has_model.astype(float)  # every point updates (contamination)
    if strategy == "B2":
        return (has_model & ~pred_fg).astype(float)
    if strategy == "A5":
        return ((qv_raw >= params["eta"]) & has_model).astype(float)
    return qv_raw * (qv_raw >= params["eta"]) * has_model


def run_sequence(strategy, params, frames, init_state, grid, save_dir=None):
    """Stage-2 protocol loop.

    frames: [(fid, xyz, ver, status)] with status in {'V','O','I'} ('I' is
    not evaluated). init_state: dict from posterior.load_bg_state.
    grid: dict with vert/horiz/sigma_floor.
    """
    vert, horiz = grid["vert"], grid["horiz"]
    mu, sig = init_state["mu"].copy(), init_state["sig"].copy()
    alpha = params["alpha"]
    tpO = fpO = fnO = tpV = fnV = 0
    qv_sum_O = qv_sum_all = 0.0
    n_O = n_O_absorb = 0
    qv_O_sum = 0.0
    n_eval = n_unknown = 0

    from posterior import horiz_idx  # local import keeps module deps minimal

    for fid, xyz, ring, status in frames:
        r = np.linalg.norm(xyz, axis=1)
        ver = ring.astype(np.int32)
        hor = horiz_idx(xyz[:, 0], xyz[:, 1], horiz)
        mu_p, sig_p = mu[ver, hor], sig[ver, hor]
        has_model = ~np.isnan(mu_p)
        diff = r - mu_p

        qv_raw, qo, qm, pred_fg = compute_scores(
            strategy, params, r, diff, sig_p, has_model, ver, hor, mu, sig,
            vert, horiz)
        qv = _gated_weights(strategy, params, qv_raw, pred_fg, has_model)

        is_O = status == "O"
        is_V = status == "V"
        is_ev = is_O | is_V
        tpO += int((is_O & pred_fg).sum())
        fpO += int((is_V & pred_fg).sum())
        fnO += int((is_O & ~pred_fg).sum())
        tpV += int((is_V & ~pred_fg).sum())
        fnV += int((is_V & pred_fg).sum())
        n_O += int(is_O.sum())
        n_O_absorb += int((is_O & (qv > 0)).sum())
        qv_O_sum += float(qv[is_O].sum())
        qv_sum_all += float(qv.sum())
        n_eval += int(is_ev.sum())
        if strategy.startswith("A"):
            n_unknown += int((is_ev & (qm > qv_raw) & (qm > qo)).sum())

        apply_update(mu, sig, ver, hor, r, qv, alpha, horiz,
                     grid.get("sigma_floor", 1e-3))

        if save_dir is not None:
            np.savez_compressed(Path(save_dir) / f"frame_{fid}.npz",
                                frame_id=fid, pred_fg=pred_fg,
                                q_v_post=qv_raw.astype(np.float32),
                                q_v=qv.astype(np.float32),
                                q_o=qo.astype(np.float32),
                                q_m=qm.astype(np.float32))

    def prf(tp, fp, fn):
        P = tp / max(tp + fp, 1)
        R = tp / max(tp + fn, 1)
        return P, R, 2 * P * R / max(P + R, 1e-9)

    PO, RO, F1O = prf(tpO, fpO, fnO)
    PV, RV, F1V = prf(tpV, fnO, fnV)  # V's FP = O points called background
    unknown_rate = n_unknown / max(n_eval, 1)
    return {
        "O": {"P": PO, "R": RO, "F1": F1O},
        "V": {"P": PV, "R": RV, "F1": F1V},
        "f1_O": F1O,
        "absorb_rate": n_O_absorb / max(n_O, 1),
        "qv_O_mean": qv_O_sum / max(n_O, 1),
        "false_update_rate": qv_O_sum / max(qv_sum_all, EPS),
        "unknown_rate": unknown_rate,
        "coverage": 1.0 - unknown_rate,
        "n_O": n_O,
        "n_eval": n_eval,
    }


def run_strategy(strategy, params, frames, init_state, grid, save_dir=None,
                 log_path=None, log_every=1):
    """Stage-4 protocol loop with per-point logging.

    frames: [(fid, xyz, ver, status, dist)]. Log columns: frame_id, beam_id,
    q_V, q_O, q_M, label, update_eligible, update_weight. log_every > 1
    subsamples frames by frame id (fid % log_every == 0) to control size.
    """
    vert, horiz = grid["vert"], grid["horiz"]
    mu, sig = init_state["mu"].copy(), init_state["sig"].copy()
    alpha, eta = params["alpha"], params["eta"]

    from posterior import horiz_idx

    tpO = fpO = fnO = tpV = fnV = 0
    n_O = n_O_abs = n_V = n_nonV = n_nonV_elig = 0
    n_eval = n_unknown = n_fg = n_bg = 0
    n_obs = n_elig = 0
    w_sum = w_O_sum = 0.0
    w_Mpost_sum = n_Mpost = 0
    log_bufs = []

    for fid, xyz, ver, status, dist in frames:
        r = np.linalg.norm(xyz, axis=1)
        ver = ver.astype(np.int32)
        hor = horiz_idx(xyz[:, 0], xyz[:, 1], horiz)
        mu_p, sig_p = mu[ver, hor], sig[ver, hor]
        has_model = ~np.isnan(mu_p)
        diff = r - mu_p

        # pre-update posterior (identical call for all five strategies)
        qv, qo, qm, _ = compute_scores(
            "A2", params, r, diff, sig_p, has_model, ver, hor, mu, sig,
            vert, horiz)

        lab = decode_labels(strategy, qv, qo, qm, has_model)
        w = update_weights(strategy, qv, qo, qm, has_model, eta)
        elig = w > 0

        is_O = status == "O"
        is_V = status == "V"
        is_ev = is_O | is_V
        is_fg = lab == LABEL["fg"]
        is_unk = lab == LABEL["unknown"]
        is_bg = lab == LABEL["bg"]
        tpO += int((is_O & is_fg).sum()); fpO += int((is_V & is_fg).sum())
        fnO += int((is_O & ~is_fg).sum())
        tpV += int((is_V & is_bg).sum()); fnV += int((is_V & ~is_bg).sum())
        n_O += int(is_O.sum()); n_V += int(is_V.sum())
        n_O_abs += int((is_O & elig).sum())
        n_nonV += int((~is_V).sum()); n_nonV_elig += int(((~is_V) & elig).sum())
        n_eval += int(is_ev.sum())
        n_unknown += int((is_ev & is_unk).sum())
        n_fg += int((is_ev & is_fg).sum()); n_bg += int((is_ev & is_bg).sum())
        n_obs += int(has_model.sum()); n_elig += int(elig.sum())
        w_sum += float(w.sum()); w_O_sum += float(w[is_O].sum())
        m_post = (~is_V) & (qm > qv) & (qm > qo)
        w_Mpost_sum += float(w[m_post].sum()); n_Mpost += int(m_post.sum())

        if log_path is not None and (fid % log_every == 0 or log_every == 1):
            log_bufs.append((np.full(len(r), fid, dtype=np.int32),
                             (ver * horiz + hor).astype(np.int32),
                             qv.astype(np.float32), qo.astype(np.float32),
                             qm.astype(np.float32), lab, elig,
                             w.astype(np.float32)))

        apply_update(mu, sig, ver, hor, r, w, alpha, horiz,
                     grid.get("sigma_floor", 1e-3))

        if save_dir is not None:
            np.savez_compressed(Path(save_dir) / f"frame_{fid}.npz",
                                frame_id=fid, label=lab,
                                q_v=qv.astype(np.float32),
                                q_o=qo.astype(np.float32),
                                q_m=qm.astype(np.float32),
                                w=w.astype(np.float32))

    if log_path is not None and log_bufs:
        cols = list(zip(*log_bufs))
        np.savez_compressed(
            log_path,
            frame_id=np.concatenate(cols[0]), beam_id=np.concatenate(cols[1]),
            q_V=np.concatenate(cols[2]), q_O=np.concatenate(cols[3]),
            q_M=np.concatenate(cols[4]), label=np.concatenate(cols[5]),
            update_eligible=np.concatenate(cols[6]),
            update_weight=np.concatenate(cols[7]))
    if save_dir is not None:
        np.savez_compressed(Path(save_dir) / "_final_state.npz", mu=mu, sigma=sig)

    def prf(tp, fp, fn):
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        return P, R, 2 * P * R / max(P + R, 1e-9)

    PO, RO, F1O = prf(tpO, fpO, fnO)
    PV, RV, F1V = prf(tpV, fnO, fnV)
    return {
        "O": {"P": PO, "R": RO, "F1": F1O},
        "V": {"P": PV, "R": RV, "F1": F1V},
        "update_rate": n_elig / max(n_obs, 1),
        "mean_update_mass": w_sum / max(n_obs, 1),
        "O_cond_update_mass": w_O_sum / max(n_O, 1),
        "Mpost_cond_update_mass": w_Mpost_sum / max(n_Mpost, 1),
        "OAR": n_O_abs / max(n_O, 1),
        "FUR": n_nonV_elig / max(n_nonV, 1),
        "coverage": 1.0 - n_unknown / max(n_eval, 1),
        "unknown_rate": n_unknown / max(n_eval, 1),
        "fg_rate": n_fg / max(n_eval, 1),
        "bg_rate": n_bg / max(n_eval, 1),
        "n_O": n_O, "n_eval": n_eval, "n_obs": n_obs,
    }
