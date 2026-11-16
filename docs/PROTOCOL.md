# Evaluation Protocol

This document fixes the evaluation protocol used for all reported numbers.
It consolidates the frozen-protocol and statistical-protocol decisions so
that results can be regenerated independently of the original workspace.

## 1. Splits

| Dataset | Split | Frames | Used for |
|---|---|---|---|
| TUMTraf-I (s03 south) | validation | 827–930 (104 frames) | parameter selection only |
| TUMTraf-I (s03 south) | test | 931–1033 (103 frames) | reported results, run exactly once |
| HighwayScene (heavy_traffic/ouster) | test | official test split, ids 5001–5998 (998 frames) | reported results |

Both slices are temporal holdouts. The proxy masks (TUMTraf-I V/O/ignore)
are generated without any fitted parameters, so mask construction does not
leak information between slices. The earlier draft split (validation 451–600)
overlapped the background-reference training window and is deprecated.

## 2. State independence

Every run (every strategy, every parameter set, both datasets) reloads the
initial per-beam background state from the cached GMM npz before the first
frame. Runs never inherit mu/sigma from each other. The initial state comes
from a beam-wise 3-component GMM cached after a 600-frame training window;
the background component is the qualifying component
(weight >= min_weight_bg and sigma <= sigma_max_bg) with the largest weight
(TUMTraf-I: 0.1/0.58; HighwayScene: 0.36/0.22, from the external repo's
per-sensor baseline configs; these are baseline-model settings, not tuned
hyperparameters).

## 3. Parameter selection (validation only)

- Objective: `J = F1_O + 0.2 * V_Recall - 0.5 * OAR`
- Hard constraints: `V_Recall >= 0.90` and `Coverage >= 0.95`; infeasible
  parameter sets are excluded from selection.
- Grids (full factorial, see `scripts/tune.py`):
  `eta {0.3,0.5,0.7}`, `wV {0.5,1,2}`, `wO {0.5,1,2}`,
  `beta {0.1,1,2,5,10}`, `c in [1.0,6.0] step 0.5`.
- Hard-gate (B2): the validation J response is flat over c in [3, 6]; the
  most conservative end `c = 6.0` is selected for both datasets.
- Selected Ours (A2) parameters:
  TUMTraf-I `eta=0.7, wV=2.0, wO=0.5, beta=10.0`;
  HighwayScene `eta=0.5, wV=2.0, wO=0.5, beta=0.1`
  (shared: `alpha=0.05, bM=-2.0, sigma_r=0.05`).
- The test slice is evaluated exactly once after selection. The
  matched-Allow-O control (`eta_O* = 0.85`) was likewise selected on the
  validation slice (matching the overall update rate of Ours) and run once
  on the test slice; additional test-slice eta points (0.80, 0.90) were used
  only for a trend diagnostic, never for selection.

## 4. Metrics and aggregation

- OAR (occluder absorption rate): fraction of O-proxy points whose beam is
  updated (update weight > 0).
- Pooled metrics sum numerators/denominators over all frames before
  dividing; per-frame means average per-frame values over frames with a
  non-zero denominator. The two need not coincide and are reported
  separately where both exist.
- 40 m+ O-R: recall on O-proxy points with dist >= 40 m.
- HighwayScene labels come from the lane-OBB proxy protocol (inside any of
  the 7 lane OBBs = foreground class, outside = background, no ignore);
  TUMTraf-I labels come from the V/O/ignore proxy masks.

## 5. Bootstrap design

- Resampling unit: the valid frame (frames whose metric denominator is zero
  are excluded BEFORE resampling; e.g. 341 of 998 HighwayScene frames have
  no 40 m+ foreground point and are excluded from the 40 m+ O-R bootstrap,
  leaving 657 frames).
- Scheme: paired per-frame bootstrap, 1000 resamples, seed 42, same-frame
  pairing (difference = A - B).
- Intervals: percentile CI is the primary report; BCa (jackknife
  acceleration) is reported as a supplement.
- For TUMTraf-I overall O-R, a track-level bootstrap (resampling annotated
  object tracks) is used instead, since overall O-R aggregates over tracks.

## 6. Logging subsampling (HighwayScene)

Per-point logs are subsampled at write time with a fixed stride on the frame
id (`fid % 10 == 0`), giving 99 of the 998 frames (ids 5010, 5020, ..., 5990,
deterministic, not random). TUMTraf-I logs every frame (103/103).
Subsampling exists only to control log volume; ratios estimated from the
subsampled logs are unbiased, but absolute counts are not comparable across
datasets. Full-run update rates (all 998 frames) come from the online
accumulators in the runner, not from the subsampled logs.

## 7. Provenance

This protocol file summarizes the decisions recorded in the original
workspace documents `upload/04_protocols/frozen_protocol.md` and
`upload/04_protocols/statistical_protocol.md`, and in the stage notes under
`scripts/stage*/NOTES.md`.
