# Fast Random-Cone Post-Recovery Conflict Diagnosis V1

Primary classification: **NO_STRONG_CONFLICT_FOUND**

This is a frozen-checkpoint, existing-data-only simulator-policy diagnosis. No training, optimizer step, checkpoint write, simulator execution, rosbag read, or data collection occurred.

## Preserved inputs and subsets

| Subset | Count | Role |
|---|---:|---|
| EXPERT_BASELINE | 6706 | D1 source / pressure only |
| DAGGER1_ALL | 1483 | D1 source |
| DAGGER1_AVOIDANCE | 1443 | approach + avoidance + pass-return |
| DAGGER1_AVOIDANCE_ONLY | 155 | avoidance phase separately identified |
| DAGGER1_RECOVERY_OR_FAILURE | 40 | available post-recovery rows |
| DAGGER2_POST_RECOVERY | 109 | all existing post-recovery rows |
| EXPERT_NOMINAL | 512 | deterministic nominal reference |
| S09_AVOIDANCE_VALIDATION | 40 | evaluation only |
| S10_ALL_VALIDATION | 418 | identity audit only; never used for gradients |

The previous DAgger2 coverage gate remains **FAIL**: 109 sequences, 18 beyond route s=20 m, and 0 beyond s=26 m. This diagnosis does not reinterpret that result.

## Label distributions

| Source | Mean | Median | Std | Min | Max | p05 | p25 | p75 | p95 | Mean abs | Neg / Zero / Pos | Saturated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DAGGER1_AVOIDANCE | 0.0470449 | 0.0108076 | 0.152249 | -0.160973 | 0.349066 | -0.155329 | -0.0914787 | 0.134328 | 0.325688 | 0.125279 | 594 / 0 / 849 | 0.023562 |
| DAGGER1_AVOIDANCE_ONLY | 0.0756384 | 0.027552 | 0.160452 | -0.151149 | 0.3214 | -0.144101 | -0.054509 | 0.244198 | 0.315986 | 0.13887 | 64 / 0 / 91 | 0 |
| DAGGER2_POST_RECOVERY | 0.00953971 | 0.118474 | 0.224365 | -0.349066 | 0.349066 | -0.349066 | -0.240957 | 0.182884 | 0.303058 | 0.200067 | 45 / 0 / 64 | 0.137615 |

Units are radians. Route, CTE, heading, per-scenario, and per-phase distributions are in `label_distribution.json`.

DAGGER2_POST_RECOVERY asks for a broader, higher-magnitude correction distribution: its mean absolute steering is 1.59698x the broad DAgger1 avoidance subset, while its mean shifts by -0.0375052 rad. The positive/negative proportions remain similar, so the marginal labels are not simply opposite in sign.

## Frozen-D1 feature conflicts

- Nearest DAgger1-avoidance cosine distance: median 2.42258e-05, p95 0.0464414.
- Nearest-pair Expert label difference: median 0.0845331 rad, p95 0.588475 rad.
- Descriptive thresholds: cosine distance <= 8.85933e-08 (observed p25), label difference >= 0.201496 rad (observed p75).
- Conflicts: 7/109 overall; 0.25 of the high-similarity tail.
- Quartile-flag independence would predict 7.44954 intersections; observed enrichment is 0.939655x.
- Euclidean cross-check: nearest distance median 0.13113, p95 0.940093; only 1/109 Euclidean-near/high-disagreement pair.
- The seven cosine candidates have Euclidean distance median 1.76045 (minimum 1.26921), exceeding the nearest-Euclidean p95; their near-zero cosine values primarily reflect activation direction, not full feature proximity.
- Feature-aliasing evidence rule satisfied: no.

Top-20 bounded conflict table:

| D2 sample | D1 avoidance neighbor | Cosine distance | D2 Expert | D1 Expert | Abs diff | Sign agrees | D2 scenario/route | D1 scenario/route |
|---|---|---:|---:|---:|---:|---|---|---|
| dagger2_s07_r01_seq_000039 | dagger1_s03_r01_seq_000172 | 7.12472e-09 | -0.332369 | 0.319045 | 0.651414 | no | S07 / 20.1687 m | S03 / 11.4692 m |
| dagger2_s08_r01_seq_000021 | dagger1_s01_r01_seq_000169 | 5.09453e-09 | -0.270247 | 0.324772 | 0.595019 | no | S08 / 11.1039 m | S01 / 11.3032 m |
| dagger2_s05_r01_seq_000025 | dagger1_s07_r01_seq_000030 | 1.52062e-08 | -0.167499 | 0.349066 | 0.516565 | no | S05 / 20.6033 m | S07 / 1.89992 m |
| dagger2_s06_r01_seq_000017 | dagger1_s03_r01_seq_000173 | 7.90291e-08 | -0.196941 | 0.317368 | 0.51431 | no | S06 / 20.6886 m | S03 / 11.5211 m |
| dagger2_s05_r01_seq_000024 | dagger1_s08_r01_seq_000167 | 1.28166e-09 | -0.240957 | 0.216618 | 0.457575 | no | S05 / 20.5359 m | S08 / 11.1813 m |
| dagger2_s08_r01_seq_000022 | dagger1_s02_r01_seq_000164 | 1.00585e-08 | -0.131048 | 0.290884 | 0.421933 | no | S08 / 11.1039 m | S02 / 11.0565 m |
| dagger2_s07_r01_seq_000032 | dagger1_s05_r01_seq_000024 | 5.70896e-08 | 0.0545429 | 0.256205 | 0.201662 | yes | S07 / 18.7165 m | S05 / 1.58915 m |
| dagger2_s08_r01_seq_000019 | dagger1_s07_r01_seq_000031 | 0.0325561 | -0.349066 | 0.349066 | 0.698132 | no | S08 / 11.1039 m | S07 / 1.95116 m |
| dagger2_s07_r01_seq_000038 | dagger1_s02_r01_seq_000172 | 0.018612 | -0.349066 | 0.319085 | 0.668151 | no | S07 / 20.077 m | S02 / 11.5524 m |
| dagger2_s06_r01_seq_000015 | dagger1_s03_r01_seq_000174 | 0.0490181 | -0.306926 | 0.314549 | 0.621475 | no | S06 / 20.4545 m | S03 / 11.6189 m |
| dagger2_s08_r01_seq_000020 | dagger1_s06_r01_seq_000169 | 0.00364064 | -0.348177 | 0.242157 | 0.590335 | no | S08 / 11.1039 m | S06 / 11.2379 m |
| dagger2_s05_r01_seq_000023 | dagger1_s06_r01_seq_000171 | 0.0400555 | -0.34525 | 0.240435 | 0.585685 | no | S05 / 20.3407 m | S06 / 11.4263 m |
| dagger2_s07_r01_seq_000040 | dagger1_s03_r01_seq_000173 | 1.19738e-07 | -0.243855 | 0.317368 | 0.561224 | no | S07 / 20.2155 m | S03 / 11.5211 m |
| dagger2_s06_r01_seq_000016 | dagger1_s02_r01_seq_000165 | 0.000794731 | -0.259392 | 0.290884 | 0.550276 | no | S06 / 20.5802 m | S02 / 11.0565 m |
| dagger2_s06_r01_seq_000014 | dagger1_s08_r01_seq_000140 | 0.087967 | -0.34344 | 0.094687 | 0.438127 | no | S06 / 20.3014 m | S08 / 9.48495 m |
| dagger2_s07_r01_seq_000041 | dagger1_s02_r01_seq_000031 | 0.000226984 | -0.107329 | 0.268533 | 0.375862 | no | S07 / 20.1898 m | S02 / 2.36124 m |
| dagger2_s07_r01_seq_000037 | dagger1_s07_r01_seq_000087 | 0.0700906 | -0.349066 | -0.0276545 | 0.321411 | yes | S07 / 19.9594 m | S07 / 5.80747 m |
| dagger2_s05_r01_seq_000022 | dagger1_s07_r01_seq_000087 | 0.0899309 | -0.349066 | -0.0276545 | 0.321411 | yes | S05 / 20.1986 m | S07 / 5.80747 m |
| dagger2_s08_r01_seq_000016 | dagger1_s05_r01_seq_000055 | 0.0273299 | -0.349066 | -0.10194 | 0.247126 | yes | S08 / 11.1039 m | S05 / 3.89605 m |
| dagger2_s06_r01_seq_000013 | dagger1_s05_r01_seq_000055 | 0.0425765 | -0.349066 | -0.10194 | 0.247126 | yes | S06 / 20.1611 m | S05 / 3.89605 m |

DAGGER1 includes CTE and heading but not learner x/y/yaw, so the temporal-state check cannot make a complete pose-to-pose comparison. The JSON records all available timing, state, learner steering, and Expert steering for these pairs.

## Frozen-D1 gradient conflict

Six deterministic, disjoint 16-sample batches were evaluated with normalized steering MSE.

| Pair | Scope | Mean cosine | Median | Min | Max | Negative fraction |
|---|---|---:|---:|---:|---:|---:|
| DAGGER1_AVOIDANCE_vs_DAGGER2_POST_RECOVERY | head | 0.562908 | 0.677434 | -0.175543 | 0.830854 | 0.166667 |
| DAGGER1_AVOIDANCE_vs_DAGGER2_POST_RECOVERY | full | 0.421978 | 0.492665 | -0.107307 | 0.742681 | 0.166667 |
| DAGGER1_AVOIDANCE_ONLY_vs_DAGGER2_POST_RECOVERY | head | 0.661772 | 0.693926 | 0.327374 | 0.87128 | 0 |
| DAGGER1_AVOIDANCE_ONLY_vs_DAGGER2_POST_RECOVERY | full | 0.516841 | 0.540364 | 0.204881 | 0.750117 | 0 |
| DAGGER1_AVOIDANCE_vs_EXPERT_NOMINAL | head | -0.252608 | -0.323328 | -0.793038 | 0.362111 | 0.666667 |
| DAGGER1_AVOIDANCE_vs_EXPERT_NOMINAL | full | -0.249168 | -0.289582 | -0.711133 | 0.303775 | 0.833333 |
| DAGGER2_POST_RECOVERY_vs_EXPERT_NOMINAL | head | -0.210468 | -0.343242 | -0.690083 | 0.360154 | 0.666667 |
| DAGGER2_POST_RECOVERY_vs_EXPERT_NOMINAL | full | -0.134272 | -0.172518 | -0.526685 | 0.245378 | 0.666667 |

Consistent DAgger2-vs-avoidance opposition rule satisfied: **no**. D1 tensors were exactly equal before and after the gradient passes.

The cone-critical avoidance-only sensitivity check is also aligned: head median cosine 0.693926 and full median 0.540364, with negative fractions 0 and 0. Thus the broad subset is not hiding phase-specific gradient opposition in these batches.

## Source loss and pressure at frozen D1

| Source | Count | MSE (rad²) | Normalized MSE | MAE (rad) | p95 abs residual | Head grad norm median | Full grad norm median |
|---|---:|---:|---:|---:|---:|---:|---:|
| EXPERT_BASELINE | 6706 | 9.46667e-05 | 0.000776931 | 0.00615446 | 0.0191652 | 0.0502417 | 0.0545315 |
| DAGGER1_AVOIDANCE | 1443 | 0.000384432 | 0.00315504 | 0.0126839 | 0.0461402 | 0.132277 | 0.141084 |
| DAGGER1_AVOIDANCE_ONLY | 155 | 0.00121437 | 0.00996636 | 0.0270804 | 0.0703574 | 0.450695 | 0.495422 |
| DAGGER1_ALL | 1483 | 0.000390544 | 0.0032052 | 0.0129095 | 0.0468635 | 0.121082 | 0.139158 |
| DAGGER2_POST_RECOVERY | 109 | 0.0345039 | 0.283174 | 0.115553 | 0.449072 | 1.59496 | 2.09782 |

The 109 D2 rows are 1.31357% of the 8,298 rows but account for 75.5974% of frozen-D1 squared-error mass. Relative to broad DAgger1 avoidance, D2 has 89.7528x normalized MSE, 12.0577x head gradient norm, and 14.8692x full-network gradient norm. This is strong pressure evidence, but pressure magnitude alone is not directional gradient conflict.

## S09 avoidance prediction context

| Model | MAE | RMSE | Bias | Corrective magnitude ratio | Sign disagreement | Best lag (frames) |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0.0255136 | 0.031485 | 0.0142849 | 1.04078 | 0.15 | -1 |
| D2_FE | 0.0263476 | 0.0323595 | 0.00804115 | 1.00517 | 0.075 | -2 |

Offline pattern: **weaker_avoidance_magnitude_than_D1**. Indicators: weaker_avoidance_magnitude_than_D1=yes, more_wrong_sign_than_D1=no, delay_indicator=no, larger_absolute_bias_than_D1=no. This is mild and non-uniform: D2-FE is weaker in absolute magnitude on 80% of rows, and its mean absolute prediction is 0.965793x D1. At the nearest frozen-validation row to route s=12.750 m (12.7607 m), D2-FE minus D1 is only -0.00287568 rad. The known collision is therefore consistency evidence only, not a causal reconstruction across rollouts.

## Conclusion

The primary classification is **NO_STRONG_CONFLICT_FOUND**. The diagnosis establishes associations at frozen checkpoints; it does not prove that the 109 samples alone caused the scratch-trained D2-FE live collision.

DAGGER2_POST_RECOVERY has disproportionate frozen-D1 loss and gradient magnitude, but its gradient direction aligns with rather than opposes D1 avoidance in these batches.

## One next experiment

Measure per-sample, layerwise D1 gradient cosine on route-progress-matched DAgger1 avoidance and DAgger2 post-recovery pairs, without changing weights.

It was not implemented. No DAgger3 is recommended or created.

## Safety and limitations

- All frozen file hashes matched both before and after diagnosis.
- S09/S10 were excluded from every gradient batch; S10 was identity-audited only.
- S11/S12 were not accessed.
- This is simulator-policy evidence, not real-robot evidence.
