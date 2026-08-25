# D1-Preserving Post-Recovery Adaptation V1

Final category: **VALIDATION_FAIL**

## Frozen inputs and adaptation

Frozen D1 source checkpoint: `b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434`. Adapted D1-R checkpoint: `5f686e83aa03c36113d34b2a9063c399f2aa166a38afcef64a2d0fbcb52ce46d`.
The model initialized exactly from D1, kept all 134,948 convolutional parameters bitwise unchanged, and trained only the 120,871-parameter fully-connected steering head for exactly five epochs.

The loss was normalized post-recovery Expert MSE + normalized frozen-D1 retention MSE at coefficients 1.0/1.0. No validation labels, holdout data, new collection, scratch initialization, D2-FE initialization, or DAgger3 were used.

## Offline retention

Structural retention: **PASS**. Post-recovery MAE changed from 0.115550 to 0.009645 rad. Aggregate MAE changed from 0.007378 to 0.011774 rad; aggregate mean/max |D1-R − D1| were 0.008924/0.149743 rad.

| Dataset | Samples | D1 MAE | D1-R MAE | Mean abs delta | Max abs delta |
|---|---:|---:|---:|---:|---:|
| S09 | 419 | 0.008304 | 0.013189 | 0.009778 | 0.085603 |
| S10 | 418 | 0.007402 | 0.011037 | 0.008129 | 0.062094 |

Full phase, route-bin, RMSE, bias, corrective-ratio, and sign-disagreement metrics are in `offline_retention.json`.

## Export and freeze

Checkpoint: `5f686e83aa03c36113d34b2a9063c399f2aa166a38afcef64a2d0fbcb52ce46d`. ONNX: `2b85fe6de53383c01363be44c5101d00512bc0ca17d2028060894c3479fae18c`. ONNX checker/equivalence: PASS/PASS. Freeze seal: `684e626dfeaf89010e552aa4141bdbe712e6d21dfb035b42edd9dd244304cfeb`.

## Strictly gated simulator evaluation

| Scenario | Role | Result | Completion | Route s m | Clearance m | Recovery | Safe stop |
|---|---|---|---:|---:|---:|---:|---:|
| S09 | VALIDATION | RANDOM_CONE_POLICY_FAIL | 0.4177 | 12.741212 | 0.000000 | False | True |


The historical DAgger2 coverage gate remains FAIL (109 total, 18 beyond 20 m, zero beyond 26 m). D2-FE remains a frozen REGRESSION and was not used for initialization.

## Verification

Tests: 464 passed, 2 warnings, 39 subtests passed in 103.06s (0:01:43). `git diff --check`: PASS. No commit or push occurred. These are simulator results, not real-robot evidence.

Final Git status:

```text
## experiment/random-cone-d1-preserving-recovery-v1
?? configs/random_cone_d1_preserving_recovery_1p0_v1.json
?? results/pilotnet_e2e_d1_r_random_cone_1p0/
?? results/pilotnet_training_d1_r_random_cone_1p0/
?? scripts/run_random_cone_d1_preserving_recovery_1p0_v1.py
?? src/physicar_e2e/random_cone_d1_preserving_recovery.py
?? tests/test_random_cone_d1_preserving_recovery.py
```
