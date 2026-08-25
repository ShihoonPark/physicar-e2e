# Random-Cone DAgger Iteration 1 — Final Report

Final category: **D1_VALIDATION_FAIL**

R1 was preserved byte-for-byte with the requested TRAIN/validation/checkpoint/ONNX/freeze hashes. The original S09 failure evidence was not overwritten.

## S09 R1 diagnosis

The available aggregate evidence supports learner-state distribution shift: nominal S09 offline MAE was 0.005641 rad, while closed-loop R1 failed before the cone with max CTE 0.803517 m and no temporal/API/contact fault. No per-tick R1 trace or live images were preserved, so counterfactual steering windows and live feature distance were honestly marked unavailable; no S09 label was generated.

## DAgger collection

Disk: 7.230 GiB before, 6.479 GiB after collection. Raw total: 799279246 bytes. Infra replacements: 0.

| Episode | Learner outcome | Completion | Teacher valid / invalid | R1↔Expert MAE rad | magnitude ratio |
|---|---:|---:|---:|---:|---:|
| dagger1_s01_r01 | RANDOM_CONE_POLICY_FAIL | 0.4023 | 193 / 0 | 0.09506 | 0.356 |
| dagger1_s02_r01 | RANDOM_CONE_POLICY_FAIL | 0.4010 | 194 / 0 | 0.09527 | 0.355 |
| dagger1_s03_r01 | RANDOM_CONE_POLICY_FAIL | 0.4083 | 195 / 0 | 0.09426 | 0.360 |
| dagger1_s04_r01 | RANDOM_CONE_POLICY_FAIL | 0.4069 | 195 / 0 | 0.09458 | 0.360 |
| dagger1_s05_r01 | RANDOM_CONE_POLICY_FAIL | 0.2949 | 137 / 0 | 0.07499 | 0.426 |
| dagger1_s06_r01 | RANDOM_CONE_POLICY_FAIL | 0.3988 | 193 / 0 | 0.08985 | 0.363 |
| dagger1_s07_r01 | RANDOM_CONE_POLICY_FAIL | 0.3981 | 194 / 0 | 0.08432 | 0.383 |
| dagger1_s08_r01 | RANDOM_CONE_POLICY_FAIL | 0.3999 | 194 / 0 | 0.07877 | 0.405 |

All eight R1 genuine failures were retained as valid DAgger evidence. S05 terminated on cone intersection; the others terminated on sustained off-track. R1 alone controlled the vehicle; Expert commands were shadow labels only.

## Dataset and training

DAgger temporal sequences: **1483**. Aggregate: **8189** = 6,706 EXPERT_BASELINE + 1483 DAGGER1. Future teacher labels: 0.

Aggregate SHA-256: `8442af7dd41e5033722981249a906e8cb68287f2b83e3e2d35eb31f305b3dfc6`

D1 trained once from scratch, early-stopped at epoch 25 with best epoch 18. Parameter count: 255,819.

| Scenario | R1 MAE | D1 MAE | R1 RMSE | D1 RMSE | R1 magnitude ratio | D1 magnitude ratio |
|---|---:|---:|---:|---:|---:|---:|
| S09 | 0.005641 | 0.008304 | 0.009701 | 0.013514 | 0.996 | 1.025 |
| S10 | 0.005057 | 0.007402 | 0.007885 | 0.011232 | 0.991 | 1.001 |

Checkpoint SHA-256: `b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434`

ONNX SHA-256: `3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c`; equivalence PASS, max rad difference 4.161e-08.

Freeze SHA-256: `66dbf7762ab089f111e2c02d22240d861e575730dcb416692bf6fac4e1e3fdc8`; freeze seal SHA-256: `7781423c7ba69f381e91120687d07d93d006393ff3c0c74af751085ce6ea1840`.

R1 penultimate-feature nearest-distance mean ratio (DAGGER1 / nominal split): 21.21× (diagnostic only).

## Frozen D1 live gate

S09: `RANDOM_CONE_POLICY_FAIL`. Completion 0.8838; minimum clearance 0.064655 m; contact=False; recovery=True; failure=`sustained off-track: boundary distance 0.181m`; safe stop=True.

D1 passed the cone and recovered, then genuinely went sustained off-track late in the lap at route s=29.307 m. Therefore S10 and unseen S11/S12 were not run. No retry, retraining, tuning, or DAgger iteration 2 occurred.

## Leakage, verification, and disposition

Leakage audit PASS: training contains S01–S08 only; validation remains S09/S10 only; S09/S10 neural-live data and S11/S12 bags/images/labels are absent from training.

Tests: 382 passed, 2 warnings, 39 subtests passed in 79.38s (0:01:19). `git diff --check`: PASS.

Tracked simulator source: PASS (runtime `userdata/last_world` may be modified; tracked source changes: []).

D1 does **not** become the random-cone simulator baseline. Repeatability and real-robot work are not justified by this failed simulator gate. D2 is not yet justified; inspect this late-lap failure first.

Limitations: the original R1 S09 trace was aggregate-only; learner rollouts often terminated before their scenario's full obstacle phases; simulator results are not real-robot evidence.

Final Git status (no commit or push):

```text
## experiment/random-cone-dagger1-1p0-v1
?? configs/random_cone_dagger1_1p0_v1.json
?? results/pilotnet_e2e_d1_random_cone_1p0/
?? results/pilotnet_training_d1_random_cone_1p0/
?? results/random_cone_dagger1_collection_1p0_v1/
?? results/random_cone_dagger1_dataset_1p0_v1/
?? results/random_cone_r1_failure_diagnosis_1p0_v1/
?? scripts/run_random_cone_dagger1_1p0_v1.py
?? src/physicar_e2e/random_cone_dagger1.py
?? tests/test_random_cone_dagger1_1p0.py
```
