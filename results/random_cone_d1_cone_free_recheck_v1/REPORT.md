# Random-Cone D1 Cone-Free Validity Recheck V1

Simulator-only diagnostic evidence; this report makes no real-robot claim.

## 1. Preserved R1/D1 hashes

- R1 checkpoint/ONNX: `b50d5d3c3cdb4f7aa730b2a44c1ffd46d7e0deb7aa0328cb7d40b090ae9022a0` / `2ebb6faf79ff015ae79c31d404c1fc7eb932b726c60c9f0b6dc7d7e02e51c993`
- D1 checkpoint/ONNX: `b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434` / `3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c`
- Both freeze records and seals also matched their preserved hashes.

## 2. Previous invalid isolation evidence

The previous D1 cone-free run remains **INFRA_FAIL**, not policy evidence: clock/pose stopped for 0.803 s at s=19.926 m (65.32%), with no off-track event, temporal failure, >100 ms slip, or saturation. Max CTE before interruption was 0.3393 m and safe stop passed.

## 3. Disk state

`df -h /` was recorded. Free space was 6.317 GiB against the 5.5 GiB gate: **PASS**.

## 4. Fresh preflight

1 policy-attempt preflight(s) passed the canonical-world, API/schema, clock, pose-motion, camera, control, spawn/reset, three-frame temporal-buffer, and safe-stop checks.

## 5. D1 attempt(s)

- Attempt 1: FULL_LAP_PASS (432 compact cycles; file `results/random_cone_d1_cone_free_recheck_v1/attempts/d1_attempt_01.json`).

Infrastructure replacement count: 0 (maximum one).

## 6. Final valid D1 cone-free result

FULL_LAP_PASS. The canonical full-lap gate closed after 28.866 s and 30.125 m (98.75% completion), with mean/max CTE 0.0999/0.3412 m and 0 off-track events. Temporal, API, clock, pose, control-loop, and safe-stop health all passed. A previous infrastructure-invalid run was not counted as policy evidence.

## 7. D1 late-route telemetry

Late-route analysis: AVAILABLE; DAgger1 samples after 20 m: 0.
- 20-26 m: n=85, mean/max CTE 0.059335/0.180485 m; policy/shadow mean steering -0.017584/-0.012536 rad; MAE/bias 0.043142/-0.005048 rad; corrective ratio 1.0054; sign agreement 0.9059; CTE growth -0.134734 m; saturation 0.0000; temporal/timing/liveness True/True/True.
- 26-30.504611 m: n=55, mean/max CTE 0.123097/0.334815 m; policy/shadow mean steering 0.066775/0.021204 rad; MAE/bias 0.066636/0.045571 rad; corrective ratio 0.7727; sign agreement 0.8364; CTE growth 0.018583 m; saturation 0.1091; temporal/timing/liveness True/True/True.

## 8. Conditional R1 result

R1 authorized: False. No physical policy attempt was recorded.

## 9. Policy versus shadow Expert

The Expert remained telemetry-only and never crossed the command boundary. Full-run D1 comparison: `{"camera_oldest_to_current_span_s": {"count": 432, "max": 0.1364030479962821, "mean": 0.13281240200693126, "median": 0.13332439550140407, "p95": 0.13494136115259608}, "corrective_magnitude_ratio": 0.7525934869763471, "count": 432, "cte_end_m": 0.03999236028486852, "cte_growth_m": 0.03999217051915602, "cte_start_m": 1.8976571250758219e-07, "end_elapsed_s": 28.82496812700265, "end_route_s_m": 30.23161923789002, "label": "full_run", "maximum_cte_m": 0.34116400914188466, "mean_absolute_error_rad": 0.04821960471194656, "mean_model_steering_rad": 0.040815243236507044, "mean_shadow_expert_steering_rad": 0.03110791941722764, "mean_signed_error_rad": 0.009707323819279409, "off_track_fraction": 0.0, "onnx_inference_ms": {"count": 432, "max": 5.204759996559005, "mean": 1.1230654653344472, "median": 0.5198539984121453, "p95": 4.048454049916472}, "result": "AVAILABLE", "start_elapsed_s": 0.14267564799956745, "start_route_s_m": 0.10687500067560156, "steering_saturation_fraction": 0.013888888888888888, "steering_sign_agreement_fraction": 0.8310185185185185, "temporal_model_path_ms": {"count": 432, "max": 7.5107599986949936, "mean": 2.8368507362217685, "median": 2.3073875017871615, "p95": 5.793405647636973}}`. R1 comparison: `null`.

## 10. Final classification

**POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED** — Generic 1.00 m/s cone-free lane-following failure is not supported; the preserved S09 late failure likely depends on post-avoidance or scenario-specific closed-loop state.

## 11. Exactly one next direction

Run one bounded TRAIN-only DAgger2 experiment targeting post-recovery and late-route learner states actually visited after successful cone avoidance. It was not implemented here.

## 12. No learning or data collection

No training, fine tuning, weighting, balancing, dataset editing, DAgger collection/iteration, rosbag, or persisted camera image occurred. No checkpoint or ONNX file was created or modified. Frozen artifact and manifest hashes still match.

## 13. S11/S12 protection

**PASS** — zero activation, neural evaluation, bag, camera inspection, label generation, or manifest addition for S11/S12.

## 14. Tests

PASS: focused recheck: 13 passed; pre-live full regression: 407 passed and 39 subtests passed; post-live full regression: 407 passed and 39 subtests passed; final report-code regression: 407 passed and 39 subtests passed; 2 existing ONNX deprecation warnings in each full run. `git diff --check`: PASS.

## 15. Files changed

- `configs/random_cone_d1_cone_free_recheck_v1.json`
- `results/random_cone_d1_cone_free_recheck_v1/`
- `scripts/run_random_cone_d1_cone_free_recheck_v1.py`
- `src/physicar_e2e/random_cone_d1_cone_free_recheck.py`
- `tests/test_random_cone_d1_cone_free_recheck.py`

## 16. Final Git status

```text
## experiment/random-cone-d1-cone-free-recheck-v1
?? configs/random_cone_d1_cone_free_recheck_v1.json
?? results/random_cone_d1_cone_free_recheck_v1/
?? scripts/run_random_cone_d1_cone_free_recheck_v1.py
?? src/physicar_e2e/random_cone_d1_cone_free_recheck.py
?? tests/test_random_cone_d1_cone_free_recheck.py
```

No commit or push was performed.
