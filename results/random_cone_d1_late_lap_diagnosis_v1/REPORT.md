# Random-Cone D1 Late-Lap Isolation V1

Diagnostic-only simulator milestone. No simulator result is presented as real-robot evidence.

## 1. Preserved R1/D1 hashes

- R1 checkpoint / ONNX: `b50d5d3c3cdb4f7aa730b2a44c1ffd46d7e0deb7aa0328cb7d40b090ae9022a0` / `2ebb6faf79ff015ae79c31d404c1fc7eb932b726c60c9f0b6dc7d7e02e51c993`
- D1 checkpoint / ONNX: `b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434` / `3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c`
- R1 freeze / seal: `ac622a793ce2bc4794170b53cc9421cc343e1691eeb4d2b85e01609714e7e0d7` / `3d1c1f647b587ae6a8788c2d545a3ee8a2b0a3b90b9b465059f38ed5b687c798`
- D1 freeze / seal: `66dbf7762ab089f111e2c02d22240d861e575730dcb416692bf6fac4e1e3fdc8` / `7781423c7ba69f381e91120687d07d93d006393ff3c0c74af751085ce6ea1840`

## 2. Disk state

Root free space: `6.332 GiB`; required: `5.500 GiB`; gate: **PASS**.

## 3. Expert vs DAgger1 route-bin distribution

| Route bin | Expert | DAgger1 | Aggregate | DAgger1 fraction |
|---|---:|---:|---:|---:|
| 0-10 m | 2340 | 1182 | 3522 | 0.3356 |
| 10-20 m | 2176 | 301 | 2477 | 0.1215 |
| 20-26 m | 1313 | 0 | 1313 | 0.0000 |
| 26-30.504611 m | 877 | 0 | 877 | 0.0000 |

## 4. DAgger1 late-route coverage

DAgger1 late-bin sequences: **0**. Zero late-lap contribution confirmed: **true**. Learner rollouts ended from 29.49% to 40.83% completion; all genuine failures were retained without retry.

Cone-phase counts and per-scenario/per-rollout route bins are preserved in `offline_distribution.json`. The frozen Expert aggregate does not encode cone phase, so no Expert phase values were invented.

## 5. Matched R1/D1 offline route bins

Exact frozen S09/S10 current-frame targets and the same causal frame triplets were used for both models.

| Group | Route bin | N | R1 MAE | D1 MAE | R1 RMSE | D1 RMSE | D1/R1 MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| S09 | 0-10 m | 144 | 0.002343 | 0.003801 | 0.005289 | 0.006053 | 1.622 |
| S09 | 10-20 m | 139 | 0.007752 | 0.013016 | 0.012219 | 0.019259 | 1.679 |
| S09 | 20-26 m | 82 | 0.007145 | 0.008120 | 0.009166 | 0.010503 | 1.136 |
| S09 | 26-30.504611 m | 54 | 0.006716 | 0.008462 | 0.011993 | 0.014038 | 1.260 |
| S10 | 0-10 m | 146 | 0.003814 | 0.008232 | 0.005349 | 0.012029 | 2.158 |
| S10 | 10-20 m | 135 | 0.005514 | 0.006782 | 0.008630 | 0.011091 | 1.230 |
| S10 | 20-26 m | 82 | 0.005329 | 0.006645 | 0.008277 | 0.009210 | 1.247 |
| S10 | 26-30.504611 m | 55 | 0.006828 | 0.007848 | 0.010564 | 0.012093 | 1.149 |
| combined | 0-10 m | 290 | 0.003084 | 0.006032 | 0.005319 | 0.009542 | 1.956 |
| combined | 10-20 m | 274 | 0.006649 | 0.009944 | 0.010603 | 0.015772 | 1.496 |
| combined | 20-26 m | 164 | 0.006237 | 0.007383 | 0.008733 | 0.009878 | 1.184 |
| combined | 26-30.504611 m | 109 | 0.006773 | 0.008152 | 0.011295 | 0.013093 | 1.204 |

D1 late-bin regression: **true**; disproportionate by the registered ratio rule: **false**. Bias, maximum error, correlation, and corrective-magnitude ratio for every group/bin are in `offline_route_bins.json`.

## 6. Preserved S09 comparison

R1 failed before the cone at 39.40% completion. D1 passed the cone with 0.064655 m clearance, recovered in 0.796 s, then failed at s=29.307 m (88.38%).

DAgger1 solved or materially improved the original cone-approach failure, but introduced or exposed a new late-lap failure frontier.

## 7. D1 cone-free live result

Classification: **INFRA_FAIL**; completion `65.32%`; final s `19.926 m`; max CTE `0.3393 m`; safe stop `true`.
Stop reason: `simulator clock did not advance for 0.803s while motion was commanded`.

## 8. Conditional R1 result

R1: **NOT_RUN_D1_INFRASTRUCTURE_INVALID**.

## 9. Shadow-Expert and late-window findings

D1 vs shadow Expert: mean signed error `0.026270 rad`, mean absolute error `0.054988 rad`, corrective ratio `0.6341`, sign agreement `0.8007`. The shadow Expert never commanded the vehicle.
Final 2 s before the infrastructure stop: model/shadow means `-0.066797` / `-0.237679` rad; CTE growth `0.0050 m`; saturation `0.0000`. There was no off-track event, so an off-track-stop window is not applicable.

Historical S09 synchronized late windows are unavailable; the preserved S09 file is aggregate-only. Only the new cone-free telemetry supports per-iteration analysis.

## 10. Final classification

**MIXED_OR_INCONCLUSIVE**

Evidence does not satisfy one registered separation rule. Missing/conflicting: a D1 run with valid infrastructure/temporal inputs

## 11. Generic regression vs post-avoidance residual

The available evidence does not separate generic lane regression from residual post-avoidance shift.

## 12. Exactly one recommended next intervention

Acquire only the single missing valid cone-free comparison identified by the classification evidence before choosing a learning intervention.

This intervention was not implemented here.

## 13. No-training/data-collection confirmation

Training invocations: `0`; bags: `0`; persisted camera images: `0`; DAgger data: `0`; DAgger2: `false`; checkpoint/ONNX changes: `false`.

## 14. S11/S12 protection audit

**PASS** — no live activation, bag, camera inspection, Expert label, or manifest row involved S11/S12.

## 15. Tests

Result: **PASS**; 394 passed, 39 subtests passed, 2 existing ONNX deprecation warnings; focused diagnosis tests 12 passed.

`git diff --check`: **PASS**.

## 16. Files changed

- `configs/random_cone_d1_late_lap_diagnosis_v1.json`
- `results/random_cone_d1_late_lap_diagnosis_v1/`
- `scripts/run_random_cone_d1_late_lap_diagnosis_v1.py`
- `src/physicar_e2e/random_cone_d1_late_lap_diagnosis.py`
- `tests/test_random_cone_d1_late_lap_diagnosis.py`

## 17. Final Git status

```text
## experiment/random-cone-d1-late-lap-diagnosis-v1
?? configs/random_cone_d1_late_lap_diagnosis_v1.json
?? results/random_cone_d1_late_lap_diagnosis_v1/
?? scripts/run_random_cone_d1_late_lap_diagnosis_v1.py
?? src/physicar_e2e/random_cone_d1_late_lap_diagnosis.py
?? tests/test_random_cone_d1_late_lap_diagnosis.py
```

No commit or push was performed.
