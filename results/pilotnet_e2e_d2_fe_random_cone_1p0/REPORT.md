# Random-Cone D2 Frontier Expansion V1

Final category: **REGRESSION**

## Prior negative result and disk

The earlier Targeted DAgger2 Post-Recovery V1 result remains FAIL: 109 valid sequences, 18 with route s>20 m, zero with route s>26 m, and training unauthorized. Its collection/dataset tree hashes remain `ebe4d24dbab5cd3155676e5af8a4cca77ca45b1ddee7be52ed3977518e394fd1` / `b9f8f993e0b30153af611720dddc00a4acb718319d1b1607ec7bce5d7dacca48`.

Root free space was 5.870 GiB before training and is 5.853 GiB at final reporting.

## DAgger2 audit and aggregate

All 109 rows passed provenance, actual-D1-state, post-recovery, causal-label, and temporal-integrity checks. Coverage: <=20 m 91, >20 m 18, >26 m 0.

Aggregate: **8298** = 6,706 EXPERT_BASELINE + 1,483 DAGGER1 + 109 DAGGER2_POST_RECOVERY. SHA-256: `0c0b231264b39cbf3b09fcf44877e6d185dca860f8756732d6585e1ba8cda3db`. No images were duplicated.

## Training, export, and freeze

D2-FE has 255,819 parameters and trained once from scratch. Best epoch 12 after 19 completed epochs; early stopped=True.

Checkpoint: `7fcd021bd314640deff44a4fda6184eb944e55381d868a12764a20232966eba7`. ONNX: `a76de58c74ad361ef7ce182a4f4b35669f2e46215f38cf208cf4a4b24fe584a2`. Freeze: `fbbfa7b4a9e204799a33bfc6ec042170477b88225bbf029d3962bfaeee812d96`. Freeze seal: `11321a4d76ebe3910d957e5ad402eb97d35f215a37376fb94a2eb63eb7cead7a`. ONNX checker/equivalence: PASS/PASS.

## Frozen offline S09/S10

Values are MAE / RMSE / bias in radians on the identical frozen manifest.

| Scenario | R1 | D1 | D2-FE |
|---|---:|---:|---:|
| S09 | 0.005641 / 0.009701 / 0.000411 | 0.008304 / 0.013513 / 0.000955 | 0.008593 / 0.013951 / -0.002622 |
| S10 | 0.005057 / 0.007885 / 0.000274 | 0.007402 / 0.011233 / -0.000294 | 0.008865 / 0.012748 / -0.005309 |

| Route bin | R1 count / MAE | D1 count / MAE | D2-FE count / MAE |
|---|---:|---:|---:|
| 0-10m | 290 / 0.003084 | 290 / 0.006032 | 290 / 0.006614 |
| 10-20m | 274 / 0.006649 | 274 / 0.009945 | 274 / 0.010925 |
| 20-26m | 164 / 0.006236 | 164 / 0.007384 | 164 / 0.008351 |
| 26-30.5046107008m | 109 / 0.006773 | 109 / 0.008151 | 109 / 0.009401 |

## Strictly gated live results

| Scenario | Role | Result | Completion | Route s m | Clearance m | Recovery | Safe stop |
|---|---|---|---:|---:|---:|---:|---:|
| S09 | VALIDATION | RANDOM_CONE_POLICY_FAIL | 0.4180 | 12.750109 | 0.000000 | False | True |


Direct D1↔D2-FE frontier comparison: `{'classification': 'REGRESSION', 'comparison_rule_preregistered_before_live': True, 'd1_baseline': {'completion_fraction': 0.8838025637925251, 'final_route_s_m': 29.307113445990467, 'progress_m': 26.960053144868205}, 'd2_fe_minus_d1': {'completion_fraction': -0.46582938152986375, 'final_route_s_m': -16.557004237689547, 'progress_m': -14.209943936567285}, 'd2_fe_observed': {'completion_fraction': 0.41797318226266134, 'final_route_s_m': 12.750109208300922, 'progress_m': 12.75010920830092}, 'recovery_success': False}`.

## Disposition and verification

Frontier expansion supported: False. D2-FE becomes the simulator baseline: False. Another DAgger iteration: NOT_AUTOMATICALLY_AUTHORIZED. No DAgger3 was started.

Leakage audit PASS; no new training collection, bags, persisted live cameras, or Expert labels were produced. S11/S12 remained gated until S09/S10 both passed.

Tests: 448 passed, 2 warnings, 39 subtests passed in 79.93s (0:01:19). `git diff --check`: PASS. No commit or push occurred.

Limitations: The 109 DAgger2 sequences contain no targets beyond route s=26 m. Offline validation is diagnostic and was not used as a model-selection gate. A single closed-loop run per scenario cannot establish repeatability. Simulator performance is not evidence of real-robot success.

External artifacts: /home/a/physicar-ai-sim-docker/userdata/physicar_e2e/random_cone_1p0_v1/d2_frontier_expansion.

Final Git status:

```text
## experiment/random-cone-d2-frontier-expansion-1p0-v1
?? configs/random_cone_d2_frontier_expansion_1p0_v1.json
?? results/pilotnet_e2e_d2_fe_random_cone_1p0/
?? results/pilotnet_training_d2_fe_random_cone_1p0/
?? scripts/run_random_cone_d2_frontier_expansion_1p0_v1.py
?? src/physicar_e2e/random_cone_d2_frontier_expansion.py
?? tests/test_random_cone_d2_frontier_expansion.py
```
