# High-Speed DAgger V1 Data Report

Result: **PASS — exactly two independent V5-controlled shadow rollouts collected and extracted**.

PilotNet V5 alone controlled steering at 1.80 m/s using camera input. High-Speed Expert V1 used the frozen 0.90 m lookahead, 15 Hz rate, 0.18 m wheelbase, and ±0.349066 rad steering limit in shadow mode with no command authority. No nominal Expert lap was collected and no rollout was retried.

| Rollout | Frozen role | V5 result | Progress | Divergence onset | Raw bag bytes / SHA-256 | Selected samples |
|---|---|---|---:|---:|---|---:|
| A | V6 training | POLICY_FAIL | 20.904 m / 68.53% | 10.867 s / index 163 | 97,656,518 / `54f23c6…43a95` | 50 |
| B | on-policy holdout | POLICY_FAIL | 20.724 m / 67.94% | 11.067 s / index 166 | 95,578,928 / `f885ba7e…70bdd` | 43 |

Rollout A passed the reproduction gate because its valid failure was beyond 30% and closely reproduced V5's original 67.82% failure region. Rollout B was independently recorded and assigned holdout status before V6 training. Both recorders finalized gracefully; both policy and final safe stops passed; API and pose/clock liveness failure counts were zero.

| Rollout | Objective selected progress | Window counts pre/divergence/late | Label age mean/median/p95/max | Stale / future |
|---|---|---:|---|---:|
| A | 51.33%–68.30% | 31 / 15 / 4 | 49.9 / 52.5 / 95 / 100 ms | 0 / 0 |
| B | 53.13%–67.83% | 30 / 13 / 0 | 53.26 / 55 / 95 / 100 ms | 0 / 0 |

Each objective interval begins 2.0 seconds before the final persistent CTE divergence and ends at the final valid pre-safe-stop frame, with retained progress at least 30%. Raw ROS RGB frames were labeled by causal simulator-time zero-order hold (`expert_timestamp <= camera_timestamp`). Per-sample metadata preserves image and Expert timestamps, label age, progress, CTE, pose/yaw, V5 and Expert steering, their signed difference, objective-window role, and source hashes.

A/B source hashes are distinct. Training/evaluation image paths do not overlap. Contact sheets show coherent late-route progression into the expected off-track region with valid crops. Large bags, telemetry, extracted images, manifests, and previews remain outside Git under `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_dagger_v1/`.
