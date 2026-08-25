# High-Speed PilotNet DAgger V1 Final Report

## Outcome

Final result: **PARTIAL SUPPORT — V6 run #1 was POLICY_FAIL at 83.40%**.

V6 progressed 25.439 m before sustained off-track, 4.751 m and 15.57 percentage points farther than nominal-only V5's 20.688 m / 67.82%. This exceeds the preregistered 5-percentage-point material-improvement threshold, but it is not a lap. Per the hard gate, no V6 run #2, run #3, retry, tuning, or DAgger Iteration 2 was performed.

V6 cannot be frozen as the canonical 1.80 m/s policy. V4 remains the canonical 0.50 m/s policy. Cone Avoidance V1 is not yet justified.

## Preserved baselines

- High-Speed Expert V1 remained 1.80 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad, 0.18 m wheelbase, and 3/3 same-map/same-spawn simulation PASS.
- High-Speed PilotNet V5 and its 12 nominal bags, 2,911-sample dataset, 8/2/2 split, checkpoint, ONNX, and valid 67.82% POLICY_FAIL evidence were not modified or rerun.
- PilotNet V4, its 0.50 m/s data/config/model/ONNX/results, and its 3/3 PASS evidence were untouched. Its ONNX SHA-256 remains `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a`.

## Shadow rollouts and data

Exactly two V5-controlled 1.80 m/s rollouts were collected. The frozen High-Speed Expert shadow-labeled them and never commanded the vehicle.

| Rollout | Role | Result | Completion | Raw bytes / SHA-256 | Divergence | Selected samples |
|---|---|---|---:|---|---:|---:|
| A | training | POLICY_FAIL | 68.53% | 97,656,518 / `54f23c6…43a95` | 10.867 s | 50 |
| B | holdout | POLICY_FAIL | 67.94% | 95,578,928 / `f885ba7e…70bdd` | 11.067 s | 43 |

Rollout A passed the ≥30% reproduction gate. Its selected window spans 51.33%–68.30% progress with 31/15/4 pre-divergence/divergence/late samples. Rollout B spans 53.13%–67.83% with 30/13/0 samples. A/B label age mean/median/p95/max was 49.9/52.5/95/100 ms and 53.26/55/95/100 ms respectively. Stale rejects and future-label violations were zero for both.

The V6 training composition was nominal train 001–008 (1,940) plus rollout A (50), totaling 1,990 samples. Nominal validation 009–010 (486), nominal holdout 011–012 (485), and rollout B (43) stayed separate. No V4, low-speed DAgger, recovery, or nominal holdout data entered training.

## Training and offline comparison

V6 used the unchanged 252,219-parameter camera-only PilotNet, MSE, Adam 1e-3, batch 64, maximum 35 epochs, frozen early stopping, deterministic seed, and identical preprocessing. It was trained from scratch, not fine-tuned from V5. Best epoch was 9; training stopped after epoch 16.

| Stratum | Model | MAE | RMSE | Bias | Max | Correlation | Corrective ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Nominal validation | V5 | 0.010116 | 0.015698 | -0.003108 | 0.073177 | 0.994939 | 0.9811 |
| Nominal validation | V6 | 0.009514 | 0.015437 | 0.000713 | 0.083784 | 0.994907 | 0.9967 |
| Nominal holdout | V5 | 0.010401 | 0.015538 | -0.002893 | 0.071923 | 0.994967 | 0.9896 |
| Nominal holdout | V6 | 0.009676 | 0.015726 | 0.001130 | 0.082389 | 0.994695 | 1.0042 |
| V5 rollout-B holdout | V5 | 0.107968 | 0.165718 | 0.091996 | 0.358098 | 0.799835 | 0.5887 |
| V5 rollout-B holdout | V6 | 0.035264 | 0.047124 | 0.013179 | 0.150992 | 0.979884 | 0.9249 |

On B's 30 pre-divergence samples, V5/V6 MAE was 0.038302/0.028178 rad. On its 13 divergence samples, MAE was 0.268735/0.051616 rad. The late/failure group has zero samples because B safe-stopped before one full second elapsed after the objective onset. The independent on-policy evidence strongly favors V6 offline, but the live result shows that one small aggregation did not fully solve closed-loop robustness.

V6 checkpoint SHA-256 is `79e21210e984fac1a88fa910987e6b562888d3317e571708e05e900df5f5aa55` (1,017,973 bytes). V6 ONNX SHA-256 is `3e168565b05b3925e3ab26d9643cdd936cefec34a11e074b918036ba96c3acf6` (1,012,518 bytes). ONNX checker, `[batch,3,66,200] -> [batch,1]` I/O, and numerical equivalence passed; maximum PyTorch↔ONNX difference was 1.25e-7 rad.

## V6 live gate

Dedicated preflight and run-1 preflight both passed the exact world, switching=false, cone-free route, 388-point route, camera, clock, model identity, and safe-stop gates.

| Metric | V6 run #1 |
|---|---:|
| Classification | POLICY_FAIL |
| Elapsed / progress / completion | 14.749 s / 25.439 m / 83.40% |
| Final distance to start | 3.92258 m |
| Mean / max CTE | 0.07632 / 0.86993 m |
| Off-track events / duration | 1 / 0.53555 s |
| Mean / max absolute steering | 0.11291 / 0.349066 rad |
| Steering saturation / mean command delta | 1.36% / 0.02739 rad |
| Camera latency mean/p95/max | 1.902 / 2.972 / 3.762 ms |
| Preprocessing latency mean/p95/max | 1.421 / 1.790 / 2.273 ms |
| ONNX latency mean/p95/max | 0.859 / 3.718 / 4.471 ms |
| Loop frequency / period p95/max | 15.00031 Hz / 66.679 / 67.736 ms |
| Timing slips | 0 |
| API / pose-clock liveness failures | 0 / 0 |
| Per-run / final safe stop | PASS / PASS |

Conditional runs #2 and #3 were not authorized because run #1 failed.

## Progression and decision

| Controller | Speed/configuration | Result |
|---|---|---|
| High-Speed Expert V1 | 1.80 m/s, lookahead 0.90 | 3/3 PASS |
| PilotNet V4 | 0.50 m/s | 3/3 PASS |
| PilotNet V4 | 1.80 m/s | FAIL at 10.48% |
| PilotNet V5 nominal-only | 1.80 m/s | FAIL at 67.82% |
| PilotNet V6 nominal + actual on-policy DAgger | 1.80 m/s | FAIL at 83.40%; PARTIAL SUPPORT |

The experiment answers the central question with bounded partial support: Expert labels on V5-visited states substantially reduced independent on-policy Expert error and moved the live failure materially later, but one iteration with 50 added training samples was insufficient for a full lap. More data or another iteration requires a separate analysis and authorization.

## Storage, tests, files, and limitations

Large bags, telemetry, extracted images, V6 checkpoint, and V6 ONNX remain outside Git under `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_dagger_v1/`. Compact evidence is under `results/high_speed_dagger_v1/`, `results/pilotnet_training_v6_high_speed_dagger/`, and `results/pilotnet_e2e_v6_high_speed/`.

Focused High-Speed DAgger tests: 15/15 PASS. Full regression: 248/248 PASS. `git diff --check` passed.

Added three isolated configs, one orchestration module and wrapper, focused tests, documentation, and compact result evidence. No preserved config, source, model, dataset, or result was edited. No simulator tracked source was modified.

Limitations: this is same-map/same-spawn simulation, not real-robot evidence. Training uses raw ROS RGB-derived PNG while live inference uses HTTP JPEG, consistent with the preserved pipeline. Only two V5 rollouts and one valid V6 live run were permitted. Rollout B has no post-one-second late/failure samples.

Final repository branch is `experiment/pilotnet-high-speed-dagger-v1`. Experiment additions remain untracked with no tracked modifications. No commit or push was performed. The external simulator checkout remains read-only apart from runtime userdata state.
