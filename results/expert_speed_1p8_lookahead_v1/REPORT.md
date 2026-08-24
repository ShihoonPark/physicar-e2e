# Expert 1.80 m/s Lookahead Characterization V1

## Final result

**PASS. A larger lookahead produced one full 1.80 m/s Expert lap.** The pre-registered sweep ran 0.60, 0.75, and 0.90 m in ascending order. The first two candidates failed valid evaluations; 0.90 m passed and ended the sweep. No repeatability, extra candidate, retry, training, DAgger, data collection, speed profile, steering-limit change, control-rate change, simulator change, commit, or push occurred.

The first passing and provisional high-speed Expert candidate is **0.90 m lookahead**. This is one-lap evidence only.

## Preserved contract

- Speed: fixed at exactly 1.80 m/s.
- Control frequency: fixed at 15 Hz.
- Steering authority: fixed at ±0.349066 rad (±20° contract).
- Wheelbase: fixed at 0.18 m.
- Candidate lookaheads: exactly 0.60, 0.75, 0.90 m.
- Pure Pursuit implementation, route, world, spawn, vehicle, off-track/liveness safety, lap completion, and safe stops: canonical and unchanged.
- Canonical Expert config SHA-256: `63814d3a30f8753092cd33fc53d44414cfb343e39caf805e624dbaf33a4bd050` — unchanged.

The canonical Expert/V4 0.50 m/s baselines, V4 1.80 m/s failure, and Expert 1.80 m/s 0.45 m-lookahead failure evidence remained unchanged.

## Preflight and infrastructure

The independent preflight and every candidate preflight passed safe stop → reset → full preflight. Each verified the exact cone-free world `custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1`, `switching=false`, 388 route points, 30.504611 m route, 12×7 m bounds, canonical spawn, valid pose/control API, and advancing unpaused simulator clock. There were zero API, pose-liveness, clock-liveness, or timing failures.

## Candidate results

| Metric | 0.60 m | 0.75 m | 0.90 m |
|---|---:|---:|---:|
| Classification | EXPERT_FAIL | EXPERT_FAIL | EXPERT_PASS |
| Elapsed | 17.077 s | 16.143 s | 16.019 s |
| Progress | 28.356 m / 92.96% | 28.277 m / 92.70% | 30.100 m / 98.67% |
| Final distance to start | 2.7740 m | 2.1056 m | 0.2121 m |
| Mean / max CTE | 0.10323 / 0.77381 m | 0.07209 / 0.54443 m | 0.05562 / 0.23236 m |
| Highest CTE route s | 28.358 m | 28.474 m | 28.999 m |
| Off-track events / duration | 1 / 0.5304 s | 1 / 0.5316 s | 0 / 0 s |
| Mean / max absolute steering | 0.16514 / 0.349066 rad | 0.14440 / 0.349066 rad | 0.11402 / 0.349066 rad |
| Saturation fraction | 25.78% | 14.88% | 5.39% |
| Terminal saturation | 53.33% | 46.67% | 0% |
| First saturation route s | 1.830 m | 1.798 m | 1.769 m |
| CTE change across saturated samples | +0.57609 m | +0.35090 m | −0.03005 m |
| Mean command delta | 0.03242 rad | 0.02833 rad | 0.02509 rad |
| Loop frequency | 15.001 Hz | 15.000 Hz | 15.001 Hz |
| Period mean / p95 / max | 66.664 / 66.690 / 68.504 ms | 66.665 / 66.692 / 67.567 ms | 66.663 / 66.689 / 69.165 ms |
| Timing slips >100 ms | 0 | 0 | 0 |
| API / pose / clock failures | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Runtime safe stop | PASS | PASS | PASS |

Final orchestrator safe stop: **PASS**.

## Baseline comparison

| Expert condition | Result | Progress | Mean / max CTE | Saturation |
|---|---:|---:|---:|---:|
| 0.50 m/s, lookahead 0.45 m | 5/5 PASS | Full laps | mean 0.01861 / worst 0.11836 m | mean 10.11% |
| 1.80 m/s, lookahead 0.45 m | EXPERT_FAIL | 12.769 m / 41.86% | 0.22262 / 0.83379 m | 59.38% |
| 1.80 m/s, lookahead 0.60 m | EXPERT_FAIL | 28.356 m / 92.96% | 0.10323 / 0.77381 m | 25.78% |
| 1.80 m/s, lookahead 0.75 m | EXPERT_FAIL | 28.277 m / 92.70% | 0.07209 / 0.54443 m | 14.88% |
| 1.80 m/s, lookahead 0.90 m | EXPERT_PASS | 30.100 m / 98.67% | 0.05562 / 0.23236 m | 5.39% |

Increasing lookahead progressively reduced mean CTE, maximum CTE, steering magnitude, saturation, terminal saturation, and command delta. At 0.90 m, CTE did not grow across saturated samples and the lap completed without an off-track event. This supports the anticipation hypothesis for this one run, but does not establish repeatability.

## Decision

Lookahead alone was sufficient to produce **one** full 1.80 m/s Expert lap at 0.90 m. The next experiment should be an Expert 1.80 m/s repeatability test with lookahead frozen at 0.90 m. Only after repeatability should high-speed PilotNet DAgger or data collection be considered.

High-speed PilotNet training is **not yet justified**, and Cone Avoidance V1 remains **blocked** pending a repeatable high-speed Expert baseline and subsequent neural-policy validation.

## Steering observation

The actual steering target remained on the unchanged `/steering` topic in radians:

```bash
docker exec -it physicar-sim bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml
ros2 topic echo /steering
'
```

## Files, tests, and Git

The isolated experiment added a minimal config, bounded runner, lookahead orchestration module, focused tests, protocol document, and compact result directory. The result directory contains `preflight.json`, `attempt_01.json`, `attempt_02.json`, `attempt_03.json`, `summary.json`, `experiment.started.json`, and this report. No bag, dataset, model, or training artifact was created.

- Focused experiment and shared-safety tests: 34 passed.
- Full regression: 217 passed.
- `git diff --check` and no-index whitespace checks for every new experiment file: PASS before and after live execution.
- Final branch: `experiment/expert-speed-1p8-lookahead-v1`; only the six intended isolated experiment paths are untracked.
- No commit or push was performed.
