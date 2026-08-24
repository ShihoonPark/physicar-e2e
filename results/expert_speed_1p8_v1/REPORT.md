# Expert Driver 1.80 m/s Feasibility Gate V1

## Final result

**EXPERT_FAIL.** The privileged canonical Pure Pursuit Expert did not complete the one permitted 1.80 m/s lap. It stopped safely after sustained off-track. No second attempt, retry, tuning, training, DAgger, rosbag collection, PilotNet change, simulator-source change, or cone-avoidance work occurred.

## Preserved contract

The canonical Expert config SHA-256 remained `63814d3a30f8753092cd33fc53d44414cfb343e39caf805e624dbaf33a4bd050`. The experiment loaded that configuration and replaced only `fixed_speed_mps`:

- Speed: 0.50 → exactly 1.80 m/s.
- Control frequency: unchanged at 15 Hz.
- Lookahead: unchanged at 0.45 m.
- Steering limit: unchanged at ±0.349066 rad.
- Wheelbase: unchanged at 0.18 m.
- Pure Pursuit implementation, route, world, spawn, lap completion, safety geometry, liveness, and safe-stop logic: unchanged.

Canonical Expert 0.50 m/s config/results, canonical V4 0.50 m/s config/results, and preserved V4 1.80 m/s failure evidence remained untouched.

## Preflight

The independent preflight and live-run preflight passed the required safe lifecycle: safe stop → reset → settle/full preflight → drive. Both verified the exact cone-free world `custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1`, `switching=false`, 388 route points, 30.504611 m route length, 12×7 m bounds, valid canonical spawn/pose, valid numeric control API, and an advancing unpaused simulator clock. Source and derived environment integrity checks passed.

## Attempt #1

| Metric | Measured value |
|---|---:|
| Classification | EXPERT_FAIL |
| Failure | Sustained off-track; 0.473 m beyond track band |
| Elapsed | 8.542 s |
| Progress | 12.769 m / 41.86% |
| Final distance to start | 10.2616 m |
| Mean / max CTE | 0.22262 / 0.83379 m |
| Highest CTE route location | s=12.664 m |
| Off-track events / total duration | 3 / 0.8640 s |
| Mean / max absolute steering | 0.26316 / 0.349066 rad |
| Steering saturation | 59.38% overall; 66.67% terminal 15 samples |
| First maximum-steering route location | s=1.838 m |
| CTE change across saturated samples | +0.73941 m; CTE grew while saturated |
| Mean absolute steering-command delta | 0.04690 rad |
| Loop frequency | 15.001 Hz |
| Loop period mean / p95 / max | 66.664 / 66.698 / 76.115 ms |
| Timing slips >100 ms | 0 |
| API failures | 0 |
| Pose / clock liveness failures | 0 / 0 |
| Runtime safe stop / final safe stop | PASS / PASS |

No infrastructure failure invalidated the evaluation. No attempt #2 exists.

## Comparison

| Controller and speed | Outcome | Progress | Mean / max CTE | Saturation |
|---|---:|---:|---:|---:|
| Expert 0.50 m/s | 5/5 PASS | Full laps | 0.01861 / worst 0.11836 m | mean 10.11% |
| PilotNet V4 0.50 m/s | 3/3 PASS | Full laps | 0.01868 / worst 0.11646 m | mean 6.54% |
| PilotNet V4 1.80 m/s | 0/1 POLICY_FAIL | 3.196 m / 10.48% | 0.17538 / 0.65820 m | 18.75%; terminal 40% |
| Expert 1.80 m/s | 0/1 EXPERT_FAIL | 12.769 m / 41.86% | 0.22262 / 0.83379 m | 59.38%; terminal 66.67% |

The Expert traveled farther than PilotNet V4 at 1.80 m/s, but both valid runs failed. One Expert run does not establish repeatability or performance outside this same-map condition.

## Gate decision

**The current Expert/control configuration itself cannot establish a stable 1.80 m/s lane-following baseline.** Its extensive clamp saturation, increasing CTE while saturated, and healthy loop/infrastructure timing support studying steering authority, fixed 15 Hz control frequency, speed-aware lookahead, vehicle response/simulator dynamics, or a trajectory/speed profile before neural adaptation.

High-speed PilotNet DAgger is therefore **not justified yet** by this feasibility gate. Cone Avoidance V1 should **remain blocked** at a presumed 1.80 m/s baseline. None of those follow-up changes were implemented here.

## Steering observation

The existing `/steering` interface remained unchanged and exposed the actual Pure Pursuit target in radians (`data: ...`):

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

The isolated experiment added a minimal speed overlay, one-shot runner, Expert validation/telemetry wrapper, focused tests, protocol document, and compact result directory. The result directory contains `preflight.json`, `attempt_01.json`, `summary.json`, `experiment.started.json`, and this report. No model, bag, dataset, image, or checkpoint artifact was created.

- Focused experiment and shared-safety tests: 28 passed (5 new plus 23 existing safety tests).
- Full regression: 211 passed.
- `git diff --check` and no-index whitespace checks for every new experiment file: PASS before and after live execution.
- Final branch: `experiment/expert-speed-1p8-v1`; only the six intended isolated experiment paths are untracked.
- No commit or push was performed.
