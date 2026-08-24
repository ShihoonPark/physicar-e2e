# PilotNet V4 High-Speed 1.80 m/s Validation V1

## Final result

**FAIL.** The first valid 1.80 m/s neural-policy evaluation ended in sustained off-track. Per protocol, the complete experiment stopped immediately. Attempts #2 and #3 were not run, no infrastructure replacement was used, and no retry, tuning, training, DAgger, data collection, or cone-avoidance work occurred.

## Isolation and identity

- PilotNet V4 remained camera-only with 252,219 parameters.
- Canonical ONNX SHA-256: `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a` — PASS.
- Canonical ONNX size: 1,012,518 bytes — PASS.
- Canonical 0.50 m/s config SHA-256: `5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45` — unchanged.
- Canonical `results/pilotnet_e2e_v4/` and `results/pilotnet_v4_repeatability_v1/` remained unchanged.
- Fixed speed was exactly 1.80 m/s. Control frequency remained 15 Hz and the steering clamp remained ±0.349066 rad.
- HTTP JPEG 480×360 camera input, ROI y=160:360, bilinear 200×66 resize, RGB→YUV, normalization, route, world, spawn, and all safety semantics were inherited from canonical V4.

At 15 Hz, nominal travel per neural update is approximately 0.0333 m at 0.50 m/s and exactly 0.1200 m at 1.80 m/s. The 1.80 m/s vehicle therefore travels 3.6× farther between updates. No frequency change or tuning was made.

## Preflight

The independent preflight and the fresh attempt #1 preflight both passed the normal safe lifecycle: safe stop → reset → full preflight → drive. Evidence verified the exact world `custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1`, `switching=false`, zero cones, 388 route points, 30.504611 m route length, 12×7 m bounds, canonical spawn and valid pose, valid numeric control API, 480×360 JPEG camera, and advancing unpaused simulator clock. Source and derived environment integrity checks passed.

## Attempt #1

| Metric | Measured value |
|---|---:|
| Classification | POLICY_FAIL |
| Failure | Sustained off-track; terminal boundary distance 0.208 m |
| Elapsed | 2.161 s |
| Progress | 3.196 m / 10.48% |
| Final distance to start | 2.9068 m |
| Mean / max CTE | 0.17538 / 0.65820 m |
| Off-track events / sustained duration | 1 / 0.5350 s |
| Mean / max absolute steering | 0.11793 / 0.349066 rad |
| Steering saturation | 18.75% overall; 40.00% over terminal 15 samples |
| Mean command delta | 0.04650 rad overall; 0.09214 rad terminal |
| Camera latency mean / p95 / max | 2.218 / 3.428 / 4.238 ms |
| Preprocessing latency mean / p95 / max | 1.697 / 2.471 / 3.149 ms |
| ONNX latency mean / p95 / max | 1.103 / 3.549 / 4.040 ms |
| Loop frequency | 15.002 Hz |
| Loop period mean / p95 / max | 66.656 / 66.694 / 66.704 ms |
| Timing slips >100 ms | 0 |
| API failures | 0 |
| Pose / clock liveness failures | 0 / 0 |
| Runtime safe stop / final safe stop | PASS / PASS |

Attempts #2 and #3: **not executed**, because attempt #1 was a valid policy failure. Infrastructure attempts: **zero**.

## Failure analysis

For an objective divergence reference, telemetry used the first CTE exceeding the canonical 0.50 m/s worst observed maximum of 0.116457 m:

- Divergence: route s=2.183 m, CTE=0.14352 m.
- First off-track boundary breach: route s=2.342 m, CTE=0.46282 m.
- Sustained off-track stop: progress=3.196 m.
- The final 15-sample CTE mean rose from 0.17778 m in the early half to 0.61158 m in the late half, an abrupt 0.43380 m terminal increase.
- Predicted steering reached the ±0.349066 rad physical clamp; terminal saturation was 40%.
- Terminal mean absolute command delta was 0.09214 rad, showing aggressive alternating/corrective command movement rather than a compute stall.
- Loop p95/max remained 66.694/66.704 ms, ONNX p95/max remained 3.549/4.040 ms, and there were no >100 ms slips, API failures, or liveness failures.

The best evidence-based category is **likely insufficient steering authority at speed**. The policy reached the fixed physical limit while CTE grew abruptly, and compute/control timing was healthy. Higher-speed closed-loop distribution shift or delayed correction may contribute, but timing/compute failure is not supported by the evidence.

## 0.50 vs 1.80 m/s

| Metric | Canonical 0.50 m/s | 1.80 m/s attempt #1 |
|---|---:|---:|
| Valid full laps | 3/3 POLICY_PASS | 0/1; POLICY_FAIL |
| Mean lap time / elapsed to stop | 59.230 s | 2.161 s |
| Mean-of-mean / run mean CTE | 0.01868 m | 0.17538 m |
| Worst / run max CTE | 0.11646 m | 0.65820 m |
| Mean / run saturation | 6.54% | 18.75% |
| Loop rate | ≈15 Hz | 15.002 Hz |

This result applies only to the same simulator map, route, spawn, and camera contract. It does not establish generalization elsewhere.

## Decision

The 1.80 m/s lane-following baseline cannot be frozen: it failed its first valid evaluation. Cone Avoidance V1 should not start on a presumed-stable 1.80 m/s baseline. This does not automatically justify DAgger; whether high-speed adaptation is worthwhile should be decided from the speed objective and the observed steering-authority limitation under a separately authorized task.

## Steering observation

The existing `/steering` path was unchanged and continued to carry the actual clamped neural target in radians (`data: ...`):

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

The 1.80 m/s experiment added an isolated config, source wrapper, launch script, focused test, protocol document, and compact result directory. Only `preflight.json`, `attempt_01.json`, `summary.json`, `experiment.started.json`, and this report were created; no later attempt files, bags, images, checkpoints, or training artifacts exist.

- Focused experiment/shared-safety tests: 23 passed.
- Full regression: 212 passed.
- `git diff --check` and no-index checks for every new experiment file: PASS before and after live execution.
- Final branch: `experiment/pilotnet-v4-speed-1p8-v1`; intended 1.80 m/s paths are untracked, alongside preserved untracked 1.50 m/s evidence from the preceding experiment.
- No commit or push was performed.
