# High-Speed PilotNet V5 V1 Final Report

## Outcome

Final result: **FAIL — first valid live run was POLICY_FAIL**. Per the preregistered gate, no second or third V5 run, retry, retraining, tuning, or DAgger was performed.

The first run reached 20.688 m / 67.82% in 12.020 s, then sustained off-track for 0.511 s. Mean/max CTE was 0.07289/1.12066 m. Final distance to start was 5.82873 m.

## Reboot recovery

The repository and external userdata were recovered after the host reboot. Evidence shows that stages 1–14 had already completed before the reboot and stage 15 had reached its mandatory stop condition: live attempt 1 was a valid `POLICY_FAIL`. No collection, extraction, training, export, preflight, or policy run was interrupted. The recovery therefore resumed at the gate assessment and final verification only; stages 16 and 17 were not eligible to run.

All 12 finalized MCAPs were reused in place. Their SHA-256 identities match the extraction records, their opening and closing MCAP magic is intact, and the completed extractor had read all 12 with zero decode failures. Dataset metadata, all 12 episode manifests, the V5 checkpoint, and the V5 ONNX also match their recorded hashes and sizes. No valid bag episode was recollected, no completed live operation was duplicated, and no external artifact was regenerated or overwritten.

## Frozen inputs and preservation

High-Speed Expert V1 remained 1.80 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad, 0.18 m wheelbase, same spawn/route/world/safety. Collection was 12/12 PASS and produced 12 finalized bags.

Canonical V4 was not edited or initialized into V5. Its config SHA-256 remains `5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45`; its ONNX remains 1,012,518 bytes with SHA-256 `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a`. The 0.50 m/s dataset is absent from the V5 manifests.

## Data and model gates

- Collection: 12/12 Expert laps and bags PASS; total raw size 1,741,580,488 bytes; camera 3,330 messages at mean 15.122 Hz; steering 2,912 messages at mean 13.224 Hz; safe stop 12/12.
- Extraction: 2,911 accepted samples; 87.42% overall and 100% active-window retention; zero future labels, stale labels, and decode failures. Representative contact sheets passed review.
- Split: 001–008 training (1,940), 009–010 validation (486), 011–012 closed holdout (485), with no episode leakage.
- Model: unchanged 252,219-parameter PilotNet, trained from scratch. Best epoch 14; validation/holdout MAE 0.01012/0.01040 rad.
- Artifacts: checkpoint `04cc5934…3a101`; ONNX `404b2ea2…8a1fd`. ONNX equivalence PASS, maximum error 6.24e-8 rad.

## Simulator preflight

PASS before the live marker and again before attempt 1: exact expected world, switching=false, cones=0, 388 route points, route length 30.504611 m, bounds 12×7 m, valid spawn/pose, 480×360 HTTP camera, advancing clock, and safe stop.

## Live attempt 1

| Metric | Result |
|---|---:|
| Classification | POLICY_FAIL |
| Elapsed / progress / completion | 12.020 s / 20.688 m / 67.82% |
| Final distance to start | 5.82873 m |
| Mean / max CTE | 0.07289 / 1.12066 m |
| Off-track events / duration | 1 / 0.51083 s |
| Mean / max absolute steering | 0.08057 / 0.349066 rad |
| Steering saturation / mean command delta | 2.78% / 0.02195 rad |
| Camera latency mean/p95/max | 2.790 / 3.869 / 5.257 ms |
| Preprocessing latency mean/p95/max | 2.060 / 2.762 / 4.762 ms |
| ONNX latency mean/p95/max | 1.772 / 4.265 / 4.660 ms |
| Loop frequency | 15.00038 Hz |
| Loop period mean/p95/max | 66.665 / 66.685 / 88.093 ms |
| Timing slips >100 ms | 0 |
| API / combined pose-clock liveness failures | 0 / 0 |
| Per-run / final safe stop | PASS / PASS |

The run reached the physical steering clamp, but saturation was only 2.78%. The Expert completed the same route while averaging about 4.97% saturation over the 12 collection laps. Combined with healthy timing and the large terminal max CTE relative to mean CTE, this evidence is more consistent with closed-loop distribution shift or delayed corrective response than compute failure or globally insufficient steering authority. The available compact telemetry does not support a more specific causal claim.

## Progression

| Controller | Speed/configuration | Result |
|---|---|---|
| Expert | 0.50 m/s, lookahead 0.45 | 5/5 PASS |
| PilotNet V4 | 0.50 m/s | 3/3 PASS |
| PilotNet V4 | 1.80 m/s | 0/1; FAIL at 3.196 m / 10.48% |
| Expert | 1.80 m/s, lookahead 0.45 | FAIL at 41.86% |
| High-Speed Expert V1 | 1.80 m/s, lookahead 0.90 | 3/3 validation PASS; 12/12 collection PASS |
| High-Speed PilotNet V5 | 1.80 m/s | 0/1; FAIL at 20.688 m / 67.82% |

V5 cannot be frozen as the canonical 1.80 m/s policy, and Cone Avoidance V1 is not yet justified. The dataset and model are valid negative-evidence artifacts. A future bounded high-speed on-policy diagnosis/DAgger experiment is justified, but was not performed here.

## Steering observability

The simulator's existing `/steering` topic remained the actual target path. Monitor it with:

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

## Limitations

The collection and live runs use different camera transports (raw ROS RGB for training, HTTP JPEG live), as in the established pipeline. Only one valid neural live run was permitted after the failure, so no repeatability claim is possible. The evidence is same-map/same-spawn simulation only.

## Tests and repository state

Focused V5 tests: 10/10 PASS. Full regression: 233/233 PASS. `git diff --check` passed. The ONNX checker/equivalence and live preflight gates also passed.

Added isolated configuration files for the Expert, collector, extractor, V5 training, and V5 inference; one orchestration module and CLI wrapper; focused tests; documentation; and compact evidence under:

- `results/high_speed_collection_v1/`
- `results/high_speed_dataset_v1/`
- `results/pilotnet_training_v5_high_speed/`
- `results/pilotnet_e2e_v5_high_speed/`

Large raw bags, images, checkpoint, and ONNX remain under external simulator userdata. Final `physicar-e2e` branch is `feature/pilotnet-high-speed-v5-v1`; all experiment files are untracked additions and there are no tracked modifications. No commit or push was performed. The simulator source checkout has no tracked source-code edits from this task; its runtime `userdata/last_world` state is reported modified.
