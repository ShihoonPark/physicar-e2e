# Real PhysiCar Temporal PilotNet Runtime V1

Result: **PASS — offline integration candidate only.** This milestone did not
train a model, modify REAL_DATASET_V1, drive a simulator, publish ROS control
commands, or move a physical vehicle. It is not a real-vehicle success claim.

## Frozen model audit

Only the selected REAL-SCRATCH-V1 ONNX was used. Its observed SHA-256 is
`b860afe396c8e48001339b4f99c8b3daa272500725d48d79b9c22b859c6fd339`.
The frozen checkpoint hash is
`02881b5b2d21768c4cf93b71e5d6c2a666043e34c08b71f4247b9545df3dc8e3`;
the freeze and seal hashes are preserved in `summary.json`. ONNX checker,
single float input `N×9×66×200`, single `N×1` radians output, and 255,819
initializer parameters all passed. No simulator D1/D1-R/D2-FE model was used
at runtime.

## Exact runtime contract

The ROS-independent core validates 480×360 `rgb8`, removes row padding, applies
only ROI `x=0:480, y=80:360`, and uses Pillow bilinear resize to 200×66. It then
uses the canonical float32 full-range BT.601 YUV conversion, normalization
`(channel - 0.5) * 2`, CHW layout, and causal channel order `[t-2,t-1,t]`.
Speed is never a neural input.

Three genuine frames are required; there is no startup padding. Arrival times
must increase strictly. A gap over 0.120 s commands a safe stop, invalidates
history, retains the current frame as the first fresh frame, and requires two
more fresh frames before inference. The ROS adapter uses monotonic callback
arrival time rather than an unverified camera-header clock.

Model radians are first bounded to `[-0.35,+0.35]`, then divided by 0.35 once
to create the normalized `/steering` command. Positive remains LEFT. The core
records raw model radians and the calculated normalized command. Speed is a
separate parameter; the default physical-motion authorization is false, so the
runtime speed output is 0.

## Complete canonical bag replay

| Bag | Frames | Warm-up | Predictions | Invalid buffers | Tensor mismatch | Prediction max diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bag_01 | 651 | 2 | 649 | 0 | 0 | 0 rad |
| bag_02 | 1,068 | 4 | 1,064 | 1 | 0 | 0 rad |
| bag_03 | 454 | 4 | 450 | 1 | 0 | 0 rad |

All 2,173 raw camera messages were replayed. Every raw-message runtime tensor
was bit-identical to its frozen REAL_DATASET_V1 PNG tensor, and all 2,163 raw
runtime ONNX predictions were identical to ONNX predictions from the stored
training/evaluation inputs.

For bag_03, the maximum runtime ONNX versus frozen PyTorch checkpoint
difference was `7.45e-8` rad. Runtime validation MAE was `0.0478417432` rad,
RMSE `0.0818924131` rad, and sign agreement `0.808888889`. The maximum
difference among overall frozen metrics was `6.42e-6`; the small five-sample
directional-bin Pearson sensitivity remained below the registered `2e-4`
derived-metric tolerance.

## Recorded-timing no-publish dry run

bag_03's 30.359963 s camera stream replayed at 1× in 30.380520 s (ratio
1.000677). It produced 450 predictions, one camera-watchdog safe stop during
the recorded dropout, zero NaN/Inf values, zero saturation, and normalized
range `[-0.708258,+0.713838]`. No `/steering` or `/speed` message was published.

Measured on this development x86 host only:

| Stage | Mean | p95 | Max |
| --- | ---: | ---: | ---: |
| Preprocessing | 1.256 ms | 1.684 ms | 3.252 ms |
| ONNX | 1.132 ms | 3.930 ms | 7.487 ms |
| Total callback path | 2.446 ms | 5.186 ms | 9.056 ms |

These are host offline measurements, not target-computer or real-vehicle
latency claims.

## Safety and deployment status

The core and publisher contract are separate and import without ROS 2. The ROS
adapter creates control publishers only with explicit `--publish-control`; the
canonical config defaults to `publish_control=false`. It rejects combining
control publication with the development start-gate bypass. Watchdog and fault
tests cover camera loss, ordering/dropout, preprocessing, ONNX failure,
non-finite output, stale-history prevention, publisher failure boundaries, and
neutral speed/steering safe stop.

The real GREEN-signal topic/API is still unverified. No topic was invented.
Default deployment remains `WAITING_FOR_START`, and a verified adapter must be
provided before a physical milestone. The deploy recipe is in
`deploy/real_runtime_v1/`; the deliberately unexecuted first physical procedure
is in `docs/real_runtime_v1_first_physical_test.md`.

Focused tests passed (25 tests, 8 subtests). The full regression passed (568
tests, 47 subtests). Final diff/status evidence is recorded at handoff.
