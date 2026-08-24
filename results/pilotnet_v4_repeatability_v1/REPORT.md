# PilotNet V4 Repeatability V1 — Validation Report

## Result

**PASS. PilotNet V4 achieved 3/3 valid same-map, same-spawn, 0.50 m/s simulation laps.** The two new attempts were both valid `POLICY_PASS`; there were zero policy failures and zero infrastructure failures. No third or fourth live attempt was needed.

## Identity and immutable setup

- Checkpoint SHA-256: `a581c1a6cb13643a0af0ee2d568244291e1eb858516f685ef492a3016501d1d9` — PASS.
- ONNX SHA-256: `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a` — PASS.
- ONNX size: 1,012,518 bytes — PASS.
- Existing inference config SHA-256: `5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45`.
- Architecture/model/preprocessing/ROI/controller/speed/watchdog were unchanged. No model training, DAgger, or data collection occurred.

The read-only lane-follow verifier passed canonical-input integrity, cone-free derived world, byte-identical derived route, metadata, and source-integrity checks. Live environment checks passed: expected world, `switching=false`, 388 route points, 30.504611 m route length, 12×7 m bounds, cone count zero, valid pose, 480×360 HTTP JPEG camera, and valid control API.

The first preflight stopped before driving because orchestration checked spawn before invoking the normal reset lifecycle; the vehicle was still 5.649 m from spawn after the prior experiment. Safe stop succeeded. This implementation ordering error is preserved in `preflight_initial_not_spawn.json`. Only the order was corrected to existing safe-stop/reset/full-preflight behavior; no experimental parameter changed.

## Clock-health method

Before preflight and every live attempt, `/sim/api/clock` was sampled approximately every 50 ms for about two seconds while stopped. Required evidence was forward simulator-time progress, no backward motion, `paused=false`, and maximum observed stall below 0.50 s. This did not weaken the existing 0.75 s runtime watchdog.

| Gate | Samples | Wall / sim elapsed | RTF | Max clock stall | Result |
|---|---:|---:|---:|---:|---|
| Static preflight | 40 | 2.039 / 2.060 s | 1.010 | 0.105 s | PASS |
| Run setup #1 | 40 | 2.035 / 2.055 s | 1.010 | 0.106 s | PASS |
| Run setup #2 | 40 | 2.024 / 2.065 s | 1.020 | 0.105 s | PASS |

## New valid attempts

| Metric | Attempt #1 | Attempt #2 |
|---|---:|---:|
| Classification | POLICY_PASS | POLICY_PASS |
| Elapsed time | 59.759 s | 58.892 s |
| Progress | 30.228 m / 99.09% | 30.181 m / 98.94% |
| Final distance to start | 0.2755 m | 0.2934 m |
| Mean / max CTE | 0.01825 / 0.10182 m | 0.01889 / 0.11646 m |
| Off-track events | 0 | 0 |
| Mean / max absolute steering | 0.10512 / 0.34907 rad | 0.10531 / 0.34907 rad |
| Saturation fraction | 6.97% | 6.11% |
| Mean command delta | 0.00939 rad | 0.00945 rad |
| Camera mean / p95 / max | 2.709 / 4.595 / 39.521 ms | 2.341 / 3.583 / 5.712 ms |
| Preprocess mean / p95 / max | 1.815 / 2.624 / 15.697 ms | 1.760 / 2.305 / 2.906 ms |
| ONNX mean / p95 / max | 1.316 / 4.016 / 15.745 ms | 1.184 / 3.968 / 5.895 ms |
| Loop frequency | 14.883 Hz | 15.000 Hz |
| Loop period mean / p95 / max | 67.192 / 66.694 / 140.515 ms | 66.666 / 66.685 / 68.385 ms |
| Timing slips (>100 ms) | 6 | 0 |
| API / pose-liveness / clock-liveness failures | 0 / 0 / 0 | 0 / 0 / 0 |
| Safe stop | PASS | PASS |

Attempt #1's timing outliers did not produce a clock/API/liveness failure, off-track event, or safety failure. Attempt #2 showed nominal loop timing. No infrastructure attempt was excluded from aggregation because none occurred after the corrected preflight.

## Three-valid-run aggregate

The historical Iteration-2 full-lap pass was included exactly once, with the two new passes:

- Policy success: **3/3**.
- Lap time mean: 59.230 s; sample standard deviation: 0.464 s; range: 58.892–59.759 s.
- Mean CTE by run: 0.01889, 0.01825, 0.01889 m; mean of means: 0.01868 m.
- Worst max CTE: 0.11646 m.
- Steering saturation mean: 6.54%; range: 6.11–6.97%.
- Worst loop p95/max: 66.694/140.515 ms.
- API/liveness failures in valid policy runs: 0/0.
- Safe stops: 3/3.

This establishes repeatability only for the same simulator map, spawn, environment, camera transport, and 0.50 m/s condition. It does not establish unseen-start, lighting, other-map, real-robot, or Raspberry Pi performance.

Static regression testing: 190 tests passed. No checkpoint, ONNX, dataset, canonical V4 evidence, expert controller, or tracked simulator source was modified. No commit or push was performed.
