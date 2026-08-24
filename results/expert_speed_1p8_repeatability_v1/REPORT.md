# Expert 1.80 m/s Repeatability V1

## Final result

**PASS. Expert 1.80 m/s / lookahead 0.90 m achieved 3/3 valid same-map, same-spawn simulation laps.** The preserved characterization pass was counted exactly once and both new valid evaluations passed. No infrastructure replacement, fifth attempt, tuning, data collection, training, DAgger, simulator change, commit, or push occurred.

The configuration can now be frozen as **High-Speed Expert V1**.

## Frozen configuration and provenance

- Speed: exactly 1.80 m/s.
- Pure Pursuit lookahead: exactly 0.90 m.
- Control frequency: exactly 15 Hz.
- Steering limit: ±0.349066 rad.
- Wheelbase: 0.18 m.
- World: `custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1`.
- Route, spawn, controller implementation, safety geometry, liveness, lap completion, and safe-stop semantics: unchanged.
- Canonical Expert config SHA-256: `63814d3a30f8753092cd33fc53d44414cfb343e39caf805e624dbaf33a4bd050` — unchanged.

Historical provenance: `results/expert_speed_1p8_lookahead_v1/attempt_03.json`, SHA-256 `7e1b5e60fbceac6167c70c6643d4c6563a71ef92e13960c907527ddb67a1b345`. It was an `EXPERT_PASS` under the exact frozen contract and was counted once.

## Preflight and attempts

The independent preflight and both live-attempt preflights passed safe stop → reset → full environment/clock preflight. Each verified the exact cone-free world, `switching=false`, 388 route points, 30.504611 m route, 12×7 m bounds, canonical spawn, valid pose/control API, and advancing unpaused simulator clock.

| Metric | Historical PASS | New attempt #1 | New attempt #2 |
|---|---:|---:|---:|
| Classification | EXPERT_PASS | EXPERT_PASS | EXPERT_PASS |
| Lap time | 16.019 s | 16.085 s | 16.013 s |
| Progress | 30.100 m / 98.67% | 30.255 m / 99.18% | 30.059 m / 98.54% |
| Final distance to start | 0.2121 m | 0.2522 m | 0.2522 m |
| Mean / max CTE | 0.05562 / 0.23236 m | 0.04923 / 0.21736 m | 0.05148 / 0.23152 m |
| Off-track events / duration | 0 / 0 s | 0 / 0 s | 0 / 0 s |
| Mean / max absolute steering | 0.11402 / 0.349066 rad | 0.10854 / 0.349066 rad | 0.11183 / 0.349066 rad |
| Steering saturation | 5.39% | 5.37% | 4.98% |
| Mean command delta | 0.02509 rad | 0.02414 rad | 0.02505 rad |
| Loop frequency | 15.001 Hz | 15.001 Hz | 15.000 Hz |
| Period mean / p95 / max | 66.663 / 66.689 / 69.165 ms | 66.664 / 66.686 / 67.399 ms | 66.665 / 66.695 / 82.978 ms |
| Timing slips >100 ms | 0 | 0 | 0 |
| API / pose / clock failures | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Safe stop | PASS | PASS | PASS |

Infrastructure attempts: **zero**. Only `attempt_01.json` and `attempt_02.json` were created; no replacement attempt files exist. Final orchestrator safe stop: **PASS**.

## Three-valid-lap aggregate

- Expert success: **3/3**.
- Lap-time mean: 16.03881 s; sample standard deviation: 0.03974 s; range: 16.01303–16.08457 s.
- Mean CTE per run: 0.05562, 0.04923, 0.05148 m.
- Mean of run-mean CTE: 0.05211 m.
- Worst max CTE: 0.23236 m.
- Steering saturation mean: 5.25%; range: 4.98–5.39%.
- Successful safe stops: 3/3.
- Infrastructure failures: 0.

CTE, saturation, lap time, and loop timing were consistent across the three valid runs. This establishes repeatability only for the same simulator map, spawn, route, vehicle, 1.80 m/s speed, and 0.90 m lookahead configuration.

## Decision and future model note

High-Speed Expert V1 can be frozen. The next stage is justified: **1.80 m/s Expert ROS bag collection of camera + steering**, followed by **separate High-Speed PilotNet V5 training**. Neither stage was executed here.

Future training should not blindly mix the speed-dependent 1.80 m/s/0.90 m Expert labels with canonical 0.50 m/s/0.45 m labels in the existing speed-unaware V4. Preserve V4 as the canonical 0.50 m/s model and create a separate V5 unless a future architecture explicitly includes speed as an input.

## Files, tests, and Git

The isolated validation added a frozen config, bounded runner, repeatability module, focused tests, protocol document, and compact result directory. No MCAP, bag, images, labels, model, or checkpoint was created.

- Focused and shared-safety tests: 40 passed.
- Full regression: 223 passed.
- `git diff --check` and no-index whitespace checks for every new experiment file: PASS before and after live execution.
- Final branch: `test/expert-speed-1p8-repeatability-v1`; only the six intended isolated validation paths are untracked.
- No commit or push was performed.
