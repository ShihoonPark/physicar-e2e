# PilotNet DAgger Iteration 2 V1 — Executed Report

## Outcome

The bounded Iteration-2 experiment passed its primary viability gate: V4 completed its first same-map 0.50 m/s lap. Conditional run 2 stopped at 21.33 m because simulator clock did not advance for 0.803 s. It had no off-track event and low CTE at that point. Safe stop succeeded, run 3 was not executed, and no retry or tuning was performed. Therefore this is a one-lap viability result, not 3/3 repeatability or generalization evidence.

## Data collection and extraction

V3 exclusively controlled both collection runs; canonical Pure Pursuit was shadow-label-only. A and B independently reproduced the later failure at 19.818 m (64.97%) after about 40.69 s. Their final-failure divergence onsets were both 39.400 s. The fixed window began 2.0 s earlier, ended at the final valid pre-stop frame, and then enforced route progress at least 10 m.

| Rollout | Role | Selected progress | Samples | Label age mean / p95 / max | Raw bytes | MCAP SHA-256 |
|---|---|---:|---:|---:|---:|---|
| A | train | 18.835–19.938 m | 49 | 48.27 / 95 / 95 ms | 319,432,257 | `9a7792a91ad12875fdce8f909c4be9753133bc6eb5eef7379cf7fdce63d5f77e` |
| B | holdout | 18.840–19.968 m | 49 | 49.39 / 95 / 100 ms | 319,951,391 | `498b4ef7cbad59deb7174061ffe6745a5032d5380720d812210e490d63a00b5d` |

Both had zero future-label violations and no stale-label rejection. The distinct source hashes and frozen roles prove A/B isolation. Contact sheets showed intact later-route curved views, increasing off-nominal state, intact ROI, no cones, and no reset/teleport frame.

## V4 training and offline validation

V4 retained the exact 252,219-parameter PilotNet architecture and fixed canonical preprocessing/training contract. It trained from scratch on 1,849 samples: 1,754 nominal + 46 DAgger1 + 49 DAgger2 A. V2 recovery, both holdouts, and episode_003 were excluded. CUDA PyTorch 2.13.0+cu130 was used. Early stopping completed 22 epochs; best epoch was 15 (train/validation normalized MSE 0.0009447/0.0004765).

| Policy | Nominal MAE / RMSE / bias / max (rad) | Iter2 B MAE / RMSE / bias / max (rad) | Corrective ratio |
|---|---|---|---:|
| V1 | 0.004819 / 0.007332 / +0.001123 / 0.03435 | 0.16853 / 0.23282 / +0.16543 / 0.43296 | 0.381 |
| V3 | 0.004408 / 0.006967 / -0.001860 / 0.03415 | 0.27773 / 0.38918 / +0.27293 / 0.63324 | 0.826 |
| V4 | 0.005698 / 0.007619 / -0.000825 / 0.03801 | 0.02339 / 0.03480 / +0.01698 / 0.10636 | 0.920 |

V4's holdout ratios were 0.967 pre-divergence, 0.887 during divergence, and 0.747 late/failure. Its small nominal regression is reported without masking the large independent on-policy improvement.

Checkpoint: 1,017,389 bytes, SHA-256 `a581c1a6cb13643a0af0ee2d568244291e1eb858516f685ef492a3016501d1d9`.

ONNX: 1,012,518 bytes, SHA-256 `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a`. Checker and I/O contract passed; PyTorch↔ONNX mean/max difference was 2.21e-8/6.24e-8 rad.

## V4 closed loop

| Run | Result | Time | Progress | Mean / max CTE | Off-track | Key timing p95 | Safe stop |
|---|---|---:|---:|---:|---:|---|---|
| 1 | PASS full lap | 59.04 s | 30.208 m (99.03%) | 0.0189 / 0.1099 m | 0 | camera 4.04 ms; preprocess 2.46 ms; ONNX 4.21 ms; loop 66.69 ms | PASS |
| 2 | FAIL liveness | 43.02 s | 21.328 m (69.92%) | 0.0161 / 0.1080 m | 0 | camera 4.13 ms; preprocess 2.63 ms; ONNX 4.21 ms; loop 66.69 ms | PASS |
| 3 | NOT EXECUTED | — | — | — | — | blocked by run-2 failure | — |

Run 1 mean/max absolute steering was 0.1046/0.3491 rad, saturation 6.55%, and mean command delta 0.00937 rad. Run 2 was 0.0807/0.3491 rad, 4.50%, and 0.00681 rad. Neither run had API failures. Run 2 had one liveness failure; it is not classified as an off-track neural-policy failure.

Closed-loop progression was V1 2.953 m, V2 2.591 m, V3 19.819 m, and V4 one full lap at 30.208 m. This supports the causal usefulness of cumulative actual visited-state labels for the primary one-lap gate. It does not establish repeatability because simulator liveness interrupted run 2.

## Interpretation and efficiency

A third DAgger iteration is not currently justified: the targeted policy failure was crossed and a full lap was completed. The next bounded action should be a clean repeatability-only validation after simulator clock liveness is independently stable; it should not collect or train on new data. Fifty blind nominal laps are also not justified. Just 95 cumulative DAgger samples (5.4% of the nominal training count) moved progress from 2.95 m to a full lap.

Iteration-2 raw bags total 639,383,648 bytes (mean 319,691,824); extracted data are 2,143,405 bytes. Based on the measured 460,521,988-byte mean of the three pilot nominal bags, 50 comparable nominal bags would be about 23.03 GB. Iteration-2 raw collection is about 2.78% of that estimate. This is an information-density comparison, not an assertion that the datasets are statistically interchangeable.

Limitations: one successful V4 lap; run-2 liveness interruption; same map/spawn/light/speed only; HTTP JPEG remains live while training images derive from raw ROS RGB; no unseen starts, recovery robustness, other maps, real robot, or Raspberry Pi measurement.
