# PilotNet DAgger Iteration 2 V1

This bounded experiment asks whether cumulative on-policy labels extend V3's same-map 0.50 m/s robustness to a full lap. V3 controls exactly two new shadow rollouts: A is training-only and B is holdout-only. Pure Pursuit supplies labels but has no command authority.

Rollout A must reproduce the later distribution: failure progress must be at least 10 m. An earlier failure or unexpected full lap stops the experiment before extraction and V4 training. Both extraction windows are continuous from 2.0 seconds before the fixed objective divergence onset through the final valid pre-safe-stop frame, after which every accepted sample must have route progress at least 10 m. Fewer than 20 frames is a hard failure; there is no fallback.

Raw camera header simulator time is causally aligned to the latest prior shadow-expert simulator-clock telemetry, with a 150 ms age gate and zero future labels. Stored images retain the canonical raw-RGB ROI and resize. Contact sheets are visual gates, not inputs to sample selection.

V4 trains from scratch on nominal episodes 001–002, DAgger1 rollout A, and Iteration-2 rollout A. DAgger1/2 holdouts and V2 recovery data are excluded. Architecture, preprocessing, MSE, Adam, learning rate, batch size, seed, and early stopping remain fixed.

V4 live validation permits one first 0.50 m/s run. Failure prevents retries; a full-lap pass permits exactly two more independent reset laps. All findings are same-map simulator evidence only.

## Executed result

Both V3 shadow rollouts reproduced the later failure at 19.818 m. Their frozen A/B windows each contained 49 causally labeled frames at 18.84–19.97 m, with zero future labels. V4 was trained from scratch on 1,754 nominal, 46 Iteration-1, and 49 Iteration-2 samples. It completed its first 0.50 m/s lap. The second conditional run stopped at 21.33 m because simulator clock liveness failed, not because of off-track behavior; the third run was therefore not executed. This establishes one-lap Iteration-2 viability but not 3/3 repeatability.
