# Random-Cone R1 S09 Failure Diagnosis

Result: **PASS_WITH_TELEMETRY_LIMITATION**

The immutable R1/hash audit passed. S09 failed during the moderate-left avoidance approach at s=12.017588 m, 0.982412 m before the cone. Maximum CTE reached 0.803517 m and recovery failed; contact, saturation, timing, temporal-input, API, pose, clock and liveness faults did not explain the stop.

Offline nominal S09 remained strong (MAE 0.005641 rad, correlation 0.996664); avoidance MAE was 0.014503 rad.

The preserved live evidence has no per-tick pose/steering trace and no live images. Therefore the requested counterfactual Expert commands, stable/divergence/final-2s comparisons, CTE-growth rate and feature distance cannot be reconstructed honestly. No S09 label was generated or admitted to training.

Conclusion: aggregate evidence supports learner-state distribution shift/closed-loop error accumulation. It does not distinguish under-command, wrong sign, delay, trace-level temporal instability, or a constant-bias mechanism.
