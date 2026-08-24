# PilotNet V4 Repeatability V1

This validation-only experiment reuses the immutable V4 checkpoint, ONNX model, and inference configuration. One preserved full-lap pass counts exactly once; the bounded runner seeks exactly two new valid 0.50 m/s passes in at most four attempts.

Every attempt resets to canonical spawn, revalidates the world/route/bounds/cone-free environment, and samples simulator clock for about two seconds while stopped. A clock-health failure prevents driving and is recorded as `INFRA_FAIL`. Runtime API, simulator-state, safe-stop, or pose/clock-liveness failures are also infrastructure failures and are excluded from policy aggregation. A genuine `POLICY_FAIL` stops the experiment immediately. No model, data, preprocessing, controller, speed, or simulator source is changed.

## Executed result

Both new evaluations were valid `POLICY_PASS` full laps, with no infrastructure or policy failures. Together with the preserved Iteration-2 pass, V4 achieved 3/3 valid same-map, same-spawn, 0.50 m/s simulation laps. The initial validation preflight exposed and preserved an orchestration ordering error—spawn was checked before the normal reset lifecycle—and no policy drove during it. After changing only that order, all environment and clock-health gates passed. No model, configuration, watchdog, controller, data, or simulator source changed.
