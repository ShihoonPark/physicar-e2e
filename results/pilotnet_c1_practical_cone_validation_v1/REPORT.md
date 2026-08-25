# Temporal PilotNet C1 Practical Cone Validation V1

## Decision

**PASS: 3/3 valid simulation runs were `PRACTICAL_CONE_PASS`.**

Temporal PilotNet C1 achieved 3/3 valid fixed-one-cone, 1.80 m/s simulation
laps without cone contact using only causal camera observations.

The minimum conservative footprint-to-cone clearance across the three runs
was `0.037644101 m`. Clearance remained a measured metric, but the historical
`0.050000 m` experimental margin was not a pass/fail threshold in this
practical validation.

This is simulator evidence only and is not a real-robot success claim.

## Preserved C1 identities and historical result

- C1 checkpoint SHA-256:
  `1e90002ca139b3cfb0f34074e013e52b6754df33ed0e3b438ca81809c9e2ee39`.
- C1 ONNX SHA-256:
  `22440ad61f6e5136b33016eb0781d79ab71637e659478ac0c92cc04cffc98e5f`.
- C1 training-summary SHA-256:
  `ea2f3c1dd65357cb2c5c8b12d8035515f0d612b6146a14b0bc4c2e8ece217f38`.
- Architecture/input: unchanged Temporal PilotNet, `255,819` parameters,
  `9x66x200`, three causal frames.
- Frozen world:
  `custom_71e69ee938032295503bfed557fde18c_e2e_cone_avoidance_v1`.
- Frozen cone pose: route `s=6.9 m`,
  `(x, y)=(6.165700204349249, 1.2298027858176892) m`.
- Frozen speed/control/steering: `1.80 m/s`, `15 Hz`, `+/-0.349066 rad`.

The preserved historical result remains `CONE_POLICY_FAIL` under the 5 cm
experimental clearance contract: minimum clearance `0.043582667 m`, vehicle
intersection false, safe stop PASS. Exact SHA-256 guards for all four files in
`results/pilotnet_e2e_c1_cone_temporal/` passed before and after driving; none
were edited or relabeled.

No training, bag/data collection, DAgger, world/cone/speed/architecture, or
steering changes were performed.

## Practical contract

A run passes only when the full lap completes, vehicle/cone collision or
intersection remains false, unchanged sustained off-track safety does not
trigger, return-to-nominal-route succeeds, temporal and infrastructure checks
pass, and safe stop succeeds. An actual intersection remains an immediate
`PRACTICAL_CONE_FAIL`; a positive clearance below 0.05 m alone does not stop or
fail a run.

## Per-run results

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Classification | PRACTICAL_CONE_PASS | PRACTICAL_CONE_PASS | PRACTICAL_CONE_PASS |
| Lap time (s) | 17.278231 | 17.297029 | 17.231087 |
| Completion fraction | 0.992201 | 0.991717 | 0.989935 |
| Progress (m) | 30.266714 | 30.251955 | 30.197591 |
| Minimum clearance (m) | 0.037644101 | 0.044882583 | 0.038405232 |
| Route s at minimum (m) | 7.086583 | 7.087119 | 7.073593 |
| Vehicle/cone intersection | false | false | false |
| Nominal mean/max CTE (m) | 0.091619 / 0.527772 | 0.095005 / 0.523592 | 0.097485 / 0.539339 |
| Off-track events / total duration (s) | 1 / 0.530454 | 1 / 0.468924 | 1 / 0.536403 |
| Recovery success / time (s) | true / 0.529193 | true / 0.531916 | true / 0.530213 |
| Recovery max CTE (m) | 0.003248 | 0.004619 | 0.003890 |
| Mean/max absolute steering (rad) | 0.129302 / 0.349066 | 0.129986 / 0.349066 | 0.131163 / 0.349066 |
| Steering saturation fraction | 0.054264 | 0.062016 | 0.062257 |
| Temporal adjacent gap maxima (s) | 0.071106 / 0.069561 | 0.076397 / 0.076397 | 0.070555 / 0.070054 |
| Oldest-current span max (s) | 0.135823 | 0.141484 | 0.136340 |
| ONNX inference mean/max (ms) | 1.288805 / 5.090453 | 1.394312 / 5.180342 | 1.369813 / 5.357222 |
| Full model path mean/max (ms) | 3.056165 / 7.050480 | 3.143677 / 7.471006 | 3.161816 / 7.189329 |
| Control period mean/max (ms) | 66.513873 / 66.733693 | 66.565255 / 74.550536 | 66.571754 / 66.753520 |
| Control frequency (Hz) | 15.034458 | 15.022852 | 15.021386 |
| Timing slips over 100 ms | 0 | 0 | 0 |
| Temporal/API/pose/clock failures | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| Safe stop | PASS | PASS | PASS |

The unchanged off-track monitor recorded one event in each run, but none
triggered its sustained off-track failure. The reported total durations include
the monitor's finalization interval after its last safety sample and are
preserved as measured. All laps completed and all return-route criteria passed;
road safety thresholds and monitor behavior were not changed.

Run 1 passed, so independent reset runs 2 and 3 were executed. There were no
`INFRA_FAIL` or `TEMPORAL_INPUT_FAIL` attempts and therefore no replacements.
Exactly three total attempts and three valid policy runs were used.

## Baseline and next stage

The fixed-one-cone practical simulation baseline is complete. Random/unseen
cone placement work is justified as the next engineering target, but no such
training or implementation was performed in this task.

## Verification and Git status

- Focused tests: `20 passed` before live validation.
- Full regression after live validation: `328 passed`, `39 subtests passed`,
  with two pre-existing ONNX deprecation warnings.
- `git diff --check`: PASS.
- Branch: `test/cone-avoidance-c1-practical-v1`.
- E2E worktree contains only the intended uncommitted practical-validation
  source/config/test/result changes.
- External simulator tracked source changes: none. Its only tracked runtime
  difference remains `userdata/last_world`.
- No commit or push was performed.
