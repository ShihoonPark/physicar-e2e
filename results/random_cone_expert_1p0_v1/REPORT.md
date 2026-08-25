# Random Cone Expert 1.0 m/s V1

Final simulation gate: **PASS**.

This is simulator-only evidence. It is not real-robot evidence.

## Frozen contract

- Seed: `20260825`
- Map family: `71e69ee938032295503bfed557fde18c`
- Split: scenarios 01–08 TRAIN, 09–10 VALIDATION, 11–12 UNSEEN_HOLDOUT
- Control: 1.00 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad, 0.18 m wheelbase
- Practical pass condition: positive footprint clearance and no cone contact/intersection
- No neural training and no training-bag collection

## Scenario results

| ID | Role | Geometry | s (m) | Side | Offset (m) | Actual clearance (m) | Collision | Recovery | Result |
|---:|---|---|---:|---|---:|---:|---|---|---|
| 01 | TRAIN | moderate_left_curve | 19.20 | right | 0.254 | 0.0193 | false | true | RANDOM_CONE_EXPERT_PASS |
| 02 | TRAIN | moderate_left_curve | 18.55 | left | 0.319 | 0.1237 | false | true | RANDOM_CONE_EXPERT_PASS |
| 03 | TRAIN | low_curvature | 15.10 | left | 0.254 | 0.0544 | false | true | RANDOM_CONE_EXPERT_PASS |
| 04 | TRAIN | moderate_right_curve | 21.15 | left | 0.254 | 0.0521 | false | true | RANDOM_CONE_EXPERT_PASS |
| 05 | TRAIN | low_curvature | 9.25 | right | 0.254 | 0.0565 | false | true | RANDOM_CONE_EXPERT_PASS |
| 06 | TRAIN | moderate_right_curve | 13.75 | right | 0.299 | 0.1601 | false | true | RANDOM_CONE_EXPERT_PASS |
| 07 | TRAIN | low_curvature | 7.40 | left | 0.254 | 0.0570 | false | true | RANDOM_CONE_EXPERT_PASS |
| 08 | TRAIN | moderate_left_curve | 12.30 | right | 0.254 | 0.0037 | false | true | RANDOM_CONE_EXPERT_PASS |
| 09 | VALIDATION | moderate_left_curve | 13.00 | right | 0.254 | 0.0809 | false | true | RANDOM_CONE_EXPERT_PASS |
| 10 | VALIDATION | low_curvature | 6.25 | right | 0.254 | 0.0555 | false | true | RANDOM_CONE_EXPERT_PASS |
| 11 | UNSEEN_HOLDOUT | low_curvature | 11.00 | right | 0.254 | 0.0152 | false | true | RANDOM_CONE_EXPERT_PASS |
| 12 | UNSEEN_HOLDOUT | moderate_right_curve | 14.15 | right | 0.294 | 0.1375 | false | true | RANDOM_CONE_EXPERT_PASS |

## Detailed live metrics

| ID | Lap time | Completion | Mean/max CTE (m) | Off-track events | Recovery/time | Steering saturation | Loop/slips | API/pose/clock failures | Safe stop |
|---:|---:|---:|---:|---:|---|---:|---|---|---|
| 01 | 28.283 | 99.21% | 0.0634/0.2709 | 0 | true/0.733s | 3.76% | 15.000 Hz/0 | 0/0/0 | true |
| 02 | 27.685 | 99.19% | 0.0701/0.3885 | 0 | true/0.333s | 0.00% | 15.000 Hz/0 | 0/0/0 | true |
| 03 | 27.680 | 99.21% | 0.0681/0.3302 | 0 | true/0.399s | 1.68% | 15.001 Hz/0 | 0/0/0 | true |
| 04 | 28.222 | 99.19% | 0.0619/0.3768 | 0 | true/0.334s | 0.00% | 15.000 Hz/0 | 0/0/0 | true |
| 05 | 27.955 | 99.19% | 0.0814/0.2580 | 0 | true/0.400s | 0.24% | 15.000 Hz/0 | 0/0/0 | true |
| 06 | 28.088 | 99.20% | 0.0804/0.3802 | 0 | true/0.334s | 0.00% | 15.000 Hz/0 | 0/0/0 | true |
| 07 | 27.886 | 99.08% | 0.0842/0.2583 | 0 | true/0.333s | 0.95% | 15.000 Hz/0 | 0/0/0 | true |
| 08 | 28.279 | 99.27% | 0.0738/0.2861 | 0 | true/0.332s | 0.94% | 15.000 Hz/0 | 0/0/0 | true |
| 09 | 28.220 | 99.24% | 0.0732/0.3234 | 0 | true/0.799s | 0.24% | 15.000 Hz/0 | 0/0/0 | true |
| 10 | 27.882 | 99.18% | 0.0849/0.2569 | 0 | true/0.333s | 0.24% | 15.000 Hz/0 | 0/0/0 | true |
| 11 | 28.219 | 99.23% | 0.0736/0.2538 | 0 | true/0.665s | 0.94% | 15.000 Hz/0 | 0/0/0 | true |
| 12 | 28.086 | 99.28% | 0.0790/0.3730 | 0 | true/0.338s | 0.95% | 15.000 Hz/0 | 0/0/0 | true |

## Aggregate and failures

- Valid policy runs: `12/12`
- Scenario passes: `12/12`
- Cone contact/intersection: `0/12`
- Minimum actual footprint clearance: `0.003716 m`
- Successful recoveries: `12/12`
- Per-scenario safe stops: `12/12`
- Final safe stop: `true`

No genuine policy failure occurred in the twelve valid runs.

## Infrastructure-only attempts

- Scenario 08 attempt 1: RuntimeError: derived world did not become ready before timeout: {'running': True, 'websocket': True, 'current': 'custom_71e69ee938032295503bfed557fde18c_e2e_random_cone_v1_07', 'switching': False, 'assets_version': 1787650931}

Each listed attempt stopped safely and received at most the one permitted fresh replacement. No genuine policy run was retried.

## Baselines and disposition

Temporal PilotNet V9, Fixed Cone Avoidance Expert V1, practical Temporal PilotNet C1, and the 1.8 m/s Random Cone stress evidence were hash-audited as applicable and left unchanged.

- Config SHA-256: `a6f2573bfaa041dafaca25d957c07b5ea1acbd33399e2aece6e1bda06d238cf5`
- Offline evidence SHA-256: `64bcebe136597337fe6256946b765ac2ceb57733bdb64e7a0f17004c34ac030d`
- Baseline audit before/after: `PASS/PASS`
- Tracked simulator source changes after run: `0`
- Neural training performed: `false`
- Training bags collected: `false`

Random Cone Expert 1.0 m/s V1 frozen: **true**.
PASS-qualified 8/2/2 split release frozen: **true**.
Random-cone bag collection justified: **true**.

## Verification and repository state

- Post-live full regression: `346 passed, 39 subtests passed`, with two existing
  ONNX deprecation warnings.
- Exact world verification: PASS; 12 byte-identical routes and exactly one cone
  in each scenario world.
- Scenario 01 preflight-only gate: PASS, including safe stop and world restore.
- Parent 1.8 m/s results manifest before/after: unchanged at
  `f776182f8d6b92ca62d9ddc2b1f1f1ebf748af5f32dc51d5d0163f84d8f82e5a`.
- Tracked simulator source changes: none.
- Neural training and training-bag collection: not performed.
- No commit or push was performed.

## Limitations

The bicycle preview is an offline feasibility model; the reported live rows are simulator measurements. Curvature classes use a 0.5 m route window on the original piecewise-polyline route. The 5 cm value is a planning margin, not a practical success threshold. Scenario 08's `0.003716 m` minimum live clearance is positive and collision-free but extremely narrow; this simulation PASS is not a real-robot safety claim.
