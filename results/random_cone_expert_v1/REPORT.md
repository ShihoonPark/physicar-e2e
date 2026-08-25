# Random Cone Expert V1

Final simulation gate: **FAIL**.

This is simulator-only evidence. It is not real-robot evidence.

## Frozen contract

- Seed: `20260825`
- Map family: `71e69ee938032295503bfed557fde18c`
- Split: scenarios 01–08 TRAIN, 09–10 VALIDATION, 11–12 UNSEEN_HOLDOUT
- Control: 1.80 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad, 0.18 m wheelbase
- Practical pass condition: positive footprint clearance and no cone contact/intersection
- No neural training and no training-bag collection

## Scenario results

| ID | Role | Geometry | s (m) | Side | Offset (m) | Actual clearance (m) | Collision | Recovery | Result |
|---:|---|---|---:|---|---:|---:|---|---|---|
| 01 | TRAIN | moderate_left_curve | 19.20 | right | 0.254 | 0.0256 | false | false | RANDOM_CONE_EXPERT_FAIL |
| 02 | TRAIN | moderate_left_curve | 18.55 | left | 0.319 | 0.1599 | false | true | RANDOM_CONE_EXPERT_PASS |
| 03 | TRAIN | low_curvature | 15.10 | left | 0.254 | 0.0597 | false | true | RANDOM_CONE_EXPERT_PASS |
| 04 | TRAIN | moderate_right_curve | 21.15 | left | 0.254 | 0.0868 | false | true | RANDOM_CONE_EXPERT_PASS |
| 05 | TRAIN | low_curvature | 9.25 | right | 0.254 | 0.0567 | false | true | RANDOM_CONE_EXPERT_PASS |
| 06 | TRAIN | moderate_right_curve | 13.75 | right | 0.299 | 0.1454 | false | true | RANDOM_CONE_EXPERT_PASS |
| 07 | TRAIN | low_curvature | 7.40 | left | 0.254 | 0.0569 | false | true | RANDOM_CONE_EXPERT_PASS |
| 08 | TRAIN | moderate_left_curve | 12.30 | right | 0.254 | 0.0370 | false | true | RANDOM_CONE_EXPERT_PASS |
| 09 | VALIDATION | moderate_left_curve | 13.00 | right | 0.254 | 0.0916 | false | true | RANDOM_CONE_EXPERT_PASS |
| 10 | VALIDATION | low_curvature | 6.25 | right | 0.254 | 0.0555 | false | true | RANDOM_CONE_EXPERT_PASS |
| 11 | UNSEEN_HOLDOUT | low_curvature | 11.00 | right | 0.254 | 0.0318 | false | true | RANDOM_CONE_EXPERT_PASS |
| 12 | UNSEEN_HOLDOUT | moderate_right_curve | 14.15 | right | 0.294 | 0.1251 | false | true | RANDOM_CONE_EXPERT_PASS |

## Detailed live metrics

| ID | Lap time | Completion | Mean/max CTE (m) | Off-track events | Recovery/time | Steering saturation | Loop/slips | API/pose/clock failures | Safe stop |
|---:|---:|---:|---:|---:|---|---:|---|---|---|
| 01 | 11.818 | 67.82% | 0.0787/0.7273 | 1 | false | 7.34% | 15.001 Hz/0 | 0/0/0 | true |
| 02 | 16.156 | 99.39% | 0.0658/0.4140 | 1 | true/1.400s | 4.94% | 15.001 Hz/0 | 0/0/0 | true |
| 03 | 16.485 | 99.32% | 0.0684/0.2965 | 0 | true/0.331s | 7.66% | 15.001 Hz/0 | 0/0/0 | true |
| 04 | 16.483 | 99.39% | 0.0710/0.3097 | 0 | true/0.868s | 6.45% | 15.001 Hz/0 | 0/0/0 | true |
| 05 | 16.215 | 99.11% | 0.0791/0.2574 | 0 | true/0.333s | 5.74% | 15.000 Hz/0 | 0/0/0 | true |
| 06 | 16.350 | 99.37% | 0.0805/0.3683 | 0 | true/0.332s | 4.88% | 15.001 Hz/0 | 0/0/0 | true |
| 07 | 16.215 | 99.04% | 0.0806/0.2574 | 0 | true/0.333s | 5.33% | 15.000 Hz/0 | 0/0/0 | true |
| 08 | 16.483 | 99.10% | 0.0758/0.2771 | 0 | true/0.332s | 5.24% | 15.001 Hz/0 | 0/0/0 | true |
| 09 | 16.493 | 99.31% | 0.0744/0.3082 | 0 | true/0.532s | 4.84% | 15.001 Hz/0 | 0/0/0 | true |
| 10 | 16.219 | 99.06% | 0.0807/0.2566 | 0 | true/0.332s | 4.92% | 15.001 Hz/0 | 0/0/0 | true |
| 11 | 16.353 | 99.10% | 0.0723/0.2545 | 0 | true/0.532s | 4.07% | 15.001 Hz/0 | 0/0/0 | true |
| 12 | 16.282 | 99.09% | 0.0760/0.3500 | 0 | true/0.798s | 4.49% | 15.000 Hz/0 | 0/0/0 | true |

## Aggregate and failures

- Valid policy runs: `12/12`
- Scenario passes: `11/12`
- Cone contact/intersection: `0/12`
- Minimum actual footprint clearance: `0.025587 m`
- Successful recoveries: `11/12`
- Per-scenario safe stops: `12/12`
- Final safe stop: `true`
- Scenario 01: sustained off-track: 0.333m beyond track band exceeds 0.050m margin

Scenario 01 cleared the cone but failed during the return: this is a moderate-left/right-side bypass in the complex multi-turn region around s=19–21 m. The other moderate-left scenarios passed, so the observed failure is not a general cone-collision or curvature-class failure. The offline ideal-bicycle preview underpredicted simulator return-path tracking divergence at this site.

## Infrastructure-only attempts

- Scenario 03 attempt 1: RuntimeError: derived world did not become ready before timeout: {'running': True, 'websocket': True, 'current': 'custom_71e69ee938032295503bfed557fde18c_e2e_random_cone_v1_02', 'switching': False, 'assets_version': 1787650931}
- Scenario 11 attempt 1: RuntimeError: derived world did not become ready before timeout: {'running': True, 'websocket': True, 'current': 'custom_71e69ee938032295503bfed557fde18c_e2e_random_cone_v1_10', 'switching': False, 'assets_version': 1787650931}

Each listed attempt stopped safely and received at most the one permitted fresh replacement. No genuine policy run was retried.

## Baselines and disposition

Temporal PilotNet V9, Fixed Cone Avoidance Expert V1, and practical Temporal PilotNet C1 evidence/models were hash-audited and left unchanged.

- Config SHA-256: `97b6460eb8c4ec63f573f30073403efd671395f45fa142905841d292c763430f`
- Offline evidence SHA-256: `d1cfd992f2fb52c85a98550d1ffeae283e6b58274ff2baa1d2ffab5640e98e0d`
- Baseline audit before/after: `PASS/PASS`
- Tracked simulator source changes after run: `0`
- Neural training performed: `false`
- Training bags collected: `false`

Random Cone Expert V1 frozen: **false**.
PASS-qualified 8/2/2 split release frozen: **false**.
Random-cone bag collection justified: **false**.

## Verification and repository state

- Full regression: `340 passed, 39 subtests passed` with two existing ONNX
  deprecation warnings.
- Focused Random Cone Expert tests: `12 passed`.
- Python compile gate: PASS.
- `git diff --check`: PASS.
- Repository branch: `feature/random-cone-expert-v1`.
- Simulator tracked-source changes: none. The external simulator worktree still
  reports only its runtime `userdata/last_world` state file as modified.
- No commit or push was performed.

Changed implementation/evidence paths are `.gitignore`, `pyproject.toml`,
`configs/random_cone_expert_v1.json`, `docs/random_cone_expert_v1.md`,
`scripts/run_random_cone_expert_v1.py`,
`src/physicar_e2e/random_cone_expert.py`,
`tests/test_random_cone_expert.py`, and `results/random_cone_expert_v1/`.
The twelve reconstructed simulator worlds/routes/models are ignored derived
assets outside this repository; no tracked simulator source was edited.

## Limitations

The bicycle preview is an offline feasibility model; the reported live rows are simulator measurements. Curvature classes use a 0.5 m route window on the original piecewise-polyline route. The 5 cm value is a planning margin, not a practical success threshold.
