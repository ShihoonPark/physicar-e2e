# Random Cone Expert 1.0 m/s V1

This is the simulator qualification for the new `1.0 m/s` operational target.
It is separate from, and does not rewrite, the `1.8 m/s` stress evidence in
`results/random_cone_expert_v1`.

## Preserved parent evidence

The target config fail-closes on both the exact parent config SHA-256 and a
manifest hash covering every file in the parent result directory. The
registered parent outcome remains:

- Random Cone Expert at `1.8 m/s`: `11/12`
- Cone contact/intersection: `0/12`
- Genuine failure: scenario 01, return off-track

The earlier V9 lane, fixed-cone Expert, and fixed-cone C1 baselines remain
under their existing hash gates as well.

## Frozen target contract

- Map, route, spawn, vehicle, seed, 12 cone coordinates, roles, provenance,
  automatic sides, and derived scenario worlds are inherited exactly from the
  parent stress config.
- Roles remain 01–08 TRAIN, 09–10 VALIDATION, and 11–12 UNSEEN_HOLDOUT.
- Control is `1.0 m/s`, `0.90 m` lookahead, `15 Hz`, `±0.349066 rad`, and
  `0.18 m` wheelbase.
- The automatic spatial bypass geometry remains unchanged: `1.80 m` quintic
  transitions and a `0.90 m` half-plateau.
- The offline continuous-steering-saturation time gate is `0.90 s`. This is
  the speed-normalized equivalent of the stress gate's `0.50 s`: both permit
  the same `0.90 m` maximum saturated travel distance. It is an offline gate,
  not a controller change or per-scenario tuning.
- No neural training or bag collection is permitted in this qualification.

After the offline evidence is generated, the official command permits one
valid policy run per scenario and only one infrastructure replacement. A
genuine policy failure is not retried. The experiment marker prevents an
official rerun or overwrite.

```bash
.venv/bin/python scripts/run_random_cone_expert_1p0_v1.py \
  --config configs/random_cone_expert_1p0_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --verify-worlds

.venv/bin/python scripts/run_random_cone_expert_1p0_v1.py \
  --config configs/random_cone_expert_1p0_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --offline-geometry

.venv/bin/python scripts/run_random_cone_expert_1p0_v1.py \
  --config configs/random_cone_expert_1p0_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --preflight-only --scenario-id 01

.venv/bin/python scripts/run_random_cone_expert_1p0_v1.py \
  --config configs/random_cone_expert_1p0_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --run
```

Only a clean `12/12` simulator PASS can justify the next, separately defined
camera-and-steering bag task. Future training bags may use TRAIN 01–08 only;
09–10 remain validation positions and 11–12 remain sealed neural holdouts.
Simulator PASS is not real-robot evidence.

## Official result

The official frozen run achieved **12/12 PASS** with cone contact/intersection
`0/12`, recovery `12/12`, per-scenario safe stop `12/12`, and final safe stop
and world restoration PASS. Scenario 01, which failed the 1.8 m/s stress run,
completed at 1.0 m/s with `0.0193 m` minimum practical clearance and no
off-track event.

Scenario 08 had one infrastructure-only world-switch timeout; its safe stop
passed and the single permitted fresh replacement produced its one valid
policy PASS. No genuine policy run was retried.

The minimum observed clearance was only `0.003716 m` in scenario 08. This is a
PASS under the preregistered practical collision-only contract (`> 0`, no
intersection), but it is a narrow simulator margin and must not be represented
as a real-robot safety margin.

Random Cone Expert 1.0 m/s V1 and the exact 8/2/2 split are frozen. A separate
TRAIN-only camera-and-steering bag task is now justified by the simulator
gate; no bags or neural model were produced by this task.
