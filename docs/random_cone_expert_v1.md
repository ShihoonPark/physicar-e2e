# Random Cone Expert V1

Random Cone Expert V1 extends the validated fixed-cone Expert to twelve
deterministic, previously unused cone positions on the same
`71e69ee938032295503bfed557fde18c` map family. It is a privileged
simulation benchmark. It does not train a neural network, collect training
bags, modify V9/C1 models, or claim real-robot success.

## Preserved baselines

Every gate hashes and audits the preserved artifacts before use:

- Temporal PilotNet V9 remains the `1.80 m/s`, causal three-frame, cone-free
  3/3 PASS baseline.
- Fixed Cone Avoidance Expert V1 remains `1.80 m/s`, `0.90 m` lookahead,
  `15 Hz`, `±0.349066 rad`, `0.18 m` wheelbase, fixed cone at `s=6.9 m`, and
  3/3 PASS.
- Temporal PilotNet C1 remains the fixed-one-cone, practical collision-only
  3/3 PASS model with cone contact/intersection 0/3.

The fixed-cone worlds, configurations, results, V9/C1 checkpoints, and V9/C1
ONNX files are inputs to hash gates only. Random Cone V1 never rewrites or
loads the neural artifacts.

## Seeded frozen scenarios

The seed is `20260825`. Candidate locations lie on a `0.05 m` canonical-route
grid and are ranked with SHA-256 over the seed, algorithm identity, and grid
index. This avoids runtime or Python-version-dependent pseudorandom state.

Before ranking, the sampler excludes the start and route-closure regions,
reserves the complete approach/bypass/return span, excludes the historical
fixed cone, rejects nonlocal route ambiguity and explicit world-object
conflicts, and requires a feasible footprint-aware bypass. It freezes a
5/4/3 mix of low-curvature, moderate-left, and moderate-right sites. A second
seeded SHA-256 rank assigns the twelve accepted locations to IDs, after which
the roles are immutable:

| ID | Role | s (m) | x (m) | y (m) | Curvature (1/m) | Class | Side |
|---:|---|---:|---:|---:|---:|---|---|
| 01 | TRAIN | 19.20 | 8.3420 | 4.3697 | +0.2018 | moderate left | right |
| 02 | TRAIN | 18.55 | 8.4379 | 4.9836 | +0.7039 | moderate left | left |
| 03 | TRAIN | 15.10 | 10.4952 | 4.6259 | 0.0000 | low | left |
| 04 | TRAIN | 21.15 | 7.1371 | 4.0633 | -0.2456 | moderate right | left |
| 05 | TRAIN | 9.25 | 8.5157 | 1.2195 | 0.0000 | low | right |
| 06 | TRAIN | 13.75 | 10.5512 | 3.2840 | -0.3867 | moderate right | right |
| 07 | TRAIN | 7.40 | 6.6657 | 1.2276 | 0.0000 | low | left |
| 08 | TRAIN | 12.30 | 10.9000 | 1.8876 | +0.2561 | moderate left | right |
| 09 | VALIDATION | 13.00 | 10.7770 | 2.5688 | +0.1112 | moderate left | right |
| 10 | VALIDATION | 6.25 | 5.5157 | 1.2327 | 0.0000 | low | right |
| 11 | UNSEEN_HOLDOUT | 11.00 | 10.2657 | 1.2118 | 0.0000 | low | right |
| 12 | UNSEEN_HOLDOUT | 14.15 | 10.4954 | 3.6759 | -0.3500 | moderate right | right |

All full-precision coordinates, seed ranks, track clearances, and nearest
world-collision clearances are stored in
`configs/random_cone_expert_v1.json`. Loading the production config reruns the
sampler and rejects any mismatch. Once frozen, no failing location may be
moved or replaced.

## Automatic bypass and geometry gate

The same `automatic_symmetric_quintic_route_normal_v1` algorithm and constants
serve all twelve positions. It uses the real `0.18 × 0.18 m` cone collision and
the conservative `0.270000 × 0.218541 m` vehicle footprint. Each side starts
from the fixed-cone planning offset and, only when curved footprint geometry
requires it, increases offset in deterministic `0.005 m` steps until the
reference recovers the `0.055 m` planning clearance or track feasibility is
exhausted. This is one geometry rule, not per-scenario tuning.

Departure and return use the existing symmetric quintic profile over `1.80 m`
with a `0.90 m` half-plateau. Side selection compares the feasible left and
right plans deterministically using predicted minimum clearance, steering
saturation, and logical-track containment. Both left and right sides occur in
the frozen set.

The offline gate requires one cone in every reconstructed world, finite smooth
geometry, positive footprint clearance, no predicted intersection, track and
world-object feasibility, bounded added curvature, steering feasibility, a
smooth exact reference return, a stable nominal-route recovery, and completion
of the remaining-lap ideal-bicycle preview. The preview is a feasibility gate,
not live evidence.

The `0.05 m` term remains a conservative planning margin. It is not a live
success threshold. Live success requires strictly positive clearance and no
cone contact/intersection. Curved-route recovery is pre-registered as nominal
CTE at most `0.08 m` for at least `0.30 s`; this is a measurement contract and
does not alter control.

## Reproduction

The twelve ignored derived worlds are reconstructed from config without
editing tracked simulator source:

```bash
uv run python scripts/run_random_cone_expert_v1.py \
  --config configs/random_cone_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --generate-worlds

uv run python scripts/run_random_cone_expert_v1.py \
  --config configs/random_cone_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --verify-worlds
```

Generate offline evidence and the 12-panel overview before driving:

```bash
uv run python scripts/run_random_cone_expert_v1.py \
  --config configs/random_cone_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/random_cone_expert_v1 \
  --offline-geometry
```

The official live command performs safe stop, exact world activation, reset,
settle, full preflight, one valid Expert run, and safe stop for each frozen
scenario. A genuine driving failure is never retried. One fresh replacement is
permitted only for an infrastructure-only failure.

```bash
uv run python scripts/run_random_cone_expert_v1.py \
  --config configs/random_cone_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/random_cone_expert_v1 \
  --run
```

The command refuses to overwrite or repeat an experiment once its start marker
exists. PASS requires all 12 valid scenarios to pass, final safe stop, world
restoration, unchanged preserved baselines/models, and no tracked simulator
source changes. Only that PASS can justify a separate future bag-collection
task; this milestone itself collects no bags.

## Official one-shot result

The preregistered run completed with **11/12 PASS**, so the final gate is
**FAIL**. All twelve valid policy runs retained positive practical cone
clearance, and cone contact/footprint intersection was `0/12`. Scenario 01 was
the sole genuine failure: on the moderate-left section at `s=19.20 m`, the
right-side bypass cleared the cone by `0.0256 m` but developed a sustained
off-track excursion during its return. It stopped safely at `67.8%` route
completion and was not retried.

Scenario 03 and scenario 11 each had one infrastructure-only attempt in which
the simulator remained on the preceding world after the activation timeout.
Both attempts stopped safely; each then received the one permitted fresh
replacement and produced its single valid policy PASS. These were not policy
retries.

The failure is specific to the moderate-left/right-side bypass in the complex
multi-turn region around `s=19–21 m`, rather than cone collision: the other
three moderate-left scenarios passed. The ideal-bicycle offline preview
therefore underpredicted the simulator's return-path tracking divergence for
this site.

The preregistered locations and roles remain immutable evidence, but they are
not promoted as a PASS-qualified Random Cone Expert V1 release. Random Cone
Expert V1 is **not frozen**, random-cone bag collection is **not justified**,
and no neural training or bag collection was performed. Full live metrics are
in `results/random_cone_expert_v1/summary.json` and the compact report is in
`results/random_cone_expert_v1/REPORT.md`.
