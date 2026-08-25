# Cone Avoidance Expert V1 — Final Report

Result: **PASS — 3/3 `OBSTACLE_EXPERT_PASS`**.

High-Speed Cone Avoidance Expert V1 is frozen. The one-cone scenario and all
Expert parameters were identical across three independent resets. There were
three total live attempts, three valid evaluations, and no infrastructure
replacement. No neural training or ROS bag collection occurred.

## Preserved baselines and asset integrity

- PilotNet V4 remains the canonical `0.50 m/s`, single-frame, 3/3 policy.
- Temporal PilotNet V9 remains the canonical `1.80 m/s`, causal three-frame,
  255,819-parameter, 3/3 same-map/same-spawn policy. It was not loaded, changed,
  or retrained.
- High-Speed Expert V1 remains `1.80 m/s`, `0.90 m` lookahead, `15 Hz`,
  `±0.349066 rad`, `0.18 m` wheelbase, and 3/3 lane-following PASS.
- The preserved Expert config SHA-256 remained
  `3afdc5d3204143e8d1f64c1ec68b2bda2de08912b779f2ece69a7d86c99503c9`.
- The preserved cone-free world SHA-256 remained
  `8a4e2b21f1cb2e4ee97684d152442610cb05f03e2ef58f6b805c5bc527d309e1`.
- The route SHA-256 remained
  `07309dab65d64719ba833bcb43a3dc28276928ae16cc6dfaa9c6882f497a4213`.

The new ignored custom world is
`custom_71e69ee938032295503bfed557fde18c_e2e_cone_avoidance_v1`. Verification
proved a byte-identical route, identical copied track-model metadata, and
exactly one intended cone. The canonical cone-free world was not edited.

## Real geometry and frozen scenario

The cone is the original world's real `cone2` model and visual mesh
`model://meshes/custom_71e69ee938032295503bfed557fde18c/cone2.dae`. Its one
collision is an explicit `0.18 × 0.18 × 0.38 m` box; the 2D half extents are
`0.09 × 0.09 m`.

The vehicle footprint comes from the real PhysiCar SDF body, lidar, and four
wheel collisions. The conservative rectangle encloses the front-wheel
collisions over the complete `±0.349066 rad` Expert steering clamp:

```text
x extent: [-0.135000, +0.135000] m
y extent: [-0.109270, +0.109270] m
length × width: 0.270000 × 0.218541 m
```

The deterministic site is route `s=6.900000 m`, world
`x/y=6.165700204/1.229802786 m`, yaw `-0.004394029 rad`, with effectively zero
local curvature. It is the midpoint sample of the longest eligible low-curvature
run (`s=3.150–10.600 m`), after spawn/final-region exclusions. Its nonlocal
route clearance is `2.007280 m`. Track-normal clearance is `0.3500003 m` left
and `0.3500046 m` right, so the greater-clearance right side was frozen before
live testing. All five explicit non-track world box collisions pass the
conflict check; the nearest is `wall_0` at `1.139408 m` clearance.

The center-offset requirement is vehicle half-width `0.109270 m` + cone
half-width `0.090000 m` + experimental clearance `0.050000 m` = `0.249270 m`.
The reference adds a deterministic `0.005000 m` planning margin, yielding
`0.254270 m` maximum right offset.

The symmetric quintic-with-flat-pass geometry is:

| Location | Nominal route s |
|---|---:|
| Departure start | 4.200 m |
| Flat-pass start | 6.000 m |
| Cone | 6.900 m |
| Flat-pass end | 7.800 m |
| Return end | 9.600 m |

Total span is `5.400 m`. Each transition is `1.800 m`; the centered flat pass is
`1.800 m`. Maximum local reference curvature is `0.444256 m⁻¹`, equivalent to
`0.079796 rad` steering, below the `2.022058 m⁻¹` / `0.349066 rad` feasibility
limit.

Offline minimum conservative footprint-to-cone clearance is `0.055000 m` and
minimum logical reference-center track clearance is `0.095734 m`. Geometry is
finite, C2 at every profile junction, returns exactly to nominal position, and
has no intersection. The ideal-bicycle sanity preview predicts `0.063339 m`
clearance and recovery, but it is not counted as live evidence.

Geometry evidence:
[`geometry_plot.png`](../cone_avoidance_environment_v1/geometry_plot.png) and
[`summary.json`](../cone_avoidance_environment_v1/summary.json).

## Live runs

The original nominal route supplied CTE, progress, lap completion, and track
safety. Only Pure Pursuit targets used the local bypass. Recovery required
absolute nominal CTE `≤0.05 m` for `≥0.50 s` after the return endpoint.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Classification | PASS | PASS | PASS |
| Lap time (s) | 16.1448 | 16.2184 | 16.1493 |
| Completion | 99.371% | 99.143% | 99.082% |
| Progress (m) | 30.3127 | 30.2432 | 30.2244 |
| Nominal mean / max CTE (m) | .08123 / .25717 | .08588 / .28798 | .08270 / .25726 |
| Off-track events / duration (s) | 0 / 0 | 0 / 0 | 0 / 0 |
| Mean / max abs steering (rad) | .11474 / .34907 | .11681 / .34907 | .11414 / .34907 |
| Steering saturation | 4.527% | 5.328% | 4.938% |
| Mean command delta (rad) | .02593 | .02586 | .02539 |
| Minimum cone clearance (m) | .056250 | .056338 | .056401 |
| Route s at minimum (m) | 6.92885 | 6.84459 | 6.92108 |
| Cone intersection | no | no | no |
| Maximum right offset reached (m) | .25717 | .25721 | .25726 |
| Recovery CTE (m) | .01714 | .01499 | .01711 |
| Recovery success / time (s) | yes / .5324 | yes / .5304 | yes / .5288 |
| Loop frequency (Hz) | 15.0009 | 15.0004 | 15.0008 |
| Loop p95 / max (ms) | 66.700 / 69.443 | 66.690 / 79.102 | 66.692 / 68.130 |
| Timing slips | 0 | 0 | 0 |
| API / pose / clock failures | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Safe stop | PASS | PASS | PASS |

Per-run JSON is preserved in `attempt_01.json`, `attempt_02.json`, and
`attempt_03.json`.

## Aggregate and decision

- Success: 3/3.
- Lap time: mean `16.170858 s`, sample std `0.041266 s`.
- Mean nominal CTE across run means: `0.083270 m`.
- Worst maximum nominal CTE: `0.287985 m`.
- Minimum cone clearance across runs: `0.056250 m`.
- Cone clearance mean/range: `0.056330 m` / `0.056250–0.056401 m`.
- Steering saturation mean/range: `4.931%` / `4.527–5.328%`.
- Recovery: 3/3.
- Safe stop: 3/3.

High-Speed Cone Avoidance Expert V1 is frozen. The next justified task is a
separate Obstacle Expert ROS bag collection milestone:

```text
camera + avoidance steering
→ causal temporal dataset
→ camera-only Temporal PilotNet cone-avoidance training
```

No bag was collected and no V10/V11 or other neural model was trained here.
The future policy must not receive cone GT coordinates, route, pose, CTE, or
Expert command.

## Tests, repository state, and limitations

Focused tests passed 18/18. The final full regression passed 308 tests plus 39
subtests; two existing PyTorch ONNX deprecation warnings were emitted.
`git diff --check` and final evidence/environment verification passed.

Files added are the two configs, two implementation modules, two wrappers, two
focused test modules, documentation, offline compact evidence/plot, and live
compact evidence/report. `sim_client.py` gained only a validated world-switch
method. No preserved baseline file changed. No commit or push was performed.

The external simulator checkout has no new tracked source changes; only the
expected tracked runtime `userdata/last_world` state is modified. Derived
custom assets are in the simulator repository's established ignored paths.

Limitations: this is simulation-only, one map, one spawn, one frozen cone, and
one avoidance side at fixed speed. The footprint is a conservative rectangular
envelope, and track clearance is the existing logical reference-center safety
definition. The `0.05 m` margin is experimental. There is no real-robot,
map-generalization, spawn-generalization, multi-obstacle, or camera-only neural
avoidance claim.
