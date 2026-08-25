# Cone Avoidance Expert V1

Cone Avoidance Expert V1 is the first obstacle-avoidance milestone. It adds one
reproducible real simulator cone and a deterministic privileged local bypass to
the preserved High-Speed Pure Pursuit Expert. It does not train a neural model
or collect a ROS bag.

## Preserved contracts

The implementation verifies compact evidence hashes before evaluation:

- PilotNet V4 remains the canonical `0.50 m/s`, single-frame, 3/3 policy.
- Temporal PilotNet V9 remains the canonical `1.80 m/s`, causal three-frame,
  255,819-parameter, 3/3 same-map/same-spawn policy. Its wider final-corner
  trajectory remains an accepted limitation; V9 is neither loaded nor changed.
- High-Speed Expert V1 remains `1.80 m/s`, `0.90 m` lookahead, `15 Hz`,
  `±0.349066 rad`, and `0.18 m` wheelbase. Only its control reference changes
  locally in this experiment.
- The preserved cone-free world
  `custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1` is hash-gated and
  never edited.

The derived ignored simulator world is:

```text
custom_71e69ee938032295503bfed557fde18c_e2e_cone_avoidance_v1
```

Its route is byte-identical to the preserved route and it contains exactly one
intended top-level model, `cone_obstacle_v1`.

## Real collision geometry

The cone is copied from model `cone2` in the original simulator world
`custom_71e69ee938032295503bfed557fde18c`. Its collision is the simulator's
explicit `0.18 × 0.18 × 0.38 m` box and its visual remains the real
`cone2.dae` mesh. The 2D collision half extents are `0.09 × 0.09 m`.

The vehicle footprint is parsed from
`src/physicar-sim/share/models/physicar/model.sdf`. It encloses the base box,
lidar cylinder, rear wheel cylinders, and front wheel cylinders over the full
Expert steering clamp. The resulting conservative base-frame rectangle is:

```text
x: [-0.135000, +0.135000] m
y: [-0.109270, +0.109270] m
length × width: 0.270000 × 0.218541 m
```

This bounding rectangle is conservative relative to the parsed collision union.
The `0.05 m` cone margin is an experimental simulator margin, not calibrated
physical truth.

## Frozen cone and bypass

The deterministic selector excludes the spawn and final region, finds the
longest route run whose local curvature gate passes, and chooses its midpoint.
The frozen cone is on the nominal route at:

```text
s:       6.900000 m
x/y:     6.165700 / 1.229803 m
yaw:    -0.004394 rad
curvature: approximately 0.0 m^-1
straight run: s = 3.150–10.600 m
```

All five explicit non-track world box collisions were checked in 2D. The
nearest is `wall_0`, with `1.139408 m` cone-collision clearance; there is no
light, wall, or other explicit model conflict.

Normal-ray clearance is `0.3500003 m` left and `0.3500046 m` right. Right is
therefore frozen as the avoidance side before live testing.

The minimum geometry-only center offset is:

```text
0.109270 m vehicle half-width
+ 0.090000 m cone half-width
+ 0.050000 m required margin
= 0.249270 m
```

V1 adds a frozen `0.005 m` planning margin, giving a maximum reference offset of
`0.254270 m`. A quintic smoothstep departs over `1.80 m`, holds the lateral
offset through a `1.80 m` flat pass centered on the cone, and returns over
`1.80 m`:

```text
departure start: s = 4.200 m
flat-pass start: s = 6.000 m
cone:            s = 6.900 m
flat-pass end:   s = 7.800 m
return end:      s = 9.600 m
total span:          5.400 m
```

The transition length is the maximum of the conservative curvature minimum,
one second of travel at `1.80 m/s`, and two `0.90 m` lookaheads. The offline
reference peaks at `0.444256 m^-1`, equivalent to `0.079796 rad` steering,
below the physical `2.022058 m^-1` / `0.349066 rad` limit. Planned minimum
footprint-to-cone clearance is `0.055000 m`; minimum logical reference-center
track clearance is `0.095734 m`. The ideal-bicycle sanity preview predicts
`0.063339 m` cone clearance and successful recovery, but is not live evidence.

The actual geometry plot is
[`results/cone_avoidance_environment_v1/geometry_plot.png`](../results/cone_avoidance_environment_v1/geometry_plot.png).

## Live simulator result

The bounded gate executed three total attempts and three valid independent-reset
evaluations. All three were `OBSTACLE_EXPERT_PASS`; no infrastructure
replacement attempt was needed. Minimum conservative footprint-to-cone
clearance was `0.056250`, `0.056338`, and `0.056401 m`. There were zero
footprint intersections, zero off-track events, recovery 3/3, and safe stop
3/3. Mean lap time was `16.1709 ± 0.0413 s` (sample standard deviation).

High-Speed Cone Avoidance Expert V1 is frozen. Obstacle Expert ROS bag
collection is justified as a separate next task, but this task collected no bag
and trained no model. These are simulator-only results.

## Progress, recovery, and information boundary

The original nominal route supplies projection, CTE, progress, lap completion,
and track-safety metrics. The bypass supplies only local Pure Pursuit target
points. Recovery is pre-registered as absolute nominal-route CTE no greater
than `0.05 m` continuously for at least `0.50 s` after `s = 9.600 m`.

The teacher may use GT vehicle pose, nominal route, cone GT pose, the generated
bypass, and track geometry. A future neural student remains temporal-camera
only and must not receive cone coordinates, route, pose, CTE, or Expert command.

## Reproduction and hard gates

Generate or verify only the ignored simulator assets:

```bash
python scripts/setup_cone_avoidance_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --config configs/cone_avoidance_environment_v1.json \
  --generate

python scripts/setup_cone_avoidance_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --config configs/cone_avoidance_environment_v1.json \
  --verify-only
```

Regenerate offline geometry evidence without driving:

```bash
python scripts/run_cone_avoidance_expert_v1.py \
  --config configs/cone_avoidance_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/cone_avoidance_environment_v1 \
  --offline-geometry
```

The non-driving live preflight may activate the derived world explicitly:

```bash
python scripts/run_cone_avoidance_expert_v1.py \
  --config configs/cone_avoidance_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/cone_avoidance_expert_v1 \
  --activate-world --preflight-only
```

The live mode allows at most five attempts and three valid evaluations. The
first valid driving failure stops the experiment without retry or tuning. Only
infrastructure failures have bounded replacement semantics. Runs two and three
occur only after every preceding valid run passes.

```bash
python scripts/run_cone_avoidance_expert_v1.py \
  --config configs/cone_avoidance_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/cone_avoidance_expert_v1 \
  --activate-world --run
```

After the experiment is frozen, a single visual-check lap can be run without
deleting, overwriting, or extending the official 3/3 evidence. Demo metrics are
printed to the terminal only and are explicitly non-evidentiary:

```bash
uv run python scripts/run_cone_avoidance_expert_v1.py \
  --config configs/cone_avoidance_expert_v1.json \
  --sim-root ~/physicar-ai-sim-docker \
  --result-dir results/cone_avoidance_expert_v1 \
  --activate-world --demo
```

`/steering` remains the unchanged simulator command topic and reports actual
Expert steering radians. This milestone makes simulation-only claims; it is not
real-robot evidence.
