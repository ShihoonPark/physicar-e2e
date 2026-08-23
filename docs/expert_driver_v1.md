# Automated Expert Driver V1

V1 follows the simulator-provided closed centerline with deterministic Pure
Pursuit. It projects GT vehicle XY onto the full closed polyline, tracks wrapped
arc length as monotonic progress, selects a target `lookahead_m` ahead in arc
length, and computes:

`kappa = 2 * y_target_vehicle / distance(vehicle, target)^2`

`steering = atan(wheelbase * kappa)`

The denominator uses actual Euclidean target distance, not nominal lookahead,
because corner geometry can make them differ. Steering is clamped before being
sent. Positive steering means left, matching the verified live OpenAPI.

## Safety and preflight

The client uses only Python's standard library and does not require host ROS 2.
Before motion it verifies API reachability, running/not-switching state, exact
world identity (unless `--allow-unexpected-world` is explicit), closed route,
bounds, finite pose near the route, zero cone-prefixed objects, and the live
OpenAPI request schemas for `POST /speed` and `POST /steering` with numeric JSON
`value` fields.

Speed is refreshed every control iteration. Driving always attempts independent
zero speed and zero steering commands on exit. Runtime stops on API/controller
errors, localization-liveness failure, sustained off-track state, world changes,
timeout, or Ctrl+C.

The simulator's `/pose` cache exposes no source timestamp or update sequence, so
HTTP response time is not treated as freshness. After the first successful
nonzero speed command, the driver requires either 0.005 m translation or 0.01 rad
yaw change within the configured 0.75 s pose timeout. It also requires Gazebo's
independent `/clock` simulation time to advance within that interval, allowing a
frozen simulator clock to be distinguished from a cached/frozen pose stream.
This watchdog is inactive before motion and while intentionally stopped.

Off-track safety treats `outer` as the track polygon shell and `inner` as its
hole. A pose inside that closed band is accepted. A pose outside is accepted
only when its minimum Euclidean distance to the actual inner or outer boundary
polyline is no greater than `off_track_margin_m`; the grace timer still filters
transient excursions. Explicitly and implicitly closed inputs are normalized,
and unusable, intersecting, or non-enclosed boundaries fail preflight. V1 assumes
the supplied boundaries form valid non-self-intersecting polygons. Centerline
CTE remains an unsigned diagnostic metric; vehicle-footprint containment is not
implemented.

Because V1 uses commanded-motion evidence as the fallback for missing pose
timestamps, it intentionally treats a physically stuck vehicle as a
localization-liveness failure and safe-stops it.

Lap completion requires both at least 90% unwrapped route progress and return to
the 0.30 m start gate. Projection jumps larger than the configured 1.0 m per
sample are rejected. This protects against an accidental projection jump, but
full branch-aware projection is a remaining limitation for self-intersecting or
very closely folded routes.

`--reset-before-run` explicitly sends independent zero speed and steering
commands before reset and again immediately afterward, then waits for a valid
near-route spawn and repeats preflight. The ordinary path also requires a safe
stop before initial preflight. A failed stop prevents reset, preflight, and
driving from proceeding. Every client lifecycle exit—including preflight,
reset, and dry-run failures—attempts a final best-effort safe stop. `--dry-run SECONDS` computes
finite geometry and metrics without sending nonzero drive commands; it may send
zero-valued safety commands. Runtime result JSON includes loop period statistics,
CTE, steering, saturation, off-track events, progress, and safe-stop status.

The installed `physicar-expert-v1` command requires `--config PATH`; it has no
working-directory-dependent default. The repository convenience launcher
`scripts/run_expert_driver_v1.py` supplies the single canonical checkout config
unless the caller explicitly overrides it.

The earlier one-lap metrics are retained as
`results/expert_driver_v1_pre_safety_run.json`. That run predates the pose/clock
liveness watchdog and is historical evidence only, not a canonical result for
the current implementation. No current canonical baseline exists until a new
lap is explicitly authorized after review.
