# Recovery Data V1 and PilotNet V2

## Experimental question

This experiment tests whether a small fixed set of targeted recovery
demonstrations fixes PilotNet V1's same-map 0.50 m/s failure without changing
its 252,219-parameter architecture, `200×66` ROI, direct RGB-to-YUV matrix,
normalization, target, optimizer, or early-stopping rule. V1 evidence and
artifacts remain unchanged.

## Frozen anchors and perturbations

The 388-point route is parameterized by centerline arc length. Tangent yaw is
computed from symmetric points around each arc position; curvature is a finite
difference of wrapped tangent yaw. The mandatory anchor is `s=2.95 m`. Two
local absolute-curvature peaks are selected in descending magnitude with at
least 5 m circular arc separation from prior anchors. Selection happens before
collection or V2 performance is observed.

For route yaw `ψ`, left normal is `[-sin(ψ), cos(ψ)]`. Positive lateral offset
is left and positive yaw offset is counter-clockwise. Each anchor receives
exactly `(+0.10 m, 0°)`, `(-0.10 m, 0°)`, `(0 m, +6°)`, and `(0 m, -6°)`.
All candidate positions must lie inside the track band and retain at least
0.15 m boundary clearance; magnitudes are never silently reduced.

The selected roles are `failure`, `curvature_near`, and `curvature_far`. The
far curvature anchor was frozen as the four-episode recovery holdout before V2
training. Failure and near-curvature recovery episodes form the eight-episode
recovery training subset.

## Collection and extraction

Each episode independently performs safe stop, confirmed pose teleport, one
second settle, pose/world/cone verification, rosbag startup, unchanged
canonical Pure Pursuit at 0.50 m/s, convergence monitoring, safe stop,
graceful recorder shutdown, and MCAP topic verification. No reset occurs after
recording begins, and collection stops on the first failure without retry.

Recovery requires absolute CTE ≤0.03 m and absolute heading error ≤0.05 rad
continuously for 0.75 s plus at least 1.0 m progress. Duration is limited to
10 s and progress to 4.0 m. These are fixed experiment criteria.

Extraction directly reuses Dataset Extractor V1: raw `rgb8`, ROI `x=0:480,
y=160:360`, bilinear `200×66` RGB PNG output, causal MCAP record-time ZOH, and
unchanged 150 ms steering/speed age gates. Generated contact sheets are a hard
manual visual gate.

## V2 and outcome

V2 is initialized from scratch. V1 checkpoint weights are loaded only into a
separate comparison model. Best-checkpoint selection uses the unchanged
nominal episode 003; the recovery holdout does not select the checkpoint.

The experiment produced 12/12 valid recoveries and 571 extracted samples, but
the hypothesis was not supported. V2 slightly improved nominal offline error
while worsening held-out recovery error. Its single authorized 0.50 m/s lap
failed in the same early route region at 8.49% progress, earlier than V1's
9.68%. Conditional laps two and three were therefore not run. This result does
not show that recovery data is generally ineffective; it shows that this fixed
three-anchor, twelve-episode distribution did not solve the failure.

The next dataset should be designed from the observed state mismatch rather
than by blindly repeating fifty nominal laps. Useful follow-up evidence would
include denser trajectory states through the initial 2–3 m region, controlled
closed-loop deviation/recovery sequences matching the actual failure, and
separate transport/illumination/start variation strata. No such collection is
part of this experiment.
