# Cone Avoidance Temporal PilotNet C1 V1 — Final Report

## Decision

**FAIL.** C1 passed dataset, training, offline comparison, and ONNX gates, but
its first valid live one-cone run was `CONE_POLICY_FAIL`. The vehicle did not
intersect the cone, yet minimum conservative footprint clearance was
`0.043582667 m`, below the frozen `0.050000000 m` contract. Evaluation stopped
after exactly one valid policy run. C1 is not frozen and multi-location cone
work is not justified by this milestone.

## Preserved identities

- V9 training result SHA-256:
  `4c22c0b7f2d408b44b4698ff98d394ff3bde3f8d40c5dc34e6edd0a30d906f87`
- V9 live result SHA-256:
  `56829cfc312f5cfe353458c60afcd146a54f406374d2e02546e20406cccaa6d2`
- V9 checkpoint SHA-256:
  `1cded5fcc7f3d13242de096c4868fc576d03fa6bc86df6e5c8c7c235d9faa6cc`
- V9 ONNX SHA-256:
  `7f6aa4c2d8c9b3615c580f065660c674efff94ff4cd0b9bdc9357df904000888`
- V9 train temporal manifest SHA-256:
  `07d2f6cb6dd668352ae988dcfb771fa42739faed745681c0ed9b682b669834b4`
- Cone Expert config SHA-256:
  `77deb963369b34aba917c3db6f559a63f85e80ae6e78dd2cf20a34dfe54e9831`
- Cone Expert 3/3 result SHA-256:
  `b1b0603ba34e50644802b6b2c46fdc92c9290f31273eed2f2ac0387830ce7082`
- Cone environment config SHA-256:
  `0778f735fd431f7befcf0ed17f59379e48883635914dea6c91b8087cc285830a`

V9, the canonical cone-free world, Cone Avoidance Expert V1, cone pose, bypass,
speed, lookahead, controller rate, architecture, and simulator tracked source
were not modified.

## Disk and collection

- Pre-collection available: `15,235,858,432 bytes` (`14.1895 GiB`), PASS against
  the 8 GiB gate.
- Post-collection available: `13,678,927,872 bytes` (`12.7395 GiB`), PASS against
  the 5 GiB training gate.
- Final available: approximately `13,601,603,584 bytes` (`12.67 GiB`).
- Exactly 12 independent MCAP bags were finalized.
- Total bag-directory storage: `1,555,045,047 bytes`.
- Expert recovery: 12/12; safe stop: 12/12; cone intersection: 0/12.
- Clearance mean/range: `0.056200246 m`,
  `0.056076810–0.056404518 m`.
- Mean Expert lap time: `16.179486 s`.
- Total camera/steering messages: `2,973 / 2,933`.
- Mean recorded camera/steering rates: `15.1084 / 14.9052 Hz`.

Full per-bag sizes, hashes, rates, lap/CTE/steering/recovery metrics are in
`results/cone_avoidance_collection_v1/cone_episode_*.json` and its summary.

## Dataset

- Accepted images: `2,931`; rejected: `42` (17 before and 25 after the drive
  window).
- Future steering labels: `0`.
- Maximum steering label age: `70.896810 ms`; maximum speed label age:
  `123.072406 ms`, both within the unchanged 150 ms gates.
- Temporal sequences: train `1,938`, validation `485`, holdout `484`.
- Maximum adjacent temporal gap: `0.070 s`; temporal gap rejects: `0`.
- The first two frames of every episode were excluded; duplicate padding and
  cross-episode sequences were prohibited.
- Frozen split: train 001–008, validation 009–010, holdout 011–012.
- Contact sheets from all three splits passed visual inspection.
- Evaluation-only route-s was recovered by a rigid transform from odom frame to
  the frozen per-episode preflight world pose. It was never a neural input.

## Training and offline evaluation

- V9 train sequences reused: `3,510`.
- Cone train sequences: `1,938`.
- Total unweighted/unresampled C1 train sequences: `5,448`.
- Architecture: exact V9 temporal PilotNet, input `N×9×66×200`, output `N×1`,
  `255,819` parameters.
- Initialization: from scratch; seed `20260824`; MSE, Adam, 1e-3, batch 64,
  35 epochs, unchanged early-stopping semantics.
- Best epoch: `29`; best validation normalized MSE: `0.001906518`.

Matched MAE in radians (`V9 → C1`):

- Nominal validation, n=482: `0.0106342 → 0.00955692`.
- Nominal holdout, n=481: `0.0107254 → 0.00957572`.
- Cone validation, n=485: `0.0288191 → 0.0100104`.
- Cone holdout, n=484: `0.0294492 → 0.00975891`.

Cone holdout relative MAE improvement was `66.8619%`, above the pre-registered
10% gate. Nominal C1/V9 MAE ratios were `0.8987` validation and `0.8928`
holdout, below the catastrophic-regression ratio of 1.50.

Cone holdout regional C1 MAE/corrective ratio:

- Approach, n=16: `0.00480050 / 0.90953`.
- Departure/avoidance, n=52: `0.00259700 / 0.99352`.
- Pass/return, n=40: `0.00145598 / 1.02320`.
- Recovered, n=16: `0.00221939 / 1.11230`.

Checkpoint SHA-256:
`1e90002ca139b3cfb0f34074e013e52b6754df33ed0e3b438ca81809c9e2ee39`.

ONNX SHA-256:
`22440ad61f6e5136b33016eb0781d79ab71637e659478ac0c92cc04cffc98e5f`.

ONNX checker and dynamic batch I/O contract passed. PyTorch↔ONNX maximum
difference was `2.38419e-7` normalized (`8.32238e-8 rad`).

## First and only live C1 run

- Classification: `CONE_POLICY_FAIL`.
- Runtime before safety stop: `3.992651 s`.
- Progress/completion: `6.590910 m / 0.216063`.
- Mean/max nominal CTE: `0.072263 / 0.242740 m`.
- Minimum clearance: `0.043582667 m` at route `s=6.700090 m`.
- Vehicle/cone intersection: false.
- Maximum actual right avoidance offset: `0.242740 m`.
- Frozen minimum valid center offset: `0.249270 m`; shortfall about `6.53 mm`.
- Frozen reference maximum offset: `0.254270 m`; live shortfall about `11.53 mm`.
- Recovery: not reached because the clearance gate stopped the run.
- Mean/max absolute steering: `0.063088 / 0.349066 rad`.
- Saturation fraction: `0.034483`; mean command delta: `0.015438 rad`.
- Camera/preprocess/ONNX mean latency: `2.494 / 1.973 / 1.881 ms`.
- Adjacent temporal gap maxima: `0.071013 / 0.070480 s`.
- Loop frequency: `15.1856 Hz`; loop max: `66.751 ms`; slips: `0`.
- Temporal/API/pose/clock failures: `0/0/0/0`.
- Off-track events/duration: `0 / 0 s`.
- Safe stop: PASS.

Runs #2 and #3 were not executed, so there is no three-run C1 aggregate.

## Interpretation and next decision

The offline imitation result shows that cone-specific steering was learned on
matched Expert samples without nominal-lane regression. The live failure shows
that this was insufficient under closed-loop camera-domain and trajectory
feedback: the policy produced too little lateral displacement for the narrow
5 cm physical-clearance contract. The result is a genuine policy failure, not
an infrastructure or temporal-buffer failure.

Per the frozen protocol, no retry, retraining, extra bag collection, DAgger, or
threshold adjustment was performed. C1 is not canonical/frozen. Multi-location
cone work is not justified yet. A future separately authorized task should
diagnose visual-domain shift, anticipation timing, corrective displacement, and
clearance margin using this preserved negative result before choosing any new
data or training intervention.

## Storage, source, and Git status

- Raw bags, extracted images/manifests, C1 checkpoint, ONNX, and plots are
  outside Git under `userdata/physicar_e2e/cone_avoidance_v1/`.
- Compact configs, source, tests, docs, and metrics are in the E2E worktree.
- Simulator tracked diff remains only runtime `userdata/last_world`; no tracked
  simulator source changed.
- No commit or push was performed.
- Final branch: `feature/cone-avoidance-pilotnet-v1`.
