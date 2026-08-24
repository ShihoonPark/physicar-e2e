# PilotNet E2E Smoke V1

## Scope and gates

This milestone is a pipeline and same-map simulator smoke test, not final
generalization evidence. Its hard-gated order is dataset integrity, tiny
overfit, baseline training, offline validation, ONNX export/equivalence, host
CPU benchmark, simulator preflight, Smoke A at 0.30 m/s, and—only if A
passes—Smoke B at 0.50 m/s. No hyperparameter sweep, augmentation, steering
rebalance, near-zero removal, expert tuning, new data collection, or automatic
live retry is permitted.

Training uses whole episodes 001 and 002 (1,754 samples); validation uses whole
episode 003 (878 samples). There is no invented test split and no frame-level
mixing. `steering_rad` remains the physical label. Optimization uses
`steering_rad / 0.349066`; metrics convert unconstrained predictions back to
radians. Only live commands are clamped to ±0.349066 rad.

## Model

Input is NCHW `float32`, `3×66×200`. The exact network is:

```
Conv2d(3,24,5,stride=2), ReLU
Conv2d(24,36,5,stride=2), ReLU
Conv2d(36,48,5,stride=2), ReLU
Conv2d(48,64,3,stride=1), ReLU
Conv2d(64,64,3,stride=1), ReLU
Flatten(64×1×18)
Linear(1152,100), ReLU
Linear(100,50), ReLU
Linear(50,10), ReLU
Linear(10,1)
```

The exact parameter count is 252,219. The output is linear normalized steering.

## Deterministic preprocessing

Stored PNGs remain unchanged RGB `200×66`. RGB values are converted to
`float32` in [0,1], then to a documented full-range BT.601-style analog YUV:

```
Y =  0.29900 R + 0.58700 G + 0.11400 B
U = -0.14713 R - 0.28886 G + 0.43600 B + 0.5
V =  0.61500 R - 0.51499 G - 0.10001 B + 0.5
```

Each channel is normalized as `(value - 0.5) * 2`, then transposed to CHW.
This is not JPEG YCbCr. The same matrix and normalization function is shared by
training, validation, ONNX inputs, and live inference. No statistics are learned
from validation data.

The HTTP live path decodes the expected `480×360` JPEG as RGB, crops
`x=0:480, y=160:360`, resizes bilinearly to `200×66`, then applies the shared
conversion. Training PNGs came from raw ROS RGB frames, so HTTP JPEG transport
is an honest potential domain difference.

## Neural and privileged boundaries

`CameraOnlyOnnxModel.predict` accepts exactly one `float32 (3,66,200)` camera
array. Route, GT pose, track boundaries, simulator clock, and world status are
used only after inference for safety, liveness, progress, lap completion, and
metrics. Speed is fixed and separately published; it is not learned.

The runner checks the expected cone-free derived world, route and track
geometry, spawn, camera contract, pose/clock liveness, sustained off-track
state, and safe stop. It performs no more than two neural laps and never runs B
after an A failure.

## Interpretation

One completed 0.30 m/s lap establishes only low-speed end-to-end pipeline
viability. A subsequent 0.50 m/s lap establishes a same-map nominal simulator
baseline. Neither establishes recovery, unseen-start, brightness, other-map,
Raspberry Pi 5, or real-robot performance. The x86 ONNX Runtime timing is
explicitly a host CPU benchmark; Pi performance remains unmeasured.
