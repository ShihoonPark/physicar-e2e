# PilotNet DAgger V1

This experiment tests one intervention: add canonical shadow-expert labels for raw-camera states actually visited by failing PilotNet V1. It does not change the 252,219-parameter architecture, preprocessing, optimizer, loss, learning rate, speed, or safety layer.

`dagger_rollout_A` is the preserved Failure Diagnosis V1 run and is training-only. `dagger_rollout_B` is exactly one new independent V1 run and is holdout-only. V1 controls both vehicles; Pure Pursuit never reaches the control command boundary.

Raw camera header simulator timestamps and telemetry simulator-clock timestamps share the same domain. Each raw frame receives the latest shadow-expert label at or before its header time. Future labels are forbidden and label age is capped at 150 ms. The objective window is continuous from 1.0 second before fixed-rule divergence onset through the final valid pre-safe-stop telemetry. If that window has fewer than 20 raw frames, the full active rollout is used and recorded as a fallback.

V3 trains from scratch on nominal episodes 001–002 plus rollout A. Nominal episode 003 and rollout B remain independent validation sets. Recovery V2 anchor data is excluded. A first V3 0.50 m/s failure prevents every later run; only a first pass enables runs two and three, for a maximum of three.

Raw bags, extracted images, checkpoints, ONNX files, and plots stay under external simulator userdata. Compact configs, metrics, and provenance remain in this repository. Simulator results do not imply real-robot performance.
