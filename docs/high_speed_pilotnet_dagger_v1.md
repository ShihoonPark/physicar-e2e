# High-Speed PilotNet DAgger V1

This bounded experiment asks whether High-Speed Expert V1 labels on states actually visited by failing camera-only PilotNet V5 improve 1.80 m/s closed-loop robustness. V5 controls exactly two frozen shadow rollouts; the 1.80 m/s, 0.90 m-lookahead Expert has no command authority. Rollout A is training-only and rollout B is holdout-only.

Raw ROS RGB frames use causal simulator-time zero-order-hold Expert labels. The objective interval begins 2.0 seconds before CTE divergence and ends at the final valid pre-safe-stop frame, with every retained sample at or beyond 30% route progress. V6 trains from scratch on nominal episodes 001–008 plus rollout A, with unchanged PilotNet architecture and training settings. Nominal validation 009–010, nominal holdout 011–012, and rollout B remain separate.

Large rollouts, images, checkpoint, ONNX, and plots remain under external simulator userdata. A first valid V6 policy failure stops live testing; a first pass permits exactly two more valid laps for a 3/3 gate. V4, V5, the nominal High-Speed dataset, and High-Speed Expert V1 remain immutable.
