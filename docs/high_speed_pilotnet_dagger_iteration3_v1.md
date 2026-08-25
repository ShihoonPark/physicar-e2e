# High-Speed PilotNet DAgger Iteration 3 V1

This final bounded iteration records exactly two V7-controlled, Expert-shadow-labeled rollouts. The target is pre-registered before collection: every valid raw ROS camera frame from 85% route completion through the final driving frame. A is training-only and B is holdout-only.

V8 is trained from scratch on the unchanged nominal high-speed train split plus DAgger1-A, DAgger2-A, and DAgger3-A. Architecture, preprocessing, optimizer, loss, seed, and speed remain unchanged. A policy failure ends live evaluation; three valid passes are required to freeze V8. Iteration 4 is forbidden.
