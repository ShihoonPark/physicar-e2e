# High-Speed PilotNet DAgger Iteration 2 V1

This is the second and final automatic high-speed DAgger iteration. PilotNet V6 controls exactly two 1.80 m/s shadow rollouts while frozen High-Speed Expert V1 (0.90 m lookahead, 15 Hz) supplies causal labels without command authority. Rollout A freezes a continuous 2-second-pre-divergence through final-frame interval at ≥60% completion; rollout B is evaluated only over that frozen route-s interval.

V7 trains from scratch on nominal high-speed training data, existing DAgger1 rollout A, and new DAgger2 rollout A. Nominal validation/holdout and both DAgger holdouts remain separate. The PilotNet architecture and preprocessing are unchanged. A first valid V7 policy failure stops this experiment permanently; no Iteration 3 is automatic.
