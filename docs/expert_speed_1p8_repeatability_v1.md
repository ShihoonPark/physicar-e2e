# Expert 1.80 m/s Repeatability V1

This validation freezes the first-passing high-speed Expert at 1.80 m/s, 0.90 m Pure Pursuit lookahead, 15 Hz, ±0.349066 rad steering, and 0.18 m wheelbase. The preserved characterization pass is counted exactly once, and exactly two additional valid passes are required for 3/3.

Any valid Expert failure stops immediately. Infrastructure failures are retained but excluded from the policy aggregate, with at most four new live attempts. No bags, images, labels, training, DAgger, or controller tuning are part of this task.
