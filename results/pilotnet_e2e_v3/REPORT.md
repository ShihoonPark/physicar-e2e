# PilotNet DAgger V1 integrated result

PilotNet V3 preserved the exact 252,219-parameter architecture and all V1 preprocessing/training hyperparameters. It trained from scratch on 1,754 nominal samples plus 46 causal shadow-expert samples from preserved V1 rollout A. The independent 44-sample rollout B was never used for training.

Both extractions used raw ROS RGB and the common simulator-time domain. The latest shadow-expert telemetry at or before each camera header supplied the target. There were zero future labels and no stale-label rejection. A/B source MCAP hashes differ.

V3 improved nominal episode-003 MAE from V1's `0.004819` to `0.004408 rad`. More importantly, rollout-B MAE fell from V1 `0.23207` and V2 `0.22783` to V3 `0.02732 rad`; corrective magnitude ratio improved from V1 `0.309` to V3 `0.997`. In the divergence window V1 MAE/ratio were `0.26866/0.244`, versus V3 `0.02664/0.966`.

ONNX equivalence passed with mean/max physical differences `2.72e-8/1.25e-7 rad`.

The first V3 live 0.50 m/s run failed after `40.758 s` and `19.819 m` (`64.97%`) with sustained off-track. Runs two and three were correctly not executed. This is materially farther than V1 `2.953 m` and V2 `2.591 m`, so the on-policy hypothesis receives **partial support**, not a closed-loop pass. Safe stop succeeded and there were no API or liveness failures.

One later, bounded DAgger iteration using actual V3 states near its new failure around 19.82 m is justified. It was not run here. Collecting 50 duplicate nominal laps is not justified. The remaining failure may reflect uncovered later-route on-policy states, compounding error, or the still-unresolved raw/JPEG transport contribution.

This remains same-map simulator evidence and does not establish real-robot performance.
