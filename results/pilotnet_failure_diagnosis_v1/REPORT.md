# PilotNet Failure Diagnosis V1 result

The single permitted V1 run at 0.50 m/s failed after 6.082 s and 2.736 m (8.97% route completion), reproducing the established early-route failure. The objective divergence rule found onset at 4.000 s. PilotNet V1 alone controlled steering; the canonical Pure Pursuit expert ran only in shadow mode. Safe stop succeeded, and there were no API or liveness failures.

The strongest evidence supports on-policy distribution shift. Nearest nominal-training feature distance had median `1.51e-5` in the stable window, `6.90e-5` in the two seconds before onset, and `0.1968` late in the failure. Its correlation with CTE was `0.742` and with absolute network–expert steering error was `0.927`.

Under-correction is weakly supported as an interacting live symptom, not as a global offline calibration defect. In the critical two seconds before onset, network and shadow expert steering remained highly correlated (`0.987`), but PilotNet produced only `80.9%` of expert corrective magnitude: means were `0.0482` versus `0.0624 rad`, with signed difference `-0.0142 rad`. On nominal `episode_003`, however, V1 slope was `1.0046` and the high-steering magnitude ratio was `1.0092`; V2 was similarly calibrated.

Temporal mismatch is not supported. Offline V1 error was best at `0 ms` with no improvement. Live network↔expert alignment was nominally best at `+150 ms`, but MAE improved only `3.99%` and correlation remained low, so this is not strong phase evidence.

The JPEG/raw hypothesis remains inconclusive. All 92 HTTP frames had near raw matches, but median association error was `37.6 ms`, so exact frame identity was not demonstrated. Prediction-difference median was `0.00282 rad` overall and `0.000259 rad` while stable, but rose to `0.0199 rad` late. The growth could reflect transport or ordinary scene evolution over the association gap. Signed mean difference was only `0.000521 rad`, arguing against a simple transport bias.

The preprocessing audit passed every fixed contract: ROI, bilinear resize, RGB interpretation, direct YUV matrix, normalization, CHW ordering, and steering scale. H5 is not supported.

Classification: H1 **NOT SUPPORTED**; H2 **WEAKLY SUPPORTED**; H3 **INCONCLUSIVE**; H4 **SUPPORTED**; H5 **NOT SUPPORTED**.

The one recommended next intervention is a controlled on-policy expert-labeling/DAgger experiment using actual V1 deviation states near the measured divergence. Blind nominal laps and arbitrary recovery anchors are not justified: nominal offline calibration is already strong, while the prior fixed-anchor recovery intervention failed and did not cover the policy's observed state trajectory.

External diagnostic storage totaled 57,351,065 bytes: 49,355,026 bytes for the raw diagnostic directory, 4,169,357 bytes for 92 HTTP JPEGs, 96,964 bytes for telemetry, and 3,729,718 bytes for the nominal feature reference. None is training data or tracked Git evidence.

This is same-map simulator evidence only. It does not establish real-robot behavior.
