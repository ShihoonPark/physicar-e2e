# Temporal PilotNet V9 live evidence

**PASS: 3/3 valid 1.80 m/s POLICY_PASS.** Each attempt reset and preflighted independently, acquired three real HTTP camera frames while stopped, used no duplicate padding, and moved only after a valid causal buffer existed. There were zero temporal-input, API, pose/clock-liveness, and >100 ms timing failures. Safe stop passed for every run.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Completion / progress | 99.06% / 30.217 m | 98.58% / 30.070 m | 98.70% / 30.108 m |
| Elapsed | 17.133 s | 16.920 s | 16.935 s |
| Final start distance | .292 m | .253 m | .211 m |
| Mean/max CTE | .0703/.5022 m | .0650/.4556 m | .0604/.4216 m |
| Off-track events/duration | 1/.468 s | 1/.330 s | 1/.199 s |
| Mean/max steering | .1273/.3491 rad | .1276/.3491 rad | .1260/.3491 rad |
| Saturation / command delta | 3.52%/.02687 | 2.37%/.02710 | 2.37%/.02686 |
| Camera mean/p95/max | 2.324/3.650/6.089 ms | 2.587/4.126/6.340 ms | 2.569/4.014/5.756 ms |
| Preprocess mean/p95/max | 1.713/2.321/2.812 ms | 1.879/2.576/3.894 ms | 1.802/2.296/3.244 ms |
| ONNX mean/p95/max | 1.364/4.162/5.285 ms | 1.472/4.179/4.612 ms | 1.405/4.130/5.318 ms |
| Total model path mean/p95/max | 3.084/6.294/7.090 ms | 3.360/6.294/6.691 ms | 3.219/6.324/7.571 ms |
| Loop Hz / period p95/max | 15.038 / 66.691/66.750 ms | 15.044 / 66.689/68.224 ms | 15.032 / 66.689/76.260 ms |
| Adjacent gap max | 71.6 ms | 72.7 ms | 76.2 ms |

For comparison, preserved V8 live preprocessing mean/p95/max was 1.96/2.49/3.00 ms (run 1) and 2.03/2.57/3.78 ms (run 2); ONNX was 1.52/3.98/5.27 and 1.71/4.13/4.75 ms. V9 remains far inside the 66.67 ms current-host simulator budget. These are not Raspberry Pi 5 measurements.
