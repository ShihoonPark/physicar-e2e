# High-Speed Expert Collection V1

Result: **PASS — 12/12 independent Expert laps and bags**.

Frozen Expert: 1.80 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad steering, 0.18 m wheelbase. Every episode used the expected cone-free world, reset/spawn lifecycle, 388-point 30.504611 m route, and passed clock/environment checks. All runs had zero off-track events and a successful safe stop.

| Episode | Bag bytes | Camera count/rate | Steering count/rate | Lap time | Mean/max CTE | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 144,869,439 | 277 / 15.156 Hz | 242 / 13.241 Hz | 16.018 s | 0.04919 / 0.19619 m | 4.56% |
| 002 | 144,872,970 | 277 / 15.094 Hz | 242 / 13.187 Hz | 16.016 s | 0.04833 / 0.18170 m | 5.39% |
| 003 | 145,912,048 | 279 / 15.171 Hz | 243 / 13.213 Hz | 16.085 s | 0.04926 / 0.20608 m | 4.96% |
| 004 | 145,391,161 | 278 / 15.147 Hz | 243 / 13.240 Hz | 16.085 s | 0.05135 / 0.22002 m | 5.37% |
| 005 | 144,349,692 | 276 / 15.115 Hz | 242 / 13.253 Hz | 16.017 s | 0.04981 / 0.19355 m | 4.98% |
| 006 | 144,871,917 | 277 / 15.115 Hz | 242 / 13.205 Hz | 16.019 s | 0.04937 / 0.22710 m | 4.56% |
| 007 | 144,868,725 | 277 / 15.081 Hz | 243 / 13.230 Hz | 16.084 s | 0.04974 / 0.18757 m | 4.96% |
| 008 | 144,871,169 | 277 / 15.098 Hz | 243 / 13.245 Hz | 16.082 s | 0.04918 / 0.21781 m | 4.96% |
| 009 | 145,395,272 | 278 / 15.122 Hz | 243 / 13.218 Hz | 16.079 s | 0.04996 / 0.21281 m | 4.96% |
| 010 | 145,394,383 | 278 / 15.123 Hz | 243 / 13.219 Hz | 16.081 s | 0.04913 / 0.19562 m | 5.37% |
| 011 | 145,393,353 | 278 / 15.095 Hz | 243 / 13.195 Hz | 16.079 s | 0.05080 / 0.19960 m | 4.96% |
| 012 | 145,390,359 | 278 / 15.147 Hz | 243 / 13.240 Hz | 16.088 s | 0.04997 / 0.22873 m | 4.55% |

Total raw storage is 1,741,580,488 bytes; mean bag size is 145,131,707 bytes. Camera total/mean rate is 3,330 / 15.122 Hz. Steering total/mean recorded rate is 2,912 / 13.224 Hz; extraction narrows labels to the active driving window. The eight required topics and types are recorded in `summary.json` and each episode report.

Raw bags remain outside Git at `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_v1/raw/`.
