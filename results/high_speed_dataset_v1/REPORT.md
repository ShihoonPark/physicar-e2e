# High-Speed Dataset V1

Result: **PASS**. The canonical extractor produced 2,911 RGB 200×66 samples from 3,330 camera messages. Overall retention is 87.42%; active-window retention is 100%. The 419 excluded frames are lifecycle boundary frames (396 before and 23 after the drive window), not stale/decode/future-label failures.

Causal zero-order hold was enforced (`steering_time <= camera_time`). Future-label violations, stale steering rejects, stale speed rejects, and image decode failures are all zero. Steering label age mean/median/p95/max is 34.75/34.55/63.48/71.82 ms. Speed label age mean/median/p95/max is 34.52/34.40/62.94/87.88 ms.

| Episode | Camera | Accepted | Retention | Steering age mean/p95/max ms | Steering min/max | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| 001 | 277 | 242 | 87.36% | 33.26/60.98/68.94 | -0.349066/0.349066 | 4.55% |
| 002 | 277 | 242 | 87.36% | 37.37/64.68/67.54 | -0.349066/0.349066 | 6.20% |
| 003 | 279 | 244 | 87.46% | 34.40/63.27/69.80 | -0.349066/0.349066 | 5.33% |
| 004 | 278 | 243 | 87.41% | 33.72/62.20/68.56 | -0.349066/0.349066 | 7.00% |
| 005 | 276 | 242 | 87.68% | 32.68/62.78/69.39 | -0.349066/0.349066 | 4.96% |
| 006 | 277 | 242 | 87.36% | 33.41/63.59/69.29 | -0.349066/0.349066 | 4.55% |
| 007 | 277 | 242 | 87.36% | 38.98/65.03/71.82 | -0.349066/0.349066 | 5.37% |
| 008 | 277 | 243 | 87.73% | 31.09/61.96/67.69 | -0.349066/0.349066 | 4.94% |
| 009 | 278 | 243 | 87.41% | 37.42/63.40/70.00 | -0.349066/0.349066 | 6.58% |
| 010 | 278 | 243 | 87.41% | 33.74/62.65/67.69 | -0.349066/0.349066 | 5.76% |
| 011 | 278 | 243 | 87.41% | 36.12/64.53/68.11 | -0.349066/0.349066 | 5.76% |
| 012 | 278 | 242 | 87.05% | 34.84/63.26/67.52 | -0.349066/0.349066 | 4.55% |

Split is frozen at episode level: training 001–008 (1,940 samples), validation 009–010 (486), holdout 011–012 (485). Manifest hashes are in `summary.json`; MCAP hashes are in the per-episode files. No V4 or V2 recovery data is referenced.

Twelve contact sheets were generated. Representative sheets 001, 006, and 012 were visually reviewed: cone-free route and full-lap coverage were intact, with no reset frames or malformed crops observed. Derived data remains outside Git at `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_v1/dataset/`.
