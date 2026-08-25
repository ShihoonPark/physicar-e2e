# High-Speed PilotNet DAgger Iteration 3 V1

## Decision

**PARTIAL SUPPORT / NOT REPEATABLE.** V8 live run #1 passed, but run #2 was a genuine policy failure. The hard gate stopped evaluation: run #3 and DAgger Iteration 4 were not executed. V8 is not frozen and Cone Avoidance V1 remains blocked.

## Preserved evidence

- Expert config SHA-256: `3afdc5d3204143e8d1f64c1ec68b2bda2de08912b779f2ece69a7d86c99503c9`; 1.80 m/s, 0.90 m lookahead, 15 Hz, 3/3 PASS.
- V4 checkpoint/ONNX: `a581c1a6...01d1d9` / `5dd2b88b...3f396a`; 0.50 m/s 3/3 PASS.
- V5 checkpoint: `04cc5934...3a101`; 1.80 m/s FAIL 67.82%.
- V6 checkpoint: `79e21210...5aa55`; FAIL 83.40%.
- V7 checkpoint/ONNX: `99a4671c...b4ea93` / `d6925fa4...4fe6bad`; run #1 PASS, run #2 FAIL 96.05%.

## Pre-registered collection

The config froze selection before collection as every valid raw ROS `/camera/image_raw` frame at route completion >=85% through the final driving frame. PASS and late POLICY_FAIL rollouts were both admissible. V7 alone controlled steering; the fixed High-Speed Expert was shadow-only.

| Evidence | Rollout A (train) | Rollout B (holdout) |
|---|---:|---:|
| Policy result | FAIL | PASS |
| Completion | 96.22% | 99.11% |
| Progress | 29.350 m | 30.234 m |
| Raw size | 893,866,080 B | 643,005,611 B |
| MCAP SHA-256 | `118a3ae9...955e3` | `0fd9f582...7f749` |
| Raw camera frames | 1,721 | 1,238 |
| Target-region/accepted | 1,499 / 1,499 | 1,015 / 1,015 |
| Label age mean/median/p95/max | 48.98/50/95/145 ms | 48.68/50/95/130 ms |
| Stale/future/decode failures | 0/0/0 | 0/0/0 |
| Safe stop | PASS | PASS |

A/B source hashes, files, and derived image hashes are disjoint. B is absent from training and did not affect selection. Late-region counts were A: 1,478/15/6 and B: 994/7/14 for 85–90/90–95/95–100%.

## V8 training and offline evaluation

V8 was initialized from scratch with the unchanged 252,219-parameter architecture and frozen training configuration. Composition: 1,940 nominal + 50 DAgger1-A + 45 DAgger2-A + 1,499 DAgger3-A = **3,534** samples. All validation/holdout and low-speed/recovery sources were excluded. Best epoch 12; early stopping after 19 epochs; best validation normalized MSE 0.00274324.

Metrics are `MAE / RMSE / bias / max / correlation / corrective ratio` in radians where applicable:

| Stratum (count) | V5 | V6 | V7 | V8 |
|---|---|---|---|---|
| Nominal validation (486) | .01012/.01570/-.00311/.07318/.99494/.9811 | .00951/.01544/.00071/.08378/.99491/.9967 | .00976/.01519/-.00137/.07043/.99529/1.0152 | .01121/.01828/.00378/.09614/.99328/1.0316 |
| Nominal holdout (485) | .01040/.01554/-.00289/.07192/.99497/.9896 | .00968/.01573/.00113/.08239/.99469/1.0042 | .00982/.01517/-.00085/.06949/.99537/1.0234 | .01193/.01947/.00431/.09793/.99250/1.0408 |
| DAgger1-B (43) | .10797/.16572/.09200/.35810/.79983/.5887 | .03526/.04712/.01318/.15099/.97988/.9249 | .03608/.05894/.01639/.22367/.96688/.9196 | .02624/.03989/.01088/.12287/.98492/.9640 |
| DAgger2-B (46) | .16210/.21516/-.06386/.43999/.58828/.6592 | .13740/.19746/-.09196/.46745/.72966/.8074 | .05018/.08164/-.01741/.30514/.94836/.9393 | .04014/.06313/-.00618/.22550/.96789/.9491 |
| DAgger3-B (1,015) | .04107/.08737/.01429/.45581/.89184/.8500 | .03547/.06850/.00787/.40047/.93493/.8840 | .03905/.08669/.01254/.53674/.89264/.9176 | .02090/.04395/.00446/.46254/.97325/.9985 |

DAgger3-B V7 → V8 MAE by subregion: 85–90% `.03900 → .02059` (994); 90–95% `.05263 → .03835` (7); 95–100% `.03531 → .03475` (14). V8 improved both earlier DAgger holdouts, while nominal MAE increased modestly and is reported without an invented threshold.

V8 checkpoint: 1,018,293 bytes, SHA-256 `536ebbfa3b33ae0d532b0044a1aff7d255d5c5c0e92fa8f7ef3773f608a605ae`. V8 ONNX: 1,012,518 bytes, SHA-256 `e79afeb021a7d56fff20049026b3f05850906dc371ce01bf9811be77cf8385ca`. Checker and I/O contract (`batch×3×66×200` → `batch×1`) passed. PyTorch/ONNX maximum difference was `4.16e-8 rad`; equivalence passed.

## V8 live gate

| Metric | Run #1 | Run #2 |
|---|---:|---:|
| Classification | POLICY_PASS | POLICY_FAIL |
| Elapsed | 17.239 s | 16.422 s |
| Progress/completion | 30.267 m / 99.22% | 28.799 m / 94.41% |
| Final start distance | 0.240 m | 1.622 m |
| Mean/max CTE | .0683/.5107 m | .0712/.5924 m |
| Off-track events/duration | 1/.470 s | 1/.535 s |
| Mean/max steering | .1336/.3491 rad | .1358/.3491 rad |
| Saturation | 5.79% | 5.28% |
| Mean command delta | .02847 rad | .02797 rad |
| Camera mean/p95/max | 2.536/3.701/5.379 ms | 2.815/4.380/5.255 ms |
| Preprocess mean/p95/max | 1.957/2.485/2.997 ms | 2.033/2.568/3.783 ms |
| ONNX mean/p95/max | 1.515/3.983/5.274 ms | 1.715/4.131/4.754 ms |
| Loop Hz; p95/max period | 15.0004; 66.691/66.730 ms | 15.0006; 66.686/92.880 ms |
| >100 ms slips | 0 | 0 |
| API / pose-clock liveness failures | 0 / 0 | 0 / 0 |
| Safe stop | PASS | PASS |

## Progression and conclusion

| Policy | Result |
|---|---|
| High-Speed Expert V1, 1.80 m/s, lookahead .90 | 3/3 PASS |
| PilotNet V4, .50 m/s | 3/3 PASS |
| PilotNet V4, 1.80 m/s | FAIL 10.48% |
| PilotNet V5 nominal | FAIL 67.82% |
| PilotNet V6 + DAgger1 | FAIL 83.40% |
| PilotNet V7 + DAgger1+2 | run #1 PASS; run #2 FAIL 96.05% |
| PilotNet V8 + DAgger1+2+3 | run #1 PASS; run #2 FAIL 94.41% |

Cumulative image-only DAgger reached its pre-registered stopping point. Further work should be structural—speed input, temporal observations, speed-aware targets/trajectory, or higher sensing/control frequency—not another automatic arbitrary-data iteration. This is simulation evidence only.

External storage: `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_dagger_iteration3_v1/`. Compact JSON evidence is in this repository. Full regression and final Git status are recorded at handoff. No commit or push was performed.
