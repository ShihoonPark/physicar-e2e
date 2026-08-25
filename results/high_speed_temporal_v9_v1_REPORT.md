# High-Speed Temporal PilotNet V9 V1 — Final Report

Result: **PASS**.

Temporal PilotNet V9 achieved 3/3 valid same-map, same-spawn, 1.80 m/s simulation laps using only causal camera observations.

V4 remains the canonical 0.50 m/s single-frame policy. V9 is frozen as the canonical 1.80 m/s temporal policy. Cone Avoidance V1 is justified, but was not implemented here.

## Data reuse and causal sequence construction

No new training driving data was collected. There were zero new bags, Expert laps, neural rollouts, synthetic recovery states, or DAgger iterations. DAgger remains closed and Iteration 4 was not created. The external V9 root contains only six temporal CSV manifests, the V9 checkpoint, and the V9 ONNX; it contains no `.mcap`, `.db3`, or `.bag` file.

Training reused exactly:

- nominal High-Speed episodes 001–008;
- DAgger Iteration-1 rollout A;
- DAgger Iteration-2 rollout A;
- DAgger Iteration-3 rollout A.

Evaluation retained exactly the frozen roles:

- nominal validation episodes 009–010;
- nominal holdout episodes 011–012;
- DAgger Iteration-1 rollout B;
- DAgger Iteration-2 rollout B;
- DAgger Iteration-3 rollout B.

Every manifest record is `[image(t-2), image(t-1), image(t)] -> Expert steering(t)`. All timestamps are canonical source simulator timestamps, strictly increasing within one episode/rollout, with no reset/source boundary crossing and no future image. The adjacent-gap gate was frozen at `0.120 s`. Missing history was rejected, never padded. Each frame was independently processed as `480x360 RGB -> crop y=160:360 -> bilinear 200x66 -> canonical RGB-to-YUV -> normalization -> CHW`; tensors were concatenated oldest-to-current into `9x66x200`.

| Stratum | Temporal candidates | Accepted | Gap rejects | Boundary rejects | Adjacent gap mean / median / p95 / max (s) | Oldest-current span mean / median / p95 / max (s) |
|---|---:|---:|---:|---:|---|---|
| Train | 3,512 | 3,510 | 2 | 22 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |
| Nominal validation | 482 | 482 | 0 | 4 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |
| Nominal holdout | 481 | 481 | 0 | 4 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |
| DAgger1-B | 41 | 41 | 0 | 2 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |
| DAgger2-B | 44 | 44 | 0 | 2 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |
| DAgger3-B | 1,013 | 1,011 | 2 | 2 | .066 / .065 / .070 / .070 | .132 / .130 / .135 / .135 |

The 3,510 training sequences comprise 1,924 nominal, 48 DAgger1-A, 43 DAgger2-A, and 1,495 DAgger3-A sequences. Source-ID overlap and source-hash overlap between training and evaluation were both false. All B holdouts and all low-speed/V4 data were excluded. The temporal manifests reference existing images rather than duplicating PNGs.

## V9 architecture and training

V9 changed only the first convolution from `3 -> 24` to `9 -> 24` channels. Its 5x5 kernel and stride 2 remain unchanged; every later convolution and fully connected layer is identical to V8. The verified parameter count is **255,819**, exactly 3,600 more than V8's 252,219.

Training began from scratch with the frozen seed `20260824`, MSE, Adam, learning rate `1e-3`, batch 64, maximum 35 epochs, and the unchanged early-stopping rule. There was no initialization from V8, augmentation, weighting, balancing, resampling, or sweep. Best validation normalized MSE was `0.00261318` at epoch 17; early stopping ended training after epoch 24. Best-epoch training normalized MSE was `0.00475068`.

Checkpoint:

- path: `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_temporal_v1/v9/checkpoints/pilotnet_v9_high_speed_temporal_best.pt`
- size: 1,032,805 bytes
- SHA-256: `1cded5fcc7f3d13242de096c4868fc576d03fa6bc86df6e5c8c7c235d9faa6cc`

ONNX:

- path: `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_temporal_v1/v9/onnx/pilotnet_v9_high_speed_temporal.onnx`
- size: 1,026,938 bytes
- SHA-256: `7f6aa4c2d8c9b3615c580f065660c674efff94ff4cd0b9bdc9357df904000888`
- checker: PASS
- contract: `N x 9 x 66 x 200 -> N x 1`
- PyTorch/ONNX equivalence: PASS on 128 samples; mean/max difference `5.51e-9 / 6.24e-8 rad`

For preserved-baseline identity, the V8 checkpoint SHA-256 was `536ebbfa3b33ae0d532b0044a1aff7d255d5c5c0e92fa8f7ef3773f608a605ae` and its ONNX SHA-256 was `e79afeb021a7d56fff20049026b3f05850906dc371ce01bf9811be77cf8385ca`.

## Matched V8 versus V9 offline comparison

V8 consumed only each sequence's current frame `t`; V9 consumed `[t-2,t-1,t]`; both used the exact same Expert target at `t`. Target identity was verified for every matched row. Values below are `MAE / RMSE / bias / max error / correlation / corrective magnitude ratio`.

| Stratum (matched count) | V8 current frame | V9 causal temporal |
|---|---|---|
| Nominal validation (482) | .01129 / .01836 / .00380 / .09614 / .99328 / 1.0315 | .01063 / .01784 / .00108 / .09578 / .99329 / .9898 |
| Nominal holdout (481) | .01202 / .01955 / .00434 / .09793 / .99249 / 1.0407 | .01073 / .01747 / .00140 / .07622 / .99349 / .9990 |
| DAgger1-B (41) | .02688 / .04065 / .01078 / .12287 / .98363 / .9588 | .02130 / .03271 / .00127 / .11467 / .98865 / .9955 |
| DAgger2-B (44) | .04047 / .06407 / -.00724 / .22550 / .96847 / .9453 | .03533 / .05200 / -.01165 / .16576 / .98021 / .9573 |
| DAgger3-B (1,011) | .02080 / .04388 / .00447 / .46254 / .97315 / .9985 | .01817 / .04060 / .00225 / .48289 / .97680 / .9799 |

DAgger3-B route-region values use the same metric order:

| Region (count) | V8 | V9 |
|---|---|---|
| 85–90% (990) | .02048 / .04371 / .00441 / .46254 / .97310 / 1.0012 | .01801 / .04072 / .00199 / .48289 / .97640 / .9803 |
| 90–95% (7) | .03835 / .06090 / -.03491 / .15001 / .51952 / .8988 | .02067 / .02886 / -.00646 / .06754 / .38538 / .9813 |
| 95–100% (14) | .03475 / .04550 / .02852 / .10427 / .98128 / .9436 | .02837 / .03721 / .02529 / .07448 / .98926 / .9544 |

The optional repeated-current-frame ablation `[t,t,t]` on DAgger3-B produced MAE/RMSE/bias/max/correlation/ratio `.02086 / .04746 / .00546 / .53525 / .96847 / .9786`, versus causal V9's `.01817 / .04060 / .00225 / .48289 / .97680 / .9799`. It was inference-only and did not affect model selection or training.

## Live temporal gate

Each attempt performed an independent reset and full preflight in the canonical cone-free world. While safely stopped, it acquired three real HTTP camera frames; startup used zero duplicate padding and motion began only after the causal buffer was valid. Only acquisition of a new frame advanced the buffer. All adjacent gaps remained below 0.120 s. The neural observation fields were exactly camera YUV at `t-2`, `t-1`, and `t`; pose, route, CTE, clock, speed, and Expert commands were not neural inputs.

| Metric | Run 1 | Run 2 | Run 3 |
|---|---:|---:|---:|
| Classification | POLICY_PASS | POLICY_PASS | POLICY_PASS |
| Completion / progress | 99.06% / 30.217 m | 98.58% / 30.070 m | 98.70% / 30.108 m |
| Elapsed | 17.133 s | 16.920 s | 16.935 s |
| Final distance to start | .292 m | .253 m | .211 m |
| Mean / max CTE | .0703 / .5022 m | .0650 / .4556 m | .0604 / .4216 m |
| Off-track events / duration | 1 / .468 s | 1 / .330 s | 1 / .199 s |
| Mean / max absolute steering | .1273 / .3491 rad | .1276 / .3491 rad | .1260 / .3491 rad |
| Saturation / mean command delta | 3.52% / .02687 rad | 2.37% / .02710 rad | 2.37% / .02686 rad |
| Camera acquisition mean / p95 / max | 2.324 / 3.650 / 6.089 ms | 2.587 / 4.126 / 6.340 ms | 2.569 / 4.014 / 5.756 ms |
| Preprocessing mean / p95 / max | 1.713 / 2.321 / 2.812 ms | 1.879 / 2.576 / 3.894 ms | 1.802 / 2.296 / 3.244 ms |
| ONNX mean / p95 / max | 1.364 / 4.162 / 5.285 ms | 1.472 / 4.179 / 4.612 ms | 1.405 / 4.130 / 5.318 ms |
| Total temporal model path mean / p95 / max | 3.084 / 6.294 / 7.090 ms | 3.360 / 6.294 / 6.691 ms | 3.219 / 6.324 / 7.571 ms |
| Loop Hz / period p95 / max | 15.038 / 66.691 / 66.750 ms | 15.044 / 66.689 / 68.224 ms | 15.032 / 66.689 / 76.260 ms |
| Adjacent gaps mean / p95 / max | 66.29 / 68.40 / 71.56 ms | 66.25 / 68.40 / 72.67 ms | 66.30 / 68.78 / 76.22 ms |
| Oldest-current span mean / p95 / max | 132.57 / 135.06 / 136.95 ms | 132.47 / 135.03 / 139.00 ms | 132.57 / 135.49 / 142.48 ms |
| Temporal invalid histories | 0 | 0 | 0 |
| >100 ms loop slips | 0 | 0 | 0 |
| API / pose-clock liveness failures | 0 / 0 | 0 / 0 | 0 / 0 |
| Safe stop | PASS | PASS | PASS |

There were three total attempts and three valid neural evaluations; no replacement attempt was needed. There were zero `INFRA_FAIL` and zero `TEMPORAL_INPUT_FAIL` outcomes.

## Current-host compute comparison

Preserved V8 live evidence reported preprocessing mean/p95/max of `1.957/2.485/2.997 ms` and `2.033/2.568/3.783 ms` across its two runs. V8 ONNX inference was `1.515/3.983/5.274 ms` and `1.715/4.131/4.754 ms`. The corresponding V9 component ranges across three runs were preprocessing `1.713–1.879 / 2.296–2.576 / 2.812–3.894 ms` and ONNX `1.364–1.472 / 4.130–4.179 / 4.612–5.318 ms` for mean/p95/max.

V9 directly measured total temporal model-path mean/p95/max at `3.084–3.360 / 6.294–6.324 / 6.691–7.571 ms`, far inside the approximately 66.67 ms control budget. Historical V8 evidence did not preserve a directly sampled total-path distribution; its component mean sums were 3.472 and 3.748 ms, so no unsupported V8 total p95/max is inferred. These are current-host simulator measurements, not Raspberry Pi 5 or real-robot measurements.

## High-Speed progression and decision

| Progression | Result |
|---|---|
| High-Speed Expert V1, 1.80 m/s / 0.90 m lookahead | 3/3 PASS |
| V4, 0.50 m/s | 3/3 PASS |
| V4, 1.80 m/s | FAIL 10.48% |
| V5 single-frame nominal | FAIL 67.82% |
| V6 single-frame + DAgger1 | FAIL 83.40% |
| V7 single-frame + DAgger1+2 | PASS, then FAIL 96.05% |
| V8 single-frame + DAgger1+2+3 | PASS, then FAIL 94.41% |
| V9 three-frame, same preserved High-Speed sources | **3/3 PASS** |

The structural A/B result supports causal three-frame observation over V8's matched single-frame input for this simulator condition. V9 can be frozen as the 1.80 m/s temporal baseline. Cone Avoidance V1 is permitted by the requested gate; no cone work was performed.

## Storage, verification, files, and limitations

External manifests and model artifacts are under `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_temporal_v1/`. Compact evidence is under:

- `results/high_speed_temporal_dataset_v1/`
- `results/pilotnet_training_v9_high_speed_temporal/`
- `results/pilotnet_e2e_v9_high_speed_temporal/`

Focused temporal tests passed 12/12. The final full regression passed 276/276 tests. `git diff --check` passed.

Added files are the three V9 configs, temporal model and orchestration modules, runner, focused tests, documentation, the three compact result directories, and this report. No tracked baseline file was modified.

Final Git state: branch `experiment/pilotnet-high-speed-temporal-v1`, HEAD `9df5aa11adefe175703e14ca560864c1748a6e61`. The tracked diff is empty; the twelve intended V9 path groups are untracked. No files were staged, committed, or pushed.

Limitations: this is only same-map, same-spawn simulation repeatability. Training used preserved raw-ROS-derived images while live inference used HTTP JPEG. DAgger1-B, DAgger2-B, and the upper DAgger3-B route bins are small. No real robot, Raspberry Pi 5, map-generalization, spawn-generalization, or perturbation claim is made.
