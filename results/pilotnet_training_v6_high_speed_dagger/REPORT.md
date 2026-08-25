# PilotNet V6 High-Speed DAgger Training Report

Result: **PASS**. V6 was initialized from scratch with the unchanged 252,219-parameter PilotNet architecture and frozen V5 training settings. It trained on 1,940 nominal samples from episodes 001–008 plus 50 rollout-A shadow-Expert samples. Nominal validation 009–010, nominal holdout 011–012, rollout B, V4 data, low-speed DAgger data, and recovery data were excluded from training.

Training early-stopped after epoch 16; best epoch 9 had train/validation normalized MSE 0.00217171/0.00195582. Tiny-overfit sanity passed.

| Evaluation stratum | Model | MAE | RMSE | Bias | Max error | Correlation | Corrective ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Nominal validation, 486 | V5 | 0.010116 | 0.015698 | -0.003108 | 0.073177 | 0.994939 | 0.9811 |
| Nominal validation, 486 | V6 | 0.009514 | 0.015437 | 0.000713 | 0.083784 | 0.994907 | 0.9967 |
| Nominal holdout, 485 | V5 | 0.010401 | 0.015538 | -0.002893 | 0.071923 | 0.994967 | 0.9896 |
| Nominal holdout, 485 | V6 | 0.009676 | 0.015726 | 0.001130 | 0.082389 | 0.994695 | 1.0042 |
| V5 rollout-B holdout, 43 | V5 | 0.107968 | 0.165718 | 0.091996 | 0.358098 | 0.799835 | 0.5887 |
| V5 rollout-B holdout, 43 | V6 | 0.035264 | 0.047124 | 0.013179 | 0.150992 | 0.979884 | 0.9249 |

On rollout B pre-divergence samples, V5/V6 MAE was 0.038302/0.028178 rad and corrective ratio was 0.8480/0.9246. On divergence samples, MAE was 0.268735/0.051616 rad and corrective ratio was 0.2050/0.9253. B contained no samples beyond the configured 1.0-second divergence interval because safe stop occurred 0.825 seconds after onset; the late/failure stratum is honestly reported as zero samples.

Checkpoint: `79e21210e984fac1a88fa910987e6b562888d3317e571708e05e900df5f5aa55` (1,017,973 bytes).

ONNX: `3e168565b05b3925e3ab26d9643cdd936cefec34a11e074b918036ba96c3acf6` (1,012,518 bytes). ONNX checker and I/O contract passed. PyTorch↔ONNX equivalence passed on 128 samples; maximum difference was 1.25e-7 rad.

Offline improvement is not treated as closed-loop proof. Artifacts remain outside Git under `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_dagger_v1/v6/`.
