# Real Temporal PilotNet V1

Result: **SELECTED**

This is an offline real-data training comparison. No physical-car drive, simulator drive, data collection, raw-bag modification, or runtime integration was performed. Offline validation is not real-robot driving success.

## Frozen data and split

REAL_DATASET_V1 manifest SHA-256: `ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597` (PASS).

| Split | Bags | Sequences | Steering mean / median / std rad | Left / right / near-zero | Exact-zero speed |
|---|---|---:|---|---|---:|
| TRAIN | bag_01, bag_02 | 1694 | 0.038839 / 0.024438 / 0.116354 | 1086 / 417 / 191 | 0 |
| VALIDATION | bag_03 | 450 | 0.004800 / 0.022102 / 0.096382 | 269 / 155 / 26 | 0 |

TRAIN began with 1,713 grouped sequences; the only selection filter removed 19 target-time speeds exactly equal to 0.0 m/s, leaving 1,694. Validation retained all 450 bag_03 sequences and contained no exact-zero speeds. Near-zero steering means abs(steering_rad) <= 0.01. The left-heavy distribution was not rebalanced.

Speed distribution (metadata only; m/s):

| Split | Min | p05 | Mean | Median | Std | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 0.200000 | 0.507999 | 0.792025 | 0.846182 | 0.187551 | 0.988654 | 1.100000 |
| VALIDATION | 0.200000 | 0.592853 | 0.804529 | 0.830452 | 0.150780 | 0.971077 | 0.998168 |

Speed remains metadata only and its command-versus-feedback semantics remain unresolved.

Magnitude-bin counts:

| Split | <0.05 | 0.05–<0.15 | 0.15–<0.25 | ≥0.25 rad |
|---|---:|---:|---:|---:|
| TRAIN | 810 | 499 | 249 | 136 |
| VALIDATION | 189 | 209 | 46 | 6 |

## Training

Both models use N×9×66×200 Temporal PilotNet with 255,819 parameters and produce physical steering radians. Targets were the manifest `steering_rad` values without additional scaling or clipping.

Transfer initialization used only the frozen simulator D1 checkpoint `b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434`; all 255,819 parameters were fine-tuned on real TRAIN data.

| Model | Initialization | LR | Epochs / best | Best validation MSE rad² | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---|
| REAL-SCRATCH-V1 | from_scratch | 0.001 | 13 / 6 | 0.006706446 | `02881b5b2d21768c4cf93b71e5d6c2a666043e34c08b71f4247b9545df3dc8e3` |
| REAL-D1-TRANSFER-V1 | exact_frozen_simulator_D1 | 0.0001 | 20 / 18 | 0.012205225 | `e6063eb5c681edf945f8ec4258b77f25f2e456c1140045f501e3b7ab18458ffc` |

## Bag_03 validation

| Model | n | MAE | RMSE | Bias | Median AE | p95 AE | Max AE | Pearson | Magnitude ratio | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REAL-SCRATCH-V1 | 450 | 0.047842 | 0.081893 | 0.014950 | 0.021560 | 0.174105 | 0.393489 | 0.569241 | 0.743113 | 0.808889 |
| REAL-D1-TRANSFER-V1 | 450 | 0.071795 | 0.110477 | 0.014428 | 0.046164 | 0.199043 | 0.487228 | 0.179731 | 0.687646 | 0.644444 |

Trivial baselines:

| Baseline | Prediction rad | n | MAE rad | RMSE rad |
|---|---:|---:|---:|---:|
| ZERO | 0.000000 | 450 | 0.076174 | 0.096502 |
| MEAN | 0.038839 | 450 | 0.075136 | 0.102216 |

Per-bin combined/left/right results (MAE rad; count):

### REAL-SCRATCH-V1

| Target magnitude | Combined | LEFT | RIGHT |
|---|---:|---:|---:|
| <0.05 | 0.019541 (189) | 0.015733 (133) | 0.028585 (56) |
| 0.05–<0.15 | 0.050641 (209) | 0.039399 (129) | 0.068769 (80) |
| 0.15–<0.25 | 0.114848 (46) | 0.091822 (15) | 0.125989 (31) |
| ≥0.25 | 0.328104 (6) | 0.046173 (1) | 0.384490 (5) |

### REAL-D1-TRANSFER-V1

| Target magnitude | Combined | LEFT | RIGHT |
|---|---:|---:|---:|
| <0.05 | 0.039706 (189) | 0.037314 (133) | 0.045388 (56) |
| 0.05–<0.15 | 0.068197 (209) | 0.058677 (129) | 0.083548 (80) |
| 0.15–<0.25 | 0.177340 (46) | 0.091418 (15) | 0.218915 (31) |
| ≥0.25 | 0.398760 (6) | 0.029652 (1) | 0.472582 (5) |

## Selection and export

REAL-SCRATCH-V1 had a clear 33.36% lower bag_03 validation MAE. Selected candidate: **REAL-SCRATCH-V1**.

Under this controlled run, simulator D1 pretraining did not help the real visual domain.

Both model ONNX files passed the N×9×66×200 → N×1 shape gate, 255,819-parameter gate, ONNX checker, and PyTorch↔ONNX equivalence gate.

Selected checkpoint SHA-256: `02881b5b2d21768c4cf93b71e5d6c2a666043e34c08b71f4247b9545df3dc8e3`.

Selected ONNX SHA-256: `b860afe396c8e48001339b4f99c8b3daa272500725d48d79b9c22b859c6fd339`.

## Current x86 batch=1 timing

| Component | Mean ms | p95 ms |
|---|---:|---:|
| Preprocess | 2.264 | 3.953 |
| ONNX inference | 0.664 | 2.120 |
| Total | 2.930 | 5.424 |

These measurements are from the current x86 CPU with batch=1; they are not Raspberry Pi timing and make no Pi performance claim.

## Deferred runtime work

Camera acquisition, the three-frame runtime buffer, steering publication, speed policy, GREEN traffic-light start gate, watchdog, and safe stop remain for the separately authorized runtime-integration milestone. The neural observation remains camera-only.
