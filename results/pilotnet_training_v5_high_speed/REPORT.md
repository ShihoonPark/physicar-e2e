# High-Speed PilotNet V5 Training

Result: **PASS**. V5 was initialized from scratch and trained only on 1.80 m/s episodes 001–008. It uses the unchanged V4 PilotNet topology (3×66×200 input; 3→24→36→48→64→64 convolutions; 1152→100→50→10→1 fully connected layers), totaling 252,219 parameters.

Training used MSE, Adam 1e-3, batch size 64, deterministic seed, maximum 35 epochs, and unchanged early stopping. It early-stopped after epoch 21; the best checkpoint is epoch 14 with train/validation normalized MSE 0.00173870/0.00202253. Tiny-overfit sanity passed.

| Offline set | Samples | MAE | RMSE | Bias | Max error | Correlation | Pred/GT slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation 009–010 | 486 | 0.01012 | 0.01570 | -0.00311 | 0.07318 | 0.99494 | 0.98507 |
| Holdout 011–012 | 485 | 0.01040 | 0.01554 | -0.00289 | 0.07192 | 0.99497 | 0.99253 |

| Magnitude bin | Validation count/MAE/ratio | Holdout count/MAE/ratio |
|---|---:|---:|
| <0.05 rad | 226 / 0.00469 / 1.128 | 222 / 0.00459 / 1.166 |
| 0.05–0.15 rad | 100 / 0.01609 / 0.930 | 100 / 0.01538 / 0.937 |
| 0.15–0.25 rad | 86 / 0.01563 / 0.985 | 89 / 0.01813 / 0.995 |
| >=0.25 rad | 74 / 0.01223 / 0.987 | 74 / 0.01180 / 0.993 |

Checkpoint SHA-256: `04cc593426d2e79a703e4218c7041d2cf1317c2039254643d7e9fb612fd3a101` (1,017,549 bytes).

ONNX SHA-256: `404b2ea24d25d0178c60ba9167496f93ba50b10ade78cfcf6edfc0f64658a1fd` (1,012,518 bytes). ONNX contract/checker and numerical equivalence passed across 128 samples; maximum PyTorch↔ONNX difference was 6.24e-8 rad. Artifacts remain outside Git under `/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/high_speed_v1/v5/`.

Offline accuracy is not treated as closed-loop proof.
