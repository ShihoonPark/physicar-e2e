# PilotNet Raspberry Pi 5 Deployment V1

## Outcome

Deployment preparation passed. The standalone CPU path accepts a decoded 480×360 RGB image, performs the exact V4 crop/resize/YUV/normalization contract, executes the immutable V4 ONNX, and returns physically clamped `steering_rad`. Startup rejects any model whose SHA-256 or size differs from the canonical artifact.

The runtime set is Python 3, `numpy==2.4.6`, `Pillow==12.3.0`, and `onnxruntime==1.29.0`. It contains no PyTorch, ROS, OpenCV, CUDA, training code, controller, or actuator protocol.

## Identity and parity

- ONNX SHA-256: `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a`.
- Size: 1,012,518 bytes.
- Input/output names and shapes: `camera_yuv` N×3×66×200 → `steering_normalized` N×1.
- ROI: x=0:480, y=160:360; resize 200×66 with Pillow bilinear.
- Direct full-range BT.601-style RGB→YUV matrix, U/V +0.5, `(YUV-0.5)*2`, NCHW float32.
- Output: normalized value × 0.349066 rad, then the same ±0.349066 physical clamp used by live V4.

Twelve fixed existing 480×360 JPEG frames spanning the diagnostic rollout were evaluated. Deployment tensors were bit-for-bit equal to canonical live tensors for all 12. Tensor mean/max absolute difference was 0/0. Physical steering mean/p95/max difference was 0/0/0 rad, including saturated high-steering frames. The parity Gate passed.

## Host CPU benchmark

This is explicitly a **HOST CPU BENCHMARK — NOT RASPBERRY PI 5**. Environment: Linux x86_64, Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0, ONNX Runtime 1.29.0. The benchmark used 20 warmups and 250 measured full-file pipeline iterations.

| Stage | Mean | Median | p95 | Max |
|---|---:|---:|---:|---:|
| File read + JPEG decode | 0.808 ms | 0.656 ms | 1.248 ms | 5.733 ms |
| Crop/resize/YUV/normalize | 0.839 ms | 0.671 ms | 1.854 ms | 4.074 ms |
| ONNX CPU inference | 0.605 ms | 0.258 ms | 2.330 ms | 7.739 ms |
| Total pipeline | 2.252 ms | 1.656 ms | 5.369 ms | 9.356 ms |

The nominal 15 Hz budget is 66.67 ms, leaving 61.30 ms relative to host total p95. This does not predict Pi 5 speed and excludes physical camera acquisition, scheduling, actuator communication, and control integration.

## Raspberry Pi command

Copy `deploy/pi5/` to a 64-bit Raspberry Pi OS/Linux system, then place `pilotnet_v4_dagger.onnx` beside the scripts. The ONNX binary is intentionally absent from Git.

```bash
cd pi5
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sha256sum pilotnet_v4_dagger.onnx
python pilotnet_pi5.py --model pilotnet_v4_dagger.onnx --image frame.jpg
python benchmark_pi5.py --model pilotnet_v4_dagger.onnx --image frame.jpg --warmup 20 --iterations 250 --output pi5_benchmark.json
```

The required hash is `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a`; runtime verifies it again and fails loudly on mismatch.

## Scope and limitations

V4's prior 3/3 same-map simulator repeatability remains historical context and was not changed. This milestone proves ONNX identity, standalone import/inference, deterministic canonical parity, and current x86 host timing. It does not prove Raspberry Pi 5 timing, a physical camera, actuator integration, real-vehicle closed-loop behavior, unseen maps/starts, lighting robustness, or generalization.

The next physical step is to copy the bundle and canonical external ONNX to an actual Pi 5 8 GB, capture its platform/thermal/governor context, run the exact 250-iteration benchmark, and then measure camera plus intended actuator/communication overhead before making a 15 Hz deployment judgment.
