# PilotNet V4 on Raspberry Pi 5

This directory is a standalone CPU inference bundle. It does not include the ONNX binary. Place `pilotnet_v4_dagger.onnx` here; startup requires SHA-256 `5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a` and size 1,012,518 bytes.

On 64-bit Raspberry Pi OS/Linux:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sha256sum pilotnet_v4_dagger.onnx
python pilotnet_pi5.py --model pilotnet_v4_dagger.onnx --image frame.jpg
python benchmark_pi5.py --model pilotnet_v4_dagger.onnx --image frame.jpg --warmup 20 --iterations 250 --output pi5_benchmark.json
```

The input image must decode to RGB 480×360. Output is a clamped physical steering angle in radians; this is only an interface boundary and sends no actuator command. The benchmark reports preprocessing, inference, and total p95 against the 66.67 ms nominal control-cycle budget, but camera/control overhead must be measured separately.
