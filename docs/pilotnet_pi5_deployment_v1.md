# PilotNet Raspberry Pi 5 Deployment V1

The deployment core preserves V4's 480×360 RGB input, ROI `(0,160,480,360)`, Pillow bilinear 200×66 resize, direct full-range BT.601-style YUV matrix, `(YUV-0.5)*2` normalization, NCHW float32 layout, ONNX names, `0.349066` rad denormalization, and live physical clamp. It requires only Python, NumPy, Pillow, and ONNX Runtime; PyTorch, ROS, OpenCV, and CUDA are absent from runtime requirements.

Startup verifies the exact canonical ONNX hash and size. The callable boundary is `PilotNetPi5.infer_rgb(image) -> steering_rad`; acquisition and actuation remain outside it. The CLI and benchmark accept image files without assuming ROS, serial, HTTP, or a physical camera implementation.

Proven before this milestone: V4 achieved 3/3 valid same-map, same-spawn 0.50 m/s simulator laps. This milestone proves deterministic preprocessing/output parity and an x86 host CPU benchmark. It does not prove Raspberry Pi speed, physical-camera behavior, actuator integration, real-vehicle closed loop, unseen-map performance, or other generalization.
