"""Minimal CPU-only PilotNet V4 deployment core for Raspberry Pi 5 preparation."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


EXPECTED_ONNX_SHA256 = "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
EXPECTED_ONNX_SIZE_BYTES = 1_012_518
SOURCE_SIZE = (480, 360)
ROI = (0, 160, 480, 360)
MODEL_SIZE = (200, 66)
MAX_STEERING_RAD = 0.349066
RGB_TO_YUV_BT601 = np.asarray([
    [0.29900, 0.58700, 0.11400],
    [-0.14713, -0.28886, 0.43600],
    [0.61500, -0.51499, -0.10001],
], dtype=np.float32)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: str | Path, expected_sha256: str = EXPECTED_ONNX_SHA256) -> dict[str, Any]:
    model_path = Path(path)
    actual = sha256_file(model_path)
    if actual != expected_sha256:
        raise RuntimeError(f"V4 ONNX SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    if expected_sha256 == EXPECTED_ONNX_SHA256 and model_path.stat().st_size != EXPECTED_ONNX_SIZE_BYTES:
        raise RuntimeError(f"V4 ONNX size mismatch: expected {EXPECTED_ONNX_SIZE_BYTES}, got {model_path.stat().st_size}")
    return {"result": "PASS", "sha256": actual, "size_bytes": model_path.stat().st_size}


def decode_rgb_image(payload: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"cannot decode input image: {exc}") from exc
    if rgb.size != SOURCE_SIZE:
        raise ValueError(f"input image must be 480x360, got {rgb.size}")
    return rgb


def preprocess_rgb_image(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        raise ValueError(f"input image must be RGB, got mode {image.mode!r}")
    if image.size != SOURCE_SIZE:
        raise ValueError(f"input image must be 480x360, got {image.size}")
    resized = image.crop(ROI).resize(MODEL_SIZE, Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.uint8).astype(np.float32) / np.float32(255.0)
    yuv = rgb @ RGB_TO_YUV_BT601.T
    yuv[..., 1:] += np.float32(0.5)
    normalized = (yuv - np.float32(0.5)) * np.float32(2.0)
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)
    return np.expand_dims(chw, 0)


def preprocess_image_bytes(payload: bytes) -> np.ndarray:
    return preprocess_rgb_image(decode_rgb_image(payload))


def steering_normalized_to_rad(value: float) -> float:
    physical = float(np.float32(value) * np.float32(MAX_STEERING_RAD))
    if not math.isfinite(physical):
        raise RuntimeError("ONNX produced non-finite steering")
    return max(-MAX_STEERING_RAD, min(MAX_STEERING_RAD, physical))


class PilotNetPi5:
    """Image-source-independent inference boundary: RGB image -> steering radians."""

    def __init__(
        self, model_path: str | Path, *, expected_sha256: str = EXPECTED_ONNX_SHA256,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.identity = verify_model(self.model_path, expected_sha256)
        if session_factory is None:
            import onnxruntime as ort
            session_factory = lambda path: ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.session = session_factory(str(self.model_path))
        input_meta = self.session.get_inputs()[0]; output_meta = self.session.get_outputs()[0]
        if input_meta.name != "camera_yuv" or list(input_meta.shape) != ["batch", 3, 66, 200]:
            if input_meta.name != "camera_yuv" or list(input_meta.shape)[-3:] != [3, 66, 200]:
                raise RuntimeError(f"unexpected ONNX input contract: {input_meta.name} {input_meta.shape}")
        if output_meta.name != "steering_normalized" or list(output_meta.shape)[-1:] != [1]:
            raise RuntimeError(f"unexpected ONNX output contract: {output_meta.name} {output_meta.shape}")

    def infer_tensor(self, tensor: np.ndarray) -> float:
        if tensor.shape != (1, 3, 66, 200) or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
            raise ValueError(f"model tensor must be contiguous float32 1x3x66x200, got {tensor.shape} {tensor.dtype}")
        output = self.session.run(["steering_normalized"], {"camera_yuv": tensor})[0]
        return steering_normalized_to_rad(float(np.asarray(output).reshape(-1)[0]))

    def infer_rgb(self, image: Image.Image) -> float:
        return self.infer_tensor(preprocess_rgb_image(image))

    def infer_bytes(self, payload: bytes) -> float:
        return self.infer_tensor(preprocess_image_bytes(payload))


def _timing(values: Sequence[float]) -> dict[str, float | int]:
    milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
    return {"count": len(values), "mean_ms": float(np.mean(milliseconds)), "median_ms": float(np.median(milliseconds)),
            "p95_ms": float(np.percentile(milliseconds, 95)), "max_ms": float(np.max(milliseconds))}


def infer_file(model: PilotNetPi5, image_path: str | Path) -> dict[str, Any]:
    total_start = time.perf_counter(); decode_start = total_start
    payload = Path(image_path).read_bytes(); image = decode_rgb_image(payload); decode_s = time.perf_counter() - decode_start
    preprocess_start = time.perf_counter(); tensor = preprocess_rgb_image(image); preprocess_s = time.perf_counter() - preprocess_start
    inference_start = time.perf_counter(); steering = model.infer_tensor(tensor); inference_s = time.perf_counter() - inference_start
    return {"steering_rad": steering, "steering_deg": math.degrees(steering), "decode_ms": decode_s * 1000.0,
            "preprocess_ms": preprocess_s * 1000.0, "inference_ms": inference_s * 1000.0,
            "total_ms": (time.perf_counter() - total_start) * 1000.0}


def benchmark_file(model: PilotNetPi5, image_path: str | Path, warmup: int = 20, iterations: int = 250) -> dict[str, Any]:
    if warmup < 0 or iterations < 200:
        raise ValueError("benchmark requires nonnegative warmup and at least 200 measured iterations")
    path = Path(image_path)
    for _ in range(warmup):
        model.infer_bytes(path.read_bytes())
    decode: list[float] = []; preprocess: list[float] = []; inference: list[float] = []; total: list[float] = []
    outputs: list[float] = []
    for _ in range(iterations):
        total_start = time.perf_counter(); decode_start = total_start
        image = decode_rgb_image(path.read_bytes()); decode.append(time.perf_counter() - decode_start)
        start = time.perf_counter(); tensor = preprocess_rgb_image(image); preprocess.append(time.perf_counter() - start)
        start = time.perf_counter(); outputs.append(model.infer_tensor(tensor)); inference.append(time.perf_counter() - start)
        total.append(time.perf_counter() - total_start)
    import onnxruntime as ort
    return {"benchmark_label": "HOST CPU BENCHMARK — NOT RASPBERRY PI 5",
            "runtime_environment": {"system": platform.system(), "machine": platform.machine(), "processor": platform.processor(),
                                    "python": platform.python_version(), "numpy": np.__version__, "Pillow": Image.__version__, "onnxruntime": ort.__version__},
            "warmup_iterations": warmup,
            "measured_iterations": iterations, "decode": _timing(decode), "preprocessing": _timing(preprocess),
            "onnx_inference": _timing(inference), "total_pipeline": _timing(total),
            "deterministic_output_range_rad": [min(outputs), max(outputs)], "control_period_budget_ms": 1000.0 / 15.0,
            "total_p95_headroom_ms": 1000.0 / 15.0 - _timing(total)["p95_ms"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True); parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true"); parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=250); parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        model = PilotNetPi5(args.model)
        result = benchmark_file(model, args.image, args.warmup, args.iterations) if args.benchmark else infer_file(model, args.image)
        result["model_identity"] = model.identity
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
