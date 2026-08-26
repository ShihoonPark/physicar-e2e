"""Offline-safe runtime core for the selected REAL-SCRATCH-V1 model.

The module deliberately has no ROS imports and no actuator access.  It owns the
frozen camera/model contracts, causal buffering, inference, steering scaling,
start-gate state, watchdog decisions, and MCAP replay validation.  Publication
is represented by a separately testable callback dispatcher and is disabled by
the canonical configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image

from .pilotnet import IMAGE_HEIGHT, IMAGE_WIDTH, preprocess_png, preprocess_rgb


VERSION = "real_runtime_v1"
SELECTED_MODEL = "REAL-SCRATCH-V1"
SELECTED_ONNX_SHA256 = "b860afe396c8e48001339b4f99c8b3daa272500725d48d79b9c22b859c6fd339"
SELECTED_ONNX_SIZE_BYTES = 1_026_924
SELECTED_CHECKPOINT_SHA256 = "02881b5b2d21768c4cf93b71e5d6c2a666043e34c08b71f4247b9545df3dc8e3"
SELECTED_CHECKPOINT_SIZE_BYTES = 1_031_661
SELECTED_FREEZE_SHA256 = "635cea0c5b974a2dae6bcc680af4bff98ff745bab578b5fb30cdda838e907f7b"
SELECTED_FREEZE_SEAL_SHA256 = "03f56c997d77189dab11c9e4f9e1f6a22b140aebc1e35e1049be34f7523e43db"
REAL_DATASET_MANIFEST_SHA256 = "ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597"
SOURCE_WIDTH = 480
SOURCE_HEIGHT = 360
SOURCE_ENCODING = "rgb8"
REAL_ROI = (0, 80, 480, 360)
TEMPORAL_FRAMES = 3
TEMPORAL_CHANNELS = 9
TEMPORAL_PARAMETER_COUNT = 255_819
MAX_ADJACENT_GAP_NS = 120_000_000
STEERING_SCALE_RAD = 0.35
NEUTRAL_STEERING_NORMALIZED = 0.0
SAFE_SPEED_MPS = 0.0


class RealRuntimeError(RuntimeError):
    """A frozen runtime contract or safe operation was violated."""


class ImageContractError(RealRuntimeError):
    """A camera message does not satisfy the exact real-camera contract."""


class TemporalBufferError(RealRuntimeError):
    """A temporal input cannot be formed causally."""


class InferenceError(RealRuntimeError):
    """The selected ONNX could not produce one finite steering value."""


class PublisherError(RealRuntimeError):
    """A configured control callback failed."""


class SafetyState(str, Enum):
    INIT = "INIT"
    WAITING_FOR_START = "WAITING_FOR_START"
    WARMING_TEMPORAL_BUFFER = "WARMING_TEMPORAL_BUFFER"
    RUNNING = "RUNNING"
    FAULT = "FAULT"
    SAFE_STOPPED = "SAFE_STOPPED"


class BufferStatus(str, Enum):
    WARMING = "WARMING"
    READY = "READY"
    RESET_GAP = "RESET_GAP"
    INVALID_ORDER = "INVALID_ORDER"
    DUPLICATE_FRAME = "DUPLICATE_FRAME"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealRuntimeError(f"cannot read JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise RealRuntimeError(f"JSON root must be an object: {source}")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RealRuntimeError(f"{label} changed: expected {expected!r}, got {actual!r}")


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fixed safety contracts while retaining explicit deploy settings."""
    value = dict(config)
    _require_equal(value.get("version"), VERSION, "runtime version")
    _require_equal(value.get("expected_branch"), "feature/real-runtime-v1", "expected branch")

    model = value.get("model", {})
    _require_equal(model.get("selected_model"), SELECTED_MODEL, "selected model")
    _require_equal(model.get("onnx_sha256"), SELECTED_ONNX_SHA256, "selected ONNX hash")
    _require_equal(model.get("onnx_size_bytes"), SELECTED_ONNX_SIZE_BYTES, "selected ONNX size")
    _require_equal(model.get("freeze_sha256"), SELECTED_FREEZE_SHA256, "freeze hash")
    _require_equal(
        model.get("freeze_seal_sha256"), SELECTED_FREEZE_SEAL_SHA256, "freeze seal hash"
    )
    _require_equal(model.get("checkpoint_sha256"), SELECTED_CHECKPOINT_SHA256, "checkpoint hash")
    _require_equal(
        model.get("checkpoint_size_bytes"), SELECTED_CHECKPOINT_SIZE_BYTES, "checkpoint size"
    )
    for name in ("onnx_path", "freeze_path", "freeze_seal_path"):
        if not isinstance(model.get(name), str) or not model[name]:
            raise RealRuntimeError(f"model.{name} must be a non-empty path")

    _require_equal(
        value.get("model_contract"),
        {
            "input_name": "camera_yuv_temporal",
            "input_shape": [1, 9, 66, 200],
            "input_frames": ["t_minus_2", "t_minus_1", "t"],
            "output_name": "steering_rad",
            "output_shape": [1, 1],
            "output_unit": "radians",
            "parameter_count": TEMPORAL_PARAMETER_COUNT,
            "speed_is_neural_input": False,
        },
        "model contract",
    )
    _require_equal(
        value.get("camera"),
        {
            "topic": "/camera/image_raw",
            "type": "sensor_msgs/msg/Image",
            "width": SOURCE_WIDTH,
            "height": SOURCE_HEIGHT,
            "encoding": SOURCE_ENCODING,
            "nominal_rate_hz": 15.0,
        },
        "real camera contract",
    )
    _require_equal(
        value.get("preprocessing"),
        {
            "roi": [0, 80, 480, 360],
            "crop_width": 480,
            "crop_height": 280,
            "output_width": IMAGE_WIDTH,
            "output_height": IMAGE_HEIGHT,
            "resize": "Pillow_Image.Resampling.BILINEAR",
            "stored_color_space": "RGB",
            "model_color_space": "YUV_BT601_full_range",
            "normalization": "(channel - 0.5) * 2.0",
            "layout": "CHW",
            "horizontal_crop": False,
            "undistortion": False,
            "simulator_roi_permitted": False,
        },
        "preprocessing contract",
    )
    _require_equal(
        value.get("temporal"),
        {
            "frame_count": TEMPORAL_FRAMES,
            "frame_order": ["t_minus_2", "t_minus_1", "t"],
            "timestamp_source_live": "camera_callback_arrival_monotonic",
            "strictly_increasing_timestamps": True,
            "duplicate_frame_padding": False,
            "maximum_adjacent_gap_s": 0.12,
            "reset_if_gap_strictly_greater_than_s": 0.12,
        },
        "temporal contract",
    )

    steering = value.get("steering", {})
    _require_equal(steering.get("topic"), "/steering", "steering topic")
    _require_equal(steering.get("type"), "std_msgs/msg/Float64", "steering type")
    _require_equal(steering.get("model_output_unit"), "radians", "steering output unit")
    _require_equal(steering.get("physical_bound_rad"), [-0.35, 0.35], "steering bounds")
    _require_equal(steering.get("radians_per_normalized_command"), 0.35, "steering scale")
    _require_equal(steering.get("normalized_bound"), [-1.0, 1.0], "normalized bounds")
    _require_equal(steering.get("positive_direction"), "LEFT", "positive steering direction")
    _require_equal(steering.get("negative_direction"), "RIGHT", "negative steering direction")
    _require_equal(steering.get("safe_stop_normalized"), 0.0, "safe steering")

    speed = value.get("speed", {})
    _require_equal(speed.get("topic"), "/speed", "speed topic")
    _require_equal(speed.get("type"), "std_msgs/msg/Float64", "speed type")
    _require_equal(speed.get("unit"), "m/s", "speed unit")
    _require_equal(speed.get("neural_input"), False, "speed neural-input flag")
    _require_equal(speed.get("safe_stop_mps"), 0.0, "safe speed")
    commanded_speed = speed.get("commanded_speed_mps")
    if not isinstance(commanded_speed, (int, float)) or not math.isfinite(float(commanded_speed)):
        raise RealRuntimeError("speed.commanded_speed_mps must be finite")
    if float(commanded_speed) < 0.0:
        raise RealRuntimeError("reverse speed is outside Runtime V1")
    if not isinstance(speed.get("physical_motion_authorized"), bool):
        raise RealRuntimeError("speed.physical_motion_authorized must be boolean")

    gate = value.get("start_gate", {})
    for name in ("required", "development_bypass", "allow_development_bypass"):
        if not isinstance(gate.get(name), bool):
            raise RealRuntimeError(f"start_gate.{name} must be boolean")
    if gate.get("adapter") is not None:
        raise RealRuntimeError("unverified traffic-light adapter must remain null in Runtime V1")
    if gate.get("topic") is not None:
        raise RealRuntimeError("Runtime V1 must not invent a traffic-light ROS topic")

    safety = value.get("safety", {})
    if not isinstance(safety.get("publish_control"), bool):
        raise RealRuntimeError("safety.publish_control must be boolean")
    for name in ("camera_timeout_s", "inference_timeout_s", "publisher_liveness_timeout_s"):
        setting = safety.get(name)
        if not isinstance(setting, (int, float)) or not math.isfinite(float(setting)) or setting <= 0:
            raise RealRuntimeError(f"safety.{name} must be positive and finite")
    failures = safety.get("maximum_consecutive_inference_failures")
    if not isinstance(failures, int) or failures < 1:
        raise RealRuntimeError("maximum_consecutive_inference_failures must be a positive integer")
    _require_equal(safety.get("fault_speed_mps"), 0.0, "fault speed")
    _require_equal(safety.get("fault_steering_normalized"), 0.0, "fault steering")
    _require_equal(safety.get("hold_stale_prediction"), False, "stale-prediction policy")

    bags = value.get("offline_validation", {}).get("bags", {})
    _require_equal(set(bags), {"bag_01", "bag_02", "bag_03"}, "offline bag set")
    _require_equal(
        value.get("offline_validation", {}).get("real_dataset_manifest_sha256"),
        REAL_DATASET_MANIFEST_SHA256,
        "REAL_DATASET_V1 manifest hash",
    )
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(_read_json(path))


def audit_selected_model(
    config: Mapping[str, Any], *, require_checkpoint_file: bool = True
) -> dict[str, Any]:
    """Verify the selected freeze, seal, checkpoint evidence, and ONNX graph."""
    config = validate_config(config)
    selected = config["model"]
    onnx_path = Path(selected["onnx_path"])
    freeze_path = Path(selected["freeze_path"])
    seal_path = Path(selected["freeze_seal_path"])
    for path, expected_hash in (
        (onnx_path, SELECTED_ONNX_SHA256),
        (freeze_path, SELECTED_FREEZE_SHA256),
        (seal_path, SELECTED_FREEZE_SEAL_SHA256),
    ):
        if not path.is_file():
            raise RealRuntimeError(f"missing frozen artifact: {path}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise RealRuntimeError(f"frozen artifact hash mismatch for {path}: {observed}")
    if onnx_path.stat().st_size != SELECTED_ONNX_SIZE_BYTES:
        raise RealRuntimeError("selected ONNX size changed")

    freeze = _read_json(freeze_path)
    seal = _read_json(seal_path)
    _require_equal(freeze.get("selected_model"), SELECTED_MODEL, "freeze selected model")
    _require_equal(freeze.get("onnx", {}).get("sha256"), SELECTED_ONNX_SHA256, "freeze ONNX hash")
    _require_equal(
        freeze.get("checkpoint", {}).get("sha256"),
        SELECTED_CHECKPOINT_SHA256,
        "freeze checkpoint hash",
    )
    _require_equal(freeze.get("architecture", {}).get("parameter_count"), 255_819, "freeze parameters")
    _require_equal(freeze.get("architecture", {}).get("input_shape"), ["N", 9, 66, 200], "freeze input")
    _require_equal(freeze.get("architecture", {}).get("output_unit"), "radians", "freeze output")
    _require_equal(freeze.get("training_config", {}).get("speed_is_neural_input"), False, "freeze speed input")
    _require_equal(freeze.get("runtime_status", {}).get("real_vehicle_run_performed"), False, "prior real run")
    _require_equal(seal.get("selected_model"), SELECTED_MODEL, "seal selected model")
    _require_equal(seal.get("onnx_sha256"), SELECTED_ONNX_SHA256, "seal ONNX hash")
    _require_equal(seal.get("checkpoint_sha256"), SELECTED_CHECKPOINT_SHA256, "seal checkpoint hash")
    _require_equal(seal.get("freeze_sha256"), SELECTED_FREEZE_SHA256, "seal freeze hash")
    _require_equal(seal.get("retraining_permitted"), False, "seal retraining permission")

    checkpoint_path = Path(freeze["checkpoint"]["path"])
    checkpoint_observed: str | None = None
    if checkpoint_path.is_file():
        if checkpoint_path.stat().st_size != SELECTED_CHECKPOINT_SIZE_BYTES:
            raise RealRuntimeError("selected checkpoint size changed")
        checkpoint_observed = sha256_file(checkpoint_path)
        if checkpoint_observed != SELECTED_CHECKPOINT_SHA256:
            raise RealRuntimeError("selected checkpoint hash mismatch")
    elif require_checkpoint_file:
        raise RealRuntimeError(f"missing frozen selected checkpoint: {checkpoint_path}")

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - deployment dependency diagnostic
        raise RealRuntimeError("onnx is required for the startup graph audit") from exc
    graph = onnx.load(onnx_path)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RealRuntimeError("selected ONNX must have exactly one input and one output")
    graph_input = graph.graph.input[0]
    graph_output = graph.graph.output[0]
    input_dims = [dimension.dim_value for dimension in graph_input.type.tensor_type.shape.dim[1:]]
    output_dims = [dimension.dim_value for dimension in graph_output.type.tensor_type.shape.dim[1:]]
    parameter_count = sum(math.prod(initializer.dims) for initializer in graph.graph.initializer)
    if graph_input.name != "camera_yuv_temporal" or input_dims != [9, 66, 200]:
        raise RealRuntimeError("selected ONNX input contract changed")
    if graph_output.name != "steering_rad" or output_dims != [1]:
        raise RealRuntimeError("selected ONNX output contract changed")
    if parameter_count != TEMPORAL_PARAMETER_COUNT:
        raise RealRuntimeError(f"selected ONNX has {parameter_count} parameters")

    return {
        "result": "PASS",
        "selected_model": SELECTED_MODEL,
        "offline_integration_candidate_only": True,
        "real_vehicle_success_claimed": False,
        "onnx": {
            "path": str(onnx_path),
            "sha256_expected": SELECTED_ONNX_SHA256,
            "sha256_observed": sha256_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "checker": "PASS",
            "input_name": graph_input.name,
            "input_shape": ["N", 9, 66, 200],
            "output_name": graph_output.name,
            "output_shape": ["N", 1],
            "parameter_count": parameter_count,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_expected": SELECTED_CHECKPOINT_SHA256,
            "sha256_observed": checkpoint_observed,
            "file_verified": checkpoint_observed is not None,
        },
        "freeze": {"path": str(freeze_path), "sha256": SELECTED_FREEZE_SHA256},
        "freeze_seal": {"path": str(seal_path), "sha256": SELECTED_FREEZE_SEAL_SHA256},
    }


def unpack_rgb8_image(message: Any) -> np.ndarray:
    """Validate and unpack a sensor_msgs/Image-like object without ROS."""
    try:
        width = int(message.width)
        height = int(message.height)
        encoding = str(message.encoding)
        step = int(message.step)
        data = bytes(message.data)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ImageContractError(f"invalid image message fields: {exc}") from exc
    if (width, height) != (SOURCE_WIDTH, SOURCE_HEIGHT):
        raise ImageContractError(
            f"camera dimensions {width}x{height}, expected {SOURCE_WIDTH}x{SOURCE_HEIGHT}"
        )
    if encoding != SOURCE_ENCODING:
        raise ImageContractError(f"camera encoding {encoding!r}, expected {SOURCE_ENCODING!r}")
    packed_row_bytes = SOURCE_WIDTH * 3
    if step < packed_row_bytes:
        raise ImageContractError(f"camera step {step} is shorter than {packed_row_bytes}")
    required = step * SOURCE_HEIGHT
    if len(data) < required:
        raise ImageContractError(f"truncated camera payload: {len(data)} bytes, need {required}")
    rows = np.frombuffer(data, dtype=np.uint8, count=required).reshape(SOURCE_HEIGHT, step)
    packed = np.ascontiguousarray(rows[:, :packed_row_bytes])
    return packed.reshape(SOURCE_HEIGHT, SOURCE_WIDTH, 3)


def resize_real_camera_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply only Real Camera ROI V1 and canonical Pillow bilinear resize."""
    value = np.asarray(rgb)
    if value.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3) or value.dtype != np.uint8:
        raise ImageContractError("real camera RGB input must be uint8 360x480x3")
    source = Image.frombytes("RGB", (SOURCE_WIDTH, SOURCE_HEIGHT), value.tobytes())
    cropped = source.crop(REAL_ROI)
    source.close()
    if cropped.size != (480, 280):
        cropped.close()
        raise ImageContractError("Real Camera ROI did not produce 480x280")
    resized = cropped.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    cropped.close()
    result = np.asarray(resized, dtype=np.uint8).copy()
    resized.close()
    if result.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ImageContractError("real camera resize did not produce RGB 200x66")
    return result


def preprocess_camera_message(message: Any) -> np.ndarray:
    """Exact live equivalent of REAL_DATASET_V1 extraction plus training preprocessing."""
    resized_rgb = resize_real_camera_rgb(unpack_rgb8_image(message))
    tensor = preprocess_rgb(resized_rgb)
    if tensor.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH) or tensor.dtype != np.float32:
        raise ImageContractError("camera preprocessing did not produce float32 3x66x200")
    return tensor


@dataclass(frozen=True)
class BufferedFrame:
    arrival_time_ns: int
    frame_id: str
    tensor: np.ndarray


@dataclass(frozen=True)
class BufferUpdate:
    status: BufferStatus
    size: int
    adjacent_gap_s: float | None = None
    reason: str | None = None


class CausalFrameBuffer:
    """Exactly three independent, strictly ordered real-camera acquisitions."""

    def __init__(self, maximum_adjacent_gap_ns: int = MAX_ADJACENT_GAP_NS) -> None:
        if maximum_adjacent_gap_ns != MAX_ADJACENT_GAP_NS:
            raise ValueError("temporal gap threshold must remain exactly 120,000,000 ns")
        self.maximum_adjacent_gap_ns = maximum_adjacent_gap_ns
        self._frames: deque[BufferedFrame] = deque(maxlen=TEMPORAL_FRAMES)

    def clear(self) -> None:
        self._frames.clear()

    @property
    def size(self) -> int:
        return len(self._frames)

    @property
    def ready(self) -> bool:
        return len(self._frames) == TEMPORAL_FRAMES

    @property
    def timestamps_ns(self) -> tuple[int, ...]:
        return tuple(frame.arrival_time_ns for frame in self._frames)

    def append(self, arrival_time_ns: int, frame_id: str, tensor: np.ndarray) -> BufferUpdate:
        value = np.asarray(tensor)
        if value.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH) or value.dtype != np.float32:
            raise TemporalBufferError("buffer frame must be float32 3x66x200")
        timestamp = int(arrival_time_ns)
        if timestamp < 0:
            raise TemporalBufferError("camera arrival timestamp must be non-negative")
        identifier = str(frame_id)
        if any(frame.frame_id == identifier for frame in self._frames):
            self.clear()
            return BufferUpdate(
                BufferStatus.DUPLICATE_FRAME, 0, reason="duplicate frame padding is forbidden"
            )
        if self._frames:
            gap_ns = timestamp - self._frames[-1].arrival_time_ns
            gap_s = gap_ns / 1e9
            if gap_ns <= 0:
                self.clear()
                return BufferUpdate(
                    BufferStatus.INVALID_ORDER,
                    0,
                    adjacent_gap_s=gap_s,
                    reason="camera arrival timestamps must be strictly increasing",
                )
            if gap_ns > self.maximum_adjacent_gap_ns:
                self.clear()
                self._frames.append(
                    BufferedFrame(timestamp, identifier, np.ascontiguousarray(value))
                )
                return BufferUpdate(
                    BufferStatus.RESET_GAP,
                    1,
                    adjacent_gap_s=gap_s,
                    reason="adjacent camera gap exceeded 0.120 s; history reset",
                )
        self._frames.append(BufferedFrame(timestamp, identifier, np.ascontiguousarray(value)))
        return BufferUpdate(
            BufferStatus.READY if self.ready else BufferStatus.WARMING,
            self.size,
            adjacent_gap_s=(
                None
                if len(self._frames) < 2
                else (self._frames[-1].arrival_time_ns - self._frames[-2].arrival_time_ns) / 1e9
            ),
        )

    def tensor(self) -> np.ndarray:
        if not self.ready:
            raise TemporalBufferError("three genuine real camera frames are not available")
        value = np.ascontiguousarray(
            np.concatenate([frame.tensor for frame in self._frames], axis=0), dtype=np.float32
        )
        if value.shape != (TEMPORAL_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise TemporalBufferError(f"invalid temporal tensor shape {value.shape}")
        return value

    def adjacent_gaps_s(self) -> tuple[float, float]:
        if not self.ready:
            raise TemporalBufferError("three genuine real camera frames are not available")
        times = self.timestamps_ns
        return (times[1] - times[0]) / 1e9, (times[2] - times[1]) / 1e9


@dataclass(frozen=True)
class SteeringCommand:
    model_steering_rad: float
    bounded_steering_rad: float
    published_steering_normalized: float
    saturated: bool


def steering_command_from_radians(model_steering_rad: float) -> SteeringCommand:
    """Clamp physical radians first, then divide by 0.35 exactly once."""
    raw = float(model_steering_rad)
    if not math.isfinite(raw):
        raise InferenceError("model steering output is NaN or infinite")
    bounded = max(-STEERING_SCALE_RAD, min(STEERING_SCALE_RAD, raw))
    normalized = bounded / STEERING_SCALE_RAD
    normalized = max(-1.0, min(1.0, normalized))
    return SteeringCommand(raw, bounded, normalized, bounded != raw)


class SteeringModel(Protocol):
    def infer(self, temporal_tensor: np.ndarray) -> float: ...


class SelectedOnnxModel:
    """Hash-pinned CPU ONNX inference for the selected real model only."""

    def __init__(self, config: Mapping[str, Any], *, audit: bool = True) -> None:
        self.config = validate_config(config)
        self.identity = (
            # The checkpoint hash remains sealed evidence, but the inference
            # deploy bundle only needs ONNX + freeze + seal at runtime.
            audit_selected_model(self.config, require_checkpoint_file=False)
            if audit
            else {"result": "NOT_RUN", "selected_model": SELECTED_MODEL}
        )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - deployment dependency diagnostic
            raise RealRuntimeError("onnxruntime is required for selected-model inference") from exc
        path = Path(self.config["model"]["onnx_path"])
        if sha256_file(path) != SELECTED_ONNX_SHA256:
            raise RealRuntimeError("selected ONNX hash mismatch immediately before session creation")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "camera_yuv_temporal":
            raise RealRuntimeError("unexpected ONNX Runtime input")
        if list(inputs[0].shape)[-3:] != [9, 66, 200] or inputs[0].type != "tensor(float)":
            raise RealRuntimeError("unexpected ONNX Runtime input shape/type")
        if len(outputs) != 1 or outputs[0].name != "steering_rad":
            raise RealRuntimeError("unexpected ONNX Runtime output")
        if list(outputs[0].shape)[-1:] != [1] or outputs[0].type != "tensor(float)":
            raise RealRuntimeError("unexpected ONNX Runtime output shape/type")

    def infer(self, temporal_tensor: np.ndarray) -> float:
        value = np.asarray(temporal_tensor)
        if value.shape != (9, 66, 200) or value.dtype != np.float32:
            raise InferenceError("ONNX input must be float32 9x66x200")
        try:
            output = self.session.run(
                ["steering_rad"], {"camera_yuv_temporal": np.expand_dims(value, axis=0)}
            )[0]
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc
        if np.asarray(output).shape != (1, 1):
            raise InferenceError(f"ONNX output shape changed: {np.asarray(output).shape}")
        steering = float(np.asarray(output, dtype=np.float32).reshape(-1)[0])
        if not math.isfinite(steering):
            raise InferenceError("ONNX produced NaN or infinite steering")
        return steering

    def infer_batch(self, temporal_tensors: np.ndarray) -> np.ndarray:
        values = np.asarray(temporal_tensors)
        if values.ndim != 4 or values.shape[1:] != (9, 66, 200) or values.dtype != np.float32:
            raise InferenceError("ONNX batch input must be float32 Nx9x66x200")
        output = self.session.run(["steering_rad"], {"camera_yuv_temporal": values})[0]
        result = np.asarray(output, dtype=np.float32).reshape(-1)
        if result.size != values.shape[0] or not np.all(np.isfinite(result)):
            raise InferenceError("ONNX batch output is invalid")
        return result


def _latency_distribution(values_s: Sequence[float]) -> dict[str, Any]:
    if not values_s:
        return {"count": 0, "min_ms": None, "mean_ms": None, "median_ms": None,
                "p95_ms": None, "max_ms": None}
    values = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {
        "count": int(values.size),
        "min_ms": float(np.min(values)),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


@dataclass
class RuntimeStatistics:
    frames: int = 0
    valid_preprocessed_frames: int = 0
    warmup_frames: int = 0
    predictions: int = 0
    invalid_temporal_buffers: int = 0
    dropouts: int = 0
    invalid_orderings: int = 0
    duplicate_frames: int = 0
    safe_stop_decisions: int = 0
    preprocessing_failures: int = 0
    inference_failures: int = 0
    nonfinite_model_outputs: int = 0
    watchdog_faults: int = 0
    publisher_failures: int = 0
    saturation_count: int = 0
    model_steering_rad: list[float] = field(default_factory=list)
    normalized_steering: list[float] = field(default_factory=list)
    frame_intervals_s: list[float] = field(default_factory=list)
    preprocessing_s: list[float] = field(default_factory=list)
    inference_s: list[float] = field(default_factory=list)
    total_latency_s: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "warmup_frames": self.warmup_frames,
            "predictions": self.predictions,
            "invalid_temporal_buffers": self.invalid_temporal_buffers,
            "dropouts": self.dropouts,
            "invalid_orderings": self.invalid_orderings,
            "duplicate_frames": self.duplicate_frames,
            "safe_stop_decisions": self.safe_stop_decisions,
            "preprocessing_failures": self.preprocessing_failures,
            "inference_failures": self.inference_failures,
            "nan_inf_model_output_count": self.nonfinite_model_outputs,
            "watchdog_faults": self.watchdog_faults,
            "publisher_failures": self.publisher_failures,
            "steering_rad_min": min(self.model_steering_rad) if self.model_steering_rad else None,
            "steering_rad_max": max(self.model_steering_rad) if self.model_steering_rad else None,
            "normalized_command_min": min(self.normalized_steering) if self.normalized_steering else None,
            "normalized_command_max": max(self.normalized_steering) if self.normalized_steering else None,
            "saturation_count": self.saturation_count,
            "frame_interval": _latency_distribution(self.frame_intervals_s),
            "preprocessing_latency": _latency_distribution(self.preprocessing_s),
            "onnx_latency": _latency_distribution(self.inference_s),
            "total_latency": _latency_distribution(self.total_latency_s),
        }


@dataclass(frozen=True)
class RuntimeStep:
    frame_id: str | None
    arrival_time_ns: int | None
    state: SafetyState
    reason: str
    safe_stop: bool
    model_steering_rad: float | None
    bounded_steering_rad: float
    published_steering_normalized: float
    speed_command_mps: float
    saturated: bool
    buffer_size: int
    adjacent_gaps_s: tuple[float, float] | None
    preprocessing_ms: float | None
    onnx_inference_ms: float | None
    total_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["state"] = self.state.value
        return result


class RealRuntimeCore:
    """Pure causal inference and safety state machine; never publishes controls."""

    def __init__(
        self,
        config: Mapping[str, Any],
        model: SteeringModel,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.config = validate_config(config)
        self.model = model
        self.clock_ns = clock_ns
        self.buffer = CausalFrameBuffer()
        self.state = SafetyState.INIT
        self.statistics = RuntimeStatistics()
        self.last_camera_receipt_ns: int | None = None
        self.last_inference_completed_ns: int | None = None
        self.inference_in_progress_since_ns: int | None = None
        self.last_frame_interval_s: float | None = None
        self.last_fault: str | None = None
        self.consecutive_inference_failures = 0
        self.development_bypass_active = False
        self.start_authorized = False
        self.activated_at_ns: int | None = None
        self._recoverable_fault = False
        self._frame_counter = 0
        self._last_preprocessed_tensor: np.ndarray | None = None

    @property
    def last_preprocessed_tensor(self) -> np.ndarray | None:
        return self._last_preprocessed_tensor

    def activate(self, *, development_bypass: bool = False, now_ns: int | None = None) -> None:
        if self.state != SafetyState.INIT:
            raise RealRuntimeError(f"runtime can only activate from INIT, not {self.state.value}")
        gate = self.config["start_gate"]
        if development_bypass:
            if not gate["allow_development_bypass"]:
                raise RealRuntimeError("development start-gate bypass is disabled")
            if self.config["safety"]["publish_control"]:
                raise RealRuntimeError("development start-gate bypass requires publish_control=false")
            self.development_bypass_active = True
            self.start_authorized = True
        elif not gate["required"]:
            self.start_authorized = True
        self.activated_at_ns = int(self.clock_ns() if now_ns is None else now_ns)
        self.state = (
            SafetyState.WARMING_TEMPORAL_BUFFER
            if self.start_authorized
            else SafetyState.WAITING_FOR_START
        )

    def set_green_authorized(self, green: bool) -> RuntimeStep:
        """Start-gate adapter boundary.  No ROS topic is assumed here."""
        now = int(self.clock_ns())
        if bool(green):
            self.start_authorized = True
            self.buffer.clear()
            self.state = SafetyState.WARMING_TEMPORAL_BUFFER
            return self._safe_step("green start authorization received", now_ns=now)
        self.start_authorized = False
        self.buffer.clear()
        self.state = SafetyState.WAITING_FOR_START
        return self._safe_step("start authorization absent or revoked", now_ns=now)

    def stop(self, reason: str = "explicit safe stop") -> RuntimeStep:
        self.buffer.clear()
        self._recoverable_fault = False
        self.last_fault = reason
        self.state = SafetyState.SAFE_STOPPED
        return self._safe_step(reason, now_ns=int(self.clock_ns()))

    def _enter_fault(self, reason: str, *, recoverable: bool, clear_buffer: bool = True) -> None:
        if clear_buffer:
            self.buffer.clear()
        self.last_fault = reason
        self._recoverable_fault = recoverable
        self.state = SafetyState.FAULT

    def _safe_step(
        self,
        reason: str,
        *,
        frame_id: str | None = None,
        now_ns: int | None = None,
        preprocessing_s: float | None = None,
        inference_s: float | None = None,
        started_s: float | None = None,
    ) -> RuntimeStep:
        total_s = 0.0 if started_s is None else max(0.0, time.perf_counter() - started_s)
        if started_s is not None:
            self.statistics.total_latency_s.append(total_s)
        self.statistics.safe_stop_decisions += 1
        return RuntimeStep(
            frame_id=frame_id,
            arrival_time_ns=now_ns,
            state=self.state,
            reason=reason,
            safe_stop=True,
            model_steering_rad=None,
            bounded_steering_rad=0.0,
            published_steering_normalized=NEUTRAL_STEERING_NORMALIZED,
            speed_command_mps=SAFE_SPEED_MPS,
            saturated=False,
            buffer_size=self.buffer.size,
            adjacent_gaps_s=self.buffer.adjacent_gaps_s() if self.buffer.ready else None,
            preprocessing_ms=None if preprocessing_s is None else preprocessing_s * 1000.0,
            onnx_inference_ms=None if inference_s is None else inference_s * 1000.0,
            total_latency_ms=total_s * 1000.0,
        )

    def process_camera(
        self,
        message: Any,
        *,
        arrival_time_ns: int | None = None,
        frame_id: str | int | None = None,
    ) -> RuntimeStep:
        started_s = time.perf_counter()
        if self.state == SafetyState.INIT:
            self.activate(now_ns=arrival_time_ns)
        now_ns = int(self.clock_ns() if arrival_time_ns is None else arrival_time_ns)
        identifier = str(self._frame_counter if frame_id is None else frame_id)
        self._frame_counter += 1
        self.statistics.frames += 1
        previous_receipt = self.last_camera_receipt_ns
        self.last_camera_receipt_ns = now_ns
        if previous_receipt is not None and now_ns > previous_receipt:
            self.last_frame_interval_s = (now_ns - previous_receipt) / 1e9
            self.statistics.frame_intervals_s.append(self.last_frame_interval_s)

        if self.state == SafetyState.SAFE_STOPPED:
            return self._safe_step(
                "runtime is latched SAFE_STOPPED", frame_id=identifier, now_ns=now_ns,
                started_s=started_s,
            )
        if self.state == SafetyState.FAULT:
            if not self._recoverable_fault:
                return self._safe_step(
                    self.last_fault or "non-recoverable fault", frame_id=identifier,
                    now_ns=now_ns, started_s=started_s,
                )
            self.state = SafetyState.WARMING_TEMPORAL_BUFFER
            self._recoverable_fault = False

        preprocess_started = time.perf_counter()
        try:
            tensor = preprocess_camera_message(message)
        except Exception as exc:
            preprocessing_s = time.perf_counter() - preprocess_started
            self.statistics.preprocessing_s.append(preprocessing_s)
            self.statistics.preprocessing_failures += 1
            self._last_preprocessed_tensor = None
            self._enter_fault(f"invalid image/preprocessing: {exc}", recoverable=True)
            return self._safe_step(
                self.last_fault or "preprocessing fault", frame_id=identifier, now_ns=now_ns,
                preprocessing_s=preprocessing_s, started_s=started_s,
            )
        preprocessing_s = time.perf_counter() - preprocess_started
        self.statistics.preprocessing_s.append(preprocessing_s)
        self.statistics.valid_preprocessed_frames += 1
        self._last_preprocessed_tensor = tensor

        if self.state == SafetyState.WAITING_FOR_START or not self.start_authorized:
            self.buffer.clear()
            self.state = SafetyState.WAITING_FOR_START
            return self._safe_step(
                "waiting for verified GREEN start authorization",
                frame_id=identifier,
                now_ns=now_ns,
                preprocessing_s=preprocessing_s,
                started_s=started_s,
            )

        try:
            update = self.buffer.append(now_ns, identifier, tensor)
        except Exception as exc:
            self.statistics.invalid_temporal_buffers += 1
            self._enter_fault(f"temporal buffer exception: {exc}", recoverable=True)
            return self._safe_step(
                self.last_fault or "temporal fault", frame_id=identifier, now_ns=now_ns,
                preprocessing_s=preprocessing_s, started_s=started_s,
            )

        if update.status in (
            BufferStatus.RESET_GAP,
            BufferStatus.INVALID_ORDER,
            BufferStatus.DUPLICATE_FRAME,
        ):
            self.statistics.invalid_temporal_buffers += 1
            self.statistics.warmup_frames += 1
            if update.status == BufferStatus.RESET_GAP:
                self.statistics.dropouts += 1
            elif update.status == BufferStatus.INVALID_ORDER:
                self.statistics.invalid_orderings += 1
            else:
                self.statistics.duplicate_frames += 1
            self._enter_fault(update.reason or update.status.value, recoverable=True, clear_buffer=False)
            return self._safe_step(
                self.last_fault or "temporal fault", frame_id=identifier, now_ns=now_ns,
                preprocessing_s=preprocessing_s, started_s=started_s,
            )
        if update.status == BufferStatus.WARMING:
            self.statistics.warmup_frames += 1
            self.state = SafetyState.WARMING_TEMPORAL_BUFFER
            return self._safe_step(
                "waiting for three genuine causal frames",
                frame_id=identifier,
                now_ns=now_ns,
                preprocessing_s=preprocessing_s,
                started_s=started_s,
            )

        inference_started = time.perf_counter()
        self.inference_in_progress_since_ns = int(self.clock_ns())
        try:
            model_steering = float(self.model.infer(self.buffer.tensor()))
            command = steering_command_from_radians(model_steering)
        except Exception as exc:
            inference_s = time.perf_counter() - inference_started
            self.statistics.inference_s.append(inference_s)
            self.statistics.inference_failures += 1
            self.consecutive_inference_failures += 1
            if "NaN" in str(exc) or "infinite" in str(exc) or "non-finite" in str(exc):
                self.statistics.nonfinite_model_outputs += 1
            threshold = int(self.config["safety"]["maximum_consecutive_inference_failures"])
            recoverable = self.consecutive_inference_failures < threshold
            self._enter_fault(f"inference fault: {exc}", recoverable=recoverable)
            self.inference_in_progress_since_ns = None
            return self._safe_step(
                self.last_fault or "inference fault", frame_id=identifier, now_ns=now_ns,
                preprocessing_s=preprocessing_s, inference_s=inference_s, started_s=started_s,
            )
        inference_s = time.perf_counter() - inference_started
        self.inference_in_progress_since_ns = None
        self.last_inference_completed_ns = int(self.clock_ns())
        self.statistics.inference_s.append(inference_s)
        self.consecutive_inference_failures = 0
        self.statistics.predictions += 1
        self.statistics.model_steering_rad.append(command.model_steering_rad)
        self.statistics.normalized_steering.append(command.published_steering_normalized)
        self.statistics.saturation_count += int(command.saturated)
        self.state = SafetyState.RUNNING
        speed = (
            float(self.config["speed"]["commanded_speed_mps"])
            if self.config["speed"]["physical_motion_authorized"]
            else SAFE_SPEED_MPS
        )
        total_s = time.perf_counter() - started_s
        self.statistics.total_latency_s.append(total_s)
        return RuntimeStep(
            frame_id=identifier,
            arrival_time_ns=now_ns,
            state=self.state,
            reason="fresh temporal inference",
            safe_stop=False,
            model_steering_rad=command.model_steering_rad,
            bounded_steering_rad=command.bounded_steering_rad,
            published_steering_normalized=command.published_steering_normalized,
            speed_command_mps=speed,
            saturated=command.saturated,
            buffer_size=self.buffer.size,
            adjacent_gaps_s=self.buffer.adjacent_gaps_s(),
            preprocessing_ms=preprocessing_s * 1000.0,
            onnx_inference_ms=inference_s * 1000.0,
            total_latency_ms=total_s * 1000.0,
        )

    def check_watchdog(self, *, now_ns: int | None = None) -> RuntimeStep | None:
        now = int(self.clock_ns() if now_ns is None else now_ns)
        if self.state in (
            SafetyState.INIT,
            SafetyState.WAITING_FOR_START,
            SafetyState.SAFE_STOPPED,
        ):
            return None
        camera_reference = self.last_camera_receipt_ns or self.activated_at_ns
        camera_timeout_ns = int(float(self.config["safety"]["camera_timeout_s"]) * 1e9)
        if camera_reference is not None and now - camera_reference > camera_timeout_ns:
            self.statistics.watchdog_faults += 1
            self.statistics.invalid_temporal_buffers += int(self.buffer.size > 0)
            self._enter_fault("camera watchdog timeout", recoverable=True)
            return self._safe_step("camera watchdog timeout", now_ns=now)
        inference_timeout_ns = int(float(self.config["safety"]["inference_timeout_s"]) * 1e9)
        if (
            self.inference_in_progress_since_ns is not None
            and now - self.inference_in_progress_since_ns > inference_timeout_ns
        ):
            self.statistics.watchdog_faults += 1
            self._enter_fault("inference watchdog timeout", recoverable=False)
            return self._safe_step("inference watchdog timeout", now_ns=now)
        return None

    def mark_publisher_failure(self, reason: str) -> RuntimeStep:
        self.statistics.publisher_failures += 1
        self._enter_fault(f"publisher/liveness failure: {reason}", recoverable=False)
        return self._safe_step(self.last_fault or "publisher fault", now_ns=int(self.clock_ns()))


@dataclass(frozen=True)
class DispatchResult:
    published: bool
    steering_messages: int
    speed_messages: int
    last_publish_ns: int | None


class ControlDispatcher:
    """Pure publisher contract; callback invocation requires explicit opt-in."""

    def __init__(
        self,
        *,
        publish_control: bool,
        steering_publish: Callable[[float], None],
        speed_publish: Callable[[float], None],
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.publish_control = bool(publish_control)
        self.steering_publish = steering_publish
        self.speed_publish = speed_publish
        self.clock_ns = clock_ns
        self.steering_messages = 0
        self.speed_messages = 0
        self.last_publish_ns: int | None = None

    def dispatch(self, step: RuntimeStep) -> DispatchResult:
        if not self.publish_control:
            return DispatchResult(False, self.steering_messages, self.speed_messages, None)
        try:
            if step.safe_stop:
                self.speed_publish(SAFE_SPEED_MPS)
                self.speed_messages += 1
                self.steering_publish(NEUTRAL_STEERING_NORMALIZED)
                self.steering_messages += 1
            else:
                self.steering_publish(step.published_steering_normalized)
                self.steering_messages += 1
                self.speed_publish(step.speed_command_mps)
                self.speed_messages += 1
        except Exception as exc:
            # Best effort only: the runtime must separately enter FAULT because
            # callback failure means the safe command cannot be guaranteed.
            try:
                self.speed_publish(SAFE_SPEED_MPS)
                self.steering_publish(NEUTRAL_STEERING_NORMALIZED)
            except Exception:
                pass
            raise PublisherError(f"control publication failed: {exc}") from exc
        self.last_publish_ns = int(self.clock_ns())
        return DispatchResult(
            True, self.steering_messages, self.speed_messages, self.last_publish_ns
        )

    def liveness_ok(self, now_ns: int, maximum_age_s: float) -> bool:
        if not self.publish_control:
            return True
        return (
            self.last_publish_ns is not None
            and int(now_ns) - self.last_publish_ns <= int(float(maximum_age_s) * 1e9)
        )


def _iter_camera_messages(mcap_path: Path, topic: str, expected_type: str):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=[topic], log_time_order=True
        ):
            actual = None if schema is None else str(schema.name)
            if actual != expected_type:
                raise RealRuntimeError(
                    f"{mcap_path}: {topic} type {actual!r}, expected {expected_type!r}"
                )
            yield int(message.log_time), decoded


def _manifest_rows(config: Mapping[str, Any], bag_id: str) -> list[dict[str, str]]:
    validation = config["offline_validation"]
    manifest = Path(validation["real_dataset_manifest_path"])
    if sha256_file(manifest) != REAL_DATASET_MANIFEST_SHA256:
        raise RealRuntimeError("REAL_DATASET_V1 manifest hash mismatch")
    with manifest.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream) if row["source_bag"] == bag_id]


def _canonical_temporal_inputs(
    config: Mapping[str, Any], rows: Sequence[dict[str, str]]
) -> Iterable[np.ndarray]:
    root = Path(config["offline_validation"]["real_dataset_root"])
    for row in rows:
        paths = [root / row[name] for name in ("image_t_minus_2", "image_t_minus_1", "image_t")]
        value = np.ascontiguousarray(
            np.concatenate([preprocess_png(path) for path in paths], axis=0), dtype=np.float32
        )
        if value.shape != (9, 66, 200):
            raise RealRuntimeError("canonical temporal input shape changed")
        yield value


def _canonical_onnx_predictions(
    model: SelectedOnnxModel,
    config: Mapping[str, Any],
    rows: Sequence[dict[str, str]],
    batch_size: int = 64,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    for tensor in _canonical_temporal_inputs(config, rows):
        batch.append(tensor)
        if len(batch) == batch_size:
            predictions.append(model.infer_batch(np.stack(batch)))
            batch.clear()
    if batch:
        predictions.append(model.infer_batch(np.stack(batch)))
    return np.concatenate(predictions).astype(np.float64) if predictions else np.empty(0)


def _checkpoint_predictions(
    config: Mapping[str, Any], rows: Sequence[dict[str, str]], batch_size: int = 64
) -> np.ndarray:
    """Re-run the frozen training evaluator without training or modifying data."""
    import torch

    from .pilotnet_temporal import build_temporal_pilotnet

    freeze = _read_json(config["model"]["freeze_path"])
    checkpoint_path = Path(freeze["checkpoint"]["path"])
    if sha256_file(checkpoint_path) != SELECTED_CHECKPOINT_SHA256:
        raise RealRuntimeError("checkpoint changed before frozen evaluator comparison")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != SELECTED_MODEL:
        raise RealRuntimeError("checkpoint is not REAL-SCRATCH-V1")
    network = build_temporal_pilotnet().eval()
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    values: list[np.ndarray] = []
    batch: list[np.ndarray] = []
    with torch.no_grad():
        for tensor in _canonical_temporal_inputs(config, rows):
            batch.append(tensor)
            if len(batch) == batch_size:
                values.append(network(torch.from_numpy(np.stack(batch))).numpy().reshape(-1))
                batch.clear()
        if batch:
            values.append(network(torch.from_numpy(np.stack(batch))).numpy().reshape(-1))
    return np.concatenate(values).astype(np.float64) if values else np.empty(0)


def _numeric_metric_differences(actual: Any, expected: Any, prefix: str = "") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(expected, dict):
        for key, expected_value in expected.items():
            if key in actual:
                differences.extend(
                    _numeric_metric_differences(actual[key], expected_value, f"{prefix}.{key}".strip("."))
                )
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        actual_value = actual
        difference = abs(float(actual_value) - float(expected))
        differences.append(
            {"metric": prefix, "actual": float(actual_value), "expected": float(expected),
             "absolute_difference": difference}
        )
    elif expected is None and actual is not None:
        differences.append(
            {"metric": prefix, "actual": actual, "expected": None,
             "absolute_difference": math.inf}
        )
    return differences


def replay_bag(
    config: Mapping[str, Any],
    model: SelectedOnnxModel,
    bag_id: str,
    *,
    realtime: bool = False,
    playback_rate: float = 1.0,
    verify_checkpoint_pipeline: bool = False,
) -> dict[str, Any]:
    """Replay one complete canonical camera stream without any publisher callbacks."""
    config = validate_config(config)
    if playback_rate <= 0 or not math.isfinite(playback_rate):
        raise RealRuntimeError("playback_rate must be positive and finite")
    bag_spec = config["offline_validation"]["bags"].get(bag_id)
    if not isinstance(bag_spec, dict):
        raise RealRuntimeError(f"unknown canonical bag {bag_id!r}")
    mcap_path = Path(bag_spec["mcap_path"])
    if not mcap_path.is_file():
        raise RealRuntimeError(f"missing canonical bag: {mcap_path}")
    observed_bag_hash = sha256_file(mcap_path)
    if observed_bag_hash != bag_spec["mcap_sha256"]:
        raise RealRuntimeError(f"canonical bag hash mismatch for {bag_id}")

    rows = _manifest_rows(config, bag_id)
    expected_count = int(bag_spec["accepted_sequence_count"])
    if len(rows) != expected_count:
        raise RealRuntimeError(f"{bag_id} manifest sequence count changed")
    expected_by_index = {int(row["target_camera_index"]): row for row in rows}
    dataset_root = Path(config["offline_validation"]["real_dataset_root"])

    core = RealRuntimeCore(config, model)
    predictions: list[float] = []
    prediction_indices: list[int] = []
    normalized: list[float] = []
    preprocessing_exact_count = 0
    preprocessing_mismatch_count = 0
    preprocessing_max_abs = 0.0
    source_first_ns: int | None = None
    source_last_ns: int | None = None
    wall_started = time.perf_counter()
    pacing_origin_wall: float | None = None
    realtime_watchdog_safe_stops = 0

    camera_topic = config["camera"]
    for index, (log_time_ns, message) in enumerate(
        _iter_camera_messages(mcap_path, camera_topic["topic"], camera_topic["type"])
    ):
        if source_first_ns is None:
            source_first_ns = log_time_ns
            pacing_origin_wall = time.perf_counter()
            core.activate(development_bypass=True, now_ns=log_time_ns)
        if realtime:
            assert pacing_origin_wall is not None and source_first_ns is not None
            if core.last_camera_receipt_ns is not None:
                camera_timeout_ns = int(float(config["safety"]["camera_timeout_s"]) * 1e9)
                watchdog_time_ns = core.last_camera_receipt_ns + camera_timeout_ns + 1
                if watchdog_time_ns < log_time_ns:
                    watchdog_wall = (
                        pacing_origin_wall
                        + (watchdog_time_ns - source_first_ns) / 1e9 / playback_rate
                    )
                    watchdog_remaining = watchdog_wall - time.perf_counter()
                    if watchdog_remaining > 0:
                        time.sleep(watchdog_remaining)
                    if core.check_watchdog(now_ns=watchdog_time_ns) is not None:
                        realtime_watchdog_safe_stops += 1
            target_wall = pacing_origin_wall + (log_time_ns - source_first_ns) / 1e9 / playback_rate
            remaining = target_wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        source_last_ns = log_time_ns
        step = core.process_camera(message, arrival_time_ns=log_time_ns, frame_id=index)

        runtime_frame = core.last_preprocessed_tensor
        expected_frame_path = dataset_root / "images" / bag_id / f"frame_{index:06d}.png"
        if runtime_frame is not None:
            expected_frame = preprocess_png(expected_frame_path)
            if np.array_equal(runtime_frame, expected_frame):
                preprocessing_exact_count += 1
            else:
                preprocessing_mismatch_count += 1
                preprocessing_max_abs = max(
                    preprocessing_max_abs,
                    float(np.max(np.abs(runtime_frame.astype(np.float64) - expected_frame))),
                )
        if step.model_steering_rad is not None:
            predictions.append(step.model_steering_rad)
            prediction_indices.append(index)
            normalized.append(step.published_steering_normalized)

    replay_wall_elapsed = time.perf_counter() - wall_started
    if core.statistics.frames != int(bag_spec["camera_frame_count"]):
        raise RealRuntimeError(f"{bag_id} camera frame count changed")
    expected_indices = list(expected_by_index)
    temporal_order_result = "PASS" if prediction_indices == expected_indices else "FAIL"
    runtime_predictions = np.asarray(predictions, dtype=np.float64)
    canonical_onnx = _canonical_onnx_predictions(model, config, rows)
    if runtime_predictions.shape != canonical_onnx.shape:
        raise RealRuntimeError(f"{bag_id} runtime/canonical prediction count mismatch")
    onnx_difference = np.abs(runtime_predictions - canonical_onnx)
    prediction_equivalence = {
        "result": "PASS" if (not onnx_difference.size or float(np.max(onnx_difference)) <= 1e-7) else "FAIL",
        "count": int(runtime_predictions.size),
        "runtime_raw_camera_vs_stored_rgb_onnx_mean_absolute_difference_rad": (
            float(np.mean(onnx_difference)) if onnx_difference.size else 0.0
        ),
        "runtime_raw_camera_vs_stored_rgb_onnx_max_absolute_difference_rad": (
            float(np.max(onnx_difference)) if onnx_difference.size else 0.0
        ),
        "tolerance_rad": 1e-7,
    }

    checkpoint_equivalence: dict[str, Any] | None = None
    validation_metrics: dict[str, Any] | None = None
    if verify_checkpoint_pipeline:
        checkpoint_values = _checkpoint_predictions(config, rows)
        checkpoint_difference = np.abs(runtime_predictions - checkpoint_values)
        from .real_temporal_pilotnet import error_metrics, magnitude_bin_metrics

        labels = np.asarray([np.float32(float(row["steering_rad"])) for row in rows], dtype=np.float64)
        validation_metrics = {
            "overall": error_metrics(runtime_predictions, labels),
            "by_target_magnitude_and_direction": magnitude_bin_metrics(runtime_predictions, labels),
        }
        freeze = _read_json(config["model"]["freeze_path"])
        expected_metrics = freeze["validation_metrics"]
        differences = _numeric_metric_differences(validation_metrics, expected_metrics)
        finite_differences = [item["absolute_difference"] for item in differences]
        metric_max_difference = max(finite_differences, default=0.0)
        overall_differences = [
            item["absolute_difference"]
            for item in differences
            if item["metric"].startswith("overall.")
        ]
        overall_metric_max_difference = max(overall_differences, default=0.0)
        checkpoint_equivalence = {
            "result": "PASS" if (
                (not checkpoint_difference.size or float(np.max(checkpoint_difference)) <= 1e-4)
                and overall_metric_max_difference <= 1e-5
                # Directional bins as small as five samples make Pearson
                # correlation sensitive to sub-1e-7 ONNX/PyTorch output drift.
                and metric_max_difference <= 2e-4
            ) else "FAIL",
            "runtime_onnx_vs_frozen_checkpoint_mean_absolute_difference_rad": (
                float(np.mean(checkpoint_difference)) if checkpoint_difference.size else 0.0
            ),
            "runtime_onnx_vs_frozen_checkpoint_max_absolute_difference_rad": (
                float(np.max(checkpoint_difference)) if checkpoint_difference.size else 0.0
            ),
            "prediction_tolerance_rad": 1e-4,
            "freeze_overall_metric_max_absolute_difference": overall_metric_max_difference,
            "overall_metric_tolerance": 1e-5,
            "freeze_metric_max_absolute_difference": metric_max_difference,
            "all_metric_tolerance": 2e-4,
        }

    stats = core.statistics.summary()
    normalized_in_bounds = all(-1.0 <= value <= 1.0 for value in normalized)
    source_gap_count = sum(
        gap > float(config["temporal"]["maximum_adjacent_gap_s"])
        for gap in core.statistics.frame_intervals_s
    )
    source_duration_s = (
        0.0
        if source_first_ns is None or source_last_ns is None
        else (source_last_ns - source_first_ns) / 1e9
    )
    result = (
        preprocessing_mismatch_count == 0
        and temporal_order_result == "PASS"
        and prediction_equivalence["result"] == "PASS"
        and stats["predictions"] == expected_count
        and stats["nan_inf_model_output_count"] == 0
        and normalized_in_bounds
        and (checkpoint_equivalence is None or checkpoint_equivalence["result"] == "PASS")
    )
    wall_elapsed = time.perf_counter() - wall_started
    return {
        "version": VERSION,
        "result": "PASS" if result else "FAIL",
        "bag_id": bag_id,
        "mode": "recorded_timing_no_publish_dry_run" if realtime else "offline_no_publish_replay",
        "source": {
            "mcap_path": str(mcap_path),
            "mcap_sha256": observed_bag_hash,
            "camera_frame_count": core.statistics.frames,
            "recorded_camera_duration_s": source_duration_s,
        },
        "safety": {
            "publish_control": False,
            "steering_publish_count": 0,
            "speed_publish_count": 0,
            "physical_motion_authorized": bool(config["speed"]["physical_motion_authorized"]),
            "development_start_gate_bypass": True,
            "real_vehicle_motion_performed": False,
            "simulator_motion_performed": False,
        },
        "temporal": {
            "result": temporal_order_result,
            "expected_predictions": expected_count,
            "observed_predictions": len(predictions),
            "target_camera_indices_match_manifest": prediction_indices == expected_indices,
            "source_adjacent_gap_gt_0p120_s_count": source_gap_count,
            "realtime_watchdog_safe_stop_count": realtime_watchdog_safe_stops,
        },
        "preprocessing_equivalence": {
            "result": "PASS" if preprocessing_mismatch_count == 0 else "FAIL",
            "frames_compared": preprocessing_exact_count + preprocessing_mismatch_count,
            "exact_tensor_matches": preprocessing_exact_count,
            "mismatch_count": preprocessing_mismatch_count,
            "max_absolute_tensor_difference": preprocessing_max_abs,
        },
        "prediction_equivalence": prediction_equivalence,
        "checkpoint_pipeline_equivalence": checkpoint_equivalence,
        "validation_metrics": validation_metrics,
        "statistics": stats,
        "normalized_commands_within_bounds": normalized_in_bounds,
        "wall_elapsed_s": wall_elapsed,
        "camera_replay_wall_elapsed_s": replay_wall_elapsed,
        "playback_rate": playback_rate if realtime else None,
        "realtime_pacing_ratio": (
            replay_wall_elapsed / (source_duration_s / playback_rate)
            if realtime and source_duration_s > 0
            else None
        ),
    }


def validate_all_bags(config: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config(config)
    audit = audit_selected_model(config)
    model = SelectedOnnxModel(config, audit=False)
    bags = {
        bag_id: replay_bag(
            config, model, bag_id, verify_checkpoint_pipeline=(bag_id == "bag_03")
        )
        for bag_id in ("bag_01", "bag_02", "bag_03")
    }
    passed = audit["result"] == "PASS" and all(item["result"] == "PASS" for item in bags.values())
    return {
        "version": VERSION,
        "result": "PASS" if passed else "FAIL",
        "selected_model_audit": audit,
        "bags": bags,
        "publication": {
            "publish_control": False,
            "steering_publish_count": 0,
            "speed_publish_count": 0,
        },
        "real_vehicle_motion_performed": False,
        "simulator_motion_performed": False,
        "training_performed": False,
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="audit the frozen selected model")
    audit.add_argument("--output", type=Path)
    validate = commands.add_parser("validate", help="offline replay all canonical bags")
    validate.add_argument("--output", type=Path)
    replay = commands.add_parser("replay", help="offline replay one canonical bag")
    replay.add_argument("--bag", required=True, choices=("bag_01", "bag_02", "bag_03"))
    replay.add_argument("--realtime", action="store_true")
    replay.add_argument("--playback-rate", type=float, default=1.0)
    replay.add_argument("--verify-checkpoint-pipeline", action="store_true")
    replay.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
        if arguments.command == "audit":
            result = audit_selected_model(config)
        elif arguments.command == "validate":
            result = validate_all_bags(config)
        else:
            model = SelectedOnnxModel(config)
            result = replay_bag(
                config,
                model,
                arguments.bag,
                realtime=arguments.realtime,
                playback_rate=arguments.playback_rate,
                verify_checkpoint_pipeline=arguments.verify_checkpoint_pipeline,
            )
        if arguments.output:
            write_json(arguments.output, result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result.get("result") == "PASS" else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
