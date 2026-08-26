"""Real Dataset V1: strict-causal three-frame extraction from approved real bags.

This module only derives an RGB image dataset and compact QC evidence.  It does
not invoke training, a simulator, Docker, or any process that can modify a bag.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw


VERSION = "real_dataset_v1"
SPEED_SEMANTICS = "UNKNOWN_COMMAND_OR_FEEDBACK"
STEERING_SCALE_RAD = 0.35
MAXIMUM_AGE_OR_GAP_S = 0.120
NEURAL_INPUT_FIELDS = ("image_t_minus_2", "image_t_minus_1", "image_t")
TRAINING_INVOCATION_PERMITTED = False

REJECTION_REASON_ORDER = (
    "non_increasing_camera_timestamps",
    "adjacent_camera_gap_gt_0p120_s",
    "no_causal_steering",
    "future_steering_label",
    "steering_age_gt_0p120_s",
)

MANIFEST_COLUMNS = (
    "sequence_id",
    "source_bag",
    "source_mcap_sha256",
    "target_camera_index",
    "target_camera_log_time_ns",
    "image_t_minus_2",
    "image_t_minus_1",
    "image_t",
    "camera_t_minus_2_log_time_ns",
    "camera_t_minus_1_log_time_ns",
    "camera_t_log_time_ns",
    "adjacent_gap_t_minus_2_to_t_minus_1_s",
    "adjacent_gap_t_minus_1_to_t_s",
    "oldest_to_current_span_s",
    "steering_recorded_raw",
    "steering_rad",
    "steering_direction",
    "steering_log_time_ns",
    "steering_age_s",
    "speed_mps",
    "speed_log_time_ns",
    "speed_age_s",
    "speed_available",
    "speed_valid",
    "speed_stale",
    "speed_state",
    "speed_semantics",
)

IMAGE_INVENTORY_COLUMNS = (
    "source_bag",
    "camera_index",
    "camera_log_time_ns",
    "image_path",
    "image_sha256",
    "image_size_bytes",
    "width",
    "height",
    "color_space",
)


class RealDatasetError(RuntimeError):
    """Raised when an extraction or QC contract cannot be satisfied."""


@dataclass(frozen=True)
class ScalarRecord:
    time_ns: int
    value: float
    index: int = 0


@dataclass(frozen=True)
class CameraFrame:
    bag_id: str
    index: int
    time_ns: int
    image_path: str
    image_sha256: str
    image_size_bytes: int


@dataclass(frozen=True)
class SteeringTarget:
    recorded_raw: float
    radians: float
    direction: str


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise RealDatasetError(f"{name} changed: expected {expected!r}, got {actual!r}")


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    _require_equal(config.get("version"), VERSION, "version")

    topics = config.get("topics", {})
    _require_equal(
        topics,
        {
            "camera": {"name": "/camera/image_raw", "type": "sensor_msgs/msg/Image"},
            "steering": {"name": "/steering", "type": "std_msgs/msg/Float64"},
            "speed": {"name": "/speed", "type": "std_msgs/msg/Float64"},
        },
        "topic contract",
    )

    timing = config.get("timestamp_contract", {})
    _require_equal(timing.get("domain"), "MCAP_log_time", "timestamp domain")
    _require_equal(
        timing.get("causal_rule"),
        "latest_scalar_with_log_time_lte_camera_log_time",
        "causal rule",
    )
    _require_equal(timing.get("future_steering_labels_permitted"), False, "future labels")
    _require_equal(timing.get("maximum_steering_age_s"), MAXIMUM_AGE_OR_GAP_S, "steering age")
    _require_equal(
        timing.get("reject_steering_age_strictly_greater_than_s"),
        MAXIMUM_AGE_OR_GAP_S,
        "steering rejection threshold",
    )

    steering = config.get("steering_contract", {})
    _require_equal(steering.get("recorded_representation"), "normalized_steering_command", "steering representation")
    _require_equal(steering.get("recorded_to_radians_scale"), STEERING_SCALE_RAD, "steering scale")
    _require_equal(steering.get("physical_target_unit"), "radians", "steering unit")
    _require_equal(steering.get("positive_direction"), "LEFT", "positive steering direction")
    _require_equal(steering.get("negative_direction"), "RIGHT", "negative steering direction")
    _require_equal(steering.get("selective_clipping"), False, "steering clipping")
    _require_equal(steering.get("expected_physical_range_rad"), [-0.35, 0.35], "steering range")

    speed = config.get("speed_contract", {})
    _require_equal(speed.get("unit"), "m/s", "speed unit")
    _require_equal(speed.get("semantics"), SPEED_SEMANTICS, "speed semantics")
    _require_equal(speed.get("maximum_age_s"), MAXIMUM_AGE_OR_GAP_S, "speed age")
    _require_equal(speed.get("stale_if_strictly_greater_than_s"), MAXIMUM_AGE_OR_GAP_S, "speed stale threshold")
    _require_equal(speed.get("reject_sequence_if_missing_or_stale"), False, "speed rejection policy")
    _require_equal(speed.get("active_driving_filter"), False, "speed active-driving filter")
    _require_equal(speed.get("neural_input"), False, "speed neural-input policy")
    _require_equal(speed.get("steering_target"), False, "speed target policy")

    camera = config.get("camera_contract", {})
    expected_camera = {
        "source_width": 480,
        "source_height": 360,
        "source_encoding": "rgb8",
        "roi": {
            "x_start": 0,
            "x_end": 480,
            "y_start": 80,
            "y_end": 360,
            "end_coordinates_exclusive": True,
        },
        "cropped_width": 480,
        "cropped_height": 280,
        "output_width": 200,
        "output_height": 66,
        "resize": "Pillow_Image.Resampling.BILINEAR",
        "stored_color_space": "RGB",
        "model_preprocessing_later": "RGB_to_YUV",
        "horizontal_crop": False,
        "undistortion": False,
        "simulator_y_160_360_crop": False,
    }
    _require_equal(camera, expected_camera, "Real Camera ROI V1")

    temporal = config.get("temporal_contract", {})
    _require_equal(temporal.get("history_frames"), 3, "history frames")
    _require_equal(temporal.get("frame_order"), ["t_minus_2", "t_minus_1", "t"], "frame order")
    _require_equal(temporal.get("target_frame"), "t", "target frame")
    _require_equal(temporal.get("maximum_adjacent_gap_s"), MAXIMUM_AGE_OR_GAP_S, "camera gap")
    _require_equal(
        temporal.get("reject_gap_strictly_greater_than_s"),
        MAXIMUM_AGE_OR_GAP_S,
        "camera gap rejection threshold",
    )
    _require_equal(temporal.get("allow_episode_crossing"), False, "episode crossing")
    _require_equal(temporal.get("allow_duplicate_frame_padding"), False, "duplicate padding")

    guards = config.get("scope_guards", {})
    for key in (
        "training_permitted",
        "raw_bag_copy_permitted",
        "raw_bag_modification_permitted",
        "simulator_permitted",
        "docker_permitted",
    ):
        _require_equal(guards.get(key), False, key)

    bags = config.get("source_bags", [])
    _require_equal([item.get("bag_id") for item in bags], ["bag_01", "bag_02", "bag_03"], "source bag order")
    for item in bags:
        if len(str(item.get("expected_mcap_sha256", ""))) != 64:
            raise RealDatasetError(f"{item.get('bag_id')}: invalid expected SHA-256")
        if int(item.get("expected_mcap_size_bytes", 0)) <= 0:
            raise RealDatasetError(f"{item.get('bag_id')}: invalid expected MCAP size")
    return config


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": len(values), "finite_count": 0, "min": None, "p01": None,
            "p05": None, "p25": None, "mean": None, "median": None, "std": None,
            "p75": None, "p95": None, "p99": None, "max": None,
        }
    return {
        "count": len(values),
        "finite_count": len(finite),
        "min": min(finite),
        "p01": percentile(finite, 0.01),
        "p05": percentile(finite, 0.05),
        "p25": percentile(finite, 0.25),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite),
        "p75": percentile(finite, 0.75),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "max": max(finite),
    }


def latest_causal(records: Sequence[ScalarRecord], target_time_ns: int) -> ScalarRecord | None:
    """Return the last scalar record whose MCAP log_time is not in the future."""
    index = bisect.bisect_right(records, int(target_time_ns), key=lambda item: item.time_ns) - 1
    if index < 0:
        return None
    selected = records[index]
    if selected.time_ns > target_time_ns:
        raise RealDatasetError("causal ZOH selected a future scalar")
    return selected


def steering_recorded_to_radians(recorded: float, scale: float = STEERING_SCALE_RAD) -> float:
    """Apply the approved normalized-command conversion exactly once, without clipping."""
    if scale != STEERING_SCALE_RAD:
        raise RealDatasetError(f"steering scale must remain exactly {STEERING_SCALE_RAD}")
    value = float(recorded)
    if not math.isfinite(value):
        raise RealDatasetError("non-finite steering command")
    return value * scale


def build_steering_target(recorded: float) -> SteeringTarget:
    radians = steering_recorded_to_radians(recorded)
    direction = "LEFT" if radians > 0 else "RIGHT" if radians < 0 else "ZERO"
    return SteeringTarget(float(recorded), radians, direction)


def speed_metadata(
    target_time_ns: int,
    records: Sequence[ScalarRecord],
    maximum_age_s: float = MAXIMUM_AGE_OR_GAP_S,
) -> dict[str, Any]:
    selected = latest_causal(records, target_time_ns)
    if selected is None:
        return {
            "speed_mps": None,
            "speed_log_time_ns": None,
            "speed_age_s": None,
            "speed_available": False,
            "speed_valid": False,
            "speed_stale": False,
            "speed_state": "MISSING",
            "speed_semantics": SPEED_SEMANTICS,
        }
    age_s = (target_time_ns - selected.time_ns) / 1e9
    if age_s < 0:
        raise RealDatasetError("future speed metadata selected")
    stale = age_s > maximum_age_s
    return {
        "speed_mps": selected.value,
        "speed_log_time_ns": selected.time_ns,
        "speed_age_s": age_s,
        "speed_available": True,
        "speed_valid": not stale,
        "speed_stale": stale,
        "speed_state": "STALE" if stale else "VALID",
        "speed_semantics": SPEED_SEMANTICS,
    }


def preprocess_real_camera_image(message: Any, config: dict[str, Any]) -> Image.Image:
    """Decode exact 480x360 rgb8, apply Real Camera ROI V1, and store RGB 200x66."""
    camera = config["camera_contract"]
    width = int(message.width)
    height = int(message.height)
    encoding = str(message.encoding)
    step = int(message.step)
    data = bytes(message.data)
    expected = (int(camera["source_width"]), int(camera["source_height"]))
    if (width, height) != expected:
        raise RealDatasetError(f"camera dimensions {width}x{height}, expected {expected[0]}x{expected[1]}")
    if encoding != camera["source_encoding"] or encoding != "rgb8":
        raise RealDatasetError(f"camera encoding {encoding!r}, expected 'rgb8'")
    packed_row_bytes = width * 3
    if step < packed_row_bytes:
        raise RealDatasetError(f"camera step {step} is shorter than {packed_row_bytes}")
    required = step * height
    if len(data) < required:
        raise RealDatasetError(f"truncated camera payload: {len(data)} bytes, need {required}")

    packed = bytearray(packed_row_bytes * height)
    for row in range(height):
        source = row * step
        target = row * packed_row_bytes
        packed[target : target + packed_row_bytes] = data[source : source + packed_row_bytes]
    source_image = Image.frombytes("RGB", (width, height), bytes(packed))
    source_image.load()
    roi = camera["roi"]
    box = (
        int(roi["x_start"]),
        int(roi["y_start"]),
        int(roi["x_end"]),
        int(roi["y_end"]),
    )
    cropped = source_image.crop(box)
    source_image.close()
    expected_crop = (int(camera["cropped_width"]), int(camera["cropped_height"]))
    if cropped.size != expected_crop:
        cropped.close()
        raise RealDatasetError(f"Real Camera ROI produced {cropped.size}, expected {expected_crop}")
    output_size = (int(camera["output_width"]), int(camera["output_height"]))
    derived = cropped.resize(output_size, resample=Image.Resampling.BILINEAR)
    cropped.close()
    derived.load()
    if derived.mode != "RGB" or derived.size != output_size:
        derived.close()
        raise RealDatasetError("camera preprocessing did not produce RGB 200x66")
    return derived


def _iter_decoded(mcap_path: Path, topics: Iterable[str]):
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        yield from reader.iter_decoded_messages(topics=list(topics), log_time_order=True)


def scan_scalar_records(
    mcap_path: Path, config: dict[str, Any]
) -> tuple[list[ScalarRecord], list[ScalarRecord]]:
    steering_topic = config["topics"]["steering"]
    speed_topic = config["topics"]["speed"]
    expected = {
        steering_topic["name"]: steering_topic["type"],
        speed_topic["name"]: speed_topic["type"],
    }
    records: dict[str, list[ScalarRecord]] = {topic: [] for topic in expected}
    for schema, channel, message, decoded in _iter_decoded(mcap_path, expected):
        topic = str(channel.topic)
        if schema is None or str(schema.name) != expected[topic]:
            actual = None if schema is None else str(schema.name)
            raise RealDatasetError(f"{mcap_path}: {topic} type {actual!r}, expected {expected[topic]!r}")
        value = float(decoded.data)
        if not math.isfinite(value):
            raise RealDatasetError(f"{mcap_path}: non-finite {topic} value")
        records[topic].append(ScalarRecord(int(message.log_time), value, len(records[topic])))
    steering = sorted(records[steering_topic["name"]], key=lambda item: (item.time_ns, item.index))
    speed = sorted(records[speed_topic["name"]], key=lambda item: (item.time_ns, item.index))
    if not steering:
        raise RealDatasetError(f"{mcap_path}: no steering records")
    if not speed:
        raise RealDatasetError(f"{mcap_path}: no speed records")
    return steering, speed


def extract_camera_frames(
    *,
    bag_id: str,
    mcap_path: Path,
    dataset_root: Path,
    config: dict[str, Any],
) -> list[CameraFrame]:
    camera_topic = config["topics"]["camera"]
    image_dir = dataset_root / "images" / bag_id
    image_dir.mkdir(parents=True, exist_ok=False)
    frames: list[CameraFrame] = []
    for schema, channel, message, decoded in _iter_decoded(mcap_path, [camera_topic["name"]]):
        if schema is None or str(schema.name) != camera_topic["type"]:
            actual = None if schema is None else str(schema.name)
            raise RealDatasetError(
                f"{bag_id}: {channel.topic} type {actual!r}, expected {camera_topic['type']!r}"
            )
        camera_index = len(frames)
        relative_path = Path("images") / bag_id / f"frame_{camera_index:06d}.png"
        image = preprocess_real_camera_image(decoded, config)
        destination = dataset_root / relative_path
        image.save(destination, format="PNG", optimize=False)
        image.close()
        frames.append(
            CameraFrame(
                bag_id=bag_id,
                index=camera_index,
                time_ns=int(message.log_time),
                image_path=relative_path.as_posix(),
                image_sha256=sha256_file(destination),
                image_size_bytes=destination.stat().st_size,
            )
        )
    if len(frames) < 3:
        raise RealDatasetError(f"{bag_id}: only {len(frames)} camera frames")
    return frames


def _sequence_common_fields(
    frames: Sequence[CameraFrame], target_index: int, source_mcap_sha256: str
) -> dict[str, Any]:
    first, middle, current = frames[target_index - 2 : target_index + 1]
    gap_a = (middle.time_ns - first.time_ns) / 1e9
    gap_b = (current.time_ns - middle.time_ns) / 1e9
    return {
        "sequence_id": f"{current.bag_id}_t{current.index:06d}",
        "source_bag": current.bag_id,
        "source_mcap_sha256": source_mcap_sha256,
        "target_camera_index": current.index,
        "target_camera_log_time_ns": current.time_ns,
        "image_t_minus_2": first.image_path,
        "image_t_minus_1": middle.image_path,
        "image_t": current.image_path,
        "camera_t_minus_2_log_time_ns": first.time_ns,
        "camera_t_minus_1_log_time_ns": middle.time_ns,
        "camera_t_log_time_ns": current.time_ns,
        "adjacent_gap_t_minus_2_to_t_minus_1_s": gap_a,
        "adjacent_gap_t_minus_1_to_t_s": gap_b,
        "oldest_to_current_span_s": (current.time_ns - first.time_ns) / 1e9,
    }


def evaluate_sequence_candidate(
    *,
    frames: Sequence[CameraFrame],
    target_index: int,
    steering_records: Sequence[ScalarRecord],
    speed_records: Sequence[ScalarRecord],
    source_mcap_sha256: str,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return either an accepted manifest row or a fully explained rejection."""
    if target_index < 2 or target_index >= len(frames):
        raise RealDatasetError(f"invalid target camera index {target_index}")
    common = _sequence_common_fields(frames, target_index, source_mcap_sha256)
    reasons: list[str] = []
    camera_times = (
        common["camera_t_minus_2_log_time_ns"],
        common["camera_t_minus_1_log_time_ns"],
        common["camera_t_log_time_ns"],
    )
    if not camera_times[0] < camera_times[1] < camera_times[2]:
        reasons.append("non_increasing_camera_timestamps")
    maximum_gap = float(config["temporal_contract"]["maximum_adjacent_gap_s"])
    if (
        common["adjacent_gap_t_minus_2_to_t_minus_1_s"] > maximum_gap
        or common["adjacent_gap_t_minus_1_to_t_s"] > maximum_gap
    ):
        reasons.append("adjacent_camera_gap_gt_0p120_s")

    target_time = int(common["target_camera_log_time_ns"])
    steering_record = latest_causal(steering_records, target_time)
    steering_fields: dict[str, Any]
    if steering_record is None:
        reasons.append("no_causal_steering")
        steering_fields = {
            "steering_recorded_raw": None,
            "steering_rad": None,
            "steering_direction": None,
            "steering_log_time_ns": None,
            "steering_age_s": None,
        }
    else:
        if steering_record.time_ns > target_time:
            reasons.append("future_steering_label")
        steering_age_s = (target_time - steering_record.time_ns) / 1e9
        target = build_steering_target(steering_record.value)
        steering_fields = {
            "steering_recorded_raw": target.recorded_raw,
            "steering_rad": target.radians,
            "steering_direction": target.direction,
            "steering_log_time_ns": steering_record.time_ns,
            "steering_age_s": steering_age_s,
        }
        maximum_age = float(config["timestamp_contract"]["maximum_steering_age_s"])
        if steering_age_s > maximum_age:
            reasons.append("steering_age_gt_0p120_s")

    speed_fields = speed_metadata(
        target_time,
        speed_records,
        float(config["speed_contract"]["maximum_age_s"]),
    )
    complete = {**common, **steering_fields, **speed_fields}
    ordered_reasons = [reason for reason in REJECTION_REASON_ORDER if reason in reasons]
    if ordered_reasons:
        rejection = {
            **complete,
            "reasons": ordered_reasons,
            "reason_combination": "+".join(ordered_reasons),
        }
        return None, rejection
    assert complete["steering_log_time_ns"] <= complete["target_camera_log_time_ns"]
    assert complete["steering_age_s"] <= MAXIMUM_AGE_OR_GAP_S
    assert complete["adjacent_gap_t_minus_2_to_t_minus_1_s"] <= MAXIMUM_AGE_OR_GAP_S
    assert complete["adjacent_gap_t_minus_1_to_t_s"] <= MAXIMUM_AGE_OR_GAP_S
    assert len({complete[field] for field in NEURAL_INPUT_FIELDS}) == 3
    expected_rad = float(complete["steering_recorded_raw"]) * STEERING_SCALE_RAD
    if not math.isclose(float(complete["steering_rad"]), expected_rad, rel_tol=0.0, abs_tol=1e-15):
        raise RealDatasetError("steering target is omitted, clipped, or scaled more than once")
    return complete, None


def _manifest_value(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _manifest_value(row.get(key)) for key in MANIFEST_COLUMNS})


def write_rejections(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row))


def write_image_inventory(path: Path, frames_by_bag: dict[str, Sequence[CameraFrame]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=IMAGE_INVENTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for bag_id in sorted(frames_by_bag):
            for frame in frames_by_bag[bag_id]:
                writer.writerow(
                    {
                        "source_bag": bag_id,
                        "camera_index": frame.index,
                        "camera_log_time_ns": frame.time_ns,
                        "image_path": frame.image_path,
                        "image_sha256": frame.image_sha256,
                        "image_size_bytes": frame.image_size_bytes,
                        "width": 200,
                        "height": 66,
                        "color_space": "RGB",
                    }
                )


def _speed_completeness(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available = sum(bool(row["speed_available"]) for row in rows)
    valid = sum(bool(row["speed_valid"]) for row in rows)
    stale = sum(bool(row["speed_stale"]) for row in rows)
    missing = sum(not bool(row["speed_available"]) for row in rows)
    values = [float(row["speed_mps"]) for row in rows if row["speed_mps"] is not None]
    ages = [float(row["speed_age_s"]) for row in rows if row["speed_age_s"] is not None]
    return {
        "sequence_count": len(rows),
        "available_count": available,
        "missing_count": missing,
        "valid_count": valid,
        "stale_count": stale,
        "semantics": SPEED_SEMANTICS,
        "unit": "m/s",
        "value_mps": distribution(values),
        "age_s": distribution(ages),
    }


def _sign_counts(values: Sequence[float]) -> dict[str, int]:
    return {
        "negative_RIGHT": sum(value < 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "positive_LEFT": sum(value > 0 for value in values),
    }


def build_bag_sequences(
    *,
    bag_id: str,
    frames: Sequence[CameraFrame],
    steering_records: Sequence[ScalarRecord],
    speed_records: Sequence[ScalarRecord],
    source_mcap_sha256: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for target_index in range(2, len(frames)):
        row, rejection = evaluate_sequence_candidate(
            frames=frames,
            target_index=target_index,
            steering_records=steering_records,
            speed_records=speed_records,
            source_mcap_sha256=source_mcap_sha256,
            config=config,
        )
        if row is not None:
            accepted.append(row)
        else:
            assert rejection is not None
            rejected.append(rejection)

    reason_occurrences = Counter(reason for item in rejected for reason in item["reasons"])
    reason_combinations = Counter(item["reason_combination"] for item in rejected)
    raw = [float(row["steering_recorded_raw"]) for row in accepted]
    radians = [float(row["steering_rad"]) for row in accepted]
    steering_ages = [float(row["steering_age_s"]) for row in accepted]
    accepted_gaps = [
        float(row[field])
        for row in accepted
        for field in (
            "adjacent_gap_t_minus_2_to_t_minus_1_s",
            "adjacent_gap_t_minus_1_to_t_s",
        )
    ]
    spans = [float(row["oldest_to_current_span_s"]) for row in accepted]
    source_gaps = [(right.time_ns - left.time_ns) / 1e9 for left, right in zip(frames, frames[1:])]
    threshold = float(config["temporal_contract"]["maximum_adjacent_gap_s"])
    source_dropouts = [
        {
            "previous_camera_index": index,
            "current_camera_index": index + 1,
            "previous_camera_log_time_ns": frames[index].time_ns,
            "current_camera_log_time_ns": frames[index + 1].time_ns,
            "current_offset_from_first_camera_s": (frames[index + 1].time_ns - frames[0].time_ns) / 1e9,
            "gap_s": gap,
        }
        for index, gap in enumerate(source_gaps)
        if gap > threshold
    ]
    physical_min, physical_max = config["steering_contract"]["expected_physical_range_rad"]
    future_violations = sum(
        item["steering_log_time_ns"] is not None
        and item["steering_log_time_ns"] > item["target_camera_log_time_ns"]
        for item in [*accepted, *rejected]
    )
    duplicate_padding_violations = sum(
        len({row[field] for field in NEURAL_INPUT_FIELDS}) != 3 for row in accepted
    )
    episode_crossing_violations = sum(
        any(not str(row[field]).startswith(f"images/{bag_id}/") for field in NEURAL_INPUT_FIELDS)
        for row in accepted
    )
    candidate_rows = [*accepted, *rejected]
    metrics = {
        "bag_id": bag_id,
        "result": "PASS",
        "source": {
            "mcap_sha256": source_mcap_sha256,
            "camera_frame_count": len(frames),
            "steering_record_count": len(steering_records),
            "speed_record_count": len(speed_records),
        },
        "counts": {
            "candidate_sequences": max(0, len(frames) - 2),
            "accepted_sequences": len(accepted),
            "rejected_sequences": len(rejected),
            "rejection_reason_occurrences_note": "reason occurrences can overlap; combinations are mutually exclusive",
            "rejection_reason_occurrences": {
                reason: reason_occurrences[reason] for reason in REJECTION_REASON_ORDER
            },
            "rejection_reason_combinations": dict(sorted(reason_combinations.items())),
        },
        "steering": {
            "recorded_raw": distribution(raw),
            "radians": distribution(radians),
            "sign_counts": _sign_counts(radians),
            "conversion": "steering_rad = steering_recorded_raw * 0.35",
            "conversion_applied_exactly_once_for_every_accepted_sequence": all(
                math.isclose(
                    float(row["steering_rad"]),
                    float(row["steering_recorded_raw"]) * STEERING_SCALE_RAD,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for row in accepted
            ),
            "selective_clipping_applied": False,
            "physical_outside_expected_range_count": sum(
                value < physical_min or value > physical_max for value in radians
            ),
            "label_age_s": distribution(steering_ages),
        },
        "timing": {
            "timestamp_domain": "MCAP log_time",
            "causal_rule": "latest steering/speed with scalar log_time <= target camera log_time",
            "source_adjacent_camera_gap_s": distribution(source_gaps),
            "accepted_sequence_adjacent_camera_gap_s": distribution(accepted_gaps),
            "accepted_oldest_to_current_span_s": distribution(spans),
            "source_adjacent_gap_gt_0p120_s_count": len(source_dropouts),
            "source_adjacent_gap_gt_0p120_s": source_dropouts,
            "future_steering_label_violations": future_violations,
        },
        "speed_metadata": {
            "accepted_sequences": _speed_completeness(accepted),
            "all_candidate_sequences": _speed_completeness(candidate_rows),
            "used_as_active_driving_filter": False,
            "used_as_neural_input": False,
            "used_as_steering_target": False,
            "missing_or_stale_caused_sequence_rejection": False,
        },
        "temporal_integrity": {
            "episode_crossing_violations": episode_crossing_violations,
            "duplicate_frame_padding_violations": duplicate_padding_violations,
            "frame_order": list(config["temporal_contract"]["frame_order"]),
        },
        "exact_rejected_sequences": rejected,
    }
    if len(accepted) + len(rejected) != max(0, len(frames) - 2):
        raise RealDatasetError(f"{bag_id}: candidate accounting mismatch")
    if future_violations or duplicate_padding_violations or episode_crossing_violations:
        raise RealDatasetError(f"{bag_id}: temporal/causal invariant failed")
    return accepted, rejected, metrics


def verify_image_dataset(
    *,
    dataset_root: Path,
    frames_by_bag: dict[str, Sequence[CameraFrame]],
    accepted_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    stored = {
        frame.image_path: frame
        for frames in frames_by_bag.values()
        for frame in frames
    }
    referenced = {
        str(row[field])
        for row in accepted_rows
        for field in NEURAL_INPUT_FIELDS
    }
    missing = sorted(referenced - stored.keys())
    orphaned = sorted(stored.keys() - referenced)
    corrupt: list[dict[str, str]] = []
    bad_contract: list[dict[str, Any]] = []
    expected_size = (
        int(config["camera_contract"]["output_width"]),
        int(config["camera_contract"]["output_height"]),
    )
    verified = 0
    for relative_path in sorted(stored):
        path = dataset_root / relative_path
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != expected_size or image.mode != "RGB" or image.format != "PNG":
                    bad_contract.append(
                        {
                            "path": relative_path,
                            "size": list(image.size),
                            "mode": image.mode,
                            "format": image.format,
                        }
                    )
                else:
                    verified += 1
        except Exception as error:
            corrupt.append({"path": relative_path, "error": f"{type(error).__name__}: {error}"})
    duplicate_padding = sum(
        len({row[field] for field in NEURAL_INPUT_FIELDS}) != 3 for row in accepted_rows
    )
    result = "PASS" if not missing and not orphaned and not corrupt and not bad_contract and not duplicate_padding else "FAIL"
    return {
        "result": result,
        "stored_image_count": len(stored),
        "unique_referenced_image_count": len(referenced),
        "verified_rgb_200x66_png_count": verified,
        "missing_referenced_image_count": len(missing),
        "orphaned_stored_image_count": len(orphaned),
        "corrupt_image_count": len(corrupt),
        "bad_contract_image_count": len(bad_contract),
        "duplicate_frame_padding_violation_count": duplicate_padding,
        "missing_referenced_images": missing,
        "orphaned_stored_images": orphaned,
        "corrupt_images": corrupt,
        "bad_contract_images": bad_contract,
        "roi": config["camera_contract"]["roi"],
        "derived_contract": {
            "width": expected_size[0],
            "height": expected_size[1],
            "color_space": "RGB",
            "format": "PNG",
            "resize": "Pillow Image.Resampling.BILINEAR",
        },
        "simulator_roi_used": False,
        "horizontal_crop_used": False,
        "undistortion_used": False,
    }


def _select_preview_rows(rows: Sequence[dict[str, Any]], maximum_count: int) -> list[dict[str, Any]]:
    if not rows or maximum_count <= 0:
        return []
    maximum_count = min(maximum_count, len(rows))
    selected: set[int] = {0, len(rows) - 1}
    selected.add(min(range(len(rows)), key=lambda index: float(rows[index]["steering_rad"])))
    selected.add(max(range(len(rows)), key=lambda index: float(rows[index]["steering_rad"])))
    divisions = max(maximum_count - 1, 1)
    for position in range(maximum_count):
        selected.add(round((len(rows) - 1) * position / divisions))
    if len(selected) < maximum_count:
        for index in range(len(rows)):
            selected.add(index)
            if len(selected) == maximum_count:
                break
    ordered = sorted(selected)
    if len(ordered) > maximum_count:
        mandatory = {0, len(rows) - 1}
        minimum = min(range(len(rows)), key=lambda index: float(rows[index]["steering_rad"]))
        maximum = max(range(len(rows)), key=lambda index: float(rows[index]["steering_rad"]))
        mandatory.update((minimum, maximum))
        keep = sorted(mandatory)
        for index in ordered:
            if index not in mandatory and len(keep) < maximum_count:
                keep.append(index)
        ordered = sorted(keep[:maximum_count])
    return [rows[index] for index in ordered]


def create_contact_sheet(
    *,
    bag_id: str,
    rows: Sequence[dict[str, Any]],
    dataset_root: Path,
    maximum_count: int,
) -> dict[str, Any]:
    selected = _select_preview_rows(rows, maximum_count)
    if not selected:
        raise RealDatasetError(f"{bag_id}: no accepted sequence for contact-sheet QC")
    image_width, image_height = 400, 132
    label_height = 50
    columns = 3
    row_count = math.ceil(len(selected) / columns)
    sheet = Image.new("RGB", (columns * image_width, row_count * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, row in enumerate(selected):
        with Image.open(dataset_root / str(row["image_t"])) as source:
            source.load()
            enlarged = source.resize((image_width, image_height), Image.Resampling.BILINEAR)
        x = (position % columns) * image_width
        y = (position // columns) * (image_height + label_height)
        sheet.paste(enlarged, (x, y))
        enlarged.close()
        draw.text(
            (x + 4, y + image_height + 3),
            f"#{int(row['target_camera_index']):06d} raw={float(row['steering_recorded_raw']):+.4f} "
            f"rad={float(row['steering_rad']):+.4f}",
            fill="black",
        )
        speed = "missing" if row["speed_mps"] is None else f"{float(row['speed_mps']):.3f} m/s"
        draw.text(
            (x + 4, y + image_height + 21),
            f"gap={float(row['adjacent_gap_t_minus_1_to_t_s'])*1000:.1f}ms v={speed} {row['speed_state']}",
            fill="black",
        )
    relative_path = Path("previews") / f"{bag_id}_contact_sheet.png"
    destination = dataset_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", optimize=False)
    sheet.close()
    return {
        "source_bag": bag_id,
        "path": relative_path.as_posix(),
        "selected_sequence_count": len(selected),
        "selected_sequence_ids": [str(row["sequence_id"]) for row in selected],
        "selection_scope": "bounded early/middle/late plus steering extrema across accepted sequences",
        "shows_derived_real_roi": True,
    }


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def aggregate_summary(
    *,
    config: dict[str, Any],
    config_sha256: str,
    final_dataset_root: Path,
    working_dataset_root: Path,
    source_bags: Sequence[dict[str, Any]],
    bag_metrics: Sequence[dict[str, Any]],
    accepted_rows: Sequence[dict[str, Any]],
    rejected_rows: Sequence[dict[str, Any]],
    image_qc: dict[str, Any],
    previews: Sequence[dict[str, Any]],
    manifest_relative_path: str,
    manifest_sha256: str,
    image_inventory_relative_path: str,
    image_inventory_sha256: str,
) -> dict[str, Any]:
    reasons = Counter(reason for row in rejected_rows for reason in row["reasons"])
    combinations = Counter(row["reason_combination"] for row in rejected_rows)
    raw = [float(row["steering_recorded_raw"]) for row in accepted_rows]
    radians = [float(row["steering_rad"]) for row in accepted_rows]
    steering_ages = [float(row["steering_age_s"]) for row in accepted_rows]
    accepted_gaps = [
        float(row[field])
        for row in accepted_rows
        for field in (
            "adjacent_gap_t_minus_2_to_t_minus_1_s",
            "adjacent_gap_t_minus_1_to_t_s",
        )
    ]
    spans = [float(row["oldest_to_current_span_s"]) for row in accepted_rows]
    candidate_rows = [*accepted_rows, *rejected_rows]
    source_dropouts = [
        {"source_bag": item["bag_id"], **dropout}
        for item in bag_metrics
        for dropout in item["timing"]["source_adjacent_gap_gt_0p120_s"]
    ]
    exact_gap_rejections = [
        {
            "sequence_id": row["sequence_id"],
            "source_bag": row["source_bag"],
            "target_camera_index": row["target_camera_index"],
            "target_camera_log_time_ns": row["target_camera_log_time_ns"],
            "gap_t_minus_2_to_t_minus_1_s": row["adjacent_gap_t_minus_2_to_t_minus_1_s"],
            "gap_t_minus_1_to_t_s": row["adjacent_gap_t_minus_1_to_t_s"],
            "steering_age_s": row["steering_age_s"],
            "speed_age_s": row["speed_age_s"],
            "speed_state": row["speed_state"],
            "reasons": row["reasons"],
        }
        for row in rejected_rows
        if "adjacent_camera_gap_gt_0p120_s" in row["reasons"]
    ]
    future_violations = sum(
        item["timing"]["future_steering_label_violations"] for item in bag_metrics
    )
    physical_range = config["steering_contract"]["expected_physical_range_rad"]
    physical_outside = sum(
        value < physical_range[0] or value > physical_range[1] for value in radians
    )
    conversion_valid = all(
        math.isclose(
            float(row["steering_rad"]),
            float(row["steering_recorded_raw"]) * STEERING_SCALE_RAD,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        for row in accepted_rows
    )
    accepted_speed = _speed_completeness(accepted_rows)
    candidate_speed = _speed_completeness(candidate_rows)
    candidate_count = sum(item["counts"]["candidate_sequences"] for item in bag_metrics)
    accepted_count = len(accepted_rows)
    rejected_count = len(rejected_rows)
    no_mcap_copies = not any(working_dataset_root.rglob("*.mcap"))
    gates = {
        "all_three_source_hashes_match": all(item["hash_matches_expected"] for item in source_bags),
        "candidate_accounting_exact": candidate_count == accepted_count + rejected_count,
        "nonzero_accepted_sequences_each_bag": all(
            item["counts"]["accepted_sequences"] > 0 for item in bag_metrics
        ),
        "whole_stream_steering_scale_exactly_once": conversion_valid,
        "physical_steering_range_qc": physical_outside == 0,
        "future_steering_label_violations_zero": future_violations == 0,
        "accepted_steering_ages_within_0p120_s": bool(steering_ages)
        and max(steering_ages) <= MAXIMUM_AGE_OR_GAP_S,
        "accepted_adjacent_camera_gaps_within_0p120_s": bool(accepted_gaps)
        and max(accepted_gaps) <= MAXIMUM_AGE_OR_GAP_S,
        "no_episode_crossing": all(
            item["temporal_integrity"]["episode_crossing_violations"] == 0
            for item in bag_metrics
        ),
        "no_duplicate_frame_padding": all(
            item["temporal_integrity"]["duplicate_frame_padding_violations"] == 0
            for item in bag_metrics
        ),
        "every_stored_image_passes_qc": image_qc["result"] == "PASS",
        "speed_semantics_preserved_unknown": all(
            row["speed_semantics"] == SPEED_SEMANTICS for row in candidate_rows
        ),
        "speed_not_used_as_filter_input_or_target": True,
        "no_raw_mcap_copies": no_mcap_copies,
        "training_not_invoked": True,
    }
    result = "PASS" if all(gates.values()) else "FAIL"
    return {
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "real_dataset_v1_passes_qc": result == "PASS",
        "config_sha256": config_sha256,
        "external_dataset_root": str(final_dataset_root),
        "source_bags": list(source_bags),
        "extraction_configuration": {
            "timestamp_domain": "MCAP log_time",
            "causal_rule": "latest scalar log_time <= target camera log_time",
            "steering_conversion": "steering_rad = steering_recorded * 0.35",
            "selective_steering_clipping": False,
            "maximum_steering_age_s": MAXIMUM_AGE_OR_GAP_S,
            "temporal_frames": ["t_minus_2", "t_minus_1", "t"],
            "maximum_adjacent_camera_gap_s": MAXIMUM_AGE_OR_GAP_S,
            "speed_semantics": SPEED_SEMANTICS,
            "speed_is_neural_input": False,
            "speed_is_steering_target": False,
            "speed_is_active_driving_filter": False,
            "real_camera_roi": config["camera_contract"],
        },
        "counts": {
            "candidate_sequences": candidate_count,
            "accepted_sequences": accepted_count,
            "rejected_sequences": rejected_count,
            "retention_fraction": accepted_count / candidate_count if candidate_count else 0.0,
            "rejection_reason_occurrences_note": "occurrence counts overlap; mutually exclusive combination counts sum to rejected_sequences",
            "rejection_reason_occurrences": {
                reason: reasons[reason] for reason in REJECTION_REASON_ORDER
            },
            "rejection_reason_combinations": dict(sorted(combinations.items())),
        },
        "per_bag": [
            {
                "bag_id": item["bag_id"],
                "camera_frames": item["source"]["camera_frame_count"],
                "candidate_sequences": item["counts"]["candidate_sequences"],
                "accepted_sequences": item["counts"]["accepted_sequences"],
                "rejected_sequences": item["counts"]["rejected_sequences"],
                "rejection_reason_occurrences": item["counts"]["rejection_reason_occurrences"],
                "steering_recorded_raw": item["steering"]["recorded_raw"],
                "steering_rad": item["steering"]["radians"],
                "steering_label_age_s": item["steering"]["label_age_s"],
                "accepted_sequence_adjacent_camera_gap_s": item["timing"]["accepted_sequence_adjacent_camera_gap_s"],
                "accepted_oldest_to_current_span_s": item["timing"]["accepted_oldest_to_current_span_s"],
                "speed_metadata": item["speed_metadata"],
            }
            for item in bag_metrics
        ],
        "steering": {
            "recorded_raw": distribution(raw),
            "radians": distribution(radians),
            "sign_counts": _sign_counts(radians),
            "conversion_applied_exactly_once_for_every_accepted_sequence": conversion_valid,
            "selective_clipping_applied": False,
            "physical_outside_expected_range_count": physical_outside,
        },
        "timing": {
            "steering_label_age_s": distribution(steering_ages),
            "accepted_sequence_adjacent_camera_gap_s": distribution(accepted_gaps),
            "accepted_oldest_to_current_span_s": distribution(spans),
            "source_adjacent_gap_gt_0p120_s_count": len(source_dropouts),
            "source_adjacent_gap_gt_0p120_s": source_dropouts,
            "exact_gap_rejected_sequence_count": len(exact_gap_rejections),
            "exact_gap_rejected_sequences": exact_gap_rejections,
            "future_steering_label_violations": future_violations,
        },
        "speed_metadata": {
            "accepted_sequences": accepted_speed,
            "all_candidate_sequences": candidate_speed,
            "semantics": SPEED_SEMANTICS,
            "missing_or_stale_never_causes_rejection": True,
            "neural_input_fields": list(NEURAL_INPUT_FIELDS),
        },
        "image_qc": image_qc,
        "previews": list(previews),
        "artifacts": {
            "manifest": manifest_relative_path,
            "manifest_sha256": manifest_sha256,
            "manifest_row_count": accepted_count,
            "rejections": "manifests/rejections.jsonl",
            "rejection_row_count": rejected_count,
            "image_inventory": image_inventory_relative_path,
            "image_inventory_sha256": image_inventory_sha256,
        },
        "qc_gates": gates,
        "training_stage": {
            "training_invoked": False,
            "training_authorized_by_this_task": False,
            "can_be_considered_next": result == "PASS",
            "decision": (
                "CAN BE CONSIDERED AFTER HUMAN REVIEW AND FINAL SUBSET SELECTION"
                if result == "PASS"
                else "NOT READY"
            ),
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _range_summary(stats: dict[str, Any]) -> str:
    return (
        f"min={_fmt(stats['min'])}, mean={_fmt(stats['mean'])}, "
        f"median={_fmt(stats['median'])}, p95={_fmt(stats['p95'])}, max={_fmt(stats['max'])}"
    )


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Real Dataset Extraction V1",
        "",
        f"Extraction/QC result: **{summary['result']}**. No training was performed.",
        "",
        "## 1. Exact source bag hashes",
        "",
        "| Bag | Bytes | SHA-256 | Verified |",
        "|---|---:|---|---|",
    ]
    for item in summary["source_bags"]:
        lines.append(
            f"| {item['bag_id']} | {item['mcap_size_bytes']} | `{item['mcap_sha256']}` | "
            f"{'yes' if item['hash_matches_expected'] else 'no'} |"
        )
    config = summary["extraction_configuration"]
    roi = config["real_camera_roi"]["roi"]
    lines.extend(
        [
            "",
            "## 2. Extraction configuration",
            "",
            "MCAP `log_time`; causal scalar ZOH; steering `recorded * 0.35` rad without clipping; "
            "steering age and each adjacent camera gap must be <=0.120 s. Speed is metadata-only "
            f"with semantics `{SPEED_SEMANTICS}`. Real ROI is "
            f"`x={roi['x_start']}:{roi['x_end']}, y={roi['y_start']}:{roi['y_end']}` -> RGB 200x66 bilinear.",
            "",
            "## 3. Accepted temporal sequence count",
            "",
            f"**{summary['counts']['accepted_sequences']}** of {summary['counts']['candidate_sequences']} candidates "
            f"({summary['counts']['retention_fraction']:.6f}).",
            "",
            "## 4. Rejected count and reasons",
            "",
            f"Rejected unique candidates: **{summary['counts']['rejected_sequences']}**.",
            "",
            f"Reason occurrences (overlap allowed): `{summary['counts']['rejection_reason_occurrences']}`.",
            "",
            f"Mutually exclusive reason combinations: `{summary['counts']['rejection_reason_combinations']}`.",
            "",
            "## 5. Per-bag sequence counts",
            "",
            "| Bag | Camera frames | Candidates | Accepted | Rejected |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in summary["per_bag"]:
        lines.append(
            f"| {item['bag_id']} | {item['camera_frames']} | {item['candidate_sequences']} | "
            f"{item['accepted_sequences']} | {item['rejected_sequences']} |"
        )
    steering = summary["steering"]
    lines.extend(
        [
            "",
            "## 6. Steering raw/radian distributions",
            "",
            f"Recorded normalized command: {_range_summary(steering['recorded_raw'])}.",
            "",
            f"Physical radians: {_range_summary(steering['radians'])}; signs={steering['sign_counts']}; "
            f"outside +/-0.35={steering['physical_outside_expected_range_count']}. Conversion-once QC="
            f"{steering['conversion_applied_exactly_once_for_every_accepted_sequence']}.",
            "",
            "## 7. Steering label-age distribution",
            "",
            _range_summary(summary["timing"]["steering_label_age_s"]) + " s.",
            "",
            "## 8. Camera gap distribution",
            "",
            "Accepted sequence adjacent gaps: "
            + _range_summary(summary["timing"]["accepted_sequence_adjacent_camera_gap_s"])
            + " s. Oldest-to-current spans: "
            + _range_summary(summary["timing"]["accepted_oldest_to_current_span_s"])
            + " s.",
            "",
            "## 9. Exact >120 ms rejects",
            "",
        ]
    )
    for item in summary["timing"]["exact_gap_rejected_sequences"]:
        lines.append(
            f"- {item['sequence_id']}: gaps={item['gap_t_minus_2_to_t_minus_1_s']:.9f}/"
            f"{item['gap_t_minus_1_to_t_s']:.9f} s, steering_age={_fmt(item['steering_age_s'], 9)} s, "
            f"reasons={item['reasons']}"
        )
    accepted_speed = summary["speed_metadata"]["accepted_sequences"]
    candidate_speed = summary["speed_metadata"]["all_candidate_sequences"]
    lines.extend(
        [
            "",
            "## 10. Speed metadata completeness/staleness",
            "",
            f"Accepted: available={accepted_speed['available_count']}, missing={accepted_speed['missing_count']}, "
            f"valid={accepted_speed['valid_count']}, stale={accepted_speed['stale_count']}. "
            f"All candidates: available={candidate_speed['available_count']}, missing={candidate_speed['missing_count']}, "
            f"valid={candidate_speed['valid_count']}, stale={candidate_speed['stale_count']}. "
            f"Semantics remain `{SPEED_SEMANTICS}`; speed is not an input, target, or filter.",
            "",
            "## 11. ROI and image QC",
            "",
            f"{summary['image_qc']['verified_rgb_200x66_png_count']} images verified as RGB 200x66 PNG; "
            f"missing={summary['image_qc']['missing_referenced_image_count']}, "
            f"orphaned={summary['image_qc']['orphaned_stored_image_count']}, "
            f"corrupt={summary['image_qc']['corrupt_image_count']}. Simulator ROI, horizontal crop, and "
            "undistortion were not used.",
            "",
            "## 12. Future-label violations",
            "",
            f"**{summary['timing']['future_steering_label_violations']}**.",
            "",
            "## 13. Dataset manifest hash",
            "",
            f"`{summary['artifacts']['manifest_sha256']}` ({summary['artifacts']['manifest_row_count']} rows, "
            f"`{summary['artifacts']['manifest']}`).",
            "",
            "## 14. REAL_DATASET_V1 QC decision",
            "",
            f"**{summary['result']}**: all QC gates={summary['qc_gates']}.",
            "",
            "## 15. Whether training can be considered next",
            "",
            f"**{summary['training_stage']['decision']}**. Training remains unauthorized and was not invoked.",
            "",
            "## 16. Tests",
            "",
            "The extractor does not invoke tests or training. Focused/full regression results are recorded in the task handoff.",
            "",
            "## 17. External artifacts",
            "",
            f"Dataset root: `{summary['external_dataset_root']}`. Manifest, image inventory, rejection records, "
            "derived images, metadata, and bounded contact sheets are stored there; no raw MCAP was copied.",
            "",
            "## 18. Git status",
            "",
            "No commit or push is performed by the extractor. Exact final status is recorded in the task handoff.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_source_bags(config: dict[str, Any], output_root: Path) -> list[tuple[dict[str, Any], Path]]:
    resolved: list[tuple[dict[str, Any], Path]] = []
    output = output_root.resolve()
    for spec in config["source_bags"]:
        bag_root = Path(spec["bag_root"]).resolve()
        mcap_path = bag_root / str(spec["mcap_filename"])
        if not mcap_path.is_file():
            raise RealDatasetError(f"{spec['bag_id']}: missing source MCAP {mcap_path}")
        if output == bag_root or bag_root in output.parents:
            raise RealDatasetError("external dataset root must not be inside a source bag")
        size = mcap_path.stat().st_size
        if size != int(spec["expected_mcap_size_bytes"]):
            raise RealDatasetError(
                f"{spec['bag_id']}: source size changed: expected {spec['expected_mcap_size_bytes']}, got {size}"
            )
        digest = sha256_file(mcap_path)
        if digest != spec["expected_mcap_sha256"]:
            raise RealDatasetError(
                f"{spec['bag_id']}: source hash changed: expected {spec['expected_mcap_sha256']}, got {digest}"
            )
        evidence = {
            "bag_id": spec["bag_id"],
            "bag_root": str(bag_root),
            "mcap_path": str(mcap_path),
            "mcap_size_bytes": size,
            "mcap_sha256": digest,
            "expected_mcap_sha256": spec["expected_mcap_sha256"],
            "hash_matches_expected": True,
            "source_access": "read_only; not copied or modified",
        }
        resolved.append((evidence, mcap_path))
    return resolved


def run_extraction(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path | None = None,
    result_root: Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_sha256 = sha256_file(config_path)
    final_output = (
        output_root.resolve()
        if output_root is not None
        else Path(config["storage"]["external_dataset_root"]).resolve()
    )
    configured_result = Path(config["storage"]["compact_result_root"])
    final_results = (
        result_root.resolve()
        if result_root is not None
        else (repo_root / configured_result).resolve()
    )
    if final_output.exists():
        raise FileExistsError(f"external dataset already exists: {final_output}")
    if final_results.exists():
        raise FileExistsError(f"compact result already exists: {final_results}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_results.parent.mkdir(parents=True, exist_ok=True)
    resolved_sources = _resolve_source_bags(config, final_output)

    output_stage = Path(
        tempfile.mkdtemp(prefix=f".{VERSION}.incomplete.", dir=final_output.parent)
    )
    result_stage = Path(
        tempfile.mkdtemp(prefix=f".{VERSION}.incomplete.", dir=final_results.parent)
    )
    try:
        (output_stage / "manifests").mkdir()
        (output_stage / "previews").mkdir()
        frames_by_bag: dict[str, list[CameraFrame]] = {}
        bag_metrics: list[dict[str, Any]] = []
        all_accepted: list[dict[str, Any]] = []
        all_rejected: list[dict[str, Any]] = []
        source_evidence: list[dict[str, Any]] = []
        for evidence, mcap_path in resolved_sources:
            bag_id = str(evidence["bag_id"])
            steering, speed = scan_scalar_records(mcap_path, config)
            frames = extract_camera_frames(
                bag_id=bag_id,
                mcap_path=mcap_path,
                dataset_root=output_stage,
                config=config,
            )
            accepted, rejected, metrics = build_bag_sequences(
                bag_id=bag_id,
                frames=frames,
                steering_records=steering,
                speed_records=speed,
                source_mcap_sha256=str(evidence["mcap_sha256"]),
                config=config,
            )
            metrics["source"].update(
                {
                    "mcap_path": evidence["mcap_path"],
                    "mcap_size_bytes": evidence["mcap_size_bytes"],
                    "hash_matches_expected": True,
                }
            )
            frames_by_bag[bag_id] = frames
            bag_metrics.append(metrics)
            all_accepted.extend(accepted)
            all_rejected.extend(rejected)
            source_evidence.append(evidence)
            write_manifest(output_stage / "manifests" / f"{bag_id}.csv", accepted)

        manifest_relative = Path("manifests") / "real_dataset_v1.csv"
        manifest_path = output_stage / manifest_relative
        write_manifest(manifest_path, all_accepted)
        rejections_path = output_stage / "manifests" / "rejections.jsonl"
        write_rejections(rejections_path, all_rejected)
        inventory_relative = Path("manifests") / "image_inventory.csv"
        inventory_path = output_stage / inventory_relative
        write_image_inventory(inventory_path, frames_by_bag)

        image_qc = verify_image_dataset(
            dataset_root=output_stage,
            frames_by_bag=frames_by_bag,
            accepted_rows=all_accepted,
            config=config,
        )
        previews = [
            create_contact_sheet(
                bag_id=bag_id,
                rows=[row for row in all_accepted if row["source_bag"] == bag_id],
                dataset_root=output_stage,
                maximum_count=int(config["qc"]["preview_sequences_per_bag"]),
            )
            for bag_id in ("bag_01", "bag_02", "bag_03")
        ]
        summary = aggregate_summary(
            config=config,
            config_sha256=config_sha256,
            final_dataset_root=final_output,
            working_dataset_root=output_stage,
            source_bags=source_evidence,
            bag_metrics=bag_metrics,
            accepted_rows=all_accepted,
            rejected_rows=all_rejected,
            image_qc=image_qc,
            previews=previews,
            manifest_relative_path=manifest_relative.as_posix(),
            manifest_sha256=sha256_file(manifest_path),
            image_inventory_relative_path=inventory_relative.as_posix(),
            image_inventory_sha256=sha256_file(inventory_path),
        )
        report = build_report(summary)
        metadata = {
            **summary,
            "config": config,
            "bag_metrics": bag_metrics,
            "scope_attestation": {
                "training_performed": False,
                "simulator_used": False,
                "docker_used": False,
                "raw_bag_copied": False,
                "raw_bag_modified": False,
                "speed_semantics_guessed": False,
            },
        }
        write_json(output_stage / "dataset_metadata.json", metadata)
        write_json(output_stage / "summary.json", summary)
        (output_stage / "REPORT.md").write_text(report, encoding="utf-8")
        (output_stage / "config_snapshot.json").write_bytes(canonical_json_bytes(config))

        write_json(result_stage / "summary.json", summary)
        write_json(result_stage / "rejections.json", all_rejected)
        write_json(result_stage / "image_qc.json", image_qc)
        for metrics in bag_metrics:
            write_json(result_stage / f"{metrics['bag_id']}.json", metrics)
        (result_stage / "REPORT.md").write_text(report, encoding="utf-8")

        if summary["result"] != "PASS":
            raise RealDatasetError("REAL_DATASET_V1 failed one or more QC gates")
        if any(output_stage.rglob("*.mcap")):
            raise RealDatasetError("derived dataset unexpectedly contains a raw MCAP")

        output_stage.rename(final_output)
        result_stage.rename(final_results)
        summary["external_dataset_size_bytes"] = _directory_size(final_output)
        return summary
    except Exception as error:
        failure = {
            "version": VERSION,
            "result": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "external_incomplete_path": str(output_stage),
            "result_incomplete_path": str(result_stage),
        }
        try:
            write_json(output_stage / "FAILED.json", failure)
            write_json(result_stage / "FAILED.json", failure)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--result-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (args.config or repo_root / "configs" / "real_dataset_v1.json").resolve()
    summary = run_extraction(
        repo_root=repo_root,
        config_path=config_path,
        output_root=args.output_root,
        result_root=args.result_root,
    )
    print(
        json.dumps(
            {
                "result": summary["result"],
                "accepted_sequences": summary["counts"]["accepted_sequences"],
                "rejected_sequences": summary["counts"]["rejected_sequences"],
                "manifest_sha256": summary["artifacts"]["manifest_sha256"],
                "external_dataset_root": summary["external_dataset_root"],
                "training_invoked": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
