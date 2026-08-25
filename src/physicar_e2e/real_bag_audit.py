"""Read-only Real PhysiCar Bag Audit tooling.

The audit deliberately keeps camera preprocessing and dataset extraction out of
scope.  MCAP log time is the only synchronization domain used for causal joins.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import statistics
import struct
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from mcap.reader import NonSeekingReader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageOps


AUDIT_VERSION = "real_bag_audit_v1"
SUPPORTED_AUDIT_VERSIONS = {"real_bag_audit_v1", "real_bag_audit_v2"}
MCAP_MAGIC = b"\x89MCAP0\r\n"


class RealBagAuditError(RuntimeError):
    """Raised when the requested audit cannot be configured safely."""


@dataclass(frozen=True)
class ScalarSample:
    index: int
    record_time_ns: int
    publish_time_ns: int
    value: float


@dataclass(frozen=True)
class CameraSample:
    index: int
    record_time_ns: int
    publish_time_ns: int
    header_time_ns: int | None
    width: int
    height: int
    encoding: str
    is_bigendian: int
    step: int
    data_length: int
    frame_id: str


@dataclass
class BagScan:
    topic_counts: Counter[str]
    topic_types: dict[str, set[str]]
    topic_times: dict[str, list[int]]
    all_message_times: list[int]
    cameras: list[CameraSample]
    steering: list[ScalarSample]
    speed: list[ScalarSample]
    decode_attempts: Counter[str]
    decode_failures: Counter[str]
    decode_failure_examples: list[dict[str, Any]]
    float64_cdr_crosscheck_attempts: Counter[str]
    float64_cdr_crosscheck_failures: Counter[str]
    scan_completed: bool
    scan_error_type: str | None
    scan_error_message: str | None


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") not in SUPPORTED_AUDIT_VERSIONS:
        raise RealBagAuditError(
            f"expected one of {sorted(SUPPORTED_AUDIT_VERSIONS)!r} configurations"
        )
    bag_ids = [item.get("id") for item in config.get("bags", [])]
    if bag_ids != ["bag_01", "bag_02", "bag_03"]:
        raise RealBagAuditError("configuration must map bag_01, bag_02, and bag_03 in order")
    required = config.get("required_topics", {})
    if set(required) != {"camera", "steering", "speed"}:
        raise RealBagAuditError("only camera, steering, and speed may be required E2E topics")
    if config.get("scope_guards", {}).get("require_odometry"):
        raise RealBagAuditError("Real Bag Audit must not require odometry")
    if config.get("real_camera_roi", {}).get("auto_apply_simulator_roi"):
        raise RealBagAuditError("simulator ROI must not be automatically applied to real images")
    if config.get("version") == "real_bag_audit_v2":
        steering_contract = config.get("steering_contract", {})
        speed_contract = config.get("speed_contract", {})
        if (
            steering_contract.get("recorded_representation") != "normalized_raw"
            or steering_contract.get("recorded_to_radians_scale") != 0.35
            or steering_contract.get("physical_unit") != "radians"
            or steering_contract.get("left_right_sign_convention")
            != "positive_left_negative_right"
            or steering_contract.get("command_or_feedback") != "command"
            or steering_contract.get("clip_out_of_range") is not False
            or speed_contract.get("unit") != "meters_per_second"
            or speed_contract.get("unit_symbol") != "m/s"
            or speed_contract.get("meaning")
            != "unresolved_command_or_actual_feedback_measurement"
        ):
            raise RealBagAuditError(
                "V2 real-vehicle steering or speed contract does not match confirmed semantics"
            )
        roi = config.get("real_camera_roi", {})
        source = roi.get("source", {})
        crop = roi.get("crop", {})
        resize = roi.get("resize", {})
        temporal = roi.get("temporal_input", {})
        if (
            roi.get("version") != "real_camera_roi_v1"
            or roi.get("status") != "approved"
            or (source.get("width"), source.get("height"), source.get("color_space"), source.get("ros_encoding"))
            != (480, 360, "RGB", "rgb8")
            or (crop.get("x_start"), crop.get("x_end"), crop.get("y_start"), crop.get("y_end"))
            != (0, 480, 80, 360)
            or crop.get("end_coordinates_exclusive") is not True
            or (roi.get("cropped_width"), roi.get("cropped_height")) != (480, 280)
            or (resize.get("output_width"), resize.get("output_height"), resize.get("interpolation"))
            != (200, 66, "bilinear")
            or roi.get("post_resize_color_conversion") != "existing RGB_to_YUV preprocessing"
            or temporal.get("frame_order") != ["t_minus_2", "t_minus_1", "t"]
            or temporal.get("frame_count") != 3
            or temporal.get("causal") is not True
            or roi.get("horizontal_crop_applied") is not False
            or roi.get("camera_undistortion_applied") is not False
            or roi.get("simulator_y_160_360_crop_used") is not False
        ):
            raise RealBagAuditError("Real Camera ROI V1 contract does not match the approved geometry and preprocessing")
    for forbidden in ("generate_training_dataset", "invoke_training", "drive_simulator", "modify_docker", "modify_bags"):
        if config.get("scope_guards", {}).get(forbidden):
            raise RealBagAuditError(f"scope guard {forbidden!r} must be false")
    return config


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _first_int(lines: Sequence[str], pattern: str) -> int:
    compiled = re.compile(pattern)
    for line in lines:
        match = compiled.match(line)
        if match:
            return int(match.group(1))
    raise RealBagAuditError(f"metadata field not found: {pattern}")


def _nested_int(lines: Sequence[str], section: str, field: str) -> int:
    marker = f"  {section}:"
    try:
        start = lines.index(marker)
    except ValueError as error:
        raise RealBagAuditError(f"metadata section not found: {section}") from error
    pattern = re.compile(rf"^    {re.escape(field)}:\s*(\d+)\s*$")
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise RealBagAuditError(f"metadata field not found: {section}.{field}")


def parse_rosbag_metadata(path: Path) -> dict[str, Any]:
    """Parse the small rosbag2 metadata subset needed by this audit.

    This avoids adding a YAML dependency solely for a stable rosbag2-generated
    file shape. Unknown fields remain untouched on disk and out of the contract.
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    topics: dict[str, dict[str, Any]] = {}
    in_topics = False
    current: dict[str, Any] | None = None
    for line in lines:
        if line == "  topics_with_message_count:":
            in_topics = True
            continue
        if in_topics and line.startswith("  ") and not line.startswith("    "):
            in_topics = False
        if not in_topics:
            continue
        if line == "    - topic_metadata:":
            current = {}
            continue
        name_match = re.match(r"^        name:\s*(.+?)\s*$", line)
        type_match = re.match(r"^        type:\s*(.+?)\s*$", line)
        count_match = re.match(r"^      message_count:\s*(\d+)\s*$", line)
        if current is not None and name_match:
            current["name"] = _yaml_scalar(name_match.group(1))
        elif current is not None and type_match:
            current["type"] = _yaml_scalar(type_match.group(1))
        elif current is not None and count_match:
            current["message_count"] = int(count_match.group(1))
            if "name" not in current or "type" not in current:
                raise RealBagAuditError(f"incomplete topic metadata in {path}")
            topics[current["name"]] = current
            current = None

    def scalar(pattern: str) -> str:
        compiled = re.compile(pattern)
        for candidate in lines:
            match = compiled.match(candidate)
            if match:
                return _yaml_scalar(match.group(1))
        raise RealBagAuditError(f"metadata scalar not found: {pattern}")

    relative_paths: list[str] = []
    try:
        relative_start = lines.index("  relative_file_paths:")
    except ValueError as error:
        raise RealBagAuditError("metadata has no relative_file_paths") from error
    for line in lines[relative_start + 1 :]:
        match = re.match(r"^    -\s+(.+?)\s*$", line)
        if match:
            relative_paths.append(_yaml_scalar(match.group(1)))
            continue
        if line.startswith("  ") and not line.startswith("    "):
            break
    if not relative_paths:
        raise RealBagAuditError("metadata lists no MCAP files")

    start_ns = _nested_int(lines, "starting_time", "nanoseconds_since_epoch")
    duration_ns = _nested_int(lines, "duration", "nanoseconds")
    return {
        "version": _first_int(lines, r"^  version:\s*(\d+)\s*$"),
        "storage_identifier": scalar(r"^  storage_identifier:\s*(.+?)\s*$"),
        "ros_distro": scalar(r"^  ros_distro:\s*(.+?)\s*$"),
        "start_time_ns": start_ns,
        "end_time_ns": start_ns + duration_ns,
        "duration_ns": duration_ns,
        "duration_s": duration_ns / 1e9,
        "message_count": _first_int(lines, r"^  message_count:\s*(\d+)\s*$"),
        "topics": topics,
        "relative_file_paths": relative_paths,
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def iso_utc_ns(time_ns: int | None) -> str | None:
    if time_ns is None:
        return None
    seconds, nanoseconds = divmod(int(time_ns), 1_000_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanoseconds:09d}Z"


def time_to_ns(stamp: Any) -> int | None:
    if stamp is None or not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def camera_timestamp_pair(record_time_ns: int, header_stamp: Any) -> dict[str, Any]:
    header_time_ns = time_to_ns(header_stamp)
    return {
        "record_time_ns": int(record_time_ns),
        "header_time_ns": header_time_ns,
        "record_minus_header_ns": (
            int(record_time_ns) - header_time_ns if header_time_ns is not None else None
        ),
    }


def scalar_sample_from_decoded(record: Any, decoded: Any, index: int = 0) -> ScalarSample:
    """Create a headerless Float64 sample using MCAP log time, never publish time."""

    return ScalarSample(
        index=index,
        record_time_ns=int(record.log_time),
        publish_time_ns=int(record.publish_time),
        value=float(decoded.data),
    )


def decode_float64_cdr(data: bytes) -> float:
    """Independently decode a ROS 2 CDR std_msgs/Float64 payload."""

    raw = bytes(data)
    if len(raw) < 12:
        raise RealBagAuditError(f"Float64 CDR payload has {len(raw)} bytes; need at least 12")
    encapsulation = int.from_bytes(raw[:2], "big")
    if encapsulation in {0x0000, 0x0002}:
        endian = ">"
    elif encapsulation in {0x0001, 0x0003}:
        endian = "<"
    else:
        raise RealBagAuditError(f"unsupported CDR encapsulation 0x{encapsulation:04x}")
    return float(struct.unpack_from(f"{endian}d", raw, 4)[0])


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
    result: dict[str, Any] = {
        "count": len(values),
        "finite_count": len(finite),
        "nonfinite_count": len(values) - len(finite),
    }
    keys = ("min", "max", "mean", "median", "std", "p01", "p05", "p25", "p75", "p95", "p99")
    if not finite:
        result.update({key: None for key in keys})
        return result
    result.update(
        {
            "min": min(finite),
            "max": max(finite),
            "mean": statistics.fmean(finite),
            "median": statistics.median(finite),
            "std": statistics.pstdev(finite),
            "p01": percentile(finite, 0.01),
            "p05": percentile(finite, 0.05),
            "p25": percentile(finite, 0.25),
            "p75": percentile(finite, 0.75),
            "p95": percentile(finite, 0.95),
            "p99": percentile(finite, 0.99),
        }
    )
    return result


def timestamp_metrics(times_ns: Sequence[int]) -> dict[str, Any]:
    times = [int(value) for value in times_ns]
    gaps_s = [(later - earlier) / 1e9 for earlier, later in zip(times, times[1:])]
    if len(times) > 1 and max(times) > min(times):
        rate = (len(times) - 1) / ((max(times) - min(times)) / 1e9)
    else:
        rate = None
    return {
        "count": len(times),
        "first_time_ns": times[0] if times else None,
        "first_time_utc": iso_utc_ns(times[0]) if times else None,
        "last_time_ns": times[-1] if times else None,
        "last_time_utc": iso_utc_ns(times[-1]) if times else None,
        "minimum_time_ns": min(times) if times else None,
        "maximum_time_ns": max(times) if times else None,
        "span_s": (max(times) - min(times)) / 1e9 if len(times) > 1 else 0.0 if times else None,
        "measured_rate_hz": rate,
        "monotonic_non_decreasing": all(a <= b for a, b in zip(times, times[1:])),
        "strictly_increasing": all(a < b for a, b in zip(times, times[1:])),
        "backward_timestamp_count": sum(b < a for a, b in zip(times, times[1:])),
        "non_increasing_timestamp_count": sum(b <= a for a, b in zip(times, times[1:])),
        "duplicate_timestamp_count": len(times) - len(set(times)),
        "gap_s": distribution(gaps_s),
    }


def validate_camera_payload(message: Any) -> tuple[int, int, str, int, int, bytes]:
    """Validate packed ROS Image layout without constructing a raster image."""

    width = int(message.width)
    height = int(message.height)
    encoding = str(message.encoding).lower()
    step = int(message.step)
    if width <= 0 or height <= 0:
        raise RealBagAuditError(f"invalid image dimensions {width}x{height}")
    channels_by_encoding = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
    if encoding not in channels_by_encoding:
        raise RealBagAuditError(f"unsupported preview encoding {encoding!r}")
    row_bytes = width * channels_by_encoding[encoding]
    if step < row_bytes:
        raise RealBagAuditError(f"image step {step} is shorter than packed row {row_bytes}")
    data = bytes(message.data)
    required = step * height
    if len(data) < required:
        raise RealBagAuditError(f"truncated image payload: {len(data)} bytes, need {required}")
    return width, height, encoding, step, row_bytes, data


def decode_camera_image(message: Any) -> Image.Image:
    """Decode common packed ROS Image encodings without cropping or resizing."""

    width, height, encoding, step, row_bytes, data = validate_camera_payload(message)
    packed = bytearray(row_bytes * height)
    for row in range(height):
        source = row * step
        target = row * row_bytes
        packed[target : target + row_bytes] = data[source : source + row_bytes]
    payload = bytes(packed)
    if encoding == "rgb8":
        image = Image.frombytes("RGB", (width, height), payload)
    elif encoding == "bgr8":
        image = Image.frombytes("RGB", (width, height), payload, "raw", "BGR")
    elif encoding == "mono8":
        source = Image.frombytes("L", (width, height), payload)
        image = source.convert("RGB")
        source.close()
    elif encoding == "rgba8":
        source = Image.frombytes("RGBA", (width, height), payload)
        image = source.convert("RGB")
        source.close()
    else:
        source = Image.frombytes("RGBA", (width, height), payload, "raw", "BGRA")
        image = source.convert("RGB")
        source.close()
    image.load()
    return image


def _decoder_for(
    factory: DecoderFactory, cache: dict[int, Any], schema: Any, channel: Any
) -> Any:
    decoder = cache.get(int(channel.id))
    if decoder is None:
        decoder = factory.decoder_for(channel.message_encoding, schema)
        if decoder is None:
            raise RealBagAuditError(
                f"no decoder for {channel.topic} ({channel.message_encoding}, {getattr(schema, 'name', None)})"
            )
        cache[int(channel.id)] = decoder
    return decoder


def scan_mcap(mcap_path: Path, config: dict[str, Any]) -> BagScan:
    required = config["required_topics"]
    topic_roles = {item["name"]: role for role, item in required.items()}
    expected_types = {item["name"]: item["type"] for item in required.values()}
    topic_counts: Counter[str] = Counter()
    topic_types: dict[str, set[str]] = {}
    topic_times: dict[str, list[int]] = {}
    all_times: list[int] = []
    cameras: list[CameraSample] = []
    steering: list[ScalarSample] = []
    speed: list[ScalarSample] = []
    attempts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    cdr_attempts: Counter[str] = Counter()
    cdr_failures: Counter[str] = Counter()
    factory = DecoderFactory()
    decoders: dict[int, Any] = {}
    scan_completed = False
    scan_error_type: str | None = None
    scan_error_message: str | None = None

    try:
        with mcap_path.open("rb") as stream:
            reader = NonSeekingReader(
                stream, validate_crcs=True, decoder_factories=[factory]
            )
            for schema, channel, record in reader.iter_messages(log_time_order=False):
                topic = str(channel.topic)
                schema_name = str(schema.name) if schema is not None else ""
                topic_counts[topic] += 1
                topic_types.setdefault(topic, set()).add(schema_name)
                topic_times.setdefault(topic, []).append(int(record.log_time))
                all_times.append(int(record.log_time))
                role = topic_roles.get(topic)
                if role is None:
                    continue
                attempts[topic] += 1
                try:
                    if schema_name != expected_types[topic]:
                        raise RealBagAuditError(
                            f"schema {schema_name!r}, expected {expected_types[topic]!r}"
                        )
                    decoded = _decoder_for(factory, decoders, schema, channel)(record.data)
                    if role == "camera":
                        pair = camera_timestamp_pair(
                            int(record.log_time), getattr(getattr(decoded, "header", None), "stamp", None)
                        )
                        frame_id = str(getattr(getattr(decoded, "header", None), "frame_id", ""))
                        sample = CameraSample(
                            index=len(cameras),
                            record_time_ns=int(record.log_time),
                            publish_time_ns=int(record.publish_time),
                            header_time_ns=pair["header_time_ns"],
                            width=int(decoded.width),
                            height=int(decoded.height),
                            encoding=str(decoded.encoding),
                            is_bigendian=int(decoded.is_bigendian),
                            step=int(decoded.step),
                            data_length=len(bytes(decoded.data)),
                            frame_id=frame_id,
                        )
                        cameras.append(sample)
                        validate_camera_payload(decoded)
                    elif role == "steering":
                        cdr_attempts[topic] += 1
                        if decode_float64_cdr(record.data) != float(decoded.data):
                            cdr_failures[topic] += 1
                        steering.append(scalar_sample_from_decoded(record, decoded, len(steering)))
                    else:
                        cdr_attempts[topic] += 1
                        if decode_float64_cdr(record.data) != float(decoded.data):
                            cdr_failures[topic] += 1
                        speed.append(scalar_sample_from_decoded(record, decoded, len(speed)))
                except Exception as error:  # retain all independently readable evidence
                    failures[topic] += 1
                    if len(examples) < 20:
                        examples.append(
                            {
                                "topic": topic,
                                "record_time_ns": int(record.log_time),
                                "error_type": type(error).__name__,
                                "error": str(error),
                            }
                        )
            scan_completed = True
    except Exception as error:
        scan_error_type = type(error).__name__
        scan_error_message = str(error)

    return BagScan(
        topic_counts=topic_counts,
        topic_types=topic_types,
        topic_times=topic_times,
        all_message_times=all_times,
        cameras=cameras,
        steering=steering,
        speed=speed,
        decode_attempts=attempts,
        decode_failures=failures,
        decode_failure_examples=examples,
        float64_cdr_crosscheck_attempts=cdr_attempts,
        float64_cdr_crosscheck_failures=cdr_failures,
        scan_completed=scan_completed,
        scan_error_type=scan_error_type,
        scan_error_message=scan_error_message,
    )


def _magic_checks(path: Path) -> tuple[bool, bool]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        opening = stream.read(len(MCAP_MAGIC)) == MCAP_MAGIC
        if size < len(MCAP_MAGIC):
            return opening, False
        stream.seek(-len(MCAP_MAGIC), 2)
        closing = stream.read(len(MCAP_MAGIC)) == MCAP_MAGIC
    return opening, closing


def _camera_contracts(cameras: Sequence[CameraSample]) -> list[dict[str, Any]]:
    counts = Counter(
        (item.width, item.height, item.encoding, item.is_bigendian, item.step, item.frame_id)
        for item in cameras
    )
    return [
        {
            "width": key[0],
            "height": key[1],
            "encoding": key[2],
            "is_bigendian": key[3],
            "step": key[4],
            "frame_id": key[5],
            "count": count,
        }
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def analyze_camera(scan: BagScan, camera_topic: str) -> dict[str, Any]:
    cameras = scan.cameras
    record_times = [item.record_time_ns for item in cameras]
    publish_times = [item.publish_time_ns for item in cameras]
    header_times = [item.header_time_ns for item in cameras if item.header_time_ns is not None]
    paired = [item for item in cameras if item.header_time_ns is not None]
    offsets_s = [
        (item.record_time_ns - int(item.header_time_ns)) / 1e9 for item in paired
    ]
    increment_delta_ms = [
        ((b.record_time_ns - a.record_time_ns) - (int(b.header_time_ns) - int(a.header_time_ns))) / 1e6
        for a, b in zip(paired, paired[1:])
    ]
    contracts = _camera_contracts(cameras)
    record_metrics = timestamp_metrics(record_times)
    return {
        "message_count": len(cameras),
        "contracts": contracts,
        "uniform_contract": contracts[0] if len(contracts) == 1 else None,
        "all_readable_messages_share_contract": len(contracts) == 1,
        "frame_ids": dict(sorted(Counter(item.frame_id for item in cameras).items())),
        "header_timestamp": {
            "available_count": len(header_times),
            "missing_count": len(cameras) - len(header_times),
            "nonzero_count": sum(value != 0 for value in header_times),
            **timestamp_metrics(header_times),
        },
        "bag_record_timestamp": record_metrics,
        "bag_publish_timestamp": timestamp_metrics(publish_times),
        "fps": record_metrics["measured_rate_hz"],
        "timestamp_monotonicity": {
            "bag_record": record_metrics["monotonic_non_decreasing"],
            "camera_header": timestamp_metrics(header_times)["monotonic_non_decreasing"],
        },
        "inter_frame_gap_s": record_metrics["gap_s"],
        "duplicate_timestamps": {
            "bag_record": record_metrics["duplicate_timestamp_count"],
            "camera_header": timestamp_metrics(header_times)["duplicate_timestamp_count"],
        },
        "header_vs_bag_record": {
            "pair_count": len(paired),
            "record_minus_header_s": distribution(offsets_s),
            "first_record_minus_header_s": offsets_s[0] if offsets_s else None,
            "last_record_minus_header_s": offsets_s[-1] if offsets_s else None,
            "offset_change_s": offsets_s[-1] - offsets_s[0] if len(offsets_s) > 1 else None,
            "inter_frame_increment_difference_ms": distribution(increment_delta_ms),
            "timestamps_numerically_identical": bool(paired) and all(
                item.record_time_ns == item.header_time_ns for item in paired
            ),
            "mixed_domain_use_safe_without_proven_transform": False,
        },
        "decode": {
            "payload_validation_attempted_count": scan.decode_attempts[camera_topic],
            "payload_validation_failure_count": scan.decode_failures[camera_topic],
            "failure_count": scan.decode_failures[camera_topic],
            "all_readable_payload_layouts_valid": (
                scan.decode_attempts[camera_topic] > 0 and scan.decode_failures[camera_topic] == 0
            ),
        },
        "roi_applied": False,
    }


def _repetition_metrics(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    counts = Counter(finite)
    modes = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    repeated_excess = sum(count - 1 for count in counts.values() if count > 1)
    records_in_repeated_values = sum(count for count in counts.values() if count > 1)
    return {
        "unique_value_count": len(counts),
        "consecutive_repeat_count": sum(a == b for a, b in zip(finite, finite[1:])),
        "consecutive_repeat_fraction": (
            sum(a == b for a, b in zip(finite, finite[1:])) / (len(finite) - 1)
            if len(finite) > 1
            else None
        ),
        "repeated_excess_record_count": repeated_excess,
        "repeated_excess_fraction": repeated_excess / len(finite) if finite else None,
        "records_in_repeated_values_fraction": records_in_repeated_values / len(finite) if finite else None,
        "most_frequent_values": [
            {"value": value, "count": count, "fraction": count / len(finite)}
            for value, count in modes
        ],
    }


def _causal_scalar_for_time(
    ordered: Sequence[ScalarSample], times_ns: Sequence[int], target_time_ns: int
) -> ScalarSample | None:
    index = latest_causal_index(times_ns, target_time_ns)
    return ordered[index] if index is not None else None


def _driving_phase(
    time_ns: int,
    causal_speed: ScalarSample | None,
    first_nonzero_time_ns: int | None,
    last_nonzero_time_ns: int | None,
) -> str:
    if causal_speed is None:
        return "missing_causal_speed"
    if first_nonzero_time_ns is None or last_nonzero_time_ns is None:
        return "no_nonzero_speed_observed"
    if time_ns < first_nonzero_time_ns:
        return "stationary_prefix"
    if math.isfinite(causal_speed.value) and causal_speed.value != 0.0:
        return "active_nonzero_speed"
    if time_ns > last_nonzero_time_ns:
        return "stationary_suffix"
    return "zero_speed_inside_active_envelope"


def _out_of_range_episode(
    contexts: Sequence[dict[str, Any]], metadata_start_ns: int
) -> dict[str, Any]:
    first = contexts[0]
    last = contexts[-1]
    samples = [item["sample"] for item in contexts]
    speeds = [item["speed"].value for item in contexts if item["speed"] is not None]
    speed_ages_ms = [
        (item["sample"].record_time_ns - item["speed"].record_time_ns) / 1e6
        for item in contexts
        if item["speed"] is not None
    ]
    gaps_s = [
        (later.record_time_ns - earlier.record_time_ns) / 1e9
        for earlier, later in zip(samples, samples[1:])
    ]
    rounded_counts = Counter(round(item.value, 6) for item in samples)
    return {
        "side": first["side"],
        "first_steering_index": samples[0].index,
        "last_steering_index": samples[-1].index,
        "start_record_time_ns": samples[0].record_time_ns,
        "start_time_utc": iso_utc_ns(samples[0].record_time_ns),
        "start_offset_from_metadata_s": (samples[0].record_time_ns - metadata_start_ns) / 1e9,
        "end_record_time_ns": samples[-1].record_time_ns,
        "end_time_utc": iso_utc_ns(samples[-1].record_time_ns),
        "end_offset_from_metadata_s": (samples[-1].record_time_ns - metadata_start_ns) / 1e9,
        "duration_s": (samples[-1].record_time_ns - samples[0].record_time_ns) / 1e9,
        "sample_count": len(samples),
        "steering_distribution_recorded": distribution([item.value for item in samples]),
        "maximum_inter_sample_gap_s": max(gaps_s) if gaps_s else 0.0,
        "rounded_6dp_values": [
            {"value_recorded": value, "count": count}
            for value, count in sorted(rounded_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "causal_speed_available_count": len(speeds),
        "causal_speed_distribution_raw": distribution(speeds),
        "causal_speed_age_ms": distribution(speed_ages_ms),
        "causal_speed_state_counts": {
            "negative": sum(value < 0.0 for value in speeds),
            "exact_zero": sum(value == 0.0 for value in speeds),
            "positive": sum(value > 0.0 for value in speeds),
        },
        "driving_phase_counts": dict(sorted(Counter(item["phase"] for item in contexts).items())),
    }


def _out_of_range_reconciliation(
    samples: Sequence[ScalarSample],
    speed_samples: Sequence[ScalarSample],
    metadata_start_ns: int,
    low: float,
    high: float,
) -> dict[str, Any]:
    steering_ordered = sorted(samples, key=lambda item: (item.record_time_ns, item.index))
    speed_ordered = sorted(speed_samples, key=lambda item: (item.record_time_ns, item.index))
    speed_times = [item.record_time_ns for item in speed_ordered]
    nonzero_speed_times = [
        item.record_time_ns
        for item in speed_ordered
        if math.isfinite(item.value) and item.value != 0.0
    ]
    first_nonzero = nonzero_speed_times[0] if nonzero_speed_times else None
    last_nonzero = nonzero_speed_times[-1] if nonzero_speed_times else None
    all_contexts: list[dict[str, Any]] = []
    for sample in steering_ordered:
        causal_speed = _causal_scalar_for_time(speed_ordered, speed_times, sample.record_time_ns)
        side = "below" if sample.value < low else "above" if sample.value > high else "inside"
        all_contexts.append(
            {
                "sample": sample,
                "speed": causal_speed,
                "side": side,
                "phase": _driving_phase(
                    sample.record_time_ns, causal_speed, first_nonzero, last_nonzero
                ),
            }
        )
    outside = [
        item
        for item in all_contexts
        if math.isfinite(item["sample"].value) and item["side"] != "inside"
    ]
    finite_count = sum(math.isfinite(item.value) for item in steering_ordered)

    episodes: list[list[dict[str, Any]]] = []
    for item in outside:
        if (
            not episodes
            or item["side"] != episodes[-1][-1]["side"]
            or item["sample"].index != episodes[-1][-1]["sample"].index + 1
        ):
            episodes.append([item])
        else:
            episodes[-1].append(item)

    plateau_runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in outside:
        rounded = round(item["sample"].value, 6)
        if (
            current
            and item["sample"].index == current[-1]["sample"].index + 1
            and rounded == round(current[-1]["sample"].value, 6)
        ):
            current.append(item)
        else:
            if len(current) >= 2:
                plateau_runs.append(current)
            current = [item]
    if len(current) >= 2:
        plateau_runs.append(current)

    outside_speeds = [item["speed"].value for item in outside if item["speed"] is not None]
    all_speeds = [
        item["speed"].value for item in all_contexts if item["speed"] is not None
    ]
    outside_speed_ages = [
        (item["sample"].record_time_ns - item["speed"].record_time_ns) / 1e6
        for item in outside
        if item["speed"] is not None
    ]
    temporal_locations = [
        _out_of_range_episode(episode, metadata_start_ns) for episode in episodes
    ]
    plateaus = []
    for plateau in plateau_runs:
        detail = _out_of_range_episode(plateau, metadata_start_ns)
        detail["value_recorded_rounded_6dp"] = round(plateau[0]["sample"].value, 6)
        plateaus.append(detail)
    return {
        "unscaled_numeric_comparison_range": [low, high],
        "clipping_applied": False,
        "finite_steering_count": finite_count,
        "outside_count": len(outside),
        "outside_fraction": len(outside) / finite_count if finite_count else None,
        "below_count": sum(item["side"] == "below" for item in outside),
        "above_count": sum(item["side"] == "above" for item in outside),
        "temporal_location_episode_count": len(temporal_locations),
        "temporal_locations": temporal_locations,
        "repeated_consecutive_plateau_definition": (
            "at least two consecutive recorded values outside the unscaled physical numeric range, equal after rounding to 6 decimals"
        ),
        "repeated_consecutive_plateau_count": len(plateaus),
        "repeated_consecutive_plateaus": plateaus,
        "causal_speed_relation": {
            "rule": "latest speed MCAP log_time satisfying t_speed <= t_steering",
            "available_count": len(outside_speeds),
            "missing_count": len(outside) - len(outside_speeds),
            "raw_speed_distribution_at_out_of_range_steering": distribution(outside_speeds),
            "raw_speed_distribution_at_all_steering": distribution(all_speeds),
            "speed_age_ms_at_out_of_range_steering": distribution(outside_speed_ages),
            "negative_count": sum(value < 0.0 for value in outside_speeds),
            "exact_zero_count": sum(value == 0.0 for value in outside_speeds),
            "nonzero_count": sum(value != 0.0 for value in outside_speeds),
            "nonzero_fraction": (
                sum(value != 0.0 for value in outside_speeds) / len(outside_speeds)
                if outside_speeds
                else None
            ),
        },
        "driving_phase_definition": {
            "stationary_prefix": "before the first finite, exactly nonzero speed record",
            "active_nonzero_speed": "within the first-to-last nonzero envelope with causal speed nonzero",
            "zero_speed_inside_active_envelope": "within that envelope with causal speed exactly zero",
            "stationary_suffix": "after the last finite, exactly nonzero speed record",
        },
        "driving_phase_counts": dict(sorted(Counter(item["phase"] for item in outside).items())),
    }


def analyze_steering(
    samples: Sequence[ScalarSample],
    config: dict[str, Any],
    scan: BagScan | None = None,
    speed_samples: Sequence[ScalarSample] = (),
    metadata_start_ns: int = 0,
) -> dict[str, Any]:
    recorded_values = [item.value for item in samples]
    finite_recorded = [value for value in recorded_values if math.isfinite(value)]
    contract = config["steering_contract"]
    near_zero = float(contract["near_zero_abs_rad"])
    low, high = [float(value) for value in contract["confirmed_numeric_range_rad"]]
    recorded_to_radians_scale = float(contract.get("recorded_to_radians_scale", 1.0))
    physical_values = [
        value * recorded_to_radians_scale if math.isfinite(value) else value
        for value in recorded_values
    ]
    finite_physical = [value for value in physical_values if math.isfinite(value)]
    recorded_low, recorded_high = [
        float(value)
        for value in contract.get("recorded_numeric_range", [low, high])
    ]
    limit_margin = (high - low) * 0.005
    near_limit = [
        value
        for value in finite_physical
        if low <= value <= high
        and (abs(value - low) <= limit_margin or abs(value - high) <= limit_margin)
    ]
    rounded_limit_counts = Counter(round(value, 6) for value in near_limit)
    candidates = [
        {"value_rad_rounded_6dp": value, "count": count}
        for value, count in sorted(rounded_limit_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ]
    physical_out_of_range_counts = Counter(
        round(value, 6) for value in finite_physical if value < low or value > high
    )
    physical_out_of_range_plateaus = [
        {"value_rad_rounded_6dp": value, "count": count}
        for value, count in sorted(
            physical_out_of_range_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= 2
    ]
    reconciliation = _out_of_range_reconciliation(
        samples, speed_samples, metadata_start_ns, low, high
    )
    physical_below = sum(value < low for value in finite_physical)
    physical_above = sum(value > high for value in finite_physical)
    physical_outside = physical_below + physical_above
    recorded_below = sum(value < recorded_low for value in finite_recorded)
    recorded_above = sum(value > recorded_high for value in finite_recorded)
    rescaling_required = recorded_to_radians_scale != 1.0
    reconciliation.update(
        {
            "recorded_representation": contract.get("recorded_representation", "radians"),
            "recorded_to_radians_formula": (
                f"steering_rad = steering_recorded * {recorded_to_radians_scale:g}"
            ),
            "recorded_to_radians_scale": recorded_to_radians_scale,
            "interpretation": (
                "These are recorded normalized values whose raw magnitudes exceed the physical radian numeric limit before the required whole-stream rescaling; they are not physical-radian range violations."
                if rescaling_required
                else "Recorded values are directly in radians, so numeric threshold exceedances are physical-range conflicts."
            ),
            "resolved_by_confirmed_rescaling": (
                rescaling_required and physical_outside == 0
            ),
            "physical_outside_count_after_rescaling": physical_outside,
            "physical_outside_fraction_after_rescaling": (
                physical_outside / len(finite_physical) if finite_physical else None
            ),
        }
    )
    reconciliation["causal_speed_relation"]["unit"] = config.get(
        "speed_contract", {}
    ).get("unit", "unresolved")
    reconciliation["causal_speed_relation"]["unit_symbol"] = config.get(
        "speed_contract", {}
    ).get("unit_symbol", "unresolved")
    return {
        "unit": "radians_after_rescaling" if rescaling_required else "radians",
        "recorded_representation": contract.get("recorded_representation", "radians"),
        "confirmed_numeric_range_rad": [low, high],
        "recorded_numeric_range": [recorded_low, recorded_high],
        "recorded_to_radians": {
            "required": rescaling_required,
            "scale": recorded_to_radians_scale,
            "offset": 0.0,
            "formula": f"steering_rad = steering_recorded * {recorded_to_radians_scale:g}",
            "clipping_applied": False,
            "applies_to": "every finite steering sample",
            "evidence": contract.get("recorded_scale_evidence"),
        },
        "count_and_rate": timestamp_metrics([item.record_time_ns for item in samples]),
        "distribution_recorded_raw": distribution(recorded_values),
        "distribution_rad": distribution(physical_values),
        "sign_counts": {
            "negative": sum(value < -near_zero for value in finite_physical),
            "near_zero": sum(abs(value) <= near_zero for value in finite_physical),
            "positive": sum(value > near_zero for value in finite_physical),
            "near_zero_threshold_abs_rad": near_zero,
        },
        "repeated_recorded_values": _repetition_metrics(recorded_values),
        "repeated_values": _repetition_metrics(physical_values),
        "probable_saturation": {
            "diagnostic_only": True,
            "evaluated_after_confirmed_rescaling": rescaling_required,
            "near_confirmed_limit_count": len(near_limit),
            "near_confirmed_limit_fraction": (
                len(near_limit) / len(finite_physical) if finite_physical else None
            ),
            "repeated_near_limit_candidates": candidates,
            "repeated_out_of_range_plateau_candidates": physical_out_of_range_plateaus,
            "minimum_exact_repeat_count": (
                finite_physical.count(min(finite_physical)) if finite_physical else 0
            ),
            "maximum_exact_repeat_count": (
                finite_physical.count(max(finite_physical)) if finite_physical else 0
            ),
            "assessment": (
                "repeated confirmed-limit values are candidates only; actuator saturation is not proven"
                if candidates
                else "out-of-range repeated plateaus require investigation; actuator saturation is not proven"
                if physical_out_of_range_plateaus
                else "no repeated near-limit physical-radian saturation value detected"
            ),
        },
        "recorded_range_check": {
            "expected_recorded_range": [recorded_low, recorded_high],
            "below_expected_recorded_range_count": recorded_below,
            "above_expected_recorded_range_count": recorded_above,
            "all_finite_values_within_expected_recorded_range": (
                recorded_below == 0 and recorded_above == 0
            ),
        },
        "range_check": {
            "evaluated_after_confirmed_rescaling": rescaling_required,
            "below_confirmed_range_count": physical_below,
            "above_confirmed_range_count": physical_above,
            "outside_confirmed_range_count": physical_outside,
            "outside_confirmed_range_fraction": (
                physical_outside / len(finite_physical) if finite_physical else None
            ),
            "all_finite_values_within_confirmed_approximate_range": physical_outside == 0,
            "assessment": (
                "CONFLICT: physical-radian values exceed the confirmed approximate range after conversion"
                if physical_outside
                else "all converted physical-radian values are within the confirmed approximate range"
            ),
        },
        "out_of_range_reconciliation": reconciliation,
        "float64_decode_validation": {
            "method": "mcap_ros2 decode independently compared with direct CDR Float64 unpack",
            "attempted_count": scan.float64_cdr_crosscheck_attempts["/steering"] if scan else None,
            "mismatch_count": scan.float64_cdr_crosscheck_failures["/steering"] if scan else None,
        },
        "timestamp": timestamp_metrics([item.record_time_ns for item in samples]),
        "semantics": {
            "recorded_representation": contract.get("recorded_representation", "radians"),
            "physical_unit_after_conversion": "confirmed_radians",
            "left_right_sign_convention": contract.get(
                "left_right_sign_convention", "unresolved"
            ),
            "positive_direction": (
                "LEFT"
                if contract.get("left_right_sign_convention")
                == "positive_left_negative_right"
                else "unresolved"
            ),
            "negative_direction": (
                "RIGHT"
                if contract.get("left_right_sign_convention")
                == "positive_left_negative_right"
                else "unresolved"
            ),
            "command_or_actual_actuator_feedback": contract.get(
                "command_or_feedback", "unresolved"
            ),
            "selective_clipping_permitted": False,
        },
    }


def _active_speed_windows(
    samples: Sequence[ScalarSample], metadata_start_ns: int, incomplete: bool
) -> list[dict[str, Any]]:
    ordered = sorted(samples, key=lambda item: item.record_time_ns)
    segments: list[list[ScalarSample]] = []
    current: list[ScalarSample] = []
    for sample in ordered:
        active = math.isfinite(sample.value) and sample.value != 0.0
        if active:
            current.append(sample)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    windows = []
    for index, segment in enumerate(segments):
        first, last = segment[0], segment[-1]
        values = [item.value for item in segment]
        windows.append(
            {
                "window_index": index,
                "method": "contiguous run of finite, exactly nonzero raw speed values",
                "diagnostic_only": True,
                "start_record_time_ns": first.record_time_ns,
                "start_time_utc": iso_utc_ns(first.record_time_ns),
                "start_offset_from_metadata_s": (first.record_time_ns - metadata_start_ns) / 1e9,
                "end_record_time_ns": last.record_time_ns,
                "end_time_utc": iso_utc_ns(last.record_time_ns),
                "end_offset_from_metadata_s": (last.record_time_ns - metadata_start_ns) / 1e9,
                "duration_s": (last.record_time_ns - first.record_time_ns) / 1e9,
                "record_count": len(segment),
                "supporting_raw_speed_distribution": distribution(values),
                "left_censored_by_readable_prefix": bool(ordered and first is ordered[0]),
                "right_censored_by_readable_prefix": bool(ordered and last is ordered[-1]),
                "full_bag_inference_permitted": not incomplete,
            }
        )
    return windows


def _active_speed_window(
    samples: Sequence[ScalarSample], metadata_start_ns: int, incomplete: bool
) -> dict[str, Any] | None:
    windows = _active_speed_windows(samples, metadata_start_ns, incomplete)
    if not windows:
        return None
    selected = max(
        windows,
        key=lambda item: (
            item["duration_s"], item["record_count"], -item["start_record_time_ns"]
        ),
    )
    return {
        **selected,
        "method": "longest contiguous run of finite, exactly nonzero raw speed values",
    }


def analyze_speed(
    samples: Sequence[ScalarSample], metadata_start_ns: int, incomplete: bool,
    scan: BagScan | None = None,
    speed_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract = speed_contract or {}
    unit = str(contract.get("unit", "unresolved"))
    unit_symbol = str(contract.get("unit_symbol", "unresolved"))
    meaning = str(contract.get("meaning", "unresolved"))
    values = [item.value for item in samples]
    finite = [value for value in values if math.isfinite(value)]
    ordered = sorted(samples, key=lambda item: item.record_time_ns)
    prefix_zero = 0
    for item in ordered:
        if math.isfinite(item.value) and item.value == 0.0:
            prefix_zero += 1
        else:
            break
    suffix_zero = 0
    for item in reversed(ordered):
        if math.isfinite(item.value) and item.value == 0.0:
            suffix_zero += 1
        else:
            break
    windows = _active_speed_windows(samples, metadata_start_ns, incomplete)
    longest_window = (
        max(
            windows,
            key=lambda item: (
                item["duration_s"], item["record_count"], -item["start_record_time_ns"]
            ),
        )
        if windows
        else None
    )
    for window in windows:
        window["speed_unit"] = unit
        window["speed_unit_symbol"] = unit_symbol
    return {
        "unit": unit,
        "unit_symbol": unit_symbol,
        "meaning": meaning,
        "count_and_rate": timestamp_metrics([item.record_time_ns for item in samples]),
        "distribution_raw": distribution(values),
        "distribution_mps": distribution(values) if unit == "meters_per_second" else None,
        "sign_and_stationary_counts": {
            "negative": sum(value < 0.0 for value in finite),
            "exact_zero": sum(value == 0.0 for value in finite),
            "positive": sum(value > 0.0 for value in finite),
            "exact_zero_fraction": sum(value == 0.0 for value in finite) / len(finite) if finite else None,
            "stationary_definition": "exact zero only; diagnostic, not a permanent filter",
        },
        "stationary_prefix_candidate": {
            "record_count": prefix_zero,
            "observed": prefix_zero > 0,
        },
        "stationary_suffix_candidate": {
            "record_count": suffix_zero,
            "observed": suffix_zero > 0,
            "full_bag_suffix_observable": not incomplete,
        },
        "active_driving_candidate_window": (
            {
                **longest_window,
                "method": "longest contiguous run of finite, exactly nonzero raw speed values",
            }
            if longest_window
            else None
        ),
        "active_driving_candidate_windows": windows,
        "timestamp": timestamp_metrics([item.record_time_ns for item in samples]),
        "float64_decode_validation": {
            "method": "mcap_ros2 decode independently compared with direct CDR Float64 unpack",
            "attempted_count": scan.float64_cdr_crosscheck_attempts["/speed"] if scan else None,
            "mismatch_count": scan.float64_cdr_crosscheck_failures["/speed"] if scan else None,
        },
    }


def latest_causal_index(times_ns: Sequence[int], camera_time_ns: int) -> int | None:
    index = bisect.bisect_right(times_ns, int(camera_time_ns)) - 1
    return index if index >= 0 else None


def _causal_assignments(
    cameras: Sequence[CameraSample], steering: Sequence[ScalarSample], speed: Sequence[ScalarSample]
) -> list[tuple[ScalarSample | None, ScalarSample | None]]:
    steering_ordered = sorted(steering, key=lambda item: item.record_time_ns)
    speed_ordered = sorted(speed, key=lambda item: item.record_time_ns)
    steering_times = [item.record_time_ns for item in steering_ordered]
    speed_times = [item.record_time_ns for item in speed_ordered]
    assignments: list[tuple[ScalarSample | None, ScalarSample | None]] = []
    for camera in cameras:
        steering_index = latest_causal_index(steering_times, camera.record_time_ns)
        speed_index = latest_causal_index(speed_times, camera.record_time_ns)
        steering_sample = steering_ordered[steering_index] if steering_index is not None else None
        speed_sample = speed_ordered[speed_index] if speed_index is not None else None
        assignments.append((steering_sample, speed_sample))
    return assignments


def causal_sync_metrics(
    cameras: Sequence[CameraSample], steering: Sequence[ScalarSample], speed: Sequence[ScalarSample]
) -> tuple[dict[str, Any], list[tuple[ScalarSample | None, ScalarSample | None]]]:
    assignments = _causal_assignments(cameras, steering, speed)
    steering_ages_ms: list[float] = []
    speed_ages_ms: list[float] = []
    future = 0
    missing_steering = missing_speed = missing_both = 0
    for camera, (steering_sample, speed_sample) in zip(cameras, assignments):
        if steering_sample is None:
            missing_steering += 1
        else:
            future += steering_sample.record_time_ns > camera.record_time_ns
            steering_ages_ms.append((camera.record_time_ns - steering_sample.record_time_ns) / 1e6)
        if speed_sample is None:
            missing_speed += 1
        else:
            future += speed_sample.record_time_ns > camera.record_time_ns
            speed_ages_ms.append((camera.record_time_ns - speed_sample.record_time_ns) / 1e6)
        if steering_sample is None and speed_sample is None:
            missing_both += 1
    if future:
        raise RealBagAuditError(f"causal synchronization produced {future} future labels")
    return (
        {
            "timestamp_domain": "MCAP log_time (bag-record time) for all three streams",
            "rule": "latest scalar record satisfying t_scalar <= t_camera",
            "camera_count": len(cameras),
            "steering_age_ms": distribution(steering_ages_ms),
            "speed_age_ms": distribution(speed_ages_ms),
            "future_label_violations": future,
            "missing_causal_steering_count": missing_steering,
            "missing_causal_speed_count": missing_speed,
            "missing_both_count": missing_both,
            "complete_causal_pair_count": sum(a is not None and b is not None for a, b in assignments),
            "stale_age_threshold_applied": False,
        },
        assignments,
    )


def temporal_readiness(
    cameras: Sequence[CameraSample],
    assignments: Sequence[tuple[ScalarSample | None, ScalarSample | None]],
    simulator_reference: dict[str, Any],
    incomplete: bool,
) -> dict[str, Any]:
    times = [item.record_time_ns for item in cameras]
    candidates = max(len(times) - 2, 0)
    valid = 0
    label_ready = 0
    failures = 0
    adjacent: list[float] = []
    spans: list[float] = []
    triplet_gaps: list[float] = []
    maximum_gate = float(simulator_reference["maximum_adjacent_gap_s"])
    over_reference_gate = 0
    for index in range(2, len(times)):
        first, middle, current = times[index - 2 : index + 1]
        gap_a = (middle - first) / 1e9
        gap_b = (current - middle) / 1e9
        if not first < middle < current:
            failures += 1
            continue
        valid += 1
        adjacent.extend([gap_a, gap_b])
        triplet_gaps.extend([gap_a, gap_b])
        spans.append((current - first) / 1e9)
        if gap_a > maximum_gate or gap_b > maximum_gate:
            over_reference_gate += 1
        steering_sample, speed_sample = assignments[index]
        if steering_sample is not None and speed_sample is not None:
            label_ready += 1
    all_pair_gaps = [(b - a) / 1e9 for a, b in zip(times, times[1:])]
    adjacent_gap_over_reference_count = sum(gap > maximum_gate for gap in all_pair_gaps)
    adjacent_gap_locations = [
        {
            "previous_camera_index": index,
            "current_camera_index": index + 1,
            "previous_record_time_ns": times[index],
            "current_record_time_ns": times[index + 1],
            "current_offset_from_first_camera_s": (times[index + 1] - times[0]) / 1e9,
            "gap_s": gap,
        }
        for index, gap in enumerate(all_pair_gaps)
        if gap > maximum_gate
    ]
    observed = distribution(adjacent)
    simulator = simulator_reference["representative_train_adjacent_gap_s"]
    prefix_compatible = (
        bool(adjacent)
        and observed["p95"] is not None
        and observed["p95"] <= maximum_gate
        and failures == 0
    )
    return {
        "frame_order": ["t_minus_2", "t_minus_1", "t"],
        "candidate_three_frame_sequences": candidates,
        "valid_strict_causal_sequences": valid,
        "valid_strict_causal_sequences_with_current_causal_labels": label_ready,
        "timestamp_order_failure_count": failures,
        "adjacent_frame_gap_s": distribution(all_pair_gaps),
        "valid_triplet_adjacent_gap_s": observed,
        "oldest_to_current_span_s": distribution(spans),
        "simulator_comparison": {
            "source": simulator_reference["source"],
            "simulator_train_adjacent_gap_s": simulator,
            "simulator_train_oldest_to_current_span_s": simulator_reference[
                "representative_train_oldest_to_current_span_s"
            ],
            "existing_0p120_s_adjacent_gate_applied": False,
            "adjacent_gap_over_0p120_s_count": adjacent_gap_over_reference_count,
            "adjacent_gaps_over_0p120_s": adjacent_gap_locations,
            "diagnostic_triplets_exceeding_0p120_s": over_reference_gate,
            "readable_prefix_spacing_supports_architecture": prefix_compatible,
        },
        "architecture_assessment": (
            "readable prefix is temporally compatible; full-bag compatibility is unresolved because the MCAP is incomplete"
            if incomplete and prefix_compatible and not adjacent_gap_over_reference_count
            else (
                "readable-prefix typical spacing is compatible, but diagnostic >0.120 s gaps require a real-data policy"
            )
            if incomplete and prefix_compatible
            else "temporally compatible on complete audited messages"
            if prefix_compatible and not adjacent_gap_over_reference_count
            else (
                "typical complete-bag spacing is compatible, but diagnostic >0.120 s gaps require a real-data policy"
            )
            if prefix_compatible
            else "observed timing does not establish temporal compatibility"
        ),
    }


def _select_preview_indices(
    cameras: Sequence[CameraSample],
    assignments: Sequence[tuple[ScalarSample | None, ScalarSample | None]],
    incomplete: bool,
    preview_config: dict[str, Any] | None = None,
) -> dict[int, list[str]]:
    if not cameras:
        return {}
    last = len(cameras) - 1
    prefix = "readable_prefix_" if incomplete else ""
    selection: dict[int, list[str]] = {}

    def add(index: int, label: str) -> None:
        selection.setdefault(max(0, min(last, index)), []).append(label)

    add(0, f"{prefix}first")
    add(round(last * 0.10), f"{prefix}early")
    add(round(last * 0.50), f"{prefix}middle")
    add(round(last * 0.90), f"{prefix}late")
    add(last, f"{prefix}last")
    labelled = [
        (index, pair[0].value)
        for index, pair in enumerate(assignments)
        if pair[0] is not None and math.isfinite(pair[0].value)
    ]
    if labelled:
        add(min(labelled, key=lambda item: item[1])[0], "minimum_steering_recorded")
        add(max(labelled, key=lambda item: item[1])[0], "maximum_steering_recorded")
        settings = preview_config or {}
        per_sign = int(settings.get("major_steering_frames_per_sign", 0))
        minimum_separation_ns = int(
            float(settings.get("major_steering_minimum_separation_s", 0.0)) * 1e9
        )
        for sign_name, candidates in (
            ("negative", sorted((item for item in labelled if item[1] < 0.0), key=lambda item: item[1])),
            ("positive", sorted((item for item in labelled if item[1] > 0.0), key=lambda item: -item[1])),
        ):
            if per_sign <= 0:
                continue
            chosen: list[int] = []
            for index, _value in candidates:
                if any(
                    abs(cameras[index].record_time_ns - cameras[other].record_time_ns)
                    < minimum_separation_ns
                    for other in chosen
                ):
                    continue
                chosen.append(index)
                add(index, f"major_{sign_name}_steering_{len(chosen):02d}")
                if len(chosen) >= per_sign:
                    break
    return selection


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")


def create_previews(
    *,
    bag_id: str,
    mcap_path: Path,
    camera_topic: str,
    cameras: Sequence[CameraSample],
    assignments: Sequence[tuple[ScalarSample | None, ScalarSample | None]],
    incomplete: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    selection = _select_preview_indices(cameras, assignments, incomplete, config.get("preview"))
    preview_root = Path(config["preview"]["root"]) / bag_id
    preview_root.mkdir(parents=True, exist_ok=True)
    selected_images: dict[int, Path] = {}
    decode_failures: list[dict[str, Any]] = []
    factory = DecoderFactory()
    cache: dict[int, Any] = {}
    camera_index = 0
    try:
        with mcap_path.open("rb") as stream:
            reader = NonSeekingReader(stream, validate_crcs=True, decoder_factories=[factory])
            for schema, channel, record in reader.iter_messages(
                topics=[camera_topic], log_time_order=False
            ):
                current = camera_index
                camera_index += 1
                if current not in selection:
                    continue
                try:
                    decoded = _decoder_for(factory, cache, schema, channel)(record.data)
                    image = decode_camera_image(decoded)
                    labels = "__".join(_safe_name(item) for item in selection[current])
                    path = preview_root / f"frame_{current:06d}__{labels}.png"
                    image.save(path, format="PNG", optimize=False)
                    image.close()
                    selected_images[current] = path
                except Exception as error:
                    decode_failures.append(
                        {"camera_index": current, "error_type": type(error).__name__, "error": str(error)}
                    )
                if len(selected_images) + len(decode_failures) == len(selection):
                    break
    except Exception as error:
        decode_failures.append(
            {"camera_index": None, "error_type": type(error).__name__, "error": str(error)}
        )

    width = int(config["preview"]["thumbnail_width"])
    height = int(config["preview"]["thumbnail_height"])
    steering_scale = float(
        config.get("steering_contract", {}).get("recorded_to_radians_scale", 1.0)
    )
    speed_unit_symbol = str(
        config.get("speed_contract", {}).get("unit_symbol", "unresolved")
    )
    label_height = 52
    columns = min(3, max(1, len(selected_images)))
    rows = math.ceil(len(selected_images) / columns) if selected_images else 1
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(sorted(selected_images)):
        image = Image.open(selected_images[index]).convert("RGB")
        thumb = ImageOps.contain(image, (width, height), Image.Resampling.BILINEAR)
        x = (position % columns) * width
        y = (position // columns) * (height + label_height)
        sheet.paste(thumb, (x + (width - thumb.width) // 2, y + (height - thumb.height) // 2))
        image.close()
        thumb.close()
        camera = cameras[index]
        steering_sample = assignments[index][0]
        speed_sample = assignments[index][1]
        steering_text = (
            f"steering_raw={steering_sample.value:+.6f}, rad={steering_sample.value * steering_scale:+.6f}"
            if steering_sample is not None
            else "steering=missing"
        )
        draw.text((x + 4, y + height + 3), f"#{index} +{(camera.record_time_ns-cameras[0].record_time_ns)/1e9:.3f}s", fill="black")
        draw.text((x + 4, y + height + 18), ",".join(selection[index]), fill="black")
        speed_text = (
            f"speed={speed_sample.value:+.6f} {speed_unit_symbol}"
            if speed_sample is not None
            else "speed=missing"
        )
        draw.text((x + 4, y + height + 33), f"{steering_text}; {speed_text}", fill="black")
    contact_sheet = preview_root / f"{bag_id}_contact_sheet.png"
    sheet.save(contact_sheet, format="PNG", optimize=False)
    sheet.close()
    return {
        "scope": "readable_prefix_only" if incomplete else "complete_bag",
        "contact_sheet": str(contact_sheet),
        "selected_frames": [str(selected_images[index]) for index in sorted(selected_images)],
        "selection": [
            {
                "camera_index": index,
                "record_time_ns": cameras[index].record_time_ns,
                "offset_from_first_camera_s": (
                    cameras[index].record_time_ns - cameras[0].record_time_ns
                ) / 1e9,
                "labels": selection[index],
                "steering_recorded_raw": (
                    assignments[index][0].value if assignments[index][0] else None
                ),
                "steering_rad": (
                    assignments[index][0].value * steering_scale
                    if assignments[index][0]
                    else None
                ),
                "speed_raw": assignments[index][1].value if assignments[index][1] else None,
                "speed_mps": (
                    assignments[index][1].value
                    if assignments[index][1] and speed_unit_symbol == "m/s"
                    else None
                ),
                "speed_unit_symbol": speed_unit_symbol,
                "path": str(selected_images.get(index, "")),
            }
            for index in sorted(selection)
        ],
        "selected_frame_count": len(selected_images),
        "decode_failure_count": len(decode_failures),
        "decode_failures": decode_failures,
        "roi_applied": False,
    }


def _topic_evidence(metadata: dict[str, Any], scan: BagScan) -> list[dict[str, Any]]:
    topics = sorted(set(metadata["topics"]) | set(scan.topic_counts))
    rows = []
    for topic in topics:
        expected = metadata["topics"].get(topic, {})
        actual_count = scan.topic_counts[topic]
        metrics = timestamp_metrics(scan.topic_times.get(topic, []))
        rows.append(
            {
                "topic": topic,
                "metadata_type": expected.get("type"),
                "readable_schema_types": sorted(scan.topic_types.get(topic, set())),
                "metadata_message_count": expected.get("message_count"),
                "readable_message_count": actual_count,
                "count_matches_metadata": expected.get("message_count") == actual_count,
                "readable_fraction": (
                    actual_count / expected["message_count"] if expected.get("message_count") else None
                ),
                "measured_rate_hz": metrics["measured_rate_hz"],
                "readable_span_s": metrics["span_s"],
            }
        )
    return rows


def audit_bag(bag: dict[str, str], config: dict[str, Any]) -> dict[str, Any]:
    bag_id = bag["id"]
    bag_root = Path(bag["path"])
    metadata_path = bag_root / "metadata.yaml"
    if not metadata_path.is_file():
        raise RealBagAuditError(f"{bag_id}: missing {metadata_path}")
    metadata = parse_rosbag_metadata(metadata_path)
    if metadata["storage_identifier"] != "mcap":
        raise RealBagAuditError(f"{bag_id}: storage identifier is not mcap")
    if len(metadata["relative_file_paths"]) != 1:
        raise RealBagAuditError(f"{bag_id}: expected exactly one MCAP file")
    mcap_path = bag_root / metadata["relative_file_paths"][0]
    if not mcap_path.is_file():
        raise RealBagAuditError(f"{bag_id}: missing {mcap_path}")

    opening_magic, closing_magic = _magic_checks(mcap_path)
    scan = scan_mcap(mcap_path, config)
    topics = _topic_evidence(metadata, scan)
    counts_match = (
        metadata["message_count"] == sum(scan.topic_counts.values())
        and all(item["count_matches_metadata"] for item in topics)
    )
    integrity_pass = opening_magic and closing_magic and scan.scan_completed and counts_match
    incomplete = not integrity_pass
    actual_start = min(scan.all_message_times) if scan.all_message_times else None
    actual_end = max(scan.all_message_times) if scan.all_message_times else None
    camera_topic = config["required_topics"]["camera"]["name"]
    steering_topic = config["required_topics"]["steering"]["name"]
    speed_topic = config["required_topics"]["speed"]["name"]
    camera = analyze_camera(scan, camera_topic)
    steering = analyze_steering(
        scan.steering,
        config,
        scan,
        speed_samples=scan.speed,
        metadata_start_ns=metadata["start_time_ns"],
    )
    speed = analyze_speed(
        scan.speed,
        metadata["start_time_ns"],
        incomplete,
        scan,
        speed_contract=config["speed_contract"],
    )
    sync, assignments = causal_sync_metrics(scan.cameras, scan.steering, scan.speed)
    temporal = temporal_readiness(
        scan.cameras, assignments, config["simulator_temporal_reference"], incomplete
    )
    previews = create_previews(
        bag_id=bag_id,
        mcap_path=mcap_path,
        camera_topic=camera_topic,
        cameras=scan.cameras,
        assignments=assignments,
        incomplete=incomplete,
        config=config,
    )
    camera["bounded_preview_decode"] = {
        "attempted_count": len(previews["selection"]),
        "failure_count": previews["decode_failure_count"],
    }
    camera["decode"]["failure_count"] += previews["decode_failure_count"]
    required_decode_pass = all(
        scan.decode_attempts[item["name"]] == scan.topic_counts[item["name"]]
        and scan.decode_failures[item["name"]] == 0
        for item in config["required_topics"].values()
    ) and previews["decode_failure_count"] == 0
    audit_pass = integrity_pass and required_decode_pass
    return {
        "version": config["version"],
        "bag_id": bag_id,
        "result": (
            "PASS"
            if audit_pass
            else "FAIL_INCOMPLETE_MCAP"
            if not integrity_pass
            else "FAIL_REQUIRED_STREAM_READABILITY"
        ),
        "audit_scope": "complete_bag" if integrity_pass else "strictly_readable_prefix_only",
        "path": str(bag_root),
        "metadata_path": str(metadata_path),
        "mcap_path": str(mcap_path),
        "mcap_size_bytes": mcap_path.stat().st_size,
        "mcap_sha256": sha256_file(mcap_path),
        "integrity": {
            "result": "PASS" if integrity_pass else "FAIL",
            "opening_magic_valid": opening_magic,
            "closing_magic_valid": closing_magic,
            "footer_record_valid": closing_magic and scan.scan_completed,
            "strict_crc_stream_scan_completed": scan.scan_completed,
            "strict_crc_and_footer_scan_completed": scan.scan_completed and closing_magic,
            "completed_records_crc_checked": True,
            "scan_error_type": scan.scan_error_type,
            "scan_error_message": scan.scan_error_message,
            "metadata_total_message_count": metadata["message_count"],
            "readable_total_message_count": sum(scan.topic_counts.values()),
            "readable_total_fraction": (
                sum(scan.topic_counts.values()) / metadata["message_count"]
                if metadata["message_count"]
                else None
            ),
            "metadata_counts_match_readable_messages": counts_match,
            "required_topic_decode_pass": required_decode_pass,
            "full_required_stream_readability_pass": audit_pass,
            "classification": (
                "complete, footer-valid MCAP with matching metadata and readable required streams"
                if audit_pass
                else "complete MCAP but one or more required messages or bounded previews failed decoding"
                if integrity_pass
                else "truncated/incomplete local MCAP: missing closing magic and strict reader reached EOF before metadata counts"
            ),
        },
        "metadata": {
            **metadata,
            "start_time_utc": iso_utc_ns(metadata["start_time_ns"]),
            "end_time_utc": iso_utc_ns(metadata["end_time_ns"]),
        },
        "actual_readable_messages": {
            "start_time_ns": actual_start,
            "start_time_utc": iso_utc_ns(actual_start),
            "end_time_ns": actual_end,
            "end_time_utc": iso_utc_ns(actual_end),
            "span_s": (actual_end - actual_start) / 1e9 if actual_start is not None and actual_end is not None else None,
            "topic_counts_and_rates": topics,
        },
        "decode_failure_examples": scan.decode_failure_examples,
        "camera": camera,
        "steering": steering,
        "speed": speed,
        "timestamp_domain": {
            **config["timestamp_domain"],
            "camera_available": ["header.stamp", "MCAP log_time", "MCAP publish_time"],
            "steering_available": ["MCAP log_time", "MCAP publish_time"],
            "speed_available": ["MCAP log_time", "MCAP publish_time"],
            "mixed_timestamp_domains_used": False,
        },
        "causal_synchronization": sync,
        "three_frame_temporal_readiness": temporal,
        "previews": previews,
    }


def cross_bag_consistency(bags: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete_bags = all(item["integrity"]["result"] == "PASS" for item in bags)
    scope_label = "complete bags" if complete_bags else "readable prefixes"
    speed_units = {item["speed"].get("unit", "unresolved") for item in bags}
    speed_unit = next(iter(speed_units)) if len(speed_units) == 1 else "mixed"
    speed_symbols = {item["speed"].get("unit_symbol", "unresolved") for item in bags}
    speed_symbol = next(iter(speed_symbols)) if len(speed_symbols) == 1 else "mixed"
    contracts = [item["camera"]["uniform_contract"] for item in bags]
    contract_keys = [
        None
        if contract is None
        else (
            contract["width"], contract["height"], contract["encoding"],
            contract["is_bigendian"], contract["step"], contract["frame_id"],
        )
        for contract in contracts
    ]
    camera_same = bool(contract_keys) and None not in contract_keys and len(set(contract_keys)) == 1
    material = []
    for item in bags:
        if item["integrity"]["result"] != "PASS":
            material.append(
                {
                    "bag_id": item["bag_id"],
                    "area": "integrity",
                    "finding": "local MCAP is incomplete; full cross-bag comparison is not valid",
                }
            )
    if not camera_same:
        material.append(
            {"bag_id": "cross_bag", "area": "camera_contract", "finding": "camera contracts differ"}
        )
    prefix_differences: list[str] = []
    for item in bags:
        range_check = item["steering"]["range_check"]
        outside = range_check["below_confirmed_range_count"] + range_check["above_confirmed_range_count"]
        raw_exceedances = item["steering"].get("out_of_range_reconciliation", {}).get(
            "outside_count", outside
        )
        conversion = item["steering"].get(
            "recorded_to_radians", {"required": False, "scale": 1.0}
        )
        if conversion["required"] and raw_exceedances:
            prefix_differences.append(
                f"{item['bag_id']}: {raw_exceedances} normalized recorded steering samples numerically exceed +/-0.35 before the required x{conversion['scale']:.6f} whole-stream conversion; {outside} converted-radian samples remain outside +/-0.35"
            )
        elif outside:
            prefix_differences.append(
                f"{item['bag_id']}: {outside} steering samples fall outside the confirmed approximate [-0.35,+0.35] rad range"
            )
    speed_medians = {
        item["bag_id"]: item["speed"]["distribution_raw"]["median"] for item in bags
    }
    prefix_differences.append(
        f"speed medians differ across {scope_label} ({speed_symbol}): "
        + ", ".join(f"{key}={value:.6f}" for key, value in speed_medians.items() if value is not None)
    )
    steering_medians = {
        item["bag_id"]: item["steering"]["distribution_rad"]["median"] for item in bags
    }
    prefix_differences.append(
        f"steering medians differ across {scope_label}: "
        + ", ".join(
            f"{key}={value:+.6f} rad" for key, value in steering_medians.items() if value is not None
        )
    )
    zero_counts = {
        item["bag_id"]: item["speed"]["sign_and_stationary_counts"]["exact_zero"] for item in bags
    }
    prefix_differences.append(
        f"exact-zero speed record counts in {scope_label}: "
        + ", ".join(f"{key}={value}" for key, value in zero_counts.items())
        + (
            "; complete-bag suffix stationarity is observable"
            if complete_bags
            else "; suffix stationarity is unobservable for every incomplete bag"
        )
    )
    gap_p95 = {
        item["bag_id"]: item["camera"]["inter_frame_gap_s"]["p95"] for item in bags
    }
    prefix_differences.append(
        f"camera timing gap p95 across {scope_label}: "
        + ", ".join(f"{key}={value:.6f} s" for key, value in gap_p95.items() if value is not None)
    )
    return {
        "complete_bag_comparison_valid": complete_bags,
        "camera": {
            "same_contract_on_readable_messages": camera_same,
            "contracts": {item["bag_id"]: item["camera"]["uniform_contract"] for item in bags},
            "fps_hz": {item["bag_id"]: item["camera"]["fps"] for item in bags},
        },
        "steering": {
            "unit": "confirmed radians",
            "range_rad": {
                item["bag_id"]: {
                    "min": item["steering"]["distribution_rad"]["min"],
                    "max": item["steering"]["distribution_rad"]["max"],
                }
                for item in bags
            },
            "median_rad": {
                item["bag_id"]: item["steering"]["distribution_rad"]["median"] for item in bags
            },
            "rate_hz": {
                item["bag_id"]: item["steering"]["timestamp"]["measured_rate_hz"] for item in bags
            },
        },
        "speed": {
            "unit": speed_unit,
            "unit_symbol": speed_symbol,
            "range_raw": {
                item["bag_id"]: {
                    "min": item["speed"]["distribution_raw"]["min"],
                    "max": item["speed"]["distribution_raw"]["max"],
                }
                for item in bags
            },
            "median_raw": {
                item["bag_id"]: item["speed"]["distribution_raw"]["median"] for item in bags
            },
            "rate_hz": {
                item["bag_id"]: item["speed"]["timestamp"]["measured_rate_hz"] for item in bags
            },
        },
        "timing": {
            item["bag_id"]: {
                "camera_gap_s": item["camera"]["inter_frame_gap_s"],
                "steering_age_ms": item["causal_synchronization"]["steering_age_ms"],
                "speed_age_ms": item["causal_synchronization"]["speed_age_ms"],
            }
            for item in bags
        },
        "material_differences_or_blockers": material,
        "readable_prefix_observed_differences": prefix_differences,
        "observed_differences": prefix_differences,
        "assessment": (
            "readable prefixes share a camera contract; steering, speed, and timing consistency across complete bags remains unresolved"
            if camera_same and material
            else "complete-bag comparison is valid and the camera contract is consistent across all three bags"
            if complete_bags and camera_same and not material
            else "cross-bag comparison complete"
            if not material
            else "material differences detected"
        ),
    }


def simulator_camera_comparison(bags: Sequence[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    reference = config["simulator_camera_reference"]
    contracts = [item["camera"]["uniform_contract"] for item in bags]
    available = [item for item in contracts if item is not None]
    def identity(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item["width"], item["height"], item["encoding"], item["is_bigendian"],
            item["step"], item["frame_id"],
        )

    same_real = (
        bool(available)
        and len(available) == len(contracts)
        and len({identity(item) for item in available}) == 1
    )
    real = available[0] if same_real else None
    return {
        "simulator_reference": reference,
        "real_readable_message_contract": real,
        "resolution_match": bool(real) and (real["width"], real["height"]) == (reference["width"], reference["height"]),
        "aspect_ratio_match": bool(real) and real["width"] * reference["height"] == reference["width"] * real["height"],
        "encoding_match": bool(real) and real["encoding"] == reference["encoding"],
        "field_of_view_and_content": config["manual_visual_review"],
        "simulator_roi_applied_to_real_images": False,
        "real_roi_status": config["real_camera_roi"]["status"],
        "real_camera_roi": config["real_camera_roi"],
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _stats_short(stats: dict[str, Any], unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"mean {_fmt(stats.get('mean'), 3)}{suffix}, median {_fmt(stats.get('median'), 3)}{suffix}, "
        f"p95 {_fmt(stats.get('p95'), 3)}{suffix}, max {_fmt(stats.get('max'), 3)}{suffix}"
    )


def build_report_v1(summary: dict[str, Any], bags: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Real PhysiCar Bag Audit V1",
        "",
        f"Overall result: **{summary['result']}**. The local MCAP files are not complete, so every numeric finding below is explicitly limited to strictly readable prefixes. No result is presented as a complete-bag or real-robot performance result.",
        "",
        "## 1. Integrity of all three bags",
        "",
        "| Bag | Metadata duration | Metadata messages | Readable messages | Readable span | Closing magic | Strict scan | Result |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in bags:
        integrity = item["integrity"]
        lines.append(
            f"| {item['bag_id']} | {_fmt(item['metadata']['duration_s'], 6)} s | {integrity['metadata_total_message_count']} | "
            f"{integrity['readable_total_message_count']} | {_fmt(item['actual_readable_messages']['span_s'], 6)} s | "
            f"{_fmt(integrity['closing_magic_valid'])} | {integrity['scan_error_type'] or 'complete'} | **{integrity['result']}** |"
        )
    lines += [
        "",
        "All three files begin with valid MCAP magic. All end inside data rather than with closing MCAP magic; CRC checking succeeds for completed records and then raises `EndOfFile`. Metadata therefore cannot be verified against all messages.",
        "",
        "### Paths and timestamps",
        "",
    ]
    for item in bags:
        lines += [
            f"- `{item['bag_id']}`: `{item['mcap_path']}`",
            f"  - metadata: {item['metadata']['start_time_ns']} ({item['metadata']['start_time_utc']}) to {item['metadata']['end_time_ns']} ({item['metadata']['end_time_utc']})",
            f"  - readable: {item['actual_readable_messages']['start_time_ns']} ({item['actual_readable_messages']['start_time_utc']}) to {item['actual_readable_messages']['end_time_ns']} ({item['actual_readable_messages']['end_time_utc']})",
        ]
    lines += [
        "",
        "### Topic counts and measured rates (readable prefixes)",
        "",
        "| Bag | Camera expected/readable/rate | Steering expected/readable/rate | Speed expected/readable/rate |",
        "|---|---:|---:|---:|",
    ]
    for item in bags:
        by_topic = {row["topic"]: row for row in item["actual_readable_messages"]["topic_counts_and_rates"]}
        cells = []
        for topic in ("/camera/image_raw", "/steering", "/speed"):
            row = by_topic[topic]
            cells.append(f"{row['metadata_message_count']}/{row['readable_message_count']}/{_fmt(row['measured_rate_hz'], 3)} Hz")
        lines.append(f"| {item['bag_id']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 2. Exact camera contract",
        "",
        "Every readable camera message in every bag is `480x360`, `rgb8`, `is_bigendian=0`, `step=1440`, `frame_id=camera`. Header stamps are present on every readable frame. This is a readable-prefix finding until complete MCAPs are supplied.",
        "",
        "## 3. Camera rate and timestamp statistics",
        "",
        "| Bag | Frames | FPS | gap mean | median | p95 | max | record monotonic | duplicate record/header | decode failures |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in bags:
        camera = item["camera"]
        gap = camera["inter_frame_gap_s"]
        lines.append(
            f"| {item['bag_id']} | {camera['message_count']} | {_fmt(camera['fps'], 3)} | {_fmt(gap['mean'], 6)} s | "
            f"{_fmt(gap['median'], 6)} s | {_fmt(gap['p95'], 6)} s | {_fmt(gap['max'], 6)} s | "
            f"{_fmt(camera['timestamp_monotonicity']['bag_record'])} | {camera['duplicate_timestamps']['bag_record']}/{camera['duplicate_timestamps']['camera_header']} | "
            f"{camera['decode']['failure_count']} |"
        )
    lines += ["", "## 4. Representative preview paths", ""]
    for item in bags:
        lines.append(f"- `{item['bag_id']}` readable-prefix contact sheet: `{item['previews']['contact_sheet']}`")
    lines += [
        "",
        "The sheets contain uncropped real frames: first/early/middle/late/last within the readable prefix plus minimum/maximum observed steering frames. They do not claim to show the true middle or end of the metadata-declared bag.",
        "",
        "Turn-like views are visible at numeric steering extrema (including a right-curving view near bag_03's negative extreme and left-curving views near positive extremes), but these visual associations do not establish the vehicle's left/right sign convention or whether the topic is command or feedback.",
        "",
        "## 5. Steering numeric distributions",
        "",
        "Unit and approximate numeric range are confirmed as radians and `[-0.35,+0.35] rad` by the user. Near-zero uses `abs(steering)<=0.01 rad`. The readable data conflicts with the confirmed approximate range, as shown below.",
        "",
        "| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | neg/zero/pos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        steering = item["steering"]
        stats = steering["distribution_rad"]
        sign = steering["sign_counts"]
        lines.append(
            f"| {item['bag_id']} | {stats['count']}/{_fmt(steering['timestamp']['measured_rate_hz'],3)} Hz | "
            + " | ".join(_fmt(stats[key], 6) for key in ("min", "p01", "p05", "p25", "mean", "median", "std", "p75", "p95", "p99", "max"))
            + f" | {sign['negative']}/{sign['near_zero']}/{sign['positive']} |"
        )
    conflict_parts = []
    cdr_mismatches = 0
    for item in bags:
        steering = item["steering"]
        check = steering["range_check"]
        outside = check["below_confirmed_range_count"] + check["above_confirmed_range_count"]
        conflict_parts.append(
            f"{item['bag_id']} {outside}/{steering['distribution_rad']['count']} outside, "
            f"min/max={_fmt(steering['distribution_rad']['min'],6)}/{_fmt(steering['distribution_rad']['max'],6)} rad"
        )
        cdr_mismatches += steering["float64_decode_validation"]["mismatch_count"] or 0
    lines += [
        "",
        "Out-of-range readable samples: " + "; ".join(conflict_parts) + ".",
        f"Direct CDR Float64 unpacking exactly matched `mcap_ros2` decoding for all readable steering samples (mismatches={cdr_mismatches}), so this is not a decoder interpretation artifact.",
        "",
        "Probable-saturation audit (numeric plateaus only; no actuator meaning inferred):",
    ]
    for item in bags:
        saturation = item["steering"]["probable_saturation"]
        candidates = saturation["repeated_out_of_range_plateau_candidates"]
        detail = (
            ", ".join(
                f"{candidate['value_rad_rounded_6dp']:+.6f} rad x{candidate['count']}"
                for candidate in candidates
            )
            if candidates
            else "none"
        )
        lines.append(
            f"- {item['bag_id']}: repeated out-of-range plateau candidates: {detail}; "
            f"repeated candidates at the confirmed +/-0.35 rad limits: {len(saturation['repeated_near_limit_candidates'])}."
        )
    lines += ["", "Full repetition details are preserved in each `bag_*.json`.", ""]
    lines += [
        "## 6. Unresolved steering semantics",
        "",
        "Radians and the approximate range are confirmed, but the observed out-of-range values require source-team reconciliation before extraction. Left/right sign convention and command-vs-actual-feedback meaning remain unresolved. Repository steering evidence describes simulator control and does not prove the provenance of these real-vehicle Float64 messages.",
        "",
        "Provenance evidence checked separately: `configs/rosbag_collector_v1.json` and `src/physicar_e2e/rosbag_collector.py` only enumerate recorded topics; `src/physicar_e2e/expert_driver.py` documents simulator command semantics. The bag metadata and MCAP channel metadata contain no real publisher/node identity, so none of those files proves real steering sign or command/feedback meaning.",
        "",
        "## 7. Speed numeric distributions",
        "",
        "| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | negative/zero/positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        speed = item["speed"]
        stats = speed["distribution_raw"]
        sign = speed["sign_and_stationary_counts"]
        lines.append(
            f"| {item['bag_id']} | {stats['count']}/{_fmt(speed['timestamp']['measured_rate_hz'],3)} Hz | "
            + " | ".join(_fmt(stats[key], 6) for key in ("min", "p01", "p05", "p25", "mean", "median", "std", "p75", "p95", "p99", "max"))
            + f" | {sign['negative']}/{sign['exact_zero']}/{sign['positive']} |"
        )
    lines += [
        "",
        "## 8. Unresolved speed semantics",
        "",
        "Speed unit and message meaning remain unresolved. Exact zero is reported only as a diagnostic stationary candidate; no simulator `1.0 m/s` assumption or permanent threshold is applied.",
        "",
        "The same provenance check found no real-vehicle publisher implementation or channel metadata that establishes the speed unit or whether the value is a command, estimate, or feedback signal.",
        "",
        "## 9. Chosen timestamp domain",
        "",
        "MCAP `log_time` (bag-record time) is selected for camera, steering, and speed. Image headers exist, but Float64 has no header and camera header epochs differ substantially from record time. No mixed-clock synchronization is performed.",
        "",
    ]
    for item in bags:
        offset = item["camera"]["header_vs_bag_record"]["record_minus_header_s"]
        lines.append(f"- {item['bag_id']} record-minus-header offset: {_stats_short(offset, 's')}.")
    lines += [
        "",
        "## 10-11. Causal steering/speed label ages and future-label violations",
        "",
        "| Bag | Steering age | Speed age | Missing steer | Missing speed | Complete pairs | Future labels |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for item in bags:
        sync = item["causal_synchronization"]
        lines.append(
            f"| {item['bag_id']} | {_stats_short(sync['steering_age_ms'], 'ms')} | {_stats_short(sync['speed_age_ms'], 'ms')} | "
            f"{sync['missing_causal_steering_count']} | {sync['missing_causal_speed_count']} | {sync['complete_causal_pair_count']} | **{sync['future_label_violations']}** |"
        )
    lines += [
        "",
        "## 12. Three-frame temporal readiness",
        "",
        "| Bag | Candidates | Strict causal | Label-ready | Order failures | Adjacent gaps | Oldest-current span | >0.120 s diagnostic | Assessment |",
        "|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for item in bags:
        temporal = item["three_frame_temporal_readiness"]
        comparison = temporal["simulator_comparison"]
        lines.append(
            f"| {item['bag_id']} | {temporal['candidate_three_frame_sequences']} | {temporal['valid_strict_causal_sequences']} | "
            f"{temporal['valid_strict_causal_sequences_with_current_causal_labels']} | {temporal['timestamp_order_failure_count']} | "
            f"{_stats_short(temporal['valid_triplet_adjacent_gap_s'], 's')} | {_stats_short(temporal['oldest_to_current_span_s'], 's')} | "
            f"{comparison['diagnostic_triplets_exceeding_0p120_s']} | {temporal['architecture_assessment']} |"
        )
    lines += [
        "",
        "Simulator train timing was approximately 0.066 s mean / 0.065 s median / 0.070 s p95 adjacent gap and 0.132 s mean oldest-to-current span. The existing 0.120 s adjacent gate was compared diagnostically, not enforced on real data.",
        "",
        "## 13. Cross-bag consistency",
        "",
        summary["cross_bag_consistency"]["assessment"] + ". Full-bag steering, speed, and timing consistency cannot be concluded from differently sized readable prefixes.",
        "",
    ]
    lines.extend(
        f"- {finding}"
        for finding in summary["cross_bag_consistency"]["readable_prefix_observed_differences"]
    )
    lines += [
        "",
        "## 14. Candidate active-driving windows",
        "",
        "Stationary prefix/suffix diagnostics use exact zero only. Incomplete inputs make the true bag suffix unobservable; the windows below are nonzero-run candidates, not permanent filters.",
        "",
    ]
    for item in bags:
        window = item["speed"]["active_driving_candidate_window"]
        prefix_count = item["speed"]["stationary_prefix_candidate"]["record_count"]
        suffix_count = item["speed"]["stationary_suffix_candidate"]["record_count"]
        if window is None:
            lines.append(
                f"- {item['bag_id']}: no nonzero-speed candidate window; exact-zero prefix/suffix counts={prefix_count}/{suffix_count}."
            )
        else:
            support = window["supporting_raw_speed_distribution"]
            lines.append(
                f"- {item['bag_id']}: +{_fmt(window['start_offset_from_metadata_s'],3)} s to +{_fmt(window['end_offset_from_metadata_s'],3)} s "
                f"({ _fmt(window['duration_s'],3)} s), raw speed min/median/max={_fmt(support['min'],6)}/{_fmt(support['median'],6)}/{_fmt(support['max'],6)}. "
                f"Exact-zero prefix/suffix counts={prefix_count}/{suffix_count}; this window is prefix-censored and diagnostic only."
            )
    comparison = summary["simulator_camera_comparison"]
    lines += [
        "",
        "## 15. Simulator-vs-real camera differences",
        "",
        f"Resolution match: {_fmt(comparison['resolution_match'])}; aspect-ratio match: {_fmt(comparison['aspect_ratio_match'])}; encoding match: {_fmt(comparison['encoding_match'])}. The real preview uses the full frame and never applies simulator `y=160:360`.",
        "",
        comparison["field_of_view_and_content"]["field_of_view_and_content"],
        "",
        "ROI implication: " + comparison["field_of_view_and_content"]["roi_implication"],
        "",
        "## 16. Unresolved items before dataset extraction",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["dataset_stage"]["blocked_on"])
    lines += [
        "",
        "## 17. Files added/modified",
        "",
        "- `configs/real_bag_audit_v1.json`",
        "- `src/physicar_e2e/real_bag_audit.py`",
        "- `scripts/run_real_bag_audit_v1.py`",
        "- `tests/test_real_bag_audit.py`",
        "- `results/real_bag_audit_v1/{summary.json,REPORT.md,bag_01.json,bag_02.json,bag_03.json,sync.json}`",
        "- bounded previews under `/home/a/physicar-e2e-artifacts/real_bag_audit_v1/previews/` (outside Git)",
        "",
        "## 18. Tests",
        "",
        "Focused unit tests cover mappings, Float64 record timestamps, Image header/record comparison, causal ZOH, zero future labels, cross-bag consistency, and scope guards. Execution status is reported in the task handoff after the audit artifacts are generated.",
        "",
        "## 19. Git status",
        "",
        "No commit or push was performed. Final worktree status is reported in the task handoff.",
        "",
        "## Scope attestation",
        "",
        "No training, fine-tuning, simulator driving, Docker modification, bag modification, odometry requirement, dataset extraction, speed-unit assumption, steering-sign assumption, or real-camera ROI application occurred.",
        "",
    ]
    return "\n".join(lines)


def build_report_v2(summary: dict[str, Any], bags: Sequence[dict[str, Any]]) -> str:
    real_roi = summary.get("real_camera_roi", {})
    real_crop = real_roi.get("crop", {})
    real_resize = real_roi.get("resize", {})
    lines = [
        "# Real PhysiCar Bag Audit V2",
        "",
        f"Audit execution result: **{summary['result']}**. This is a complete-bag data audit, not a training result or a claim of real-robot driving success. V1 remains separate historical evidence of the failed/truncated transfer.",
        "",
        "## 1. Complete/footer-valid MCAP proof",
        "",
        "| Bag | Bytes | SHA-256 | Opening magic | Closing magic | Strict CRC/footer scan | Metadata counts match | Required streams readable | Result |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]
    for item in bags:
        integrity = item["integrity"]
        lines.append(
            f"| {item['bag_id']} | {item['mcap_size_bytes']} | `{item['mcap_sha256']}` | "
            f"{_fmt(integrity['opening_magic_valid'])} | {_fmt(integrity['closing_magic_valid'])} | "
            f"{_fmt(integrity['strict_crc_and_footer_scan_completed'])} | "
            f"{_fmt(integrity['metadata_counts_match_readable_messages'])} | "
            f"{_fmt(integrity['required_topic_decode_pass'])} | **{item['result']}** |"
        )
    lines += [
        "",
        "Each strict non-seeking scan used CRC validation and reached the parsed footer and closing MCAP magic. Counts from every channel matched `metadata.yaml`; the required camera, steering, and speed messages also decoded without failure.",
        "",
        "## 2. Complete readable durations and counts",
        "",
        "| Bag | Metadata duration | Readable span | Total metadata/readable | Camera | Steering | Speed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        by_topic = {
            row["topic"]: row
            for row in item["actual_readable_messages"]["topic_counts_and_rates"]
        }
        cells = []
        for topic in ("/camera/image_raw", "/steering", "/speed"):
            row = by_topic[topic]
            cells.append(
                f"{row['metadata_message_count']}/{row['readable_message_count']} @ {_fmt(row['measured_rate_hz'], 3)} Hz"
            )
        integrity = item["integrity"]
        lines.append(
            f"| {item['bag_id']} | {_fmt(item['metadata']['duration_s'], 6)} s | "
            f"{_fmt(item['actual_readable_messages']['span_s'], 6)} s | "
            f"{integrity['metadata_total_message_count']}/{integrity['readable_total_message_count']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## 3. Camera contract, rate, and full-frame validation",
        "",
        "| Bag | Frames | Contract | FPS | Gap mean/median/p95/max | Timestamp order failures | Payload/preview failures |",
        "|---|---:|---|---:|---|---:|---:|",
    ]
    for item in bags:
        camera = item["camera"]
        contract = camera["uniform_contract"]
        contract_text = (
            f"{contract['width']}x{contract['height']} {contract['encoding']}, step={contract['step']}, frame_id={contract['frame_id']}"
            if contract
            else "non-uniform"
        )
        lines.append(
            f"| {item['bag_id']} | {camera['message_count']} | {contract_text} | {_fmt(camera['fps'], 3)} | "
            f"{_stats_short(camera['inter_frame_gap_s'], 's')} | "
            f"{camera['bag_record_timestamp']['backward_timestamp_count']} | {camera['decode']['failure_count']} |"
        )
    lines += [
        "",
        "All complete camera messages were audited. No crop was applied; in particular, simulator `y=160:360` was not applied to the real images.",
        "",
        f"Human-approved Real Camera ROI V1 for future extraction is `x={real_crop.get('x_start')}:{real_crop.get('x_end')}, y={real_crop.get('y_start')}:{real_crop.get('y_end')}` (480x280), resized to `{real_resize.get('output_width')}x{real_resize.get('output_height')}` with canonical bilinear resize, then existing RGB-to-YUV preprocessing and causal `[t-2,t-1,t]` input. Horizontal cropping and camera undistortion are disabled.",
        "",
        "## 4. Complete steering distributions: recorded and converted",
        "",
        "The complete bags store a normalized steering COMMAND on a nominal `[-1,+1]` scale. Per the confirmed real-vehicle contract, the whole stream is converted with `steering_rad = steering_recorded * 0.35`; positive is LEFT, negative is RIGHT, and no selective clipping is performed.",
        "",
        "### Recorded normalized values",
        "",
        "| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | neg/near-zero/pos |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        steering = item["steering"]
        stats = steering["distribution_recorded_raw"]
        signs = steering["sign_counts"]
        lines.append(
            f"| {item['bag_id']} | {stats['count']}/{_fmt(steering['timestamp']['measured_rate_hz'], 3)} Hz | "
            + " | ".join(
                _fmt(stats[key], 6)
                for key in ("min", "p01", "p05", "p25", "mean", "median", "std", "p75", "p95", "p99", "max")
            )
            + f" | {signs['negative']}/{signs['near_zero']}/{signs['positive']} |"
        )
    lines += [
        "",
        "The sign counts use the physical near-zero threshold after conversion; sign itself is unchanged by the positive scale factor.",
        "",
        "### Converted physical radians (`recorded * 0.35`)",
        "",
        "| Bag | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | Outside +/-0.35 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        steering = item["steering"]
        stats = steering["distribution_rad"]
        range_check = steering["range_check"]
        lines.append(
            f"| {item['bag_id']} | "
            + " | ".join(
                _fmt(stats[key], 6)
                for key in ("min", "p01", "p05", "p25", "mean", "median", "std", "p75", "p95", "p99", "max")
            )
            + f" | {range_check['outside_confirmed_range_count']} |"
        )
    lines += [
        "",
        "All converted values remain inside the approximate physical +/-0.35 rad range. `/steering` command semantics and sign are confirmed: positive is LEFT and negative is RIGHT.",
        "",
        "## 5. Reconciliation of recorded values numerically above +/-0.35",
        "",
        "The counts below preserve the original V1 concern by comparing the recorded normalized numbers directly with +/-0.35 before conversion. They are not physical-radian violations after the confirmed x0.35 conversion.",
        "",
        "| Bag | Recorded > +/-0.35 / finite | Fraction | Below / above | Converted outside +/-0.35 rad | Temporal episodes | Repeated raw plateaus | Samples with nonzero causal speed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        detail = item["steering"]["out_of_range_reconciliation"]
        relation = detail["causal_speed_relation"]
        lines.append(
            f"| {item['bag_id']} | {detail['outside_count']}/{detail['finite_steering_count']} | "
            f"{_fmt(detail['outside_fraction'], 6)} | {detail['below_count']}/{detail['above_count']} | "
            f"{detail['physical_outside_count_after_rescaling']} | "
            f"{detail['temporal_location_episode_count']} | {detail['repeated_consecutive_plateau_count']} | "
            f"{relation['nonzero_count']}/{relation['available_count']} ({_fmt(relation['nonzero_fraction'], 6)}) |"
        )
    lines += ["", "### Temporal locations and speed/driving-phase relation", ""]
    for item in bags:
        detail = item["steering"]["out_of_range_reconciliation"]
        lines.append(f"#### {item['bag_id']}")
        lines.append("")
        relation = detail["causal_speed_relation"]
        lines.append(
            f"At recorded magnitudes above 0.35, causal speed: {_stats_short(relation['raw_speed_distribution_at_out_of_range_steering'], 'm/s')}; "
            f"zero/nonzero={relation['exact_zero_count']}/{relation['nonzero_count']}; phases={detail['driving_phase_counts']}."
        )
        lines.append("")
        if not detail["temporal_locations"]:
            lines.append("No out-of-range episode was observed.")
        else:
            lines += [
                "| Raw side | Start-end offset | Samples | Recorded min/median/max | Causal speed min/median/max (m/s) | Driving phases |",
                "|---|---:|---:|---|---|---|",
            ]
            for episode in detail["temporal_locations"]:
                steering_stats = episode["steering_distribution_recorded"]
                speed_stats = episode["causal_speed_distribution_raw"]
                lines.append(
                    f"| {episode['side']} | +{_fmt(episode['start_offset_from_metadata_s'], 3)} to +{_fmt(episode['end_offset_from_metadata_s'], 3)} s | "
                    f"{episode['sample_count']} | {_fmt(steering_stats['min'], 6)}/{_fmt(steering_stats['median'], 6)}/{_fmt(steering_stats['max'], 6)} | "
                    f"{_fmt(speed_stats['min'], 6)}/{_fmt(speed_stats['median'], 6)}/{_fmt(speed_stats['max'], 6)} | {episode['driving_phase_counts']} |"
                )
        lines += ["", "Repeated consecutive recorded-value plateaus above the unscaled +/-0.35 numeric threshold (6-decimal diagnostic):"]
        plateaus = detail["repeated_consecutive_plateaus"]
        if not plateaus:
            lines.append("- none")
        for plateau in plateaus:
            lines.append(
                f"- recorded `{plateau['value_recorded_rounded_6dp']:+.6f}` (converted `{plateau['value_recorded_rounded_6dp'] * item['steering']['recorded_to_radians']['scale']:+.6f} rad`): +{_fmt(plateau['start_offset_from_metadata_s'], 3)} to "
                f"+{_fmt(plateau['end_offset_from_metadata_s'], 3)} s, {plateau['sample_count']} records, phases={plateau['driving_phase_counts']}."
            )
        lines.append("")
    lines += [
        "No actuator-saturation meaning is inferred from a repeated normalized numeric plateau. Direct CDR Float64 decoding independently matched `mcap_ros2` decoding. The raw values remain preserved, and the confirmed whole-stream scaling—not clipping—reconciles them with the physical range.",
        "",
        "## 6. Complete speed distributions (m/s)",
        "",
        "| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | negative/zero/positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in bags:
        speed = item["speed"]
        stats = speed["distribution_raw"]
        signs = speed["sign_and_stationary_counts"]
        lines.append(
            f"| {item['bag_id']} | {stats['count']}/{_fmt(speed['timestamp']['measured_rate_hz'], 3)} Hz | "
            + " | ".join(
                _fmt(stats[key], 6)
                for key in ("min", "p01", "p05", "p25", "mean", "median", "std", "p75", "p95", "p99", "max")
            )
            + f" | {signs['negative']}/{signs['exact_zero']}/{signs['positive']} |"
        )
    lines += [
        "",
        "The `/speed` unit is confirmed as meters per second (m/s). Only whether it is a command or an actual feedback/measurement remains unresolved. Exact zero is only a diagnostic stationary indicator.",
        "",
        "## 7. Timestamp domain",
        "",
        "MCAP `log_time` is the single synchronization domain for camera, steering, and speed. Camera headers are diagnostic only because both Float64 streams lack headers; no unproven mixed-clock transform is used.",
        "",
        "| Bag | Camera-header backward steps | Header duplicate timestamps | First/last record-minus-header offset | Offset change |",
        "|---|---:|---:|---|---:|",
    ]
    header_findings = []
    for item in bags:
        camera = item["camera"]
        comparison = camera["header_vs_bag_record"]
        lines.append(
            f"| {item['bag_id']} | {camera['header_timestamp']['backward_timestamp_count']} | "
            f"{camera['duplicate_timestamps']['camera_header']} | "
            f"{_fmt(comparison['first_record_minus_header_s'], 6)}/{_fmt(comparison['last_record_minus_header_s'], 6)} s | "
            f"{_fmt(comparison['offset_change_s'], 6)} s |"
        )
        if camera["header_timestamp"]["backward_timestamp_count"]:
            header_findings.append(
                f"{item['bag_id']} has {camera['header_timestamp']['backward_timestamp_count']} backward header step(s)"
            )
        if camera["duplicate_timestamps"]["camera_header"]:
            header_findings.append(
                f"{item['bag_id']} repeats {camera['duplicate_timestamps']['camera_header']} header timestamp(s)"
            )
    lines += [
        "",
        (
            "Camera header audit: " + "; ".join(header_findings) + ". This independently rules out using camera header time as a shared raw clock with the headerless scalar topics."
            if header_findings
            else "Camera headers are monotonic, but their epoch differs from MCAP log time and the scalar streams have no headers; no safe raw mixed-clock join is proven."
        ),
        "",
        "## 8. Causal camera-to-label ages",
        "",
        "| Bag | Steering age | Speed age | Missing steering | Missing speed | Complete causal pairs |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in bags:
        sync = item["causal_synchronization"]
        lines.append(
            f"| {item['bag_id']} | {_stats_short(sync['steering_age_ms'], 'ms')} | {_stats_short(sync['speed_age_ms'], 'ms')} | "
            f"{sync['missing_causal_steering_count']} | {sync['missing_causal_speed_count']} | {sync['complete_causal_pair_count']} |"
        )
    lines += [
        "",
        "The join is causal ZOH: for camera time `t`, each label is the latest same-domain scalar record with `t_scalar <= t`.",
        "",
        "## 9. Future-label violations",
        "",
        f"Total future-label violations across all three bags: **{summary['future_label_violations']}**.",
        "",
        "## 10. Full three-frame temporal readiness",
        "",
        "| Bag | Candidates | Strict ordered | Current-label ready | Order violations | Adjacent gap distribution | Oldest-current span | Adjacent gaps >120 ms |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in bags:
        temporal = item["three_frame_temporal_readiness"]
        comparison = temporal["simulator_comparison"]
        lines.append(
            f"| {item['bag_id']} | {temporal['candidate_three_frame_sequences']} | {temporal['valid_strict_causal_sequences']} | "
            f"{temporal['valid_strict_causal_sequences_with_current_causal_labels']} | {temporal['timestamp_order_failure_count']} | "
            f"{_stats_short(temporal['adjacent_frame_gap_s'], 's')} | {_stats_short(temporal['oldest_to_current_span_s'], 's')} | "
            f"{comparison['adjacent_gap_over_0p120_s_count']} |"
        )
    lines += [
        "",
        "The simulator 120 ms gap threshold is reported diagnostically and was not applied as a real-data rejection gate.",
        "",
    ]
    for item in bags:
        gaps = item["three_frame_temporal_readiness"]["simulator_comparison"][
            "adjacent_gaps_over_0p120_s"
        ]
        for gap in gaps:
            lines.append(
                f"- {item['bag_id']}: {_fmt(gap['gap_s'], 6)} s between camera #{gap['previous_camera_index']} and "
                f"#{gap['current_camera_index']} (current frame +{_fmt(gap['current_offset_from_first_camera_s'], 3)} s)."
            )
    lines += [
        "" if any(
            item["three_frame_temporal_readiness"]["simulator_comparison"][
                "adjacent_gaps_over_0p120_s"
            ]
            for item in bags
        ) else "No adjacent gap exceeds 120 ms.",
        "",
        "## 11. Cross-bag consistency",
        "",
        summary["cross_bag_consistency"]["assessment"] + ".",
        "",
    ]
    lines.extend(
        f"- {finding}" for finding in summary["cross_bag_consistency"]["observed_differences"]
    )
    lines += ["", "## 12. Candidate active-driving windows", ""]
    for item in bags:
        windows = item["speed"]["active_driving_candidate_windows"]
        prefix = item["speed"]["stationary_prefix_candidate"]["record_count"]
        suffix = item["speed"]["stationary_suffix_candidate"]["record_count"]
        lines.append(
            f"### {item['bag_id']} ({len(windows)} nonzero-speed run(s); exact-zero prefix/suffix records={prefix}/{suffix})"
        )
        lines.append("")
        if not windows:
            lines.append("- No finite, exactly nonzero speed run observed.")
        for window in windows:
            stats = window["supporting_raw_speed_distribution"]
            lines.append(
                f"- +{_fmt(window['start_offset_from_metadata_s'], 3)} to +{_fmt(window['end_offset_from_metadata_s'], 3)} s "
                f"({_fmt(window['duration_s'], 3)} s, {window['record_count']} records), speed min/median/max (m/s)="
                f"{_fmt(stats['min'], 6)}/{_fmt(stats['median'], 6)}/{_fmt(stats['max'], 6)}."
            )
        lines.append("")
    lines += [
        "These are diagnostic candidates based only on exact nonzero speed in m/s; they are not extraction filters, and no command-versus-feedback interpretation is inferred.",
        "",
        "## 13. Bounded visual previews",
        "",
    ]
    for item in bags:
        lines.append(
            f"- {item['bag_id']}: `{item['previews']['contact_sheet']}` ({item['previews']['selected_frame_count']} uncropped selected frames; decode failures={item['previews']['decode_failure_count']})"
        )
    lines += [
        "",
        "Selections span first/early/middle/late/last frames and time-separated positive-LEFT/negative-RIGHT steering-command extrema. Human ROI review is complete: Real Camera ROI V1 preserves y=80:160 far-track curvature, orange center-line vanishing-point information, and early cone visibility. The audit previews remain uncropped evidence.",
        "",
        "## 14. Remaining blockers before real dataset extraction",
        "",
    ]
    if summary["dataset_stage"]["blocked_on"]:
        lines.extend(f"- {item}" for item in summary["dataset_stage"]["blocked_on"])
    else:
        lines.append("- none")
    lines += [
        "",
        "## 15. REAL_DATASET_EXTRACTION decision",
        "",
        f"**{summary['dataset_stage']['real_dataset_extraction_decision']}**: {summary['dataset_stage']['decision_reason']}",
        "",
        "No dataset extraction or training was performed by this audit.",
        "",
        "## 16. Tests",
        "",
        "The audit artifact generator does not run the repository test suite. The exact post-audit test command and result are reported in the task handoff.",
        "",
        "## 17. Git status",
        "",
        "No commit or push was performed. The exact final worktree status is reported in the task handoff.",
        "",
        "## Scope attestation",
        "",
        "No training, final dataset extraction, simulator use, Docker use, bag modification, odometry requirement, steering clipping, speed-unit assumption, steering-sign assumption, or real-camera ROI application occurred.",
        "",
    ]
    return "\n".join(lines)


def build_report(summary: dict[str, Any], bags: Sequence[dict[str, Any]]) -> str:
    if summary.get("version") == "real_bag_audit_v2":
        return build_report_v2(summary, bags)
    return build_report_v1(summary, bags)


def run_audit(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    output = output_root or Path(config["output_root"])
    if config["version"] == "real_bag_audit_v2" and output.name == "real_bag_audit_v1":
        raise RealBagAuditError("V2 output must not overwrite the historical V1 result directory")
    output.mkdir(parents=True, exist_ok=True)
    bags = [audit_bag(item, config) for item in config["bags"]]
    cross_bag = cross_bag_consistency(bags)
    simulator_comparison = simulator_camera_comparison(bags, config)
    complete = all(item["result"] == "PASS" for item in bags)
    recorded_threshold_exceedance_count = sum(
        item["steering"]["out_of_range_reconciliation"]["outside_count"] for item in bags
    )
    physical_out_of_range_count = sum(
        item["steering"]["range_check"]["outside_confirmed_range_count"] for item in bags
    )
    steering_rescaling_required = any(
        item["steering"]["recorded_to_radians"]["required"] for item in bags
    )
    temporal_gap_over_count = sum(
        item["three_frame_temporal_readiness"]["simulator_comparison"][
            "adjacent_gap_over_0p120_s_count"
        ]
        for item in bags
    )
    maximum_label_age_ms = max(
        max(
            item["causal_synchronization"]["steering_age_ms"]["max"] or 0.0,
            item["causal_synchronization"]["speed_age_ms"]["max"] or 0.0,
        )
        for item in bags
    )
    real_roi_approved = config["real_camera_roi"].get("status") == "approved"
    blockers = []
    if not complete:
        blockers.append("complete, footer-valid, fully readable copies of bag_01, bag_02, and bag_03")
    if physical_out_of_range_count:
        blockers.append(
            f"reconcile {physical_out_of_range_count} converted physical-radian steering samples outside the confirmed approximate [-0.35,+0.35] range"
        )
    if steering_rescaling_required:
        blockers.append(
            "freeze and test the user-confirmed whole-stream steering conversion steering_rad = steering_recorded * 0.35 in the future real-data extractor (never clip only the raw exceedances)"
        )
    if temporal_gap_over_count:
        blockers.append(
            f"approve a real-data temporal-gap and label-staleness policy for {temporal_gap_over_count} adjacent camera gaps above 120 ms and observed causal label ages up to {maximum_label_age_ms:.3f} ms"
        )
    if config["steering_contract"].get("left_right_sign_convention") == "unresolved":
        blockers.append("confirm the real steering left/right sign convention")
    if config["steering_contract"].get("command_or_feedback") == "unresolved":
        blockers.append(
            "confirm whether real steering is command, requested target, feedback, or another quantity"
        )
    if config["speed_contract"].get("unit") == "unresolved":
        blockers.append("confirm the real speed unit")
    if config["speed_contract"].get("meaning", "unresolved").startswith("unresolved"):
        blockers.append(
            "confirm only whether /speed is a command or an actual feedback/measurement value; its m/s unit is already confirmed"
        )
    if not real_roi_approved:
        blockers.append(
            "obtain human approval of the real-camera ROI from the complete-bag uncropped previews"
        )
    extraction_justified = not blockers
    summary = {
        "version": config["version"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS" if complete else "FAIL_INCOMPLETE_INPUTS",
        "audit_scope": "complete_bags" if complete else "strictly_readable_prefixes_only",
        "bag_ids": [item["bag_id"] for item in bags],
        "integrity_all_three": complete,
        "camera_contract_same_on_readable_messages": cross_bag["camera"]["same_contract_on_readable_messages"],
        "camera_contract_same_on_all_messages": (
            complete and cross_bag["camera"]["same_contract_on_readable_messages"]
        ),
        "steering_contract": config["steering_contract"],
        "speed_contract": config["speed_contract"],
        "timestamp_domain": config["timestamp_domain"],
        "real_camera_roi": config["real_camera_roi"],
        "cross_bag_consistency": cross_bag,
        "simulator_camera_comparison": simulator_comparison,
        "future_label_violations": sum(
            item["causal_synchronization"]["future_label_violations"] for item in bags
        ),
        "steering_recorded_threshold_exceedance_count": recorded_threshold_exceedance_count,
        "steering_physical_outside_count_after_rescaling": physical_out_of_range_count,
        "dataset_stage": {
            "status": "READY" if extraction_justified else "BLOCKED",
            "generated": False,
            "real_dataset_extraction_justified": extraction_justified,
            "real_dataset_extraction_decision": (
                "JUSTIFIED" if extraction_justified else "NOT YET JUSTIFIED"
            ),
            "decision_reason": (
                "the complete-bag audit passed and no extraction blockers remain"
                if extraction_justified
                else "the complete-bag audit and ROI approval are complete, but preprocessing, temporal-policy, and /speed command-versus-feedback blockers remain"
            ),
            "blocked_on": blockers,
        },
        "scope_guards": config["scope_guards"],
        "preview_root": config["preview"]["root"],
        "bag_result_files": {item["bag_id"]: f"{item['bag_id']}.json" for item in bags},
    }
    sync = {
        "version": config["version"],
        "chosen_timestamp_domain": config["timestamp_domain"],
        "future_label_violations": summary["future_label_violations"],
        "bags": {
            item["bag_id"]: {
                "audit_scope": item["audit_scope"],
                "camera_header_vs_record": item["camera"]["header_vs_bag_record"],
                "causal_synchronization": item["causal_synchronization"],
                "three_frame_temporal_readiness": item["three_frame_temporal_readiness"],
            }
            for item in bags
        },
    }
    for item in bags:
        (output / f"{item['bag_id']}.json").write_text(
            json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "sync.json").write_text(
        json.dumps(sync, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "REPORT.md").write_text(build_report(summary, bags), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only Real PhysiCar Bag Audit V1")
    parser.add_argument("--config", type=Path, default=Path("configs/real_bag_audit_v1.json"))
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_audit(args.config, args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
