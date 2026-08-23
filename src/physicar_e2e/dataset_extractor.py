"""Offline, deterministic PhysiCar Dataset Extractor V1."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image


EXTRACTOR_VERSION = "dataset_extractor_v1"
REJECTION_REASONS = [
    "before_active_drive_window",
    "after_active_drive_window",
    "no_causal_steering",
    "no_causal_speed",
    "stale_steering",
    "stale_speed",
    "below_drive_speed",
    "image_decode_error",
    "future_record_violation",
]
MANIFEST_COLUMNS = [
    "episode_id",
    "sample_index",
    "image_path",
    "camera_record_time_ns",
    "camera_header_time_ns",
    "steering_record_time_ns",
    "steering_age_ms",
    "steering_rad",
    "steering_normalized",
    "speed_record_time_ns",
    "speed_age_ms",
    "speed_mps",
    "source_mcap_sha256",
]


class ExtractionError(RuntimeError):
    """A clear data-quality or input failure."""


@dataclass(frozen=True)
class ScalarRecord:
    time_ns: int
    value: float


@dataclass(frozen=True)
class DriveWindow:
    start_ns: int
    end_ns: int
    record_count: int

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "camera_topic", "steering_topic", "speed_topic",
        "minimum_drive_speed_mps", "maximum_steering_age_s",
        "maximum_speed_age_s", "source_width", "source_height",
        "source_encoding", "roi", "output_width", "output_height",
        "maximum_steering_rad", "near_zero_steering_rad",
        "saturation_fraction_of_limit", "preview_frame_count",
        "steering_histogram_bin_edges_rad",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ExtractionError(f"configuration missing keys: {', '.join(missing)}")
    return config


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def time_to_ns(stamp: Any) -> int | None:
    if stamp is None or not hasattr(stamp, "sec") or not hasattr(stamp, "nanosec"):
        return None
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def latest_causal(records: Sequence[ScalarRecord], time_ns: int) -> ScalarRecord | None:
    """Return the latest record at or before time_ns; never a future record."""
    index = bisect.bisect_right(records, time_ns, key=lambda item: item.time_ns) - 1
    if index < 0:
        return None
    selected = records[index]
    assert selected.time_ns <= time_ns, "causal ZOH selected a future record"
    return selected


def synchronize_frame(
    camera_time_ns: int,
    steering: Sequence[ScalarRecord],
    speed: Sequence[ScalarRecord],
    window: DriveWindow,
    config: dict[str, Any],
) -> tuple[str | None, ScalarRecord | None, ScalarRecord | None]:
    """Apply the V1 window, causal ZOH, freshness, and driving-state gates."""
    if camera_time_ns < window.start_ns:
        return "before_active_drive_window", None, None
    if camera_time_ns > window.end_ns:
        return "after_active_drive_window", None, None
    steering_record = latest_causal(steering, camera_time_ns)
    if steering_record is None:
        return "no_causal_steering", None, None
    speed_record = latest_causal(speed, camera_time_ns)
    if speed_record is None:
        return "no_causal_speed", steering_record, None
    assert steering_record.time_ns <= camera_time_ns
    assert speed_record.time_ns <= camera_time_ns
    if (camera_time_ns - steering_record.time_ns) / 1e9 > float(config["maximum_steering_age_s"]):
        return "stale_steering", steering_record, speed_record
    if (camera_time_ns - speed_record.time_ns) / 1e9 > float(config["maximum_speed_age_s"]):
        return "stale_speed", steering_record, speed_record
    if abs(speed_record.value) < float(config["minimum_drive_speed_mps"]):
        return "below_drive_speed", steering_record, speed_record
    return None, steering_record, speed_record


def detect_dominant_drive_window(
    speed_records: Sequence[ScalarRecord], minimum_speed_mps: float
) -> DriveWindow:
    """Select the longest threshold-active segment bounded by speed records."""
    if not speed_records:
        raise ExtractionError("no speed records available for drive-window detection")
    ordered = sorted(speed_records, key=lambda item: item.time_ns)
    segments: list[DriveWindow] = []
    start: int | None = None
    end: int | None = None
    count = 0
    for record in ordered:
        active = abs(record.value) >= minimum_speed_mps
        if active:
            if start is None:
                start = record.time_ns
                count = 0
            end = record.time_ns
            count += 1
        elif start is not None:
            segments.append(DriveWindow(start, end if end is not None else start, count))
            start = end = None
            count = 0
    if start is not None:
        segments.append(DriveWindow(start, end if end is not None else start, count))
    if not segments:
        raise ExtractionError(
            f"no speed segment meets minimum drive speed {minimum_speed_mps:.3f} m/s"
        )
    return max(segments, key=lambda item: (item.end_ns - item.start_ns, item.record_count, -item.start_ns))


def decode_rgb8_image(message: Any, config: dict[str, Any]) -> Image.Image:
    width = int(message.width)
    height = int(message.height)
    encoding = str(message.encoding)
    step = int(message.step)
    data = bytes(message.data)
    expected_width = int(config["source_width"])
    expected_height = int(config["source_height"])
    expected_encoding = str(config["source_encoding"])
    if (width, height) != (expected_width, expected_height):
        raise ExtractionError(
            f"unexpected image dimensions {width}x{height}; expected {expected_width}x{expected_height}"
        )
    if encoding != expected_encoding or encoding != "rgb8":
        raise ExtractionError(f"unsupported image encoding {encoding!r}; expected 'rgb8'")
    packed_row_bytes = width * 3
    if step < packed_row_bytes:
        raise ExtractionError(f"invalid image step {step}; RGB row needs at least {packed_row_bytes} bytes")
    required_bytes = step * height
    if len(data) < required_bytes:
        raise ExtractionError(f"truncated image: {len(data)} bytes; need {required_bytes}")
    packed = bytearray(packed_row_bytes * height)
    for row in range(height):
        source = row * step
        target = row * packed_row_bytes
        packed[target : target + packed_row_bytes] = data[source : source + packed_row_bytes]
    return Image.frombytes("RGB", (width, height), bytes(packed))


def preprocess_image(image: Image.Image, config: dict[str, Any]) -> Image.Image:
    roi = config["roi"]
    box = (
        int(roi["x_start"]), int(roi["y_start"]),
        int(roi["x_end"]), int(roi["y_end"]),
    )
    cropped = image.crop(box)
    expected_crop = (box[2] - box[0], box[3] - box[1])
    if cropped.size != expected_crop:
        raise ExtractionError(f"ROI produced {cropped.size}, expected {expected_crop}")
    output_size = (int(config["output_width"]), int(config["output_height"]))
    result = cropped.resize(output_size, resample=Image.Resampling.BILINEAR)
    if result.mode != "RGB" or result.size != output_size:
        raise ExtractionError("image preprocessing did not produce required RGB output")
    return result


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


def numeric_distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def steering_distribution(values: Sequence[float], config: dict[str, Any]) -> dict[str, Any]:
    near_zero = float(config["near_zero_steering_rad"])
    limit = float(config["maximum_steering_rad"])
    saturation_threshold = limit * float(config["saturation_fraction_of_limit"])
    edges = [float(value) for value in config["steering_histogram_bin_edges_rad"]]
    bins = [0] * (len(edges) - 1)
    below = above = 0
    for value in values:
        if value < edges[0]:
            below += 1
        elif value > edges[-1]:
            above += 1
        else:
            index = bisect.bisect_right(edges, value) - 1
            index = min(max(index, 0), len(bins) - 1)
            bins[index] += 1
    saturation_count = sum(abs(value) >= saturation_threshold for value in values)
    count = len(values)
    return {
        "sample_count": count,
        "mean_rad": statistics.fmean(values) if values else None,
        "std_rad": statistics.pstdev(values) if values else None,
        "min_rad": min(values) if values else None,
        "max_rad": max(values) if values else None,
        "mean_absolute_rad": statistics.fmean(abs(value) for value in values) if values else None,
        "saturation_threshold_rad": saturation_threshold,
        "saturation_count": saturation_count,
        "saturation_fraction": saturation_count / count if count else None,
        "negative_count": sum(value < -near_zero for value in values),
        "near_zero_count": sum(abs(value) <= near_zero for value in values),
        "positive_count": sum(value > near_zero for value in values),
        "near_zero_threshold_rad": near_zero,
        "histogram": {"bin_edges_rad": edges, "bin_counts": bins, "below_range": below, "above_range": above},
    }


def synchronization_diagnostics(
    steering_ages_ms: Sequence[float], speed_ages_ms: Sequence[float],
    accepted_camera_times_ns: Sequence[int],
) -> dict[str, Any]:
    intervals_ms = [
        (later - earlier) / 1e6
        for earlier, later in zip(accepted_camera_times_ns, accepted_camera_times_ns[1:])
    ]
    interval = numeric_distribution(intervals_ms)
    interval.pop("median")
    return {
        "steering_age_ms": numeric_distribution(steering_ages_ms),
        "speed_age_ms": numeric_distribution(speed_ages_ms),
        "accepted_camera_interval_ms": interval,
    }


def _iter_decoded(mcap_path: Path, topics: Iterable[str]):
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        yield from reader.iter_decoded_messages(topics=list(topics), log_time_order=True)


def scan_control_records(mcap_path: Path, config: dict[str, Any]) -> tuple[list[ScalarRecord], list[ScalarRecord], dict[str, Any]]:
    steering: list[ScalarRecord] = []
    speed: list[ScalarRecord] = []
    clock_pairs: list[tuple[int, int]] = []
    expected_types = {
        config["steering_topic"]: "std_msgs/msg/Float64",
        config["speed_topic"]: "std_msgs/msg/Float64",
        config.get("clock_topic", "/clock"): "rosgraph_msgs/msg/Clock",
    }
    for schema, channel, record, decoded in _iter_decoded(mcap_path, expected_types):
        expected = expected_types[channel.topic]
        if schema.name != expected:
            raise ExtractionError(f"{channel.topic} type is {schema.name}, expected {expected}")
        if channel.topic == config["steering_topic"]:
            steering.append(ScalarRecord(record.log_time, float(decoded.data)))
        elif channel.topic == config["speed_topic"]:
            speed.append(ScalarRecord(record.log_time, float(decoded.data)))
        else:
            stamp_ns = time_to_ns(decoded.clock)
            if stamp_ns is not None:
                clock_pairs.append((record.log_time, stamp_ns))
    if not steering:
        raise ExtractionError(f"no records on {config['steering_topic']}")
    if not speed:
        raise ExtractionError(f"no records on {config['speed_topic']}")
    clock = {
        "record_count": len(clock_pairs),
        "first_record_time_ns": clock_pairs[0][0] if clock_pairs else None,
        "first_sim_time_ns": clock_pairs[0][1] if clock_pairs else None,
        "last_record_time_ns": clock_pairs[-1][0] if clock_pairs else None,
        "last_sim_time_ns": clock_pairs[-1][1] if clock_pairs else None,
        "sim_duration_s": (clock_pairs[-1][1] - clock_pairs[0][1]) / 1e9 if len(clock_pairs) > 1 else None,
    }
    return steering, speed, clock


def _camera_header_time_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    return time_to_ns(getattr(header, "stamp", None)) if header is not None else None


def _manifest_value(value: Any) -> Any:
    return "" if value is None else value


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _manifest_value(row.get(key)) for key in MANIFEST_COLUMNS})


def create_contact_sheet(dataset_root: Path, rows: Sequence[dict[str, Any]], output: Path, count: int) -> None:
    if not rows:
        raise ExtractionError("cannot create preview without accepted images")
    actual = min(count, len(rows))
    indices = sorted({round(index * (len(rows) - 1) / max(actual - 1, 1)) for index in range(actual)})
    images = [Image.open(dataset_root / rows[index]["image_path"]).convert("RGB") for index in indices]
    columns = min(3, len(images))
    rows_count = math.ceil(len(images) / columns)
    cell = images[0].size
    sheet = Image.new("RGB", (columns * cell[0], rows_count * cell[1]))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * cell[0], (index // columns) * cell[1]))
        image.close()
    sheet.save(output, format="PNG", optimize=False)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def extract_episode(
    *, episode_id: str, mcap_path: Path, collector_metadata_path: Path,
    dataset_root: Path, config: dict[str, Any], config_sha256: str,
    source_path_identity: str, collector_metadata_identity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collector = json.loads(collector_metadata_path.read_text(encoding="utf-8"))
    source_size = mcap_path.stat().st_size
    source_sha = sha256_file(mcap_path)
    steering, speed, clock = scan_control_records(mcap_path, config)
    window = detect_dominant_drive_window(speed, float(config["minimum_drive_speed_mps"]))
    image_dir = dataset_root / "images" / episode_id
    image_dir.mkdir(parents=True)
    rejection: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    camera_times: list[int] = []
    header_times: list[int] = []
    total_camera = 0
    active_window_camera = 0
    future_violations = 0
    expected_camera_type = "sensor_msgs/msg/Image"
    for schema, channel, record, decoded in _iter_decoded(mcap_path, [config["camera_topic"]]):
        total_camera += 1
        camera_time = int(record.log_time)
        header_time = _camera_header_time_ns(decoded)
        if header_time is not None:
            header_times.append(header_time)
        if schema.name != expected_camera_type:
            raise ExtractionError(f"{channel.topic} type is {schema.name}, expected {expected_camera_type}")
        if window.start_ns <= camera_time <= window.end_ns:
            active_window_camera += 1
        reason, steering_record, speed_record = synchronize_frame(
            camera_time, steering, speed, window, config
        )
        if reason is not None:
            rejection[reason] += 1
            continue
        assert steering_record is not None and speed_record is not None
        steering_age_s = (camera_time - steering_record.time_ns) / 1e9
        speed_age_s = (camera_time - speed_record.time_ns) / 1e9
        if steering_age_s > float(config["maximum_steering_age_s"]):
            rejection["stale_steering"] += 1
            continue
        if speed_age_s > float(config["maximum_speed_age_s"]):
            rejection["stale_speed"] += 1
            continue
        if abs(speed_record.value) < float(config["minimum_drive_speed_mps"]):
            rejection["below_drive_speed"] += 1
            continue
        try:
            image = preprocess_image(decode_rgb8_image(decoded, config), config)
        except Exception as error:
            rejection["image_decode_error"] += 1
            raise ExtractionError(f"{episode_id} camera at {camera_time}: {error}") from error
        sample_index = len(rows)
        relative_path = Path("images") / episode_id / f"frame_{sample_index:06d}.png"
        image.save(dataset_root / relative_path, format="PNG", optimize=False)
        image.close()
        row = {
            "episode_id": episode_id,
            "sample_index": sample_index,
            "image_path": relative_path.as_posix(),
            "camera_record_time_ns": camera_time,
            "camera_header_time_ns": header_time,
            "steering_record_time_ns": steering_record.time_ns,
            "steering_age_ms": steering_age_s * 1000,
            "steering_rad": steering_record.value,
            "steering_normalized": steering_record.value / float(config["maximum_steering_rad"]),
            "speed_record_time_ns": speed_record.time_ns,
            "speed_age_ms": speed_age_s * 1000,
            "speed_mps": speed_record.value,
            "source_mcap_sha256": source_sha,
        }
        assert row["steering_record_time_ns"] <= row["camera_record_time_ns"]
        assert row["speed_record_time_ns"] <= row["camera_record_time_ns"]
        rows.append(row)
        camera_times.append(camera_time)
    if not rows:
        raise ExtractionError(f"{episode_id}: no accepted camera samples")
    if future_violations:
        raise ExtractionError(f"{episode_id}: {future_violations} future-label violations")
    if camera_times != sorted(camera_times):
        raise ExtractionError(f"{episode_id}: camera samples are not timestamp-sorted")
    manifest_path = dataset_root / "manifests" / f"{episode_id}.csv"
    write_manifest(manifest_path, rows)
    preview_path = dataset_root / "previews" / f"{episode_id}.png"
    create_contact_sheet(dataset_root, rows, preview_path, int(config["preview_frame_count"]))
    steering_values = [float(row["steering_rad"]) for row in rows]
    steering_ages = [float(row["steering_age_ms"]) for row in rows]
    speed_ages = [float(row["speed_age_ms"]) for row in rows]
    header_monotonic = all(a <= b for a, b in zip(header_times, header_times[1:])) if header_times else None
    metadata_start = datetime.fromisoformat(collector["recording_start_utc"])
    mcap_start = datetime.fromtimestamp(min(steering[0].time_ns, speed[0].time_ns) / 1e9, tz=timezone.utc)
    expert_duration = float(collector["expert_result_metrics"]["elapsed_s"])
    episode_dir = dataset_root / "images" / episode_id
    metrics = {
        "extractor_version": EXTRACTOR_VERSION,
        "result": "PASS",
        "episode_id": episode_id,
        "source": {
            "mcap_path": source_path_identity,
            "mcap_size_bytes": source_size,
            "mcap_sha256": source_sha,
            "collector_metadata": collector_metadata_identity,
            "collector_reported_bag_size_bytes": collector.get("bag_size_bytes"),
            "physicar_e2e_source_commit": collector.get("physicar_e2e_git_commit"),
            "extractor_source_commit": _git_commit_or_none(),
            "expert_config_sha256": collector.get("canonical_expert_config_sha256"),
            "extractor_config_sha256": config_sha256,
        },
        "drive_window": {
            "method": "dominant contiguous abs(speed) >= threshold segment in MCAP record-time domain",
            "minimum_drive_speed_mps": config["minimum_drive_speed_mps"],
            "start_record_time_ns": window.start_ns,
            "end_record_time_ns": window.end_ns,
            "duration_s": window.duration_s,
            "speed_record_count": window.record_count,
            "collector_expert_duration_s": expert_duration,
            "duration_difference_s": window.duration_s - expert_duration,
        },
        "counts": {
            "total_camera_frames": total_camera,
            "active_window_camera_frames": active_window_camera,
            "accepted_camera_samples": len(rows),
            "rejected_camera_frames": total_camera - len(rows),
            "rejection_by_reason": {reason: rejection[reason] for reason in REJECTION_REASONS},
            "retention_fraction": len(rows) / total_camera,
            "active_window_retention_fraction": len(rows) / active_window_camera if active_window_camera else 0.0,
            "manifest_row_count": len(rows),
        },
        "synchronization": {
            **synchronization_diagnostics(steering_ages, speed_ages, camera_times),
            "future_label_violations": future_violations,
            "rule": "causal zero-order hold on MCAP record timestamps",
        },
        "steering_distribution": steering_distribution(steering_values, config),
        "camera_headers": {
            "available_count": len(header_times),
            "monotonic_non_decreasing": header_monotonic,
            "first_time_ns": header_times[0] if header_times else None,
            "last_time_ns": header_times[-1] if header_times else None,
            "domain_use": "diagnostic only; never used for synchronization",
        },
        "clock": {**clock, "domain_use": "diagnostic only; never compared directly to MCAP record time"},
        "clock_domain_findings": {
            "mcap_record_time_is_unix_epoch_like": min(steering[0].time_ns, speed[0].time_ns) > 1_000_000_000_000_000_000,
            "collector_start_minus_mcap_start_ms": (metadata_start - mcap_start).total_seconds() * 1000,
            "camera_header_and_clock_ranges_overlap": bool(
                header_times and clock["first_sim_time_ns"] is not None
                and header_times[-1] >= clock["first_sim_time_ns"]
                and header_times[0] <= clock["last_sim_time_ns"]
            ),
            "first_camera_header_minus_first_clock_ms": (
                (header_times[0] - clock["first_sim_time_ns"]) / 1e6
                if header_times and clock["first_sim_time_ns"] is not None else None
            ),
            "last_camera_header_minus_last_clock_ms": (
                (header_times[-1] - clock["last_sim_time_ns"]) / 1e6
                if header_times and clock["last_sim_time_ns"] is not None else None
            ),
            "domains_proven_identical": False,
        },
        "artifacts": {
            "manifest_path": manifest_path.relative_to(dataset_root).as_posix(),
            "preview_path": preview_path.relative_to(dataset_root).as_posix(),
            "extracted_image_size_bytes": _directory_size(episode_dir),
        },
    }
    return metrics, rows


def _git_commit_or_none() -> str | None:
    head = Path(".git/HEAD")
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            return (Path(".git") / value[5:]).read_text(encoding="utf-8").strip()
        return value
    except OSError:
        return None


def aggregate_summary(
    episode_metrics: Sequence[dict[str, Any]], all_rows: Sequence[dict[str, Any]],
    dataset_root: Path, config: dict[str, Any], config_sha256: str,
) -> dict[str, Any]:
    total_camera = sum(item["counts"]["total_camera_frames"] for item in episode_metrics)
    active_camera = sum(item["counts"]["active_window_camera_frames"] for item in episode_metrics)
    accepted = len(all_rows)
    steering = [float(row["steering_rad"]) for row in all_rows]
    steering_ages = [float(row["steering_age_ms"]) for row in all_rows]
    speed_ages = [float(row["speed_age_ms"]) for row in all_rows]
    by_episode_times = [
        [int(row["camera_record_time_ns"]) for row in all_rows if row["episode_id"] == item["episode_id"]]
        for item in episode_metrics
    ]
    intervals = [
        (b - a) / 1e6 for times in by_episode_times for a, b in zip(times, times[1:])
    ]
    source_bytes = sum(item["source"]["mcap_size_bytes"] for item in episode_metrics)
    derived_bytes = _directory_size(dataset_root)
    rejections = Counter()
    for item in episode_metrics:
        rejections.update(item["counts"]["rejection_by_reason"])
    interval_stats = numeric_distribution(intervals)
    interval_stats.pop("median")
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "episode_count": len(episode_metrics),
        "episode_ids": [item["episode_id"] for item in episode_metrics],
        "extractor_config_sha256": config_sha256,
        "split_policy": "No frame-level split. Future splitting must preserve whole episode_id groups.",
        "counts": {
            "total_camera_frames": total_camera,
            "active_window_camera_frames": active_camera,
            "accepted_camera_samples": accepted,
            "rejected_camera_frames": total_camera - accepted,
            "rejection_by_reason": dict(sorted(rejections.items())),
            "retention_fraction": accepted / total_camera,
            "active_window_retention_fraction": accepted / active_camera if active_camera else 0.0,
        },
        "synchronization": {
            "steering_age_ms": numeric_distribution(steering_ages),
            "speed_age_ms": numeric_distribution(speed_ages),
            "accepted_camera_interval_ms": interval_stats,
            "future_label_violations": sum(item["synchronization"]["future_label_violations"] for item in episode_metrics),
            "rule": "causal zero-order hold on MCAP record timestamps",
        },
        "steering_distribution": steering_distribution(steering, config),
        "storage": {
            "source_mcap_total_size_bytes": source_bytes,
            "derived_dataset_total_size_bytes": derived_bytes,
            "derived_bytes_per_accepted_frame": derived_bytes / accepted if accepted else None,
            "derived_mebibytes_per_accepted_frame": derived_bytes / accepted / (1024 * 1024) if accepted else None,
            "raw_to_derived_ratio": source_bytes / derived_bytes if derived_bytes else None,
            "derived_to_raw_fraction": derived_bytes / source_bytes if source_bytes else None,
        },
        "episodes": [
            {"episode_id": item["episode_id"], "result": item["result"], "accepted_samples": item["counts"]["accepted_camera_samples"]}
            for item in episode_metrics
        ],
        "pilot_success_gate": {
            "all_three_bags_readable": len(episode_metrics) == 3,
            "nonzero_samples_each": all(item["counts"]["accepted_camera_samples"] > 0 for item in episode_metrics),
            "future_label_violations_zero": all(item["synchronization"]["future_label_violations"] == 0 for item in episode_metrics),
            "age_gates_satisfied": all(
                item["synchronization"]["steering_age_ms"]["max"] <= config["maximum_steering_age_s"] * 1000
                and item["synchronization"]["speed_age_ms"]["max"] <= config["maximum_speed_age_s"] * 1000
                for item in episode_metrics
            ),
            "episode_level_separation_preserved": len({row["episode_id"] for row in all_rows}) == 3,
        },
    }


def resolve_paths(repo_root: Path, sim_root: Path | None, input_root: Path | None, output_root: Path | None):
    if input_root is None:
        if sim_root is None:
            raise ExtractionError("provide --sim-root or --input-root")
        input_root = sim_root / "userdata" / "physicar_e2e" / "rosbag_collector_v1_pilot"
    if output_root is None:
        if sim_root is None:
            raise ExtractionError("provide --output-root when using --input-root without --sim-root")
        output_root = sim_root / "userdata" / "physicar_e2e" / "dataset_extractor_v1_pilot"
    metadata_root = repo_root / "results" / "rosbag_collector_v1_pilot"
    return input_root.resolve(), output_root.resolve(), metadata_root.resolve()


def prepare_output_root(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output dataset already exists: {path}; use --force to replace it")
        if path == Path(path.anchor) or len(path.parts) < 4:
            raise ExtractionError(f"refusing to remove unsafe output path: {path}")
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "manifests").mkdir()
    (path / "previews").mkdir()


def run_extraction(
    *, repo_root: Path, config_path: Path, input_root: Path,
    output_root: Path, force: bool = False, write_repo_results: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_sha = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    prepare_output_root(output_root, force)
    episode_metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    try:
        for number in range(1, 4):
            episode_id = f"episode_{number:03d}"
            episode_root = input_root / episode_id / "bag"
            mcap_files = sorted(episode_root.glob("*.mcap"))
            if len(mcap_files) != 1:
                raise ExtractionError(f"{episode_id}: expected exactly one MCAP under {episode_root}, found {len(mcap_files)}")
            collector_path = repo_root / "results" / "rosbag_collector_v1_pilot" / f"{episode_id}.json"
            metrics, rows = extract_episode(
                episode_id=episode_id,
                mcap_path=mcap_files[0],
                collector_metadata_path=collector_path,
                dataset_root=output_root,
                config=config,
                config_sha256=config_sha,
                source_path_identity=mcap_files[0].relative_to(input_root).as_posix(),
                collector_metadata_identity=collector_path.relative_to(repo_root).as_posix(),
            )
            episode_metrics.append(metrics)
            all_rows.extend(rows)
        write_manifest(output_root / "manifest.csv", all_rows)
        summary = aggregate_summary(episode_metrics, all_rows, output_root, config, config_sha)
        metadata = {
            **summary,
            "config": config,
            "dataset_root": output_root.as_posix(),
            "input_root": input_root.as_posix(),
            "episode_metrics": episode_metrics,
        }
        (output_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
        # Recompute storage after the metadata itself exists.
        final_size = _directory_size(output_root)
        metadata["storage"]["derived_dataset_total_size_bytes"] = final_size
        metadata["storage"]["derived_bytes_per_accepted_frame"] = final_size / len(all_rows)
        metadata["storage"]["derived_mebibytes_per_accepted_frame"] = final_size / len(all_rows) / (1024 * 1024)
        metadata["storage"]["raw_to_derived_ratio"] = metadata["storage"]["source_mcap_total_size_bytes"] / final_size
        metadata["storage"]["derived_to_raw_fraction"] = final_size / metadata["storage"]["source_mcap_total_size_bytes"]
        (output_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
        compact_summary = {key: value for key, value in metadata.items() if key not in {"config", "dataset_root", "input_root", "episode_metrics"}}
        compact_summary["storage"] = metadata["storage"]
        if write_repo_results:
            result_dir = repo_root / "results" / "dataset_extractor_v1_pilot"
            result_dir.mkdir(parents=True, exist_ok=True)
            for item in episode_metrics:
                (result_dir / f"{item['episode_id']}.json").write_bytes(canonical_json_bytes(item))
            (repo_root / "results" / "dataset_extractor_v1_pilot_summary.json").write_bytes(canonical_json_bytes(compact_summary))
        return metadata
    except Exception:
        # Preserve partial output for diagnosis; never touch source bags.
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, help="simulator repository root containing userdata/")
    parser.add_argument("--input-root", type=Path, help="explicit rosbag_collector_v1_pilot root")
    parser.add_argument("--output-root", type=Path, help="external derived dataset root")
    parser.add_argument("--config", type=Path, help="extractor JSON configuration")
    parser.add_argument("--force", action="store_true", help="replace an existing output dataset")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = args.config or repo_root / "configs" / "dataset_extractor_v1.json"
    input_root, output_root, _ = resolve_paths(repo_root, args.sim_root, args.input_root, args.output_root)
    result = run_extraction(
        repo_root=repo_root, config_path=config_path.resolve(), input_root=input_root,
        output_root=output_root, force=args.force,
    )
    print(json.dumps({
        "result": result["result"],
        "accepted_samples": result["counts"]["accepted_camera_samples"],
        "dataset_root": output_root.as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
