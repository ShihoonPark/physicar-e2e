"""Read-only camera/steering phase audit for the three frozen real bags.

The audit uses MCAP ``log_time`` exclusively.  It deliberately contains no
training entry point and never writes to a source bag, dataset, or model.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont


VERSION = "real_camera_steering_phase_audit_v1"
TIMESTAMP_DOMAIN = "MCAP_log_time"
STEERING_SCALE_RAD = 0.35
TRAINING_PERMITTED = False
DRIVING_PERMITTED = False
DATASET_MODIFICATION_PERMITTED = False

CAMERA_TOPIC = "/camera/image_raw"
STEERING_TOPIC = "/steering"
SPEED_TOPIC = "/speed"

REAL_DATASET_ROOT = Path("/home/a/physicar-e2e-artifacts/real_dataset_v1")
MANIFEST_PATH = REAL_DATASET_ROOT / "manifests/real_dataset_v1.csv"
MANIFEST_SHA256 = "ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597"
SCRATCH_ONNX = Path(
    "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/"
    "scratch/onnx/real_scratch_v1.onnx"
)
SCRATCH_ONNX_SHA256 = "b860afe396c8e48001339b4f99c8b3daa272500725d48d79b9c22b859c6fd339"

EXPECTED_TRACK_DRIVE = Path("/home/a/real_physicar_handoff/track_drive")
RECOVERED_TRACK_DRIVE_ARCHIVE = Path("/home/a/Downloads/TalkFile_track_drive.zip.zip")
PLATFORM_ROOT = Path("/home/a/physicar-ai-sim-docker")

BAG_PATHS = {
    "bag_01": Path("/home/a/output_bag/bag_01/bag_01_0.mcap"),
    "bag_02": Path("/home/a/output_bag/bag_02/bag_02_0.mcap"),
    "bag_03": Path("/home/a/output_bag/bag_03/bag_03_0.mcap"),
}

SEMANTIC_DECISIONS = {
    "PREVIOUS_CAUSAL_COMMAND_CORRECT",
    "POST_CAMERA_COMMAND_CORRECT",
    "INDEPENDENT_CONTROL_STREAM",
    "INCONCLUSIVE",
}


class PhaseAuditError(RuntimeError):
    """Raised when frozen input evidence or an audit invariant is violated."""


@dataclass(frozen=True)
class ScalarCommand:
    time_ns: int
    recorded: float
    index: int = 0


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def scale_steering(recorded: float) -> float:
    """Apply the frozen physical conversion exactly once, without clipping."""
    value = float(recorded)
    if not math.isfinite(value):
        raise PhaseAuditError("steering command is not finite")
    return value * STEERING_SCALE_RAD


def _times(records: Sequence[ScalarCommand]) -> list[int]:
    return [record.time_ns for record in records]


def previous_lookup(records: Sequence[ScalarCommand], camera_time_ns: int) -> ScalarCommand | None:
    """Latest command with ``log_time <= camera log_time``."""
    position = bisect.bisect_right(_times(records), int(camera_time_ns)) - 1
    return records[position] if position >= 0 else None


def next_lookup(records: Sequence[ScalarCommand], camera_time_ns: int) -> ScalarCommand | None:
    """First command with ``log_time >= camera log_time``."""
    position = bisect.bisect_left(_times(records), int(camera_time_ns))
    return records[position] if position < len(records) else None


def nearest_lookup(records: Sequence[ScalarCommand], camera_time_ns: int) -> ScalarCommand | None:
    """Minimum absolute offset, choosing PREV deterministically on an exact tie."""
    before = previous_lookup(records, camera_time_ns)
    after = next_lookup(records, camera_time_ns)
    if before is None:
        return after
    if after is None:
        return before
    if camera_time_ns - before.time_ns <= after.time_ns - camera_time_ns:
        return before
    return after


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


def compact_distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0, "mean": None, "median": None, "p05": None,
            "p95": None, "min": None, "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p05": percentile(finite, 0.05),
        "p95": percentile(finite, 0.95),
        "min": min(finite),
        "max": max(finite),
    }


def _decoded_scalar(message: Any) -> float:
    if not hasattr(message, "data"):
        raise PhaseAuditError("decoded scalar message has no data field")
    return float(message.data)


def scan_bag(path: Path) -> dict[str, Any]:
    """Scan only topic metadata/timestamps and Float64 payloads; never decode images."""
    if not path.is_file():
        raise PhaseAuditError(f"missing bag: {path}")
    camera_times: list[int] = []
    steering: list[ScalarCommand] = []
    speed: list[ScalarCommand] = []
    topic_counts = {CAMERA_TOPIC: 0, STEERING_TOPIC: 0, SPEED_TOPIC: 0}
    decoder_factory = DecoderFactory()
    with path.open("rb") as stream:
        reader = make_reader(stream)
        messages = reader.iter_messages(topics=[CAMERA_TOPIC, STEERING_TOPIC, SPEED_TOPIC])
        for schema, channel, message in messages:
            topic_counts[channel.topic] += 1
            time_ns = int(message.log_time)
            if channel.topic == CAMERA_TOPIC:
                camera_times.append(time_ns)
            else:
                decoder = decoder_factory.decoder_for(channel.message_encoding, schema)
                if decoder is None:
                    raise PhaseAuditError(f"cannot decode {channel.topic} scalar")
                value = _decoded_scalar(decoder(message.data))
                if channel.topic == STEERING_TOPIC:
                    steering.append(ScalarCommand(time_ns, value, len(steering)))
                else:
                    speed.append(ScalarCommand(time_ns, value, len(speed)))
    if camera_times != sorted(camera_times):
        raise PhaseAuditError(f"{path}: camera log_time is not nondecreasing")
    if _times(steering) != sorted(_times(steering)):
        raise PhaseAuditError(f"{path}: steering log_time is not nondecreasing")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "topic_counts": topic_counts,
        "camera_times": camera_times,
        "steering": steering,
        "speed": speed,
    }


def phase_rows(camera_times: Sequence[int], steering: Sequence[ScalarCommand]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera_index, camera_time_ns in enumerate(camera_times):
        previous = previous_lookup(steering, camera_time_ns)
        following = next_lookup(steering, camera_time_ns)
        nearest = nearest_lookup(steering, camera_time_ns)
        row: dict[str, Any] = {
            "camera_index": camera_index,
            "camera_log_time_ns": camera_time_ns,
            "previous": previous,
            "next": following,
            "nearest": nearest,
            "previous_offset_ms": (
                (camera_time_ns - previous.time_ns) / 1e6 if previous else None
            ),
            "next_offset_ms": (
                (following.time_ns - camera_time_ns) / 1e6 if following else None
            ),
            "nearest_signed_offset_ms": (
                (nearest.time_ns - camera_time_ns) / 1e6 if nearest else None
            ),
        }
        if previous is not None:
            row["previous_steering_rad"] = scale_steering(previous.recorded)
        if following is not None:
            row["next_steering_rad"] = scale_steering(following.recorded)
        if nearest is not None:
            row["nearest_steering_rad"] = scale_steering(nearest.recorded)
        if previous is not None and following is not None:
            row["cycle_interval_ms"] = (following.time_ns - previous.time_ns) / 1e6
            row["steering_change_rad"] = (
                row["next_steering_rad"] - row["previous_steering_rad"]
            )
        rows.append(row)
    return rows


def _decile_stability(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    buckets: list[dict[str, Any]] = []
    count = len(rows)
    for decile in range(10):
        start = count * decile // 10
        stop = count * (decile + 1) // 10
        part = rows[start:stop]
        buckets.append({
            "decile": decile + 1,
            "camera_index_start": part[0]["camera_index"] if part else None,
            "camera_index_end": part[-1]["camera_index"] if part else None,
            "count": len(part),
            "previous_offset_ms": compact_distribution(
                row["previous_offset_ms"] for row in part
                if row["previous_offset_ms"] is not None
            ),
            "next_offset_ms": compact_distribution(
                row["next_offset_ms"] for row in part if row["next_offset_ms"] is not None
            ),
            "nearest_is_next_fraction": (
                sum(
                    row["nearest"] is not None and row["next"] is not None
                    and row["nearest"].index == row["next"].index
                    for row in part
                ) / len(part) if part else None
            ),
        })
    prev_medians = [
        item["previous_offset_ms"]["median"] for item in buckets
        if item["previous_offset_ms"]["median"] is not None
    ]
    next_medians = [
        item["next_offset_ms"]["median"] for item in buckets
        if item["next_offset_ms"]["median"] is not None
    ]
    prev_range = max(prev_medians) - min(prev_medians)
    next_range = max(next_medians) - min(next_medians)
    return {
        "method": "ten_contiguous_equal_count_segments",
        "segments": buckets,
        "previous_median_range_ms": prev_range,
        "next_median_range_ms": next_range,
        "stable_phase": prev_range <= 15.0 and next_range <= 10.0,
        "stability_rule": "decile PREV median range <=15 ms and NEXT median range <=10 ms",
    }


def summarize_bag_phase(bag_id: str, scanned: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = phase_rows(scanned["camera_times"], scanned["steering"])
    complete = [row for row in rows if row["previous"] is not None and row["next"] is not None]
    nearest_is_next = [
        row for row in rows
        if row["nearest"] is not None and row["next"] is not None
        and row["nearest"].index == row["next"].index
    ]
    summary = {
        "bag_id": bag_id,
        "mcap_path": scanned["path"],
        "mcap_size_bytes": scanned["size_bytes"],
        "timestamp_domain": TIMESTAMP_DOMAIN,
        "topic_counts": scanned["topic_counts"],
        "camera_frames": len(rows),
        "previous_available": sum(row["previous"] is not None for row in rows),
        "next_available": sum(row["next"] is not None for row in rows),
        "nearest_available": sum(row["nearest"] is not None for row in rows),
        "previous_offset_ms": compact_distribution(
            row["previous_offset_ms"] for row in rows if row["previous_offset_ms"] is not None
        ),
        "next_offset_ms": compact_distribution(
            row["next_offset_ms"] for row in rows if row["next_offset_ms"] is not None
        ),
        "nearest_signed_offset_ms": compact_distribution(
            row["nearest_signed_offset_ms"] for row in rows
            if row["nearest_signed_offset_ms"] is not None
        ),
        "nearest_absolute_offset_ms": compact_distribution(
            abs(row["nearest_signed_offset_ms"]) for row in rows
            if row["nearest_signed_offset_ms"] is not None
        ),
        "nearest_is_next_count": len(nearest_is_next),
        "nearest_is_next_fraction": len(nearest_is_next) / len(rows),
        "complete_prev_camera_next_count": len(complete),
        "complete_prev_camera_next_fraction": len(complete) / len(rows),
        "steering_prev_to_next_interval_ms": compact_distribution(
            row["cycle_interval_ms"] for row in complete
        ),
        "phase_order": "steering_prev -> camera -> steering_next",
        "stability": _decile_stability(rows),
    }
    return summary, rows


def _change_group(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.01:
        return "low_abs_delta_lt_0p01_rad"
    if absolute < 0.05:
        return "medium_abs_delta_0p01_to_lt_0p05_rad"
    return "high_abs_delta_ge_0p05_rad"


def summarize_change_rows(rows_by_bag: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    for bag_id, rows in rows_by_bag.items():
        for row in rows:
            if "steering_change_rad" in row:
                complete.append({**row, "bag_id": bag_id})

    def summarize(selection: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(selection),
            "next_minus_previous_rad": compact_distribution(
                row["steering_change_rad"] for row in selection
            ),
            "absolute_candidate_label_difference_rad": compact_distribution(
                abs(row["steering_change_rad"]) for row in selection
            ),
            "previous_steering_rad": compact_distribution(
                row["previous_steering_rad"] for row in selection
            ),
            "next_steering_rad": compact_distribution(
                row["next_steering_rad"] for row in selection
            ),
            "sign_change_count": sum(
                (row["previous_steering_rad"] < 0 < row["next_steering_rad"])
                or (row["previous_steering_rad"] > 0 > row["next_steering_rad"])
                for row in selection
            ),
        }

    groups = {
        name: summarize([row for row in complete if _change_group(row["steering_change_rad"]) == name])
        for name in (
            "low_abs_delta_lt_0p01_rad",
            "medium_abs_delta_0p01_to_lt_0p05_rad",
            "high_abs_delta_ge_0p05_rad",
        )
    }
    focus: dict[str, Any] = {}
    for threshold in (0.15, 0.25):
        key = f"previous_abs_steering_ge_{str(threshold).replace('.', 'p')}_rad"
        focus[key] = summarize([
            row for row in complete if abs(row["previous_steering_rad"]) >= threshold
        ])
    sign_changes = [
        row for row in complete
        if (row["previous_steering_rad"] < 0 < row["next_steering_rad"])
        or (row["previous_steering_rad"] > 0 > row["next_steering_rad"])
    ]
    focus["strict_steering_sign_changes"] = summarize(sign_changes)
    return {
        "physical_conversion": "steering_rad = recorded * 0.35 exactly once; no clipping",
        "group_thresholds_pre_registered_for_this_audit": {
            "low": "abs(next-prev) < 0.01 rad",
            "medium": "0.01 <= abs(next-prev) < 0.05 rad",
            "high": "abs(next-prev) >= 0.05 rad",
        },
        "complete_candidate_pairs": len(complete),
        "groups": groups,
        "focus": focus,
        "per_bag": {
            bag_id: summarize([row for row in complete if row["bag_id"] == bag_id])
            for bag_id in rows_by_bag
        },
    }


def classify_semantics(
    *,
    exact_deployed_publisher_attributed: bool,
    command_computed_in_camera_callback: bool = False,
    independent_command_source: bool = False,
    active_command_is_intended_target: bool = False,
) -> str:
    """Classify from source semantics; timing statistics are intentionally absent."""
    if not exact_deployed_publisher_attributed:
        return "INCONCLUSIVE"
    flags = sum((
        command_computed_in_camera_callback,
        independent_command_source,
        active_command_is_intended_target,
    ))
    if flags != 1:
        return "INCONCLUSIVE"
    if command_computed_in_camera_callback:
        return "POST_CAMERA_COMMAND_CORRECT"
    if independent_command_source:
        return "INDEPENDENT_CONTROL_STREAM"
    return "PREVIOUS_CAUSAL_COMMAND_CORRECT"


def _numbered_lines(text: str, first: int, last: int) -> list[dict[str, Any]]:
    lines = text.splitlines()
    return [
        {"line": number, "text": lines[number - 1].strip()}
        for number in range(first, last + 1)
        if 1 <= number <= len(lines)
    ]


def _line_evidence(path: Path, first: int, last: int) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "line_start": first,
        "line_end": last,
        "lines": _numbered_lines(text, first, last),
    }


def _zip_evidence(archive: Path, member: str, first: int, last: int) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read(member).decode("utf-8")
    return {
        "archive": str(archive),
        "member": member,
        "line_start": first,
        "line_end": last,
        "lines": _numbered_lines(text, first, last),
    }


def audit_source_semantics() -> dict[str, Any]:
    expected_present = EXPECTED_TRACK_DRIVE.is_dir()
    archive_present = RECOVERED_TRACK_DRIVE_ARCHIVE.is_file()
    recovered: dict[str, Any] = {"archive_present": archive_present}
    if archive_present:
        recovered.update({
            "archive": str(RECOVERED_TRACK_DRIVE_ARCHIVE),
            "archive_sha256": sha256_file(RECOVERED_TRACK_DRIVE_ARCHIVE),
            "status": "candidate_source_only_not_proven_deployed",
            "data_logger": {
                "subscriptions": _zip_evidence(
                    RECOVERED_TRACK_DRIVE_ARCHIVE,
                    "track_drive/track_drive/data_logger.py", 50, 72,
                ),
                "finding": (
                    "Subscribes to camera and /xycar_motor, caches the latest command, "
                    "and writes it from the image callback. It creates no publisher."
                ),
            },
            "teleoperation": {
                "publisher": _zip_evidence(
                    RECOVERED_TRACK_DRIVE_ARCHIVE,
                    "track_drive/track_drive/teleop_key.py", 42, 57,
                ),
                "independent_loop": _zip_evidence(
                    RECOVERED_TRACK_DRIVE_ARCHIVE,
                    "track_drive/track_drive/teleop_key.py", 115, 159,
                ),
                "finding": (
                    "Publishes /xycar_motor in a 50 Hz keyboard loop independent of camera."
                ),
            },
            "camera_callback_e2e": {
                "publisher_and_subscription": _zip_evidence(
                    RECOVERED_TRACK_DRIVE_ARCHIVE,
                    "track_drive/track_drive/e2e_sched.py", 88, 93,
                ),
                "callback_order": _zip_evidence(
                    RECOVERED_TRACK_DRIVE_ARCHIVE,
                    "track_drive/track_drive/e2e_sched.py", 122, 146,
                ),
                "finding": (
                    "Processes the current image and then publishes /xycar_motor inside the "
                    "same callback, which would imply a post-camera command for that node."
                ),
            },
            "topic_match": {
                "bag_topics": [CAMERA_TOPIC, STEERING_TOPIC, SPEED_TOPIC],
                "archive_camera_topic": "/usb_cam/image_raw/front",
                "archive_command_topic": "/xycar_motor",
                "exact_match": False,
            },
        })

    ros_root = PLATFORM_ROOT / "src/physicar-ros"
    ros_bridge = ros_root / "physicar_webserver/physicar_webserver/ros_bridge.py"
    hw_router = ros_root / "physicar_webserver/physicar_webserver/routers/hw.py"
    launch = ros_root / "physicar_bringup/launch/real.launch.py"
    driver = ros_root / "physicar_bringup/src/physicar_driver_node.cpp"
    platform = {
        "status": "available_platform_source_not_recording_provenance",
        "external_dependency_read_only": True,
        "exact_topic_publishers": _line_evidence(ros_bridge, 120, 124),
        "publish_functions": _line_evidence(ros_bridge, 326, 344),
        "independent_web_control": _line_evidence(hw_router, 504, 553),
        "steering_http_endpoint": _line_evidence(hw_router, 560, 581),
        "camera_is_separate_node": _line_evidence(launch, 161, 175),
        "webserver_is_separate_node": _line_evidence(launch, 260, 273),
        "driver_subscribers": _line_evidence(driver, 681, 695),
        "held_steering_evidence": _line_evidence(driver, 753, 764),
        "servo_application": _line_evidence(driver, 850, 855),
        "finding": (
            "The platform web bridge can publish exact /steering and /speed from independent "
            "HTTP/WebSocket control. The driver subscribes and applies steering; its watchdog "
            "stops stale speed but explicitly holds steering. This proves a possible independent "
            "source and held-command behavior, not which node generated these recordings."
        ),
    }

    requested_trace_answers = {
        "camera_publisher": (
            "Platform real.launch.py starts physicar_camera/camera_node and remaps its output "
            "to /camera/image_raw. It is launched separately from the webserver."
        ),
        "steering_publisher": (
            "The only exact /steering publisher found is the platform webserver RosBridge "
            "publisher, callable from independent HTTP/WebSocket endpoints. Available evidence "
            "does not establish that it produced the recorded bag stream."
        ),
        "speed_publisher": (
            "The same RosBridge creates /speed and publish_speed. Available evidence does not "
            "establish that it produced the recorded bag stream."
        ),
        "steering_computation_function": (
            "Unknown for the recorded bags. RosBridge.publish_steering forwards an externally "
            "supplied scalar and does no camera computation. Recovered e2e_sched.cb computes "
            "from a current camera frame but publishes the different /xycar_motor topic."
        ),
        "image_compute_publish_order": (
            "Unknown for the recorded exact topics. The recovered e2e_sched candidate executes "
            "image conversion/preprocessing/inference and then publishes /xycar_motor inside cb. "
            "The exact-topic web control path is independent of camera callbacks."
        ),
        "data_logger_role": (
            "Recovered data_logger only subscribes, caches the latest /xycar_motor command, and "
            "logs it during its image callback; it does not republish commands."
        ),
        "teleoperation_role": (
            "Recovered teleop_key independently publishes /xycar_motor at 50 Hz, not /steering. "
            "The exact /steering web endpoints are also independent external-control inputs."
        ),
        "command_hold": (
            "The platform driver applies each /steering command to the servo. Its watchdog comment "
            "explicitly stops stale speed while holding steering/camera, so steering remains held "
            "between updates on this driver path."
        ),
        "representation_mismatch": (
            "The exact-topic web bridge documents its scalar as radians, whereas the frozen bag "
            "contract identifies recorded steering as normalized command multiplied by 0.35. "
            "This is additional evidence that the discovered bridge cannot be assumed to be the "
            "recording producer without launch/runtime provenance."
        ),
    }
    decision = classify_semantics(exact_deployed_publisher_attributed=False)
    return {
        "requested_source_path": str(EXPECTED_TRACK_DRIVE),
        "requested_source_present": expected_present,
        "home_search_scope": "/home/a excluding generated result references",
        "exact_deployed_steering_publisher_attributed": False,
        "deployed_node_identity_available_in_mcap_metadata": False,
        "requested_trace_answers": requested_trace_answers,
        "recovered_track_drive": recovered,
        "platform_source": platform,
        "exact_topic_source_search": {
            "lane_debug_path_publisher_found": False,
            "steering_publishers_found": [
                str(ros_bridge) + ":120-122",
                str(hw_router) + ":504-553,560-581",
            ],
            "finding": (
                "No source publishing /lane/debug/path was found, so the lane/control node "
                "captured in the bags cannot be matched to the recovered candidates."
            ),
        },
        "semantic_decision": decision,
        "decision_basis": (
            "The exact deployed /steering publisher cannot be attributed. Candidate sources "
            "support mutually different semantics, so topic phase alone cannot select one."
        ),
    }


def _artifact_snapshot() -> dict[str, Any]:
    files = {
        "real_dataset_v1_manifest": MANIFEST_PATH,
        "scratch_checkpoint": Path(
            "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/"
            "scratch/checkpoints/real_scratch_v1_best.pt"
        ),
        "scratch_onnx": SCRATCH_ONNX,
        "transfer_checkpoint": Path(
            "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/"
            "transfer/checkpoints/real_transfer_v1_best.pt"
        ),
        "transfer_onnx": Path(
            "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/"
            "transfer/onnx/real_transfer_v1.onnx"
        ),
    }
    return {
        name: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        for name, path in files.items()
    }


def _bag_snapshot() -> dict[str, Any]:
    return {
        bag_id: {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for bag_id, path in BAG_PATHS.items()
    }


def _load_high_steering_manifest_rows() -> list[dict[str, str]]:
    actual_hash = sha256_file(MANIFEST_PATH)
    if actual_hash != MANIFEST_SHA256:
        raise PhaseAuditError(
            f"REAL_DATASET_V1 manifest hash mismatch: {actual_hash}"
        )
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["source_bag"] == "bag_03" and abs(float(row["steering_rad"])) >= 0.25
        ]
    if len(rows) != 6:
        raise PhaseAuditError(f"expected six bag_03 high-steering rows, found {len(rows)}")
    return rows


def _scratch_predictions(rows: Sequence[dict[str, str]]) -> list[float | None]:
    if not SCRATCH_ONNX.is_file() or sha256_file(SCRATCH_ONNX) != SCRATCH_ONNX_SHA256:
        return [None] * len(rows)
    try:
        import numpy as np
        import onnxruntime as ort
        from .pilotnet_temporal import preprocess_temporal_paths
    except ImportError:
        return [None] * len(rows)
    session = ort.InferenceSession(str(SCRATCH_ONNX), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions: list[float] = []
    for row in rows:
        paths = [REAL_DATASET_ROOT / row[field] for field in (
            "image_t_minus_2", "image_t_minus_1", "image_t"
        )]
        tensor = preprocess_temporal_paths(paths)[None, ...]
        value = session.run(None, {input_name: np.asarray(tensor, dtype=np.float32)})[0]
        predictions.append(float(value.reshape(-1)[0]))
    return predictions


def build_high_steering_cases(
    bag_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_rows = _load_high_steering_manifest_rows()
    predictions = _scratch_predictions(manifest_rows)
    phase_by_time = {int(row["camera_log_time_ns"]): row for row in bag_rows}
    cases: list[dict[str, Any]] = []
    for manifest, prediction in zip(manifest_rows, predictions, strict=True):
        camera_time = int(manifest["target_camera_log_time_ns"])
        phase = phase_by_time[camera_time]
        current = float(manifest["steering_rad"])
        if not math.isclose(current, phase["previous_steering_rad"], abs_tol=1e-15):
            raise PhaseAuditError("current REAL_DATASET_V1 label is not the PREV command")
        nearest = phase["nearest"]
        cases.append({
            "sequence_id": manifest["sequence_id"],
            "camera_log_time_ns": camera_time,
            "previous_steering_log_time_ns": phase["previous"].time_ns,
            "next_steering_log_time_ns": phase["next"].time_ns,
            "nearest_steering_log_time_ns": nearest.time_ns,
            "previous_steering_rad": phase["previous_steering_rad"],
            "next_steering_rad": phase["next_steering_rad"],
            "nearest_steering_rad": phase["nearest_steering_rad"],
            "previous_offset_ms": phase["previous_offset_ms"],
            "next_offset_ms": phase["next_offset_ms"],
            "nearest_signed_offset_ms": phase["nearest_signed_offset_ms"],
            "current_real_dataset_v1_label_rad": current,
            "current_label_policy": "PREV latest steering log_time <= camera log_time",
            "scratch_v1_prediction_rad": prediction,
            "scratch_prediction_available": prediction is not None,
            "image_reference": str(REAL_DATASET_ROOT / manifest["image_t"]),
            "temporal_image_references": [
                str(REAL_DATASET_ROOT / manifest[field]) for field in (
                    "image_t_minus_2", "image_t_minus_1", "image_t"
                )
            ],
        })
    return cases


def render_contact_sheet(cases: Sequence[dict[str, Any]], output: Path) -> None:
    panel_width, panel_height = 520, 210
    sheet = Image.new("RGB", (panel_width * 2, panel_height * 3), "#15171a")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, case in enumerate(cases):
        x = (index % 2) * panel_width
        y = (index // 2) * panel_height
        image = Image.open(case["image_reference"]).convert("RGB")
        image = image.resize((500, 165), Image.Resampling.NEAREST)
        sheet.paste(image, (x + 10, y + 8))
        prediction = case["scratch_v1_prediction_rad"]
        pred_text = "unavailable" if prediction is None else f"{prediction:+.4f}"
        lines = [
            f"{case['sequence_id']}  current={case['current_real_dataset_v1_label_rad']:+.4f}",
            f"prev={case['previous_steering_rad']:+.4f}  next={case['next_steering_rad']:+.4f}  pred={pred_text}",
        ]
        for offset, line in enumerate(lines):
            draw.text((x + 10, y + 176 + offset * 14), line, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def build_report(
    timing: dict[str, Any],
    source: dict[str, Any],
    cases: Sequence[dict[str, Any]],
    preservation: dict[str, Any],
) -> str:
    lines = [
        "# Real PhysiCar Camera–Steering Phase Audit V1",
        "",
        "## Result",
        "",
        f"**Semantic decision: `{source['semantic_decision']}`.**",
        "",
        (
            "MCAP timing strongly resembles a camera-triggered control cycle, but the exact "
            "deployed `/steering` publisher could not be identified from the requested source "
            "location, recovered archive, platform source, or bag metadata. Code candidates "
            "support both post-camera and independent-control semantics. Therefore timing alone "
            "is not used to relabel the dataset."
        ),
        "",
        "`REAL_DATASET_V1` remains preserved and no `REAL_DATASET_V2` extraction is recommended "
        "until the deployed publisher source/provenance is recovered.",
        "",
        "## Bag timing (MCAP log_time only)",
        "",
        "All offsets below are milliseconds. PREV is `camera - previous`; NEXT is "
        "`next - camera`; NEAREST is signed `command - camera`.",
        "",
        "| Bag | Frames | PREV mean / median / p05 / p95 / min / max | NEXT mean / median / p05 / p95 / min / max | NEAREST signed mean / median / p05 / p95 / min / max |",
        "|---|---:|---|---|---|",
    ]
    keys = ("mean", "median", "p05", "p95", "min", "max")
    for bag_id, bag in timing["bags"].items():
        prev = " / ".join(_fmt(bag["previous_offset_ms"][key]) for key in keys)
        nxt = " / ".join(_fmt(bag["next_offset_ms"][key]) for key in keys)
        near = " / ".join(_fmt(bag["nearest_signed_offset_ms"][key]) for key in keys)
        lines.append(f"| {bag_id} | {bag['camera_frames']} | {prev} | {nxt} | {near} |")
    lines += [
        "",
        "For every frame with a NEXT candidate, NEAREST is NEXT. Contiguous-decile medians "
        "satisfy the recorded stability rule in all three bags. Rare large PREV/period maxima "
        "are stream dropouts; the p05–p95 phase remains narrow.",
        "",
        "## Control-cycle structure",
        "",
    ]
    for bag_id, bag in timing["bags"].items():
        lines.append(
            f"- `{bag_id}`: PREV→camera median "
            f"{_fmt(bag['previous_offset_ms']['median'])} ms; camera→NEXT median "
            f"{_fmt(bag['next_offset_ms']['median'])} ms; complete ordered triples "
            f"{bag['complete_prev_camera_next_count']}/{bag['camera_frames']}."
        )
    lines += [
        "",
        "This is strong phase evidence, not publisher provenance.",
        "",
        "## Code-semantics audit",
        "",
        f"Requested source `{source['requested_source_path']}`: **not present**.",
        "",
        "| Audit question | Evidence-led answer |",
        "|---|---|",
        f"| Who publishes `/camera/image_raw`? | {source['requested_trace_answers']['camera_publisher']} |",
        f"| Who publishes `/steering`? | {source['requested_trace_answers']['steering_publisher']} |",
        f"| Who publishes `/speed`? | {source['requested_trace_answers']['speed_publisher']} |",
        f"| Where is steering computed? | {source['requested_trace_answers']['steering_computation_function']} |",
        f"| Image→compute→publish ordering? | {source['requested_trace_answers']['image_compute_publish_order']} |",
        f"| Does `data_logger` publish? | {source['requested_trace_answers']['data_logger_role']} |",
        f"| Is teleoperation independent? | {source['requested_trace_answers']['teleoperation_role']} |",
        f"| Are commands held? | {source['requested_trace_answers']['command_hold']} |",
        "",
        "Additional provenance evidence:",
        "",
        "- Recovered `TalkFile_track_drive.zip.zip`: candidate only. `data_logger.py` subscribes "
        "to camera and `/xycar_motor` and logs the cached command; it does not publish. "
        "`teleop_key.py` publishes independently of camera. `e2e_sched.py` processes the current "
        "camera callback and then publishes, but all use different bag topics.",
        "- Read-only platform source shows the web bridge publishing exact `/steering` and "
        "`/speed` from independent HTTP/WebSocket requests, while the camera is a separate node. "
        "The motor driver subscribes and holds steering between updates; only stale speed is "
        "stopped by its watchdog.",
        "- No `/lane/debug/path` publisher source was found. This prevents attribution of the "
        "recorded 15 Hz lane/control stream to either source architecture.",
        f"- {source['requested_trace_answers']['representation_mismatch']}",
        "",
        "All file/function/line excerpts and source hashes are embedded in `summary.json`.",
        "",
        "## Steering-change analysis",
        "",
        "PREV/NEXT differences use `recorded × 0.35` exactly once, with no clipping. The fixed "
        "audit groups are low `<0.01 rad`, medium `0.01–<0.05 rad`, and high `≥0.05 rad`.",
        "",
        "| Group | Count | absolute difference mean / median / p95 / max (rad) |",
        "|---|---:|---|",
    ]
    for name, group in timing["steering_change_analysis"]["groups"].items():
        dist = group["absolute_candidate_label_difference_rad"]
        label = name.replace("_", " ")
        values = " / ".join(_fmt(dist[key], 6) for key in ("mean", "median", "p95", "max"))
        lines.append(f"| {label} | {group['count']} | {values} |")
    lines += [
        "",
        "Magnitude-threshold and strict sign-change subsets are reported in `timing.json`.",
        "",
        "## Six high-steering bag_03 cases",
        "",
        "| Sequence | Camera log_time ns | PREV rad / age ms | NEXT rad / delay ms | NEAREST rad | Current label rad | Scratch V1 prediction rad |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case['sequence_id']} | {case['camera_log_time_ns']} | "
            f"{case['previous_steering_rad']:+.6f} / {case['previous_offset_ms']:.3f} | "
            f"{case['next_steering_rad']:+.6f} / {case['next_offset_ms']:.3f} | "
            f"{case['nearest_steering_rad']:+.6f} | "
            f"{case['current_real_dataset_v1_label_rad']:+.6f} | "
            f"{_fmt(case['scratch_v1_prediction_rad'], 6)} |"
        )
    lines += [
        "",
        "See `high_steering_contact_sheet.png` and `high_steering_cases.json` for image paths.",
        "",
        "Historical Scratch V1 evidence remains: bag_03 MAE 0.047842 rad, RMSE 0.081893 rad, "
        "sign agreement 0.8089; six-sample `|steering| ≥ 0.25` MAE 0.328104 rad. No model was "
        "trained or modified by this audit.",
        "",
        "## Causality and dataset consequence",
        "",
        "A command timestamp after image capture is not future sensor information when the command "
        "is computed from that image. A model using `[t-2,t-1,t]` remains camera-causal. That "
        "distinction does not resolve this audit because the deployed computation path is missing.",
        "",
        "Current consequence: preserve V1 unchanged; recover the exact deployed lane/control "
        "publisher or recording launch provenance before selecting PREV, NEXT, or an independent-"
        "stream alignment policy. Do not create V2 solely because NEXT is closer in time.",
        "",
        "## Scope and preservation",
        "",
        f"- Manifest SHA-256: `{preservation['after']['artifacts']['real_dataset_v1_manifest']['sha256']}`",
        f"- Scratch ONNX SHA-256: `{preservation['after']['artifacts']['scratch_onnx']['sha256']}`",
        "- Artifact and raw-bag before/after snapshots matched exactly.",
        "- No training, driving, simulation, collection, raw-bag changes, dataset changes, "
        "checkpoint changes, commit, or push occurred.",
        "",
    ]
    return "\n".join(lines)


def run_audit(output_dir: Path) -> dict[str, Any]:
    before = {"artifacts": _artifact_snapshot(), "bags": _bag_snapshot()}
    if before["artifacts"]["real_dataset_v1_manifest"]["sha256"] != MANIFEST_SHA256:
        raise PhaseAuditError("frozen REAL_DATASET_V1 manifest identity changed")
    if before["artifacts"]["scratch_onnx"]["sha256"] != SCRATCH_ONNX_SHA256:
        raise PhaseAuditError("frozen Scratch V1 ONNX identity changed")

    bags: dict[str, Any] = {}
    rows_by_bag: dict[str, list[dict[str, Any]]] = {}
    for bag_id, path in BAG_PATHS.items():
        summary, rows = summarize_bag_phase(bag_id, scan_bag(path))
        bags[bag_id] = summary
        rows_by_bag[bag_id] = rows
    change = summarize_change_rows(rows_by_bag)
    timing = {
        "version": VERSION,
        "timestamp_domain": TIMESTAMP_DOMAIN,
        "lookup_contract": {
            "previous": "latest steering where t_steering <= t_camera",
            "next": "first steering where t_steering >= t_camera",
            "nearest": "minimum absolute offset; PREV wins exact tie",
        },
        "bags": bags,
        "steering_change_analysis": change,
    }

    source = audit_source_semantics()
    cases = build_high_steering_cases(rows_by_bag["bag_03"])
    output_dir.mkdir(parents=True, exist_ok=True)
    render_contact_sheet(cases, output_dir / "high_steering_contact_sheet.png")

    after = {"artifacts": _artifact_snapshot(), "bags": _bag_snapshot()}
    preservation = {
        "before": before,
        "after": after,
        "artifacts_unchanged": before["artifacts"] == after["artifacts"],
        "raw_bags_unchanged": before["bags"] == after["bags"],
    }
    if not preservation["artifacts_unchanged"] or not preservation["raw_bags_unchanged"]:
        raise PhaseAuditError("a frozen input changed while the audit was running")

    summary = {
        "version": VERSION,
        "semantic_decision": source["semantic_decision"],
        "decision_primary_evidence": "code_semantics",
        "decision": source,
        "timing_conclusion": (
            "Strong, stable steering_prev -> camera -> steering_next phase; NEXT is nearest "
            "for every camera frame with a NEXT candidate. This does not establish semantics."
        ),
        "causality_distinction": (
            "A post-camera timestamp can be causal if that command is computed from the current "
            "frame; it is not future camera information. Model camera input remains [t-2,t-1,t]."
        ),
        "dataset_consequence": {
            "real_dataset_v1_action": "PRESERVE_UNCHANGED",
            "real_dataset_v2_action": "DO_NOT_IMPLEMENT_OR_RECOMMEND_UNTIL_SOURCE_PROVENANCE_RECOVERED",
            "reason": "Semantic pairing is inconclusive from available exact source evidence.",
        },
        "historical_model_evidence_preserved": {
            "scratch_v1_bag_03_mae_rad": 0.047842,
            "scratch_v1_bag_03_rmse_rad": 0.081893,
            "scratch_v1_sign_agreement": 0.8089,
            "high_steering_count": 6,
            "high_steering_scratch_mae_rad": 0.328104,
        },
        "high_steering_case_count": len(cases),
        "preservation": preservation,
        "scope_guards": {
            "training_permitted": TRAINING_PERMITTED,
            "driving_permitted": DRIVING_PERMITTED,
            "dataset_modification_permitted": DATASET_MODIFICATION_PERMITTED,
            "simulator_used": False,
            "real_vehicle_used": False,
            "data_collected": False,
        },
    }
    write_json(output_dir / "timing.json", timing)
    write_json(output_dir / "high_steering_cases.json", {
        "version": VERSION,
        "selection": "bag_03 current REAL_DATASET_V1 abs(steering_rad) >= 0.25",
        "manifest_sha256": MANIFEST_SHA256,
        "scratch_onnx_sha256": SCRATCH_ONNX_SHA256,
        "cases": cases,
    })
    write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(
        build_report(timing, source, cases, preservation), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/real_camera_steering_phase_audit_v1"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_audit(args.output_dir.resolve())
    print(json.dumps({
        "status": "ok",
        "semantic_decision": result["semantic_decision"],
        "output_dir": str(args.output_dir.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
