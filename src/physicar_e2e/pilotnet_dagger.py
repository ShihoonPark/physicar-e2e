"""Extract causal on-policy DAgger data and collect one frozen V1 holdout rollout."""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics
from typing import Any, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image

from .dataset_extractor import decode_rgb8_image, preprocess_image
from .expert_driver import wait_after_reset
from .pilotnet_failure_diagnosis import (
    _collector_config as diagnosis_collector_config,
    _driver_config,
    detect_divergence_windows,
    load_model,
    run_live_loop,
    utc_now,
)
from .pilotnet_inference import CameraOnlyOnnxModel, sha256_file
from .rosbag_collector import CollectorConfig, DockerRosBackend, RecorderHandle, directory_size, verify_bag
from .sim_client import SimClient


VERSION = "pilotnet_dagger_v1"
MANIFEST_FIELDS = [
    "episode_id", "sample_index", "image_path", "camera_record_time_ns", "camera_header_time_ns",
    "expert_label_time_ns", "expert_label_age_ms", "steering_rad", "steering_normalized",
    "route_progress_m", "cte_m", "v1_steering_rad", "network_expert_difference_rad",
    "window_role", "source_mcap_sha256",
]


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION:
        raise ValueError("unexpected DAgger config version")
    if payload.get("rollout_assignment") != {"dagger_rollout_A": "training", "dagger_rollout_B": "holdout"}:
        raise ValueError("rollout A/B assignment must remain frozen")
    if payload.get("maximum_collection_rollouts") != 1 or payload.get("diagnostic_speed_mps") != 0.5:
        raise ValueError("DAgger V1 permits exactly one new V1 holdout rollout at 0.50 m/s")
    if payload.get("diagnostic_ros_topics") != ["/camera/image_raw", "/clock"]:
        raise ValueError("DAgger diagnostic bag must remain raw camera plus clock only")
    return payload


def sim_time_ns(row: dict[str, Any]) -> int:
    return int(round(float(row["sim_time_s"]) * 1e9))


def latest_causal_shadow(telemetry: Sequence[dict[str, Any]], camera_time_ns: int) -> dict[str, Any] | None:
    times = [sim_time_ns(row) for row in telemetry]
    index = bisect.bisect_right(times, int(camera_time_ns)) - 1
    return telemetry[index] if index >= 0 else None


def select_dagger_window(
    telemetry: Sequence[dict[str, Any]], divergence_index: int, pre_divergence_s: float,
) -> dict[str, Any]:
    if not telemetry or not 0 <= divergence_index < len(telemetry):
        raise ValueError("invalid telemetry/divergence index")
    divergence_ns = sim_time_ns(telemetry[divergence_index])
    start_ns = divergence_ns - int(pre_divergence_s * 1e9)
    end_ns = sim_time_ns(telemetry[-1])
    return {
        "start_sim_time_ns": start_ns, "divergence_sim_time_ns": divergence_ns,
        "end_sim_time_ns": end_ns, "rule": "continuous from 1.0 s before objective divergence onset through final valid pre-safe-stop telemetry",
    }


def window_role(camera_ns: int, divergence_ns: int, divergence_duration_s: float) -> str:
    if camera_ns < divergence_ns:
        return "pre_divergence"
    if camera_ns < divergence_ns + int(divergence_duration_s * 1e9):
        return "divergence"
    return "late_failure"


def _raw_frames(bag_path: Path):
    files = sorted(bag_path.glob("*.mcap"))
    if len(files) != 1:
        raise RuntimeError(f"expected one MCAP file, found {len(files)}")
    with files[0].open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, record, decoded in reader.iter_decoded_messages(topics=["/camera/image_raw"], log_time_order=True):
            if schema.name != "sensor_msgs/msg/Image":
                raise RuntimeError(f"unexpected raw camera type {schema.name}")
            stamp = decoded.header.stamp
            header_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            yield files[0], int(record.log_time), header_ns, decoded


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def extract_rollout(
    *, rollout_id: str, bag_path: Path, telemetry_path: Path, output_root: Path,
    config_path: Path, source_result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in config["rollout_assignment"]:
        raise ValueError("unknown frozen rollout id")
    manifest_dir = output_root / "manifests"
    image_dir = output_root / "images" / rollout_id
    preview_dir = output_root / "previews"
    manifest_path = manifest_dir / f"{rollout_id}.csv"
    if manifest_path.exists() or image_dir.exists():
        raise RuntimeError(f"refusing to overwrite extracted rollout {rollout_id}")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    windows = detect_divergence_windows(
        [row["elapsed_s"] for row in telemetry], [row["cte_m"] for row in telemetry],
        stable_window_s=config["stable_window_s"], cte_floor_m=config["divergence_cte_floor_m"],
        persistence_samples=config["divergence_persistence_samples"],
    )
    selected = select_dagger_window(telemetry, windows["divergence_index"], config["window_pre_divergence_s"])
    frame_records = list(_raw_frames(bag_path))
    eligible = [item for item in frame_records if selected["start_sim_time_ns"] <= item[2] <= selected["end_sim_time_ns"]]
    fallback = False
    if len(eligible) < int(config["minimum_selected_samples"]):
        fallback = True
        selected["start_sim_time_ns"] = sim_time_ns(telemetry[0])
        eligible = [item for item in frame_records if selected["start_sim_time_ns"] <= item[2] <= selected["end_sim_time_ns"]]
        selected["rule"] = "full active rollout fallback because objective window had fewer than 20 raw frames"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    image_config = {
        "source_width": config["source_width"], "source_height": config["source_height"],
        "source_encoding": config["source_encoding"], "roi": config["roi"],
        "output_width": config["output_width"], "output_height": config["output_height"],
    }
    rows: list[dict[str, Any]] = []
    violations = 0
    rejected_stale = 0
    source_mcap = frame_records[0][0]
    source_sha = sha256_file(source_mcap)
    for _, record_ns, header_ns, decoded in eligible:
        label = latest_causal_shadow(telemetry, header_ns)
        if label is None:
            continue
        label_ns = sim_time_ns(label)
        age_ns = header_ns - label_ns
        if age_ns < 0:
            violations += 1
            continue
        if age_ns / 1e9 > float(config["maximum_expert_label_age_s"]):
            rejected_stale += 1
            continue
        expert = float(label["shadow_expert_steering_rad"])
        network = float(label["network_steering_rad"])
        if not all(math.isfinite(value) for value in (expert, network)) or abs(expert) > config["max_steering_rad"] + 1e-9:
            raise RuntimeError("non-finite or out-of-bounds DAgger label")
        image = preprocess_image(decode_rgb8_image(decoded, image_config), image_config)
        index = len(rows)
        relative = Path("images") / rollout_id / f"frame_{index:06d}.png"
        image.save(output_root / relative, format="PNG", optimize=False)
        image.close()
        rows.append({
            "episode_id": rollout_id, "sample_index": index, "image_path": relative.as_posix(),
            "camera_record_time_ns": record_ns, "camera_header_time_ns": header_ns,
            "expert_label_time_ns": label_ns, "expert_label_age_ms": age_ns / 1e6,
            "steering_rad": expert, "steering_normalized": expert / config["max_steering_rad"],
            "route_progress_m": label["unwrapped_progress_m"], "cte_m": label["cte_m"],
            "v1_steering_rad": network, "network_expert_difference_rad": network - expert,
            "window_role": window_role(header_ns, selected["divergence_sim_time_ns"], config["divergence_evaluation_duration_s"]),
            "source_mcap_sha256": source_sha,
        })
    if violations or not rows:
        raise RuntimeError(f"DAgger quality gate failed: samples={len(rows)}, future violations={violations}")
    if rollout_id == "dagger_rollout_B" and config["rollout_assignment"][rollout_id] != "holdout":
        raise RuntimeError("rollout B role changed")
    _write_manifest(manifest_path, rows)
    preview_indices = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    images = [Image.open(output_root / rows[int(i)]["image_path"]).convert("RGB") for i in preview_indices]
    sheet = Image.new("RGB", (600, math.ceil(len(images) / 3) * 66))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * 200, (index // 3) * 66)); image.close()
    preview_path = preview_dir / f"{rollout_id}.png"
    sheet.save(preview_path)
    ages = [row["expert_label_age_ms"] for row in rows]
    result = {
        "rollout_id": rollout_id, "role": config["rollout_assignment"][rollout_id], "result": "PASS",
        "source_bag": str(bag_path), "source_mcap_sha256": source_sha,
        "source_result": str(source_result_path), "source_result_sha256": sha256_file(source_result_path),
        "source_raw_size_bytes": directory_size(bag_path), "telemetry_sha256": sha256_file(telemetry_path),
        "raw_camera_frames": len(frame_records), "objective_window_raw_frames": len(eligible),
        "accepted_samples": len(rows), "full_rollout_fallback": fallback,
        "future_label_violations": violations, "stale_label_rejections": rejected_stale,
        "expert_label_age_ms": _distribution(ages), "window": selected,
        "window_counts": {name: sum(row["window_role"] == name for row in rows) for name in ("pre_divergence", "divergence", "late_failure")},
        "manifest": str(manifest_path), "preview": str(preview_path),
    }
    metadata_path = output_root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"version": VERSION, "episodes": []}
    metadata["episodes"].append(result)
    metadata["future_label_violations"] = sum(item["future_label_violations"] for item in metadata["episodes"])
    metadata["result"] = "PASS" if all(item["result"] == "PASS" for item in metadata["episodes"]) else "FAIL"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _rollout_b_collector(config: dict[str, Any]) -> CollectorConfig:
    base = diagnosis_collector_config(config)
    return CollectorConfig(**{**asdict(base), "data_relative_root": "physicar_e2e/pilotnet_dagger_v1/rollout_B/raw"})


def collect_rollout_b(
    *, config_path: Path, v1_checkpoint: Path, v1_onnx: Path, sim_root: Path,
    artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    marker = result_path.with_suffix(".started.json")
    if result_path.exists() or marker.exists():
        raise RuntimeError("refusing a second rollout_B collection")
    if sha256_file(v1_checkpoint) != config["v1_checkpoint_sha256"] or sha256_file(v1_onnx) != config["v1_onnx_sha256"]:
        raise RuntimeError("canonical V1 artifact hash mismatch")
    client = SimClient(config["base_url"], config["api_timeout_s"])
    backend = DockerRosBackend(_rollout_b_collector(config), sim_root)
    model = CameraOnlyOnnxModel(v1_onnx)
    handle: RecorderHandle | None = None
    report: dict[str, Any] = {"version": VERSION, "rollout_id": "dagger_rollout_B", "role": "holdout", "result": "FAIL", "collection_run_count": 0}
    telemetry: list[dict[str, Any]] = []
    try:
        if errors := client.safe_stop():
            raise RuntimeError("initial safe-stop failed: " + "; ".join(errors))
        initial = wait_after_reset(client, _driver_config(config), False)
        jpeg = client.camera_jpeg(config["camera_path"])
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.size != (480, 360) or image.format != "JPEG":
                raise RuntimeError("HTTP camera contract mismatch")
        topics = backend.preflight(config["diagnostic_ros_topics"])
        if shutil.disk_usage(backend.host_userdata_root).free < config["minimum_free_bytes"]:
            raise RuntimeError("insufficient external userdata free space")
        report["preflight"] = {"result": "PASS", "world": initial.world, "cones": initial.cone_count,
                               "route_points": initial.route_points, "topics": topics}
        handle = backend.start_recorder("dagger_rollout_B", config["diagnostic_ros_topics"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"status": "ROLLOUT_B_STARTED_DO_NOT_RETRY", "started_utc": utc_now()}, indent=2) + "\n")
        report["collection_run_count"] = 1
        metrics, telemetry = run_live_loop(client, model, config, initial, artifact_root / "rollout_B" / "http_frames")
        report["live"] = metrics
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if handle is not None:
            stopped = backend.stop_recorder(handle)
            report["recorder_stop"] = asdict(stopped)
            if not stopped.graceful or stopped.orphaned:
                raise RuntimeError(f"rollout_B recorder did not finalize gracefully: {stopped}")
            info = backend.bag_info(handle)
            verify_bag(info, config["diagnostic_ros_topics"], 2)
            report["bag"] = {"path": str(handle.host_bag_path), "size_bytes": directory_size(handle.host_bag_path),
                             "duration_s": info.duration_s, "topic_counts": info.topic_counts}
    if report["collection_run_count"] != 1 or not telemetry:
        raise RuntimeError("rollout_B did not produce exactly one telemetry sequence")
    telemetry_path = artifact_root / "rollout_B" / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n")
    windows = detect_divergence_windows([row["elapsed_s"] for row in telemetry], [row["cte_m"] for row in telemetry],
        stable_window_s=config["stable_window_s"], cte_floor_m=config["divergence_cte_floor_m"],
        persistence_samples=config["divergence_persistence_samples"])
    report["objective_divergence"] = windows
    report["telemetry"] = str(telemetry_path)
    collection_ok = (
        report["live"]["safe_stop_success"] and report["final_safe_stop_success"]
        and report["live"]["api_failures"] == 0 and report["live"]["liveness_failures"] == 0
        and report["recorder_stop"]["graceful"] and not report["recorder_stop"]["orphaned"]
    )
    report["result"] = "PASS" if collection_ok else "FAIL"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--extract", action="store_true")
    modes.add_argument("--collect-rollout-b", action="store_true")
    parser.add_argument("--rollout-id")
    parser.add_argument("--bag", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--v1-checkpoint", type=Path)
    parser.add_argument("--v1-onnx", type=Path)
    parser.add_argument("--sim-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.extract:
            if not all((args.rollout_id, args.bag, args.telemetry, args.source_result, args.output_root)):
                raise ValueError("extract mode requires rollout/source/output arguments")
            report = extract_rollout(rollout_id=args.rollout_id, bag_path=args.bag, telemetry_path=args.telemetry,
                                     output_root=args.output_root, config_path=args.config, source_result_path=args.source_result)
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        else:
            if not all((args.v1_checkpoint, args.v1_onnx, args.sim_root, args.artifact_root)):
                raise ValueError("collection mode requires model/simulator/artifact arguments")
            report = collect_rollout_b(config_path=args.config, v1_checkpoint=args.v1_checkpoint, v1_onnx=args.v1_onnx,
                                       sim_root=args.sim_root, artifact_root=args.artifact_root, result_path=args.result)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
