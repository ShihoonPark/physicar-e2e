"""Bounded V3 shadow-rollout collection and causal DAgger Iteration-2 extraction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .dataset_extractor import decode_rgb8_image, preprocess_image
from .expert_driver import wait_after_reset
from .pilotnet_dagger import (
    _distribution, _raw_frames, latest_causal_shadow, sim_time_ns, window_role,
)
from .pilotnet_failure_diagnosis import (
    _collector_config as diagnosis_collector_config, _driver_config,
    run_live_loop, utc_now,
)
from .pilotnet_inference import CameraOnlyOnnxModel, sha256_file
from .rosbag_collector import CollectorConfig, DockerRosBackend, RecorderHandle, directory_size, verify_bag
from .sim_client import SimClient


VERSION = "pilotnet_dagger_iteration2_v1"
ROLLOUTS = {"dagger_iter2_rollout_A": "training", "dagger_iter2_rollout_B": "holdout"}
FIELDS = [
    "episode_id", "sample_index", "image_path", "camera_record_time_ns", "camera_header_time_ns",
    "expert_label_time_ns", "expert_label_age_ms", "steering_rad", "steering_normalized",
    "route_progress_m", "cte_m", "heading_error_rad", "v3_steering_rad",
    "network_expert_difference_rad", "window_role", "source_mcap_sha256",
]


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("rollout_assignment") != ROLLOUTS:
        raise ValueError("Iteration-2 version or frozen A/B assignment changed")
    if payload.get("maximum_collection_rollouts") != 2 or payload.get("diagnostic_speed_mps") != 0.5:
        raise ValueError("Iteration-2 permits exactly two V3 0.50 m/s collection rollouts")
    if payload.get("minimum_reproduction_progress_m") != 10.0 or payload.get("minimum_selected_progress_m") != 10.0:
        raise ValueError("Iteration-2 10 m reproduction/selection gate changed")
    if payload.get("window_pre_divergence_s") != 2.0 or payload.get("minimum_selected_samples") != 20:
        raise ValueError("Iteration-2 objective window contract changed")
    return payload


def enforce_reproduction_gate(role: str, live: dict[str, Any]) -> dict[str, Any]:
    if role != "training":
        return {"result": "NOT_APPLICABLE", "reason": "holdout progress is reported but does not define the training reproduction gate"}
    if live["result"] == "PASS":
        return {"result": "FAIL", "reason": "V3 unexpectedly completed a full lap; repeatability requires investigation before Iteration-2 training"}
    progress = float(live["route_progress_m"])
    if progress < 10.0:
        return {"result": "FAIL", "reason": f"V3 later failure distribution not reproduced: progress {progress:.3f} m < 10 m"}
    return {"result": "PASS", "progress_m": progress, "later_failure_distribution_reproduced": True}


def select_iteration2_window(
    telemetry: Sequence[dict[str, Any]], divergence_index: int, pre_divergence_s: float = 2.0,
) -> dict[str, Any]:
    if not telemetry or not 0 <= divergence_index < len(telemetry):
        raise ValueError("invalid Iteration-2 telemetry/divergence index")
    divergence_ns = sim_time_ns(telemetry[divergence_index])
    return {
        "start_sim_time_ns": divergence_ns - int(pre_divergence_s * 1e9),
        "divergence_sim_time_ns": divergence_ns,
        "end_sim_time_ns": sim_time_ns(telemetry[-1]),
        "minimum_route_progress_m": 10.0,
        "rule": "continuous from 2.0 s before objective V3 divergence through final valid pre-safe-stop frame, then require route progress >=10 m",
    }


def filter_progress_rows(rows: Sequence[dict[str, Any]], minimum_progress_m: float) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["unwrapped_progress_m"]) >= minimum_progress_m]


def detect_final_failure_divergence(
    times_s: Sequence[float], ctes_m: Sequence[float], *, stable_window_s: float,
    cte_floor_m: float, persistence_samples: int,
) -> dict[str, Any]:
    """Find the start of the final persistent CTE excursion that reaches failure."""
    times = np.asarray(times_s, dtype=np.float64)
    ctes = np.abs(np.asarray(ctes_m, dtype=np.float64))
    if times.size != ctes.size or times.size < persistence_samples + 2:
        raise ValueError("insufficient telemetry for final-failure divergence")
    stable = np.flatnonzero(times <= times[0] + stable_window_s)
    threshold = max(float(cte_floor_m), float(np.mean(ctes[stable]) + 3.0 * np.std(ctes[stable])))
    if ctes[-1] < threshold:
        raise RuntimeError("final telemetry is below divergence threshold")
    onset = times.size - 1
    while onset > 0 and ctes[onset - 1] >= threshold:
        onset -= 1
    if times.size - onset < persistence_samples:
        raise RuntimeError("final CTE excursion is shorter than persistence gate")
    slope = float(np.polyfit(times[onset : onset + persistence_samples], ctes[onset : onset + persistence_samples], 1)[0])
    if slope <= 0:
        raise RuntimeError("final persistent CTE excursion does not begin with positive growth")
    critical = np.flatnonzero((times >= times[onset] - 2.0) & (times < times[onset]))
    return {
        "method": "start of final continuous above-threshold CTE excursion that persists to failure and begins with positive slope",
        "threshold_m": threshold, "divergence_index": int(onset), "divergence_time_s": float(times[onset]),
        "initial_persistence_slope_m_per_s": slope, "stable_indices": stable.tolist(),
        "critical_pre_onset_indices": critical.tolist(), "final_excursion_indices": list(range(int(onset), int(times.size))),
        "threshold_crossing_reproduced": True,
    }


def _collector(config: dict[str, Any], rollout_id: str) -> CollectorConfig:
    base = diagnosis_collector_config(config)
    return CollectorConfig(**{
        **asdict(base),
        "data_relative_root": f"physicar_e2e/pilotnet_dagger_iteration2_v1/{rollout_id}/raw",
    })


def collect_rollout(
    *, rollout_id: str, config_path: Path, v3_checkpoint: Path, v3_onnx: Path,
    sim_root: Path, artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in ROLLOUTS:
        raise ValueError("unknown Iteration-2 rollout")
    role = ROLLOUTS[rollout_id]
    marker = result_path.with_suffix(".started.json")
    if result_path.exists() or marker.exists():
        raise RuntimeError(f"refusing to recollect {rollout_id}")
    if sha256_file(v3_checkpoint) != config["v3_checkpoint_sha256"] or sha256_file(v3_onnx) != config["v3_onnx_sha256"]:
        raise RuntimeError("canonical V3 artifact hash mismatch")
    client = SimClient(config["base_url"], config["api_timeout_s"])
    backend = DockerRosBackend(_collector(config, rollout_id), sim_root)
    model = CameraOnlyOnnxModel(v3_onnx)
    handle: RecorderHandle | None = None
    telemetry: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "version": VERSION, "rollout_id": rollout_id, "role": role,
        "result": "FAIL", "collection_run_count": 0,
        "v3_checkpoint_sha256": sha256_file(v3_checkpoint), "v3_onnx_sha256": sha256_file(v3_onnx),
    }
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
        handle = backend.start_recorder(rollout_id, config["diagnostic_ros_topics"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"status": f"{rollout_id}_STARTED_DO_NOT_RETRY", "started_utc": utc_now()}, indent=2) + "\n")
        report["collection_run_count"] = 1
        metrics, telemetry = run_live_loop(
            client, model, config, initial, artifact_root / rollout_id / "http_frames", policy_name="PilotNet V3",
        )
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
                result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
                raise RuntimeError(f"recorder did not finalize gracefully: {stopped}")
            info = backend.bag_info(handle)
            verify_bag(info, config["diagnostic_ros_topics"], 2)
            report["bag"] = {"path": str(handle.host_bag_path), "size_bytes": directory_size(handle.host_bag_path),
                             "duration_s": info.duration_s, "topic_counts": info.topic_counts}
    if report["collection_run_count"] != 1 or handle is None or not telemetry:
        raise RuntimeError("rollout did not produce exactly one telemetry sequence")
    telemetry_path = artifact_root / rollout_id / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n")
    windows = detect_final_failure_divergence(
        [row["elapsed_s"] for row in telemetry], [row["cte_m"] for row in telemetry],
        stable_window_s=config["stable_window_s"], cte_floor_m=config["divergence_cte_floor_m"],
        persistence_samples=config["divergence_persistence_samples"],
    )
    divergence_reproduced = windows["threshold_crossing_reproduced"]
    report["objective_divergence"] = windows
    report["telemetry"] = str(telemetry_path)
    report["reproduction_gate"] = enforce_reproduction_gate(role, report["live"])
    runtime_ok = (
        report["live"]["safe_stop_success"] and report["final_safe_stop_success"]
        and report["live"]["api_failures"] == 0 and report["live"]["liveness_failures"] == 0
        and report["recorder_stop"]["graceful"] and not report["recorder_stop"]["orphaned"]
    )
    gate_ok = report["reproduction_gate"]["result"] in {"PASS", "NOT_APPLICABLE"}
    report["result"] = "PASS" if runtime_ok and gate_ok and divergence_reproduced else "FAIL"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def reanalyze_collection(result_path: Path) -> dict[str, Any]:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    telemetry = json.loads(Path(report["telemetry"]).read_text(encoding="utf-8"))
    config_path = Path(__file__).resolve().parents[2] / "configs" / "pilotnet_dagger_iteration2_v1.json"
    config = load_config(config_path)
    report["objective_divergence"] = detect_final_failure_divergence(
        [row["elapsed_s"] for row in telemetry], [row["cte_m"] for row in telemetry],
        stable_window_s=config["stable_window_s"], cte_floor_m=config["divergence_cte_floor_m"],
        persistence_samples=config["divergence_persistence_samples"],
    )
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def extract_rollout(
    *, rollout_id: str, bag_path: Path, telemetry_path: Path, source_result_path: Path,
    output_root: Path, config_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in ROLLOUTS:
        raise ValueError("unknown Iteration-2 rollout")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("result") != "PASS" or source_result.get("collection_run_count") != 1:
        raise RuntimeError("collection/reproduction gate is not PASS")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    divergence = source_result["objective_divergence"]
    if divergence.get("threshold_crossing_reproduced") is not True:
        raise RuntimeError("objective V3 divergence was not reproduced")
    selected = select_iteration2_window(telemetry, int(divergence["divergence_index"]), config["window_pre_divergence_s"])
    frames = list(_raw_frames(bag_path))
    in_time = [item for item in frames if selected["start_sim_time_ns"] <= item[2] <= selected["end_sim_time_ns"]]
    manifest_dir = output_root / "manifests"
    image_dir = output_root / "images" / rollout_id
    preview_dir = output_root / "previews"
    manifest_path = manifest_dir / f"{rollout_id}.csv"
    if manifest_path.exists() or image_dir.exists():
        raise RuntimeError(f"refusing to overwrite extracted {rollout_id}")
    manifest_dir.mkdir(parents=True, exist_ok=True); image_dir.mkdir(parents=True); preview_dir.mkdir(parents=True, exist_ok=True)
    image_config = {key: config[key] for key in ("source_width", "source_height", "source_encoding", "roi", "output_width", "output_height")}
    source_mcap = frames[0][0]
    source_sha = sha256_file(source_mcap)
    rows: list[dict[str, Any]] = []
    future = stale = below_progress = 0
    for _, record_ns, header_ns, decoded in in_time:
        label = latest_causal_shadow(telemetry, header_ns)
        if label is None:
            continue
        if float(label["unwrapped_progress_m"]) < config["minimum_selected_progress_m"]:
            below_progress += 1; continue
        label_ns = sim_time_ns(label)
        age_ns = header_ns - label_ns
        if age_ns < 0:
            future += 1; continue
        if age_ns / 1e9 > config["maximum_expert_label_age_s"]:
            stale += 1; continue
        expert = float(label["shadow_expert_steering_rad"])
        network = float(label["network_steering_rad"])
        if not all(math.isfinite(value) for value in (expert, network)) or abs(expert) > config["max_steering_rad"] + 1e-9:
            raise RuntimeError("invalid Iteration-2 expert label")
        image = preprocess_image(decode_rgb8_image(decoded, image_config), image_config)
        index = len(rows); relative = Path("images") / rollout_id / f"frame_{index:06d}.png"
        image.save(output_root / relative, format="PNG", optimize=False); image.close()
        rows.append({
            "episode_id": rollout_id, "sample_index": index, "image_path": relative.as_posix(),
            "camera_record_time_ns": record_ns, "camera_header_time_ns": header_ns,
            "expert_label_time_ns": label_ns, "expert_label_age_ms": age_ns / 1e6,
            "steering_rad": expert, "steering_normalized": expert / config["max_steering_rad"],
            "route_progress_m": label["unwrapped_progress_m"], "cte_m": label["cte_m"],
            "heading_error_rad": label.get("heading_error_rad", ""), "v3_steering_rad": network,
            "network_expert_difference_rad": network - expert,
            "window_role": window_role(header_ns, selected["divergence_sim_time_ns"], config["divergence_evaluation_duration_s"]),
            "source_mcap_sha256": source_sha,
        })
    if future or stale or len(rows) < config["minimum_selected_samples"]:
        raise RuntimeError(f"Iteration-2 extraction gate failed: samples={len(rows)}, future={future}, stale={stale}")
    if min(float(row["route_progress_m"]) for row in rows) < 10.0:
        raise RuntimeError("selected sample below 10 m progress gate")
    _write_manifest(manifest_path, rows)
    indices = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    images = [Image.open(output_root / rows[int(index)]["image_path"]).convert("RGB") for index in indices]
    sheet = Image.new("RGB", (600, math.ceil(len(images) / 3) * 66))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * 200, (index // 3) * 66)); image.close()
    preview = preview_dir / f"{rollout_id}.png"; sheet.save(preview)
    ages = [float(row["expert_label_age_ms"]) for row in rows]
    result = {
        "version": VERSION, "rollout_id": rollout_id, "role": ROLLOUTS[rollout_id], "result": "PASS",
        "source_bag": str(bag_path), "source_raw_size_bytes": directory_size(bag_path),
        "source_mcap_sha256": source_sha, "source_result_sha256": sha256_file(source_result_path),
        "telemetry_sha256": sha256_file(telemetry_path), "raw_camera_frames": len(frames),
        "objective_window_raw_frames_before_progress_gate": len(in_time), "below_10m_rejections": below_progress,
        "accepted_samples": len(rows), "future_label_violations": future, "stale_label_rejections": stale,
        "expert_label_age_ms": _distribution(ages), "window": selected,
        "selected_progress_range_m": [min(float(row["route_progress_m"]) for row in rows), max(float(row["route_progress_m"]) for row in rows)],
        "window_counts": {name: sum(row["window_role"] == name for row in rows) for name in ("pre_divergence", "divergence", "late_failure")},
        "manifest": str(manifest_path), "preview": str(preview),
    }
    metadata_path = output_root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"version": VERSION, "episodes": []}
    metadata["episodes"].append(result)
    metadata["future_label_violations"] = sum(item["future_label_violations"] for item in metadata["episodes"])
    metadata["result"] = "PASS" if all(item["result"] == "PASS" for item in metadata["episodes"]) else "FAIL"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--collect", action="store_true"); modes.add_argument("--extract", action="store_true"); modes.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--rollout-id", required=True)
    parser.add_argument("--v3-checkpoint", type=Path); parser.add_argument("--v3-onnx", type=Path)
    parser.add_argument("--sim-root", type=Path); parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--bag", type=Path); parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--source-result", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.reanalyze:
            report = reanalyze_collection(args.result)
        elif args.collect:
            report = collect_rollout(rollout_id=args.rollout_id, config_path=args.config,
                v3_checkpoint=args.v3_checkpoint, v3_onnx=args.v3_onnx, sim_root=args.sim_root,
                artifact_root=args.artifact_root, result_path=args.result)
        else:
            report = extract_rollout(rollout_id=args.rollout_id, bag_path=args.bag,
                telemetry_path=args.telemetry, source_result_path=args.source_result,
                output_root=args.output_root, config_path=args.config)
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
