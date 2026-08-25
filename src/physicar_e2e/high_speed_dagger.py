"""Bounded High-Speed PilotNet DAgger V1 collection, training, and live gates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch

from .dataset_extractor import decode_rgb8_image, preprocess_image
from .expert_driver import wait_after_reset
from .high_speed_v5 import (
    EPISODES, HOLDOUT_EPISODES, SPEED_MPS, TRAIN_EPISODES, VALIDATION_EPISODES,
    classify_policy_run, live_preflight, validate_v5_dataset, verify_frozen_expert, write_json,
)
from .pilotnet import PILOTNET_PARAMETER_COUNT
from .pilotnet_dagger import _distribution, _raw_frames, latest_causal_shadow, sim_time_ns, window_role
from .pilotnet_dagger_iteration2 import detect_final_failure_divergence
from .pilotnet_dagger_training import read_dagger_rows
from .pilotnet_failure_diagnosis import (
    _collector_config as diagnosis_collector_config,
    _driver_config,
    detect_divergence_windows,
    run_live_loop,
)
from .pilotnet_inference import CameraOnlyOnnxModel, InferenceConfig, run_smoke
from .pilotnet_recovery_training import load_checkpoint
from .pilotnet_training import (
    GateFailure, error_metrics, export_onnx, load_config as load_training_config, predict_rows,
    sha256_file, tiny_overfit_sanity, train_baseline, validate_onnx_equivalence,
)
from .pilotnet_v4_repeatability import clock_health_preflight, verify_static_environment
from .rosbag_collector import CollectorConfig, DockerRosBackend, RecorderHandle, directory_size, verify_bag
from .sim_client import SimClient


VERSION = "high_speed_dagger_v1"
ROLLOUTS = {
    "high_speed_dagger_rollout_A": "training",
    "high_speed_dagger_rollout_B": "holdout",
}
V5_CHECKPOINT_SHA256 = "04cc593426d2e79a703e4218c7041d2cf1317c2039254643d7e9fb612fd3a101"
V5_ONNX_SHA256 = "404b2ea24d25d0178c60ba9167496f93ba50b10ade78cfcf6edfc0f64658a1fd"
V4_ONNX_SHA256 = "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
MAX_LIVE_ATTEMPTS = 5
TARGET_POLICY_PASSES = 3
FIELDS = [
    "episode_id", "rollout_id", "sample_index", "image_path", "raw_image_source",
    "camera_record_time_ns", "camera_header_time_ns", "expert_label_time_ns", "expert_label_age_ms",
    "steering_rad", "steering_normalized", "route_progress_m", "route_progress_fraction", "cte_m",
    "x_m", "y_m", "yaw_rad", "heading_error_rad", "v5_steering_rad", "shadow_expert_steering_rad",
    "v5_minus_expert_rad", "window_role", "source_mcap_sha256", "telemetry_sha256",
    "source_result_sha256", "config_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("rollout_assignment") != ROLLOUTS:
        raise ValueError("High-Speed DAgger version or frozen A/B assignment changed")
    fixed = (
        payload.get("maximum_collection_rollouts"), payload.get("diagnostic_speed_mps"),
        payload.get("lookahead_m"), payload.get("control_frequency_hz"), payload.get("max_steering_rad"),
        payload.get("window_pre_divergence_s"), payload.get("minimum_selected_samples"),
        payload.get("minimum_reproduction_progress_fraction"), payload.get("minimum_selected_progress_fraction"),
    )
    if fixed != (2, 1.8, 0.9, 15.0, 0.349066, 2.0, 20, 0.3, 0.3):
        raise ValueError(f"High-Speed DAgger frozen contract changed: {fixed}")
    if payload.get("diagnostic_ros_topics") != ["/camera/image_raw", "/clock"]:
        raise ValueError("High-Speed DAgger raw bag must contain camera and simulator clock")
    if payload.get("camera_only_model_observation") is not True:
        raise ValueError("V5/V6 neural observation must remain camera-only")
    return payload


def verify_preserved_baselines(repo: Path, sim_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    expert = verify_frozen_expert(repo, repo / "configs/high_speed_expert_v1.json")
    collection = json.loads((repo / "results/high_speed_collection_v1/summary.json").read_text())
    dataset = json.loads((repo / "results/high_speed_dataset_v1/summary.json").read_text())
    training = json.loads((repo / "results/pilotnet_training_v5_high_speed/summary.json").read_text())
    live = json.loads((repo / "results/pilotnet_e2e_v5_high_speed/summary.json").read_text())
    v5_checkpoint = Path(training["artifacts"]["checkpoint"]["path"])
    v5_onnx = Path(training["artifacts"]["onnx"]["path"])
    v4_onnx = sim_root / "userdata/physicar_e2e/pilotnet_dagger_iteration2_v1/v4/onnx/pilotnet_v4_dagger.onnx"
    checks = {
        "expert_3_of_3_pass": expert.fixed_speed_mps == 1.8 and expert.lookahead_m == 0.9,
        "nominal_collection_12_of_12": collection.get("result") == "PASS" and collection.get("passed_episode_count") == 12,
        "nominal_dataset_2911": dataset.get("result") == "PASS" and dataset.get("counts", {}).get("accepted_camera_samples") == 2911,
        "v5_training_pass": training.get("result") == "PASS" and training.get("training_from_scratch") is True,
        "v5_policy_fail_preserved": live.get("result") == "FAIL" and live["attempts"][0]["classification"] == "POLICY_FAIL",
        "v5_checkpoint_hash": sha256_file(v5_checkpoint) == config["v5_checkpoint_sha256"] == V5_CHECKPOINT_SHA256,
        "v5_onnx_hash": sha256_file(v5_onnx) == config["v5_onnx_sha256"] == V5_ONNX_SHA256,
        "v4_onnx_hash": sha256_file(v4_onnx) == V4_ONNX_SHA256,
    }
    if not all(checks.values()):
        raise GateFailure(f"preserved baseline gate failed: {checks}")
    return {"result": "PASS", "checks": checks, "v5_checkpoint": str(v5_checkpoint), "v5_onnx": str(v5_onnx)}


def reproduction_gate(role: str, live: dict[str, Any]) -> dict[str, Any]:
    if role != "training":
        return {"result": "NOT_APPLICABLE", "reason": "rollout B is independent holdout evidence"}
    if live.get("result") == "PASS":
        return {"result": "FAIL", "reason": "V5 unexpectedly completed a full lap; repeatability requires investigation"}
    fraction = float(live["route_completion_fraction"])
    if fraction < 0.30:
        return {"result": "FAIL", "reason": f"V5 failure distribution not reproduced: {fraction:.2%} < 30%"}
    return {"result": "PASS", "route_completion_fraction": fraction, "late_high_speed_failure_reproduced": True}


def control_authority_contract() -> dict[str, Any]:
    return {"vehicle_controller": "PilotNet V5", "shadow_expert_control_authority": False,
            "neural_observation": ["camera"]}


def passes_progress_gate(progress_m: float, route_length_m: float) -> bool:
    return float(progress_m) / float(route_length_m) >= 0.30


def select_objective_window(telemetry: Sequence[dict[str, Any]], divergence_index: int, route_length_m: float) -> dict[str, Any]:
    if not telemetry or not 0 <= divergence_index < len(telemetry):
        raise ValueError("invalid High-Speed DAgger telemetry/divergence index")
    divergence_ns = sim_time_ns(telemetry[divergence_index])
    return {
        "start_sim_time_ns": divergence_ns - 2_000_000_000,
        "divergence_sim_time_ns": divergence_ns,
        "end_sim_time_ns": sim_time_ns(telemetry[-1]),
        "minimum_route_progress_m": route_length_m * 0.30,
        "minimum_route_progress_fraction": 0.30,
        "rule": "continuous from 2.0 s before objective V5 divergence through final valid pre-safe-stop frame, then require route progress >=30%",
    }


def _objective_divergence(telemetry: Sequence[dict[str, Any]], live_result: str, config: dict[str, Any]) -> dict[str, Any]:
    arguments = {
        "stable_window_s": config["stable_window_s"],
        "cte_floor_m": config["divergence_cte_floor_m"],
        "persistence_samples": config["divergence_persistence_samples"],
    }
    times = [row["elapsed_s"] for row in telemetry]
    ctes = [row["cte_m"] for row in telemetry]
    if live_result == "FAIL":
        return detect_final_failure_divergence(times, ctes, **arguments)
    result = detect_divergence_windows(times, ctes, **arguments)
    result["threshold_crossing_reproduced"] = True
    result["method"] += "; full-lap holdout uses maximum/persistent CTE evaluation window"
    return result


def _collector(config: dict[str, Any], rollout_id: str) -> CollectorConfig:
    base = diagnosis_collector_config(config)
    return CollectorConfig(**{
        **asdict(base),
        "data_relative_root": f"physicar_e2e/high_speed_dagger_v1/{rollout_id}/raw",
    })


def collect_rollout(
    *, repo: Path, sim_root: Path, rollout_id: str, config_path: Path, artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in ROLLOUTS:
        raise ValueError("unknown High-Speed DAgger rollout")
    role = ROLLOUTS[rollout_id]
    marker = result_path.with_suffix(".started.json")
    if result_path.exists() or marker.exists():
        raise RuntimeError(f"refusing to recollect {rollout_id}")
    preserved = verify_preserved_baselines(repo, sim_root, config)
    v5_checkpoint = Path(preserved["v5_checkpoint"])
    v5_onnx = Path(preserved["v5_onnx"])
    client = SimClient(config["base_url"], config["api_timeout_s"])
    backend = DockerRosBackend(_collector(config, rollout_id), sim_root)
    model = CameraOnlyOnnxModel(v5_onnx)
    handle: RecorderHandle | None = None
    telemetry: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "version": VERSION, "rollout_id": rollout_id, "role": role, "result": "FAIL",
        "collection_run_count": 0, "preserved_baselines": preserved,
        "control_authority": control_authority_contract(),
        "v5_checkpoint_sha256": sha256_file(v5_checkpoint), "v5_onnx_sha256": sha256_file(v5_onnx),
    }
    try:
        if errors := client.safe_stop():
            raise RuntimeError("initial safe-stop failed: " + "; ".join(errors))
        initial = wait_after_reset(client, _driver_config(config), False)
        environment = verify_static_environment(initial)
        clock = clock_health_preflight(client)
        if clock.get("result") != "PASS":
            raise RuntimeError("simulator clock preflight failed")
        jpeg = client.camera_jpeg(config["camera_path"])
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.size != (480, 360) or image.format != "JPEG":
                raise RuntimeError("HTTP camera contract mismatch")
        topics = backend.preflight(config["diagnostic_ros_topics"])
        if shutil.disk_usage(backend.host_userdata_root).free < config["minimum_free_bytes"]:
            raise RuntimeError("insufficient external userdata free space")
        report["preflight"] = {"result": "PASS", "environment": environment, "clock_health": clock,
                               "camera": {"transport": "HTTP JPEG", "dimensions": [480, 360]}, "topics": topics}
        handle = backend.start_recorder(rollout_id, config["diagnostic_ros_topics"])
        marker.parent.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"status": f"{rollout_id}_STARTED_DO_NOT_RETRY_POLICY_FAILURE", "started_utc": utc_now()})
        report["collection_run_count"] = 1
        metrics, telemetry = run_live_loop(
            client, model, config, initial, artifact_root / rollout_id / "http_frames", policy_name="PilotNet V5",
        )
        report["live"] = metrics
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if handle is not None:
            try:
                stopped = backend.stop_recorder(handle)
                report["recorder_stop"] = asdict(stopped)
                info = backend.bag_info(handle)
                verify_bag(info, config["diagnostic_ros_topics"], 2)
                report["bag"] = {"path": str(handle.host_bag_path), "size_bytes": directory_size(handle.host_bag_path),
                                 "duration_s": info.duration_s, "topic_counts": info.topic_counts}
            except Exception as exc:
                report["recorder_failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(result_path, report)
    if report["collection_run_count"] != 1 or handle is None or not telemetry or "live" not in report:
        raise GateFailure(f"{rollout_id} did not produce one valid telemetry sequence")
    telemetry_path = artifact_root / rollout_id / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    runtime_ok = (
        report["live"]["api_failures"] == 0 and report["live"]["liveness_failures"] == 0
        and report["live"]["safe_stop_success"] and report["final_safe_stop_success"]
        and report.get("recorder_stop", {}).get("graceful") is True
        and report.get("recorder_stop", {}).get("orphaned") is False
        and "recorder_failure" not in report
    )
    report["classification"] = (
        "INFRA_FAIL" if not runtime_ok else ("POLICY_PASS" if report["live"]["result"] == "PASS" else "POLICY_FAIL")
    )
    report["reproduction_gate"] = reproduction_gate(role, report["live"])
    if runtime_ok and report["reproduction_gate"]["result"] in {"PASS", "NOT_APPLICABLE"}:
        report["objective_divergence"] = _objective_divergence(telemetry, report["live"]["result"], config)
        report["telemetry"] = str(telemetry_path)
        report["telemetry_sha256"] = sha256_file(telemetry_path)
        report["result"] = "PASS"
    write_json(result_path, report)
    return report


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def extract_rollout(
    *, rollout_id: str, config_path: Path, bag_path: Path, telemetry_path: Path,
    source_result_path: Path, output_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in ROLLOUTS:
        raise ValueError("unknown High-Speed DAgger rollout")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if source_result.get("result") != "PASS" or source_result.get("collection_run_count") != 1:
        raise GateFailure("collection/reproduction gate is not PASS")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    divergence = source_result["objective_divergence"]
    route_length = float(source_result["live"]["route_length_m"])
    selected = select_objective_window(telemetry, int(divergence["divergence_index"]), route_length)
    frames = list(_raw_frames(bag_path))
    if not frames:
        raise GateFailure("raw rollout bag has no camera frames")
    in_time = [item for item in frames if selected["start_sim_time_ns"] <= item[2] <= selected["end_sim_time_ns"]]
    source_mcap = frames[0][0]
    source_sha = sha256_file(source_mcap)
    telemetry_sha = sha256_file(telemetry_path)
    result_sha = sha256_file(source_result_path)
    config_sha = sha256_file(config_path)
    accepted: list[tuple[Any, dict[str, Any], int]] = []
    future = stale = below_progress = 0
    for item in in_time:
        header_ns = item[2]
        label = latest_causal_shadow(telemetry, header_ns)
        if label is None:
            continue
        if not passes_progress_gate(float(label["unwrapped_progress_m"]), route_length):
            below_progress += 1
            continue
        age_ns = header_ns - sim_time_ns(label)
        if age_ns < 0:
            future += 1
            continue
        if age_ns / 1e9 > config["maximum_expert_label_age_s"]:
            stale += 1
            continue
        accepted.append((item, label, age_ns))
    if future != 0 or len(accepted) < config["minimum_selected_samples"]:
        raise GateFailure(f"DAgger extraction gate failed: samples={len(accepted)}, future={future}, stale={stale}")
    manifest_dir = output_root / "manifests"
    image_dir = output_root / "images" / rollout_id
    preview_dir = output_root / "previews"
    manifest_path = manifest_dir / f"{rollout_id}.csv"
    if manifest_path.exists() or image_dir.exists():
        raise RuntimeError(f"refusing to overwrite extracted {rollout_id}")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    image_config = {key: config[key] for key in ("source_width", "source_height", "source_encoding", "roi", "output_width", "output_height")}
    rows: list[dict[str, Any]] = []
    for (mcap, record_ns, header_ns, decoded), label, age_ns in accepted:
        expert = float(label["shadow_expert_steering_rad"])
        v5 = float(label["network_steering_rad"])
        if not all(math.isfinite(value) for value in (expert, v5)) or abs(expert) > config["max_steering_rad"] + 1e-9:
            raise GateFailure("invalid shadow Expert label")
        image = preprocess_image(decode_rgb8_image(decoded, image_config), image_config)
        index = len(rows)
        relative = Path("images") / rollout_id / f"frame_{index:06d}.png"
        image.save(output_root / relative, format="PNG", optimize=False)
        image.close()
        progress = float(label["unwrapped_progress_m"])
        rows.append({
            "episode_id": rollout_id, "rollout_id": rollout_id, "sample_index": index,
            "image_path": relative.as_posix(), "raw_image_source": f"{mcap}:/camera/image_raw",
            "camera_record_time_ns": record_ns, "camera_header_time_ns": header_ns,
            "expert_label_time_ns": sim_time_ns(label), "expert_label_age_ms": age_ns / 1e6,
            "steering_rad": expert, "steering_normalized": expert / config["max_steering_rad"],
            "route_progress_m": progress, "route_progress_fraction": progress / route_length,
            "cte_m": label["cte_m"], "x_m": label["x"], "y_m": label["y"], "yaw_rad": label["yaw"],
            "heading_error_rad": label.get("heading_error_rad", ""), "v5_steering_rad": v5,
            "shadow_expert_steering_rad": expert, "v5_minus_expert_rad": v5 - expert,
            "window_role": window_role(header_ns, selected["divergence_sim_time_ns"], config["divergence_evaluation_duration_s"]),
            "source_mcap_sha256": source_sha, "telemetry_sha256": telemetry_sha,
            "source_result_sha256": result_sha, "config_sha256": config_sha,
        })
    _write_manifest(manifest_path, rows)
    indices = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    images = [Image.open(output_root / rows[int(index)]["image_path"]).convert("RGB") for index in indices]
    sheet = Image.new("RGB", (600, math.ceil(len(images) / 3) * 66))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % 3) * 200, (index // 3) * 66))
        image.close()
    preview = preview_dir / f"{rollout_id}.png"
    sheet.save(preview)
    result = {
        "version": VERSION, "rollout_id": rollout_id, "role": ROLLOUTS[rollout_id], "result": "PASS",
        "source_bag": str(bag_path), "source_raw_size_bytes": directory_size(bag_path),
        "source_mcap_sha256": source_sha, "source_result_sha256": result_sha,
        "telemetry_sha256": telemetry_sha, "config_sha256": config_sha,
        "raw_camera_frames": len(frames), "objective_window_raw_frames_before_progress_gate": len(in_time),
        "below_30_percent_rejections": below_progress, "accepted_samples": len(rows),
        "future_label_violations": future, "stale_label_rejections": stale,
        "expert_label_age_ms": _distribution([float(row["expert_label_age_ms"]) for row in rows]),
        "window": selected,
        "selected_progress_range_fraction": [min(float(row["route_progress_fraction"]) for row in rows),
                                             max(float(row["route_progress_fraction"]) for row in rows)],
        "window_counts": {name: sum(row["window_role"] == name for row in rows)
                          for name in ("pre_divergence", "divergence", "late_failure")},
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "preview": str(preview),
    }
    metadata_path = output_root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"version": VERSION, "episodes": []}
    if any(item["rollout_id"] == rollout_id for item in metadata["episodes"]):
        raise RuntimeError(f"metadata already contains {rollout_id}")
    metadata["episodes"].append(result)
    metadata["future_label_violations"] = sum(item["future_label_violations"] for item in metadata["episodes"])
    metadata["result"] = "PASS" if all(item["result"] == "PASS" for item in metadata["episodes"]) else "FAIL"
    write_json(metadata_path, metadata)
    return result


def update_compact_dagger_summary(repo: Path, output_root: Path) -> None:
    metadata = json.loads((output_root / "dataset_metadata.json").read_text())
    result_dir = repo / "results/high_speed_dagger_v1"
    collections = []
    for rollout_id in ROLLOUTS:
        path = result_dir / f"{rollout_id}_collection.json"
        if path.exists():
            item = json.loads(path.read_text())
            collections.append({"rollout_id": rollout_id, "classification": item.get("classification"),
                                "live": item.get("live"), "bag": item.get("bag"),
                                "objective_divergence": item.get("objective_divergence")})
    summary = {
        "version": VERSION, "generated_utc": utc_now(), "result": metadata["result"],
        "rollout_assignment": ROLLOUTS, "collections": collections, "extractions": metadata["episodes"],
        "future_label_violations": metadata["future_label_violations"],
        "dataset_metadata_path": str(output_root / "dataset_metadata.json"),
        "dataset_metadata_sha256": sha256_file(output_root / "dataset_metadata.json"),
    }
    write_json(result_dir / "summary.json", summary)


def validate_composition(nominal_train, nominal_validation, nominal_holdout, dagger_train, dagger_holdout) -> dict[str, Any]:
    if {row["episode_id"] for row in nominal_train} != set(TRAIN_EPISODES):
        raise GateFailure("nominal V6 training split changed")
    if {row["episode_id"] for row in nominal_validation} != set(VALIDATION_EPISODES):
        raise GateFailure("nominal validation split changed")
    if {row["episode_id"] for row in nominal_holdout} != set(HOLDOUT_EPISODES):
        raise GateFailure("nominal holdout split changed")
    if any(row["episode_id"] != "high_speed_dagger_rollout_A" for row in dagger_train):
        raise GateFailure("non-A data entered DAgger training")
    if any(row["episode_id"] != "high_speed_dagger_rollout_B" for row in dagger_holdout):
        raise GateFailure("non-B data entered on-policy holdout")
    training_paths = {row["image_path"] for row in [*nominal_train, *dagger_train]}
    evaluation_paths = {row["image_path"] for row in [*nominal_validation, *nominal_holdout, *dagger_holdout]}
    if training_paths & evaluation_paths:
        raise GateFailure("V6 train/evaluation image leakage")
    train_sources = {row["source_mcap_sha256"] for row in dagger_train}
    holdout_sources = {row["source_mcap_sha256"] for row in dagger_holdout}
    if train_sources & holdout_sources:
        raise GateFailure("rollout A/B source leakage")
    forbidden = {"dagger_rollout_A", "dagger_rollout_B", "dagger_iter2_rollout_A", "dagger_iter2_rollout_B"}
    if {row["episode_id"] for row in [*nominal_train, *dagger_train]} & forbidden:
        raise GateFailure("low-speed DAgger data entered V6 training")
    if any("recovery" in row["episode_id"] for row in [*nominal_train, *dagger_train]):
        raise GateFailure("recovery data entered V6 training")
    return {"result": "PASS", "holdout_leakage": False, "v4_data_included": False,
            "low_speed_dagger_included": False, "recovery_data_included": False}


def _metric_arrays(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result = error_metrics(predictions, labels)
    result["correlation"] = float(np.corrcoef(predictions, labels)[0, 1]) if len(labels) > 1 else None
    magnitude = float(np.mean(np.abs(labels)))
    result["corrective_magnitude_ratio"] = float(np.mean(np.abs(predictions)) / magnitude) if magnitude else None
    return result


def model_metrics(model, rows, config, device, *, grouped: bool = False) -> dict[str, Any]:
    predictions, labels = predict_rows(model, rows, config, device)
    result: dict[str, Any] = {"overall": _metric_arrays(predictions, labels)}
    if grouped:
        result["windows"] = {}
        for role in ("pre_divergence", "divergence", "late_failure"):
            mask = np.asarray([row["window_role"] == role for row in rows], dtype=bool)
            result["windows"][role] = _metric_arrays(predictions[mask], labels[mask]) if np.any(mask) else {"sample_count": 0}
    return result


def training_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_path = repo / "results/pilotnet_training_v6_high_speed_dagger/summary.json"
    if result_path.exists():
        raise RuntimeError("refusing to repeat completed V6 training")
    config_path = repo / "configs/pilotnet_training_v6_high_speed_dagger.json"
    config = load_training_config(config_path)
    if config.get("version") != "pilotnet_training_v6_high_speed_dagger" or config.get("initialization") != "from_scratch":
        raise GateFailure("V6 must use the frozen from-scratch training config")
    nominal_root = sim_root / "userdata/physicar_e2e/high_speed_v1/dataset"
    dagger_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_v1/extracted"
    artifact_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_v1/v6"
    if artifact_root.exists():
        raise RuntimeError("refusing to overwrite V6 artifacts")
    dagger_metadata = json.loads((dagger_root / "dataset_metadata.json").read_text())
    roles = {item["rollout_id"]: item["role"] for item in dagger_metadata.get("episodes", [])}
    if roles != ROLLOUTS or dagger_metadata.get("future_label_violations") != 0 or dagger_metadata.get("result") != "PASS":
        raise GateFailure("High-Speed DAgger dataset metadata gate failed")
    nominal_train, nominal_validation, nominal_holdout, nominal_integrity = validate_v5_dataset(nominal_root, config)
    dagger_train = read_dagger_rows(dagger_root, config["dagger_training_rollout"])
    dagger_holdout = read_dagger_rows(dagger_root, config["dagger_holdout_rollout"])
    composition = validate_composition(nominal_train, nominal_validation, nominal_holdout, dagger_train, dagger_holdout)
    combined = [*nominal_train, *dagger_train]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v5_summary = json.loads((repo / "results/pilotnet_training_v5_high_speed/summary.json").read_text())
    v5_checkpoint = Path(v5_summary["artifacts"]["checkpoint"]["path"])
    report: dict[str, Any] = {
        "version": "pilotnet_training_v6_high_speed_dagger", "generated_utc": utc_now(), "result": "FAIL",
        "gate_reached": "dataset", "device": str(device),
        "architecture": {"input_shape": [3, 66, 200], "parameter_count": PILOTNET_PARAMETER_COUNT,
                         "identical_to_v5": True},
        "training_from_scratch": True, "v5_checkpoint_used_for_training": False,
        "dataset": {"nominal_integrity": nominal_integrity,
                    "dagger_metadata_sha256": sha256_file(dagger_root / "dataset_metadata.json"),
                    "nominal_training_samples": len(nominal_train), "dagger_A_training_samples": len(dagger_train),
                    "combined_training_samples": len(combined), "nominal_validation_samples": len(nominal_validation),
                    "nominal_holdout_samples": len(nominal_holdout), "dagger_B_holdout_samples": len(dagger_holdout),
                    "training_rollout": config["dagger_training_rollout"], "holdout_rollout": config["dagger_holdout_rollout"],
                    **composition},
    }
    try:
        report["tiny_overfit"] = tiny_overfit_sanity(combined, config, device)
        checkpoint = artifact_root / "checkpoints/pilotnet_v6_high_speed_dagger_best.pt"
        model, training, history = train_baseline(combined, nominal_validation, config, device, checkpoint)
        report["training"] = {**training, "initialized_from_scratch": True}
        report["epochs"] = history
        report["gate_reached"] = "training"
        v5 = load_checkpoint(v5_checkpoint, device)
        report["offline_comparison"] = {
            "nominal_validation": {"v5": model_metrics(v5, nominal_validation, config, device),
                                   "v6": model_metrics(model, nominal_validation, config, device)},
            "nominal_holdout": {"v5": model_metrics(v5, nominal_holdout, config, device),
                                "v6": model_metrics(model, nominal_holdout, config, device)},
            "v5_on_policy_rollout_B": {"v5": model_metrics(v5, dagger_holdout, config, device, grouped=True),
                                       "v6": model_metrics(model, dagger_holdout, config, device, grouped=True)},
        }
        report["gate_reached"] = "offline_evaluation"
        onnx_path = artifact_root / "onnx/pilotnet_v6_high_speed_dagger.onnx"
        export_onnx(model, onnx_path, config)
        report["onnx_equivalence"] = validate_onnx_equivalence(
            model, [*nominal_validation, *nominal_holdout, *dagger_holdout], onnx_path, config,
        )
        report["onnx_contract"] = {"checker": "PASS", "input": ["batch", 3, 66, 200],
                                   "output": ["batch", 1], "opset": config["onnx_opset"]}
        report["gate_reached"] = "onnx_equivalence"
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
            "v5_checkpoint_comparison_sha256": sha256_file(v5_checkpoint),
        }
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(result_path, report)
    return report


def load_v6_inference(repo: Path) -> InferenceConfig:
    payload = json.loads((repo / "configs/pilotnet_inference_v6_high_speed_dagger.json").read_text())
    canonical = json.loads((repo / "configs/pilotnet_inference_v5_high_speed.json").read_text())
    differences = {key for key in set(payload) | set(canonical) if payload.get(key) != canonical.get(key)}
    if differences != {"version", "material_progress_improvement_fraction"}:
        raise GateFailure(f"V6 inference differs unexpectedly from V5: {sorted(differences)}")
    if payload.get("smoke_speeds_mps") != [1.8, 1.8, 1.8] or payload.get("maximum_smoke_runs") != 3:
        raise GateFailure("V6 permits exactly three conditional valid runs at 1.80 m/s")
    return InferenceConfig(payload)


def run_v6_attempts(
    client, model, config: InferenceConfig, result_dir: Path,
    *, preflight_one: Callable = live_preflight, run_one: Callable = run_smoke,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    passes = 0
    for number in range(1, MAX_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight_result = preflight_one(client, config)
            run = run_one(client, model, config, initial, SPEED_MPS)
            classification = classify_policy_run(run)
            attempt = {"attempt_number": number, "valid_policy_run_number": passes + 1 if classification != "INFRA_FAIL" else None,
                       "classification": classification, "preflight": preflight_result, "run": run}
        except Exception as exc:
            errors = client.safe_stop()
            attempt = {"attempt_number": number, "valid_policy_run_number": None, "classification": "INFRA_FAIL", "run": None,
                       "preflight": {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                                     "safe_stop_success": not errors, "safe_stop_errors": errors}}
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "POLICY_FAIL":
            return attempts, "FAIL"
        if attempt["classification"] == "POLICY_PASS":
            passes += 1
            if passes == TARGET_POLICY_PASSES:
                return attempts, "PASS"
    return attempts, "INCONCLUSIVE"


def _training_artifact(repo: Path) -> tuple[dict[str, Any], Path]:
    training = json.loads((repo / "results/pilotnet_training_v6_high_speed_dagger/summary.json").read_text())
    if training.get("result") != "PASS" or training.get("onnx_equivalence", {}).get("result") != "PASS":
        raise GateFailure("V6 training/ONNX gate is not PASS")
    path = Path(training["artifacts"]["onnx"]["path"])
    if sha256_file(path) != training["artifacts"]["onnx"]["sha256"]:
        raise GateFailure("V6 ONNX identity mismatch")
    return training, path


def live_preflight_stage(repo: Path) -> dict[str, Any]:
    training, onnx_path = _training_artifact(repo)
    config = load_v6_inference(repo)
    result_dir = repo / "results/pilotnet_e2e_v6_high_speed"
    result_path = result_dir / "preflight.json"
    if result_path.exists() or (result_dir / "experiment.started.json").exists():
        raise RuntimeError("refusing to repeat V6 preflight/live experiment")
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
    report = {"version": "pilotnet_e2e_v6_high_speed_preflight", "generated_utc": utc_now(), "result": "FAIL",
              "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop():
            raise GateFailure("initial safe stop failed: " + "; ".join(errors))
        _, checks = live_preflight(client, config)
        report.update(checks)
    finally:
        errors = client.safe_stop()
        report["safe_stop_success"] = not errors
        report["safe_stop_errors"] = errors
        if errors:
            report["result"] = "FAIL"
        write_json(result_path, report)
    return report


def live_stage(repo: Path) -> dict[str, Any]:
    training, onnx_path = _training_artifact(repo)
    config = load_v6_inference(repo)
    result_dir = repo / "results/pilotnet_e2e_v6_high_speed"
    marker = result_dir / "experiment.started.json"
    summary_path = result_dir / "summary.json"
    if marker.exists() or summary_path.exists():
        raise RuntimeError("refusing to repeat V6 live validation")
    model = CameraOnlyOnnxModel(onnx_path)
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
    report: dict[str, Any] = {"version": "pilotnet_e2e_v6_high_speed", "generated_utc": utc_now(),
                              "result": "INCONCLUSIVE", "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop():
            raise GateFailure("initial safe stop failed: " + "; ".join(errors))
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"status": "V6_LIVE_STARTED", "maximum_attempts": MAX_LIVE_ATTEMPTS,
                            "maximum_valid_policy_runs": TARGET_POLICY_PASSES, "started_utc": utc_now()})
        attempts, result = run_v6_attempts(client, model, config, result_dir)
        passes = sum(item["classification"] == "POLICY_PASS" for item in attempts)
        report.update({"result": result, "attempts": attempts, "policy_pass_count": passes,
                       "infrastructure_failure_count": sum(item["classification"] == "INFRA_FAIL" for item in attempts)})
        if result == "PASS":
            report["decision"] = {"classification": "FULL_SUPPORT", "v6_can_be_frozen": True,
                                  "cone_avoidance_v1_justified": True}
        elif result == "FAIL":
            failed = next(item["run"] for item in attempts if item["classification"] == "POLICY_FAIL")
            v5_fraction = 0.6782077179124942
            improvement = float(failed["route_completion_fraction"]) - v5_fraction
            material = float(config.payload["material_progress_improvement_fraction"])
            partial = improvement >= material
            report["decision"] = {
                "classification": "PARTIAL_SUPPORT" if partial else "INSUFFICIENT",
                "v5_completion_fraction": v5_fraction, "v6_completion_fraction": failed["route_completion_fraction"],
                "completion_fraction_improvement": improvement, "material_threshold": material,
                "v6_can_be_frozen": False, "cone_avoidance_v1_justified": False,
            }
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if errors:
            report["result"] = "INCONCLUSIVE"
        write_json(summary_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "extract", "train", "live-preflight", "live"))
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--rollout-id", choices=tuple(ROLLOUTS))
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    sim_root = args.sim_root.resolve()
    artifact_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_v1"
    result_dir = repo / "results/high_speed_dagger_v1"
    config_path = repo / "configs/high_speed_dagger_v1.json"
    try:
        if args.stage in {"collect", "extract"} and not args.rollout_id:
            raise ValueError("collect/extract requires --rollout-id")
        if args.stage == "collect":
            result = collect_rollout(repo=repo, sim_root=sim_root, rollout_id=args.rollout_id,
                config_path=config_path, artifact_root=artifact_root,
                result_path=result_dir / f"{args.rollout_id}_collection.json")
        elif args.stage == "extract":
            rollout_root = artifact_root / args.rollout_id
            result = extract_rollout(rollout_id=args.rollout_id, config_path=config_path,
                bag_path=rollout_root / "raw" / args.rollout_id / "bag",
                telemetry_path=rollout_root / "telemetry.json",
                source_result_path=result_dir / f"{args.rollout_id}_collection.json",
                output_root=artifact_root / "extracted")
            write_json(result_dir / f"{args.rollout_id}_extraction.json", result)
            update_compact_dagger_summary(repo, artifact_root / "extracted")
        elif args.stage == "train":
            result = training_stage(repo, sim_root)
        elif args.stage == "live-preflight":
            result = live_preflight_stage(repo)
        else:
            result = live_stage(repo)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("result") in {"PASS", "PREFLIGHT_PASS"} else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
