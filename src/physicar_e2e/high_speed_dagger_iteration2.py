"""Bounded High-Speed PilotNet DAgger Iteration-2 pipeline."""

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
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from .dataset_extractor import decode_rgb8_image, preprocess_image
from .expert_driver import wait_after_reset
from .high_speed_dagger import (
    _collector as dagger_collector, _distribution, load_config as load_dagger1_config,
    model_metrics, run_v6_attempts, verify_preserved_baselines,
)
from .high_speed_v5 import (
    EPISODES, SPEED_MPS, classify_policy_run, live_preflight, validate_v5_dataset, write_json,
)
from .pilotnet_dagger import _raw_frames, latest_causal_shadow, sim_time_ns, window_role
from .pilotnet_dagger_iteration2 import detect_final_failure_divergence
from .pilotnet_dagger_training import read_dagger_rows
from .pilotnet_failure_diagnosis import _driver_config, run_live_loop
from .pilotnet_inference import CameraOnlyOnnxModel, InferenceConfig
from .pilotnet_recovery_training import load_checkpoint
from .pilotnet_training import (
    GateFailure, error_metrics, export_onnx, load_config as load_training_config,
    sha256_file, tiny_overfit_sanity, train_baseline, validate_onnx_equivalence,
)
from .pilotnet_v4_repeatability import clock_health_preflight, verify_static_environment
from .rosbag_collector import DockerRosBackend, RecorderHandle, directory_size, verify_bag
from .sim_client import SimClient


VERSION = "high_speed_dagger_iteration2_v1"
ROLLOUTS = {"high_speed_dagger_iter2_rollout_A": "training", "high_speed_dagger_iter2_rollout_B": "holdout"}
MAX_LIVE_ATTEMPTS = 5
TARGET_POLICY_PASSES = 3
FIELDS = [
    "episode_id", "rollout_id", "sample_index", "image_path", "raw_image_source",
    "camera_record_time_ns", "camera_header_time_ns", "expert_label_time_ns", "expert_label_age_ms",
    "steering_rad", "steering_normalized", "route_s_m", "completion_fraction", "cte_m",
    "x_m", "y_m", "yaw_rad", "v6_steering_rad", "shadow_expert_steering_rad", "v6_minus_expert_rad",
    "window_role", "source_mcap_sha256", "telemetry_sha256", "source_result_sha256", "config_sha256",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("rollout_assignment") != ROLLOUTS:
        raise ValueError("Iteration-2 version or A/B assignment changed")
    fixed = (payload.get("maximum_collection_rollouts"), payload.get("diagnostic_speed_mps"),
             payload.get("lookahead_m"), payload.get("control_frequency_hz"), payload.get("max_steering_rad"),
             payload.get("minimum_reproduction_progress_fraction"), payload.get("minimum_selected_progress_fraction"),
             payload.get("window_pre_divergence_s"), payload.get("minimum_selected_samples"))
    if fixed != (2, 1.8, 0.9, 15.0, 0.349066, 0.6, 0.6, 2.0, 20):
        raise ValueError(f"Iteration-2 frozen contract changed: {fixed}")
    if payload.get("diagnostic_ros_topics") != ["/camera/image_raw", "/clock"]:
        raise ValueError("Iteration-2 raw bag must contain camera and clock")
    if payload.get("camera_only_model_observation") is not True:
        raise ValueError("V6/V7 observation must remain camera-only")
    return payload


def control_authority_contract() -> dict[str, Any]:
    return {"vehicle_controller": "PilotNet V6", "shadow_expert_control_authority": False,
            "neural_observation": ["camera"]}


def reproduction_gate(role: str, live: dict[str, Any]) -> dict[str, Any]:
    if role != "training":
        return {"result": "NOT_APPLICABLE", "reason": "B is independent holdout evidence"}
    if live.get("result") == "PASS":
        return {"result": "FAIL", "reason": "V6 unexpectedly completed a full lap; repeatability requires investigation"}
    fraction = float(live.get("route_completion_fraction", 0.0))
    if fraction < 0.60:
        return {"result": "FAIL", "reason": f"V6 failure before 60% reproduction gate: {fraction:.2%}"}
    return {"result": "PASS", "completion_fraction": fraction, "late_failure_distribution_reproduced": True}


def freeze_a_window(telemetry: Sequence[dict[str, Any]], divergence_index: int, route_length_m: float) -> dict[str, Any]:
    if not telemetry or not 0 <= divergence_index < len(telemetry):
        raise ValueError("invalid A divergence index")
    divergence_ns = sim_time_ns(telemetry[divergence_index])
    start_ns = divergence_ns - 2_000_000_000
    candidates = [row for row in telemetry if start_ns <= sim_time_ns(row) <= sim_time_ns(telemetry[-1])
                  and float(row["unwrapped_progress_m"]) / route_length_m >= 0.60]
    if len(candidates) < 20:
        raise GateFailure(f"A objective window has only {len(candidates)} valid telemetry frames")
    route_values = [float(row["unwrapped_progress_m"]) for row in candidates]
    return {
        "start_sim_time_ns": start_ns, "divergence_sim_time_ns": divergence_ns,
        "end_sim_time_ns": sim_time_ns(telemetry[-1]),
        "route_s_start_m": min(route_values), "route_s_end_m": max(route_values),
        "minimum_completion_fraction": 0.60, "route_length_m": route_length_m,
        "telemetry_frame_count": len(candidates),
        "rule": "continuous 2.0 s pre-divergence through final pre-safe-stop interval, then require completion >=60%",
    }


def route_in_frozen_window(route_s_m: float, window: dict[str, Any]) -> bool:
    """Return whether a sample lies in the immutable A-derived route interval."""
    return float(window["route_s_start_m"]) <= float(route_s_m) <= float(window["route_s_end_m"])


def _collector(config: dict[str, Any], rollout_id: str):
    base = dagger_collector(config, rollout_id)
    return type(base)(**{**asdict(base), "data_relative_root": f"physicar_e2e/high_speed_dagger_iteration2_v1/{rollout_id}/raw"})


def _baseline(repo: Path, sim_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    # Reuse the prior gate, adapting its expected V5 hash fields without touching its files.
    return verify_preserved_baselines(repo, sim_root, config)


def collect_rollout(*, repo: Path, sim_root: Path, rollout_id: str, config_path: Path,
                    artifact_root: Path, result_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    if rollout_id not in ROLLOUTS:
        raise ValueError("unknown Iteration-2 rollout")
    role = ROLLOUTS[rollout_id]
    marker = result_path.with_suffix(".started.json")
    if result_path.exists() or marker.exists():
        raise RuntimeError(f"refusing to recollect {rollout_id}")
    baseline = _baseline(repo, sim_root, config)
    v6_summary = json.loads((repo / "results/pilotnet_training_v6_high_speed_dagger/summary.json").read_text())
    v6_checkpoint = Path(v6_summary["artifacts"]["checkpoint"]["path"])
    v6_onnx = Path(v6_summary["artifacts"]["onnx"]["path"])
    if sha256_file(v6_checkpoint) != config["v6_checkpoint_sha256"] or sha256_file(v6_onnx) != config["v6_onnx_sha256"]:
        raise GateFailure("V6 artifact hash mismatch")
    if role == "holdout":
        a_path = repo / "results/high_speed_dagger_iteration2_v1/high_speed_dagger_iter2_rollout_A_collection.json"
        if not a_path.exists():
            raise GateFailure("A frozen route interval is required before B")
        a = json.loads(a_path.read_text())
        if a.get("result") != "PASS" or "frozen_window" not in a:
            raise GateFailure("A route interval is not frozen PASS")
        frozen_window = a["frozen_window"]
    else:
        frozen_window = None
    client = SimClient(config["base_url"], config["api_timeout_s"])
    backend = DockerRosBackend(_collector(config, rollout_id), sim_root)
    model = CameraOnlyOnnxModel(v6_onnx)
    handle: RecorderHandle | None = None
    telemetry: list[dict[str, Any]] = []
    report: dict[str, Any] = {"version": VERSION, "rollout_id": rollout_id, "role": role, "result": "FAIL",
                              "collection_run_count": 0, "preserved_baseline": baseline,
                              "control_authority": control_authority_contract(), "frozen_a_window": frozen_window}
    try:
        if errors := client.safe_stop():
            raise RuntimeError("initial safe-stop failed: " + "; ".join(errors))
        initial = wait_after_reset(client, _driver_config(config), False)
        environment = verify_static_environment(initial)
        clock = clock_health_preflight(client)
        if clock.get("result") != "PASS":
            raise RuntimeError("clock health preflight failed")
        jpeg = client.camera_jpeg(config["camera_path"])
        with Image.open(io.BytesIO(jpeg)) as image:
            if image.size != (480, 360) or image.format != "JPEG":
                raise RuntimeError("camera contract mismatch")
        topics = backend.preflight(config["diagnostic_ros_topics"])
        if shutil.disk_usage(backend.host_userdata_root).free < config["minimum_free_bytes"]:
            raise RuntimeError("insufficient external userdata free space")
        report["preflight"] = {"result": "PASS", "environment": environment, "clock_health": clock,
                               "camera": {"transport": "HTTP JPEG", "dimensions": [480, 360]}, "topics": topics}
        handle = backend.start_recorder(rollout_id, config["diagnostic_ros_topics"])
        write_json(marker, {"status": f"{rollout_id}_STARTED_DO_NOT_RETRY_POLICY_FAILURE", "started_utc": utc_now()})
        report["collection_run_count"] = 1
        metrics, telemetry = run_live_loop(client, model, config, initial,
                                           artifact_root / rollout_id / "http_frames", policy_name="PilotNet V6")
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
    if "live" not in report or not telemetry or handle is None:
        raise GateFailure(f"{rollout_id} did not produce one valid rollout")
    telemetry_path = artifact_root / rollout_id / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    runtime_ok = (report["live"]["api_failures"] == 0 and report["live"]["liveness_failures"] == 0
                  and report["live"]["safe_stop_success"] and report["final_safe_stop_success"]
                  and report.get("recorder_stop", {}).get("graceful") is True
                  and not report.get("recorder_stop", {}).get("orphaned", True)
                  and "recorder_failure" not in report)
    report["classification"] = "INFRA_FAIL" if not runtime_ok else classify_policy_run(report["live"])
    report["reproduction_gate"] = reproduction_gate(role, report["live"])
    if runtime_ok and report["reproduction_gate"]["result"] in {"PASS", "NOT_APPLICABLE"}:
        divergence = detect_final_failure_divergence(
            [row["elapsed_s"] for row in telemetry], [row["cte_m"] for row in telemetry],
            stable_window_s=config["stable_window_s"], cte_floor_m=config["divergence_cte_floor_m"],
            persistence_samples=config["divergence_persistence_samples"],
        ) if report["live"]["result"] == "FAIL" else None
        report["objective_divergence"] = divergence
        if role == "training":
            report["frozen_window"] = freeze_a_window(telemetry, divergence["divergence_index"], report["live"]["route_length_m"])
        else:
            target = frozen_window
            reached = max(float(row["unwrapped_progress_m"]) for row in telemetry)
            report["holdout_region_gate"] = {"result": "PASS" if reached >= target["route_s_start_m"] else "FAIL",
                                              "target_route_s_start_m": target["route_s_start_m"],
                                              "target_route_s_end_m": target["route_s_end_m"], "maximum_route_s_m": reached}
            if report["holdout_region_gate"]["result"] != "PASS":
                report["result"] = "FAIL"
                write_json(result_path, report)
                return report
        report["telemetry"] = str(telemetry_path)
        report["telemetry_sha256"] = sha256_file(telemetry_path)
        report["result"] = "PASS"
    write_json(result_path, report)
    return report


def _write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def extract_rollout(*, rollout_id: str, config_path: Path, bag_path: Path, telemetry_path: Path,
                    source_result_path: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source = json.loads(source_result_path.read_text())
    if source.get("result") != "PASS" or source.get("collection_run_count") != 1:
        raise GateFailure("collection gate is not PASS")
    telemetry = json.loads(telemetry_path.read_text())
    frames = list(_raw_frames(bag_path))
    if not frames:
        raise GateFailure("rollout has no raw camera frames")
    target = source.get("frozen_window") or source.get("frozen_a_window")
    if not target:
        raise GateFailure("missing frozen A route window")
    if rollout_id.endswith("_A"):
        candidates = [item for item in frames if target["start_sim_time_ns"] <= item[2] <= target["end_sim_time_ns"]]
        selection_rule = "A frozen simulator-time window plus >=60% completion"
    else:
        candidates = [item for item in frames if route_in_frozen_window(
            (latest_causal_shadow(telemetry, item[2]) or {"unwrapped_progress_m": -1})["unwrapped_progress_m"], target)]
        selection_rule = "B frames in the frozen A route-s interval"
    source_mcap = frames[0][0]
    source_hash, telemetry_hash = sha256_file(source_mcap), sha256_file(telemetry_path)
    result_hash, config_hash = sha256_file(source_result_path), sha256_file(config_path)
    accepted: list[tuple[Any, dict[str, Any], int]] = []
    stale = future = decode_failures = 0
    for item in candidates:
        label = latest_causal_shadow(telemetry, item[2])
        if label is None:
            continue
        progress = float(label["unwrapped_progress_m"])
        if rollout_id.endswith("_A") and progress / float(source["live"]["route_length_m"]) < 0.60:
            continue
        age_ns = item[2] - sim_time_ns(label)
        if age_ns < 0:
            future += 1; continue
        if age_ns / 1e9 > config["maximum_expert_label_age_s"]:
            stale += 1; continue
        try:
            # Decode before creating output so malformed frames are counted and never silently accepted.
            decode_rgb8_image(item[3], config)
        except Exception:
            decode_failures += 1; continue
        accepted.append((item, label, age_ns))
    if future != 0 or decode_failures != 0 or len(accepted) < config["minimum_selected_samples"]:
        raise GateFailure(f"Iteration-2 extraction gate failed: samples={len(accepted)}, stale={stale}, future={future}, decode={decode_failures}")
    manifest_dir, image_dir, preview_dir = output_root / "manifests", output_root / "images" / rollout_id, output_root / "previews"
    manifest_path = manifest_dir / f"{rollout_id}.csv"
    if manifest_path.exists() or image_dir.exists():
        raise RuntimeError(f"refusing to overwrite {rollout_id}")
    manifest_dir.mkdir(parents=True, exist_ok=True); image_dir.mkdir(parents=True); preview_dir.mkdir(parents=True, exist_ok=True)
    image_config = {key: config[key] for key in ("source_width", "source_height", "source_encoding", "roi", "output_width", "output_height")}
    rows: list[dict[str, Any]] = []
    route_length = float(source["live"]["route_length_m"])
    for item, label, age_ns in accepted:
        image = preprocess_image(decode_rgb8_image(item[3], image_config), image_config)
        index = len(rows); relative = Path("images") / rollout_id / f"frame_{index:06d}.png"
        image.save(output_root / relative, format="PNG", optimize=False); image.close()
        progress = float(label["unwrapped_progress_m"]); expert = float(label["shadow_expert_steering_rad"]); v6 = float(label["network_steering_rad"])
        rows.append({"episode_id": rollout_id, "rollout_id": rollout_id, "sample_index": index,
                     "image_path": relative.as_posix(), "raw_image_source": f"{item[0]}:/camera/image_raw",
                     "camera_record_time_ns": item[1], "camera_header_time_ns": item[2],
                     "expert_label_time_ns": sim_time_ns(label), "expert_label_age_ms": age_ns / 1e6,
                     "steering_rad": expert, "steering_normalized": expert / config["max_steering_rad"],
                     "route_s_m": progress, "completion_fraction": progress / route_length, "cte_m": label["cte_m"],
                     "x_m": label["x"], "y_m": label["y"], "yaw_rad": label["yaw"],
                     "v6_steering_rad": v6, "shadow_expert_steering_rad": expert, "v6_minus_expert_rad": v6 - expert,
                     "window_role": window_role(item[2], target["divergence_sim_time_ns"], config["divergence_evaluation_duration_s"]),
                     "source_mcap_sha256": source_hash, "telemetry_sha256": telemetry_hash,
                     "source_result_sha256": result_hash, "config_sha256": config_hash})
    _write_manifest(manifest_path, rows)
    indices = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    images = [Image.open(output_root / rows[int(i)]["image_path"]).convert("RGB") for i in indices]
    sheet = Image.new("RGB", (600, math.ceil(len(images) / 3) * 66))
    for i, image in enumerate(images): sheet.paste(image, ((i % 3) * 200, (i // 3) * 66)); image.close()
    preview = preview_dir / f"{rollout_id}.png"; sheet.save(preview)
    result = {"version": VERSION, "rollout_id": rollout_id, "role": ROLLOUTS[rollout_id], "result": "PASS",
              "selection_rule": selection_rule, "source_bag": str(bag_path), "source_raw_size_bytes": directory_size(bag_path),
              "source_mcap_sha256": source_hash, "telemetry_sha256": telemetry_hash, "source_result_sha256": result_hash,
              "config_sha256": config_hash, "raw_camera_frames": len(frames), "candidate_frames": len(candidates),
              "accepted_samples": len(rows), "stale_label_rejections": stale, "future_label_violations": future,
              "image_decode_failures": decode_failures,
              "expert_label_age_ms": _distribution([float(r["expert_label_age_ms"]) for r in rows]),
              "route_s_range_m": [min(float(r["route_s_m"]) for r in rows), max(float(r["route_s_m"]) for r in rows)],
              "completion_fraction_range": [min(float(r["completion_fraction"]) for r in rows), max(float(r["completion_fraction"]) for r in rows)],
              "window_counts": {name: sum(r["window_role"] == name for r in rows) for name in ("pre_divergence", "divergence", "late_failure")},
              "frozen_a_window": target, "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "preview": str(preview)}
    metadata_path = output_root / "dataset_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {"version": VERSION, "episodes": []}
    if any(x["rollout_id"] == rollout_id for x in metadata["episodes"]): raise RuntimeError("duplicate metadata rollout")
    metadata["episodes"].append(result); metadata["future_label_violations"] = sum(x["future_label_violations"] for x in metadata["episodes"]); metadata["result"] = "PASS"
    write_json(metadata_path, metadata)
    return result


def validate_cumulative_composition(nominal_train, nominal_val, nominal_hold, d1a, d1b, d2a, d2b) -> dict[str, Any]:
    if {x["episode_id"] for x in d1a} != {"high_speed_dagger_rollout_A"} or {x["episode_id"] for x in d2a} != {"high_speed_dagger_iter2_rollout_A"}:
        raise GateFailure("DAgger A training roles changed")
    if {x["episode_id"] for x in d1b} != {"high_speed_dagger_rollout_B"} or {x["episode_id"] for x in d2b} != {"high_speed_dagger_iter2_rollout_B"}:
        raise GateFailure("DAgger B holdout roles changed")
    train = [*nominal_train, *d1a, *d2a]; evaluation = [*nominal_val, *nominal_hold, *d1b, *d2b]
    if {x["image_path"] for x in train} & {x["image_path"] for x in evaluation}: raise GateFailure("image leakage")
    train_sources = {x["source_mcap_sha256"] for x in [*d1a, *d2a]}; hold_sources = {x["source_mcap_sha256"] for x in [*d1b, *d2b]}
    if train_sources & hold_sources: raise GateFailure("source hash leakage")
    ids = {x["episode_id"] for x in train}
    if any("recovery" in x or x.startswith("dagger_") for x in ids): raise GateFailure("forbidden low-speed/recovery data")
    return {"result": "PASS", "nominal_train_unchanged": True, "dagger1_A_retained": True,
            "dagger1_B_excluded": True, "dagger2_B_excluded": True, "v4_data_included": False, "low_speed_data_included": False}


def training_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_path = repo / "results/pilotnet_training_v7_high_speed_dagger/summary.json"
    if result_path.exists(): raise RuntimeError("refusing to repeat V7 training")
    config_path = repo / "configs/pilotnet_training_v7_high_speed_dagger.json"; config = load_training_config(config_path)
    if config.get("version") != "pilotnet_training_v7_high_speed_dagger" or config.get("initialization") != "from_scratch": raise GateFailure("V7 config gate failed")
    nominal_root = sim_root / "userdata/physicar_e2e/high_speed_v1/dataset"
    d1_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_v1/extracted"
    d2_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_iteration2_v1/extracted"
    artifact_root = sim_root / "userdata/physicar_e2e/high_speed_dagger_iteration2_v1/v7"
    if artifact_root.exists(): raise RuntimeError("refusing to overwrite V7 artifacts")
    d2meta = json.loads((d2_root / "dataset_metadata.json").read_text())
    if d2meta.get("result") != "PASS" or d2meta.get("future_label_violations") != 0 or {x["role"] for x in d2meta.get("episodes", [])} != {"training", "holdout"}:
        raise GateFailure("Iteration-2 dataset gate failed")
    nominal_train, nominal_val, nominal_hold, nominal_integrity = validate_v5_dataset(nominal_root, config)
    d1a, d1b = read_dagger_rows(d1_root, config["dagger1_training_rollout"]), read_dagger_rows(d1_root, config["dagger1_holdout_rollout"])
    d2a, d2b = read_dagger_rows(d2_root, config["dagger2_training_rollout"]), read_dagger_rows(d2_root, config["dagger2_holdout_rollout"])
    composition = validate_cumulative_composition(nominal_train, nominal_val, nominal_hold, d1a, d1b, d2a, d2b)
    combined = [*nominal_train, *d1a, *d2a]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v5 = json.loads((repo / "results/pilotnet_training_v5_high_speed/summary.json").read_text()); v6 = json.loads((repo / "results/pilotnet_training_v6_high_speed_dagger/summary.json").read_text())
    v5_model_path, v6_model_path = Path(v5["artifacts"]["checkpoint"]["path"]), Path(v6["artifacts"]["checkpoint"]["path"])
    report: dict[str, Any] = {"version": "pilotnet_training_v7_high_speed_dagger", "generated_utc": utc_now(), "result": "FAIL", "device": str(device),
      "architecture": {"input_shape": [3, 66, 200], "parameter_count": 252219, "identical_to_v5_v6": True}, "training_from_scratch": True, "v6_checkpoint_used_for_training": False,
      "dataset": {"nominal_integrity": nominal_integrity, "nominal_training_samples": len(nominal_train), "dagger1_A_samples": len(d1a), "dagger2_A_samples": len(d2a), "combined_training_samples": len(combined), "nominal_validation_samples": len(nominal_val), "nominal_holdout_samples": len(nominal_hold), "dagger1_B_samples": len(d1b), "dagger2_B_samples": len(d2b), "dagger2_metadata_sha256": sha256_file(d2_root / "dataset_metadata.json"), **composition}}
    try:
        report["tiny_overfit"] = tiny_overfit_sanity(combined, config, device)
        checkpoint = artifact_root / "checkpoints/pilotnet_v7_high_speed_dagger_best.pt"; model, training, history = train_baseline(combined, nominal_val, config, device, checkpoint)
        report["training"] = {**training, "initialized_from_scratch": True}; report["epochs"] = history; report["gate_reached"] = "training"
        v5m, v6m = load_checkpoint(v5_model_path, device), load_checkpoint(v6_model_path, device)
        report["offline_comparison"] = {
            "nominal_validation": {"v5": model_metrics(v5m, nominal_val, config, device), "v6": model_metrics(v6m, nominal_val, config, device), "v7": model_metrics(model, nominal_val, config, device)},
            "nominal_holdout": {"v5": model_metrics(v5m, nominal_hold, config, device), "v6": model_metrics(v6m, nominal_hold, config, device), "v7": model_metrics(model, nominal_hold, config, device)},
            "dagger1_rollout_B": {"v5": model_metrics(v5m, d1b, config, device, grouped=True), "v6": model_metrics(v6m, d1b, config, device, grouped=True), "v7": model_metrics(model, d1b, config, device, grouped=True)},
            "dagger2_rollout_B": {"v5": model_metrics(v5m, d2b, config, device, grouped=True), "v6": model_metrics(v6m, d2b, config, device, grouped=True), "v7": model_metrics(model, d2b, config, device, grouped=True)},
        }
        onnx_path = artifact_root / "onnx/pilotnet_v7_high_speed_dagger.onnx"; export_onnx(model, onnx_path, config)
        report["onnx_contract"] = {"checker": "PASS", "input": ["batch", 3, 66, 200], "output": ["batch", 1]}
        report["onnx_equivalence"] = validate_onnx_equivalence(model, [*nominal_val, *nominal_hold, *d1b, *d2b], onnx_path, config)
        report["artifacts"] = {"checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)}, "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)}, "v5_checkpoint_sha256": sha256_file(v5_model_path), "v6_checkpoint_sha256": sha256_file(v6_model_path)}
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}; raise
    finally: write_json(result_path, report)
    return report


def load_v7_inference(repo: Path) -> InferenceConfig:
    payload = json.loads((repo / "configs/pilotnet_inference_v7_high_speed_dagger.json").read_text()); canonical = json.loads((repo / "configs/pilotnet_inference_v6_high_speed_dagger.json").read_text())
    if {key for key in set(payload) | set(canonical) if payload.get(key) != canonical.get(key)} != {"version"}: raise GateFailure("V7 inference contract differs from V6")
    return InferenceConfig(payload)


def _v7_artifact(repo: Path):
    training = json.loads((repo / "results/pilotnet_training_v7_high_speed_dagger/summary.json").read_text())
    if training.get("result") != "PASS" or training.get("onnx_equivalence", {}).get("result") != "PASS": raise GateFailure("V7 ONNX gate failed")
    path = Path(training["artifacts"]["onnx"]["path"])
    if sha256_file(path) != training["artifacts"]["onnx"]["sha256"]: raise GateFailure("V7 ONNX hash mismatch")
    return training, path


def live_preflight_stage(repo: Path) -> dict[str, Any]:
    training, path = _v7_artifact(repo); config = load_v7_inference(repo); result_dir = repo / "results/pilotnet_e2e_v7_high_speed"; result_path = result_dir / "preflight.json"
    if result_path.exists() or (result_dir / "experiment.started.json").exists(): raise RuntimeError("refusing to repeat V7 preflight")
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"]); report = {"version": "pilotnet_e2e_v7_high_speed_preflight", "generated_utc": utc_now(), "result": "FAIL", "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop(): raise GateFailure("initial safe-stop failed: " + "; ".join(errors))
        _, checks = live_preflight(client, config); report.update(checks)
    finally:
        errors = client.safe_stop(); report["safe_stop_success"] = not errors; report["safe_stop_errors"] = errors; report["result"] = "FAIL" if errors else report.get("result", "PASS"); write_json(result_path, report)
    return report


def live_stage(repo: Path) -> dict[str, Any]:
    training, path = _v7_artifact(repo); config = load_v7_inference(repo); result_dir = repo / "results/pilotnet_e2e_v7_high_speed"; marker = result_dir / "experiment.started.json"; summary_path = result_dir / "summary.json"
    if marker.exists() or summary_path.exists(): raise RuntimeError("refusing to repeat V7 live validation")
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"]); model = CameraOnlyOnnxModel(path); report: dict[str, Any] = {"version": "pilotnet_e2e_v7_high_speed", "generated_utc": utc_now(), "result": "INCONCLUSIVE", "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop(): raise GateFailure("initial safe-stop failed: " + "; ".join(errors))
        result_dir.mkdir(parents=True, exist_ok=True); write_json(marker, {"status": "V7_LIVE_STARTED", "maximum_attempts": MAX_LIVE_ATTEMPTS, "maximum_valid_policy_runs": TARGET_POLICY_PASSES, "iteration_3_automatic": False, "started_utc": utc_now()})
        attempts, result = run_v6_attempts(client, model, config, result_dir, preflight_one=live_preflight)
        report.update({"result": result, "attempts": attempts, "policy_pass_count": sum(x["classification"] == "POLICY_PASS" for x in attempts), "infrastructure_failure_count": sum(x["classification"] == "INFRA_FAIL" for x in attempts)})
        if result == "PASS": report["decision"] = {"classification": "PASS", "v7_can_be_frozen": True, "cone_avoidance_v1_justified": True, "automatic_iteration3": False}
        elif result == "FAIL":
            failed = next(x["run"] for x in attempts if x["classification"] == "POLICY_FAIL"); v6_progress = 25.439457538760948; improvement_m = float(failed["total_unwrapped_progress_m"]) - v6_progress; improvement_fraction = float(failed["route_completion_fraction"]) - 0.8339545057064432
            report["decision"] = {"classification": "PARTIAL_SUPPORT" if improvement_m >= 1.5 or improvement_fraction >= .05 else "NOT_SUFFICIENT", "v6_progress_m": v6_progress, "v7_progress_m": failed["total_unwrapped_progress_m"], "improvement_m": improvement_m, "improvement_fraction": improvement_fraction, "v7_can_be_frozen": False, "cone_avoidance_v1_justified": False, "automatic_iteration3": False}
    finally:
        errors = client.safe_stop(); report["final_safe_stop_success"] = not errors; report["final_safe_stop_errors"] = errors; write_json(summary_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("stage", choices=("collect", "extract", "train", "live-preflight", "live")); parser.add_argument("--sim-root", type=Path, required=True); parser.add_argument("--rollout-id", choices=tuple(ROLLOUTS)); args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]; sim_root = args.sim_root.resolve(); root = sim_root / "userdata/physicar_e2e/high_speed_dagger_iteration2_v1"; result_dir = repo / "results/high_speed_dagger_iteration2_v1"; config_path = repo / "configs/high_speed_dagger_iteration2_v1.json"
    try:
        if args.stage in {"collect", "extract"} and not args.rollout_id: raise ValueError("--rollout-id is required")
        if args.stage == "collect": result = collect_rollout(repo=repo, sim_root=sim_root, rollout_id=args.rollout_id, config_path=config_path, artifact_root=root, result_path=result_dir / f"{args.rollout_id}_collection.json")
        elif args.stage == "extract":
            rr = root / args.rollout_id; result = extract_rollout(rollout_id=args.rollout_id, config_path=config_path, bag_path=rr / "raw" / args.rollout_id / "bag", telemetry_path=rr / "telemetry.json", source_result_path=result_dir / f"{args.rollout_id}_collection.json", output_root=root / "extracted"); write_json(result_dir / f"{args.rollout_id}_extraction.json", result)
        elif args.stage == "train": result = training_stage(repo, sim_root)
        elif args.stage == "live-preflight": result = live_preflight_stage(repo)
        else: result = live_stage(repo)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result.get("result") in {"PASS", "PREFLIGHT_PASS"} else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {type(exc).__name__}: {exc}", file=__import__("sys").stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
