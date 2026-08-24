"""PilotNet Failure Diagnosis V1: offline evidence and one gated live run."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import time
from typing import Any, Callable, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image
import torch

from .dataset_extractor import decode_rgb8_image, preprocess_image
from .expert_driver import DriverConfig, PoseLivenessMonitor, preflight, wait_after_reset
from .pilotnet import (
    IMAGE_HEIGHT, IMAGE_WIDTH, MAX_STEERING_RAD, PILOTNET_PARAMETER_COUNT,
    RGB_TO_YUV_BT601, build_pilotnet, clamp_steering_rad, preprocess_live_jpeg,
    preprocess_png, preprocess_rgb, steering_normalized_to_rad,
)
from .pilotnet_inference import CameraOnlyOnnxModel, sha256_file
from .pilotnet_training import read_episode_rows
from .rosbag_collector import (
    CollectorConfig, DockerRosBackend, RecorderHandle, directory_size, verify_bag,
)
from .route_geometry import OffTrackMonitor, ProgressTracker, pure_pursuit_steering
from .sim_client import SimClient


VERSION = "pilotnet_failure_diagnosis_v1"
MAGNITUDE_BIN_NAMES = ("abs_lt_0.05", "abs_0.05_to_0.15", "abs_0.15_to_0.25", "abs_ge_0.25")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "version", "base_url", "expected_world", "camera_only_model_observation",
        "diagnostic_speed_mps", "maximum_live_runs", "temporal_shift_ms",
        "magnitude_bin_edges_rad", "feature_layer", "diagnostic_ros_topics",
        "v1_checkpoint_sha256", "v1_onnx_sha256", "v2_checkpoint_sha256",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"diagnostic config missing fields: {sorted(missing)}")
    if payload["version"] != VERSION:
        raise ValueError("unexpected diagnostic config version")
    if payload["camera_only_model_observation"] is not True:
        raise ValueError("neural observation must remain camera only")
    if payload["diagnostic_speed_mps"] != 0.50 or payload["maximum_live_runs"] != 1:
        raise ValueError("diagnostic permits exactly one V1 run at 0.50 m/s")
    if payload["diagnostic_ros_topics"] != ["/camera/image_raw", "/clock"]:
        raise ValueError("diagnostic bag contract must remain raw camera plus clock only")
    return payload


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_pilotnet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_rows(model, rows: Sequence[dict[str, Any]], device: torch.device, batch_size: int = 64):
    predictions: list[float] = []
    labels: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            images = torch.from_numpy(np.stack([preprocess_png(row["image_path"]) for row in batch])).to(device)
            output = model(images).cpu().numpy().reshape(-1)
            predictions.extend(steering_normalized_to_rad(output).tolist())
            labels.extend(float(row["steering_rad"]) for row in batch)
    return np.asarray(predictions, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def basic_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    if predictions.shape != labels.shape or predictions.size == 0:
        raise ValueError("metrics require equal non-empty arrays")
    error = predictions - labels
    return {
        "count": int(error.size),
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(error * error))),
        "signed_bias_rad": float(np.mean(error)),
        "max_abs_error_rad": float(np.max(np.abs(error))),
        "correlation": safe_correlation(predictions, labels),
    }


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def steering_calibration(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    result = basic_metrics(predictions, labels)
    slope, intercept = np.polyfit(labels, predictions, 1)
    result.update({"regression_slope": float(slope), "regression_intercept_rad": float(intercept)})
    absolute = np.abs(labels)
    masks = (
        absolute < 0.05,
        (absolute >= 0.05) & (absolute < 0.15),
        (absolute >= 0.15) & (absolute < 0.25),
        absolute >= 0.25,
    )
    bins: dict[str, Any] = {}
    for name, mask in zip(MAGNITUDE_BIN_NAMES, masks, strict=True):
        if not np.any(mask):
            bins[name] = {"count": 0}
            continue
        pred = predictions[mask]
        label = labels[mask]
        mean_gt = float(np.mean(np.abs(label)))
        bins[name] = {
            **basic_metrics(pred, label),
            "gt_mean_abs_rad": mean_gt,
            "prediction_mean_abs_rad": float(np.mean(np.abs(pred))),
            "absolute_magnitude_ratio": float(np.mean(np.abs(pred)) / mean_gt) if mean_gt else None,
        }
    result["magnitude_bins"] = bins
    return result


def temporal_shift_diagnostic(
    timestamps_ns: np.ndarray, predictions: np.ndarray, labels: np.ndarray, shifts_ms: Sequence[float]
) -> dict[str, Any]:
    if not (timestamps_ns.size == predictions.size == labels.size) or timestamps_ns.size < 2:
        raise ValueError("temporal shift requires ordered equal-length sequences")
    order = np.argsort(timestamps_ns)
    t = timestamps_ns[order].astype(np.float64)
    pred = predictions[order]
    label = labels[order]
    rows: list[dict[str, Any]] = []
    for shift_ms in shifts_ms:
        query = t + float(shift_ms) * 1e6
        valid = (query >= t[0]) & (query <= t[-1])
        aligned = np.interp(query[valid], t, label)
        metric = basic_metrics(pred[valid], aligned)
        rows.append({"shift_ms": float(shift_ms), **metric})
    zero = next(row for row in rows if row["shift_ms"] == 0.0)
    best = min(rows, key=lambda row: row["mae_rad"])
    return {
        "sign_convention": "positive shift compares prediction at t with expert label at t+shift",
        "alignment": "linear interpolation on actual camera record timestamps; diagnostic only",
        "shifts": rows,
        "zero_shift": zero,
        "best_shift": best,
        "mae_improvement_fraction": float((zero["mae_rad"] - best["mae_rad"]) / zero["mae_rad"]),
    }


def extract_features(model, tensors: np.ndarray, device: torch.device, batch_size: int = 64) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tensors.shape[0], batch_size):
            batch = torch.from_numpy(tensors[start : start + batch_size]).to(device)
            values.append(model.features(batch).flatten(1).cpu().numpy())
    return np.ascontiguousarray(np.concatenate(values), dtype=np.float32)


def l2_normalize_features(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, np.float32(1e-12))


def nearest_cosine_distances(query: np.ndarray, reference: np.ndarray, chunk_size: int = 256) -> np.ndarray:
    q = l2_normalize_features(np.asarray(query, dtype=np.float32))
    r = l2_normalize_features(np.asarray(reference, dtype=np.float32))
    output = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], chunk_size):
        similarities = q[start : start + chunk_size] @ r.T
        output[start : start + chunk_size] = 1.0 - np.max(similarities, axis=1)
    return output


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "max": float(np.max(array)),
    }


def detect_divergence_windows(
    times_s: Sequence[float], ctes_m: Sequence[float], *, stable_window_s: float = 1.0,
    cte_floor_m: float = 0.03, persistence_samples: int = 5,
) -> dict[str, Any]:
    t = np.asarray(times_s, dtype=np.float64)
    cte = np.abs(np.asarray(ctes_m, dtype=np.float64))
    if t.size != cte.size or t.size < persistence_samples + 2:
        raise ValueError("insufficient telemetry for divergence detection")
    stable = np.flatnonzero(t <= t[0] + stable_window_s)
    baseline = cte[stable]
    threshold = max(float(cte_floor_m), float(np.mean(baseline) + 3.0 * np.std(baseline)))
    onset: int | None = None
    for index in range(max(1, len(stable)), t.size - persistence_samples + 1):
        window = cte[index : index + persistence_samples]
        if window[0] >= threshold and np.polyfit(t[index : index + persistence_samples], window, 1)[0] > 0:
            onset = index
            break
    if onset is None:
        onset = int(np.argmax(cte))
    critical = np.flatnonzero((t >= t[onset] - 2.0) & (t < t[onset]))
    late = np.flatnonzero(t >= max(t[0], t[-1] - 1.0))
    return {
        "method": "first persistent above-baseline CTE window with positive linear slope",
        "threshold_m": threshold, "divergence_index": onset, "divergence_time_s": float(t[onset]),
        "stable_indices": stable.tolist(), "critical_pre_onset_indices": critical.tolist(),
        "late_indices": late.tolist(),
    }


def steering_window_metrics(rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    net = np.asarray([rows[i]["network_steering_rad"] for i in indices], dtype=np.float64)
    expert = np.asarray([rows[i]["shadow_expert_steering_rad"] for i in indices], dtype=np.float64)
    if not net.size:
        return {"count": 0}
    difference = net - expert
    expert_abs = float(np.mean(np.abs(expert)))
    return {
        "count": int(net.size), "mean_network_rad": float(np.mean(net)),
        "mean_shadow_expert_rad": float(np.mean(expert)), "mean_signed_difference_rad": float(np.mean(difference)),
        "mean_absolute_difference_rad": float(np.mean(np.abs(difference))),
        "corrective_magnitude_ratio": float(np.mean(np.abs(net)) / expert_abs) if expert_abs else None,
        "correlation": safe_correlation(net, expert),
        "same_sign_fraction": float(np.mean(np.sign(net) == np.sign(expert))),
    }


def associate_frames(http_times_ns: Sequence[int], raw_times_ns: Sequence[int], tolerance_ms: float) -> list[tuple[int, int, float]]:
    raw = np.asarray(raw_times_ns, dtype=np.int64)
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    limit_ns = tolerance_ms * 1e6
    for http_index, value in enumerate(http_times_ns):
        order = np.argsort(np.abs(raw - int(value)))
        selected = next((int(i) for i in order if int(i) not in used and abs(int(raw[i]) - int(value)) <= limit_ns), None)
        if selected is not None:
            used.add(selected)
            matches.append((http_index, selected, (int(raw[selected]) - int(value)) / 1e6))
    return matches


def compare_transport_predictions(
    raw_rgb: np.ndarray, jpeg_bytes: bytes, predictor: Callable[[np.ndarray], float], roi=(0, 160, 480, 360),
) -> dict[str, float]:
    extractor_config = {
        "source_width": 480, "source_height": 360, "source_encoding": "rgb8",
        "roi": {"x_start": roi[0], "y_start": roi[1], "x_end": roi[2], "y_end": roi[3]},
        "output_width": 200, "output_height": 66,
    }
    raw_image = Image.fromarray(np.asarray(raw_rgb, dtype=np.uint8), "RGB")
    raw_resized = np.asarray(preprocess_image(raw_image, extractor_config), dtype=np.uint8)
    with Image.open(io.BytesIO(jpeg_bytes)) as image:
        jpeg_resized = np.asarray(image.convert("RGB").crop(roi).resize((200, 66), Image.Resampling.BILINEAR), dtype=np.uint8)
    raw_prediction = float(predictor(preprocess_rgb(raw_resized)))
    jpeg_prediction = float(predictor(preprocess_rgb(jpeg_resized)))
    return {
        "raw_prediction_rad": raw_prediction, "jpeg_prediction_rad": jpeg_prediction,
        "signed_prediction_difference_rad": jpeg_prediction - raw_prediction,
        "absolute_prediction_difference_rad": abs(jpeg_prediction - raw_prediction),
        "mean_absolute_pixel_difference_0_255": float(np.mean(np.abs(jpeg_resized.astype(float) - raw_resized.astype(float)))),
    }


def audit_preprocessing(dataset_config_path: Path, training_config_path: Path, inference_config_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_config_path.read_text(encoding="utf-8"))
    training = json.loads(training_config_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_config_path.read_text(encoding="utf-8"))
    checks = {
        "roi": dataset["roi"] == inference["roi"],
        "resize_dimensions": [dataset["output_width"], dataset["output_height"]] == [training["image_width"], training["image_height"]] == [inference["model_width"], inference["model_height"]],
        "resize_interpolation": dataset["resize_interpolation"] == inference["resize_interpolation"] == "bilinear",
        "rgb_channel_interpretation": dataset["stored_color_space"] == training["preprocessing"]["stored_color_space"] == "RGB",
        "yuv_convention": training["preprocessing"]["model_color_space"] == inference["model_color_space"] == "YUV_BT601_full_range",
        "normalization": training["preprocessing"]["normalization"] == inference["normalization"] == "(channel - 0.5) * 2.0",
        "steering_denormalization": training["max_steering_rad"] == inference["max_steering_rad"] == MAX_STEERING_RAD,
        "chw_shape": [3, IMAGE_HEIGHT, IMAGE_WIDTH] == [3, inference["model_height"], inference["model_width"]],
    }
    synthetic = np.arange(IMAGE_HEIGHT * IMAGE_WIDTH * 3, dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    checks["shared_preprocess_is_deterministic"] = bool(np.array_equal(preprocess_rgb(synthetic), preprocess_rgb(synthetic.copy())))
    return {
        "result": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "rgb_to_yuv_matrix": RGB_TO_YUV_BT601.tolist(),
        "pipeline": {
            "training_offline": "RGB PNG 200x66 -> preprocess_rgb -> BT.601-style YUV -> normalization -> CHW",
            "live": "HTTP JPEG 480x360 -> RGB -> ROI -> bilinear 200x66 -> same preprocess_rgb",
        },
    }


def _timestamps(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    values: list[int] = []
    for row in rows:
        with Path(row["image_path"]).parent.parent.parent.joinpath("manifests", f"{row['episode_id']}.csv").open(newline="", encoding="utf-8") as stream:
            manifest = {int(item["sample_index"]): item for item in csv.DictReader(stream)}
        values.append(int(manifest[int(row["sample_index"])]["camera_record_time_ns"]))
    return np.asarray(values, dtype=np.int64)


def _read_timestamps(dataset_root: Path, episode: str) -> dict[int, int]:
    with (dataset_root / "manifests" / f"{episode}.csv").open(newline="", encoding="utf-8") as stream:
        return {int(row["sample_index"]): int(row["camera_record_time_ns"]) for row in csv.DictReader(stream)}


def run_offline(
    *, config_path: Path, dataset_root: Path, v1_checkpoint: Path, v2_checkpoint: Path,
    artifact_root: Path, result_path: Path, repo_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    report: dict[str, Any] = {"version": VERSION, "generated_utc": utc_now(), "result": "FAIL", "gate_reached": "provenance"}
    hashes = {
        "v1_checkpoint_sha256": sha256_file(v1_checkpoint),
        "v2_checkpoint_sha256": sha256_file(v2_checkpoint),
        "diagnostic_config_sha256": sha256_file(config_path),
    }
    if hashes["v1_checkpoint_sha256"] != config["v1_checkpoint_sha256"] or hashes["v2_checkpoint_sha256"] != config["v2_checkpoint_sha256"]:
        raise RuntimeError("canonical checkpoint hash mismatch")
    report["provenance"] = hashes
    audit = audit_preprocessing(repo_root / "configs/dataset_extractor_v1.json", repo_root / "configs/pilotnet_training_v1.json", repo_root / "configs/pilotnet_inference_v1.json")
    report["preprocessing_audit"] = audit
    report["gate_reached"] = "preprocessing_audit"
    if audit["result"] != "PASS":
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    train_rows = read_episode_rows(dataset_root, ["episode_001", "episode_002"])
    validation_rows = read_episode_rows(dataset_root, ["episode_003"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v1 = load_model(v1_checkpoint, device)
    v2 = load_model(v2_checkpoint, device)
    v1_predictions, labels = predict_rows(v1, validation_rows, device)
    v2_predictions, labels_v2 = predict_rows(v2, validation_rows, device)
    if not np.array_equal(labels, labels_v2):
        raise RuntimeError("V1/V2 validation labels differ")
    timestamps_by_index = _read_timestamps(dataset_root, "episode_003")
    timestamps = np.asarray([timestamps_by_index[int(row["sample_index"])] for row in validation_rows], dtype=np.int64)
    report["offline"] = {
        "route_section_0_to_5m": {"result": "UNAVAILABLE", "reason": "nominal manifest has no pose or route-progress association; geometry was not manufactured"},
        "v1_calibration": steering_calibration(v1_predictions, labels),
        "v2_calibration": steering_calibration(v2_predictions, labels),
        "temporal_shift_v1": temporal_shift_diagnostic(timestamps, v1_predictions, labels, config["temporal_shift_ms"]),
        "sample_count": len(validation_rows),
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    train_tensors = np.stack([preprocess_png(row["image_path"]) for row in train_rows])
    train_features = extract_features(v1, train_tensors, device)
    embeddings_path = artifact_root / "v1_nominal_train_features.npz"
    np.savez_compressed(embeddings_path, features=train_features)
    report["feature_reference"] = {
        "representation": config["feature_layer"], "distance": config["feature_distance"],
        "training_samples": len(train_rows), "path": str(embeddings_path),
        "sha256": sha256_file(embeddings_path),
    }
    report["environment"] = {"device": str(device), "torch": torch.__version__, "parameter_count": PILOTNET_PARAMETER_COUNT}
    report["gate_reached"] = "offline_diagnostics"
    report["result"] = "OFFLINE_PASS"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _driver_config(config: dict[str, Any]) -> DriverConfig:
    names = DriverConfig.__dataclass_fields__.keys()
    values = {name: config[name] for name in names if name != "fixed_speed_mps"}
    values["fixed_speed_mps"] = config["diagnostic_speed_mps"]
    result = DriverConfig(**values)
    result.validate()
    return result


def _collector_config(config: dict[str, Any]) -> CollectorConfig:
    return CollectorConfig(
        expected_world=config["expected_world"],
        required_topics=("/camera/image_raw", "/steering", "/speed", "/cmd_vel", "/odom", "/clock", "/tf", "/tf_static"),
        container_name="physicar-sim", compose_service="sim", container_userdata_root="/opt/physicar/userdata",
        data_relative_root="physicar_e2e/pilotnet_failure_diagnosis_v1/raw", storage_id="mcap",
        recorder_startup_timeout_s=config["recorder_startup_timeout_s"], recorder_shutdown_timeout_s=config["recorder_shutdown_timeout_s"],
        settle_duration_s=0.0, pilot_episode_count=1, minimum_free_bytes=config["minimum_free_bytes"], minimum_camera_messages=2,
    )


def shadow_expert_steering(route, pose: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    projection = route.project((pose["x"], pose["y"]))
    target = route.point_at(projection.s + config["lookahead_m"])
    steering, curvature, target_distance = pure_pursuit_steering(
        (pose["x"], pose["y"]), pose["yaw"], target, config["wheelbase_m"], config["max_steering_rad"],
    )
    route_ahead = route.point_at(projection.s + 0.05)
    route_yaw = math.atan2(route_ahead[1] - projection.point[1], route_ahead[0] - projection.point[0])
    heading_error = math.atan2(math.sin(pose["yaw"] - route_yaw), math.cos(pose["yaw"] - route_yaw))
    return steering, {"route_s_m": projection.s, "cte_m": projection.distance, "signed_cte_m": projection.signed_error,
                      "route_heading_rad": route_yaw, "heading_error_rad": heading_error,
                      "shadow_curvature": curvature, "shadow_target_distance_m": target_distance}


def issue_neural_commands(client, network_steering_rad: float, speed_mps: float) -> None:
    """The only command boundary: shadow expert values are intentionally absent."""
    client.command_steering(clamp_steering_rad(network_steering_rad))
    client.command_speed(float(speed_mps))


def run_count_guard(existing_live_report: Path, run_marker: Path | None = None) -> None:
    marker = run_marker or existing_live_report.with_suffix(".started.json")
    if existing_live_report.exists() or marker.exists():
        raise RuntimeError("maximum_live_runs=1: refusing a second diagnostic run because live evidence already exists")


def _latency(values: Sequence[float]) -> dict[str, Any]:
    scaled = np.asarray(values, dtype=float) * 1000.0
    if not scaled.size:
        return {"count": 0}
    return {"count": int(scaled.size), "mean_ms": float(np.mean(scaled)), "median_ms": float(np.median(scaled)),
            "p95_ms": float(np.percentile(scaled, 95)), "max_ms": float(np.max(scaled))}


def run_live_loop(
    client, model, config: dict[str, Any], initial, frame_root: Path, *, policy_name: str = "PilotNet V1",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route = initial.route
    safety = _driver_config(config)
    tracker = ProgressTracker(route.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s)
    liveness = PoseLivenessMonitor(safety.pose_stale_timeout_s, safety.pose_motion_translation_threshold_m, safety.pose_motion_yaw_threshold_rad)
    telemetry: list[dict[str, Any]] = []
    camera_latencies: list[float] = []
    preprocessing_latencies: list[float] = []
    inference_latencies: list[float] = []
    periods: list[float] = []
    api_failures = liveness_failures = 0
    motion_commanded = False
    failure: str | None = None
    result = "FAIL"
    started = time.monotonic()
    next_tick = started
    previous_tick: float | None = None
    final_pose = initial.pose
    frame_root.mkdir(parents=True, exist_ok=False)
    try:
        while True:
            now = time.monotonic()
            if now - started >= safety.maximum_runtime_s:
                raise RuntimeError("maximum runtime exceeded")
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            if previous_tick is not None:
                periods.append(tick - previous_tick)
            previous_tick = tick
            request_ns = time.time_ns()
            camera_started = time.perf_counter()
            jpeg = client.camera_jpeg(config["camera_path"])
            receive_ns = time.time_ns()
            camera_latencies.append(time.perf_counter() - camera_started)
            frame_path = frame_root / f"frame_{len(telemetry):04d}.jpg"
            frame_path.write_bytes(jpeg)
            pre_started = time.perf_counter()
            tensor = preprocess_live_jpeg(jpeg, roi=(0, 160, 480, 360))
            preprocessing_latencies.append(time.perf_counter() - pre_started)
            infer_started = time.perf_counter()
            normalized = model.predict(tensor)
            inference_latencies.append(time.perf_counter() - infer_started)
            network = clamp_steering_rad(float(steering_normalized_to_rad(normalized)))
            pose = client.pose()
            clock = client.clock()
            final_pose = pose
            try:
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=motion_commanded)
            except RuntimeError:
                liveness_failures += 1
                raise
            expert, geometry = shadow_expert_steering(route, pose, config)
            tracker.update(geometry["route_s_m"])
            boundary_distance = route.track_boundary_distance((pose["x"], pose["y"]))
            if boundary_distance is None:
                raise RuntimeError("track boundary unavailable")
            row = {
                "iteration": len(telemetry), "elapsed_s": tick - started,
                "host_request_time_ns": request_ns, "host_receive_time_ns": receive_ns,
                "sim_time_s": float(clock["sim_time"]), "x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"]),
                **geometry, "unwrapped_progress_m": tracker.unwrapped,
                "network_steering_rad": network, "shadow_expert_steering_rad": expert,
                "network_minus_expert_rad": network - expert, "fixed_speed_mps": config["diagnostic_speed_mps"],
                "camera_acquisition_ms": camera_latencies[-1] * 1000.0,
                "preprocessing_ms": preprocessing_latencies[-1] * 1000.0,
                "inference_ms": inference_latencies[-1] * 1000.0, "frame_path": str(frame_path),
            }
            telemetry.append(row)
            if off_track.update(boundary_distance > safety.off_track_margin_m, time.monotonic()):
                raise RuntimeError(f"sustained off-track: boundary distance {boundary_distance:.3f}m")
            issue_neural_commands(client, network, config["diagnostic_speed_mps"])
            if not motion_commanded:
                motion_commanded = True
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=True)
            distance_to_start = math.dist((pose["x"], pose["y"]), route.points[0])
            if tracker.lap_complete(distance_to_start, safety.start_gate_radius_m, safety.minimum_lap_progress_fraction):
                result = "PASS"
                break
            next_tick += 1.0 / safety.control_frequency_hz
            if next_tick < time.monotonic() - 1.0 / safety.control_frequency_hz:
                next_tick = time.monotonic()
    except Exception as exc:
        failure = str(exc)
        if any(token in failure.lower() for token in ("get ", "post ", "unavailable", "control rejected")):
            api_failures += 1
    finally:
        ended = time.monotonic()
        off_track.finalize(ended)
        stop_errors = client.safe_stop()
        if stop_errors:
            failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
            result = "FAIL"
            api_failures += len(stop_errors)
    ctes = [row["cte_m"] for row in telemetry]
    steering = [row["network_steering_rad"] for row in telemetry]
    return {
        "result": result, "failure": failure, "elapsed_s": time.monotonic() - started,
        "route_length_m": route.length, "route_progress_m": tracker.unwrapped,
        "route_completion_fraction": tracker.unwrapped / route.length,
        "final_distance_to_start_m": math.dist((final_pose["x"], final_pose["y"]), route.points[0]),
        "mean_cte_m": statistics.fmean(ctes) if ctes else None, "max_cte_m": max(ctes, default=None),
        "off_track_events": off_track.event_count, "off_track_total_duration_s": off_track.total_duration_s,
        "mean_abs_steering_rad": statistics.fmean(abs(value) for value in steering) if steering else None,
        "max_abs_steering_rad": max((abs(value) for value in steering), default=None),
        "steering_saturation_fraction": sum(math.isclose(abs(value), MAX_STEERING_RAD, abs_tol=1e-8) for value in steering) / len(steering) if steering else 0.0,
        "mean_abs_command_delta_rad": statistics.fmean(abs(steering[i] - steering[i-1]) for i in range(1, len(steering))) if len(steering) > 1 else 0.0,
        "camera_latency": _latency(camera_latencies), "preprocessing_latency": _latency(preprocessing_latencies),
        "inference_latency": _latency(inference_latencies), "control_period": _latency(periods),
        "control_frequency_hz": 1.0 / statistics.fmean(periods) if periods else 0.0,
        "control_iterations": len(telemetry), "api_failures": api_failures, "liveness_failures": liveness_failures,
        "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors,
        "policy_controlling_vehicle": f"{policy_name} only", "shadow_expert_control_authority": False,
        "neural_observation_fields": ["HTTP camera JPEG"],
        "privileged_fields_outside_model": ["GT pose", "route", "CTE", "sim clock", "track boundaries"],
    }, telemetry


def read_raw_camera_bag(bag_path: Path, extractor_config: dict[str, Any]) -> list[dict[str, Any]]:
    mcap_files = sorted(bag_path.glob("*.mcap"))
    if len(mcap_files) != 1:
        raise RuntimeError(f"expected one MCAP file, found {len(mcap_files)}")
    frames: list[dict[str, Any]] = []
    with mcap_files[0].open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, record, decoded in reader.iter_decoded_messages(topics=["/camera/image_raw"], log_time_order=True):
            image = decode_rgb8_image(decoded, extractor_config)
            frames.append({"record_time_ns": int(record.log_time), "rgb": np.asarray(image, dtype=np.uint8)})
    return frames


def analyze_live(
    *, telemetry: list[dict[str, Any]], config: dict[str, Any], v1_model, device: torch.device,
    reference_features_path: Path, bag_path: Path, extractor_config_path: Path,
) -> dict[str, Any]:
    times = [row["elapsed_s"] for row in telemetry]
    ctes = [row["cte_m"] for row in telemetry]
    windows = detect_divergence_windows(times, ctes, stable_window_s=config["stable_window_s"],
                                         cte_floor_m=config["divergence_cte_floor_m"],
                                         persistence_samples=config["divergence_persistence_samples"])
    window_metrics = {
        "stable_initial": steering_window_metrics(telemetry, windows["stable_indices"]),
        "critical_pre_onset": steering_window_metrics(telemetry, windows["critical_pre_onset_indices"]),
        "late_pre_offtrack": steering_window_metrics(telemetry, windows["late_indices"]),
    }
    net = np.asarray([row["network_steering_rad"] for row in telemetry])
    expert = np.asarray([row["shadow_expert_steering_rad"] for row in telemetry])
    host = np.asarray([row["host_receive_time_ns"] for row in telemetry], dtype=np.int64)
    shift = temporal_shift_diagnostic(host, net, expert, config["temporal_shift_ms"])
    tensors = np.stack([preprocess_live_jpeg(Path(row["frame_path"]).read_bytes()) for row in telemetry])
    live_features = extract_features(v1_model, tensors, device)
    reference = np.load(reference_features_path)["features"]
    distances = nearest_cosine_distances(live_features, reference)
    for row, distance in zip(telemetry, distances, strict=True):
        row["nearest_nominal_feature_cosine_distance"] = float(distance)
    feature_windows = {name: distribution(distances[data["stable_indices"] if name == "stable_initial" else data["critical_pre_onset_indices"] if name == "divergence_onset" else data["late_indices"]]) for name, data in (("stable_initial", windows), ("divergence_onset", windows), ("late_failure", windows))}
    feature_report = {
        "metric": config["feature_distance"], "no_pass_threshold_defined": True, "windows": feature_windows,
        "correlation_with_cte": safe_correlation(distances, np.asarray(ctes)),
        "correlation_with_progress": safe_correlation(distances, np.asarray([row["unwrapped_progress_m"] for row in telemetry])),
        "correlation_with_abs_network_expert_error": safe_correlation(distances, np.abs(net - expert)),
    }
    extractor_config = json.loads(extractor_config_path.read_text(encoding="utf-8"))
    raw_frames = read_raw_camera_bag(bag_path, extractor_config)
    matches = associate_frames(host, [frame["record_time_ns"] for frame in raw_frames], config["frame_association_tolerance_ms"])
    def predictor(tensor: np.ndarray) -> float:
        with torch.no_grad():
            normalized = float(v1_model(torch.from_numpy(tensor[None]).to(device)).cpu().numpy().reshape(-1)[0])
        return float(steering_normalized_to_rad(normalized))
    comparisons = []
    for http_i, raw_i, error_ms in matches:
        comparison = compare_transport_predictions(
            raw_frames[raw_i]["rgb"], Path(telemetry[http_i]["frame_path"]).read_bytes(), predictor
        )
        comparison.update({"http_index": http_i, "raw_index": raw_i, "association_error_ms": error_ms})
        comparisons.append(comparison)
    association_errors = [abs(item[2]) for item in matches]
    prediction_abs = [item["absolute_prediction_difference_rad"] for item in comparisons]
    prediction_signed = [item["signed_prediction_difference_rad"] for item in comparisons]
    pixels = [item["mean_absolute_pixel_difference_0_255"] for item in comparisons]
    index_groups = {
        "stable_initial": set(windows["stable_indices"]),
        "critical_pre_onset": set(windows["critical_pre_onset_indices"]),
        "late_failure": set(windows["late_indices"]),
    }
    transport_windows: dict[str, Any] = {}
    for name, indices in index_groups.items():
        selected = [item for item in comparisons if item["http_index"] in indices]
        transport_windows[name] = {
            "absolute_prediction_difference_rad": distribution([item["absolute_prediction_difference_rad"] for item in selected]),
            "signed_prediction_difference_rad": distribution([item["signed_prediction_difference_rad"] for item in selected]),
            "mean_absolute_pixel_difference_0_255": distribution([item["mean_absolute_pixel_difference_0_255"] for item in selected]),
            "association_absolute_error_ms": distribution([abs(item["association_error_ms"]) for item in selected]),
        }
    transport = {
        "association_method": "one-to-one nearest MCAP record timestamp to HTTP receive wall time",
        "exact_frame_identity_claimed": False, "tolerance_ms": config["frame_association_tolerance_ms"],
        "http_frames": len(telemetry), "raw_frames": len(raw_frames), "matched_pairs": len(matches),
        "association_absolute_error_ms": distribution(association_errors),
        "absolute_prediction_difference_rad": distribution(prediction_abs),
        "signed_prediction_difference_rad": {**distribution(prediction_signed), "mean": float(np.mean(prediction_signed)) if prediction_signed else None},
        "mean_absolute_pixel_difference_0_255": distribution(pixels),
        "windows": transport_windows,
    }
    return {"divergence_detection": windows, "steering_windows": window_metrics,
            "network_shadow_expert_temporal_shift": shift, "feature_distance": feature_report,
            "raw_rgb_vs_http_jpeg": transport}


def classify_hypotheses(offline: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    temporal = analysis["network_shadow_expert_temporal_shift"]
    best_lag = abs(float(temporal["best_shift"]["shift_ms"]))
    lag_improvement = float(temporal["mae_improvement_fraction"])
    if best_lag >= 50 and lag_improvement >= 0.10:
        h1 = "SUPPORTED"
    elif best_lag >= 50 and lag_improvement >= 0.05:
        h1 = "WEAKLY SUPPORTED"
    else:
        h1 = "NOT SUPPORTED"
    v1 = offline["offline"]["v1_calibration"]
    high = v1["magnitude_bins"]["abs_ge_0.25"]
    critical = analysis["steering_windows"]["critical_pre_onset"]
    offline_under = v1["regression_slope"] < 0.90 or high.get("absolute_magnitude_ratio", 1.0) < 0.90
    live_under = critical.get("corrective_magnitude_ratio") is not None and critical["corrective_magnitude_ratio"] < 0.85
    h2 = "SUPPORTED" if offline_under and live_under else "WEAKLY SUPPORTED" if offline_under or live_under else "NOT SUPPORTED"
    transport = analysis["raw_rgb_vs_http_jpeg"]
    transport_mae = transport["absolute_prediction_difference_rad"]["mean"]
    nominal_mae = float(v1["mae_rad"])
    association_median = transport["association_absolute_error_ms"]["median"]
    late_transport = transport.get("windows", {}).get("late_failure", {}).get("absolute_prediction_difference_rad", {}).get("median")
    if transport_mae is None or transport["matched_pairs"] < 3:
        h3 = "INCONCLUSIVE"
    elif association_median is not None and association_median > 1000.0 / (2.0 * 15.0):
        h3 = "INCONCLUSIVE"
    elif transport_mae > nominal_mae or (late_transport is not None and late_transport > nominal_mae):
        h3 = "WEAKLY SUPPORTED"
    else:
        h3 = "NOT SUPPORTED"
    feature = analysis["feature_distance"]
    early = feature["windows"]["stable_initial"]["median"]
    late = feature["windows"]["late_failure"]["median"]
    corr = feature["correlation_with_cte"]
    if early is None or late is None:
        h4 = "INCONCLUSIVE"
    elif late > early and corr is not None and corr > 0.30:
        h4 = "SUPPORTED"
    elif late > early or (corr is not None and corr > 0.30):
        h4 = "WEAKLY SUPPORTED"
    else:
        h4 = "NOT SUPPORTED"
    h5 = "NOT SUPPORTED" if offline["preprocessing_audit"]["result"] == "PASS" else "SUPPORTED"
    evidence = {
        "H1_temporal_phase_mismatch": {"classification": h1, "best_lag_ms": temporal["best_shift"]["shift_ms"], "mae_improvement_fraction": lag_improvement},
        "H2_regression_to_mean_under_correction": {"classification": h2, "offline_slope": v1["regression_slope"], "offline_high_steering_ratio": high.get("absolute_magnitude_ratio"), "critical_live_ratio": critical.get("corrective_magnitude_ratio")},
        "H3_http_jpeg_vs_raw_rgb": {"classification": h3, "matched_pairs": transport["matched_pairs"], "mean_abs_prediction_difference_rad": transport_mae, "median_association_error_ms": association_median, "late_median_abs_prediction_difference_rad": late_transport, "nominal_validation_mae_rad_for_context": nominal_mae, "confound": "Near-matched transport frames are not proven identical; temporal scene change can inflate the difference."},
        "H4_on_policy_distribution_shift": {"classification": h4, "stable_median_distance": early, "late_median_distance": late, "distance_cte_correlation": corr, "threshold_note": "No feature-distance PASS threshold was defined; classification uses relative evolution and correlation."},
        "H5_preprocessing_runtime_mismatch": {"classification": h5, "audit_result": offline["preprocessing_audit"]["result"]},
    }
    supported = {key for key, value in evidence.items() if value["classification"] in {"SUPPORTED", "WEAKLY SUPPORTED"}}
    if "H4_on_policy_distribution_shift" in supported:
        recommendation = "Collect a tightly scoped on-policy expert-labeling/DAgger dataset from actual V1 deviation states; do not collect blind nominal or arbitrary-anchor recovery laps."
    elif "H1_temporal_phase_mismatch" in supported:
        recommendation = "Run one controlled label-alignment A/B experiment; do not change architecture or collect blind laps."
    elif "H3_http_jpeg_vs_raw_rgb" in supported:
        recommendation = "Match training and live image transport in a controlled A/B, or apply transport-specific augmentation; do not collect blind laps."
    elif "H2_regression_to_mean_under_correction" in supported:
        recommendation = "Run a controlled loss/sampling/target-distribution A/B using existing data; do not collect blind recovery anchors."
    else:
        recommendation = "Do not collect more data yet; first repeat the offline/runtime instrumentation as a non-driving replay study because no tested hypothesis is supported."
    return {"hypotheses": evidence, "exact_next_intervention": recommendation}


def run_live(
    *, config_path: Path, offline_result_path: Path, v1_checkpoint: Path, v1_onnx: Path,
    sim_root: Path, artifact_root: Path, live_result_path: Path, repo_root: Path,
) -> dict[str, Any]:
    run_marker = live_result_path.with_suffix(".started.json")
    run_count_guard(live_result_path, run_marker)
    config = load_config(config_path)
    offline = json.loads(offline_result_path.read_text(encoding="utf-8"))
    if offline.get("result") != "OFFLINE_PASS" or offline.get("preprocessing_audit", {}).get("result") != "PASS":
        raise RuntimeError("offline diagnostic gates have not passed")
    if sha256_file(v1_checkpoint) != config["v1_checkpoint_sha256"] or sha256_file(v1_onnx) != config["v1_onnx_sha256"]:
        raise RuntimeError("canonical V1 artifact hash mismatch")
    client = SimClient(config["base_url"], config["api_timeout_s"])
    model = CameraOnlyOnnxModel(v1_onnx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_model = load_model(v1_checkpoint, device)
    collector = _collector_config(config)
    backend = DockerRosBackend(collector, sim_root)
    handle: RecorderHandle | None = None
    report: dict[str, Any] = {"version": VERSION, "generated_utc": utc_now(), "result": "FAIL", "live_run_count": 0}
    try:
        initial_stop = client.safe_stop()
        if initial_stop:
            raise RuntimeError("initial safe-stop failed: " + "; ".join(initial_stop))
        initial = wait_after_reset(client, _driver_config(config), False)
        camera = client.camera_jpeg(config["camera_path"])
        with Image.open(io.BytesIO(camera)) as image:
            if image.size != (480, 360) or image.format != "JPEG":
                raise RuntimeError(f"HTTP camera contract mismatch: {image.format} {image.size}")
        topics = backend.preflight(config["diagnostic_ros_topics"])
        if shutil.disk_usage(backend.host_userdata_root).free < config["minimum_free_bytes"]:
            raise RuntimeError("insufficient external userdata free space")
        report["preflight"] = {"result": "PASS", "world": initial.world, "cones": initial.cone_count,
                               "route_points": initial.route_points, "route_length_m": initial.route.length,
                               "camera_dimensions": [480, 360], "diagnostic_topics": topics,
                               "v1_checkpoint_sha256": sha256_file(v1_checkpoint), "v1_onnx_sha256": sha256_file(v1_onnx)}
        handle = backend.start_recorder("diagnostic_run_001", config["diagnostic_ros_topics"])
        report["live_run_count"] = 1
        live_result_path.parent.mkdir(parents=True, exist_ok=True)
        run_marker.write_text(json.dumps({"version": VERSION, "started_utc": utc_now(), "live_run_count": 1,
                                         "status": "LIVE_RUN_STARTED_DO_NOT_RETRY"}, indent=2) + "\n", encoding="utf-8")
        frame_root = artifact_root / "http_frames"
        metrics, telemetry = run_live_loop(client, model, config, initial, frame_root)
        report["live_run"] = metrics
        live_result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        stop_errors = client.safe_stop()
        report["final_safe_stop_success"] = not stop_errors
        report["final_safe_stop_errors"] = stop_errors
        if handle is not None:
            stopped = backend.stop_recorder(handle)
            report["recorder_stop"] = asdict(stopped)
            if not stopped.graceful or stopped.orphaned:
                live_result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                raise RuntimeError(f"diagnostic recorder did not finalize gracefully: {stopped}")
            info = backend.bag_info(handle)
            verify_bag(info, config["diagnostic_ros_topics"], 2)
            report["diagnostic_bag"] = {"path": str(handle.host_bag_path), "duration_s": info.duration_s,
                                        "size_bytes": directory_size(handle.host_bag_path), "topic_counts": info.topic_counts}
    if report.get("live_run_count") != 1 or handle is None:
        raise RuntimeError("live diagnostic did not execute exactly once")
    telemetry_path = artifact_root / "telemetry.json"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    report["analysis"] = analyze_live(
        telemetry=telemetry, config=config, v1_model=torch_model, device=device,
        reference_features_path=Path(offline["feature_reference"]["path"]), bag_path=handle.host_bag_path,
        extractor_config_path=repo_root / "configs/dataset_extractor_v1.json",
    )
    report["interpretation"] = classify_hypotheses(offline, report["analysis"])
    report["external_artifacts"] = {"root": str(artifact_root), "telemetry": str(telemetry_path),
                                    "http_frames": str(artifact_root / "http_frames"), "diagnostic_bag": str(handle.host_bag_path)}
    report["result"] = "DIAGNOSIS_COMPLETE"
    live_result_path.parent.mkdir(parents=True, exist_ok=True)
    live_result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path)
    parser.add_argument("--v1-onnx", type=Path)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--offline-result", type=Path)
    parser.add_argument("--sim-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sum((args.offline, args.live, args.reanalyze)) != 1:
        raise SystemExit("choose exactly one of --offline, --live, or --reanalyze")
    repo_root = Path(__file__).resolve().parents[2]
    if args.offline:
        if args.dataset_root is None or args.v2_checkpoint is None:
            raise SystemExit("--offline requires --dataset-root and --v2-checkpoint")
        result = run_offline(config_path=args.config, dataset_root=args.dataset_root,
                             v1_checkpoint=args.v1_checkpoint, v2_checkpoint=args.v2_checkpoint,
                             artifact_root=args.artifact_root, result_path=args.result, repo_root=repo_root)
    elif args.live:
        if args.v1_onnx is None or args.offline_result is None or args.sim_root is None:
            raise SystemExit("--live requires --v1-onnx, --offline-result, and --sim-root")
        result = run_live(config_path=args.config, offline_result_path=args.offline_result,
                          v1_checkpoint=args.v1_checkpoint, v1_onnx=args.v1_onnx,
                          sim_root=args.sim_root, artifact_root=args.artifact_root,
                          live_result_path=args.result, repo_root=repo_root)
    else:
        if args.offline_result is None:
            raise SystemExit("--reanalyze requires --offline-result")
        prior = json.loads(args.result.read_text(encoding="utf-8"))
        if prior.get("live_run_count") != 1 or "diagnostic_bag" not in prior:
            raise SystemExit("existing result does not contain exactly one completed live run")
        config = load_config(args.config)
        offline = json.loads(args.offline_result.read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = load_model(args.v1_checkpoint, device)
        telemetry_path = Path(prior["external_artifacts"]["telemetry"])
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        prior["analysis"] = analyze_live(
            telemetry=telemetry, config=config, v1_model=torch_model, device=device,
            reference_features_path=Path(offline["feature_reference"]["path"]),
            bag_path=Path(prior["diagnostic_bag"]["path"]),
            extractor_config_path=repo_root / "configs/dataset_extractor_v1.json",
        )
        prior["interpretation"] = classify_hypotheses(offline, prior["analysis"])
        args.result.write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = prior
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] in {"OFFLINE_PASS", "DIAGNOSIS_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
