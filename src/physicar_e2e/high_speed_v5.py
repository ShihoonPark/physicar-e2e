"""Integrated gated High-Speed Expert dataset and PilotNet V5 pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch

from .dataset_extractor import (
    aggregate_summary, canonical_json_bytes, extract_episode, load_config as load_extractor_config,
    prepare_output_root, write_manifest,
)
from .expert_driver import DriverConfig, run_driver, wait_after_reset
from .pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from .pilotnet_inference import CameraOnlyOnnxModel, InferenceConfig, live_camera_preflight, run_smoke
from .pilotnet_training import (
    GateFailure, benchmark_host_cpu, error_metrics, export_onnx, load_config as load_training_config,
    predict_rows, read_episode_rows, sha256_file, tiny_overfit_sanity, train_baseline,
    validate_onnx_equivalence,
)
from .pilotnet_v4_repeatability import clock_health_preflight, verify_static_environment
from .rosbag_collector import (
    CollectorConfig, CollectorError, DockerRosBackend, collect_sequence, git_commit, summarize,
    verify_environment,
)
from .sim_client import SimClient


EPISODES = tuple(f"episode_{number:03d}" for number in range(1, 13))
TRAIN_EPISODES = EPISODES[:8]
VALIDATION_EPISODES = EPISODES[8:10]
HOLDOUT_EPISODES = EPISODES[10:12]
REQUIRED_TOPICS = ("/camera/image_raw", "/steering", "/speed", "/cmd_vel", "/odom", "/clock", "/tf", "/tf_static")
SPEED_MPS = 1.80
LOOKAHEAD_M = 0.90
MAX_LIVE_ATTEMPTS = 5
TARGET_POLICY_PASSES = 3
EXPECTED_V4_ONNX_SHA256 = "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
EXPECTED_V4_CONFIG_SHA256 = "5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_frozen_expert(repo: Path, expert_path: Path) -> DriverConfig:
    expert = DriverConfig.load(expert_path)
    canonical = DriverConfig.load(repo / "configs/expert_driver_v1.json")
    changed = [name for name in canonical.__dataclass_fields__ if getattr(canonical, name) != getattr(expert, name)]
    if changed != ["fixed_speed_mps", "lookahead_m"]:
        raise GateFailure(f"High-Speed Expert differs unexpectedly: {changed}")
    expected = (expert.fixed_speed_mps, expert.lookahead_m, expert.control_frequency_hz,
                expert.max_steering_rad, expert.wheelbase_m)
    if expected != (1.8, 0.9, 15.0, 0.349066, 0.18):
        raise GateFailure(f"High-Speed Expert frozen values mismatch: {expected}")
    repeat = json.loads((repo / "results/expert_speed_1p8_repeatability_v1/summary.json").read_text())
    if repeat.get("result") != "PASS" or repeat.get("aggregate", {}).get("expert_success") != "3/3":
        raise GateFailure("High-Speed Expert repeatability evidence is not 3/3 PASS")
    return expert


def collection_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_dir = repo / "results/high_speed_collection_v1"
    summary_path = result_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"collection evidence already exists: {summary_path}")
    collector = CollectorConfig.load(repo / "configs/high_speed_rosbag_v1.json")
    expert_path = repo / "configs/high_speed_expert_v1.json"
    expert = verify_frozen_expert(repo, expert_path)
    if collector.pilot_episode_count != 12 or collector.required_topics != REQUIRED_TOPICS:
        raise GateFailure("collector does not request exactly 12 episodes and the canonical eight topics")
    environment = verify_environment(repo, sim_root)
    backend = DockerRosBackend(collector, sim_root)
    topic_types = backend.preflight(collector.required_topics)
    client = SimClient(expert.base_url, expert.api_timeout_s)
    try:
        if errors := client.safe_stop():
            raise CollectorError("initial safe stop failed: " + "; ".join(errors))

        def gated_driver(live_client, live_config, initial):
            static = verify_static_environment(initial)
            clock = clock_health_preflight(live_client)
            if clock["result"] != "PASS":
                raise CollectorError(str(clock.get("failure_reason", "clock gate failed")))
            metrics = run_driver(live_client, live_config, initial)
            metrics["environment_preflight"] = static
            metrics["clock_health_preflight"] = clock
            return metrics

        episodes, summary = collect_sequence(
            12, collector, expert, expert_path.resolve(), git_commit(repo), backend, client, result_dir,
            driver=gated_driver,
        )
        summary.update({
            "version": "high_speed_collection_v1", "expert_config": asdict(expert),
            "expert_config_sha256": sha256_file(expert_path), "raw_root": str(backend.host_data_root),
            "topic_types": topic_types, "environment_verification": environment,
            "all_required_topics": list(REQUIRED_TOPICS),
        })
        write_json(summary_path, summary)
        if summary["result"] != "PASS" or len(episodes) != 12:
            raise GateFailure("12/12 collection gate failed")
        return summary
    finally:
        errors = client.safe_stop()
        if errors:
            raise GateFailure("collection final safe stop failed: " + "; ".join(errors))


def extraction_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    collection = json.loads((repo / "results/high_speed_collection_v1/summary.json").read_text())
    if collection.get("result") != "PASS" or collection.get("passed_episode_count") != 12:
        raise GateFailure("collection gate is not 12/12 PASS")
    input_root = sim_root / "userdata/physicar_e2e/high_speed_v1/raw"
    output_root = sim_root / "userdata/physicar_e2e/high_speed_v1/dataset"
    config_path = repo / "configs/high_speed_dataset_v1.json"
    config = load_extractor_config(config_path)
    prepare_output_root(output_root, False)
    (output_root / "manifests").mkdir(exist_ok=True)
    (output_root / "previews").mkdir(exist_ok=True)
    config_sha = sha256_file(config_path)
    episode_metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for episode_id in EPISODES:
        bag_root = input_root / episode_id / "bag"
        mcap_files = sorted(bag_root.glob("*.mcap"))
        if len(mcap_files) != 1:
            raise GateFailure(f"{episode_id}: expected one MCAP, got {len(mcap_files)}")
        collector_path = repo / "results/high_speed_collection_v1" / f"{episode_id}.json"
        metrics, rows = extract_episode(
            episode_id=episode_id, mcap_path=mcap_files[0], collector_metadata_path=collector_path,
            dataset_root=output_root, config=config, config_sha256=config_sha,
            source_path_identity=mcap_files[0].relative_to(input_root).as_posix(),
            collector_metadata_identity=collector_path.relative_to(repo).as_posix(),
        )
        episode_metrics.append(metrics); all_rows.extend(rows)
    write_manifest(output_root / "manifest.csv", all_rows)
    metadata = aggregate_summary(episode_metrics, all_rows, output_root, config, config_sha)
    metadata.pop("pilot_success_gate", None)
    metadata.update({"version": "high_speed_dataset_v1", "episode_count": 12,
                     "episode_ids": list(EPISODES), "input_root": str(input_root),
                     "dataset_root": str(output_root), "config": config, "episode_metrics": episode_metrics,
                     "split": {"training": list(TRAIN_EPISODES), "validation": list(VALIDATION_EPISODES),
                               "holdout": list(HOLDOUT_EPISODES)}})
    metadata["high_speed_quality_gate"] = {
        "all_twelve_bags_readable": len(episode_metrics) == 12,
        "nonzero_samples_each": all(item["counts"]["accepted_camera_samples"] > 0 for item in episode_metrics),
        "future_label_violations_zero": all(item["synchronization"]["future_label_violations"] == 0 for item in episode_metrics),
        "age_gates_satisfied": all(
            item["synchronization"]["steering_age_ms"]["max"] <= config["maximum_steering_age_s"] * 1000
            and item["synchronization"]["speed_age_ms"]["max"] <= config["maximum_speed_age_s"] * 1000
            for item in episode_metrics
        ),
        "episode_level_separation_preserved": len({row["episode_id"] for row in all_rows}) == 12,
    }
    if metadata["synchronization"]["future_label_violations"] != 0:
        raise GateFailure("future steering labels detected")
    if any(item["counts"]["rejection_by_reason"]["image_decode_error"] for item in episode_metrics):
        raise GateFailure("image decode failure detected")
    if len({row["episode_id"] for row in all_rows}) != 12:
        raise GateFailure("episode coverage is incomplete")
    (output_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
    result_dir = repo / "results/high_speed_dataset_v1"; result_dir.mkdir(parents=True, exist_ok=True)
    for item in episode_metrics:
        write_json(result_dir / f"{item['episode_id']}.json", item)
    compact = {key: value for key, value in metadata.items() if key not in ("config", "episode_metrics")}
    compact["dataset_metadata_sha256"] = sha256_file(output_root / "dataset_metadata.json")
    write_json(result_dir / "summary.json", compact)
    return compact


def validate_v5_dataset(dataset_root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = tuple(config["train_episodes"]); validation = tuple(config["validation_episodes"]); holdout = tuple(config["holdout_episodes"])
    if (train, validation, holdout) != (TRAIN_EPISODES, VALIDATION_EPISODES, HOLDOUT_EPISODES):
        raise GateFailure("V5 split is not the frozen 8/2/2 episode split")
    if set(train) & set(validation) or set(train) & set(holdout) or set(validation) & set(holdout):
        raise GateFailure("episode leakage across V5 splits")
    groups = [read_episode_rows(dataset_root, episodes) for episodes in (train, validation, holdout)]
    seen: set[Path] = set()
    hashes: dict[str, str] = {}
    for episode in EPISODES:
        manifest = dataset_root / "manifests" / f"{episode}.csv"
        hashes[episode] = sha256_file(manifest)
    for row in [item for group in groups for item in group]:
        path = row["image_path"]
        if path in seen: raise GateFailure(f"duplicate/leaked frame path: {path}")
        seen.add(path)
        with Image.open(path) as image:
            if image.size != (200, 66) or image.mode != "RGB":
                raise GateFailure(f"invalid V5 image contract: {path}")
    return groups[0], groups[1], groups[2], {
        "result": "PASS", "training_samples": len(groups[0]), "validation_samples": len(groups[1]),
        "holdout_samples": len(groups[2]), "episode_level_separation": True,
        "manifest_sha256_by_episode": hashes,
    }


def calibration(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    overall = error_metrics(predictions, labels)
    if len(labels) > 1 and float(np.std(labels)) > 0 and float(np.std(predictions)) > 0:
        correlation = float(np.corrcoef(labels, predictions)[0, 1])
        slope = float(np.polyfit(labels, predictions, 1)[0])
    else:
        correlation = 0.0; slope = 0.0
    bins = [("abs_lt_0p05", 0.0, 0.05), ("abs_0p05_0p15", 0.05, 0.15),
            ("abs_0p15_0p25", 0.15, 0.25), ("abs_ge_0p25", 0.25, math.inf)]
    magnitude_bins: dict[str, Any] = {}
    absolute_labels = np.abs(labels)
    for index, (name, low, high) in enumerate(bins):
        mask = (absolute_labels >= low) & (absolute_labels < high if index < len(bins) - 1 else absolute_labels >= low)
        gt = absolute_labels[mask]; pred = np.abs(predictions[mask])
        magnitude_bins[name] = {"count": int(mask.sum()), "mae_rad": float(np.mean(np.abs(predictions[mask] - labels[mask]))) if mask.any() else None,
                                "gt_mean_magnitude_rad": float(np.mean(gt)) if mask.any() else None,
                                "predicted_mean_magnitude_rad": float(np.mean(pred)) if mask.any() else None,
                                "magnitude_ratio": float(np.mean(pred) / np.mean(gt)) if mask.any() and float(np.mean(gt)) > 0 else None}
    return {**overall, "correlation": correlation, "prediction_vs_gt_slope": slope, "magnitude_bins": magnitude_bins}


def training_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    dataset_summary = json.loads((repo / "results/high_speed_dataset_v1/summary.json").read_text())
    if dataset_summary.get("result") != "PASS" or dataset_summary.get("episode_count") != 12:
        raise GateFailure("dataset quality gate is not 12-episode PASS")
    dataset_root = sim_root / "userdata/physicar_e2e/high_speed_v1/dataset"
    artifact_root = sim_root / "userdata/physicar_e2e/high_speed_v1/v5"
    if artifact_root.exists(): raise FileExistsError(f"V5 artifact root already exists: {artifact_root}")
    config_path = repo / "configs/pilotnet_training_v5_high_speed.json"
    config = load_training_config(config_path)
    if config.get("initialization") != "from_scratch": raise GateFailure("V5 must train from scratch")
    train_rows, validation_rows, holdout_rows, integrity = validate_v5_dataset(dataset_root, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {"version": "pilotnet_training_v5_high_speed", "generated_utc": utc_now(),
                              "result": "FAIL", "architecture": {"input_shape": [3, 66, 200],
                              "parameter_count": PILOTNET_PARAMETER_COUNT}, "dataset_integrity": integrity,
                              "training_from_scratch": True, "device": str(device)}
    result_path = repo / "results/pilotnet_training_v5_high_speed/summary.json"
    try:
        report["tiny_overfit"] = tiny_overfit_sanity(train_rows, config, device)
        checkpoint = artifact_root / "checkpoints/pilotnet_v5_high_speed_best.pt"
        model, training, history = train_baseline(train_rows, validation_rows, config, device, checkpoint)
        report["training"] = training; report["epochs"] = history
        val_predictions, val_labels = predict_rows(model, validation_rows, config, device)
        hold_predictions, hold_labels = predict_rows(model, holdout_rows, config, device)
        report["offline_validation"] = calibration(val_predictions, val_labels)
        report["offline_holdout"] = calibration(hold_predictions, hold_labels)
        onnx_path = artifact_root / "onnx/pilotnet_v5_high_speed.onnx"
        export_onnx(model, onnx_path, config)
        report["onnx_equivalence"] = validate_onnx_equivalence(model, [*validation_rows, *holdout_rows], onnx_path, config)
        report["host_cpu_benchmark"] = benchmark_host_cpu(onnx_path, validation_rows[0], config)
        report["artifacts"] = {"checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                                                "sha256": sha256_file(checkpoint)},
                               "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size,
                                        "sha256": sha256_file(onnx_path)}}
        report["result"] = "PASS"
        return report
    finally:
        write_json(result_path, report)


def load_v5_inference(repo: Path) -> InferenceConfig:
    payload = json.loads((repo / "configs/pilotnet_inference_v5_high_speed.json").read_text())
    if payload.get("version") != "pilotnet_inference_v5_high_speed" or payload.get("smoke_speeds_mps") != [1.8, 1.8, 1.8]:
        raise GateFailure("V5 live config speed/run contract mismatch")
    canonical = json.loads((repo / "configs/pilotnet_inference_v4_dagger.json").read_text())
    if sha256_file(repo / "configs/pilotnet_inference_v4_dagger.json") != EXPECTED_V4_CONFIG_SHA256:
        raise GateFailure("canonical V4 config changed")
    permitted = {"version", "smoke_speeds_mps"}
    differences = {key for key in set(payload) | set(canonical) if payload.get(key) != canonical.get(key)}
    if differences != permitted:
        raise GateFailure(f"V5 inference differs unexpectedly from V4: {sorted(differences)}")
    return InferenceConfig(payload)


def live_preflight(client: SimClient, config: InferenceConfig) -> tuple[Any, dict[str, Any]]:
    initial = wait_after_reset(client, config.safety_config(SPEED_MPS), False)
    environment = verify_static_environment(initial)
    camera = live_camera_preflight(client, config)
    clock = clock_health_preflight(client)
    if clock["result"] != "PASS": raise GateFailure("clock health preflight failed")
    return initial, {"result": "PASS", "environment": environment, "camera": camera, "clock_health": clock}


def classify_policy_run(run: dict[str, Any]) -> str:
    if run.get("result") == "PASS": return "POLICY_PASS"
    if run.get("api_failures") or run.get("liveness_failures") or not run.get("safe_stop_success", False):
        return "INFRA_FAIL"
    return "POLICY_FAIL"


def run_live_attempts(
    client: SimClient, model: CameraOnlyOnnxModel, config: InferenceConfig, result_dir: Path,
    *, preflight_one: Callable[[SimClient, InferenceConfig], tuple[Any, dict[str, Any]]] = live_preflight,
    run_one: Callable[..., dict[str, Any]] = run_smoke,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []; passes = 0
    for number in range(1, MAX_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight_result = preflight_one(client, config)
            run = run_one(client, model, config, initial, SPEED_MPS)
            attempt = {"attempt_number": number, "classification": classify_policy_run(run),
                       "preflight": preflight_result, "run": run}
        except Exception as exc:
            errors = client.safe_stop()
            attempt = {"attempt_number": number, "classification": "INFRA_FAIL", "run": None,
                       "preflight": {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                                     "safe_stop_success": not errors, "safe_stop_errors": errors}}
        attempts.append(attempt); write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "POLICY_FAIL": return attempts, "FAIL"
        if attempt["classification"] == "POLICY_PASS":
            passes += 1
            if passes == TARGET_POLICY_PASSES: return attempts, "PASS"
    return attempts, "INCONCLUSIVE"


def live_preflight_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    training = json.loads((repo / "results/pilotnet_training_v5_high_speed/summary.json").read_text())
    if training.get("result") != "PASS" or training.get("onnx_equivalence", {}).get("result") != "PASS":
        raise GateFailure("training/ONNX gate is not PASS")
    onnx_path = Path(training["artifacts"]["onnx"]["path"])
    if sha256_file(onnx_path) != training["artifacts"]["onnx"]["sha256"]:
        raise GateFailure("V5 ONNX identity mismatch")
    config = load_v5_inference(repo)
    result_dir = repo / "results/pilotnet_e2e_v5_high_speed"
    result_path = result_dir / "preflight.json"
    if result_path.exists() or (result_dir / "experiment.started.json").exists():
        raise FileExistsError("refusing to repeat V5 live preflight")
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
    report: dict[str, Any] = {"version": "pilotnet_e2e_v5_high_speed_preflight",
                              "generated_utc": utc_now(), "result": "FAIL",
                              "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop():
            raise GateFailure("initial safe stop failed: " + "; ".join(errors))
        _, checks = live_preflight(client, config)
        report.update(checks)
        return report
    finally:
        errors = client.safe_stop()
        report["safe_stop_success"] = not errors
        report["safe_stop_errors"] = errors
        if errors:
            report["result"] = "FAIL"
        write_json(result_path, report)


def live_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    training = json.loads((repo / "results/pilotnet_training_v5_high_speed/summary.json").read_text())
    if training.get("result") != "PASS" or training.get("onnx_equivalence", {}).get("result") != "PASS":
        raise GateFailure("training/ONNX gate is not PASS")
    onnx_path = Path(training["artifacts"]["onnx"]["path"])
    if sha256_file(onnx_path) != training["artifacts"]["onnx"]["sha256"]:
        raise GateFailure("V5 ONNX identity mismatch")
    config = load_v5_inference(repo)
    result_dir = repo / "results/pilotnet_e2e_v5_high_speed"
    marker = result_dir / "experiment.started.json"
    if marker.exists() or (result_dir / "summary.json").exists():
        raise FileExistsError("refusing to repeat V5 live validation")
    model = CameraOnlyOnnxModel(onnx_path)
    client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
    report: dict[str, Any] = {"version": "pilotnet_e2e_v5_high_speed", "generated_utc": utc_now(),
                              "result": "INCONCLUSIVE", "onnx": training["artifacts"]["onnx"]}
    try:
        if errors := client.safe_stop(): raise GateFailure("initial safe stop failed: " + "; ".join(errors))
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(marker, {"status": "V5_LIVE_STARTED", "maximum_attempts": MAX_LIVE_ATTEMPTS,
                            "target_policy_passes": TARGET_POLICY_PASSES, "started_utc": utc_now()})
        attempts, result = run_live_attempts(client, model, config, result_dir)
        passes = sum(item["classification"] == "POLICY_PASS" for item in attempts)
        report["result"] = result
        report.update({"attempts": attempts, "policy_pass_count": passes,
                       "infrastructure_failure_count": sum(x["classification"] == "INFRA_FAIL" for x in attempts)})
        return report
    finally:
        errors = client.safe_stop(); report["final_safe_stop_success"] = not errors; report["final_safe_stop_errors"] = errors
        if errors: report["result"] = "INCONCLUSIVE"
        write_json(result_dir / "summary.json", report)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "extract", "train", "live-preflight", "live"))
    parser.add_argument("--sim-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    try:
        if args.stage == "collect": result = collection_stage(repo, args.sim_root.resolve())
        elif args.stage == "extract": result = extraction_stage(repo, args.sim_root.resolve())
        elif args.stage == "train": result = training_stage(repo, args.sim_root.resolve())
        elif args.stage == "live-preflight": result = live_preflight_stage(repo, args.sim_root.resolve())
        else: result = live_stage(repo, args.sim_root.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("result") == "PASS" else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
