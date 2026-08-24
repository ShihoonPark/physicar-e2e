"""Train PilotNet V3 from scratch on nominal plus frozen rollout-A DAgger data."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .pilotnet import PILOTNET_PARAMETER_COUNT
from .pilotnet_failure_diagnosis import steering_calibration
from .pilotnet_recovery_training import load_checkpoint
from .pilotnet_training import (
    GateFailure, create_plots, error_metrics, export_onnx, load_config, predict_rows,
    read_episode_rows, sha256_file, train_baseline, validate_dataset_integrity,
    validate_offline, validate_onnx_equivalence,
)


def read_dagger_rows(root: Path, rollout_id: str) -> list[dict[str, Any]]:
    manifest = root / "manifests" / f"{rollout_id}.csv"
    if not manifest.is_file():
        raise GateFailure(f"missing DAgger manifest {manifest}")
    rows: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            if raw["episode_id"] != rollout_id:
                raise GateFailure("DAgger manifest episode mismatch")
            image = root / raw["image_path"]
            steering = float(raw["steering_rad"])
            label_time = int(raw["expert_label_time_ns"])
            camera_time = int(raw["camera_header_time_ns"])
            if not image.is_file() or not math.isfinite(steering):
                raise GateFailure("DAgger image/label integrity failure")
            if label_time > camera_time:
                raise GateFailure("future DAgger expert label")
            rows.append({
                "episode_id": rollout_id, "sample_index": int(raw["sample_index"]),
                "image_path": image, "steering_rad": steering, "window_role": raw["window_role"],
                "source_mcap_sha256": raw["source_mcap_sha256"],
            })
    if not rows:
        raise GateFailure("empty DAgger rollout")
    return rows


def dagger_split(metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    training = str(config["dagger_training_rollout"])
    holdout = str(config["dagger_holdout_rollout"])
    roles = {item["rollout_id"]: item["role"] for item in metadata.get("episodes", [])}
    if training != "dagger_rollout_A" or holdout != "dagger_rollout_B":
        raise GateFailure("frozen A/B split changed")
    if roles.get(training) != "training" or roles.get(holdout) != "holdout" or training == holdout:
        raise GateFailure("DAgger A/B roles are incomplete or overlapping")
    return {"training": training, "holdout": holdout}


def prediction_metrics(model, rows, config, device) -> dict[str, Any]:
    predictions, labels = predict_rows(model, rows, config, device)
    base = error_metrics(predictions, labels)
    label_magnitude = float(np.mean(np.abs(labels)))
    base["corrective_magnitude_ratio"] = float(np.mean(np.abs(predictions)) / label_magnitude) if label_magnitude else None
    return base


def grouped_on_policy_metrics(model, rows, config, device) -> dict[str, Any]:
    predictions, labels = predict_rows(model, rows, config, device)
    result: dict[str, Any] = {"overall": error_metrics(predictions, labels), "windows": {}}
    label_magnitude = float(np.mean(np.abs(labels)))
    result["overall"]["corrective_magnitude_ratio"] = float(np.mean(np.abs(predictions)) / label_magnitude) if label_magnitude else None
    for role in ("pre_divergence", "divergence", "late_failure"):
        mask = np.asarray([row["window_role"] == role for row in rows], dtype=bool)
        if not np.any(mask):
            result["windows"][role] = {"sample_count": 0}
            continue
        metric = error_metrics(predictions[mask], labels[mask])
        magnitude = float(np.mean(np.abs(labels[mask])))
        metric["corrective_magnitude_ratio"] = float(np.mean(np.abs(predictions[mask])) / magnitude) if magnitude else None
        result["windows"][role] = metric
    return result


def run_training(
    *, config_path: Path, nominal_root: Path, dagger_root: Path, v1_checkpoint: Path,
    v2_checkpoint: Path, artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("version") != "pilotnet_training_v3_dagger":
        raise GateFailure("unexpected V3 training config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "version": "pilotnet_training_v3_dagger", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL", "gate_reached": "environment",
        "architecture": {"parameter_count": PILOTNET_PARAMETER_COUNT, "identical_to_v1_v2": True},
        "environment": {"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "training_config_sha256": sha256_file(config_path),
    }
    try:
        nominal = validate_dataset_integrity(nominal_root, config)
        nominal_train = nominal.pop("train_rows")
        nominal_validation = nominal.pop("validation_rows")
        metadata_path = dagger_root / "dataset_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("result") != "PASS" or metadata.get("future_label_violations") != 0:
            raise GateFailure("DAgger dataset quality metadata is not PASS")
        split = dagger_split(metadata, config)
        dagger_train = read_dagger_rows(dagger_root, split["training"])
        dagger_holdout = read_dagger_rows(dagger_root, split["holdout"])
        if {row["source_mcap_sha256"] for row in dagger_train} & {row["source_mcap_sha256"] for row in dagger_holdout}:
            raise GateFailure("rollout A/B source leakage detected")
        combined = [*nominal_train, *dagger_train]
        report["dataset"] = {
            "nominal": nominal, "dagger_metadata_sha256": sha256_file(metadata_path),
            "nominal_training_samples": len(nominal_train), "dagger_training_samples": len(dagger_train),
            "combined_training_samples": len(combined), "nominal_validation_samples": len(nominal_validation),
            "dagger_holdout_samples": len(dagger_holdout), "training_rollout": split["training"],
            "holdout_rollout": split["holdout"], "holdout_absent_from_training": True,
            "recovery_v2_data_included": False,
        }
        checkpoint_path = artifact_root / "checkpoints" / "pilotnet_v3_dagger_best.pt"
        model, training, history = train_baseline(combined, nominal_validation, config, device, checkpoint_path)
        report["training"] = {**training, "initialized_from_scratch": True, "v1_checkpoint_loaded_for_training": False}
        report["gate_reached"] = "training"
        nominal_v3, nominal_predictions, nominal_labels = validate_offline(model, nominal_validation, config, device)
        v1 = load_checkpoint(v1_checkpoint, device)
        v2 = load_checkpoint(v2_checkpoint, device)
        v1_nominal_predictions, labels = predict_rows(v1, nominal_validation, config, device)
        v2_nominal_predictions, _ = predict_rows(v2, nominal_validation, config, device)
        v3_nominal_predictions, _ = predict_rows(model, nominal_validation, config, device)
        report["validation"] = {
            "nominal_v1": {"overall": error_metrics(v1_nominal_predictions, labels), "calibration": steering_calibration(v1_nominal_predictions, labels)},
            "nominal_v2": {"overall": error_metrics(v2_nominal_predictions, labels), "calibration": steering_calibration(v2_nominal_predictions, labels)},
            "nominal_v3": {**nominal_v3, "calibration": steering_calibration(v3_nominal_predictions, labels)},
            "on_policy_holdout_v1": grouped_on_policy_metrics(v1, dagger_holdout, config, device),
            "on_policy_holdout_v2": grouped_on_policy_metrics(v2, dagger_holdout, config, device),
            "on_policy_holdout_v3": grouped_on_policy_metrics(model, dagger_holdout, config, device),
        }
        report["gate_reached"] = "offline_validation"
        onnx_path = artifact_root / "onnx" / "pilotnet_v3_dagger.onnx"
        export_onnx(model, onnx_path, config)
        report["onnx_equivalence"] = validate_onnx_equivalence(model, dagger_holdout, onnx_path, config)
        report["gate_reached"] = "onnx_equivalence"
        report["plots"] = create_plots(history, nominal_predictions, nominal_labels, artifact_root / "plots")
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
            "v1_checkpoint_sha256": sha256_file(v1_checkpoint), "v2_checkpoint_sha256": sha256_file(v2_checkpoint),
        }
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--nominal-root", type=Path, required=True)
    parser.add_argument("--dagger-root", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_training(config_path=args.config, nominal_root=args.nominal_root, dagger_root=args.dagger_root,
                              v1_checkpoint=args.v1_checkpoint, v2_checkpoint=args.v2_checkpoint,
                              artifact_root=args.artifact_root, result_path=args.result)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
