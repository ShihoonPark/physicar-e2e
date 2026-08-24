"""Train V4 from scratch on cumulative nominal + DAgger1 + DAgger2 aggregation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch

from .pilotnet import PILOTNET_PARAMETER_COUNT
from .pilotnet_dagger_training import grouped_on_policy_metrics, read_dagger_rows
from .pilotnet_failure_diagnosis import steering_calibration
from .pilotnet_recovery_training import load_checkpoint
from .pilotnet_training import (
    GateFailure, create_plots, error_metrics, export_onnx, load_config, predict_rows,
    sha256_file, train_baseline, validate_dataset_integrity, validate_offline,
    validate_onnx_equivalence,
)


def iteration2_split(metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    roles = {item["rollout_id"]: item["role"] for item in metadata.get("episodes", [])}
    training = str(config["dagger2_training_rollout"])
    holdout = str(config["dagger2_holdout_rollout"])
    if training != "dagger_iter2_rollout_A" or holdout != "dagger_iter2_rollout_B":
        raise GateFailure("Iteration-2 frozen A/B assignment changed")
    if roles.get(training) != "training" or roles.get(holdout) != "holdout":
        raise GateFailure("Iteration-2 A/B metadata roles invalid")
    return {"training": training, "holdout": holdout}


def validate_cumulative_composition(
    nominal_rows, dagger1_rows, dagger2_rows, holdout_rows,
) -> dict[str, Any]:
    training_sources = {row["source_mcap_sha256"] for row in [*dagger1_rows, *dagger2_rows]}
    holdout_sources = {row["source_mcap_sha256"] for row in holdout_rows}
    if training_sources & holdout_sources:
        raise GateFailure("Iteration-2 holdout source leaked into cumulative training")
    episodes = {row["episode_id"] for row in [*nominal_rows, *dagger1_rows, *dagger2_rows]}
    if "dagger_rollout_B" in episodes or "dagger_iter2_rollout_B" in episodes:
        raise GateFailure("a DAgger holdout entered cumulative training")
    if any("recovery" in episode for episode in episodes):
        raise GateFailure("V2 recovery data entered V4 training")
    return {"holdout_leakage": False, "dagger1_retained": True, "v2_recovery_excluded": True}


def run_training(
    *, config_path: Path, nominal_root: Path, dagger1_root: Path, dagger2_root: Path,
    v1_checkpoint: Path, v3_checkpoint: Path, artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    if config.get("version") != "pilotnet_training_v4_dagger":
        raise GateFailure("unexpected V4 training config")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "version": "pilotnet_training_v4_dagger", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL", "gate_reached": "environment",
        "architecture": {"parameter_count": PILOTNET_PARAMETER_COUNT, "identical_to_v1_v3": True},
        "environment": {"device": str(device), "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "training_config_sha256": sha256_file(config_path),
    }
    try:
        nominal = validate_dataset_integrity(nominal_root, config)
        nominal_train = nominal.pop("train_rows"); nominal_validation = nominal.pop("validation_rows")
        if config["dagger1_training_rollout"] != "dagger_rollout_A":
            raise GateFailure("DAgger1 training rollout was not retained")
        dagger1_rows = read_dagger_rows(dagger1_root, "dagger_rollout_A")
        metadata_path = dagger2_root / "dataset_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("result") != "PASS" or metadata.get("future_label_violations") != 0:
            raise GateFailure("Iteration-2 dataset quality metadata is not PASS")
        split = iteration2_split(metadata, config)
        dagger2_train = read_dagger_rows(dagger2_root, split["training"])
        dagger2_holdout = read_dagger_rows(dagger2_root, split["holdout"])
        composition = validate_cumulative_composition(nominal_train, dagger1_rows, dagger2_train, dagger2_holdout)
        combined = [*nominal_train, *dagger1_rows, *dagger2_train]
        report["dataset"] = {
            "nominal": nominal, "dagger2_metadata_sha256": sha256_file(metadata_path),
            "nominal_training_samples": len(nominal_train), "dagger1_training_samples": len(dagger1_rows),
            "dagger2_training_samples": len(dagger2_train), "combined_training_samples": len(combined),
            "cumulative_dagger_samples": len(dagger1_rows) + len(dagger2_train),
            "nominal_validation_samples": len(nominal_validation), "dagger2_holdout_samples": len(dagger2_holdout),
            "dagger2_training_rollout": split["training"], "dagger2_holdout_rollout": split["holdout"],
            **composition,
        }
        report["gate_reached"] = "dataset_composition"
        checkpoint_path = artifact_root / "checkpoints" / "pilotnet_v4_dagger_best.pt"
        model, training, history = train_baseline(combined, nominal_validation, config, device, checkpoint_path)
        report["training"] = {**training, "initialized_from_scratch": True, "v3_checkpoint_loaded_for_training": False}
        report["gate_reached"] = "training"
        v1 = load_checkpoint(v1_checkpoint, device); v3 = load_checkpoint(v3_checkpoint, device)
        nominal_v4, nominal_predictions, nominal_labels = validate_offline(model, nominal_validation, config, device)
        p1, labels = predict_rows(v1, nominal_validation, config, device)
        p3, _ = predict_rows(v3, nominal_validation, config, device)
        p4, _ = predict_rows(model, nominal_validation, config, device)
        report["validation"] = {
            "nominal_v1": {"overall": error_metrics(p1, labels), "calibration": steering_calibration(p1, labels)},
            "nominal_v3": {"overall": error_metrics(p3, labels), "calibration": steering_calibration(p3, labels)},
            "nominal_v4": {**nominal_v4, "calibration": steering_calibration(p4, labels)},
            "iteration2_holdout_v1": grouped_on_policy_metrics(v1, dagger2_holdout, config, device),
            "iteration2_holdout_v3": grouped_on_policy_metrics(v3, dagger2_holdout, config, device),
            "iteration2_holdout_v4": grouped_on_policy_metrics(model, dagger2_holdout, config, device),
        }
        report["gate_reached"] = "offline_validation"
        onnx_path = artifact_root / "onnx" / "pilotnet_v4_dagger.onnx"
        export_onnx(model, onnx_path, config)
        report["onnx_equivalence"] = validate_onnx_equivalence(model, dagger2_holdout, onnx_path, config)
        report["gate_reached"] = "onnx_equivalence"
        report["plots"] = create_plots(history, nominal_predictions, nominal_labels, artifact_root / "plots")
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
            "v1_checkpoint_sha256": sha256_file(v1_checkpoint), "v3_checkpoint_sha256": sha256_file(v3_checkpoint),
        }
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--nominal-root", type=Path, required=True)
    parser.add_argument("--dagger1-root", type=Path, required=True); parser.add_argument("--dagger2-root", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True); parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_training(config_path=args.config, nominal_root=args.nominal_root, dagger1_root=args.dagger1_root,
            dagger2_root=args.dagger2_root, v1_checkpoint=args.v1_checkpoint, v3_checkpoint=args.v3_checkpoint,
            artifact_root=args.artifact_root, result_path=args.result)
        print(json.dumps(report, indent=2, sort_keys=True)); return 0
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
