"""Train PilotNet V2 from scratch on frozen nominal plus recovery composition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from .pilotnet_training import (
    GateFailure,
    create_plots,
    error_metrics,
    export_onnx,
    load_config,
    predict_rows,
    read_episode_rows,
    sha256_file,
    train_baseline,
    validate_dataset_integrity,
    validate_offline,
    validate_onnx_equivalence,
)


def recovery_split(metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for item in metadata.get("episodes", []):
        recovery = item.get("recovery", {})
        roles.setdefault(str(recovery.get("anchor_role")), []).append(str(item["episode_id"]))
    training_roles = list(config["recovery_training_anchor_roles"])
    holdout_role = str(config["recovery_holdout_anchor_role"])
    if training_roles != ["failure", "curvature_near"] or holdout_role != "curvature_far":
        raise GateFailure("frozen Recovery V1 train/holdout policy changed")
    training = sorted(episode for role in training_roles for episode in roles.get(role, []))
    holdout = sorted(roles.get(holdout_role, []))
    if len(training) != 8 or len(holdout) != 4 or set(training) & set(holdout):
        raise GateFailure("recovery split must be 8 training and 4 non-overlapping holdout episodes")
    return {"training": training, "holdout": holdout}


def grouped_recovery_metrics(model, rows, config, device) -> dict[str, Any]:
    predictions, labels = predict_rows(model, rows, config, device)
    episode_types = {
        "lat_p10": "lateral_positive", "lat_m10": "lateral_negative",
        "yaw_p06": "heading_positive", "yaw_m06": "heading_negative",
    }
    groups: dict[str, Any] = {}
    for suffix, name in episode_types.items():
        mask = np.asarray([row["episode_id"].endswith(suffix) for row in rows], dtype=bool)
        groups[name] = error_metrics(predictions[mask], labels[mask])
    return {"overall": error_metrics(predictions, labels), "by_perturbation_type": groups}


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_pilotnet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_v2_training(
    *, config_path: Path, nominal_root: Path, recovery_root: Path,
    v1_checkpoint: Path, artifact_root: Path, result_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "version": "pilotnet_training_v2_recovery", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL", "gate_reached": "environment",
        "architecture": {"parameter_count": PILOTNET_PARAMETER_COUNT, "identical_to_v1": True},
        "environment": {"torch": torch.__version__, "device": str(device), "cuda_available": torch.cuda.is_available()},
        "training_config_sha256": sha256_file(config_path),
    }
    try:
        nominal = validate_dataset_integrity(nominal_root, config)
        nominal_train = nominal.pop("train_rows")
        nominal_validation = nominal.pop("validation_rows")
        recovery_metadata_path = recovery_root / "dataset_metadata.json"
        recovery_metadata = json.loads(recovery_metadata_path.read_text(encoding="utf-8"))
        if recovery_metadata.get("result") != "PASS" or recovery_metadata.get("future_label_violations") != 0:
            raise GateFailure("recovery dataset metadata gate is not PASS")
        split = recovery_split(recovery_metadata, config)
        recovery_train = read_episode_rows(recovery_root, split["training"])
        recovery_holdout = read_episode_rows(recovery_root, split["holdout"])
        if {row["episode_id"] for row in recovery_train} & {row["episode_id"] for row in recovery_holdout}:
            raise GateFailure("recovery anchor leakage detected")
        combined_train = [*nominal_train, *recovery_train]
        report["dataset"] = {
            "nominal": nominal, "recovery_metadata_sha256": sha256_file(recovery_metadata_path),
            "recovery_training_episodes": split["training"], "recovery_holdout_episodes": split["holdout"],
            "nominal_training_samples": len(nominal_train), "recovery_training_samples": len(recovery_train),
            "combined_training_samples": len(combined_train), "nominal_validation_samples": len(nominal_validation),
            "recovery_holdout_samples": len(recovery_holdout), "anchor_leakage": False,
        }
        report["gate_reached"] = "dataset_composition"
        checkpoint_path = artifact_root / "checkpoints" / "pilotnet_v2_recovery_best.pt"
        model, training, history = train_baseline(
            combined_train, nominal_validation, config, device, checkpoint_path,
        )
        report["training"] = {**training, "initialized_from_scratch": True, "v1_checkpoint_loaded_for_training": False}
        report["gate_reached"] = "training"
        nominal_v2, nominal_predictions, nominal_labels = validate_offline(model, nominal_validation, config, device)
        recovery_v2 = grouped_recovery_metrics(model, recovery_holdout, config, device)
        v1_model = load_checkpoint(v1_checkpoint, device)
        nominal_v1, _, _ = validate_offline(v1_model, nominal_validation, config, device)
        recovery_v1 = grouped_recovery_metrics(v1_model, recovery_holdout, config, device)
        report["validation"] = {
            "nominal_v1": nominal_v1, "nominal_v2": nominal_v2,
            "recovery_holdout_v1": recovery_v1, "recovery_holdout_v2": recovery_v2,
        }
        report["gate_reached"] = "offline_validation"
        onnx_path = artifact_root / "onnx" / "pilotnet_v2_recovery.onnx"
        export_onnx(model, onnx_path, config)
        report["onnx_equivalence"] = validate_onnx_equivalence(model, recovery_holdout, onnx_path, config)
        report["gate_reached"] = "onnx_equivalence"
        report["plots"] = create_plots(history, nominal_predictions, nominal_labels, artifact_root / "plots")
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
            "v1_checkpoint_sha256": sha256_file(v1_checkpoint),
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
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_v2_training(
            config_path=args.config, nominal_root=args.nominal_root, recovery_root=args.recovery_root,
            v1_checkpoint=args.v1_checkpoint, artifact_root=args.artifact_root, result_path=args.result,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
