"""D1-Preserving Post-Recovery Adaptation V1 at 1.00 m/s.

This workflow has one narrow learning surface: it initializes from the exact
frozen Random-Cone D1 checkpoint, freezes the complete convolutional feature
extractor, and adapts only the fully-connected steering head.  It never
collects data and never uses D2-FE as an initialization source.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_extractor import canonical_json_bytes, sha256_file
from .high_speed_temporal import (
    TemporalOnnxModel,
    export_temporal_onnx,
    validate_equivalence,
)
from .pilotnet import steering_normalized_to_rad
from .pilotnet_temporal import (
    TEMPORAL_CHANNELS,
    TEMPORAL_FRAMES,
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
    preprocess_temporal_paths,
)
from .random_cone_dagger1 import (
    _phase,
    _read_model_rows,
    load_config as load_dagger1_config,
)
from .random_cone_dagger2_post_recovery import (
    HOLDOUT_SCENARIOS,
    PROVENANCE,
    ROUTE_BINS,
    TRAIN_SCENARIOS,
    VALIDATION_SCENARIOS,
)
from . import random_cone_d2_frontier_expansion as prior_fe
from .random_cone_expert import _restore_world, simulator_tracked_status
from .random_cone_temporal_r1 import (
    load_config as load_r1_config,
    run_live_once,
)
from .random_cone_train_data import audit_frozen_expert, disk_state, load_task_config
from .sim_client import SimClient


VERSION = "random_cone_d1_preserving_recovery_1p0_v1"
TRAINING_VERSION = "pilotnet_training_d1_r_random_cone_1p0"
LIVE_VERSION = "pilotnet_e2e_d1_r_random_cone_1p0"
EXPECTED_BRANCH = "experiment/random-cone-d1-preserving-recovery-v1"
EXPECTED_BASE_COMMIT = "2c52f4dd3634fe59233acf95df42789a35da1828"
EXPECTED_REPLAY_COUNT = 8_189
EXPECTED_POST_COUNT = 109
FROZEN_PARAMETER_COUNT = 134_948
TRAINABLE_PARAMETER_COUNT = 120_871

FULL_PASS = "FULL_PASS"
VALIDATION_FAIL = "VALIDATION_FAIL"
UNSEEN_FAIL = "UNSEEN_FAIL"
OFFLINE_RETENTION_FAIL = "OFFLINE_RETENTION_FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
VALID_POLICY_CLASSIFICATIONS = ("RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL")


class AdaptationGateError(RuntimeError):
    """A frozen identity, adaptation, retention, export, or live gate failed."""


@dataclass(frozen=True)
class AdaptationConfig:
    path: Path
    payload: dict[str, Any]
    dagger1: Any
    prior_frontier: Any

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def inputs(self) -> dict[str, Any]:
        return self.payload["frozen_inputs"]

    @property
    def training(self) -> dict[str, Any]:
        return self.payload["training"]

    @property
    def artifact_root(self) -> Path:
        return Path(self.payload["artifact_root"])

    def result_dir(self, repo: Path, key: str) -> Path:
        return repo / self.payload["result_directories"][key]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptationGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdaptationGateError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _hash_gate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise AdaptationGateError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AdaptationGateError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.rstrip()


def load_config(path: Path, repo: Path) -> AdaptationConfig:
    payload = _read_json(path)
    required = {
        "version", "expected_branch", "expected_base_commit", "objective",
        "frozen_inputs", "prior_dagger2_negative", "architecture", "training",
        "offline", "disk_gate", "artifact_root", "result_directories", "live",
        "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise AdaptationGateError("D1-R config fields/version changed")
    if (
        payload["expected_branch"] != EXPECTED_BRANCH
        or payload["expected_base_commit"] != EXPECTED_BASE_COMMIT
    ):
        raise AdaptationGateError("branch/base-commit contract changed")
    expected_training = {
        "seed": 20260826,
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "maximum_epochs": 5,
        "batch_size": 64,
        "max_steering_rad": 0.349066,
        "initialization": "exact_frozen_D1",
        "frozen_module": "features",
        "trainable_module": "regressor",
        "post_recovery_target": "frozen_Expert_steering_normalized",
        "retention_target": "frozen_D1_output_normalized",
        "post_recovery_coefficient": 1.0,
        "retention_coefficient": 1.0,
        "loss": "post_recovery_supervised_MSE_plus_retention_MSE",
        "augmentation": False,
        "regularizers": [],
        "hyperparameter_sweep": False,
        "model_selection_from_validation": False,
        "onnx_opset": 17,
        "onnx_equivalence_samples": 128,
        "onnx_mean_abs_difference_limit": 0.00001,
        "onnx_max_abs_difference_limit": 0.0001,
    }
    if payload["training"] != expected_training:
        raise AdaptationGateError("preregistered optimization contract changed")
    if payload["architecture"] != {
        "name": "Temporal PilotNet",
        "input_shape": [9, 66, 200],
        "output_shape": [1],
        "parameter_count": 255819,
        "frozen_module": "features",
        "frozen_parameter_count": FROZEN_PARAMETER_COUNT,
        "trainable_module": "regressor",
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
    }:
        raise AdaptationGateError("architecture/freeze contract changed")
    permissions = payload["permissions"]
    if permissions != {
        "adaptation_logical_runs_permitted": 1,
        "train_from_scratch_permitted": False,
        "d2_fe_initialization_permitted": False,
        "new_data_collection_permitted": False,
        "new_expert_rollouts_permitted": False,
        "new_dagger_rollouts_permitted": False,
        "dagger3_permitted": False,
        "scenario_changes_permitted": False,
        "speed_changes_permitted": False,
        "architecture_changes_permitted": False,
        "retraining_after_offline_or_live_permitted": False,
        "holdout_access_before_validation_pass_permitted": False,
        "commit_permitted": False,
        "push_permitted": False,
    }:
        raise AdaptationGateError("permission boundary changed")
    if tuple(tuple(float(v) for v in item) for item in payload["offline"]["route_bins_m"]) != ROUTE_BINS:
        raise AdaptationGateError("route-bin contract changed")
    if Path(payload["artifact_root"]).is_relative_to(repo):
        raise AdaptationGateError("checkpoint/ONNX artifact root must remain outside Git")
    dagger_ref = payload["frozen_inputs"]["dagger1_config"]
    dagger_path = _resolve(repo, dagger_ref["path"])
    _hash_gate(dagger_path, dagger_ref["sha256"], "DAgger1 config")
    prior_ref = payload["frozen_inputs"]["d2_fe_config"]
    prior_path = _resolve(repo, prior_ref["path"])
    _hash_gate(prior_path, prior_ref["sha256"], "D2-FE task config")
    return AdaptationConfig(
        path.resolve(), payload, load_dagger1_config(dagger_path, repo),
        prior_fe.load_config(prior_path, repo),
    )


def disk_gate(config: AdaptationConfig, *, live: bool = False) -> dict[str, Any]:
    key = "minimum_before_live_bytes" if live else "minimum_before_adaptation_bytes"
    required = int(config.payload["disk_gate"][key])
    report = disk_state(config.payload["disk_gate"]["path"])
    report.update({
        "required_available_bytes": required,
        "required_available_gib": required / 1024**3,
        "result": "PASS" if int(report["available_bytes"]) >= required else "FAIL",
    })
    if report["result"] != "PASS":
        raise AdaptationGateError(
            f"disk gate failed: {report['available_gib']:.3f} GiB available; "
            f"{required / 1024**3:.3f} GiB required"
        )
    return report


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parameter_contract(model: Any, *, freeze: bool) -> dict[str, Any]:
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad_(False)
        for parameter in model.regressor.parameters():
            parameter.requires_grad_(True)
    frozen = [
        {"name": name, "count": parameter.numel(), "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters() if not parameter.requires_grad
    ]
    trainable = [
        {"name": name, "count": parameter.numel(), "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    result = {
        "frozen_parameters": frozen,
        "trainable_parameters": trainable,
        "frozen_parameter_count": sum(item["count"] for item in frozen),
        "trainable_parameter_count": sum(item["count"] for item in trainable),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    expected_frozen = {
        "features.0.weight", "features.0.bias", "features.2.weight", "features.2.bias",
        "features.4.weight", "features.4.bias", "features.6.weight", "features.6.bias",
        "features.8.weight", "features.8.bias",
    }
    expected_trainable = {
        "regressor.1.weight", "regressor.1.bias", "regressor.3.weight", "regressor.3.bias",
        "regressor.5.weight", "regressor.5.bias", "regressor.7.weight", "regressor.7.bias",
    }
    if (
        {item["name"] for item in frozen} != expected_frozen
        or {item["name"] for item in trainable} != expected_trainable
        or result["frozen_parameter_count"] != FROZEN_PARAMETER_COUNT
        or result["trainable_parameter_count"] != TRAINABLE_PARAMETER_COUNT
        or result["total_parameter_count"] != TEMPORAL_PARAMETER_COUNT
    ):
        raise AdaptationGateError("frozen/trainable parameter contract changed")
    return result


def _load_checkpoint_payload(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    _hash_gate(path, expected_hash, label)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise AdaptationGateError(f"{label} lacks model_state_dict")
    if payload.get("parameter_count") != TEMPORAL_PARAMETER_COUNT:
        raise AdaptationGateError(f"{label} parameter-count metadata changed")
    return payload


def load_exact_d1_initialization(config: AdaptationConfig, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    """Strictly load D1 twice: one frozen teacher and one head-adaptation copy."""
    import torch

    d1 = config.inputs["d1"]
    d2 = config.inputs["d2_fe_negative"]
    source_path = Path(d1["checkpoint_path"])
    if (
        source_path.resolve() == Path(d2["checkpoint_path"]).resolve()
        or d1["checkpoint_sha256"] == d2["checkpoint_sha256"]
    ):
        raise AdaptationGateError("D2-FE initialization is forbidden")
    payload = _load_checkpoint_payload(source_path, d1["checkpoint_sha256"], "D1 checkpoint")
    source_state = payload["model_state_dict"]
    adapted = build_temporal_pilotnet().to(device)
    teacher = build_temporal_pilotnet().to(device)
    adapted.load_state_dict(source_state, strict=True)
    teacher.load_state_dict(source_state, strict=True)
    equality = {
        name: bool(torch.equal(adapted.state_dict()[name].cpu(), value.cpu()))
        for name, value in source_state.items()
    }
    if len(equality) != len(adapted.state_dict()) or not all(equality.values()):
        raise AdaptationGateError("adapted model is not an exact D1 initialization")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    contract = parameter_contract(adapted, freeze=True)
    contract.update({
        "initialization_source": str(source_path),
        "initialization_source_sha256": d1["checkpoint_sha256"],
        "strict_state_dict_load": True,
        "all_initial_tensors_bitwise_equal": True,
        "initial_tensor_count": len(equality),
        "d2_fe_initialization_rejected": True,
    })
    return adapted, teacher, contract


def _audit_d1_preserved_facts(repo: Path, config: AdaptationConfig) -> dict[str, Any]:
    s09_ref = config.inputs["d1_s09_live"]
    cone_ref = config.inputs["d1_cone_free"]
    s09_path = _resolve(repo, s09_ref["path"])
    cone_path = _resolve(repo, cone_ref["path"])
    _hash_gate(s09_path, s09_ref["sha256"], "D1 S09 result")
    _hash_gate(cone_path, cone_ref["sha256"], "D1 cone-free result")
    s09 = _read_json(s09_path)
    run = s09.get("run") or {}
    cone = _read_json(cone_path)
    metrics = cone.get("metrics") or {}
    gates = {
        "s09_genuine_policy_fail": s09.get("classification") == "RANDOM_CONE_POLICY_FAIL",
        "s09_avoidance_positive_clearance": math.isclose(
            float(run.get("minimum_footprint_to_cone_clearance_m", -1.0)),
            0.06465524295557254, rel_tol=0.0, abs_tol=1e-12,
        ),
        "s09_recovery_pass": run.get("recovery_success") is True,
        "s09_recovery_time_preserved": math.isclose(
            float(run.get("recovery_time_s", -1.0)), 0.7956490869983099,
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "s09_late_sustained_offtrack": "sustained off-track" in str(run.get("failure")),
        "s09_final_route_s_preserved": math.isclose(
            float(run.get("final_route_s_m", -1.0)), 29.307113445990467,
            rel_tol=0.0, abs_tol=1e-12,
        ),
        "cone_free_full_lap_pass": cone.get("classification") == "FULL_LAP_PASS",
        "cone_free_no_sustained_offtrack": metrics.get("off_track_events") == 0,
    }
    if not all(gates.values()):
        raise AdaptationGateError("preserved D1 facts changed")
    return {"result": "PASS", "gates": gates}


def _audit_d2_fe_negative(repo: Path, config: AdaptationConfig) -> dict[str, Any]:
    ref = config.inputs["d2_fe_negative"]
    _hash_gate(Path(ref["checkpoint_path"]), ref["checkpoint_sha256"], "frozen D2-FE checkpoint")
    training_path = _resolve(repo, ref["training_summary_path"])
    final_path = _resolve(repo, ref["final_summary_path"])
    _hash_gate(training_path, ref["training_summary_sha256"], "D2-FE training summary")
    _hash_gate(final_path, ref["final_summary_sha256"], "D2-FE final summary")
    training = _read_json(training_path)
    final = _read_json(final_path)
    s09 = (final.get("live_metrics") or {}).get("09") or {}
    gates = {
        "trained_from_scratch": training.get("initialized_from_scratch") is True,
        "exact_8298_sequences": (training.get("training_sources") or {}).get("aggregate_sequence_count") == 8298,
        "s09_genuine_intersection": s09.get("cone_contact_or_intersection_occurred") is True,
        "s09_completion_preserved": math.isclose(
            float(s09.get("route_completion_fraction", -1.0)),
            float(ref["s09_completion_fraction"]), rel_tol=0.0, abs_tol=1e-12,
        ),
        "final_category_regression": final.get("final_category") == ref["expected_final_category"] == "REGRESSION",
        "not_initialization_source": ref["checkpoint_sha256"] != config.inputs["d1"]["checkpoint_sha256"],
    }
    if not all(gates.values()):
        raise AdaptationGateError("frozen D2-FE negative evidence changed")
    return {
        "result": "PASS", "gates": gates, "classification": "REGRESSION",
        "checkpoint_sha256": ref["checkpoint_sha256"], "used_for_initialization": False,
        "reinterpreted": False,
    }


def audit_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_BASE_COMMIT:
        raise AdaptationGateError(f"branch/HEAD mismatch: {branch} at {head}")
    prior_trees_before = prior_fe._prior_tree_gate(repo, config.prior_frontier)
    prior_failure = prior_fe._audit_prior_failure(repo, config.prior_frontier)
    dagger2_data, _ = prior_fe.audit_dagger2_sequences(repo, config.prior_frontier)
    prior_trees_after = prior_fe._prior_tree_gate(repo, config.prior_frontier)
    if prior_trees_after != prior_trees_before:
        raise AdaptationGateError("prior DAgger2 evidence changed during read-only audit")
    hashes: dict[str, Any] = {}
    for key in ("expert_train_manifest", "retention_replay_manifest", "post_recovery_manifest", "validation_manifest"):
        item = config.inputs[key]
        hashes[key] = _hash_gate(_resolve(repo, item["path"]), item["sha256"], key)
    d1 = config.inputs["d1"]
    for key, suffix in (
        ("checkpoint", "checkpoint"), ("onnx", "onnx"),
        ("freeze", "freeze"), ("freeze_seal", "freeze_seal"),
    ):
        hashes[f"d1_{key}"] = _hash_gate(
            Path(d1[f"{suffix}_path"]), d1[f"{suffix}_sha256"], f"D1 {key}",
        )
    hashes["d1_training_summary"] = _hash_gate(
        _resolve(repo, d1["training_summary_path"]), d1["training_summary_sha256"],
        "D1 training summary",
    )
    report = {
        "version": VERSION + "_audit", "generated_utc": utc_now(), "result": "PASS",
        "branch": branch, "head": head, "task_config_sha256": config.sha256,
        "disk_before_adaptation": disk_gate(config),
        "frozen_input_hashes": hashes,
        "d1_preserved_facts": _audit_d1_preserved_facts(repo, config),
        "d2_fe_negative_evidence": _audit_d2_fe_negative(repo, config),
        "prior_dagger2_coverage_gate": {
            **prior_failure,
            "directory_identities_before": prior_trees_before,
            "directory_identities_after": prior_trees_after,
            "result_remains": "FAIL",
            "training_authorized_remains": False,
            "reinterpreted": False,
        },
        "dagger2_109_integrity": dagger2_data,
        "new_data_collection_performed": False,
        "external_dependency_modified": False,
    }
    write_json(config.result_dir(repo, "training") / "audit.json", report)
    return report


def replay_identity_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    audit = audit_stage(repo, sim_root, config)
    replay_ref = config.inputs["retention_replay_manifest"]
    post_ref = config.inputs["post_recovery_manifest"]
    replay_path = _resolve(repo, replay_ref["path"])
    post_path = _resolve(repo, post_ref["path"])
    replay = _read_csv(replay_path)
    post = _read_csv(post_path)
    provenance = {
        name: sum(row.get("provenance") == name for row in replay)
        for name in ("EXPERT_BASELINE", "DAGGER1")
    }
    replay_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in replay})
    post_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in post})
    gates = {
        "replay_count_exact": len(replay) == EXPECTED_REPLAY_COUNT,
        "replay_provenance_exact": provenance == {"EXPERT_BASELINE": 6706, "DAGGER1": 1483},
        "post_count_exact": len(post) == EXPECTED_POST_COUNT,
        "post_provenance_exact": all(row.get("provenance") == PROVENANCE for row in post),
        "training_s01_s08_only": replay_scenarios == list(TRAIN_SCENARIOS) and set(post_scenarios) <= set(TRAIN_SCENARIOS),
        "s09_s10_excluded": not bool(set(replay_scenarios + post_scenarios) & set(VALIDATION_SCENARIOS)),
        "s11_s12_excluded": not bool(set(replay_scenarios + post_scenarios) & set(HOLDOUT_SCENARIOS)),
        "images_referenced_in_place": all(Path(row[key]).is_file() for row in [*replay, *post] for key in ("frame_t_minus_2", "frame_t_minus_1", "frame_t")),
    }
    if not all(gates.values()):
        raise AdaptationGateError("retention/post-recovery source identity failed")
    identity = {
        "version": VERSION + "_replay_identity", "generated_utc": utc_now(),
        "result": "PASS", "gates": gates,
        "retention_replay": {
            "path": str(replay_path), "sha256": sha256_file(replay_path),
            "sequence_count": len(replay), "provenance_counts": provenance,
            "target": "frozen_D1_output_generated_deterministically_at_adaptation_time",
        },
        "post_recovery": {
            "path": str(post_path), "sha256": sha256_file(post_path),
            "sequence_count": len(post), "scenario_ids": post_scenarios,
            "target": "already_stored_frozen_Expert_steering",
        },
        "image_files_copied": 0, "new_images_created": 0,
        "new_bags_created": 0, "new_labels_generated": 0,
        "prior_dagger2_gate": "FAIL_UNCHANGED",
        "prior_directory_identities": audit["prior_dagger2_coverage_gate"]["directory_identities_after"],
    }
    external_path = config.artifact_root / "replay_identity.json"
    compact_path = config.result_dir(repo, "training") / "replay_identity.json"
    write_json(external_path, identity)
    write_json(compact_path, identity)
    identity["external_path"] = str(external_path)
    identity["external_sha256"] = sha256_file(external_path)
    identity["compact_path"] = str(compact_path)
    identity["compact_sha256"] = sha256_file(compact_path)
    return identity


def set_deterministic_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_rows(repo: Path, config: AdaptationConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay_path = _resolve(repo, config.inputs["retention_replay_manifest"]["path"])
    post_path = _resolve(repo, config.inputs["post_recovery_manifest"]["path"])
    replay = _read_model_rows(replay_path, expected_scenarios=TRAIN_SCENARIOS)
    post = _read_model_rows(post_path, expected_scenarios=TRAIN_SCENARIOS)
    if len(replay) != EXPECTED_REPLAY_COUNT or len(post) != EXPECTED_POST_COUNT:
        raise AdaptationGateError("adaptation source row counts changed")
    if any(row.get("provenance") != PROVENANCE for row in post):
        raise AdaptationGateError("post-recovery source provenance changed")
    scenarios = {str(row["scenario_id"]).zfill(2) for row in [*replay, *post]}
    if scenarios & set(VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS):
        raise AdaptationGateError("validation/holdout row entered adaptation")
    return replay, post


def _extract_frozen_features(
    teacher: Any,
    rows: Sequence[dict[str, Any]],
    device: Any,
    batch_size: int,
    maximum: float,
    *,
    target_kind: str,
) -> tuple[Any, Any]:
    import torch

    features: list[Any] = []
    targets: list[Any] = []
    teacher.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = torch.from_numpy(np.stack([
                preprocess_temporal_paths(row["paths"]) for row in batch
            ])).to(device)
            values = teacher.features(images)
            features.append(values.detach().cpu())
            if target_kind == "frozen_D1":
                targets.append(teacher.regressor(values).detach().cpu())
            elif target_kind == "frozen_Expert":
                targets.append(torch.tensor(
                    [[float(row["steering_rad"]) / maximum] for row in batch],
                    dtype=torch.float32,
                ))
            else:
                raise ValueError(target_kind)
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _full_mse(head: Any, features: Any, targets: Any, device: Any, batch_size: int) -> float:
    import torch

    total = 0.0
    count = 0
    head.eval()
    with torch.inference_mode():
        for start in range(0, int(features.shape[0]), batch_size):
            values = features[start:start + batch_size].to(device)
            labels = targets[start:start + batch_size].to(device)
            errors = head(values) - labels
            total += float(torch.sum(errors * errors).detach().cpu())
            count += int(labels.numel())
    return total / count


def _infinite_batches(loader: Any):
    while True:
        yield from loader


def _train_head_epochs(
    adapted: Any,
    teacher: Any,
    replay_rows: Sequence[dict[str, Any]],
    post_rows: Sequence[dict[str, Any]],
    config: AdaptationConfig,
    device: Any,
    state_path: Path,
    identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    training = config.training
    seed = int(training["seed"])
    batch_size = int(training["batch_size"])
    maximum = float(training["max_steering_rad"])
    replay_features, replay_targets = _extract_frozen_features(
        teacher, replay_rows, device, batch_size, maximum, target_kind="frozen_D1",
    )
    post_features, post_targets = _extract_frozen_features(
        teacher, post_rows, device, batch_size, maximum, target_kind="frozen_Expert",
    )
    if tuple(replay_features.shape[1:]) != (64, 1, 18) or tuple(post_features.shape[1:]) != (64, 1, 18):
        raise AdaptationGateError("frozen temporal visual representation shape changed")
    optimizer = torch.optim.Adam(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=float(training["learning_rate"]),
    )
    replay_generator = torch.Generator().manual_seed(seed)
    post_generator = torch.Generator().manual_seed(seed + 1)
    history: list[dict[str, Any]] = []
    epoch_completed = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("identity") != dict(identity) or state.get("training_config") != training:
            raise AdaptationGateError("interrupted adaptation state identity/config changed")
        adapted.load_state_dict(state["model_state_dict"], strict=True)
        parameter_contract(adapted, freeze=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        replay_generator.set_state(state["replay_generator_state"].cpu())
        post_generator.set_state(state["post_generator_state"].cpu())
        history = list(state["history"])
        epoch_completed = int(state["epoch_completed"])
        if bool(state.get("completed")) and epoch_completed != int(training["maximum_epochs"]):
            raise AdaptationGateError("completed adaptation state has wrong epoch count")
    replay_loader = DataLoader(
        TensorDataset(replay_features, replay_targets), batch_size=batch_size,
        shuffle=True, generator=replay_generator, num_workers=0,
    )
    post_loader = DataLoader(
        TensorDataset(post_features, post_targets), batch_size=batch_size,
        shuffle=True, generator=post_generator, num_workers=0,
    )
    criterion = torch.nn.MSELoss(reduction="mean")
    maximum_epochs = int(training["maximum_epochs"])
    for epoch in range(epoch_completed + 1, maximum_epochs + 1):
        adapted.regressor.train()
        post_batches = _infinite_batches(post_loader)
        post_step_sum = 0.0
        retention_step_sum = 0.0
        step_count = 0
        for replay_features_batch, replay_targets_batch in replay_loader:
            post_features_batch, post_targets_batch = next(post_batches)
            replay_features_batch = replay_features_batch.to(device)
            replay_targets_batch = replay_targets_batch.to(device)
            post_features_batch = post_features_batch.to(device)
            post_targets_batch = post_targets_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            post_loss = criterion(adapted.regressor(post_features_batch), post_targets_batch)
            retention_loss = criterion(
                adapted.regressor(replay_features_batch), replay_targets_batch,
            )
            loss = (
                float(training["post_recovery_coefficient"]) * post_loss
                + float(training["retention_coefficient"]) * retention_loss
            )
            if not torch.isfinite(loss):
                raise AdaptationGateError("non-finite adaptation loss")
            loss.backward()
            optimizer.step()
            post_step_sum += float(post_loss.detach().cpu())
            retention_step_sum += float(retention_loss.detach().cpu())
            step_count += 1
        full_post = _full_mse(
            adapted.regressor, post_features, post_targets, device, batch_size,
        )
        full_retention = _full_mse(
            adapted.regressor, replay_features, replay_targets, device, batch_size,
        )
        item = {
            "epoch": epoch,
            "optimizer_steps": step_count,
            "mean_step_post_recovery_supervised_normalized_mse": post_step_sum / step_count,
            "mean_step_retention_normalized_mse": retention_step_sum / step_count,
            "full_post_recovery_supervised_normalized_mse": full_post,
            "full_retention_normalized_mse": full_retention,
            "full_equal_coefficient_objective": full_post + full_retention,
            "post_recovery_coefficient": 1.0,
            "retention_coefficient": 1.0,
        }
        history.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)
        _atomic_torch_save(state_path, {
            "version": TRAINING_VERSION + "_resumable_state",
            "identity": dict(identity), "training_config": training,
            "epoch_completed": epoch, "completed": epoch == maximum_epochs,
            "model_state_dict": adapted.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_generator_state": replay_generator.get_state(),
            "post_generator_state": post_generator.get_state(),
            "history": history,
        })
    if len(history) != maximum_epochs or int(history[-1]["epoch"]) != maximum_epochs:
        raise AdaptationGateError("single adaptation did not complete exactly five epochs")
    return history, epoch_completed


def _adaptation_artifact_valid(report: Mapping[str, Any]) -> bool:
    checkpoint = (report.get("artifacts") or {}).get("checkpoint") or {}
    path = Path(str(checkpoint.get("path", "")))
    return bool(
        report.get("result") == "PASS"
        and report.get("training_runs") == 1
        and report.get("epochs_completed") == 5
        and report.get("initialized_from_exact_D1") is True
        and report.get("backbone_unchanged_after_training") is True
        and path.is_file()
        and sha256_file(path) == checkpoint.get("sha256")
    )


def training_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    disk_before = disk_gate(config)
    replay_identity = replay_identity_stage(repo, sim_root, config)
    result_dir = config.result_dir(repo, "training")
    report_path = result_dir / "adaptation.json"
    if report_path.is_file():
        report = _read_json(report_path)
        if _adaptation_artifact_valid(report):
            return report
        raise AdaptationGateError("existing adaptation evidence is incomplete or changed")
    if config.result_dir(repo, "live").exists() and any(config.result_dir(repo, "live").rglob("*attempt*.json")):
        raise AdaptationGateError("live attempt exists before adaptation/freeze")
    checkpoint = config.artifact_root / "checkpoints" / "random_cone_temporal_d1_r.pt"
    state_path = config.artifact_root / "checkpoints" / "random_cone_temporal_d1_r_training_state.pt"
    config_snapshot = config.artifact_root / "training_config_snapshot.json"
    marker = result_dir / "training.started.json"
    identity = {
        "task_config_sha256": config.sha256,
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "retention_replay_manifest_sha256": config.inputs["retention_replay_manifest"]["sha256"],
        "post_recovery_manifest_sha256": config.inputs["post_recovery_manifest"]["sha256"],
    }
    if marker.is_file():
        previous = _read_json(marker)
        if previous.get("source_identity") != identity:
            raise AdaptationGateError("interrupted adaptation marker identity changed")
        if previous.get("status") == "COMPLETED":
            raise AdaptationGateError("completed marker exists without valid adaptation evidence")
    else:
        if checkpoint.exists() or state_path.exists():
            raise AdaptationGateError("unregistered D1-R artifact exists before the single run")
        write_json(marker, {
            "status": "ONE_LOGICAL_ADAPTATION_RUN_STARTED",
            "started_utc": utc_now(), "source_identity": identity,
            "maximum_epochs": 5, "retraining_permitted": False,
        })
    write_json(config_snapshot, {
        "version": TRAINING_VERSION + "_config_snapshot",
        "task_config_sha256": config.sha256,
        "training": config.training,
        "architecture": config.payload["architecture"],
        "sources": replay_identity,
        "excluded": [
            "scratch initialization", "D2-FE initialization", "S09/S10 training",
            "S11/S12", "new bags", "new images", "new labels", "DAgger3",
            "architecture changes", "speed changes", "scenario changes",
        ],
    })
    import torch

    set_deterministic_seed(int(config.training["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapted, teacher, contract = load_exact_d1_initialization(config, device)
    source_state = _cpu_state_dict(teacher)
    replay_rows, post_rows = _model_rows(repo, config)
    history, resumed_epoch = _train_head_epochs(
        adapted, teacher, replay_rows, post_rows, config, device, state_path, identity,
    )
    post_state = _cpu_state_dict(adapted)
    backbone_equal = all(
        torch.equal(post_state[name], source_state[name])
        for name in post_state if name.startswith("features.")
    )
    head_changed = any(
        not torch.equal(post_state[name], source_state[name])
        for name in post_state if name.startswith("regressor.")
    )
    if not backbone_equal or not head_changed:
        raise AdaptationGateError("backbone changed or steering head failed to adapt")
    _atomic_torch_save(checkpoint, {
        "version": TRAINING_VERSION + "_checkpoint",
        "model_state_dict": post_state,
        "epoch": 5,
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "input_shape": [TEMPORAL_CHANNELS, 66, 200],
        "output_shape": [1],
        "training_config": config.training,
        "identity": identity,
        "initialized_from_exact_D1": True,
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "d2_fe_initialization_used": False,
        "frozen_parameter_names": [item["name"] for item in contract["frozen_parameters"]],
        "trainable_parameter_names": [item["name"] for item in contract["trainable_parameters"]],
    })
    report = {
        "version": TRAINING_VERSION + "_adaptation", "generated_utc": utc_now(),
        "result": "PASS", "task_config_sha256": config.sha256,
        "device": str(device), "training_runs": 1, "epochs_completed": len(history),
        "maximum_epochs": 5, "early_stopping_used": False,
        "resumed_from_completed_epoch": resumed_epoch,
        "optimizer": "Adam", "learning_rate": 0.0001, "batch_size": 64,
        "loss": {
            "formula": "post_recovery_supervised_MSE + retention_MSE",
            "post_recovery_coefficient": 1.0, "retention_coefficient": 1.0,
            "additional_regularizers": [],
        },
        "history": history,
        "parameter_contract": contract,
        "initialized_from_exact_D1": True,
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "d2_fe_initialization_used": False,
        "trained_from_scratch": False,
        "backbone_unchanged_after_training": backbone_equal,
        "steering_head_changed_after_training": head_changed,
        "sources": {
            "retention_replay_sequence_count": len(replay_rows),
            "post_recovery_sequence_count": len(post_rows),
            "retention_target": "frozen_D1_output",
            "post_recovery_target": "frozen_Expert_steering",
            "validation_rows_used_for_optimization": 0,
            "holdout_rows_used": 0,
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint), "sha256": sha256_file(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            },
            "training_config_snapshot": {
                "path": str(config_snapshot), "sha256": sha256_file(config_snapshot),
                "size_bytes": config_snapshot.stat().st_size,
            },
        },
        "replay_identity": replay_identity,
        "model_selection_from_validation": False,
        "new_data_collection_performed": False,
        "prior_dagger2_coverage_gate": "FAIL_UNCHANGED",
        "disk_before_adaptation": disk_before,
        "disk_after_adaptation": disk_state("/"),
        "frozen_before_live": False,
    }
    write_json(report_path, report)
    write_json(marker, {
        "status": "COMPLETED", "completed_utc": utc_now(),
        "source_identity": identity, "checkpoint_sha256": sha256_file(checkpoint),
        "epochs_completed": 5, "retraining_permitted": False,
    })
    if state_path.is_file():
        state_path.unlink()
    report["temporary_training_state_removed"] = not state_path.exists()
    write_json(report_path, report)
    return report


def _load_model_from_checkpoint(path: Path, expected_hash: str, device: Any, label: str) -> Any:
    payload = _load_checkpoint_payload(path, expected_hash, label)
    model = build_temporal_pilotnet().to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model


def _predict_pair(
    d1: Any,
    adapted: Any,
    rows: Sequence[dict[str, Any]],
    device: Any,
    batch_size: int,
    maximum: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    d1_predictions: list[np.ndarray] = []
    adapted_predictions: list[np.ndarray] = []
    labels: list[float] = []
    d1.eval()
    adapted.eval()
    d1_state = d1.state_dict()
    adapted_state = adapted.state_dict()
    if not all(
        torch.equal(d1_state[name].cpu(), adapted_state[name].cpu())
        for name in d1_state if name.startswith("features.")
    ):
        raise AdaptationGateError("offline models do not share the exact frozen backbone")
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            images = torch.from_numpy(np.stack([
                preprocess_temporal_paths(row["paths"]) for row in batch
            ])).to(device)
            features = d1.features(images)
            d1_normalized = d1.regressor(features).detach().cpu().numpy().reshape(-1)
            adapted_normalized = adapted.regressor(features).detach().cpu().numpy().reshape(-1)
            d1_predictions.extend(steering_normalized_to_rad(d1_normalized, maximum))
            adapted_predictions.extend(steering_normalized_to_rad(adapted_normalized, maximum))
            labels.extend(float(row["steering_rad"]) for row in batch)
    return (
        np.asarray(d1_predictions, dtype=np.float64),
        np.asarray(adapted_predictions, dtype=np.float64),
        np.asarray(labels, dtype=np.float64),
    )


def steering_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if predictions.shape != labels.shape:
        raise ValueError("prediction/label shape mismatch")
    if labels.size == 0:
        return {
            "sample_count": 0, "mae_rad": None, "rmse_rad": None,
            "bias_mean_signed_error_rad": None, "max_absolute_error_rad": None,
            "corrective_magnitude_ratio": None,
        }
    errors = predictions - labels
    denominator = float(np.mean(np.abs(labels)))
    return {
        "sample_count": int(labels.size),
        "mae_rad": float(np.mean(np.abs(errors))),
        "rmse_rad": float(np.sqrt(np.mean(errors * errors))),
        "bias_mean_signed_error_rad": float(np.mean(errors)),
        "max_absolute_error_rad": float(np.max(np.abs(errors))),
        "corrective_magnitude_ratio": (
            float(np.mean(np.abs(predictions))) / denominator if denominator > 0.0 else None
        ),
    }


def comparison_metrics(
    d1: np.ndarray, adapted: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    if d1.shape != adapted.shape or d1.shape != labels.shape:
        raise ValueError("comparison arrays differ in shape")
    if labels.size == 0:
        return {
            "sample_count": 0,
            "D1": steering_metrics(d1, labels),
            "D1_R": steering_metrics(adapted, labels),
            "adapted_minus_D1_prediction_delta_rad": None,
            "sign_disagreement": {"count": 0, "fraction": None},
        }
    delta = adapted - d1
    signs = np.sign(adapted) != np.sign(d1)
    d1_metrics = steering_metrics(d1, labels)
    adapted_metrics = steering_metrics(adapted, labels)
    return {
        "sample_count": int(labels.size),
        "D1": d1_metrics,
        "D1_R": adapted_metrics,
        "D1_R_minus_D1": {
            "mae_rad": adapted_metrics["mae_rad"] - d1_metrics["mae_rad"],
            "rmse_rad": adapted_metrics["rmse_rad"] - d1_metrics["rmse_rad"],
            "bias_rad": (
                adapted_metrics["bias_mean_signed_error_rad"]
                - d1_metrics["bias_mean_signed_error_rad"]
            ),
            "corrective_magnitude_ratio": (
                None if adapted_metrics["corrective_magnitude_ratio"] is None
                or d1_metrics["corrective_magnitude_ratio"] is None
                else adapted_metrics["corrective_magnitude_ratio"]
                - d1_metrics["corrective_magnitude_ratio"]
            ),
        },
        "adapted_minus_D1_prediction_delta_rad": {
            "mean_signed": float(np.mean(delta)),
            "mean_absolute": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_absolute": float(np.max(np.abs(delta))),
        },
        "sign_disagreement": {
            "count": int(np.sum(signs)),
            "fraction": float(np.mean(signs)),
        },
    }


def _route_bin_label(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:g}m"


def _route_bin_indices(rows: Sequence[dict[str, Any]], lower: float, upper: float, last: bool) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(rows):
        value = float(row["route_progress_m"])
        if lower <= value <= upper if last else lower <= value < upper:
            result.append(index)
    return result


def _selected_report(
    d1: np.ndarray, adapted: np.ndarray, labels: np.ndarray, indices: Sequence[int],
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    return comparison_metrics(d1[selected], adapted[selected], labels[selected])


def _structural_group(
    d1: np.ndarray,
    adapted: np.ndarray,
    *,
    zero_tolerance: float,
    full_span: float,
) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(d1)) and np.all(np.isfinite(adapted)))
    if not finite:
        return {
            "sample_count": int(d1.size), "finite": False,
            "broad_sign_reversal": False, "severe_output_collapse": False,
            "gross_prediction_shift": False, "result": "FAIL",
        }
    nonzero = (np.abs(d1) > zero_tolerance) & (np.abs(adapted) > zero_tolerance)
    reversal_count = int(np.sum(np.sign(d1[nonzero]) != np.sign(adapted[nonzero])))
    compared = int(np.sum(nonzero))
    reversal_fraction = reversal_count / compared if compared else 0.0
    collapse = bool(
        d1.size
        and np.any(np.abs(d1) > zero_tolerance)
        and np.all(np.abs(adapted) <= zero_tolerance)
    )
    max_delta = float(np.max(np.abs(adapted - d1))) if d1.size else 0.0
    broad = reversal_fraction > 0.5
    gross = max_delta > full_span
    return {
        "sample_count": int(d1.size), "finite": True,
        "nonzero_sign_comparison_count": compared,
        "opposite_sign_count": reversal_count,
        "opposite_sign_fraction": reversal_fraction,
        "broad_sign_reversal": broad,
        "severe_output_collapse": collapse,
        "max_absolute_prediction_delta_rad": max_delta,
        "gross_prediction_shift": gross,
        "result": "FAIL" if broad or collapse or gross else "PASS",
    }


def structural_retention_check(
    groups: Mapping[str, tuple[np.ndarray, np.ndarray]],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    zero = float(definition["zero_tolerance_rad"])
    span = float(definition["full_signed_steering_span_rad"])
    reports = {
        name: _structural_group(d1, adapted, zero_tolerance=zero, full_span=span)
        for name, (d1, adapted) in groups.items()
    }
    failure_modes = {
        "nonfinite_output": any(not item["finite"] for item in reports.values()),
        "broad_sign_reversal": any(item["broad_sign_reversal"] for item in reports.values()),
        "severe_output_collapse": any(item["severe_output_collapse"] for item in reports.values()),
        "gross_prediction_shift": any(item["gross_prediction_shift"] for item in reports.values()),
    }
    passed = not any(failure_modes.values())
    return {
        "result": "PASS" if passed else "FAIL",
        "classification": "OFFLINE_RETENTION_PASS" if passed else OFFLINE_RETENTION_FAIL,
        "definitions": dict(definition),
        "failure_modes": failure_modes,
        "groups": reports,
        "performance_threshold_tuned": False,
        "closed_loop_validation_decisive_if_pass": True,
    }


def offline_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    adaptation = training_stage(repo, sim_root, config)
    report_path = config.result_dir(repo, "training") / "offline_retention.json"
    if report_path.is_file():
        existing = _read_json(report_path)
        if existing.get("checkpoint_sha256") == adaptation["artifacts"]["checkpoint"]["sha256"]:
            return existing
        raise AdaptationGateError("existing offline audit checkpoint identity changed")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d1_ref = config.inputs["d1"]
    d1 = _load_model_from_checkpoint(
        Path(d1_ref["checkpoint_path"]), d1_ref["checkpoint_sha256"], device, "D1 checkpoint",
    )
    checkpoint = adaptation["artifacts"]["checkpoint"]
    adapted = _load_model_from_checkpoint(
        Path(checkpoint["path"]), checkpoint["sha256"], device, "D1-R checkpoint",
    )
    parameter_contract(adapted, freeze=True)
    replay_rows, post_rows = _model_rows(repo, config)
    validation_ref = config.inputs["validation_manifest"]
    validation_path = _resolve(repo, validation_ref["path"])
    _hash_gate(validation_path, validation_ref["sha256"], "frozen S09/S10 validation")
    validation_rows = _read_model_rows(validation_path, expected_scenarios=VALIDATION_SCENARIOS)
    if len(validation_rows) != int(validation_ref["sequence_count"]):
        raise AdaptationGateError("validation row count changed")
    maximum = float(config.training["max_steering_rad"])
    batch_size = int(config.training["batch_size"])
    replay_predictions = _predict_pair(d1, adapted, replay_rows, device, batch_size, maximum)
    post_predictions = _predict_pair(d1, adapted, post_rows, device, batch_size, maximum)
    validation_predictions = _predict_pair(d1, adapted, validation_rows, device, batch_size, maximum)
    d1_replay, adapted_replay, labels_replay = replay_predictions
    d1_post, adapted_post, labels_post = post_predictions
    d1_validation, adapted_validation, labels_validation = validation_predictions

    train_task_ref = config.inputs["expert_task_config"]
    train_task_path = _resolve(repo, train_task_ref["path"])
    _hash_gate(train_task_path, train_task_ref["sha256"], "frozen Expert task config")
    train_task = load_task_config(train_task_path, repo)
    _expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in VALIDATION_SCENARIOS
    }
    if set(bundles) != set(VALIDATION_SCENARIOS):
        raise AdaptationGateError("frozen S09/S10 scenario bundle identity changed")
    s09_indices = [index for index, row in enumerate(validation_rows) if row["scenario_id"] == "09"]
    s10_indices = [index for index, row in enumerate(validation_rows) if row["scenario_id"] == "10"]
    s09_phases: dict[str, Any] = {}
    phase_indices: dict[str, list[int]] = {}
    for phase in config.payload["offline"]["s09_phases"]:
        indices = [
            index for index in s09_indices
            if _phase(bundles["09"], float(validation_rows[index]["route_progress_m"])) == phase
        ]
        phase_indices[phase] = indices
        s09_phases[phase] = _selected_report(
            d1_validation, adapted_validation, labels_validation, indices,
        )
    validation_bins: dict[str, Any] = {}
    replay_bins: dict[str, Any] = {}
    for bin_index, (lower, upper) in enumerate(ROUTE_BINS):
        label = _route_bin_label(lower, upper)
        validation_indices = _route_bin_indices(
            validation_rows, lower, upper, bin_index == len(ROUTE_BINS) - 1,
        )
        replay_indices = _route_bin_indices(
            replay_rows, lower, upper, bin_index == len(ROUTE_BINS) - 1,
        )
        validation_bins[label] = {
            "bounds_m": [lower, upper],
            **_selected_report(d1_validation, adapted_validation, labels_validation, validation_indices),
        }
        replay_bins[label] = {
            "bounds_m": [lower, upper],
            **_selected_report(d1_replay, adapted_replay, labels_replay, replay_indices),
        }
    protected: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "original_D1_aggregate": (d1_replay, adapted_replay),
        "S09": (d1_validation[np.asarray(s09_indices)], adapted_validation[np.asarray(s09_indices)]),
        "S10": (d1_validation[np.asarray(s10_indices)], adapted_validation[np.asarray(s10_indices)]),
    }
    for phase, indices in phase_indices.items():
        selected = np.asarray(indices, dtype=np.int64)
        protected[f"S09_{phase}"] = (d1_validation[selected], adapted_validation[selected])
    check = structural_retention_check(protected, config.payload["offline"]["structural_check"])
    result = "PASS" if check["result"] == "PASS" else OFFLINE_RETENTION_FAIL
    report = {
        "version": VERSION + "_offline_retention", "generated_utc": utc_now(),
        "result": result, "checkpoint_sha256": checkpoint["sha256"],
        "d1_source_checkpoint_sha256": d1_ref["checkpoint_sha256"],
        "matched_preprocessing_and_temporal_contract": True,
        "datasets": {
            "original_D1_aggregate": comparison_metrics(d1_replay, adapted_replay, labels_replay),
            "S09": _selected_report(d1_validation, adapted_validation, labels_validation, s09_indices),
            "S10": _selected_report(d1_validation, adapted_validation, labels_validation, s10_indices),
            "S09_phases": s09_phases,
            "validation_route_bins": validation_bins,
            "original_D1_aggregate_route_bins": replay_bins,
            "DAGGER2_POST_RECOVERY_109": comparison_metrics(d1_post, adapted_post, labels_post),
        },
        "structural_retention_check": check,
        "key_question": {
            "post_recovery_expert_error_reduced": (
                steering_metrics(adapted_post, labels_post)["mae_rad"]
                < steering_metrics(d1_post, labels_post)["mae_rad"]
            ),
            "D1_avoidance_and_nominal_predictions_reported_without_tuned_gate": True,
        },
        "validation_used_for_training_or_model_selection": False,
        "retraining_after_metrics_permitted": False,
        "retraining_after_metrics_performed": False,
        "expert_metadata_audit": expert_audit,
        "holdout_camera_or_steering_data_accessed": False,
    }
    write_json(report_path, report)
    write_json(config.result_dir(repo, "training") / "structural_retention_check.json", check)
    return report


def leakage_audit(
    repo: Path, sim_root: Path, config: AdaptationConfig, *, stage: str,
) -> dict[str, Any]:
    if stage == "before_holdout":
        validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
        if not validation_path.is_file() or not validation_allows_holdout(_read_json(validation_path)):
            raise AdaptationGateError("S09/S10 have not both passed; holdout audit remains blocked")
    expert_path = _resolve(repo, config.inputs["expert_train_manifest"]["path"])
    replay_path = _resolve(repo, config.inputs["retention_replay_manifest"]["path"])
    post_path = _resolve(repo, config.inputs["post_recovery_manifest"]["path"])
    validation_path = _resolve(repo, config.inputs["validation_manifest"]["path"])
    expert = _read_csv(expert_path)
    replay = _read_csv(replay_path)
    post = _read_csv(post_path)
    validation = _read_csv(validation_path)
    training_rows = [*replay, *post]
    train_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in training_rows})
    validation_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in validation})
    holdout = set(HOLDOUT_SCENARIOS)
    gates = {
        "expert_training_s01_s08_only": {str(row["scenario_id"]).zfill(2) for row in expert} == set(TRAIN_SCENARIOS),
        "dagger1_and_retention_s01_s08_only": {str(row["scenario_id"]).zfill(2) for row in replay} == set(TRAIN_SCENARIOS),
        "dagger2_s01_s08_only": {str(row["scenario_id"]).zfill(2) for row in post} <= set(TRAIN_SCENARIOS),
        "adaptation_s01_s08_only": train_scenarios == list(TRAIN_SCENARIOS),
        "s09_s10_absent_from_training": not bool(set(train_scenarios) & set(VALIDATION_SCENARIOS)),
        "validation_s09_s10_only": validation_scenarios == list(VALIDATION_SCENARIOS),
        "s11_s12_absent_from_expert_training": not any(str(row["scenario_id"]).zfill(2) in holdout for row in expert),
        "s11_s12_absent_from_dagger1": not any(str(row["scenario_id"]).zfill(2) in holdout for row in replay if row.get("provenance") == "DAGGER1"),
        "s11_s12_absent_from_dagger2": not any(str(row["scenario_id"]).zfill(2) in holdout for row in post),
        "s11_s12_absent_from_retention_adaptation": not any(str(row["scenario_id"]).zfill(2) in holdout for row in training_rows),
        "s11_s12_absent_from_validation_training": not any(str(row["scenario_id"]).zfill(2) in holdout for row in validation),
        "source_hashes_exact": all((
            sha256_file(expert_path) == config.inputs["expert_train_manifest"]["sha256"],
            sha256_file(replay_path) == config.inputs["retention_replay_manifest"]["sha256"],
            sha256_file(post_path) == config.inputs["post_recovery_manifest"]["sha256"],
            sha256_file(validation_path) == config.inputs["validation_manifest"]["sha256"],
        )),
    }
    report = {
        "version": VERSION + "_leakage_audit", "generated_utc": utc_now(),
        "stage": stage, "result": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates, "training_scenarios": train_scenarios,
        "validation_scenarios": validation_scenarios,
        "holdout_scenarios": list(HOLDOUT_SCENARIOS),
        "holdout_camera_rows_read": 0, "holdout_steering_rows_read": 0,
        "holdout_bags_opened": 0, "holdout_worlds_activated": 0,
        "new_data_collection_performed": False,
    }
    path = config.result_dir(repo, "training") / "audits" / f"leakage_{stage}.json"
    write_json(path, report)
    if report["result"] != "PASS":
        raise AdaptationGateError(f"leakage audit failed at {stage}")
    return report


def _frozen_artifacts_valid(report: Mapping[str, Any]) -> bool:
    if (
        report.get("result") != "PASS"
        or report.get("model_frozen_before_live") is not True
        or (report.get("onnx_equivalence") or {}).get("result") != "PASS"
        or (report.get("architecture") or {}).get("parameter_count") != TEMPORAL_PARAMETER_COUNT
    ):
        return False
    for key in ("checkpoint", "onnx", "training_config_snapshot", "training_summary_snapshot", "replay_identity"):
        item = (report.get("artifacts") or {}).get(key) or {}
        path = Path(str(item.get("path", "")))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    for key in ("freeze", "freeze_seal"):
        item = report.get(key) or {}
        external = Path(str(item.get("path", "")))
        compact = Path(str(item.get("compact_path", "")))
        if (
            not external.is_file() or not compact.is_file()
            or sha256_file(external) != item.get("sha256")
            or sha256_file(compact) != item.get("sha256")
        ):
            return False
    return True


def verify_frozen_d1_r(repo: Path, config: AdaptationConfig) -> dict[str, Any]:
    report = _read_json(config.result_dir(repo, "training") / "summary.json")
    if not _frozen_artifacts_valid(report):
        raise AdaptationGateError("D1-R is not a complete frozen model")
    if report.get("task_config_sha256") != config.sha256:
        raise AdaptationGateError("D1-R task-config identity changed")
    freeze_path = Path(report["freeze"]["path"])
    seal_path = Path(report["freeze_seal"]["path"])
    freeze = _read_json(freeze_path)
    seal = _read_json(seal_path)
    expected_seal = {
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": report["artifacts"]["checkpoint"]["sha256"],
        "onnx_sha256": report["artifacts"]["onnx"]["sha256"],
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "post_recovery_manifest_sha256": config.inputs["post_recovery_manifest"]["sha256"],
        "retention_replay_manifest_sha256": config.inputs["retention_replay_manifest"]["sha256"],
        "task_config_sha256": config.sha256,
        "training_summary_snapshot_sha256": report["artifacts"]["training_summary_snapshot"]["sha256"],
        "live_attempt_count_before_seal": 0,
        "model_changes_after_seal_permitted": False,
    }
    if any(seal.get(key) != value for key, value in expected_seal.items()):
        raise AdaptationGateError("D1-R freeze seal contract changed")
    if (
        freeze.get("model_name") != "Random-Cone Temporal PilotNet D1-R"
        or freeze.get("initialized_from_exact_D1") is not True
        or freeze.get("d2_fe_initialization_used") is not False
        or freeze.get("frozen_before_S09") is not True
        or freeze.get("single_logical_adaptation_run") is not True
        or (freeze.get("architecture") or {}).get("parameter_count") != TEMPORAL_PARAMETER_COUNT
    ):
        raise AdaptationGateError("D1-R freeze contract changed")
    return report


def freeze_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    offline = offline_stage(repo, sim_root, config)
    if offline.get("structural_retention_check", {}).get("result") != "PASS":
        raise AdaptationGateError("OFFLINE_RETENTION_FAIL: live/export freeze is blocked")
    leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    summary_path = config.result_dir(repo, "training") / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if _frozen_artifacts_valid(existing):
            return verify_frozen_d1_r(repo, config)
        raise AdaptationGateError("existing freeze evidence is incomplete or changed")
    adaptation = _read_json(config.result_dir(repo, "training") / "adaptation.json")
    checkpoint = adaptation["artifacts"]["checkpoint"]
    import torch

    device = torch.device("cpu")
    model = _load_model_from_checkpoint(
        Path(checkpoint["path"]), checkpoint["sha256"], device, "D1-R checkpoint",
    )
    contract = parameter_contract(model, freeze=True)
    if contract["total_parameter_count"] != TEMPORAL_PARAMETER_COUNT:
        raise AdaptationGateError("D1-R parameter count changed before export")
    onnx_path = config.artifact_root / "onnx" / "random_cone_temporal_d1_r.onnx"
    if onnx_path.exists():
        raise AdaptationGateError("unsealed D1-R ONNX already exists")
    export_temporal_onnx(model, onnx_path, config.training)
    replay_rows, _post_rows = _model_rows(repo, config)
    equivalence = validate_equivalence(model, replay_rows, onnx_path, config.training)
    snapshot_path = config.artifact_root / "training_summary_snapshot.json"
    snapshot = {
        "version": TRAINING_VERSION + "_snapshot", "generated_utc": utc_now(),
        "adaptation": adaptation,
        "offline_retention_path": str(config.result_dir(repo, "training") / "offline_retention.json"),
        "offline_retention_sha256": sha256_file(config.result_dir(repo, "training") / "offline_retention.json"),
        "structural_retention_result": offline["structural_retention_check"]["result"],
        "leakage_audit": leakage,
        "no_retraining_after_offline_metrics": True,
    }
    write_json(snapshot_path, snapshot)
    config_snapshot = Path(adaptation["artifacts"]["training_config_snapshot"]["path"])
    replay_identity_path = Path(adaptation["replay_identity"]["external_path"])
    artifacts = {
        "checkpoint": checkpoint,
        "onnx": {
            "path": str(onnx_path), "sha256": sha256_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
        },
        "training_config_snapshot": {
            "path": str(config_snapshot), "sha256": sha256_file(config_snapshot),
            "size_bytes": config_snapshot.stat().st_size,
        },
        "training_summary_snapshot": {
            "path": str(snapshot_path), "sha256": sha256_file(snapshot_path),
            "size_bytes": snapshot_path.stat().st_size,
        },
        "replay_identity": {
            "path": str(replay_identity_path), "sha256": sha256_file(replay_identity_path),
            "size_bytes": replay_identity_path.stat().st_size,
        },
    }
    freeze = {
        "version": VERSION + "_freeze", "generated_utc": utc_now(),
        "model_name": "Random-Cone Temporal PilotNet D1-R",
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "post_recovery_manifest_sha256": config.inputs["post_recovery_manifest"]["sha256"],
        "retention_replay_manifest_sha256": config.inputs["retention_replay_manifest"]["sha256"],
        "replay_identity_sha256": artifacts["replay_identity"]["sha256"],
        "task_config_sha256": config.sha256,
        "training_config_snapshot_sha256": artifacts["training_config_snapshot"]["sha256"],
        "training_summary_snapshot_sha256": artifacts["training_summary_snapshot"]["sha256"],
        "architecture": {
            "input": ["batch", 9, 66, 200], "output": ["batch", 1],
            "parameter_count": TEMPORAL_PARAMETER_COUNT,
            "frozen_parameter_count": FROZEN_PARAMETER_COUNT,
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        },
        "initialized_from_exact_D1": True, "d2_fe_initialization_used": False,
        "single_logical_adaptation_run": True, "epochs": 5,
        "frozen_before_S09": True, "live_attempt_count_before_freeze": 0,
        "model_changes_after_freeze_permitted": False,
    }
    external_freeze = config.artifact_root / "freeze.json"
    compact_freeze = config.result_dir(repo, "training") / "freeze.json"
    write_json(external_freeze, freeze)
    write_json(compact_freeze, freeze)
    freeze_sha = sha256_file(external_freeze)
    seal = {
        "version": VERSION + "_freeze_seal", "generated_utc": utc_now(),
        "freeze_sha256": freeze_sha,
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "d1_source_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "post_recovery_manifest_sha256": config.inputs["post_recovery_manifest"]["sha256"],
        "retention_replay_manifest_sha256": config.inputs["retention_replay_manifest"]["sha256"],
        "task_config_sha256": config.sha256,
        "training_summary_snapshot_sha256": artifacts["training_summary_snapshot"]["sha256"],
        "live_attempt_count_before_seal": 0,
        "model_changes_after_seal_permitted": False,
    }
    external_seal = config.artifact_root / "freeze_seal.json"
    compact_seal = config.result_dir(repo, "training") / "freeze_seal.json"
    write_json(external_seal, seal)
    write_json(compact_seal, seal)
    report = {
        "version": TRAINING_VERSION, "generated_utc": utc_now(), "result": "PASS",
        "task_config_sha256": config.sha256,
        "architecture": config.payload["architecture"],
        "parameter_contract": contract,
        "adaptation": adaptation,
        "offline_retention": offline,
        "leakage_audit_before_S09": leakage,
        "onnx_contract": {
            "checker": "PASS", "input": ["batch", 9, 66, 200],
            "output": ["batch", 1],
        },
        "onnx_equivalence": equivalence,
        "artifacts": artifacts,
        "freeze": {
            "path": str(external_freeze), "compact_path": str(compact_freeze),
            "sha256": freeze_sha,
        },
        "freeze_seal": {
            "path": str(external_seal), "compact_path": str(compact_seal),
            "sha256": sha256_file(external_seal),
        },
        "model_frozen_before_live": True,
        "model_changes_after_freeze": False,
        "prior_dagger2_coverage_gate": "FAIL_UNCHANGED",
        "d2_fe_negative_classification": "REGRESSION",
        "disk_after_freeze": disk_state("/"),
    }
    write_json(summary_path, report)
    return verify_frozen_d1_r(repo, config)


def live_retry_decision(classification: str, attempt_number: int) -> str:
    if classification == "RANDOM_CONE_POLICY_PASS":
        return "FINALIZE_PASS"
    if classification == "RANDOM_CONE_POLICY_FAIL":
        return "FINALIZE_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < 2:
        return "REPLACE_INFRA"
    return "STOP_INFRA"


def next_validation(records: Sequence[Mapping[str, Any]]) -> str | None:
    by_id = {item.get("scenario_id"): item for item in records}
    if "09" not in by_id:
        return "09"
    if by_id["09"].get("classification") != "RANDOM_CONE_POLICY_PASS":
        return None
    if "10" not in by_id:
        return "10"
    return None


def next_holdout(records: Sequence[Mapping[str, Any]]) -> str | None:
    by_id = {item.get("scenario_id"): item for item in records}
    if "11" not in by_id:
        return "11"
    if by_id["11"].get("classification") != "RANDOM_CONE_POLICY_PASS":
        return None
    if "12" not in by_id:
        return "12"
    return None


def validation_allows_holdout(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("result") == "PASS"
        and [item.get("scenario_id") for item in report.get("scenarios", [])]
        == list(VALIDATION_SCENARIOS)
        and all(
            item.get("classification") == "RANDOM_CONE_POLICY_PASS"
            for item in report.get("scenarios", [])
        )
    )


def _valid_live_record(
    record: Mapping[str, Any], scenario: str, role: str, frozen: Mapping[str, Any],
) -> bool:
    return bool(
        record.get("version") == LIVE_VERSION + "_scenario"
        and record.get("scenario_id") == scenario
        and record.get("role") == role
        and record.get("classification") in VALID_POLICY_CLASSIFICATIONS
        and record.get("onnx_sha256") == frozen["artifacts"]["onnx"]["sha256"]
        and record.get("checkpoint_sha256") == frozen["artifacts"]["checkpoint"]["sha256"]
        and record.get("freeze_seal_sha256") == frozen["freeze_seal"]["sha256"]
        and (record.get("run") or {}).get("safe_stop_success") is True
        and record.get("model_frozen_before_attempt") is True
        and record.get("valid_policy_run_number") == 1
        and record.get("bags_collected") == 0
        and record.get("camera_images_persisted") == 0
        and record.get("expert_labels_generated") == 0
    )


def _protected_holdout_world(world: str | None) -> bool:
    value = str(world or "")
    return value.endswith("_11") or value.endswith("_12")


def _live_group(
    repo: Path, sim_root: Path, config: AdaptationConfig, *, group: str,
) -> dict[str, Any]:
    if group not in {"validation", "holdout"}:
        raise ValueError(group)
    disk_before = disk_gate(config, live=True)
    frozen = verify_frozen_d1_r(repo, config)
    offline = frozen["offline_retention"]
    if offline.get("structural_retention_check", {}).get("result") != "PASS":
        raise AdaptationGateError("OFFLINE_RETENTION_FAIL blocks live driving")
    if group == "validation":
        scenario_ids: Sequence[str] = VALIDATION_SCENARIOS
        role = "VALIDATION"
        leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    else:
        validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
        if not validation_path.is_file():
            raise AdaptationGateError("S09/S10 evidence is absent; S11/S12 remain untouched")
        validation = _read_json(validation_path)
        if not validation_allows_holdout(validation):
            raise AdaptationGateError("S09/S10 did not both pass; S11/S12 remain untouched")
        scenario_ids = HOLDOUT_SCENARIOS
        role = "UNSEEN_HOLDOUT"
        leakage = leakage_audit(repo, sim_root, config, stage="before_holdout")
    train_task_ref = config.inputs["expert_task_config"]
    train_task_path = _resolve(repo, train_task_ref["path"])
    _hash_gate(train_task_path, train_task_ref["sha256"], "frozen Expert task config")
    train_task = load_task_config(train_task_path, repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in scenario_ids
    }
    if set(bundles) != set(scenario_ids):
        raise AdaptationGateError(f"frozen {group} scenario bundle set changed")
    r1_config = load_r1_config(repo / config.dagger1.r1["task_config_path"], repo)
    model = TemporalOnnxModel(Path(frozen["artifacts"]["onnx"]["path"]))
    result_dir = config.result_dir(repo, "live")
    attempts_dir = result_dir / f"{group}_attempts"
    scenarios_dir = result_dir / f"{group}_scenarios"
    states_dir = result_dir / f"{group}_states"
    summary_path = result_dir / f"live_{group}_summary.json"
    records: list[dict[str, Any]] = []
    for scenario in scenario_ids:
        final_path = scenarios_dir / f"scenario_{scenario}.json"
        if final_path.is_file():
            existing = _read_json(final_path)
            if not _valid_live_record(existing, scenario, role, frozen):
                raise AdaptationGateError(f"completed D1-R S{scenario} live identity changed")
            records.append(existing)
    if [item["scenario_id"] for item in records] != list(scenario_ids[:len(records)]):
        raise AdaptationGateError(f"existing {group} results are out of gate order")
    if group == "validation" and next_validation(records) is None and len(records) < 2:
        pending: Sequence[str] = ()
    elif group == "holdout" and next_holdout(records) is None and len(records) < 2:
        pending = ()
    else:
        pending = scenario_ids
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    if group == "validation" and _protected_holdout_world(original_world):
        errors = client.safe_stop()
        raise AdaptationGateError(
            "refusing validation while a protected S11/S12 world is active"
            + ("; safe-stop errors: " + "; ".join(errors) if errors else "")
        )
    infrastructure_replacements: list[dict[str, Any]] = []
    final_errors: list[str] = []
    restoration: dict[str, Any] = {"result": "NOT_REQUIRED"}
    try:
        for scenario in pending:
            if any(item["scenario_id"] == scenario for item in records):
                continue
            if group == "validation" and scenario == "10" and (
                not records or records[-1]["classification"] != "RANDOM_CONE_POLICY_PASS"
            ):
                break
            if group == "holdout" and scenario == "12" and (
                not records or records[-1]["scenario_id"] != "11"
                or records[-1]["classification"] != "RANDOM_CONE_POLICY_PASS"
            ):
                break
            verify_frozen_d1_r(repo, config)
            state_path = states_dir / f"scenario_{scenario}.json"
            final_path = scenarios_dir / f"scenario_{scenario}.json"
            attempt_paths = sorted(attempts_dir.glob(f"scenario_{scenario}_attempt_*.json"))
            if attempt_paths:
                latest = _read_json(attempt_paths[-1])
                if latest.get("classification") in VALID_POLICY_CLASSIFICATIONS:
                    if not _valid_live_record(latest, scenario, role, frozen):
                        raise AdaptationGateError(f"captured D1-R S{scenario} evidence changed")
                    write_json(final_path, latest)
                    records.append(latest)
                    if latest["classification"] != "RANDOM_CONE_POLICY_PASS":
                        break
                    continue
            attempts_consumed = len(attempt_paths)
            if state_path.is_file():
                state = _read_json(state_path)
                started = int(state.get("attempt_number", 0))
                if state.get("status") == "STARTED_UNFINALIZED" and started > attempts_consumed:
                    interrupted = attempts_dir / f"scenario_{scenario}_attempt_{started:02d}.json"
                    write_json(interrupted, {
                        "version": LIVE_VERSION + "_interrupted", "generated_utc": utc_now(),
                        "scenario_id": scenario, "role": role, "attempt_number": started,
                        "classification": "INFRA_FAIL", "result": "FAIL",
                        "failure_reason": "process ended before finalized live evidence",
                    })
                    attempts_consumed = started
            if attempts_consumed >= 2:
                break
            attempt_number = attempts_consumed + 1
            while attempt_number <= 2:
                frozen_now = verify_frozen_d1_r(repo, config)
                write_json(state_path, {
                    "status": "STARTED_UNFINALIZED", "scenario_id": scenario,
                    "role": role, "attempt_number": attempt_number,
                    "started_utc": utc_now(),
                    "onnx_sha256": frozen_now["artifacts"]["onnx"]["sha256"],
                    "freeze_seal_sha256": frozen_now["freeze_seal"]["sha256"],
                })
                record: dict[str, Any] = {
                    "version": LIVE_VERSION + "_scenario", "generated_utc": utc_now(),
                    "scenario_id": scenario, "role": role,
                    "attempt_number": attempt_number, "valid_policy_run_number": None,
                    "classification": "INFRA_FAIL", "result": "FAIL",
                    "failure_reason": None,
                    "onnx_sha256": frozen_now["artifacts"]["onnx"]["sha256"],
                    "checkpoint_sha256": frozen_now["artifacts"]["checkpoint"]["sha256"],
                    "freeze_sha256": frozen_now["freeze"]["sha256"],
                    "freeze_seal_sha256": frozen_now["freeze_seal"]["sha256"],
                    "model_frozen_before_attempt": True,
                    "d1_r_controls_vehicle": True, "expert_control_authority": False,
                    "bags_collected": 0, "camera_images_persisted": 0,
                    "expert_labels_generated": 0,
                    "preflight": None, "world_activation": None, "run": None,
                }
                try:
                    live = run_live_once(
                        client, model, r1_config, expert, bundles[scenario], sim_root,
                    )
                    record["preflight"] = live["preflight"]
                    record["world_activation"] = live["world_activation"]
                    record["run"] = live["run"]
                    record["classification"] = live["run"]["classification"]
                    record["result"] = (
                        "PASS" if record["classification"] == "RANDOM_CONE_POLICY_PASS"
                        else "FAIL"
                    )
                    if record["classification"] in VALID_POLICY_CLASSIFICATIONS:
                        record["valid_policy_run_number"] = 1
                except BaseException as exc:
                    errors = client.safe_stop()
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    record["safe_stop_after_exception_success"] = not errors
                    record["safe_stop_after_exception_errors"] = errors
                attempt_path = attempts_dir / f"scenario_{scenario}_attempt_{attempt_number:02d}.json"
                write_json(attempt_path, record)
                print(json.dumps({
                    "stage": f"d1_r_live_{group}", "scenario": scenario,
                    "attempt": attempt_number, "classification": record["classification"],
                    "completion": (record.get("run") or {}).get("route_completion_fraction"),
                }, sort_keys=True), flush=True)
                decision = live_retry_decision(record["classification"], attempt_number)
                if decision in {"FINALIZE_PASS", "FINALIZE_GENUINE_FAILURE"}:
                    if not _valid_live_record(record, scenario, role, frozen_now):
                        raise AdaptationGateError(f"D1-R S{scenario} policy evidence contract failed")
                    write_json(attempt_path, record)
                    write_json(final_path, record)
                    write_json(state_path, {
                        "status": "FINALIZED_VALID_POLICY_EVALUATION",
                        "scenario_id": scenario, "role": role,
                        "attempt_number": attempt_number,
                        "classification": record["classification"],
                        "finalized_utc": utc_now(), "do_not_repeat": True,
                    })
                    records.append(record)
                    break
                if decision == "REPLACE_INFRA":
                    errors = client.safe_stop()
                    if errors:
                        record["failure_reason"] = (
                            str(record.get("failure_reason") or "")
                            + "; safe stop failed before bounded replacement: " + "; ".join(errors)
                        )
                        write_json(attempt_path, record)
                        break
                    infrastructure_replacements.append({
                        "scenario_id": scenario, "invalid_attempt": attempt_number,
                        "replacement_attempt": attempt_number + 1,
                    })
                    attempt_number += 1
                    continue
                break
            if not records or records[-1].get("scenario_id") != scenario:
                break
            if records[-1]["classification"] != "RANDOM_CONE_POLICY_PASS":
                break
    finally:
        final_errors = client.safe_stop()
        try:
            restoration = _restore_world(client, original_world)
        except BaseException as exc:
            restoration = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"}
    pass_count = sum(item.get("classification") == "RANDOM_CONE_POLICY_PASS" for item in records)
    policy_failure = next(
        (item for item in records if item.get("classification") == "RANDOM_CONE_POLICY_FAIL"),
        None,
    )
    if policy_failure:
        result = "FAIL"
        category = VALIDATION_FAIL if group == "validation" else UNSEEN_FAIL
    elif pass_count == 2 and len(records) == 2 and not final_errors and restoration.get("result") == "PASS":
        result = "PASS"
        category = "VALIDATION_PASS" if group == "validation" else "UNSEEN_PASS"
    else:
        result = INCONCLUSIVE
        category = INCONCLUSIVE
    report = {
        "version": LIVE_VERSION + f"_{group}", "generated_utc": utc_now(),
        "result": result, "category": category, "role": role,
        "intended_scenario_ids": list(scenario_ids), "scenarios": records,
        "valid_policy_run_count": len(records), "pass_count": pass_count,
        "maximum_valid_policy_runs_per_scenario": 1,
        "maximum_infrastructure_replacements_per_scenario": 1,
        "infrastructure_replacements": infrastructure_replacements,
        "model_frozen_before_all_attempts": True,
        "onnx_sha256": frozen["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": frozen["artifacts"]["checkpoint"]["sha256"],
        "freeze_seal_sha256": frozen["freeze_seal"]["sha256"],
        "leakage_audit": leakage, "frozen_expert_metadata_audit": expert_audit,
        "disk_before_live": disk_before,
        "final_safe_stop_success": not final_errors,
        "final_safe_stop_errors": final_errors,
        "world_restoration": restoration,
        "bags_collected": 0, "camera_images_persisted": 0,
        "expert_training_labels_generated": 0,
    }
    write_json(summary_path, report)
    return report


def live_validation_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="validation")


def live_holdout_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="holdout")


def classify_final_category(
    offline: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    holdout: Mapping[str, Any] | None,
) -> str:
    if not offline or offline.get("structural_retention_check", {}).get("result") != "PASS":
        return OFFLINE_RETENTION_FAIL
    if not validation or validation.get("result") == INCONCLUSIVE:
        return INCONCLUSIVE
    if validation.get("result") == "FAIL":
        return VALIDATION_FAIL
    if validation.get("result") != "PASS":
        return INCONCLUSIVE
    if not holdout or holdout.get("result") == INCONCLUSIVE:
        return INCONCLUSIVE
    if holdout.get("result") == "FAIL":
        return UNSEEN_FAIL
    if holdout.get("result") == "PASS" and holdout.get("category") == "UNSEEN_PASS":
        return FULL_PASS
    return INCONCLUSIVE


def _live_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in records:
        run = item.get("run") or {}
        output[str(item["scenario_id"])] = {
            "role": item.get("role"), "classification": item.get("classification"),
            "route_completion_fraction": run.get("route_completion_fraction"),
            "final_route_s_m": run.get("final_route_s_m"),
            "total_unwrapped_progress_m": run.get("total_unwrapped_progress_m"),
            "minimum_footprint_to_cone_clearance_m": run.get("minimum_footprint_to_cone_clearance_m"),
            "cone_contact_or_intersection_occurred": run.get("cone_contact_or_intersection_occurred"),
            "recovery_success": run.get("recovery_success"),
            "recovery_time_s": run.get("recovery_time_s"),
            "off_track_events": run.get("off_track_events"),
            "off_track_total_duration_s": run.get("off_track_total_duration_s"),
            "temporal_invalid_history_count": run.get("temporal_invalid_history_count"),
            "api_failures": run.get("api_failures"), "pose_failures": run.get("pose_failures"),
            "clock_failures": run.get("clock_failures"),
            "liveness_failures": run.get("liveness_failures"),
            "safe_stop_success": run.get("safe_stop_success"),
        }
    return output


def final_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    training = verify_frozen_d1_r(repo, config)
    offline = training["offline_retention"]
    live_root = config.result_dir(repo, "live")
    validation_path = live_root / "live_validation_summary.json"
    holdout_path = live_root / "live_holdout_summary.json"
    validation = _read_json(validation_path) if validation_path.is_file() else None
    holdout = _read_json(holdout_path) if holdout_path.is_file() else None
    category = classify_final_category(offline, validation, holdout)
    if holdout is not None and not validation_allows_holdout(validation or {}):
        raise AdaptationGateError("holdout evidence exists without S09/S10 PASS authorization")
    final_leakage = leakage_audit(
        repo, sim_root, config,
        stage="before_holdout" if validation_allows_holdout(validation or {}) else "final_validation_only",
    )
    audit = _read_json(config.result_dir(repo, "training") / "audit.json")
    prior_after = prior_fe._prior_tree_gate(repo, config.prior_frontier)
    prior_before = audit["prior_dagger2_coverage_gate"]["directory_identities_before"]
    if prior_after != prior_before:
        raise AdaptationGateError("prior DAgger2 evidence changed during D1-R experiment")
    records = [
        *(validation.get("scenarios", []) if validation else []),
        *(holdout.get("scenarios", []) if holdout else []),
    ]
    report = {
        "version": LIVE_VERSION, "generated_utc": utc_now(),
        "result": category, "final_category": category,
        "training": training, "offline_retention": offline,
        "live_validation": validation, "live_holdout": holdout,
        "live_metrics": _live_metrics(records),
        "prior_dagger2_coverage_gate": {
            "result": "FAIL", "training_authorized": False,
            "sequence_count": 109, "sequences_after_20m": 18,
            "sequences_after_26m": 0, "proof_unchanged": True,
            "directory_identities_before": prior_before,
            "directory_identities_after": prior_after,
        },
        "d2_fe_negative_experiment": audit["d2_fe_negative_evidence"],
        "leakage_audit": final_leakage,
        "holdout_protection": {
            "authorized_only_after_S09_S10_PASS": True,
            "touched_before_gate": False,
            "camera_or_steering_rows_used_for_training": 0,
        },
        "single_adaptation_run": True, "retraining_after_observation": False,
        "new_data_collection_performed": False, "dagger3_started": False,
        "commit_performed": False, "push_performed": False,
        "real_robot_success_claimed": False,
        "disk_final": disk_state("/"),
        "limitations": [
            "The 109 post-recovery sequences contain no targets beyond route s=26 m.",
            "Offline retention measurements are diagnostic; closed-loop validation is decisive.",
            "One simulator run per scenario does not establish repeatability.",
            "Simulator results are not real-robot evidence.",
        ],
    }
    write_json(live_root / "summary.json", report)
    return report


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _final_markdown(report: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    offline = report["offline_retention"]
    post = offline["datasets"]["DAGGER2_POST_RECOVERY_109"]
    aggregate = offline["datasets"]["original_D1_aggregate"]
    validation_rows: list[str] = []
    for name in ("S09", "S10"):
        item = offline["datasets"][name]
        validation_rows.append(
            f"| {name} | {item['sample_count']} | {_fmt(item['D1']['mae_rad'])} | "
            f"{_fmt(item['D1_R']['mae_rad'])} | "
            f"{_fmt(item['adapted_minus_D1_prediction_delta_rad']['mean_absolute'])} | "
            f"{_fmt(item['adapted_minus_D1_prediction_delta_rad']['max_absolute'])} |"
        )
    live_rows: list[str] = []
    for scenario, item in report["live_metrics"].items():
        live_rows.append(
            f"| S{scenario} | {item['role']} | {item['classification']} | "
            f"{_fmt(item['route_completion_fraction'], 4)} | {_fmt(item['final_route_s_m'])} | "
            f"{_fmt(item['minimum_footprint_to_cone_clearance_m'])} | "
            f"{item['recovery_success']} | {item['safe_stop_success']} |"
        )
    status = verification.get("repository_status", [])
    training = report["training"]
    lines = [
        "# D1-Preserving Post-Recovery Adaptation V1", "",
        f"Final category: **{report['final_category']}**", "",
        "## Frozen inputs and adaptation", "",
        f"Frozen D1 source checkpoint: `{training['adaptation']['d1_source_checkpoint_sha256']}`. "
        f"Adapted D1-R checkpoint: `{training['artifacts']['checkpoint']['sha256']}`.",
        "The model initialized exactly from D1, kept all 134,948 convolutional parameters bitwise unchanged, "
        "and trained only the 120,871-parameter fully-connected steering head for exactly five epochs.", "",
        "The loss was normalized post-recovery Expert MSE + normalized frozen-D1 retention MSE at coefficients 1.0/1.0. "
        "No validation labels, holdout data, new collection, scratch initialization, D2-FE initialization, or DAgger3 were used.", "",
        "## Offline retention", "",
        f"Structural retention: **{offline['structural_retention_check']['result']}**. "
        f"Post-recovery MAE changed from {_fmt(post['D1']['mae_rad'])} to {_fmt(post['D1_R']['mae_rad'])} rad. "
        f"Aggregate MAE changed from {_fmt(aggregate['D1']['mae_rad'])} to {_fmt(aggregate['D1_R']['mae_rad'])} rad; "
        f"aggregate mean/max |D1-R − D1| were "
        f"{_fmt(aggregate['adapted_minus_D1_prediction_delta_rad']['mean_absolute'])}/"
        f"{_fmt(aggregate['adapted_minus_D1_prediction_delta_rad']['max_absolute'])} rad.", "",
        "| Dataset | Samples | D1 MAE | D1-R MAE | Mean abs delta | Max abs delta |",
        "|---|---:|---:|---:|---:|---:|", *validation_rows, "",
        "Full phase, route-bin, RMSE, bias, corrective-ratio, and sign-disagreement metrics are in `offline_retention.json`.", "",
        "## Export and freeze", "",
        f"Checkpoint: `{training['artifacts']['checkpoint']['sha256']}`. ONNX: "
        f"`{training['artifacts']['onnx']['sha256']}`. ONNX checker/equivalence: "
        f"PASS/{training['onnx_equivalence']['result']}. Freeze seal: `{training['freeze_seal']['sha256']}`.", "",
        "## Strictly gated simulator evaluation", "",
        "| Scenario | Role | Result | Completion | Route s m | Clearance m | Recovery | Safe stop |",
        "|---|---|---|---:|---:|---:|---:|---:|", *live_rows,
        "" if live_rows else "No infrastructure-valid policy result was available.", "",
        "The historical DAgger2 coverage gate remains FAIL (109 total, 18 beyond 20 m, zero beyond 26 m). "
        "D2-FE remains a frozen REGRESSION and was not used for initialization.", "",
        "## Verification", "",
        f"Tests: {verification.get('tests', {}).get('summary', 'pending')}. `git diff --check`: "
        f"{verification.get('git_diff_check', {}).get('result', 'pending')}. No commit or push occurred. "
        "These are simulator results, not real-robot evidence.", "",
        "Final Git status:", "", "```text", *status, "```", "",
    ]
    return "\n".join(lines)


def test_stage(repo: Path, config: AdaptationConfig) -> dict[str, Any]:
    focused = repo / "tests/test_random_cone_d1_preserving_recovery.py"
    commands = (
        [sys.executable, "-m", "pytest", "-q", str(focused)],
        [sys.executable, "-m", "pytest", "-q"],
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        process = subprocess.run(
            command, cwd=repo, text=True, capture_output=True, check=False,
        )
        item = {
            "command": command, "returncode": process.returncode,
            "stdout_tail": process.stdout.splitlines()[-12:],
            "stderr_tail": process.stderr.splitlines()[-12:],
        }
        results.append(item)
        if process.returncode:
            raise AdaptationGateError(
                f"test gate failed: {' '.join(command)}\n{process.stdout}\n{process.stderr}"
            )
    report = {
        "version": VERSION + "_pretraining_tests", "generated_utc": utc_now(),
        "result": "PASS", "commands": results,
    }
    write_json(config.result_dir(repo, "training") / "pretraining_tests.json", report)
    return report


def verification_stage(repo: Path, sim_root: Path, config: AdaptationConfig) -> dict[str, Any]:
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    summary_line = next(
        (line for line in reversed(tests.stdout.splitlines()) if " passed" in line),
        "pytest summary unavailable",
    )
    simulator = simulator_tracked_status(sim_root)
    status = _git(repo, "status", "--short", "--branch").splitlines()
    result = {
        "version": VERSION + "_verification", "generated_utc": utc_now(),
        "result": "PASS" if tests.returncode == 0 and diff.returncode == 0
        and simulator.get("result") == "PASS" else "FAIL",
        "tests": {
            "result": "PASS" if tests.returncode == 0 else "FAIL",
            "returncode": tests.returncode, "summary": summary_line,
            "stdout_tail": tests.stdout.splitlines()[-20:],
            "stderr_tail": tests.stderr.splitlines()[-20:],
        },
        "git_diff_check": {
            "result": "PASS" if diff.returncode == 0 else "FAIL",
            "returncode": diff.returncode, "stdout": diff.stdout,
            "stderr": diff.stderr,
        },
        "simulator_tracked_source": simulator,
        "repository_status": status,
        "external_artifacts_root": str(config.artifact_root),
        "commit_performed": False, "push_performed": False,
    }
    live_root = config.result_dir(repo, "live")
    write_json(live_root / "verification.json", result)
    summary_path = live_root / "summary.json"
    if summary_path.is_file():
        report = _read_json(summary_path)
        _write_text(live_root / "REPORT.md", _final_markdown(report, result))
    if result["result"] != "PASS":
        raise AdaptationGateError("final regression/diff/simulator-source verification failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "audit", "replay", "test", "train", "offline", "freeze", "leakage",
        "live-validation", "live-holdout", "final", "verify", "all",
    ))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    sim_root = args.sim_root.resolve()
    config_path = (
        args.config or repo / "configs/random_cone_d1_preserving_recovery_1p0_v1.json"
    ).resolve()
    config = load_config(config_path, repo)
    if args.stage == "audit":
        result = audit_stage(repo, sim_root, config)
    elif args.stage == "replay":
        result = replay_identity_stage(repo, sim_root, config)
    elif args.stage == "test":
        result = test_stage(repo, config)
    elif args.stage == "train":
        result = training_stage(repo, sim_root, config)
    elif args.stage == "offline":
        result = offline_stage(repo, sim_root, config)
    elif args.stage == "freeze":
        result = freeze_stage(repo, sim_root, config)
    elif args.stage == "leakage":
        result = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    elif args.stage == "live-validation":
        result = live_validation_stage(repo, sim_root, config)
    elif args.stage == "live-holdout":
        result = live_holdout_stage(repo, sim_root, config)
    elif args.stage == "final":
        result = final_stage(repo, sim_root, config)
    elif args.stage == "verify":
        result = verification_stage(repo, sim_root, config)
    else:
        audit_stage(repo, sim_root, config)
        replay_identity_stage(repo, sim_root, config)
        test_stage(repo, config)
        training_stage(repo, sim_root, config)
        offline = offline_stage(repo, sim_root, config)
        if offline.get("structural_retention_check", {}).get("result") != "PASS":
            result = offline
        else:
            freeze_stage(repo, sim_root, config)
            validation = live_validation_stage(repo, sim_root, config)
            if validation.get("result") == "PASS":
                live_holdout_stage(repo, sim_root, config)
            result = final_stage(repo, sim_root, config)
            verification_stage(repo, sim_root, config)
    print(json.dumps({"stage": args.stage, "result": result.get("result")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
