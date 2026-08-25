"""Random-Cone D2 Frontier Expansion V1 at 1.00 m/s.

This experiment never collects training data.  It evaluates a separately
preregistered frontier-expansion hypothesis using the immutable 109-sequence
DAgger2 post-recovery dataset whose original >26 m coverage gate failed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset_extractor import canonical_json_bytes, sha256_file
from .high_speed_temporal import (
    TemporalOnnxModel,
    export_temporal_onnx,
    metrics as error_metrics,
    predict_temporal,
    validate_equivalence,
)
from .pilotnet_temporal import TEMPORAL_PARAMETER_COUNT, build_temporal_pilotnet
from .random_cone_dagger1 import (
    Dagger1Config,
    _load_temporal_checkpoint,
    _phase,
    _read_model_rows,
    load_config as load_dagger1_config,
)
from .random_cone_dagger2_post_recovery import (
    AGGREGATE_FIELDS,
    HOLDOUT_SCENARIOS,
    PROVENANCE,
    ROUTE_BINS,
    TRAIN_SCENARIOS,
    VALIDATION_SCENARIOS,
    _csv_rows,
    _directory_tree_identity,
    _route_bin_key,
    _validate_episode_dataset,
    episode_specs,
    load_config as load_dagger2_config,
    validate_existing_episode,
)
from .random_cone_expert import ScenarioBundle, _restore_world, simulator_tracked_status
from .random_cone_temporal_r1 import (
    load_config as load_r1_config,
    run_live_once,
    train_temporal_resumable,
)
from .random_cone_train_data import audit_frozen_expert, disk_state, load_task_config
from .sim_client import SimClient


VERSION = "random_cone_d2_frontier_expansion_1p0_v1"
TRAINING_VERSION = "pilotnet_training_d2_fe_random_cone_1p0"
LIVE_VERSION = "pilotnet_e2e_d2_fe_random_cone_1p0"
EXPECTED_BRANCH = "experiment/random-cone-d2-frontier-expansion-1p0-v1"
EXPECTED_SEQUENCE_COUNT = 8_298
EXPECTED_PROVENANCE = {
    "EXPERT_BASELINE": 6_706,
    "DAGGER1": 1_483,
    PROVENANCE: 109,
}
VALID_POLICY_CLASSIFICATIONS = (
    "RANDOM_CONE_POLICY_PASS",
    "RANDOM_CONE_POLICY_FAIL",
)

D2_FE_FULL_PASS = "D2_FE_FULL_PASS"
D2_FE_VALIDATION_FAIL = "D2_FE_VALIDATION_FAIL"
D2_FE_UNSEEN_FAIL = "D2_FE_UNSEEN_FAIL"
FRONTIER_EXPANSION_PARTIAL_SUPPORT = "FRONTIER_EXPANSION_PARTIAL_SUPPORT"
NO_CLEAR_FRONTIER_EXPANSION = "NO_CLEAR_FRONTIER_EXPANSION"
REGRESSION = "REGRESSION"
INCONCLUSIVE = "INCONCLUSIVE"


class FrontierGateError(RuntimeError):
    """A preregistered identity, data, training, or live gate failed."""


@dataclass(frozen=True)
class FrontierConfig:
    path: Path
    payload: dict[str, Any]
    prior_dagger2: Any
    dagger1: Dagger1Config

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def inputs(self) -> dict[str, Any]:
        return self.payload["frozen_inputs"]

    @property
    def training(self) -> dict[str, Any]:
        return self.payload["training"]

    def result_dir(self, repo: Path, key: str) -> Path:
        return repo / self.payload["result_directories"][key]

    def external_root(self, sim_root: Path) -> Path:
        return sim_root / "userdata" / self.payload["external_relative_root"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrontierGateError(f"JSON root is not an object: {path}")
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=AGGREGATE_FIELDS, extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field)
                             for field in AGGREGATE_FIELDS})
    temporary.replace(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _hash_gate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FrontierGateError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise FrontierGateError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.rstrip()


def disk_gate(config: FrontierConfig, *, live: bool = False) -> dict[str, Any]:
    gate = config.payload["disk_gate"]
    required = int(
        gate["minimum_before_live_bytes" if live else "minimum_before_training_bytes"]
    )
    report = disk_state(gate["path"])
    report.update({
        "required_available_bytes": required,
        "required_available_gib": required / 1024**3,
        "result": "PASS" if int(report["available_bytes"]) >= required else "FAIL",
        "df_h": subprocess.run(
            ["df", "-h", gate["path"]], text=True, capture_output=True, check=True,
        ).stdout.strip(),
    })
    if report["result"] != "PASS":
        raise FrontierGateError(
            f"disk gate failed: {report['available_bytes'] / 1024**3:.3f} GiB available; "
            f"{required / 1024**3:.3f} GiB required"
        )
    return report


def load_config(path: Path, repo: Path) -> FrontierConfig:
    payload = _read_json(path)
    required = {
        "version", "expected_branch", "expected_base_commit", "scientific_question",
        "hypothesis", "prior_negative_result", "frozen_inputs", "aggregate", "training",
        "offline", "frontier", "disk_gate", "external_relative_root",
        "result_directories", "live", "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise FrontierGateError("frontier-expansion config fields/version changed")
    if payload["expected_branch"] != EXPECTED_BRANCH:
        raise FrontierGateError("expected branch contract changed")
    if payload["aggregate"] != {
        "sequence_count": EXPECTED_SEQUENCE_COUNT,
        "provenance_counts": EXPECTED_PROVENANCE,
        "scenario_ids": list(TRAIN_SCENARIOS),
        "weighting": False,
        "balancing": False,
        "resampling": False,
    }:
        raise FrontierGateError("exact frontier aggregate contract changed")
    expected_training = {
        "seed": 20260824, "image_width": 200, "image_height": 66,
        "input_channels": 9, "history_frames": 3, "maximum_adjacent_gap_s": 0.12,
        "max_steering_rad": 0.349066,
        "target": "frozen_expert_steering_normalized_at_t",
        "optimizer": "Adam", "loss": "MSE", "learning_rate": 0.001,
        "batch_size": 64, "max_epochs": 35, "early_stopping_patience": 7,
        "minimum_improvement": 0.000001, "initialization": "from_scratch",
        "augmentation": False, "sample_weighting": False, "source_weighting": False,
        "scenario_weighting": False, "oversampling": False, "undersampling": False,
        "hyperparameter_sweep": False, "onnx_opset": 17,
        "onnx_equivalence_samples": 128,
        "onnx_mean_abs_difference_limit": 0.00001,
        "onnx_max_abs_difference_limit": 0.0001,
    }
    if payload["training"] != expected_training:
        raise FrontierGateError("D2-FE scratch training contract changed")
    permissions = payload["permissions"]
    if permissions != {
        "new_data_collection_permitted": False,
        "new_expert_rollouts_permitted": False,
        "new_dagger_rollouts_permitted": False,
        "validation_recollection_permitted": False,
        "d2_fe_logical_training_runs_permitted": 1,
        "retraining_after_freeze_permitted": False,
        "dagger3_permitted": False,
        "holdout_access_before_validation_pass_permitted": False,
        "commit_permitted": False,
        "push_permitted": False,
    }:
        raise FrontierGateError("frontier-expansion permission boundary changed")
    if tuple(tuple(float(v) for v in item) for item in payload["offline"]["route_bins_m"]) != ROUTE_BINS:
        raise FrontierGateError("frozen route bins changed")
    prior_ref = payload["frozen_inputs"]["dagger2_config"]
    prior_path = _resolve(repo, prior_ref["path"])
    _hash_gate(prior_path, prior_ref["sha256"], "prior DAgger2 config")
    prior = load_dagger2_config(prior_path, repo)
    d1_ref = payload["frozen_inputs"]["dagger1_config"]
    d1_path = _resolve(repo, d1_ref["path"])
    _hash_gate(d1_path, d1_ref["sha256"], "DAgger1 config")
    return FrontierConfig(path.resolve(), payload, prior, load_dagger1_config(d1_path, repo))


def _prior_tree_gate(repo: Path, config: FrontierConfig) -> dict[str, Any]:
    prior = config.payload["prior_negative_result"]
    output: dict[str, Any] = {}
    for key in ("collection", "dataset"):
        path = _resolve(repo, prior[f"{key}_directory"])
        identity = _directory_tree_identity(path)
        if identity["sha256"] != prior[f"{key}_directory_sha256"]:
            raise FrontierGateError(f"immutable prior {key} directory identity changed")
        output[key] = identity
    return output


def _audit_prior_failure(repo: Path, config: FrontierConfig) -> dict[str, Any]:
    prior = config.payload["prior_negative_result"]
    collection_path = _resolve(repo, prior["collection_summary"])
    dataset_path = _resolve(repo, prior["dataset_summary"])
    _hash_gate(collection_path, prior["collection_summary_sha256"], "prior collection summary")
    _hash_gate(dataset_path, prior["dataset_summary_sha256"], "prior dataset summary")
    collection = _read_json(collection_path)
    dataset = _read_json(dataset_path)
    gates = {
        "collection_passed": collection.get("result") == "PASS",
        "dataset_result_remains_fail": dataset.get("result") == "FAIL",
        "training_remains_unauthorized": dataset.get("training_authorized") is False,
        "sequence_count_exact": (dataset.get("dagger2_temporal_manifest") or {}).get("sequence_count") == 109,
        "after_20_exact": dataset.get("sequences_after_20m") == 18,
        "after_26_exact": dataset.get("sequences_after_26m") == 0,
        "historical_after_26_gate_remains_fail":
            (dataset.get("gates") or {}).get("at_least_one_sequence_after_s26") is False,
        "historical_aggregate_remains_unbuilt":
            (dataset.get("aggregate") or {}).get("status") == "NOT_BUILT_COVERAGE_OR_INTEGRITY_GATE_FAILED",
    }
    if not all(gates.values()):
        raise FrontierGateError("prior DAgger2 negative-result semantics changed")
    return {
        "result": "PASS", "gates": gates,
        "collection_summary_sha256": sha256_file(collection_path),
        "dataset_summary_sha256": sha256_file(dataset_path),
        "historical_result": "FAIL", "historical_training_authorized": False,
        "historical_requirement_after_26m": 1,
        "new_experiment_does_not_modify_historical_gate": True,
    }


def audit_dagger2_sequences(repo: Path, config: FrontierConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    collection_path = _resolve(
        repo, config.payload["prior_negative_result"]["collection_summary"],
    )
    collection = _read_json(collection_path)
    records = collection.get("episodes") or []
    if [item.get("episode_id") for item in records] != [item.episode_id for item in episode_specs()]:
        raise FrontierGateError("prior episode order is not exact S01-S08")
    metrics: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    compact_dir = _resolve(repo, config.payload["prior_negative_result"]["collection_directory"])
    for episode, summary_record in zip(episode_specs(), records, strict=True):
        path = compact_dir / "episodes" / f"{episode.episode_id}.json"
        record = validate_existing_episode(path, episode, config.prior_dagger2)
        if record is None:
            raise FrontierGateError(f"missing finalized prior episode {episode.episode_id}")
        metric, episode_rows = _validate_episode_dataset(record, episode)
        expected = next(
            item for item in _read_json(
                _resolve(repo, config.payload["prior_negative_result"]["dataset_summary"])
            )["episodes"] if item["episode_id"] == episode.episode_id
        )
        if any(metric[key] != expected[key] for key in (
            "policy_outcome", "cone_pass_success", "recovery_success",
            "temporal_sequence_count", "sequences_after_20m", "sequences_after_26m",
        )):
            raise FrontierGateError(f"episode dataset semantics changed: {episode.episode_id}")
        metrics.append(metric)
        rows.extend(episode_rows)
    manifest_ref = config.inputs["dagger2_temporal_manifest"]
    manifest_path = _resolve(repo, manifest_ref["path"])
    _hash_gate(manifest_path, manifest_ref["sha256"], "DAgger2 temporal manifest")
    manifest_rows = _csv_rows(manifest_path)
    if [row["sequence_id"] for row in rows] != [row["sequence_id"] for row in manifest_rows]:
        raise FrontierGateError("revalidated DAgger2 sequence order differs from frozen manifest")
    scenario_ids = sorted({str(row["scenario_id"]).zfill(2) for row in rows})
    at_or_below_20 = sum(float(row["route_s_m"]) <= 20.0 for row in rows)
    after_20 = sum(float(row["route_s_m"]) > 20.0 for row in rows)
    after_26 = sum(float(row["route_s_m"]) > 26.0 for row in rows)
    gates = {
        "exact_109_sequences": len(rows) == 109 == len(manifest_rows),
        "only_s01_s08": set(scenario_ids) <= set(TRAIN_SCENARIOS),
        "provenance_exact": all(row.get("provenance") == PROVENANCE for row in rows),
        "captured_after_cone_pass_and_recovery_pass": all(
            row.get("cone_phase") == "post_recovery"
            and row.get("recovery_state") == "PASS"
            and str(row.get("post_recovery_target")).lower() == "true"
            for row in rows
        ),
        "actual_d1_learner_states": all(
            str(row.get("teacher_uses_actual_learner_pose")).lower() == "true" for row in rows
        ),
        "frozen_expert_shadow_labels": all(
            row.get("target_steering_rad") not in (None, "") for row in rows
        ) and all(item.get("d1_vs_expert", {}).get("sample_count") == item["temporal_sequence_count"] for item in metrics),
        "future_label_violations_zero": sum(item["future_teacher_label_violations"] for item in metrics) == 0,
        "temporal_corruption_zero": sum(item["temporal_corruption_count"] for item in metrics) == 0,
        "duplicate_padding_zero": sum(item["duplicate_padding_count"] for item in metrics) == 0,
        "episode_boundary_crossings_zero": sum(item["episode_boundary_crossings"] for item in metrics) == 0,
        "no_s09_s12_leakage": not bool(set(scenario_ids) & set(VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS)),
        "historical_after_20_exact": after_20 == 18,
        "historical_after_26_exact": after_26 == 0,
    }
    if not all(gates.values()):
        raise FrontierGateError("DAgger2 109-sequence integrity audit failed")
    return ({
        "result": "PASS", "gates": gates, "episodes": metrics,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "sequence_count": len(rows), "scenario_ids": scenario_ids,
        "route_coverage": {"at_or_below_20m": at_or_below_20, "after_20m": after_20, "after_26m": after_26},
        "future_teacher_label_violations": 0, "temporal_corruption_count": 0,
        "new_collection_performed": False,
    }, rows)


def audit_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    branch = _git(repo, "branch", "--show-current")
    head = _git(repo, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != config.payload["expected_base_commit"]:
        raise FrontierGateError(f"branch/HEAD mismatch: {branch} at {head}")
    trees_before = _prior_tree_gate(repo, config)
    prior = _audit_prior_failure(repo, config)
    hashes: dict[str, Any] = {}
    for key, item in config.inputs.items():
        if key in {"r1", "d1"}:
            hashes[key] = {
                "checkpoint": _hash_gate(Path(item["checkpoint_path"]), item["checkpoint_sha256"], f"{key} checkpoint"),
                "onnx": _hash_gate(Path(item["onnx_path"]), item["onnx_sha256"], f"{key} ONNX"),
                "training_summary": _hash_gate(
                    _resolve(repo, item["training_summary_path"]), item["training_summary_sha256"],
                    f"{key} training summary",
                ),
            }
        elif key not in {"r1_s09_live", "d1_s09_live", "d1_cone_free"}:
            hashes[key] = _hash_gate(_resolve(repo, item["path"]), item["sha256"], key)
    for key in ("r1_s09_live", "d1_s09_live", "d1_cone_free"):
        item = config.inputs[key]
        hashes[key] = _hash_gate(_resolve(repo, item["path"]), item["sha256"], key)
    data_audit, _rows = audit_dagger2_sequences(repo, config)
    trees_after = _prior_tree_gate(repo, config)
    if trees_after != trees_before:
        raise FrontierGateError("prior DAgger2 evidence changed during read-only audit")
    report = {
        "version": VERSION + "_audit", "generated_utc": utc_now(), "result": "PASS",
        "branch": branch, "head": head, "task_config_sha256": config.sha256,
        "prior_negative_result": prior,
        "prior_directory_identities_before": trees_before,
        "prior_directory_identities_after": trees_after,
        "frozen_input_hashes": hashes, "dagger2_data": data_audit,
        "disk_before_training": disk_gate(config),
        "scientific_hypothesis": config.payload["hypothesis"],
        "new_data_collection_permitted": False, "new_data_collection_performed": False,
        "historical_coverage_gate_reinterpreted": False,
    }
    write_json(config.result_dir(repo, "training") / "audit.json", report)
    return report


def build_aggregate_stage(
    repo: Path, sim_root: Path, config: FrontierConfig,
) -> dict[str, Any]:
    audit = audit_stage(repo, sim_root, config)
    _data_audit, dagger2_rows = audit_dagger2_sequences(repo, config)
    d1_ref = config.inputs["dagger1_aggregate_manifest"]
    d1_path = _resolve(repo, d1_ref["path"])
    _hash_gate(d1_path, d1_ref["sha256"], "D1 aggregate")
    baseline = _csv_rows(d1_path)
    baseline_counts = {
        name: sum(row.get("provenance") == name for row in baseline)
        for name in ("EXPERT_BASELINE", "DAGGER1")
    }
    if len(baseline) != 8_189 or baseline_counts != {
        "EXPERT_BASELINE": 6_706, "DAGGER1": 1_483,
    }:
        raise FrontierGateError("frozen D1 aggregate provenance/count changed")
    source_hash = config.inputs["dagger2_temporal_manifest"]["sha256"]
    converted: list[dict[str, Any]] = []
    for row in dagger2_rows:
        scenario = str(row["scenario_id"]).zfill(2)
        if scenario not in TRAIN_SCENARIOS or row.get("provenance") != PROVENANCE:
            raise FrontierGateError("forbidden DAgger2 row entered frontier aggregate")
        converted.append({
            "sequence_id": row["sequence_id"], "provenance": PROVENANCE,
            "episode_id": row["episode_id"], "scenario_id": scenario,
            "scenario_role": "TRAIN", "repeat_id": "R01",
            "frame_t_minus_2": row["frame_t_minus_2"],
            "frame_t_minus_1": row["frame_t_minus_1"], "frame_t": row["frame_t"],
            "timestamp_t_minus_2_ns": row["timestamp_t_minus_2_ns"],
            "timestamp_t_minus_1_ns": row["timestamp_t_minus_1_ns"],
            "timestamp_t_ns": row["timestamp_t_ns"],
            "target_steering_rad": row["target_steering_rad"],
            "route_progress_m": row["route_s_m"], "cone_phase": "post_recovery",
            "source_mcap_sha256": "", "source_manifest_sha256": source_hash,
        })
    aggregate = [*baseline, *converted]
    counts = {
        name: sum(row.get("provenance") == name for row in aggregate)
        for name in EXPECTED_PROVENANCE
    }
    sequence_ids = [row["sequence_id"] for row in aggregate]
    train_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in aggregate})
    if (
        len(aggregate) != EXPECTED_SEQUENCE_COUNT
        or counts != EXPECTED_PROVENANCE
        or len(sequence_ids) != len(set(sequence_ids))
        or train_scenarios != list(TRAIN_SCENARIOS)
    ):
        raise FrontierGateError("exact 8,298-sequence aggregate gate failed")
    path = config.external_root(sim_root) / "aggregate" / "manifests" / "aggregate.csv"
    if path.is_file():
        expected_path = path.with_suffix(".expected.csv")
        _write_csv(expected_path, aggregate)
        expected_hash = sha256_file(expected_path)
        expected_path.unlink()
        if sha256_file(path) != expected_hash:
            raise FrontierGateError("existing frontier aggregate differs from deterministic rebuild")
    else:
        _write_csv(path, aggregate)
    identity = {
        "version": VERSION + "_aggregate_identity", "generated_utc": utc_now(),
        "result": "PASS", "path": str(path), "sha256": sha256_file(path),
        "sequence_count": len(aggregate), "provenance_counts": counts,
        "scenario_ids": train_scenarios, "excluded_scenarios": ["09", "10", "11", "12"],
        "d1_aggregate_manifest_sha256": d1_ref["sha256"],
        "dagger2_post_recovery_manifest_sha256": source_hash,
        "new_images_created": 0, "existing_images_duplicated": 0,
        "weighting": False, "balancing": False, "resampling": False,
        "only_new_training_information_relative_to_d1": "109 DAGGER2_POST_RECOVERY sequences",
        "prior_negative_result_directory_identities": audit["prior_directory_identities_after"],
    }
    identity_path = path.parent / "identity.json"
    write_json(identity_path, identity)
    identity["identity_path"] = str(identity_path)
    identity["identity_sha256"] = sha256_file(identity_path)
    write_json(config.result_dir(repo, "training") / "aggregate.json", identity)
    if _prior_tree_gate(repo, config) != audit["prior_directory_identities_before"]:
        raise FrontierGateError("prior DAgger2 evidence changed during aggregate construction")
    return identity


def leakage_audit(
    repo: Path, sim_root: Path, config: FrontierConfig, *, stage: str,
) -> dict[str, Any]:
    aggregate = _read_json(config.result_dir(repo, "training") / "aggregate.json")
    aggregate_path = Path(aggregate["path"])
    aggregate_rows = _csv_rows(aggregate_path)
    expert_path = _resolve(repo, config.inputs["expert_train_manifest"]["path"])
    d1_path = _resolve(repo, config.inputs["dagger1_aggregate_manifest"]["path"])
    d2_path = _resolve(repo, config.inputs["dagger2_temporal_manifest"]["path"])
    validation_path = _resolve(repo, config.inputs["validation_manifest"]["path"])
    expert_rows = _csv_rows(expert_path)
    d1_rows = _csv_rows(d1_path)
    d2_rows = _csv_rows(d2_path)
    validation_rows = _csv_rows(validation_path)
    provenance_counts = {
        name: sum(row.get("provenance") == name for row in aggregate_rows)
        for name in EXPECTED_PROVENANCE
    }
    train_rows = [*expert_rows, *d1_rows, *d2_rows, *aggregate_rows]
    train_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in aggregate_rows})
    validation_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in validation_rows})
    d2_ids = sorted({row["episode_id"] for row in d2_rows})
    gates = {
        "expert_hash_exact": sha256_file(expert_path) == config.inputs["expert_train_manifest"]["sha256"],
        "dagger1_hash_exact": sha256_file(d1_path) == config.inputs["dagger1_aggregate_manifest"]["sha256"],
        "dagger2_hash_exact": sha256_file(d2_path) == config.inputs["dagger2_temporal_manifest"]["sha256"],
        "aggregate_hash_exact": sha256_file(aggregate_path) == aggregate["sha256"],
        "validation_hash_exact": sha256_file(validation_path) == config.inputs["validation_manifest"]["sha256"],
        "exact_aggregate_count": len(aggregate_rows) == EXPECTED_SEQUENCE_COUNT,
        "exact_aggregate_provenance": provenance_counts == EXPECTED_PROVENANCE,
        "training_s01_s08_only": train_scenarios == list(TRAIN_SCENARIOS),
        "dagger2_episode_ids_s01_s08_only": all(
            item.startswith("dagger2_s") and item[9:11] in TRAIN_SCENARIOS for item in d2_ids
        ),
        "validation_s09_s10_only": validation_scenarios == list(VALIDATION_SCENARIOS),
        "s09_s10_absent_from_training": not any(
            str(row["scenario_id"]).zfill(2) in VALIDATION_SCENARIOS for row in train_rows
        ),
        "s11_s12_absent_from_training_and_validation": not any(
            str(row["scenario_id"]).zfill(2) in HOLDOUT_SCENARIOS
            for row in [*train_rows, *validation_rows]
        ),
        "no_weighting_balancing_resampling": all(
            aggregate.get(key) is False for key in ("weighting", "balancing", "resampling")
        ),
        "prior_failure_still_immutable": _audit_prior_failure(repo, config)["result"] == "PASS",
    }
    report = {
        "version": VERSION + "_leakage_audit", "generated_utc": utc_now(),
        "stage": stage, "result": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "aggregate": {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path),
                      "sequence_count": len(aggregate_rows), "provenance_counts": provenance_counts,
                      "scenario_ids": train_scenarios},
        "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path),
                       "sequence_count": len(validation_rows), "scenario_ids": validation_scenarios},
        "dagger2_episode_ids": d2_ids,
        "holdout_protection": {
            "scenario_ids": list(HOLDOUT_SCENARIOS),
            "camera_content_inspected": False, "expert_labels_generated": 0,
            "bags_collected": 0, "training_or_validation_rows": 0,
        },
        "new_data_collection_performed": False,
    }
    write_json(config.result_dir(repo, "training") / "audits" / f"leakage_{stage}.json", report)
    if report["result"] != "PASS":
        raise FrontierGateError(f"frontier leakage audit failed at {stage}")
    return report


def _metrics_for_indices(
    predictions: np.ndarray, labels: np.ndarray, indices: Sequence[int],
) -> dict[str, Any]:
    if not indices:
        return {"sample_count": 0, "result": "NO_SAMPLES"}
    selected = np.asarray(indices, dtype=np.int64)
    return error_metrics(predictions[selected], labels[selected])


def _offline_model_report(
    model: Any, rows: Sequence[dict[str, Any]], training: dict[str, Any], device: Any,
    bundles: Mapping[str, ScenarioBundle], phases: Sequence[str],
) -> dict[str, Any]:
    predictions, labels = predict_temporal(model, rows, training, device)
    per_scenario: dict[str, Any] = {}
    for scenario in VALIDATION_SCENARIOS:
        indices = [index for index, row in enumerate(rows) if row["scenario_id"] == scenario]
        route_bins: dict[str, Any] = {}
        for bin_index, (lower, upper) in enumerate(ROUTE_BINS):
            selected = [
                index for index in indices
                if (lower <= float(rows[index]["route_progress_m"]) <= upper
                    if bin_index == len(ROUTE_BINS) - 1
                    else lower <= float(rows[index]["route_progress_m"]) < upper)
            ]
            route_bins[_route_bin_key(lower, upper)] = {
                "route_s_m": [lower, upper],
                **_metrics_for_indices(predictions, labels, selected),
            }
        phase_metrics = {
            phase: _metrics_for_indices(
                predictions, labels,
                [index for index in indices
                 if _phase(bundles[scenario], float(rows[index]["route_progress_m"])) == phase],
            ) for phase in phases
        }
        per_scenario[scenario] = {
            **_metrics_for_indices(predictions, labels, indices),
            "route_bins": route_bins, "obstacle_phases": phase_metrics,
        }
    combined_bins: dict[str, Any] = {}
    for bin_index, (lower, upper) in enumerate(ROUTE_BINS):
        indices = [
            index for index, row in enumerate(rows)
            if (lower <= float(row["route_progress_m"]) <= upper
                if bin_index == len(ROUTE_BINS) - 1
                else lower <= float(row["route_progress_m"]) < upper)
        ]
        combined_bins[_route_bin_key(lower, upper)] = {
            "route_s_m": [lower, upper], **_metrics_for_indices(predictions, labels, indices),
        }
    combined_phases = {
        phase: _metrics_for_indices(
            predictions, labels,
            [index for index, row in enumerate(rows)
             if _phase(bundles[row["scenario_id"]], float(row["route_progress_m"])) == phase],
        ) for phase in phases
    }
    return {
        "combined": error_metrics(predictions, labels),
        "per_scenario": per_scenario,
        "combined_route_bins": combined_bins,
        "combined_obstacle_phases": combined_phases,
    }


def _training_artifacts_valid(report: Mapping[str, Any]) -> bool:
    if (
        report.get("result") != "PASS"
        or report.get("model_frozen_before_live") is not True
        or (report.get("onnx_equivalence") or {}).get("result") != "PASS"
        or (report.get("architecture") or {}).get("parameter_count") != TEMPORAL_PARAMETER_COUNT
    ):
        return False
    for key in ("checkpoint", "onnx", "training_config_snapshot", "training_summary_snapshot"):
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


def verify_frozen_d2_fe(repo: Path, config: FrontierConfig) -> dict[str, Any]:
    report = _read_json(config.result_dir(repo, "training") / "summary.json")
    if not _training_artifacts_valid(report):
        raise FrontierGateError("D2-FE is not a complete frozen model")
    if report.get("task_config_sha256") != config.sha256:
        raise FrontierGateError("D2-FE task config identity changed")
    aggregate = _read_json(config.result_dir(repo, "training") / "aggregate.json")
    aggregate_path = Path(aggregate["path"])
    validation_path = _resolve(repo, config.inputs["validation_manifest"]["path"])
    freeze_path = Path(report["freeze"]["path"])
    seal_path = Path(report["freeze_seal"]["path"])
    freeze = _read_json(freeze_path)
    seal = _read_json(seal_path)
    expected_seal = {
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": report["artifacts"]["checkpoint"]["sha256"],
        "onnx_sha256": report["artifacts"]["onnx"]["sha256"],
        "aggregate_manifest_sha256": sha256_file(aggregate_path),
        "validation_manifest_sha256": sha256_file(validation_path),
        "training_summary_snapshot_sha256": report["artifacts"]["training_summary_snapshot"]["sha256"],
        "task_config_sha256": config.sha256,
        "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    if any(seal.get(key) != value for key, value in expected_seal.items()):
        raise FrontierGateError("D2-FE freeze seal contract failed")
    if (
        freeze.get("frozen_before_any_new_s09_live_run") is not True
        or freeze.get("training_from_scratch") is not True
        or freeze.get("single_logical_training_run") is not True
        or freeze.get("model_name") != "Random-Cone Temporal PilotNet D2-FE"
        or (freeze.get("architecture") or {}).get("parameter_count") != TEMPORAL_PARAMETER_COUNT
    ):
        raise FrontierGateError("D2-FE freeze contract changed")
    return report


def training_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    disk_before = disk_gate(config)
    aggregate = _read_json(config.result_dir(repo, "training") / "aggregate.json")
    if (
        aggregate.get("result") != "PASS"
        or aggregate.get("sequence_count") != EXPECTED_SEQUENCE_COUNT
        or aggregate.get("provenance_counts") != EXPECTED_PROVENANCE
    ):
        raise FrontierGateError("exact frontier aggregate is absent")
    result_dir = config.result_dir(repo, "training")
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if _training_artifacts_valid(existing):
            return verify_frozen_d2_fe(repo, config)
        raise FrontierGateError("existing D2-FE training evidence is incomplete or changed")
    leakage = leakage_audit(repo, sim_root, config, stage="before_training")
    aggregate_path = Path(aggregate["path"])
    aggregate_rows = _read_model_rows(aggregate_path, expected_scenarios=TRAIN_SCENARIOS)
    if len(aggregate_rows) != EXPECTED_SEQUENCE_COUNT:
        raise FrontierGateError("D2-FE model-row count is not 8,298")
    validation_ref = config.inputs["validation_manifest"]
    validation_path = _resolve(repo, validation_ref["path"])
    _hash_gate(validation_path, validation_ref["sha256"], "frozen S09/S10 validation")
    validation_rows = _read_model_rows(validation_path, expected_scenarios=VALIDATION_SCENARIOS)
    if len(validation_rows) != 837:
        raise FrontierGateError("frozen validation is not exactly 837 sequences")
    parameter_count = sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters())
    if parameter_count != 255_819 or parameter_count != TEMPORAL_PARAMETER_COUNT:
        raise FrontierGateError("D2-FE architecture is not exactly 255,819 parameters")
    live_root = config.result_dir(repo, "live")
    if live_root.exists() and any(live_root.rglob("*attempt*.json")):
        raise FrontierGateError("live attempt exists before D2-FE training/freeze")
    external = config.external_root(sim_root)
    checkpoint = external / "checkpoints" / "random_cone_temporal_d2_fe_best.pt"
    state_path = external / "checkpoints" / "random_cone_temporal_d2_fe_training_state.pt"
    onnx_path = external / "onnx" / "random_cone_temporal_d2_fe.onnx"
    config_snapshot = external / "training_config_snapshot.json"
    training_snapshot = external / "training_summary_snapshot.json"
    marker = result_dir / "training.started.json"
    identity = {
        "task_config_sha256": config.sha256,
        "train_manifest_sha256": aggregate["sha256"],
        "validation_manifest_sha256": validation_ref["sha256"],
    }
    if marker.is_file():
        previous = _read_json(marker)
        if previous.get("source_identity") != identity:
            raise FrontierGateError("interrupted D2-FE training marker identity changed")
        if previous.get("status") == "D2_FE_COMPLETED_AND_FROZEN":
            raise FrontierGateError("completed marker exists without valid frozen summary")
    else:
        write_json(marker, {
            "status": "ONE_LOGICAL_D2_FE_TRAINING_RUN_STARTED",
            "started_utc": utc_now(), "source_identity": identity,
            "initialization": "from_scratch", "resumable_epoch_transactions": True,
            "retraining_permitted": False,
        })
    write_json(config_snapshot, {
        "version": TRAINING_VERSION + "_config_snapshot",
        "task_config_sha256": config.sha256, "training": config.training,
        "sources": {
            "aggregate_manifest": str(aggregate_path), "aggregate_sha256": aggregate["sha256"],
            "validation_manifest": str(validation_path), "validation_sha256": validation_ref["sha256"],
            "provenance_counts": aggregate["provenance_counts"],
        },
        "only_new_training_information_relative_to_d1": "109 DAGGER2_POST_RECOVERY sequences",
        "previous_dagger2_coverage_gate": "FAIL_UNCHANGED",
        "excluded": [
            "new data collection", "new Expert laps", "S09/S10 training", "S11/S12",
            "1.8 m/s", "V9", "C1", "fixed-cone", "unrelated DAgger",
            "weighting", "balancing", "resampling", "augmentation", "fine-tuning",
        ],
    })
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_result, history = train_temporal_resumable(
        aggregate_rows, validation_rows, config.training, device,
        state_path, checkpoint, identity,
    )
    train_task = load_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    _expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in VALIDATION_SCENARIOS
    }
    if set(bundles) != set(VALIDATION_SCENARIOS):
        raise FrontierGateError("frozen S09/S10 scenario bundle set changed")
    r1_model = _load_temporal_checkpoint(Path(config.inputs["r1"]["checkpoint_path"]), device)
    d1_model = _load_temporal_checkpoint(Path(config.inputs["d1"]["checkpoint_path"]), device)
    phases = tuple(config.payload["offline"]["phases"])
    offline = {
        "R1": _offline_model_report(r1_model, validation_rows, config.training, device, bundles, phases),
        "D1": _offline_model_report(d1_model, validation_rows, config.training, device, bundles, phases),
        "D2_FE": _offline_model_report(model, validation_rows, config.training, device, bundles, phases),
    }
    export_temporal_onnx(model, onnx_path, config.training)
    equivalence = validate_equivalence(model, validation_rows, onnx_path, config.training)
    if equivalence.get("result") != "PASS":
        raise FrontierGateError("D2-FE PyTorch/ONNX equivalence failed")
    artifacts = {
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
                       "sha256": sha256_file(checkpoint)},
        "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size,
                 "sha256": sha256_file(onnx_path)},
        "training_config_snapshot": {
            "path": str(config_snapshot), "size_bytes": config_snapshot.stat().st_size,
            "sha256": sha256_file(config_snapshot),
        },
    }
    write_json(training_snapshot, {
        "version": TRAINING_VERSION + "_training_summary_snapshot",
        "generated_utc": utc_now(), "training": training_result, "epochs": history,
        "architecture": {"input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
                         "parameter_count": parameter_count},
        "offline_validation": offline, "onnx_equivalence": equivalence,
        "source_identity": identity,
    })
    artifacts["training_summary_snapshot"] = {
        "path": str(training_snapshot), "size_bytes": training_snapshot.stat().st_size,
        "sha256": sha256_file(training_snapshot),
    }
    freeze_payload = {
        "version": TRAINING_VERSION + "_freeze", "frozen_utc": utc_now(),
        "frozen_before_any_new_s09_live_run": True,
        "model_name": "Random-Cone Temporal PilotNet D2-FE",
        "architecture": {"input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
                         "parameter_count": parameter_count,
                         "architecture_identity": "frozen Temporal PilotNet R1/D1"},
        "training_from_scratch": True, "single_logical_training_run": True,
        "task_config_sha256": config.sha256, "aggregate_manifest": aggregate,
        "validation_manifest": {"path": str(validation_path), "sha256": validation_ref["sha256"],
                                "sequence_count": 837},
        "training_config_snapshot": artifacts["training_config_snapshot"],
        "training_summary_snapshot": artifacts["training_summary_snapshot"],
        "checkpoint": artifacts["checkpoint"], "onnx": artifacts["onnx"],
        "onnx_equivalence": equivalence, "offline_validation": offline,
        "prior_dagger2_coverage_gate": "FAIL_UNCHANGED",
        "holdout_scenarios_observed_by_model_before_freeze": [],
    }
    external_freeze = external / "freeze.json"
    compact_freeze = result_dir / "freeze.json"
    write_json(external_freeze, freeze_payload)
    write_json(compact_freeze, freeze_payload)
    freeze_sha = sha256_file(external_freeze)
    if freeze_sha != sha256_file(compact_freeze):
        raise FrontierGateError("external/compact D2-FE freeze mismatch")
    seal_payload = {
        "version": TRAINING_VERSION + "_freeze_seal", "sealed_utc": utc_now(),
        "freeze_sha256": freeze_sha,
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "aggregate_manifest_sha256": aggregate["sha256"],
        "validation_manifest_sha256": validation_ref["sha256"],
        "training_summary_snapshot_sha256": artifacts["training_summary_snapshot"]["sha256"],
        "task_config_sha256": config.sha256, "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    external_seal = external / "freeze_seal.json"
    compact_seal = result_dir / "freeze_seal.json"
    write_json(external_seal, seal_payload)
    write_json(compact_seal, seal_payload)
    seal_sha = sha256_file(external_seal)
    if seal_sha != sha256_file(compact_seal):
        raise FrontierGateError("external/compact D2-FE freeze seal mismatch")
    report = {
        "version": TRAINING_VERSION, "generated_utc": utc_now(), "result": "PASS",
        "task_config_sha256": config.sha256,
        "training_sources": {
            "aggregate_manifest": str(aggregate_path), "aggregate_manifest_sha256": aggregate["sha256"],
            "aggregate_sequence_count": len(aggregate_rows),
            "provenance_counts": aggregate["provenance_counts"],
            "validation_manifest": str(validation_path),
            "validation_manifest_sha256": validation_ref["sha256"],
            "validation_sequence_count": len(validation_rows),
        },
        "only_new_training_information_relative_to_d1": "109 DAGGER2_POST_RECOVERY sequences",
        "architecture": {"input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
                         "parameter_count": parameter_count, "first_conv": "9->24, 5x5, stride 2"},
        "training": training_result, "epochs": history, "device": str(device),
        "offline_validation": offline,
        "onnx_contract": {"checker": "PASS", "input": ["batch", 9, 66, 200],
                          "output": ["batch", 1]},
        "onnx_equivalence": equivalence, "artifacts": artifacts,
        "freeze": {"path": str(external_freeze), "compact_path": str(compact_freeze),
                   "sha256": freeze_sha},
        "freeze_seal": {"path": str(external_seal), "compact_path": str(compact_seal),
                        "sha256": seal_sha},
        "leakage_audit_before_training": leakage,
        "frozen_expert_metadata_audit": expert_audit,
        "model_frozen_before_live": True, "training_runs": 1,
        "initialized_from_scratch": True, "fine_tuned_from_d1": False,
        "weighting_balancing_resampling": False, "retraining_performed": False,
        "holdout_data_used": False, "new_data_collection_performed": False,
        "prior_dagger2_coverage_gate": "FAIL_UNCHANGED",
        "disk_before_training": disk_before, "disk_after_training": disk_state("/"),
    }
    write_json(summary_path, report)
    write_json(marker, {
        "status": "D2_FE_COMPLETED_AND_FROZEN", "completed_utc": utc_now(),
        "source_identity": identity, "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"], "freeze_seal_sha256": seal_sha,
        "retraining_permitted": False,
    })
    if state_path.is_file():
        state_path.unlink()
    report["temporary_resumable_training_state_removed_after_freeze"] = not state_path.exists()
    write_json(summary_path, report)
    _prior_tree_gate(repo, config)
    return verify_frozen_d2_fe(repo, config)


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
        and all(item.get("classification") == "RANDOM_CONE_POLICY_PASS"
                for item in report.get("scenarios", []))
    )


def classify_s09_frontier(
    record: Mapping[str, Any], frontier: Mapping[str, Any],
) -> dict[str, Any]:
    classification = record.get("classification")
    run = record.get("run") or {}
    observed = {
        "progress_m": float(run.get("total_unwrapped_progress_m") or 0.0),
        "final_route_s_m": float(run.get("final_route_s_m") or 0.0),
        "completion_fraction": float(run.get("route_completion_fraction") or 0.0),
    }
    baseline = {
        "progress_m": float(frontier["d1_s09_progress_m"]),
        "final_route_s_m": float(frontier["d1_s09_final_route_s_m"]),
        "completion_fraction": float(frontier["d1_s09_completion_fraction"]),
    }
    deltas = {key: observed[key] - baseline[key] for key in observed}
    if classification == "RANDOM_CONE_POLICY_PASS":
        result = "S09_FULL_LAP_PASS"
    elif classification != "RANDOM_CONE_POLICY_FAIL":
        result = INCONCLUSIVE
    else:
        material_gain = bool(
            deltas["progress_m"] >= float(frontier["partial_support_minimum_progress_gain_m"])
            or deltas["final_route_s_m"] >= float(frontier["partial_support_minimum_route_s_gain_m"])
            or deltas["completion_fraction"] >= float(frontier["partial_support_minimum_completion_gain"])
        )
        recovered = run.get("recovery_success") is True
        progress_loss = -deltas["progress_m"]
        completion_loss = -deltas["completion_fraction"]
        route_loss = -deltas["final_route_s_m"]
        substantial_regression = bool(
            completion_loss >= float(frontier["regression_minimum_completion_loss"])
            or (
                progress_loss >= float(frontier["regression_minimum_progress_loss_m"])
                and route_loss >= float(frontier["partial_support_minimum_route_s_gain_m"])
            )
        )
        if material_gain and recovered:
            result = FRONTIER_EXPANSION_PARTIAL_SUPPORT
        elif substantial_regression:
            result = REGRESSION
        else:
            result = NO_CLEAR_FRONTIER_EXPANSION
    return {
        "classification": result, "d1_baseline": baseline, "d2_fe_observed": observed,
        "d2_fe_minus_d1": deltas,
        "recovery_success": run.get("recovery_success"),
        "comparison_rule_preregistered_before_live": True,
    }


def _valid_live_record(
    record: Mapping[str, Any], scenario: str, role: str, training: Mapping[str, Any],
) -> bool:
    return bool(
        record.get("version") == LIVE_VERSION + "_scenario"
        and record.get("scenario_id") == scenario
        and record.get("role") == role
        and record.get("classification") in VALID_POLICY_CLASSIFICATIONS
        and record.get("onnx_sha256") == training["artifacts"]["onnx"]["sha256"]
        and record.get("checkpoint_sha256") == training["artifacts"]["checkpoint"]["sha256"]
        and record.get("freeze_seal_sha256") == training["freeze_seal"]["sha256"]
        and (record.get("run") or {}).get("safe_stop_success") is True
        and record.get("model_frozen_before_attempt") is True
        and record.get("valid_policy_run_number") == 1
        and record.get("bags_collected") == 0
        and record.get("camera_images_persisted") == 0
        and record.get("expert_labels_generated") == 0
    )


def _live_group(
    repo: Path, sim_root: Path, config: FrontierConfig, *, group: str,
) -> dict[str, Any]:
    if group not in {"validation", "holdout"}:
        raise ValueError(group)
    disk_before = disk_gate(config, live=True)
    training = verify_frozen_d2_fe(repo, config)
    if group == "validation":
        scenario_ids: Sequence[str] = VALIDATION_SCENARIOS
        role = "VALIDATION"
        leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    else:
        validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
        if not validation_path.is_file():
            raise FrontierGateError("S09/S10 evidence is absent; S11/S12 remain untouched")
        validation = _read_json(validation_path)
        if not validation_allows_holdout(validation):
            raise FrontierGateError("S09/S10 did not both pass; S11/S12 remain untouched")
        scenario_ids = HOLDOUT_SCENARIOS
        role = "UNSEEN_HOLDOUT"
        leakage = leakage_audit(repo, sim_root, config, stage="before_untouched_holdout")
    train_task = load_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in scenario_ids
    }
    if set(bundles) != set(scenario_ids):
        raise FrontierGateError(f"frozen {group} scenario bundle set changed")
    r1_config = load_r1_config(repo / config.dagger1.r1["task_config_path"], repo)
    model = TemporalOnnxModel(Path(training["artifacts"]["onnx"]["path"]))
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
            if not _valid_live_record(existing, scenario, role, training):
                raise FrontierGateError(f"completed D2-FE S{scenario} live identity changed")
            records.append(existing)
    if [item["scenario_id"] for item in records] != list(scenario_ids[:len(records)]):
        raise FrontierGateError(f"existing {group} live results are out of gate order")
    if group == "validation" and next_validation(records) is None and len(records) < 2:
        pending: Sequence[str] = ()
    elif group == "holdout" and next_holdout(records) is None and len(records) < 2:
        pending = ()
    else:
        pending = scenario_ids
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
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
            verify_frozen_d2_fe(repo, config)
            state_path = states_dir / f"scenario_{scenario}.json"
            final_path = scenarios_dir / f"scenario_{scenario}.json"
            attempt_paths = sorted(attempts_dir.glob(f"scenario_{scenario}_attempt_*.json"))
            if attempt_paths:
                latest = _read_json(attempt_paths[-1])
                if latest.get("classification") in VALID_POLICY_CLASSIFICATIONS:
                    if not _valid_live_record(latest, scenario, role, training):
                        raise FrontierGateError(f"captured D2-FE S{scenario} evidence changed")
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
                        "infrastructure_classification": "HOST_OR_PROCESS_INTERRUPTION",
                        "failure_reason": "process ended before finalized live evidence",
                    })
                    attempts_consumed = started
            if attempts_consumed >= 2:
                break
            attempt_number = attempts_consumed + 1
            while attempt_number <= 2:
                frozen = verify_frozen_d2_fe(repo, config)
                write_json(state_path, {
                    "status": "STARTED_UNFINALIZED", "scenario_id": scenario, "role": role,
                    "attempt_number": attempt_number, "started_utc": utc_now(),
                    "onnx_sha256": frozen["artifacts"]["onnx"]["sha256"],
                    "freeze_seal_sha256": frozen["freeze_seal"]["sha256"],
                })
                record: dict[str, Any] = {
                    "version": LIVE_VERSION + "_scenario", "generated_utc": utc_now(),
                    "scenario_id": scenario, "role": role, "attempt_number": attempt_number,
                    "valid_policy_run_number": None, "classification": "INFRA_FAIL",
                    "result": "FAIL", "failure_reason": None,
                    "onnx_sha256": frozen["artifacts"]["onnx"]["sha256"],
                    "checkpoint_sha256": frozen["artifacts"]["checkpoint"]["sha256"],
                    "freeze_sha256": frozen["freeze"]["sha256"],
                    "freeze_seal_sha256": frozen["freeze_seal"]["sha256"],
                    "model_frozen_before_attempt": True,
                    "d2_fe_controls_vehicle": True, "expert_control_authority": False,
                    "bags_collected": 0, "camera_images_persisted": 0,
                    "expert_labels_generated": 0,
                    "preflight": None, "world_activation": None, "run": None,
                }
                try:
                    live = run_live_once(client, model, r1_config, expert, bundles[scenario], sim_root)
                    record["preflight"] = live["preflight"]
                    record["world_activation"] = live["world_activation"]
                    record["run"] = live["run"]
                    record["classification"] = live["run"]["classification"]
                    record["result"] = (
                        "PASS" if record["classification"] == "RANDOM_CONE_POLICY_PASS" else "FAIL"
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
                    "stage": f"d2_fe_live_{group}", "scenario": scenario,
                    "attempt": attempt_number, "classification": record["classification"],
                    "completion": (record.get("run") or {}).get("route_completion_fraction"),
                    "route_s_m": (record.get("run") or {}).get("final_route_s_m"),
                }), flush=True)
                decision = live_retry_decision(record["classification"], attempt_number)
                if decision in {"FINALIZE_PASS", "FINALIZE_GENUINE_FAILURE"}:
                    if not _valid_live_record(record, scenario, role, training):
                        raise FrontierGateError(f"D2-FE S{scenario} policy evidence contract failed")
                    if scenario == "09":
                        record["frontier_comparison"] = classify_s09_frontier(
                            record, config.payload["frontier"],
                        )
                    write_json(attempt_path, record)
                    write_json(final_path, record)
                    write_json(state_path, {
                        "status": "FINALIZED_VALID_POLICY_EVALUATION",
                        "scenario_id": scenario, "role": role,
                        "attempt_number": attempt_number,
                        "classification": record["classification"], "finalized_utc": utc_now(),
                        "do_not_repeat": True,
                    })
                    records.append(record)
                    break
                if decision == "REPLACE_INFRA":
                    errors = client.safe_stop()
                    if errors:
                        record["failure_reason"] = (
                            (record.get("failure_reason") or "")
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
        (item for item in records if item.get("classification") == "RANDOM_CONE_POLICY_FAIL"), None,
    )
    frontier_comparison = next(
        (item.get("frontier_comparison") for item in records if item.get("scenario_id") == "09"), None,
    )
    if policy_failure:
        result = "FAIL"
        category = D2_FE_VALIDATION_FAIL if group == "validation" else D2_FE_UNSEEN_FAIL
    elif pass_count == 2 and len(records) == 2 and not final_errors and restoration.get("result") == "PASS":
        result = "PASS"
        category = "VALIDATION_PASS" if group == "validation" else "UNSEEN_PASS"
    else:
        result = INCONCLUSIVE
        category = INCONCLUSIVE
    report = {
        "version": LIVE_VERSION + f"_{group}", "generated_utc": utc_now(),
        "result": result, "category": category, "role": role,
        "intended_scenario_ids": list(
            VALIDATION_SCENARIOS if group == "validation" else HOLDOUT_SCENARIOS
        ),
        "scenarios": records, "valid_policy_run_count": len(records), "pass_count": pass_count,
        "frontier_comparison": frontier_comparison,
        "maximum_valid_runs_per_scenario": 1,
        "maximum_infrastructure_replacements_per_scenario": 1,
        "infrastructure_replacements": infrastructure_replacements,
        "model_frozen_before_all_attempts": True,
        "onnx_sha256": training["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": training["artifacts"]["checkpoint"]["sha256"],
        "freeze_seal_sha256": training["freeze_seal"]["sha256"],
        "leakage_audit": leakage, "frozen_expert_metadata_audit": expert_audit,
        "disk_before_live": disk_before,
        "final_safe_stop_success": not final_errors, "final_safe_stop_errors": final_errors,
        "world_restoration": restoration,
        "bags_collected": 0, "camera_images_persisted": 0,
        "expert_training_labels_generated": 0,
    }
    write_json(summary_path, report)
    return report


def live_validation_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="validation")


def live_holdout_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="holdout")


def classify_final_category(
    validation: Mapping[str, Any] | None, holdout: Mapping[str, Any] | None,
) -> str:
    if not validation or validation.get("result") == INCONCLUSIVE:
        return INCONCLUSIVE
    scenarios = validation.get("scenarios") or []
    if scenarios and scenarios[0].get("classification") == "RANDOM_CONE_POLICY_FAIL":
        comparison = scenarios[0].get("frontier_comparison") or validation.get("frontier_comparison") or {}
        return str(comparison.get("classification") or NO_CLEAR_FRONTIER_EXPANSION)
    if validation.get("result") == "FAIL":
        return D2_FE_VALIDATION_FAIL
    if validation.get("result") != "PASS":
        return INCONCLUSIVE
    if not holdout or holdout.get("result") == INCONCLUSIVE:
        return INCONCLUSIVE
    if holdout.get("result") == "FAIL":
        return D2_FE_UNSEEN_FAIL
    if holdout.get("result") == "PASS" and holdout.get("category") == "UNSEEN_PASS":
        return D2_FE_FULL_PASS
    return INCONCLUSIVE


def _live_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in records:
        run = item.get("run") or {}
        output[str(item["scenario_id"])] = {
            "role": item.get("role"), "classification": item.get("classification"),
            "elapsed_s": run.get("elapsed_s"),
            "total_unwrapped_progress_m": run.get("total_unwrapped_progress_m"),
            "final_route_s_m": run.get("final_route_s_m"),
            "route_completion_fraction": run.get("route_completion_fraction"),
            "minimum_footprint_to_cone_clearance_m": run.get("minimum_footprint_to_cone_clearance_m"),
            "cone_contact_or_intersection_occurred": run.get("cone_contact_or_intersection_occurred"),
            "recovery_success": run.get("recovery_success"),
            "recovery_time_s": run.get("recovery_time_s"),
            "mean_cte_m": run.get("mean_cte_m"), "max_cte_m": run.get("max_cte_m"),
            "off_track_events": run.get("off_track_events"),
            "off_track_total_duration_s": run.get("off_track_total_duration_s"),
            "mean_absolute_predicted_steering_rad": run.get("mean_absolute_predicted_steering_rad"),
            "max_absolute_predicted_steering_rad": run.get("max_absolute_predicted_steering_rad"),
            "steering_saturation_fraction": run.get("steering_saturation_fraction"),
            "temporal_frame_gaps": run.get("temporal_frame_gaps"),
            "temporal_invalid_history_count": run.get("temporal_invalid_history_count"),
            "preprocessing_latency": run.get("preprocessing_latency"),
            "onnx_inference_latency": run.get("onnx_inference_latency"),
            "control_loop_frequency_hz": run.get("control_loop_frequency_hz"),
            "timing_slips_over_100ms": run.get("timing_slips_over_100ms"),
            "api_failures": run.get("api_failures"), "pose_failures": run.get("pose_failures"),
            "clock_failures": run.get("clock_failures"),
            "liveness_failures": run.get("liveness_failures"),
            "safe_stop_success": run.get("safe_stop_success"),
            "frontier_comparison": item.get("frontier_comparison"),
        }
    return output


def final_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    training = verify_frozen_d2_fe(repo, config)
    aggregate = _read_json(config.result_dir(repo, "training") / "aggregate.json")
    audit = _read_json(config.result_dir(repo, "training") / "audit.json")
    validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
    validation = _read_json(validation_path) if validation_path.is_file() else None
    holdout_path = config.result_dir(repo, "live") / "live_holdout_summary.json"
    holdout = _read_json(holdout_path) if holdout_path.is_file() else None
    category = classify_final_category(validation, holdout)
    final_leakage = leakage_audit(repo, sim_root, config, stage="final")
    prior_after = _prior_tree_gate(repo, config)
    if prior_after != audit["prior_directory_identities_before"]:
        raise FrontierGateError("prior DAgger2 failure evidence changed during D2-FE experiment")
    records = [
        *(validation.get("scenarios", []) if validation else []),
        *(holdout.get("scenarios", []) if holdout else []),
    ]
    frontier = validation.get("frontier_comparison") if validation else None
    frontier_supported = category in {D2_FE_FULL_PASS, FRONTIER_EXPANSION_PARTIAL_SUPPORT}
    report = {
        "version": LIVE_VERSION, "generated_utc": utc_now(),
        "result": category, "final_category": category,
        "prior_dagger2_coverage_gate_failure": {
            **audit["prior_negative_result"],
            "directory_identities_before": audit["prior_directory_identities_before"],
            "directory_identities_after": prior_after,
            "proof_unchanged": True,
        },
        "disk": {
            "before_training": audit["disk_before_training"],
            "after_training": training["disk_after_training"],
            "final": disk_state("/"),
        },
        "dagger2_109_sequence_integrity": audit["dagger2_data"],
        "aggregate": aggregate,
        "training": training,
        "offline_validation": training["offline_validation"],
        "live_validation": validation, "live_holdout": holdout,
        "live_metrics": _live_metrics(records),
        "d1_versus_d2_fe_frontier": frontier,
        "leakage_audit": final_leakage,
        "holdout_leakage_audit_performed_after_s09_s10_pass": bool(
            holdout and validation_allows_holdout(validation or {})
        ),
        "holdout_protection": {
            "s11_s12_authorized_only_after_s09_s10_pass": True,
            "touched_before_gate": False,
            "bags_collected": 0, "expert_training_labels_generated": 0,
            "training_or_validation_rows": 0,
        },
        "frontier_expansion_supported": frontier_supported,
        "d2_fe_becomes_simulator_baseline": category == D2_FE_FULL_PASS,
        "another_dagger_iteration_justified": (
            "SEPARATE_DECISION_MAY_BE_JUSTIFIED_BY_PARTIAL_SUPPORT"
            if category == FRONTIER_EXPANSION_PARTIAL_SUPPORT
            else "NOT_AUTOMATICALLY_AUTHORIZED"
        ),
        "new_data_collection_performed": False, "dagger3_started": False,
        "training_after_freeze": False, "commit_performed": False, "push_performed": False,
        "real_robot_success_claimed": False,
        "limitations": [
            "The 109 DAgger2 sequences contain no targets beyond route s=26 m.",
            "Offline validation is diagnostic and was not used as a model-selection gate.",
            "A single closed-loop run per scenario cannot establish repeatability.",
            "Simulator performance is not evidence of real-robot success.",
        ],
    }
    write_json(config.result_dir(repo, "live") / "summary.json", report)
    return report


def _display(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _final_markdown(report: Mapping[str, Any], verification: Mapping[str, Any]) -> str:
    training = report["training"]
    data = report["dagger2_109_sequence_integrity"]
    aggregate = report["aggregate"]
    offline_rows = []
    for scenario in VALIDATION_SCENARIOS:
        values = [training["offline_validation"][name]["per_scenario"][scenario]
                  for name in ("R1", "D1", "D2_FE")]
        offline_rows.append(
            f"| S{scenario} | " + " | ".join(
                f"{_display(item.get('mae_rad'))} / {_display(item.get('rmse_rad'))} / "
                f"{_display(item.get('bias_mean_signed_error_rad'))}"
                for item in values
            ) + " |"
        )
    route_rows = []
    for lower, upper in ROUTE_BINS:
        key = _route_bin_key(lower, upper)
        route_rows.append(
            f"| {key} | " + " | ".join(
                f"{training['offline_validation'][name]['combined_route_bins'][key].get('sample_count', 0)} / "
                f"{_display(training['offline_validation'][name]['combined_route_bins'][key].get('mae_rad'))}"
                for name in ("R1", "D1", "D2_FE")
            ) + " |"
        )
    live_rows = []
    for scenario, item in report["live_metrics"].items():
        live_rows.append(
            f"| S{scenario} | {item['role']} | {item['classification']} | "
            f"{_display(item['route_completion_fraction'], 4)} | "
            f"{_display(item['final_route_s_m'])} | "
            f"{_display(item['minimum_footprint_to_cone_clearance_m'])} | "
            f"{item['recovery_success']} | {item['safe_stop_success']} |"
        )
    status = verification.get("repository_status", [])
    prior = report["prior_dagger2_coverage_gate_failure"]
    lines = [
        "# Random-Cone D2 Frontier Expansion V1", "",
        f"Final category: **{report['final_category']}**", "",
        "## Prior negative result and disk", "",
        "The earlier Targeted DAgger2 Post-Recovery V1 result remains FAIL: "
        "109 valid sequences, 18 with route s>20 m, zero with route s>26 m, and training unauthorized. "
        f"Its collection/dataset tree hashes remain `{prior['directory_identities_after']['collection']['sha256']}` / "
        f"`{prior['directory_identities_after']['dataset']['sha256']}`.", "",
        f"Root free space was {report['disk']['before_training']['available_gib']:.3f} GiB before training "
        f"and is {report['disk']['final']['available_gib']:.3f} GiB at final reporting.", "",
        "## DAgger2 audit and aggregate", "",
        f"All {data['sequence_count']} rows passed provenance, actual-D1-state, post-recovery, causal-label, "
        f"and temporal-integrity checks. Coverage: <=20 m {data['route_coverage']['at_or_below_20m']}, "
        f">20 m {data['route_coverage']['after_20m']}, >26 m {data['route_coverage']['after_26m']}.", "",
        f"Aggregate: **{aggregate['sequence_count']}** = 6,706 EXPERT_BASELINE + 1,483 DAGGER1 + "
        f"109 DAGGER2_POST_RECOVERY. SHA-256: `{aggregate['sha256']}`. No images were duplicated.", "",
        "## Training, export, and freeze", "",
        f"D2-FE has {training['architecture']['parameter_count']:,} parameters and trained once from scratch. "
        f"Best epoch {training['training']['best_epoch']} after {training['training']['epochs_completed']} completed epochs; "
        f"early stopped={training['training']['early_stopped']}.", "",
        f"Checkpoint: `{training['artifacts']['checkpoint']['sha256']}`. ONNX: "
        f"`{training['artifacts']['onnx']['sha256']}`. Freeze: `{training['freeze']['sha256']}`. "
        f"Freeze seal: `{training['freeze_seal']['sha256']}`. ONNX checker/equivalence: "
        f"PASS/{training['onnx_equivalence']['result']}.", "",
        "## Frozen offline S09/S10", "",
        "Values are MAE / RMSE / bias in radians on the identical frozen manifest.", "",
        "| Scenario | R1 | D1 | D2-FE |", "|---|---:|---:|---:|", *offline_rows, "",
        "| Route bin | R1 count / MAE | D1 count / MAE | D2-FE count / MAE |",
        "|---|---:|---:|---:|", *route_rows, "",
        "## Strictly gated live results", "",
        "| Scenario | Role | Result | Completion | Route s m | Clearance m | Recovery | Safe stop |",
        "|---|---|---|---:|---:|---:|---:|---:|", *live_rows,
        "" if live_rows else "No infrastructure-valid policy result was available.", "",
        f"Direct D1↔D2-FE frontier comparison: `{report.get('d1_versus_d2_fe_frontier')}`.", "",
        "## Disposition and verification", "",
        f"Frontier expansion supported: {report['frontier_expansion_supported']}. D2-FE becomes the simulator "
        f"baseline: {report['d2_fe_becomes_simulator_baseline']}. Another DAgger iteration: "
        f"{report['another_dagger_iteration_justified']}. No DAgger3 was started.", "",
        "Leakage audit PASS; no new training collection, bags, persisted live cameras, or Expert labels were produced. "
        "S11/S12 remained gated until S09/S10 both passed.", "",
        f"Tests: {verification.get('tests', {}).get('summary', 'pending')}. `git diff --check`: "
        f"{verification.get('git_diff_check', {}).get('result', 'pending')}. No commit or push occurred.", "",
        "Limitations: " + " ".join(report["limitations"]), "",
        "External artifacts: " + str(verification.get("external_artifacts_root", "")) + ".", "",
        "Final Git status:", "", "```text", *status, "```", "",
    ]
    return "\n".join(lines)


def test_stage(repo: Path, config: FrontierConfig) -> dict[str, Any]:
    focused = repo / "tests/test_random_cone_d2_frontier_expansion.py"
    commands = (
        [sys.executable, "-m", "pytest", "-q", str(focused)],
        [sys.executable, "-m", "pytest", "-q"],
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        process = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=False)
        item = {
            "command": command, "returncode": process.returncode,
            "stdout_tail": process.stdout.splitlines()[-12:],
            "stderr_tail": process.stderr.splitlines()[-12:],
        }
        results.append(item)
        if process.returncode:
            raise FrontierGateError(f"test gate failed: {' '.join(command)}\n{process.stdout}\n{process.stderr}")
    report = {
        "version": VERSION + "_pretraining_tests", "generated_utc": utc_now(),
        "result": "PASS", "commands": results,
    }
    write_json(config.result_dir(repo, "training") / "pretraining_tests.json", report)
    return report


def verification_stage(repo: Path, sim_root: Path, config: FrontierConfig) -> dict[str, Any]:
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo, text=True, capture_output=True, check=False,
    )
    summary_line = next(
        (line for line in reversed(tests.stdout.splitlines()) if " passed" in line),
        "pytest summary unavailable",
    )
    simulator = simulator_tracked_status(sim_root)
    result = {
        "version": VERSION + "_verification", "generated_utc": utc_now(),
        "result": "PASS" if tests.returncode == 0 and diff.returncode == 0
                  and simulator.get("result") == "PASS" else "FAIL",
        "tests": {"result": "PASS" if tests.returncode == 0 else "FAIL",
                  "returncode": tests.returncode, "summary": summary_line,
                  "stdout_tail": tests.stdout.splitlines()[-12:],
                  "stderr_tail": tests.stderr.splitlines()[-12:]},
        "git_diff_check": {"result": "PASS" if diff.returncode == 0 else "FAIL",
                           "returncode": diff.returncode,
                           "output": (diff.stdout + diff.stderr).splitlines()},
        "simulator_tracked_source_status": simulator,
        "repository_status": [], "commit_performed": False, "push_performed": False,
        "external_artifacts_root": str(config.external_root(sim_root)),
        "disk_final": disk_state("/"),
    }
    live_root = config.result_dir(repo, "live")
    write_json(live_root / "verification.json", result)
    final_path = live_root / "summary.json"
    final = _read_json(final_path)
    final["verification"] = result
    write_json(final_path, final)
    _write_text(live_root / "REPORT.md", _final_markdown(final, result))
    final_diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo, text=True, capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    result["git_diff_check"] = {
        "result": "PASS" if final_diff.returncode == 0 else "FAIL",
        "returncode": final_diff.returncode,
        "output": (final_diff.stdout + final_diff.stderr).splitlines(),
    }
    result["repository_status"] = status
    result["result"] = "PASS" if tests.returncode == 0 and final_diff.returncode == 0 \
        and simulator.get("result") == "PASS" else "FAIL"
    write_json(live_root / "verification.json", result)
    final["verification"] = result
    final["final_git_status"] = status
    final["files_added_or_modified"] = [line[3:] for line in status[1:] if len(line) > 3]
    final["external_artifacts_root"] = str(config.external_root(sim_root))
    final["disk_final"] = disk_state("/")
    write_json(final_path, final)
    _write_text(live_root / "REPORT.md", _final_markdown(final, result))
    if result["result"] != "PASS":
        raise FrontierGateError("final regression/diff/simulator-source verification failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "audit", "aggregate", "test", "train", "live-validation", "live-holdout",
        "final", "verify", "all",
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
        args.config or repo / "configs/random_cone_d2_frontier_expansion_1p0_v1.json"
    ).resolve()
    config = load_config(config_path, repo)
    if args.stage == "audit":
        result = audit_stage(repo, sim_root, config)
    elif args.stage == "aggregate":
        result = build_aggregate_stage(repo, sim_root, config)
    elif args.stage == "test":
        result = test_stage(repo, config)
    elif args.stage == "train":
        result = training_stage(repo, sim_root, config)
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
        build_aggregate_stage(repo, sim_root, config)
        test_stage(repo, config)
        training_stage(repo, sim_root, config)
        validation = live_validation_stage(repo, sim_root, config)
        if validation.get("result") == "PASS":
            live_holdout_stage(repo, sim_root, config)
        result = final_stage(repo, sim_root, config)
        verification_stage(repo, sim_root, config)
    print(json.dumps({"stage": args.stage, "result": result.get("result")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
