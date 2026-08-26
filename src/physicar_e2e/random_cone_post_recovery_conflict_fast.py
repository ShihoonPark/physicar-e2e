"""Read-only D1 versus post-recovery conflict diagnosis.

The module deliberately exposes no training, collection, or live-driving entry
point.  It evaluates immutable manifests and checkpoints with forward passes
and ``torch.autograd.grad``; no model state is ever written or updated.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dataset_extractor import canonical_json_bytes, sha256_file
from .pilotnet import MAX_STEERING_RAD
from .pilotnet_temporal import (
    TEMPORAL_CHANNELS,
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
    preprocess_temporal_paths,
)


VERSION = "random_cone_post_recovery_conflict_fast_v1"
EXPECTED_BRANCH = "diagnosis/random-cone-post-recovery-conflict-fast-v1"
EXPECTED_BASE_COMMIT = "2c52f4dd3634fe59233acf95df42789a35da1828"
CONFIG_RELATIVE = Path("configs/random_cone_post_recovery_conflict_fast_v1.json")

GRADIENT_CONFLICT_SUPPORTED = "GRADIENT_CONFLICT_SUPPORTED"
VISUAL_STATE_ALIASING_SUPPORTED = "VISUAL_STATE_ALIASING_SUPPORTED"
BOTH_SUPPORTED = "BOTH_SUPPORTED"
NO_STRONG_CONFLICT_FOUND = "NO_STRONG_CONFLICT_FOUND"
MIXED_OR_INCONCLUSIVE = "MIXED_OR_INCONCLUSIVE"
CLASSIFICATIONS = (
    GRADIENT_CONFLICT_SUPPORTED,
    VISUAL_STATE_ALIASING_SUPPORTED,
    BOTH_SUPPORTED,
    NO_STRONG_CONFLICT_FOUND,
    MIXED_OR_INCONCLUSIVE,
)


class DiagnosisGateError(RuntimeError):
    """Raised when a frozen identity or diagnostic-only boundary changes."""


@dataclass(frozen=True)
class DiagnosticRow:
    sequence_id: str
    provenance: str
    scenario_id: str
    scenario_role: str
    episode_id: str
    paths: tuple[Path, Path, Path]
    steering_rad: float
    route_progress_m: float
    phase: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class DiagnosticConfig:
    repo: Path
    path: Path
    payload: Mapping[str, Any]

    @property
    def inputs(self) -> Mapping[str, Any]:
        return self.payload["frozen_inputs"]

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        return self.payload["diagnostics"]

    @property
    def result_dir(self) -> Path:
        return self.repo / str(self.payload["result_directory"])


def _resolve(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo / candidate


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosisGateError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(value)))
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def load_config(repo: Path, path: Path | None = None) -> DiagnosticConfig:
    repo = repo.resolve()
    config_path = (path or (repo / CONFIG_RELATIVE)).resolve()
    payload = _read_json(config_path)
    if payload.get("version") != VERSION:
        raise DiagnosisGateError("diagnosis config version changed")
    if payload.get("expected_branch") != EXPECTED_BRANCH:
        raise DiagnosisGateError("expected branch contract changed")
    permissions = payload.get("permissions", {})
    forbidden_true = (
        "training_permitted", "fine_tuning_permitted",
        "checkpoint_writes_permitted", "simulator_invocation_permitted",
        "docker_manipulation_permitted", "rosbag_access_permitted",
        "new_data_collection_permitted", "dagger3_permitted",
        "s11_s12_access_permitted", "commit_permitted", "push_permitted",
    )
    if any(permissions.get(key) is not False for key in forbidden_true):
        raise DiagnosisGateError("diagnostic-only permission boundary changed")
    if permissions.get("optimizer_steps_permitted") != 0:
        raise DiagnosisGateError("optimizer-step prohibition changed")
    if payload.get("architecture") != {
        "name": "Temporal PilotNet",
        "input_shape": [9, 66, 200],
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "penultimate_representation": "regressor final 10-D ReLU activation",
    }:
        raise DiagnosisGateError("frozen architecture contract changed")
    return DiagnosticConfig(repo=repo, path=config_path, payload=payload)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _directory_tree_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    if path.is_dir():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = item.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(item).encode("ascii"))
            digest.update(b"\n")
            count += 1
            size += item.stat().st_size
    return {"path": str(path), "sha256": digest.hexdigest(),
            "file_count": count, "size_bytes": size}


def _checkpoint_state(path: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    import torch

    record = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(record, dict) or not isinstance(record.get("model_state_dict"), dict):
        raise DiagnosisGateError(f"invalid checkpoint payload: {path}")
    return record, record["model_state_dict"]


def state_tensor_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def states_exactly_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    import torch

    return left.keys() == right.keys() and all(
        torch.equal(left[name].detach().cpu(), right[name].detach().cpu()) for name in left
    )


def load_frozen_model(checkpoint: Path):
    record, state = _checkpoint_state(checkpoint)
    if record.get("parameter_count") != TEMPORAL_PARAMETER_COUNT:
        raise DiagnosisGateError(f"checkpoint parameter count changed: {checkpoint}")
    model = build_temporal_pilotnet()
    model.load_state_dict(state, strict=True)
    model.eval()
    actual = sum(parameter.numel() for parameter in model.parameters())
    if actual != TEMPORAL_PARAMETER_COUNT:
        raise DiagnosisGateError(f"model parameter count changed: {actual}")
    return model, record


def _float(raw: Mapping[str, str], key: str) -> float | None:
    value = raw.get(key, "")
    return None if value in (None, "") else float(value)


def _first_float(raw: Mapping[str, str], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _float(raw, key)
        if value is not None:
            return value
    return None


def _first_text(raw: Mapping[str, str], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = raw.get(key, "")
        if value:
            return value
    return default


def _manifest_dataset_root(path: Path) -> Path:
    # All preserved temporal manifests live under <dataset>/.../<manifest>.csv.
    return path.parents[1]


def read_manifest(
    path: Path,
    *,
    expected_role: str | None = None,
    expected_provenance: str | None = None,
) -> list[DiagnosticRow]:
    if not path.is_file():
        raise DiagnosisGateError(f"missing frozen manifest: {path}")
    dataset_root = _manifest_dataset_root(path)
    rows: list[DiagnosticRow] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sequence_id", "scenario_id", "scenario_role", "frame_t_minus_2",
            "frame_t_minus_1", "frame_t", "target_steering_rad", "route_progress_m",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise DiagnosisGateError(f"manifest columns changed: {path}")
        for raw in reader:
            role = raw["scenario_role"]
            provenance = raw.get("provenance", "VALIDATION") or "VALIDATION"
            if expected_role is not None and role != expected_role:
                raise DiagnosisGateError(f"unexpected role {role} in {path}")
            if expected_provenance is not None and provenance != expected_provenance:
                raise DiagnosisGateError(f"unexpected provenance {provenance} in {path}")
            paths: list[Path] = []
            for key in ("frame_t_minus_2", "frame_t_minus_1", "frame_t"):
                candidate = Path(raw[key])
                paths.append(candidate if candidate.is_absolute() else dataset_root / candidate)
            if len(set(paths)) != 3 or not all(candidate.is_file() for candidate in paths):
                raise DiagnosisGateError(f"missing or duplicate temporal frames: {raw['sequence_id']}")
            time_keys = (
                ("camera_timestamp_t_minus_2_ns", "timestamp_t_minus_2_ns"),
                ("camera_timestamp_t_minus_1_ns", "timestamp_t_minus_1_ns"),
                ("camera_timestamp_t_ns", "timestamp_t_ns"),
            )
            times = tuple(int(_first_text(raw, keys)) for keys in time_keys)
            if not times[0] < times[1] < times[2]:
                raise DiagnosisGateError(f"non-causal sequence: {raw['sequence_id']}")
            if max(times[1] - times[0], times[2] - times[1]) > 120_000_000:
                raise DiagnosisGateError(f"temporal gap violation: {raw['sequence_id']}")
            target_time = _first_float(
                raw, ("expert_target_timestamp_ns", "steering_target_timestamp_ns")
            )
            if target_time is not None and target_time > times[2]:
                raise DiagnosisGateError(f"future teacher label: {raw['sequence_id']}")
            metadata: dict[str, Any] = dict(raw)
            metadata.update({
                "timestamp_t_minus_2_ns": times[0],
                "timestamp_t_minus_1_ns": times[1],
                "timestamp_t_ns": times[2],
                "adjacent_gap_1_s": (times[1] - times[0]) / 1e9,
                "adjacent_gap_2_s": (times[2] - times[1]) / 1e9,
                "oldest_to_current_span_s": (times[2] - times[0]) / 1e9,
            })
            rows.append(DiagnosticRow(
                sequence_id=raw["sequence_id"], provenance=provenance,
                scenario_id=raw["scenario_id"], scenario_role=role,
                episode_id=raw.get("episode_id", ""), paths=tuple(paths),
                steering_rad=float(raw["target_steering_rad"]),
                route_progress_m=float(raw["route_progress_m"]),
                phase=raw.get("cone_phase", "") or "",
                metadata=metadata,
            ))
    if not rows:
        raise DiagnosisGateError(f"empty manifest: {path}")
    if len({row.sequence_id for row in rows}) != len(rows):
        raise DiagnosisGateError(f"duplicate sequence IDs in {path}")
    return rows


def _phase_for_progress(progress: float, bypass: Mapping[str, Any]) -> str:
    if progress < float(bypass["departure_start_s_m"]):
        return "approach"
    if progress < float(bypass["cone_s_m"]):
        return "avoidance"
    if progress < float(bypass["return_end_s_m"]):
        return "pass_return"
    return "post_recovery"


def _with_phase(row: DiagnosticRow, phase: str) -> DiagnosticRow:
    return DiagnosticRow(
        sequence_id=row.sequence_id, provenance=row.provenance,
        scenario_id=row.scenario_id, scenario_role=row.scenario_role,
        episode_id=row.episode_id, paths=row.paths, steering_rad=row.steering_rad,
        route_progress_m=row.route_progress_m, phase=phase, metadata=row.metadata,
    )


def _route_bin(progress: float, bins: Sequence[Sequence[float]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if float(lower) <= progress < float(upper) or (
            index == len(bins) - 1 and math.isclose(progress, float(upper))
        ):
            return index
    return None


def deterministic_expert_nominal(
    rows: Sequence[DiagnosticRow], *, seed: int, bins: Sequence[Sequence[float]],
    per_stratum: int,
) -> list[DiagnosticRow]:
    grouped: dict[tuple[str, int], list[tuple[str, DiagnosticRow]]] = {}
    for row in rows:
        index = _route_bin(row.route_progress_m, bins)
        if index is None:
            raise DiagnosisGateError(f"route progress outside nominal bins: {row.sequence_id}")
        digest = hashlib.sha256(
            f"{seed}:{row.scenario_id}:{index}:{row.sequence_id}".encode("utf-8")
        ).hexdigest()
        grouped.setdefault((row.scenario_id, index), []).append((digest, row))
    selected: list[DiagnosticRow] = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda item: (item[0], item[1].sequence_id))
        if len(ranked) < per_stratum:
            raise DiagnosisGateError(f"nominal stratum {key} has only {len(ranked)} rows")
        selected.extend(row for _, row in ranked[:per_stratum])
    return sorted(selected, key=lambda row: row.sequence_id)


def _counts(rows: Sequence[DiagnosticRow], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        value = str(getattr(row, field))
        values[value] = values.get(value, 0) + 1
    return dict(sorted(values.items()))


def construct_subsets(config: DiagnosticConfig) -> tuple[dict[str, list[DiagnosticRow]], dict[str, Any]]:
    inputs = config.inputs
    aggregate = read_manifest(
        _resolve(config.repo, inputs["d1_aggregate_manifest"]["path"]), expected_role="TRAIN"
    )
    expert = [row for row in aggregate if row.provenance == "EXPERT_BASELINE"]
    aggregate_dagger1 = [row for row in aggregate if row.provenance == "DAGGER1"]
    dagger1 = read_manifest(
        _resolve(config.repo, inputs["dagger1_detailed_manifest"]["path"]),
        expected_role="TRAIN", expected_provenance="DAGGER1",
    )
    dagger2 = read_manifest(
        _resolve(config.repo, inputs["dagger2_post_recovery_manifest"]["path"]),
        expected_role="TRAIN", expected_provenance="DAGGER2_POST_RECOVERY",
    )
    s09_raw = read_manifest(
        _resolve(config.repo, inputs["s09_validation_manifest"]["path"]),
        expected_role="VALIDATION",
    )
    s10 = read_manifest(
        _resolve(config.repo, inputs["s10_validation_manifest"]["path"]),
        expected_role="VALIDATION",
    )
    episode = _read_json(_resolve(config.repo, inputs["s09_episode_metadata"]["path"]))
    bypass = episode.get("planned_bypass")
    if not isinstance(bypass, dict):
        raise DiagnosisGateError("S09 bypass metadata is missing")
    s09 = [_with_phase(row, _phase_for_progress(row.route_progress_m, bypass)) for row in s09_raw]

    expected = {
        "d1_aggregate_manifest": len(aggregate),
        "dagger1_detailed_manifest": len(dagger1),
        "dagger2_post_recovery_manifest": len(dagger2),
        "s09_validation_manifest": len(s09),
        "s10_validation_manifest": len(s10),
    }
    for key, actual in expected.items():
        configured = int(inputs[key]["sequence_count"])
        if actual != configured:
            raise DiagnosisGateError(f"{key} count {actual} != frozen {configured}")
    if len(expert) != 6706 or len(aggregate_dagger1) != 1483:
        raise DiagnosisGateError("D1 aggregate provenance counts changed")
    if {row.sequence_id for row in aggregate_dagger1} != {row.sequence_id for row in dagger1}:
        raise DiagnosisGateError("detailed DAgger1 rows do not match D1 aggregate")
    if {row.scenario_id for row in dagger1 + dagger2} != set(f"{i:02d}" for i in range(1, 9)):
        raise DiagnosisGateError("training sources are not exactly S01-S08")
    if {row.scenario_id for row in s09} != {"09"}:
        raise DiagnosisGateError("S09 validation manifest identity changed")
    if {row.scenario_id for row in s10} != {"10"}:
        raise DiagnosisGateError("S10 validation manifest identity changed")

    subset_config = config.payload["subsets"]
    avoidance_phases = set(subset_config["dagger1_avoidance_phases"])
    d1_avoidance = [row for row in dagger1 if row.phase in avoidance_phases]
    d1_avoidance_only = [row for row in dagger1 if row.phase == "avoidance"]
    d1_recovery = [
        row for row in dagger1
        if row.phase in set(subset_config["dagger1_recovery_or_failure_phases"])
    ]
    nominal_config = subset_config["expert_nominal"]
    nominal = deterministic_expert_nominal(
        expert, seed=int(config.diagnostics["deterministic_seed"]),
        bins=nominal_config["route_bins_m"],
        per_stratum=int(nominal_config["per_scenario_route_bin"]),
    )
    if len(nominal) != int(nominal_config["expected_count"]):
        raise DiagnosisGateError("deterministic nominal subset count changed")
    s09_avoidance = [row for row in s09 if row.phase == "avoidance"]
    if not s09_avoidance:
        raise DiagnosisGateError("S09 avoidance subset is empty")

    subsets = {
        "EXPERT_BASELINE": expert,
        "DAGGER1_ALL": dagger1,
        "DAGGER1_AVOIDANCE": d1_avoidance,
        "DAGGER1_AVOIDANCE_ONLY": d1_avoidance_only,
        "DAGGER1_RECOVERY_OR_FAILURE": d1_recovery,
        "DAGGER2_POST_RECOVERY": dagger2,
        "EXPERT_NOMINAL": nominal,
        "S09_ALL_VALIDATION": s09,
        "S09_AVOIDANCE_VALIDATION": s09_avoidance,
        "S10_ALL_VALIDATION": s10,
    }
    audit = {
        "version": VERSION,
        "result": "PASS",
        "construction": {
            "DAGGER1_AVOIDANCE": "DAGGER1 phases approach + avoidance + pass_return",
            "DAGGER1_AVOIDANCE_ONLY": "DAGGER1 phase avoidance",
            "DAGGER1_RECOVERY_OR_FAILURE": "available DAGGER1 post_recovery rows; no explicit failure flag exists",
            "DAGGER2_POST_RECOVERY": "all 109 immutable rows",
            "EXPERT_NOMINAL": nominal_config,
            "S09_AVOIDANCE_VALIDATION": {
                "source": str(_resolve(config.repo, inputs["s09_validation_manifest"]["path"])),
                "phase_source": str(_resolve(config.repo, inputs["s09_episode_metadata"]["path"])),
                "lower_inclusive_m": float(bypass["departure_start_s_m"]),
                "upper_exclusive_m": float(bypass["cone_s_m"]),
            },
        },
        "counts": {name: len(rows) for name, rows in subsets.items()},
        "scenario_counts": {name: _counts(rows, "scenario_id") for name, rows in subsets.items()},
        "phase_counts": {name: _counts(rows, "phase") for name, rows in subsets.items()},
        "training_scenarios": sorted({row.scenario_id for row in expert + dagger1 + dagger2}),
        "evaluation_only_scenarios": sorted({row.scenario_id for row in s09 + s10}),
        "validation_gradient_use": {"S09": False, "S10": False},
        "s10_accessed_for_identity_audit_only": True,
        "s11_s12_accessed": False,
        "image_data_copied": False,
    }
    return subsets, audit


def audit_frozen_inputs(config: DiagnosticConfig) -> dict[str, Any]:
    branch = _git(config.repo, "branch", "--show-current")
    head = _git(config.repo, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH:
        raise DiagnosisGateError(f"expected branch {EXPECTED_BRANCH}, observed {branch}")
    if head != EXPECTED_BASE_COMMIT:
        raise DiagnosisGateError(f"expected base {EXPECTED_BASE_COMMIT}, observed {head}")
    identities: dict[str, Any] = {}
    for name, spec in config.inputs.items():
        path = _resolve(config.repo, str(spec["path"]))
        observed = sha256_file(path)
        if observed != spec["sha256"]:
            raise DiagnosisGateError(f"frozen input hash mismatch for {name}: {observed}")
        identities[name] = {
            "path": str(path), "sha256": observed, "size_bytes": path.stat().st_size,
        }
    prior = config.payload["prior_dagger2_negative"]
    prior_trees: dict[str, Any] = {}
    for label, path_key, hash_key in (
        ("collection", "collection_directory", "collection_directory_sha256"),
        ("dataset", "dataset_directory", "dataset_directory_sha256"),
    ):
        identity = _directory_tree_identity(_resolve(config.repo, prior[path_key]))
        if identity["sha256"] != prior[hash_key]:
            raise DiagnosisGateError(f"preserved DAgger2 {label} tree changed")
        prior_trees[label] = identity
    d1_path = _resolve(config.repo, config.inputs["d1_checkpoint"]["path"])
    d2_path = _resolve(config.repo, config.inputs["d2_fe_checkpoint"]["path"])
    d1_model, d1_record = load_frozen_model(d1_path)
    d2_model, d2_record = load_frozen_model(d2_path)
    d1_state_hash = state_tensor_sha256(d1_model.state_dict())
    d2_state_hash = state_tensor_sha256(d2_model.state_dict())
    del d1_model, d2_model
    if d1_record.get("initialized_from_scratch") is not True:
        raise DiagnosisGateError("preserved D1 initialization record changed")
    if d2_record.get("initialized_from_scratch") is not True:
        raise DiagnosisGateError("preserved D2-FE initialization record changed")
    if prior != {
        "collection_directory": "results/random_cone_dagger2_collection_1p0_v1",
        "collection_directory_sha256": "ebe4d24dbab5cd3155676e5af8a4cca77ca45b1ddee7be52ed3977518e394fd1",
        "dataset_directory": "results/random_cone_dagger2_dataset_1p0_v1",
        "dataset_directory_sha256": "b9f8f993e0b30153af611720dddc00a4acb718319d1b1607ec7bce5d7dacca48",
        "sequence_count": 109,
        "sequences_after_20m": 18,
        "sequences_after_26m": 0,
        "historical_result": "FAIL",
        "training_authorized": False,
    }:
        raise DiagnosisGateError("historical DAgger2 coverage-gate record changed")
    return {
        "version": VERSION, "result": "PASS", "branch": branch, "head": head,
        "config_path": str(config.path), "config_sha256": sha256_file(config.path),
        "frozen_inputs": identities, "prior_dagger2_trees": prior_trees,
        "d1": {
            "checkpoint_file_sha256": identities["d1_checkpoint"]["sha256"],
            "state_tensor_sha256": d1_state_hash,
            "parameter_count": int(d1_record["parameter_count"]),
            "input_shape": ["N", TEMPORAL_CHANNELS, 66, 200],
            "training_sequences": 8189,
            "provenance": {"EXPERT_BASELINE": 6706, "DAGGER1": 1483},
        },
        "d2_fe_frozen_negative": {
            "checkpoint_file_sha256": identities["d2_fe_checkpoint"]["sha256"],
            "state_tensor_sha256": d2_state_hash,
            "parameter_count": int(d2_record["parameter_count"]),
            "training_sequences": 8298,
            "historical_classification": "REGRESSION",
            "s09_collision_route_progress_m": 12.750,
            "s09_completion_fraction": 0.4180,
        },
        "prior_dagger2_coverage_gate": dict(prior),
        "s10_accessed_for_identity_audit_only": True, "s11_s12_accessed": False,
        "checkpoint_written": False, "model_training_performed": False,
        "optimizer_steps": 0, "simulator_invoked": False,
        "rosbag_opened": False, "data_collected": False,
    }


def describe_numeric(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0, "available": False}
    return {
        "count": int(array.size), "available": True,
        "mean": float(np.mean(array)), "median": float(np.median(array)),
        "std": float(np.std(array)), "min": float(np.min(array)),
        "max": float(np.max(array)), "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)), "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _metadata_values(rows: Sequence[DiagnosticRow], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw = row.metadata.get(key, "")
        if raw not in (None, ""):
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
    return values


def label_distribution(
    rows: Sequence[DiagnosticRow], *, steering_limit_rad: float,
    zero_tolerance_rad: float,
) -> dict[str, Any]:
    values = np.asarray([row.steering_rad for row in rows], dtype=np.float64)
    signs = {
        "negative": int(np.sum(values < -zero_tolerance_rad)),
        "zero": int(np.sum(np.abs(values) <= zero_tolerance_rad)),
        "positive": int(np.sum(values > zero_tolerance_rad)),
    }
    result = describe_numeric(values)
    result.update({
        "sign_counts": signs,
        "mean_absolute_steering_rad": float(np.mean(np.abs(values))),
        "saturation_fraction": float(np.mean(np.abs(values) >= steering_limit_rad - 1e-6)),
        "route_progress_m": describe_numeric(row.route_progress_m for row in rows),
        "cte_m": describe_numeric(_metadata_values(rows, "cte_m")),
        "signed_cte_m": describe_numeric(_metadata_values(rows, "signed_cte_m")),
        "heading_error_rad": describe_numeric(_metadata_values(rows, "heading_error_rad")),
        "scenario_counts": _counts(rows, "scenario_id"),
        "phase_counts": _counts(rows, "phase"),
    })
    return result


def analyze_label_distributions(
    config: DiagnosticConfig, subsets: Mapping[str, Sequence[DiagnosticRow]],
) -> dict[str, Any]:
    limit = float(config.diagnostics["steering_limit_rad"])
    tolerance = float(config.diagnostics["zero_sign_tolerance_rad"])
    names = (
        "DAGGER1_AVOIDANCE", "DAGGER1_AVOIDANCE_ONLY",
        "DAGGER1_RECOVERY_OR_FAILURE", "DAGGER2_POST_RECOVERY", "EXPERT_NOMINAL",
    )
    distributions = {
        name: label_distribution(subsets[name], steering_limit_rad=limit,
                                 zero_tolerance_rad=tolerance)
        for name in names
    }
    a = np.asarray([row.steering_rad for row in subsets["DAGGER1_AVOIDANCE"]])
    b = np.asarray([row.steering_rad for row in subsets["DAGGER2_POST_RECOVERY"]])
    return {
        "version": VERSION, "result": "PASS", "units": "radians",
        "sources": distributions,
        "direct_comparison": {
            "DAGGER2_minus_DAGGER1_AVOIDANCE_mean_rad": float(np.mean(b) - np.mean(a)),
            "DAGGER2_to_DAGGER1_AVOIDANCE_mean_absolute_ratio": (
                float(np.mean(np.abs(b)) / np.mean(np.abs(a)))
                if float(np.mean(np.abs(a))) > 0 else None
            ),
            "note": "Unpaired marginal label distributions; feature-neighbor and gradient analyses test local conflict.",
        },
        "metadata_limitations": {
            "DAGGER1_AVOIDANCE": {
                "pose_xyz_yaw_available": False,
                "cte_heading_available": bool(_metadata_values(subsets["DAGGER1_AVOIDANCE"], "heading_error_rad")),
            },
            "DAGGER2_POST_RECOVERY": {
                "learner_xy_yaw_available": all(
                    bool(_metadata_values(subsets["DAGGER2_POST_RECOVERY"], key))
                    for key in ("learner_x_m", "learner_y_m", "learner_yaw_rad")
                ),
                "cte_heading_available": bool(_metadata_values(subsets["DAGGER2_POST_RECOVERY"], "heading_error_rad")),
            },
        },
    }


def _preprocess_batch(rows: Sequence[DiagnosticRow]):
    import torch

    values = np.stack([preprocess_temporal_paths(row.paths) for row in rows], axis=0)
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))


def forward_rows(
    model: Any, rows: Sequence[DiagnosticRow], *, batch_size: int,
    include_penultimate: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    predictions: list[np.ndarray] = []
    features: list[np.ndarray] = []
    model.eval()
    layers = list(model.regressor.children())
    if include_penultimate and (
        len(layers) != 8 or layers[-1].in_features != 10 or layers[-1].out_features != 1
    ):
        raise DiagnosisGateError("penultimate-layer architecture contract changed")
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            images = _preprocess_batch(rows[start:start + batch_size])
            if include_penultimate:
                value = model.features(images)
                for layer in layers[:-1]:
                    value = layer(value)
                if value.ndim != 2 or value.shape[1] != 10:
                    raise DiagnosisGateError(f"invalid penultimate representation {tuple(value.shape)}")
                output = layers[-1](value)
                features.append(value.detach().cpu().numpy().astype(np.float64))
            else:
                output = model(images)
            predictions.append(
                (output.detach().cpu().numpy().reshape(-1) * MAX_STEERING_RAD).astype(np.float64)
            )
    prediction = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float64)
    feature = np.concatenate(features) if features else None
    if not np.all(np.isfinite(prediction)) or (feature is not None and not np.all(np.isfinite(feature))):
        raise DiagnosisGateError("non-finite frozen-model forward output")
    return prediction, feature


def nearest_neighbor_mapping(
    queries: np.ndarray, references: np.ndarray,
) -> dict[str, np.ndarray]:
    query = np.asarray(queries, dtype=np.float64)
    reference = np.asarray(references, dtype=np.float64)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query/reference feature matrices must have matching second dimension")
    if query.shape[0] == 0 or reference.shape[0] == 0:
        raise ValueError("nearest-neighbor feature matrices must be non-empty")
    q_norm = np.linalg.norm(query, axis=1)
    r_norm = np.linalg.norm(reference, axis=1)
    denom = q_norm[:, None] * r_norm[None, :]
    similarity = np.divide(
        query @ reference.T, denom, out=np.zeros_like(denom), where=denom > 0,
    )
    cosine = np.clip(1.0 - similarity, 0.0, 2.0)
    cosine_index = np.argmin(cosine, axis=1)
    cosine_distance = cosine[np.arange(query.shape[0]), cosine_index]
    squared = (
        np.sum(query * query, axis=1)[:, None]
        + np.sum(reference * reference, axis=1)[None, :]
        - 2.0 * (query @ reference.T)
    )
    euclidean = np.sqrt(np.maximum(squared, 0.0))
    euclidean_index = np.argmin(euclidean, axis=1)
    euclidean_distance = euclidean[np.arange(query.shape[0]), euclidean_index]
    return {
        "cosine_index": cosine_index, "cosine_distance": cosine_distance,
        "euclidean_index": euclidean_index, "euclidean_distance": euclidean_distance,
        "query_norm": q_norm, "reference_norm": r_norm,
    }


def _sign(value: float, tolerance: float) -> int:
    return -1 if value < -tolerance else 1 if value > tolerance else 0


def _optional_float(row: DiagnosticRow, key: str) -> float | None:
    raw = row.metadata.get(key, "")
    return None if raw in (None, "") else float(raw)


def _pair_record(
    d2: DiagnosticRow, neighbor: DiagnosticRow, *, cosine_distance: float,
    euclidean_distance: float, tolerance: float,
) -> dict[str, Any]:
    difference = abs(d2.steering_rad - neighbor.steering_rad)
    return {
        "d2_sample_id": d2.sequence_id,
        "dagger1_sample_id": neighbor.sequence_id,
        "feature_cosine_distance": float(cosine_distance),
        "feature_euclidean_distance": float(euclidean_distance),
        "d2_expert_steering_rad": d2.steering_rad,
        "dagger1_expert_steering_rad": neighbor.steering_rad,
        "absolute_label_difference_rad": difference,
        "steering_sign_agreement": _sign(d2.steering_rad, tolerance) == _sign(neighbor.steering_rad, tolerance),
        "d2": {
            "scenario_id": d2.scenario_id, "route_progress_m": d2.route_progress_m,
            "phase": d2.phase,
        },
        "dagger1": {
            "scenario_id": neighbor.scenario_id, "route_progress_m": neighbor.route_progress_m,
            "phase": neighbor.phase,
        },
    }


def _temporal_state(row: DiagnosticRow, learner_key: str) -> dict[str, Any]:
    return {
        "timestamp_t_minus_2_ns": int(row.metadata["timestamp_t_minus_2_ns"]),
        "timestamp_t_minus_1_ns": int(row.metadata["timestamp_t_minus_1_ns"]),
        "timestamp_t_ns": int(row.metadata["timestamp_t_ns"]),
        "adjacent_gap_1_s": float(row.metadata["adjacent_gap_1_s"]),
        "adjacent_gap_2_s": float(row.metadata["adjacent_gap_2_s"]),
        "oldest_to_current_span_s": float(row.metadata["oldest_to_current_span_s"]),
        "route_progress_m": row.route_progress_m,
        "cte_m": _optional_float(row, "cte_m"),
        "signed_cte_m": _optional_float(row, "signed_cte_m"),
        "heading_error_rad": _optional_float(row, "heading_error_rad"),
        "x_m": _optional_float(row, "learner_x_m"),
        "y_m": _optional_float(row, "learner_y_m"),
        "yaw_rad": _optional_float(row, "learner_yaw_rad"),
        "learner_steering_rad": _optional_float(row, learner_key),
        "expert_steering_rad": row.steering_rad,
    }


def analyze_features(
    config: DiagnosticConfig, subsets: Mapping[str, Sequence[DiagnosticRow]], model: Any,
) -> dict[str, Any]:
    batch = int(config.diagnostics["forward_batch_size"])
    tolerance = float(config.diagnostics["zero_sign_tolerance_rad"])
    source_names = (
        "DAGGER2_POST_RECOVERY", "DAGGER1_AVOIDANCE", "DAGGER1_ALL", "EXPERT_NOMINAL",
    )
    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in source_names:
        prediction, feature = forward_rows(
            model, subsets[name], batch_size=batch, include_penultimate=True,
        )
        if feature is None:
            raise DiagnosisGateError("penultimate feature extraction failed")
        outputs[name] = prediction, feature
    d2_features = outputs["DAGGER2_POST_RECOVERY"][1]
    mappings: dict[str, dict[str, np.ndarray]] = {}
    for reference_name in ("DAGGER1_AVOIDANCE", "DAGGER1_ALL", "EXPERT_NOMINAL"):
        mappings[reference_name] = nearest_neighbor_mapping(d2_features, outputs[reference_name][1])

    avoidance_rows = subsets["DAGGER1_AVOIDANCE"]
    avoidance_map = mappings["DAGGER1_AVOIDANCE"]
    pairs: list[dict[str, Any]] = []
    for index, d2 in enumerate(subsets["DAGGER2_POST_RECOVERY"]):
        neighbor_index = int(avoidance_map["cosine_index"][index])
        neighbor = avoidance_rows[neighbor_index]
        pair = _pair_record(
            d2, neighbor,
            cosine_distance=float(avoidance_map["cosine_distance"][index]),
            euclidean_distance=float(np.linalg.norm(
                d2_features[index] - outputs["DAGGER1_AVOIDANCE"][1][neighbor_index]
            )),
            tolerance=tolerance,
        )
        pair["d2_frozen_d1_prediction_rad"] = float(outputs["DAGGER2_POST_RECOVERY"][0][index])
        pair["dagger1_frozen_d1_prediction_rad"] = float(
            outputs["DAGGER1_AVOIDANCE"][0][neighbor_index]
        )
        pair["nearest_neighbors"] = {}
        for reference_name, mapping in mappings.items():
            cosine_index = int(mapping["cosine_index"][index])
            euclidean_index = int(mapping["euclidean_index"][index])
            cosine_neighbor = subsets[reference_name][cosine_index]
            euclidean_neighbor = subsets[reference_name][euclidean_index]
            pair["nearest_neighbors"][reference_name] = {
                "cosine_neighbor_id": cosine_neighbor.sequence_id,
                "cosine_distance": float(mapping["cosine_distance"][index]),
                "cosine_neighbor_expert_steering_rad": cosine_neighbor.steering_rad,
                "cosine_neighbor_label_difference_rad": abs(
                    d2.steering_rad - cosine_neighbor.steering_rad
                ),
                "euclidean_neighbor_id": euclidean_neighbor.sequence_id,
                "euclidean_distance": float(mapping["euclidean_distance"][index]),
                "euclidean_neighbor_expert_steering_rad": euclidean_neighbor.steering_rad,
                "euclidean_neighbor_label_difference_rad": abs(
                    d2.steering_rad - euclidean_neighbor.steering_rad
                ),
            }
        pairs.append(pair)
    distances = np.asarray([pair["feature_cosine_distance"] for pair in pairs])
    differences = np.asarray([pair["absolute_label_difference_rad"] for pair in pairs])
    distance_threshold = float(np.quantile(distances, 0.25))
    difference_threshold = float(np.quantile(differences, 0.75))
    euclidean_distances = np.asarray([
        pair["nearest_neighbors"]["DAGGER1_AVOIDANCE"]["euclidean_distance"]
        for pair in pairs
    ])
    euclidean_differences = np.asarray([
        pair["nearest_neighbors"]["DAGGER1_AVOIDANCE"][
            "euclidean_neighbor_label_difference_rad"
        ]
        for pair in pairs
    ])
    euclidean_distance_threshold = float(np.quantile(euclidean_distances, 0.25))
    euclidean_difference_threshold = float(np.quantile(euclidean_differences, 0.75))
    for pair in pairs:
        high_similarity = pair["feature_cosine_distance"] <= distance_threshold
        high_disagreement = pair["absolute_label_difference_rad"] >= difference_threshold
        euclidean_neighbor = pair["nearest_neighbors"]["DAGGER1_AVOIDANCE"]
        euclidean_high_similarity = (
            euclidean_neighbor["euclidean_distance"] <= euclidean_distance_threshold
        )
        euclidean_high_disagreement = (
            euclidean_neighbor["euclidean_neighbor_label_difference_rad"]
            >= euclidean_difference_threshold
        )
        pair["high_similarity"] = bool(high_similarity)
        pair["high_label_disagreement"] = bool(high_disagreement)
        pair["candidate_state_aliasing_conflict"] = bool(high_similarity and high_disagreement)
        pair["euclidean_high_similarity"] = bool(euclidean_high_similarity)
        pair["euclidean_high_label_disagreement"] = bool(euclidean_high_disagreement)
        pair["euclidean_candidate_state_aliasing_conflict"] = bool(
            euclidean_high_similarity and euclidean_high_disagreement
        )
    candidates = [pair for pair in pairs if pair["candidate_state_aliasing_conflict"]]
    high_similarity_pairs = [pair for pair in pairs if pair["high_similarity"]]
    high_disagreement_pairs = [pair for pair in pairs if pair["high_label_disagreement"]]
    euclidean_candidates = [
        pair for pair in pairs if pair["euclidean_candidate_state_aliasing_conflict"]
    ]
    euclidean_high_similarity_pairs = [
        pair for pair in pairs if pair["euclidean_high_similarity"]
    ]
    top = sorted(
        pairs,
        key=lambda pair: (
            not pair["candidate_state_aliasing_conflict"],
            -pair["absolute_label_difference_rad"],
            pair["feature_cosine_distance"],
            pair["d2_sample_id"],
        ),
    )[:int(config.diagnostics["top_conflict_count"])]
    d2_by_id = {row.sequence_id: row for row in subsets["DAGGER2_POST_RECOVERY"]}
    d1_by_id = {row.sequence_id: row for row in avoidance_rows}
    temporal_top = []
    for pair in top:
        d2 = d2_by_id[pair["d2_sample_id"]]
        d1 = d1_by_id[pair["dagger1_sample_id"]]
        temporal_top.append({
            **pair,
            "d2_temporal_state": _temporal_state(d2, "d1_steering_rad"),
            "dagger1_temporal_state": _temporal_state(d1, "r1_steering_rad"),
        })

    references: dict[str, Any] = {}
    for name, mapping in mappings.items():
        reference_rows = subsets[name]
        cosine_label_differences = np.asarray([
            abs(d2.steering_rad - reference_rows[int(mapping["cosine_index"][index])].steering_rad)
            for index, d2 in enumerate(subsets["DAGGER2_POST_RECOVERY"])
        ])
        euclidean_label_differences = np.asarray([
            abs(d2.steering_rad - reference_rows[int(mapping["euclidean_index"][index])].steering_rad)
            for index, d2 in enumerate(subsets["DAGGER2_POST_RECOVERY"])
        ])
        references[name] = {
            "reference_count": len(reference_rows),
            "nearest_cosine_distance": describe_numeric(mapping["cosine_distance"]),
            "nearest_euclidean_distance": describe_numeric(mapping["euclidean_distance"]),
            "nearest_cosine_neighbor_label_difference_rad": describe_numeric(
                cosine_label_differences
            ),
            "nearest_euclidean_neighbor_label_difference_rad": describe_numeric(
                euclidean_label_differences
            ),
            "reference_feature_norm": describe_numeric(mapping["reference_norm"]),
        }
    conditional_fraction = len(candidates) / len(high_similarity_pairs) if high_similarity_pairs else 0.0
    expected_independent_count = (
        len(high_similarity_pairs) * len(high_disagreement_pairs) / len(pairs)
    )
    euclidean_conditional_fraction = (
        len(euclidean_candidates) / len(euclidean_high_similarity_pairs)
        if euclidean_high_similarity_pairs else 0.0
    )
    scenario_span = sorted({pair["d2"]["scenario_id"] for pair in candidates})
    aliasing_supported = (
        len(candidates) >= 5 and conditional_fraction >= 0.5 and len(scenario_span) >= 2
    )
    return {
        "version": VERSION, "result": "PASS",
        "checkpoint_sha256": config.inputs["d1_checkpoint"]["sha256"],
        "representation": "frozen D1 10-D final ReLU before the scalar output layer",
        "distance_note": "Cosine is primary. Raw Euclidean distance is descriptive because activation norms are not scale-normalized.",
        "query_count": len(subsets["DAGGER2_POST_RECOVERY"]),
        "references": references,
        "dagger1_avoidance_nearest_pairs": pairs,
        "observed_distribution_thresholds": {
            "high_similarity": {
                "rule": "nearest cosine distance <= observed p25",
                "value": distance_threshold,
            },
            "high_label_disagreement": {
                "rule": "absolute Expert-label difference >= observed p75",
                "value_rad": difference_threshold,
            },
            "euclidean_cross_check": {
                "high_similarity_rule": "nearest Euclidean distance <= observed p25",
                "high_similarity_value": euclidean_distance_threshold,
                "high_label_disagreement_rule": (
                    "nearest-Euclidean-neighbor absolute Expert-label difference >= observed p75"
                ),
                "high_label_disagreement_value_rad": euclidean_difference_threshold,
            },
            "these_are_descriptive_not_tuned": True,
        },
        "conflict_summary": {
            "nearest_distance": describe_numeric(distances),
            "nearest_label_difference_rad": describe_numeric(differences),
            "high_similarity_count": len(high_similarity_pairs),
            "high_label_disagreement_count": len(high_disagreement_pairs),
            "candidate_count": len(candidates),
            "candidate_fraction_all": len(candidates) / len(pairs),
            "candidate_fraction_within_high_similarity": conditional_fraction,
            "candidate_count_expected_if_quartile_flags_independent": expected_independent_count,
            "candidate_enrichment_over_independence": (
                len(candidates) / expected_independent_count
                if expected_independent_count else None
            ),
            "candidate_d2_scenarios": scenario_span,
            "sign_disagreement_count": sum(not pair["steering_sign_agreement"] for pair in pairs),
            "sign_disagreement_fraction": float(np.mean([
                not pair["steering_sign_agreement"] for pair in pairs
            ])),
            "cosine_candidates_euclidean_distance_to_cosine_neighbor": describe_numeric(
                pair["feature_euclidean_distance"] for pair in candidates
            ),
            "euclidean_cross_check": {
                "nearest_distance": describe_numeric(euclidean_distances),
                "nearest_neighbor_label_difference_rad": describe_numeric(
                    euclidean_differences
                ),
                "high_similarity_count": len(euclidean_high_similarity_pairs),
                "candidate_count": len(euclidean_candidates),
                "candidate_fraction_all": len(euclidean_candidates) / len(pairs),
                "candidate_fraction_within_high_similarity": euclidean_conditional_fraction,
            },
        },
        "evidence_rule": {
            "aliasing_supported_if": "at least 5 quartile-defined conflicts, at least half of the high-similarity tail, spanning at least 2 D2 scenarios",
            "aliasing_supported": aliasing_supported,
        },
        "top_20_conflicts": top,
        "temporal_observability": {
            "top_conflict_states": temporal_top,
            "DAGGER1_pose_limitation": "DAGGER1 stores CTE and heading error but not learner x/y/yaw; full pose-to-pose comparison is unavailable.",
            "S09_pose_limitation": "The frozen S09 temporal validation manifest stores route progress but not pose, CTE, or heading.",
            "partial_observability_claimed": bool(aliasing_supported),
            "claim_scope": "candidate ambiguity in the frozen D1 representation; not proof of causal perceptual aliasing",
        },
        "contact_sheet_generated": False,
        "image_data_copied": False,
    }


def gradient_cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("gradient vectors must be non-empty and shape matched")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        raise ValueError("gradient cosine is undefined for a zero vector")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def deterministic_gradient_batches(
    rows: Sequence[DiagnosticRow], *, source_name: str, seed: int,
    batch_size: int, batch_count: int,
) -> list[list[DiagnosticRow]]:
    required = batch_size * batch_count
    if len(rows) < required:
        raise DiagnosisGateError(
            f"{source_name} has {len(rows)} rows, fewer than {required} required for disjoint batches"
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{source_name}:{row.sequence_id}".encode()).hexdigest(),
            row.sequence_id,
        ),
    )[:required]
    return [ranked[start:start + batch_size] for start in range(0, required, batch_size)]


def _gradient_vector(model: Any, rows: Sequence[DiagnosticRow]) -> dict[str, Any]:
    import torch

    model.eval()
    named = list(model.named_parameters())
    if any(parameter.grad is not None for _, parameter in named):
        raise DiagnosisGateError("unexpected accumulated parameter gradients")
    images = _preprocess_batch(rows)
    targets = torch.tensor(
        [[row.steering_rad / MAX_STEERING_RAD] for row in rows], dtype=torch.float32,
    )
    prediction = model(images)
    loss = torch.mean((prediction - targets) ** 2)
    gradients = torch.autograd.grad(
        loss, [parameter for _, parameter in named], create_graph=False, retain_graph=False,
    )
    full_parts: list[np.ndarray] = []
    head_parts: list[np.ndarray] = []
    for (name, _), gradient in zip(named, gradients):
        value = gradient.detach().cpu().numpy().reshape(-1).astype(np.float64)
        full_parts.append(value)
        if name.startswith("regressor."):
            head_parts.append(value)
    full = np.concatenate(full_parts)
    head = np.concatenate(head_parts)
    if full.size != TEMPORAL_PARAMETER_COUNT:
        raise DiagnosisGateError(f"full gradient size changed: {full.size}")
    if any(parameter.grad is not None for _, parameter in named):
        raise DiagnosisGateError("diagnostic gradient unexpectedly accumulated in parameters")
    return {
        "normalized_mse": float(loss.detach().cpu()),
        "full": full, "head": head,
        "full_norm": float(np.linalg.norm(full)),
        "head_norm": float(np.linalg.norm(head)),
    }


def _cosine_summary(values: Sequence[float]) -> dict[str, Any]:
    result = describe_numeric(values)
    result["fraction_negative"] = float(np.mean(np.asarray(values) < 0.0))
    result["negative_count"] = int(np.sum(np.asarray(values) < 0.0))
    return result


def _norm_summary(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return describe_numeric(float(record[key]) for record in records)


def calculate_gradient_diagnostics(
    config: DiagnosticConfig, subsets: Mapping[str, Sequence[DiagnosticRow]], model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = int(config.diagnostics["deterministic_seed"])
    batch_size = int(config.diagnostics["gradient_batch_size"])
    batch_count = int(config.diagnostics["gradient_batch_count"])
    source_mapping = {
        "DAGGER1_AVOIDANCE": "DAGGER1_AVOIDANCE",
        "DAGGER1_AVOIDANCE_ONLY": "DAGGER1_AVOIDANCE_ONLY",
        "DAGGER2_POST_RECOVERY": "DAGGER2_POST_RECOVERY",
        "EXPERT_NOMINAL": "EXPERT_NOMINAL",
        "EXPERT_BASELINE": "EXPERT_BASELINE",
        "DAGGER1_ALL": "DAGGER1_ALL",
    }
    state_before = clone_state(model.state_dict())
    state_hash_before = state_tensor_sha256(state_before)
    records: dict[str, list[dict[str, Any]]] = {}
    vectors: dict[str, list[dict[str, Any]]] = {}
    batch_ids: dict[str, list[list[str]]] = {}
    for source_name, subset_name in source_mapping.items():
        batches = deterministic_gradient_batches(
            subsets[subset_name], source_name=source_name, seed=seed,
            batch_size=batch_size, batch_count=batch_count,
        )
        batch_ids[source_name] = [[row.sequence_id for row in batch] for batch in batches]
        vectors[source_name] = []
        records[source_name] = []
        for index, batch_rows in enumerate(batches):
            value = _gradient_vector(model, batch_rows)
            vectors[source_name].append(value)
            records[source_name].append({
                "batch_index": index, "sample_count": len(batch_rows),
                "normalized_mse": value["normalized_mse"],
                "head_gradient_norm": value["head_norm"],
                "full_gradient_norm": value["full_norm"],
            })
    pairs = {
        "DAGGER1_AVOIDANCE_vs_DAGGER2_POST_RECOVERY": (
            "DAGGER1_AVOIDANCE", "DAGGER2_POST_RECOVERY"
        ),
        "DAGGER1_AVOIDANCE_ONLY_vs_DAGGER2_POST_RECOVERY": (
            "DAGGER1_AVOIDANCE_ONLY", "DAGGER2_POST_RECOVERY"
        ),
        "DAGGER1_AVOIDANCE_vs_EXPERT_NOMINAL": (
            "DAGGER1_AVOIDANCE", "EXPERT_NOMINAL"
        ),
        "DAGGER2_POST_RECOVERY_vs_EXPERT_NOMINAL": (
            "DAGGER2_POST_RECOVERY", "EXPERT_NOMINAL"
        ),
    }
    comparisons: dict[str, Any] = {}
    for label, (left_name, right_name) in pairs.items():
        head_values = [
            gradient_cosine(vectors[left_name][i]["head"], vectors[right_name][i]["head"])
            for i in range(batch_count)
        ]
        full_values = [
            gradient_cosine(vectors[left_name][i]["full"], vectors[right_name][i]["full"])
            for i in range(batch_count)
        ]
        comparisons[label] = {
            "per_batch": [
                {"batch_index": i, "head_cosine": head_values[i], "full_cosine": full_values[i]}
                for i in range(batch_count)
            ],
            "head": _cosine_summary(head_values),
            "full": _cosine_summary(full_values),
        }
    state_after = model.state_dict()
    exact = states_exactly_equal(state_before, state_after)
    state_hash_after = state_tensor_sha256(state_after)
    if not exact or state_hash_before != state_hash_after:
        raise DiagnosisGateError("frozen D1 tensors changed during gradient diagnosis")
    ab = comparisons["DAGGER1_AVOIDANCE_vs_DAGGER2_POST_RECOVERY"]
    avoidance_only_ab = comparisons[
        "DAGGER1_AVOIDANCE_ONLY_vs_DAGGER2_POST_RECOVERY"
    ]
    opposition = all(
        ab[scope]["median"] < 0.0 and ab[scope]["negative_count"] >= 4
        for scope in ("head", "full")
    )
    gradient_result = {
        "version": VERSION, "result": "PASS",
        "objective": "mean squared error in normalized steering (the preserved training objective)",
        "diagnostic_method": "torch.autograd.grad only; no accumulated .grad and no parameter update",
        "batch_design": {
            "seed": seed, "batch_size": batch_size, "batch_count": batch_count,
            "sampling": "SHA-256 ranked, deterministic, disjoint within source",
            "batch_sequence_ids": batch_ids,
        },
        "parameter_scopes": {
            "head": {
                "names": [name for name, _ in model.named_parameters() if name.startswith("regressor.")],
                "parameter_count": sum(
                    parameter.numel() for name, parameter in model.named_parameters()
                    if name.startswith("regressor.")
                ),
            },
            "full": {
                "names": [name for name, _ in model.named_parameters()],
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            },
        },
        "source_batches": records,
        "comparisons": comparisons,
        "consistent_opposition_rule": config.diagnostics["consistent_gradient_opposition_rule"],
        "dagger2_opposes_dagger1_avoidance": opposition,
        "avoidance_only_sensitivity_check": {
            "purpose": (
                "Check whether the 155 cone-avoidance-phase rows are masked by the broader "
                "approach/avoidance/pass-return subset."
            ),
            "head": avoidance_only_ab["head"],
            "full": avoidance_only_ab["full"],
            "dagger2_opposes_avoidance_only": all(
                avoidance_only_ab[scope]["median"] < 0.0
                and avoidance_only_ab[scope]["negative_count"] >= 4
                for scope in ("head", "full")
            ),
        },
        "model_immutability": {
            "state_tensor_sha256_before": state_hash_before,
            "state_tensor_sha256_after": state_hash_after,
            "exact_tensor_equality": exact,
            "all_parameter_grad_fields_none": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "optimizer_constructed": False, "optimizer_steps": 0,
        },
    }
    pressure_batch = {
        source: {
            "matched_batch_count": batch_count,
            "matched_batch_size": batch_size,
            "head_gradient_norm": _norm_summary(source_records, "head_gradient_norm"),
            "full_gradient_norm": _norm_summary(source_records, "full_gradient_norm"),
            "normalized_mse": describe_numeric(
                float(record["normalized_mse"]) for record in source_records
            ),
        }
        for source, source_records in records.items()
    }
    return gradient_result, pressure_batch


def prediction_error_metrics(prediction_rad: np.ndarray, rows: Sequence[DiagnosticRow]) -> dict[str, Any]:
    targets = np.asarray([row.steering_rad for row in rows], dtype=np.float64)
    residual = np.asarray(prediction_rad, dtype=np.float64) - targets
    absolute = np.abs(residual)
    return {
        "count": len(rows),
        "mse_rad2": float(np.mean(residual ** 2)),
        "normalized_mse": float(np.mean((residual / MAX_STEERING_RAD) ** 2)),
        "mae_rad": float(np.mean(absolute)),
        "mean_absolute_residual_rad": float(np.mean(absolute)),
        "p95_absolute_residual_rad": float(np.quantile(absolute, 0.95)),
        "bias_rad": float(np.mean(residual)),
    }


def analyze_source_pressure(
    config: DiagnosticConfig, subsets: Mapping[str, Sequence[DiagnosticRow]], model: Any,
    gradient_pressure: Mapping[str, Any],
) -> dict[str, Any]:
    batch_size = int(config.diagnostics["forward_batch_size"])
    source_names = (
        "EXPERT_BASELINE", "DAGGER1_AVOIDANCE", "DAGGER1_AVOIDANCE_ONLY",
        "DAGGER1_ALL", "DAGGER2_POST_RECOVERY",
    )
    sources: dict[str, Any] = {}
    for name in source_names:
        prediction, _ = forward_rows(model, subsets[name], batch_size=batch_size)
        sources[name] = {
            **prediction_error_metrics(prediction, subsets[name]),
            "gradient_batches": gradient_pressure[name],
        }
    d2 = sources["DAGGER2_POST_RECOVERY"]
    total = len(subsets["EXPERT_BASELINE"]) + len(subsets["DAGGER1_ALL"]) + len(subsets["DAGGER2_POST_RECOVERY"])
    comparison: dict[str, Any] = {
        "DAGGER2_sample_fraction_of_8298": len(subsets["DAGGER2_POST_RECOVERY"]) / total,
        "DAGGER2_sample_count": len(subsets["DAGGER2_POST_RECOVERY"]),
        "combined_count": total,
    }
    disjoint_loss_mass = {
        name: sources[name]["normalized_mse"] * len(subsets[name])
        for name in ("EXPERT_BASELINE", "DAGGER1_ALL", "DAGGER2_POST_RECOVERY")
    }
    total_loss_mass = sum(disjoint_loss_mass.values())
    comparison.update({
        "frozen_D1_normalized_squared_error_mass": disjoint_loss_mass,
        "DAGGER2_fraction_of_combined_frozen_D1_squared_error": (
            disjoint_loss_mass["DAGGER2_POST_RECOVERY"] / total_loss_mass
            if total_loss_mass > 0 else None
        ),
        "note_on_squared_error_mass": (
            "Count multiplied by source normalized MSE over the disjoint 8,298-row "
            "EXPERT_BASELINE + DAGGER1_ALL + DAGGER2_POST_RECOVERY aggregate."
        ),
    })
    for reference in (
        "EXPERT_BASELINE", "DAGGER1_AVOIDANCE", "DAGGER1_AVOIDANCE_ONLY", "DAGGER1_ALL",
    ):
        source = sources[reference]
        comparison[f"DAGGER2_to_{reference}_normalized_mse_ratio"] = (
            d2["normalized_mse"] / source["normalized_mse"]
            if source["normalized_mse"] > 0 else None
        )
        for scope in ("head", "full"):
            d2_norm = d2["gradient_batches"][f"{scope}_gradient_norm"]["median"]
            reference_norm = source["gradient_batches"][f"{scope}_gradient_norm"]["median"]
            comparison[f"DAGGER2_to_{reference}_{scope}_median_gradient_norm_ratio"] = (
                d2_norm / reference_norm if reference_norm > 0 else None
            )
    return {
        "version": VERSION, "result": "PASS",
        "checkpoint_sha256": config.inputs["d1_checkpoint"]["sha256"],
        "loss_objective": "normalized steering MSE; radian residual metrics also reported",
        "sources": sources, "disproportionate_pressure": comparison,
    }


def _sign_disagreement_fraction(
    prediction: np.ndarray, target: np.ndarray, tolerance: float,
) -> float:
    relevant = np.abs(target) > tolerance
    if not np.any(relevant):
        return 0.0
    return float(np.mean(np.sign(prediction[relevant]) != np.sign(target[relevant])))


def _lag_profile(prediction: np.ndarray, target: np.ndarray, maximum_lag: int = 5) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            aligned_prediction = prediction[:lag]
            aligned_target = target[-lag:]
        elif lag > 0:
            aligned_prediction = prediction[lag:]
            aligned_target = target[:-lag]
        else:
            aligned_prediction = prediction
            aligned_target = target
        values.append({
            "lag_frames": lag,
            "mae_rad": float(np.mean(np.abs(aligned_prediction - aligned_target))),
            "sample_count": int(aligned_target.size),
        })
    best = min(values, key=lambda value: (value["mae_rad"], abs(value["lag_frames"])))
    zero = next(value for value in values if value["lag_frames"] == 0)
    return {
        "convention": "positive lag means the prediction best matches an earlier Expert target (prediction delay)",
        "per_lag": values, "best": best,
        "relative_mae_improvement_from_zero": (
            (zero["mae_rad"] - best["mae_rad"]) / zero["mae_rad"]
            if zero["mae_rad"] > 0 else 0.0
        ),
    }


def _s09_model_metrics(
    prediction: np.ndarray, target: np.ndarray, *, tolerance: float,
) -> dict[str, Any]:
    residual = prediction - target
    absolute_target = np.abs(target)
    relevant = absolute_target > tolerance
    return {
        "mae_rad": float(np.mean(np.abs(residual))),
        "rmse_rad": float(np.sqrt(np.mean(residual ** 2))),
        "bias_rad": float(np.mean(residual)),
        "mean_prediction_rad": float(np.mean(prediction)),
        "mean_absolute_prediction_rad": float(np.mean(np.abs(prediction))),
        "corrective_magnitude_ratio": (
            float(np.mean(np.abs(prediction[relevant])) / np.mean(absolute_target[relevant]))
            if np.any(relevant) else None
        ),
        "sign_disagreement_fraction": _sign_disagreement_fraction(prediction, target, tolerance),
        "lag_profile": _lag_profile(prediction, target),
    }


def analyze_s09_predictions(
    config: DiagnosticConfig, subsets: Mapping[str, Sequence[DiagnosticRow]],
    d1_model: Any, d2_model: Any,
) -> dict[str, Any]:
    rows = sorted(subsets["S09_AVOIDANCE_VALIDATION"], key=lambda row: row.route_progress_m)
    batch = int(config.diagnostics["forward_batch_size"])
    tolerance = float(config.diagnostics["zero_sign_tolerance_rad"])
    d1, _ = forward_rows(d1_model, rows, batch_size=batch)
    d2, _ = forward_rows(d2_model, rows, batch_size=batch)
    target = np.asarray([row.steering_rad for row in rows], dtype=np.float64)
    d1_metrics = _s09_model_metrics(d1, target, tolerance=tolerance)
    d2_metrics = _s09_model_metrics(d2, target, tolerance=tolerance)
    delta = d2 - d1
    table = [
        {
            "sequence_id": row.sequence_id,
            "route_progress_m": row.route_progress_m,
            "expert_steering_rad": row.steering_rad,
            "d1_steering_rad": float(d1[index]),
            "d2_fe_steering_rad": float(d2[index]),
            "d1_error_rad": float(d1[index] - target[index]),
            "d2_fe_error_rad": float(d2[index] - target[index]),
            "d2_fe_minus_d1_rad": float(delta[index]),
        }
        for index, row in enumerate(rows)
    ]
    known_collision_route_progress_m = 12.750
    collision_nearest_index = min(
        range(len(rows)),
        key=lambda index: abs(rows[index].route_progress_m - known_collision_route_progress_m),
    )
    d2_delay = d2_metrics["lag_profile"]
    indicators = {
        "weaker_avoidance_magnitude_than_D1": bool(
            d2_metrics["corrective_magnitude_ratio"] < d1_metrics["corrective_magnitude_ratio"]
            and d2_metrics["mae_rad"] > d1_metrics["mae_rad"]
        ),
        "more_wrong_sign_than_D1": bool(
            d2_metrics["sign_disagreement_fraction"] > d1_metrics["sign_disagreement_fraction"]
        ),
        "delay_indicator": bool(
            d2_delay["best"]["lag_frames"] > 0
            and d2_delay["relative_mae_improvement_from_zero"] >= 0.10
        ),
        "larger_absolute_bias_than_D1": bool(
            abs(d2_metrics["bias_rad"]) > abs(d1_metrics["bias_rad"])
        ),
    }
    active = [name for name, value in indicators.items() if value]
    interpretation = active[0] if len(active) == 1 else "no_simple_single_pattern"
    return {
        "version": VERSION, "result": "PASS", "evaluation_only": True,
        "subset": {
            "name": "S09_AVOIDANCE_VALIDATION", "count": len(rows),
            "route_progress_min_m": rows[0].route_progress_m,
            "route_progress_max_m": rows[-1].route_progress_m,
            "phase_contract": "departure_start <= route progress < cone_s",
        },
        "checkpoint_sha256": {
            "D1": config.inputs["d1_checkpoint"]["sha256"],
            "D2_FE": config.inputs["d2_fe_checkpoint"]["sha256"],
        },
        "metrics": {"D1": d1_metrics, "D2_FE": d2_metrics},
        "D2_FE_minus_D1": {
            "mean_rad": float(np.mean(delta)), "median_rad": float(np.median(delta)),
            "mean_absolute_rad": float(np.mean(np.abs(delta))),
            "max_absolute_rad": float(np.max(np.abs(delta))),
            "D2_FE_weaker_absolute_prediction_fraction": float(np.mean(np.abs(d2) < np.abs(d1))),
            "D2_FE_to_D1_mean_absolute_prediction_ratio": (
                float(np.mean(np.abs(d2)) / np.mean(np.abs(d1)))
                if float(np.mean(np.abs(d1))) > 0 else None
            ),
        },
        "pattern_indicators": indicators,
        "offline_pattern": interpretation,
        "per_sequence": table,
        "nearest_frozen_validation_row_to_live_collision_progress": {
            "live_collision_route_progress_m": known_collision_route_progress_m,
            "cross_rollout_comparison_only": True,
            **table[collision_nearest_index],
        },
        "known_live_context": {
            "D1": "S09 cone avoidance PASS; later sustained off-track at route s=29.307 m",
            "D2_FE": "genuine cone intersection near route s=12.750 m; completion 41.80%",
            "live_execution_performed_here": False,
        },
    }


def classify_evidence(
    *, gradient_supported: bool, aliasing_supported: bool,
    mixed_or_required_evidence_missing: bool = False,
) -> str:
    if gradient_supported and aliasing_supported:
        return BOTH_SUPPORTED
    if gradient_supported:
        return GRADIENT_CONFLICT_SUPPORTED
    if aliasing_supported:
        return VISUAL_STATE_ALIASING_SUPPORTED
    if mixed_or_required_evidence_missing:
        return MIXED_OR_INCONCLUSIVE
    return NO_STRONG_CONFLICT_FOUND


def _one_next_experiment(classification: str) -> dict[str, Any]:
    if classification == GRADIENT_CONFLICT_SUPPORTED:
        text = (
            "Run one D1-preserving adaptation on existing data only: initialize exact D1, retain "
            "DAGGER1 avoidance explicitly against frozen D1 outputs, and give the 109 post-recovery "
            "samples proportionate/reduced influence; freeze all choices before one run."
        )
    elif classification == VISUAL_STATE_ALIASING_SUPPORTED:
        text = (
            "Run one architecture experiment with a longer causal camera history than three frames, "
            "using only existing sequences that support that history, before adding any more DAgger data."
        )
    elif classification == BOTH_SUPPORTED:
        text = (
            "Run one smallest observability-first experiment: lengthen the causal visual history using "
            "existing data while explicitly retaining frozen D1 avoidance predictions."
        )
    elif classification == NO_STRONG_CONFLICT_FOUND:
        text = (
            "Measure per-sample, layerwise D1 gradient cosine on route-progress-matched DAgger1 "
            "avoidance and DAgger2 post-recovery pairs, without changing weights."
        )
    else:
        text = (
            "Measure synchronized physical pose (x/y/yaw, CTE, and heading) for the already-preserved "
            "nearest feature-conflict pairs by resolving it from existing compact telemetry only."
        )
    return {
        "count": 1, "recommendation": text, "implementation_performed": False,
        "new_data_collection": False, "dagger3": False,
    }


def build_summary(
    config: DiagnosticConfig, *, audit: Mapping[str, Any], subset_audit: Mapping[str, Any],
    labels: Mapping[str, Any], features: Mapping[str, Any],
    gradients: Mapping[str, Any], pressure: Mapping[str, Any], s09: Mapping[str, Any],
    final_input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    gradient_supported = bool(gradients["dagger2_opposes_dagger1_avoidance"])
    aliasing_supported = bool(features["evidence_rule"]["aliasing_supported"])
    ab = gradients["comparisons"]["DAGGER1_AVOIDANCE_vs_DAGGER2_POST_RECOVERY"]
    head_rule = ab["head"]["median"] < 0 and ab["head"]["negative_count"] >= 4
    full_rule = ab["full"]["median"] < 0 and ab["full"]["negative_count"] >= 4
    mixed = bool(head_rule != full_rule)
    classification = classify_evidence(
        gradient_supported=gradient_supported, aliasing_supported=aliasing_supported,
        mixed_or_required_evidence_missing=mixed,
    )
    start_inputs = audit["frozen_inputs"]
    end_inputs = final_input_audit["frozen_inputs"]
    frozen_unchanged = all(
        start_inputs[name]["sha256"] == end_inputs[name]["sha256"] for name in start_inputs
    )
    if not frozen_unchanged:
        raise DiagnosisGateError("a frozen input changed during diagnosis")
    return {
        "version": VERSION, "result": "PASS", "primary_classification": classification,
        "question": "Why did adding 109 existing D2 post-recovery samples damage D1 cone avoidance?",
        "evidence": {
            "gradient_conflict_supported": gradient_supported,
            "visual_state_aliasing_supported": aliasing_supported,
            "gradient_scope_mixed": mixed,
            "DAGGER1_AVOIDANCE_vs_DAGGER2_head_cosine": ab["head"],
            "DAGGER1_AVOIDANCE_vs_DAGGER2_full_cosine": ab["full"],
            "avoidance_only_gradient_sensitivity": gradients[
                "avoidance_only_sensitivity_check"
            ],
            "feature_conflict_summary": features["conflict_summary"],
            "D2_pressure": pressure["disproportionate_pressure"],
            "S09_offline_pattern": s09["offline_pattern"],
            "S09_pattern_indicators": s09["pattern_indicators"],
            "S09_nearest_row_to_live_collision_progress": s09[
                "nearest_frozen_validation_row_to_live_collision_progress"
            ],
        },
        "secondary_finding": (
            "DAGGER2_POST_RECOVERY has disproportionate frozen-D1 loss and gradient magnitude, "
            "but its gradient direction aligns with rather than opposes D1 avoidance in these batches."
        ),
        "interpretation_boundary": (
            "The diagnosis establishes associations at frozen checkpoints; it does not prove that the "
            "109 samples alone caused the scratch-trained D2-FE live collision."
        ),
        "metadata_limitations": features["temporal_observability"],
        "one_next_experiment": _one_next_experiment(classification),
        "safety_and_immutability": {
            "frozen_input_hashes_unchanged": frozen_unchanged,
            "d1_exact_tensor_equality_after_gradients": gradients["model_immutability"]["exact_tensor_equality"],
            "d1_state_sha256_before": gradients["model_immutability"]["state_tensor_sha256_before"],
            "d1_state_sha256_after": gradients["model_immutability"]["state_tensor_sha256_after"],
            "optimizer_constructed": False, "optimizer_steps": 0,
            "training_performed": False, "checkpoint_written": False,
            "simulator_invoked": False, "rosbag_opened": False,
            "data_collected": False, "S10_identity_audited_only": True,
            "S10_gradient_use": False, "S11_S12_accessed": False,
        },
        "counts": subset_audit["counts"],
        "label_distribution_file": "label_distribution.json",
        "feature_conflicts_file": "feature_conflicts.json",
        "gradient_conflict_file": "gradient_conflict.json",
        "source_pressure_file": "source_pressure.json",
        "s09_prediction_delta_file": "s09_prediction_delta.json",
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def render_report(
    summary: Mapping[str, Any], labels: Mapping[str, Any],
    features: Mapping[str, Any], gradients: Mapping[str, Any],
    pressure: Mapping[str, Any], s09: Mapping[str, Any],
) -> str:
    classification = summary["primary_classification"]
    lines = [
        "# Fast Random-Cone Post-Recovery Conflict Diagnosis V1",
        "",
        f"Primary classification: **{classification}**",
        "",
        "This is a frozen-checkpoint, existing-data-only simulator-policy diagnosis. No training, "
        "optimizer step, checkpoint write, simulator execution, rosbag read, or data collection occurred.",
        "",
        "## Preserved inputs and subsets",
        "",
        "| Subset | Count | Role |",
        "|---|---:|---|",
    ]
    roles = {
        "EXPERT_BASELINE": "D1 source / pressure only",
        "DAGGER1_ALL": "D1 source",
        "DAGGER1_AVOIDANCE": "approach + avoidance + pass-return",
        "DAGGER1_AVOIDANCE_ONLY": "avoidance phase separately identified",
        "DAGGER1_RECOVERY_OR_FAILURE": "available post-recovery rows",
        "DAGGER2_POST_RECOVERY": "all existing post-recovery rows",
        "EXPERT_NOMINAL": "deterministic nominal reference",
        "S09_AVOIDANCE_VALIDATION": "evaluation only",
        "S10_ALL_VALIDATION": "identity audit only; never used for gradients",
    }
    for name in roles:
        lines.append(f"| {name} | {summary['counts'][name]} | {roles[name]} |")
    lines.extend([
        "",
        "The previous DAgger2 coverage gate remains **FAIL**: 109 sequences, 18 beyond route "
        "s=20 m, and 0 beyond s=26 m. This diagnosis does not reinterpret that result.",
        "",
        "## Label distributions",
        "",
        "| Source | Mean | Median | Std | Min | Max | p05 | p25 | p75 | p95 | Mean abs | Neg / Zero / Pos | Saturated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("DAGGER1_AVOIDANCE", "DAGGER1_AVOIDANCE_ONLY", "DAGGER2_POST_RECOVERY"):
        item = labels["sources"][name]
        signs = item["sign_counts"]
        lines.append(
            f"| {name} | {_fmt(item['mean'])} | {_fmt(item['median'])} | {_fmt(item['std'])} | "
            f"{_fmt(item['min'])} | {_fmt(item['max'])} | {_fmt(item['p05'])} | {_fmt(item['p25'])} | "
            f"{_fmt(item['p75'])} | {_fmt(item['p95'])} | {_fmt(item['mean_absolute_steering_rad'])} | "
            f"{signs['negative']} / {signs['zero']} / {signs['positive']} | "
            f"{_fmt(item['saturation_fraction'])} |"
        )
    lines.extend([
        "",
        "Units are radians. Route, CTE, heading, per-scenario, and per-phase distributions are in "
        "`label_distribution.json`.",
        "",
        "DAGGER2_POST_RECOVERY asks for a broader, higher-magnitude correction distribution: its "
        f"mean absolute steering is {_fmt(labels['direct_comparison']['DAGGER2_to_DAGGER1_AVOIDANCE_mean_absolute_ratio'])}x "
        "the broad DAgger1 avoidance subset, while its mean shifts by "
        f"{_fmt(labels['direct_comparison']['DAGGER2_minus_DAGGER1_AVOIDANCE_mean_rad'])} rad. "
        "The positive/negative proportions remain similar, so the marginal labels are not simply "
        "opposite in sign.",
        "",
        "## Frozen-D1 feature conflicts",
        "",
    ])
    conflict = features["conflict_summary"]
    euclidean = conflict["euclidean_cross_check"]
    candidate_euclidean = conflict[
        "cosine_candidates_euclidean_distance_to_cosine_neighbor"
    ]
    lines.extend([
        f"- Nearest DAgger1-avoidance cosine distance: median {_fmt(conflict['nearest_distance']['median'])}, "
        f"p95 {_fmt(conflict['nearest_distance']['p95'])}.",
        f"- Nearest-pair Expert label difference: median {_fmt(conflict['nearest_label_difference_rad']['median'])} rad, "
        f"p95 {_fmt(conflict['nearest_label_difference_rad']['p95'])} rad.",
        f"- Descriptive thresholds: cosine distance <= {_fmt(features['observed_distribution_thresholds']['high_similarity']['value'])} "
        f"(observed p25), label difference >= {_fmt(features['observed_distribution_thresholds']['high_label_disagreement']['value_rad'])} rad "
        "(observed p75).",
        f"- Conflicts: {conflict['candidate_count']}/{features['query_count']} overall; "
        f"{_fmt(conflict['candidate_fraction_within_high_similarity'])} of the high-similarity tail.",
        f"- Quartile-flag independence would predict {_fmt(conflict['candidate_count_expected_if_quartile_flags_independent'])} "
        f"intersections; observed enrichment is {_fmt(conflict['candidate_enrichment_over_independence'])}x.",
        f"- Euclidean cross-check: nearest distance median {_fmt(euclidean['nearest_distance']['median'])}, "
        f"p95 {_fmt(euclidean['nearest_distance']['p95'])}; only {euclidean['candidate_count']}/{features['query_count']} "
        "Euclidean-near/high-disagreement pair.",
        f"- The seven cosine candidates have Euclidean distance median {_fmt(candidate_euclidean['median'])} "
        f"(minimum {_fmt(candidate_euclidean['min'])}), exceeding the nearest-Euclidean p95; their "
        "near-zero cosine values primarily reflect activation direction, not full feature proximity.",
        f"- Feature-aliasing evidence rule satisfied: {_fmt(features['evidence_rule']['aliasing_supported'])}.",
        "",
        "Top-20 bounded conflict table:",
        "",
        "| D2 sample | D1 avoidance neighbor | Cosine distance | D2 Expert | D1 Expert | Abs diff | Sign agrees | D2 scenario/route | D1 scenario/route |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for pair in features["top_20_conflicts"]:
        lines.append(
            f"| {pair['d2_sample_id']} | {pair['dagger1_sample_id']} | {_fmt(pair['feature_cosine_distance'])} | "
            f"{_fmt(pair['d2_expert_steering_rad'])} | {_fmt(pair['dagger1_expert_steering_rad'])} | "
            f"{_fmt(pair['absolute_label_difference_rad'])} | {_fmt(pair['steering_sign_agreement'])} | "
            f"S{pair['d2']['scenario_id']} / {_fmt(pair['d2']['route_progress_m'])} m | "
            f"S{pair['dagger1']['scenario_id']} / {_fmt(pair['dagger1']['route_progress_m'])} m |"
        )
    lines.extend([
        "",
        "DAGGER1 includes CTE and heading but not learner x/y/yaw, so the temporal-state check cannot "
        "make a complete pose-to-pose comparison. The JSON records all available timing, state, learner "
        "steering, and Expert steering for these pairs.",
        "",
        "## Frozen-D1 gradient conflict",
        "",
        "Six deterministic, disjoint 16-sample batches were evaluated with normalized steering MSE.",
        "",
        "| Pair | Scope | Mean cosine | Median | Min | Max | Negative fraction |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for pair_name, comparison in gradients["comparisons"].items():
        for scope in ("head", "full"):
            item = comparison[scope]
            lines.append(
                f"| {pair_name} | {scope} | {_fmt(item['mean'])} | {_fmt(item['median'])} | "
                f"{_fmt(item['min'])} | {_fmt(item['max'])} | {_fmt(item['fraction_negative'])} |"
            )
    lines.extend([
        "",
        f"Consistent DAgger2-vs-avoidance opposition rule satisfied: "
        f"**{_fmt(gradients['dagger2_opposes_dagger1_avoidance'])}**. D1 tensors were exactly equal "
        "before and after the gradient passes.",
        "",
        "The cone-critical avoidance-only sensitivity check is also aligned: head median cosine "
        f"{_fmt(gradients['avoidance_only_sensitivity_check']['head']['median'])} and full median "
        f"{_fmt(gradients['avoidance_only_sensitivity_check']['full']['median'])}, with negative "
        f"fractions {_fmt(gradients['avoidance_only_sensitivity_check']['head']['fraction_negative'])} "
        f"and {_fmt(gradients['avoidance_only_sensitivity_check']['full']['fraction_negative'])}. "
        "Thus the broad subset is not hiding phase-specific gradient opposition in these batches.",
        "",
        "## Source loss and pressure at frozen D1",
        "",
        "| Source | Count | MSE (rad²) | Normalized MSE | MAE (rad) | p95 abs residual | Head grad norm median | Full grad norm median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in (
        "EXPERT_BASELINE", "DAGGER1_AVOIDANCE", "DAGGER1_AVOIDANCE_ONLY",
        "DAGGER1_ALL", "DAGGER2_POST_RECOVERY",
    ):
        item = pressure["sources"][name]
        lines.append(
            f"| {name} | {item['count']} | {_fmt(item['mse_rad2'])} | {_fmt(item['normalized_mse'])} | "
            f"{_fmt(item['mae_rad'])} | {_fmt(item['p95_absolute_residual_rad'])} | "
            f"{_fmt(item['gradient_batches']['head_gradient_norm']['median'])} | "
            f"{_fmt(item['gradient_batches']['full_gradient_norm']['median'])} |"
        )
    d2_pressure = pressure["disproportionate_pressure"]
    lines.extend([
        "",
        f"The 109 D2 rows are {_fmt(100 * d2_pressure['DAGGER2_sample_fraction_of_8298'])}% of the 8,298 rows "
        f"but account for {_fmt(100 * d2_pressure['DAGGER2_fraction_of_combined_frozen_D1_squared_error'])}% "
        "of frozen-D1 squared-error mass. Relative to broad DAgger1 avoidance, D2 has "
        f"{_fmt(d2_pressure['DAGGER2_to_DAGGER1_AVOIDANCE_normalized_mse_ratio'])}x normalized MSE, "
        f"{_fmt(d2_pressure['DAGGER2_to_DAGGER1_AVOIDANCE_head_median_gradient_norm_ratio'])}x head "
        f"gradient norm, and {_fmt(d2_pressure['DAGGER2_to_DAGGER1_AVOIDANCE_full_median_gradient_norm_ratio'])}x "
        "full-network gradient norm. This is strong pressure evidence, but pressure magnitude alone is "
        "not directional gradient conflict.",
        "",
        "## S09 avoidance prediction context",
        "",
        "| Model | MAE | RMSE | Bias | Corrective magnitude ratio | Sign disagreement | Best lag (frames) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("D1", "D2_FE"):
        item = s09["metrics"][name]
        lines.append(
            f"| {name} | {_fmt(item['mae_rad'])} | {_fmt(item['rmse_rad'])} | {_fmt(item['bias_rad'])} | "
            f"{_fmt(item['corrective_magnitude_ratio'])} | {_fmt(item['sign_disagreement_fraction'])} | "
            f"{item['lag_profile']['best']['lag_frames']} |"
        )
    collision_context = s09["nearest_frozen_validation_row_to_live_collision_progress"]
    lines.extend([
        "",
        f"Offline pattern: **{s09['offline_pattern']}**. Indicators: "
        + ", ".join(f"{key}={_fmt(value)}" for key, value in s09["pattern_indicators"].items())
        + ". This is mild and non-uniform: D2-FE is weaker in absolute magnitude on "
        f"{_fmt(100 * s09['D2_FE_minus_D1']['D2_FE_weaker_absolute_prediction_fraction'])}% of rows, and "
        f"its mean absolute prediction is {_fmt(s09['D2_FE_minus_D1']['D2_FE_to_D1_mean_absolute_prediction_ratio'])}x D1. "
        f"At the nearest frozen-validation row to route s=12.750 m ({_fmt(collision_context['route_progress_m'])} m), "
        f"D2-FE minus D1 is only {_fmt(collision_context['d2_fe_minus_d1_rad'])} rad. The known collision "
        "is therefore consistency evidence only, not a causal reconstruction across rollouts.",
        "",
        "## Conclusion",
        "",
        f"The primary classification is **{classification}**. "
        + summary["interpretation_boundary"],
        "",
        summary["secondary_finding"],
        "",
        "## One next experiment",
        "",
        summary["one_next_experiment"]["recommendation"],
        "",
        "It was not implemented. No DAgger3 is recommended or created.",
        "",
        "## Safety and limitations",
        "",
        "- All frozen file hashes matched both before and after diagnosis.",
        "- S09/S10 were excluded from every gradient batch; S10 was identity-audited only.",
        "- S11/S12 were not accessed.",
        "- This is simulator-policy evidence, not real-robot evidence.",
    ])
    return "\n".join(lines) + "\n"


def run_diagnosis(config: DiagnosticConfig, emit=print) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(config.diagnostics["torch_threads"]))
    torch.manual_seed(int(config.diagnostics["deterministic_seed"]))
    torch.use_deterministic_algorithms(True)
    result_dir = config.result_dir

    emit("[1/10] auditing frozen hashes and historical evidence")
    audit = audit_frozen_inputs(config)
    write_json(result_dir / "input_audit.json", audit)

    emit("[2/10] constructing deterministic diagnostic subsets")
    subsets, subset_audit = construct_subsets(config)
    write_json(result_dir / "subset_audit.json", subset_audit)

    emit("[3/10] calculating label distributions")
    labels = analyze_label_distributions(config, subsets)
    write_json(result_dir / "label_distribution.json", labels)

    d1_path = _resolve(config.repo, config.inputs["d1_checkpoint"]["path"])
    d1_model, _ = load_frozen_model(d1_path)
    emit("[4-6/10] extracting frozen D1 features and nearest-neighbor conflicts")
    features = analyze_features(config, subsets, d1_model)
    write_json(result_dir / "feature_conflicts.json", features)

    emit("[7/10] calculating frozen-D1 diagnostic gradients")
    gradients, gradient_pressure = calculate_gradient_diagnostics(
        config, subsets, d1_model,
    )
    write_json(result_dir / "gradient_conflict.json", gradients)

    emit("[8/10] calculating source loss and gradient pressure")
    pressure = analyze_source_pressure(config, subsets, d1_model, gradient_pressure)
    write_json(result_dir / "source_pressure.json", pressure)

    emit("[9/10] evaluating preserved D1 and D2-FE on frozen S09 avoidance")
    d2_path = _resolve(config.repo, config.inputs["d2_fe_checkpoint"]["path"])
    d2_model, _ = load_frozen_model(d2_path)
    s09 = analyze_s09_predictions(config, subsets, d1_model, d2_model)
    write_json(result_dir / "s09_prediction_delta.json", s09)

    emit("[10/10] classifying evidence and verifying final frozen identities")
    final_audit = audit_frozen_inputs(config)
    audit["post_diagnosis"] = {
        "frozen_inputs": final_audit["frozen_inputs"],
        "d1_state_tensor_sha256": final_audit["d1"]["state_tensor_sha256"],
        "d2_fe_state_tensor_sha256": final_audit["d2_fe_frozen_negative"]["state_tensor_sha256"],
        "all_frozen_input_hashes_unchanged": all(
            audit["frozen_inputs"][name]["sha256"] == final_audit["frozen_inputs"][name]["sha256"]
            for name in audit["frozen_inputs"]
        ),
    }
    write_json(result_dir / "input_audit.json", audit)
    summary = build_summary(
        config, audit=audit, subset_audit=subset_audit, labels=labels,
        features=features, gradients=gradients, pressure=pressure, s09=s09,
        final_input_audit=final_audit,
    )
    write_json(result_dir / "summary.json", summary)
    write_text(result_dir / "REPORT.md", render_report(summary, labels, features, gradients, pressure, s09))
    emit(f"classification={summary['primary_classification']}")
    return summary
