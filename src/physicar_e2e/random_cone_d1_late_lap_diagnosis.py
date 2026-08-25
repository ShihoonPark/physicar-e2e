"""Diagnostic-only isolation of the Random-Cone D1 late-lap failure.

The module has three deliberately separate boundaries:

* frozen manifests and model artifacts are read-only inputs;
* offline audits never invoke optimization or data collection;
* live evaluation retains only bounded numeric telemetry and lets the selected
  neural policy, never the shadow Expert, cross the command boundary.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .cone_avoidance_environment import (
    EnvironmentConfig,
    share_path,
    verify_canonical_hashes,
)
from .cone_avoidance_expert import activate_world
from .dataset_extractor import canonical_json_bytes, sha256_file
from .expert_driver import DriverConfig, PoseLivenessMonitor, wait_after_reset
from .high_speed_temporal import TemporalOnnxModel, warm_temporal_buffer
from .pilotnet import clamp_steering_rad, steering_normalized_to_rad
from .pilotnet_temporal import (
    TEMPORAL_PARAMETER_COUNT,
    TemporalInputError,
    append_live_jpeg,
    build_temporal_pilotnet,
    preprocess_temporal_paths,
)
from .pilotnet_v4_repeatability import clock_health_preflight
from .route_geometry import OffTrackMonitor, ProgressTracker, pure_pursuit_steering
from .sim_client import SimClient


VERSION = "random_cone_d1_late_lap_diagnosis_v1"
EXPECTED_BRANCH = "experiment/random-cone-d1-late-lap-diagnosis-v1"
CANONICAL_WORLD = "custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1"
ROUTE_LENGTH_M = 30.50461070080936
ROUTE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 10.0),
    (10.0, 20.0),
    (20.0, 26.0),
    (26.0, ROUTE_LENGTH_M),
)
TRAIN_SCENARIOS = tuple(f"{value:02d}" for value in range(1, 9))
VALIDATION_SCENARIOS = ("09", "10")
PROTECTED_SCENARIOS = ("11", "12")
MINIMUM_FREE_BYTES = 11 * 1024**3 // 2
MAX_STEERING_RAD = 0.349066
CONTROL_HZ = 15.0
SPEED_MPS = 1.0
LOOKAHEAD_M = 0.9

POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED = "POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED"
DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED = "DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED"
SHARED_LANE_WEAKNESS_SUPPORTED = "SHARED_LANE_WEAKNESS_SUPPORTED"
MIXED_OR_INCONCLUSIVE = "MIXED_OR_INCONCLUSIVE"


class DiagnosisGateError(RuntimeError):
    """A frozen identity, safety, or execution-order gate failed."""


@dataclass(frozen=True)
class DiagnosisConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def sources(self) -> dict[str, Any]:
        return self.payload["sources"]

    @property
    def models(self) -> dict[str, Any]:
        return self.payload["models"]

    @property
    def control(self) -> dict[str, Any]:
        return self.payload["control"]

    def result_dir(self, repo: Path) -> Path:
        return repo / self.payload["result_directory"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosisGateError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def load_config(path: Path, repo: Path) -> DiagnosisConfig:
    payload = _read_json(path)
    required = {
        "version", "expected_branch", "map_family", "result_directory",
        "canonical_cone_free_world", "canonical_route_length_m",
        "canonical_route_points", "route_bins_m", "disk_gate", "control",
        "canonical_assets", "sources", "models", "preserved_evidence",
        "live_run_policy", "protected_scenarios", "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise DiagnosisGateError("diagnosis config fields/version changed")
    configured_bins = tuple(tuple(float(value) for value in item) for item in payload["route_bins_m"])
    if configured_bins != ROUTE_BINS:
        raise DiagnosisGateError(f"route bins must remain exactly {ROUTE_BINS}")
    if (
        payload["expected_branch"] != EXPECTED_BRANCH
        or payload["canonical_cone_free_world"] != CANONICAL_WORLD
        or float(payload["canonical_route_length_m"]) != ROUTE_LENGTH_M
        or int(payload["canonical_route_points"]) != 388
        or tuple(payload["protected_scenarios"]) != PROTECTED_SCENARIOS
    ):
        raise DiagnosisGateError("branch/world/route/holdout contract changed")
    control = payload["control"]
    fixed_control = (
        float(control["speed_mps"]), float(control["control_frequency_hz"]),
        float(control["lookahead_m"]), float(control["steering_limit_rad"]),
        float(control["wheelbase_m"]), int(control["history_frames"]),
        float(control["maximum_adjacent_gap_s"]),
    )
    if fixed_control != (1.0, 15.0, 0.9, 0.349066, 0.18, 3, 0.12):
        raise DiagnosisGateError(f"fixed control/temporal contract changed: {fixed_control}")
    if (
        int(payload["disk_gate"]["minimum_free_bytes"]) != MINIMUM_FREE_BYTES
        or payload["live_run_policy"] != {
            "d1_runs": 1,
            "r1_runs_if_d1_policy_fail": 1,
            "r1_runs_if_d1_pass": 0,
            "retry_genuine_policy_failure": False,
            "shadow_expert_control_authority": False,
            "record_images": False,
            "record_bags": False,
        }
        or any(value is not False for value in payload["permissions"].values())
    ):
        raise DiagnosisGateError("disk/live/permission boundary changed")
    if set(payload["models"]) != {"R1", "D1"}:
        raise DiagnosisGateError("model set must be exactly frozen R1 and D1")
    return DiagnosisConfig(path.resolve(), payload)


def disk_gate(
    path: str | Path = "/",
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    *,
    disk_usage: Callable[[str | Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    usage = disk_usage(path)
    free = int(usage.free)
    report = {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": free,
        "free_gib": free / 1024**3,
        "minimum_free_bytes": int(minimum_free_bytes),
        "minimum_free_gib": int(minimum_free_bytes) / 1024**3,
        "result": "PASS" if free >= minimum_free_bytes else "FAIL",
    }
    try:
        report["df_h"] = subprocess.run(
            ["df", "-h", str(path)], text=True, capture_output=True, check=True,
        ).stdout.strip()
        report["df_bytes"] = subprocess.run(
            ["df", "-B1", str(path)], text=True, capture_output=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        report["df_recording_error"] = str(exc)
    if report["result"] != "PASS":
        raise DiagnosisGateError(
            f"root disk has {report['free_gib']:.3f} GiB free; "
            f"at least {report['minimum_free_gib']:.3f} GiB is required"
        )
    return report


def _git_branch(repo: Path) -> str:
    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.strip()


def _hash_gate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise DiagnosisGateError(f"missing frozen {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise DiagnosisGateError(f"frozen {label} hash changed: {observed}")
    return observed


def _scenario(value: object) -> str:
    return str(value).zfill(2)


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except (OSError, csv.Error) as exc:
        raise DiagnosisGateError(f"cannot read frozen manifest {path}: {exc}") from exc


def _validate_preserved_s09(config: DiagnosisConfig, repo: Path) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for name in ("R1", "D1"):
        model = config.models[name]
        path = _resolve(repo, model["s09_live_path"])
        _hash_gate(path, model["s09_live_sha256"], f"{name} S09 live record")
        records[name] = _read_json(path)
    r1 = records["R1"]
    r1_run = r1.get("run") or {}
    d1 = records["D1"]
    d1_run = d1.get("run") or {}
    r1_expected = {
        "classification": "RANDOM_CONE_POLICY_FAIL",
        "route_completion_fraction": 0.3939597182924379,
        "max_cte_m": 0.8035169999999994,
        "minimum_footprint_to_cone_clearance_m": 1.107698830004098,
        "recovery_success": False,
    }
    d1_expected = {
        "classification": "RANDOM_CONE_POLICY_FAIL",
        "route_completion_fraction": 0.8838025637925251,
        "final_route_s_m": 29.307113445990467,
        "max_cte_m": 0.6906843277290918,
        "minimum_footprint_to_cone_clearance_m": 0.06465524295557254,
        "recovery_success": True,
        "recovery_time_s": 0.7956490869983099,
    }
    for key, expected in r1_expected.items():
        observed = r1.get(key) if key == "classification" else r1_run.get(key)
        if observed != expected:
            raise DiagnosisGateError(f"preserved R1 S09 evidence changed: {key}")
    for key, expected in d1_expected.items():
        observed = d1.get(key) if key == "classification" else d1_run.get(key)
        if observed != expected:
            raise DiagnosisGateError(f"preserved D1 S09 evidence changed: {key}")
    healthy = (
        not d1_run.get("temporal_input_failure")
        and d1_run.get("timing_slips_over_100ms") == 0
        and sum(int(d1_run.get(key, 0)) for key in (
            "api_failures", "pose_failures", "clock_failures", "liveness_failures",
        )) == 0
        and d1_run.get("steering_saturation_fraction") == 0.0
        and d1_run.get("safe_stop_success") is True
    )
    if not healthy:
        raise DiagnosisGateError("preserved D1 S09 health/saturation evidence changed")
    return {
        "result": "PASS",
        "R1": {
            "classification": r1["classification"],
            "failed_before_cone": True,
            "completion_fraction": r1_run["route_completion_fraction"],
            "completion_percent": 100.0 * r1_run["route_completion_fraction"],
            "max_cte_m": r1_run["max_cte_m"],
            "minimum_cone_clearance_m": r1_run["minimum_footprint_to_cone_clearance_m"],
            "sha256": config.models["R1"]["s09_live_sha256"],
        },
        "D1": {
            "classification": d1["classification"],
            "cone_avoidance": "PASS",
            "route_recovery": "PASS",
            "recovery_time_s": d1_run["recovery_time_s"],
            "completion_fraction": d1_run["route_completion_fraction"],
            "completion_percent": 100.0 * d1_run["route_completion_fraction"],
            "failure_route_s_m": d1_run["final_route_s_m"],
            "max_cte_m": d1_run["max_cte_m"],
            "minimum_cone_clearance_m": d1_run["minimum_footprint_to_cone_clearance_m"],
            "infrastructure_temporal_timing_or_saturation_failure": False,
            "sha256": config.models["D1"]["s09_live_sha256"],
        },
        "comparison": (
            "DAgger1 solved or materially improved the original cone-approach failure, "
            "but introduced or exposed a new late-lap failure frontier."
        ),
        "new_failure_cause_pre_isolation": "UNCLASSIFIED",
    }


def audit_preserved_inputs(config: DiagnosisConfig, repo: Path) -> dict[str, Any]:
    branch = _git_branch(repo)
    if branch != config.payload["expected_branch"]:
        raise DiagnosisGateError(
            f"expected branch {config.payload['expected_branch']!r}, observed {branch!r}"
        )
    hashes: dict[str, Any] = {"models": {}, "sources": {}, "canonical_assets": {}}
    for model_name, model in config.models.items():
        hashes["models"][model_name] = {
            key.removesuffix("_path"): _hash_gate(
                _resolve(repo, value), model[key.replace("_path", "_sha256")],
                f"{model_name} {key.removesuffix('_path')}",
            )
            for key, value in model.items()
            if key.endswith("_path") and key.replace("_path", "_sha256") in model
        }
        seal = _read_json(_resolve(repo, model["freeze_seal_path"]))
        if (
            seal.get("checkpoint_sha256") != model["checkpoint_sha256"]
            or seal.get("onnx_sha256") != model["onnx_sha256"]
            or seal.get("freeze_sha256") != model["freeze_sha256"]
            or seal.get("retraining_or_tuning_after_seal_permitted") is not False
        ):
            raise DiagnosisGateError(f"{model_name} freeze seal contract changed")
    for name, source in config.sources.items():
        path = _resolve(repo, source["path"])
        hashes["sources"][name] = _hash_gate(path, source["sha256"], name)
    assets = config.payload["canonical_assets"]
    environment_path = _resolve(repo, assets["environment_config_path"])
    hashes["canonical_assets"]["environment_config"] = _hash_gate(
        environment_path, assets["environment_config_sha256"], "canonical environment config",
    )
    train_rows = _load_csv(_resolve(repo, config.sources["expert_train_manifest"]["path"]))
    validation_rows = _load_csv(_resolve(repo, config.sources["expert_validation_manifest"]["path"]))
    aggregate_rows = _load_csv(_resolve(repo, config.sources["dagger1_aggregate_manifest"]["path"]))
    if (
        len(train_rows) != 6706
        or {_scenario(row["scenario_id"]) for row in train_rows} != set(TRAIN_SCENARIOS)
        or len(validation_rows) != 837
        or {_scenario(row["scenario_id"]) for row in validation_rows} != set(VALIDATION_SCENARIOS)
        or len(aggregate_rows) != 8189
        or {_scenario(row["scenario_id"]) for row in aggregate_rows} != set(TRAIN_SCENARIOS)
        or any(_scenario(row["scenario_id"]) in PROTECTED_SCENARIOS for row in [*train_rows, *validation_rows, *aggregate_rows])
    ):
        raise DiagnosisGateError("frozen 8/2/2 manifest role/count contract changed")
    if sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) != TEMPORAL_PARAMETER_COUNT:
        raise DiagnosisGateError("Temporal PilotNet architecture changed")
    preserved = _validate_preserved_s09(config, repo)
    return {
        "version": VERSION + "_preserved_input_audit",
        "generated_utc": utc_now(),
        "result": "PASS",
        "branch": branch,
        "hashes": hashes,
        "architecture": {
            "parameter_count": TEMPORAL_PARAMETER_COUNT,
            "input_shape": ["N", 9, 66, 200],
        },
        "dataset_counts": {"Expert": len(train_rows), "DAgger1": 1483, "aggregate": len(aggregate_rows)},
        "preserved_s09_comparison": preserved,
        "training_or_data_collection_performed": False,
        "protected_holdout": {
            "scenario_ids": list(PROTECTED_SCENARIOS),
            "rows_in_any_accessed_manifest": 0,
            "camera_data_inspected": False,
            "expert_labels_generated": 0,
        },
    }


def _bin_label(bounds: tuple[float, float]) -> str:
    lower, upper = bounds
    upper_text = "30.504611" if upper == ROUTE_LENGTH_M else f"{upper:g}"
    return f"{lower:g}-{upper_text} m"


def route_bin_index(route_s_m: float) -> int | None:
    value = float(route_s_m)
    if not math.isfinite(value) or value < ROUTE_BINS[0][0] or value > ROUTE_BINS[-1][1] + 1e-9:
        return None
    for index, (lower, upper) in enumerate(ROUTE_BINS):
        if lower <= value < upper or (index == len(ROUTE_BINS) - 1 and value <= upper + 1e-9):
            return index
    return None


def _bin_counts(rows: Sequence[Mapping[str, Any]], field: str = "route_progress_m") -> dict[str, int]:
    counts = {_bin_label(bounds): 0 for bounds in ROUTE_BINS}
    for row in rows:
        raw = row.get(field)
        if raw in (None, ""):
            raise DiagnosisGateError(f"route-progress metadata unavailable in field {field}")
        index = route_bin_index(float(raw))
        if index is None:
            raise DiagnosisGateError(f"route progress outside fixed route bins: {raw}")
        counts[_bin_label(ROUTE_BINS[index])] += 1
    return counts


def offline_distribution_audit(config: DiagnosisConfig, repo: Path) -> dict[str, Any]:
    expert_source = config.sources["expert_train_manifest"]
    aggregate_source = config.sources["dagger1_aggregate_manifest"]
    expert_path = _resolve(repo, expert_source["path"])
    aggregate_path = _resolve(repo, aggregate_source["path"])
    _hash_gate(expert_path, expert_source["sha256"], "Expert TRAIN manifest")
    _hash_gate(aggregate_path, aggregate_source["sha256"], "DAgger1 aggregate manifest")
    expert = _load_csv(expert_path)
    aggregate = _load_csv(aggregate_path)
    aggregate_expert = [row for row in aggregate if row.get("provenance") == "EXPERT_BASELINE"]
    dagger = [row for row in aggregate if row.get("provenance") == "DAGGER1"]
    if (len(expert), len(aggregate_expert), len(dagger), len(aggregate)) != (6706, 6706, 1483, 8189):
        raise DiagnosisGateError("Expert/DAgger aggregate source counts changed")
    expert_by_id = {row["sequence_id"]: row for row in expert}
    if set(expert_by_id) != {row["sequence_id"] for row in aggregate_expert}:
        raise DiagnosisGateError("aggregate Expert sequence identities differ from frozen TRAIN")
    for row in aggregate_expert:
        original = expert_by_id[row["sequence_id"]]
        if (
            float(row["route_progress_m"]) != float(original["route_progress_m"])
            or float(row["target_steering_rad"]) != float(original["target_steering_rad"])
        ):
            raise DiagnosisGateError("aggregate Expert route progress/target changed")

    expert_counts = _bin_counts(expert)
    dagger_counts = _bin_counts(dagger)
    aggregate_counts = _bin_counts(aggregate)
    bins: dict[str, Any] = {}
    for bounds in ROUTE_BINS:
        label = _bin_label(bounds)
        total = aggregate_counts[label]
        bins[label] = {
            "bounds_m": list(bounds),
            "expert_baseline_sequence_count": expert_counts[label],
            "dagger1_sequence_count": dagger_counts[label],
            "aggregate_sequence_count": total,
            "dagger1_fraction": dagger_counts[label] / total if total else None,
        }

    scenarios: dict[str, Any] = {}
    for scenario in TRAIN_SCENARIOS:
        scenario_expert = [row for row in expert if _scenario(row["scenario_id"]) == scenario]
        scenario_dagger = [row for row in dagger if _scenario(row["scenario_id"]) == scenario]
        scenarios[scenario] = {
            "expert_baseline_sequence_count": len(scenario_expert),
            "dagger1_sequence_count": len(scenario_dagger),
            "aggregate_sequence_count": len(scenario_expert) + len(scenario_dagger),
            "expert_baseline_route_bins": _bin_counts(scenario_expert),
            "dagger1_route_bins": _bin_counts(scenario_dagger),
        }

    phases: dict[str, Any] = {}
    for phase in ("approach", "avoidance", "pass_return", "post_recovery"):
        selected = [row for row in dagger if row.get("cone_phase") == phase]
        phases[phase] = {"sequence_count": len(selected), "route_bins": _bin_counts(selected)}
    blank_expert_phase_count = sum(not row.get("cone_phase") for row in aggregate_expert)
    if blank_expert_phase_count != 6706:
        raise DiagnosisGateError("Expert aggregate cone-phase provenance changed")

    dataset_source = config.sources["dagger1_dataset_summary"]
    collection_source = config.sources["dagger1_collection_summary"]
    dataset_path = _resolve(repo, dataset_source["path"])
    collection_path = _resolve(repo, collection_source["path"])
    _hash_gate(dataset_path, dataset_source["sha256"], "DAgger1 dataset summary")
    _hash_gate(collection_path, collection_source["sha256"], "DAgger1 collection summary")
    dataset = _read_json(dataset_path)
    collection = _read_json(collection_path)
    collection_episodes = {
        item["episode_id"]: item for item in collection.get("episodes", [])
        if isinstance(item, dict) and "episode_id" in item
    }
    completion_rows: list[dict[str, Any]] = []
    for item in dataset.get("episodes", []):
        episode_id = item["episode_id"]
        selected = [row for row in dagger if row["episode_id"] == episode_id]
        collected = collection_episodes.get(episode_id) or {}
        learner_run = collected.get("learner_run") or {}
        completion_rows.append({
            "episode_id": episode_id,
            "scenario_id": item["scenario_id"],
            "classification": item["learner_rollout_classification"],
            "completion_fraction": item["rollout_progress"],
            "completion_percent": 100.0 * float(item["rollout_progress"]),
            "final_route_s_m": learner_run.get("final_route_s_m"),
            "accepted_temporal_sequences": item["accepted_temporal_sequences"],
            "route_bins": _bin_counts(selected),
            "cone_phase_sequence_counts": item["cone_phase_sequence_counts"],
            "genuine_failure_preserved_without_retry": (
                item["learner_rollout_classification"] == "RANDOM_CONE_POLICY_FAIL"
                and collected.get("genuine_policy_failure_preserved_as_valid_evidence") is True
                and int(collected.get("attempt_number", 0)) == 1
            ),
        })
    if (
        len(completion_rows) != 8
        or any(not row["genuine_failure_preserved_without_retry"] for row in completion_rows)
        or any(row["scenario_id"] not in TRAIN_SCENARIOS for row in completion_rows)
    ):
        raise DiagnosisGateError("DAgger1 learner rollout completion/provenance changed")
    completion_values = [float(row["completion_fraction"]) for row in completion_rows]
    late_label = _bin_label(ROUTE_BINS[-1])
    zero_late = dagger_counts[late_label] == 0
    report = {
        "version": VERSION + "_offline_distribution",
        "generated_utc": utc_now(),
        "result": "PASS",
        "fixed_route_bins_m": [list(bounds) for bounds in ROUTE_BINS],
        "route_bin_inclusion": "left-closed/right-open; final bin includes the route endpoint",
        "bins": bins,
        "by_scenario": scenarios,
        "by_cone_phase": {
            "DAGGER1": phases,
            "EXPERT_BASELINE": {
                "result": "NOT_ENCODED_IN_FROZEN_MANIFEST",
                "blank_phase_sequence_count": blank_expert_phase_count,
                "phase_values_were_not_invented": True,
            },
        },
        "learner_rollout_completion": {
            "episodes": completion_rows,
            "minimum_completion_fraction": min(completion_values),
            "maximum_completion_fraction": max(completion_values),
            "minimum_completion_percent": 100.0 * min(completion_values),
            "maximum_completion_percent": 100.0 * max(completion_values),
            "all_genuine_failures_preserved_without_retry": True,
        },
        "source_provenance": {
            "expert_train_manifest": {"path": str(expert_path), "sha256": expert_source["sha256"], "sequence_count": len(expert)},
            "dagger1_aggregate_manifest": {"path": str(aggregate_path), "sha256": aggregate_source["sha256"], "sequence_count": len(aggregate)},
            "provenance_counts": {"EXPERT_BASELINE": len(aggregate_expert), "DAGGER1": len(dagger)},
            "scenario_ids": list(TRAIN_SCENARIOS),
        },
        "dagger1_late_lap_sequence_count": dagger_counts[late_label],
        "dagger1_contributes_zero_late_lap_samples": zero_late,
        "conclusion": (
            "DAgger1 contributes zero samples in 26-30.504611 m; all eight learner rollouts "
            "ended at approximately 29.49-40.83% completion."
        ) if zero_late else "DAgger1 contributes late-lap samples.",
        "dataset_modified": False,
    }
    if not zero_late:
        raise DiagnosisGateError("DAgger1 unexpectedly contributes late-lap samples")
    _write_json(config.result_dir(repo) / "offline_distribution.json", report)
    return report


def _validation_rows(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []
    for raw in _load_csv(path):
        route_raw = raw.get("route_progress_m")
        if route_raw in (None, ""):
            return [], "frozen validation manifest has no route_progress_m metadata"
        scenario = _scenario(raw.get("scenario_id"))
        if scenario not in VALIDATION_SCENARIOS:
            raise DiagnosisGateError(f"validation manifest contains forbidden S{scenario}")
        root = path.parents[1]
        paths = tuple(root / raw[key] for key in ("frame_t_minus_2", "frame_t_minus_1", "frame_t"))
        timestamps = tuple(int(raw[key]) for key in (
            "camera_timestamp_t_minus_2_ns", "camera_timestamp_t_minus_1_ns", "camera_timestamp_t_ns",
        ))
        if (
            not timestamps[0] < timestamps[1] < timestamps[2]
            or max(timestamps[1] - timestamps[0], timestamps[2] - timestamps[1]) > 120_000_000
            or not all(image.is_file() for image in paths)
        ):
            raise DiagnosisGateError("validation temporal causality/gap/image contract changed")
        route_s = float(route_raw)
        if route_bin_index(route_s) is None:
            raise DiagnosisGateError(f"validation route progress outside fixed bins: {route_s}")
        rows.append({
            "sequence_id": raw["sequence_id"],
            "scenario_id": scenario,
            "paths": paths,
            "target_steering_rad": float(raw["target_steering_rad"]),
            "route_progress_m": route_s,
        })
    return rows, None


def _load_checkpoint(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise DiagnosisGateError(f"checkpoint lacks model_state_dict: {path}")
    model = build_temporal_pilotnet().to("cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_checkpoint(model: Any, rows: Sequence[dict[str, Any]], batch_size: int = 64) -> np.ndarray:
    import torch

    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            tensor = torch.from_numpy(np.stack([
                preprocess_temporal_paths(row["paths"]) for row in batch
            ])).to("cpu")
            normalized = model(tensor).detach().cpu().numpy().reshape(-1)
            predictions.append(np.asarray(
                steering_normalized_to_rad(normalized, MAX_STEERING_RAD), dtype=np.float64,
            ))
    return np.concatenate(predictions) if predictions else np.asarray([], dtype=np.float64)


def steering_metrics(predictions: Sequence[float], targets: Sequence[float]) -> dict[str, Any]:
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if not target.size:
        return {
            "count": 0, "mae_rad": None, "rmse_rad": None,
            "bias_mean_signed_error_rad": None, "max_absolute_error_rad": None,
            "correlation": None, "corrective_magnitude_ratio": None,
        }
    errors = predicted - target
    correlation: float | None = None
    if target.size > 1 and float(np.std(predicted)) > 0.0 and float(np.std(target)) > 0.0:
        value = float(np.corrcoef(predicted, target)[0, 1])
        correlation = value if math.isfinite(value) else None
    target_magnitude = float(np.mean(np.abs(target)))
    return {
        "count": int(target.size),
        "mae_rad": float(np.mean(np.abs(errors))),
        "rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_mean_signed_error_rad": float(np.mean(errors)),
        "max_absolute_error_rad": float(np.max(np.abs(errors))),
        "correlation": correlation,
        "corrective_magnitude_ratio": (
            float(np.mean(np.abs(predicted))) / target_magnitude if target_magnitude > 0.0 else None
        ),
    }


def _model_route_metrics(
    rows: Sequence[dict[str, Any]], predictions: np.ndarray,
) -> dict[str, dict[str, Any]]:
    targets = np.asarray([row["target_steering_rad"] for row in rows], dtype=np.float64)
    result: dict[str, dict[str, Any]] = {}
    groups = {
        "S09": [index for index, row in enumerate(rows) if row["scenario_id"] == "09"],
        "S10": [index for index, row in enumerate(rows) if row["scenario_id"] == "10"],
        "combined": list(range(len(rows))),
    }
    for group, indices in groups.items():
        bins: dict[str, Any] = {}
        for bin_index, bounds in enumerate(ROUTE_BINS):
            selected = [index for index in indices if route_bin_index(rows[index]["route_progress_m"]) == bin_index]
            bins[_bin_label(bounds)] = {
                "bounds_m": list(bounds),
                **steering_metrics(predictions[selected], targets[selected]),
            }
        result[group] = bins
    return result


def _ratio(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return value / reference


def _late_bin_assessment(models: Mapping[str, Any]) -> dict[str, Any]:
    labels = [_bin_label(bounds) for bounds in ROUTE_BINS]
    comparisons: dict[str, Any] = {}
    ratios: list[float] = []
    for label in labels:
        r1 = models["R1"]["combined"][label]
        d1 = models["D1"]["combined"][label]
        mae_ratio = _ratio(d1["mae_rad"], r1["mae_rad"])
        rmse_ratio = _ratio(d1["rmse_rad"], r1["rmse_rad"])
        comparisons[label] = {
            "count": d1["count"],
            "d1_minus_r1_mae_rad": d1["mae_rad"] - r1["mae_rad"] if d1["mae_rad"] is not None else None,
            "d1_to_r1_mae_ratio": mae_ratio,
            "d1_minus_r1_rmse_rad": d1["rmse_rad"] - r1["rmse_rad"] if d1["rmse_rad"] is not None else None,
            "d1_to_r1_rmse_ratio": rmse_ratio,
        }
        if label != labels[-1] and mae_ratio is not None:
            ratios.append(mae_ratio)
    late = comparisons[labels[-1]]
    late_regression = bool(late["d1_minus_r1_mae_rad"] is not None and late["d1_minus_r1_mae_rad"] > 0.0)
    disproportionate = bool(
        late_regression
        and late["d1_to_r1_mae_ratio"] is not None
        and ratios
        and late["d1_to_r1_mae_ratio"] > max(ratios)
    )
    return {
        "per_bin_comparison": comparisons,
        "late_bin": labels[-1],
        "d1_late_bin_regression": late_regression,
        "maximum_earlier_bin_d1_to_r1_mae_ratio": max(ratios) if ratios else None,
        "d1_late_bin_regression_is_disproportionate": disproportionate,
        "rule": (
            "disproportionate iff D1 late-bin MAE exceeds R1 and the D1/R1 late-bin "
            "MAE ratio exceeds every earlier-bin ratio"
        ),
    }


def offline_route_bin_analysis(config: DiagnosisConfig, repo: Path) -> dict[str, Any]:
    source = config.sources["expert_validation_manifest"]
    path = _resolve(repo, source["path"])
    _hash_gate(path, source["sha256"], "Expert validation manifest")
    rows, limitation = _validation_rows(path)
    if limitation:
        report = {
            "version": VERSION + "_offline_route_bins",
            "generated_utc": utc_now(),
            "result": "UNAVAILABLE",
            "route_progress_metadata_available": False,
            "limitation": limitation,
            "continued_to_live_isolation": True,
            "route_progress_values_invented": False,
        }
        _write_json(config.result_dir(repo) / "offline_route_bins.json", report)
        return report
    if len(rows) != int(source["sequence_count"]):
        raise DiagnosisGateError("frozen validation sequence count changed")
    predictions: dict[str, np.ndarray] = {}
    for name in ("R1", "D1"):
        model_config = config.models[name]
        checkpoint = _resolve(repo, model_config["checkpoint_path"])
        _hash_gate(checkpoint, model_config["checkpoint_sha256"], f"{name} checkpoint")
        predictions[name] = _predict_checkpoint(_load_checkpoint(checkpoint), rows)
    model_metrics = {
        name: _model_route_metrics(rows, model_predictions)
        for name, model_predictions in predictions.items()
    }
    report = {
        "version": VERSION + "_offline_route_bins",
        "generated_utc": utc_now(),
        "result": "PASS",
        "route_progress_metadata_available": True,
        "route_progress_values_invented": False,
        "fixed_route_bins_m": [list(bounds) for bounds in ROUTE_BINS],
        "evaluation_contract": {
            "manifest_path": str(path),
            "manifest_sha256": source["sha256"],
            "sequence_count": len(rows),
            "scenarios": list(VALIDATION_SCENARIOS),
            "matched_sequence_ids": True,
            "target": "exact frozen current-frame Expert steering target at frame_t",
            "temporal_input": "exact same causal frames t-2,t-1,t for R1 and D1",
        },
        "models": model_metrics,
        "late_bin_assessment": _late_bin_assessment(model_metrics),
    }
    _write_json(config.result_dir(repo) / "offline_route_bins.json", report)
    return report


@dataclass(frozen=True)
class _LiveInferenceConfig:
    payload: dict[str, Any]

    @property
    def roi(self) -> tuple[int, int, int, int]:
        roi = self.payload["roi"]
        return (
            int(roi["x_start"]), int(roi["y_start"]),
            int(roi["x_end"]), int(roi["y_end"]),
        )


def _driver_config(config: DiagnosisConfig) -> DriverConfig:
    raw = config.control
    result = DriverConfig(
        base_url="http://localhost:8080",
        expected_world=CANONICAL_WORLD,
        wheelbase_m=float(raw["wheelbase_m"]),
        max_steering_rad=float(raw["steering_limit_rad"]),
        fixed_speed_mps=float(raw["speed_mps"]),
        control_frequency_hz=float(raw["control_frequency_hz"]),
        lookahead_m=float(raw["lookahead_m"]),
        start_gate_radius_m=float(raw["start_gate_radius_m"]),
        minimum_lap_progress_fraction=float(raw["minimum_lap_progress_fraction"]),
        off_track_margin_m=float(raw["off_track_margin_m"]),
        off_track_grace_s=float(raw["off_track_grace_s"]),
        api_timeout_s=float(raw["api_timeout_s"]),
        pose_stale_timeout_s=float(raw["pose_stale_timeout_s"]),
        pose_motion_translation_threshold_m=float(raw["pose_motion_translation_threshold_m"]),
        pose_motion_yaw_threshold_rad=float(raw["pose_motion_yaw_threshold_rad"]),
        maximum_runtime_s=float(raw["maximum_runtime_s"]),
        closed_route_tolerance_m=float(raw["closed_route_tolerance_m"]),
        spawn_route_tolerance_m=float(raw["spawn_route_tolerance_m"]),
        minimum_route_points=int(raw["minimum_route_points"]),
        maximum_progress_jump_m=float(raw["maximum_progress_jump_m"]),
        world_check_interval_s=float(raw["world_check_interval_s"]),
        reset_wait_timeout_s=float(raw["reset_wait_timeout_s"]),
    )
    result.validate()
    return result


def _live_inference_config(config: DiagnosisConfig) -> _LiveInferenceConfig:
    return _LiveInferenceConfig({
        "camera_path": config.control["camera_path"],
        "control_frequency_hz": float(config.control["control_frequency_hz"]),
        "roi": dict(config.control["roi"]),
    })


def _summary(values: Sequence[float], *, scale: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64) * scale
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _heading_error(route: Any, route_s_m: float, yaw: float) -> tuple[float, float]:
    before = route.point_at(route_s_m - 0.1)
    after = route.point_at(route_s_m + 0.1)
    route_heading = math.atan2(after[1] - before[1], after[0] - before[0])
    error = math.atan2(math.sin(yaw - route_heading), math.cos(yaw - route_heading))
    return route_heading, error


def shadow_expert_steering(
    route: Any,
    pose: Mapping[str, Any],
    *,
    lookahead_m: float = LOOKAHEAD_M,
    wheelbase_m: float = 0.18,
    maximum_steering_rad: float = MAX_STEERING_RAD,
) -> tuple[float, dict[str, float]]:
    """Evaluate the frozen lane Expert on the actual learner pose, without commanding."""
    projection = route.project((float(pose["x"]), float(pose["y"])))
    target = route.point_at(projection.s + lookahead_m)
    steering, curvature, target_distance = pure_pursuit_steering(
        (float(pose["x"]), float(pose["y"])), float(pose["yaw"]), target,
        wheelbase_m, maximum_steering_rad,
    )
    route_heading, heading_error = _heading_error(route, projection.s, float(pose["yaw"]))
    return steering, {
        "route_s_m": float(projection.s),
        "cte_m": float(projection.distance),
        "signed_cte_m": float(projection.signed_error),
        "route_heading_rad": route_heading,
        "heading_error_rad": heading_error,
        "shadow_curvature_per_m": float(curvature),
        "shadow_target_distance_m": float(target_distance),
    }


def issue_policy_commands(client: Any, policy_steering_rad: float, speed_mps: float) -> None:
    """The sole driving boundary; no shadow-Expert value is accepted here."""
    client.command_steering(clamp_steering_rad(float(policy_steering_rad), MAX_STEERING_RAD))
    client.command_speed(float(speed_mps))


def _same_steering_sign(first: float, second: float, tolerance: float = 1e-6) -> bool:
    if abs(first) <= tolerance and abs(second) <= tolerance:
        return True
    if abs(first) <= tolerance or abs(second) <= tolerance:
        return False
    return math.copysign(1.0, first) == math.copysign(1.0, second)


def _classify_live_run(metrics: Mapping[str, Any]) -> str:
    if metrics.get("result") == "PASS":
        return "FULL_LAP_PASS"
    if (
        metrics.get("temporal_input_failure")
        or int(metrics.get("api_failures", 0))
        or int(metrics.get("pose_failures", 0))
        or int(metrics.get("clock_failures", 0))
        or int(metrics.get("liveness_failures", 0))
        or metrics.get("safe_stop_success") is not True
    ):
        return "INFRA_FAIL"
    return "POLICY_FAIL"


def run_compact_live_loop(
    client: Any,
    model: Any,
    config: DiagnosisConfig,
    initial: Any,
    *,
    policy_name: str,
    monotonic: Callable[[], float] = time.monotonic,
    perf_counter: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    warm_buffer: Callable[..., tuple[Any, dict[str, Any]]] = warm_temporal_buffer,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drive once with an ONNX policy while retaining numeric telemetry only."""
    if policy_name not in ("D1", "R1"):
        raise ValueError("policy_name must be D1 or R1")
    safety = _driver_config(config)
    inference = _live_inference_config(config)
    route = initial.route
    tracker = ProgressTracker(route.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s)
    liveness = PoseLivenessMonitor(
        safety.pose_stale_timeout_s,
        safety.pose_motion_translation_threshold_m,
        safety.pose_motion_yaw_threshold_rad,
    )
    telemetry: list[dict[str, Any]] = []
    periods: list[float] = []
    camera_times: list[float] = []
    preprocessing_times: list[float] = []
    inference_times: list[float] = []
    total_model_times: list[float] = []
    gap1_values: list[float] = []
    gap2_values: list[float] = []
    span_values: list[float] = []
    api_failures = pose_failures = clock_failures = liveness_failures = 0
    temporal_failure = False
    invalid_history = 0
    timing_slips = 0
    failure: str | None = None
    result = "FAIL"
    warmup: dict[str, Any] | None = None
    motion_commanded = False
    previous_tick: float | None = None
    final_pose = initial.pose
    final_projection = route.project((float(final_pose["x"]), float(final_pose["y"])))
    stop_errors: list[str] = []
    started = monotonic()
    next_tick = started
    next_world_check = started
    try:
        buffer, warmup = warm_buffer(client, inference)
        while True:
            now = monotonic()
            if now - started >= safety.maximum_runtime_s:
                raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick:
                sleep(next_tick - now)
            tick = monotonic()
            if previous_tick is not None:
                period = tick - previous_tick
                periods.append(period)
                timing_slips += period > 0.1
            previous_tick = tick
            if tick >= next_world_check:
                status = client.status()
                if (
                    status.get("running") is not True
                    or status.get("switching") is not False
                    or status.get("current") != initial.world
                ):
                    raise RuntimeError(f"simulator state changed while driving: {status}")
                next_world_check = tick + safety.world_check_interval_s

            camera_started = perf_counter()
            jpeg = client.camera_jpeg(inference.payload["camera_path"])
            camera_times.append(perf_counter() - camera_started)
            model_started = perf_counter()
            preprocessing_started = perf_counter()
            append_live_jpeg(buffer, jpeg, monotonic(), roi=inference.roi)
            preprocessing_times.append(perf_counter() - preprocessing_started)
            gap1, gap2, span = buffer.gaps()
            gap1_values.append(gap1)
            gap2_values.append(gap2)
            span_values.append(span)
            inference_started = perf_counter()
            normalized = model.predict(buffer.tensor())
            inference_times.append(perf_counter() - inference_started)
            total_model_times.append(perf_counter() - model_started)
            policy_steering = clamp_steering_rad(
                float(steering_normalized_to_rad(normalized, MAX_STEERING_RAD)),
                MAX_STEERING_RAD,
            )
            try:
                pose = client.pose()
            except Exception:
                pose_failures += 1
                raise
            try:
                clock_payload = client.clock()
            except Exception:
                clock_failures += 1
                raise
            try:
                liveness.update(
                    pose, float(clock_payload["sim_time"]), monotonic(),
                    motion_commanded=motion_commanded,
                )
            except RuntimeError:
                liveness_failures += 1
                raise
            final_pose = pose
            final_projection = route.project((float(pose["x"]), float(pose["y"])))
            tracker.update(final_projection.s)
            shadow, geometry = shadow_expert_steering(
                route, pose, lookahead_m=LOOKAHEAD_M,
                wheelbase_m=safety.wheelbase_m,
                maximum_steering_rad=safety.max_steering_rad,
            )
            boundary_distance = route.track_boundary_distance((float(pose["x"]), float(pose["y"])))
            if boundary_distance is None or not math.isfinite(boundary_distance):
                raise RuntimeError("invalid track boundary geometry")
            off_track_now = boundary_distance > safety.off_track_margin_m
            sustained = off_track.update(off_track_now, monotonic())
            off_track_duration = (
                max(0.0, monotonic() - off_track.started_at)
                if off_track.started_at is not None else 0.0
            )
            signed_error = policy_steering - shadow
            shadow_magnitude = abs(shadow)
            row = {
                "iteration": len(telemetry),
                "monotonic_timestamp_s": tick,
                "elapsed_s": tick - started,
                "sim_time_s": float(clock_payload["sim_time"]),
                "route_s_m": geometry["route_s_m"],
                "completion_fraction": tracker.unwrapped / route.length,
                "unwrapped_progress_m": tracker.unwrapped,
                "x_m": float(pose["x"]),
                "y_m": float(pose["y"]),
                "yaw_rad": float(pose["yaw"]),
                "cte_m": geometry["cte_m"],
                "signed_cte_m": geometry["signed_cte_m"],
                "heading_error_rad": geometry["heading_error_rad"],
                "policy_name": policy_name,
                "policy_steering_rad": policy_steering,
                f"{policy_name.lower()}_steering_rad": policy_steering,
                "shadow_expert_steering_rad": shadow,
                "signed_steering_error_rad": signed_error,
                "absolute_steering_error_rad": abs(signed_error),
                "corrective_magnitude_ratio": (
                    abs(policy_steering) / shadow_magnitude if shadow_magnitude > 1e-9 else None
                ),
                "steering_sign_agreement": _same_steering_sign(policy_steering, shadow),
                "camera_gap_t_minus_2_to_t_minus_1_s": gap1,
                "camera_gap_t_minus_1_to_t_s": gap2,
                "camera_oldest_to_current_span_s": span,
                "camera_acquisition_ms": camera_times[-1] * 1000.0,
                "preprocessing_ms": preprocessing_times[-1] * 1000.0,
                "onnx_inference_ms": inference_times[-1] * 1000.0,
                "temporal_model_path_ms": total_model_times[-1] * 1000.0,
                "boundary_distance_m": float(boundary_distance),
                "off_track": off_track_now,
                "off_track_continuous_duration_s": off_track_duration,
                "steering_saturated": math.isclose(
                    abs(policy_steering), safety.max_steering_rad, abs_tol=1e-8,
                ),
            }
            telemetry.append(row)
            if sustained:
                raise RuntimeError(f"sustained off-track: boundary distance {boundary_distance:.3f}m")
            issue_policy_commands(client, policy_steering, SPEED_MPS)
            if not motion_commanded:
                motion_commanded = True
                liveness.update(
                    pose, float(clock_payload["sim_time"]), monotonic(),
                    motion_commanded=True,
                )
            distance_to_start = math.dist(
                (float(pose["x"]), float(pose["y"])), route.points[0],
            )
            if tracker.lap_complete(
                distance_to_start,
                safety.start_gate_radius_m,
                safety.minimum_lap_progress_fraction,
            ):
                result = "PASS"
                break
            next_tick += 1.0 / safety.control_frequency_hz
            if next_tick < monotonic() - 1.0 / safety.control_frequency_hz:
                next_tick = monotonic()
    except TemporalInputError as exc:
        temporal_failure = True
        invalid_history += 1
        failure = str(exc)
    except Exception as exc:
        failure = str(exc)
        lowered = failure.lower()
        if any(token in lowered for token in (
            "get ", "post ", "control rejected", "unavailable", "simulator state changed",
        )) and not (pose_failures or clock_failures):
            api_failures += 1
    finally:
        ended = monotonic()
        off_track.finalize(ended)
        stop_errors = client.safe_stop()
        if stop_errors:
            api_failures += len(stop_errors)
            result = "FAIL"
            failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
    ctes = [float(row["cte_m"]) for row in telemetry]
    steerings = [float(row["policy_steering_rad"]) for row in telemetry]
    shadow = [float(row["shadow_expert_steering_rad"]) for row in telemetry]
    errors = [float(row["signed_steering_error_rad"]) for row in telemetry]
    metrics: dict[str, Any] = {
        "result": result,
        "failure": failure,
        "policy_name": policy_name,
        "elapsed_s": monotonic() - started,
        "route_length_m": route.length,
        "final_route_s_m": float(final_projection.s),
        "total_unwrapped_progress_m": tracker.unwrapped,
        "route_completion_fraction": tracker.unwrapped / route.length,
        "final_distance_to_start_m": math.dist(
            (float(final_pose["x"]), float(final_pose["y"])), route.points[0],
        ),
        "control_iterations": len(telemetry),
        "mean_cte_m": statistics.fmean(ctes) if ctes else None,
        "max_cte_m": max(ctes, default=None),
        "mean_policy_steering_rad": statistics.fmean(steerings) if steerings else None,
        "mean_shadow_expert_steering_rad": statistics.fmean(shadow) if shadow else None,
        "mean_signed_steering_error_rad": statistics.fmean(errors) if errors else None,
        "mean_absolute_steering_error_rad": (
            statistics.fmean(abs(value) for value in errors) if errors else None
        ),
        "corrective_magnitude_ratio": (
            statistics.fmean(abs(value) for value in steerings)
            / statistics.fmean(abs(value) for value in shadow)
            if shadow and statistics.fmean(abs(value) for value in shadow) > 0.0 else None
        ),
        "steering_sign_agreement_fraction": (
            statistics.fmean(bool(row["steering_sign_agreement"]) for row in telemetry)
            if telemetry else None
        ),
        "steering_saturation_fraction": (
            statistics.fmean(bool(row["steering_saturated"]) for row in telemetry)
            if telemetry else 0.0
        ),
        "off_track_events": off_track.event_count,
        "off_track_total_duration_s": off_track.total_duration_s,
        "temporal_input_failure": temporal_failure,
        "temporal_invalid_history_count": invalid_history,
        "timing_slips_over_100ms": timing_slips,
        "api_failures": api_failures,
        "pose_failures": pose_failures,
        "clock_failures": clock_failures,
        "liveness_failures": liveness_failures,
        "safe_stop_success": not stop_errors,
        "safe_stop_errors": stop_errors,
        "warmup": warmup,
        "control_loop_period_ms": _summary(periods, scale=1000.0),
        "camera_acquisition_ms": _summary(camera_times, scale=1000.0),
        "preprocessing_ms": _summary(preprocessing_times, scale=1000.0),
        "onnx_inference_ms": _summary(inference_times, scale=1000.0),
        "temporal_model_path_ms": _summary(total_model_times, scale=1000.0),
        "temporal_frame_gaps_s": {
            "oldest_to_middle": _summary(gap1_values),
            "middle_to_current": _summary(gap2_values),
            "oldest_to_current": _summary(span_values),
        },
        "speed_mps": SPEED_MPS,
        "control_frequency_hz": CONTROL_HZ,
        "neural_observation_fields": [
            "camera_yuv_t_minus_2", "camera_yuv_t_minus_1", "camera_yuv_t",
        ],
        "shadow_expert_control_authority": False,
        "camera_images_recorded": 0,
        "bags_recorded": 0,
    }
    metrics["classification"] = _classify_live_run(metrics)
    return metrics, telemetry


def _preflight_cone_free(
    client: SimClient,
    config: DiagnosisConfig,
    repo: Path,
    sim_root: Path,
) -> tuple[Any, dict[str, Any]]:
    disk = disk_gate(
        config.payload["disk_gate"]["path"],
        int(config.payload["disk_gate"]["minimum_free_bytes"]),
    )
    assets = config.payload["canonical_assets"]
    environment_path = _resolve(repo, assets["environment_config_path"])
    _hash_gate(environment_path, assets["environment_config_sha256"], "environment config")
    environment = EnvironmentConfig.load(environment_path)
    hashes = verify_canonical_hashes(environment, share_path(sim_root))
    if (
        hashes["canonical_cone_free_world"] != assets["canonical_cone_free_world_sha256"]
        or hashes["canonical_route"] != assets["canonical_route_sha256"]
    ):
        raise DiagnosisGateError("canonical cone-free world/route identity changed")
    activation = activate_world(client, CANONICAL_WORLD)
    initial = wait_after_reset(client, _driver_config(config), False)
    if (
        initial.world != CANONICAL_WORLD
        or initial.cone_count != 0
        or initial.route_points != int(config.payload["canonical_route_points"])
        or not math.isclose(initial.route.length, ROUTE_LENGTH_M, abs_tol=1e-9)
    ):
        raise DiagnosisGateError("cone-free world route/cone preflight changed")
    clock = clock_health_preflight(client)
    if clock.get("result") != "PASS":
        raise DiagnosisGateError("simulator clock health preflight failed")
    camera_started = time.perf_counter()
    jpeg = client.camera_jpeg(config.control["camera_path"])
    with Image.open(BytesIO(jpeg)) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
        image_format = image.format
    if dimensions != [480, 360] or image_format != "JPEG":
        raise DiagnosisGateError(
            f"camera preflight differs from 480x360 JPEG: {image_format} {dimensions}"
        )
    return initial, {
        "result": "PASS",
        "generated_utc": utc_now(),
        "disk_gate": disk,
        "world_activation": activation,
        "world": initial.world,
        "cone_count": initial.cone_count,
        "route_points": initial.route_points,
        "route_length_m": initial.route.length,
        "pose": initial.pose,
        "bounds": initial.bounds,
        "clock_health": clock,
        "camera": {
            "transport": "HTTP JPEG",
            "dimensions": dimensions,
            "mode": mode,
            "acquisition_ms": (time.perf_counter() - camera_started) * 1000.0,
            "images_persisted": 0,
        },
        "canonical_hashes": hashes,
        "fixed_control": {
            "speed_mps": SPEED_MPS,
            "control_frequency_hz": CONTROL_HZ,
            "lookahead_m": LOOKAHEAD_M,
            "steering_limit_rad": MAX_STEERING_RAD,
            "wheelbase_m": 0.18,
        },
        "temporal_contract": {
            "history_frames": 3,
            "maximum_adjacent_gap_s": 0.12,
            "duplicate_padding": False,
            "warmup_while_stopped": True,
        },
        "recording": {"images": False, "bags": False},
    }


def _window_metrics(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    if not rows:
        return {"label": label, "result": "NO_SAMPLES", "count": 0}
    policy = [float(row["policy_steering_rad"]) for row in rows]
    expert = [float(row["shadow_expert_steering_rad"]) for row in rows]
    signed = [float(row["signed_steering_error_rad"]) for row in rows]
    ctes = [float(row["cte_m"]) for row in rows]
    expert_mean_abs = statistics.fmean(abs(value) for value in expert)
    return {
        "label": label,
        "result": "AVAILABLE",
        "count": len(rows),
        "start_elapsed_s": float(rows[0]["elapsed_s"]),
        "end_elapsed_s": float(rows[-1]["elapsed_s"]),
        "start_route_s_m": float(rows[0]["route_s_m"]),
        "end_route_s_m": float(rows[-1]["route_s_m"]),
        "mean_model_steering_rad": statistics.fmean(policy),
        "mean_shadow_expert_steering_rad": statistics.fmean(expert),
        "mean_signed_error_rad": statistics.fmean(signed),
        "mean_absolute_error_rad": statistics.fmean(abs(value) for value in signed),
        "corrective_magnitude_ratio": (
            statistics.fmean(abs(value) for value in policy) / expert_mean_abs
            if expert_mean_abs > 0.0 else None
        ),
        "steering_sign_agreement_fraction": statistics.fmean(
            bool(row["steering_sign_agreement"]) for row in rows
        ),
        "cte_start_m": ctes[0],
        "cte_end_m": ctes[-1],
        "cte_growth_m": ctes[-1] - ctes[0],
        "maximum_cte_m": max(ctes),
        "steering_saturation_fraction": statistics.fmean(
            bool(row["steering_saturated"]) for row in rows
        ),
        "off_track_fraction": statistics.fmean(bool(row["off_track"]) for row in rows),
        "temporal_model_path_ms": _summary([
            float(row["temporal_model_path_ms"]) / 1000.0 for row in rows
        ], scale=1000.0),
        "onnx_inference_ms": _summary([
            float(row["onnx_inference_ms"]) / 1000.0 for row in rows
        ], scale=1000.0),
        "camera_oldest_to_current_span_s": _summary([
            float(row["camera_oldest_to_current_span_s"]) for row in rows
        ]),
    }


def _cte_growth_onset_index(rows: Sequence[Mapping[str, Any]]) -> int | None:
    """Find transparent, sustained pre-failure CTE growth in the final 10 seconds."""
    if len(rows) < 3:
        return None
    end_time = float(rows[-1]["elapsed_s"])
    first_candidate = next(
        (index for index, row in enumerate(rows) if float(row["elapsed_s"]) >= end_time - 10.0),
        0,
    )
    for index in range(first_candidate, len(rows) - 1):
        target_time = float(rows[index]["elapsed_s"]) + 1.0
        later = next(
            (candidate for candidate in range(index + 1, len(rows))
             if float(rows[candidate]["elapsed_s"]) >= target_time),
            None,
        )
        if later is None:
            continue
        start_cte = float(rows[index]["cte_m"])
        end_cte = float(rows[later]["cte_m"])
        deltas = [
            float(rows[value]["cte_m"]) - float(rows[value - 1]["cte_m"])
            for value in range(index + 1, later + 1)
        ]
        nondecreasing_fraction = sum(value >= -0.002 for value in deltas) / len(deltas)
        if (
            end_cte - start_cte >= 0.10
            and nondecreasing_fraction >= 0.75
            and float(rows[-1]["cte_m"]) >= start_cte + 0.05
        ):
            return index
    return None


def analyze_live_telemetry(
    metrics: Mapping[str, Any], telemetry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_rows = list(telemetry)
    route_bins: dict[str, Any] = {}
    for index, bounds in enumerate(ROUTE_BINS):
        selected = [row for row in all_rows if route_bin_index(float(row["route_s_m"])) == index]
        route_bins[_bin_label(bounds)] = _window_metrics(selected, label=_bin_label(bounds))
    result: dict[str, Any] = {
        "result": "PASS" if all_rows else "UNAVAILABLE",
        "full_run_shadow_comparison": _window_metrics(all_rows, label="full_run"),
        "route_bin_shadow_comparison": route_bins,
        "historical_d1_s09_synchronized_telemetry": {
            "result": "UNAVAILABLE",
            "reason": (
                "the preserved D1 S09 record contains aggregate metrics but no synchronized "
                "per-iteration pose/model/shadow-Expert series"
            ),
        },
    }
    if all_rows and metrics.get("classification") != "FULL_LAP_PASS":
        stop_time = float(all_rows[-1]["elapsed_s"])
        final_run_window = [
            row for row in all_rows if float(row["elapsed_s"]) >= stop_time - 2.0
        ]
        result["final_2_seconds_before_run_stop"] = _window_metrics(
            final_run_window, label="final_2_seconds_before_run_stop",
        )
    if metrics.get("classification") == "POLICY_FAIL" and all_rows:
        stop_time = float(all_rows[-1]["elapsed_s"])
        final_window = [row for row in all_rows if float(row["elapsed_s"]) >= stop_time - 2.0]
        onset_index = _cte_growth_onset_index(all_rows)
        if onset_index is None:
            growth = {
                "label": "final_2_seconds_before_cte_growth",
                "result": "UNAVAILABLE",
                "reason": "no onset met the stated 0.10 m/1 s sustained-growth rule",
            }
        else:
            onset_time = float(all_rows[onset_index]["elapsed_s"])
            growth_rows = [
                row for row in all_rows
                if onset_time - 2.0 <= float(row["elapsed_s"]) <= onset_time
            ]
            growth = {
                **_window_metrics(growth_rows, label="final_2_seconds_before_cte_growth"),
                "cte_growth_onset_iteration": int(all_rows[onset_index]["iteration"]),
                "cte_growth_onset_elapsed_s": onset_time,
                "onset_rule": (
                    "CTE increases at least 0.10 m over the next 1.0 s, at least 75% of "
                    "step deltas are >= -0.002 m, and final CTE remains >= onset+0.05 m"
                ),
            }
        result["failure_windows"] = {
            "final_2_seconds_before_cte_growth": growth,
            "final_2_seconds_before_off_track_stop": _window_metrics(
                final_window, label="final_2_seconds_before_off_track_stop",
            ),
        }
    elif metrics.get("classification") == "INFRA_FAIL":
        result["failure_windows"] = {
            "result": "INFRA_FAILURE_NO_POLICY_WINDOW",
            "final_2_seconds_before_cte_growth": {
                "result": "NOT_ATTRIBUTABLE",
                "reason": "the run ended on invalid infrastructure rather than a policy CTE-growth stop",
            },
            "final_2_seconds_before_off_track_stop": {
                "result": "NOT_APPLICABLE",
                "reason": "no off-track event or off-track stop occurred before the infrastructure failure",
            },
        }
    else:
        result["failure_windows"] = {
            "result": "NOT_APPLICABLE",
            "reason": "cone-free run did not end in a genuine policy failure",
        }
    return result


def _refresh_live_analysis(path: Path) -> None:
    if not path.is_file():
        return
    report = _read_json(path)
    metrics = report.get("metrics") or {}
    telemetry = report.get("telemetry") or []
    if isinstance(metrics, dict) and isinstance(telemetry, list):
        report["analysis"] = analyze_live_telemetry(metrics, telemetry)
        _write_json(path, report)


def _run_marker_path(output_path: Path) -> Path:
    return output_path.with_suffix(".started.json")


def _run_one_live(
    *,
    client: SimClient,
    config: DiagnosisConfig,
    repo: Path,
    sim_root: Path,
    policy_name: str,
    output_path: Path,
) -> dict[str, Any]:
    marker = _run_marker_path(output_path)
    if output_path.exists() or marker.exists():
        raise DiagnosisGateError(
            f"{policy_name} cone-free run is limited to exactly one attempt; evidence already exists"
        )
    model_config = config.models[policy_name]
    onnx_path = _resolve(repo, model_config["onnx_path"])
    _hash_gate(onnx_path, model_config["onnx_sha256"], f"{policy_name} ONNX")
    model = TemporalOnnxModel(onnx_path)
    initial, preflight = _preflight_cone_free(client, config, repo, sim_root)
    _write_json(marker, {
        "version": VERSION + "_live_marker",
        "policy": policy_name,
        "started_utc": utc_now(),
        "status": "LIVE_RUN_STARTED_DO_NOT_RETRY",
        "maximum_runs": 1,
        "record_images": False,
        "record_bags": False,
    })
    metrics, telemetry = run_compact_live_loop(
        client, model, config, initial, policy_name=policy_name,
    )
    analysis = analyze_live_telemetry(metrics, telemetry)
    report = {
        "version": VERSION + "_cone_free_run",
        "generated_utc": utc_now(),
        "policy": policy_name,
        "valid_run_number": 1,
        "preflight": preflight,
        "classification": metrics["classification"],
        "metrics": metrics,
        "telemetry_schema": {
            "format": "compact JSON numeric rows",
            "camera_images": 0,
            "bags": 0,
            "shadow_expert_control_authority": False,
        },
        "telemetry": telemetry,
        "analysis": analysis,
    }
    _write_json(output_path, report)
    return report


def run_conditional_live(
    run_one: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Pure orchestration gate used by production and focused tests."""
    counts = {"D1": 0, "R1": 0}
    counts["D1"] += 1
    d1 = run_one("D1")
    d1_classification = d1.get("classification")
    r1: dict[str, Any] | None = None
    r1_gate_reason: str
    if d1_classification == "POLICY_FAIL":
        counts["R1"] += 1
        r1 = run_one("R1")
        r1_gate_reason = "RUN_ONCE_AFTER_D1_POLICY_FAIL"
    elif d1_classification == "FULL_LAP_PASS":
        r1_gate_reason = "NOT_RUN_D1_FULL_LAP_PASS"
    else:
        r1_gate_reason = "NOT_RUN_D1_INFRASTRUCTURE_INVALID"
    if counts["D1"] != 1 or counts["R1"] not in (0, 1):
        raise DiagnosisGateError("conditional live-run cardinality violated")
    if d1_classification == "FULL_LAP_PASS" and counts["R1"] != 0:
        raise DiagnosisGateError("R1 ran despite D1 full-lap pass")
    return {
        "D1": d1,
        "R1": r1,
        "run_counts": counts,
        "r1_gate_reason": r1_gate_reason,
    }


def _protected_world(world: object) -> bool:
    value = str(world or "")
    return any(value.endswith(f"_e2e_random_cone_v1_{scenario}") for scenario in PROTECTED_SCENARIOS)


def execute_live_isolation(
    config: DiagnosisConfig,
    repo: Path,
    sim_root: Path,
) -> dict[str, Any]:
    result_dir = config.result_dir(repo)
    d1_path = result_dir / "d1_cone_free_run.json"
    r1_path = result_dir / "r1_cone_free_run.json"
    if d1_path.is_file():
        d1 = _read_json(d1_path)
        r1 = _read_json(r1_path) if r1_path.is_file() else None
        expected_r1 = d1.get("classification") == "POLICY_FAIL"
        if expected_r1 != (r1 is not None):
            raise DiagnosisGateError("existing conditional live evidence is incomplete")
        return {
            "D1": d1,
            "R1": r1,
            "run_counts": {"D1": 1, "R1": int(r1 is not None)},
            "r1_gate_reason": (
                "RUN_ONCE_AFTER_D1_POLICY_FAIL" if r1 is not None
                else "NOT_RUN_D1_FULL_LAP_PASS"
            ),
            "resumed_from_completed_compact_evidence": True,
        }
    if _run_marker_path(d1_path).exists():
        raise DiagnosisGateError("D1 live marker exists without finalized evidence; retry is prohibited")
    if r1_path.exists() or _run_marker_path(r1_path).exists():
        raise DiagnosisGateError("R1 evidence exists before its D1 conditional gate")
    offline_distribution = _read_json(result_dir / "offline_distribution.json")
    offline_bins = _read_json(result_dir / "offline_route_bins.json")
    if offline_distribution.get("result") != "PASS" or offline_bins.get("result") not in ("PASS", "UNAVAILABLE"):
        raise DiagnosisGateError("offline audits must complete before live isolation")
    audit_preserved_inputs(config, repo)
    disk_gate(
        config.payload["disk_gate"]["path"],
        int(config.payload["disk_gate"]["minimum_free_bytes"]),
    )
    client = SimClient("http://localhost:8080", float(config.control["api_timeout_s"]))
    initial_status = client.status()
    original_world = initial_status.get("current")
    if _protected_world(original_world):
        client.safe_stop()
        raise DiagnosisGateError("refusing live isolation while a protected S11/S12 world is active")
    final_stop_errors: list[str] = []
    restoration: dict[str, Any] | None = None
    try:
        if errors := client.safe_stop():
            raise DiagnosisGateError("initial live safe stop failed: " + "; ".join(errors))

        def run_one(policy_name: str) -> dict[str, Any]:
            disk_gate(
                config.payload["disk_gate"]["path"],
                int(config.payload["disk_gate"]["minimum_free_bytes"]),
            )
            output = d1_path if policy_name == "D1" else r1_path
            return _run_one_live(
                client=client, config=config, repo=repo, sim_root=sim_root,
                policy_name=policy_name, output_path=output,
            )

        result = run_conditional_live(run_one)
    finally:
        final_stop_errors = client.safe_stop()
        if original_world and original_world != CANONICAL_WORLD and not _protected_world(original_world):
            try:
                restoration = activate_world(client, str(original_world))
            except Exception as exc:
                restoration = {"result": "FAIL", "failure": str(exc)}
            final_stop_errors.extend(client.safe_stop())
    result["initial_world"] = original_world
    result["world_restoration"] = restoration or {
        "result": "PASS", "action": "canonical cone-free world remained active",
    }
    result["final_safe_stop_success"] = not final_stop_errors
    result["final_safe_stop_errors"] = final_stop_errors
    if final_stop_errors:
        raise DiagnosisGateError("final live safe stop failed: " + "; ".join(final_stop_errors))
    return result


def classify_evidence(
    *,
    d1_run: Mapping[str, Any] | None,
    r1_run: Mapping[str, Any] | None,
    offline_route_bins: Mapping[str, Any],
    preserved_s09: Mapping[str, Any],
) -> dict[str, Any]:
    d1_classification = None if d1_run is None else d1_run.get("classification")
    r1_classification = None if r1_run is None else r1_run.get("classification")
    d1_metrics = {} if d1_run is None else (d1_run.get("metrics") or {})
    late_failure = bool(
        d1_classification == "POLICY_FAIL"
        and float(d1_metrics.get("final_route_s_m", -math.inf)) >= ROUTE_BINS[-1][0]
    )
    offline_available = offline_route_bins.get("result") == "PASS"
    offline_late_regression = bool(
        (offline_route_bins.get("late_bin_assessment") or {}).get("d1_late_bin_regression")
    )
    offline_condition = offline_late_regression if offline_available else True
    d1_s09 = preserved_s09.get("D1") or {}
    recovered_after_cone = (
        d1_s09.get("cone_avoidance") == "PASS"
        and d1_s09.get("route_recovery") == "PASS"
        and float(d1_s09.get("failure_route_s_m", 0.0)) >= ROUTE_BINS[-1][0]
    )
    if d1_classification == "FULL_LAP_PASS" and recovered_after_cone:
        classification = POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED
        reason = (
            "D1 completed the identical 1.00 m/s cone-free lap, while preserved S09 shows "
            "successful cone avoidance/recovery followed by the late failure."
        )
    elif (
        late_failure
        and r1_classification == "FULL_LAP_PASS"
        and offline_condition
    ):
        classification = DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED
        reason = (
            "D1 failed cone-free in the late route, R1 completed the matched cone-free lap, "
            + (
                "and the frozen matched validation late bin shows D1 regression."
                if offline_available else
                "while route-bin metadata was unavailable for the optional offline corroboration."
            )
        )
    elif d1_classification == "POLICY_FAIL" and r1_classification == "POLICY_FAIL":
        classification = SHARED_LANE_WEAKNESS_SUPPORTED
        reason = "D1 and R1 both genuinely failed the identical cone-free 1.00 m/s isolation run."
    else:
        classification = MIXED_OR_INCONCLUSIVE
        missing: list[str] = []
        if d1_run is None:
            missing.append("a valid D1 cone-free run")
        elif d1_classification == "INFRA_FAIL":
            missing.append("a D1 run with valid infrastructure/temporal inputs")
        if d1_classification == "POLICY_FAIL" and r1_run is None:
            missing.append("the conditionally required R1 cone-free run")
        elif r1_classification == "INFRA_FAIL":
            missing.append("an R1 run with valid infrastructure/temporal inputs")
        if offline_available and not offline_late_regression and r1_classification == "FULL_LAP_PASS":
            missing.append("offline late-bin corroboration of a D1 regression")
        reason = "Evidence does not satisfy one registered separation rule. Missing/conflicting: " + (
            "; ".join(missing) if missing else "the observed live/offline pattern is conflicting"
        )
    return {
        "classification": classification,
        "reason": reason,
        "conditions": {
            "d1_cone_free_classification": d1_classification,
            "d1_cone_free_failure_in_late_region": late_failure,
            "r1_cone_free_classification": r1_classification,
            "offline_route_bins_available": offline_available,
            "offline_matched_late_bin_d1_regression": offline_late_regression if offline_available else None,
            "preserved_d1_s09_cone_avoidance_and_recovery_pass": recovered_after_cone,
        },
    }


def recommended_next_intervention(classification: str) -> dict[str, Any]:
    if classification == POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED:
        direction = (
            "Run one bounded TRAIN-only DAgger2 targeting post-recovery and late-route states "
            "actually visited after successful cone avoidance."
        )
    elif classification == DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED:
        direction = (
            "Run one existing-data source-mixing/route-coverage training A/B that controls the "
            "DAgger contribution across route regions; do not collect DAgger2 first."
        )
    elif classification == SHARED_LANE_WEAKNESS_SUPPORTED:
        direction = (
            "Run one dedicated 1.00 m/s cone-free temporal lane-baseline diagnosis before any "
            "additional random-cone learning."
        )
    else:
        direction = (
            "Acquire only the single missing valid cone-free comparison identified by the "
            "classification evidence before choosing a learning intervention."
        )
    return {
        "count": 1,
        "direction": direction,
        "implemented_in_this_milestone": False,
    }


def _git_status(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.rstrip()


def _changed_files(repo: Path) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--short"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    return [line[3:] if len(line) > 3 else line for line in output]


def _load_live_evidence(result_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    d1_path = result_dir / "d1_cone_free_run.json"
    r1_path = result_dir / "r1_cone_free_run.json"
    return (
        _read_json(d1_path) if d1_path.is_file() else None,
        _read_json(r1_path) if r1_path.is_file() else None,
    )


def _compact_live_summary(report: Mapping[str, Any] | None, telemetry_file: str) -> dict[str, Any] | None:
    if report is None:
        return None
    telemetry = report.get("telemetry") or []
    return {
        key: report[key]
        for key in (
            "version", "generated_utc", "policy", "valid_run_number", "preflight",
            "classification", "metrics", "telemetry_schema", "analysis",
        )
        if key in report
    } | {
        "telemetry_file": telemetry_file,
        "telemetry_sample_count": len(telemetry) if isinstance(telemetry, list) else None,
        "telemetry_embedded_in_summary": False,
    }


def build_summary(
    config: DiagnosisConfig,
    repo: Path,
    *,
    tests: Mapping[str, Any] | None = None,
    diff_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_dir = config.result_dir(repo)
    audit_path = result_dir / "audit.json"
    distribution_path = result_dir / "offline_distribution.json"
    bins_path = result_dir / "offline_route_bins.json"
    audit = _read_json(audit_path) if audit_path.is_file() else audit_preserved_inputs(config, repo)
    distribution = _read_json(distribution_path)
    offline_bins = _read_json(bins_path)
    d1, r1 = _load_live_evidence(result_dir)
    preserved = audit["preserved_s09_comparison"]
    classification = classify_evidence(
        d1_run=d1,
        r1_run=r1,
        offline_route_bins=offline_bins,
        preserved_s09=preserved,
    )
    next_intervention = recommended_next_intervention(classification["classification"])
    if classification["classification"] == POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED:
        cause_statement = (
            "The measured isolation supports a residual post-avoidance learner-state shift, "
            "not a generic cone-free late-lap lane regression."
        )
    elif classification["classification"] == DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED:
        cause_statement = "The measured isolation supports a DAgger1-induced nominal late-lap regression."
    elif classification["classification"] == SHARED_LANE_WEAKNESS_SUPPORTED:
        cause_statement = "The measured isolation supports a shared 1.00 m/s cone-free lane-following weakness."
    else:
        cause_statement = "The available evidence does not separate generic lane regression from residual post-avoidance shift."
    current_hash_audit = audit_preserved_inputs(config, repo)
    summary = {
        "version": VERSION,
        "generated_utc": utc_now(),
        "result": "DIAGNOSIS_COMPLETE" if d1 is not None else "INCOMPLETE",
        "diagnostic_only": True,
        "simulator_evidence_only": True,
        "real_robot_success_claimed": False,
        "preserved_hashes": current_hash_audit["hashes"],
        "disk_state": disk_gate(
            config.payload["disk_gate"]["path"],
            int(config.payload["disk_gate"]["minimum_free_bytes"]),
        ),
        "offline_distribution": distribution,
        "offline_route_bins": offline_bins,
        "preserved_s09_comparison": preserved,
        "d1_cone_free": _compact_live_summary(d1, "d1_cone_free_run.json"),
        "r1_cone_free": _compact_live_summary(r1, "r1_cone_free_run.json"),
        "r1_conditional_result": (
            "RUN_ONCE_AFTER_D1_POLICY_FAIL" if r1 is not None
            else (
                "NOT_RUN_D1_FULL_LAP_PASS" if d1 and d1.get("classification") == "FULL_LAP_PASS"
                else "NOT_RUN_D1_INFRASTRUCTURE_INVALID"
            )
        ),
        "historical_s09_late_window": {
            "result": "UNAVAILABLE",
            "reason": (
                "no synchronized historical D1 S09 per-iteration telemetry exists; only the "
                "new cone-free run supports per-iteration steering/CTE analysis"
            ),
        },
        "classification": classification,
        "d1_s09_failure_interpretation": cause_statement,
        "recommended_next_intervention": next_intervention,
        "prohibited_actions_audit": {
            "training_invocations": 0,
            "bags_collected": 0,
            "camera_images_persisted": 0,
            "dagger_data_collected": 0,
            "dagger_iteration2_created": False,
            "checkpoints_or_onnx_created_or_modified": False,
            "frozen_scenarios_modified": False,
            "controller_route_lookahead_speed_or_safety_changed": False,
            "commit_performed": False,
            "push_performed": False,
        },
        "s11_s12_protection_audit": {
            "result": "PASS",
            "scenarios": list(PROTECTED_SCENARIOS),
            "live_runs": 0,
            "bags_collected": 0,
            "camera_data_inspected": False,
            "expert_labels_generated": 0,
            "manifest_rows_added": 0,
            "present_in_accessed_train_or_validation_manifests": False,
        },
        "tests": dict(tests or {"result": "PENDING_FINAL_REGRESSION"}),
        "git_diff_check": dict(diff_check or {"result": "PENDING_FINAL_REGRESSION"}),
        "files_changed": _changed_files(repo),
        "final_git_status": _git_status(repo),
    }
    return summary


def _format_count(value: object) -> str:
    return "—" if value is None else str(value)


def _format_float(value: object, digits: int = 6) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _report_markdown(summary: Mapping[str, Any]) -> str:
    distribution = summary["offline_distribution"]
    offline = summary["offline_route_bins"]
    d1 = summary.get("d1_cone_free")
    r1 = summary.get("r1_cone_free")
    classification = summary["classification"]
    hashes = summary["preserved_hashes"]["models"]
    lines = [
        "# Random-Cone D1 Late-Lap Isolation V1",
        "",
        "Diagnostic-only simulator milestone. No simulator result is presented as real-robot evidence.",
        "",
        "## 1. Preserved R1/D1 hashes",
        "",
        f"- R1 checkpoint / ONNX: `{hashes['R1']['checkpoint']}` / `{hashes['R1']['onnx']}`",
        f"- D1 checkpoint / ONNX: `{hashes['D1']['checkpoint']}` / `{hashes['D1']['onnx']}`",
        f"- R1 freeze / seal: `{hashes['R1']['freeze']}` / `{hashes['R1']['freeze_seal']}`",
        f"- D1 freeze / seal: `{hashes['D1']['freeze']}` / `{hashes['D1']['freeze_seal']}`",
        "",
        "## 2. Disk state",
        "",
        f"Root free space: `{summary['disk_state']['free_gib']:.3f} GiB`; required: `5.500 GiB`; gate: **{summary['disk_state']['result']}**.",
        "",
        "## 3. Expert vs DAgger1 route-bin distribution",
        "",
        "| Route bin | Expert | DAgger1 | Aggregate | DAgger1 fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in distribution["bins"].items():
        lines.append(
            f"| {label} | {item['expert_baseline_sequence_count']} | {item['dagger1_sequence_count']} | "
            f"{item['aggregate_sequence_count']} | {_format_float(item['dagger1_fraction'], 4)} |"
        )
    completion = distribution["learner_rollout_completion"]
    lines.extend([
        "",
        "## 4. DAgger1 late-route coverage",
        "",
        f"DAgger1 late-bin sequences: **{distribution['dagger1_late_lap_sequence_count']}**. "
        f"Zero late-lap contribution confirmed: **{str(distribution['dagger1_contributes_zero_late_lap_samples']).lower()}**. "
        f"Learner rollouts ended from {completion['minimum_completion_percent']:.2f}% to {completion['maximum_completion_percent']:.2f}% completion; all genuine failures were retained without retry.",
        "",
        "Cone-phase counts and per-scenario/per-rollout route bins are preserved in `offline_distribution.json`. The frozen Expert aggregate does not encode cone phase, so no Expert phase values were invented.",
        "",
        "## 5. Matched R1/D1 offline route bins",
        "",
    ])
    if offline.get("result") == "PASS":
        lines.extend([
            "Exact frozen S09/S10 current-frame targets and the same causal frame triplets were used for both models.",
            "",
            "| Group | Route bin | N | R1 MAE | D1 MAE | R1 RMSE | D1 RMSE | D1/R1 MAE |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        comparisons = offline["late_bin_assessment"]["per_bin_comparison"]
        for group in ("S09", "S10", "combined"):
            for label in (_bin_label(bounds) for bounds in ROUTE_BINS):
                r1_metrics = offline["models"]["R1"][group][label]
                d1_metrics = offline["models"]["D1"][group][label]
                ratio = _ratio(d1_metrics["mae_rad"], r1_metrics["mae_rad"])
                lines.append(
                    f"| {group} | {label} | {r1_metrics['count']} | {_format_float(r1_metrics['mae_rad'])} | "
                    f"{_format_float(d1_metrics['mae_rad'])} | {_format_float(r1_metrics['rmse_rad'])} | "
                    f"{_format_float(d1_metrics['rmse_rad'])} | {_format_float(ratio, 3)} |"
                )
        assessment = offline["late_bin_assessment"]
        lines.extend([
            "",
            f"D1 late-bin regression: **{str(assessment['d1_late_bin_regression']).lower()}**; "
            f"disproportionate by the registered ratio rule: **{str(assessment['d1_late_bin_regression_is_disproportionate']).lower()}**. "
            "Bias, maximum error, correlation, and corrective-magnitude ratio for every group/bin are in `offline_route_bins.json`.",
        ])
    else:
        lines.append(f"Unavailable: {offline.get('limitation')}. No route progress was invented.")
    preserved = summary["preserved_s09_comparison"]
    lines.extend([
        "",
        "## 6. Preserved S09 comparison",
        "",
        f"R1 failed before the cone at {preserved['R1']['completion_percent']:.2f}% completion. D1 passed the cone with "
        f"{preserved['D1']['minimum_cone_clearance_m']:.6f} m clearance, recovered in "
        f"{preserved['D1']['recovery_time_s']:.3f} s, then failed at s={preserved['D1']['failure_route_s_m']:.3f} m "
        f"({preserved['D1']['completion_percent']:.2f}%).",
        "",
        preserved["comparison"],
        "",
        "## 7. D1 cone-free live result",
        "",
    ])
    if d1:
        metrics = d1["metrics"]
        lines.append(
            f"Classification: **{d1['classification']}**; completion `{100.0 * metrics['route_completion_fraction']:.2f}%`; "
            f"final s `{metrics['final_route_s_m']:.3f} m`; max CTE `{_format_float(metrics['max_cte_m'], 4)} m`; "
            f"safe stop `{str(metrics['safe_stop_success']).lower()}`."
        )
        if metrics.get("failure"):
            lines.append(f"Stop reason: `{metrics['failure']}`.")
    else:
        lines.append("No finalized D1 cone-free evidence exists.")
    lines.extend([
        "",
        "## 8. Conditional R1 result",
        "",
    ])
    if r1:
        metrics = r1["metrics"]
        lines.append(
            f"R1 ran exactly once after D1 POLICY_FAIL: **{r1['classification']}**; completion "
            f"`{100.0 * metrics['route_completion_fraction']:.2f}%`; max CTE `{_format_float(metrics['max_cte_m'], 4)} m`."
        )
    else:
        lines.append(f"R1: **{summary['r1_conditional_result']}**.")
    lines.extend([
        "",
        "## 9. Shadow-Expert and late-window findings",
        "",
    ])
    if d1:
        comparison = d1["analysis"]["full_run_shadow_comparison"]
        lines.append(
            f"D1 vs shadow Expert: mean signed error `{_format_float(comparison.get('mean_signed_error_rad'))} rad`, "
            f"mean absolute error `{_format_float(comparison.get('mean_absolute_error_rad'))} rad`, corrective ratio "
            f"`{_format_float(comparison.get('corrective_magnitude_ratio'), 4)}`, sign agreement "
            f"`{_format_float(comparison.get('steering_sign_agreement_fraction'), 4)}`. The shadow Expert never commanded the vehicle."
        )
        failure_windows = d1["analysis"].get("failure_windows") or {}
        if failure_windows.get("result") == "NOT_APPLICABLE":
            lines.append("No cone-free failure window applies because D1 completed the lap.")
        elif failure_windows.get("result") == "INFRA_FAILURE_NO_POLICY_WINDOW":
            run_stop = d1["analysis"].get("final_2_seconds_before_run_stop") or {}
            lines.append(
                f"Final 2 s before the infrastructure stop: model/shadow means "
                f"`{_format_float(run_stop.get('mean_model_steering_rad'))}` / "
                f"`{_format_float(run_stop.get('mean_shadow_expert_steering_rad'))}` rad; CTE growth "
                f"`{_format_float(run_stop.get('cte_growth_m'), 4)} m`; saturation "
                f"`{_format_float(run_stop.get('steering_saturation_fraction'), 4)}`. "
                "There was no off-track event, so an off-track-stop window is not applicable."
            )
        else:
            stop_window = failure_windows.get("final_2_seconds_before_off_track_stop") or {}
            lines.append(
                f"Final 2 s before stop: model/shadow means `{_format_float(stop_window.get('mean_model_steering_rad'))}` / "
                f"`{_format_float(stop_window.get('mean_shadow_expert_steering_rad'))}` rad; CTE growth "
                f"`{_format_float(stop_window.get('cte_growth_m'), 4)} m`; saturation "
                f"`{_format_float(stop_window.get('steering_saturation_fraction'), 4)}`."
            )
    lines.extend([
        "",
        "Historical S09 synchronized late windows are unavailable; the preserved S09 file is aggregate-only. Only the new cone-free telemetry supports per-iteration analysis.",
        "",
        "## 10. Final classification",
        "",
        f"**{classification['classification']}**",
        "",
        classification["reason"],
        "",
        "## 11. Generic regression vs post-avoidance residual",
        "",
        summary["d1_s09_failure_interpretation"],
        "",
        "## 12. Exactly one recommended next intervention",
        "",
        summary["recommended_next_intervention"]["direction"],
        "",
        "This intervention was not implemented here.",
        "",
        "## 13. No-training/data-collection confirmation",
        "",
        "Training invocations: `0`; bags: `0`; persisted camera images: `0`; DAgger data: `0`; DAgger2: `false`; checkpoint/ONNX changes: `false`.",
        "",
        "## 14. S11/S12 protection audit",
        "",
        "**PASS** — no live activation, bag, camera inspection, Expert label, or manifest row involved S11/S12.",
        "",
        "## 15. Tests",
        "",
        f"Result: **{summary['tests'].get('result')}**; {summary['tests'].get('summary', 'final regression pending')}.",
        "",
        f"`git diff --check`: **{summary['git_diff_check'].get('result')}**.",
        "",
        "## 16. Files changed",
        "",
        *[f"- `{path}`" for path in summary["files_changed"]],
        "",
        "## 17. Final Git status",
        "",
        "```text",
        summary["final_git_status"],
        "```",
        "",
        "No commit or push was performed.",
        "",
    ])
    return "\n".join(lines)


def write_summary_and_report(
    config: DiagnosisConfig,
    repo: Path,
    *,
    tests: Mapping[str, Any] | None = None,
    diff_check: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_dir = config.result_dir(repo)
    _refresh_live_analysis(result_dir / "d1_cone_free_run.json")
    _refresh_live_analysis(result_dir / "r1_cone_free_run.json")
    summary = build_summary(config, repo, tests=tests, diff_check=diff_check)
    _write_json(result_dir / "summary.json", summary)
    _write_text(result_dir / "REPORT.md", _report_markdown(summary))
    return summary


def _run_audit(config: DiagnosisConfig, repo: Path) -> dict[str, Any]:
    audit = audit_preserved_inputs(config, repo)
    audit["disk_gate"] = disk_gate(
        config.payload["disk_gate"]["path"],
        int(config.payload["disk_gate"]["minimum_free_bytes"]),
    )
    _write_json(config.result_dir(repo) / "audit.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--audit", action="store_true")
    stage.add_argument("--offline", action="store_true")
    stage.add_argument("--preflight", action="store_true")
    stage.add_argument("--live", action="store_true")
    stage.add_argument("--report", action="store_true")
    stage.add_argument("--finalize", action="store_true")
    stage.add_argument("--all", action="store_true")
    parser.add_argument("--test-summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    config_path = (args.config or repo / "configs/random_cone_d1_late_lap_diagnosis_v1.json").resolve()
    config = load_config(config_path, repo)
    sim_root = args.sim_root.expanduser().resolve()
    try:
        if args.audit:
            result = _run_audit(config, repo)
        elif args.offline:
            _run_audit(config, repo)
            distribution = offline_distribution_audit(config, repo)
            route_bins = offline_route_bin_analysis(config, repo)
            result = {"distribution": distribution["result"], "route_bins": route_bins["result"]}
        elif args.preflight:
            _run_audit(config, repo)
            client = SimClient("http://localhost:8080", float(config.control["api_timeout_s"]))
            original = client.status().get("current")
            if _protected_world(original):
                raise DiagnosisGateError("protected S11/S12 world is active")
            try:
                _initial, result = _preflight_cone_free(client, config, repo, sim_root)
            finally:
                client.safe_stop()
                if original and original != CANONICAL_WORLD and not _protected_world(original):
                    activate_world(client, str(original))
                    client.safe_stop()
            _write_json(config.result_dir(repo) / "cone_free_preflight.json", result)
        elif args.live:
            result = execute_live_isolation(config, repo, sim_root)
        elif args.report:
            result = write_summary_and_report(config, repo)
        elif args.finalize:
            if not args.test_summary:
                raise DiagnosisGateError("--finalize requires --test-summary")
            diff = subprocess.run(
                ["git", "diff", "--check"], cwd=repo,
                text=True, capture_output=True,
            )
            if diff.returncode:
                raise DiagnosisGateError("git diff --check failed: " + diff.stdout + diff.stderr)
            result = write_summary_and_report(
                config, repo,
                tests={"result": "PASS", "summary": args.test_summary},
                diff_check={"result": "PASS", "output": diff.stdout.strip()},
            )
        else:
            _run_audit(config, repo)
            offline_distribution_audit(config, repo)
            offline_route_bin_analysis(config, repo)
            execute_live_isolation(config, repo, sim_root)
            result = write_summary_and_report(config, repo)
        displayed: Any
        if not isinstance(result, dict):
            displayed = result
        elif "result" in result:
            displayed = result["result"]
        elif isinstance(result.get("D1"), dict):
            displayed = {
                "D1": result["D1"].get("classification"),
                "R1": None if result.get("R1") is None else result["R1"].get("classification"),
                "run_counts": result.get("run_counts"),
                "r1_gate_reason": result.get("r1_gate_reason"),
            }
        else:
            displayed = result
        print(json.dumps({"version": VERSION, "result": displayed}, indent=2, sort_keys=True))
        return 0
    except DiagnosisGateError as exc:
        print(f"ERROR: {exc}")
        return 2
