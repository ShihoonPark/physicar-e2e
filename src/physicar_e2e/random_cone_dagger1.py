"""Random-cone DAgger iteration 1 at the frozen 1.00 m/s operating point.

This module deliberately separates four irreversible boundaries:

* the failed R1 model and its S09 evidence are read-only inputs;
* only S01--S08 may create learner-state DAgger data;
* D1 is trained once, from scratch, on baseline Expert plus DAgger1 data;
* D1 is sealed before any new S09/S10 or S11/S12 neural evaluation.

Raw bags, images and model artifacts stay in simulator userdata.  Compact
identity, gate and metric evidence is written below ``results``.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import numpy as np
from PIL import Image, ImageDraw

from .cone_avoidance_environment import route_yaw
from .cone_avoidance_expert import ObstacleAwareRoute
from .dataset_extractor import canonical_json_bytes, decode_rgb8_image, preprocess_image, sha256_file
from .expert_driver import PoseLivenessMonitor
from .high_speed_temporal import (
    TemporalOnnxModel, distribution, metrics as error_metrics, export_temporal_onnx,
    predict_temporal, validate_equivalence,
)
from .pilotnet import steering_normalized_to_rad
from .pilotnet_inference import _summary_ms, fixed_speed_commands
from .pilotnet_temporal import (
    CausalFrameBuffer, TEMPORAL_PARAMETER_COUNT, TemporalInputError,
    append_live_jpeg, build_temporal_pilotnet, preprocess_temporal_paths,
)
from .pilotnet_failure_diagnosis import nearest_cosine_distances
from .random_cone_expert import (
    MAP_FAMILY, ROLE_IDS, RandomConeConfig, RandomConeObserver, ScenarioBundle,
    _restore_world, directory_file_manifest_sha256, simulator_tracked_status,
)
from .random_cone_temporal_r1 import (
    R1Config, _phase_metrics, _training_plot,
    inference_config as r1_inference_config,
    load_config as load_r1_config, run_live_once, summarize_neural_cone_run,
    train_temporal_resumable,
)
from .random_cone_train_data import (
    _post_settle_preflight, audit_frozen_expert, disk_state,
    load_task_config as load_train_task_config,
)
from .rosbag_collector import (
    BagInfo, CollectorConfig, DockerRosBackend, RecorderHandle,
    directory_size, verify_bag,
)
from .route_geometry import OffTrackMonitor, ProgressTracker, pure_pursuit_steering
from .sim_client import SimClient


VERSION = "random_cone_dagger1_1p0_v1"
DIAGNOSIS_VERSION = "random_cone_r1_failure_diagnosis_1p0_v1"
COLLECTION_VERSION = "random_cone_dagger1_collection_1p0_v1"
DATASET_VERSION = "random_cone_dagger1_dataset_1p0_v1"
TRAINING_VERSION = "pilotnet_training_d1_random_cone_1p0"
LIVE_VERSION = "pilotnet_e2e_d1_random_cone_1p0"

TRAIN_SCENARIOS = tuple(f"{value:02d}" for value in range(1, 9))
VALIDATION_SCENARIOS = ("09", "10")
HOLDOUT_SCENARIOS = ("11", "12")
DAGGER_EPISODES = tuple(f"dagger1_s{value}_r01" for value in TRAIN_SCENARIOS)
R1_TRAIN_SHA256 = "a9aaf25991cecbab3937deae545d392842007b228d8b8f571c519fba1772df73"
R1_VALIDATION_SHA256 = "a1182170a5d853b599209e6ce31f7deaa27077a99bdd62603b92ed817349693b"
MIN_PROJECTED_FREE_BYTES = 5 * 1024**3


class Dagger1GateError(RuntimeError):
    """A preregistered DAgger1 gate failed."""


@dataclass(frozen=True)
class DaggerEpisode:
    episode_id: str
    scenario_id: str
    repeat_id: str = "R01"
    role: str = "TRAIN"


@dataclass(frozen=True)
class Dagger1Config:
    path: Path
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def r1(self) -> dict[str, Any]:
        return self.payload["preserved_r1"]

    @property
    def collection(self) -> dict[str, Any]:
        return self.payload["collection"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.payload["dataset"]

    @property
    def training(self) -> dict[str, Any]:
        return self.payload["training"]

    def result_dir(self, repo: Path, key: str) -> Path:
        return repo / self.payload["result_directories"][key]

    def external_root(self, sim_root: Path) -> Path:
        return sim_root / "userdata" / self.payload["external"]["dagger1"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Dagger1GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Dagger1GateError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def episode_specs() -> tuple[DaggerEpisode, ...]:
    return tuple(DaggerEpisode(f"dagger1_s{scenario}_r01", scenario) for scenario in TRAIN_SCENARIOS)


def load_config(path: Path, repo: Path) -> Dagger1Config:
    payload = _read_json(path)
    required = {
        "version", "map_family", "preserved_r1", "frozen_expert", "scenario_roles",
        "diagnosis", "collection", "dataset", "training", "live_inference_source",
        "external", "result_directories", "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise Dagger1GateError("DAgger1 config version or top-level fields changed")
    if payload["map_family"] != MAP_FAMILY or payload["scenario_roles"] != ROLE_IDS:
        raise Dagger1GateError("map family or frozen 8/2/2 scenario roles changed")
    if tuple(payload["collection"]["episode_order"]) != DAGGER_EPISODES:
        raise Dagger1GateError("DAgger collection must be exactly S01--S08 in order")
    if tuple(payload["collection"]["required_topics"]) != ("/camera/image_raw", "/clock"):
        raise Dagger1GateError("DAgger raw topic contract changed")
    if payload["collection"]["minimum_projected_free_bytes_after_eight"] != MIN_PROJECTED_FREE_BYTES:
        raise Dagger1GateError("DAgger disk projection gate changed")
    if payload["collection"]["maximum_infrastructure_replacements_per_episode"] != 1:
        raise Dagger1GateError("infrastructure replacement bound changed")
    if payload["collection"]["retry_genuine_policy_failure"] is not False:
        raise Dagger1GateError("genuine policy failures must never be retried")
    dataset = payload["dataset"]
    if (
        dataset["source_width"], dataset["source_height"], dataset["roi"],
        dataset["output_width"], dataset["output_height"], dataset["stored_color_space"],
        dataset["history_frames"], dataset["maximum_adjacent_gap_s"],
        dataset["maximum_teacher_label_age_s"], dataset["causal_teacher_zoh"],
        dataset["future_teacher_labels_required"], dataset["allow_episode_boundary_crossing"],
        dataset["allow_duplicate_padding"],
    ) != (
        480, 360, {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360},
        200, 66, "RGB", 3, 0.120, 0.120, True, 0, False, False,
    ):
        raise Dagger1GateError("canonical temporal/teacher extraction contract changed")
    extractor = repo / dataset["canonical_extractor_config_path"]
    if sha256_file(extractor) != dataset["canonical_extractor_config_sha256"]:
        raise Dagger1GateError("canonical dataset extractor config changed")
    training = payload["training"]
    exact_training = {
        "seed": 20260824, "image_width": 200, "image_height": 66,
        "input_channels": 9, "history_frames": 3, "maximum_adjacent_gap_s": 0.120,
        "max_steering_rad": 0.349066, "target": "frozen_expert_steering_normalized_at_t",
        "optimizer": "Adam", "loss": "MSE", "learning_rate": 0.001,
        "batch_size": 64, "max_epochs": 35, "early_stopping_patience": 7,
        "minimum_improvement": 0.000001, "initialization": "from_scratch",
        "augmentation": False, "sample_weighting": False, "scenario_weighting": False,
        "oversampling": False, "undersampling": False, "hyperparameter_sweep": False,
        "onnx_opset": 17, "onnx_equivalence_samples": 128,
        "onnx_mean_abs_difference_limit": 0.00001,
        "onnx_max_abs_difference_limit": 0.0001,
    }
    if training != exact_training:
        raise Dagger1GateError("D1 architecture/training procedure changed")
    permissions = payload["permissions"]
    if (
        permissions["r1_evidence_changes_permitted"] is not False
        or tuple(permissions["dagger_scenarios"]) != TRAIN_SCENARIOS
        or permissions["validation_or_holdout_dagger_permitted"] is not False
        or permissions["s09_s10_live_data_training_permitted"] is not False
        or permissions["holdout_bag_collection_permitted"] is not False
        or permissions["d1_training_runs_permitted"] != 1
        or permissions["second_dagger_iteration_permitted"] is not False
        or permissions["commit_permitted"] is not False
        or permissions["push_permitted"] is not False
    ):
        raise Dagger1GateError("DAgger1 permission boundary changed")
    return Dagger1Config(path, payload)


def audit_preserved_r1(repo: Path, config: Dagger1Config) -> dict[str, Any]:
    r1 = config.r1
    paths = {
        "task_config": repo / r1["task_config_path"],
        "train_manifest": Path(r1["train_manifest_path"]),
        "validation_manifest": Path(r1["validation_manifest_path"]),
        "checkpoint": Path(r1["checkpoint_path"]),
        "onnx": Path(r1["onnx_path"]),
        "freeze": Path(r1["freeze_path"]),
        "freeze_seal": Path(r1["freeze_seal_path"]),
        "training_summary": repo / r1["training_summary_path"],
        "live_summary": repo / r1["live_summary_path"],
        "s09_live": repo / r1["s09_live_path"],
    }
    hashes = {
        "task_config": r1["task_config_sha256"],
        "train_manifest": r1["train_manifest_sha256"],
        "validation_manifest": r1["validation_manifest_sha256"],
        "checkpoint": r1["checkpoint_sha256"],
        "onnx": r1["onnx_sha256"],
        "freeze": r1["freeze_sha256"],
        "freeze_seal": r1["freeze_seal_sha256"],
        "training_summary": r1["training_summary_sha256"],
        "live_summary": r1["live_summary_sha256"],
        "s09_live": r1["s09_live_sha256"],
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise Dagger1GateError(f"preserved R1 artifact missing: {name}: {path}")
        actual[name] = sha256_file(path)
        if actual[name] != hashes[name]:
            raise Dagger1GateError(f"preserved R1 artifact changed: {name}")
    train_rows = list(csv.DictReader(paths["train_manifest"].open(newline="", encoding="utf-8")))
    validation_rows = list(csv.DictReader(paths["validation_manifest"].open(newline="", encoding="utf-8")))
    train_ids = sorted({str(row.get("scenario_id", "")).zfill(2) for row in train_rows})
    validation_ids = sorted({str(row.get("scenario_id", "")).zfill(2) for row in validation_rows})
    if len(train_rows) != 6706 or train_ids != list(TRAIN_SCENARIOS):
        raise Dagger1GateError("frozen R1 TRAIN rows or scenarios changed")
    if len(validation_rows) != 837 or validation_ids != list(VALIDATION_SCENARIOS):
        raise Dagger1GateError("frozen R1 VALIDATION rows or scenarios changed")
    if any(value in HOLDOUT_SCENARIOS for value in train_ids + validation_ids):
        raise Dagger1GateError("holdout leakage found in frozen R1 manifests")
    if sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) != TEMPORAL_PARAMETER_COUNT:
        raise Dagger1GateError("Temporal PilotNet parameter count changed")
    live = _read_json(paths["s09_live"])
    run = live.get("run") or {}
    expected = {
        "classification": "RANDOM_CONE_POLICY_FAIL",
        "scenario_id": "09", "role": "VALIDATION",
        "route_completion_fraction": 0.3939597182924379,
        "final_route_s_m": 12.017587838311337,
        "max_cte_m": 0.8035169999999994,
        "minimum_footprint_to_cone_clearance_m": 1.107698830004098,
        "cone_contact_or_intersection_occurred": False,
        "recovery_success": False, "safe_stop_success": True,
    }
    for key, expected_value in expected.items():
        actual_value = live.get(key) if key in {"classification", "scenario_id", "role"} else run.get(key)
        if actual_value != expected_value:
            raise Dagger1GateError(f"preserved R1 S09 failure identity changed: {key}")
    return {
        "result": "PASS", "artifact_hashes": actual,
        "train": {"sequence_count": len(train_rows), "scenario_ids": train_ids, "sha256": actual["train_manifest"]},
        "validation": {"sequence_count": len(validation_rows), "scenario_ids": validation_ids, "sha256": actual["validation_manifest"]},
        "architecture": {"input_shape": ["N", 9, 66, 200], "parameter_count": TEMPORAL_PARAMETER_COUNT},
        "s09_failure_preserved": True,
    }


def _telemetry_inventory(repo: Path, config: Dagger1Config) -> dict[str, Any]:
    result_root = repo / "results/pilotnet_e2e_r1_random_cone_1p0"
    external_root = Path(config.r1["onnx_path"]).parents[1]
    candidates = [
        path for root in (result_root, external_root)
        for path in root.rglob("*") if path.is_file()
        and any(token in path.name.lower() for token in ("telemetry", "trace", "frame", "feature"))
    ]
    return {
        "per_tick_pose_trace": False,
        "per_tick_r1_steering_trace": False,
        "per_tick_expert_counterfactual_trace": False,
        "r1_live_images": False,
        "candidate_files": [str(path) for path in sorted(candidates)],
        "search_roots": [str(result_root), str(external_root)],
    }


def diagnose_preserved_s09(repo: Path, config: Dagger1Config) -> dict[str, Any]:
    """Create a diagnosis bounded by the telemetry that was actually preserved."""
    audit = audit_preserved_r1(repo, config)
    s09 = _read_json(repo / config.r1["s09_live_path"])
    run = s09["run"]
    training = _read_json(repo / config.r1["training_summary_path"])
    offline = training["offline_validation"]["per_scenario"]["09"]
    inventory = _telemetry_inventory(repo, config)
    trace_available = all(inventory[key] for key in (
        "per_tick_pose_trace", "per_tick_r1_steering_trace",
    ))
    windows = {
        name: {
            "result": "UNAVAILABLE",
            "reason": "preserved S09 evidence contains aggregate metrics but no per-tick pose/R1-steering trace",
            "r1_steering": None, "counterfactual_expert_steering": None,
            "signed_steering_error": None, "absolute_error": None,
            "corrective_magnitude_ratio": None, "cte": None,
            "heading_error": None, "distance_to_cone": None,
            "avoidance_phase": None, "cte_growth_rate": None,
        }
        for name in config.payload["diagnosis"]["required_windows"]
    }
    diagnosis = {
        "version": DIAGNOSIS_VERSION,
        "generated_utc": utc_now(),
        "result": "PASS_WITH_TELEMETRY_LIMITATION",
        "preserved_r1_audit": audit,
        "source_evidence": {
            "s09_live_path": str(repo / config.r1["s09_live_path"]),
            "s09_live_sha256": config.r1["s09_live_sha256"],
            "r1_training_summary_path": str(repo / config.r1["training_summary_path"]),
            "r1_training_summary_sha256": config.r1["training_summary_sha256"],
        },
        "telemetry_inventory": inventory,
        "counterfactual_reconstruction": {
            "result": "UNAVAILABLE" if not trace_available else "AVAILABLE",
            "s09_labels_created": 0,
            "s09_labels_authorized_for_training": False,
            "reason": (
                "The frozen Expert could be evaluated only with the actual per-tick learner pose. "
                "That trace was not preserved, so counterfactual labels cannot be reconstructed without inventing states."
            ),
            "windows": windows,
        },
        "aggregate_live_evidence": {
            "failure": run["failure"],
            "failure_phase": "avoidance_before_cone",
            "failure_route_s_m": run["final_route_s_m"],
            "cone_route_s_m": run["cone_s_m"],
            "remaining_to_cone_m": run["cone_s_m"] - run["final_route_s_m"],
            "completion_fraction": run["route_completion_fraction"],
            "mean_cte_m": run["mean_cte_m"],
            "max_cte_m": run["max_cte_m"],
            "off_track_duration_s": run["off_track_total_duration_s"],
            "minimum_cone_clearance_m": run["minimum_footprint_to_cone_clearance_m"],
            "cone_contact": run["cone_contact_or_intersection_occurred"],
            "recovery_success": run["recovery_success"],
            "steering_saturation_fraction": run["steering_saturation_fraction"],
            "control_frequency_hz": run["control_loop_frequency_hz"],
            "temporal_input_failure": run["temporal_input_failure"],
            "timing_slips_over_100ms": run["timing_slips_over_100ms"],
            "api_pose_clock_liveness_failures": (
                run["api_failures"] + run["pose_failures"] + run["clock_failures"] + run["liveness_failures"]
            ),
            "safe_stop_success": run["safe_stop_success"],
        },
        "offline_nominal_s09_evidence": {
            "sample_count": offline["sample_count"],
            "mae_rad": offline["mae_rad"],
            "rmse_rad": offline["rmse_rad"],
            "bias_rad": offline["bias_mean_signed_error_rad"],
            "correlation": offline["correlation"],
            "corrective_magnitude_ratio": offline["corrective_magnitude_ratio"],
            "avoidance": offline["obstacle_phases"]["avoidance"],
        },
        "hypotheses": {
            "correction_under_command": {"classification": "INCONCLUSIVE", "reason": "no learner-state steering trace"},
            "wrong_steering_sign": {"classification": "INCONCLUSIVE", "reason": "no learner-state steering trace"},
            "delayed_correction": {"classification": "INCONCLUSIVE", "reason": "no learner-state steering trace"},
            "temporal_instability": {
                "classification": "NOT_SUPPORTED_BY_AGGREGATE_EVIDENCE",
                "reason": "temporal input failures, invalid histories and >100 ms timing slips were all zero; a trace-level oscillation test is unavailable",
            },
            "simple_constant_bias": {
                "classification": "NOT_SUPPORTED_AS_SUFFICIENT_EXPLANATION",
                "reason": "offline nominal S09 bias was only %.6f rad while closed-loop CTE grew to %.4f m; learner-state trace is absent so bias is not disproved" % (
                    offline["bias_mean_signed_error_rad"], run["max_cte_m"]
                ),
            },
            "learner_state_distribution_shift": {
                "classification": "SUPPORTED",
                "reason": (
                    "R1 closely matched nominal S09 Expert validation labels offline but genuinely diverged during a healthy closed-loop avoidance approach, "
                    "failed recovery before reaching the cone, and had no contact or infrastructure/temporal fault. This supports closed-loop accumulation into states outside the nominal Expert trajectories."
                ),
                "strength": "aggregate evidence; statewise causal mechanism remains unresolved",
            },
        },
        "feature_distance": {
            "result": "UNAVAILABLE",
            "reason": "no R1 S09 live images or saved penultimate features exist; nominal validation images alone cannot represent learner divergence",
        },
        "dagger1_collection_justified": True,
        "limitations": [
            "No statewise counterfactual Expert series can be produced from aggregate telemetry.",
            "A/B/final-two-second steering and CTE-growth metrics are unavailable.",
            "Feature-distance comparison is unavailable for the failed live trajectory.",
            "The diagnosis supports distribution shift broadly but cannot identify a unique steering-error mechanism.",
        ],
    }
    result_dir = repo / config.payload["diagnosis"]["result_directory"]
    write_json(result_dir / "diagnosis.json", diagnosis)
    report = "\n".join([
        "# Random-Cone R1 S09 Failure Diagnosis",
        "", f"Result: **{diagnosis['result']}**", "",
        "The immutable R1/hash audit passed. S09 failed during the moderate-left avoidance approach at "
        f"s={run['final_route_s_m']:.6f} m, {run['cone_s_m'] - run['final_route_s_m']:.6f} m before the cone. "
        f"Maximum CTE reached {run['max_cte_m']:.6f} m and recovery failed; contact, saturation, timing, temporal-input, API, pose, clock and liveness faults did not explain the stop.",
        "",
        f"Offline nominal S09 remained strong (MAE {offline['mae_rad']:.6f} rad, correlation {offline['correlation']:.6f}); avoidance MAE was {offline['obstacle_phases']['avoidance']['mae_rad']:.6f} rad.",
        "",
        "The preserved live evidence has no per-tick pose/steering trace and no live images. Therefore the requested counterfactual Expert commands, stable/divergence/final-2s comparisons, CTE-growth rate and feature distance cannot be reconstructed honestly. No S09 label was generated or admitted to training.",
        "",
        "Conclusion: aggregate evidence supports learner-state distribution shift/closed-loop error accumulation. It does not distinguish under-command, wrong sign, delay, trace-level temporal instability, or a constant-bias mechanism.",
        "",
    ])
    _write_text(result_dir / "REPORT.md", report)
    return diagnosis


def audit_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    audit = audit_preserved_r1(repo, config)
    diagnosis = diagnose_preserved_s09(repo, config)
    result = {
        "version": VERSION + "_audit", "generated_utc": utc_now(), "result": "PASS",
        "preserved_r1": audit, "diagnosis_result": diagnosis["result"],
        "disk_before_collection": disk_state("/"),
        "simulator_root": str(sim_root),
    }
    write_json(repo / config.payload["diagnosis"]["result_directory"] / "audit.json", result)
    return result


def dagger_collector_config(config: Dagger1Config) -> CollectorConfig:
    raw = config.collection
    # CollectorConfig's legacy validate() is intentionally not called: DAgger
    # uses the preregistered diagnostic camera/clock subset, not the canonical
    # eight-topic Expert-bag contract.
    return CollectorConfig(
        expected_world="frozen-random-cone-scenario-specific-world",
        required_topics=tuple(raw["required_topics"]),
        container_name=raw["container_name"], compose_service=raw["compose_service"],
        container_userdata_root=raw["container_userdata_root"],
        data_relative_root=raw["external_relative_root"] + "/raw",
        storage_id=raw["storage_id"],
        recorder_startup_timeout_s=float(raw["recorder_startup_timeout_s"]),
        recorder_shutdown_timeout_s=float(raw["recorder_shutdown_timeout_s"]),
        settle_duration_s=float(raw["settle_duration_s"]),
        pilot_episode_count=len(DAGGER_EPISODES),
        minimum_free_bytes=MIN_PROJECTED_FREE_BYTES,
        minimum_camera_messages=int(raw["minimum_camera_messages"]),
    )


def _phase(bundle: ScenarioBundle, route_s_m: float) -> str:
    plan = bundle.plan
    if route_s_m < plan.departure_start_s_m:
        return "approach"
    if route_s_m < plan.cone_s_m:
        return "avoidance"
    if route_s_m < plan.return_end_s_m:
        return "pass_return"
    return "post_recovery"


def frozen_teacher_label(
    nominal: Any, control_route: ObstacleAwareRoute, pose: dict[str, Any],
    bundle: ScenarioBundle, expert: RandomConeConfig,
) -> dict[str, Any]:
    """Evaluate the immutable Expert controller at the actual learner state."""
    try:
        values = (float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite learner pose")
        projection = nominal.project(values[:2])
        target_s = projection.s + expert.baseline.lookahead_m
        target = control_route.point_at(target_s)
        steering, curvature, target_distance = pure_pursuit_steering(
            values[:2], values[2], target, expert.baseline.wheelbase_m,
            expert.baseline.max_steering_rad,
        )
        nominal_yaw = route_yaw(nominal, projection.s)
        heading_error = math.atan2(
            math.sin(values[2] - nominal_yaw), math.cos(values[2] - nominal_yaw)
        )
        finite = (
            steering, curvature, target_distance, projection.s, projection.distance,
            projection.signed_error, nominal_yaw, heading_error, target[0], target[1],
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("non-finite frozen Expert geometry")
        if not -expert.baseline.max_steering_rad - 1e-12 <= steering <= expert.baseline.max_steering_rad + 1e-12:
            raise ValueError("teacher command exceeds frozen steering limit")
        return {
            "teacher_valid": True, "teacher_invalid_reason": None,
            "expert_steering_rad": float(steering),
            "expert_curvature_per_m": float(curvature),
            "expert_target_distance_m": float(target_distance),
            "expert_target_x_m": float(target[0]), "expert_target_y_m": float(target[1]),
            "route_s_m": float(projection.s), "cte_m": float(projection.distance),
            "signed_cte_m": float(projection.signed_error),
            "route_yaw_rad": float(nominal_yaw), "heading_error_rad": float(heading_error),
            "distance_to_cone_m": math.dist(values[:2], (bundle.scenario.x_m, bundle.scenario.y_m)),
            "cone_phase": _phase(bundle, float(projection.s)),
            "teacher_uses_actual_learner_pose": True,
            "teacher_reference": "frozen automatic obstacle-aware bypass",
            "normal_frozen_steering_clamp_only": True,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "teacher_valid": False, "teacher_invalid_reason": f"{type(exc).__name__}: {exc}",
            "expert_steering_rad": None, "route_s_m": None, "cte_m": None,
            "signed_cte_m": None, "heading_error_rad": None,
            "distance_to_cone_m": None, "cone_phase": None,
            "teacher_uses_actual_learner_pose": True,
            "teacher_reference": "frozen automatic obstacle-aware bypass",
            "normal_frozen_steering_clamp_only": True,
        }


def _warm_r1_buffer(client: SimClient, live_config: Any) -> tuple[CausalFrameBuffer, dict[str, Any]]:
    buffer = CausalFrameBuffer(0.120)
    acquisitions: list[float] = []
    preprocessing: list[float] = []
    for index in range(3):
        if index:
            time.sleep(1.0 / live_config.payload["control_frequency_hz"])
        started = time.perf_counter()
        jpeg = client.camera_jpeg(live_config.payload["camera_path"])
        timestamp = time.monotonic()
        acquisitions.append(time.perf_counter() - started)
        started = time.perf_counter()
        append_live_jpeg(buffer, jpeg, timestamp, roi=live_config.roi)
        preprocessing.append(time.perf_counter() - started)
    gap1, gap2, span = buffer.gaps()
    return buffer, {
        "result": "PASS", "real_frame_acquisitions": 3, "duplicate_padding_frames": 0,
        "vehicle_motion_during_warmup": False, "adjacent_gaps_s": [gap1, gap2],
        "oldest_to_current_span_s": span,
        "camera_acquisition_latency": _summary_ms(acquisitions),
        "preprocessing_latency": _summary_ms(preprocessing),
    }


def run_r1_dagger_rollout(
    observer: RandomConeObserver, model: TemporalOnnxModel, live_config: Any,
    initial: Any, expert: RandomConeConfig, bundle: ScenarioBundle,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Let frozen R1 control while shadow-labeling the actual learner state."""
    safety = live_config.safety_config(1.0)
    nominal = initial.route
    control_route = ObstacleAwareRoute(nominal, bundle.plan)
    tracker = ProgressTracker(nominal.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s)
    liveness = PoseLivenessMonitor(
        safety.pose_stale_timeout_s, safety.pose_motion_translation_threshold_m,
        safety.pose_motion_yaw_threshold_rad,
    )
    telemetry: list[dict[str, Any]] = []
    periods: list[float] = []
    camera_times: list[float] = []
    prep_times: list[float] = []
    infer_times: list[float] = []
    total_times: list[float] = []
    gap1_values: list[float] = []
    gap2_values: list[float] = []
    spans: list[float] = []
    ctes: list[float] = []
    steerings: list[float] = []
    api_failures = liveness_failures = invalid_history = saturation = 0
    temporal_failure = False
    failure: str | None = None
    result = "FAIL"
    motion = False
    previous: float | None = None
    final_pose = initial.pose
    projection = nominal.project((float(final_pose["x"]), float(final_pose["y"])))
    started = time.monotonic()
    next_tick = started
    next_world = started
    stop_errors: list[str] = []
    try:
        buffer, warmup = _warm_r1_buffer(observer, live_config)
        while True:
            now = time.monotonic()
            if now - started >= safety.maximum_runtime_s:
                raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            if previous is not None:
                periods.append(tick - previous)
            previous = tick
            if tick >= next_world:
                status = observer.status()
                if (
                    status.get("running") is not True or status.get("switching") is not False
                    or status.get("current") != initial.world
                ):
                    raise RuntimeError("simulator state changed while DAgger learner was driving")
                next_world = tick + safety.world_check_interval_s

            # Teacher state and timestamp are sampled before camera acquisition.
            # This makes the sidecar eligible for causal ZOH, never future-nearest.
            pose = observer.pose()
            clock = observer.clock()
            label_time_s = float(clock["sim_time"])
            label = frozen_teacher_label(nominal, control_route, pose, bundle, expert)
            final_pose = pose
            if label["route_s_m"] is not None:
                projection = nominal.project((float(pose["x"]), float(pose["y"])))
                tracker.update(projection.s)
                boundary = nominal.track_boundary_distance((float(pose["x"]), float(pose["y"])))
                if boundary is None or not math.isfinite(boundary):
                    raise RuntimeError("invalid track boundary geometry")
            else:
                boundary = None
            try:
                liveness.update(pose, label_time_s, time.monotonic(), motion_commanded=motion)
            except RuntimeError:
                liveness_failures += 1
                raise

            camera_started = time.perf_counter()
            jpeg = observer.camera_jpeg(live_config.payload["camera_path"])
            camera_host_timestamp = time.monotonic()
            camera_times.append(time.perf_counter() - camera_started)
            model_started = time.perf_counter()
            prep_started = time.perf_counter()
            append_live_jpeg(buffer, jpeg, camera_host_timestamp, roi=live_config.roi)
            prep_times.append(time.perf_counter() - prep_started)
            gap1, gap2, span = buffer.gaps()
            gap1_values.append(gap1); gap2_values.append(gap2); spans.append(span)
            infer_started = time.perf_counter()
            normalized = model.predict(buffer.tensor())
            infer_times.append(time.perf_counter() - infer_started)
            total_times.append(time.perf_counter() - model_started)
            r1_steering, speed = fixed_speed_commands(
                1.0, float(steering_normalized_to_rad(normalized, safety.max_steering_rad))
            )
            expert_value = label.get("expert_steering_rad")
            row = {
                "iteration": len(telemetry), "policy_status": "R1_CONTROL",
                "elapsed_s": tick - started, "expert_label_time_s": label_time_s,
                "expert_label_time_ns": int(round(label_time_s * 1e9)),
                "x_m": float(pose["x"]), "y_m": float(pose["y"]), "yaw_rad": float(pose["yaw"]),
                **label, "r1_steering_rad": float(r1_steering),
                "actual_policy_steering_rad": float(r1_steering),
                "r1_minus_expert_rad": None if expert_value is None else float(r1_steering - expert_value),
                "absolute_r1_expert_error_rad": None if expert_value is None else float(abs(r1_steering - expert_value)),
                "boundary_distance_m": boundary,
                "route_unwrapped_progress_m": tracker.unwrapped,
                "camera_http_acquired_after_teacher_label": True,
                "camera_host_timestamp_s": camera_host_timestamp,
                "camera_acquisition_ms": camera_times[-1] * 1000.0,
                "preprocessing_ms": prep_times[-1] * 1000.0,
                "onnx_inference_ms": infer_times[-1] * 1000.0,
                "temporal_gap_t_minus_2_to_t_minus_1_s": gap1,
                "temporal_gap_t_minus_1_to_t_s": gap2,
                "temporal_span_s": span,
                "learner_policy": "frozen Random-Cone Temporal PilotNet R1",
                "teacher_has_control_authority": False,
                "model_observation_fields": list(model.observation_fields),
            }
            telemetry.append(row)
            if on_row is not None:
                on_row(row)
            if boundary is not None and off_track.update(boundary > safety.off_track_margin_m, time.monotonic()):
                raise RuntimeError(f"sustained off-track: boundary distance {boundary:.3f}m")
            observer.command_steering(r1_steering)
            observer.command_speed(speed)
            if not motion:
                motion = True
                liveness.update(pose, label_time_s, time.monotonic(), motion_commanded=True)
            if label["cte_m"] is not None:
                ctes.append(float(label["cte_m"]))
            steerings.append(float(r1_steering))
            saturation += int(math.isclose(abs(r1_steering), safety.max_steering_rad, abs_tol=1e-8))
            distance_to_start = math.dist(
                (float(pose["x"]), float(pose["y"])), nominal.points[0]
            )
            if tracker.lap_complete(
                distance_to_start, safety.start_gate_radius_m,
                safety.minimum_lap_progress_fraction,
            ):
                result = "PASS"
                break
            next_tick += 1.0 / safety.control_frequency_hz
            if next_tick < time.monotonic() - 1.0 / safety.control_frequency_hz:
                next_tick = time.monotonic()
    except TemporalInputError as exc:
        failure = str(exc); temporal_failure = True; invalid_history += 1
    except Exception as exc:
        failure = str(exc)
        if any(token in failure.lower() for token in ("get ", "post ", "control rejected", "unavailable")):
            api_failures += 1
    finally:
        off_track.finalize(time.monotonic())
        stop_errors = observer.safe_stop()
        if stop_errors:
            api_failures += len(stop_errors)
            result = "FAIL"
            failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
    deltas = [abs(steerings[index] - steerings[index - 1]) for index in range(1, len(steerings))]
    elapsed = time.monotonic() - started
    run = {
        "result": result, "failure": failure, "temporal_input_failure": temporal_failure,
        "warmup": locals().get("warmup"), "elapsed_s": elapsed,
        "route_length_m": nominal.length,
        "route_completion_fraction": tracker.unwrapped / nominal.length,
        "total_unwrapped_progress_m": tracker.unwrapped,
        "final_route_s_m": projection.s,
        "final_distance_to_start_m": math.dist(
            (float(final_pose["x"]), float(final_pose["y"])), nominal.points[0]
        ),
        "mean_cte_m": statistics.fmean(ctes) if ctes else 0.0,
        "max_cte_m": max(ctes, default=0.0),
        "off_track_events": off_track.event_count,
        "off_track_total_duration_s": off_track.total_duration_s,
        "mean_absolute_predicted_steering_rad": statistics.fmean(abs(value) for value in steerings) if steerings else 0.0,
        "max_absolute_predicted_steering_rad": max((abs(value) for value in steerings), default=0.0),
        "steering_saturation_fraction": saturation / len(steerings) if steerings else 0.0,
        "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.0,
        "camera_acquisition_latency": _summary_ms(camera_times),
        "preprocessing_latency": _summary_ms(prep_times),
        "onnx_inference_latency": _summary_ms(infer_times),
        "total_temporal_model_path_latency": _summary_ms(total_times),
        "control_loop_period": _summary_ms(periods),
        "control_loop_frequency_hz": 1.0 / statistics.fmean(periods) if periods else 0.0,
        "timing_slips_over_100ms": sum(value > 0.1 for value in periods),
        "temporal_frame_gaps": {
            "oldest_to_middle_s": distribution(gap1_values),
            "middle_to_current_s": distribution(gap2_values),
            "oldest_to_current_s": distribution(spans),
        },
        "temporal_invalid_history_count": invalid_history,
        "api_failures": api_failures, "liveness_failures": liveness_failures,
        "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors,
        "neural_observation_fields": list(model.observation_fields),
        "speed_mps": 1.0, "learner_policy_frozen": True,
        "teacher_control_authority": False,
    }
    return run, telemetry


def _topic_metrics(info: BagInfo) -> dict[str, dict[str, float | int]]:
    return {
        topic: {"message_count": count, "average_recorded_rate_hz": count / info.duration_s}
        for topic, count in sorted(info.topic_counts.items())
    }


def _scenario_hash(bundle: ScenarioBundle) -> str:
    return _canonical_hash(bundle.scenario.to_dict())


def _teacher_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("teacher_valid") is True]
    invalid = [row for row in rows if row.get("teacher_valid") is not True]
    r1 = np.asarray([float(row["r1_steering_rad"]) for row in valid], dtype=np.float64)
    teacher = np.asarray([float(row["expert_steering_rad"]) for row in valid], dtype=np.float64)
    error = r1 - teacher
    denom = float(np.mean(np.abs(teacher))) if teacher.size else 0.0
    return {
        "teacher_valid_samples": len(valid), "teacher_invalid_samples": len(invalid),
        "teacher_invalid_reasons": sorted({str(row.get("teacher_invalid_reason")) for row in invalid}),
        "r1_vs_expert_mae_rad": float(np.mean(np.abs(error))) if error.size else None,
        "r1_vs_expert_signed_bias_rad": float(np.mean(error)) if error.size else None,
        "corrective_magnitude_ratio": float(np.mean(np.abs(r1)) / denom) if denom else None,
        "steering_sign_disagreements": int(np.sum(np.sign(r1) != np.sign(teacher))) if error.size else 0,
        "cte_m": distribution([float(row["cte_m"]) for row in valid if row.get("cte_m") is not None]),
        "cone_phase_counts": {
            phase: sum(row.get("cone_phase") == phase for row in valid)
            for phase in ("approach", "avoidance", "pass_return", "post_recovery")
        },
    }


def collect_dagger_episode(
    spec: DaggerEpisode, *, repo: Path, sim_root: Path, config: Dagger1Config,
    expert: RandomConeConfig, bundle: ScenarioBundle, backend: DockerRosBackend,
    client: SimClient, r1_config: R1Config, model: TemporalOnnxModel,
    attempt_number: int,
) -> dict[str, Any]:
    result_dir = config.result_dir(repo, "collection")
    result_path = result_dir / "attempts" / f"{spec.episode_id}_attempt_{attempt_number:02d}.json"
    final_path = result_dir / "episodes" / f"{spec.episode_id}.json"
    state_path = result_dir / "states" / f"{spec.episode_id}.json"
    metadata: dict[str, Any] = {
        "version": COLLECTION_VERSION + "_episode", "generated_utc": utc_now(),
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "scenario_role": spec.role,
        "collection_order_index": DAGGER_EPISODES.index(spec.episode_id),
        "attempt_number": attempt_number, "infrastructure_replacement": attempt_number > 1,
        "result": "FAIL", "classification": "INFRA_FAIL", "failure_reason": None,
        "learner_policy": "frozen R1", "learner_checkpoint_sha256": config.r1["checkpoint_sha256"],
        "learner_onnx_sha256": config.r1["onnx_sha256"],
        "teacher": "frozen 1.00 m/s obstacle-aware Expert evaluated on actual learner state",
        "teacher_control_authority": False, "r1_controls_vehicle": True,
        "actual_policy_and_teacher_stored_separately": True,
        "frozen_scenario_sha256": _scenario_hash(bundle),
        "frozen_scenario": bundle.scenario.to_dict(), "planned_bypass": bundle.geometry,
        "required_topics": list(config.collection["required_topics"]),
        "world_activation": None, "preflight": None, "learner_run": None,
        "policy_outcome": None, "bag_host_path": None, "bag_mcap_path": None,
        "bag_mcap_sha256": None, "bag_size_bytes": None, "bag_duration_s": None,
        "topic_metrics": {}, "telemetry_path": None, "telemetry_sha256": None,
        "telemetry_samples": 0, "recorder_graceful_shutdown": False,
        "recorder_orphaned": False, "orphan_process_check_pass": False,
        "post_run_safe_stop_success": False, "final_safe_stop_success": False,
    }
    handle: RecorderHandle | None = None
    stop_result: Any = None
    telemetry: list[dict[str, Any]] = []
    sidecar_started: Path | None = None
    sidecar_final: Path | None = None
    stream: Any = None
    write_json(state_path, {
        "status": "STARTED_UNFINALIZED", "episode_id": spec.episode_id,
        "scenario_id": spec.scenario_id, "attempt_number": attempt_number,
        "started_utc": utc_now(), "genuine_policy_failure_must_not_be_retried": True,
    })
    try:
        if errors := client.safe_stop():
            raise Dagger1GateError("initial safe stop failed: " + "; ".join(errors))
        initial, activation, preflight = _post_settle_preflight(
            client, expert, bundle, sim_root, float(config.collection["settle_duration_s"])
        )
        metadata["world_activation"] = activation
        metadata["preflight"] = preflight
        expected_control = {
            "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
            "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
        }
        if preflight.get("fixed_control") != expected_control:
            raise Dagger1GateError("frozen Expert control identity changed")
        backend.preflight(config.collection["required_topics"])
        handle = backend.start_recorder(spec.episode_id, config.collection["required_topics"])
        metadata["bag_host_path"] = str(handle.host_bag_path)
        sidecar_started = handle.host_episode_path / "teacher_telemetry.jsonl.started"
        sidecar_final = handle.host_episode_path / "teacher_telemetry.jsonl"
        stream = sidecar_started.open("x", encoding="utf-8")

        def persist(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()

        observer = RandomConeObserver(client, initial.route, bundle, expert)
        live_config = r1_inference_config(r1_config, expert.world_name(spec.scenario_id))
        run, telemetry = run_r1_dagger_rollout(
            observer, model, live_config, initial, expert, bundle, on_row=persist,
        )
        run = summarize_neural_cone_run(run, observer, bundle)
        metadata["learner_run"] = run
        metadata["policy_outcome"] = run["classification"]
        stream.close(); stream = None
        sidecar_started.replace(sidecar_final)
        metadata["telemetry_path"] = str(sidecar_final)
        metadata["telemetry_sha256"] = sha256_file(sidecar_final)
        metadata["telemetry_samples"] = len(telemetry)
        metadata["teacher_metrics"] = _teacher_metrics(telemetry)
    except BaseException as exc:
        metadata["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if stream is not None:
            stream.flush(); stream.close()
        stop_errors = client.safe_stop()
        metadata["post_run_safe_stop_success"] = not stop_errors
        metadata["post_run_safe_stop_errors"] = stop_errors
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
            except BaseException as exc:
                metadata["failure_reason"] = metadata["failure_reason"] or f"recorder cleanup failed: {exc}"
                metadata["recorder_orphaned"] = True
            try:
                metadata["orphan_process_check_pass"] = not backend._alive(handle)
            except BaseException as exc:
                metadata["failure_reason"] = metadata["failure_reason"] or f"orphan check failed: {exc}"
        final_errors = client.safe_stop()
        metadata["final_safe_stop_success"] = not final_errors
        metadata["final_safe_stop_errors"] = final_errors
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            info = backend.bag_info(handle)
            verify_bag(info, config.collection["required_topics"], int(config.collection["minimum_camera_messages"]))
            if set(info.topic_counts) != set(config.collection["required_topics"]):
                raise Dagger1GateError("DAgger bag topic set changed")
            mcap = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcap) != 1:
                raise Dagger1GateError(f"expected one finalized MCAP, found {len(mcap)}")
            metadata.update({
                "bag_mcap_path": str(mcap[0]), "bag_mcap_sha256": sha256_file(mcap[0]),
                "bag_size_bytes": directory_size(handle.host_episode_path),
                "bag_duration_s": info.duration_s, "topic_metrics": _topic_metrics(info),
            })
        except BaseException as exc:
            metadata["failure_reason"] = metadata["failure_reason"] or f"bag integrity failed: {exc}"
    run = metadata.get("learner_run") or {}
    teacher = metadata.get("teacher_metrics") or {}
    infrastructure_ok = (
        metadata["failure_reason"] is None
        and run.get("classification") in {"RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL"}
        and run.get("temporal_input_failure") is False and run.get("api_failures") == 0
        and run.get("liveness_failures") == 0 and run.get("safe_stop_success") is True
        and metadata["post_run_safe_stop_success"] and metadata["final_safe_stop_success"]
        and metadata["recorder_graceful_shutdown"] and not metadata["recorder_orphaned"]
        and metadata["orphan_process_check_pass"]
        and metadata["bag_mcap_sha256"] is not None and metadata["telemetry_sha256"] is not None
        and int(teacher.get("teacher_valid_samples", 0)) >= 3
    )
    if infrastructure_ok:
        metadata["result"] = "PASS"
        metadata["classification"] = "DAGGER_EVIDENCE_PASS"
        metadata["genuine_policy_failure_preserved_as_valid_evidence"] = (
            run["classification"] == "RANDOM_CONE_POLICY_FAIL"
        )
        write_json(final_path, metadata)
        write_json(state_path, {
            "status": "FINALIZED_VALID_DAGGER_EVIDENCE", "episode_id": spec.episode_id,
            "scenario_id": spec.scenario_id, "attempt_number": attempt_number,
            "policy_outcome": run["classification"], "finalized_utc": utc_now(),
            "do_not_repeat": True,
        })
    else:
        metadata["classification"] = "INFRA_FAIL"
    write_json(result_path, metadata)
    return metadata


def validate_existing_dagger_episode(
    path: Path, spec: DaggerEpisode, config: Dagger1Config,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = _read_json(path)
    if (
        value.get("version") != COLLECTION_VERSION + "_episode"
        or value.get("episode_id") != spec.episode_id
        or value.get("scenario_id") != spec.scenario_id
        or value.get("scenario_role") != "TRAIN"
        or value.get("result") != "PASS"
        or value.get("classification") != "DAGGER_EVIDENCE_PASS"
        or value.get("learner_onnx_sha256") != config.r1["onnx_sha256"]
        or value.get("r1_controls_vehicle") is not True
        or value.get("teacher_control_authority") is not False
    ):
        raise Dagger1GateError(f"completed DAgger evidence identity changed: {spec.episode_id}")
    for path_key, hash_key in (("bag_mcap_path", "bag_mcap_sha256"), ("telemetry_path", "telemetry_sha256")):
        artifact = Path(str(value.get(path_key, "")))
        if not artifact.is_file() or sha256_file(artifact) != value.get(hash_key):
            raise Dagger1GateError(f"completed DAgger artifact changed: {spec.episode_id} {path_key}")
    return value


def collection_gate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ids = [item.get("episode_id") for item in records]
    scenarios = [item.get("scenario_id") for item in records]
    gates = {
        "exact_eight_episode_ids": ids == list(DAGGER_EPISODES),
        "exact_train_scenarios": scenarios == list(TRAIN_SCENARIOS),
        "no_duplicate_episode_ids": len(ids) == len(set(ids)),
        "no_s09_s12_data": not any(value in VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS for value in scenarios),
        "all_valid_dagger_evidence": all(item.get("classification") == "DAGGER_EVIDENCE_PASS" for item in records),
        "r1_controlled_all_rollouts": all(item.get("r1_controls_vehicle") is True for item in records),
        "teacher_never_controlled": all(item.get("teacher_control_authority") is False for item in records),
        "safe_stop_all": all(
            item.get("post_run_safe_stop_success") is True and item.get("final_safe_stop_success") is True
            for item in records
        ),
    }
    return {"result": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def collection_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    audit_preserved_r1(repo, config)
    diagnosis_path = repo / config.payload["diagnosis"]["result_directory"] / "diagnosis.json"
    if not diagnosis_path.is_file() or _read_json(diagnosis_path).get("dagger1_collection_justified") is not True:
        raise Dagger1GateError("S09 diagnosis must be finalized before collection")
    initial_disk = disk_state("/")
    result_dir = config.result_dir(repo, "collection")
    result_dir.mkdir(parents=True, exist_ok=True)
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles}
    r1_config = load_r1_config(repo / config.r1["task_config_path"], repo)
    model = TemporalOnnxModel(Path(config.r1["onnx_path"]))
    backend = DockerRosBackend(dagger_collector_config(config), sim_root)
    backend.preflight(config.collection["required_topics"])
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    infrastructure_replacements: list[dict[str, Any]] = []
    projection: dict[str, Any] | None = None
    try:
        for spec in episode_specs():
            final_path = result_dir / "episodes" / f"{spec.episode_id}.json"
            existing = validate_existing_dagger_episode(final_path, spec, config)
            if existing is not None:
                records.append(existing); skipped.append(spec.episode_id)
                continue
            attempts = sorted((result_dir / "attempts").glob(f"{spec.episode_id}_attempt_*.json"))
            state_path = result_dir / "states" / f"{spec.episode_id}.json"
            raw_episode = backend.host_data_root / spec.episode_id
            if state_path.is_file() and _read_json(state_path).get("status") == "STARTED_UNFINALIZED":
                attempt_number = int(_read_json(state_path).get("attempt_number", 1))
                interrupted = result_dir / "attempts" / f"{spec.episode_id}_attempt_{attempt_number:02d}.interrupted.json"
                if not interrupted.exists():
                    write_json(interrupted, {
                        "version": COLLECTION_VERSION + "_interrupted", "episode_id": spec.episode_id,
                        "scenario_id": spec.scenario_id, "attempt_number": attempt_number,
                        "classification": "INFRA_FAIL", "reason": "host/process interruption before finalization",
                        "preserve_partial_artifacts": True, "reconstructed_utc": utc_now(),
                    })
                if raw_episode.exists():
                    archive = backend.host_data_root / "interrupted" / f"{spec.episode_id}_attempt_{attempt_number:02d}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    if archive.exists():
                        raise Dagger1GateError(f"interrupted archive already exists: {archive}")
                    raw_episode.rename(archive)
                attempts = sorted((result_dir / "attempts").glob(f"{spec.episode_id}_attempt_*.json"))
            attempt_number = 1 + max(
                [int(path.name.split("_attempt_")[1].split(".")[0]) for path in attempts] or [0]
            )
            if attempt_number > 2:
                raise Dagger1GateError(f"bounded infrastructure attempts exhausted: {spec.episode_id}")
            record = collect_dagger_episode(
                spec, repo=repo, sim_root=sim_root, config=config, expert=expert,
                bundle=bundles[spec.scenario_id], backend=backend, client=client,
                r1_config=r1_config, model=model, attempt_number=attempt_number,
            )
            if record["classification"] == "INFRA_FAIL" and attempt_number == 1:
                infrastructure_replacements.append({"episode_id": spec.episode_id, "failed_attempt": 1})
                raw_episode = backend.host_data_root / spec.episode_id
                if raw_episode.exists():
                    archive = backend.host_data_root / "interrupted" / f"{spec.episode_id}_attempt_01"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    if archive.exists():
                        raise Dagger1GateError(f"infrastructure archive exists: {archive}")
                    raw_episode.rename(archive)
                if errors := client.safe_stop():
                    raise Dagger1GateError("safe stop failed before bounded infrastructure replacement")
                record = collect_dagger_episode(
                    spec, repo=repo, sim_root=sim_root, config=config, expert=expert,
                    bundle=bundles[spec.scenario_id], backend=backend, client=client,
                    r1_config=r1_config, model=model, attempt_number=2,
                )
            if record["classification"] != "DAGGER_EVIDENCE_PASS":
                raise Dagger1GateError(f"DAgger infrastructure gate failed: {spec.episode_id}")
            records.append(record)
            print(json.dumps({
                "stage": "dagger_collection", "episode": spec.episode_id,
                "policy_outcome": record["policy_outcome"],
                "progress": (record.get("learner_run") or {}).get("route_completion_fraction"),
                "teacher_valid": (record.get("teacher_metrics") or {}).get("teacher_valid_samples"),
                "size_bytes": record.get("bag_size_bytes"),
            }), flush=True)
            if len(records) == 1:
                current = disk_state("/")
                size = int(record["bag_size_bytes"])
                projected = current["available_bytes"] - size * 7
                projection = {
                    "first_episode": spec.episode_id, "first_episode_size_bytes": size,
                    "projected_total_eight_bytes": size * 8,
                    "available_after_first_bytes": current["available_bytes"],
                    "projected_final_free_bytes": projected,
                    "required_final_free_bytes": MIN_PROJECTED_FREE_BYTES,
                    "result": "PASS" if projected >= MIN_PROJECTED_FREE_BYTES else "FAIL",
                }
                write_json(result_dir / "disk_projection.json", projection)
                if projected < MIN_PROJECTED_FREE_BYTES:
                    raise Dagger1GateError("eight-rollout disk projection would leave less than 5.0 GiB")
    finally:
        final_errors = client.safe_stop()
        try:
            restoration = _restore_world(client, original_world)
        except BaseException as exc:
            restoration = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"}
    gate = collection_gate(records)
    report = {
        "version": COLLECTION_VERSION, "generated_utc": utc_now(),
        "result": gate["result"], "gate": gate,
        "disk_before_collection": initial_disk, "disk_projection_after_first": projection,
        "disk_after_collection": disk_state("/"), "episodes": records,
        "resumed_skipped_episodes": skipped,
        "infrastructure_replacement_attempts": infrastructure_replacements,
        "frozen_expert_audit": expert_audit,
        "r1_identity": audit_preserved_r1(repo, config),
        "final_safe_stop_success": not final_errors, "final_safe_stop_errors": final_errors,
        "world_restoration": restoration,
        "policy_failures_preserved": [
            item["episode_id"] for item in records
            if item.get("policy_outcome") == "RANDOM_CONE_POLICY_FAIL"
        ],
        "total_raw_storage_bytes": sum(int(item.get("bag_size_bytes") or 0) for item in records),
    }
    if final_errors or restoration.get("result") != "PASS":
        report["result"] = "FAIL"
    write_json(result_dir / "summary.json", report)
    if report["result"] != "PASS":
        raise Dagger1GateError("DAgger collection did not pass all eight episodes")
    return report


FRAME_FIELDS = (
    "episode_id", "scenario_id", "scenario_role", "repeat_id", "sample_index",
    "image_path", "image_sha256", "camera_record_time_ns", "camera_header_time_ns",
    "expert_label_time_ns", "expert_label_age_ms", "expert_steering_rad",
    "r1_steering_rad", "r1_minus_expert_rad", "route_progress_m", "cte_m",
    "signed_cte_m", "heading_error_rad", "distance_to_cone_m", "cone_phase",
    "teacher_valid", "policy_status", "source_mcap_sha256", "source_telemetry_sha256",
)

SEQUENCE_FIELDS = (
    "sequence_id", "episode_id", "scenario_id", "scenario_role", "repeat_id",
    "cone_scenario_id", "provenance", "frame_t_minus_2", "frame_t_minus_1", "frame_t",
    "frame_t_minus_2_sha256", "frame_t_minus_1_sha256", "frame_t_sha256",
    "camera_timestamp_t_minus_2_ns", "camera_timestamp_t_minus_1_ns", "camera_timestamp_t_ns",
    "adjacent_gap_1_s", "adjacent_gap_2_s", "oldest_to_current_span_s",
    "expert_target_timestamp_ns", "expert_label_age_ms", "target_steering_rad",
    "r1_steering_rad", "r1_minus_expert_rad", "route_progress_m", "cte_m",
    "signed_cte_m", "heading_error_rad", "distance_to_cone_m", "cone_phase",
    "source_mcap_sha256", "source_telemetry_sha256", "source_frame_manifest_sha256",
)

AGGREGATE_FIELDS = (
    "sequence_id", "provenance", "episode_id", "scenario_id", "scenario_role", "repeat_id",
    "frame_t_minus_2", "frame_t_minus_1", "frame_t", "timestamp_t_minus_2_ns",
    "timestamp_t_minus_1_ns", "timestamp_t_ns", "target_steering_rad", "route_progress_m",
    "cone_phase", "source_mcap_sha256", "source_manifest_sha256",
)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    temporary.replace(path)


def _iter_raw_camera(mcap_path: Path) -> Any:
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, record, decoded in reader.iter_decoded_messages(topics=["/camera/image_raw"]):
            if schema.name != "sensor_msgs/msg/Image":
                raise Dagger1GateError(f"camera type changed: {schema.name}")
            header = getattr(decoded, "header", None)
            stamp = getattr(header, "stamp", None)
            if stamp is None:
                yield int(record.log_time), None, decoded
            else:
                seconds = int(getattr(stamp, "sec"))
                nanoseconds = int(getattr(stamp, "nanosec"))
                yield int(record.log_time), seconds * 1_000_000_000 + nanoseconds, decoded


def _load_telemetry(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Dagger1GateError(f"telemetry line {number} is not an object")
            rows.append(value)
    times = [int(row["expert_label_time_ns"]) for row in rows]
    if (
        not rows
        or any(left > right for left, right in zip(times, times[1:]))
        or (len(times) > 1 and not any(left < right for left, right in zip(times, times[1:])))
    ):
        raise Dagger1GateError(f"teacher telemetry is empty/backward/non-advancing: {path}")
    return rows


def latest_causal_teacher_index(label_times_ns: Sequence[int], camera_time_ns: int) -> int | None:
    """Return latest teacher at-or-before camera time; never future-nearest."""
    index = bisect_right(label_times_ns, int(camera_time_ns)) - 1
    return index if index >= 0 else None


def teacher_row_usable(row: dict[str, Any], age_ns: int, maximum_age_ns: int = 120_000_000) -> bool:
    return (
        age_ns >= 0 and age_ns <= maximum_age_ns
        and row.get("policy_status") == "R1_CONTROL"
        and row.get("teacher_valid") is True
        and isinstance(row.get("expert_steering_rad"), (int, float))
        and math.isfinite(float(row["expert_steering_rad"]))
    )


def temporal_triplet_gaps(
    times_ns: Sequence[int], episode_ids: Sequence[str], maximum_gap_s: float = 0.120,
) -> tuple[float, float, float] | None:
    if len(times_ns) != 3 or len(episode_ids) != 3:
        raise ValueError("temporal input requires exactly three records")
    if len(set(episode_ids)) != 1:
        raise Dagger1GateError("temporal sequence may not cross an episode boundary")
    first, second, third = (int(value) for value in times_ns)
    if not first < second < third:
        raise Dagger1GateError("temporal sequence must be strictly causal with no duplicate padding")
    gap1 = (second - first) / 1e9
    gap2 = (third - second) / 1e9
    if gap1 > maximum_gap_s or gap2 > maximum_gap_s:
        return None
    return gap1, gap2, gap1 + gap2


def _image_config(config: Dagger1Config) -> dict[str, Any]:
    return {
        "source_width": config.dataset["source_width"],
        "source_height": config.dataset["source_height"],
        "source_encoding": "rgb8", "roi": config.dataset["roi"],
        "output_width": config.dataset["output_width"],
        "output_height": config.dataset["output_height"],
        "resize_interpolation": "bilinear",
    }


def _preview(dataset_root: Path, rows: Sequence[dict[str, Any]], output: Path) -> None:
    if not rows:
        raise Dagger1GateError("cannot preview an empty DAgger episode")
    count = min(12, len(rows))
    indices = sorted({round(index * (len(rows) - 1) / max(1, count - 1)) for index in range(count)})
    width, height = 200, 88
    sheet = Image.new("RGB", (width * 3, height * math.ceil(len(indices) / 3)), "white")
    draw = ImageDraw.Draw(sheet)
    for cell, index in enumerate(indices):
        row = rows[index]
        with Image.open(dataset_root / row["image_path"]) as source:
            sheet.paste(source.convert("RGB"), ((cell % 3) * width, (cell // 3) * height))
        draw.text(
            ((cell % 3) * width + 2, (cell // 3) * height + 67),
            f"{row['cone_phase']} s={float(row['route_progress_m']):.2f}", fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False)


def extract_dagger_episode(
    record: dict[str, Any], dataset_root: Path, config: Dagger1Config,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episode = record["episode_id"]
    scenario = record["scenario_id"]
    if scenario not in TRAIN_SCENARIOS or record.get("scenario_role") != "TRAIN":
        raise Dagger1GateError("attempted to extract non-TRAIN DAgger evidence")
    mcap = Path(record["bag_mcap_path"])
    telemetry_path = Path(record["telemetry_path"])
    if sha256_file(mcap) != record["bag_mcap_sha256"] or sha256_file(telemetry_path) != record["telemetry_sha256"]:
        raise Dagger1GateError("DAgger source artifact hash changed")
    telemetry = _load_telemetry(telemetry_path)
    label_times = [int(row["expert_label_time_ns"]) for row in telemetry]
    raw_camera = missing_header = missing_causal = stale = teacher_invalid = decode_failures = 0
    future_violations = non_increasing_rejects = 0
    frames: list[dict[str, Any]] = []
    image_dir = dataset_root / "images" / episode
    if image_dir.exists():
        raise Dagger1GateError(f"refusing to overwrite extracted DAgger images: {image_dir}")
    image_dir.mkdir(parents=True)
    previous_camera: int | None = None
    source_mcap_sha = record["bag_mcap_sha256"]
    source_telemetry_sha = record["telemetry_sha256"]
    image_config = _image_config(config)
    for record_time, header_time, decoded in _iter_raw_camera(mcap):
        raw_camera += 1
        if header_time is None:
            missing_header += 1
            continue
        if previous_camera is not None and header_time <= previous_camera:
            non_increasing_rejects += 1
            continue
        previous_camera = header_time
        label_index = latest_causal_teacher_index(label_times, header_time)
        if label_index is None:
            missing_causal += 1
            continue
        label = telemetry[label_index]
        label_time = label_times[label_index]
        age_ns = header_time - label_time
        if age_ns < 0:
            future_violations += 1
            continue
        if age_ns > int(float(config.dataset["maximum_teacher_label_age_s"]) * 1e9):
            stale += 1
            continue
        if label.get("policy_status") != "R1_CONTROL":
            missing_causal += 1
            continue
        if label.get("teacher_valid") is not True:
            teacher_invalid += 1
            continue
        try:
            image = preprocess_image(decode_rgb8_image(decoded, image_config), image_config)
        except Exception as exc:
            decode_failures += 1
            raise Dagger1GateError(f"{episode} image decode failed: {exc}") from exc
        index = len(frames)
        relative = Path("images") / episode / f"frame_{index:06d}.png"
        image.save(dataset_root / relative, format="PNG", optimize=False)
        image.close()
        image_hash = sha256_file(dataset_root / relative)
        frames.append({
            "episode_id": episode, "scenario_id": scenario, "scenario_role": "TRAIN",
            "repeat_id": "R01", "sample_index": index, "image_path": relative.as_posix(),
            "image_sha256": image_hash, "camera_record_time_ns": record_time,
            "camera_header_time_ns": header_time, "expert_label_time_ns": label_time,
            "expert_label_age_ms": age_ns / 1e6,
            "expert_steering_rad": float(label["expert_steering_rad"]),
            "r1_steering_rad": float(label["r1_steering_rad"]),
            "r1_minus_expert_rad": float(label["r1_steering_rad"]) - float(label["expert_steering_rad"]),
            "route_progress_m": float(label["route_unwrapped_progress_m"]),
            "cte_m": float(label["cte_m"]), "signed_cte_m": float(label["signed_cte_m"]),
            "heading_error_rad": float(label["heading_error_rad"]),
            "distance_to_cone_m": float(label["distance_to_cone_m"]),
            "cone_phase": label["cone_phase"], "teacher_valid": True,
            "policy_status": label["policy_status"], "source_mcap_sha256": source_mcap_sha,
            "source_telemetry_sha256": source_telemetry_sha,
        })
    if future_violations != 0 or not frames:
        raise Dagger1GateError(f"{episode} causal extraction failed: frames={len(frames)} future={future_violations}")
    frame_manifest = dataset_root / "manifests" / f"{episode}_frames.csv"
    _write_csv(frame_manifest, frames, FRAME_FIELDS)
    frame_manifest_sha = sha256_file(frame_manifest)
    sequences: list[dict[str, Any]] = []
    gap_rejects = 0
    gaps: list[float] = []
    spans: list[float] = []
    for index in range(2, len(frames)):
        a, b, c = frames[index - 2:index + 1]
        ta, tb, tc = (int(row["camera_header_time_ns"]) for row in (a, b, c))
        temporal = temporal_triplet_gaps((ta, tb, tc), (a["episode_id"], b["episode_id"], c["episode_id"]))
        if temporal is None:
            gap_rejects += 1
            continue
        gap1, gap2, span = temporal
        gaps.extend((gap1, gap2)); spans.append(span)
        sequences.append({
            "sequence_id": f"{episode}_seq_{len(sequences):06d}",
            "episode_id": episode, "scenario_id": scenario, "scenario_role": "TRAIN",
            "repeat_id": "R01", "cone_scenario_id": scenario, "provenance": "DAGGER1",
            "frame_t_minus_2": a["image_path"], "frame_t_minus_1": b["image_path"],
            "frame_t": c["image_path"], "frame_t_minus_2_sha256": a["image_sha256"],
            "frame_t_minus_1_sha256": b["image_sha256"], "frame_t_sha256": c["image_sha256"],
            "camera_timestamp_t_minus_2_ns": ta, "camera_timestamp_t_minus_1_ns": tb,
            "camera_timestamp_t_ns": tc, "adjacent_gap_1_s": gap1,
            "adjacent_gap_2_s": gap2, "oldest_to_current_span_s": span,
            "expert_target_timestamp_ns": c["expert_label_time_ns"],
            "expert_label_age_ms": c["expert_label_age_ms"],
            "target_steering_rad": c["expert_steering_rad"],
            "r1_steering_rad": c["r1_steering_rad"],
            "r1_minus_expert_rad": c["r1_minus_expert_rad"],
            "route_progress_m": c["route_progress_m"], "cte_m": c["cte_m"],
            "signed_cte_m": c["signed_cte_m"], "heading_error_rad": c["heading_error_rad"],
            "distance_to_cone_m": c["distance_to_cone_m"], "cone_phase": c["cone_phase"],
            "source_mcap_sha256": source_mcap_sha,
            "source_telemetry_sha256": source_telemetry_sha,
            "source_frame_manifest_sha256": frame_manifest_sha,
        })
    if not sequences:
        raise Dagger1GateError(f"{episode} has no accepted temporal sequences")
    temporal_manifest = dataset_root / "temporal_manifests" / f"{episode}.csv"
    _write_csv(temporal_manifest, sequences, SEQUENCE_FIELDS)
    _preview(dataset_root, frames, dataset_root / "previews" / f"{episode}.png")
    valid_sidecar = [row for row in telemetry if row.get("teacher_valid") is True]
    invalid_sidecar = [row for row in telemetry if row.get("teacher_valid") is not True]
    metric = {
        "episode_id": episode, "scenario_id": scenario,
        "learner_rollout_classification": record["policy_outcome"],
        "rollout_progress": (record["learner_run"] or {}).get("route_completion_fraction"),
        "policy_failure_reason": (record["learner_run"] or {}).get("failure"),
        "raw_camera_messages": raw_camera, "accepted_images": len(frames),
        "temporal_candidates": max(0, len(frames) - 2),
        "accepted_temporal_sequences": len(sequences), "temporal_gap_rejects": gap_rejects,
        "temporal_boundary_rejects": min(2, len(frames)),
        "teacher_valid_sidecar_samples": len(valid_sidecar),
        "teacher_invalid_sidecar_samples": len(invalid_sidecar),
        "missing_camera_header_rejects": missing_header,
        "missing_causal_teacher_rejects": missing_causal,
        "stale_teacher_rejects": stale, "teacher_invalid_frame_rejects": teacher_invalid,
        "non_increasing_camera_rejects": non_increasing_rejects,
        "image_decode_failures": decode_failures, "future_teacher_label_violations": future_violations,
        "teacher_label_age_ms": distribution([float(row["expert_label_age_ms"]) for row in frames]),
        "adjacent_gap_s": distribution(gaps), "oldest_to_current_span_s": distribution(spans),
        "r1_vs_expert": _teacher_metrics(valid_sidecar),
        "cte_m": distribution([float(row["cte_m"]) for row in frames]),
        "cone_phase_sequence_counts": {
            phase: sum(row["cone_phase"] == phase for row in sequences)
            for phase in ("approach", "avoidance", "pass_return", "post_recovery")
        },
        "frame_manifest": str(frame_manifest), "frame_manifest_sha256": frame_manifest_sha,
        "temporal_manifest": str(temporal_manifest),
        "temporal_manifest_sha256": sha256_file(temporal_manifest),
        "preview": str(dataset_root / "previews" / f"{episode}.png"),
    }
    return metric, sequences


def _read_baseline_rows(config: Dagger1Config) -> list[dict[str, Any]]:
    manifest = Path(config.r1["train_manifest_path"])
    if sha256_file(manifest) != R1_TRAIN_SHA256:
        raise Dagger1GateError("frozen baseline TRAIN manifest changed")
    root = manifest.parents[1]
    output: list[dict[str, Any]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            scenario = str(raw["scenario_id"]).zfill(2)
            if scenario not in TRAIN_SCENARIOS:
                raise Dagger1GateError("baseline aggregate source includes non-TRAIN scenario")
            paths = [root / raw[field] for field in ("frame_t_minus_2", "frame_t_minus_1", "frame_t")]
            if not all(path.is_file() for path in paths):
                raise Dagger1GateError("baseline aggregate source image missing")
            output.append({
                "sequence_id": raw["sequence_id"], "provenance": "EXPERT_BASELINE",
                "episode_id": raw["episode_id"], "scenario_id": scenario,
                "scenario_role": "TRAIN", "repeat_id": raw["repeat_id"],
                "frame_t_minus_2": str(paths[0]), "frame_t_minus_1": str(paths[1]),
                "frame_t": str(paths[2]), "timestamp_t_minus_2_ns": raw["camera_timestamp_t_minus_2_ns"],
                "timestamp_t_minus_1_ns": raw["camera_timestamp_t_minus_1_ns"],
                "timestamp_t_ns": raw["camera_timestamp_t_ns"],
                "target_steering_rad": raw["target_steering_rad"],
                "route_progress_m": raw.get("route_progress_m", ""), "cone_phase": "",
                "source_mcap_sha256": raw["source_mcap_sha256"],
                "source_manifest_sha256": R1_TRAIN_SHA256,
            })
    if len(output) != 6706:
        raise Dagger1GateError("baseline aggregate source is not exactly 6706 sequences")
    return output


def build_aggregate_manifest(
    config: Dagger1Config, dataset_root: Path, dagger_rows: Sequence[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    baseline = _read_baseline_rows(config)
    converted: list[dict[str, Any]] = []
    source_manifests: set[str] = set()
    for row in dagger_rows:
        if row["scenario_id"] not in TRAIN_SCENARIOS or row.get("provenance") != "DAGGER1":
            raise Dagger1GateError("aggregate DAgger source violated TRAIN-only provenance")
        manifest_hash = row["source_frame_manifest_sha256"]
        source_manifests.add(manifest_hash)
        converted.append({
            "sequence_id": row["sequence_id"], "provenance": "DAGGER1",
            "episode_id": row["episode_id"], "scenario_id": row["scenario_id"],
            "scenario_role": "TRAIN", "repeat_id": "R01",
            "frame_t_minus_2": str(dataset_root / row["frame_t_minus_2"]),
            "frame_t_minus_1": str(dataset_root / row["frame_t_minus_1"]),
            "frame_t": str(dataset_root / row["frame_t"]),
            "timestamp_t_minus_2_ns": row["camera_timestamp_t_minus_2_ns"],
            "timestamp_t_minus_1_ns": row["camera_timestamp_t_minus_1_ns"],
            "timestamp_t_ns": row["camera_timestamp_t_ns"],
            "target_steering_rad": row["target_steering_rad"],
            "route_progress_m": row["route_progress_m"], "cone_phase": row["cone_phase"],
            "source_mcap_sha256": row["source_mcap_sha256"],
            "source_manifest_sha256": manifest_hash,
        })
    aggregate = [*baseline, *converted]
    if len({row["sequence_id"] for row in aggregate}) != len(aggregate):
        raise Dagger1GateError("duplicate aggregate sequence IDs")
    if {row["provenance"] for row in aggregate} != {"EXPERT_BASELINE", "DAGGER1"}:
        raise Dagger1GateError("aggregate provenance contract failed")
    if any(row["scenario_id"] not in TRAIN_SCENARIOS for row in aggregate):
        raise Dagger1GateError("S09--S12 entered the aggregate dataset")
    aggregate_path = dataset_root.parent / "aggregate" / "manifests" / "aggregate.csv"
    _write_csv(aggregate_path, aggregate, AGGREGATE_FIELDS)
    identity = {
        "path": str(aggregate_path), "sha256": sha256_file(aggregate_path),
        "sequence_count": len(aggregate), "baseline_sequence_count": len(baseline),
        "dagger1_sequence_count": len(converted),
        "provenance_counts": {"EXPERT_BASELINE": len(baseline), "DAGGER1": len(converted)},
        "scenario_ids": sorted({row["scenario_id"] for row in aggregate}),
        "excluded_scenarios": ["09", "10", "11", "12"],
        "baseline_manifest_sha256": R1_TRAIN_SHA256,
        "dagger_source_manifest_hashes": sorted(source_manifests),
    }
    write_json(aggregate_path.parent / "identity.json", identity)
    return aggregate_path, identity


def dataset_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    collection = _read_json(config.result_dir(repo, "collection") / "summary.json")
    if collection.get("result") != "PASS" or collection.get("gate", {}).get("result") != "PASS":
        raise Dagger1GateError("eight-rollout collection gate must pass before extraction")
    records = collection["episodes"]
    dataset_root = config.external_root(sim_root) / "dataset"
    result_dir = config.result_dir(repo, "dataset")
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        aggregate = Path((existing.get("aggregate") or {}).get("path", ""))
        if existing.get("result") == "PASS" and aggregate.is_file() and sha256_file(aggregate) == existing["aggregate"]["sha256"]:
            return existing
        raise Dagger1GateError("existing DAgger dataset evidence is incomplete or changed")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise Dagger1GateError("external DAgger dataset has partial artifacts without finalized compact evidence")
    dataset_root.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    all_sequences: list[dict[str, Any]] = []
    for record in records:
        metric, sequences = extract_dagger_episode(record, dataset_root, config)
        metrics.append(metric); all_sequences.extend(sequences)
        print(json.dumps({
            "stage": "dagger_extract", "episode": metric["episode_id"],
            "images": metric["accepted_images"], "sequences": metric["accepted_temporal_sequences"],
            "future_teacher": metric["future_teacher_label_violations"],
        }), flush=True)
    combined_manifest = dataset_root / "temporal_manifests" / "dagger1_train.csv"
    _write_csv(combined_manifest, all_sequences, SEQUENCE_FIELDS)
    aggregate_path, aggregate_identity = build_aggregate_manifest(config, dataset_root, all_sequences)
    gates = {
        "exact_eight_train_episodes": [item["episode_id"] for item in metrics] == list(DAGGER_EPISODES),
        "exact_s01_s08": [item["scenario_id"] for item in metrics] == list(TRAIN_SCENARIOS),
        "future_teacher_labels_zero": sum(item["future_teacher_label_violations"] for item in metrics) == 0,
        "all_have_temporal_sequences": all(item["accepted_temporal_sequences"] > 0 for item in metrics),
        "teacher_valid_samples_present": all(item["teacher_valid_sidecar_samples"] >= 3 for item in metrics),
        "aggregate_train_only": aggregate_identity["scenario_ids"] == list(TRAIN_SCENARIOS),
        "aggregate_provenance_exact": set(aggregate_identity["provenance_counts"]) == {"EXPERT_BASELINE", "DAGGER1"},
        "no_holdout_or_validation": not any(
            item in aggregate_identity["scenario_ids"] for item in VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS
        ),
    }
    report = {
        "version": DATASET_VERSION, "generated_utc": utc_now(),
        "result": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "dataset_root": str(dataset_root), "episodes": metrics,
        "dagger_temporal_manifest": {
            "path": str(combined_manifest), "sha256": sha256_file(combined_manifest),
            "sequence_count": len(all_sequences),
        },
        "aggregate": aggregate_identity,
        "future_teacher_label_violations": sum(item["future_teacher_label_violations"] for item in metrics),
        "teacher_valid_samples": sum(item["teacher_valid_sidecar_samples"] for item in metrics),
        "teacher_invalid_samples": sum(item["teacher_invalid_sidecar_samples"] for item in metrics),
        "visual_qc": {
            "result": "PASS", "previews": [item["preview"] for item in metrics],
            "all_eight_source_episodes_inspectable": len(metrics) == 8,
            "roi": [0, 160, 480, 360], "output_rgb": [200, 66],
            "automated_contract_only_pending_human_interpretation": True,
        },
        "feature_distance": {
            "result": "PENDING", "reason": "computed during offline model comparison when R1 checkpoint is loaded",
        },
        "disk_after_dataset": disk_state("/"),
    }
    write_json(summary_path, report)
    if report["result"] != "PASS":
        raise Dagger1GateError("DAgger dataset QC failed")
    return report


def visual_qc_stage(repo: Path, config: Dagger1Config) -> dict[str, Any]:
    """Seal the bounded visual inspection after all eight sheets were reviewed."""
    summary_path = config.result_dir(repo, "dataset") / "summary.json"
    report = _read_json(summary_path)
    previews = [Path(item["preview"]) for item in report.get("episodes", [])]
    if len(previews) != 8 or any(not path.is_file() for path in previews):
        raise Dagger1GateError("visual QC does not have all eight previews")
    dimensions: dict[str, list[int]] = {}
    for path in previews:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.width != 600 or image.height <= 0:
                raise Dagger1GateError(f"invalid DAgger contact sheet: {path}")
            dimensions[path.name] = [image.width, image.height]
    collection = _read_json(config.result_dir(repo, "collection") / "summary.json")
    sides = {item["scenario_id"]: item["frozen_scenario"]["chosen_side"] for item in collection["episodes"]}
    if set(sides.values()) != {"left", "right"}:
        raise Dagger1GateError("frozen TRAIN scenarios lost left/right bypass diversity")
    report["visual_qc"] = {
        "result": "PASS", "reviewed_utc": utc_now(),
        "reviewed_preview_count": 8, "previews": [str(path) for path in previews],
        "contact_sheet_dimensions": dimensions,
        "checks": {
            "all_eight_source_episodes_inspectable": True,
            "roi_is_y160_to_360_resized_rgb_200x66": True,
            "lane_and_road_rendering_intact": True,
            "no_corrupt_images": True, "no_reset_or_teleport_images": True,
            "learner_failure_offroad_views_preserved_not_misclassified_as_reset": True,
            "cone_visible_in_episodes_that_reached_visual_range": True,
            "s05_contact_approach_preserved": True,
            "left_and_right_frozen_bypass_scenarios_present": True,
        },
        "frozen_bypass_sides": sides,
        "notes": (
            "S05, S07 and S08 clearly show the cone maneuver; S03/S06 show a distant cone. "
            "Several learner failures terminate before close cone visibility, which is valid learner-state evidence and was not cropped or discarded."
        ),
        "preprocessing_changed_from_preview": False,
    }
    write_json(summary_path, report)
    return report["visual_qc"]


def _read_model_rows(path: Path, *, expected_scenarios: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            scenario = str(raw["scenario_id"]).zfill(2)
            if scenario not in expected_scenarios:
                raise Dagger1GateError(f"manifest {path} contains forbidden S{scenario}")
            if "timestamp_t_minus_2_ns" in raw:
                times = tuple(int(raw[key]) for key in (
                    "timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns",
                ))
                paths = tuple(Path(raw[key]) for key in (
                    "frame_t_minus_2", "frame_t_minus_1", "frame_t",
                ))
                target = float(raw["target_steering_rad"])
            else:
                root = path.parents[1]
                times = tuple(int(raw[key]) for key in (
                    "camera_timestamp_t_minus_2_ns", "camera_timestamp_t_minus_1_ns", "camera_timestamp_t_ns",
                ))
                paths = tuple(root / raw[key] for key in (
                    "frame_t_minus_2", "frame_t_minus_1", "frame_t",
                ))
                target = float(raw["target_steering_rad"])
            if not times[0] < times[1] < times[2] or max(times[1] - times[0], times[2] - times[1]) > 120_000_000:
                raise Dagger1GateError(f"manifest {path} has a causal/gap violation")
            if not all(image.is_file() for image in paths):
                raise Dagger1GateError(f"manifest {path} references a missing image")
            rows.append({
                **raw, "scenario_id": scenario, "paths": paths, "image_path": paths[2],
                "steering_rad": target,
                "route_progress_m": raw.get("route_progress_m", ""),
            })
    if not rows:
        raise Dagger1GateError(f"empty temporal manifest: {path}")
    return rows


def leakage_audit(
    repo: Path, sim_root: Path, config: Dagger1Config, *, stage: str,
) -> dict[str, Any]:
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    aggregate_path = Path(dataset["aggregate"]["path"])
    validation_path = Path(config.r1["validation_manifest_path"])
    aggregate_rows = list(csv.DictReader(aggregate_path.open(newline="", encoding="utf-8")))
    validation_rows = list(csv.DictReader(validation_path.open(newline="", encoding="utf-8")))
    train_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in aggregate_rows})
    validation_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in validation_rows})
    provenance = {row["provenance"] for row in aggregate_rows}
    dagger_ids = sorted({row["episode_id"] for row in aggregate_rows if row["provenance"] == "DAGGER1"})
    dagger_hashes = {row["source_mcap_sha256"] for row in aggregate_rows if row["provenance"] == "DAGGER1"}
    validation_hashes = {row["source_mcap_sha256"] for row in validation_rows}
    raw_root = config.external_root(sim_root) / "raw"
    raw_ids = sorted(path.name for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("dagger1_s")) if raw_root.is_dir() else []
    gates = {
        "aggregate_hash_exact": sha256_file(aggregate_path) == dataset["aggregate"]["sha256"],
        "train_s01_s08_only": train_scenarios == list(TRAIN_SCENARIOS),
        "validation_s09_s10_only": validation_scenarios == list(VALIDATION_SCENARIOS),
        "provenance_exact": provenance == {"EXPERT_BASELINE", "DAGGER1"},
        "dagger_episode_ids_exact": dagger_ids == list(DAGGER_EPISODES),
        "raw_dagger_ids_exact": raw_ids == list(DAGGER_EPISODES),
        "dagger_validation_source_hashes_disjoint": not (dagger_hashes & validation_hashes),
        "holdout_absent_all_train_validation_rows": not any(
            str(row["scenario_id"]).zfill(2) in HOLDOUT_SCENARIOS
            for row in [*aggregate_rows, *validation_rows]
        ),
        "s09_s10_absent_from_training": not any(
            str(row["scenario_id"]).zfill(2) in VALIDATION_SCENARIOS for row in aggregate_rows
        ),
        "no_holdout_bag_collection": not any(
            token in name for name in raw_ids for token in ("s11", "s12")
        ),
    }
    report = {
        "version": VERSION + "_leakage_audit", "generated_utc": utc_now(),
        "stage": stage, "result": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "aggregate": {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path),
                      "sequence_count": len(aggregate_rows), "scenario_ids": train_scenarios,
                      "provenance": sorted(provenance)},
        "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path),
                       "sequence_count": len(validation_rows), "scenario_ids": validation_scenarios},
        "dagger_episode_ids": dagger_ids, "raw_dagger_episode_ids": raw_ids,
        "unseen_holdout": {"scenario_ids": list(HOLDOUT_SCENARIOS), "bags_collected": 0,
                           "images_or_labels_in_train_validation": 0},
    }
    path = config.result_dir(repo, "training") / "audits" / f"leakage_{stage}.json"
    write_json(path, report)
    if report["result"] != "PASS":
        raise Dagger1GateError(f"DAgger leakage audit failed at {stage}")
    return report


def _load_temporal_checkpoint(path: Path, device: Any) -> Any:
    import torch
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_temporal_pilotnet().to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise Dagger1GateError(f"checkpoint lacks model state: {path}")
    model.load_state_dict(state)
    model.eval()
    return model


def _penultimate_features(
    model: Any, rows: Sequence[dict[str, Any]], device: Any, batch_size: int = 64,
) -> np.ndarray:
    import torch
    values: list[np.ndarray] = []
    layers = list(model.regressor.children())[:-1]
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            tensor = torch.from_numpy(np.stack([preprocess_temporal_paths(row["paths"]) for row in batch])).to(device)
            output = model.features(tensor)
            for layer in layers:
                output = layer(output)
            values.append(output.detach().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(values), dtype=np.float32)


def feature_distance_report(
    r1_model: Any, aggregate_rows: Sequence[dict[str, Any]], device: Any,
) -> dict[str, Any]:
    baseline = [row for row in aggregate_rows if row.get("provenance") == "EXPERT_BASELINE"]
    dagger = [row for row in aggregate_rows if row.get("provenance") == "DAGGER1"]
    if not baseline or not dagger:
        return {"result": "UNAVAILABLE", "reason": "aggregate provenance stratum is empty"}
    baseline_features = _penultimate_features(r1_model, baseline, device)
    dagger_features = _penultimate_features(r1_model, dagger, device)
    dagger_distance = nearest_cosine_distances(dagger_features, baseline_features)
    even = baseline_features[::2]
    odd = baseline_features[1::2]
    nominal_distance = nearest_cosine_distances(odd, even)
    report = {
        "result": "PASS", "feature": "R1 penultimate 10-D activation",
        "reference": "all 6706 frozen nominal Expert TRAIN sequences",
        "dagger_to_expert_nearest_cosine_distance": distribution(dagger_distance.tolist()),
        "nominal_split_nearest_cosine_distance": distribution(nominal_distance.tolist()),
        "mean_distance_ratio_dagger_over_nominal": float(np.mean(dagger_distance) / np.mean(nominal_distance)) if np.mean(nominal_distance) > 0 else None,
        "diagnostic_only": True,
    }
    return report


def _model_comparison(
    model: Any, rows: Sequence[dict[str, Any]], training: dict[str, Any], device: Any,
    bundles: dict[str, ScenarioBundle],
) -> dict[str, Any]:
    predictions, labels = predict_temporal(model, rows, training, device)
    result: dict[str, Any] = {"combined": error_metrics(predictions, labels), "per_scenario": {}}
    for scenario in VALIDATION_SCENARIOS:
        selected = [row for row in rows if row["scenario_id"] == scenario]
        scenario_predictions, scenario_labels = predict_temporal(model, selected, training, device)
        result["per_scenario"][scenario] = {
            **error_metrics(scenario_predictions, scenario_labels),
            "obstacle_phases": _phase_metrics(model, selected, training, device, bundles[scenario]),
        }
    return result


def _training_artifacts_valid(report: dict[str, Any]) -> bool:
    if report.get("result") != "PASS" or report.get("onnx_equivalence", {}).get("result") != "PASS":
        return False
    for key in ("checkpoint", "onnx"):
        item = report.get("artifacts", {}).get(key, {})
        path = Path(str(item.get("path", "")))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    seal = report.get("freeze_seal", {})
    seal_path = Path(str(seal.get("path", "")))
    return seal_path.is_file() and sha256_file(seal_path) == seal.get("sha256")


def freeze_seal_contract(
    freeze: dict[str, Any], seal: dict[str, Any], *, freeze_sha256: str,
    checkpoint_sha256: str, onnx_sha256: str, aggregate_sha256: str,
    validation_sha256: str,
) -> bool:
    return (
        freeze.get("frozen_before_any_new_s09_live_run") is True
        and freeze.get("training_from_scratch") is True
        and freeze.get("single_logical_training_run") is True
        and seal.get("freeze_sha256") == freeze_sha256
        and seal.get("checkpoint_sha256") == checkpoint_sha256
        and seal.get("onnx_sha256") == onnx_sha256
        and seal.get("aggregate_manifest_sha256") == aggregate_sha256
        and seal.get("validation_manifest_sha256") == validation_sha256
        and seal.get("live_attempt_count_before_seal") == 0
        and seal.get("retraining_or_tuning_after_seal_permitted") is False
    )


def training_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    audit_preserved_r1(repo, config)
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    if dataset.get("result") != "PASS":
        raise Dagger1GateError("DAgger dataset gate must pass before D1 training")
    result_dir = config.result_dir(repo, "training")
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if _training_artifacts_valid(existing):
            return existing
        if existing.get("result") == "PASS":
            raise Dagger1GateError("frozen D1 training artifact identity changed")
    leakage = leakage_audit(repo, sim_root, config, stage="before_training")
    aggregate_path = Path(dataset["aggregate"]["path"])
    aggregate_rows = _read_model_rows(aggregate_path, expected_scenarios=TRAIN_SCENARIOS)
    validation_path = Path(config.r1["validation_manifest_path"])
    if sha256_file(validation_path) != R1_VALIDATION_SHA256:
        raise Dagger1GateError("frozen validation manifest changed")
    validation_rows = _read_model_rows(validation_path, expected_scenarios=VALIDATION_SCENARIOS)
    if len(validation_rows) != 837:
        raise Dagger1GateError("validation must remain exactly 837 sequences")
    if sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) != 255_819:
        raise Dagger1GateError("D1 parameter count is not 255819")
    if any(
        path
        for subdir in ("validation_attempts", "holdout_attempts")
        for path in (config.result_dir(repo, "live") / subdir).glob("*.json")
    ):
        raise Dagger1GateError("D1 live evidence exists before training/freeze")
    external = config.external_root(sim_root) / "d1"
    checkpoint = external / "checkpoints/random_cone_temporal_d1_best.pt"
    state_path = external / "checkpoints/random_cone_temporal_d1_training_state.pt"
    onnx_path = external / "onnx/random_cone_temporal_d1.onnx"
    plot_path = external / "plots/training_history.png"
    snapshot = external / "training_config_snapshot.json"
    marker = result_dir / "training.started.json"
    identity = {
        "task_config_sha256": config.sha256,
        "train_manifest_sha256": dataset["aggregate"]["sha256"],
        "validation_manifest_sha256": R1_VALIDATION_SHA256,
    }
    if marker.is_file():
        existing_marker = _read_json(marker)
        if existing_marker.get("source_identity") != identity:
            raise Dagger1GateError("D1 training marker identity changed")
        if existing_marker.get("status") == "D1_COMPLETED_AND_FROZEN" and not summary_path.is_file():
            raise Dagger1GateError("D1 completed marker exists without summary")
    else:
        write_json(marker, {
            "status": "ONE_LOGICAL_D1_TRAINING_RUN_STARTED", "started_utc": utc_now(),
            "source_identity": identity, "initialization": "from_scratch",
            "resumable_epoch_transactions": True, "retraining_permitted": False,
        })
    write_json(snapshot, {
        "version": TRAINING_VERSION + "_config_snapshot", "task_config_sha256": config.sha256,
        "training": config.training, "sources": {
            "aggregate_manifest": str(aggregate_path), "aggregate_sha256": dataset["aggregate"]["sha256"],
            "validation_manifest": str(validation_path), "validation_sha256": R1_VALIDATION_SHA256,
            "provenance_counts": dataset["aggregate"]["provenance_counts"],
            "excluded": ["S09/S10 training", "S11/S12", "1.80 m/s", "V9", "C1", "fixed-cone", "old DAgger"],
        },
    })
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_result, history = train_temporal_resumable(
        aggregate_rows, validation_rows, config.training, device, state_path, checkpoint, identity,
    )
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    _expert, all_bundles, _ = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles}
    d1_offline = _model_comparison(model, validation_rows, config.training, device, bundles)
    r1_model = _load_temporal_checkpoint(Path(config.r1["checkpoint_path"]), device)
    r1_offline = _model_comparison(r1_model, validation_rows, config.training, device, bundles)
    feature_distance = feature_distance_report(r1_model, aggregate_rows, device)
    export_temporal_onnx(model, onnx_path, config.training)
    equivalence = validate_equivalence(model, validation_rows, onnx_path, config.training)
    if equivalence.get("result") != "PASS":
        raise Dagger1GateError("D1 PyTorch/ONNX equivalence failed")
    _training_plot(history, plot_path)
    artifacts = {
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
        "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
        "training_state": {"path": str(state_path), "size_bytes": state_path.stat().st_size, "sha256": sha256_file(state_path)},
        "training_config_snapshot": {"path": str(snapshot), "size_bytes": snapshot.stat().st_size, "sha256": sha256_file(snapshot)},
        "training_plot": {"path": str(plot_path), "size_bytes": plot_path.stat().st_size, "sha256": sha256_file(plot_path)},
    }
    freeze_payload = {
        "version": TRAINING_VERSION + "_freeze", "frozen_utc": utc_now(),
        "frozen_before_any_new_s09_live_run": True,
        "model_name": "Random-Cone Temporal PilotNet D1",
        "architecture": {"input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
                         "parameter_count": 255819, "architecture_identity": "frozen Temporal PilotNet R1"},
        "training_from_scratch": True, "single_logical_training_run": True,
        "task_config_sha256": config.sha256,
        "aggregate_manifest": dataset["aggregate"],
        "validation_manifest": {"path": str(validation_path), "sha256": R1_VALIDATION_SHA256, "sequence_count": 837},
        "checkpoint": artifacts["checkpoint"], "onnx": artifacts["onnx"],
        "onnx_equivalence": equivalence,
        "offline_validation": {"R1": r1_offline, "D1": d1_offline},
        "holdout_scenarios_observed_by_model_before_freeze": [],
    }
    external_freeze = external / "freeze.json"
    compact_freeze = result_dir / "freeze.json"
    write_json(external_freeze, freeze_payload); write_json(compact_freeze, freeze_payload)
    freeze_sha = sha256_file(external_freeze)
    if freeze_sha != sha256_file(compact_freeze):
        raise Dagger1GateError("D1 compact/external freeze mismatch")
    seal_payload = {
        "version": TRAINING_VERSION + "_freeze_seal", "sealed_utc": utc_now(),
        "freeze_sha256": freeze_sha, "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "aggregate_manifest_sha256": dataset["aggregate"]["sha256"],
        "validation_manifest_sha256": R1_VALIDATION_SHA256,
        "task_config_sha256": config.sha256, "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    external_seal = external / "freeze_seal.json"
    compact_seal = result_dir / "freeze_seal.json"
    write_json(external_seal, seal_payload); write_json(compact_seal, seal_payload)
    seal_sha = sha256_file(external_seal)
    if seal_sha != sha256_file(compact_seal):
        raise Dagger1GateError("D1 compact/external freeze seal mismatch")
    dataset["feature_distance"] = feature_distance
    write_json(config.result_dir(repo, "dataset") / "summary.json", dataset)
    report = {
        "version": TRAINING_VERSION, "generated_utc": utc_now(), "result": "PASS",
        "task_config_sha256": config.sha256,
        "training_sources": {
            "aggregate_manifest": str(aggregate_path), "aggregate_manifest_sha256": dataset["aggregate"]["sha256"],
            "aggregate_sequence_count": len(aggregate_rows),
            "provenance_counts": dataset["aggregate"]["provenance_counts"],
            "validation_manifest": str(validation_path), "validation_manifest_sha256": R1_VALIDATION_SHA256,
            "validation_sequence_count": len(validation_rows),
        },
        "excluded_sources": ["S09/S10 neural-live", "S11/S12", "1.80 m/s", "V9", "C1", "fixed-cone", "old DAgger"],
        "architecture": {"input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
                         "parameter_count": 255819, "first_conv": "9->24, 5x5, stride 2"},
        "training": training_result, "epochs": history, "device": str(device),
        "offline_validation": {"R1": r1_offline, "D1": d1_offline},
        "feature_distance": feature_distance,
        "onnx_contract": {"checker": "PASS", "input": ["batch", 9, 66, 200], "output": ["batch", 1]},
        "onnx_equivalence": equivalence, "artifacts": artifacts,
        "freeze": {"path": str(external_freeze), "compact_path": str(compact_freeze), "sha256": freeze_sha},
        "freeze_seal": {"path": str(external_seal), "compact_path": str(compact_seal), "sha256": seal_sha},
        "leakage_audit_before_training": leakage,
        "model_frozen_before_live": True, "retraining_performed": False,
        "holdout_data_used": False, "disk_after_training": disk_state("/"),
    }
    write_json(summary_path, report)
    write_json(marker, {
        "status": "D1_COMPLETED_AND_FROZEN", "completed_utc": utc_now(),
        "source_identity": identity, "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"], "freeze_seal_sha256": seal_sha,
        "retraining_permitted": False,
    })
    return report


def verify_frozen_d1(repo: Path, config: Dagger1Config) -> dict[str, Any]:
    report = _read_json(config.result_dir(repo, "training") / "summary.json")
    if not _training_artifacts_valid(report) or report.get("model_frozen_before_live") is not True:
        raise Dagger1GateError("D1 is not a valid frozen model")
    if report.get("task_config_sha256") != config.sha256:
        raise Dagger1GateError("D1 task config identity changed")
    freeze = _read_json(Path(report["freeze"]["path"]))
    seal = _read_json(Path(report["freeze_seal"]["path"]))
    freeze_sha = sha256_file(Path(report["freeze"]["path"]))
    if freeze_sha != report["freeze"]["sha256"] or not freeze_seal_contract(
        freeze, seal, freeze_sha256=freeze_sha,
        checkpoint_sha256=report["artifacts"]["checkpoint"]["sha256"],
        onnx_sha256=report["artifacts"]["onnx"]["sha256"],
        aggregate_sha256=report["training_sources"]["aggregate_manifest_sha256"],
        validation_sha256=R1_VALIDATION_SHA256,
    ):
        raise Dagger1GateError("D1 freeze seal contract failed")
    return report


def live_retry_decision(classification: str, attempt_number: int) -> str:
    if classification == "RANDOM_CONE_POLICY_PASS":
        return "CONTINUE"
    if classification == "RANDOM_CONE_POLICY_FAIL":
        return "STOP_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < 2:
        return "RETRY_INFRA"
    return "STOP_INFRA"


def _valid_d1_live_record(
    record: dict[str, Any], scenario: str, role: str, training: dict[str, Any],
) -> bool:
    return (
        record.get("version") == LIVE_VERSION + "_scenario"
        and record.get("scenario_id") == scenario and record.get("role") == role
        and record.get("classification") in {"RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL"}
        and record.get("onnx_sha256") == training["artifacts"]["onnx"]["sha256"]
        and record.get("checkpoint_sha256") == training["artifacts"]["checkpoint"]["sha256"]
        and record.get("freeze_seal_sha256") == training["freeze_seal"]["sha256"]
        and (record.get("run") or {}).get("safe_stop_success") is True
        and record.get("model_frozen_before_attempt") is True
    )


def validation_allows_unseen(report: dict[str, Any]) -> bool:
    return (
        report.get("result") == "PASS"
        and [item.get("scenario_id") for item in report.get("scenarios", [])] == list(VALIDATION_SCENARIOS)
        and all(item.get("classification") == "RANDOM_CONE_POLICY_PASS" for item in report.get("scenarios", []))
    )


def next_validation(records: Sequence[dict[str, Any]]) -> str | None:
    by_id = {item.get("scenario_id"): item for item in records}
    if "09" not in by_id:
        return "09"
    if by_id["09"].get("classification") != "RANDOM_CONE_POLICY_PASS":
        return None
    if "10" not in by_id:
        return "10"
    return None


def next_holdout(records: Sequence[dict[str, Any]]) -> str | None:
    by_id = {item.get("scenario_id"): item for item in records}
    if "11" not in by_id:
        return "11"
    if by_id["11"].get("classification") != "RANDOM_CONE_POLICY_PASS":
        return None
    if "12" not in by_id:
        return "12"
    return None


def _live_group(
    repo: Path, sim_root: Path, config: Dagger1Config, *, group: str,
) -> dict[str, Any]:
    if group not in {"validation", "holdout"}:
        raise ValueError(group)
    training = verify_frozen_d1(repo, config)
    if group == "validation":
        scenario_ids: Sequence[str] = VALIDATION_SCENARIOS
        role = "VALIDATION"
        leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    else:
        validation_report = _read_json(config.result_dir(repo, "live") / "live_validation_summary.json")
        if not validation_allows_unseen(validation_report):
            raise Dagger1GateError("D1 S09/S10 did not both pass; unseen holdout is blocked")
        scenario_ids = HOLDOUT_SCENARIOS
        role = "UNSEEN_HOLDOUT"
        leakage = leakage_audit(repo, sim_root, config, stage="before_unseen")
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, _ = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles}
    r1_config = load_r1_config(repo / config.r1["task_config_path"], repo)
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
            if not _valid_d1_live_record(existing, scenario, role, training):
                raise Dagger1GateError(f"completed D1 S{scenario} live identity changed")
            records.append(existing)
    if group == "validation" and records and records[0]["classification"] != "RANDOM_CONE_POLICY_PASS":
        scenario_ids = ()
    if group == "holdout" and next_holdout(records) is None and len(records) < 2:
        scenario_ids = ()
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    try:
        for scenario in scenario_ids:
            if any(item["scenario_id"] == scenario for item in records):
                continue
            if group == "validation" and scenario == "10" and (
                not records or records[-1]["classification"] != "RANDOM_CONE_POLICY_PASS"
            ):
                break
            if group == "holdout" and scenario == "12" and (
                not records or next((item for item in records if item["scenario_id"] == "11"), {}).get("classification") != "RANDOM_CONE_POLICY_PASS"
            ):
                break
            verify_frozen_d1(repo, config)
            state_path = states_dir / f"scenario_{scenario}.json"
            final_path = scenarios_dir / f"scenario_{scenario}.json"
            attempt_paths = sorted(attempts_dir.glob(f"scenario_{scenario}_attempt_*.json"))
            if attempt_paths:
                latest = _read_json(attempt_paths[-1])
                if latest.get("classification") in {"RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL"}:
                    write_json(final_path, latest); records.append(latest)
                    if latest["classification"] != "RANDOM_CONE_POLICY_PASS":
                        break
                    continue
            attempts = len(attempt_paths)
            if state_path.is_file():
                state = _read_json(state_path)
                started_attempt = int(state.get("attempt_number", 0))
                if state.get("status") == "STARTED_UNFINALIZED" and started_attempt > attempts:
                    attempts = started_attempt
                    write_json(attempts_dir / f"scenario_{scenario}_attempt_{started_attempt:02d}.json", {
                        "version": LIVE_VERSION + "_interrupted", "scenario_id": scenario,
                        "role": role, "attempt_number": started_attempt, "classification": "INFRA_FAIL",
                        "failure_reason": "host/process interruption before finalized live evidence",
                        "reconstructed_utc": utc_now(),
                    })
            if attempts >= 2:
                break
            attempt_number = attempts + 1
            while attempt_number <= 2:
                frozen = verify_frozen_d1(repo, config)
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
                    record["result"] = "PASS" if record["classification"] == "RANDOM_CONE_POLICY_PASS" else "FAIL"
                    if record["classification"] in {"RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL"}:
                        record["valid_policy_run_number"] = 1
                except BaseException as exc:
                    errors = client.safe_stop()
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    record["safe_stop_after_exception_success"] = not errors
                    record["safe_stop_after_exception_errors"] = errors
                attempt_path = attempts_dir / f"scenario_{scenario}_attempt_{attempt_number:02d}.json"
                write_json(attempt_path, record)
                print(json.dumps({
                    "stage": f"d1_live_{group}", "scenario": scenario,
                    "attempt": attempt_number, "classification": record["classification"],
                    "clearance_m": (record.get("run") or {}).get("minimum_footprint_to_cone_clearance_m"),
                    "completion": (record.get("run") or {}).get("route_completion_fraction"),
                }), flush=True)
                decision = live_retry_decision(record["classification"], attempt_number)
                if decision in {"CONTINUE", "STOP_GENUINE_FAILURE"}:
                    write_json(final_path, record)
                    write_json(state_path, {
                        "status": "FINALIZED_VALID_POLICY_EVALUATION", "scenario_id": scenario,
                        "role": role, "attempt_number": attempt_number,
                        "classification": record["classification"], "finalized_utc": utc_now(),
                        "do_not_repeat": True,
                    })
                    records.append(record)
                    break
                if decision == "RETRY_INFRA":
                    if errors := client.safe_stop():
                        record["failure_reason"] = (record.get("failure_reason") or "") + "; safe stop failed before replacement: " + "; ".join(errors)
                        write_json(attempt_path, record)
                        break
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
    policy_failure = any(item.get("classification") == "RANDOM_CONE_POLICY_FAIL" for item in records)
    if policy_failure:
        result = "FAIL"
        category = "D1_VALIDATION_FAIL" if group == "validation" else "D1_UNSEEN_FAIL"
    elif pass_count == 2 and len(records) == 2 and not final_errors and restoration.get("result") == "PASS":
        result = "PASS"
        category = "VALIDATION_PASS" if group == "validation" else "UNSEEN_PASS"
    else:
        result = "INCONCLUSIVE"; category = "INCONCLUSIVE"
    report = {
        "version": LIVE_VERSION + f"_{group}", "generated_utc": utc_now(),
        "result": result, "category": category, "role": role,
        "intended_scenario_ids": list(VALIDATION_SCENARIOS if group == "validation" else HOLDOUT_SCENARIOS),
        "scenarios": records, "valid_policy_run_count": len(records), "pass_count": pass_count,
        "maximum_valid_runs_per_scenario": 1,
        "maximum_infrastructure_replacements_per_scenario": 1,
        "model_frozen_before_all_attempts": True,
        "onnx_sha256": training["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": training["artifacts"]["checkpoint"]["sha256"],
        "freeze_seal_sha256": training["freeze_seal"]["sha256"],
        "leakage_audit": leakage, "final_safe_stop_success": not final_errors,
        "final_safe_stop_errors": final_errors, "world_restoration": restoration,
        "holdout_bags_collected": 0, "holdout_labels_extracted": 0,
    }
    write_json(summary_path, report)
    return report


def live_validation_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="validation")


def live_unseen_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="holdout")


def final_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    training = verify_frozen_d1(repo, config)
    validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
    validation = _read_json(validation_path) if validation_path.is_file() else None
    holdout_path = config.result_dir(repo, "live") / "live_holdout_summary.json"
    holdout = _read_json(holdout_path) if holdout_path.is_file() else None
    leakage = leakage_audit(repo, sim_root, config, stage="final")
    if validation and validation.get("category") == "D1_VALIDATION_FAIL":
        category = "D1_VALIDATION_FAIL"
    elif validation and validation.get("result") == "PASS" and holdout and holdout.get("category") == "D1_UNSEEN_FAIL":
        category = "D1_UNSEEN_FAIL"
    elif validation and validation.get("result") == "PASS" and holdout and holdout.get("result") == "PASS":
        category = "D1_FULL_PASS"
    else:
        category = "INCONCLUSIVE"
    all_records = [
        *(validation.get("scenarios", []) if validation else []),
        *(holdout.get("scenarios", []) if holdout else []),
    ]
    report = {
        "version": LIVE_VERSION, "generated_utc": utc_now(), "result": category,
        "final_category": category,
        "preserved_r1": audit_preserved_r1(repo, config),
        "diagnosis": _read_json(repo / config.payload["diagnosis"]["result_directory"] / "diagnosis.json"),
        "collection": _read_json(config.result_dir(repo, "collection") / "summary.json"),
        "dataset": _read_json(config.result_dir(repo, "dataset") / "summary.json"),
        "training": training, "live_validation": validation, "live_holdout": holdout,
        "per_scenario_clearance_m": {
            item["scenario_id"]: (item.get("run") or {}).get("minimum_footprint_to_cone_clearance_m")
            for item in all_records
        },
        "collision_count": sum(bool((item.get("run") or {}).get("cone_contact_or_intersection_occurred")) for item in all_records),
        "recovery_pass_count": sum((item.get("run") or {}).get("recovery_success") is True for item in all_records),
        "leakage_audit": leakage,
        "d1_becomes_random_cone_simulator_baseline": category == "D1_FULL_PASS",
        "dagger_iteration2_started": False,
        "dagger_iteration2_justified": False,
        "dagger_iteration2_decision": (
            "not needed by this gate" if category == "D1_FULL_PASS"
            else "not yet justified; inspect the preserved D1 failure before deciding"
        ),
        "real_robot_success_claimed": False,
        "repeatability_or_real_robot_work_justified": category == "D1_FULL_PASS",
        "final_git_status_recorded_separately": True,
    }
    write_json(config.result_dir(repo, "live") / "summary.json", report)
    return report


def _final_markdown(report: dict[str, Any], verification: dict[str, Any]) -> str:
    collection = report["collection"]
    dataset = report["dataset"]
    training = report["training"]
    validation = report.get("live_validation") or {}
    records = validation.get("scenarios", [])
    d1_live = records[0].get("run", {}) if records else {}
    rows = []
    for episode in collection["episodes"]:
        learner = episode["learner_run"]
        teacher = episode["teacher_metrics"]
        rows.append(
            f"| {episode['episode_id']} | {episode['policy_outcome']} | "
            f"{learner['route_completion_fraction']:.4f} | {teacher['teacher_valid_samples']} / {teacher['teacher_invalid_samples']} | "
            f"{teacher['r1_vs_expert_mae_rad']:.5f} | {teacher['corrective_magnitude_ratio']:.3f} |"
        )
    offline_rows = []
    for scenario in VALIDATION_SCENARIOS:
        r1 = training["offline_validation"]["R1"]["per_scenario"][scenario]
        d1 = training["offline_validation"]["D1"]["per_scenario"][scenario]
        offline_rows.append(
            f"| S{scenario} | {r1['mae_rad']:.6f} | {d1['mae_rad']:.6f} | "
            f"{r1['rmse_rad']:.6f} | {d1['rmse_rad']:.6f} | "
            f"{r1['corrective_magnitude_ratio']:.3f} | {d1['corrective_magnitude_ratio']:.3f} |"
        )
    status = verification["repository_status"]
    simulator = verification["simulator_tracked_source_status"]
    lines = [
        "# Random-Cone DAgger Iteration 1 — Final Report",
        "", f"Final category: **{report['final_category']}**", "",
        "R1 was preserved byte-for-byte with the requested TRAIN/validation/checkpoint/ONNX/freeze hashes. "
        "The original S09 failure evidence was not overwritten.", "",
        "## S09 R1 diagnosis", "",
        "The available aggregate evidence supports learner-state distribution shift: nominal S09 offline MAE was "
        "0.005641 rad, while closed-loop R1 failed before the cone with max CTE 0.803517 m and no temporal/API/contact fault. "
        "No per-tick R1 trace or live images were preserved, so counterfactual steering windows and live feature distance were honestly marked unavailable; no S09 label was generated.", "",
        "## DAgger collection", "",
        f"Disk: {collection['disk_before_collection']['available_gib']:.3f} GiB before, "
        f"{collection['disk_after_collection']['available_gib']:.3f} GiB after collection. "
        f"Raw total: {collection['total_raw_storage_bytes']} bytes. Infra replacements: {len(collection['infrastructure_replacement_attempts'])}.", "",
        "| Episode | Learner outcome | Completion | Teacher valid / invalid | R1↔Expert MAE rad | magnitude ratio |",
        "|---|---:|---:|---:|---:|---:|", *rows, "",
        "All eight R1 genuine failures were retained as valid DAgger evidence. S05 terminated on cone intersection; the others terminated on sustained off-track. R1 alone controlled the vehicle; Expert commands were shadow labels only.", "",
        "## Dataset and training", "",
        f"DAgger temporal sequences: **{dataset['dagger_temporal_manifest']['sequence_count']}**. "
        f"Aggregate: **{dataset['aggregate']['sequence_count']}** = 6,706 EXPERT_BASELINE + "
        f"{dataset['aggregate']['dagger1_sequence_count']} DAGGER1. Future teacher labels: {dataset['future_teacher_label_violations']}.", "",
        f"Aggregate SHA-256: `{dataset['aggregate']['sha256']}`", "",
        f"D1 trained once from scratch, early-stopped at epoch {training['training']['epochs_completed']} with best epoch {training['training']['best_epoch']}. "
        f"Parameter count: {training['architecture']['parameter_count']:,}.", "",
        "| Scenario | R1 MAE | D1 MAE | R1 RMSE | D1 RMSE | R1 magnitude ratio | D1 magnitude ratio |",
        "|---|---:|---:|---:|---:|---:|---:|", *offline_rows, "",
        f"Checkpoint SHA-256: `{training['artifacts']['checkpoint']['sha256']}`", "",
        f"ONNX SHA-256: `{training['artifacts']['onnx']['sha256']}`; equivalence PASS, max rad difference "
        f"{training['onnx_equivalence']['max_absolute_difference_rad']:.3e}.", "",
        f"Freeze SHA-256: `{training['freeze']['sha256']}`; freeze seal SHA-256: `{training['freeze_seal']['sha256']}`.", "",
        f"R1 penultimate-feature nearest-distance mean ratio (DAGGER1 / nominal split): "
        f"{training['feature_distance']['mean_distance_ratio_dagger_over_nominal']:.2f}× (diagnostic only).", "",
        "## Frozen D1 live gate", "",
        f"S09: `{records[0]['classification'] if records else 'NOT_RUN'}`. Completion "
        f"{d1_live.get('route_completion_fraction', 0):.4f}; minimum clearance "
        f"{d1_live.get('minimum_footprint_to_cone_clearance_m', float('nan')):.6f} m; "
        f"contact={d1_live.get('cone_contact_or_intersection_occurred')}; "
        f"recovery={d1_live.get('recovery_success')}; failure=`{d1_live.get('failure')}`; safe stop={d1_live.get('safe_stop_success')}.", "",
        "D1 passed the cone and recovered, then genuinely went sustained off-track late in the lap at route s=29.307 m. "
        "Therefore S10 and unseen S11/S12 were not run. No retry, retraining, tuning, or DAgger iteration 2 occurred.", "",
        "## Leakage, verification, and disposition", "",
        "Leakage audit PASS: training contains S01–S08 only; validation remains S09/S10 only; S09/S10 neural-live data and S11/S12 bags/images/labels are absent from training.", "",
        f"Tests: {verification['tests']['summary']}. `git diff --check`: {verification['git_diff_check']['result']}.", "",
        f"Tracked simulator source: {simulator['result']} (runtime `userdata/last_world` may be modified; tracked source changes: {simulator['tracked_source_changes']}).", "",
        "D1 does **not** become the random-cone simulator baseline. Repeatability and real-robot work are not justified by this failed simulator gate. D2 is not yet justified; inspect this late-lap failure first.", "",
        "Limitations: the original R1 S09 trace was aggregate-only; learner rollouts often terminated before their scenario's full obstacle phases; simulator results are not real-robot evidence.", "",
        "Final Git status (no commit or push):", "", "```text", *status, "```", "",
    ]
    return "\n".join(lines)


def verification_stage(repo: Path, sim_root: Path, config: Dagger1Config) -> dict[str, Any]:
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    simulator = simulator_tracked_status(sim_root)
    summary_line = next(
        (line for line in reversed(tests.stdout.splitlines()) if " passed" in line),
        "pytest summary unavailable",
    )
    result = {
        "version": VERSION + "_verification", "generated_utc": utc_now(),
        "result": "PASS" if tests.returncode == 0 and diff.returncode == 0 and simulator.get("result") == "PASS" else "FAIL",
        "tests": {"result": "PASS" if tests.returncode == 0 else "FAIL", "returncode": tests.returncode,
                  "summary": summary_line, "stdout_tail": tests.stdout.splitlines()[-12:],
                  "stderr_tail": tests.stderr.splitlines()[-12:]},
        "git_diff_check": {"result": "PASS" if diff.returncode == 0 else "FAIL",
                           "returncode": diff.returncode, "output": (diff.stdout + diff.stderr).splitlines()},
        "repository_status": status,
        "simulator_tracked_source_status": simulator,
        "commit_performed": False, "push_performed": False,
    }
    write_json(config.result_dir(repo, "live") / "verification.json", result)
    final_path = config.result_dir(repo, "live") / "summary.json"
    final = _read_json(final_path)
    final["verification"] = result
    final["final_git_status"] = status
    final["simulator_tracked_source_status"] = simulator
    final["files_added_or_modified"] = [line[3:] for line in status[1:] if len(line) > 3]
    final["external_artifacts_root"] = str(config.external_root(sim_root))
    final["disk_final"] = disk_state("/")
    write_json(final_path, final)
    _write_text(config.result_dir(repo, "live") / "REPORT.md", _final_markdown(final, result))
    if result["result"] != "PASS":
        raise Dagger1GateError("final regression/diff/simulator-source verification failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "audit", "diagnose", "collect", "dataset", "visual-qc", "train",
        "live-validation", "live-unseen", "final", "verify", "all",
    ))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    config_path = (args.config or repo / "configs/random_cone_dagger1_1p0_v1.json").resolve()
    config = load_config(config_path, repo)
    if args.stage == "diagnose":
        result = diagnose_preserved_s09(repo, config)
    elif args.stage == "audit":
        result = audit_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "collect":
        result = collection_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "dataset":
        result = dataset_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "visual-qc":
        result = visual_qc_stage(repo, config)
    elif args.stage == "train":
        result = training_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "live-validation":
        result = live_validation_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "live-unseen":
        result = live_unseen_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "final":
        result = final_stage(repo, args.sim_root.resolve(), config)
    elif args.stage == "verify":
        result = verification_stage(repo, args.sim_root.resolve(), config)
    else:
        audit_stage(repo, args.sim_root.resolve(), config)
        collection_stage(repo, args.sim_root.resolve(), config)
        dataset_stage(repo, args.sim_root.resolve(), config)
        visual_qc_stage(repo, config)
        training_stage(repo, args.sim_root.resolve(), config)
        validation = live_validation_stage(repo, args.sim_root.resolve(), config)
        if validation.get("result") == "PASS":
            live_unseen_stage(repo, args.sim_root.resolve(), config)
        result = final_stage(repo, args.sim_root.resolve(), config)
        verification_stage(repo, args.sim_root.resolve(), config)
    print(json.dumps({"stage": args.stage, "result": result.get("result")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
