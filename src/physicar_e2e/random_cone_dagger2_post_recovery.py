"""Targeted Random-Cone DAgger iteration 2: post-recovery learner states only.

The frozen D1 policy drives one TRAIN rollout in each frozen S01--S08
scenario.  Full-resolution camera responses exist only in memory.  After the
existing recovery observer passes, the most recent three causal frames are
materialized as canonical 200x66 RGB PNGs and the frozen Expert labels the
actual D1 state.  D2 is one deterministic scratch training run over the exact
D1 aggregate plus this single new provenance stratum.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from .cone_avoidance_expert import ObstacleAwareRoute
from .dataset_extractor import canonical_json_bytes, preprocess_image, sha256_file
from .expert_driver import PoseLivenessMonitor
from .high_speed_temporal import (
    TemporalOnnxModel,
    distribution,
    export_temporal_onnx,
    metrics as error_metrics,
    predict_temporal,
    validate_equivalence,
)
from .pilotnet import steering_normalized_to_rad
from .pilotnet_inference import _summary_ms, fixed_speed_commands
from .pilotnet_temporal import (
    CausalFrameBuffer,
    TEMPORAL_PARAMETER_COUNT,
    TemporalInputError,
    append_live_jpeg,
    build_temporal_pilotnet,
)
from .random_cone_dagger1 import (
    Dagger1Config,
    _load_temporal_checkpoint,
    _phase,
    _read_model_rows,
    _scenario_hash,
    _warm_r1_buffer,
    frozen_teacher_label,
    load_config as load_dagger1_config,
    verify_frozen_d1,
)
from .random_cone_expert import (
    MAP_FAMILY,
    ROLE_IDS,
    RandomConeConfig,
    RandomConeObserver,
    ScenarioBundle,
    _restore_world,
    simulator_tracked_status,
)
from .random_cone_temporal_r1 import (
    R1Config,
    inference_config as r1_inference_config,
    load_config as load_r1_config,
    run_live_once,
    summarize_neural_cone_run,
    train_temporal_resumable,
)
from .random_cone_train_data import (
    _post_settle_preflight,
    audit_frozen_expert,
    disk_state,
    load_task_config as load_train_task_config,
)
from .route_geometry import OffTrackMonitor, ProgressTracker
from .sim_client import SimClient


VERSION = "random_cone_dagger2_post_recovery_1p0_v1"
COLLECTION_VERSION = "random_cone_dagger2_collection_1p0_v1"
DATASET_VERSION = "random_cone_dagger2_dataset_1p0_v1"
TRAINING_VERSION = "pilotnet_training_d2_random_cone_1p0"
LIVE_VERSION = "pilotnet_e2e_d2_random_cone_1p0"
EXPECTED_BRANCH = "experiment/random-cone-dagger2-post-recovery-1p0-v1"
TRAIN_SCENARIOS = tuple(f"{value:02d}" for value in range(1, 9))
VALIDATION_SCENARIOS = ("09", "10")
HOLDOUT_SCENARIOS = ("11", "12")
DAGGER2_EPISODES = tuple(f"dagger2_s{value}_r01" for value in TRAIN_SCENARIOS)
ROUTE_END_M = 30.50461070080936
ROUTE_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 26.0), (26.0, ROUTE_END_M))
PROVENANCE = "DAGGER2_POST_RECOVERY"
MINIMUM_BEFORE_BYTES = 11 * 1024**3 // 2
MINIMUM_PROJECTED_BYTES = 5 * 1024**3
VALID_POLICY_CLASSIFICATIONS = ("RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL")
INFRASTRUCTURE_INTERRUPTION = "INFRASTRUCTURE_INTERRUPTION"
HOST_CRASH = "HOST_CRASH"

D2_FULL_PASS = "D2_FULL_PASS"
D2_VALIDATION_FAIL = "D2_VALIDATION_FAIL"
D2_UNSEEN_FAIL = "D2_UNSEEN_FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


class Dagger2GateError(RuntimeError):
    """A frozen identity, collection, coverage, training, or live gate failed."""


@dataclass(frozen=True)
class Dagger2Episode:
    episode_id: str
    scenario_id: str
    repeat_id: str = "R01"
    role: str = "TRAIN"


@dataclass(frozen=True)
class Dagger2Config:
    path: Path
    payload: dict[str, Any]
    dagger1: Dagger1Config

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def inputs(self) -> dict[str, Any]:
        return self.payload["preserved_inputs"]

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
        return sim_root / "userdata" / self.payload["external_relative_root"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Dagger2GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Dagger2GateError(f"JSON root is not an object: {path}")
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    temporary.replace(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _hash_gate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise Dagger2GateError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise Dagger2GateError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.rstrip()


def episode_specs() -> tuple[Dagger2Episode, ...]:
    return tuple(Dagger2Episode(f"dagger2_s{scenario}_r01", scenario) for scenario in TRAIN_SCENARIOS)


def load_config(path: Path, repo: Path) -> Dagger2Config:
    payload = _read_json(path)
    required = {
        "version", "expected_branch", "map_family", "preserved_inputs", "scenario_roles",
        "collection", "dataset", "coverage_gate", "disk_gate", "training", "offline",
        "external_relative_root", "result_directories", "live", "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise Dagger2GateError("DAgger2 config fields/version changed")
    if (
        payload["expected_branch"] != EXPECTED_BRANCH
        or payload["map_family"] != MAP_FAMILY
        or payload["scenario_roles"] != ROLE_IDS
        or tuple(payload["collection"]["episode_order"]) != DAGGER2_EPISODES
    ):
        raise Dagger2GateError("branch/map/scenario/episode contract changed")
    collection = payload["collection"]
    collection_contract = (
        collection["learner"], collection["teacher"], collection["teacher_control_authority"],
        float(collection["speed_mps"]), float(collection["control_frequency_hz"]),
        float(collection["steering_limit_rad"]), float(collection["wheelbase_m"]),
        float(collection["lookahead_m"]), int(collection["maximum_infrastructure_replacements_per_episode"]),
        collection["retry_genuine_policy_failure"], collection["start_persistence_after_recovery_pass"],
        collection["preserve_two_history_frames_at_boundary"], collection["record_rosbag"],
        collection["persist_full_resolution_camera"],
    )
    if collection_contract != (
        "frozen_D1", "frozen_1p0_expert_actual_learner_state", False,
        1.0, 15.0, 0.349066, 0.18, 0.9, 1, False, True, True, False, False,
    ):
        raise Dagger2GateError("targeted collection/control contract changed")
    dataset = payload["dataset"]
    dataset_contract = (
        dataset["source_width"], dataset["source_height"], dataset["source_transport"],
        dataset["roi"], dataset["output_width"], dataset["output_height"],
        dataset["stored_color_space"], dataset["stored_format"], dataset["resize_interpolation"],
        dataset["history_frames"], float(dataset["maximum_adjacent_gap_s"]),
        float(dataset["maximum_teacher_label_age_s"]), dataset["causal_teacher_zoh"],
        dataset["future_teacher_labels_required"], dataset["allow_episode_boundary_crossing"],
        dataset["allow_duplicate_padding"], dataset["target_recovery_state"], dataset["provenance"],
    )
    if dataset_contract != (
        480, 360, "HTTP_JPEG_IN_MEMORY_ONLY", {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360},
        200, 66, "RGB", "PNG", "bilinear", 3, 0.12, 0.12, True, 0, False, False,
        "PASS", PROVENANCE,
    ):
        raise Dagger2GateError("derived image/temporal/teacher contract changed")
    if tuple(tuple(float(v) for v in item) for item in payload["offline"]["route_bins_m"]) != ROUTE_BINS:
        raise Dagger2GateError("fixed route bins changed")
    if (
        int(payload["disk_gate"]["minimum_before_bytes"]) != MINIMUM_BEFORE_BYTES
        or int(payload["disk_gate"]["minimum_projected_final_bytes"]) != MINIMUM_PROJECTED_BYTES
    ):
        raise Dagger2GateError("disk gates changed")
    if payload["training"] != {
        "seed": 20260824, "image_width": 200, "image_height": 66,
        "input_channels": 9, "history_frames": 3, "maximum_adjacent_gap_s": 0.12,
        "max_steering_rad": 0.349066, "target": "frozen_expert_steering_normalized_at_t",
        "optimizer": "Adam", "loss": "MSE", "learning_rate": 0.001,
        "batch_size": 64, "max_epochs": 35, "early_stopping_patience": 7,
        "minimum_improvement": 0.000001, "initialization": "from_scratch",
        "augmentation": False, "sample_weighting": False, "scenario_weighting": False,
        "oversampling": False, "undersampling": False, "hyperparameter_sweep": False,
        "onnx_opset": 17, "onnx_equivalence_samples": 128,
        "onnx_mean_abs_difference_limit": 0.00001,
        "onnx_max_abs_difference_limit": 0.0001,
    }:
        raise Dagger2GateError("D2 scratch training contract changed")
    permissions = payload["permissions"]
    if (
        permissions["d1_artifact_changes_permitted"] is not False
        or tuple(permissions["dagger2_collection_scenarios"]) != TRAIN_SCENARIOS
        or permissions["s09_s12_dagger_data_permitted"] is not False
        or permissions["new_expert_nominal_data_permitted"] is not False
        or permissions["d2_logical_training_runs_permitted"] != 1
        or permissions["retraining_after_freeze_permitted"] is not False
        or permissions["d3_permitted"] is not False
        or permissions["holdout_access_before_validation_pass_permitted"] is not False
        or permissions["commit_permitted"] is not False
        or permissions["push_permitted"] is not False
    ):
        raise Dagger2GateError("permission boundary changed")
    dagger1_ref = payload["preserved_inputs"]["dagger1_config"]
    dagger1_path = _resolve(repo, dagger1_ref["path"])
    _hash_gate(dagger1_path, dagger1_ref["sha256"], "DAgger1 config")
    return Dagger2Config(path.resolve(), payload, load_dagger1_config(dagger1_path, repo))


def disk_gate(config: Dagger2Config, *, minimum_bytes: int = MINIMUM_BEFORE_BYTES) -> dict[str, Any]:
    report = disk_state(config.payload["disk_gate"]["path"])
    report["required_available_bytes"] = int(minimum_bytes)
    report["required_available_gib"] = minimum_bytes / 1024**3
    report["result"] = "PASS" if int(report["available_bytes"]) >= minimum_bytes else "FAIL"
    report["df_h"] = subprocess.run(
        ["df", "-h", config.payload["disk_gate"]["path"]],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if report["result"] != "PASS":
        raise Dagger2GateError(
            f"disk gate failed: {report['available_bytes'] / 1024**3:.3f} GiB available; "
            f"{minimum_bytes / 1024**3:.3f} GiB required"
        )
    return report


def audit_preserved_inputs(config: Dagger2Config, repo: Path) -> dict[str, Any]:
    branch = _git(repo, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise Dagger2GateError(f"expected branch {EXPECTED_BRANCH!r}, active branch is {branch!r}")
    hashes: dict[str, Any] = {}
    simple = (
        "expert_train_manifest", "dagger1_aggregate_manifest", "validation_manifest",
        "dagger1_collection_summary", "dagger1_dataset_summary",
        "cone_free_recheck_summary", "cone_free_valid_result",
    )
    for key in simple:
        item = config.inputs[key]
        hashes[key] = _hash_gate(_resolve(repo, item["path"]), item["sha256"], key.replace("_", " "))
    for model_name in ("r1", "d1"):
        model = config.inputs[model_name]
        hashes[model_name] = {}
        for artifact in ("checkpoint", "onnx", "freeze", "freeze_seal"):
            hashes[model_name][artifact] = _hash_gate(
                _resolve(repo, model[f"{artifact}_path"]),
                model[f"{artifact}_sha256"],
                f"{model_name.upper()} {artifact}",
            )
    hashes["d1"]["training_summary"] = _hash_gate(
        _resolve(repo, config.inputs["d1"]["training_summary_path"]),
        config.inputs["d1"]["training_summary_sha256"],
        "D1 training summary",
    )
    d1_report = verify_frozen_d1(repo, config.dagger1)
    aggregate = config.inputs["dagger1_aggregate_manifest"]
    if (
        d1_report["training_sources"]["aggregate_sequence_count"] != 8189
        or d1_report["training_sources"]["provenance_counts"]
        != {"EXPERT_BASELINE": 6706, "DAGGER1": 1483}
        or d1_report["artifacts"]["checkpoint"]["sha256"] != config.inputs["d1"]["checkpoint_sha256"]
        or d1_report["artifacts"]["onnx"]["sha256"] != config.inputs["d1"]["onnx_sha256"]
        or aggregate["sequence_count"] != 8189
    ):
        raise Dagger2GateError("preserved D1 aggregate/model identity changed")
    recheck = _read_json(_resolve(repo, config.inputs["cone_free_recheck_summary"]["path"]))
    valid = _read_json(_resolve(repo, config.inputs["cone_free_valid_result"]["path"]))
    d1_metrics = valid.get("metrics") or {}
    if (
        (recheck.get("classification") or {}).get("classification")
        != "POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED"
        or valid.get("classification") != "FULL_LAP_PASS"
        or d1_metrics.get("off_track_events") != 0
        or d1_metrics.get("safe_stop_success") is not True
    ):
        raise Dagger2GateError("preserved cone-free conclusion changed")
    for actual, expected, label in (
        (d1_metrics.get("elapsed_s"), 28.865773855999578, "lap time"),
        (d1_metrics.get("total_unwrapped_progress_m"), 30.124744237214415, "progress"),
        (d1_metrics.get("route_completion_fraction"), 0.9875472443388742, "completion"),
        (d1_metrics.get("mean_cte_m"), 0.09992980804696788, "mean CTE"),
        (d1_metrics.get("max_cte_m"), 0.34116400914188466, "max CTE"),
    ):
        if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
            raise Dagger2GateError(f"preserved cone-free {label} changed")
    return {
        "version": VERSION + "_audit", "generated_utc": utc_now(), "result": "PASS",
        "branch": branch, "head_commit": _git(repo, "rev-parse", "HEAD"),
        "hashes": hashes,
        "frozen_counts": {
            "expert_baseline": 6706, "dagger1": 1483, "d1_aggregate": 8189,
            "validation_s09_s10": 837,
        },
        "preserved_conclusion": "POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED",
        "generic_cone_free_regression_supported": False,
        "cone_free_recheck": {
            "classification": valid["classification"], "elapsed_s": d1_metrics["elapsed_s"],
            "progress_m": d1_metrics["total_unwrapped_progress_m"],
            "completion_fraction": d1_metrics["route_completion_fraction"],
            "mean_cte_m": d1_metrics["mean_cte_m"], "max_cte_m": d1_metrics["max_cte_m"],
            "off_track_events": d1_metrics["off_track_events"],
            "infrastructure_healthy": True, "safe_stop_success": True,
        },
        "d1_frozen_report_sha256": config.inputs["d1"]["training_summary_sha256"],
        "s11_s12_camera_or_live_access": 0,
    }


def audit_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    report = audit_preserved_inputs(config, repo)
    report["disk_before"] = disk_gate(config)
    report["external_root"] = str(config.external_root(sim_root))
    report["external_root_existed_before_milestone"] = config.external_root(sim_root).exists()
    write_json(config.result_dir(repo, "collection") / "audit.json", report)
    return report


FRAME_FIELDS = (
    "episode_id", "scenario_id", "scenario_role", "repeat_id", "frame_index",
    "capture_iteration", "image_path", "image_sha256", "camera_timestamp_ns",
    "expert_label_timestamp_ns", "expert_label_age_ms", "learner_x_m", "learner_y_m",
    "learner_yaw_rad", "route_progress_m", "route_s_m", "cte_m", "signed_cte_m",
    "heading_error_rad", "cone_phase", "cone_passed", "recovery_state",
    "recovery_success_at_capture", "history_context_only", "d1_steering_rad",
    "expert_steering_rad", "d1_minus_expert_rad", "absolute_steering_error_rad",
    "teacher_uses_actual_learner_pose", "teacher_valid", "provenance",
)

SEQUENCE_FIELDS = (
    "sequence_id", "episode_id", "scenario_id", "scenario_role", "repeat_id",
    "cone_scenario_id", "provenance", "frame_t_minus_2", "frame_t_minus_1", "frame_t",
    "frame_t_minus_2_sha256", "frame_t_minus_1_sha256", "frame_t_sha256",
    "timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns",
    "adjacent_gap_1_s", "adjacent_gap_2_s", "oldest_to_current_span_s",
    "expert_target_timestamp_ns", "expert_label_age_ms", "target_steering_rad",
    "d1_steering_rad", "d1_minus_expert_rad", "route_progress_m", "route_s_m",
    "learner_x_m", "learner_y_m", "learner_yaw_rad", "cte_m", "signed_cte_m",
    "heading_error_rad", "cone_phase", "recovery_state", "post_recovery_target",
    "teacher_uses_actual_learner_pose", "source_capture_metadata_sha256",
)

AGGREGATE_FIELDS = (
    "sequence_id", "provenance", "episode_id", "scenario_id", "scenario_role", "repeat_id",
    "frame_t_minus_2", "frame_t_minus_1", "frame_t", "timestamp_t_minus_2_ns",
    "timestamp_t_minus_1_ns", "timestamp_t_ns", "target_steering_rad", "route_progress_m",
    "cone_phase", "source_mcap_sha256", "source_manifest_sha256",
)


def temporal_triplet_contract(
    timestamps_ns: Sequence[int], episode_ids: Sequence[str], maximum_gap_s: float = 0.12,
) -> tuple[float, float, float] | None:
    if len(timestamps_ns) != 3 or len(episode_ids) != 3:
        raise ValueError("temporal sequence requires exactly three frames")
    if len(set(episode_ids)) != 1:
        raise Dagger2GateError("temporal sequence crossed an episode boundary")
    first, second, third = (int(value) for value in timestamps_ns)
    if not first < second < third:
        raise Dagger2GateError("temporal sequence is not strictly causal")
    gap1 = (second - first) / 1e9
    gap2 = (third - second) / 1e9
    if gap1 > maximum_gap_s or gap2 > maximum_gap_s:
        return None
    return gap1, gap2, gap1 + gap2


def teacher_pair_is_causal(
    teacher_timestamp_ns: int,
    camera_timestamp_ns: int,
    maximum_age_s: float = 0.12,
) -> bool:
    age = int(camera_timestamp_ns) - int(teacher_timestamp_ns)
    return 0 <= age <= int(maximum_age_s * 1e9)


def post_recovery_target_eligible(capture: Mapping[str, Any], return_end_s_m: float) -> bool:
    return bool(
        capture.get("teacher_valid") is True
        and capture.get("recovery_success_at_capture") is True
        and capture.get("cone_passed") is True
        and float(capture.get("route_s_m", -math.inf)) >= float(return_end_s_m)
        and teacher_pair_is_causal(
            int(capture["expert_label_timestamp_ns"]), int(capture["camera_timestamp_ns"]),
        )
    )


def _derived_rgb(jpeg: bytes, config: Dagger2Config) -> Image.Image:
    with Image.open(BytesIO(jpeg)) as source:
        source.load()
        if source.size != (480, 360) or source.format != "JPEG":
            raise Dagger2GateError(
                f"camera source changed: format={source.format!r}, size={source.size}"
            )
        rgb = source.convert("RGB")
    image_config = {
        "roi": config.dataset["roi"], "output_width": 200, "output_height": 66,
    }
    result = preprocess_image(rgb, image_config)
    rgb.close()
    if result.mode != "RGB" or result.size != (200, 66):
        result.close()
        raise Dagger2GateError("derived camera frame is not canonical 200x66 RGB")
    return result


def _persist_capture(
    capture: dict[str, Any], *, staging: Path, episode: Dagger2Episode,
    frame_rows: list[dict[str, Any]], saved: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    iteration = int(capture["capture_iteration"])
    if iteration in saved:
        return saved[iteration]
    image = capture.get("_image")
    if not isinstance(image, Image.Image):
        raise Dagger2GateError("capture no longer has an in-memory derived RGB image")
    frame_index = len(frame_rows)
    relative = Path("images") / f"frame_{frame_index:06d}.png"
    output = staging / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=False)
    row = {
        key: value for key, value in capture.items() if not key.startswith("_")
    }
    row.update({
        "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
        "scenario_role": "TRAIN", "repeat_id": "R01", "frame_index": frame_index,
        "image_path": relative.as_posix(), "image_sha256": sha256_file(output),
        "history_context_only": not bool(capture["recovery_success_at_capture"]),
        "provenance": PROVENANCE,
    })
    frame_rows.append(row)
    saved[iteration] = row
    return row


def _sequence_from_ring(
    ring: Sequence[dict[str, Any]], *, episode: Dagger2Episode,
    staging: Path, frame_rows: list[dict[str, Any]], sequence_rows: list[dict[str, Any]],
    saved: dict[int, dict[str, Any]], return_end_s_m: float,
) -> bool:
    if len(ring) != 3 or not post_recovery_target_eligible(ring[-1], return_end_s_m):
        return False
    temporal = temporal_triplet_contract(
        [int(item["camera_timestamp_ns"]) for item in ring],
        [episode.episode_id] * 3,
    )
    if temporal is None:
        return False
    rows = [
        _persist_capture(item, staging=staging, episode=episode, frame_rows=frame_rows, saved=saved)
        for item in ring
    ]
    first, second, target = rows
    gap1, gap2, span = temporal
    if int(target["expert_label_timestamp_ns"]) > int(target["camera_timestamp_ns"]):
        raise Dagger2GateError("future teacher target reached sequence construction")
    sequence_rows.append({
        "sequence_id": f"{episode.episode_id}_seq_{len(sequence_rows):06d}",
        "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
        "scenario_role": "TRAIN", "repeat_id": "R01", "cone_scenario_id": episode.scenario_id,
        "provenance": PROVENANCE,
        "frame_t_minus_2": first["image_path"], "frame_t_minus_1": second["image_path"],
        "frame_t": target["image_path"], "frame_t_minus_2_sha256": first["image_sha256"],
        "frame_t_minus_1_sha256": second["image_sha256"], "frame_t_sha256": target["image_sha256"],
        "timestamp_t_minus_2_ns": first["camera_timestamp_ns"],
        "timestamp_t_minus_1_ns": second["camera_timestamp_ns"],
        "timestamp_t_ns": target["camera_timestamp_ns"], "adjacent_gap_1_s": gap1,
        "adjacent_gap_2_s": gap2, "oldest_to_current_span_s": span,
        "expert_target_timestamp_ns": target["expert_label_timestamp_ns"],
        "expert_label_age_ms": target["expert_label_age_ms"],
        "target_steering_rad": target["expert_steering_rad"],
        "d1_steering_rad": target["d1_steering_rad"],
        "d1_minus_expert_rad": target["d1_minus_expert_rad"],
        "route_progress_m": target["route_progress_m"], "route_s_m": target["route_s_m"],
        "learner_x_m": target["learner_x_m"], "learner_y_m": target["learner_y_m"],
        "learner_yaw_rad": target["learner_yaw_rad"], "cte_m": target["cte_m"],
        "signed_cte_m": target["signed_cte_m"], "heading_error_rad": target["heading_error_rad"],
        "cone_phase": target["cone_phase"], "recovery_state": "PASS",
        "post_recovery_target": True, "teacher_uses_actual_learner_pose": True,
        "source_capture_metadata_sha256": "",
    })
    return True


def run_d1_targeted_rollout(
    observer: RandomConeObserver,
    model: TemporalOnnxModel,
    live_config: Any,
    initial: Any,
    expert: RandomConeConfig,
    bundle: ScenarioBundle,
    episode: Dagger2Episode,
    staging: Path,
    config: Dagger2Config,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """D1 controls; only recovery-qualified derived frames cross the storage boundary."""
    safety = live_config.safety_config(1.0)
    nominal = initial.route
    control_route = ObstacleAwareRoute(nominal, bundle.plan)
    tracker = ProgressTracker(nominal.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s)
    liveness = PoseLivenessMonitor(
        safety.pose_stale_timeout_s,
        safety.pose_motion_translation_threshold_m,
        safety.pose_motion_yaw_threshold_rad,
    )
    telemetry: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    saved: dict[int, dict[str, Any]] = {}
    ring: list[dict[str, Any]] = []
    periods: list[float] = []
    camera_times: list[float] = []
    prep_times: list[float] = []
    inference_times: list[float] = []
    total_times: list[float] = []
    gap1_values: list[float] = []
    gap2_values: list[float] = []
    span_values: list[float] = []
    ctes: list[float] = []
    steerings: list[float] = []
    invalid_teacher = future_teacher = stale_teacher = saturation = 0
    api_failures = liveness_failures = invalid_history = timing_slips = 0
    temporal_failure = False
    result = "FAIL"
    failure: str | None = None
    motion = False
    previous_tick: float | None = None
    previous_camera_ns: int | None = None
    final_pose = initial.pose
    final_projection = nominal.project((float(final_pose["x"]), float(final_pose["y"])))
    started = time.monotonic()
    next_tick = started
    next_world = started
    stop_errors: list[str] = []
    recovery_pass_iteration: int | None = None
    try:
        buffer, warmup = _warm_r1_buffer(observer, live_config)
        while True:
            now = time.monotonic()
            if now - started >= safety.maximum_runtime_s:
                raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            if previous_tick is not None:
                period = tick - previous_tick
                periods.append(period)
                timing_slips += int(period > 0.1)
            previous_tick = tick
            if tick >= next_world:
                status = observer.status()
                if (
                    status.get("running") is not True
                    or status.get("switching") is not False
                    or status.get("current") != initial.world
                ):
                    raise RuntimeError("simulator state changed while D1 was driving")
                next_world = tick + safety.world_check_interval_s

            pose = observer.pose()
            label_clock = observer.clock()
            label_time_s = float(label_clock["sim_time"])
            label_time_ns = int(round(label_time_s * 1e9))
            label = frozen_teacher_label(nominal, control_route, pose, bundle, expert)
            final_pose = pose
            if label.get("route_s_m") is None:
                invalid_teacher += 1
                raise RuntimeError("frozen Expert could not label the actual D1 state")
            final_projection = nominal.project((float(pose["x"]), float(pose["y"])))
            tracker.update(final_projection.s)
            boundary = nominal.track_boundary_distance((float(pose["x"]), float(pose["y"])))
            if boundary is None or not math.isfinite(boundary):
                raise RuntimeError("invalid track boundary geometry")
            try:
                liveness.update(pose, label_time_s, time.monotonic(), motion_commanded=motion)
            except RuntimeError:
                liveness_failures += 1
                raise

            camera_started = time.perf_counter()
            jpeg = observer.camera_jpeg(live_config.payload["camera_path"])
            camera_host_timestamp = time.monotonic()
            camera_clock = observer.clock()
            camera_timestamp_ns = int(round(float(camera_clock["sim_time"]) * 1e9))
            camera_times.append(time.perf_counter() - camera_started)
            derived = _derived_rgb(jpeg, config)
            prep_started = time.perf_counter()
            append_live_jpeg(buffer, jpeg, camera_host_timestamp, roi=live_config.roi)
            prep_times.append(time.perf_counter() - prep_started)
            gap1, gap2, span = buffer.gaps()
            gap1_values.append(gap1); gap2_values.append(gap2); span_values.append(span)
            model_started = time.perf_counter()
            inference_started = time.perf_counter()
            normalized = model.predict(buffer.tensor())
            inference_times.append(time.perf_counter() - inference_started)
            total_times.append(time.perf_counter() - model_started)
            d1_steering, speed = fixed_speed_commands(
                1.0, float(steering_normalized_to_rad(normalized, safety.max_steering_rad)),
            )
            expert_steering = float(label["expert_steering_rad"])
            teacher_age_ns = camera_timestamp_ns - label_time_ns
            teacher_valid = bool(
                label.get("teacher_valid") is True
                and teacher_pair_is_causal(label_time_ns, camera_timestamp_ns)
            )
            if teacher_age_ns < 0:
                future_teacher += 1
            elif teacher_age_ns > 120_000_000:
                stale_teacher += 1
            elif not teacher_valid:
                invalid_teacher += 1
            capture = {
                "capture_iteration": len(telemetry), "camera_timestamp_ns": camera_timestamp_ns,
                "expert_label_timestamp_ns": label_time_ns,
                "expert_label_age_ms": teacher_age_ns / 1e6,
                "learner_x_m": float(pose["x"]), "learner_y_m": float(pose["y"]),
                "learner_yaw_rad": float(pose["yaw"]),
                "route_progress_m": float(tracker.unwrapped), "route_s_m": float(label["route_s_m"]),
                "cte_m": float(label["cte_m"]), "signed_cte_m": float(label["signed_cte_m"]),
                "heading_error_rad": float(label["heading_error_rad"]),
                "cone_phase": label["cone_phase"],
                "cone_passed": float(label["route_s_m"]) >= float(bundle.plan.cone_s_m),
                "recovery_state": "PASS" if observer.recovery_success else "NOT_YET_PASS",
                "recovery_success_at_capture": bool(observer.recovery_success),
                "d1_steering_rad": float(d1_steering),
                "expert_steering_rad": expert_steering,
                "d1_minus_expert_rad": float(d1_steering - expert_steering),
                "absolute_steering_error_rad": float(abs(d1_steering - expert_steering)),
                "teacher_uses_actual_learner_pose": True, "teacher_valid": teacher_valid,
                "_image": derived,
            }
            if teacher_valid:
                if previous_camera_ns is not None and (
                    camera_timestamp_ns <= previous_camera_ns
                    or camera_timestamp_ns - previous_camera_ns > 120_000_000
                ):
                    for old in ring:
                        image = old.get("_image")
                        if isinstance(image, Image.Image):
                            image.close()
                    ring = []
                previous_camera_ns = camera_timestamp_ns
                ring.append(capture)
                if len(ring) > 3:
                    old = ring.pop(0)
                    image = old.get("_image")
                    if isinstance(image, Image.Image):
                        image.close()
                if observer.recovery_success:
                    if recovery_pass_iteration is None:
                        recovery_pass_iteration = len(telemetry)
                    _sequence_from_ring(
                        ring, episode=episode, staging=staging, frame_rows=frame_rows,
                        sequence_rows=sequence_rows, saved=saved,
                        return_end_s_m=float(bundle.plan.return_end_s_m),
                    )
            else:
                derived.close()
                for old in ring:
                    image = old.get("_image")
                    if isinstance(image, Image.Image):
                        image.close()
                ring = []
                previous_camera_ns = None

            row = {
                key: value for key, value in capture.items() if not key.startswith("_")
            }
            row.update({
                "iteration": len(telemetry), "elapsed_s": tick - started,
                "policy_status": "D1_CONTROL", "actual_policy_steering_rad": float(d1_steering),
                "teacher_has_control_authority": False, "boundary_distance_m": float(boundary),
                "camera_acquisition_ms": camera_times[-1] * 1000.0,
                "preprocessing_ms": prep_times[-1] * 1000.0,
                "onnx_inference_ms": inference_times[-1] * 1000.0,
                "temporal_gap_t_minus_2_to_t_minus_1_s": gap1,
                "temporal_gap_t_minus_1_to_t_s": gap2, "temporal_span_s": span,
            })
            telemetry.append(row)
            if off_track.update(boundary > safety.off_track_margin_m, time.monotonic()):
                raise RuntimeError(f"sustained off-track: boundary distance {boundary:.3f}m")
            observer.command_steering(d1_steering)
            observer.command_speed(speed)
            if not motion:
                motion = True
                liveness.update(pose, label_time_s, time.monotonic(), motion_commanded=True)
            ctes.append(float(label["cte_m"])); steerings.append(float(d1_steering))
            saturation += int(math.isclose(abs(d1_steering), safety.max_steering_rad, abs_tol=1e-8))
            if tracker.lap_complete(
                math.dist((float(pose["x"]), float(pose["y"])), nominal.points[0]),
                safety.start_gate_radius_m, safety.minimum_lap_progress_fraction,
            ):
                result = "PASS"
                break
            next_tick += 1.0 / safety.control_frequency_hz
            if next_tick < time.monotonic() - 1.0 / safety.control_frequency_hz:
                next_tick = time.monotonic()
    except TemporalInputError as exc:
        temporal_failure = True; invalid_history += 1; failure = str(exc)
    except Exception as exc:
        failure = str(exc)
        if any(token in failure.lower() for token in (
            "get ", "post ", "control rejected", "unavailable", "simulator state changed",
        )):
            api_failures += 1
    finally:
        off_track.finalize(time.monotonic())
        stop_errors = observer.safe_stop()
        for capture in ring:
            image = capture.get("_image")
            if isinstance(image, Image.Image):
                image.close()
    errors = np.asarray([
        float(row["d1_minus_expert_rad"]) for row in telemetry if row.get("teacher_valid") is True
    ], dtype=np.float64)
    d1_values = np.asarray([
        float(row["d1_steering_rad"]) for row in telemetry if row.get("teacher_valid") is True
    ], dtype=np.float64)
    expert_values = np.asarray([
        float(row["expert_steering_rad"]) for row in telemetry if row.get("teacher_valid") is True
    ], dtype=np.float64)
    expert_abs = float(np.mean(np.abs(expert_values))) if expert_values.size else 0.0
    run = {
        "result": result, "failure": failure, "elapsed_s": time.monotonic() - started,
        "route_length_m": nominal.length, "route_completion_fraction": tracker.unwrapped / nominal.length,
        "total_unwrapped_progress_m": tracker.unwrapped, "final_route_s_m": float(final_projection.s),
        "final_distance_to_start_m": math.dist(
            (float(final_pose["x"]), float(final_pose["y"])), nominal.points[0],
        ),
        "mean_cte_m": statistics.fmean(ctes) if ctes else None,
        "max_cte_m": max(ctes, default=None), "off_track_events": off_track.event_count,
        "off_track_event_count": off_track.event_count,
        "off_track_total_duration_s": off_track.total_duration_s,
        "mean_d1_steering_rad": statistics.fmean(steerings) if steerings else None,
        "mean_absolute_predicted_steering_rad": statistics.fmean(abs(v) for v in steerings) if steerings else None,
        "max_absolute_predicted_steering_rad": max((abs(v) for v in steerings), default=None),
        "steering_saturation_fraction": saturation / len(steerings) if steerings else 0.0,
        "shadow_expert": {
            "sample_count": int(errors.size),
            "mean_expert_steering_rad": float(np.mean(expert_values)) if expert_values.size else None,
            "mean_signed_error_rad": float(np.mean(errors)) if errors.size else None,
            "mean_absolute_error_rad": float(np.mean(np.abs(errors))) if errors.size else None,
            "corrective_magnitude_ratio": (
                float(np.mean(np.abs(d1_values)) / expert_abs) if expert_abs > 0 else None
            ),
            "steering_sign_agreement_fraction": (
                float(np.mean(np.sign(d1_values) == np.sign(expert_values))) if errors.size else None
            ),
            "teacher_control_authority": False,
        },
        "camera_acquisition_latency": _summary_ms(camera_times),
        "preprocessing_latency": _summary_ms(prep_times),
        "onnx_inference_latency": _summary_ms(inference_times),
        "total_temporal_model_path_latency": _summary_ms(total_times),
        "control_loop_period": _summary_ms(periods),
        "control_loop_frequency_hz": 1.0 / statistics.fmean(periods) if periods else 0.0,
        "timing_slips_over_100ms": timing_slips,
        "temporal_frame_gaps": {
            "oldest_to_middle_s": distribution(gap1_values),
            "middle_to_current_s": distribution(gap2_values),
            "oldest_to_current_s": distribution(span_values),
        },
        "temporal_input_failure": temporal_failure,
        "temporal_invalid_history_count": invalid_history,
        "api_failures": api_failures, "liveness_failures": liveness_failures,
        "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors,
        "speed_mps": 1.0, "control_frequency_hz": 15.0,
        "learner_policy": "frozen D1", "learner_policy_frozen": True,
        "teacher_control_authority": False, "rosbags_recorded": 0,
        "raw_full_resolution_images_persisted": 0,
        "derived_images_persisted": len(frame_rows),
        "post_recovery_temporal_sequences": len(sequence_rows),
        "recovery_pass_iteration": recovery_pass_iteration,
        "teacher_invalid_frames_excluded": invalid_teacher,
        "future_teacher_label_violations": future_teacher,
        "stale_teacher_frames_excluded": stale_teacher,
        "warmup": locals().get("warmup"),
    }
    return run, telemetry, frame_rows, sequence_rows


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def _directory_tree_identity(path: Path) -> dict[str, Any]:
    """Hash names and contents without changing preserved external evidence."""
    digest = hashlib.sha256()
    file_count = 0
    size_bytes = 0
    if path.is_dir():
        for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = item.relative_to(path).as_posix()
            item_sha256 = sha256_file(item)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(item_sha256.encode("ascii"))
            digest.update(b"\n")
            file_count += 1
            size_bytes += item.stat().st_size
    return {
        "path": str(path), "sha256": digest.hexdigest(),
        "file_count": file_count, "size_bytes": size_bytes,
    }


def _finalized_episode_identity(
    result_dir: Path, episode: Dagger2Episode, record: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = result_dir / "states" / f"{episode.episode_id}.json"
    episode_path = result_dir / "episodes" / f"{episode.episode_id}.json"
    attempts = sorted(
        (result_dir / "attempts").glob(f"{episode.episode_id}_attempt_*.json")
    )
    if not state_path.is_file() or not episode_path.is_file() or not attempts:
        raise Dagger2GateError(f"finalized compact evidence is incomplete: {episode.episode_id}")
    state = _read_json(state_path)
    if (
        state.get("status") != "FINALIZED_VALID_POLICY_OUTCOME"
        or state.get("episode_id") != episode.episode_id
        or state.get("policy_outcome") != record.get("policy_outcome")
        or state.get("do_not_repeat") is not True
    ):
        raise Dagger2GateError(f"finalized state identity changed: {episode.episode_id}")
    return {
        "episode_id": episode.episode_id,
        "state_sha256": sha256_file(state_path),
        "episode_sha256": sha256_file(episode_path),
        "attempt_sha256": {path.name: sha256_file(path) for path in attempts},
        "external_tree": _directory_tree_identity(Path(str(record["external_episode_root"]))),
    }


def host_crash_replacement_attempt(
    state: Mapping[str, Any], *, maximum_replacements: int = 1,
) -> int | None:
    """Return the sole allowed replacement number for an unfinished host-crash attempt."""
    if state.get("status") != "STARTED_UNFINALIZED":
        return None
    attempt_number = int(state.get("attempt_number", 0))
    if attempt_number < 1:
        raise Dagger2GateError("invalid interrupted attempt number")
    next_attempt = attempt_number + 1
    maximum_attempt = 1 + int(maximum_replacements)
    return next_attempt if next_attempt <= maximum_attempt else None


def _host_crash_attempt_record(
    state: Mapping[str, Any], episode: Dagger2Episode, *, state_path: Path,
    staging: Path, archive: Path | None,
) -> dict[str, Any]:
    """Preserve an unfinished attempt as infrastructure evidence, never policy evidence."""
    if (
        state.get("status") != "STARTED_UNFINALIZED"
        or state.get("episode_id") != episode.episode_id
        or state.get("scenario_id") != episode.scenario_id
    ):
        raise Dagger2GateError(f"state is not an interrupted {episode.episode_id} attempt")
    replacement = host_crash_replacement_attempt(state)
    return {
        "version": COLLECTION_VERSION + "_interrupted",
        "generated_utc": utc_now(),
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "scenario_role": "TRAIN",
        "attempt_number": int(state["attempt_number"]),
        "classification": INFRASTRUCTURE_INTERRUPTION,
        "infrastructure_outcome": HOST_CRASH,
        "policy_outcome": None,
        "result": "INVALIDATED_INFRASTRUCTURE",
        "failure_reason": "host/OS crash before any captured genuine policy outcome",
        "counts_as_genuine_policy_outcome": False,
        "do_not_reinterpret_as_policy_fail": True,
        "interrupted_state": dict(state),
        "interrupted_state_path": str(state_path),
        "interrupted_state_sha256": sha256_file(state_path),
        "staging_path_before_archive": str(staging),
        "staging_file_count_before_archive": sum(
            item.is_file() for item in staging.rglob("*")
        ) if staging.is_dir() else 0,
        "archive_path": None if archive is None else str(archive),
        "bounded_replacement_authorized": replacement is not None,
        "replacement_attempt_number": replacement,
        "maximum_infrastructure_replacements": 1,
    }


def _attempt_paths(
    config: Dagger2Config, repo: Path, sim_root: Path,
    episode: Dagger2Episode, attempt_number: int,
) -> tuple[Path, Path, Path]:
    result = config.result_dir(repo, "collection") / "attempts" / (
        f"{episode.episode_id}_attempt_{attempt_number:02d}.json"
    )
    state = config.result_dir(repo, "collection") / "states" / f"{episode.episode_id}.json"
    staging = config.external_root(sim_root) / "staging" / (
        f"{episode.episode_id}_attempt_{attempt_number:02d}"
    )
    return result, state, staging


def _compact_attempt(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key) for key in (
            "version", "generated_utc", "episode_id", "scenario_id", "scenario_role",
            "attempt_number", "classification", "result", "failure_reason", "learner_policy",
            "learner_checkpoint_sha256", "learner_onnx_sha256", "d1_controls_vehicle",
            "teacher_control_authority", "world_activation", "preflight", "run",
            "outcome_path", "outcome_sha256", "post_run_safe_stop_success",
            "post_run_safe_stop_errors",
        )
    }


def collect_dagger2_attempt(
    episode: Dagger2Episode,
    *,
    repo: Path,
    sim_root: Path,
    config: Dagger2Config,
    expert: RandomConeConfig,
    bundle: ScenarioBundle,
    client: SimClient,
    r1_config: R1Config,
    model: TemporalOnnxModel,
    attempt_number: int,
) -> dict[str, Any]:
    attempt_path, state_path, staging = _attempt_paths(
        config, repo, sim_root, episode, attempt_number,
    )
    if attempt_path.exists() or staging.exists():
        raise Dagger2GateError(f"refusing to overwrite DAgger2 attempt {episode.episode_id} #{attempt_number}")
    staging.mkdir(parents=True)
    write_json(state_path, {
        "status": "STARTED_UNFINALIZED", "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id, "attempt_number": attempt_number,
        "started_utc": utc_now(), "genuine_policy_outcome_must_not_be_retried": True,
        "staging_path": str(staging),
    })
    record: dict[str, Any] = {
        "version": COLLECTION_VERSION + "_attempt", "generated_utc": utc_now(),
        "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
        "scenario_role": "TRAIN", "repeat_id": "R01", "attempt_number": attempt_number,
        "classification": "INFRA_FAIL", "result": "FAIL", "failure_reason": None,
        "learner_policy": "frozen D1", "learner_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "learner_onnx_sha256": config.inputs["d1"]["onnx_sha256"],
        "d1_controls_vehicle": True, "teacher_control_authority": False,
        "rosbag_created": False, "raw_full_resolution_images_persisted": 0,
        "world_activation": None, "preflight": None, "run": None,
        "outcome_path": None, "outcome_sha256": None,
        "post_run_safe_stop_success": False, "post_run_safe_stop_errors": [],
    }
    try:
        if errors := client.safe_stop():
            raise Dagger2GateError("initial collection safe stop failed: " + "; ".join(errors))
        initial, activation, preflight = _post_settle_preflight(
            client, expert, bundle, sim_root, float(config.collection["settle_duration_s"]),
        )
        record["world_activation"] = activation
        record["preflight"] = preflight
        if preflight.get("fixed_control") != {
            "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
            "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
        }:
            raise Dagger2GateError("frozen 1.00 m/s preflight control changed")
        observer = RandomConeObserver(client, initial.route, bundle, expert)
        live_config = r1_inference_config(r1_config, expert.world_name(episode.scenario_id))
        run, telemetry, frames, sequences = run_d1_targeted_rollout(
            observer, model, live_config, initial, expert, bundle, episode, staging, config,
        )
        run = summarize_neural_cone_run(run, observer, bundle)
        record["run"] = run
        record["classification"] = run["classification"]
        record["result"] = "PASS" if run["classification"] in VALID_POLICY_CLASSIFICATIONS else "FAIL"
        outcome = {
            "version": COLLECTION_VERSION + "_captured_outcome", "generated_utc": utc_now(),
            "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
            "attempt_number": attempt_number, "classification": run["classification"],
            "run": run, "telemetry": telemetry, "frames": frames, "sequences": sequences,
            "frozen_scenario_sha256": _scenario_hash(bundle),
            "frozen_scenario": bundle.scenario.to_dict(), "planned_bypass": bundle.geometry,
            "bypass_side": bundle.plan.side, "teacher_control_authority": False,
            "d1_controls_vehicle": True, "rosbags_recorded": 0,
            "raw_full_resolution_images_persisted": 0,
        }
        outcome_path = staging / "outcome.json"
        write_json(outcome_path, outcome)
        record["outcome_path"] = str(outcome_path)
        record["outcome_sha256"] = sha256_file(outcome_path)
        write_json(state_path, {
            "status": "POLICY_OUTCOME_CAPTURED", "episode_id": episode.episode_id,
            "scenario_id": episode.scenario_id, "attempt_number": attempt_number,
            "classification": run["classification"], "outcome_path": str(outcome_path),
            "outcome_sha256": record["outcome_sha256"], "captured_utc": utc_now(),
            "do_not_repeat_if_policy_outcome": run["classification"] in VALID_POLICY_CLASSIFICATIONS,
        })
    except BaseException as exc:
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        errors = client.safe_stop()
        record["post_run_safe_stop_success"] = not errors
        record["post_run_safe_stop_errors"] = errors
        if errors:
            record["classification"] = "INFRA_FAIL"
            record["result"] = "FAIL"
            record["failure_reason"] = (
                (record.get("failure_reason") + "; ") if record.get("failure_reason") else ""
            ) + "; ".join(errors)
    if record.get("classification") not in (*VALID_POLICY_CLASSIFICATIONS, "INFRA_FAIL"):
        raise Dagger2GateError("unexpected collection classification")
    write_json(attempt_path, _compact_attempt(record))
    return record


def _archive_invalid_staging(
    config: Dagger2Config, sim_root: Path, episode: Dagger2Episode,
    attempt_number: int, staging: Path,
) -> Path | None:
    if not staging.exists():
        return None
    archive = config.external_root(sim_root) / "infrastructure_invalid" / (
        f"{episode.episode_id}_attempt_{attempt_number:02d}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise Dagger2GateError(f"invalid-attempt archive already exists: {archive}")
    staging.rename(archive)
    return archive


def _finalize_policy_outcome(
    record: dict[str, Any], *, config: Dagger2Config, repo: Path,
    sim_root: Path, episode: Dagger2Episode,
) -> dict[str, Any]:
    if record.get("classification") not in VALID_POLICY_CLASSIFICATIONS:
        raise Dagger2GateError("only a valid policy outcome may be finalized")
    attempt_number = int(record["attempt_number"])
    attempt_path, state_path, staging = _attempt_paths(
        config, repo, sim_root, episode, attempt_number,
    )
    outcome_path = Path(str(record.get("outcome_path") or staging / "outcome.json"))
    if not outcome_path.is_file():
        candidate = staging / "outcome.json"
        if candidate.is_file():
            outcome_path = candidate
        else:
            raise Dagger2GateError("captured policy outcome metadata is missing")
    if record.get("outcome_sha256") and sha256_file(outcome_path) != record["outcome_sha256"]:
        raise Dagger2GateError("captured policy outcome hash changed")
    outcome = _read_json(outcome_path)
    if (
        outcome.get("classification") != record["classification"]
        or outcome.get("episode_id") != episode.episode_id
        or outcome.get("scenario_id") != episode.scenario_id
    ):
        raise Dagger2GateError("captured outcome identity changed")
    frames = list(outcome.get("frames") or [])
    sequences = list(outcome.get("sequences") or [])
    for frame in frames:
        image = staging / frame["image_path"]
        if not image.is_file() or sha256_file(image) != frame["image_sha256"]:
            raise Dagger2GateError("derived frame missing or corrupt before finalization")
        with Image.open(image) as opened:
            opened.load()
            if opened.mode != "RGB" or opened.size != (200, 66) or opened.format != "PNG":
                raise Dagger2GateError("persisted DAgger2 frame contract changed")
    outcome_sha = sha256_file(outcome_path)
    finalized_sequences = [
        {**row, "source_capture_metadata_sha256": outcome_sha} for row in sequences
    ]
    frame_manifest = staging / "frames.csv"
    sequence_manifest = staging / "sequences.csv"
    _write_csv(frame_manifest, frames, FRAME_FIELDS)
    _write_csv(sequence_manifest, finalized_sequences, SEQUENCE_FIELDS)
    metadata = {
        "version": COLLECTION_VERSION + "_external_episode", "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id, "classification": record["classification"],
        "outcome_sha256": outcome_sha, "frame_manifest_sha256": sha256_file(frame_manifest),
        "sequence_manifest_sha256": sha256_file(sequence_manifest),
        "derived_image_count": len(frames), "temporal_sequence_count": len(finalized_sequences),
        "provenance": PROVENANCE, "rosbags_recorded": 0,
        "raw_full_resolution_images_persisted": 0,
    }
    write_json(staging / "metadata.json", metadata)
    final_root = config.external_root(sim_root) / "episodes" / episode.episode_id
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise Dagger2GateError(f"final external episode already exists: {final_root}")
    staging.rename(final_root)
    final_outcome = final_root / "outcome.json"
    final_frame_manifest = final_root / "frames.csv"
    final_sequence_manifest = final_root / "sequences.csv"
    run = outcome["run"]
    target_frames = [frame for frame in frames if frame.get("recovery_success_at_capture") is True]
    errors = [float(row["d1_minus_expert_rad"]) for row in finalized_sequences]
    final_record = {
        "version": COLLECTION_VERSION + "_episode", "generated_utc": utc_now(),
        "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
        "scenario_role": "TRAIN", "repeat_id": "R01", "attempt_number": attempt_number,
        "result": "PASS", "classification": "DAGGER2_EVIDENCE_PASS",
        "policy_outcome": record["classification"], "learner_run": run,
        "d1_controls_vehicle": True, "teacher_control_authority": False,
        "learner_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
        "learner_onnx_sha256": config.inputs["d1"]["onnx_sha256"],
        "frozen_scenario_sha256": outcome["frozen_scenario_sha256"],
        "frozen_scenario": outcome["frozen_scenario"], "bypass_side": outcome["bypass_side"],
        "cone_pass_success": bool(
            run.get("minimum_cone_clearance_route_s_m") is not None
            and float(run.get("final_route_s_m", 0.0)) >= float(outcome["planned_bypass"]["cone_s_m"])
        ),
        "recovery_success": run.get("recovery_success") is True,
        "recovery_time_s": run.get("recovery_time_s"),
        "contributes_training_data": len(finalized_sequences) > 0,
        "zero_samples_due_to_pre_recovery_policy_failure": (
            record["classification"] == "RANDOM_CONE_POLICY_FAIL"
            and run.get("recovery_success") is not True and not finalized_sequences
        ),
        "derived_image_count": len(frames), "history_context_frame_count": sum(
            frame.get("history_context_only") is True for frame in frames
        ),
        "post_recovery_target_frame_count": len(target_frames),
        "temporal_sequence_count": len(finalized_sequences),
        "route_progress_m": distribution([
            float(row["route_progress_m"]) for row in finalized_sequences
        ]),
        "sequences_after_20m": sum(float(row["route_s_m"]) > 20.0 for row in finalized_sequences),
        "sequences_after_26m": sum(float(row["route_s_m"]) > 26.0 for row in finalized_sequences),
        "d1_vs_expert_post_recovery": {
            "sample_count": len(errors),
            "mae_rad": statistics.fmean(abs(value) for value in errors) if errors else None,
            "signed_bias_rad": statistics.fmean(errors) if errors else None,
            "maximum_absolute_error_rad": max((abs(value) for value in errors), default=None),
        },
        "cte_m": distribution([float(row["cte_m"]) for row in finalized_sequences]),
        "teacher_invalid_frames_excluded": run["teacher_invalid_frames_excluded"],
        "future_teacher_label_violations": run["future_teacher_label_violations"],
        "temporal_corruption_count": 0,
        "rosbags_recorded": 0, "raw_full_resolution_images_persisted": 0,
        "external_episode_root": str(final_root), "external_episode_size_bytes": _directory_size(final_root),
        "outcome": {"path": str(final_outcome), "sha256": sha256_file(final_outcome)},
        "frame_manifest": {"path": str(final_frame_manifest), "sha256": sha256_file(final_frame_manifest)},
        "temporal_manifest": {
            "path": str(final_sequence_manifest), "sha256": sha256_file(final_sequence_manifest),
        },
        "post_run_safe_stop_success": record.get("post_run_safe_stop_success") is True,
    }
    final_path = config.result_dir(repo, "collection") / "episodes" / f"{episode.episode_id}.json"
    write_json(final_path, final_record)
    updated_attempt = {**_compact_attempt(record)}
    updated_attempt["outcome_path"] = str(final_outcome)
    updated_attempt["outcome_sha256"] = sha256_file(final_outcome)
    write_json(attempt_path, updated_attempt)
    write_json(state_path, {
        "status": "FINALIZED_VALID_POLICY_OUTCOME", "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id, "attempt_number": attempt_number,
        "policy_outcome": record["classification"], "finalized_utc": utc_now(),
        "do_not_repeat": True,
    })
    return final_record


def validate_existing_episode(
    path: Path, episode: Dagger2Episode, config: Dagger2Config,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = _read_json(path)
    if (
        record.get("version") != COLLECTION_VERSION + "_episode"
        or record.get("episode_id") != episode.episode_id
        or record.get("scenario_id") != episode.scenario_id
        or record.get("scenario_role") != "TRAIN"
        or record.get("result") != "PASS"
        or record.get("classification") != "DAGGER2_EVIDENCE_PASS"
        or record.get("learner_onnx_sha256") != config.inputs["d1"]["onnx_sha256"]
        or record.get("d1_controls_vehicle") is not True
        or record.get("teacher_control_authority") is not False
        or record.get("rosbags_recorded") != 0
        or record.get("raw_full_resolution_images_persisted") != 0
    ):
        raise Dagger2GateError(f"completed DAgger2 episode identity changed: {episode.episode_id}")
    for key in ("outcome", "frame_manifest", "temporal_manifest"):
        item = record[key]
        path_value = Path(item["path"])
        if not path_value.is_file() or sha256_file(path_value) != item["sha256"]:
            raise Dagger2GateError(f"completed DAgger2 {key} changed: {episode.episode_id}")
    return record


def collection_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gates = {
        "exact_s01_s08_episode_order": [item.get("episode_id") for item in records] == list(DAGGER2_EPISODES),
        "exact_train_scenarios": [item.get("scenario_id") for item in records] == list(TRAIN_SCENARIOS),
        "all_valid_policy_evidence": all(
            item.get("policy_outcome") in VALID_POLICY_CLASSIFICATIONS for item in records
        ),
        "d1_controlled_all": all(item.get("d1_controls_vehicle") is True for item in records),
        "expert_shadow_only": all(item.get("teacher_control_authority") is False for item in records),
        "no_bags": all(item.get("rosbags_recorded") == 0 for item in records),
        "no_raw_full_resolution_images": all(
            item.get("raw_full_resolution_images_persisted") == 0 for item in records
        ),
        "no_s09_s12_collection": not any(
            item.get("scenario_id") in VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS for item in records
        ),
        "future_teacher_labels_zero": sum(
            int(item.get("future_teacher_label_violations", 0)) for item in records
        ) == 0,
        "safe_stop_all": all(item.get("post_run_safe_stop_success") is True for item in records),
    }
    return {"result": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def collection_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    audit_preserved_inputs(config, repo)
    initial_disk = disk_gate(config)
    result_dir = config.result_dir(repo, "collection")
    result_dir.mkdir(parents=True, exist_ok=True)
    preexisting: dict[str, dict[str, Any]] = {}
    preserved_before: dict[str, dict[str, Any]] = {}
    for episode in episode_specs():
        final_path = result_dir / "episodes" / f"{episode.episode_id}.json"
        existing = validate_existing_episode(final_path, episode, config)
        if existing is None:
            continue
        _validate_episode_dataset(existing, episode)
        preexisting[episode.episode_id] = existing
        preserved_before[episode.episode_id] = _finalized_episode_identity(
            result_dir, episode, existing,
        )
    missing_episode_count = len(DAGGER2_EPISODES) - len(preexisting)
    projection_basis = max(
        (int(record["external_episode_size_bytes"]) for record in preexisting.values()),
        default=0,
    )
    projected_final_free = int(initial_disk["available_bytes"]) - (
        projection_basis * missing_episode_count
    )
    resume_projection = {
        "basis": "largest finalized DAgger2 episode observed before resume",
        "finalized_episode_count": len(preexisting),
        "missing_episode_count": missing_episode_count,
        "projected_bytes_per_missing_episode": projection_basis,
        "available_before_remainder_bytes": int(initial_disk["available_bytes"]),
        "projected_final_free_bytes": projected_final_free,
        "required_projected_final_free_bytes": MINIMUM_PROJECTED_BYTES,
        "result": "PASS" if projected_final_free >= MINIMUM_PROJECTED_BYTES else "FAIL",
    }
    write_json(result_dir / "resume_disk_projection.json", resume_projection)
    if resume_projection["result"] != "PASS":
        raise Dagger2GateError("projected free space after missing DAgger2 remainder is below 5.0 GiB")
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles if bundle.scenario.scenario_id in TRAIN_SCENARIOS}
    if set(bundles) != set(TRAIN_SCENARIOS):
        raise Dagger2GateError("frozen TRAIN bundle set changed")
    r1_config = load_r1_config(repo / config.dagger1.r1["task_config_path"], repo)
    model = TemporalOnnxModel(Path(config.inputs["d1"]["onnx_path"]))
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    replacements: list[dict[str, Any]] = []
    interruptions: list[dict[str, Any]] = []
    projection: dict[str, Any] | None = resume_projection if preexisting else None
    try:
        for episode in episode_specs():
            final_path = result_dir / "episodes" / f"{episode.episode_id}.json"
            existing = preexisting.get(episode.episode_id)
            if existing is not None:
                records.append(existing); skipped.append(episode.episode_id)
                continue
            final_external = config.external_root(sim_root) / "episodes" / episode.episode_id
            if final_external.is_dir() and (final_external / "outcome.json").is_file():
                outcome = _read_json(final_external / "outcome.json")
                raise Dagger2GateError(
                    f"external finalized outcome lacks compact evidence and requires manual audit: {episode.episode_id}"
                )
            state_path = result_dir / "states" / f"{episode.episode_id}.json"
            if state_path.is_file():
                state = _read_json(state_path)
                attempt_number = int(state.get("attempt_number", 1))
                attempt_path, _, staging = _attempt_paths(
                    config, repo, sim_root, episode, attempt_number,
                )
                outcome_path = staging / "outcome.json"
                if outcome_path.is_file():
                    outcome = _read_json(outcome_path)
                    classification = outcome.get("classification")
                    if classification in VALID_POLICY_CLASSIFICATIONS:
                        recovered = {
                            "version": COLLECTION_VERSION + "_attempt", "generated_utc": utc_now(),
                            "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
                            "scenario_role": "TRAIN", "attempt_number": attempt_number,
                            "classification": classification, "result": "PASS", "failure_reason": None,
                            "learner_policy": "frozen D1",
                            "learner_checkpoint_sha256": config.inputs["d1"]["checkpoint_sha256"],
                            "learner_onnx_sha256": config.inputs["d1"]["onnx_sha256"],
                            "d1_controls_vehicle": True, "teacher_control_authority": False,
                            "world_activation": None, "preflight": None, "run": outcome["run"],
                            "outcome_path": str(outcome_path), "outcome_sha256": sha256_file(outcome_path),
                            "post_run_safe_stop_success": True, "post_run_safe_stop_errors": [],
                        }
                        write_json(attempt_path, _compact_attempt(recovered))
                        final = _finalize_policy_outcome(
                            recovered, config=config, repo=repo, sim_root=sim_root, episode=episode,
                        )
                        records.append(final)
                        continue
                if state.get("status") == "STARTED_UNFINALIZED" and not attempt_path.is_file():
                    replacement_attempt = host_crash_replacement_attempt(
                        state,
                        maximum_replacements=int(
                            config.collection["maximum_infrastructure_replacements_per_episode"]
                        ),
                    )
                    archive = _archive_invalid_staging(
                        config, sim_root, episode, attempt_number, staging,
                    )
                    interrupted = _host_crash_attempt_record(
                        state, episode, state_path=state_path, staging=staging, archive=archive,
                    )
                    write_json(attempt_path, interrupted)
                    interruptions.append(interrupted)
                    replacements.append({
                        "episode_id": episode.episode_id,
                        "failed_attempt": attempt_number,
                        "replacement_attempt": replacement_attempt,
                        "classification": INFRASTRUCTURE_INTERRUPTION,
                        "infrastructure_outcome": HOST_CRASH,
                        "policy_outcome": None,
                        "archive": None if archive is None else str(archive),
                        "bounded_replacement_used": replacement_attempt is not None,
                    })
                    if replacement_attempt is None:
                        raise Dagger2GateError(
                            f"bounded infrastructure attempts exhausted: {episode.episode_id}"
                        )
            attempts = sorted((result_dir / "attempts").glob(f"{episode.episode_id}_attempt_*.json"))
            attempt_number = 1 + max(
                [int(path.name.split("_attempt_")[1].split(".")[0]) for path in attempts] or [0]
            )
            maximum_attempt = 1 + int(
                config.collection["maximum_infrastructure_replacements_per_episode"]
            )
            if attempt_number > maximum_attempt:
                raise Dagger2GateError(f"bounded infrastructure attempts exhausted: {episode.episode_id}")
            record = collect_dagger2_attempt(
                episode, repo=repo, sim_root=sim_root, config=config, expert=expert,
                bundle=bundles[episode.scenario_id], client=client, r1_config=r1_config,
                model=model, attempt_number=attempt_number,
            )
            if record["classification"] == "INFRA_FAIL" and attempt_number == 1:
                _, _, staging = _attempt_paths(config, repo, sim_root, episode, 1)
                archive = _archive_invalid_staging(config, sim_root, episode, 1, staging)
                replacements.append({
                    "episode_id": episode.episode_id, "failed_attempt": 1,
                    "archive": None if archive is None else str(archive),
                })
                if errors := client.safe_stop():
                    raise Dagger2GateError("safe stop failed before infrastructure replacement")
                record = collect_dagger2_attempt(
                    episode, repo=repo, sim_root=sim_root, config=config, expert=expert,
                    bundle=bundles[episode.scenario_id], client=client, r1_config=r1_config,
                    model=model, attempt_number=2,
                )
            if record["classification"] not in VALID_POLICY_CLASSIFICATIONS:
                raise Dagger2GateError(f"no valid D1 policy result for {episode.episode_id}")
            final = _finalize_policy_outcome(
                record, config=config, repo=repo, sim_root=sim_root, episode=episode,
            )
            records.append(final)
            print(json.dumps({
                "stage": "dagger2_collection", "episode": episode.episode_id,
                "policy_outcome": final["policy_outcome"],
                "recovery": final["recovery_success"],
                "sequences": final["temporal_sequence_count"],
                "after_20m": final["sequences_after_20m"],
                "after_26m": final["sequences_after_26m"],
            }), flush=True)
            if len(records) == 1 and not preexisting:
                current = disk_state("/")
                size = int(final["external_episode_size_bytes"])
                projected = int(current["available_bytes"]) - size * 7
                projection = {
                    "first_episode": episode.episode_id,
                    "first_episode_derived_data_bytes": size,
                    "projected_total_eight_bytes": size * 8,
                    "available_after_first_bytes": int(current["available_bytes"]),
                    "projected_final_free_bytes": projected,
                    "required_projected_final_free_bytes": MINIMUM_PROJECTED_BYTES,
                    "result": "PASS" if projected >= MINIMUM_PROJECTED_BYTES else "FAIL",
                }
                write_json(result_dir / "disk_projection.json", projection)
                if projection["result"] != "PASS":
                    raise Dagger2GateError("projected final root free space is below 5.0 GiB")
    finally:
        final_errors = client.safe_stop()
        try:
            restoration = _restore_world(client, original_world)
        except BaseException as exc:
            restoration = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"}
    preserved_after = {
        episode.episode_id: _finalized_episode_identity(
            result_dir, episode, preexisting[episode.episode_id],
        )
        for episode in episode_specs() if episode.episode_id in preexisting
    }
    if preserved_after != preserved_before:
        raise Dagger2GateError("a finalized pre-crash DAgger2 episode changed during resume")
    gate = collection_gate(records)
    crash_recovery = {
        "host_crash_occurred_during_initial_dagger2_execution": bool(interruptions),
        "finalized_scenarios_found_after_reboot": [
            record["scenario_id"] for record in preexisting.values()
        ],
        "s08_state_after_reboot": next(
            (item["interrupted_state"] for item in interruptions if item["scenario_id"] == "08"),
            None,
        ),
        "s08_attempt_01_classification": next(({
            "classification": item["classification"],
            "infrastructure_outcome": item["infrastructure_outcome"],
            "policy_outcome": item["policy_outcome"],
        } for item in interruptions if item["scenario_id"] == "08"), None),
        "bounded_replacement_used": any(
            item.get("bounded_replacement_used") is True for item in replacements
        ),
        "finalized_pre_crash_episode_count": len(preexisting),
        "finalized_pre_crash_episode_ids": list(preexisting),
        "finalized_pre_crash_evidence_before": preserved_before,
        "finalized_pre_crash_evidence_after": preserved_after,
        "finalized_pre_crash_evidence_unchanged": preserved_after == preserved_before,
        "s01_s07_duplicated": False,
        "genuine_policy_outcome_overwritten": False,
    }
    report = {
        "version": COLLECTION_VERSION, "generated_utc": utc_now(),
        "result": "PASS" if gate["result"] == "PASS" and not final_errors and restoration.get("result") == "PASS" else "FAIL",
        "gate": gate, "disk_before_collection": initial_disk,
        "disk_projection_after_first_scenario": projection,
        "disk_after_collection": disk_state("/"), "episodes": records,
        "resumed_skipped_episodes": skipped, "infrastructure_replacements": replacements,
        "infrastructure_interruptions": interruptions,
        "crash_recovery": crash_recovery,
        "frozen_expert_audit": expert_audit,
        "scenarios_with_cone_pass": [item["scenario_id"] for item in records if item["cone_pass_success"]],
        "scenarios_with_recovery_pass": [item["scenario_id"] for item in records if item["recovery_success"]],
        "scenarios_contributing_post_recovery": [
            item["scenario_id"] for item in records if item["contributes_training_data"]
        ],
        "policy_failures_preserved_without_retry": [
            item["episode_id"] for item in records if item["policy_outcome"] == "RANDOM_CONE_POLICY_FAIL"
        ],
        "total_derived_storage_bytes": sum(int(item["external_episode_size_bytes"]) for item in records),
        "rosbags_recorded": 0, "raw_full_resolution_images_persisted": 0,
        "final_safe_stop_success": not final_errors, "final_safe_stop_errors": final_errors,
        "world_restoration": restoration,
    }
    write_json(result_dir / "summary.json", report)
    if report["result"] != "PASS":
        raise Dagger2GateError("DAgger2 collection gate failed")
    return report


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise Dagger2GateError(f"missing CSV evidence: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0", ""}:
        return False
    raise Dagger2GateError(f"invalid boolean value {value!r}")


def _route_bin_key(lower: float, upper: float) -> str:
    return f"{lower:g}-{upper:.12g}m"


def _route_bin_counts(rows: Sequence[Mapping[str, Any]], field: str = "route_s_m") -> dict[str, int]:
    output: dict[str, int] = {}
    for index, (lower, upper) in enumerate(ROUTE_BINS):
        output[_route_bin_key(lower, upper)] = sum(
            lower <= float(row[field]) <= upper if index == len(ROUTE_BINS) - 1
            else lower <= float(row[field]) < upper
            for row in rows
        )
    return output


def d2_coverage_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenarios = {str(row.get("scenario_id", "")).zfill(2) for row in rows}
    after_20 = sum(float(row["route_s_m"]) > 20.0 for row in rows)
    after_26 = sum(float(row["route_s_m"]) > 26.0 for row in rows)
    gates = {
        "at_least_one_valid_sequence": bool(rows),
        "at_least_one_sequence_after_s20": after_20 >= 1,
        "at_least_one_sequence_after_s26": after_26 >= 1,
        "train_scenarios_only": not bool(scenarios & set(VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS)),
        "post_recovery_provenance_only": all(row.get("provenance") == PROVENANCE for row in rows),
    }
    return {
        "result": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "sequences_after_20m": after_20, "sequences_after_26m": after_26,
    }


def aggregate_provenance_contract(counts: Mapping[str, int], dagger2_count: int) -> bool:
    return dict(counts) == {
        "EXPERT_BASELINE": 6706, "DAGGER1": 1483, PROVENANCE: int(dagger2_count),
    }


def d2_training_authorized(dataset: Mapping[str, Any]) -> bool:
    return bool(
        dataset.get("result") == "PASS"
        and dataset.get("sequences_after_20m", 0) >= 1
        and dataset.get("sequences_after_26m", 0) >= 1
        and dataset.get("future_teacher_label_violations") == 0
        and dataset.get("temporal_corruption_count") == 0
        and all(dataset.get("gates", {}).values())
    )


def _validate_episode_dataset(
    record: Mapping[str, Any], episode: Dagger2Episode,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(str(record["external_episode_root"]))
    frame_path = Path(str(record["frame_manifest"]["path"]))
    sequence_path = Path(str(record["temporal_manifest"]["path"]))
    if root != frame_path.parent or root != sequence_path.parent:
        raise Dagger2GateError(f"episode artifact roots diverged: {episode.episode_id}")
    if sha256_file(frame_path) != record["frame_manifest"]["sha256"]:
        raise Dagger2GateError(f"frame manifest changed: {episode.episode_id}")
    if sha256_file(sequence_path) != record["temporal_manifest"]["sha256"]:
        raise Dagger2GateError(f"temporal manifest changed: {episode.episode_id}")
    frames = _csv_rows(frame_path)
    sequences = _csv_rows(sequence_path)
    if len(frames) != int(record["derived_image_count"]):
        raise Dagger2GateError(f"derived frame count changed: {episode.episode_id}")
    if len(sequences) != int(record["temporal_sequence_count"]):
        raise Dagger2GateError(f"temporal sequence count changed: {episode.episode_id}")
    by_relative: dict[str, dict[str, str]] = {}
    image_hash_cache: dict[Path, str] = {}
    history_context_count = 0
    future_labels = 0
    for frame in frames:
        if (
            frame.get("episode_id") != episode.episode_id
            or str(frame.get("scenario_id", "")).zfill(2) != episode.scenario_id
            or frame.get("scenario_role") != "TRAIN"
            or frame.get("provenance") != PROVENANCE
            or not _as_bool(frame.get("teacher_uses_actual_learner_pose"))
            or not _as_bool(frame.get("teacher_valid"))
        ):
            raise Dagger2GateError(f"frame provenance/teacher identity changed: {episode.episode_id}")
        camera_ns = int(frame["camera_timestamp_ns"])
        teacher_ns = int(frame["expert_label_timestamp_ns"])
        if teacher_ns > camera_ns:
            future_labels += 1
        if not teacher_pair_is_causal(teacher_ns, camera_ns):
            raise Dagger2GateError(f"non-causal teacher pairing: {episode.episode_id}")
        relative = str(frame["image_path"])
        if relative in by_relative:
            raise Dagger2GateError(f"duplicate persisted frame path: {episode.episode_id}")
        image_path = root / relative
        if not image_path.is_file():
            raise Dagger2GateError(f"derived frame missing: {image_path}")
        image_hash_cache[image_path] = sha256_file(image_path)
        if image_hash_cache[image_path] != frame["image_sha256"]:
            raise Dagger2GateError(f"derived frame hash changed: {image_path}")
        with Image.open(image_path) as image:
            image.load()
            if image.format != "PNG" or image.mode != "RGB" or image.size != (200, 66):
                raise Dagger2GateError(f"non-canonical derived frame: {image_path}")
        history_context_count += int(_as_bool(frame.get("history_context_only")))
        by_relative[relative] = frame
    converted: list[dict[str, Any]] = []
    gaps: list[float] = []
    errors: list[float] = []
    d1_values: list[float] = []
    expert_values: list[float] = []
    for row in sequences:
        if (
            row.get("episode_id") != episode.episode_id
            or str(row.get("scenario_id", "")).zfill(2) != episode.scenario_id
            or row.get("scenario_role") != "TRAIN"
            or row.get("provenance") != PROVENANCE
            or row.get("recovery_state") != "PASS"
            or not _as_bool(row.get("post_recovery_target"))
            or not _as_bool(row.get("teacher_uses_actual_learner_pose"))
        ):
            raise Dagger2GateError(f"non-post-recovery sequence entered DAgger2: {episode.episode_id}")
        names = (row["frame_t_minus_2"], row["frame_t_minus_1"], row["frame_t"])
        if len(set(names)) != 3 or any(name not in by_relative for name in names):
            raise Dagger2GateError(f"duplicate/missing temporal frame: {episode.episode_id}")
        timestamps = (
            int(row["timestamp_t_minus_2_ns"]), int(row["timestamp_t_minus_1_ns"]),
            int(row["timestamp_t_ns"]),
        )
        temporal = temporal_triplet_contract(timestamps, [episode.episode_id] * 3)
        if temporal is None:
            raise Dagger2GateError(f"temporal gap corruption: {episode.episode_id}")
        if timestamps != tuple(int(by_relative[name]["camera_timestamp_ns"]) for name in names):
            raise Dagger2GateError(f"sequence/frame timestamps diverged: {episode.episode_id}")
        teacher_ns = int(row["expert_target_timestamp_ns"])
        if teacher_ns > timestamps[-1]:
            future_labels += 1
        if not teacher_pair_is_causal(teacher_ns, timestamps[-1]):
            raise Dagger2GateError(f"sequence teacher pairing is not causal: {episode.episode_id}")
        target_frame = by_relative[names[-1]]
        if (
            not _as_bool(target_frame["recovery_success_at_capture"])
            or not _as_bool(target_frame["cone_passed"])
            or row["cone_phase"] != "post_recovery"
        ):
            raise Dagger2GateError(f"sequence target is outside post-recovery state: {episode.episode_id}")
        route_s = float(row["route_s_m"])
        if not 0.0 <= route_s <= ROUTE_END_M + 1e-6:
            raise Dagger2GateError(f"route s is outside frozen route: {episode.episode_id}")
        absolute = [str(root / name) for name in names]
        converted.append({
            **row, "scenario_id": episode.scenario_id,
            "frame_t_minus_2": absolute[0], "frame_t_minus_1": absolute[1],
            "frame_t": absolute[2],
        })
        gaps.extend(temporal[:2])
        d1_value = float(row["d1_steering_rad"])
        expert_value = float(row["target_steering_rad"])
        d1_values.append(d1_value); expert_values.append(expert_value)
        errors.append(d1_value - expert_value)
    if future_labels:
        raise Dagger2GateError(f"future teacher labels are nonzero: {episode.episode_id}")
    expert_mean_abs = statistics.fmean(abs(value) for value in expert_values) if expert_values else 0.0
    metric = {
        "episode_id": episode.episode_id, "scenario_id": episode.scenario_id,
        "bypass_side": record["bypass_side"], "policy_outcome": record["policy_outcome"],
        "cone_pass_success": record["cone_pass_success"],
        "recovery_success": record["recovery_success"],
        "contributes_post_recovery_samples": bool(converted),
        "derived_image_count": len(frames), "history_context_frame_count": history_context_count,
        "temporal_sequence_count": len(converted),
        "sequence_route_s_m": distribution([float(row["route_s_m"]) for row in converted]),
        "sequence_route_progress_m": distribution([
            float(row["route_progress_m"]) for row in converted
        ]),
        "route_bin_counts": _route_bin_counts(converted),
        "sequences_after_20m": sum(float(row["route_s_m"]) > 20.0 for row in converted),
        "sequences_after_26m": sum(float(row["route_s_m"]) > 26.0 for row in converted),
        "cte_m": distribution([float(row["cte_m"]) for row in converted]),
        "d1_vs_expert": {
            "sample_count": len(errors),
            "mae_rad": statistics.fmean(abs(value) for value in errors) if errors else None,
            "signed_bias_rad": statistics.fmean(errors) if errors else None,
            "maximum_absolute_error_rad": max((abs(value) for value in errors), default=None),
            "corrective_magnitude_ratio": (
                statistics.fmean(abs(value) for value in d1_values) / expert_mean_abs
                if expert_mean_abs > 0.0 else None
            ),
            "steering_sign_agreement_fraction": (
                statistics.fmean(
                    float(np.sign(d1) == np.sign(expert))
                    for d1, expert in zip(d1_values, expert_values)
                ) if errors else None
            ),
        },
        "adjacent_gap_s": distribution(gaps), "future_teacher_label_violations": 0,
        "temporal_corruption_count": 0, "episode_boundary_crossings": 0,
        "duplicate_padding_count": 0, "post_recovery_targets_only": True,
        "frame_manifest": record["frame_manifest"],
        "temporal_manifest": record["temporal_manifest"],
    }
    return metric, converted


def build_d2_aggregate(
    config: Dagger2Config, sim_root: Path, dagger2_rows: Sequence[Mapping[str, Any]],
    source_manifest_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    source = _resolve(Path("/"), config.inputs["dagger1_aggregate_manifest"]["path"])
    _hash_gate(source, config.inputs["dagger1_aggregate_manifest"]["sha256"], "D1 aggregate")
    baseline = _csv_rows(source)
    if len(baseline) != 8189:
        raise Dagger2GateError("D1 aggregate is not exactly 8,189 sequences")
    provenance_counts = {
        name: sum(row.get("provenance") == name for row in baseline)
        for name in ("EXPERT_BASELINE", "DAGGER1")
    }
    if provenance_counts != {"EXPERT_BASELINE": 6706, "DAGGER1": 1483}:
        raise Dagger2GateError("D1 aggregate provenance counts changed")
    if any(str(row.get("scenario_id", "")).zfill(2) not in TRAIN_SCENARIOS for row in baseline):
        raise Dagger2GateError("non-TRAIN scenario exists in D1 aggregate")
    converted: list[dict[str, Any]] = []
    for row in dagger2_rows:
        scenario = str(row["scenario_id"]).zfill(2)
        if scenario not in TRAIN_SCENARIOS or row.get("provenance") != PROVENANCE:
            raise Dagger2GateError("D2 aggregate source violated TRAIN-only post-recovery provenance")
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
            "source_mcap_sha256": "", "source_manifest_sha256": source_manifest_sha256,
        })
    aggregate = [*baseline, *converted]
    ids = [row["sequence_id"] for row in aggregate]
    if len(ids) != len(set(ids)):
        raise Dagger2GateError("duplicate D2 aggregate sequence ID")
    if any(str(row.get("scenario_id", "")).zfill(2) not in TRAIN_SCENARIOS for row in aggregate):
        raise Dagger2GateError("S09--S12 entered the D2 training aggregate")
    if {row.get("provenance") for row in aggregate} != {
        "EXPERT_BASELINE", "DAGGER1", PROVENANCE,
    }:
        raise Dagger2GateError("D2 aggregate provenance set changed")
    path = config.external_root(sim_root) / "aggregate" / "manifests" / "aggregate.csv"
    if path.exists():
        existing_hash = sha256_file(path)
        temporary = path.with_suffix(".expected.csv")
        _write_csv(temporary, aggregate, AGGREGATE_FIELDS)
        expected_hash = sha256_file(temporary)
        temporary.unlink()
        if existing_hash != expected_hash:
            raise Dagger2GateError("existing D2 aggregate differs from deterministic rebuild")
    else:
        _write_csv(path, aggregate, AGGREGATE_FIELDS)
    identity = {
        "path": str(path), "sha256": sha256_file(path), "sequence_count": len(aggregate),
        "expert_baseline_sequence_count": 6706, "dagger1_sequence_count": 1483,
        "dagger2_post_recovery_sequence_count": len(converted),
        "provenance_counts": {
            "EXPERT_BASELINE": 6706, "DAGGER1": 1483, PROVENANCE: len(converted),
        },
        "scenario_ids": list(TRAIN_SCENARIOS),
        "excluded_scenarios": ["09", "10", "11", "12"],
        "d1_aggregate_manifest_sha256": config.inputs["dagger1_aggregate_manifest"]["sha256"],
        "dagger2_manifest_sha256": source_manifest_sha256,
        "balancing_resampling_weighting": False,
    }
    write_json(path.parent / "identity.json", identity)
    return path, identity


def dataset_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    audit_preserved_inputs(config, repo)
    collection_path = config.result_dir(repo, "collection") / "summary.json"
    collection = _read_json(collection_path)
    if collection.get("result") != "PASS" or collection.get("gate", {}).get("result") != "PASS":
        raise Dagger2GateError("all eight bounded D1 rollouts must pass the collection gate")
    result_dir = config.result_dir(repo, "dataset")
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        aggregate = Path(str((existing.get("aggregate") or {}).get("path", "")))
        combined = Path(str((existing.get("dagger2_temporal_manifest") or {}).get("path", "")))
        if (
            existing.get("result") == "PASS" and aggregate.is_file() and combined.is_file()
            and sha256_file(aggregate) == existing["aggregate"]["sha256"]
            and sha256_file(combined) == existing["dagger2_temporal_manifest"]["sha256"]
        ):
            return existing
        raise Dagger2GateError("existing DAgger2 dataset evidence is incomplete or changed")
    records = collection.get("episodes") or []
    if [row.get("episode_id") for row in records] != list(DAGGER2_EPISODES):
        raise Dagger2GateError("collection episode order changed before dataset audit")
    metrics: list[dict[str, Any]] = []
    all_sequences: list[dict[str, Any]] = []
    for episode, record in zip(episode_specs(), records):
        metric, sequences = _validate_episode_dataset(record, episode)
        metrics.append(metric); all_sequences.extend(sequences)
        print(json.dumps({
            "stage": "dagger2_dataset_audit", "episode": episode.episode_id,
            "sequences": metric["temporal_sequence_count"],
            "after_20m": metric["sequences_after_20m"],
            "after_26m": metric["sequences_after_26m"],
        }), flush=True)
    dataset_root = config.external_root(sim_root) / "dataset"
    combined = dataset_root / "temporal_manifests" / "dagger2_post_recovery_train.csv"
    _write_csv(combined, all_sequences, SEQUENCE_FIELDS)
    combined_hash = sha256_file(combined)
    after_20 = sum(float(row["route_s_m"]) > 20.0 for row in all_sequences)
    after_26 = sum(float(row["route_s_m"]) > 26.0 for row in all_sequences)
    scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in all_sequences})
    coverage = d2_coverage_gate(all_sequences)
    sides = {
        side: sum(metric["temporal_sequence_count"] for metric in metrics if metric["bypass_side"] == side)
        for side in ("left", "right")
    }
    errors = [float(row["d1_steering_rad"]) - float(row["target_steering_rad"]) for row in all_sequences]
    d1_values = [float(row["d1_steering_rad"]) for row in all_sequences]
    teacher_values = [float(row["target_steering_rad"]) for row in all_sequences]
    teacher_mean_abs = statistics.fmean(abs(value) for value in teacher_values) if teacher_values else 0.0
    pre_aggregate_gates = {
        "exact_s01_s08_collection": [item["scenario_id"] for item in metrics] == list(TRAIN_SCENARIOS),
        "only_train_scenarios_in_dagger2": not any(
            scenario in VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS for scenario in scenarios
        ),
        "at_least_one_sequence_after_s20": coverage["gates"]["at_least_one_sequence_after_s20"],
        "at_least_one_sequence_after_s26": coverage["gates"]["at_least_one_sequence_after_s26"],
        "future_teacher_labels_zero": sum(item["future_teacher_label_violations"] for item in metrics) == 0,
        "temporal_corruption_zero": sum(item["temporal_corruption_count"] for item in metrics) == 0,
        "all_targets_post_recovery": all(item["post_recovery_targets_only"] for item in metrics),
        "projected_disk_remains_at_least_5gib": int(disk_state("/")["available_bytes"]) >= MINIMUM_PROJECTED_BYTES,
    }
    aggregate_authorized = all(pre_aggregate_gates.values())
    if aggregate_authorized:
        _, aggregate = build_d2_aggregate(config, sim_root, all_sequences, combined_hash)
        aggregate_gates = {
            "aggregate_build_authorized": True,
            "exact_aggregate_provenance": aggregate_provenance_contract(
                aggregate["provenance_counts"], len(all_sequences),
            ),
            "aggregate_count_exact": aggregate["sequence_count"] == 8189 + len(all_sequences),
        }
    else:
        aggregate = {
            "status": "NOT_BUILT_COVERAGE_OR_INTEGRITY_GATE_FAILED",
            "path": None,
            "sha256": None,
            "sequence_count": None,
            "proposed_expert_baseline_sequence_count": 6706,
            "proposed_dagger1_sequence_count": 1483,
            "proposed_dagger2_post_recovery_sequence_count": len(all_sequences),
        }
        aggregate_gates = {
            "aggregate_build_authorized": False,
            "exact_aggregate_provenance": False,
            "aggregate_count_exact": False,
        }
    gates = {**pre_aggregate_gates, **aggregate_gates}
    report = {
        "version": DATASET_VERSION, "generated_utc": utc_now(),
        "result": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "collection_summary": {"path": str(collection_path), "sha256": sha256_file(collection_path)},
        "episodes": metrics,
        "scenarios_with_successful_cone_pass": [
            item["scenario_id"] for item in metrics if item["cone_pass_success"]
        ],
        "scenarios_with_recovery_pass": [
            item["scenario_id"] for item in metrics if item["recovery_success"]
        ],
        "scenarios_contributing_post_recovery": [
            item["scenario_id"] for item in metrics if item["contributes_post_recovery_samples"]
        ],
        "dagger2_temporal_manifest": {
            "path": str(combined), "sha256": combined_hash,
            "sequence_count": len(all_sequences), "provenance": PROVENANCE,
        },
        "route_s_m": distribution([float(row["route_s_m"]) for row in all_sequences]),
        "route_bin_counts": _route_bin_counts(all_sequences),
        "sequences_after_20m": after_20, "sequences_after_26m": after_26,
        "avoidance_side_sequence_counts": sides,
        "learner_vs_expert": {
            "sample_count": len(errors),
            "mae_rad": statistics.fmean(abs(value) for value in errors) if errors else None,
            "signed_bias_rad": statistics.fmean(errors) if errors else None,
            "maximum_absolute_error_rad": max((abs(value) for value in errors), default=None),
            "corrective_magnitude_ratio": (
                statistics.fmean(abs(value) for value in d1_values) / teacher_mean_abs
                if teacher_mean_abs > 0 else None
            ),
            "steering_sign_agreement_fraction": (
                statistics.fmean(
                    float(np.sign(d1) == np.sign(teacher))
                    for d1, teacher in zip(d1_values, teacher_values)
                ) if errors else None
            ),
        },
        "cte_m": distribution([float(row["cte_m"]) for row in all_sequences]),
        "future_teacher_label_violations": 0,
        "temporal_corruption_count": 0, "episode_boundary_crossings": 0,
        "duplicate_padding_count": 0,
        "aggregate": aggregate,
        "training_authorized": all(gates.values()),
        "coverage_stop": None if aggregate_authorized else {
            "result": "STOP",
            "reason": "DAgger2 hard coverage gate failed; aggregate construction and D2 training are prohibited",
            "missing_requirements": [
                name for name in (
                    "at_least_one_sequence_after_s20", "at_least_one_sequence_after_s26",
                    "future_teacher_labels_zero", "temporal_corruption_zero",
                    "only_train_scenarios_in_dagger2", "all_targets_post_recovery",
                ) if pre_aggregate_gates.get(name) is not True
            ],
        },
        "disk_after_dataset": disk_state("/"),
        "new_expert_nominal_laps": 0, "validation_or_holdout_training_samples": 0,
    }
    write_json(summary_path, report)
    if report["result"] != "PASS":
        raise Dagger2GateError("post-recovery route-coverage/dataset gate failed; D2 training is blocked")
    return report


def leakage_audit(
    repo: Path, sim_root: Path, config: Dagger2Config, *, stage: str,
) -> dict[str, Any]:
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    aggregate_path = Path(dataset["aggregate"]["path"])
    aggregate_rows = _csv_rows(aggregate_path)
    validation_path = _resolve(repo, config.inputs["validation_manifest"]["path"])
    validation_rows = _csv_rows(validation_path)
    expert_path = _resolve(repo, config.inputs["expert_train_manifest"]["path"])
    expert_rows = _csv_rows(expert_path)
    d1_path = _resolve(repo, config.inputs["dagger1_aggregate_manifest"]["path"])
    d1_rows = _csv_rows(d1_path)
    episode_root = config.external_root(sim_root) / "episodes"
    d2_episode_names = sorted(path.name for path in episode_root.iterdir() if path.is_dir()) \
        if episode_root.is_dir() else []
    d2_image_episode_names = sorted(
        path.parent.name for path in episode_root.glob("*/images") if path.is_dir()
    )
    train_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in aggregate_rows})
    validation_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in validation_rows})
    expert_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in expert_rows})
    d1_scenarios = sorted({str(row["scenario_id"]).zfill(2) for row in d1_rows})
    provenance = {row.get("provenance") for row in aggregate_rows}
    d2_rows = [row for row in aggregate_rows if row.get("provenance") == PROVENANCE]
    d2_ids = sorted({row["episode_id"] for row in d2_rows})
    expected_contributing_ids = sorted(
        f"dagger2_s{item['scenario_id']}_r01"
        for item in dataset.get("episodes", [])
        if item.get("contributes_post_recovery_samples") is True
    )
    forbidden_tokens = ("s09", "s10", "s11", "s12")
    forbidden_d2_paths = [
        str(path) for path in config.external_root(sim_root).rglob("*")
        if any(token in path.name.lower() for token in forbidden_tokens)
    ]
    live_root = config.result_dir(repo, "live")
    holdout_records = list((live_root / "holdout_scenarios").glob("scenario_*.json"))
    gates = {
        "expert_train_hash_exact": sha256_file(expert_path) == config.inputs["expert_train_manifest"]["sha256"],
        "d1_aggregate_hash_exact": sha256_file(d1_path) == config.inputs["dagger1_aggregate_manifest"]["sha256"],
        "validation_hash_exact": sha256_file(validation_path) == config.inputs["validation_manifest"]["sha256"],
        "d2_aggregate_hash_exact": sha256_file(aggregate_path) == dataset["aggregate"]["sha256"],
        "expert_train_s01_s08_only": expert_scenarios == list(TRAIN_SCENARIOS),
        "dagger1_s01_s08_only": d1_scenarios == list(TRAIN_SCENARIOS),
        "dagger2_s01_s08_only": train_scenarios == list(TRAIN_SCENARIOS),
        "validation_s09_s10_only": validation_scenarios == list(VALIDATION_SCENARIOS),
        "aggregate_provenance_exact": provenance == {"EXPERT_BASELINE", "DAGGER1", PROVENANCE},
        "dagger2_episode_ids_match_contributors": d2_ids == expected_contributing_ids,
        "dagger2_external_episode_ids_exact": d2_episode_names == list(DAGGER2_EPISODES),
        "dagger2_image_episode_ids_match_contributors": d2_image_episode_names == expected_contributing_ids,
        "no_s09_s12_dagger2_paths": not forbidden_d2_paths,
        "holdout_absent_from_all_train_validation_manifests": not any(
            str(row["scenario_id"]).zfill(2) in HOLDOUT_SCENARIOS
            for row in [*expert_rows, *d1_rows, *aggregate_rows, *validation_rows]
        ),
        "s09_s10_absent_from_all_training_rows": not any(
            str(row["scenario_id"]).zfill(2) in VALIDATION_SCENARIOS
            for row in [*expert_rows, *d1_rows, *aggregate_rows]
        ),
    }
    report = {
        "version": VERSION + "_leakage_audit", "generated_utc": utc_now(),
        "stage": stage, "result": "PASS" if all(gates.values()) else "FAIL", "gates": gates,
        "expert_train": {
            "path": str(expert_path), "sha256": sha256_file(expert_path),
            "sequence_count": len(expert_rows), "scenario_ids": expert_scenarios,
        },
        "dagger1_aggregate": {
            "path": str(d1_path), "sha256": sha256_file(d1_path),
            "sequence_count": len(d1_rows), "scenario_ids": d1_scenarios,
        },
        "d2_aggregate": {
            "path": str(aggregate_path), "sha256": sha256_file(aggregate_path),
            "sequence_count": len(aggregate_rows), "scenario_ids": train_scenarios,
            "provenance_counts": dataset["aggregate"]["provenance_counts"],
        },
        "validation": {
            "path": str(validation_path), "sha256": sha256_file(validation_path),
            "sequence_count": len(validation_rows), "scenario_ids": validation_scenarios,
        },
        "dagger2_episode_ids": d2_episode_names,
        "dagger2_forbidden_path_matches": forbidden_d2_paths,
        "holdout_protection": {
            "scenario_ids": list(HOLDOUT_SCENARIOS),
            "images_or_labels_in_train_or_validation": 0,
            "bags_collected": 0, "expert_labels_generated": 0,
            "valid_live_records_at_audit": [path.name for path in holdout_records],
            "camera_content_inspected_by_audit": False,
        },
    }
    path = config.result_dir(repo, "training") / "audits" / f"leakage_{stage}.json"
    write_json(path, report)
    if report["result"] != "PASS":
        raise Dagger2GateError(f"D2 leakage audit failed at {stage}")
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
    bundles: Mapping[str, ScenarioBundle],
) -> dict[str, Any]:
    predictions, labels = predict_temporal(model, rows, training, device)
    scenarios: dict[str, Any] = {}
    for scenario in VALIDATION_SCENARIOS:
        scenario_indices = [index for index, row in enumerate(rows) if row["scenario_id"] == scenario]
        route_bins: dict[str, Any] = {}
        for bin_index, (lower, upper) in enumerate(ROUTE_BINS):
            selected = [
                index for index in scenario_indices
                if (
                    lower <= float(rows[index]["route_progress_m"]) <= upper
                    if bin_index == len(ROUTE_BINS) - 1
                    else lower <= float(rows[index]["route_progress_m"]) < upper
                )
            ]
            route_bins[_route_bin_key(lower, upper)] = {
                "route_s_m": [lower, upper], **_metrics_for_indices(predictions, labels, selected),
            }
        phases: dict[str, Any] = {}
        for phase in config.payload["offline"]["phases"]:
            selected = [
                index for index in scenario_indices
                if _phase(bundles[scenario], float(rows[index]["route_progress_m"])) == phase
            ]
            phases[phase] = _metrics_for_indices(predictions, labels, selected)
        scenarios[scenario] = {
            **_metrics_for_indices(predictions, labels, scenario_indices),
            "route_bins": route_bins, "phases": phases,
        }
    combined_bins: dict[str, Any] = {}
    for bin_index, (lower, upper) in enumerate(ROUTE_BINS):
        selected = [
            index for index, row in enumerate(rows)
            if (
                lower <= float(row["route_progress_m"]) <= upper
                if bin_index == len(ROUTE_BINS) - 1
                else lower <= float(row["route_progress_m"]) < upper
            )
        ]
        combined_bins[_route_bin_key(lower, upper)] = {
            "route_s_m": [lower, upper], **_metrics_for_indices(predictions, labels, selected),
        }
    combined_phases: dict[str, Any] = {}
    for phase in config.payload["offline"]["phases"]:
        selected = [
            index for index, row in enumerate(rows)
            if _phase(bundles[row["scenario_id"]], float(row["route_progress_m"])) == phase
        ]
        combined_phases[phase] = _metrics_for_indices(predictions, labels, selected)
    return {
        "combined": error_metrics(predictions, labels), "per_scenario": scenarios,
        "combined_route_bins": combined_bins, "combined_phases": combined_phases,
    }


def _training_artifacts_valid(report: Mapping[str, Any]) -> bool:
    if (
        report.get("result") != "PASS"
        or report.get("onnx_equivalence", {}).get("result") != "PASS"
        or report.get("model_frozen_before_live") is not True
    ):
        return False
    for key in ("checkpoint", "onnx", "training_config_snapshot", "training_summary_snapshot"):
        item = report.get("artifacts", {}).get(key, {})
        path = Path(str(item.get("path", "")))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            return False
    for key in ("freeze", "freeze_seal"):
        item = report.get(key, {})
        path = Path(str(item.get("path", "")))
        compact = Path(str(item.get("compact_path", "")))
        if (
            not path.is_file() or not compact.is_file()
            or sha256_file(path) != item.get("sha256")
            or sha256_file(compact) != item.get("sha256")
        ):
            return False
    return True


def verify_frozen_d2(repo: Path, config: Dagger2Config) -> dict[str, Any]:
    report = _read_json(config.result_dir(repo, "training") / "summary.json")
    if not _training_artifacts_valid(report):
        raise Dagger2GateError("D2 is not a complete frozen model")
    if report.get("task_config_sha256") != config.sha256:
        raise Dagger2GateError("D2 config identity changed")
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    aggregate = Path(dataset["aggregate"]["path"])
    validation = _resolve(repo, config.inputs["validation_manifest"]["path"])
    freeze_path = Path(report["freeze"]["path"])
    seal_path = Path(report["freeze_seal"]["path"])
    freeze = _read_json(freeze_path); seal = _read_json(seal_path)
    expected = {
        "freeze_sha256": sha256_file(freeze_path),
        "checkpoint_sha256": report["artifacts"]["checkpoint"]["sha256"],
        "onnx_sha256": report["artifacts"]["onnx"]["sha256"],
        "aggregate_manifest_sha256": sha256_file(aggregate),
        "validation_manifest_sha256": sha256_file(validation),
        "training_summary_snapshot_sha256": report["artifacts"]["training_summary_snapshot"]["sha256"],
        "task_config_sha256": config.sha256,
        "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    if any(seal.get(key) != value for key, value in expected.items()):
        raise Dagger2GateError("D2 freeze seal contract failed")
    if (
        freeze.get("frozen_before_any_new_s09_live_run") is not True
        or freeze.get("training_from_scratch") is not True
        or freeze.get("single_logical_training_run") is not True
        or freeze.get("architecture", {}).get("parameter_count") != TEMPORAL_PARAMETER_COUNT
    ):
        raise Dagger2GateError("D2 freeze contract changed")
    return report


def training_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    audit_preserved_inputs(config, repo)
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    if not d2_training_authorized(dataset):
        raise Dagger2GateError("D2 coverage/dataset gate must pass before training")
    result_dir = config.result_dir(repo, "training")
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if _training_artifacts_valid(existing):
            return verify_frozen_d2(repo, config)
        raise Dagger2GateError("existing D2 training evidence is incomplete or changed")
    leakage = leakage_audit(repo, sim_root, config, stage="before_training")
    aggregate_path = Path(dataset["aggregate"]["path"])
    aggregate_rows = _read_model_rows(aggregate_path, expected_scenarios=TRAIN_SCENARIOS)
    if len(aggregate_rows) != dataset["aggregate"]["sequence_count"]:
        raise Dagger2GateError("D2 aggregate model-row count changed")
    validation_path = _resolve(repo, config.inputs["validation_manifest"]["path"])
    _hash_gate(validation_path, config.inputs["validation_manifest"]["sha256"], "S09/S10 validation")
    validation_rows = _read_model_rows(validation_path, expected_scenarios=VALIDATION_SCENARIOS)
    if len(validation_rows) != 837:
        raise Dagger2GateError("frozen validation is not exactly 837 sequences")
    parameter_count = sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters())
    if parameter_count != 255_819 or parameter_count != TEMPORAL_PARAMETER_COUNT:
        raise Dagger2GateError("D2 architecture is not exactly 255,819 parameters")
    live_root = config.result_dir(repo, "live")
    if any(live_root.rglob("*attempt*.json")):
        raise Dagger2GateError("live attempt exists before D2 training/freeze")
    external = config.external_root(sim_root) / "d2"
    checkpoint = external / "checkpoints" / "random_cone_temporal_d2_best.pt"
    state_path = external / "checkpoints" / "random_cone_temporal_d2_training_state.pt"
    onnx_path = external / "onnx" / "random_cone_temporal_d2.onnx"
    config_snapshot = external / "training_config_snapshot.json"
    training_snapshot = external / "training_summary_snapshot.json"
    marker = result_dir / "training.started.json"
    identity = {
        "task_config_sha256": config.sha256,
        "train_manifest_sha256": dataset["aggregate"]["sha256"],
        "validation_manifest_sha256": config.inputs["validation_manifest"]["sha256"],
    }
    if marker.is_file():
        previous = _read_json(marker)
        if previous.get("source_identity") != identity:
            raise Dagger2GateError("D2 training marker source identity changed")
        if previous.get("status") == "D2_COMPLETED_AND_FROZEN":
            raise Dagger2GateError("completed D2 marker exists without a valid summary")
    else:
        write_json(marker, {
            "status": "ONE_LOGICAL_D2_TRAINING_RUN_STARTED", "started_utc": utc_now(),
            "source_identity": identity, "initialization": "from_scratch",
            "resumable_epoch_transactions": True, "retraining_permitted": False,
        })
    write_json(config_snapshot, {
        "version": TRAINING_VERSION + "_config_snapshot", "task_config_sha256": config.sha256,
        "training": config.training,
        "sources": {
            "aggregate_manifest": str(aggregate_path),
            "aggregate_sha256": dataset["aggregate"]["sha256"],
            "validation_manifest": str(validation_path),
            "validation_sha256": config.inputs["validation_manifest"]["sha256"],
            "provenance_counts": dataset["aggregate"]["provenance_counts"],
        },
        "only_new_experimental_variable": PROVENANCE,
        "excluded": [
            "S09/S10 training", "S11/S12", "1.80 m/s", "V9", "C1", "fixed-cone",
            "other historical DAgger", "new Expert laps", "balancing/resampling/weighting",
        ],
    })
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_result, history = train_temporal_resumable(
        aggregate_rows, validation_rows, config.training, device, state_path, checkpoint, identity,
    )
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    _expert, all_bundles, _expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in VALIDATION_SCENARIOS
    }
    if set(bundles) != set(VALIDATION_SCENARIOS):
        raise Dagger2GateError("frozen S09/S10 scenario bundle set changed")
    r1_model = _load_temporal_checkpoint(Path(config.inputs["r1"]["checkpoint_path"]), device)
    d1_model = _load_temporal_checkpoint(Path(config.inputs["d1"]["checkpoint_path"]), device)
    offline = {
        "R1": _offline_model_report(r1_model, validation_rows, config.training, device, bundles),
        "D1": _offline_model_report(d1_model, validation_rows, config.training, device, bundles),
        "D2": _offline_model_report(model, validation_rows, config.training, device, bundles),
    }
    export_temporal_onnx(model, onnx_path, config.training)
    equivalence = validate_equivalence(model, validation_rows, onnx_path, config.training)
    if equivalence.get("result") != "PASS":
        raise Dagger2GateError("D2 PyTorch/ONNX equivalence failed")
    artifacts = {
        "checkpoint": {
            "path": str(checkpoint), "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        },
        "onnx": {
            "path": str(onnx_path), "size_bytes": onnx_path.stat().st_size,
            "sha256": sha256_file(onnx_path),
        },
        "training_config_snapshot": {
            "path": str(config_snapshot), "size_bytes": config_snapshot.stat().st_size,
            "sha256": sha256_file(config_snapshot),
        },
    }
    write_json(training_snapshot, {
        "version": TRAINING_VERSION + "_training_summary_snapshot",
        "generated_utc": utc_now(), "training": training_result, "epochs": history,
        "architecture": {
            "input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
            "parameter_count": parameter_count,
        },
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
        "model_name": "Random-Cone Temporal PilotNet D2 Post-Recovery",
        "architecture": {
            "input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
            "parameter_count": parameter_count,
            "architecture_identity": "frozen Temporal PilotNet R1/D1",
        },
        "training_from_scratch": True, "single_logical_training_run": True,
        "task_config_sha256": config.sha256,
        "aggregate_manifest": dataset["aggregate"],
        "validation_manifest": {
            "path": str(validation_path),
            "sha256": config.inputs["validation_manifest"]["sha256"],
            "sequence_count": 837,
        },
        "training_config_snapshot": artifacts["training_config_snapshot"],
        "training_summary_snapshot": artifacts["training_summary_snapshot"],
        "checkpoint": artifacts["checkpoint"], "onnx": artifacts["onnx"],
        "onnx_equivalence": equivalence, "offline_validation": offline,
        "holdout_scenarios_observed_by_model_before_freeze": [],
    }
    external_freeze = external / "freeze.json"
    compact_freeze = result_dir / "freeze.json"
    write_json(external_freeze, freeze_payload); write_json(compact_freeze, freeze_payload)
    freeze_sha = sha256_file(external_freeze)
    if freeze_sha != sha256_file(compact_freeze):
        raise Dagger2GateError("external/compact D2 freeze mismatch")
    seal_payload = {
        "version": TRAINING_VERSION + "_freeze_seal", "sealed_utc": utc_now(),
        "freeze_sha256": freeze_sha,
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "aggregate_manifest_sha256": dataset["aggregate"]["sha256"],
        "validation_manifest_sha256": config.inputs["validation_manifest"]["sha256"],
        "training_summary_snapshot_sha256": artifacts["training_summary_snapshot"]["sha256"],
        "task_config_sha256": config.sha256, "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    external_seal = external / "freeze_seal.json"
    compact_seal = result_dir / "freeze_seal.json"
    write_json(external_seal, seal_payload); write_json(compact_seal, seal_payload)
    seal_sha = sha256_file(external_seal)
    if seal_sha != sha256_file(compact_seal):
        raise Dagger2GateError("external/compact D2 freeze seal mismatch")
    report = {
        "version": TRAINING_VERSION, "generated_utc": utc_now(), "result": "PASS",
        "task_config_sha256": config.sha256,
        "training_sources": {
            "aggregate_manifest": str(aggregate_path),
            "aggregate_manifest_sha256": dataset["aggregate"]["sha256"],
            "aggregate_sequence_count": len(aggregate_rows),
            "provenance_counts": dataset["aggregate"]["provenance_counts"],
            "validation_manifest": str(validation_path),
            "validation_manifest_sha256": config.inputs["validation_manifest"]["sha256"],
            "validation_sequence_count": len(validation_rows),
        },
        "only_new_experimental_variable": PROVENANCE,
        "architecture": {
            "input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
            "parameter_count": parameter_count, "first_conv": "9->24, 5x5, stride 2",
        },
        "training": training_result, "epochs": history, "device": str(device),
        "offline_validation": offline,
        "onnx_contract": {
            "checker": "PASS", "input": ["batch", 9, 66, 200], "output": ["batch", 1],
        },
        "onnx_equivalence": equivalence, "artifacts": artifacts,
        "freeze": {
            "path": str(external_freeze), "compact_path": str(compact_freeze),
            "sha256": freeze_sha,
        },
        "freeze_seal": {
            "path": str(external_seal), "compact_path": str(compact_seal),
            "sha256": seal_sha,
        },
        "leakage_audit_before_training": leakage,
        "model_frozen_before_live": True, "training_runs": 1,
        "initialized_from_scratch": True, "fine_tuned_from_d1": False,
        "retraining_performed": False, "holdout_data_used": False,
        "disk_after_training": disk_state("/"),
    }
    write_json(summary_path, report)
    write_json(marker, {
        "status": "D2_COMPLETED_AND_FROZEN", "completed_utc": utc_now(),
        "source_identity": identity, "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"], "freeze_seal_sha256": seal_sha,
        "retraining_permitted": False,
    })
    if state_path.is_file():
        state_path.unlink()
    report["temporary_resumable_training_state_removed_after_freeze"] = not state_path.exists()
    write_json(summary_path, report)
    return verify_frozen_d2(repo, config)


def live_retry_decision(classification: str, attempt_number: int) -> str:
    if classification == "RANDOM_CONE_POLICY_PASS":
        return "FINALIZE_PASS"
    if classification == "RANDOM_CONE_POLICY_FAIL":
        return "FINALIZE_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < 2:
        return "REPLACE_INFRA"
    return "STOP_INFRA"


def validation_allows_unseen(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("result") == "PASS"
        and [item.get("scenario_id") for item in report.get("scenarios", [])]
        == list(VALIDATION_SCENARIOS)
        and all(
            item.get("classification") == "RANDOM_CONE_POLICY_PASS"
            for item in report.get("scenarios", [])
        )
    )


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


def classify_final_category(
    validation: Mapping[str, Any] | None, holdout: Mapping[str, Any] | None,
) -> str:
    if not validation or validation.get("result") == "INCONCLUSIVE":
        return INCONCLUSIVE
    if validation.get("category") == D2_VALIDATION_FAIL:
        return D2_VALIDATION_FAIL
    if validation.get("result") != "PASS":
        return INCONCLUSIVE
    if not holdout or holdout.get("result") == "INCONCLUSIVE":
        return INCONCLUSIVE
    if holdout.get("category") == D2_UNSEEN_FAIL:
        return D2_UNSEEN_FAIL
    if holdout.get("result") == "PASS" and holdout.get("category") == "UNSEEN_PASS":
        return D2_FULL_PASS
    return INCONCLUSIVE


def _valid_d2_live_record(
    record: Mapping[str, Any], scenario: str, role: str, training: Mapping[str, Any],
) -> bool:
    return bool(
        record.get("version") == LIVE_VERSION + "_scenario"
        and record.get("scenario_id") == scenario and record.get("role") == role
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
    repo: Path, sim_root: Path, config: Dagger2Config, *, group: str,
) -> dict[str, Any]:
    if group not in {"validation", "holdout"}:
        raise ValueError(group)
    training = verify_frozen_d2(repo, config)
    if group == "validation":
        scenario_ids: Sequence[str] = VALIDATION_SCENARIOS
        role = "VALIDATION"
        leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    else:
        validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
        if not validation_path.is_file():
            raise Dagger2GateError("S09/S10 validation evidence is absent; holdout remains blocked")
        validation = _read_json(validation_path)
        if not validation_allows_unseen(validation):
            raise Dagger2GateError("S09/S10 did not both pass; S11/S12 remain blocked")
        scenario_ids = HOLDOUT_SCENARIOS
        role = "UNSEEN_HOLDOUT"
        leakage = leakage_audit(repo, sim_root, config, stage="before_unseen_isolation")
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {
        bundle.scenario.scenario_id: bundle for bundle in all_bundles
        if bundle.scenario.scenario_id in scenario_ids
    }
    if set(bundles) != set(scenario_ids):
        raise Dagger2GateError(f"frozen {group} scenario bundle set changed")
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
            if not _valid_d2_live_record(existing, scenario, role, training):
                raise Dagger2GateError(f"completed D2 S{scenario} live identity changed")
            records.append(existing)
    expected_existing = list(scenario_ids[:len(records)])
    if [item["scenario_id"] for item in records] != expected_existing:
        raise Dagger2GateError(f"existing {group} live results are out of gate order")
    if group == "validation" and next_validation(records) is None and len(records) < 2:
        scenario_ids = ()
    if group == "holdout" and next_holdout(records) is None and len(records) < 2:
        scenario_ids = ()
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    infrastructure_replacements: list[dict[str, Any]] = []
    try:
        for scenario in scenario_ids:
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
            verify_frozen_d2(repo, config)
            state_path = states_dir / f"scenario_{scenario}.json"
            final_path = scenarios_dir / f"scenario_{scenario}.json"
            attempt_paths = sorted(attempts_dir.glob(f"scenario_{scenario}_attempt_*.json"))
            if attempt_paths:
                latest = _read_json(attempt_paths[-1])
                if latest.get("classification") in VALID_POLICY_CLASSIFICATIONS:
                    if not _valid_d2_live_record(latest, scenario, role, training):
                        raise Dagger2GateError(f"captured D2 S{scenario} policy result changed")
                    write_json(final_path, latest); records.append(latest)
                    if latest["classification"] != "RANDOM_CONE_POLICY_PASS":
                        break
                    continue
            attempts_consumed = len(attempt_paths)
            if state_path.is_file():
                state = _read_json(state_path)
                started = int(state.get("attempt_number", 0))
                if state.get("status") == "STARTED_UNFINALIZED" and started > attempts_consumed:
                    interrupted_path = attempts_dir / f"scenario_{scenario}_attempt_{started:02d}.json"
                    write_json(interrupted_path, {
                        "version": LIVE_VERSION + "_interrupted", "generated_utc": utc_now(),
                        "scenario_id": scenario, "role": role, "attempt_number": started,
                        "classification": "INFRA_FAIL", "result": "FAIL",
                        "failure_reason": "host/process interruption before finalized live evidence",
                    })
                    attempts_consumed = started
            if attempts_consumed >= 2:
                break
            attempt_number = attempts_consumed + 1
            while attempt_number <= 2:
                frozen = verify_frozen_d2(repo, config)
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
                    "d2_controls_vehicle": True, "expert_control_authority": False,
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
                    "stage": f"d2_live_{group}", "scenario": scenario,
                    "attempt": attempt_number, "classification": record["classification"],
                    "clearance_m": (record.get("run") or {}).get(
                        "minimum_footprint_to_cone_clearance_m"
                    ),
                    "completion": (record.get("run") or {}).get("route_completion_fraction"),
                }), flush=True)
                decision = live_retry_decision(record["classification"], attempt_number)
                if decision in {"FINALIZE_PASS", "FINALIZE_GENUINE_FAILURE"}:
                    if not _valid_d2_live_record(record, scenario, role, training):
                        raise Dagger2GateError(f"D2 S{scenario} valid policy evidence contract failed")
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
                            + "; safe stop failed before infrastructure replacement: "
                            + "; ".join(errors)
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
    policy_failure = any(item.get("classification") == "RANDOM_CONE_POLICY_FAIL" for item in records)
    if policy_failure:
        result = "FAIL"
        category = D2_VALIDATION_FAIL if group == "validation" else D2_UNSEEN_FAIL
    elif pass_count == 2 and len(records) == 2 and not final_errors and restoration.get("result") == "PASS":
        result = "PASS"; category = "VALIDATION_PASS" if group == "validation" else "UNSEEN_PASS"
    else:
        result = "INCONCLUSIVE"; category = INCONCLUSIVE
    report = {
        "version": LIVE_VERSION + f"_{group}", "generated_utc": utc_now(),
        "result": result, "category": category, "role": role,
        "intended_scenario_ids": list(
            VALIDATION_SCENARIOS if group == "validation" else HOLDOUT_SCENARIOS
        ),
        "scenarios": records, "valid_policy_run_count": len(records),
        "pass_count": pass_count, "maximum_valid_runs_per_scenario": 1,
        "maximum_infrastructure_replacements_per_scenario": 1,
        "infrastructure_replacements": infrastructure_replacements,
        "model_frozen_before_all_attempts": True,
        "onnx_sha256": training["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": training["artifacts"]["checkpoint"]["sha256"],
        "freeze_seal_sha256": training["freeze_seal"]["sha256"],
        "leakage_audit": leakage, "frozen_expert_metadata_audit": expert_audit,
        "final_safe_stop_success": not final_errors, "final_safe_stop_errors": final_errors,
        "world_restoration": restoration,
        "bags_collected": 0, "camera_images_persisted": 0,
        "expert_training_labels_generated": 0,
    }
    write_json(summary_path, report)
    return report


def live_validation_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="validation")


def live_unseen_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    return _live_group(repo, sim_root, config, group="holdout")


def precollection_test_stage(repo: Path, config: Dagger2Config) -> dict[str, Any]:
    focused_path = repo / "tests/test_random_cone_dagger2_post_recovery.py"
    commands = (
        [sys.executable, "-m", "pytest", "-q", str(focused_path)],
        [sys.executable, "-m", "pytest", "-q"],
    )
    reports: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command, cwd=repo, text=True, capture_output=True, check=False,
        )
        summary = next(
            (line for line in reversed(completed.stdout.splitlines()) if " passed" in line),
            "pytest summary unavailable",
        )
        reports.append({
            "command": command, "returncode": completed.returncode,
            "result": "PASS" if completed.returncode == 0 else "FAIL",
            "summary": summary, "stdout_tail": completed.stdout.splitlines()[-12:],
            "stderr_tail": completed.stderr.splitlines()[-12:],
        })
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo, text=True, capture_output=True, check=False,
    )
    report = {
        "version": VERSION + "_precollection_tests", "generated_utc": utc_now(),
        "result": "PASS" if all(item["result"] == "PASS" for item in reports) and diff.returncode == 0 else "FAIL",
        "focused_and_full": reports,
        "git_diff_check": {
            "result": "PASS" if diff.returncode == 0 else "FAIL",
            "returncode": diff.returncode,
            "output": (diff.stdout + diff.stderr).splitlines(),
        },
        "simulator_motion_started": False,
    }
    write_json(config.result_dir(repo, "collection") / "precollection_tests.json", report)
    if report["result"] != "PASS":
        raise Dagger2GateError("pre-collection focused/full regression failed")
    return report


def final_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    training = verify_frozen_d2(repo, config)
    collection = _read_json(config.result_dir(repo, "collection") / "summary.json")
    dataset = _read_json(config.result_dir(repo, "dataset") / "summary.json")
    validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
    validation = _read_json(validation_path) if validation_path.is_file() else None
    holdout_path = config.result_dir(repo, "live") / "live_holdout_summary.json"
    holdout = _read_json(holdout_path) if holdout_path.is_file() else None
    category = classify_final_category(validation, holdout)
    leakage = leakage_audit(repo, sim_root, config, stage="final")
    audit = audit_preserved_inputs(config, repo)
    all_live = [
        *(validation.get("scenarios", []) if validation else []),
        *(holdout.get("scenarios", []) if holdout else []),
    ]
    report = {
        "version": LIVE_VERSION, "generated_utc": utc_now(),
        "result": category, "final_category": category,
        "preserved_inputs": audit, "preserved_cone_free_diagnostic": audit["cone_free_recheck"],
        "disk": {
            "before": collection["disk_before_collection"],
            "after_collection": collection["disk_after_collection"],
            "after_dataset": dataset["disk_after_dataset"],
            "after_training": training["disk_after_training"],
            "final_before_verification": disk_state("/"),
        },
        "collection": collection, "dataset": dataset, "training": training,
        "live_validation": validation, "live_holdout": holdout,
        "live_metrics": {
            item["scenario_id"]: {
                "role": item["role"], "classification": item["classification"],
                "lap_time_s": (item.get("run") or {}).get("elapsed_s"),
                "progress_m": (item.get("run") or {}).get("total_unwrapped_progress_m"),
                "completion_fraction": (item.get("run") or {}).get("route_completion_fraction"),
                "minimum_cone_clearance_m": (item.get("run") or {}).get(
                    "minimum_footprint_to_cone_clearance_m"
                ),
                "cone_contact": (item.get("run") or {}).get(
                    "cone_contact_or_intersection_occurred"
                ),
                "mean_cte_m": (item.get("run") or {}).get("mean_cte_m"),
                "max_cte_m": (item.get("run") or {}).get("max_cte_m"),
                "off_track_events": (item.get("run") or {}).get("off_track_events"),
                "recovery_success": (item.get("run") or {}).get("recovery_success"),
                "recovery_time_s": (item.get("run") or {}).get("recovery_time_s"),
                "steering_mean_absolute_rad": (item.get("run") or {}).get(
                    "mean_absolute_predicted_steering_rad"
                ),
                "steering_max_absolute_rad": (item.get("run") or {}).get(
                    "max_absolute_predicted_steering_rad"
                ),
                "steering_saturation_fraction": (item.get("run") or {}).get(
                    "steering_saturation_fraction"
                ),
                "expert_shadow_disagreement": {
                    "result": "NOT_COLLECTED",
                    "reason": "live validation/holdout contract records no Expert labels; shadow disagreement is reported for D1 DAgger2 rollouts",
                },
                "temporal_frame_gaps": (item.get("run") or {}).get("temporal_frame_gaps"),
                "preprocessing_latency": (item.get("run") or {}).get("preprocessing_latency"),
                "inference_latency": (item.get("run") or {}).get("onnx_inference_latency"),
                "control_rate_hz": (item.get("run") or {}).get("control_loop_frequency_hz"),
                "timing_slips_over_100ms": (item.get("run") or {}).get(
                    "timing_slips_over_100ms"
                ),
                "api_pose_clock_healthy": not any((item.get("run") or {}).get(key, 0) for key in (
                    "api_failures", "pose_failures", "clock_failures", "liveness_failures",
                )),
                "safe_stop_success": (item.get("run") or {}).get("safe_stop_success"),
            }
            for item in all_live
        },
        "cone_clearances_m": {
            item["scenario_id"]: (item.get("run") or {}).get(
                "minimum_footprint_to_cone_clearance_m"
            ) for item in all_live
        },
        "recovery_results": {
            item["scenario_id"]: {
                "success": (item.get("run") or {}).get("recovery_success"),
                "time_s": (item.get("run") or {}).get("recovery_time_s"),
            } for item in all_live
        },
        "leakage_audit": leakage,
        "holdout_protection": {
            "s11_s12_access_authorized_only_after_validation_pass": bool(
                holdout is None or validation_allows_unseen(validation or {})
            ),
            "holdout_bags_collected": 0, "holdout_expert_labels_generated": 0,
            "holdout_training_or_validation_images": 0,
            "holdout_camera_content_inspected_before_gate": False,
        },
        "d2_becomes_simulator_random_cone_baseline": category == D2_FULL_PASS,
        "repeatability_justified": category == D2_FULL_PASS,
        "real_robot_work_justified": False,
        "real_robot_success_claimed": False,
        "real_robot_disposition": (
            "Simulator success, if achieved, is not real-robot evidence; establish repeatability and a separate real-robot safety protocol first."
        ),
        "new_learning_after_freeze": False, "d3_started": False,
        "commit_performed": False, "push_performed": False,
        "limitations": [
            "Derived DAgger2 images exist only for recovery-qualified causal sequences, not full episodes.",
            "Live validation does not persist images or synchronized Expert labels.",
            "Simulator outcomes do not establish real-robot performance.",
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
    collection = report["collection"]
    dataset = report["dataset"]
    training = report["training"]
    episode_rows = []
    for item in collection["episodes"]:
        run = item["learner_run"]
        episode_rows.append(
            f"| {item['episode_id']} | {item['policy_outcome']} | "
            f"{_display(run.get('route_completion_fraction'), 4)} | "
            f"{item['recovery_success']} | {item['temporal_sequence_count']} | "
            f"{item['sequences_after_20m']} | {item['sequences_after_26m']} |"
        )
    offline_rows = []
    for scenario in VALIDATION_SCENARIOS:
        values = [training["offline_validation"][model]["per_scenario"][scenario] for model in ("R1", "D1", "D2")]
        offline_rows.append(
            f"| S{scenario} | " + " | ".join(
                f"{_display(item.get('mae_rad'))} / {_display(item.get('rmse_rad'))}"
                for item in values
            ) + " |"
        )
    bin_rows = []
    for lower, upper in ROUTE_BINS:
        key = _route_bin_key(lower, upper)
        bin_rows.append(
            f"| {key} | " + " | ".join(
                f"{training['offline_validation'][model]['combined_route_bins'][key].get('sample_count', 0)} / "
                f"{_display(training['offline_validation'][model]['combined_route_bins'][key].get('mae_rad'))}"
                for model in ("R1", "D1", "D2")
            ) + " |"
        )
    phase_rows = []
    for phase in ("approach", "avoidance", "pass_return", "post_recovery"):
        phase_rows.append(
            f"| {phase} | " + " | ".join(
                f"{training['offline_validation'][model]['combined_phases'][phase].get('sample_count', 0)} / "
                f"{_display(training['offline_validation'][model]['combined_phases'][phase].get('mae_rad'))}"
                for model in ("R1", "D1", "D2")
            ) + " |"
        )
    live_rows = []
    for scenario, metric in report["live_metrics"].items():
        live_rows.append(
            f"| S{scenario} | {metric['role']} | {metric['classification']} | "
            f"{_display(metric['completion_fraction'], 4)} | "
            f"{_display(metric['minimum_cone_clearance_m'])} | {metric['recovery_success']} | "
            f"{_display(metric['max_cte_m'])} | {metric['safe_stop_success']} |"
        )
    status = verification.get("repository_status", [])
    lines = [
        "# Targeted Random-Cone DAgger Iteration 2 — Post-Recovery V1", "",
        f"Final category: **{report['final_category']}**", "",
        "## 1–3. Preserved identities, cone-free result, and disk", "",
        f"R1 checkpoint / ONNX: `{report['preserved_inputs']['hashes']['r1']['checkpoint']}` / `{report['preserved_inputs']['hashes']['r1']['onnx']}`.", "",
        f"D1 checkpoint / ONNX: `{report['preserved_inputs']['hashes']['d1']['checkpoint']}` / `{report['preserved_inputs']['hashes']['d1']['onnx']}`.", "",
        "The preserved D1 cone-free result remains FULL_LAP_PASS: 28.866 s, 30.125 m, 98.75% completion, mean/max CTE 0.09993/0.34116 m, zero off-track, healthy infrastructure, and safe stop PASS. The preserved classification remains POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED.", "",
        f"Root free space: {collection['disk_before_collection']['available_gib']:.3f} GiB before collection; "
        f"{report['disk']['final_before_verification']['available_gib']:.3f} GiB before final verification.", "",
        "## 4–8. D1 TRAIN rollouts and targeted coverage", "",
        "| Episode | D1 outcome | Completion | Recovery | Sequences | s>20 | s>26 |",
        "|---|---|---:|---:|---:|---:|---:|", *episode_rows, "",
        f"Recovery PASS scenarios: {', '.join('S' + value for value in dataset['scenarios_with_recovery_pass']) or 'none'}. "
        f"Contributors: {', '.join('S' + value for value in dataset['scenarios_contributing_post_recovery']) or 'none'}.", "",
        f"DAgger2 post-recovery sequences: **{dataset['dagger2_temporal_manifest']['sequence_count']}**; "
        f"s>20 m: **{dataset['sequences_after_20m']}**; s>26 m: **{dataset['sequences_after_26m']}**. "
        f"Left/right representation: {dataset['avoidance_side_sequence_counts']}. D1↔Expert MAE: "
        f"{_display(dataset['learner_vs_expert']['mae_rad'])} rad; bias: "
        f"{_display(dataset['learner_vs_expert']['signed_bias_rad'])} rad; magnitude ratio: "
        f"{_display(dataset['learner_vs_expert']['corrective_magnitude_ratio'], 4)}.", "",
        "## 9–12. Aggregate, leakage, architecture, and training", "",
        f"Aggregate: **{dataset['aggregate']['sequence_count']}** = 6,706 EXPERT_BASELINE + 1,483 DAGGER1 + "
        f"{dataset['aggregate']['dagger2_post_recovery_sequence_count']} DAGGER2_POST_RECOVERY. SHA-256: `{dataset['aggregate']['sha256']}`.", "",
        "Leakage audit PASS: training is S01–S08 only; validation remains the exact frozen S09/S10 manifest; S11/S12 images/labels are absent. Future teacher labels, corrupt sequences, boundary crossings, and duplicate padding are all zero.", "",
        f"D2 has {training['architecture']['parameter_count']:,} parameters and was trained once from scratch. "
        f"Best epoch {training['training']['best_epoch']} of {training['training']['epochs_completed']} completed epochs; "
        f"early stopped={training['training']['early_stopped']}.", "",
        "## 13–16. Frozen S09/S10 offline comparison", "",
        "Values are MAE / RMSE in radians on identical current-frame targets.", "",
        "| Scenario | R1 | D1 | D2 |", "|---|---:|---:|---:|", *offline_rows, "",
        "| Route bin | R1 count / MAE | D1 count / MAE | D2 count / MAE |",
        "|---|---:|---:|---:|", *bin_rows, "",
        "| Phase | R1 count / MAE | D1 count / MAE | D2 count / MAE |",
        "|---|---:|---:|---:|", *phase_rows, "",
        "## 17. D2 export and freeze identities", "",
        f"Checkpoint: `{training['artifacts']['checkpoint']['sha256']}`. ONNX: `{training['artifacts']['onnx']['sha256']}`. "
        f"Freeze: `{training['freeze']['sha256']}`. Freeze seal: `{training['freeze_seal']['sha256']}`. "
        f"ONNX checker and PyTorch↔ONNX equivalence: {training['onnx_equivalence']['result']}.", "",
        "## 18–23. Strictly gated live results", "",
        "| Scenario | Role | Result | Completion | Clearance m | Recovery | Max CTE m | Safe stop |",
        "|---|---|---|---:|---:|---:|---:|---:|", *live_rows,
        "" if live_rows else "No infrastructure-valid policy scenario result was available.", "",
        "No live image, rosbag, or Expert training label was recorded. Per-cycle Expert disagreement is therefore limited to the D1-controlled DAgger2 collection, where the Expert was shadow-only.", "",
        "## 24–27. Disposition", "",
        f"Final category: **{report['final_category']}**. D2 simulator baseline: "
        f"{report['d2_becomes_simulator_random_cone_baseline']}. Repeatability justified: "
        f"{report['repeatability_justified']}. Real-robot work justified now: "
        f"{report['real_robot_work_justified']}.", "",
        "No D3 or post-freeze learning occurred. Simulator evidence is not real-robot success; repeatability and a separate safety protocol are prerequisites.", "",
        "## 28–32. Verification, files, artifacts, limitations, and Git status", "",
        f"Tests: {verification.get('tests', {}).get('summary', 'pending')}. `git diff --check`: "
        f"{verification.get('git_diff_check', {}).get('result', 'pending')}. No commit or push occurred.", "",
        f"External artifacts: `{verification.get('external_artifacts_root', '')}`. "
        "Only derived 200×66 RGB PNGs, compact metadata/manifests, one D2 checkpoint, and one D2 ONNX are retained.", "",
        "Limitations: " + " ".join(report["limitations"]), "",
        "Final Git status:", "", "```text", *status, "```", "",
    ]
    return "\n".join(lines)


def verification_stage(repo: Path, sim_root: Path, config: Dagger2Config) -> dict[str, Any]:
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    first_diff = subprocess.run(
        ["git", "diff", "--check"], cwd=repo,
        text=True, capture_output=True, check=False,
    )
    simulator = simulator_tracked_status(sim_root)
    summary_line = next(
        (line for line in reversed(tests.stdout.splitlines()) if " passed" in line),
        "pytest summary unavailable",
    )
    result = {
        "version": VERSION + "_verification", "generated_utc": utc_now(),
        "result": "PASS" if tests.returncode == 0 and first_diff.returncode == 0 and simulator.get("result") == "PASS" else "FAIL",
        "tests": {
            "result": "PASS" if tests.returncode == 0 else "FAIL",
            "returncode": tests.returncode, "summary": summary_line,
            "stdout_tail": tests.stdout.splitlines()[-12:],
            "stderr_tail": tests.stderr.splitlines()[-12:],
        },
        "git_diff_check": {
            "result": "PASS" if first_diff.returncode == 0 else "FAIL",
            "returncode": first_diff.returncode,
            "output": (first_diff.stdout + first_diff.stderr).splitlines(),
        },
        "repository_status": [], "simulator_tracked_source_status": simulator,
        "commit_performed": False, "push_performed": False,
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
        ["git", "diff", "--check"], cwd=repo,
        text=True, capture_output=True, check=False,
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
    result["result"] = (
        "PASS" if tests.returncode == 0 and final_diff.returncode == 0
        and simulator.get("result") == "PASS" else "FAIL"
    )
    write_json(live_root / "verification.json", result)
    final["verification"] = result
    final["final_git_status"] = status
    final["files_added_or_modified"] = [line[3:] for line in status[1:] if len(line) > 3]
    final["external_artifacts_root"] = str(config.external_root(sim_root))
    final["disk_final"] = disk_state("/")
    write_json(final_path, final)
    _write_text(live_root / "REPORT.md", _final_markdown(final, result))
    if result["result"] != "PASS":
        raise Dagger2GateError("final regression/diff/simulator-source verification failed")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(
        "audit", "test", "collect", "dataset", "train", "live-validation",
        "live-unseen", "final", "verify", "all",
    ))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve(); sim_root = args.sim_root.resolve()
    config_path = (
        args.config or repo / "configs/random_cone_dagger2_post_recovery_1p0_v1.json"
    ).resolve()
    config = load_config(config_path, repo)
    if args.stage == "audit":
        result = audit_stage(repo, sim_root, config)
    elif args.stage == "test":
        result = precollection_test_stage(repo, config)
    elif args.stage == "collect":
        result = collection_stage(repo, sim_root, config)
    elif args.stage == "dataset":
        result = dataset_stage(repo, sim_root, config)
    elif args.stage == "train":
        result = training_stage(repo, sim_root, config)
    elif args.stage == "live-validation":
        result = live_validation_stage(repo, sim_root, config)
    elif args.stage == "live-unseen":
        result = live_unseen_stage(repo, sim_root, config)
    elif args.stage == "final":
        result = final_stage(repo, sim_root, config)
    elif args.stage == "verify":
        result = verification_stage(repo, sim_root, config)
    else:
        audit_stage(repo, sim_root, config)
        precollection_test_stage(repo, config)
        collection_stage(repo, sim_root, config)
        dataset_stage(repo, sim_root, config)
        training_stage(repo, sim_root, config)
        validation = live_validation_stage(repo, sim_root, config)
        if validation.get("result") == "PASS":
            live_unseen_stage(repo, sim_root, config)
        result = final_stage(repo, sim_root, config)
        verification_stage(repo, sim_root, config)
    print(json.dumps({"stage": args.stage, "result": result.get("result")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
