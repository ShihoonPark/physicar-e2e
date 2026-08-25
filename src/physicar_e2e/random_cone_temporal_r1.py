"""Final 1.0 m/s random-cone Temporal PilotNet R1 pipeline.

The pipeline deliberately keeps three boundaries explicit:

* S01--S08 are the immutable TRAIN manifest.
* S09--S10 are the only Expert bags and offline validation trajectories.
* S11--S12 are live-only, sequential, unseen holdouts.

Large bags, images, checkpoints, ONNX models, and plots live under simulator
userdata.  Only compact gates and metrics are written under ``results``.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from .dataset_extractor import (
    MANIFEST_COLUMNS,
    REJECTION_REASONS,
    canonical_json_bytes,
    extract_episode,
    load_config as load_extractor_config,
    numeric_distribution,
    prepare_output_root,
    sha256_file,
    steering_distribution,
)
from .high_speed_temporal import (
    TemporalDataset,
    TemporalOnnxModel,
    _epoch,
    export_temporal_onnx,
    metrics as error_metrics,
    predict_temporal,
    run_temporal_live,
    validate_equivalence,
)
from .pilotnet_inference import InferenceConfig
from .pilotnet_temporal import (
    MAX_ADJACENT_GAP_S,
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
)
from .pilotnet_training import GateFailure, set_reproducible_seed
from .random_cone_expert import (
    MAP_FAMILY,
    ROLE_IDS,
    RandomConeConfig,
    RandomConeObserver,
    ScenarioBundle,
    _restore_world,
    directory_file_manifest_sha256,
    run_random_cone_expert,
    simulator_tracked_status,
    verify_frozen_scenarios,
)
from .random_cone_train_data import (
    FRAME_MANIFEST_COLUMNS,
    REQUIRED_TOPICS,
    TEMPORAL_MANIFEST_COLUMNS,
    TaskConfig as TrainTaskConfig,
    _post_settle_preflight,
    attach_causal_route_s,
    audit_frozen_expert,
    create_maneuver_contact_sheet,
    disk_state,
    load_task_config as load_train_task_config,
    region_coverage,
    scan_odom_route_s,
    scenario_regions,
    write_csv,
)
from .rosbag_collector import (
    BagInfo,
    CollectorConfig,
    DockerRosBackend,
    RecorderHandle,
    directory_size,
    git_commit,
    verify_bag,
)
from .sim_client import SimClient


VERSION = "random_cone_temporal_r1_v1"
COLLECTION_VERSION = "random_cone_validation_collection_1p0_v1"
DATASET_VERSION = "random_cone_validation_dataset_1p0_v1"
TRAINING_VERSION = "pilotnet_training_r1_random_cone_1p0"
LIVE_VERSION = "pilotnet_e2e_r1_random_cone_1p0"
TRAIN_SCENARIOS = tuple(f"{number:02d}" for number in range(1, 9))
VALIDATION_SCENARIOS = ("09", "10")
HOLDOUT_SCENARIOS = ("11", "12")
VALIDATION_EPISODES = ("val_s09_r01", "val_s10_r01")
TRAIN_MANIFEST_SHA256 = "a9aaf25991cecbab3937deae545d392842007b228d8b8f571c519fba1772df73"
TRAIN_SEQUENCE_COUNT = 6706
MIN_COLLECTION_BYTES = int(6.5 * 1024**3)
MIN_TRAIN_BYTES = int(5.5 * 1024**3)
MIN_LIVE_BYTES = int(4.5 * 1024**3)


class R1GateError(RuntimeError):
    """A preregistered R1 gate failed."""


@dataclass(frozen=True)
class ValidationSpec:
    episode_id: str
    scenario_id: str
    repeat_id: str = "R01"
    role: str = "VALIDATION"


@dataclass(frozen=True)
class R1Config:
    path: Path
    payload: dict[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    @property
    def frozen_expert(self) -> dict[str, Any]:
        return self.payload["frozen_expert"]

    @property
    def frozen_train(self) -> dict[str, Any]:
        return self.payload["frozen_train"]

    @property
    def collection(self) -> dict[str, Any]:
        return self.payload["validation_collection"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.payload["validation_dataset"]

    @property
    def training(self) -> dict[str, Any]:
        return self.payload["training"]

    @property
    def live(self) -> dict[str, Any]:
        return self.payload["live_inference"]

    def result_dir(self, repo: Path, key: str) -> Path:
        return repo / self.payload["result_directories"][key]

    def external(self, sim_root: Path, key: str) -> Path:
        return sim_root / "userdata" / self.payload["external"][key]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R1GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise R1GateError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validation_specs() -> tuple[ValidationSpec, ...]:
    return tuple(
        ValidationSpec(f"val_s{scenario}_r01", scenario)
        for scenario in VALIDATION_SCENARIOS
    )


def load_config(path: Path, repo: Path) -> R1Config:
    payload = _read_json(path)
    required = {
        "version", "map_family", "frozen_expert", "frozen_train", "scenario_roles",
        "validation_collection", "validation_dataset", "training", "live_inference",
        "external", "result_directories", "permissions",
    }
    if set(payload) != required or payload["version"] != VERSION:
        raise R1GateError("R1 task version or top-level fields changed")
    if payload["map_family"] != MAP_FAMILY or payload["scenario_roles"] != ROLE_IDS:
        raise R1GateError("map family or frozen 8/2/2 roles changed")
    if payload["frozen_train"] != {
        "collection_summary_path": "results/random_cone_train_collection_1p0_v1/summary.json",
        "collection_summary_sha256": "265a20b02b5736025a4c500a29aa57181dd5530e91188a39786bc8bc8f399557",
        "dataset_summary_path": "results/random_cone_train_dataset_1p0_v1/summary.json",
        "dataset_summary_sha256": "7fa2baf5874e84db1340fa8341b641fadf6cf2e5d5f0cc0f8d29d0b3818c5511",
        "manifest_path": "/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/random_cone_1p0_v1/train_dataset/temporal_manifests/train.csv",
        "manifest_sha256": TRAIN_MANIFEST_SHA256,
        "sequence_count": TRAIN_SEQUENCE_COUNT,
        "scenario_ids": list(TRAIN_SCENARIOS),
        "repeat_ids": ["R01", "R02"],
    }:
        raise R1GateError("frozen TRAIN identity changed")
    for relative_key, hash_key in (
        ("collection_summary_path", "collection_summary_sha256"),
        ("dataset_summary_path", "dataset_summary_sha256"),
    ):
        if sha256_file(repo / payload["frozen_train"][relative_key]) != payload["frozen_train"][hash_key]:
            raise R1GateError(f"frozen TRAIN evidence changed: {relative_key}")
    expert = payload["frozen_expert"]
    for relative, expected in (
        (expert["config_path"], expert["config_sha256"]),
        (f"{expert['result_directory']}/summary.json", expert["summary_sha256"]),
        (f"{expert['result_directory']}/offline_geometry.json", expert["offline_geometry_sha256"]),
    ):
        if sha256_file(repo / relative) != expected:
            raise R1GateError(f"frozen Expert evidence changed: {relative}")
    if directory_file_manifest_sha256(repo / expert["result_directory"]) != expert["result_manifest_sha256"]:
        raise R1GateError("frozen Expert result tree changed")
    collection = payload["validation_collection"]
    if (
        tuple(collection["episode_order"]) != VALIDATION_EPISODES
        or tuple(collection["required_topics"]) != REQUIRED_TOPICS
        or collection["data_relative_root"] != "physicar_e2e/random_cone_1p0_v1/validation_raw"
        or collection["minimum_free_bytes_before_collection"] != MIN_COLLECTION_BYTES
        or collection["infrastructure_replacement_attempts_per_episode"] != 1
        or collection["retry_genuine_policy_failure"] is not False
    ):
        raise R1GateError("validation collection contract changed")
    dataset = payload["validation_dataset"]
    if (
        dataset["history_frames"], dataset["maximum_adjacent_gap_s"], dataset["causal_only"],
        dataset["allow_episode_boundary_crossing"], dataset["allow_reset_boundary_crossing"],
        dataset["allow_duplicate_padding"], dataset["minimum_free_bytes_before_training"],
    ) != (3, 0.120, True, False, False, False, MIN_TRAIN_BYTES):
        raise R1GateError("validation temporal contract changed")
    extractor = repo / dataset["canonical_extractor_config_path"]
    if sha256_file(extractor) != dataset["canonical_extractor_config_sha256"]:
        raise R1GateError("canonical extractor config changed")
    training = payload["training"]
    baseline = {
        "seed": 20260824, "input_channels": 9, "history_frames": 3,
        "maximum_adjacent_gap_s": 0.120, "optimizer": "Adam", "loss": "MSE",
        "learning_rate": 0.001, "batch_size": 64, "max_epochs": 35,
        "early_stopping_patience": 7, "minimum_improvement": 0.000001,
        "initialization": "from_scratch", "max_steering_rad": 0.349066,
    }
    if any(training.get(key) != value for key, value in baseline.items()):
        raise R1GateError("R1 baseline training semantics changed")
    if any(training.get(key) is not False for key in (
        "augmentation", "sample_weighting", "scenario_weighting", "oversampling",
        "undersampling", "hyperparameter_sweep",
    )):
        raise R1GateError("R1 balancing/tuning is prohibited")
    live = payload["live_inference"]
    if (
        live["camera_only_model_observation"] is not True
        or (live["history_frames"], live["input_channels"], live["maximum_adjacent_gap_s"])
        != (3, 9, 0.120)
        or live["duplicate_frame_padding"] is not False
        or (live["speed_mps"], live["control_frequency_hz"], live["max_steering_rad"])
        != (1.0, 15.0, 0.349066)
        or live["minimum_free_bytes_before_live"] != MIN_LIVE_BYTES
    ):
        raise R1GateError("R1 live temporal/control contract changed")
    permissions = payload["permissions"]
    if (
        permissions["train_recollection_permitted"] is not False
        or permissions["validation_bag_collection_permitted"] is not True
        or permissions["holdout_bag_collection_permitted"] is not False
        or permissions["holdout_label_extraction_permitted"] is not False
        or permissions["training_permitted_once"] is not True
        or permissions["fine_tuning_permitted"] is not False
        or permissions["retraining_after_validation_or_holdout_permitted"] is not False
        or permissions["commit_permitted"] is not False
        or permissions["push_permitted"] is not False
    ):
        raise R1GateError("R1 permission boundary changed")
    return R1Config(path.resolve(), payload)


def _read_temporal_csv(path: Path, dataset_root: Path, *, expected_role: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise R1GateError(f"missing temporal manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            times = tuple(int(raw[key]) for key in (
                "camera_timestamp_t_minus_2_ns", "camera_timestamp_t_minus_1_ns",
                "camera_timestamp_t_ns",
            ))
            if not times[0] < times[1] < times[2]:
                raise R1GateError("temporal manifest contains a non-causal sequence")
            if max(times[1] - times[0], times[2] - times[1]) > 120_000_000:
                raise R1GateError("temporal manifest exceeds the 0.120 s gap gate")
            if int(raw["steering_target_timestamp_ns"]) > times[2]:
                raise R1GateError("temporal manifest contains a future steering label")
            if raw["scenario_role"] != expected_role:
                raise R1GateError(f"unexpected temporal role {raw['scenario_role']}")
            paths = tuple(dataset_root / raw[key] for key in (
                "frame_t_minus_2", "frame_t_minus_1", "frame_t",
            ))
            if len(set(paths)) != 3 or not all(item.is_file() for item in paths):
                raise R1GateError("temporal manifest has padding or a missing image")
            rows.append({
                **raw, "paths": paths, "image_path": paths[2],
                "steering_rad": float(raw["target_steering_rad"]),
                "route_progress_m": float(raw["route_progress_m"]),
            })
    if not rows:
        raise R1GateError(f"empty temporal manifest: {path}")
    return rows


def audit_train_manifest(config: R1Config) -> dict[str, Any]:
    train = config.frozen_train
    path = Path(train["manifest_path"])
    observed_hash = sha256_file(path)
    if observed_hash != TRAIN_MANIFEST_SHA256:
        raise R1GateError(f"frozen TRAIN manifest hash mismatch: {observed_hash}")
    root = path.parents[1]
    rows = _read_temporal_csv(path, root, expected_role="TRAIN")
    scenarios = sorted({row["scenario_id"] for row in rows})
    repeats = sorted({row["repeat_id"] for row in rows})
    episodes = sorted({row["episode_id"] for row in rows})
    intended = sorted(f"train_s{s}_r{repeat}" for repeat in ("01", "02") for s in TRAIN_SCENARIOS)
    if len(rows) != TRAIN_SEQUENCE_COUNT or scenarios != list(TRAIN_SCENARIOS):
        raise R1GateError("frozen TRAIN count/scenario identity changed")
    if repeats != ["R01", "R02"] or episodes != intended:
        raise R1GateError("frozen TRAIN repeat/episode identity changed")
    if {row["scenario_role"] for row in rows} != {"TRAIN"}:
        raise R1GateError("non-TRAIN row entered frozen TRAIN manifest")
    source_hashes = sorted({row["source_mcap_sha256"] for row in rows})
    if len(source_hashes) != 16:
        raise R1GateError("frozen TRAIN does not contain exactly 16 source bags")
    return {
        "result": "PASS", "manifest": str(path), "manifest_sha256": observed_hash,
        "sequence_count": len(rows), "scenario_ids": scenarios, "repeat_ids": repeats,
        "episode_ids": episodes, "source_bag_hash_count": len(source_hashes),
        "future_label_violations": 0, "maximum_adjacent_gap_s": 0.120,
    }


def audit_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    if tuple(bundle.scenario.scenario_id for bundle in bundles) != tuple(f"{i:02d}" for i in range(1, 13)):
        raise R1GateError("frozen 12-scenario identity changed")
    train = audit_train_manifest(config)
    simulator = simulator_tracked_status(sim_root)
    if simulator.get("result") != "PASS":
        raise R1GateError("tracked simulator source has changed")
    disk = disk_state("/")
    result = "PASS" if disk["available_bytes"] >= MIN_COLLECTION_BYTES else "FAIL"
    report = {
        "version": VERSION + "_audit", "generated_utc": utc_now(), "result": result,
        "task_config": {"path": str(config.path.relative_to(repo)), "sha256": config.sha256},
        "frozen_expert_identity": expert_audit["frozen_expert"],
        "frozen_scenario_roles": ROLE_IDS, "frozen_train": train,
        "validation_episode_ids": list(VALIDATION_EPISODES),
        "holdout_live_only_ids": list(HOLDOUT_SCENARIOS),
        "disk_before_validation_collection": disk,
        "minimum_collection_free_bytes": MIN_COLLECTION_BYTES,
        "disk_gate_pass": result == "PASS", "simulator_tracked_source_status": simulator,
        "train_recollection_performed": False, "holdout_bag_collection_performed": False,
        "expert_config_loaded_not_reinterpreted": True,
        "fixed_control": {
            "speed_mps": expert.baseline.fixed_speed_mps,
            "lookahead_m": expert.baseline.lookahead_m,
            "control_frequency_hz": expert.baseline.control_frequency_hz,
            "steering_limit_rad": expert.baseline.max_steering_rad,
            "wheelbase_m": expert.baseline.wheelbase_m,
        },
    }
    write_json(config.result_dir(repo, "validation_collection") / "audit.json", report)
    if result != "PASS":
        raise R1GateError("root disk has less than 6.5 GiB before validation collection")
    return report


def collector_config(config: R1Config) -> CollectorConfig:
    raw = config.collection
    value = CollectorConfig(
        expected_world="frozen-random-cone-validation-scenario-world",
        required_topics=REQUIRED_TOPICS,
        container_name=raw["container_name"], compose_service=raw["compose_service"],
        container_userdata_root=raw["container_userdata_root"],
        data_relative_root=raw["data_relative_root"], storage_id=raw["storage_id"],
        recorder_startup_timeout_s=raw["recorder_startup_timeout_s"],
        recorder_shutdown_timeout_s=raw["recorder_shutdown_timeout_s"],
        settle_duration_s=raw["settle_duration_s"], pilot_episode_count=2,
        minimum_free_bytes=MIN_TRAIN_BYTES,
        minimum_camera_messages=raw["minimum_camera_messages"],
    )
    value.validate()
    return value


def _validation_handle(backend: DockerRosBackend, episode_id: str) -> RecorderHandle:
    host_episode = backend.host_data_root / episode_id
    container_episode = str(PurePosixPath(backend.container_data_root) / episode_id)
    return RecorderHandle(
        episode_id, host_episode, host_episode / "bag", container_episode,
        str(PurePosixPath(container_episode) / "bag"),
        str(PurePosixPath(container_episode) / ".rosbag_pid"),
        str(PurePosixPath(container_episode) / "recorder.log"),
    )


def _topic_metrics(info: BagInfo) -> dict[str, dict[str, float | int]]:
    return {
        topic: {"message_count": count, "average_recorded_rate_hz": count / info.duration_s}
        for topic, count in sorted(info.topic_counts.items())
    }


def _base_validation_metadata(
    spec: ValidationSpec, config: R1Config, expert: RandomConeConfig,
    bundle: ScenarioBundle, repo: Path, attempt_number: int,
) -> dict[str, Any]:
    return {
        "version": COLLECTION_VERSION + "_episode", "episode_id": spec.episode_id,
        "scenario_id": spec.scenario_id, "repeat_id": spec.repeat_id,
        "scenario_role": spec.role, "collection_order_index": VALIDATION_EPISODES.index(spec.episode_id),
        "attempt_number": attempt_number, "infrastructure_replacement": attempt_number > 1,
        "result": "FAIL", "classification": "INFRA_FAIL", "failure_reason": None,
        "world": expert.world_name(spec.scenario_id),
        "frozen_scenario": bundle.scenario.to_dict(),
        "frozen_scenario_sha256": _canonical_hash(bundle.scenario.to_dict()),
        "planned_bypass": bundle.geometry, "task_config_sha256": config.sha256,
        "frozen_expert_config_sha256": config.frozen_expert["config_sha256"],
        "frozen_expert_result_manifest_sha256": config.frozen_expert["result_manifest_sha256"],
        "physicar_e2e_git_commit": git_commit(repo), "required_topics": list(REQUIRED_TOPICS),
        "recording_start_utc": None, "expert_driving_start_utc": None,
        "expert_driving_end_utc": None, "recording_end_utc": None,
        "preflight": None, "world_activation": None, "expert_result_metrics": None,
        "expert_classification": None, "bag_host_path": None, "bag_container_path": None,
        "bag_mcap_path": None, "bag_mcap_sha256": None, "bag_size_bytes": None,
        "bag_duration_s": None, "actual_topic_message_counts": {}, "topic_metrics": {},
        "recorder_graceful_shutdown": False, "recorder_orphaned": False,
        "orphan_process_check_pass": False, "post_run_safe_stop_success": False,
        "post_run_safe_stop_errors": [], "final_safe_stop_success": False,
        "final_safe_stop_errors": [], "infrastructure_failures": [],
    }


def collect_validation_episode(
    spec: ValidationSpec, *, config: R1Config, repo: Path, sim_root: Path,
    expert: RandomConeConfig, bundle: ScenarioBundle, backend: DockerRosBackend,
    client: SimClient, attempt_number: int, result_path: Path,
    prepare: Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]] = _post_settle_preflight,
    run_expert: Callable[..., dict[str, Any]] = run_random_cone_expert,
) -> dict[str, Any]:
    metadata = _base_validation_metadata(spec, config, expert, bundle, repo, attempt_number)
    handle: RecorderHandle | None = None
    stop_result = None
    run_exception: BaseException | None = None
    try:
        if errors := client.safe_stop():
            raise R1GateError("initial safe stop failed: " + "; ".join(errors))
        initial, activation, preflight = prepare(
            client, expert, bundle, sim_root, float(config.collection["settle_duration_s"])
        )
        metadata["world_activation"], metadata["preflight"] = activation, preflight
        if disk_state("/")["available_bytes"] < MIN_TRAIN_BYTES:
            raise R1GateError("root disk fell below 5.5 GiB before validation recording")
        handle = backend.start_recorder(spec.episode_id, REQUIRED_TOPICS)
        metadata.update({
            "bag_host_path": str(handle.host_bag_path),
            "bag_container_path": handle.container_bag_path,
            "recording_start_utc": utc_now(), "expert_driving_start_utc": utc_now(),
        })
        result = run_expert(client, expert, initial, bundle)
        metadata["expert_driving_end_utc"] = utc_now()
        metadata["expert_result_metrics"] = result
        metadata["expert_classification"] = result.get("classification")
        if result.get("classification") == "RANDOM_CONE_EXPERT_FAIL":
            metadata["failure_reason"] = "frozen Expert produced a genuine policy failure"
        elif result.get("classification") != "RANDOM_CONE_EXPERT_PASS":
            metadata["failure_reason"] = f"frozen Expert classified {result.get('classification')}"
    except BaseException as exc:
        run_exception = exc
        metadata["failure_reason"] = metadata["failure_reason"] or f"{type(exc).__name__}: {exc}"
    finally:
        errors = client.safe_stop()
        metadata["post_run_safe_stop_success"] = not errors
        metadata["post_run_safe_stop_errors"] = errors
        if errors:
            metadata["failure_reason"] = metadata["failure_reason"] or "post-run safe stop failed: " + "; ".join(errors)
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
                if not stop_result.graceful:
                    metadata["failure_reason"] = metadata["failure_reason"] or stop_result.detail or "recorder did not finalize"
            except BaseException as exc:
                metadata["recorder_orphaned"] = True
                metadata["failure_reason"] = metadata["failure_reason"] or f"recorder cleanup failed: {exc}"
            metadata["recording_end_utc"] = utc_now()
            try:
                metadata["orphan_process_check_pass"] = not backend._alive(handle)
            except BaseException as exc:
                metadata["infrastructure_failures"].append(f"orphan check failed: {exc}")
        final_errors = client.safe_stop()
        metadata["final_safe_stop_success"] = not final_errors
        metadata["final_safe_stop_errors"] = final_errors
        if final_errors:
            metadata["failure_reason"] = metadata["failure_reason"] or "final safe stop failed: " + "; ".join(final_errors)
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            info = backend.bag_info(handle)
            verify_bag(info, REQUIRED_TOPICS, int(config.collection["minimum_camera_messages"]))
            if set(info.topic_counts) != set(REQUIRED_TOPICS):
                raise R1GateError("bag does not contain exactly the canonical eight topics")
            mcaps = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcaps) != 1:
                raise R1GateError(f"expected one finalized MCAP, found {len(mcaps)}")
            metadata.update({
                "bag_mcap_path": str(mcaps[0]), "bag_mcap_sha256": sha256_file(mcaps[0]),
                "bag_size_bytes": directory_size(handle.host_bag_path),
                "bag_duration_s": info.duration_s,
                "actual_topic_message_counts": dict(sorted(info.topic_counts.items())),
                "topic_metrics": _topic_metrics(info),
            })
        except BaseException as exc:
            metadata["failure_reason"] = metadata["failure_reason"] or f"bag integrity failed: {exc}"
    result = metadata.get("expert_result_metrics") or {}
    success = (
        metadata["failure_reason"] is None
        and metadata["expert_classification"] == "RANDOM_CONE_EXPERT_PASS"
        and result.get("result") == "PASS"
        and float(result.get("minimum_footprint_to_cone_clearance_m", -1.0)) > 0.0
        and result.get("cone_contact_or_intersection_occurred") is False
        and result.get("recovery_success") is True
        and result.get("api_failures") == result.get("pose_failures") == result.get("clock_failures") == 0
        and result.get("safe_stop_success") is True
        and metadata["post_run_safe_stop_success"] and metadata["final_safe_stop_success"]
        and metadata["recorder_graceful_shutdown"] and not metadata["recorder_orphaned"]
        and metadata["orphan_process_check_pass"] and metadata["bag_mcap_sha256"] is not None
    )
    if success:
        metadata["result"], metadata["classification"] = "PASS", "VALIDATION_EPISODE_PASS"
    elif metadata["expert_classification"] == "RANDOM_CONE_EXPERT_FAIL":
        metadata["classification"] = "GENUINE_EXPERT_FAIL"
    else:
        metadata["classification"] = "INFRA_FAIL"
        if run_exception is not None:
            metadata["infrastructure_failures"].append(f"{type(run_exception).__name__}: {run_exception}")
    write_json(result_path, metadata)
    return metadata


def validate_validation_metadata(
    metadata: dict[str, Any], spec: ValidationSpec, config: R1Config,
    expert: RandomConeConfig, bundle: ScenarioBundle,
) -> list[str]:
    errors: list[str] = []
    expected = {
        "version": COLLECTION_VERSION + "_episode", "episode_id": spec.episode_id,
        "scenario_id": spec.scenario_id, "repeat_id": "R01", "scenario_role": "VALIDATION",
        "collection_order_index": VALIDATION_EPISODES.index(spec.episode_id),
        "world": expert.world_name(spec.scenario_id),
        "frozen_scenario_sha256": _canonical_hash(bundle.scenario.to_dict()),
        "task_config_sha256": config.sha256, "result": "PASS",
        "classification": "VALIDATION_EPISODE_PASS",
        "expert_classification": "RANDOM_CONE_EXPERT_PASS",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"{key} mismatch")
    if tuple(metadata.get("required_topics") or ()) != REQUIRED_TOPICS:
        errors.append("required topics mismatch")
    counts = metadata.get("actual_topic_message_counts") or {}
    if set(counts) != set(REQUIRED_TOPICS) or any(int(counts.get(topic, 0)) <= 0 for topic in REQUIRED_TOPICS):
        errors.append("topic counts incomplete")
    if int(counts.get("/camera/image_raw", 0)) < int(config.collection["minimum_camera_messages"]):
        errors.append("camera count below minimum")
    for flag in ("recorder_graceful_shutdown", "orphan_process_check_pass", "post_run_safe_stop_success", "final_safe_stop_success"):
        if metadata.get(flag) is not True:
            errors.append(f"{flag} is not true")
    result = metadata.get("expert_result_metrics") or {}
    if (
        result.get("result") != "PASS"
        or not float(result.get("minimum_footprint_to_cone_clearance_m", -1.0)) > 0.0
        or result.get("cone_contact_or_intersection_occurred") is not False
        or result.get("recovery_success") is not True
        or result.get("safe_stop_success") is not True
    ):
        errors.append("Expert practical contract failed")
    if not int(metadata.get("bag_size_bytes") or 0) > 0 or not float(metadata.get("bag_duration_s") or 0) > 0:
        errors.append("bag size/duration invalid")
    return errors


def inspect_validation_episode(
    spec: ValidationSpec, result_path: Path, config: R1Config, expert: RandomConeConfig,
    bundle: ScenarioBundle, backend: DockerRosBackend,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    external_exists = (backend.host_data_root / spec.episode_id).exists()
    if not result_path.is_file():
        return ("PARTIAL" if external_exists else "MISSING"), None, ["compact final evidence missing"] if external_exists else []
    metadata = _read_json(result_path)
    if metadata.get("classification") == "GENUINE_EXPERT_FAIL":
        return "GENUINE_FAILURE", metadata, [str(metadata.get("failure_reason"))]
    errors = validate_validation_metadata(metadata, spec, config, expert, bundle)
    if not external_exists:
        errors.append("external episode directory missing")
    if not errors:
        try:
            handle = _validation_handle(backend, spec.episode_id)
            info = backend.bag_info(handle)
            verify_bag(info, REQUIRED_TOPICS, int(config.collection["minimum_camera_messages"]))
            mcaps = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcaps) != 1 or sha256_file(mcaps[0]) != metadata["bag_mcap_sha256"]:
                errors.append("MCAP identity mismatch")
            if dict(sorted(info.topic_counts.items())) != metadata["actual_topic_message_counts"]:
                errors.append("MCAP topic counts mismatch")
        except BaseException as exc:
            errors.append(f"MCAP integrity recheck failed: {exc}")
    return ("VALID" if not errors else "PARTIAL"), metadata, errors


def validation_retry_decision(classification: str, attempt_number: int) -> str:
    if classification == "VALIDATION_EPISODE_PASS":
        return "CONTINUE"
    if classification == "GENUINE_EXPERT_FAIL":
        return "STOP_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < 2:
        return "RETRY_INFRA"
    return "STOP_INFRA"


def _archive_partial(backend: DockerRosBackend, spec: ValidationSpec, attempt: int) -> str | None:
    source = backend.host_data_root / spec.episode_id
    if not source.exists():
        return None
    handle = _validation_handle(backend, spec.episode_id)
    try:
        if backend._alive(handle):
            backend.stop_recorder(handle)
    except BaseException:
        pass
    destination = backend.host_data_root / "_interrupted" / spec.episode_id / f"attempt_{attempt:02d}"
    if destination.exists():
        destination = destination.with_name(destination.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return str(destination)


def validation_collection_gate(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ids = [item.get("episode_id") for item in episodes]
    scenarios = [item.get("scenario_id") for item in episodes]
    gates = {
        "exactly_two_validation_episodes": ids == list(VALIDATION_EPISODES),
        "no_duplicate_episode_ids": len(ids) == len(set(ids)) == 2,
        "only_scenarios_09_10": scenarios == list(VALIDATION_SCENARIOS),
        "validation_role_only": all(item.get("scenario_role") == "VALIDATION" for item in episodes),
        "one_repeat_only": all(item.get("repeat_id") == "R01" for item in episodes),
        "all_episode_results_pass": all(item.get("classification") == "VALIDATION_EPISODE_PASS" for item in episodes),
        "no_genuine_expert_failure": all(item.get("classification") != "GENUINE_EXPERT_FAIL" for item in episodes),
        "no_holdout_collection": not set(scenarios).intersection(HOLDOUT_SCENARIOS),
        "all_safe_stops_pass": all(
            item.get("post_run_safe_stop_success") is True
            and item.get("final_safe_stop_success") is True
            and (item.get("expert_result_metrics") or {}).get("safe_stop_success") is True
            for item in episodes
        ),
        "frozen_control": all((item.get("preflight") or {}).get("fixed_control") == {
            "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
            "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
        } for item in episodes),
    }
    return {"result": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def _write_report(path: Path, title: str, result: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"# {title}", "", f"Result: **{result}**", "", *lines, ""]), encoding="utf-8")


def collection_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    audit = audit_stage(repo, sim_root, config)
    train_task: TrainTaskConfig = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, _ = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles if bundle.scenario.scenario_id in VALIDATION_SCENARIOS}
    result_dir = config.result_dir(repo, "validation_collection")
    result_dir.mkdir(parents=True, exist_ok=True)
    backend = DockerRosBackend(collector_config(config), sim_root)
    if backend.host_data_root.exists():
        forbidden = sorted(item.name for item in backend.host_data_root.iterdir() if item.is_dir() and item.name not in {*VALIDATION_EPISODES, "_interrupted"})
        if forbidden:
            raise R1GateError(f"validation raw root contains forbidden episode directories: {forbidden}")
    topic_types = backend.preflight(REQUIRED_TOPICS)
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    report: dict[str, Any] = {
        "version": COLLECTION_VERSION, "generated_utc": utc_now(), "result": "FAIL",
        "task_config_sha256": config.sha256, "frozen_audit": audit,
        "intended_episode_order": list(VALIDATION_EPISODES), "required_topics": list(REQUIRED_TOPICS),
        "topic_types": topic_types, "raw_root": str(backend.host_data_root),
        "disk_pre_collection": disk_state("/"), "resumed_skipped_episode_ids": [],
        "interrupted_archives": [], "episodes": [], "failure_reason": None,
    }
    summary_path = result_dir / "summary.json"
    try:
        for spec in validation_specs():
            bundle = bundles[spec.scenario_id]
            final_path = result_dir / "episodes" / f"{spec.episode_id}.json"
            status, existing, errors = inspect_validation_episode(spec, final_path, config, expert, bundle, backend)
            if status == "GENUINE_FAILURE":
                report["failure_reason"] = f"preserved genuine Expert failure in {spec.episode_id}"
                break
            if status == "VALID":
                report["episodes"].append(existing)
                report["resumed_skipped_episode_ids"].append(spec.episode_id)
                continue
            attempts = len(list((result_dir / "attempts").glob(f"{spec.episode_id}_attempt_*.json")))
            state = result_dir / "states" / f"{spec.episode_id}.json"
            if state.is_file():
                attempts = max(attempts, int(_read_json(state).get("attempt_number", 0)))
            if status == "PARTIAL" and attempts == 0:
                attempts = 1
                write_json(result_dir / "attempts" / f"{spec.episode_id}_attempt_01.json", {
                    "version": COLLECTION_VERSION + "_interrupted", "episode_id": spec.episode_id,
                    "scenario_id": spec.scenario_id, "classification": "INFRA_FAIL", "result": "FAIL",
                    "failure_reason": "; ".join(errors) or "unfinalized validation bag",
                    "reconstructed_utc": utc_now(),
                })
            if attempts >= 2:
                report["failure_reason"] = f"{spec.episode_id} exhausted its bounded infrastructure replacement"
                break
            if status == "PARTIAL":
                archive = _archive_partial(backend, spec, attempts)
                if archive:
                    report["interrupted_archives"].append({"episode_id": spec.episode_id, "archive_path": archive, "diagnostics": errors})
            attempt = attempts + 1
            while attempt <= 2:
                write_json(state, {
                    "status": "STARTED_UNFINALIZED", "episode_id": spec.episode_id,
                    "scenario_id": spec.scenario_id, "attempt_number": attempt,
                    "started_utc": utc_now(), "task_config_sha256": config.sha256,
                })
                attempt_path = result_dir / "attempts" / f"{spec.episode_id}_attempt_{attempt:02d}.json"
                episode = collect_validation_episode(
                    spec, config=config, repo=repo, sim_root=sim_root, expert=expert,
                    bundle=bundle, backend=backend, client=client,
                    attempt_number=attempt, result_path=attempt_path,
                )
                print(json.dumps({
                    "stage": "validation_collection", "episode_id": spec.episode_id,
                    "attempt": attempt, "classification": episode["classification"],
                    "clearance_m": (episode.get("expert_result_metrics") or {}).get("minimum_footprint_to_cone_clearance_m"),
                    "bag_size_bytes": episode.get("bag_size_bytes"),
                }), flush=True)
                decision = validation_retry_decision(episode["classification"], attempt)
                if decision == "CONTINUE":
                    write_json(final_path, episode)
                    write_json(state, {
                        "status": "FINALIZED_VALID", "episode_id": spec.episode_id,
                        "scenario_id": spec.scenario_id, "attempt_number": attempt,
                        "finalized_utc": utc_now(), "bag_mcap_sha256": episode["bag_mcap_sha256"],
                    })
                    report["episodes"].append(episode)
                    break
                if decision == "STOP_GENUINE_FAILURE":
                    write_json(final_path, episode)
                    write_json(state, {"status": "GENUINE_POLICY_FAILURE_DO_NOT_RETRY", "attempt_number": attempt, "finalized_utc": utc_now()})
                    report["episodes"].append(episode)
                    report["failure_reason"] = f"genuine Expert failure in {spec.episode_id}"
                    break
                if decision == "RETRY_INFRA":
                    archive = _archive_partial(backend, spec, attempt)
                    if archive:
                        report["interrupted_archives"].append({"episode_id": spec.episode_id, "archive_path": archive, "diagnostics": [episode.get("failure_reason")]})
                    if errors := client.safe_stop():
                        report["failure_reason"] = "safe stop failed before infrastructure retry: " + "; ".join(errors)
                        break
                    attempt += 1
                    continue
                report["failure_reason"] = f"infrastructure replacement exhausted for {spec.episode_id}"
                break
            if report["failure_reason"]:
                break
            write_json(summary_path, report)
        report["disk_after_validation_collection"] = disk_state("/")
        report["collection_gate"] = validation_collection_gate(report["episodes"])
        report["total_raw_storage_bytes"] = sum(int(item.get("bag_size_bytes") or 0) for item in report["episodes"])
        report["infrastructure_replacement_attempt_count"] = sum(max(0, len(list((result_dir / "attempts").glob(f"{spec.episode_id}_attempt_*.json"))) - 1) for spec in validation_specs())
        report["result"] = "PASS" if report["collection_gate"]["result"] == "PASS" and report["failure_reason"] is None else "FAIL"
    finally:
        final_errors = client.safe_stop()
        report["final_safe_stop_success"], report["final_safe_stop_errors"] = not final_errors, final_errors
        try:
            report["world_restoration"] = _restore_world(client, original_world)
        except BaseException as exc:
            report["world_restoration"] = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"}
        report["simulator_tracked_source_status_after"] = simulator_tracked_status(sim_root)
        report["frozen_expert_evidence_unchanged"] = directory_file_manifest_sha256(repo / config.frozen_expert["result_directory"]) == config.frozen_expert["result_manifest_sha256"]
        if final_errors or report["world_restoration"].get("result") != "PASS" or report["simulator_tracked_source_status_after"].get("result") != "PASS" or not report["frozen_expert_evidence_unchanged"]:
            report["result"] = "FAIL"
        write_json(summary_path, report)
        _write_report(result_dir / "REPORT.md", "Random-Cone 1.0 m/s Validation Bag Collection", report["result"], [
            f"Episodes: {len(report.get('episodes', []))}/2",
            f"Resumed/skipped: {', '.join(report.get('resumed_skipped_episode_ids', [])) or 'none'}",
            f"Raw bytes: {report.get('total_raw_storage_bytes')}",
            "No TRAIN or UNSEEN_HOLDOUT bag was collected.",
        ])
    if report["result"] != "PASS":
        raise R1GateError(report.get("failure_reason") or "validation collection gate failed")
    return report


def build_validation_sequences(
    rows: Sequence[dict[str, Any]], spec: ValidationSpec, source_manifest_sha256: str,
    maximum_gap_s: float = MAX_ADJACENT_GAP_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build unpadded causal [t-2,t-1,t] sequences within one validation bag."""
    if maximum_gap_s != 0.120:
        raise R1GateError("maximum adjacent temporal gap must remain 0.120 s")
    ordered = list(rows)
    if any(row.get("episode_id") != spec.episode_id for row in ordered):
        raise R1GateError("validation temporal source crosses an episode boundary")
    if any(row.get("scenario_id") != spec.scenario_id or row.get("repeat_id") != "R01" for row in ordered):
        raise R1GateError("validation temporal source crosses a scenario/repeat boundary")
    accepted: list[dict[str, Any]] = []
    gaps: list[float] = []
    spans: list[float] = []
    rejected_gap = 0
    for index in range(2, len(ordered)):
        a, b, c = ordered[index - 2:index + 1]
        times = tuple(int(row["camera_record_time_ns"]) for row in (a, b, c))
        paths = tuple(str(row["image_path"]) for row in (a, b, c))
        if not times[0] < times[1] < times[2]:
            raise R1GateError(f"non-causal camera timestamps in {spec.episode_id}")
        if len(set(paths)) != 3:
            raise R1GateError(f"duplicate-frame padding in {spec.episode_id}")
        gap_1, gap_2 = (times[1] - times[0]) / 1e9, (times[2] - times[1]) / 1e9
        if gap_1 > maximum_gap_s or gap_2 > maximum_gap_s:
            rejected_gap += 1
            continue
        row = {
            "sequence_id": f"{spec.episode_id}_seq_{len(accepted):06d}",
            "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
            "scenario_role": "VALIDATION", "repeat_id": "R01",
            "cone_scenario_id": spec.scenario_id,
            "frame_t_minus_2": paths[0], "frame_t_minus_1": paths[1], "frame_t": paths[2],
            "frame_t_minus_2_sha256": a["image_sha256"],
            "frame_t_minus_1_sha256": b["image_sha256"],
            "frame_t_sha256": c["image_sha256"],
            "camera_timestamp_t_minus_2_ns": times[0],
            "camera_timestamp_t_minus_1_ns": times[1],
            "camera_timestamp_t_ns": times[2],
            "adjacent_gap_1_s": gap_1, "adjacent_gap_2_s": gap_2,
            "oldest_to_current_span_s": gap_1 + gap_2,
            "steering_target_timestamp_ns": int(c["steering_record_time_ns"]),
            "steering_label_age_ms": float(c["steering_age_ms"]),
            "target_steering_rad": float(c["steering_rad"]),
            "speed_record_time_ns": int(c["speed_record_time_ns"]),
            "speed_age_ms": float(c["speed_age_ms"]), "speed_mps": float(c["speed_mps"]),
            "route_progress_m": float(c["route_s_m"]),
            "source_mcap_sha256": c["source_mcap_sha256"],
            "source_manifest_sha256": source_manifest_sha256,
        }
        if row["steering_target_timestamp_ns"] > row["camera_timestamp_t_ns"]:
            raise R1GateError("future steering label entered validation manifest")
        accepted.append(row)
        gaps.extend((gap_1, gap_2))
        spans.append(gap_1 + gap_2)
    return accepted, {
        "episode_id": spec.episode_id, "source_frames": len(ordered),
        "temporal_candidate_sequences": max(0, len(ordered) - 2),
        "accepted_temporal_sequences": len(accepted), "gap_rejects": rejected_gap,
        "boundary_rejects": min(2, len(ordered)),
        "adjacent_gap_s": numeric_distribution(gaps),
        "oldest_to_current_span_s": numeric_distribution(spans),
        "future_label_violations": 0,
    }


def _two_episode_overview(previews: Sequence[Path], output: Path) -> None:
    if len(previews) != 2:
        raise R1GateError(f"validation overview requires two previews, found {len(previews)}")
    images: list[Image.Image] = []
    try:
        for path in previews:
            with Image.open(path) as source:
                item = source.convert("RGB")
                label = Image.new("RGB", (item.width, item.height + 18), "black")
                label.paste(item, (0, 18))
                from PIL import ImageDraw
                ImageDraw.Draw(label).text((3, 3), path.stem, fill="white")
                images.append(label)
                item.close()
        sheet = Image.new("RGB", (max(item.width for item in images), sum(item.height for item in images)), "black")
        y = 0
        for item in images:
            sheet.paste(item, (0, y))
            y += item.height
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="PNG", optimize=False)
        sheet.close()
    finally:
        for item in images:
            item.close()


def _dataset_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _aggregate_rejections(metrics: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        reason: sum(int(item["counts"]["rejection_by_reason"].get(reason, 0)) for item in metrics)
        for reason in REJECTION_REASONS
    }


def dataset_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    collection_path = config.result_dir(repo, "validation_collection") / "summary.json"
    collection = _read_json(collection_path)
    if collection.get("result") != "PASS" or (collection.get("collection_gate") or {}).get("result") != "PASS":
        raise R1GateError("validation collection is not 2/2 PASS")
    result_dir = config.result_dir(repo, "validation_dataset")
    summary_path = result_dir / "summary.json"
    dataset_root = config.external(sim_root, "validation_dataset")
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("result") in {"PENDING_VISUAL_QC", "PASS"} and existing.get("collection_summary_sha256") == sha256_file(collection_path):
            return existing
    if dataset_root.exists():
        archive = dataset_root.parent / "_interrupted_validation_dataset" / datetime.now().strftime("%Y%m%dT%H%M%S")
        archive.parent.mkdir(parents=True, exist_ok=True)
        dataset_root.rename(archive)
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, expert_audit = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles if bundle.scenario.scenario_id in VALIDATION_SCENARIOS}
    backend = DockerRosBackend(collector_config(config), sim_root)
    source_metadata: dict[str, dict[str, Any]] = {}
    for spec in validation_specs():
        path = config.result_dir(repo, "validation_collection") / "episodes" / f"{spec.episode_id}.json"
        status, metadata, errors = inspect_validation_episode(spec, path, config, expert, bundles[spec.scenario_id], backend)
        if status != "VALID" or metadata is None:
            raise R1GateError(f"{spec.episode_id} is not a finalized source: {errors}")
        source_metadata[spec.episode_id] = metadata
    extractor_path = repo / config.dataset["canonical_extractor_config_path"]
    extractor = load_extractor_config(extractor_path)
    extractor_sha = sha256_file(extractor_path)
    prepare_output_root(dataset_root, False)
    (dataset_root / "temporal_manifests").mkdir()
    (dataset_root / "previews" / "maneuver_regions").mkdir(parents=True)
    episode_metrics: list[dict[str, Any]] = []
    compact_episodes: list[dict[str, Any]] = []
    all_frames: list[dict[str, Any]] = []
    all_temporal: list[dict[str, Any]] = []
    temporal_by_episode: dict[str, list[dict[str, Any]]] = {}
    previews: list[Path] = []
    for spec in validation_specs():
        bundle = bundles[spec.scenario_id]
        source_result = config.result_dir(repo, "validation_collection") / "episodes" / f"{spec.episode_id}.json"
        source = source_metadata[spec.episode_id]
        mcap = Path(source["bag_mcap_path"])
        if sha256_file(mcap) != source["bag_mcap_sha256"]:
            raise R1GateError(f"source MCAP changed: {spec.episode_id}")
        metrics, rows = extract_episode(
            episode_id=spec.episode_id, mcap_path=mcap,
            collector_metadata_path=source_result, dataset_root=dataset_root,
            config=extractor, config_sha256=extractor_sha,
            source_path_identity=mcap.relative_to(backend.host_data_root).as_posix(),
            collector_metadata_identity=source_result.relative_to(repo).as_posix(),
        )
        odom = scan_odom_route_s(mcap, bundle.plan.nominal, source["preflight"]["pose"])
        metrics["evaluation_only_route_s"] = attach_causal_route_s(rows, odom)
        for row in rows:
            row.update({
                "scenario_id": spec.scenario_id, "scenario_role": "VALIDATION",
                "repeat_id": "R01", "cone_scenario_id": spec.scenario_id,
            })
            row["image_sha256"] = sha256_file(dataset_root / row["image_path"])
        frame_manifest = dataset_root / "manifests" / f"{spec.episode_id}.csv"
        write_csv(frame_manifest, rows, FRAME_MANIFEST_COLUMNS)
        frame_sha = sha256_file(frame_manifest)
        temporal, temporal_stats = build_validation_sequences(rows, spec, frame_sha)
        temporal_manifest = dataset_root / "temporal_manifests" / f"{spec.episode_id}.csv"
        write_csv(temporal_manifest, temporal, TEMPORAL_MANIFEST_COLUMNS)
        regions = scenario_regions(bundle, float(source["preflight"]["route_length_m"]), float(config.dataset["route_region_margin_m"]))
        coverage = region_coverage(temporal, regions)
        preview = dataset_root / "previews" / "maneuver_regions" / f"{spec.episode_id}.png"
        selected = create_maneuver_contact_sheet(dataset_root, temporal, regions, preview)
        previews.append(preview)
        metrics.update({"scenario_id": spec.scenario_id, "scenario_role": "VALIDATION", "repeat_id": "R01", "temporal": temporal_stats, "cone_region_coverage": coverage})
        episode_metrics.append(metrics)
        all_frames.extend(rows)
        all_temporal.extend(temporal)
        temporal_by_episode[spec.episode_id] = temporal
        compact = {
            "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
            "repeat_id": "R01", "scenario_role": "VALIDATION",
            "result": "PASS" if metrics["result"] == "PASS" and coverage["result"] == "PASS" and temporal_stats["future_label_violations"] == 0 else "FAIL",
            "raw_camera_messages": metrics["counts"]["total_camera_frames"],
            "accepted_images": metrics["counts"]["accepted_camera_samples"],
            "retention_fraction": metrics["counts"]["retention_fraction"],
            "rejection_by_reason": metrics["counts"]["rejection_by_reason"],
            "steering_age_ms": metrics["synchronization"]["steering_age_ms"],
            "speed_age_ms": metrics["synchronization"]["speed_age_ms"],
            "image_interval_ms": metrics["synchronization"]["accepted_camera_interval_ms"],
            "future_label_violations": metrics["synchronization"]["future_label_violations"],
            "temporal_candidate_sequences": temporal_stats["temporal_candidate_sequences"],
            "accepted_temporal_sequences": temporal_stats["accepted_temporal_sequences"],
            "gap_rejects": temporal_stats["gap_rejects"], "boundary_rejects": temporal_stats["boundary_rejects"],
            "adjacent_gap_s": temporal_stats["adjacent_gap_s"],
            "oldest_to_current_span_s": temporal_stats["oldest_to_current_span_s"],
            "cone_region_coverage": coverage, "frame_manifest_sha256": frame_sha,
            "temporal_manifest_sha256": sha256_file(temporal_manifest),
            "source_mcap_sha256": source["bag_mcap_sha256"],
            "canonical_preview": metrics["artifacts"]["preview_path"],
            "maneuver_preview": {"path": preview.relative_to(dataset_root).as_posix(), "sha256": sha256_file(preview), "selection_count_by_region": selected},
        }
        compact_episodes.append(compact)
        write_json(result_dir / "episodes" / f"{spec.episode_id}.json", compact)
        print(json.dumps({"stage": "validation_extract", "episode_id": spec.episode_id, "images": compact["accepted_images"], "sequences": compact["accepted_temporal_sequences"], "coverage": coverage["result"]}), flush=True)
    write_csv(dataset_root / "manifest.csv", all_frames, FRAME_MANIFEST_COLUMNS)
    validation_manifest = dataset_root / "temporal_manifests" / "validation.csv"
    write_csv(validation_manifest, all_temporal, TEMPORAL_MANIFEST_COLUMNS)
    overview = dataset_root / "previews" / "all_validation_episodes_maneuver_overview.png"
    _two_episode_overview(previews, overview)
    all_gaps = [gap for row in all_temporal for gap in (float(row["adjacent_gap_1_s"]), float(row["adjacent_gap_2_s"]))]
    all_spans = [float(row["oldest_to_current_span_s"]) for row in all_temporal]
    future_violations = sum(item["synchronization"]["future_label_violations"] for item in episode_metrics)
    scenarios = {row["scenario_id"] for row in all_temporal}
    episodes = {row["episode_id"] for row in all_temporal}
    source_hashes = {row["source_mcap_sha256"] for row in all_temporal}
    train_source_hashes = {
        row["source_mcap_sha256"] for row in _read_temporal_csv(
            Path(config.frozen_train["manifest_path"]), Path(config.frozen_train["manifest_path"]).parents[1], expected_role="TRAIN"
        )
    }
    gates = {
        "two_readable_validation_bags": len(episode_metrics) == 2,
        "exact_episode_order": [item["episode_id"] for item in compact_episodes] == list(VALIDATION_EPISODES),
        "only_scenarios_09_10": scenarios == set(VALIDATION_SCENARIOS),
        "validation_role_only": {row["scenario_role"] for row in all_temporal} == {"VALIDATION"},
        "no_train_or_holdout_rows": not scenarios.intersection(set(TRAIN_SCENARIOS) | set(HOLDOUT_SCENARIOS)),
        "source_episode_set_exact": episodes == set(VALIDATION_EPISODES),
        "source_hashes_disjoint_from_train": not source_hashes.intersection(train_source_hashes),
        "future_label_violations_zero": future_violations == 0,
        "causal_steering_targets": all(int(row["steering_target_timestamp_ns"]) <= int(row["camera_timestamp_t_ns"]) for row in all_temporal),
        "strict_three_frame_order": all(int(row["camera_timestamp_t_minus_2_ns"]) < int(row["camera_timestamp_t_minus_1_ns"]) < int(row["camera_timestamp_t_ns"]) for row in all_temporal),
        "adjacent_gap_at_most_0p120_s": bool(all_gaps) and max(all_gaps) <= 0.120,
        "no_duplicate_frame_padding": all(len({row["frame_t_minus_2"], row["frame_t_minus_1"], row["frame_t"]}) == 3 for row in all_temporal),
        "two_boundary_rejects_per_episode": all(item["boundary_rejects"] == 2 for item in compact_episodes),
        "maneuver_regions_covered": all(item["cone_region_coverage"]["result"] == "PASS" for item in compact_episodes),
        "no_image_decode_failures": all(item["counts"]["rejection_by_reason"]["image_decode_error"] == 0 for item in episode_metrics),
        "canonical_roi_rgb": extractor["roi"] == {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360} and extractor["output_width"] == 200 and extractor["output_height"] == 66 and extractor["source_encoding"] == "rgb8",
        "all_episode_gates_pass": all(item["result"] == "PASS" for item in compact_episodes),
    }
    technical = "PASS" if all(gates.values()) else "FAIL"
    intervals = []
    for spec in validation_specs():
        selected = [row for row in all_frames if row["episode_id"] == spec.episode_id]
        intervals.extend((int(b["camera_record_time_ns"]) - int(a["camera_record_time_ns"])) / 1e6 for a, b in zip(selected, selected[1:]))
    report: dict[str, Any] = {
        "version": DATASET_VERSION, "generated_utc": utc_now(),
        "result": "PENDING_VISUAL_QC" if technical == "PASS" else "FAIL",
        "technical_qc_result": technical, "task_config_sha256": config.sha256,
        "collection_summary_sha256": sha256_file(collection_path),
        "frozen_expert_audit_result": expert_audit["result"],
        "dataset_root": str(dataset_root), "episode_count": 2,
        "episode_ids": list(VALIDATION_EPISODES), "scenario_ids": list(VALIDATION_SCENARIOS),
        "scenario_role": "VALIDATION", "train_sequences": 0, "holdout_sequences": 0,
        "counts": {
            "raw_camera_messages": sum(item["counts"]["total_camera_frames"] for item in episode_metrics),
            "active_window_camera_messages": sum(item["counts"]["active_window_camera_frames"] for item in episode_metrics),
            "accepted_images": len(all_frames),
            "rejected_images": sum(item["counts"]["total_camera_frames"] for item in episode_metrics) - len(all_frames),
            "rejection_by_reason": _aggregate_rejections(episode_metrics),
        },
        "retention_fraction": len(all_frames) / sum(item["counts"]["total_camera_frames"] for item in episode_metrics),
        "synchronization": {
            "steering_age_ms": numeric_distribution([float(row["steering_age_ms"]) for row in all_frames]),
            "speed_age_ms": numeric_distribution([float(row["speed_age_ms"]) for row in all_frames]),
            "accepted_camera_interval_ms": numeric_distribution(intervals),
            "future_label_violations": future_violations,
        },
        "temporal": {
            "history_frames": 3, "maximum_adjacent_gap_s": 0.120,
            "candidate_sequence_count": sum(item["temporal_candidate_sequences"] for item in compact_episodes),
            "accepted_sequence_count": len(all_temporal),
            "gap_reject_count": sum(item["gap_rejects"] for item in compact_episodes),
            "boundary_reject_count": sum(item["boundary_rejects"] for item in compact_episodes),
            "adjacent_gap_s": numeric_distribution(all_gaps),
            "oldest_to_current_span_s": numeric_distribution(all_spans),
            "manifest": str(validation_manifest), "manifest_sha256": sha256_file(validation_manifest),
        },
        "steering_distribution": {
            "overall": steering_distribution([float(row["target_steering_rad"]) for row in all_temporal], extractor),
            "by_episode": {episode: steering_distribution([float(row["target_steering_rad"]) for row in temporal_by_episode[episode]], extractor) for episode in VALIDATION_EPISODES},
        },
        "episodes": compact_episodes,
        "cone_region_coverage": {item["episode_id"]: item["cone_region_coverage"] for item in compact_episodes},
        "quality_gates": gates,
        "visual_qc": {
            "result": "PENDING_MANUAL_REVIEW", "episode_preview_count": 2,
            "overview_path": str(overview), "overview_sha256": sha256_file(overview),
        },
        "canonical_extractor": {
            "path": config.dataset["canonical_extractor_config_path"], "sha256": extractor_sha,
            "synchronization": "latest steering/speed MCAP record at or before camera record time",
            "source": "480x360 RGB /camera/image_raw", "crop": "x=0:480,y=160:360",
            "stored_image": "200x66 RGB PNG",
        },
        "external_storage": {
            "source_raw_bytes": sum(int(item["bag_size_bytes"]) for item in source_metadata.values()),
            "derived_dataset_bytes": None,
        },
        "disk_after_validation_extraction": disk_state("/"),
        "training_disk_gate_pass": disk_state("/")["available_bytes"] >= MIN_TRAIN_BYTES,
        "holdout_data_created": False, "neural_training_performed": False,
    }
    metadata_path = dataset_root / "dataset_metadata.json"
    write_json(metadata_path, report)
    report["external_storage"]["derived_dataset_bytes"] = _dataset_bytes(dataset_root)
    write_json(metadata_path, report)
    report["dataset_metadata_sha256"] = sha256_file(metadata_path)
    write_json(summary_path, report)
    _write_report(result_dir / "REPORT.md", "Random-Cone 1.0 m/s Validation Temporal Dataset", report["result"], [
        f"Sequences: {len(all_temporal)}", f"Future-label violations: {future_violations}",
        "Contains only S09/S10 VALIDATION data; S11/S12 are absent.",
    ])
    if technical != "PASS":
        raise R1GateError("validation temporal dataset technical QC failed")
    if not report["training_disk_gate_pass"]:
        raise R1GateError("root disk has less than 5.5 GiB after validation extraction")
    return report


def finalize_visual_qc(
    repo: Path, sim_root: Path, config: R1Config, *, passed: bool, review_note: str,
) -> dict[str, Any]:
    if not review_note.strip():
        raise R1GateError("visual QC requires a review note")
    result_dir = config.result_dir(repo, "validation_dataset")
    summary_path = result_dir / "summary.json"
    report = _read_json(summary_path)
    if report.get("technical_qc_result") != "PASS":
        raise R1GateError("cannot pass visual QC before technical QC")
    dataset_root = Path(report["dataset_root"])
    preview_paths = [dataset_root / item["maneuver_preview"]["path"] for item in report["episodes"]]
    canonical_paths = [dataset_root / item["canonical_preview"] for item in report["episodes"]]
    overview = Path(report["visual_qc"]["overview_path"])
    if len(preview_paths) != 2 or len(canonical_paths) != 2 or not all(path.is_file() for path in [*preview_paths, *canonical_paths, overview]):
        raise R1GateError("validation preview artifact set is incomplete")
    _two_episode_overview(preview_paths, overview)
    report["visual_qc"] = {
        **report["visual_qc"], "result": "PASS" if passed else "FAIL",
        "overview_sha256": sha256_file(overview), "reviewed_utc": utc_now(),
        "reviewer": "Codex visual inspection", "review_note": review_note.strip(),
        "checks": {
            "both_validation_episodes_inspectable": passed,
            "cone_visible_and_correct_scenario": passed,
            "lane_and_road_intact": passed, "maneuver_and_recovery_represented": passed,
            "roi_correct": passed, "no_corrupt_images": passed,
            "no_reset_or_teleport_images": passed, "temporal_order_valid": passed,
        },
    }
    report["quality_gates"]["visual_qc_pass"] = passed
    report["result"] = "PASS" if passed and all(report["quality_gates"].values()) else "FAIL"
    report["training_justified"] = report["result"] == "PASS" and report["training_disk_gate_pass"]
    report["finalized_utc"] = utc_now()
    report.pop("dataset_metadata_sha256", None)
    metadata_path = dataset_root / "dataset_metadata.json"
    write_json(metadata_path, report)
    report["external_storage"]["derived_dataset_bytes"] = _dataset_bytes(dataset_root)
    write_json(metadata_path, report)
    report["dataset_metadata_sha256"] = sha256_file(metadata_path)
    write_json(summary_path, report)
    _write_report(result_dir / "REPORT.md", "Random-Cone 1.0 m/s Validation Temporal Dataset", report["result"], [
        f"Sequences: {report['temporal']['accepted_sequence_count']}",
        f"Future-label violations: {report['synchronization']['future_label_violations']}",
        f"Visual QC: {report['visual_qc']['result']}",
        "Contains only S09/S10 VALIDATION data; S11/S12 are absent.",
    ])
    return report


def leakage_audit(repo: Path, sim_root: Path, config: R1Config, *, stage: str) -> dict[str, Any]:
    train_path = Path(config.frozen_train["manifest_path"])
    train_rows = _read_temporal_csv(train_path, train_path.parents[1], expected_role="TRAIN")
    validation_summary_path = config.result_dir(repo, "validation_dataset") / "summary.json"
    if not validation_summary_path.is_file():
        raise R1GateError("validation dataset summary is missing for leakage audit")
    validation_summary = _read_json(validation_summary_path)
    validation_path = Path(validation_summary["temporal"]["manifest"])
    validation_rows = _read_temporal_csv(validation_path, validation_path.parents[1], expected_role="VALIDATION")
    train_scenarios = sorted({row["scenario_id"] for row in train_rows})
    validation_scenarios = sorted({row["scenario_id"] for row in validation_rows})
    train_episodes = sorted({row["episode_id"] for row in train_rows})
    validation_episodes = sorted({row["episode_id"] for row in validation_rows})
    train_hashes = {row["source_mcap_sha256"] for row in train_rows}
    validation_hashes = {row["source_mcap_sha256"] for row in validation_rows}
    raw_roots = {
        "train": sim_root / "userdata/physicar_e2e/random_cone_1p0_v1/train_raw",
        "validation": config.external(sim_root, "validation_raw"),
    }
    raw_directory_names = {
        name: sorted(item.name for item in root.iterdir() if item.is_dir() and item.name != "_interrupted") if root.is_dir() else []
        for name, root in raw_roots.items()
    }
    holdout_tokens = ("s11", "s12", "scenario_11", "scenario_12")
    manifest_holdout_rows = [
        row["sequence_id"] for row in [*train_rows, *validation_rows]
        if row["scenario_id"] in HOLDOUT_SCENARIOS or any(token in row["episode_id"].lower() for token in holdout_tokens)
    ]
    raw_holdout_dirs = [
        f"{name}/{entry}" for name, entries in raw_directory_names.items() for entry in entries
        if any(token in entry.lower() for token in holdout_tokens)
    ]
    gates = {
        "frozen_train_hash_exact": sha256_file(train_path) == TRAIN_MANIFEST_SHA256,
        "train_only_s01_s08": train_scenarios == list(TRAIN_SCENARIOS),
        "train_exact_16_episode_ids": len(train_episodes) == 16 and all(item.startswith("train_s") for item in train_episodes),
        "validation_only_s09_s10": validation_scenarios == list(VALIDATION_SCENARIOS),
        "validation_exact_episode_ids": validation_episodes == sorted(VALIDATION_EPISODES),
        "source_bag_hashes_disjoint": not train_hashes.intersection(validation_hashes),
        "holdout_absent_from_manifests": not manifest_holdout_rows,
        "holdout_absent_from_train_validation_bag_roots": not raw_holdout_dirs,
        "no_holdout_expert_labels": not manifest_holdout_rows,
    }
    report = {
        "version": VERSION + "_leakage_audit", "stage": stage,
        "generated_utc": utc_now(), "result": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates, "train": {
            "manifest": str(train_path), "sha256": sha256_file(train_path),
            "sequence_count": len(train_rows), "scenario_ids": train_scenarios,
            "episode_ids": train_episodes, "source_hash_count": len(train_hashes),
        },
        "validation": {
            "manifest": str(validation_path), "sha256": sha256_file(validation_path),
            "sequence_count": len(validation_rows), "scenario_ids": validation_scenarios,
            "episode_ids": validation_episodes, "source_hash_count": len(validation_hashes),
        },
        "unseen_holdout": {
            "scenario_ids": list(HOLDOUT_SCENARIOS), "manifest_rows": manifest_holdout_rows,
            "raw_bag_directories": raw_holdout_dirs, "expert_bags_collected": 0,
            "expert_labels_extracted": 0, "images_in_train_or_validation": 0,
        },
        "raw_directory_names": raw_directory_names,
    }
    output = config.result_dir(repo, "live") / "audits" / f"leakage_{stage}.json"
    write_json(output, report)
    if report["result"] != "PASS":
        raise R1GateError(f"data leakage audit failed at {stage}")
    return report


def load_training_rows(repo: Path, config: R1Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = Path(config.frozen_train["manifest_path"])
    train_rows = _read_temporal_csv(train_path, train_path.parents[1], expected_role="TRAIN")
    validation_summary = _read_json(config.result_dir(repo, "validation_dataset") / "summary.json")
    if validation_summary.get("result") != "PASS" or (validation_summary.get("visual_qc") or {}).get("result") != "PASS":
        raise R1GateError("validation dataset and visual QC must PASS before training")
    validation_path = Path(validation_summary["temporal"]["manifest"])
    validation_rows = _read_temporal_csv(validation_path, validation_path.parents[1], expected_role="VALIDATION")
    if len(train_rows) != TRAIN_SEQUENCE_COUNT:
        raise R1GateError("R1 training must use exactly 6,706 sequences")
    if {row["scenario_id"] for row in train_rows} != set(TRAIN_SCENARIOS):
        raise R1GateError("R1 training source is not exactly S01--S08")
    if {row["scenario_id"] for row in validation_rows} != set(VALIDATION_SCENARIOS):
        raise R1GateError("R1 offline validation source is not exactly S09--S10")
    if {row["source_mcap_sha256"] for row in train_rows}.intersection({row["source_mcap_sha256"] for row in validation_rows}):
        raise R1GateError("TRAIN/VALIDATION source MCAP leakage")
    forbidden = ("v9", "c1", "1p8", "1.8", "fixed_cone", "dagger", "low_speed")
    source_strings = [str(train_path), *(row["episode_id"] for row in train_rows)]
    if any(token in value.lower() for token in forbidden for value in source_strings):
        raise R1GateError("excluded prior-model/training source entered R1 training")
    identities = {
        "train_manifest": str(train_path), "train_manifest_sha256": sha256_file(train_path),
        "train_sequence_count": len(train_rows), "train_scenarios": list(TRAIN_SCENARIOS),
        "train_episode_count": len({row["episode_id"] for row in train_rows}),
        "validation_manifest": str(validation_path), "validation_manifest_sha256": sha256_file(validation_path),
        "validation_sequence_count": len(validation_rows), "validation_scenarios": list(VALIDATION_SCENARIOS),
        "validation_episode_count": len({row["episode_id"] for row in validation_rows}),
        "excluded_sources": {
            "one_point_eight_mps": True, "V9": True, "C1": True,
            "fixed_cone": True, "DAgger": True, "low_speed": True,
        },
    }
    return train_rows, validation_rows, identities


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_temporal_resumable(
    train_rows: Sequence[dict[str, Any]], validation_rows: Sequence[dict[str, Any]],
    training_config: dict[str, Any], device: Any, state_path: Path, checkpoint_path: Path,
    identity: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """One logical optimization run with epoch-transaction crash recovery."""
    import torch
    from torch.utils.data import DataLoader

    expected_identity = {
        "task_config_sha256": identity["task_config_sha256"],
        "train_manifest_sha256": identity["train_manifest_sha256"],
        "validation_manifest_sha256": identity["validation_manifest_sha256"],
    }
    model = build_temporal_pilotnet().to(device)
    generator = torch.Generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("identity") != expected_identity or state.get("training_config") != training_config:
            raise R1GateError("interrupted training state identity/config mismatch")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        generator.set_state(state["data_loader_generator_state"].cpu())
        epoch_completed = int(state["epoch_completed"])
        best = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        best_train = float(state["best_train_loss"])
        stale = int(state["stale_epochs"])
        history = list(state["history"])
        best_state = state.get("best_model_state_dict")
        completed = bool(state.get("completed", False))
    else:
        set_reproducible_seed(int(training_config["seed"]))
        model = build_temporal_pilotnet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config["learning_rate"])
        generator.manual_seed(int(training_config["seed"]))
        epoch_completed, best, best_epoch, best_train, stale = 0, math.inf, 0, math.inf, 0
        history, best_state, completed = [], None, False
        _atomic_torch_save(state_path, {
            "version": TRAINING_VERSION + "_resumable_state", "identity": expected_identity,
            "training_config": training_config, "epoch_completed": 0,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "best_validation_loss": best, "best_epoch": best_epoch,
            "best_train_loss": best_train, "stale_epochs": stale,
            "history": history, "best_model_state_dict": best_state, "completed": False,
        })
    maximum = float(training_config["max_steering_rad"])
    train_loader = DataLoader(
        TemporalDataset(train_rows, maximum), batch_size=training_config["batch_size"],
        shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        TemporalDataset(validation_rows, maximum), batch_size=training_config["batch_size"],
        shuffle=False, num_workers=0,
    )
    if not completed:
        for epoch in range(epoch_completed + 1, int(training_config["max_epochs"]) + 1):
            train_loss = _epoch(model, train_loader, device, optimizer)
            validation_loss = _epoch(model, validation_loader, device)
            item = {
                "epoch": epoch, "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation_loss,
            }
            history.append(item)
            print(json.dumps(item), flush=True)
            if best - validation_loss > float(training_config["minimum_improvement"]):
                best, best_epoch, best_train, stale = validation_loss, epoch, train_loss, 0
                best_state = _cpu_state_dict(model)
            else:
                stale += 1
            should_stop = stale >= int(training_config["early_stopping_patience"])
            state = {
                "version": TRAINING_VERSION + "_resumable_state", "identity": expected_identity,
                "training_config": training_config, "epoch_completed": epoch,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "data_loader_generator_state": generator.get_state(),
                "best_validation_loss": best, "best_epoch": best_epoch,
                "best_train_loss": best_train, "stale_epochs": stale,
                "history": history, "best_model_state_dict": best_state,
                "completed": should_stop or epoch == int(training_config["max_epochs"]),
            }
            _atomic_torch_save(state_path, state)
            if should_stop:
                break
        state = torch.load(state_path, map_location=device, weights_only=False)
        completed = bool(state["completed"])
        best_state = state["best_model_state_dict"]
    if not completed or best_state is None or best_epoch <= 0:
        raise R1GateError("resumable training did not reach a completed best checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_payload = {
        "model_state_dict": best_state, "epoch": best_epoch,
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "training_config": training_config, "identity": expected_identity,
        "initialized_from_scratch": True,
    }
    _atomic_torch_save(checkpoint_path, checkpoint_payload)
    training = {
        "result": "PASS", "epochs_completed": len(history), "best_epoch": best_epoch,
        "train_normalized_mse_at_best": best_train,
        "validation_normalized_mse_at_best": best,
        "early_stopped": len(history) < int(training_config["max_epochs"]),
        "initialized_from_scratch": True,
        "resumed_from_completed_epoch": epoch_completed,
        "single_logical_training_run": True,
    }
    return model, training, history


def _phase_metrics(
    model: Any, rows: Sequence[dict[str, Any]], training: dict[str, Any], device: Any,
    bundle: ScenarioBundle,
) -> dict[str, Any]:
    route_length = bundle.plan.nominal.length
    regions = scenario_regions(bundle, route_length, 1.0)
    report: dict[str, Any] = {}
    for phase, (lower, upper) in regions.items():
        selected = [row for row in rows if lower <= float(row["route_progress_m"]) < upper]
        if selected:
            predictions, labels = predict_temporal(model, selected, training, device)
            report[phase] = {"route_s_m": [lower, upper], **error_metrics(predictions, labels)}
        else:
            report[phase] = {"route_s_m": [lower, upper], "sample_count": 0, "result": "NO_SAMPLES"}
    return report


def _training_plot(history: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train_normalized_mse"] for row in history], label="TRAIN")
    axis.plot([row["epoch"] for row in history], [row["validation_normalized_mse"] for row in history], label="VALIDATION S09/S10")
    axis.set_xlabel("epoch")
    axis.set_ylabel("normalized MSE")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)


def _validate_existing_training(report: dict[str, Any]) -> bool:
    if report.get("result") != "PASS" or (report.get("onnx_equivalence") or {}).get("result") != "PASS":
        return False
    for key in ("checkpoint", "onnx"):
        artifact = (report.get("artifacts") or {}).get(key) or {}
        path = Path(str(artifact.get("path", "")))
        if not path.is_file() or sha256_file(path) != artifact.get("sha256") or path.stat().st_size != artifact.get("size_bytes"):
            return False
    seal = report.get("freeze_seal") or {}
    seal_path = Path(str(seal.get("path", "")))
    return seal_path.is_file() and sha256_file(seal_path) == seal.get("sha256")


def training_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    result_dir = config.result_dir(repo, "training")
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if _validate_existing_training(existing):
            return existing
        if existing.get("result") == "PASS":
            raise R1GateError("frozen R1 training evidence/artifact identity changed")
    if disk_state("/")["available_bytes"] < MIN_TRAIN_BYTES:
        raise R1GateError("root disk has less than 5.5 GiB before R1 training")
    leakage = leakage_audit(repo, sim_root, config, stage="before_training")
    train_rows, validation_rows, sources = load_training_rows(repo, config)
    count = sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters())
    if count != 255_819:
        raise R1GateError(f"Temporal PilotNet parameter count is {count}, expected 255819")
    live_result_dir = config.result_dir(repo, "live")
    if any(
        path
        for subdir in ("validation_attempts", "holdout_attempts")
        for path in (live_result_dir / subdir).glob("*.json")
    ):
        raise R1GateError("live evidence exists before model training/freeze")
    external = config.external(sim_root, "r1")
    checkpoint = external / "checkpoints/random_cone_temporal_r1_best.pt"
    state_path = external / "checkpoints/random_cone_temporal_r1_training_state.pt"
    onnx_path = external / "onnx/random_cone_temporal_r1.onnx"
    plot_path = external / "plots/training_history.png"
    snapshot_path = external / "training_config_snapshot.json"
    marker_path = result_dir / "training.started.json"
    source_identity = {
        "task_config_sha256": config.sha256,
        "train_manifest_sha256": sources["train_manifest_sha256"],
        "validation_manifest_sha256": sources["validation_manifest_sha256"],
    }
    if marker_path.is_file():
        marker = _read_json(marker_path)
        if marker.get("source_identity") != source_identity:
            raise R1GateError("existing training marker source identity mismatch")
    else:
        write_json(marker_path, {
            "status": "ONE_LOGICAL_R1_TRAINING_RUN_STARTED", "started_utc": utc_now(),
            "source_identity": source_identity, "initialization": "from_scratch",
            "resumable_epoch_transactions": True, "retraining_permitted": False,
        })
    write_json(snapshot_path, {
        "version": TRAINING_VERSION + "_config_snapshot", "task_config_sha256": config.sha256,
        "training": config.training, "sources": sources,
    })
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, training_result, history = train_temporal_resumable(
        train_rows, validation_rows, config.training, device, state_path, checkpoint,
        {**source_identity, "task_config_sha256": config.sha256},
    )
    predictions, labels = predict_temporal(model, validation_rows, config.training, device)
    combined = error_metrics(predictions, labels)
    per_scenario: dict[str, Any] = {}
    expert_config = RandomConeConfig.load(repo / config.frozen_expert["config_path"], repo, sim_root)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in verify_frozen_scenarios(expert_config, sim_root)}
    for scenario in VALIDATION_SCENARIOS:
        selected = [row for row in validation_rows if row["scenario_id"] == scenario]
        scenario_predictions, scenario_labels = predict_temporal(model, selected, config.training, device)
        per_scenario[scenario] = {
            **error_metrics(scenario_predictions, scenario_labels),
            "obstacle_phases": _phase_metrics(model, selected, config.training, device, bundles[scenario]),
        }
    export_temporal_onnx(model, onnx_path, config.training)
    equivalence = validate_equivalence(model, validation_rows, onnx_path, config.training)
    if equivalence.get("result") != "PASS":
        raise R1GateError("R1 PyTorch/ONNX equivalence failed")
    _training_plot(history, plot_path)
    artifacts = {
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
        "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
        "training_state": {"path": str(state_path), "size_bytes": state_path.stat().st_size, "sha256": sha256_file(state_path)},
        "training_config_snapshot": {"path": str(snapshot_path), "size_bytes": snapshot_path.stat().st_size, "sha256": sha256_file(snapshot_path)},
        "training_plot": {"path": str(plot_path), "size_bytes": plot_path.stat().st_size, "sha256": sha256_file(plot_path)},
    }
    freeze_payload = {
        "version": TRAINING_VERSION + "_freeze", "frozen_utc": utc_now(),
        "frozen_before_any_live_neural_attempt": True,
        "model_name": "Random-Cone Temporal PilotNet R1", "architecture": {
            "input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
            "parameter_count": count, "architecture_identity": "Temporal PilotNet V9/C1",
        },
        "training_config_sha256": artifacts["training_config_snapshot"]["sha256"],
        "task_config_sha256": config.sha256,
        "train_manifest": {"path": sources["train_manifest"], "sha256": sources["train_manifest_sha256"], "sequence_count": len(train_rows)},
        "validation_manifest": {"path": sources["validation_manifest"], "sha256": sources["validation_manifest_sha256"], "sequence_count": len(validation_rows)},
        "checkpoint": artifacts["checkpoint"], "onnx": artifacts["onnx"],
        "onnx_equivalence": equivalence, "offline_validation": {"combined": combined, "per_scenario": per_scenario},
        "training_from_scratch": True, "single_training_run": True,
        "holdout_scenarios_observed_by_model_before_freeze": [],
    }
    external_freeze = external / "freeze.json"
    compact_freeze = result_dir / "freeze.json"
    write_json(external_freeze, freeze_payload)
    write_json(compact_freeze, freeze_payload)
    freeze_hash = sha256_file(external_freeze)
    if freeze_hash != sha256_file(compact_freeze):
        raise R1GateError("external/compact freeze identity mismatch")
    seal_payload = {
        "version": TRAINING_VERSION + "_freeze_seal", "sealed_utc": utc_now(),
        "freeze_sha256": freeze_hash, "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "task_config_sha256": config.sha256,
        "train_manifest_sha256": sources["train_manifest_sha256"],
        "validation_manifest_sha256": sources["validation_manifest_sha256"],
        "live_attempt_count_before_seal": 0, "retraining_or_tuning_after_seal_permitted": False,
    }
    external_seal = external / "freeze_seal.json"
    compact_seal = result_dir / "freeze_seal.json"
    write_json(external_seal, seal_payload)
    write_json(compact_seal, seal_payload)
    if sha256_file(external_seal) != sha256_file(compact_seal):
        raise R1GateError("external/compact freeze seal identity mismatch")
    disk_after = disk_state("/")
    report = {
        "version": TRAINING_VERSION, "generated_utc": utc_now(), "result": "PASS",
        "task_config_sha256": config.sha256, "training_sources": sources,
        "proof_excluded_sources": sources["excluded_sources"],
        "architecture": {
            "input_shape": ["N", 9, 66, 200], "output_shape": ["N", 1],
            "parameter_count": count, "first_conv": "9->24, 5x5, stride 2",
            "architecture_identity": "validated Temporal PilotNet V9/C1",
        },
        "training": training_result, "epochs": history, "device": str(device),
        "offline_validation": {"combined": combined, "per_scenario": per_scenario},
        "optional_preserved_model_comparison": {"result": "NOT_RUN", "hard_gate": False, "reason": "R1 single-run gate does not depend on contextual V9/C1 evaluation"},
        "onnx_contract": {"checker": "PASS", "input": ["batch", 9, 66, 200], "output": ["batch", 1]},
        "onnx_equivalence": equivalence, "artifacts": artifacts,
        "freeze": {"path": str(external_freeze), "sha256": freeze_hash, "compact_path": str(compact_freeze)},
        "freeze_seal": {"path": str(external_seal), "sha256": sha256_file(external_seal), "compact_path": str(compact_seal)},
        "leakage_audit_before_training": leakage, "disk_after_training": disk_after,
        "live_disk_gate_pass": disk_after["available_bytes"] >= MIN_LIVE_BYTES,
        "model_frozen_before_live": True, "retraining_performed": False,
        "holdout_data_used": False, "holdout_performance_observed": False,
    }
    write_json(summary_path, report)
    write_json(marker_path, {
        "status": "ONE_LOGICAL_R1_TRAINING_RUN_COMPLETED_AND_FROZEN",
        "completed_utc": utc_now(), "source_identity": source_identity,
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "onnx_sha256": artifacts["onnx"]["sha256"],
        "freeze_seal_sha256": report["freeze_seal"]["sha256"],
        "retraining_permitted": False,
    })
    _write_report(result_dir / "REPORT.md", "Random-Cone Temporal PilotNet R1 Training", "PASS", [
        f"TRAIN sequences: {len(train_rows)} (S01--S08 only)",
        f"VALIDATION sequences: {len(validation_rows)} (S09--S10 only)",
        f"Best epoch: {training_result['best_epoch']}",
        f"Combined validation MAE: {combined['mae_rad']:.6f} rad",
        f"Checkpoint SHA-256: {artifacts['checkpoint']['sha256']}",
        f"ONNX SHA-256: {artifacts['onnx']['sha256']}",
        "The model/config/manifests were sealed before live validation or holdout execution.",
    ])
    if not report["live_disk_gate_pass"]:
        raise R1GateError("R1 trained and frozen, but root disk is below 4.5 GiB; live runs blocked")
    return report


def verify_frozen_model(repo: Path, config: R1Config) -> dict[str, Any]:
    report = _read_json(config.result_dir(repo, "training") / "summary.json")
    if not _validate_existing_training(report) or report.get("model_frozen_before_live") is not True:
        raise R1GateError("R1 training/freeze evidence is not a valid PASS")
    if report.get("task_config_sha256") != config.sha256:
        raise R1GateError("R1 frozen task config identity changed")
    sources = report["training_sources"]
    if sha256_file(Path(sources["train_manifest"])) != sources["train_manifest_sha256"]:
        raise R1GateError("frozen TRAIN manifest changed after training")
    if sha256_file(Path(sources["validation_manifest"])) != sources["validation_manifest_sha256"]:
        raise R1GateError("frozen VALIDATION manifest changed after training")
    freeze = _read_json(Path(report["freeze"]["path"]))
    seal = _read_json(Path(report["freeze_seal"]["path"]))
    if (
        sha256_file(Path(report["freeze"]["path"])) != report["freeze"]["sha256"]
        or seal.get("freeze_sha256") != report["freeze"]["sha256"]
        or seal.get("checkpoint_sha256") != report["artifacts"]["checkpoint"]["sha256"]
        or seal.get("onnx_sha256") != report["artifacts"]["onnx"]["sha256"]
        or seal.get("live_attempt_count_before_seal") != 0
        or seal.get("retraining_or_tuning_after_seal_permitted") is not False
        or freeze.get("frozen_before_any_live_neural_attempt") is not True
    ):
        raise R1GateError("R1 freeze seal contract failed")
    return report


def inference_config(config: R1Config, world: str) -> InferenceConfig:
    payload = copy.deepcopy(config.live)
    payload.update({
        "version": TRAINING_VERSION + "_live_inference", "expected_world": world,
        "camera_transport": "HTTP JPEG", "smoke_speeds_mps": [1.0],
        "maximum_smoke_runs": 1, "maximum_total_attempts": 2,
    })
    return InferenceConfig(payload)


def summarize_neural_cone_run(
    run: dict[str, Any], observer: RandomConeObserver, bundle: ScenarioBundle,
) -> dict[str, Any]:
    rows = observer.samples
    minimum = min(rows, key=lambda row: float(row["cone_clearance_m"]), default=None)
    failure = str(run.get("failure") or "").lower()
    pose_failures = int("pose did not change meaningfully" in failure)
    clock_failures = int("clock did not advance" in failure or "clock moved backward" in failure)
    infrastructure = bool(
        run.get("temporal_input_failure") or run.get("api_failures")
        or run.get("liveness_failures") or pose_failures or clock_failures
        or not run.get("safe_stop_success")
        or any(token in failure for token in (
            "simulator state changed", "unexpected world", "invalid track boundary",
            "unavailable", "control rejected", "get ", "post ",
        ))
    )
    clearance = None if minimum is None else float(minimum["cone_clearance_m"])
    practical_pass = (
        run.get("result") == "PASS" and clearance is not None and clearance > 0.0
        and observer.intersection_occurred is False
        and observer.contact_or_movement_occurred is False
        and observer.recovery_success is True
        and not infrastructure
    )
    if infrastructure:
        classification = "INFRA_FAIL"
    elif practical_pass:
        classification = "RANDOM_CONE_POLICY_PASS"
    else:
        classification = "RANDOM_CONE_POLICY_FAIL"
    run.update({
        "classification": classification, "scenario_id": bundle.scenario.scenario_id,
        "role": bundle.scenario.role, "curvature_class": bundle.scenario.curvature_class,
        "cone_s_m": bundle.scenario.route_s_m, "cone_x_m": bundle.scenario.x_m,
        "cone_y_m": bundle.scenario.y_m, "chosen_expert_bypass_side_qa_only": bundle.plan.side,
        "minimum_footprint_to_cone_clearance_m": clearance,
        "minimum_cone_clearance_route_s_m": None if minimum is None else float(minimum["route_s_m"]),
        "footprint_cone_intersection_occurred": observer.intersection_occurred,
        "cone_contact_or_movement_occurred": observer.contact_or_movement_occurred,
        "cone_contact_or_intersection_occurred": observer.intersection_occurred or observer.contact_or_movement_occurred,
        "recovery_success": observer.recovery_success,
        "recovery_time_s": observer.recovery_time_s, "recovery_cte_m": observer.recovery_cte_m,
        "pose_failures": pose_failures, "clock_failures": clock_failures,
        "practical_success_contract": "strictly positive clearance and no cone contact/intersection",
        "clearance_0p05_m_not_required": True,
        "privileged_qa_fields_not_model_inputs": [
            "scenario_id", "cone pose", "route pose/progress", "clearance", "CTE",
            "avoidance side", "recovery telemetry",
        ],
        "neural_observation_fields": [
            "camera_yuv_t_minus_2", "camera_yuv_t_minus_1", "camera_yuv_t",
        ],
    })
    return run


def run_live_once(
    client: SimClient, model: TemporalOnnxModel, config: R1Config,
    expert: RandomConeConfig, bundle: ScenarioBundle, sim_root: Path,
    *, prepare: Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]] = _post_settle_preflight,
    run_policy: Callable[..., dict[str, Any]] = run_temporal_live,
) -> dict[str, Any]:
    if errors := client.safe_stop():
        raise R1GateError("pre-scenario safe stop failed: " + "; ".join(errors))
    initial, activation, preflight = prepare(
        client, expert, bundle, sim_root, 2.0,
    )
    expected_control = {
        "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
        "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
    }
    if preflight.get("fixed_control") != expected_control:
        raise R1GateError("frozen Expert/scenario preflight control changed")
    live_config = inference_config(config, expert.world_name(bundle.scenario.scenario_id))
    observer = RandomConeObserver(client, initial.route, bundle, expert)
    run = run_policy(observer, model, live_config, initial, 1.0)
    return {
        "world_activation": activation, "preflight": preflight,
        "run": summarize_neural_cone_run(run, observer, bundle),
    }


def live_retry_decision(classification: str, attempt_number: int) -> str:
    if classification == "RANDOM_CONE_POLICY_PASS":
        return "CONTINUE"
    if classification == "RANDOM_CONE_POLICY_FAIL":
        return "STOP_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < 2:
        return "RETRY_INFRA"
    return "STOP_INFRA"


def validation_allows_unseen(validation_report: dict[str, Any]) -> bool:
    return (
        validation_report.get("result") == "PASS"
        and [item.get("scenario_id") for item in validation_report.get("scenarios", [])] == list(VALIDATION_SCENARIOS)
        and all(item.get("classification") == "RANDOM_CONE_POLICY_PASS" for item in validation_report.get("scenarios", []))
    )


def holdout_next_scenario(records: Sequence[dict[str, Any]]) -> str | None:
    by_id = {item.get("scenario_id"): item for item in records}
    if "11" not in by_id:
        return "11"
    if by_id["11"].get("classification") != "RANDOM_CONE_POLICY_PASS":
        return None
    if "12" not in by_id:
        return "12"
    return None


def _valid_live_record(
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
    )


def _run_live_group(
    repo: Path, sim_root: Path, config: R1Config, *, group: str,
) -> dict[str, Any]:
    if group not in {"validation", "holdout"}:
        raise ValueError(group)
    training = verify_frozen_model(repo, config)
    if disk_state("/")["available_bytes"] < MIN_LIVE_BYTES:
        raise R1GateError("root disk is below 4.5 GiB; new live neural experiment blocked")
    if group == "validation":
        scenario_ids, role = VALIDATION_SCENARIOS, "VALIDATION"
        leakage = leakage_audit(repo, sim_root, config, stage="before_live_validation")
    else:
        validation_report = _read_json(config.result_dir(repo, "live") / "live_validation_summary.json")
        if not validation_allows_unseen(validation_report):
            raise R1GateError("S09/S10 live validation did not both PASS; unseen execution blocked")
        leakage = leakage_audit(repo, sim_root, config, stage="before_unseen")
        scenario_ids, role = HOLDOUT_SCENARIOS, "UNSEEN_HOLDOUT"
    train_task = load_train_task_config(repo / "configs/random_cone_train_data_1p0_v1.json", repo)
    expert, all_bundles, _ = audit_frozen_expert(repo, sim_root, train_task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles}
    onnx_path = Path(training["artifacts"]["onnx"]["path"])
    model = TemporalOnnxModel(onnx_path)
    result_dir = config.result_dir(repo, "live")
    attempts_dir = result_dir / f"{group}_attempts"
    scenarios_dir = result_dir / f"{group}_scenarios"
    states_dir = result_dir / f"{group}_states"
    summary_path = result_dir / f"live_{group}_summary.json"
    records: list[dict[str, Any]] = []
    for scenario in scenario_ids:
        path = scenarios_dir / f"scenario_{scenario}.json"
        if path.is_file():
            existing = _read_json(path)
            if not _valid_live_record(existing, scenario, role, training):
                raise R1GateError(f"completed live S{scenario} evidence identity changed")
            records.append(existing)
    if group == "validation" and records and records[0]["scenario_id"] == "09" and records[0]["classification"] != "RANDOM_CONE_POLICY_PASS":
        scenario_ids = ()
    if group == "holdout":
        next_holdout = holdout_next_scenario(records)
        if next_holdout is None and len(records) < 2:
            scenario_ids = ()
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    final_errors: list[str] = []
    try:
        for scenario in scenario_ids:
            if any(item["scenario_id"] == scenario for item in records):
                continue
            if group == "validation":
                if scenario == "10" and (not records or records[-1]["classification"] != "RANDOM_CONE_POLICY_PASS"):
                    break
            else:
                if scenario == "12" and (not records or next((item for item in records if item["scenario_id"] == "11"), {}).get("classification") != "RANDOM_CONE_POLICY_PASS"):
                    break
            verify_frozen_model(repo, config)
            state_path = states_dir / f"scenario_{scenario}.json"
            final_path = scenarios_dir / f"scenario_{scenario}.json"
            attempt_paths = sorted(attempts_dir.glob(f"scenario_{scenario}_attempt_*.json"))
            # Promote a completed policy attempt after a crash between attempt/final writes.
            if attempt_paths:
                latest = _read_json(attempt_paths[-1])
                if latest.get("classification") in {"RANDOM_CONE_POLICY_PASS", "RANDOM_CONE_POLICY_FAIL"}:
                    write_json(final_path, latest)
                    records.append(latest)
                    if latest["classification"] != "RANDOM_CONE_POLICY_PASS":
                        break
                    continue
            attempts = len(attempt_paths)
            if state_path.is_file():
                state = _read_json(state_path)
                started = int(state.get("attempt_number", 0))
                if started > attempts:
                    attempts = started
                    interrupted_path = attempts_dir / f"scenario_{scenario}_attempt_{started:02d}.json"
                    write_json(interrupted_path, {
                        "version": LIVE_VERSION + "_interrupted", "scenario_id": scenario,
                        "role": role, "attempt_number": started, "classification": "INFRA_FAIL",
                        "failure_reason": "host/process interruption before finalized live evidence",
                        "reconstructed_utc": utc_now(),
                    })
            if attempts >= 2:
                break
            attempt = attempts + 1
            while attempt <= 2:
                frozen = verify_frozen_model(repo, config)
                write_json(state_path, {
                    "status": "STARTED_UNFINALIZED", "scenario_id": scenario,
                    "role": role, "attempt_number": attempt, "started_utc": utc_now(),
                    "onnx_sha256": frozen["artifacts"]["onnx"]["sha256"],
                    "freeze_seal_sha256": frozen["freeze_seal"]["sha256"],
                })
                attempt_path = attempts_dir / f"scenario_{scenario}_attempt_{attempt:02d}.json"
                record: dict[str, Any] = {
                    "version": LIVE_VERSION + "_scenario", "generated_utc": utc_now(),
                    "scenario_id": scenario, "role": role, "attempt_number": attempt,
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
                    live = run_live_once(client, model, config, expert, bundles[scenario], sim_root)
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
                write_json(attempt_path, record)
                print(json.dumps({
                    "stage": f"live_{group}", "scenario": scenario, "attempt": attempt,
                    "classification": record["classification"],
                    "clearance_m": (record.get("run") or {}).get("minimum_footprint_to_cone_clearance_m"),
                    "completion": (record.get("run") or {}).get("route_completion_fraction"),
                }), flush=True)
                decision = live_retry_decision(record["classification"], attempt)
                if decision in {"CONTINUE", "STOP_GENUINE_FAILURE"}:
                    write_json(final_path, record)
                    write_json(state_path, {
                        "status": "FINALIZED_VALID_POLICY_EVALUATION", "scenario_id": scenario,
                        "role": role, "attempt_number": attempt, "classification": record["classification"],
                        "finalized_utc": utc_now(), "do_not_repeat": True,
                    })
                    records.append(record)
                    break
                if decision == "RETRY_INFRA":
                    if errors := client.safe_stop():
                        record["failure_reason"] = (record.get("failure_reason") or "") + "; safe stop failed before replacement: " + "; ".join(errors)
                        write_json(attempt_path, record)
                        break
                    attempt += 1
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
    intended_count = 2
    pass_count = sum(item.get("classification") == "RANDOM_CONE_POLICY_PASS" for item in records)
    policy_fail = any(item.get("classification") == "RANDOM_CONE_POLICY_FAIL" for item in records)
    if policy_fail:
        result = "FAIL"
        category = "VALIDATION_FAIL" if group == "validation" else "UNSEEN_FAIL"
    elif pass_count == intended_count and len(records) == intended_count and not final_errors and restoration.get("result") == "PASS":
        result, category = "PASS", "VALIDATION_PASS" if group == "validation" else "UNSEEN_PASS"
    else:
        result, category = "INCONCLUSIVE", "INCONCLUSIVE"
    report = {
        "version": LIVE_VERSION + f"_{group}", "generated_utc": utc_now(),
        "result": result, "category": category, "role": role,
        "intended_scenario_ids": list(VALIDATION_SCENARIOS if group == "validation" else HOLDOUT_SCENARIOS),
        "scenarios": records, "valid_policy_run_count": len(records), "pass_count": pass_count,
        "maximum_valid_runs_per_scenario": 1, "maximum_infrastructure_replacements_per_scenario": 1,
        "model_frozen_before_all_attempts": True,
        "onnx_sha256": training["artifacts"]["onnx"]["sha256"],
        "checkpoint_sha256": training["artifacts"]["checkpoint"]["sha256"],
        "freeze_seal_sha256": training["freeze_seal"]["sha256"],
        "leakage_audit": leakage, "final_safe_stop_success": not final_errors,
        "final_safe_stop_errors": final_errors, "world_restoration": restoration,
        "disk_at_completion": disk_state("/"),
        "holdout_bags_collected": 0, "holdout_labels_extracted": 0,
    }
    write_json(summary_path, report)
    _write_report(result_dir / f"LIVE_{group.upper()}_REPORT.md", f"Random-Cone Temporal R1 Live {group.title()}", category, [
        *[f"S{item['scenario_id']} ({item['role']}): {item['classification']}, clearance={(item.get('run') or {}).get('minimum_footprint_to_cone_clearance_m')} m" for item in records],
        f"Model ONNX SHA-256: {report['onnx_sha256']}",
        "Each valid policy evaluation is final and is never repeated.",
    ])
    return report


def live_validation_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    return _run_live_group(repo, sim_root, config, group="validation")


def live_unseen_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    return _run_live_group(repo, sim_root, config, group="holdout")


def final_stage(repo: Path, sim_root: Path, config: R1Config) -> dict[str, Any]:
    training = verify_frozen_model(repo, config)
    validation_path = config.result_dir(repo, "live") / "live_validation_summary.json"
    holdout_path = config.result_dir(repo, "live") / "live_holdout_summary.json"
    validation = _read_json(validation_path) if validation_path.is_file() else None
    holdout = _read_json(holdout_path) if holdout_path.is_file() else None
    leakage = leakage_audit(repo, sim_root, config, stage="final")
    if validation and validation.get("category") == "VALIDATION_FAIL":
        category = "VALIDATION_FAIL"
    elif validation and validation.get("result") == "PASS" and holdout and holdout.get("category") == "UNSEEN_FAIL":
        category = "UNSEEN_FAIL"
    elif validation and validation.get("result") == "PASS" and holdout and holdout.get("result") == "PASS":
        category = "FULL_PASS"
    else:
        category = "INCONCLUSIVE"
    scenarios = [
        *([] if validation is None else validation.get("scenarios", [])),
        *([] if holdout is None else holdout.get("scenarios", [])),
    ]
    by_scenario = {item["scenario_id"]: item for item in scenarios}
    scenario_status: dict[str, Any] = {}
    for scenario, role in (("09", "VALIDATION"), ("10", "VALIDATION"), ("11", "UNSEEN_HOLDOUT"), ("12", "UNSEEN_HOLDOUT")):
        if scenario in by_scenario:
            scenario_status[scenario] = {
                "role": role, "status": "EXECUTED",
                "classification": by_scenario[scenario]["classification"],
            }
        elif scenario == "10" and category == "VALIDATION_FAIL":
            scenario_status[scenario] = {
                "role": role, "status": "NOT_RUN",
                "reason": "blocked by genuine S09 neural policy failure",
            }
        elif scenario in HOLDOUT_SCENARIOS and category == "VALIDATION_FAIL":
            scenario_status[scenario] = {
                "role": role, "status": "NOT_RUN",
                "reason": "unseen holdout blocked because live validation did not PASS",
            }
        else:
            scenario_status[scenario] = {"role": role, "status": "NOT_RUN", "reason": "pipeline gate not reached"}
    failure_analysis: dict[str, Any] | None = None
    if category == "VALIDATION_FAIL" and "09" in by_scenario:
        failed = by_scenario["09"].get("run") or {}
        failure_analysis = {
            "scenario_id": "09", "geometry_class": failed.get("curvature_class"),
            "failure": failed.get("failure"),
            "failure_route_s_m": failed.get("final_route_s_m"),
            "cone_route_s_m": failed.get("cone_s_m"),
            "route_completion_fraction": failed.get("route_completion_fraction"),
            "failure_phase": "avoidance_before_cone" if (
                failed.get("final_route_s_m") is not None
                and failed.get("cone_s_m") is not None
                and float(failed["final_route_s_m"]) < float(failed["cone_s_m"])
            ) else "unknown",
            "maximum_cte_m": failed.get("max_cte_m"),
            "off_track_duration_s": failed.get("off_track_total_duration_s"),
            "minimum_cone_clearance_m": failed.get("minimum_footprint_to_cone_clearance_m"),
            "cone_contact_or_intersection": failed.get("cone_contact_or_intersection_occurred"),
            "recovery_success": failed.get("recovery_success"),
            "temporal_input_failure": failed.get("temporal_input_failure"),
            "api_pose_clock_failures": {
                "api": failed.get("api_failures"), "pose": failed.get("pose_failures"),
                "clock": failed.get("clock_failures"), "liveness": failed.get("liveness_failures"),
            },
            "timing_slips_over_100ms": failed.get("timing_slips_over_100ms"),
            "steering_saturation_fraction": failed.get("steering_saturation_fraction"),
            "safe_stop_success": failed.get("safe_stop_success"),
            "evidence_based_interpretation": (
                "Genuine closed-loop accumulation/distribution-shift failure on the "
                "moderate-left S09 avoidance approach; offline target agreement did not "
                "translate to stable closed-loop recovery. This is analysis only and was "
                "not used to tune or retrain R1."
            ),
        }
    report = {
        "version": LIVE_VERSION, "generated_utc": utc_now(), "result": category,
        "frozen_train": {
            "manifest": config.frozen_train["manifest_path"],
            "sha256": config.frozen_train["manifest_sha256"],
            "sequence_count": config.frozen_train["sequence_count"],
        },
        "training_summary_sha256": sha256_file(config.result_dir(repo, "training") / "summary.json"),
        "checkpoint": training["artifacts"]["checkpoint"], "onnx": training["artifacts"]["onnx"],
        "freeze": training["freeze"], "freeze_seal": training["freeze_seal"],
        "offline_validation": training["offline_validation"],
        "live_validation": validation, "live_unseen_holdout": holdout,
        "live_scenarios": scenarios, "scenario_status": scenario_status,
        "failure_analysis": failure_analysis, "leakage_audit": leakage,
        "per_scenario_clearance_m": {
            item["scenario_id"]: (item.get("run") or {}).get("minimum_footprint_to_cone_clearance_m")
            for item in scenarios
        },
        "collision_count": sum(bool((item.get("run") or {}).get("cone_contact_or_intersection_occurred")) for item in scenarios),
        "recovery_pass_count": sum((item.get("run") or {}).get("recovery_success") is True for item in scenarios),
        "random_cone_simulator_baseline": category == "FULL_PASS",
        "repeatability_work_justified": category == "FULL_PASS",
        "real_robot_work_directly_claimed": False,
        "real_robot_transfer_requires_separate_safety_validation": True,
        "holdout_bags_collected": 0, "holdout_labels_extracted": 0,
        "retraining_after_validation_or_holdout": False,
        "disk_final": disk_state("/"), "simulator_tracked_source_status": simulator_tracked_status(sim_root),
    }
    result_dir = config.result_dir(repo, "live")
    write_json(result_dir / "summary.json", report)
    conclusion = (
        "Random-Cone Temporal PilotNet R1 completed both frozen validation scenarios and both never-trained unseen holdout cone scenarios at 1.00 m/s using only causal camera observations."
        if category == "FULL_PASS" else f"Final category: {category}; no failure was reinterpreted or tuned away."
    )
    _write_report(result_dir / "REPORT.md", "Random-Cone Temporal PilotNet R1 Final Gate", category, [
        conclusion,
        f"Live valid evaluations: {len(scenarios)}/4",
        f"Collisions/contacts: {report['collision_count']}",
        *([] if failure_analysis is None else [
            f"Failure geometry: S09 {failure_analysis['geometry_class']}, {failure_analysis['failure_phase']}",
            f"Failure: {failure_analysis['failure']}",
        ]),
        f"S10 status: {scenario_status['10']['status']} ({scenario_status['10'].get('reason', 'executed')})",
        f"S11/S12 status: {scenario_status['11']['status']}/{scenario_status['12']['status']}",
        f"Leakage audit: {leakage['result']}",
        "No S11/S12 bag or Expert label was collected or extracted.",
        "This is simulator evidence, not a real-robot success claim.",
    ])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=(
        "audit", "collect-validation", "extract-validation", "visual-qc-pass",
        "visual-qc-fail", "train-freeze", "leakage-audit", "live-validation",
        "live-unseen", "final",
    ))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--review-note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    path = args.config or repo / "configs/random_cone_temporal_r1_v1.json"
    try:
        config = load_config(path.resolve(), repo)
        sim_root = args.sim_root.expanduser().resolve()
        if args.stage == "audit":
            result = audit_stage(repo, sim_root, config)
        elif args.stage == "collect-validation":
            result = collection_stage(repo, sim_root, config)
        elif args.stage == "extract-validation":
            result = dataset_stage(repo, sim_root, config)
        elif args.stage in {"visual-qc-pass", "visual-qc-fail"}:
            result = finalize_visual_qc(repo, sim_root, config, passed=args.stage == "visual-qc-pass", review_note=args.review_note)
        elif args.stage == "train-freeze":
            result = training_stage(repo, sim_root, config)
        elif args.stage == "leakage-audit":
            result = leakage_audit(repo, sim_root, config, stage="manual")
        elif args.stage == "live-validation":
            result = live_validation_stage(repo, sim_root, config)
        elif args.stage == "live-unseen":
            result = live_unseen_stage(repo, sim_root, config)
        else:
            result = final_stage(repo, sim_root, config)
        print(json.dumps({
            "stage": args.stage, "result": result.get("result"),
            "category": result.get("category"),
            "sequence_count": (result.get("temporal") or {}).get("accepted_sequence_count"),
        }, indent=2, sort_keys=True))
        return 0 if result.get("result") in {"PASS", "PENDING_VISUAL_QC", "FULL_PASS"} else 1
    except KeyboardInterrupt:
        print("ERROR: interrupted; crash-safe artifacts were preserved for resume", flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
