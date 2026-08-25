"""Random-cone 1.0 m/s TRAIN bag collection and temporal dataset QC V1.

This module intentionally has no neural-training or model-export dependency.  It
loads the frozen Random Cone Expert, records only TRAIN scenarios 01--08 in two
complete rounds, and reuses Dataset Extractor V1's record-time causal ZOH and
image preprocessing semantics.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw

from .dataset_extractor import (
    MANIFEST_COLUMNS,
    REJECTION_REASONS,
    _iter_decoded,
    canonical_json_bytes,
    extract_episode,
    load_config as load_extractor_config,
    numeric_distribution,
    prepare_output_root,
    sha256_file,
    steering_distribution,
)
from .pilotnet_v4_repeatability import clock_health_preflight
from .random_cone_expert import (
    MAP_FAMILY,
    ROLE_IDS,
    RandomConeConfig,
    ScenarioBundle,
    _restore_world,
    activate_world,
    audit_preserved_state,
    directory_file_manifest_sha256,
    scenario_structural_preflight,
    simulator_tracked_status,
    verify_frozen_scenarios,
    verify_offline_evidence,
    verify_scenario_environment,
    wait_after_scenario_reset,
    run_random_cone_expert,
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


VERSION = "random_cone_train_data_1p0_v1"
COLLECTION_VERSION = "random_cone_train_collection_1p0_v1"
DATASET_VERSION = "random_cone_train_dataset_1p0_v1"
TRAIN_SCENARIOS = tuple(f"{number:02d}" for number in range(1, 9))
REPEAT_IDS = ("R01", "R02")
REQUIRED_TOPICS = (
    "/camera/image_raw", "/steering", "/speed", "/cmd_vel",
    "/odom", "/clock", "/tf", "/tf_static",
)
EPISODE_ORDER = tuple(
    f"train_s{scenario}_r{repeat[1:]}"
    for repeat in REPEAT_IDS
    for scenario in TRAIN_SCENARIOS
)
MINIMUM_COLLECTION_FREE_BYTES = 8 * 1024**3
MINIMUM_PROJECTED_FREE_BYTES = 6 * 1024**3
MAXIMUM_ADJACENT_GAP_S = 0.120

FRAME_MANIFEST_COLUMNS = [
    *MANIFEST_COLUMNS,
    "scenario_id", "scenario_role", "repeat_id", "cone_scenario_id",
    "route_s_m", "route_s_record_time_ns", "image_sha256",
]
TEMPORAL_MANIFEST_COLUMNS = [
    "sequence_id", "episode_id", "scenario_id", "scenario_role", "repeat_id",
    "cone_scenario_id",
    "frame_t_minus_2", "frame_t_minus_1", "frame_t",
    "frame_t_minus_2_sha256", "frame_t_minus_1_sha256", "frame_t_sha256",
    "camera_timestamp_t_minus_2_ns", "camera_timestamp_t_minus_1_ns",
    "camera_timestamp_t_ns", "adjacent_gap_1_s", "adjacent_gap_2_s",
    "oldest_to_current_span_s", "steering_target_timestamp_ns",
    "steering_label_age_ms", "target_steering_rad", "speed_record_time_ns",
    "speed_age_ms", "speed_mps", "route_progress_m", "source_mcap_sha256",
    "source_manifest_sha256",
]


class TrainDataGateError(RuntimeError):
    """A hard collection, extraction, or evidence-gate failure."""


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    scenario_id: str
    repeat_id: str
    role: str = "TRAIN"


@dataclass(frozen=True)
class TaskConfig:
    path: Path
    payload: dict[str, Any]

    @property
    def collection(self) -> dict[str, Any]:
        return self.payload["collection"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.payload["dataset"]

    @property
    def frozen(self) -> dict[str, Any]:
        return self.payload["frozen_expert"]

    @property
    def sha256(self) -> str:
        return sha256_file(self.path)

    def result_dir(self, repo: Path, name: str) -> Path:
        return repo / self.payload["result_directories"][name]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def episode_specs() -> tuple[EpisodeSpec, ...]:
    return tuple(
        EpisodeSpec(
            episode_id=f"train_s{scenario}_r{repeat[1:]}",
            scenario_id=scenario,
            repeat_id=repeat,
        )
        for repeat in REPEAT_IDS
        for scenario in TRAIN_SCENARIOS
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainDataGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainDataGateError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write compact evidence atomically so a host crash cannot finalize half JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    temporary.replace(path)


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def load_task_config(path: Path, repo: Path) -> TaskConfig:
    payload = _load_json(path)
    required = {
        "version", "map_family", "frozen_expert", "scenario_roles", "collection",
        "dataset", "result_directories", "permissions",
    }
    if set(payload) != required or payload["version"] != VERSION or payload["map_family"] != MAP_FAMILY:
        raise TrainDataGateError("random-cone TRAIN task version/map/fields changed")
    if payload["scenario_roles"] != ROLE_IDS:
        raise TrainDataGateError("frozen 8/2/2 scenario roles changed")
    frozen = payload["frozen_expert"]
    expected_frozen_keys = {
        "config_path", "config_sha256", "result_directory", "result_manifest_sha256",
        "summary_sha256", "offline_geometry_sha256",
    }
    if not isinstance(frozen, dict) or set(frozen) != expected_frozen_keys:
        raise TrainDataGateError("frozen Expert identity fields changed")
    for relative, expected, label in (
        (frozen["config_path"], frozen["config_sha256"], "frozen Expert config"),
        (f"{frozen['result_directory']}/summary.json", frozen["summary_sha256"], "frozen Expert summary"),
        (f"{frozen['result_directory']}/offline_geometry.json", frozen["offline_geometry_sha256"], "frozen offline geometry"),
    ):
        observed = sha256_file(repo / relative)
        if observed != expected:
            raise TrainDataGateError(f"{label} changed: {observed} != {expected}")
    if directory_file_manifest_sha256(repo / frozen["result_directory"]) != frozen["result_manifest_sha256"]:
        raise TrainDataGateError("frozen 1.0 m/s Expert result tree changed")

    collection = payload["collection"]
    expected_collection_keys = {
        "repeat_ids", "episode_order", "required_topics", "container_name", "compose_service",
        "container_userdata_root", "data_relative_root", "storage_id",
        "recorder_startup_timeout_s", "recorder_shutdown_timeout_s", "settle_duration_s",
        "minimum_camera_messages", "minimum_free_bytes_before_collection",
        "minimum_projected_free_bytes", "infrastructure_replacement_attempts_per_episode",
        "retry_genuine_policy_failure",
    }
    if not isinstance(collection, dict) or set(collection) != expected_collection_keys:
        raise TrainDataGateError("collection configuration fields changed")
    frozen_collection = (
        tuple(collection["repeat_ids"]), tuple(collection["episode_order"]),
        tuple(collection["required_topics"]), collection["data_relative_root"],
        collection["storage_id"], collection["minimum_free_bytes_before_collection"],
        collection["minimum_projected_free_bytes"],
        collection["infrastructure_replacement_attempts_per_episode"],
        collection["retry_genuine_policy_failure"],
    )
    if frozen_collection != (
        REPEAT_IDS, EPISODE_ORDER, REQUIRED_TOPICS,
        "physicar_e2e/random_cone_1p0_v1/train_raw", "mcap",
        MINIMUM_COLLECTION_FREE_BYTES, MINIMUM_PROJECTED_FREE_BYTES, 1, False,
    ):
        raise TrainDataGateError(f"TRAIN collection contract changed: {frozen_collection}")
    if any(re.search(r"s(?:09|10|11|12)", episode) for episode in collection["episode_order"]):
        raise TrainDataGateError("VALIDATION/HOLDOUT episode leaked into collection order")

    dataset = payload["dataset"]
    expected_dataset_keys = {
        "canonical_extractor_config_path", "canonical_extractor_config_sha256",
        "data_relative_root", "history_frames", "maximum_adjacent_gap_s", "causal_only",
        "allow_episode_boundary_crossing", "allow_reset_boundary_crossing",
        "allow_duplicate_padding", "route_region_margin_m",
    }
    if not isinstance(dataset, dict) or set(dataset) != expected_dataset_keys:
        raise TrainDataGateError("dataset configuration fields changed")
    extractor_path = repo / dataset["canonical_extractor_config_path"]
    if sha256_file(extractor_path) != dataset["canonical_extractor_config_sha256"]:
        raise TrainDataGateError("canonical Dataset Extractor V1 config changed")
    temporal = (
        dataset["history_frames"], dataset["maximum_adjacent_gap_s"], dataset["causal_only"],
        dataset["allow_episode_boundary_crossing"], dataset["allow_reset_boundary_crossing"],
        dataset["allow_duplicate_padding"], dataset["data_relative_root"],
    )
    if temporal != (
        3, MAXIMUM_ADJACENT_GAP_S, True, False, False, False,
        "physicar_e2e/random_cone_1p0_v1/train_dataset",
    ):
        raise TrainDataGateError(f"temporal TRAIN contract changed: {temporal}")
    if payload["result_directories"] != {
        "collection": "results/random_cone_train_collection_1p0_v1",
        "dataset": "results/random_cone_train_dataset_1p0_v1",
    }:
        raise TrainDataGateError("compact result directories changed")
    expected_permissions = {
        "validation_bag_collection_permitted": False,
        "holdout_bag_collection_permitted": False,
        "neural_training_permitted": False,
        "onnx_export_permitted": False,
        "neural_closed_loop_permitted": False,
        "frozen_expert_evidence_changes_permitted": False,
        "tracked_simulator_source_changes_permitted": False,
        "commit_permitted": False,
        "push_permitted": False,
    }
    if payload["permissions"] != expected_permissions:
        raise TrainDataGateError("forbidden-action permissions changed")
    return TaskConfig(path.resolve(), payload)


def collector_config(task: TaskConfig) -> CollectorConfig:
    raw = task.collection
    config = CollectorConfig(
        expected_world="frozen-random-cone-scenario-specific-world",
        required_topics=REQUIRED_TOPICS,
        container_name=raw["container_name"], compose_service=raw["compose_service"],
        container_userdata_root=raw["container_userdata_root"],
        data_relative_root=raw["data_relative_root"], storage_id=raw["storage_id"],
        recorder_startup_timeout_s=raw["recorder_startup_timeout_s"],
        recorder_shutdown_timeout_s=raw["recorder_shutdown_timeout_s"],
        settle_duration_s=raw["settle_duration_s"], pilot_episode_count=len(EPISODE_ORDER),
        minimum_free_bytes=raw["minimum_projected_free_bytes"],
        minimum_camera_messages=raw["minimum_camera_messages"],
    )
    config.validate()
    return config


def disk_state(path: Path | str = "/") -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path), "total_bytes": usage.total, "used_bytes": usage.used,
        "available_bytes": usage.free, "available_gib": usage.free / 1024**3,
    }


def audit_frozen_expert(
    repo: Path, sim_root: Path, task: TaskConfig,
) -> tuple[RandomConeConfig, tuple[ScenarioBundle, ...], dict[str, Any]]:
    expert_path = repo / task.frozen["config_path"]
    expert = RandomConeConfig.load(expert_path, repo, sim_root)
    bundles = tuple(verify_frozen_scenarios(expert, sim_root))
    offline_path = repo / task.frozen["result_directory"] / "offline_geometry.json"
    offline = verify_offline_evidence(offline_path, expert, bundles)
    summary = _load_json(repo / task.frozen["result_directory"] / "summary.json")
    scenario_rows = summary.get("scenarios")
    if (
        summary.get("version") != "random_cone_expert_1p0_v1"
        or summary.get("result") != "PASS"
        or summary.get("aggregate", {}).get("success") != "12/12"
        or summary.get("aggregate", {}).get("valid_policy_runs") != 12
        or summary.get("aggregate", {}).get("cone_contact_or_intersection_count") != 0
        or summary.get("random_cone_expert_frozen") is not True
        or summary.get("exact_8_2_2_split_frozen") is not True
        or summary.get("random_cone_bag_collection_justified") is not True
        or not isinstance(scenario_rows, list) or len(scenario_rows) != 12
    ):
        raise TrainDataGateError("frozen 1.0 m/s Expert evidence is not official 12/12 PASS")
    for observed, bundle in zip(scenario_rows, bundles, strict=True):
        if observed.get("scenario") != bundle.scenario.to_dict():
            raise TrainDataGateError(f"frozen scenario {bundle.scenario.scenario_id} identity changed")
        valid = observed.get("valid_policy_run") or {}
        metrics = valid.get("metrics") or {}
        if (
            observed.get("result") != "RANDOM_CONE_EXPERT_PASS"
            or observed.get("valid_policy_run_count") != 1
            or metrics.get("cone_contact_or_intersection_occurred") is not False
            or metrics.get("recovery_success") is not True
            or not float(metrics.get("minimum_footprint_to_cone_clearance_m", 0.0)) > 0.0
        ):
            raise TrainDataGateError(f"frozen scenario {bundle.scenario.scenario_id} PASS evidence changed")
    if expert.random_seed != 20260825 or tuple(item.scenario.scenario_id for item in bundles[:8]) != TRAIN_SCENARIOS:
        raise TrainDataGateError("frozen seed or TRAIN identities changed")
    control = {
        "speed_mps": expert.baseline.fixed_speed_mps,
        "lookahead_m": expert.baseline.lookahead_m,
        "control_frequency_hz": expert.baseline.control_frequency_hz,
        "steering_limit_rad": expert.baseline.max_steering_rad,
        "wheelbase_m": expert.baseline.wheelbase_m,
    }
    if tuple(control.values()) != (1.0, 0.90, 15.0, 0.349066, 0.18):
        raise TrainDataGateError(f"frozen 1.0 m/s control changed: {control}")
    preserved = audit_preserved_state(repo, sim_root, expert)
    simulator = simulator_tracked_status(sim_root)
    if simulator.get("result") != "PASS":
        raise TrainDataGateError("simulator has tracked source changes before collection")
    audit = {
        "version": VERSION, "result": "PASS", "generated_utc": utc_now(),
        "task_config": {"path": str(task.path.relative_to(repo)), "sha256": task.sha256},
        "frozen_expert": {
            **task.frozen, "random_seed": expert.random_seed, "fixed_control": control,
            "result": summary["result"], "success": summary["aggregate"]["success"],
            "minimum_actual_clearance_m": summary["aggregate"]["minimum_actual_clearance_m"],
        },
        "train_scenarios": [bundle.scenario.to_dict() for bundle in bundles[:8]],
        "collection_order": list(EPISODE_ORDER),
        "offline_geometry": {"result": offline["result"], "sha256": sha256_file(offline_path)},
        "preserved_state": preserved, "simulator_tracked_source_status": simulator,
    }
    return expert, bundles, audit


def _scenario_hash(bundle: ScenarioBundle) -> str:
    return _canonical_hash(bundle.scenario.to_dict())


def _topic_metrics(info: BagInfo) -> dict[str, dict[str, float | int]]:
    return {
        topic: {"message_count": count, "average_recorded_rate_hz": count / info.duration_s}
        for topic, count in sorted(info.topic_counts.items())
    }


def _handle_for(backend: DockerRosBackend, episode_id: str) -> RecorderHandle:
    host_episode = backend.host_data_root / episode_id
    container_episode = str(PurePosixPath(backend.container_data_root) / episode_id)
    return RecorderHandle(
        episode_id=episode_id,
        host_episode_path=host_episode,
        host_bag_path=host_episode / "bag",
        container_episode_path=container_episode,
        container_bag_path=str(PurePosixPath(container_episode) / "bag"),
        container_pid_path=str(PurePosixPath(container_episode) / ".rosbag_pid"),
        container_log_path=str(PurePosixPath(container_episode) / "recorder.log"),
    )


def _post_settle_preflight(
    client: SimClient, expert: RandomConeConfig, bundle: ScenarioBundle,
    sim_root: Path, settle_duration_s: float,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Activate, reset, settle, then complete the full scenario preflight."""
    activation = activate_world(client, expert.world_name(bundle.scenario.scenario_id))
    initial = wait_after_scenario_reset(client, expert, bundle)
    time.sleep(settle_duration_s)
    environment = verify_scenario_environment(expert, bundle, sim_root)
    initial = scenario_structural_preflight(client, expert, bundle)
    clock = clock_health_preflight(client)
    if clock.get("result") != "PASS":
        raise TrainDataGateError(str(clock.get("failure_reason", "clock health failed")))
    started = time.monotonic()
    with Image.open(BytesIO(client.camera_jpeg())) as image:
        image.load()
        dimensions, mode = list(image.size), image.mode
    if dimensions != [480, 360] or mode != "RGB":
        raise TrainDataGateError(f"camera preflight differs from 480x360 RGB: {dimensions} {mode}")
    driver = expert.driver_for(bundle.scenario)
    preflight = {
        "result": "PASS", "scenario_id": bundle.scenario.scenario_id,
        "role": bundle.scenario.role, "world": initial.world, "environment": environment,
        "offline_geometry_gate": bundle.geometry.get("result"), "route_points": initial.route_points,
        "route_length_m": initial.route.length, "cone_count": initial.cone_count,
        "pose": initial.pose, "bounds": initial.bounds, "clock_health": clock,
        "camera": {"result": "PASS", "dimensions": dimensions, "mode": mode,
                   "acquisition_ms": (time.monotonic() - started) * 1000.0},
        "control_api": "PASS", "reset_before_recording": True,
        "settled_before_full_preflight_s": settle_duration_s,
        "fixed_control": {
            "speed_mps": driver.fixed_speed_mps, "lookahead_m": driver.lookahead_m,
            "control_frequency_hz": driver.control_frequency_hz,
            "steering_limit_rad": driver.max_steering_rad, "wheelbase_m": driver.wheelbase_m,
        },
    }
    return initial, activation, preflight


def _base_episode_metadata(
    spec: EpisodeSpec, task: TaskConfig, expert: RandomConeConfig,
    bundle: ScenarioBundle, repo: Path, attempt_number: int,
) -> dict[str, Any]:
    return {
        "version": COLLECTION_VERSION + "_episode",
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "scenario_role": spec.role,
        "collection_order_index": EPISODE_ORDER.index(spec.episode_id),
        "attempt_number": attempt_number, "infrastructure_replacement": attempt_number > 1,
        "result": "FAIL", "classification": "INFRA_FAIL", "failure_reason": None,
        "world": expert.world_name(spec.scenario_id),
        "frozen_scenario": bundle.scenario.to_dict(),
        "frozen_scenario_sha256": _scenario_hash(bundle),
        "planned_bypass": bundle.geometry,
        "task_config_sha256": task.sha256,
        "frozen_expert_config_sha256": task.frozen["config_sha256"],
        "frozen_expert_result_manifest_sha256": task.frozen["result_manifest_sha256"],
        "physicar_e2e_git_commit": git_commit(repo),
        "required_topics": list(REQUIRED_TOPICS),
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


def collect_one_episode(
    spec: EpisodeSpec, *, task: TaskConfig, repo: Path, sim_root: Path,
    expert: RandomConeConfig, bundle: ScenarioBundle, backend: DockerRosBackend,
    client: SimClient, attempt_number: int, result_path: Path,
    prepare: Callable[..., tuple[Any, dict[str, Any], dict[str, Any]]] = _post_settle_preflight,
    run_expert: Callable[..., dict[str, Any]] = run_random_cone_expert,
) -> dict[str, Any]:
    """Run one independent bag lifecycle; never retries or changes parameters."""
    metadata = _base_episode_metadata(spec, task, expert, bundle, repo, attempt_number)
    handle: RecorderHandle | None = None
    stop_result = None
    run_exception: BaseException | None = None
    try:
        if errors := client.safe_stop():
            raise TrainDataGateError("initial safe stop failed: " + "; ".join(errors))
        initial, activation, preflight = prepare(
            client, expert, bundle, sim_root, float(task.collection["settle_duration_s"])
        )
        metadata["world_activation"] = activation
        metadata["preflight"] = preflight
        if disk_state("/")["available_bytes"] < task.collection["minimum_projected_free_bytes"]:
            raise TrainDataGateError("root disk fell below 6 GiB before episode recording")
        handle = backend.start_recorder(spec.episode_id, REQUIRED_TOPICS)
        metadata.update({
            "bag_host_path": str(handle.host_bag_path),
            "bag_container_path": handle.container_bag_path,
            "recording_start_utc": utc_now(), "expert_driving_start_utc": utc_now(),
        })
        metrics = run_expert(client, expert, initial, bundle)
        metadata["expert_driving_end_utc"] = utc_now()
        metadata["expert_result_metrics"] = metrics
        metadata["expert_classification"] = metrics.get("classification")
        if metrics.get("classification") == "RANDOM_CONE_EXPERT_FAIL":
            metadata["failure_reason"] = "frozen Expert produced a genuine policy failure"
        elif metrics.get("classification") != "RANDOM_CONE_EXPERT_PASS":
            metadata["failure_reason"] = f"frozen Expert classified {metrics.get('classification')}"
    except BaseException as exc:
        run_exception = exc
        metadata["failure_reason"] = metadata["failure_reason"] or f"{type(exc).__name__}: {exc}"
    finally:
        # The vehicle is stopped before SIGINT so the bag has a bounded stationary suffix.
        stop_errors = client.safe_stop()
        metadata["post_run_safe_stop_success"] = not stop_errors
        metadata["post_run_safe_stop_errors"] = stop_errors
        if stop_errors:
            metadata["failure_reason"] = metadata["failure_reason"] or (
                "post-run safe stop failed: " + "; ".join(stop_errors)
            )
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
                if not stop_result.graceful:
                    metadata["failure_reason"] = metadata["failure_reason"] or (
                        stop_result.detail or "recorder did not finalize gracefully"
                    )
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
            metadata["failure_reason"] = metadata["failure_reason"] or (
                "final safe stop failed: " + "; ".join(final_errors)
            )
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            info = backend.bag_info(handle)
            verify_bag(info, REQUIRED_TOPICS, int(task.collection["minimum_camera_messages"]))
            if set(info.topic_counts) != set(REQUIRED_TOPICS):
                raise TrainDataGateError("bag topic set is not exactly the canonical eight topics")
            mcap_files = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcap_files) != 1:
                raise TrainDataGateError(f"expected exactly one finalized MCAP, found {len(mcap_files)}")
            metadata.update({
                "bag_mcap_path": str(mcap_files[0]),
                "bag_mcap_sha256": sha256_file(mcap_files[0]),
                "bag_size_bytes": directory_size(handle.host_bag_path),
                "bag_duration_s": info.duration_s,
                "actual_topic_message_counts": dict(sorted(info.topic_counts.items())),
                "topic_metrics": _topic_metrics(info),
            })
        except BaseException as exc:
            metadata["failure_reason"] = metadata["failure_reason"] or f"bag integrity failed: {exc}"
    metrics = metadata.get("expert_result_metrics") or {}
    genuine_failure = metadata.get("expert_classification") == "RANDOM_CONE_EXPERT_FAIL"
    success = (
        metadata["failure_reason"] is None
        and metadata.get("expert_classification") == "RANDOM_CONE_EXPERT_PASS"
        and metrics.get("result") == "PASS"
        and float(metrics.get("minimum_footprint_to_cone_clearance_m", -1.0)) > 0.0
        and metrics.get("cone_contact_or_intersection_occurred") is False
        and metrics.get("recovery_success") is True
        and metrics.get("api_failures") == 0 and metrics.get("pose_failures") == 0
        and metrics.get("clock_failures") == 0
        and metrics.get("safe_stop_success") is True
        and metadata["post_run_safe_stop_success"] and metadata["final_safe_stop_success"]
        and metadata["recorder_graceful_shutdown"] and not metadata["recorder_orphaned"]
        and metadata["orphan_process_check_pass"]
        and metadata["bag_size_bytes"] is not None and metadata["bag_mcap_sha256"] is not None
    )
    if success:
        metadata["result"] = "PASS"
        metadata["classification"] = "TRAIN_EPISODE_PASS"
    elif genuine_failure:
        metadata["classification"] = "GENUINE_EXPERT_FAIL"
    else:
        metadata["classification"] = "INFRA_FAIL"
        if run_exception is not None:
            metadata["infrastructure_failures"].append(
                f"{type(run_exception).__name__}: {run_exception}"
            )
        if metadata["failure_reason"] is None:
            metadata["failure_reason"] = "episode did not satisfy every collection gate"
    write_json(result_path, metadata)
    return metadata


def validate_finalized_metadata(
    metadata: dict[str, Any], spec: EpisodeSpec, task: TaskConfig,
    expert: RandomConeConfig, bundle: ScenarioBundle,
) -> list[str]:
    """Pure compact-evidence validation used by the crash-safe resume path."""
    metrics = metadata.get("expert_result_metrics") or {}
    errors: list[str] = []
    expected = {
        "version": COLLECTION_VERSION + "_episode", "episode_id": spec.episode_id,
        "scenario_id": spec.scenario_id, "repeat_id": spec.repeat_id,
        "scenario_role": "TRAIN", "collection_order_index": EPISODE_ORDER.index(spec.episode_id),
        "world": expert.world_name(spec.scenario_id), "frozen_scenario_sha256": _scenario_hash(bundle),
        "task_config_sha256": task.sha256,
        "frozen_expert_config_sha256": task.frozen["config_sha256"],
        "frozen_expert_result_manifest_sha256": task.frozen["result_manifest_sha256"],
        "result": "PASS", "classification": "TRAIN_EPISODE_PASS",
        "expert_classification": "RANDOM_CONE_EXPERT_PASS",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            errors.append(f"{key} mismatch")
    if metadata.get("frozen_scenario") != bundle.scenario.to_dict():
        errors.append("frozen scenario mismatch")
    if tuple(metadata.get("required_topics") or ()) != REQUIRED_TOPICS:
        errors.append("required topics mismatch")
    counts = metadata.get("actual_topic_message_counts") or {}
    if set(counts) != set(REQUIRED_TOPICS) or any(int(counts.get(topic, 0)) <= 0 for topic in REQUIRED_TOPICS):
        errors.append("actual topic counts are incomplete")
    if int(counts.get("/camera/image_raw", 0)) < int(task.collection["minimum_camera_messages"]):
        errors.append("camera count below minimum")
    if not float(metadata.get("bag_duration_s") or 0.0) > 0.0:
        errors.append("bag duration is not positive")
    if not int(metadata.get("bag_size_bytes") or 0) > 0:
        errors.append("bag size is not positive")
    if not isinstance(metadata.get("bag_mcap_sha256"), str) or len(metadata["bag_mcap_sha256"]) != 64:
        errors.append("MCAP SHA-256 missing")
    for flag in (
        "recorder_graceful_shutdown", "orphan_process_check_pass",
        "post_run_safe_stop_success", "final_safe_stop_success",
    ):
        if metadata.get(flag) is not True:
            errors.append(f"{flag} is not true")
    if metadata.get("recorder_orphaned") is not False:
        errors.append("recorder orphaned")
    if (
        metrics.get("result") != "PASS"
        or not float(metrics.get("minimum_footprint_to_cone_clearance_m", -1.0)) > 0.0
        or metrics.get("cone_contact_or_intersection_occurred") is not False
        or metrics.get("recovery_success") is not True
        or metrics.get("api_failures") != 0 or metrics.get("pose_failures") != 0
        or metrics.get("clock_failures") != 0 or metrics.get("safe_stop_success") is not True
    ):
        errors.append("Expert practical success metrics failed")
    return errors


def validate_existing_episode(
    spec: EpisodeSpec, *, result_path: Path, task: TaskConfig,
    expert: RandomConeConfig, bundle: ScenarioBundle, backend: DockerRosBackend,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    """Return VALID, MISSING, PARTIAL, or GENUINE_FAILURE without mutating artifacts."""
    root_exists = (backend.host_data_root / spec.episode_id).exists()
    if not result_path.is_file():
        return ("PARTIAL" if root_exists else "MISSING"), None, ["final compact evidence missing"] if root_exists else []
    try:
        metadata = _load_json(result_path)
    except TrainDataGateError as exc:
        return "PARTIAL", None, [str(exc)]
    if metadata.get("classification") == "GENUINE_EXPERT_FAIL":
        return "GENUINE_FAILURE", metadata, [str(metadata.get("failure_reason"))]
    errors = validate_finalized_metadata(metadata, spec, task, expert, bundle)
    if not root_exists:
        errors.append("external episode directory missing")
    if not errors:
        try:
            handle = _handle_for(backend, spec.episode_id)
            info = backend.bag_info(handle)
            verify_bag(info, REQUIRED_TOPICS, int(task.collection["minimum_camera_messages"]))
            if set(info.topic_counts) != set(REQUIRED_TOPICS):
                errors.append("live MCAP topic set differs from canonical eight")
            if dict(sorted(info.topic_counts.items())) != metadata["actual_topic_message_counts"]:
                errors.append("live MCAP counts differ from compact evidence")
            mcap_files = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcap_files) != 1:
                errors.append(f"expected one MCAP, found {len(mcap_files)}")
            elif sha256_file(mcap_files[0]) != metadata["bag_mcap_sha256"]:
                errors.append("MCAP SHA-256 differs from compact evidence")
            if directory_size(handle.host_bag_path) != metadata["bag_size_bytes"]:
                errors.append("bag directory size differs from compact evidence")
        except BaseException as exc:
            errors.append(f"MCAP integrity recheck failed: {exc}")
    return ("VALID" if not errors else "PARTIAL"), metadata, errors


def retry_decision(classification: str, attempt_number: int, maximum_attempts: int = 2) -> str:
    if classification == "TRAIN_EPISODE_PASS":
        return "CONTINUE"
    if classification == "GENUINE_EXPERT_FAIL":
        return "STOP_GENUINE_FAILURE"
    if classification == "INFRA_FAIL" and attempt_number < maximum_attempts:
        return "RETRY_INFRA"
    return "STOP_INFRA"


def _attempt_count(result_dir: Path, episode_id: str) -> int:
    count = len(list((result_dir / "attempts").glob(f"{episode_id}_attempt_*.json")))
    state_path = result_dir / "states" / f"{episode_id}.json"
    if state_path.is_file():
        try:
            count = max(count, int(_load_json(state_path).get("attempt_number", 0)))
        except (TrainDataGateError, TypeError, ValueError):
            pass
    return count


def _recover_final_from_valid_attempt(
    spec: EpisodeSpec, *, result_dir: Path, final_path: Path, task: TaskConfig,
    expert: RandomConeConfig, bundle: ScenarioBundle, backend: DockerRosBackend,
) -> bool:
    """Promote a finalized attempt after a crash between attempt and final writes."""
    attempts = sorted((result_dir / "attempts").glob(f"{spec.episode_id}_attempt_*.json"))
    if final_path.exists() or not attempts:
        return False
    status, metadata, _ = validate_existing_episode(
        spec, result_path=attempts[-1], task=task, expert=expert,
        bundle=bundle, backend=backend,
    )
    if status != "VALID" or metadata is None:
        return False
    write_json(final_path, metadata)
    write_json(result_dir / "states" / f"{spec.episode_id}.json", {
        "status": "FINALIZED_VALID_RECOVERED_AFTER_RESTART",
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "attempt_number": metadata["attempt_number"],
        "recovered_utc": utc_now(), "bag_mcap_sha256": metadata["bag_mcap_sha256"],
    })
    return True


def _archive_partial_episode(
    backend: DockerRosBackend, spec: EpisodeSpec, attempt_number: int,
) -> str | None:
    source = backend.host_data_root / spec.episode_id
    if not source.exists():
        return None
    handle = _handle_for(backend, spec.episode_id)
    try:
        if backend._alive(handle):
            backend.stop_recorder(handle)
    except BaseException:
        pass
    destination = backend.host_data_root / "_interrupted" / spec.episode_id / f"attempt_{attempt_number:02d}"
    if destination.exists():
        destination = destination.with_name(destination.name + "_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return str(destination)


def _raw_root_has_forbidden_episode(backend: DockerRosBackend) -> list[str]:
    if not backend.host_data_root.exists():
        return []
    allowed = set(EPISODE_ORDER) | {"_interrupted"}
    return sorted(
        item.name for item in backend.host_data_root.iterdir()
        if item.is_dir() and item.name not in allowed
    )


def validate_collection_gate(
    episodes: Sequence[dict[str, Any]], intended: Sequence[EpisodeSpec] = episode_specs(),
) -> dict[str, Any]:
    ids = [item.get("episode_id") for item in episodes]
    intended_ids = [spec.episode_id for spec in intended]
    role_ids = {(item.get("scenario_id"), item.get("repeat_id"), item.get("scenario_role")) for item in episodes}
    expected_roles = {(spec.scenario_id, spec.repeat_id, "TRAIN") for spec in intended}
    gates = {
        "sixteen_intended_episodes": ids == intended_ids and len(ids) == 16,
        "no_duplicate_valid_ids": len(ids) == len(set(ids)),
        "exact_scenario_repeat_roles": role_ids == expected_roles,
        "all_episode_results_pass": all(item.get("result") == "PASS" for item in episodes),
        "no_validation_or_holdout": all(item.get("scenario_id") in TRAIN_SCENARIOS for item in episodes),
        "no_genuine_expert_failure": all(item.get("classification") != "GENUINE_EXPERT_FAIL" for item in episodes),
        "no_parameter_change": all(
            (item.get("preflight") or {}).get("fixed_control") == {
                "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
                "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
            }
            for item in episodes
        ),
        "all_safe_stops_pass": all(
            item.get("post_run_safe_stop_success") is True
            and item.get("final_safe_stop_success") is True
            and (item.get("expert_result_metrics") or {}).get("safe_stop_success") is True
            for item in episodes
        ),
    }
    return {"result": "PASS" if all(gates.values()) else "FAIL", "gates": gates}


def _collection_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Random-Cone 1.0 m/s TRAIN Bag Collection V1", "",
        f"Result: **{report.get('result', 'FAIL')}**", "",
        "| Episode | Scenario | Repeat | Result | Clearance (m) | Bag bytes | Camera | Steering |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for item in report.get("episodes", []):
        metrics = item.get("expert_result_metrics") or {}
        counts = item.get("actual_topic_message_counts") or {}
        lines.append(
            f"| {item['episode_id']} | {item['scenario_id']} | {item['repeat_id']} | "
            f"{item['result']} | {metrics.get('minimum_footprint_to_cone_clearance_m')} | "
            f"{item.get('bag_size_bytes')} | {counts.get('/camera/image_raw')} | {counts.get('/steering')} |"
        )
    lines.extend([
        "", f"Total raw bytes: {report.get('total_raw_storage_bytes')}",
        f"Genuine Expert failures: {report.get('genuine_expert_failure_count')}",
        f"Infrastructure replacement attempts: {report.get('infrastructure_replacement_attempt_count')}",
        "", "No neural training, model export, validation collection, or holdout collection was performed.", "",
    ])
    return "\n".join(lines)


def _write_collection_progress(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)
    report_path = path.parent / "REPORT.md"
    report_path.write_text(_collection_report_markdown(report), encoding="utf-8")


def collection_stage(repo: Path, sim_root: Path, task: TaskConfig) -> dict[str, Any]:
    expert, all_bundles, audit = audit_frozen_expert(repo, sim_root, task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles[:8]}
    result_dir = task.result_dir(repo, "collection")
    result_dir.mkdir(parents=True, exist_ok=True)
    write_json(result_dir / "frozen_audit.json", audit)
    pre_disk = disk_state("/")
    if pre_disk["available_bytes"] < MINIMUM_COLLECTION_FREE_BYTES:
        raise TrainDataGateError("root disk has less than 8 GiB available before collection")
    backend = DockerRosBackend(collector_config(task), sim_root)
    forbidden = _raw_root_has_forbidden_episode(backend)
    if forbidden:
        raise TrainDataGateError(f"unexpected/forbidden raw episode directories: {forbidden}")
    topic_types = backend.preflight(REQUIRED_TOPICS)
    client = SimClient(expert.baseline.base_url, expert.baseline.api_timeout_s)
    original_world = str(client.status().get("current") or "") or None
    report: dict[str, Any] = {
        "version": COLLECTION_VERSION, "generated_utc": utc_now(), "result": "FAIL",
        "task_config_sha256": task.sha256, "frozen_expert_audit": "PASS",
        "frozen_expert_config_sha256": task.frozen["config_sha256"],
        "frozen_expert_result_manifest_sha256": task.frozen["result_manifest_sha256"],
        "random_seed": expert.random_seed, "map_family": MAP_FAMILY,
        "train_scenario_ids": list(TRAIN_SCENARIOS), "repeat_ids": list(REPEAT_IDS),
        "intended_episode_order": list(EPISODE_ORDER), "required_topics": list(REQUIRED_TOPICS),
        "topic_types": topic_types, "raw_root": str(backend.host_data_root),
        "disk_pre_collection": pre_disk, "resumed_skipped_episode_ids": [],
        "interrupted_archives": [], "episodes": [], "storage_projection": None,
        "failure_reason": None, "collection_gate": None,
    }
    summary_path = result_dir / "summary.json"
    maximum_attempts = 1 + int(task.collection["infrastructure_replacement_attempts_per_episode"])
    projection_recorded = False
    try:
        for spec in episode_specs():
            bundle = bundles[spec.scenario_id]
            final_path = result_dir / "episodes" / f"{spec.episode_id}.json"
            recovered_after_restart = _recover_final_from_valid_attempt(
                spec, result_dir=result_dir, final_path=final_path, task=task,
                expert=expert, bundle=bundle, backend=backend,
            )
            status, existing, errors = validate_existing_episode(
                spec, result_path=final_path, task=task, expert=expert,
                bundle=bundle, backend=backend,
            )
            if status == "GENUINE_FAILURE":
                report["failure_reason"] = f"preserved genuine Expert failure in {spec.episode_id}"
                break
            if status == "VALID":
                assert existing is not None
                report["episodes"].append(existing)
                report["resumed_skipped_episode_ids"].append(spec.episode_id)
                if recovered_after_restart:
                    report.setdefault("recovered_finalized_attempt_ids", []).append(spec.episode_id)
            else:
                started_attempts = _attempt_count(result_dir, spec.episode_id)
                if status == "PARTIAL" and started_attempts == 0:
                    started_attempts = 1
                    write_json(result_dir / "attempts" / f"{spec.episode_id}_attempt_01.json", {
                        "version": COLLECTION_VERSION + "_interruption",
                        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
                        "repeat_id": spec.repeat_id, "classification": "INFRA_FAIL",
                        "result": "FAIL", "failure_reason": "; ".join(errors) or "unfinalized bag",
                        "reconstructed_after_restart_utc": utc_now(),
                    })
                if started_attempts >= maximum_attempts:
                    report["failure_reason"] = (
                        f"{spec.episode_id} exhausted its one infrastructure replacement"
                    )
                    break
                if status == "PARTIAL":
                    archive = _archive_partial_episode(backend, spec, started_attempts)
                    if archive:
                        report["interrupted_archives"].append({
                            "episode_id": spec.episode_id, "attempt_number": started_attempts,
                            "archive_path": archive, "diagnostics": errors,
                        })
                attempt_number = started_attempts + 1
                while attempt_number <= maximum_attempts:
                    write_json(result_dir / "states" / f"{spec.episode_id}.json", {
                        "status": "STARTED_UNFINALIZED", "episode_id": spec.episode_id,
                        "scenario_id": spec.scenario_id, "repeat_id": spec.repeat_id,
                        "attempt_number": attempt_number, "started_utc": utc_now(),
                        "task_config_sha256": task.sha256,
                    })
                    attempt_path = result_dir / "attempts" / (
                        f"{spec.episode_id}_attempt_{attempt_number:02d}.json"
                    )
                    episode = collect_one_episode(
                        spec, task=task, repo=repo, sim_root=sim_root, expert=expert,
                        bundle=bundle, backend=backend, client=client,
                        attempt_number=attempt_number, result_path=attempt_path,
                    )
                    print(json.dumps({
                        "stage": "collect", "episode_id": spec.episode_id,
                        "attempt": attempt_number, "result": episode["result"],
                        "classification": episode["classification"],
                        "clearance_m": (episode.get("expert_result_metrics") or {}).get(
                            "minimum_footprint_to_cone_clearance_m"
                        ), "bag_size_bytes": episode.get("bag_size_bytes"),
                    }), flush=True)
                    decision = retry_decision(episode["classification"], attempt_number, maximum_attempts)
                    if decision == "CONTINUE":
                        write_json(final_path, episode)
                        write_json(result_dir / "states" / f"{spec.episode_id}.json", {
                            "status": "FINALIZED_VALID", "episode_id": spec.episode_id,
                            "scenario_id": spec.scenario_id, "repeat_id": spec.repeat_id,
                            "attempt_number": attempt_number, "finalized_utc": utc_now(),
                            "bag_mcap_sha256": episode["bag_mcap_sha256"],
                        })
                        report["episodes"].append(episode)
                        break
                    if decision == "STOP_GENUINE_FAILURE":
                        write_json(final_path, episode)
                        write_json(result_dir / "states" / f"{spec.episode_id}.json", {
                            "status": "GENUINE_POLICY_FAILURE_DO_NOT_RETRY",
                            "episode_id": spec.episode_id, "attempt_number": attempt_number,
                            "finalized_utc": utc_now(),
                        })
                        report["episodes"].append(episode)
                        report["failure_reason"] = f"genuine Expert failure in {spec.episode_id}"
                        break
                    if decision == "RETRY_INFRA":
                        archive = _archive_partial_episode(backend, spec, attempt_number)
                        if archive:
                            report["interrupted_archives"].append({
                                "episode_id": spec.episode_id, "attempt_number": attempt_number,
                                "archive_path": archive,
                                "diagnostics": [episode.get("failure_reason")],
                            })
                        if errors := client.safe_stop():
                            report["failure_reason"] = "safe stop failed before infrastructure replacement: " + "; ".join(errors)
                            break
                        attempt_number += 1
                        continue
                    report["failure_reason"] = f"infrastructure replacement exhausted for {spec.episode_id}"
                    break
                if report["failure_reason"]:
                    break

            if report["episodes"] and not projection_recorded:
                first = report["episodes"][0]
                first_size = int(first["bag_size_bytes"])
                current = disk_state("/")
                remaining = 16 - len(report["episodes"])
                projected_total = first_size * 16
                projected_free = current["available_bytes"] - first_size * remaining
                report["storage_projection"] = {
                    "basis_episode_id": first["episode_id"], "basis_bag_size_bytes": first_size,
                    "projected_total_raw_size_bytes": projected_total,
                    "disk_available_at_projection_bytes": current["available_bytes"],
                    "remaining_episode_count": remaining,
                    "projected_free_after_collection_bytes": projected_free,
                    "minimum_required_projected_free_bytes": MINIMUM_PROJECTED_FREE_BYTES,
                    "result": "PASS" if projected_free >= MINIMUM_PROJECTED_FREE_BYTES else "FAIL",
                }
                projection_recorded = True
                if projected_free < MINIMUM_PROJECTED_FREE_BYTES:
                    report["failure_reason"] = "first-bag storage projection would leave less than 6 GiB"
                    break
            report["completed_episode_count"] = len(report["episodes"])
            _write_collection_progress(summary_path, report)
        report["disk_post_collection"] = disk_state("/")
        gate = validate_collection_gate(report["episodes"])
        forbidden_after = _raw_root_has_forbidden_episode(backend)
        gate["gates"]["external_root_has_only_intended_ids"] = not forbidden_after
        gate["result"] = "PASS" if all(gate["gates"].values()) else "FAIL"
        report["collection_gate"] = gate
        report["completed_episode_count"] = len(report["episodes"])
        report["passed_episode_count"] = sum(item.get("result") == "PASS" for item in report["episodes"])
        report["genuine_expert_failure_count"] = sum(
            item.get("classification") == "GENUINE_EXPERT_FAIL" for item in report["episodes"]
        )
        report["infrastructure_replacement_attempt_count"] = sum(
            max(0, _attempt_count(result_dir, spec.episode_id) - 1)
            for spec in episode_specs()
        )
        report["total_raw_storage_bytes"] = sum(
            int(item.get("bag_size_bytes") or 0) for item in report["episodes"] if item.get("result") == "PASS"
        )
        report["result"] = "PASS" if gate["result"] == "PASS" and report["failure_reason"] is None else "FAIL"
    finally:
        final_stop_errors = client.safe_stop()
        report["final_safe_stop_success"] = not final_stop_errors
        report["final_safe_stop_errors"] = final_stop_errors
        try:
            report["world_restoration"] = _restore_world(client, original_world)
        except BaseException as exc:
            report["world_restoration"] = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"}
        report["simulator_tracked_source_status_after"] = simulator_tracked_status(sim_root)
        try:
            unchanged = directory_file_manifest_sha256(
                repo / task.frozen["result_directory"]
            ) == task.frozen["result_manifest_sha256"]
        except BaseException:
            unchanged = False
        report["frozen_expert_evidence_unchanged"] = unchanged
        if (
            final_stop_errors
            or report["world_restoration"].get("result") != "PASS"
            or report["simulator_tracked_source_status_after"].get("result") != "PASS"
            or not unchanged
        ):
            report["result"] = "FAIL"
        _write_collection_progress(summary_path, report)
    if report["result"] != "PASS":
        raise TrainDataGateError(report.get("failure_reason") or "16/16 collection gate failed")
    return report


def transform_odom_to_world(
    x_m: float, y_m: float, *, initial_odom: tuple[float, float, float],
    initial_world: tuple[float, float, float],
) -> tuple[float, float]:
    """Rigidly align odom XY to the frozen reset pose for evaluation-only route s."""
    angle = initial_world[2] - initial_odom[2]
    dx, dy = x_m - initial_odom[0], y_m - initial_odom[1]
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        initial_world[0] + cosine * dx - sine * dy,
        initial_world[1] + sine * dx + cosine * dy,
    )


def scan_odom_route_s(
    mcap_path: Path, route: Any, initial_world_pose: dict[str, Any], topic: str = "/odom",
) -> list[tuple[int, float]]:
    decoded_records: list[tuple[int, float, float, float]] = []
    for schema, channel, record, decoded in _iter_decoded(mcap_path, [topic]):
        if schema.name != "nav_msgs/msg/Odometry":
            raise TrainDataGateError(f"{channel.topic} type is {schema.name}, expected nav_msgs/msg/Odometry")
        position = decoded.pose.pose.position
        orientation = decoded.pose.pose.orientation
        yaw = 2.0 * math.atan2(float(orientation.z), float(orientation.w))
        decoded_records.append((int(record.log_time), float(position.x), float(position.y), yaw))
    if not decoded_records:
        raise TrainDataGateError("no odometry for evaluation-only route-progress QA")
    first = decoded_records[0]
    initial_odom = (first[1], first[2], first[3])
    initial_world = (
        float(initial_world_pose["x"]), float(initial_world_pose["y"]),
        float(initial_world_pose["yaw"]),
    )
    return [
        (
            record_time,
            float(route.project(transform_odom_to_world(
                x_m, y_m, initial_odom=initial_odom, initial_world=initial_world,
            )).s),
        )
        for record_time, x_m, y_m, _ in decoded_records
    ]


def attach_causal_route_s(
    rows: list[dict[str, Any]], odom: Sequence[tuple[int, float]],
) -> dict[str, Any]:
    times = [record[0] for record in odom]
    ages_ms: list[float] = []
    for row in rows:
        camera_time = int(row["camera_record_time_ns"])
        index = bisect.bisect_right(times, camera_time) - 1
        if index < 0:
            raise TrainDataGateError("accepted camera frame has no causal odometry")
        record_time, route_s = odom[index]
        if record_time > camera_time:
            raise TrainDataGateError("future odometry selected for evaluation-only QA")
        row["route_s_m"] = route_s
        row["route_s_record_time_ns"] = record_time
        ages_ms.append((camera_time - record_time) / 1e6)
    return {"count": len(rows), "future_violations": 0, "age_ms": numeric_distribution(ages_ms)}


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_temporal_sequences(
    rows: Sequence[dict[str, Any]], spec: EpisodeSpec, source_manifest_sha256: str,
    maximum_gap_s: float = MAXIMUM_ADJACENT_GAP_S,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build unpadded, within-episode causal [t-2,t-1,t] sequences."""
    if maximum_gap_s != MAXIMUM_ADJACENT_GAP_S:
        raise TrainDataGateError("maximum adjacent temporal gap must remain exactly 0.120 s")
    ordered = list(rows)
    if any(row.get("episode_id") != spec.episode_id for row in ordered):
        raise TrainDataGateError("temporal source rows cross an episode boundary")
    if any(row.get("scenario_id") != spec.scenario_id or row.get("repeat_id") != spec.repeat_id for row in ordered):
        raise TrainDataGateError("temporal source rows cross a scenario/repeat boundary")
    accepted: list[dict[str, Any]] = []
    rejected_gap = 0
    accepted_gaps: list[float] = []
    accepted_spans: list[float] = []
    for index in range(2, len(ordered)):
        a, b, c = ordered[index - 2:index + 1]
        times = tuple(int(row["camera_record_time_ns"]) for row in (a, b, c))
        if not times[0] < times[1] < times[2]:
            raise TrainDataGateError(f"non-causal or duplicate camera timestamps in {spec.episode_id}")
        paths = tuple(str(row["image_path"]) for row in (a, b, c))
        if len(set(paths)) != 3:
            raise TrainDataGateError(f"duplicate-frame history padding in {spec.episode_id}")
        gap_1 = (times[1] - times[0]) / 1e9
        gap_2 = (times[2] - times[1]) / 1e9
        if gap_1 > maximum_gap_s or gap_2 > maximum_gap_s:
            rejected_gap += 1
            continue
        row = {
            "sequence_id": f"{spec.episode_id}_seq_{len(accepted):06d}",
            "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
            "scenario_role": "TRAIN", "repeat_id": spec.repeat_id,
            "cone_scenario_id": spec.scenario_id,
            "frame_t_minus_2": paths[0], "frame_t_minus_1": paths[1], "frame_t": paths[2],
            "frame_t_minus_2_sha256": a["image_sha256"],
            "frame_t_minus_1_sha256": b["image_sha256"], "frame_t_sha256": c["image_sha256"],
            "camera_timestamp_t_minus_2_ns": times[0],
            "camera_timestamp_t_minus_1_ns": times[1], "camera_timestamp_t_ns": times[2],
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
            raise TrainDataGateError("future steering label entered temporal manifest")
        accepted.append(row)
        accepted_gaps.extend((gap_1, gap_2))
        accepted_spans.append(gap_1 + gap_2)
    stats = {
        "episode_id": spec.episode_id, "source_frames": len(ordered),
        "temporal_candidate_sequences": max(0, len(ordered) - 2),
        "accepted_temporal_sequences": len(accepted),
        "gap_rejects": rejected_gap, "boundary_rejects": min(2, len(ordered)),
        "adjacent_gap_s": numeric_distribution(accepted_gaps),
        "oldest_to_current_span_s": numeric_distribution(accepted_spans),
        "future_label_violations": 0,
    }
    return accepted, stats


def scenario_regions(bundle: ScenarioBundle, route_length_m: float, margin_m: float) -> dict[str, tuple[float, float]]:
    departure = float(bundle.plan.departure_start_s_m)
    cone = float(bundle.plan.cone_s_m)
    returned = float(bundle.plan.return_end_s_m)
    regions = {
        "approach": (max(0.0, departure - margin_m), departure),
        "avoidance": (departure, cone),
        "pass_return": (cone, returned),
        "post_recovery": (returned, min(route_length_m, returned + margin_m)),
    }
    if any(not lower < upper for lower, upper in regions.values()):
        raise TrainDataGateError(f"invalid QA regions for scenario {bundle.scenario.scenario_id}: {regions}")
    return regions


def region_coverage(
    temporal_rows: Sequence[dict[str, Any]], regions: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    counts = {
        name: sum(lower <= float(row["route_progress_m"]) < upper for row in temporal_rows)
        for name, (lower, upper) in regions.items()
    }
    return {
        "result": "PASS" if counts and all(count > 0 for count in counts.values()) else "FAIL",
        "regions_s_m": {name: list(bounds) for name, bounds in regions.items()},
        "accepted_temporal_sequence_counts": counts,
        "evaluation_only": True, "future_neural_input": False,
    }


def _evenly_spaced(items: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(items) <= count:
        return list(items)
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def create_maneuver_contact_sheet(
    dataset_root: Path, temporal_rows: Sequence[dict[str, Any]],
    regions: dict[str, tuple[float, float]], output: Path,
) -> dict[str, int]:
    """Create three current-frame samples per maneuver region for visual review."""
    names = list(regions)
    selected: dict[str, list[dict[str, Any]]] = {}
    for name, (lower, upper) in regions.items():
        candidates = [
            row for row in temporal_rows
            if lower <= float(row["route_progress_m"]) < upper
        ]
        if not candidates:
            raise TrainDataGateError(f"cannot preview empty maneuver region {name}")
        selected[name] = _evenly_spaced(candidates, 3)
    cell_width, image_height, label_height = 200, 66, 18
    cell_height = image_height + label_height
    sheet = Image.new("RGB", (cell_width * len(names), cell_height * 3), "black")
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(names):
        rows = selected[name]
        for row_index in range(3):
            row = rows[min(row_index, len(rows) - 1)]
            with Image.open(dataset_root / row["frame_t"]) as source:
                image = source.convert("RGB")
                if image.size != (200, 66):
                    raise TrainDataGateError(f"preview source has wrong ROI size: {image.size}")
                x, y = column * cell_width, row_index * cell_height
                sheet.paste(image, (x, y + label_height))
            draw.text((column * cell_width + 2, row_index * cell_height + 2), name, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False)
    sheet.close()
    return {name: len(rows) for name, rows in selected.items()}


def create_overview_sheet(previews: Sequence[Path], output: Path) -> None:
    if len(previews) != 16:
        raise TrainDataGateError(f"overview requires exactly 16 episode previews, found {len(previews)}")
    thumbnails: list[Image.Image] = []
    for path in previews:
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((400, 129), Image.Resampling.LANCZOS)
            labeled = Image.new("RGB", (thumbnail.width, thumbnail.height + 16), "black")
            labeled.paste(thumbnail, (0, 16))
            ImageDraw.Draw(labeled).text((3, 2), path.stem, fill="white")
            thumbnails.append(labeled)
            thumbnail.close()
    width = max(image.width for image in thumbnails)
    height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (width * 2, height * 8), "black")
    for index, image in enumerate(thumbnails):
        sheet.paste(image, ((index % 2) * width, (index // 2) * height))
        image.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=False)
    sheet.close()


def _dataset_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _aggregate_rejections(episode_metrics: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        reason: sum(int(item["counts"]["rejection_by_reason"].get(reason, 0)) for item in episode_metrics)
        for reason in REJECTION_REASONS
    }


def _per_episode_distribution(
    temporal_by_episode: dict[str, list[dict[str, Any]]], extractor_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        episode_id: steering_distribution(
            [float(row["target_steering_rad"]) for row in rows], extractor_config,
        )
        for episode_id, rows in temporal_by_episode.items()
    }


def _per_scenario_distribution(
    temporal_rows: Sequence[dict[str, Any]], extractor_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        scenario_id: steering_distribution(
            [
                float(row["target_steering_rad"]) for row in temporal_rows
                if row["scenario_id"] == scenario_id
            ],
            extractor_config,
        )
        for scenario_id in TRAIN_SCENARIOS
    }


def _dataset_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Random-Cone 1.0 m/s TRAIN Temporal Dataset QC V1", "",
        f"Result: **{report.get('result', 'FAIL')}**", "",
        "| Episode | Scenario | Repeat | Frames | Sequences | Gap rejects | Approach/Avoid/Return/Post |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("episodes", []):
        regions = item["cone_region_coverage"]["accepted_temporal_sequence_counts"]
        lines.append(
            f"| {item['episode_id']} | {item['scenario_id']} | {item['repeat_id']} | "
            f"{item['accepted_images']} | {item['accepted_temporal_sequences']} | "
            f"{item['gap_rejects']} | {regions['approach']}/{regions['avoidance']}/"
            f"{regions['pass_return']}/{regions['post_recovery']} |"
        )
    lines.extend([
        "", f"Future-label violations: {report.get('future_label_violations')}",
        f"Temporal sequences: {report.get('temporal', {}).get('accepted_sequence_count')}",
        f"Visual QC: {(report.get('visual_qc') or {}).get('result')}",
        "", "The dataset contains TRAIN scenarios 01–08 only. Route progress and cone scenario metadata are QA-only fields.",
        "No neural training, model export, or neural closed-loop run was performed.", "",
    ])
    return "\n".join(lines)


def _write_dataset_summary(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)
    path.parent.joinpath("REPORT.md").write_text(_dataset_report_markdown(report), encoding="utf-8")


def _resume_dataset_if_complete(
    dataset_root: Path, summary_path: Path, collection_sha: str,
) -> dict[str, Any] | None:
    metadata_path = dataset_root / "dataset_metadata.json"
    if not dataset_root.exists() and not summary_path.exists():
        return None
    if metadata_path.is_file() and summary_path.is_file():
        summary = _load_json(summary_path)
        metadata = _load_json(metadata_path)
        if (
            summary.get("result") in {"PENDING_VISUAL_QC", "PASS", "FAIL"}
            and metadata.get("collection_summary_sha256") == collection_sha
            and summary.get("dataset_root") == str(dataset_root)
            and summary.get("episode_ids") == list(EPISODE_ORDER)
        ):
            return summary
    # Preserve a partial extraction for diagnosis; source MCAPs remain untouched.
    if dataset_root.exists():
        archive = dataset_root.parent / "_interrupted_train_dataset" / datetime.now().strftime("%Y%m%dT%H%M%S")
        archive.parent.mkdir(parents=True, exist_ok=True)
        dataset_root.rename(archive)
    return None


def dataset_stage(repo: Path, sim_root: Path, task: TaskConfig) -> dict[str, Any]:
    collection_path = task.result_dir(repo, "collection") / "summary.json"
    collection = _load_json(collection_path)
    if (
        collection.get("result") != "PASS"
        or collection.get("passed_episode_count") != 16
        or (collection.get("collection_gate") or {}).get("result") != "PASS"
    ):
        raise TrainDataGateError("collection gate is not 16/16 PASS; extraction prohibited")
    collection_sha = sha256_file(collection_path)
    expert, all_bundles, audit = audit_frozen_expert(repo, sim_root, task)
    bundles = {bundle.scenario.scenario_id: bundle for bundle in all_bundles[:8]}
    collection_result_dir = task.result_dir(repo, "collection")
    dataset_result_dir = task.result_dir(repo, "dataset")
    dataset_result_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = sim_root / "userdata" / task.dataset["data_relative_root"]
    summary_path = dataset_result_dir / "summary.json"
    resumed = _resume_dataset_if_complete(dataset_root, summary_path, collection_sha)
    if resumed is not None:
        return resumed

    backend = DockerRosBackend(collector_config(task), sim_root)
    source_metadata: dict[str, dict[str, Any]] = {}
    for spec in episode_specs():
        final_path = collection_result_dir / "episodes" / f"{spec.episode_id}.json"
        status, metadata, errors = validate_existing_episode(
            spec, result_path=final_path, task=task, expert=expert,
            bundle=bundles[spec.scenario_id], backend=backend,
        )
        if status != "VALID" or metadata is None:
            raise TrainDataGateError(f"{spec.episode_id} is not a finalized valid source: {errors}")
        source_metadata[spec.episode_id] = metadata
    if _raw_root_has_forbidden_episode(backend):
        raise TrainDataGateError("raw root contains a non-TRAIN episode")

    extractor_path = repo / task.dataset["canonical_extractor_config_path"]
    extractor_config = load_extractor_config(extractor_path)
    extractor_sha = sha256_file(extractor_path)
    prepare_output_root(dataset_root, False)
    (dataset_root / "temporal_manifests").mkdir()
    (dataset_root / "previews" / "maneuver_regions").mkdir()
    episode_metrics: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    all_temporal_rows: list[dict[str, Any]] = []
    temporal_by_episode: dict[str, list[dict[str, Any]]] = {}
    preview_paths: list[Path] = []
    compact_episodes: list[dict[str, Any]] = []

    for spec in episode_specs():
        bundle = bundles[spec.scenario_id]
        collector_path = collection_result_dir / "episodes" / f"{spec.episode_id}.json"
        collector = source_metadata[spec.episode_id]
        mcap_path = Path(collector["bag_mcap_path"])
        if sha256_file(mcap_path) != collector["bag_mcap_sha256"]:
            raise TrainDataGateError(f"source MCAP changed before extraction: {spec.episode_id}")
        metrics, rows = extract_episode(
            episode_id=spec.episode_id, mcap_path=mcap_path,
            collector_metadata_path=collector_path, dataset_root=dataset_root,
            config=extractor_config, config_sha256=extractor_sha,
            source_path_identity=mcap_path.relative_to(backend.host_data_root).as_posix(),
            collector_metadata_identity=collector_path.relative_to(repo).as_posix(),
        )
        odom = scan_odom_route_s(mcap_path, bundle.plan.nominal, collector["preflight"]["pose"])
        metrics["evaluation_only_route_s"] = attach_causal_route_s(rows, odom)
        for row in rows:
            row.update({
                "scenario_id": spec.scenario_id, "scenario_role": "TRAIN",
                "repeat_id": spec.repeat_id, "cone_scenario_id": spec.scenario_id,
            })
            row["image_sha256"] = sha256_file(dataset_root / row["image_path"])
        frame_manifest = dataset_root / "manifests" / f"{spec.episode_id}.csv"
        write_csv(frame_manifest, rows, FRAME_MANIFEST_COLUMNS)
        frame_manifest_sha = sha256_file(frame_manifest)
        temporal, temporal_stats = build_temporal_sequences(rows, spec, frame_manifest_sha)
        temporal_manifest = dataset_root / "temporal_manifests" / f"{spec.episode_id}.csv"
        write_csv(temporal_manifest, temporal, TEMPORAL_MANIFEST_COLUMNS)
        regions = scenario_regions(
            bundle, float(collector["preflight"]["route_length_m"]),
            float(task.dataset["route_region_margin_m"]),
        )
        coverage = region_coverage(temporal, regions)
        preview = dataset_root / "previews" / "maneuver_regions" / f"{spec.episode_id}.png"
        preview_selection = create_maneuver_contact_sheet(dataset_root, temporal, regions, preview)
        preview_paths.append(preview)
        metrics.update({
            "scenario_id": spec.scenario_id, "scenario_role": "TRAIN",
            "repeat_id": spec.repeat_id, "temporal": temporal_stats,
            "cone_region_coverage": coverage,
            "maneuver_preview": {
                "path": preview.relative_to(dataset_root).as_posix(),
                "sha256": sha256_file(preview), "selection_count_by_region": preview_selection,
            },
        })
        episode_metrics.append(metrics)
        all_frame_rows.extend(rows)
        all_temporal_rows.extend(temporal)
        temporal_by_episode[spec.episode_id] = temporal
        compact = {
            "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
            "repeat_id": spec.repeat_id, "scenario_role": "TRAIN", "result": (
                "PASS" if metrics["result"] == "PASS" and coverage["result"] == "PASS"
                and temporal_stats["future_label_violations"] == 0 else "FAIL"
            ),
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
            "gap_rejects": temporal_stats["gap_rejects"],
            "boundary_rejects": temporal_stats["boundary_rejects"],
            "adjacent_gap_s": temporal_stats["adjacent_gap_s"],
            "oldest_to_current_span_s": temporal_stats["oldest_to_current_span_s"],
            "cone_region_coverage": coverage,
            "frame_manifest_sha256": frame_manifest_sha,
            "temporal_manifest_sha256": sha256_file(temporal_manifest),
            "source_mcap_sha256": collector["bag_mcap_sha256"],
            "canonical_preview": metrics["artifacts"]["preview_path"],
            "maneuver_preview": metrics["maneuver_preview"],
        }
        compact_episodes.append(compact)
        write_json(dataset_result_dir / "episodes" / f"{spec.episode_id}.json", compact)
        print(json.dumps({
            "stage": "extract", "episode_id": spec.episode_id,
            "accepted_images": compact["accepted_images"],
            "accepted_sequences": compact["accepted_temporal_sequences"],
            "gap_rejects": compact["gap_rejects"], "regions": coverage["result"],
        }), flush=True)

    write_csv(dataset_root / "manifest.csv", all_frame_rows, FRAME_MANIFEST_COLUMNS)
    train_temporal_path = dataset_root / "temporal_manifests" / "train.csv"
    write_csv(train_temporal_path, all_temporal_rows, TEMPORAL_MANIFEST_COLUMNS)
    overview = dataset_root / "previews" / "all_train_episodes_maneuver_overview.png"
    create_overview_sheet(preview_paths, overview)
    future_violations = sum(
        item["synchronization"]["future_label_violations"] for item in episode_metrics
    )
    all_steering_ages = [float(row["steering_age_ms"]) for row in all_frame_rows]
    all_speed_ages = [float(row["speed_age_ms"]) for row in all_frame_rows]
    all_intervals = [
        (int(b["camera_record_time_ns"]) - int(a["camera_record_time_ns"])) / 1e6
        for spec in episode_specs()
        for a, b in zip(
            [row for row in all_frame_rows if row["episode_id"] == spec.episode_id],
            [row for row in all_frame_rows if row["episode_id"] == spec.episode_id][1:],
        )
    ]
    all_gaps = [
        gap for row in all_temporal_rows
        for gap in (float(row["adjacent_gap_1_s"]), float(row["adjacent_gap_2_s"]))
    ]
    all_spans = [float(row["oldest_to_current_span_s"]) for row in all_temporal_rows]
    source_ids = {row["episode_id"] for row in all_temporal_rows}
    scenario_ids = {row["scenario_id"] for row in all_temporal_rows}
    repeat_ids = {row["repeat_id"] for row in all_temporal_rows}
    sequence_ids = [row["sequence_id"] for row in all_temporal_rows]
    technical_gates = {
        "sixteen_readable_train_bags": len(episode_metrics) == 16,
        "exact_train_episode_order": [item["episode_id"] for item in compact_episodes] == list(EPISODE_ORDER),
        "only_scenarios_01_through_08": scenario_ids == set(TRAIN_SCENARIOS),
        "exactly_two_repeats": repeat_ids == set(REPEAT_IDS),
        "no_validation_or_holdout": not scenario_ids.intersection({"09", "10", "11", "12"}),
        "source_episode_set_exact": source_ids == set(EPISODE_ORDER),
        "no_duplicate_sequence_ids": len(sequence_ids) == len(set(sequence_ids)),
        "future_label_violations_zero": future_violations == 0,
        "causal_steering_targets": all(
            int(row["steering_target_timestamp_ns"]) <= int(row["camera_timestamp_t_ns"])
            for row in all_temporal_rows
        ),
        "strict_three_frame_order": all(
            int(row["camera_timestamp_t_minus_2_ns"])
            < int(row["camera_timestamp_t_minus_1_ns"])
            < int(row["camera_timestamp_t_ns"])
            for row in all_temporal_rows
        ),
        "adjacent_gap_at_most_0p120_s": bool(all_gaps) and max(all_gaps) <= MAXIMUM_ADJACENT_GAP_S,
        "no_duplicate_frame_padding": all(
            len({row["frame_t_minus_2"], row["frame_t_minus_1"], row["frame_t"]}) == 3
            for row in all_temporal_rows
        ),
        "two_boundary_rejects_per_episode": all(item["boundary_rejects"] == 2 for item in compact_episodes),
        "nonempty_temporal_sequences_each": all(item["accepted_temporal_sequences"] > 0 for item in compact_episodes),
        "cone_regions_covered_each_episode": all(
            item["cone_region_coverage"]["result"] == "PASS" for item in compact_episodes
        ),
        "both_avoidance_sides_represented": {
            bundles[scenario].scenario.chosen_side for scenario in TRAIN_SCENARIOS
        } == {"left", "right"},
        "canonical_roi_and_rgb": (
            extractor_config["roi"] == {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360}
            and extractor_config["output_width"] == 200 and extractor_config["output_height"] == 66
            and extractor_config["source_encoding"] == "rgb8"
        ),
        "no_image_decode_failures": all(
            item["counts"]["rejection_by_reason"]["image_decode_error"] == 0
            for item in episode_metrics
        ),
        "all_extraction_episode_gates_pass": all(item["result"] == "PASS" for item in compact_episodes),
    }
    if not all(technical_gates.values()):
        technical_result = "FAIL"
    else:
        technical_result = "PASS"
    report: dict[str, Any] = {
        "version": DATASET_VERSION, "generated_utc": utc_now(),
        "result": "PENDING_VISUAL_QC" if technical_result == "PASS" else "FAIL",
        "technical_qc_result": technical_result,
        "task_config_sha256": task.sha256,
        "collection_summary_sha256": collection_sha,
        "frozen_expert_result_manifest_sha256": task.frozen["result_manifest_sha256"],
        "frozen_audit_result": audit["result"], "dataset_root": str(dataset_root),
        "episode_count": 16, "episode_ids": list(EPISODE_ORDER),
        "scenario_ids": list(TRAIN_SCENARIOS), "repeat_ids": list(REPEAT_IDS),
        "scenario_role": "TRAIN", "validation_sequences": 0, "holdout_sequences": 0,
        "canonical_extractor": {
            "path": task.dataset["canonical_extractor_config_path"], "sha256": extractor_sha,
            "synchronization": "latest steering/speed MCAP record at or before camera record time",
            "active_window": "dominant contiguous abs(speed) >= 0.10 m/s segment",
            "source": "480x360 RGB /camera/image_raw", "crop": "x=0:480,y=160:360",
            "stored_image": "200x66 RGB PNG",
        },
        "counts": {
            "raw_camera_messages": sum(item["counts"]["total_camera_frames"] for item in episode_metrics),
            "active_window_camera_messages": sum(item["counts"]["active_window_camera_frames"] for item in episode_metrics),
            "accepted_images": len(all_frame_rows),
            "rejected_images": sum(item["counts"]["total_camera_frames"] for item in episode_metrics) - len(all_frame_rows),
            "rejection_by_reason": _aggregate_rejections(episode_metrics),
        },
        "retention_fraction": len(all_frame_rows) / sum(
            item["counts"]["total_camera_frames"] for item in episode_metrics
        ),
        "synchronization": {
            "steering_age_ms": numeric_distribution(all_steering_ages),
            "speed_age_ms": numeric_distribution(all_speed_ages),
            "accepted_camera_interval_ms": numeric_distribution(all_intervals),
            "future_label_violations": future_violations,
        },
        "future_label_violations": future_violations,
        "temporal": {
            "history_frames": 3, "maximum_adjacent_gap_s": MAXIMUM_ADJACENT_GAP_S,
            "candidate_sequence_count": sum(item["temporal_candidate_sequences"] for item in compact_episodes),
            "accepted_sequence_count": len(all_temporal_rows),
            "gap_reject_count": sum(item["gap_rejects"] for item in compact_episodes),
            "boundary_reject_count": sum(item["boundary_rejects"] for item in compact_episodes),
            "adjacent_gap_s": numeric_distribution(all_gaps),
            "oldest_to_current_span_s": numeric_distribution(all_spans),
            "manifest": str(train_temporal_path), "manifest_sha256": sha256_file(train_temporal_path),
        },
        "steering_distribution": {
            "overall_temporal_targets": steering_distribution(
                [float(row["target_steering_rad"]) for row in all_temporal_rows], extractor_config,
            ),
            "by_episode_repeat": _per_episode_distribution(temporal_by_episode, extractor_config),
            "by_scenario": _per_scenario_distribution(all_temporal_rows, extractor_config),
        },
        "episodes": compact_episodes,
        "cone_region_coverage": {
            item["episode_id"]: item["cone_region_coverage"] for item in compact_episodes
        },
        "field_boundaries": {
            "future_neural_inputs": ["frame_t_minus_2", "frame_t_minus_1", "frame_t"],
            "future_neural_target": "target_steering_rad",
            "qa_only_not_neural_inputs": [
                "scenario_id", "repeat_id", "cone_scenario_id", "route_progress_m",
                "source_mcap_sha256", "image_sha256",
            ],
        },
        "visual_qc": {
            "result": "PENDING_MANUAL_REVIEW", "episode_preview_count": len(preview_paths),
            "canonical_preview_count": len(episode_metrics),
            "overview_path": str(overview), "overview_sha256": sha256_file(overview),
            "required_checks": [
                "cone visible where expected", "lane and road intact",
                "avoidance and recovery represented", "ROI correct", "no corrupt images",
                "no reset or teleport images", "left and right avoidance represented",
            ],
        },
        "quality_gates": technical_gates,
        "neural_training_performed": False, "model_export_performed": False,
        "neural_training_justified": False,
        "external_storage": {
            "source_raw_bytes": sum(int(item["bag_size_bytes"]) for item in source_metadata.values()),
            "derived_dataset_bytes": None,
        },
        "simulator_tracked_source_status": simulator_tracked_status(sim_root),
        "frozen_expert_evidence_unchanged": directory_file_manifest_sha256(
            repo / task.frozen["result_directory"]
        ) == task.frozen["result_manifest_sha256"],
    }
    if report["simulator_tracked_source_status"].get("result") != "PASS" or not report["frozen_expert_evidence_unchanged"]:
        report["result"] = report["technical_qc_result"] = "FAIL"
    metadata_path = dataset_root / "dataset_metadata.json"
    write_json(metadata_path, report)
    report["external_storage"]["derived_dataset_bytes"] = _dataset_size(dataset_root)
    write_json(metadata_path, report)
    report["dataset_metadata_sha256"] = sha256_file(metadata_path)
    _write_dataset_summary(summary_path, report)
    if report["technical_qc_result"] != "PASS":
        raise TrainDataGateError("temporal dataset technical QC failed")
    return report


def finalize_visual_qc(
    repo: Path, sim_root: Path, task: TaskConfig, *, passed: bool, review_note: str,
) -> dict[str, Any]:
    if not review_note.strip():
        raise TrainDataGateError("visual QC requires a non-empty review note")
    result_dir = task.result_dir(repo, "dataset")
    summary_path = result_dir / "summary.json"
    report = _load_json(summary_path)
    dataset_root = Path(report["dataset_root"])
    metadata_path = dataset_root / "dataset_metadata.json"
    if report.get("technical_qc_result") != "PASS":
        raise TrainDataGateError("cannot pass visual QC when technical QC is not PASS")
    preview_paths = [
        dataset_root / item["maneuver_preview"]["path"] for item in report.get("episodes", [])
    ]
    canonical_paths = [dataset_root / item["canonical_preview"] for item in report.get("episodes", [])]
    overview = Path(report["visual_qc"]["overview_path"])
    if len(preview_paths) != 16 or len(canonical_paths) != 16 or not all(
        path.is_file() for path in [*preview_paths, *canonical_paths, overview]
    ):
        raise TrainDataGateError("visual QC artifact set is incomplete")
    create_overview_sheet(preview_paths, overview)
    report["visual_qc"] = {
        **report["visual_qc"], "result": "PASS" if passed else "FAIL",
        "overview_sha256": sha256_file(overview),
        "reviewed_utc": utc_now(), "reviewer": "Codex visual inspection",
        "review_note": review_note.strip(),
        "checks": {
            "all_16_source_episodes_inspectable": passed,
            "cone_visible_where_expected": passed,
            "lane_and_road_intact": passed,
            "avoidance_and_recovery_represented": passed,
            "roi_correct": passed, "no_corrupt_images": passed,
            "no_reset_or_teleport_images": passed,
            "left_and_right_avoidance_represented": passed,
        },
    }
    report["quality_gates"]["visual_qc_pass"] = passed
    frozen_unchanged = directory_file_manifest_sha256(
        repo / task.frozen["result_directory"]
    ) == task.frozen["result_manifest_sha256"]
    simulator = simulator_tracked_status(sim_root)
    report["frozen_expert_evidence_unchanged"] = frozen_unchanged
    report["simulator_tracked_source_status"] = simulator
    final_pass = (
        passed and all(report["quality_gates"].values()) and frozen_unchanged
        and simulator.get("result") == "PASS"
    )
    report["result"] = "PASS" if final_pass else "FAIL"
    report["neural_training_justified"] = final_pass
    report["neural_training_performed"] = False
    report["model_export_performed"] = False
    report["finalized_utc"] = utc_now()
    report.pop("dataset_metadata_sha256", None)
    write_json(metadata_path, report)
    report["external_storage"]["derived_dataset_bytes"] = _dataset_size(dataset_root)
    write_json(metadata_path, report)
    report["dataset_metadata_sha256"] = sha256_file(metadata_path)
    _write_dataset_summary(summary_path, report)
    return report


def audit_stage(repo: Path, sim_root: Path, task: TaskConfig) -> dict[str, Any]:
    _, _, audit = audit_frozen_expert(repo, sim_root, task)
    audit["disk_pre_collection"] = disk_state("/")
    audit["disk_gate_pass"] = audit["disk_pre_collection"]["available_bytes"] >= MINIMUM_COLLECTION_FREE_BYTES
    if not audit["disk_gate_pass"]:
        audit["result"] = "FAIL"
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("audit", "collect", "dataset", "visual-qc-pass", "visual-qc-fail", "all"),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--review-note", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    config_path = args.config or repo / "configs" / "random_cone_train_data_1p0_v1.json"
    try:
        task = load_task_config(config_path.resolve(), repo)
        sim_root = args.sim_root.expanduser().resolve()
        if args.stage == "audit":
            result = audit_stage(repo, sim_root, task)
        elif args.stage == "collect":
            result = collection_stage(repo, sim_root, task)
        elif args.stage == "dataset":
            result = dataset_stage(repo, sim_root, task)
        elif args.stage in {"visual-qc-pass", "visual-qc-fail"}:
            result = finalize_visual_qc(
                repo, sim_root, task, passed=args.stage == "visual-qc-pass",
                review_note=args.review_note,
            )
        else:
            collection_stage(repo, sim_root, task)
            result = dataset_stage(repo, sim_root, task)
        print(json.dumps({
            "stage": args.stage, "result": result.get("result"),
            "episode_count": result.get("episode_count", result.get("passed_episode_count")),
            "dataset_root": result.get("dataset_root"),
        }, indent=2, sort_keys=True))
        return 0 if result.get("result") in {"PASS", "PENDING_VISUAL_QC"} else 1
    except KeyboardInterrupt:
        print("ERROR: interrupted; partial artifacts were preserved for resume", flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
