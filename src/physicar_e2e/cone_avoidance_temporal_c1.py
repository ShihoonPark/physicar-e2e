"""Integrated gated one-cone Temporal PilotNet C1 simulation pipeline."""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch

from .cone_avoidance_environment import share_path, verify_derived_environment
from .cone_avoidance_expert import (
    ClearanceObserver,
    ExpertConfig,
    activate_world,
    audit_preserved_baselines,
    build_bypass_plan,
    full_preflight,
    obstacle_structural_preflight,
    run_obstacle_expert,
    validate_geometry,
)
from .dataset_extractor import (
    MANIFEST_COLUMNS,
    _iter_decoded,
    aggregate_summary,
    canonical_json_bytes,
    extract_episode,
    load_config as load_extractor_config,
    prepare_output_root,
    write_manifest,
)
from .high_speed_temporal import (
    TEMPORAL_FIELDS,
    TemporalOnnxModel,
    build_sequences,
    distribution,
    export_temporal_onnx,
    metrics as error_metrics,
    predict_temporal,
    read_temporal_rows,
    run_temporal_live,
    train_temporal,
    utc_now,
    validate_equivalence,
    warm_temporal_buffer,
)
from .high_speed_v5 import clock_health_preflight, write_json
from .pilotnet_inference import InferenceConfig
from .pilotnet_temporal import TEMPORAL_PARAMETER_COUNT, build_temporal_pilotnet
from .pilotnet_training import GateFailure, sha256_file
from .rosbag_collector import (
    CollectorConfig,
    DockerRosBackend,
    directory_size,
    git_commit,
    verify_bag,
)
from .sim_client import SimClient


VERSION = "cone_avoidance_temporal_c1_v1"
WORLD = "custom_71e69ee938032295503bfed557fde18c_e2e_cone_avoidance_v1"
EPISODES = tuple(f"cone_episode_{index:03d}" for index in range(1, 13))
TRAIN_EPISODES = EPISODES[:8]
VALIDATION_EPISODES = EPISODES[8:10]
HOLDOUT_EPISODES = EPISODES[10:]
C1_TRAIN_STRATA = ("v9_train", "cone_train")
C1_EVAL_STRATA = ("nominal_validation", "nominal_holdout", "cone_validation", "cone_holdout")
REQUIRED_TOPICS = (
    "/camera/image_raw", "/steering", "/speed", "/cmd_vel",
    "/odom", "/clock", "/tf", "/tf_static",
)
MAXIMUM_LIVE_ATTEMPTS = 5
TARGET_VALID_POLICY_RUNS = 3
REQUIRED_CLEARANCE_M = 0.05
RETURN_CTE_M = 0.05
RETURN_DURATION_S = 0.50
EXPECTED_IDENTITIES = {
    "cone_environment_config": ("configs/cone_avoidance_environment_v1.json", "0778f735fd431f7befcf0ed17f59379e48883635914dea6c91b8087cc285830a"),
    "cone_expert_config": ("configs/cone_avoidance_expert_v1.json", "77deb963369b34aba917c3db6f559a63f85e80ae6e78dd2cf20a34dfe54e9831"),
    "cone_expert_result": ("results/cone_avoidance_expert_v1/summary.json", "b1b0603ba34e50644802b6b2c46fdc92c9290f31273eed2f2ac0387830ce7082"),
    "v9_dataset_config": ("configs/high_speed_temporal_dataset_v1.json", "619bfea3c4fcbc5b8dcd8a37edfb1383bd97c695a38b69149dd3856b646aec46"),
    "v9_training_config": ("configs/pilotnet_training_v9_high_speed_temporal.json", "c8c9821cde130a4855d81a11428e49ee1c5f3f42faad8bd4939c23b412d5a017"),
    "v9_training_result": ("results/pilotnet_training_v9_high_speed_temporal/summary.json", "4c22c0b7f2d408b44b4698ff98d394ff3bde3f8d40c5dc34e6edd0a30d906f87"),
    "v9_live_result": ("results/pilotnet_e2e_v9_high_speed_temporal/summary.json", "56829cfc312f5cfe353458c60afcd146a54f406374d2e02546e20406cccaa6d2"),
}


def disk_state(path: str | Path = "/") -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path), "total_bytes": usage.total, "used_bytes": usage.used,
        "available_bytes": usage.free, "available_gib": usage.free / (1024 ** 3),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_file(path)
    if observed != expected:
        raise GateFailure(f"{label} identity changed: {observed} != {expected}")
    return observed


def audit_frozen(repo: Path, sim_root: Path) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for label, (relative, expected) in EXPECTED_IDENTITIES.items():
        path = repo / relative
        identities[label] = {"path": relative, "sha256": _assert_hash(path, expected, label)}
    v9 = _load_json(repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json")
    v9_live = _load_json(repo / "results/pilotnet_e2e_v9_high_speed_temporal/summary.json")
    expert = _load_json(repo / "results/cone_avoidance_expert_v1/summary.json")
    if v9.get("result") != "PASS" or v9.get("architecture", {}).get("parameter_count") != 255_819:
        raise GateFailure("preserved V9 training evidence is not PASS/255,819")
    if v9_live.get("result") != "PASS" or v9_live.get("policy_pass_count") != 3:
        raise GateFailure("preserved V9 live evidence is not 3/3 PASS")
    attempts = expert.get("attempts", [])
    if expert.get("result") != "PASS" or len(attempts) != 3 or any(
        item.get("classification") != "OBSTACLE_EXPERT_PASS" for item in attempts
    ):
        raise GateFailure("frozen Cone Avoidance Expert evidence is not 3/3 PASS")
    clearances = [float(item["metrics"]["minimum_footprint_to_cone_clearance_m"]) for item in attempts]
    if min(clearances) < REQUIRED_CLEARANCE_M:
        raise GateFailure("frozen Expert clearance evidence is below 0.05 m")
    v9_artifacts = v9["artifacts"]
    for kind in ("checkpoint", "onnx"):
        artifact = Path(v9_artifacts[kind]["path"])
        _assert_hash(artifact, v9_artifacts[kind]["sha256"], f"V9 {kind}")
        identities[f"v9_{kind}"] = {"path": str(artifact), "sha256": v9_artifacts[kind]["sha256"]}
    v9_train = sim_root / "userdata/physicar_e2e/high_speed_temporal_v1/manifests/train.csv"
    identities["v9_train_manifest"] = {
        "path": str(v9_train),
        "sha256": _assert_hash(v9_train, "07d2f6cb6dd668352ae988dcfb771fa42739faed745681c0ed9b682b669834b4", "V9 train manifest"),
    }
    expert_config = ExpertConfig.load(repo / "configs/cone_avoidance_expert_v1.json", repo, sim_root)
    plan, route_data = build_bypass_plan(expert_config, sim_root)
    geometry = validate_geometry(expert_config, plan, route_data)
    environment = verify_derived_environment(expert_config.environment, share_path(sim_root))
    if expert_config.environment.derived_world != WORLD or geometry.get("result") != "PASS":
        raise GateFailure("frozen one-cone geometry identity failed")
    baseline_audit = audit_preserved_baselines(repo)
    return {
        "result": "PASS", "identities": identities, "baseline_audit": baseline_audit,
        "world": WORLD, "environment": environment,
        "cone": {"route_s_m": plan.cone_s_m, "x_m": plan.site.x_m, "y_m": plan.site.y_m},
        "avoidance": {
            "side": plan.side, "maximum_lateral_offset_m": plan.maximum_lateral_offset_m,
            "departure_s_m": plan.departure_start_s_m, "return_s_m": plan.return_end_s_m,
        },
        "expert_clearances_m": clearances,
    }


def load_collection_config(path: Path) -> tuple[dict[str, Any], CollectorConfig]:
    payload = _load_json(path)
    frozen = (
        payload.get("version"), payload.get("expected_world"), payload.get("episode_count"),
        tuple(payload.get("required_topics", [])), payload.get("data_relative_root"),
        payload.get("minimum_free_bytes_before_collection"), payload.get("retry_valid_expert_failure"),
    )
    if frozen != (
        "cone_avoidance_collection_v1", WORLD, 12, REQUIRED_TOPICS,
        "physicar_e2e/cone_avoidance_v1/raw", 8 * 1024 ** 3, False,
    ):
        raise GateFailure(f"cone collection contract changed: {frozen}")
    collector = CollectorConfig(
        expected_world=payload["expected_world"], required_topics=tuple(payload["required_topics"]),
        container_name=payload["container_name"], compose_service=payload["compose_service"],
        container_userdata_root=payload["container_userdata_root"], data_relative_root=payload["data_relative_root"],
        storage_id=payload["storage_id"], recorder_startup_timeout_s=payload["recorder_startup_timeout_s"],
        recorder_shutdown_timeout_s=payload["recorder_shutdown_timeout_s"], settle_duration_s=payload["settle_duration_s"],
        pilot_episode_count=payload["episode_count"], minimum_free_bytes=payload["minimum_free_bytes_before_training"],
        minimum_camera_messages=payload["minimum_camera_messages"],
    )
    collector.validate()
    return payload, collector


def _camera_topic_stats(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "camera_count": metadata["actual_topic_message_counts"].get("/camera/image_raw", 0),
        "camera_rate_hz": metadata["topic_metrics"].get("/camera/image_raw", {}).get("average_recorded_rate_hz"),
        "steering_count": metadata["actual_topic_message_counts"].get("/steering", 0),
        "steering_rate_hz": metadata["topic_metrics"].get("/steering", {}).get("average_recorded_rate_hz"),
    }


def collect_one_episode(
    episode_id: str, *, repo: Path, sim_root: Path, payload: dict[str, Any], collector: CollectorConfig,
    backend: DockerRosBackend, client: SimClient, expert: ExpertConfig, plan: Any,
    geometry: dict[str, Any], result_path: Path,
) -> dict[str, Any]:
    handle = None
    stop_result = None
    bag_info = None
    final_stop_errors: list[str] = []
    metadata: dict[str, Any] = {
        "version": "cone_avoidance_collection_episode_v1", "episode_id": episode_id,
        "world": WORLD, "required_topics": list(REQUIRED_TOPICS), "result": "FAIL",
        "classification": "INFRA_FAIL", "failure_reason": None, "recording_start_utc": None,
        "recording_end_utc": None, "expert_result_metrics": None, "bag_host_path": None,
        "bag_size_bytes": None, "bag_mcap_sha256": None, "bag_duration_s": None,
        "actual_topic_message_counts": {}, "topic_metrics": {}, "recorder_graceful_shutdown": False,
        "recorder_orphaned": False, "safe_stop_success": False,
        "canonical_expert_config_sha256": payload["frozen_expert_config_sha256"],
        "canonical_environment_config_sha256": payload["frozen_environment_config_sha256"],
        "physicar_e2e_git_commit": git_commit(repo),
    }
    try:
        if errors := client.safe_stop():
            raise GateFailure("episode initial safe stop failed: " + "; ".join(errors))
        activate_world(client, WORLD)
        initial, preflight = full_preflight(client, expert, plan, sim_root, geometry)
        time.sleep(collector.settle_duration_s)
        initial = obstacle_structural_preflight(client, expert, plan)
        clock = clock_health_preflight(client)
        if clock.get("result") != "PASS":
            raise GateFailure(str(clock.get("failure_reason", "clock health failed after settle")))
        metadata["preflight"] = {**preflight, "post_settle_structural_check": "PASS", "post_settle_clock": clock}
        if disk_state("/")["available_bytes"] < payload["minimum_free_bytes_before_training"]:
            raise GateFailure("root disk fell below 5 GiB before episode recording")
        handle = backend.start_recorder(episode_id, collector.required_topics)
        metadata.update({
            "bag_host_path": str(handle.host_bag_path), "bag_container_path": handle.container_bag_path,
            "recording_start_utc": utc_now(),
        })
        metrics = run_obstacle_expert(client, expert, initial, plan)
        metadata["expert_result_metrics"] = metrics
        metadata["classification"] = str(metrics["classification"])
        if metadata["classification"] != "OBSTACLE_EXPERT_PASS":
            metadata["failure_reason"] = f"frozen Expert episode classified {metadata['classification']}"
    except BaseException as exc:
        metadata["failure_reason"] = metadata["failure_reason"] or f"{type(exc).__name__}: {exc}"
    finally:
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
                if not stop_result.graceful:
                    metadata["failure_reason"] = metadata["failure_reason"] or stop_result.detail
            except BaseException as exc:
                metadata["recorder_orphaned"] = True
                metadata["failure_reason"] = metadata["failure_reason"] or f"recorder cleanup failed: {exc}"
            metadata["recording_end_utc"] = utc_now()
        final_stop_errors = client.safe_stop()
        metadata["safe_stop_success"] = not final_stop_errors
        metadata["safe_stop_errors"] = final_stop_errors
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            bag_info = backend.bag_info(handle)
            verify_bag(bag_info, collector.required_topics, collector.minimum_camera_messages)
            mcap_files = sorted(handle.host_bag_path.glob("*.mcap"))
            if len(mcap_files) != 1:
                raise GateFailure(f"expected exactly one finalized MCAP, found {len(mcap_files)}")
            metadata.update({
                "bag_duration_s": bag_info.duration_s,
                "bag_size_bytes": directory_size(handle.host_bag_path),
                "bag_mcap_path": str(mcap_files[0]), "bag_mcap_sha256": sha256_file(mcap_files[0]),
                "actual_topic_message_counts": dict(sorted(bag_info.topic_counts.items())),
                "topic_metrics": {
                    topic: {"message_count": count, "average_recorded_rate_hz": count / bag_info.duration_s}
                    for topic, count in sorted(bag_info.topic_counts.items())
                },
            })
        except BaseException as exc:
            metadata["failure_reason"] = metadata["failure_reason"] or f"bag integrity failed: {exc}"
    metrics = metadata.get("expert_result_metrics") or {}
    success = (
        metadata["failure_reason"] is None
        and metadata["classification"] == "OBSTACLE_EXPERT_PASS"
        and float(metrics.get("minimum_footprint_to_cone_clearance_m", -1)) >= REQUIRED_CLEARANCE_M
        and metrics.get("footprint_cone_intersection_occurred") is False
        and metrics.get("recovery_success") is True
        and metrics.get("off_track_event_count") == 0
        and metrics.get("api_failures") == 0 and metrics.get("pose_failures") == 0 and metrics.get("clock_failures") == 0
        and metadata["recorder_graceful_shutdown"] and not metadata["recorder_orphaned"]
        and metadata["bag_size_bytes"] is not None and metadata["safe_stop_success"]
    )
    metadata["result"] = "PASS" if success else "FAIL"
    if not success and metadata["failure_reason"] is None:
        metadata["failure_reason"] = "episode did not satisfy every frozen collection gate"
    metadata["topic_summary"] = _camera_topic_stats(metadata)
    write_json(result_path, metadata)
    return metadata


def collection_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_dir = repo / "results/cone_avoidance_collection_v1"
    summary_path = result_dir / "summary.json"
    external = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/raw"
    if summary_path.exists() or external.exists():
        raise FileExistsError("refusing to overwrite cone collection evidence or raw bags")
    audit = audit_frozen(repo, sim_root)
    payload, collector = load_collection_config(repo / "configs/cone_avoidance_collection_v1.json")
    pre_disk = disk_state("/")
    if pre_disk["available_bytes"] < payload["minimum_free_bytes_before_collection"]:
        raise GateFailure("root disk has less than 8 GiB available before collection")
    expert = ExpertConfig.load(repo / payload["frozen_expert_config"], repo, sim_root)
    plan, route_data = build_bypass_plan(expert, sim_root)
    geometry = validate_geometry(expert, plan, route_data)
    backend = DockerRosBackend(collector, sim_root)
    topic_types = backend.preflight(collector.required_topics)
    client = SimClient(expert.driver.base_url, expert.driver.api_timeout_s)
    episodes: list[dict[str, Any]] = []
    result_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "version": "cone_avoidance_collection_v1", "generated_utc": utc_now(), "result": "FAIL",
        "audit": audit, "disk_pre_collection": pre_disk, "raw_root": str(external),
        "required_topics": list(REQUIRED_TOPICS), "topic_types": topic_types, "episodes": [],
    }
    try:
        for episode_id in EPISODES:
            episode = collect_one_episode(
                episode_id, repo=repo, sim_root=sim_root, payload=payload, collector=collector,
                backend=backend, client=client, expert=expert, plan=plan, geometry=geometry,
                result_path=result_dir / f"{episode_id}.json",
            )
            episodes.append(episode)
            print(json.dumps({
                "stage": "collect", "episode_id": episode_id, "result": episode["result"],
                "classification": episode["classification"],
                "minimum_cone_clearance_m": (episode.get("expert_result_metrics") or {}).get("minimum_footprint_to_cone_clearance_m"),
                "bag_size_bytes": episode.get("bag_size_bytes"),
            }), flush=True)
            report["episodes"].append({
                "episode_id": episode_id, "result": episode["result"], "classification": episode["classification"],
                "bag_size_bytes": episode["bag_size_bytes"], "bag_mcap_sha256": episode["bag_mcap_sha256"],
                "topic_summary": episode["topic_summary"],
                "minimum_cone_clearance_m": (episode.get("expert_result_metrics") or {}).get("minimum_footprint_to_cone_clearance_m"),
                "recovery_success": (episode.get("expert_result_metrics") or {}).get("recovery_success"),
            })
            if episode["result"] != "PASS":
                break
        post_disk = disk_state("/")
        report["disk_post_collection"] = post_disk
        report["completed_episode_count"] = len(episodes)
        report["passed_episode_count"] = sum(item["result"] == "PASS" for item in episodes)
        report["total_raw_storage_bytes"] = sum(int(item.get("bag_size_bytes") or 0) for item in episodes)
        report["all_bags_finalized"] = len(episodes) == 12 and all(item["recorder_graceful_shutdown"] for item in episodes)
        report["recovery_success_count"] = sum((item.get("expert_result_metrics") or {}).get("recovery_success") is True for item in episodes)
        report["clearance_distribution_m"] = distribution([
            float(item["expert_result_metrics"]["minimum_footprint_to_cone_clearance_m"])
            for item in episodes if item.get("expert_result_metrics")
        ])
        report["result"] = "PASS" if len(episodes) == 12 and all(item["result"] == "PASS" for item in episodes) else "FAIL"
        if post_disk["available_bytes"] < payload["minimum_free_bytes_before_training"]:
            report["result"] = "FAIL"
            report["storage_risk"] = "root disk below 5 GiB after collection; training prohibited"
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if errors:
            report["result"] = "FAIL"
        write_json(summary_path, report)
    if report["result"] != "PASS":
        raise GateFailure("12/12 Cone Expert collection gate failed")
    return report


def transform_odom_to_world(
    x_m: float, y_m: float, *, initial_odom: tuple[float, float, float],
    initial_world: tuple[float, float, float],
) -> tuple[float, float]:
    """Map odom-frame XY into the frozen world frame using the reset pose."""
    angle = initial_world[2] - initial_odom[2]
    dx, dy = x_m - initial_odom[0], y_m - initial_odom[1]
    cosine, sine = math.cos(angle), math.sin(angle)
    return (
        initial_world[0] + cosine * dx - sine * dy,
        initial_world[1] + sine * dx + cosine * dy,
    )


def _scan_odom_route_s(
    mcap_path: Path, topic: str, route: Any, initial_world_pose: dict[str, Any],
) -> list[tuple[int, float]]:
    decoded_records: list[tuple[int, float, float, float]] = []
    for schema, channel, record, decoded in _iter_decoded(mcap_path, [topic]):
        if schema.name != "nav_msgs/msg/Odometry":
            raise GateFailure(f"{topic} type is {schema.name}, expected nav_msgs/msg/Odometry")
        position = decoded.pose.pose.position
        orientation = decoded.pose.pose.orientation
        yaw = 2.0 * math.atan2(float(orientation.z), float(orientation.w))
        decoded_records.append((int(record.log_time), float(position.x), float(position.y), yaw))
    if not decoded_records:
        raise GateFailure("no odometry available for evaluation-only route-s metrics")
    first = decoded_records[0]
    initial_odom = (first[1], first[2], first[3])
    initial_world = (
        float(initial_world_pose["x"]), float(initial_world_pose["y"]), float(initial_world_pose["yaw"]),
    )
    records: list[tuple[int, float]] = []
    for record_time, x_m, y_m, _ in decoded_records:
        world_xy = transform_odom_to_world(
            x_m, y_m, initial_odom=initial_odom, initial_world=initial_world
        )
        records.append((record_time, float(route.project(world_xy).s)))
    return records


def _attach_evaluation_route_s(rows: list[dict[str, Any]], odom: Sequence[tuple[int, float]]) -> dict[str, Any]:
    times = [item[0] for item in odom]
    ages_ms: list[float] = []
    future = 0
    for row in rows:
        camera_time = int(row["camera_record_time_ns"])
        index = bisect.bisect_right(times, camera_time) - 1
        if index < 0:
            raise GateFailure("camera sample has no causal odometry for evaluation")
        selected_time, route_s = odom[index]
        if selected_time > camera_time:
            future += 1
            raise GateFailure("future odometry selected for evaluation")
        row["route_s_m"] = route_s
        row["route_s_record_time_ns"] = selected_time
        ages_ms.append((camera_time - selected_time) / 1e6)
    return {"count": len(rows), "future_violations": future, "age_ms": distribution(ages_ms)}


CONE_MANIFEST_COLUMNS = [*MANIFEST_COLUMNS, "route_s_m", "route_s_record_time_ns"]
CONE_TEMPORAL_FIELDS = [*TEMPORAL_FIELDS, "route_s_m"]


def _write_rows(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _cone_temporal_sequences(
    episode_id: str, stratum: str, dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = dataset_root / "manifests" / f"{episode_id}.csv"
    built, stats = build_sequences(episode_id, stratum, dataset_root, manifest, .120)
    route_by_time: dict[int, float] = {}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            timestamp = int(raw["camera_header_time_ns"])
            if timestamp in route_by_time:
                raise GateFailure(f"duplicate camera header timestamp in {episode_id}")
            route_by_time[timestamp] = float(raw["route_s_m"])
    for row in built:
        row["route_s_m"] = route_by_time[int(row["timestamp_t_ns"])]
    return built, stats


def dataset_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    collection = _load_json(repo / "results/cone_avoidance_collection_v1/summary.json")
    if collection.get("result") != "PASS" or collection.get("passed_episode_count") != 12:
        raise GateFailure("collection gate is not 12/12 PASS")
    result_dir = repo / "results/cone_avoidance_dataset_v1"
    result_path = result_dir / "summary.json"
    input_root = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/raw"
    dataset_root = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/dataset"
    if result_path.exists() or dataset_root.exists():
        raise FileExistsError("refusing to overwrite cone dataset evidence")
    config_path = repo / "configs/cone_avoidance_dataset_v1.json"
    config = load_extractor_config(config_path)
    expected_split = (tuple(config["train_episodes"]), tuple(config["validation_episodes"]), tuple(config["holdout_episodes"]))
    if expected_split != (TRAIN_EPISODES, VALIDATION_EPISODES, HOLDOUT_EPISODES):
        raise GateFailure("cone 8/2/2 episode split changed")
    if config.get("maximum_adjacent_gap_s") != .120 or config.get("allow_duplicate_padding") is not False:
        raise GateFailure("cone temporal causal/gap contract changed")
    expert = ExpertConfig.load(repo / "configs/cone_avoidance_expert_v1.json", repo, sim_root)
    plan, _ = build_bypass_plan(expert, sim_root)
    prepare_output_root(dataset_root, False)
    config_sha = sha256_file(config_path)
    episode_metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    result_dir.mkdir(parents=True, exist_ok=False)
    for episode_id in EPISODES:
        bag_root = input_root / episode_id / "bag"
        mcap_files = sorted(bag_root.glob("*.mcap"))
        if len(mcap_files) != 1:
            raise GateFailure(f"{episode_id}: expected exactly one MCAP, found {len(mcap_files)}")
        collector_path = repo / "results/cone_avoidance_collection_v1" / f"{episode_id}.json"
        metrics, rows = extract_episode(
            episode_id=episode_id, mcap_path=mcap_files[0], collector_metadata_path=collector_path,
            dataset_root=dataset_root, config=config, config_sha256=config_sha,
            source_path_identity=mcap_files[0].relative_to(input_root).as_posix(),
            collector_metadata_identity=collector_path.relative_to(repo).as_posix(),
        )
        collector_metadata = _load_json(collector_path)
        odom = _scan_odom_route_s(
            mcap_files[0], config["odom_topic"], plan.nominal,
            collector_metadata["preflight"]["pose"],
        )
        metrics["evaluation_only_route_s"] = _attach_evaluation_route_s(rows, odom)
        _write_rows(dataset_root / "manifests" / f"{episode_id}.csv", rows, CONE_MANIFEST_COLUMNS)
        episode_metrics.append(metrics)
        all_rows.extend(rows)
        write_json(result_dir / f"{episode_id}.json", metrics)
    _write_rows(dataset_root / "manifest.csv", all_rows, CONE_MANIFEST_COLUMNS)
    temporal_root = dataset_root / "temporal_manifests"
    temporal_rows: dict[str, list[dict[str, Any]]] = {}
    temporal_stats: dict[str, Any] = {}
    for name, episode_ids in (
        ("train", TRAIN_EPISODES), ("validation", VALIDATION_EPISODES), ("holdout", HOLDOUT_EPISODES),
    ):
        rows_for_stratum: list[dict[str, Any]] = []
        source_stats = []
        for episode_id in episode_ids:
            built, stats = _cone_temporal_sequences(episode_id, f"cone_{name}", dataset_root)
            rows_for_stratum.extend(built)
            source_stats.append(stats)
        _write_rows(temporal_root / f"{name}.csv", rows_for_stratum, CONE_TEMPORAL_FIELDS)
        temporal_rows[name] = rows_for_stratum
        temporal_stats[name] = {
            "sequence_count": len(rows_for_stratum), "episode_ids": list(episode_ids),
            "manifest": str(temporal_root / f"{name}.csv"),
            "manifest_sha256": sha256_file(temporal_root / f"{name}.csv"),
            "temporal_candidate_count": sum(item["temporal_candidates"] for item in source_stats),
            "rejected_boundary_count": sum(item["rejected_boundary"] for item in source_stats),
            "rejected_gap_count": sum(item["rejected_gap"] for item in source_stats),
            "adjacent_gap_s": distribution([
                gap for row in rows_for_stratum for gap in (
                    (int(row["timestamp_t_minus_1_ns"]) - int(row["timestamp_t_minus_2_ns"])) / 1e9,
                    (int(row["timestamp_t_ns"]) - int(row["timestamp_t_minus_1_ns"])) / 1e9,
                )
            ]),
            "sources": source_stats,
        }
    source_hashes = {
        name: {row["source_mcap_sha256"] for row in rows} for name, rows in temporal_rows.items()
    }
    if source_hashes["train"] & (source_hashes["validation"] | source_hashes["holdout"]):
        raise GateFailure("cone training/evaluation source hash leakage")
    if source_hashes["validation"] & source_hashes["holdout"]:
        raise GateFailure("cone validation/holdout source hash leakage")
    metadata = aggregate_summary(episode_metrics, all_rows, dataset_root, config, config_sha)
    metadata.pop("pilot_success_gate", None)
    metadata.update({
        "version": "cone_avoidance_dataset_v1", "episode_count": 12,
        "split": {"train": list(TRAIN_EPISODES), "validation": list(VALIDATION_EPISODES), "holdout": list(HOLDOUT_EPISODES)},
        "temporal": temporal_stats, "future_label_violations": metadata["synchronization"]["future_label_violations"],
        "evaluation_only_fields": ["route_s_m", "route_s_record_time_ns"],
        "neural_input_fields": ["camera_t_minus_2", "camera_t_minus_1", "camera_t"],
        "source_bag_hashes": {item["episode_id"]: item["source"]["mcap_sha256"] for item in episode_metrics},
    })
    gates = {
        "twelve_readable_episodes": len(episode_metrics) == 12,
        "future_label_violations_zero": metadata["future_label_violations"] == 0,
        "causal_zoh": all(item["synchronization"]["future_label_violations"] == 0 for item in episode_metrics),
        "temporal_gap_maximum_0p120": all(
            stats["adjacent_gap_s"]["max"] <= .120 for stats in temporal_stats.values()
        ),
        "first_two_frames_rejected_per_episode": all(
            stats["rejected_boundary_count"] == 2 * len(stats["episode_ids"]) for stats in temporal_stats.values()
        ),
        "episode_split_no_leakage": not any(source_hashes[a] & source_hashes[b] for a, b in (("train", "validation"), ("train", "holdout"), ("validation", "holdout"))),
    }
    metadata["quality_gates"] = gates
    metadata["result"] = "PASS" if all(gates.values()) else "FAIL"
    (dataset_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
    compact = {key: value for key, value in metadata.items() if key not in {"episodes", "episode_metrics", "config"}}
    compact["dataset_metadata_sha256"] = sha256_file(dataset_root / "dataset_metadata.json")
    compact["dataset_root"] = str(dataset_root)
    write_json(result_path, compact)
    if compact["result"] != "PASS":
        raise GateFailure("cone extraction/temporal quality gate failed")
    return compact


def load_c1_training_config(repo: Path) -> dict[str, Any]:
    path = repo / "configs/pilotnet_training_c1_cone_temporal.json"
    config = _load_json(path)
    v9 = _load_json(repo / "configs/pilotnet_training_v9_high_speed_temporal.json")
    fixed = (
        config.get("version"), config.get("input_channels"), config.get("history_frames"),
        config.get("maximum_adjacent_gap_s"), config.get("initialization"),
        config.get("sample_weighting"), config.get("resampling"), config.get("augmentation"),
    )
    if fixed != ("pilotnet_training_c1_cone_temporal", 9, 3, .120, "from_scratch", False, False, False):
        raise GateFailure(f"C1 training contract changed: {fixed}")
    for key in (
        "seed", "optimizer", "loss", "learning_rate", "batch_size", "max_epochs",
        "early_stopping_patience", "minimum_improvement", "max_steering_rad", "onnx_opset",
        "onnx_equivalence_samples", "onnx_mean_abs_difference_limit", "onnx_max_abs_difference_limit",
    ):
        if config.get(key) != v9.get(key):
            raise GateFailure(f"C1 differs from V9 training semantics: {key}")
    return config


def _rebuild_cone_temporal_manifests(dataset_root: Path) -> dict[str, Any]:
    temporal_root = dataset_root / "temporal_manifests"
    output: dict[str, Any] = {}
    for name, episode_ids in (
        ("train", TRAIN_EPISODES), ("validation", VALIDATION_EPISODES), ("holdout", HOLDOUT_EPISODES),
    ):
        rows: list[dict[str, Any]] = []
        stats = []
        for episode_id in episode_ids:
            built, source_stats = _cone_temporal_sequences(episode_id, f"cone_{name}", dataset_root)
            rows.extend(built); stats.append(source_stats)
        path = temporal_root / f"{name}.csv"
        _write_rows(path, rows, CONE_TEMPORAL_FIELDS)
        output[name] = {
            "sequence_count": len(rows), "episode_ids": list(episode_ids), "manifest": str(path),
            "manifest_sha256": sha256_file(path),
            "temporal_candidate_count": sum(item["temporal_candidates"] for item in stats),
            "rejected_boundary_count": sum(item["rejected_boundary"] for item in stats),
            "rejected_gap_count": sum(item["rejected_gap"] for item in stats),
            "adjacent_gap_s": distribution([
                gap for row in rows for gap in (
                    (int(row["timestamp_t_minus_1_ns"]) - int(row["timestamp_t_minus_2_ns"])) / 1e9,
                    (int(row["timestamp_t_ns"]) - int(row["timestamp_t_minus_1_ns"])) / 1e9,
                )
            ]),
            "sources": stats,
        }
    return output


def reanalyze_route_s_and_offline(repo: Path, sim_root: Path) -> dict[str, Any]:
    """Correct evaluation-only odom/world alignment; never retrain or rewrite model artifacts."""
    dataset_root = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/dataset"
    input_root = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/raw"
    config = _load_json(repo / "configs/cone_avoidance_dataset_v1.json")
    expert = ExpertConfig.load(repo / "configs/cone_avoidance_expert_v1.json", repo, sim_root)
    plan, _ = build_bypass_plan(expert, sim_root)
    route_diagnostics: dict[str, Any] = {}
    for episode_id in EPISODES:
        manifest = dataset_root / "manifests" / f"{episode_id}.csv"
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        bag_files = sorted((input_root / episode_id / "bag").glob("*.mcap"))
        if len(bag_files) != 1:
            raise GateFailure(f"{episode_id}: cannot reanalyze without exactly one MCAP")
        collector = _load_json(repo / "results/cone_avoidance_collection_v1" / f"{episode_id}.json")
        odom = _scan_odom_route_s(
            bag_files[0], config["odom_topic"], plan.nominal, collector["preflight"]["pose"]
        )
        diagnostics = _attach_evaluation_route_s(rows, odom)
        _write_rows(manifest, rows, CONE_MANIFEST_COLUMNS)
        route_diagnostics[episode_id] = diagnostics
    temporal = _rebuild_cone_temporal_manifests(dataset_root)
    region_counts: dict[str, Any] = {}
    regions = config["evaluation_regions_s_m"]
    for stratum in ("validation", "holdout"):
        rows = read_temporal_rows(dataset_root / "temporal_manifests" / f"{stratum}.csv")
        region_counts[stratum] = {
            name: sum(float(bounds[0]) <= float(row["route_s_m"]) < float(bounds[1]) for row in rows)
            for name, bounds in regions.items()
        }
    if any(count <= 0 for counts in region_counts.values() for count in counts.values()):
        raise GateFailure(f"evaluation route-s reanalysis still has empty fixed regions: {region_counts}")
    dataset_summary_path = repo / "results/cone_avoidance_dataset_v1/summary.json"
    dataset_summary = _load_json(dataset_summary_path)
    dataset_summary["temporal"] = temporal
    dataset_summary["evaluation_route_s_reanalysis"] = {
        "result": "PASS", "generated_utc": utc_now(),
        "method": "rigid 2D transform from first odom pose to frozen per-episode preflight world pose",
        "model_input_use": False, "region_counts": region_counts,
        "episodes": route_diagnostics,
    }
    external_metadata_path = dataset_root / "dataset_metadata.json"
    external_metadata = _load_json(external_metadata_path)
    external_metadata["temporal"] = temporal
    external_metadata["evaluation_route_s_reanalysis"] = dataset_summary["evaluation_route_s_reanalysis"]
    external_metadata_path.write_bytes(canonical_json_bytes(external_metadata))
    dataset_summary["dataset_metadata_sha256"] = sha256_file(external_metadata_path)
    write_json(dataset_summary_path, dataset_summary)

    training_path = repo / "results/pilotnet_training_c1_cone_temporal/summary.json"
    training = _load_json(training_path)
    if training.get("result") != "PASS":
        raise GateFailure("cannot reanalyze a non-PASS C1 training result")
    train_config = load_c1_training_config(repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(training["artifacts"]["checkpoint"]["path"])
    onnx_path = Path(training["artifacts"]["onnx"]["path"])
    _assert_hash(checkpoint, training["artifacts"]["checkpoint"]["sha256"], "C1 checkpoint before reanalysis")
    _assert_hash(onnx_path, training["artifacts"]["onnx"]["sha256"], "C1 ONNX before reanalysis")
    v9_summary = _load_json(repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json")
    c1_model = _load_temporal_checkpoint(checkpoint, device)
    v9_model = _load_temporal_checkpoint(Path(v9_summary["artifacts"]["checkpoint"]["path"]), device)
    v9_base = sim_root / "userdata/physicar_e2e/high_speed_temporal_v1/manifests"
    rows = {
        "nominal_validation": read_temporal_rows(v9_base / "nominal_validation.csv"),
        "nominal_holdout": read_temporal_rows(v9_base / "nominal_holdout.csv"),
        "cone_validation": read_temporal_rows(dataset_root / "temporal_manifests/validation.csv"),
        "cone_holdout": read_temporal_rows(dataset_root / "temporal_manifests/holdout.csv"),
    }
    comparisons: dict[str, Any] = {}
    for name, stratum_rows in rows.items():
        v9_pred, labels = predict_temporal(v9_model, stratum_rows, train_config, device)
        c1_pred, c1_labels = predict_temporal(c1_model, stratum_rows, train_config, device)
        if not np.array_equal(labels, c1_labels):
            raise GateFailure(f"reanalysis target mismatch for {name}")
        comparisons[name] = {"matched_count": len(stratum_rows), "v9": error_metrics(v9_pred, labels), "c1": error_metrics(c1_pred, labels)}
    obstacle_regions = {
        "cone_validation": _region_comparison(v9_model, c1_model, rows["cone_validation"], train_config, device, regions),
        "cone_holdout": _region_comparison(v9_model, c1_model, rows["cone_holdout"], train_config, device, regions),
    }
    if any(region.get("count", 0) <= 0 for strata in obstacle_regions.values() for region in strata.values()):
        raise GateFailure("offline obstacle breakdown still contains an empty region")
    v9_mae = comparisons["cone_holdout"]["v9"]["mae_rad"]
    c1_mae = comparisons["cone_holdout"]["c1"]["mae_rad"]
    improvement = (v9_mae - c1_mae) / v9_mae
    ratios = {
        name: comparisons[name]["c1"]["mae_rad"] / comparisons[name]["v9"]["mae_rad"]
        for name in ("nominal_validation", "nominal_holdout")
    }
    offline_gate = {
        "training_pass": True, "cone_holdout_nonempty": bool(rows["cone_holdout"]),
        "cone_holdout_relative_mae_improvement": improvement,
        "minimum_required_relative_improvement": train_config["minimum_relative_cone_holdout_mae_improvement_fraction"],
        "cone_material_improvement_pass": improvement >= train_config["minimum_relative_cone_holdout_mae_improvement_fraction"],
        "nominal_mae_ratios_c1_over_v9": ratios,
        "catastrophic_nominal_ratio_limit": train_config["catastrophic_nominal_mae_ratio"],
        "no_catastrophic_nominal_regression": all(value <= train_config["catastrophic_nominal_mae_ratio"] for value in ratios.values()),
    }
    offline_gate["result"] = "PASS" if offline_gate["cone_material_improvement_pass"] and offline_gate["no_catastrophic_nominal_regression"] else "FAIL"
    if offline_gate["result"] != "PASS":
        raise GateFailure("corrected offline gate failed")
    training["matched_offline_comparison"] = comparisons
    training["obstacle_regions"] = obstacle_regions
    training["offline_gate"] = offline_gate
    training["evaluation_route_s_reanalysis"] = {
        "result": "PASS", "generated_utc": utc_now(), "training_reexecuted": False,
        "checkpoint_unchanged": True, "onnx_unchanged": True,
        "dataset_region_counts": region_counts,
    }
    write_json(training_path, training)
    return {
        "result": "PASS", "training_reexecuted": False,
        "checkpoint_sha256": sha256_file(checkpoint), "onnx_sha256": sha256_file(onnx_path),
        "region_counts": region_counts, "obstacle_regions": obstacle_regions,
        "offline_gate": offline_gate,
    }


def _load_temporal_checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_temporal_pilotnet().to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _region_comparison(
    v9_model: Any, c1_model: Any, rows: Sequence[dict[str, Any]], config: dict[str, Any],
    device: torch.device, regions: dict[str, Sequence[float]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, bounds in regions.items():
        low, high = map(float, bounds)
        subset = [row for row in rows if low <= float(row["route_s_m"]) < high]
        if not subset:
            output[name] = {"count": 0}
            continue
        v9_pred, labels = predict_temporal(v9_model, subset, config, device)
        c1_pred, labels_c1 = predict_temporal(c1_model, subset, config, device)
        if not np.array_equal(labels, labels_c1):
            raise GateFailure("region comparison target mismatch")
        output[name] = {"count": len(subset), "v9": error_metrics(v9_pred, labels), "c1": error_metrics(c1_pred, labels)}
    return output


def _write_training_plots(artifact_root: Path, history: Sequence[dict[str, Any]], comparisons: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_root = artifact_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    history_path = plot_root / "training_loss.png"
    figure, axis = plt.subplots(figsize=(6, 3.4))
    axis.plot([row["epoch"] for row in history], [row["train_normalized_mse"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["validation_normalized_mse"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="normalized MSE", title="Cone Temporal PilotNet C1 training")
    axis.grid(True, linewidth=.3); axis.legend(); figure.tight_layout(); figure.savefig(history_path, dpi=140); plt.close(figure)
    comparison_path = plot_root / "offline_mae.png"
    names = list(comparisons)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(names)); width = .36
    axis.bar(x - width / 2, [comparisons[name]["v9"]["mae_rad"] for name in names], width, label="V9")
    axis.bar(x + width / 2, [comparisons[name]["c1"]["mae_rad"] for name in names], width, label="C1")
    axis.set_xticks(x, names, rotation=20, ha="right"); axis.set_ylabel("MAE (rad)"); axis.legend(); axis.grid(True, axis="y", linewidth=.3)
    figure.tight_layout(); figure.savefig(comparison_path, dpi=140); plt.close(figure)
    return [str(history_path), str(comparison_path)]


def training_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    dataset_summary = _load_json(repo / "results/cone_avoidance_dataset_v1/summary.json")
    if dataset_summary.get("result") != "PASS":
        raise GateFailure("cone dataset gate is not PASS")
    visual_gate = _load_json(repo / "results/cone_avoidance_dataset_v1/visual_quality.json")
    if visual_gate.get("result") != "PASS":
        raise GateFailure("cone dataset visual quality gate is not PASS")
    result_dir = repo / "results/pilotnet_training_c1_cone_temporal"
    result_path = result_dir / "summary.json"
    artifact_root = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/c1"
    if result_path.exists() or artifact_root.exists():
        raise FileExistsError("refusing to overwrite C1 training evidence/artifacts")
    config = load_c1_training_config(repo)
    audit = audit_frozen(repo, sim_root)
    temporal_base = sim_root / "userdata/physicar_e2e/high_speed_temporal_v1/manifests"
    cone_base = sim_root / "userdata/physicar_e2e/cone_avoidance_v1/dataset/temporal_manifests"
    rows = {
        "v9_train": read_temporal_rows(temporal_base / "train.csv"),
        "nominal_validation": read_temporal_rows(temporal_base / "nominal_validation.csv"),
        "nominal_holdout": read_temporal_rows(temporal_base / "nominal_holdout.csv"),
        "cone_train": read_temporal_rows(cone_base / "train.csv"),
        "cone_validation": read_temporal_rows(cone_base / "validation.csv"),
        "cone_holdout": read_temporal_rows(cone_base / "holdout.csv"),
    }
    train_rows = [row for name in C1_TRAIN_STRATA for row in rows[name]]
    validation_rows = [*rows["nominal_validation"], *rows["cone_validation"]]
    train_sources = {row["source_mcap_sha256"] for row in train_rows}
    eval_sources = {row["source_mcap_sha256"] for key in C1_EVAL_STRATA for row in rows[key]}
    if train_sources & eval_sources:
        raise GateFailure("C1 training source overlaps validation/holdout source")
    if sha256_file(temporal_base / "train.csv") != config["preserved_v9_train_manifest_sha256"]:
        raise GateFailure("preserved V9 train manifest changed")
    if len(rows["cone_holdout"]) == 0:
        raise GateFailure("cone holdout contains no temporal samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_count = sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters())
    if parameter_count != TEMPORAL_PARAMETER_COUNT:
        raise GateFailure("C1 architecture parameter gate failed")
    checkpoint = artifact_root / "checkpoints/pilotnet_c1_cone_temporal_best.pt"
    result_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "version": "pilotnet_training_c1_cone_temporal", "generated_utc": utc_now(), "result": "FAIL",
        "audit": audit, "visual_quality_gate": visual_gate,
        "architecture": {"input_shape": [9, 66, 200], "parameter_count": parameter_count, "identical_to_v9": True},
        "training_from_scratch": True, "no_sample_weighting": True, "no_resampling": True, "no_augmentation": True,
        "dataset": {key: len(value) for key, value in rows.items()},
        "training_composition": {"v9_train_sequences": len(rows["v9_train"]), "cone_train_sequences": len(rows["cone_train"]), "total": len(train_rows)},
        "training_source_hash_overlap_with_eval": False,
    }
    try:
        model, training, history = train_temporal(train_rows, validation_rows, config, device, checkpoint)
        report["training"] = training
        report["epochs"] = history
        v9_summary = _load_json(repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json")
        v9_checkpoint = Path(v9_summary["artifacts"]["checkpoint"]["path"])
        _assert_hash(v9_checkpoint, config["preserved_v9_checkpoint_sha256"], "V9 checkpoint")
        v9_model = _load_temporal_checkpoint(v9_checkpoint, device)
        comparisons: dict[str, Any] = {}
        for name in ("nominal_validation", "nominal_holdout", "cone_validation", "cone_holdout"):
            v9_pred, labels_v9 = predict_temporal(v9_model, rows[name], config, device)
            c1_pred, labels_c1 = predict_temporal(model, rows[name], config, device)
            if not np.array_equal(labels_v9, labels_c1):
                raise GateFailure(f"matched target identity failed for {name}")
            comparisons[name] = {
                "matched_count": len(rows[name]), "v9": error_metrics(v9_pred, labels_v9),
                "c1": error_metrics(c1_pred, labels_c1),
            }
        report["matched_offline_comparison"] = comparisons
        regions = _load_json(repo / "configs/cone_avoidance_dataset_v1.json")["evaluation_regions_s_m"]
        report["obstacle_regions"] = {
            "cone_validation": _region_comparison(v9_model, model, rows["cone_validation"], config, device, regions),
            "cone_holdout": _region_comparison(v9_model, model, rows["cone_holdout"], config, device, regions),
        }
        holdout_v9 = comparisons["cone_holdout"]["v9"]["mae_rad"]
        holdout_c1 = comparisons["cone_holdout"]["c1"]["mae_rad"]
        relative_improvement = (holdout_v9 - holdout_c1) / holdout_v9 if holdout_v9 else -math.inf
        nominal_ratios = {
            name: comparisons[name]["c1"]["mae_rad"] / comparisons[name]["v9"]["mae_rad"]
            for name in ("nominal_validation", "nominal_holdout")
        }
        gates = {
            "training_pass": training["result"] == "PASS",
            "cone_holdout_nonempty": len(rows["cone_holdout"]) > 0,
            "cone_holdout_relative_mae_improvement": relative_improvement,
            "minimum_required_relative_improvement": config["minimum_relative_cone_holdout_mae_improvement_fraction"],
            "cone_material_improvement_pass": relative_improvement >= config["minimum_relative_cone_holdout_mae_improvement_fraction"],
            "nominal_mae_ratios_c1_over_v9": nominal_ratios,
            "catastrophic_nominal_ratio_limit": config["catastrophic_nominal_mae_ratio"],
            "no_catastrophic_nominal_regression": all(value <= config["catastrophic_nominal_mae_ratio"] for value in nominal_ratios.values()),
        }
        gates["result"] = "PASS" if gates["cone_material_improvement_pass"] and gates["no_catastrophic_nominal_regression"] else "FAIL"
        report["offline_gate"] = gates
        report["plots"] = _write_training_plots(artifact_root, history, comparisons)
        if gates["result"] != "PASS":
            raise GateFailure("C1 offline gate failed; live driving and ONNX export prohibited")
        onnx_path = artifact_root / "onnx/pilotnet_c1_cone_temporal.onnx"
        export_temporal_onnx(model, onnx_path, config)
        equivalence_rows = [*rows["nominal_validation"], *rows["cone_holdout"]]
        equivalence = validate_equivalence(model, equivalence_rows, onnx_path, config)
        report["onnx_contract"] = {"checker": "PASS", "input": ["batch", 9, 66, 200], "output": ["batch", 1]}
        report["onnx_equivalence"] = equivalence
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
        }
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        write_json(result_path, report)
    return report


def load_c1_inference_config(repo: Path) -> InferenceConfig:
    payload = _load_json(repo / "configs/pilotnet_inference_c1_cone_temporal.json")
    frozen = (
        payload.get("version"), payload.get("expected_world"), payload.get("camera_only_model_observation"),
        payload.get("history_frames"), payload.get("input_channels"), payload.get("maximum_adjacent_gap_s"),
        payload.get("duplicate_frame_padding"), payload.get("warmup_while_stopped"),
        payload.get("smoke_speeds_mps"), payload.get("maximum_total_attempts"),
        payload.get("required_cone_clearance_m"), payload.get("return_maximum_absolute_nominal_cte_m"),
        payload.get("return_minimum_stable_duration_s"),
    )
    if frozen != (
        "pilotnet_inference_c1_cone_temporal", WORLD, True, 3, 9, .120, False, True,
        [1.8, 1.8, 1.8], 5, .05, .05, .50,
    ):
        raise GateFailure(f"C1 live contract changed: {frozen}")
    return InferenceConfig(payload)


def run_cone_policy(
    client: SimClient, model: TemporalOnnxModel, inference: InferenceConfig,
    initial: Any, expert: ExpertConfig, plan: Any,
) -> dict[str, Any]:
    observer = ClearanceObserver(client, initial.route, plan, expert)
    run = run_temporal_live(observer, model, inference, initial, 1.80)
    rows = observer.samples
    minimum = min(rows, key=lambda row: float(row["cone_clearance_m"]), default=None)
    avoidance = [row for row in rows if plan.departure_start_s_m <= float(row["route_s_m"]) <= plan.return_end_s_m]
    offsets = [float(row["nominal_signed_cte_m"]) * plan.side_sign for row in avoidance]
    run.update({
        "minimum_footprint_to_cone_clearance_m": None if minimum is None else minimum["cone_clearance_m"],
        "minimum_cone_clearance_route_s_m": None if minimum is None else minimum["route_s_m"],
        "footprint_cone_intersection_occurred": observer.intersection_occurred,
        "maximum_lateral_avoidance_offset_reached_m": max(offsets, default=0.0),
        "recovery_success": observer.recovery_success, "recovery_cte_m": observer.recovery_cte_m,
        "recovery_time_s": observer.recovery_time_s,
        "nominal_route_used_for_progress": True,
        "privileged_metrics_only": ["pose", "route", "cte", "cone_pose", "footprint_clearance", "recovery"],
        "model_observation_only": ["camera_yuv_t_minus_2", "camera_yuv_t_minus_1", "camera_yuv_t"],
        "pose_failures": run.get("liveness_failures", 0), "clock_failures": run.get("liveness_failures", 0),
    })
    run["classification"] = classify_cone_policy_run(run)
    return run


def classify_cone_policy_run(run: dict[str, Any]) -> str:
    if run.get("temporal_input_failure"):
        return "TEMPORAL_INPUT_FAIL"
    if run.get("api_failures") or run.get("liveness_failures") or not run.get("safe_stop_success", False):
        return "INFRA_FAIL"
    clearance = run.get("minimum_footprint_to_cone_clearance_m")
    if (
        run.get("result") == "PASS" and clearance is not None and float(clearance) >= REQUIRED_CLEARANCE_M
        and run.get("footprint_cone_intersection_occurred") is False
        and run.get("recovery_success") is True
    ):
        return "CONE_POLICY_PASS"
    return "CONE_POLICY_FAIL"


def run_c1_attempts(
    client: SimClient, model: TemporalOnnxModel, inference: InferenceConfig, expert: ExpertConfig,
    plan: Any, geometry: dict[str, Any], sim_root: Path, result_dir: Path,
    *, preflight_one: Callable[..., tuple[Any, dict[str, Any]]] = full_preflight,
    run_one: Callable[..., dict[str, Any]] = run_cone_policy,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    valid = 0
    for number in range(1, MAXIMUM_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight = preflight_one(client, expert, plan, sim_root, geometry)
            run = run_one(client, model, inference, initial, expert, plan)
            classification = str(run["classification"])
            if classification in {"CONE_POLICY_PASS", "CONE_POLICY_FAIL"}:
                valid += 1
            attempt = {
                "attempt_number": number, "valid_policy_run_number": valid if classification in {"CONE_POLICY_PASS", "CONE_POLICY_FAIL"} else None,
                "classification": classification, "preflight": preflight, "run": run,
            }
        except Exception as exc:
            errors = client.safe_stop()
            attempt = {
                "attempt_number": number, "valid_policy_run_number": None, "classification": "INFRA_FAIL", "run": None,
                "preflight": {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}", "safe_stop_success": not errors, "safe_stop_errors": errors},
            }
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "CONE_POLICY_FAIL":
            return attempts, "FAIL"
        if attempt["classification"] == "CONE_POLICY_PASS" and valid == TARGET_VALID_POLICY_RUNS:
            return attempts, "PASS"
    return attempts, "INCONCLUSIVE"


def aggregate_live(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    runs = [item["run"] for item in attempts if item["classification"] == "CONE_POLICY_PASS"]
    if len(runs) != 3:
        raise GateFailure("live aggregate requires exactly three C1 passes")
    laps = [float(run["elapsed_s"]) for run in runs]
    clearances = [float(run["minimum_footprint_to_cone_clearance_m"]) for run in runs]
    ctes = [float(run["mean_cte_m"]) for run in runs]
    saturations = [float(run["steering_saturation_fraction"]) for run in runs]
    return {
        "success": "3/3", "lap_time_mean_s": statistics.fmean(laps), "lap_time_sample_std_s": statistics.stdev(laps),
        "minimum_cone_clearance_m": min(clearances), "cone_clearance_mean_m": statistics.fmean(clearances),
        "cone_clearance_range_m": [min(clearances), max(clearances)], "recovery_success": "3/3",
        "nominal_mean_cte_m": statistics.fmean(ctes), "worst_max_cte_m": max(float(run["max_cte_m"]) for run in runs),
        "steering_saturation_mean": statistics.fmean(saturations),
        "onnx_latency_mean_ms": statistics.fmean(float(run["onnx_inference_latency"]["mean_ms"]) for run in runs),
        "camera_latency_mean_ms": statistics.fmean(float(run["camera_acquisition_latency"]["mean_ms"]) for run in runs),
        "safe_stop": "3/3",
    }


def live_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    training = _load_json(repo / "results/pilotnet_training_c1_cone_temporal/summary.json")
    if training.get("result") != "PASS" or training.get("offline_gate", {}).get("result") != "PASS" or training.get("onnx_equivalence", {}).get("result") != "PASS":
        raise GateFailure("C1 training/offline/ONNX gate is not PASS")
    result_dir = repo / "results/pilotnet_e2e_c1_cone_temporal"
    summary_path = result_dir / "summary.json"
    marker = result_dir / "experiment.started.json"
    if summary_path.exists() or marker.exists():
        raise FileExistsError("refusing to repeat C1 bounded live experiment")
    onnx_path = Path(training["artifacts"]["onnx"]["path"])
    _assert_hash(onnx_path, training["artifacts"]["onnx"]["sha256"], "C1 ONNX")
    audit = audit_frozen(repo, sim_root)
    expert = ExpertConfig.load(repo / "configs/cone_avoidance_expert_v1.json", repo, sim_root)
    plan, route_data = build_bypass_plan(expert, sim_root)
    geometry = validate_geometry(expert, plan, route_data)
    inference = load_c1_inference_config(repo)
    model = TemporalOnnxModel(onnx_path)
    client = SimClient(inference.payload["base_url"], inference.payload["api_timeout_s"])
    result_dir.mkdir(parents=True, exist_ok=False)
    write_json(marker, {
        "status": "C1_CONE_POLICY_STARTED_DO_NOT_REPEAT", "started_utc": utc_now(),
        "maximum_total_attempts": 5, "maximum_valid_policy_runs": 3,
    })
    report: dict[str, Any] = {
        "version": "pilotnet_e2e_c1_cone_temporal", "generated_utc": utc_now(), "result": "INCONCLUSIVE",
        "audit": audit, "onnx": training["artifacts"]["onnx"], "camera_only_observation": True,
    }
    try:
        activate_world(client, WORLD)
        if errors := client.safe_stop():
            raise GateFailure("initial live safe stop failed: " + "; ".join(errors))
        initial, preflight = full_preflight(client, expert, plan, sim_root, geometry)
        _, buffer_check = warm_temporal_buffer(client, inference)
        report["temporal_live_preflight"] = {
            "result": "PASS", "world": initial.world, "environment": preflight,
            "buffer": buffer_check, "model_observation_fields": list(model.observation_fields),
        }
        report["attempts"], report["result"] = run_c1_attempts(
            client, model, inference, expert, plan, geometry, sim_root, result_dir
        )
        report["total_attempts"] = len(report["attempts"])
        report["valid_policy_runs"] = sum(item["classification"] in {"CONE_POLICY_PASS", "CONE_POLICY_FAIL"} for item in report["attempts"])
        report["aggregate"] = aggregate_live(report["attempts"]) if report["result"] == "PASS" else None
        report["c1_frozen"] = report["result"] == "PASS"
        report["multi_location_work_justified"] = report["result"] == "PASS"
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if errors:
            report["result"] = "INCONCLUSIVE"
        write_json(summary_path, report)
    return report


def simulator_tracked_status(sim_root: Path) -> dict[str, Any]:
    status = subprocess.run(["git", "status", "--short"], cwd=sim_root, text=True, capture_output=True, check=True).stdout.splitlines()
    diff = subprocess.run(["git", "diff", "--name-only"], cwd=sim_root, text=True, capture_output=True, check=True).stdout.splitlines()
    source = [path for path in diff if path != "userdata/last_world"]
    return {"status_short": status, "tracked_diff_paths": diff, "tracked_source_changes": source, "result": "PASS" if not source else "FAIL"}


def pipeline(repo: Path, sim_root: Path) -> dict[str, Any]:
    audit = audit_frozen(repo, sim_root)
    collection = collection_stage(repo, sim_root)
    dataset = dataset_stage(repo, sim_root)
    training = training_stage(repo, sim_root)
    live = live_stage(repo, sim_root)
    return {"audit": audit, "collection": collection["result"], "dataset": dataset["result"], "training": training["result"], "live": live["result"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "collect", "dataset", "train", "reanalyze", "live", "pipeline"), required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    sim_root = args.sim_root.expanduser().resolve()
    functions = {
        "audit": audit_frozen, "collect": collection_stage, "dataset": dataset_stage,
        "train": training_stage, "reanalyze": reanalyze_route_s_and_offline,
        "live": live_stage, "pipeline": pipeline,
    }
    try:
        result = functions[args.stage](repo, sim_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.stage == "live":
            return 0 if result.get("result") == "PASS" else 1
        return 0
    except Exception as exc:
        print(json.dumps({"result": "FAIL", "stage": args.stage, "failure": f"{type(exc).__name__}: {exc}"}, indent=2), flush=True)
        return 1 if isinstance(exc, GateFailure) else 2


if __name__ == "__main__":
    raise SystemExit(main())
