"""Targeted, deterministic Recovery Data V1 collection and extraction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import time
from typing import Any, Sequence

import numpy as np

from .dataset_extractor import (
    _directory_size,
    canonical_json_bytes,
    extract_episode,
    load_config as load_extractor_config,
    prepare_output_root,
    steering_distribution,
    write_manifest,
)
from .expert_driver import DriverConfig, PoseLivenessMonitor, Preflight, wait_after_reset
from .route_geometry import ClosedRoute, OffTrackMonitor, ProgressTracker, pure_pursuit_steering
from .rosbag_collector import (
    CollectorConfig,
    DockerRosBackend,
    directory_size,
    git_commit,
    sha256_file,
    utc_now,
    verify_bag,
    verify_environment,
)
from .sim_client import SimClient


class RecoveryGateFailure(RuntimeError):
    """A recovery experiment gate failed; later stages must not run."""


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def circular_distance(a: float, b: float, length: float) -> float:
    delta = abs((a - b) % length)
    return min(delta, length - delta)


def tangent_at(route: ClosedRoute, s: float, delta_m: float = 0.05) -> float:
    before = route.point_at(s - delta_m)
    after = route.point_at(s + delta_m)
    return math.atan2(after[1] - before[1], after[0] - before[0])


def curvature_at(route: ClosedRoute, s: float, delta_m: float) -> float:
    before = tangent_at(route, s - delta_m, delta_m / 3.0)
    after = tangent_at(route, s + delta_m, delta_m / 3.0)
    return wrap_angle(after - before) / (2.0 * delta_m)


def _distance_to_segment(point, start, end) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    denominator = vx * vx + vy * vy
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / denominator))
    return math.dist(point, (start[0] + fraction * vx, start[1] + fraction * vy))


def boundary_clearance(route: ClosedRoute, point: tuple[float, float]) -> float:
    if route.inner is None or route.outer is None:
        raise RecoveryGateFailure("route has no validated inner/outer boundaries")
    boundaries = (route.inner, route.outer)
    return min(
        _distance_to_segment(point, boundary[index], boundary[(index + 1) % len(boundary)])
        for boundary in boundaries for index in range(len(boundary))
    )


@dataclass(frozen=True)
class Anchor:
    role: str
    s_m: float
    x: float
    y: float
    yaw_rad: float
    curvature_per_m: float


@dataclass(frozen=True)
class RecoveryEpisode:
    episode_id: str
    anchor_role: str
    anchor_s_m: float
    lateral_offset_m: float
    heading_offset_deg: float
    x: float
    y: float
    yaw_rad: float
    boundary_clearance_m: float

    @property
    def perturbation_type(self) -> str:
        if self.lateral_offset_m > 0:
            return "lateral_positive"
        if self.lateral_offset_m < 0:
            return "lateral_negative"
        if self.heading_offset_deg > 0:
            return "heading_positive"
        return "heading_negative"


def select_anchors(route: ClosedRoute, config: dict[str, Any]) -> list[Anchor]:
    failure_s = float(config["failure_anchor_s_m"]) % route.length
    step = float(config["anchor_curvature_sample_step_m"])
    delta = float(config["anchor_curvature_delta_m"])
    separation = float(config["minimum_anchor_separation_m"])
    samples = np.arange(0.0, route.length, step, dtype=np.float64)
    curvatures = [curvature_at(route, float(s), delta) for s in samples]
    peaks: list[tuple[float, float, float]] = []
    for index, value in enumerate(curvatures):
        magnitude = abs(value)
        if magnitude >= abs(curvatures[index - 1]) and magnitude > abs(curvatures[(index + 1) % len(curvatures)]):
            peaks.append((magnitude, float(samples[index]), value))
    peaks.sort(reverse=True)
    selected: list[tuple[float, float]] = []
    for _, s_m, curvature in peaks:
        if circular_distance(s_m, failure_s, route.length) < separation:
            continue
        if any(circular_distance(s_m, prior, route.length) < separation for prior, _ in selected):
            continue
        selected.append((s_m, curvature))
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise RecoveryGateFailure("could not select two curvature peaks with configured separation")

    failure = Anchor(
        "failure", failure_s, *route.point_at(failure_s), tangent_at(route, failure_s),
        curvature_at(route, failure_s, delta),
    )
    selected.sort(key=lambda item: circular_distance(item[0], failure_s, route.length))
    roles = ("curvature_near", "curvature_far")
    anchors = [failure]
    for role, (s_m, curvature) in zip(roles, selected):
        anchors.append(Anchor(role, s_m, *route.point_at(s_m), tangent_at(route, s_m), curvature))
    return anchors


def episode_matrix(route: ClosedRoute, anchors: Sequence[Anchor], config: dict[str, Any]) -> list[RecoveryEpisode]:
    lateral = list(config["lateral_offsets_m"])
    headings = list(config["heading_offsets_deg"])
    if lateral != [0.10, -0.10, 0.0, 0.0] or headings != [0.0, 0.0, 6.0, -6.0]:
        raise RecoveryGateFailure("Recovery V1 perturbation matrix changed")
    suffixes = ("lat_p10", "lat_m10", "yaw_p06", "yaw_m06")
    episodes: list[RecoveryEpisode] = []
    clearance_limit = float(config["minimum_boundary_clearance_m"])
    for anchor in anchors:
        normal = (-math.sin(anchor.yaw_rad), math.cos(anchor.yaw_rad))
        for offset, heading_deg, suffix in zip(lateral, headings, suffixes):
            point = (anchor.x + offset * normal[0], anchor.y + offset * normal[1])
            yaw = wrap_angle(anchor.yaw_rad + math.radians(heading_deg))
            track_distance = route.track_boundary_distance(point)
            clearance = boundary_clearance(route, point)
            if track_distance is None or track_distance > 1e-9 or clearance < clearance_limit:
                raise RecoveryGateFailure(
                    f"unsafe perturbation {anchor.role}/{suffix}: point={point}, "
                    f"track_distance={track_distance}, boundary_clearance={clearance:.6f}m "
                    f"(<{clearance_limit:.6f}m)"
                )
            episodes.append(RecoveryEpisode(
                f"recovery_{anchor.role}_{suffix}", anchor.role, anchor.s_m,
                offset, heading_deg, point[0], point[1], yaw, clearance,
            ))
    if len(episodes) != 12 or len({item.episode_id for item in episodes}) != 12:
        raise RecoveryGateFailure("recovery episode matrix must contain exactly 12 unique episodes")
    return episodes


def route_from_npy(path: Path) -> ClosedRoute:
    values = np.load(path)
    if values.ndim != 2 or values.shape[1] != 6:
        raise RecoveryGateFailure(f"route array must be Nx6, got {values.shape}")
    return ClosedRoute(values[:, :2].tolist(), values[:, 2:4].tolist(), values[:, 4:6].tolist())


def build_plan(route: ClosedRoute, config: dict[str, Any], route_source: str) -> dict[str, Any]:
    anchors = select_anchors(route, config)
    episodes = episode_matrix(route, anchors, config)
    return {
        "version": "recovery_data_v1_plan",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "route_source": route_source,
        "route_length_m": route.length,
        # The API/Numpy route contains a repeated closing point; ClosedRoute removes it.
        "route_point_count": len(route.points) + 1,
        "sign_convention": {
            "left_normal": "[-sin(route_yaw), cos(route_yaw)]",
            "lateral": "p_perturbed = p_route + d * left_normal; positive d is left",
            "heading": "yaw_perturbed = route_yaw + delta_yaw; positive is counter-clockwise",
        },
        "holdout_policy_frozen_before_training": "curvature_far; the additional anchor with greatest circular arc distance from failure",
        "anchors": [asdict(item) for item in anchors],
        "episodes": [{**asdict(item), "perturbation_type": item.perturbation_type} for item in episodes],
    }


class RecoveryCompletionGate:
    def __init__(self, cte_m: float, heading_rad: float, hold_s: float, minimum_progress_m: float) -> None:
        self.cte_m = cte_m
        self.heading_rad = heading_rad
        self.hold_s = hold_s
        self.minimum_progress_m = minimum_progress_m
        self.converged_since: float | None = None

    def update(self, *, abs_cte_m: float, abs_heading_rad: float, progress_m: float, now: float) -> bool:
        converged = abs_cte_m <= self.cte_m and abs_heading_rad <= self.heading_rad
        if not converged:
            self.converged_since = None
            return False
        if self.converged_since is None:
            self.converged_since = now
        return progress_m >= self.minimum_progress_m and now - self.converged_since >= self.hold_s


def check_recovery_limits(elapsed_s: float, progress_m: float, maximum_duration_s: float, maximum_progress_m: float) -> None:
    if elapsed_s >= maximum_duration_s:
        raise RecoveryGateFailure("recovery timeout")
    if progress_m > maximum_progress_m:
        raise RecoveryGateFailure("maximum recovery progress exceeded without convergence")


def verify_requested_pose(actual: dict[str, Any], episode: RecoveryEpisode, config: dict[str, Any]) -> dict[str, float]:
    position_error = math.dist((float(actual["x"]), float(actual["y"])), (episode.x, episode.y))
    yaw_error = abs(wrap_angle(float(actual["yaw"]) - episode.yaw_rad))
    if position_error > float(config["pose_position_tolerance_m"]) or yaw_error > float(config["pose_yaw_tolerance_rad"]):
        raise RecoveryGateFailure(
            f"pose verification failed for {episode.episode_id}: position_error={position_error:.6f}, yaw_error={yaw_error:.6f}"
        )
    return {"position_error_m": position_error, "yaw_error_rad": yaw_error}


def run_recovery_expert(
    client: SimClient, expert: DriverConfig, initial: Preflight,
    episode: RecoveryEpisode, config: dict[str, Any],
) -> dict[str, Any]:
    route = initial.route
    tracker = ProgressTracker(route.length, float(config["maximum_progress_jump_m"]))
    off_track = OffTrackMonitor(float(config["off_track_grace_s"]))
    liveness = PoseLivenessMonitor(
        float(config["pose_stale_timeout_s"]),
        float(config["pose_motion_translation_threshold_m"]),
        float(config["pose_motion_yaw_threshold_rad"]),
    )
    completion = RecoveryCompletionGate(
        float(config["recovery_cte_threshold_m"]), float(config["recovery_heading_threshold_rad"]),
        float(config["recovery_hold_duration_s"]), float(config["minimum_recovery_progress_m"]),
    )
    started = time.monotonic()
    next_tick = started
    next_world_check = started
    previous_tick: float | None = None
    periods: list[float] = []
    ctes: list[float] = []
    headings: list[float] = []
    steerings: list[float] = []
    motion_commanded = False
    failure: str | None = None
    result = "FAIL"
    liveness_failures = 0
    api_failures = 0
    final_projection = route.project((episode.x, episode.y))
    tracker.update(final_projection.s)
    final_pose = initial.pose
    try:
        while True:
            now = time.monotonic()
            check_recovery_limits(
                now - started, tracker.unwrapped, float(config["maximum_recovery_duration_s"]),
                float(config["maximum_recovery_progress_m"]),
            )
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            if previous_tick is not None:
                periods.append(tick - previous_tick)
            previous_tick = tick
            if tick >= next_world_check:
                status = client.status()
                if status.get("running") is not True or status.get("switching") is not False or status.get("current") != initial.world:
                    raise RecoveryGateFailure(f"simulator world/state changed: {status}")
                next_world_check = tick + float(config["world_check_interval_s"])
            pose = client.pose()
            clock = client.clock()
            try:
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=motion_commanded)
            except RuntimeError:
                liveness_failures += 1
                raise
            final_pose = pose
            final_projection = route.project((pose["x"], pose["y"]))
            progress = tracker.update(final_projection.s)
            check_recovery_limits(
                tick - started, progress, float(config["maximum_recovery_duration_s"]),
                float(config["maximum_recovery_progress_m"]),
            )
            boundary = route.track_boundary_distance((pose["x"], pose["y"]))
            if boundary is None or not math.isfinite(boundary):
                raise RecoveryGateFailure("invalid track boundary geometry")
            if off_track.update(boundary > float(config["off_track_margin_m"]), time.monotonic()):
                raise RecoveryGateFailure(f"sustained off-track during recovery: {boundary:.3f}m")
            route_yaw = tangent_at(route, final_projection.s)
            heading_error = wrap_angle(float(pose["yaw"]) - route_yaw)
            target = route.point_at(final_projection.s + expert.lookahead_m)
            steering, _, _ = pure_pursuit_steering(
                (pose["x"], pose["y"]), float(pose["yaw"]), target,
                expert.wheelbase_m, expert.max_steering_rad,
            )
            client.command_steering(steering)
            client.command_speed(expert.fixed_speed_mps)
            if not motion_commanded:
                motion_commanded = True
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=True)
            ctes.append(final_projection.signed_error)
            headings.append(heading_error)
            steerings.append(steering)
            if completion.update(
                abs_cte_m=abs(final_projection.signed_error), abs_heading_rad=abs(heading_error),
                progress_m=progress, now=tick,
            ):
                result = "PASS"
                break
            next_tick += 1.0 / float(config["control_frequency_hz"])
    except Exception as exc:
        failure = str(exc)
        if any(token in failure.lower() for token in ("get ", "post ", "unavailable", "control rejected")):
            api_failures += 1
    finally:
        ended = time.monotonic()
        off_track.finalize(ended)
        stop_errors = client.safe_stop()
        if stop_errors:
            result = "FAIL"
            failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
            api_failures += len(stop_errors)
    progress = tracker.unwrapped
    return {
        "result": result, "failure": failure, "elapsed_s": time.monotonic() - started,
        "recovery_progress_m": progress, "control_iterations": len(steerings),
        "initial_lateral_offset_m": episode.lateral_offset_m,
        "initial_yaw_offset_deg": episode.heading_offset_deg,
        "starting_cte_m": ctes[0] if ctes else None,
        "starting_heading_error_rad": headings[0] if headings else None,
        "final_cte_m": ctes[-1] if ctes else None,
        "final_heading_error_rad": headings[-1] if headings else None,
        "mean_absolute_cte_m": statistics.fmean(abs(value) for value in ctes) if ctes else None,
        "max_absolute_cte_m": max((abs(value) for value in ctes), default=None),
        "mean_absolute_steering_rad": statistics.fmean(abs(value) for value in steerings) if steerings else None,
        "max_absolute_steering_rad": max((abs(value) for value in steerings), default=None),
        "mean_loop_period_s": statistics.fmean(periods) if periods else 0.0,
        "off_track_events": off_track.event_count, "api_failures": api_failures,
        "liveness_failures": liveness_failures, "safe_stop_success": not stop_errors,
        "safe_stop_errors": stop_errors, "final_pose": final_pose,
    }


def collector_config(config: dict[str, Any]) -> CollectorConfig:
    return CollectorConfig(
        expected_world=config["expected_world"], required_topics=tuple(config["required_topics"]),
        container_name=config["container_name"], compose_service=config["compose_service"],
        container_userdata_root=config["container_userdata_root"], data_relative_root=config["data_relative_root"],
        storage_id=config["storage_id"], recorder_startup_timeout_s=config["recorder_startup_timeout_s"],
        recorder_shutdown_timeout_s=config["recorder_shutdown_timeout_s"], settle_duration_s=config["settle_duration_s"],
        pilot_episode_count=12, minimum_free_bytes=config["minimum_free_bytes"],
        minimum_camera_messages=config["minimum_camera_messages"],
    )


def collect_one(
    episode: RecoveryEpisode, config: dict[str, Any], expert: DriverConfig, expert_path: Path,
    source_commit: str, backend: DockerRosBackend, client: SimClient, static: Preflight, result_path: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "version": "recovery_data_v1", "episode_id": episode.episode_id,
        "episode_definition": {**asdict(episode), "perturbation_type": episode.perturbation_type},
        "world": config["expected_world"], "canonical_expert_config_path": str(expert_path),
        "canonical_expert_config_sha256": sha256_file(expert_path),
        "canonical_expert_config": asdict(expert), "physicar_e2e_git_commit": source_commit,
        "recording_start_utc": None, "expert_driving_start_utc": None,
        "expert_driving_end_utc": None, "recording_end_utc": None,
        "expert_result_metrics": None, "bag_size_bytes": None, "bag_duration_s": None,
        "actual_topic_message_counts": {}, "recorder_graceful_shutdown": False,
        "recorder_orphaned": False, "safe_stop_success": False, "result": "FAIL", "failure_reason": None,
    }
    handle = None
    stop_result = None
    try:
        stop = client.safe_stop()
        if stop:
            raise RecoveryGateFailure("pre-teleport safe-stop failed: " + "; ".join(stop))
        client.set_pose(episode.x, episode.y, episode.yaw_rad)
        time.sleep(float(config["settle_duration_s"]))
        pose = client.pose()
        metadata["pose_verification"] = verify_requested_pose(pose, episode, config)
        status = client.status()
        if status.get("current") != config["expected_world"] or status.get("running") is not True or status.get("switching") is not False:
            raise RecoveryGateFailure(f"unexpected simulator state after teleport: {status}")
        objects = client.objects()
        cones = sum(str(item.get("name", "")).lower().startswith("cone") for item in objects.get("objects", []))
        if cones:
            raise RecoveryGateFailure(f"found {cones} cones after teleport")
        if shutil.disk_usage(backend.host_userdata_root).free < int(config["minimum_free_bytes"]):
            raise RecoveryGateFailure("insufficient external userdata free space")
        handle = backend.start_recorder(episode.episode_id, tuple(config["required_topics"]))
        metadata["bag_host_path"] = str(handle.host_bag_path)
        metadata["bag_container_path"] = handle.container_bag_path
        metadata["recording_start_utc"] = utc_now()
        metadata["expert_driving_start_utc"] = utc_now()
        run_initial = Preflight(static.world, static.route, static.route_points, static.cone_count, static.bounds, pose)
        metrics = run_recovery_expert(client, expert, run_initial, episode, config)
        metadata["expert_result_metrics"] = metrics
        metadata["expert_driving_end_utc"] = utc_now()
        if metrics["result"] != "PASS":
            raise RecoveryGateFailure(f"canonical expert recovery failed: {metrics.get('failure')}")
    except BaseException as exc:
        metadata["failure_reason"] = str(exc)
    finally:
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
                if not stop_result.graceful:
                    metadata["failure_reason"] = "; ".join(filter(None, [metadata["failure_reason"], stop_result.detail]))
            except BaseException as exc:
                metadata["recorder_orphaned"] = True
                metadata["failure_reason"] = "; ".join(filter(None, [metadata["failure_reason"], str(exc)]))
            metadata["recording_end_utc"] = utc_now()
        stop_errors = client.safe_stop()
        metadata["safe_stop_success"] = not stop_errors
        if stop_errors:
            metadata["failure_reason"] = "; ".join(filter(None, [metadata["failure_reason"], "; ".join(stop_errors)]))
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            info = backend.bag_info(handle)
            verify_bag(info, tuple(config["required_topics"]), int(config["minimum_camera_messages"]))
            metadata["bag_duration_s"] = info.duration_s
            metadata["bag_size_bytes"] = directory_size(handle.host_bag_path)
            metadata["actual_topic_message_counts"] = dict(sorted(info.topic_counts.items()))
        except BaseException as exc:
            metadata["failure_reason"] = "; ".join(filter(None, [metadata["failure_reason"], f"bag integrity failed: {exc}"]))
    if (
        metadata["failure_reason"] is None and metadata["expert_result_metrics"]["result"] == "PASS"
        and metadata["recorder_graceful_shutdown"] and not metadata["recorder_orphaned"]
        and metadata["safe_stop_success"] and metadata["bag_size_bytes"] is not None
    ):
        metadata["result"] = "PASS"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_collection(
    *, config: dict[str, Any], plan: dict[str, Any], expert_path: Path,
    sim_root: Path, results_dir: Path,
) -> dict[str, Any]:
    expert = DriverConfig.load(expert_path)
    if expert.fixed_speed_mps != 0.5 or expert.expected_world != config["expected_world"]:
        raise RecoveryGateFailure("canonical expert speed/world contract changed")
    cconfig = collector_config(config)
    verify_environment(Path(__file__).resolve().parents[2], sim_root)
    backend = DockerRosBackend(cconfig, sim_root)
    topics = backend.preflight(cconfig.required_topics)
    client = SimClient(config["base_url"], config["api_timeout_s"])
    initial_stop = client.safe_stop()
    if initial_stop:
        raise RecoveryGateFailure("initial safe-stop failed: " + "; ".join(initial_stop))
    static = wait_after_reset(client, expert, False)
    if abs(static.route.length - float(plan["route_length_m"])) > 1e-6 or static.route_points != plan["route_point_count"]:
        raise RecoveryGateFailure("live route does not match frozen anchor plan")
    episodes = [RecoveryEpisode(**{key: item[key] for key in RecoveryEpisode.__dataclass_fields__}) for item in plan["episodes"]]
    # Revalidate all 12 candidates against the live route before the first teleport.
    live_anchors = [Anchor(**item) for item in plan["anchors"]]
    live_matrix = episode_matrix(static.route, live_anchors, config)
    for frozen, live in zip(episodes, live_matrix):
        if (
            frozen.episode_id != live.episode_id
            or abs(frozen.x - live.x) > 2e-5 or abs(frozen.y - live.y) > 2e-5
            or abs(wrap_angle(frozen.yaw_rad - live.yaw_rad)) > 2e-5
        ):
            raise RecoveryGateFailure("live geometry does not reproduce frozen perturbation matrix")
    results: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            item = collect_one(
                episode, config, expert, expert_path, git_commit(Path(__file__).resolve().parents[2]),
                backend, client, static, results_dir / f"{episode.episode_id}.json",
            )
            results.append(item)
            print(json.dumps({
                "episode_id": episode.episode_id, "result": item["result"],
                "recovery_time_s": (item.get("expert_result_metrics") or {}).get("elapsed_s"),
                "recovery_progress_m": (item.get("expert_result_metrics") or {}).get("recovery_progress_m"),
            }), flush=True)
            if item["result"] != "PASS":
                break
    finally:
        final_stop = client.safe_stop()
    sizes = [item["bag_size_bytes"] for item in results if item.get("bag_size_bytes")]
    summary = {
        "version": "recovery_data_v1", "generated_utc": utc_now(),
        "requested_episode_count": 12, "completed_episode_count": len(results),
        "passed_episode_count": sum(item["result"] == "PASS" for item in results),
        "result": "PASS" if len(results) == 12 and all(item["result"] == "PASS" for item in results) and not final_stop else "FAIL",
        "failure_reason": next((item["failure_reason"] for item in results if item["result"] != "PASS"), None),
        "episodes": [{
            "episode_id": item["episode_id"], "result": item["result"],
            "recovery_time_s": (item.get("expert_result_metrics") or {}).get("elapsed_s"),
            "recovery_progress_m": (item.get("expert_result_metrics") or {}).get("recovery_progress_m"),
            "bag_size_bytes": item.get("bag_size_bytes"),
        } for item in results],
        "total_raw_bag_size_bytes": sum(sizes),
        "mean_raw_bag_size_bytes": statistics.fmean(sizes) if sizes else None,
        "all_bags_finalized": len(results) == 12 and all(item["recorder_graceful_shutdown"] for item in results),
        "all_safe_stops": len(results) == 12 and all(item["safe_stop_success"] for item in results) and not final_stop,
        "topic_types": topics, "final_safe_stop_errors": final_stop,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def run_live_preflight(
    *, config: dict[str, Any], plan: dict[str, Any], expert_path: Path, sim_root: Path,
) -> dict[str, Any]:
    expert = DriverConfig.load(expert_path)
    cconfig = collector_config(config)
    environment = verify_environment(Path(__file__).resolve().parents[2], sim_root)
    backend = DockerRosBackend(cconfig, sim_root)
    topics = backend.preflight(cconfig.required_topics)
    client = SimClient(config["base_url"], config["api_timeout_s"])
    try:
        stop = client.safe_stop()
        if stop:
            raise RecoveryGateFailure("preflight safe-stop failed: " + "; ".join(stop))
        static = wait_after_reset(client, expert, False)
        if abs(static.route.length - float(plan["route_length_m"])) > 1e-6 or static.route_points != plan["route_point_count"]:
            raise RecoveryGateFailure("live route does not match frozen plan")
        anchors = [Anchor(**item) for item in plan["anchors"]]
        live = episode_matrix(static.route, anchors, config)
        frozen = [RecoveryEpisode(**{key: item[key] for key in RecoveryEpisode.__dataclass_fields__}) for item in plan["episodes"]]
        for expected, actual in zip(frozen, live):
            if (
                expected.episode_id != actual.episode_id
                or abs(expected.x - actual.x) > 2e-5 or abs(expected.y - actual.y) > 2e-5
                or abs(wrap_angle(expected.yaw_rad - actual.yaw_rad)) > 2e-5
            ):
                raise RecoveryGateFailure("live route does not reproduce frozen perturbations")
        return {
            "version": "recovery_data_v1_preflight", "generated_utc": utc_now(), "result": "PASS",
            "world": static.world, "route_length_m": static.route.length, "route_points": static.route_points,
            "cone_count": static.cone_count, "validated_perturbation_count": len(live),
            "minimum_boundary_clearance_m": min(item.boundary_clearance_m for item in live),
            "topic_types": topics, "host_userdata_root": str(backend.host_userdata_root),
            "environment_verification": environment,
        }
    finally:
        client.safe_stop()


def run_fail_fast_sequence(items: Sequence[Any], operation) -> list[Any]:
    """Run ordered experiment items once each and stop immediately on first FAIL."""
    results = []
    for item in items:
        result = operation(item)
        results.append(result)
        if result.get("result") != "PASS":
            break
    return results


def run_extraction_recovery(
    *, config: dict[str, Any], extractor_config_path: Path, raw_root: Path,
    output_root: Path, results_dir: Path, plan: dict[str, Any],
) -> dict[str, Any]:
    extractor = load_extractor_config(extractor_config_path)
    extractor_sha = hashlib.sha256(canonical_json_bytes(extractor)).hexdigest()
    prepare_output_root(output_root, False)
    all_rows: list[dict[str, Any]] = []
    episode_metrics: list[dict[str, Any]] = []
    try:
        for definition in plan["episodes"]:
            episode_id = definition["episode_id"]
            mcap_files = sorted((raw_root / episode_id / "bag").glob("*.mcap"))
            if len(mcap_files) != 1:
                raise RecoveryGateFailure(f"{episode_id}: expected one MCAP, found {len(mcap_files)}")
            collector_path = results_dir / f"{episode_id}.json"
            metrics, rows = extract_episode(
                episode_id=episode_id, mcap_path=mcap_files[0], collector_metadata_path=collector_path,
                dataset_root=output_root, config=extractor, config_sha256=extractor_sha,
                source_path_identity=mcap_files[0].relative_to(raw_root).as_posix(),
                collector_metadata_identity=collector_path.as_posix(),
            )
            retention = metrics["counts"]["active_window_retention_fraction"]
            if retention < float(config["minimum_extraction_retention_fraction"]):
                raise RecoveryGateFailure(f"{episode_id}: extraction retention {retention:.6f} is materially low")
            collector = json.loads(collector_path.read_text(encoding="utf-8"))
            metrics["recovery"] = {
                **collector["episode_definition"],
                "recovery_time_s": collector["expert_result_metrics"]["elapsed_s"],
                "recovery_progress_m": collector["expert_result_metrics"]["recovery_progress_m"],
                "starting_cte_m": collector["expert_result_metrics"]["starting_cte_m"],
                "starting_heading_error_rad": collector["expert_result_metrics"]["starting_heading_error_rad"],
            }
            episode_metrics.append(metrics)
            all_rows.extend(rows)
        write_manifest(output_root / "manifest.csv", all_rows)
        steering = [float(row["steering_rad"]) for row in all_rows]
        raw_bytes = sum(item["source"]["mcap_size_bytes"] for item in episode_metrics)
        metadata = {
            "version": "recovery_dataset_v1", "generated_utc": utc_now(), "result": "PASS",
            "episode_count": len(episode_metrics), "accepted_sample_count": len(all_rows),
            "future_label_violations": sum(item["synchronization"]["future_label_violations"] for item in episode_metrics),
            "synchronization_rule": "causal zero-order hold on MCAP record timestamps",
            "extractor_config_sha256": extractor_sha,
            "steering_distribution": steering_distribution(steering, extractor),
            "episodes": episode_metrics,
            "storage": {"raw_bag_size_bytes": raw_bytes, "extracted_size_bytes": 0},
            "preview_gate": {"contact_sheets_generated": len(episode_metrics), "visual_verification": "PENDING_MANUAL_INSPECTION"},
        }
        (output_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
        metadata["storage"]["extracted_size_bytes"] = _directory_size(output_root)
        (output_root / "dataset_metadata.json").write_bytes(canonical_json_bytes(metadata))
        compact = {key: value for key, value in metadata.items() if key != "episodes"}
        compact["episodes"] = [{
            "episode_id": item["episode_id"], "accepted_samples": item["counts"]["accepted_camera_samples"],
            "steering_age_ms": item["synchronization"]["steering_age_ms"],
            "recovery": item["recovery"],
        } for item in episode_metrics]
        (results_dir / "extraction_summary.json").write_bytes(canonical_json_bytes(compact))
        return metadata
    except Exception:
        raise


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--route-npy", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path)
    parser.add_argument("--expert-config", type=Path)
    parser.add_argument("--extractor-config", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if sum((args.plan_only, args.preflight_only, args.collect, args.extract)) != 1:
        raise SystemExit("choose exactly one of --plan-only, --preflight-only, --collect, --extract")
    config = load_json(args.config)
    try:
        if args.plan_only:
            if args.route_npy is None:
                raise ValueError("--route-npy is required for --plan-only")
            plan = build_plan(route_from_npy(args.route_npy), config, str(args.route_npy))
            args.plan.parent.mkdir(parents=True, exist_ok=True)
            args.plan.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        plan = load_json(args.plan)
        if args.preflight_only:
            if args.sim_root is None or args.expert_config is None:
                raise ValueError("--sim-root and --expert-config are required for --preflight-only")
            result = run_live_preflight(
                config=config, plan=plan, expert_path=args.expert_config, sim_root=args.sim_root,
            )
            args.results_dir.mkdir(parents=True, exist_ok=True)
            (args.results_dir / "preflight.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.collect:
            if args.sim_root is None or args.expert_config is None:
                raise ValueError("--sim-root and --expert-config are required for --collect")
            result = run_collection(
                config=config, plan=plan, expert_path=args.expert_config,
                sim_root=args.sim_root, results_dir=args.results_dir,
            )
        else:
            if args.extractor_config is None or args.raw_root is None or args.output_root is None:
                raise ValueError("--extractor-config, --raw-root, and --output-root are required for --extract")
            result = run_extraction_recovery(
                config=config, extractor_config_path=args.extractor_config, raw_root=args.raw_root,
                output_root=args.output_root, results_dir=args.results_dir, plan=plan,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["result"] == "PASS" else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
