"""Safe, deterministic Pure Pursuit expert driver V1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Protocol

from .route_geometry import ClosedRoute, OffTrackMonitor, ProgressTracker, pure_pursuit_steering
from .sim_client import SimClient, verify_control_schema


@dataclass(frozen=True)
class DriverConfig:
    base_url: str
    expected_world: str
    wheelbase_m: float
    max_steering_rad: float
    fixed_speed_mps: float
    control_frequency_hz: float
    lookahead_m: float
    start_gate_radius_m: float
    minimum_lap_progress_fraction: float
    off_track_margin_m: float
    off_track_grace_s: float
    api_timeout_s: float
    pose_stale_timeout_s: float
    pose_motion_translation_threshold_m: float
    pose_motion_yaw_threshold_rad: float
    maximum_runtime_s: float
    closed_route_tolerance_m: float
    spawn_route_tolerance_m: float
    minimum_route_points: int
    maximum_progress_jump_m: float
    world_check_interval_s: float
    reset_wait_timeout_s: float

    @classmethod
    def load(cls, path: str | Path) -> "DriverConfig":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a JSON object")
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid configuration fields: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        def is_numeric(value: object) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        positive = (
            "wheelbase_m", "max_steering_rad", "control_frequency_hz", "lookahead_m",
            "start_gate_radius_m", "api_timeout_s", "pose_stale_timeout_s",
            "pose_motion_translation_threshold_m", "pose_motion_yaw_threshold_rad",
            "maximum_runtime_s", "closed_route_tolerance_m", "spawn_route_tolerance_m",
            "maximum_progress_jump_m", "world_check_interval_s", "reset_wait_timeout_s",
        )
        for name in positive:
            value = getattr(self, name)
            if not is_numeric(value) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not is_numeric(self.fixed_speed_mps)
            or not math.isfinite(self.fixed_speed_mps)
            or self.fixed_speed_mps <= 0
            or self.fixed_speed_mps > 3.0
        ):
            raise ValueError("fixed_speed_mps must be in (0, 3.0]")
        if self.max_steering_rad > math.pi / 2:
            raise ValueError("max_steering_rad is physically impossible")
        if (
            not is_numeric(self.minimum_lap_progress_fraction)
            or not math.isfinite(self.minimum_lap_progress_fraction)
            or not 0 < self.minimum_lap_progress_fraction <= 1
        ):
            raise ValueError("minimum_lap_progress_fraction must be in (0, 1]")
        for name in ("off_track_margin_m", "off_track_grace_s"):
            value = getattr(self, name)
            if not is_numeric(value) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not isinstance(self.minimum_route_points, int)
            or isinstance(self.minimum_route_points, bool)
            or self.minimum_route_points < 3
        ):
            raise ValueError("minimum_route_points must be an integer of at least 3")
        if (
            not isinstance(self.base_url, str)
            or not self.base_url.startswith(("http://", "https://"))
            or not isinstance(self.expected_world, str)
            or not self.expected_world
        ):
            raise ValueError("base_url and expected_world are required")


@dataclass
class Preflight:
    world: str
    route: ClosedRoute
    route_points: int
    cone_count: int
    bounds: dict[str, Any]
    pose: dict[str, Any]


class PoseLivenessMonitor:
    """Require source-backed clock and meaningful pose updates during motion."""

    def __init__(self, timeout_s: float, translation_threshold_m: float, yaw_threshold_rad: float) -> None:
        values = (timeout_s, translation_threshold_m, yaw_threshold_rad)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("pose liveness limits must be finite and positive")
        self.timeout_s = timeout_s
        self.translation_threshold_m = translation_threshold_m
        self.yaw_threshold_rad = yaw_threshold_rad
        self._active = False
        self._reference_pose: tuple[float, float, float] | None = None
        self._last_pose_evidence_at: float | None = None
        self._last_clock_evidence_at: float | None = None
        self._last_sim_time: float | None = None

    def update(
        self,
        pose: dict[str, Any],
        sim_time: float,
        now: float,
        *,
        motion_commanded: bool,
    ) -> None:
        if not motion_commanded:
            self._reset()
            return
        current = (float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
        if not all(math.isfinite(value) for value in (*current, sim_time, now)):
            raise RuntimeError("pose liveness received non-finite state")
        if not self._active:
            self._active = True
            self._reference_pose = current
            self._last_pose_evidence_at = now
            self._last_clock_evidence_at = now
            self._last_sim_time = sim_time
            return

        assert self._reference_pose is not None
        assert self._last_pose_evidence_at is not None
        assert self._last_clock_evidence_at is not None
        assert self._last_sim_time is not None
        if sim_time < self._last_sim_time - 1e-9:
            raise RuntimeError("simulator clock moved backward while motion was commanded")
        if sim_time > self._last_sim_time + 1e-9:
            self._last_sim_time = sim_time
            self._last_clock_evidence_at = now

        dx = current[0] - self._reference_pose[0]
        dy = current[1] - self._reference_pose[1]
        yaw_change = abs(_wrapped_angle(current[2] - self._reference_pose[2]))
        if math.hypot(dx, dy) >= self.translation_threshold_m or yaw_change >= self.yaw_threshold_rad:
            self._reference_pose = current
            self._last_pose_evidence_at = now

        if now - self._last_clock_evidence_at > self.timeout_s:
            raise RuntimeError(
                f"simulator clock did not advance for {now - self._last_clock_evidence_at:.3f}s "
                "while motion was commanded"
            )
        if now - self._last_pose_evidence_at > self.timeout_s:
            raise RuntimeError(
                f"pose did not change meaningfully for {now - self._last_pose_evidence_at:.3f}s "
                "while motion was commanded"
            )

    def _reset(self) -> None:
        self._active = False
        self._reference_pose = None
        self._last_pose_evidence_at = None
        self._last_clock_evidence_at = None
        self._last_sim_time = None


class ClientLike(Protocol):
    def openapi(self) -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...
    def route(self) -> dict[str, Any]: ...
    def pose(self) -> dict[str, Any]: ...
    def clock(self) -> dict[str, Any]: ...
    def bounds(self) -> dict[str, Any]: ...
    def objects(self) -> dict[str, Any]: ...
    def command_steering(self, value: float) -> dict[str, Any]: ...
    def command_speed(self, value: float) -> dict[str, Any]: ...
    def safe_stop(self) -> list[str]: ...


def preflight(client: ClientLike, config: DriverConfig, allow_unexpected_world: bool = False) -> Preflight:
    verify_control_schema(client.openapi())
    status = client.status()
    if status.get("running") is not True:
        raise RuntimeError(f"simulator is not running: {status}")
    if status.get("switching") is not False:
        raise RuntimeError(f"simulator is switching worlds: {status}")
    world = status.get("current")
    if world != config.expected_world and not allow_unexpected_world:
        raise RuntimeError(f"expected world {config.expected_world!r}, active world is {world!r}")
    route_payload = client.route()
    if route_payload.get("world") != world:
        raise RuntimeError("route world does not match active world")
    waypoints = route_payload.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) < config.minimum_route_points:
        raise RuntimeError(f"route has fewer than {config.minimum_route_points} points")
    if math.dist(waypoints[0][:2], waypoints[-1][:2]) > config.closed_route_tolerance_m:
        raise RuntimeError("route endpoints are not sufficiently closed")
    route = ClosedRoute(waypoints, route_payload.get("inner"), route_payload.get("outer"))
    if route.inner is None or route.outer is None:
        raise RuntimeError("route does not provide usable inner and outer track boundaries")
    bounds = client.bounds()
    for key in ("minX", "maxX", "minY", "maxY"):
        if key not in bounds or not math.isfinite(float(bounds[key])):
            raise RuntimeError(f"bounds missing finite {key}")
    pose = client.pose()
    spawn_distance = math.dist((pose["x"], pose["y"]), route.points[0])
    if spawn_distance > config.spawn_route_tolerance_m:
        raise RuntimeError(
            f"vehicle is {spawn_distance:.3f}m from route spawn; "
            f"limit is {config.spawn_route_tolerance_m:.3f}m"
        )
    objects = client.objects()
    if objects.get("world") != world:
        raise RuntimeError("objects world does not match active world")
    names = [str(item.get("name", "")) for item in objects.get("objects", []) if isinstance(item, dict)]
    cones = sum(name.lower().startswith("cone") for name in names)
    if cones:
        raise RuntimeError(f"derived-world safety check failed: found {cones} cone objects")
    return Preflight(world, route, len(waypoints), cones, bounds, pose)


def wait_after_reset(client: SimClient, config: DriverConfig, allow_unexpected_world: bool) -> Preflight:
    try:
        pre_stop_errors = client.safe_stop()
        if pre_stop_errors:
            raise RuntimeError("pre-reset safe-stop failed: " + "; ".join(pre_stop_errors))
        client.reset()
        post_stop_errors = client.safe_stop()
        if post_stop_errors:
            raise RuntimeError("post-reset safe-stop failed: " + "; ".join(post_stop_errors))
        deadline = time.monotonic() + config.reset_wait_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                result = preflight(client, config, allow_unexpected_world)
                if math.dist(
                    (result.pose["x"], result.pose["y"]), result.route.points[0]
                ) <= config.spawn_route_tolerance_m:
                    return result
            except Exception as exc:
                last_error = exc
            time.sleep(0.1)
        raise RuntimeError(f"reset did not produce a valid spawn before timeout: {last_error}")
    except BaseException:
        client.safe_stop()
        raise


def run_driver(
    client: ClientLike,
    config: DriverConfig,
    initial: Preflight,
    *,
    dry_run_s: float | None = None,
) -> dict[str, Any]:
    dry_run = dry_run_s is not None
    runtime_limit = dry_run_s if dry_run else config.maximum_runtime_s
    assert runtime_limit is not None and runtime_limit > 0
    route = initial.route
    tracker = ProgressTracker(route.length, config.maximum_progress_jump_m)
    off_track = OffTrackMonitor(config.off_track_grace_s)
    pose_liveness = PoseLivenessMonitor(
        config.pose_stale_timeout_s,
        config.pose_motion_translation_threshold_m,
        config.pose_motion_yaw_threshold_rad,
    )
    periods: list[float] = []
    ctes: list[float] = []
    steerings: list[float] = []
    iterations = 0
    saturation_count = 0
    started = time.monotonic()
    previous_iteration: float | None = None
    next_tick = started
    next_world_check = started
    result = "DRY_RUN_PASS" if dry_run else "FAIL"
    failure: str | None = None
    final_pose = initial.pose
    final_projection = route.project((final_pose["x"], final_pose["y"]))
    stop_errors: list[str] = []
    motion_commanded = False
    try:
        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= runtime_limit:
                if dry_run:
                    break
                raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick:
                time.sleep(next_tick - now)
            iteration_start = time.monotonic()
            if previous_iteration is not None:
                periods.append(iteration_start - previous_iteration)
            previous_iteration = iteration_start
            if iteration_start >= next_world_check:
                status = client.status()
                if status.get("running") is not True or status.get("switching") is not False:
                    raise RuntimeError(f"simulator state changed while driving: {status}")
                if status.get("current") != initial.world:
                    raise RuntimeError(f"unexpected world change while driving: {status.get('current')!r}")
                next_world_check = iteration_start + config.world_check_interval_s
            pose = client.pose()
            clock = client.clock()
            pose_liveness.update(
                pose,
                float(clock["sim_time"]),
                time.monotonic(),
                motion_commanded=motion_commanded,
            )
            final_pose = pose
            projection = route.project((pose["x"], pose["y"]))
            final_projection = projection
            progress = tracker.update(projection.s)
            boundary_distance = route.track_boundary_distance((pose["x"], pose["y"]))
            if boundary_distance is None or not math.isfinite(boundary_distance):
                raise RuntimeError("invalid track boundary geometry")
            if off_track.update(boundary_distance > config.off_track_margin_m, time.monotonic()):
                raise RuntimeError(
                    f"sustained off-track: {boundary_distance:.3f}m beyond track band "
                    f"exceeds {config.off_track_margin_m:.3f}m margin"
                )
            target = route.point_at(projection.s + config.lookahead_m)
            steering, curvature, target_distance = pure_pursuit_steering(
                (pose["x"], pose["y"]), float(pose["yaw"]), target,
                config.wheelbase_m, config.max_steering_rad,
            )
            if not all(math.isfinite(v) for v in (progress, steering, curvature, target_distance)):
                raise RuntimeError("controller produced non-finite state")
            if not dry_run:
                client.command_steering(steering)
                client.command_speed(config.fixed_speed_mps)
                if not motion_commanded:
                    motion_commanded = True
                    pose_liveness.update(
                        pose,
                        float(clock["sim_time"]),
                        time.monotonic(),
                        motion_commanded=True,
                    )
            ctes.append(projection.distance)
            steerings.append(steering)
            saturation_count += math.isclose(abs(steering), config.max_steering_rad, abs_tol=1e-9)
            iterations += 1
            distance_to_start = math.dist((pose["x"], pose["y"]), route.points[0])
            if not dry_run and tracker.lap_complete(
                distance_to_start, config.start_gate_radius_m, config.minimum_lap_progress_fraction
            ):
                result = "PASS"
                break
            next_tick += 1.0 / config.control_frequency_hz
            if next_tick < time.monotonic() - 1.0 / config.control_frequency_hz:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        failure = "interrupted by user"
        result = "INTERRUPTED"
    except Exception as exc:
        failure = str(exc)
        result = "FAIL"
    finally:
        ended = time.monotonic()
        off_track.finalize(ended)
        if not dry_run:
            stop_errors = client.safe_stop()
            if stop_errors:
                result = "FAIL"
                failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
    elapsed = time.monotonic() - started
    final_distance = math.dist((final_pose["x"], final_pose["y"]), route.points[0])
    metrics = {
        "result": result,
        "failure": failure,
        "dry_run": dry_run,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "world": initial.world,
        "route_length_m": route.length,
        "route_points_api": initial.route_points,
        "cone_count": initial.cone_count,
        "elapsed_s": elapsed,
        "control_iterations": iterations,
        "mean_loop_period_s": _mean(periods),
        "p95_loop_period_s": _percentile(periods, 0.95),
        "max_loop_period_s": max(periods, default=0.0),
        "period_slip_count": sum(p > 1.5 / config.control_frequency_hz for p in periods),
        "mean_centerline_error_m": _mean(ctes),
        "max_centerline_error_m": max(ctes, default=0.0),
        "mean_absolute_steering_rad": _mean([abs(v) for v in steerings]),
        "max_absolute_steering_rad": max((abs(v) for v in steerings), default=0.0),
        "steering_saturation_fraction": saturation_count / iterations if iterations else 0.0,
        "off_track_event_count": off_track.event_count,
        "off_track_total_duration_s": off_track.total_duration_s,
        "final_distance_to_start_m": final_distance,
        "final_route_s_m": final_projection.s,
        "total_unwrapped_progress_m": tracker.unwrapped,
        "rejected_progress_jumps": tracker.rejected_jumps,
        "safe_stop_success": not stop_errors if not dry_run else None,
        "safe_stop_errors": stop_errors,
    }
    return metrics


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="path to an explicit expert-driver JSON configuration",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--dry-run", type=float, metavar="SECONDS")
    parser.add_argument("--reset-before-run", action="store_true")
    parser.add_argument("--allow-unexpected-world", action="store_true")
    parser.add_argument("--result", type=Path, help="write compact JSON metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client: SimClient | None = None
    exit_code = 2
    try:
        config = DriverConfig.load(args.config)
        if args.dry_run is not None and (not math.isfinite(args.dry_run) or args.dry_run <= 0):
            raise ValueError("--dry-run seconds must be finite and positive")
        client = SimClient(config.base_url, config.api_timeout_s)
        initial_stop_errors = client.safe_stop()
        if initial_stop_errors:
            raise RuntimeError("initial safe-stop failed: " + "; ".join(initial_stop_errors))
        initial = (
            wait_after_reset(client, config, args.allow_unexpected_world)
            if args.reset_before_run
            else preflight(client, config, args.allow_unexpected_world)
        )
        summary = {
            "preflight": "PASS", "world": initial.world, "route_points": initial.route_points,
            "route_length_m": initial.route.length, "cone_count": initial.cone_count,
            "bounds": initial.bounds, "pose": initial.pose,
        }
        if args.preflight_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
            exit_code = 0
        else:
            metrics = run_driver(client, config, initial, dry_run_s=args.dry_run)
            print(json.dumps(metrics, indent=2, sort_keys=True))
            if args.result:
                args.result.parent.mkdir(parents=True, exist_ok=True)
                args.result.write_text(
                    json.dumps({"config": asdict(config), "metrics": metrics}, indent=2, sort_keys=True) + "\n"
                )
            exit_code = 0 if metrics["result"] in ("PASS", "DRY_RUN_PASS") else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 2
    except KeyboardInterrupt:
        print("ERROR: interrupted by user", file=sys.stderr)
        exit_code = 130
    finally:
        if client is not None:
            stop_errors = client.safe_stop()
            if stop_errors:
                print("ERROR: final safe-stop failed: " + "; ".join(stop_errors), file=sys.stderr)
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
