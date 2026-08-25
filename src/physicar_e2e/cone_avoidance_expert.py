"""Privileged one-cone Pure Pursuit Expert V1 and hard-gated evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from PIL import Image

from .cone_avoidance_environment import (
    ConeGeometry,
    ConeSite,
    EnvironmentConfig,
    EnvironmentError,
    RouteData,
    VehicleFootprint,
    asset_set,
    box_polygon,
    cone_geometry_dict,
    cone_site_dict,
    footprint_polygon,
    generate_environment,
    load_route,
    parse_cone_geometry,
    parse_vehicle_footprint,
    point_track_clearance,
    polygon_clearance,
    route_curvature,
    route_yaw,
    select_cone_site,
    sha256_file,
    share_path,
    vehicle_footprint_dict,
    verify_derived_environment,
)
from .expert_driver import DriverConfig, Preflight, preflight as lane_preflight, run_driver
from .pilotnet_v4_repeatability import clock_health_preflight
from .route_geometry import ClosedRoute, Projection, pure_pursuit_steering
from .sim_client import SimClient


VERSION = "cone_avoidance_expert_v1"
EXPECTED_RESULT_DIRECTORY = "results/cone_avoidance_expert_v1"
EXPECTED_ENVIRONMENT_RESULT_DIRECTORY = "results/cone_avoidance_environment_v1"
MAXIMUM_LIVE_ATTEMPTS = 5
TARGET_VALID_EVALUATIONS = 3
PRESERVED_EVIDENCE = {
    "pilotnet_v4": (
        "results/pilotnet_v4_repeatability_v1/summary.json",
        "15fd6c61ce94f3962aeb6c7c64bd7462a1e60ab9161282ea926cd9fd60dc8503",
    ),
    "temporal_pilotnet_v9_live": (
        "results/pilotnet_e2e_v9_high_speed_temporal/summary.json",
        "56829cfc312f5cfe353458c60afcd146a54f406374d2e02546e20406cccaa6d2",
    ),
    "temporal_pilotnet_v9_training": (
        "results/pilotnet_training_v9_high_speed_temporal/summary.json",
        "4c22c0b7f2d408b44b4698ff98d394ff3bde3f8d40c5dc34e6edd0a30d906f87",
    ),
    "high_speed_expert": (
        "results/expert_speed_1p8_repeatability_v1/summary.json",
        "aa0195ebeb2a15d1e6536896afaa5115bbf3c7c7cde7df319768dcff1bb42672",
    ),
}


class ExpertError(RuntimeError):
    """A controlled Expert-contract or geometry-gate failure."""


@dataclass(frozen=True)
class ExpertConfig:
    payload: dict[str, Any]
    environment_path: Path
    environment: EnvironmentConfig
    baseline_path: Path
    baseline: DriverConfig
    driver: DriverConfig
    avoidance: dict[str, Any]
    frozen_bypass: dict[str, Any]
    return_to_route: dict[str, float]

    @classmethod
    def load(cls, path: Path, repo_root: Path, sim_root: Path) -> "ExpertConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExpertError(f"cannot load Expert config {path}: {exc}") from exc
        required = {
            "version", "baseline_expert_config", "baseline_expert_sha256",
            "environment_config", "avoidance", "frozen_bypass", "return_to_route",
            "maximum_live_attempts", "target_valid_evaluations",
            "retry_obstacle_expert_failure", "result_directory",
            "privileged_inputs", "future_policy_forbidden_inputs",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ExpertError("Expert config fields do not match the V1 protocol")
        exact = {
            "version": VERSION,
            "maximum_live_attempts": MAXIMUM_LIVE_ATTEMPTS,
            "target_valid_evaluations": TARGET_VALID_EVALUATIONS,
            "retry_obstacle_expert_failure": False,
            "result_directory": EXPECTED_RESULT_DIRECTORY,
            "privileged_inputs": [
                "gt_vehicle_pose", "nominal_route", "cone_gt_pose",
                "generated_bypass_reference", "track_geometry",
            ],
            "future_policy_forbidden_inputs": [
                "cone_gt_coordinates", "route", "pose", "cte", "expert_command",
            ],
        }
        for key, expected in exact.items():
            if payload[key] != expected:
                raise ExpertError(f"Expert config {key} differs from the frozen protocol")
        baseline_path = (repo_root / str(payload["baseline_expert_config"])).resolve()
        environment_path = (repo_root / str(payload["environment_config"])).resolve()
        if sha256_file(baseline_path) != payload["baseline_expert_sha256"]:
            raise ExpertError("preserved High-Speed Expert config identity mismatch")
        baseline = DriverConfig.load(baseline_path)
        environment = EnvironmentConfig.load(environment_path)
        driver = replace(baseline, expected_world=environment.derived_world)
        driver.validate()
        changed = [
            name for name in baseline.__dataclass_fields__
            if getattr(baseline, name) != getattr(driver, name)
        ]
        if changed != ["expected_world"]:
            raise ExpertError(f"unexpected difference from High-Speed Expert V1: {changed}")
        fixed = (
            driver.fixed_speed_mps, driver.lookahead_m, driver.control_frequency_hz,
            driver.max_steering_rad, driver.wheelbase_m,
        )
        if fixed != (1.80, 0.90, 15.0, 0.349066, 0.18):
            raise ExpertError(f"High-Speed obstacle Expert fixed contract changed: {fixed}")
        avoidance = payload["avoidance"]
        expected_avoidance_keys = {
            "profile", "transition_minimum_time_s", "transition_minimum_lookaheads",
            "plateau_half_lookaheads", "maximum_reference_curvature_fraction",
            "sample_spacing_m", "kinematic_preview_substep_s",
        }
        if not isinstance(avoidance, dict) or set(avoidance) != expected_avoidance_keys:
            raise ExpertError("avoidance geometry fields do not match V1")
        if avoidance["profile"] != "symmetric_quintic_with_flat_pass":
            raise ExpertError("V1 requires the frozen symmetric quintic profile")
        for key, value in avoidance.items():
            if key != "profile" and not _positive_number(value):
                raise ExpertError(f"avoidance field {key} must be finite and positive")
        if float(avoidance["maximum_reference_curvature_fraction"]) > 1.0:
            raise ExpertError("reference curvature fraction cannot exceed one")
        frozen = payload["frozen_bypass"]
        frozen_keys = {
            "chosen_side", "required_center_offset_m", "maximum_lateral_offset_m",
            "departure_start_s_m", "cone_s_m", "return_end_s_m",
            "transition_length_m", "plateau_half_length_m",
        }
        if not isinstance(frozen, dict) or set(frozen) != frozen_keys:
            raise ExpertError("frozen_bypass fields do not match V1")
        if frozen["chosen_side"] not in ("left", "right"):
            raise ExpertError("frozen bypass side is invalid")
        if not all(_finite_number(value) for key, value in frozen.items() if key != "chosen_side"):
            raise ExpertError("frozen bypass geometry must be finite")
        return_contract = payload["return_to_route"]
        if not isinstance(return_contract, dict) or set(return_contract) != {
            "maximum_absolute_nominal_cte_m", "minimum_stable_duration_s"
        } or not all(_positive_number(value) for value in return_contract.values()):
            raise ExpertError("return-to-route contract is invalid")
        # Resolve and inspect simulator geometry now so config loading itself is
        # a provenance gate; no simulator write is performed.
        verify_derived = False
        try:
            verify_derived_environment(environment, share_path(sim_root))
            verify_derived = True
        except EnvironmentError as exc:
            if "assets are missing" not in str(exc):
                raise
        _ = verify_derived
        return cls(
            payload=payload,
            environment_path=environment_path,
            environment=environment,
            baseline_path=baseline_path,
            baseline=baseline,
            driver=driver,
            avoidance=dict(avoidance),
            frozen_bypass=dict(frozen),
            return_to_route={key: float(value) for key, value in return_contract.items()},
        )


@dataclass(frozen=True)
class BypassPlan:
    nominal: ClosedRoute
    site: ConeSite
    cone: ConeGeometry
    footprint: VehicleFootprint
    side: str
    side_sign: float
    required_center_offset_m: float
    maximum_lateral_offset_m: float
    transition_length_m: float
    plateau_half_length_m: float
    departure_start_s_m: float
    plateau_start_s_m: float
    cone_s_m: float
    plateau_end_s_m: float
    return_end_s_m: float
    physical_curvature_limit_per_m: float
    design_curvature_limit_per_m: float
    geometric_minimum_transition_m: float
    available_left_clearance_m: float
    available_right_clearance_m: float

    def lateral_offset(self, s: float) -> float:
        if s <= self.departure_start_s_m or s >= self.return_end_s_m:
            return 0.0
        if s < self.plateau_start_s_m:
            fraction = (s - self.departure_start_s_m) / self.transition_length_m
            return self.side_sign * self.maximum_lateral_offset_m * _quintic(fraction)
        if s <= self.plateau_end_s_m:
            return self.side_sign * self.maximum_lateral_offset_m
        fraction = (self.return_end_s_m - s) / self.transition_length_m
        return self.side_sign * self.maximum_lateral_offset_m * _quintic(fraction)

    def point_at(self, s: float) -> tuple[float, float]:
        wrapped = s % self.nominal.length
        point = self.nominal.point_at(wrapped)
        yaw = route_yaw(self.nominal, wrapped)
        offset = self.lateral_offset(wrapped)
        return point[0] - math.sin(yaw) * offset, point[1] + math.cos(yaw) * offset

    def yaw_at(self, s: float, half_window_m: float = 0.005) -> float:
        before, after = self.point_at(s - half_window_m), self.point_at(s + half_window_m)
        return math.atan2(after[1] - before[1], after[0] - before[0])


class ObstacleAwareRoute:
    """Nominal projection/progress with bypass-only Pure Pursuit targets."""

    def __init__(self, nominal: ClosedRoute, plan: BypassPlan) -> None:
        self.nominal = nominal
        self.plan = plan
        self.points = nominal.points
        self.inner = nominal.inner
        self.outer = nominal.outer
        self.length = nominal.length

    def project(self, position: Any) -> Projection:
        return self.nominal.project(position)

    def point_at(self, s: float) -> tuple[float, float]:
        return self.plan.point_at(s)

    def track_boundary_distance(self, position: Any) -> float | None:
        return self.nominal.track_boundary_distance(position)


def build_bypass_plan(config: ExpertConfig, sim_root: Path) -> tuple[BypassPlan, RouteData]:
    share = share_path(sim_root)
    cone_free = asset_set(share, config.environment.canonical_cone_free_world)
    route_data = load_route(cone_free.route)
    site = select_cone_site(config.environment, route_data)
    cone, _ = parse_cone_geometry(config.environment, share)
    footprint = parse_vehicle_footprint(config.environment, share)
    side = site.chosen_side
    side_sign = 1.0 if side == "left" else -1.0
    # Cone and footprint are aligned with route yaw at the frozen pass point.
    # The footprint is already a conservative all-steering envelope.
    required = (
        max(abs(footprint.y_min_m), abs(footprint.y_max_m))
        + cone.half_width_m
        + config.environment.required_cone_clearance_m
    )
    maximum_offset = required + config.environment.extra_cone_clearance_m
    available = site.left_clearance_m if side == "left" else site.right_clearance_m
    if maximum_offset >= available:
        raise ExpertError(
            f"no centerline-contained bypass: required {maximum_offset:.6f}m, available {available:.6f}m"
        )
    physical_limit = math.tan(config.driver.max_steering_rad) / config.driver.wheelbase_m
    design_limit = physical_limit * float(config.avoidance["maximum_reference_curvature_fraction"])
    geometric_minimum = _minimum_transition_for_curvature(maximum_offset, design_limit)
    transition = max(
        geometric_minimum,
        config.driver.fixed_speed_mps * float(config.avoidance["transition_minimum_time_s"]),
        config.driver.lookahead_m * float(config.avoidance["transition_minimum_lookaheads"]),
    )
    plateau_half = config.driver.lookahead_m * float(config.avoidance["plateau_half_lookaheads"])
    start = site.route_s_m - transition - plateau_half
    end = site.route_s_m + transition + plateau_half
    if start <= float(config.environment.selector["start_exclusion_m"]) or end >= route_data.route.length - float(config.environment.selector["end_exclusion_m"]):
        raise ExpertError("avoidance span violates the frozen spawn/final-region exclusions")
    if start < site.straight_run_start_s_m or end > site.straight_run_end_s_m:
        raise ExpertError("avoidance span does not fit entirely within the selected straight run")
    plan = BypassPlan(
        nominal=route_data.route,
        site=site,
        cone=cone,
        footprint=footprint,
        side=side,
        side_sign=side_sign,
        required_center_offset_m=required,
        maximum_lateral_offset_m=maximum_offset,
        transition_length_m=transition,
        plateau_half_length_m=plateau_half,
        departure_start_s_m=start,
        plateau_start_s_m=site.route_s_m - plateau_half,
        cone_s_m=site.route_s_m,
        plateau_end_s_m=site.route_s_m + plateau_half,
        return_end_s_m=end,
        physical_curvature_limit_per_m=physical_limit,
        design_curvature_limit_per_m=design_limit,
        geometric_minimum_transition_m=geometric_minimum,
        available_left_clearance_m=site.left_clearance_m,
        available_right_clearance_m=site.right_clearance_m,
    )
    verify_frozen_bypass(config, plan)
    return plan, route_data


def verify_frozen_bypass(config: ExpertConfig, plan: BypassPlan, tolerance: float = 1e-6) -> None:
    observed: dict[str, Any] = {
        "chosen_side": plan.side,
        "required_center_offset_m": plan.required_center_offset_m,
        "maximum_lateral_offset_m": plan.maximum_lateral_offset_m,
        "departure_start_s_m": plan.departure_start_s_m,
        "cone_s_m": plan.cone_s_m,
        "return_end_s_m": plan.return_end_s_m,
        "transition_length_m": plan.transition_length_m,
        "plateau_half_length_m": plan.plateau_half_length_m,
    }
    mismatches: dict[str, Any] = {}
    for key, value in observed.items():
        frozen = config.frozen_bypass[key]
        if isinstance(value, str):
            if value != frozen:
                mismatches[key] = {"frozen": frozen, "computed": value}
        elif not math.isclose(float(frozen), value, rel_tol=0.0, abs_tol=tolerance):
            mismatches[key] = {"frozen": frozen, "computed": value}
    if mismatches:
        raise ExpertError(f"computed bypass differs from frozen V1 geometry: {mismatches}")


def validate_geometry(config: ExpertConfig, plan: BypassPlan, route_data: RouteData) -> dict[str, Any]:
    spacing = float(config.avoidance["sample_spacing_m"])
    count = int(math.ceil(plan.nominal.length / spacing))
    samples_s = [index * plan.nominal.length / count for index in range(count)]
    cone_polygon = box_polygon(
        plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
        plan.cone.half_length_m, plan.cone.half_width_m,
    )
    minimum_clearance = math.inf
    minimum_clearance_s = 0.0
    intersection = False
    minimum_track = math.inf
    maximum_curvature = 0.0
    nonfinite_count = 0
    for s in samples_s:
        point = plan.point_at(s)
        yaw = plan.yaw_at(s)
        if not all(math.isfinite(value) for value in (*point, yaw)):
            nonfinite_count += 1
            continue
        vehicle = footprint_polygon(plan.footprint, point[0], point[1], yaw)
        clearance, intersects = polygon_clearance(vehicle, cone_polygon)
        if clearance < minimum_clearance:
            minimum_clearance, minimum_clearance_s = clearance, s
        intersection = intersection or intersects
        minimum_track = min(minimum_track, point_track_clearance(route_data.route, point))
        if plan.departure_start_s_m - spacing <= s <= plan.return_end_s_m + spacing:
            first, middle, last = plan.point_at(s - spacing), point, plan.point_at(s + spacing)
            maximum_curvature = max(maximum_curvature, abs(_three_point_curvature(first, middle, last)))
    start_error = math.dist(plan.point_at(plan.departure_start_s_m), plan.nominal.point_at(plan.departure_start_s_m))
    end_error = math.dist(plan.point_at(plan.return_end_s_m), plan.nominal.point_at(plan.return_end_s_m))
    start_heading_error = abs(_wrapped(plan.yaw_at(plan.departure_start_s_m) - route_yaw(plan.nominal, plan.departure_start_s_m)))
    end_heading_error = abs(_wrapped(plan.yaw_at(plan.return_end_s_m) - route_yaw(plan.nominal, plan.return_end_s_m)))
    preview = kinematic_preview(config, plan, route_data.route)
    gates = {
        "finite_geometry": nonfinite_count == 0,
        "no_footprint_cone_intersection": not intersection,
        "minimum_planned_cone_clearance": minimum_clearance + 1e-9 >= config.environment.required_cone_clearance_m,
        "reference_center_within_track": minimum_track >= -1e-9,
        "reference_curvature_feasible": maximum_curvature <= plan.physical_curvature_limit_per_m + 1e-9,
        "smooth_start_end_position": max(start_error, end_error) <= 1e-9,
        "smooth_start_end_heading": max(start_heading_error, end_heading_error) <= 1e-4,
        "kinematic_preview_clearance": preview["minimum_footprint_to_cone_clearance_m"] + 1e-9 >= config.environment.required_cone_clearance_m,
        "kinematic_preview_return": bool(preview["recovery_success"]),
    }
    result = {
        "result": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "nominal_route_length_m": plan.nominal.length,
        "profile": config.avoidance["profile"],
        "chosen_side": plan.side,
        "available_left_clearance_m": plan.available_left_clearance_m,
        "available_right_clearance_m": plan.available_right_clearance_m,
        "required_center_offset_m": plan.required_center_offset_m,
        "maximum_lateral_offset_m": plan.maximum_lateral_offset_m,
        "departure_start_s_m": plan.departure_start_s_m,
        "plateau_start_s_m": plan.plateau_start_s_m,
        "cone_s_m": plan.cone_s_m,
        "plateau_end_s_m": plan.plateau_end_s_m,
        "return_end_s_m": plan.return_end_s_m,
        "total_avoidance_span_m": plan.return_end_s_m - plan.departure_start_s_m,
        "transition_length_m": plan.transition_length_m,
        "geometric_minimum_transition_m": plan.geometric_minimum_transition_m,
        "maximum_reference_curvature_per_m": maximum_curvature,
        "physical_curvature_limit_per_m": plan.physical_curvature_limit_per_m,
        "maximum_reference_equivalent_steering_rad": math.atan(config.driver.wheelbase_m * maximum_curvature),
        "minimum_planned_footprint_to_cone_clearance_m": minimum_clearance,
        "minimum_clearance_route_s_m": minimum_clearance_s,
        "minimum_reference_center_track_clearance_m": minimum_track,
        "footprint_intersection": intersection,
        "start_position_error_m": start_error,
        "end_position_error_m": end_error,
        "start_heading_error_rad": start_heading_error,
        "end_heading_error_rad": end_heading_error,
        "sample_spacing_m": spacing,
        "kinematic_preview": preview,
        "cone_geometry": cone_geometry_dict(plan.cone),
        "vehicle_footprint": vehicle_footprint_dict(plan.footprint),
        "cone_site": cone_site_dict(plan.site),
        "track_clearance_definition": "Euclidean reference-center distance to the nearest logical track boundary; nonnegative points are inside the existing logical track band.",
        "clearance_contract_note": "0.05 m is an experimental simulator margin, not calibrated physical truth.",
    }
    if result["result"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise ExpertError(f"offline geometry gate failed: {failed}")
    return result


def kinematic_preview(config: ExpertConfig, plan: BypassPlan, nominal: ClosedRoute) -> dict[str, Any]:
    """Frozen ideal-bicycle preflight; this is not treated as live evidence."""
    control_period = 1.0 / config.driver.control_frequency_hz
    substep = float(config.avoidance["kinematic_preview_substep_s"])
    x, y = nominal.point_at(plan.departure_start_s_m - config.driver.lookahead_m)
    yaw = route_yaw(nominal, plan.departure_start_s_m - config.driver.lookahead_m)
    end_s = plan.return_end_s_m + 1.5
    cone_polygon = box_polygon(plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
                               plan.cone.half_length_m, plan.cone.half_width_m)
    minimum_clearance = math.inf
    minimum_s = 0.0
    intersection = False
    minimum_track = math.inf
    maximum_actual_offset = 0.0
    recovery_started: float | None = None
    recovery_success = False
    recovery_time = None
    elapsed = 0.0
    maximum_steps = int(math.ceil((end_s - (plan.departure_start_s_m - config.driver.lookahead_m) + 2.0) / (config.driver.fixed_speed_mps * control_period)))
    for _ in range(maximum_steps):
        projection = nominal.project((x, y))
        target = plan.point_at(projection.s + config.driver.lookahead_m)
        steering, _, _ = pure_pursuit_steering(
            (x, y), yaw, target, config.driver.wheelbase_m, config.driver.max_steering_rad
        )
        remaining = control_period
        while remaining > 1e-12:
            dt = min(substep, remaining)
            curvature = math.tan(steering) / config.driver.wheelbase_m
            if abs(curvature) <= 1e-12:
                x += config.driver.fixed_speed_mps * dt * math.cos(yaw)
                y += config.driver.fixed_speed_mps * dt * math.sin(yaw)
            else:
                next_yaw = yaw + config.driver.fixed_speed_mps * curvature * dt
                x += (math.sin(next_yaw) - math.sin(yaw)) / curvature
                y += (-math.cos(next_yaw) + math.cos(yaw)) / curvature
                yaw = next_yaw
            elapsed += dt
            remaining -= dt
            projection = nominal.project((x, y))
            clearance, intersects = polygon_clearance(
                footprint_polygon(plan.footprint, x, y, yaw), cone_polygon
            )
            if clearance < minimum_clearance:
                minimum_clearance, minimum_s = clearance, projection.s
            intersection = intersection or intersects
            minimum_track = min(minimum_track, point_track_clearance(nominal, (x, y)))
            if plan.departure_start_s_m <= projection.s <= plan.return_end_s_m:
                directional = projection.signed_error * plan.side_sign
                maximum_actual_offset = max(maximum_actual_offset, directional)
            if projection.s >= plan.return_end_s_m:
                threshold = config.return_to_route["maximum_absolute_nominal_cte_m"]
                if projection.distance <= threshold:
                    recovery_started = elapsed if recovery_started is None else recovery_started
                    if elapsed - recovery_started >= config.return_to_route["minimum_stable_duration_s"]:
                        recovery_success = True
                        recovery_time = elapsed - recovery_started
                else:
                    recovery_started = None
        if projection.s >= end_s and recovery_success:
            break
    return {
        "model": "ideal kinematic bicycle with zero delay; controller updates at frozen 15 Hz",
        "evidence_role": "offline sanity check only; not a dynamic or live guarantee",
        "minimum_footprint_to_cone_clearance_m": minimum_clearance,
        "minimum_clearance_route_s_m": minimum_s,
        "footprint_intersection": intersection,
        "minimum_reference_center_track_clearance_m": minimum_track,
        "maximum_lateral_offset_reached_m": maximum_actual_offset,
        "recovery_success": recovery_success,
        "recovery_stable_duration_s": recovery_time,
    }


def write_geometry_plot(path: Path, plan: BypassPlan, route_data: RouteData, geometry: dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:
        raise ExpertError(f"matplotlib is required for the geometry evidence plot: {exc}") from exc
    spacing = 0.02
    span = plan.return_end_s_m - plan.departure_start_s_m
    count = int(math.ceil(span / spacing))
    reference = [
        plan.point_at(plan.departure_start_s_m + index * span / count)
        for index in range(count + 1)
    ]
    figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
    center = [*route_data.center, route_data.center[0]]
    inner = [*route_data.inner, route_data.inner[0]]
    outer = [*route_data.outer, route_data.outer[0]]
    axis.plot([p[0] for p in outer], [p[1] for p in outer], color="black", linewidth=1.0, label="track boundaries")
    axis.plot([p[0] for p in inner], [p[1] for p in inner], color="black", linewidth=1.0)
    axis.plot([p[0] for p in center], [p[1] for p in center], color="#777777", linewidth=1.0, label="nominal route")
    axis.plot([p[0] for p in reference], [p[1] for p in reference], color="#1769aa", linewidth=2.0, label="bypass reference")
    cone = box_polygon(plan.site.x_m, plan.site.y_m, plan.site.yaw_rad, plan.cone.half_length_m, plan.cone.half_width_m)
    axis.add_patch(Polygon(cone, closed=True, facecolor="#ff7f0e", edgecolor="#9a4b00", label="cone collision"))
    for label, s, marker in (
        ("departure", plan.departure_start_s_m, "o"),
        ("cone s", plan.cone_s_m, "x"),
        ("return", plan.return_end_s_m, "s"),
    ):
        point = plan.point_at(s)
        axis.scatter([point[0]], [point[1]], marker=marker, s=45, label=label)
    minimum_s = float(geometry["minimum_clearance_route_s_m"])
    minimum_point = plan.point_at(minimum_s)
    minimum_vehicle = footprint_polygon(plan.footprint, minimum_point[0], minimum_point[1], plan.yaw_at(minimum_s))
    axis.add_patch(Polygon(minimum_vehicle, closed=True, fill=False, linestyle="--", linewidth=1.2,
                           edgecolor="#d62728", label="vehicle envelope at min clearance"))
    axis.scatter([minimum_point[0]], [minimum_point[1]], marker="*", s=75, color="#d62728", label="minimum-clearance point")
    axis.annotate(
        f"pass {plan.side}\nplanned clearance {geometry['minimum_planned_footprint_to_cone_clearance_m']:.3f} m",
        xy=(plan.site.x_m, plan.site.y_m), xytext=(plan.site.x_m + 0.35, plan.site.y_m + 0.55),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8}, fontsize=8,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title("Cone Avoidance Expert V1 — frozen simulator geometry")
    axis.grid(True, linewidth=0.3, alpha=0.5)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


class _ConeMaskingClient:
    def __init__(self, client: SimClient) -> None:
        self.client = client

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def objects(self) -> dict[str, Any]:
        payload = self.client.objects()
        return {
            **payload,
            "objects": [
                item for item in payload.get("objects", [])
                if not str(item.get("name", "")).lower().startswith("cone")
            ],
        }


def obstacle_structural_preflight(client: SimClient, config: ExpertConfig, plan: BypassPlan) -> Preflight:
    masked = lane_preflight(_ConeMaskingClient(client), config.driver, False)
    payload = client.objects()
    if payload.get("world") != masked.world:
        raise RuntimeError("objects world does not match the active world")
    cones = [
        item for item in payload.get("objects", [])
        if isinstance(item, dict) and str(item.get("name", "")).lower().startswith("cone")
    ]
    if len(cones) != 1 or cones[0].get("name") != config.environment.derived_cone_model:
        raise RuntimeError(f"expected exactly one intended cone object, observed {cones}")
    cone = cones[0]
    size = cone.get("size") or {}
    expected_size = plan.cone.size_xyz_m
    if any(not math.isclose(float(size.get(key, math.nan)), expected_size[index], abs_tol=1e-9)
           for index, key in enumerate(("x", "y", "z"))):
        raise RuntimeError(f"live cone collision size mismatch: {size}")
    origin = cone.get("origin") or {}
    current = cone.get("current") or {}
    for label, pose in (("origin", origin), ("current", current)):
        if math.dist((float(pose.get("x", math.nan)), float(pose.get("y", math.nan))),
                     (plan.site.x_m, plan.site.y_m)) > 0.01:
            raise RuntimeError(f"cone {label} position differs from frozen geometry: {pose}")
    return Preflight(masked.world, masked.route, masked.route_points, 1, masked.bounds, masked.pose)


def wait_after_obstacle_reset(client: SimClient, config: ExpertConfig, plan: BypassPlan) -> Preflight:
    try:
        if errors := client.safe_stop():
            raise RuntimeError("pre-reset safe stop failed: " + "; ".join(errors))
        response = client.reset()
        if response.get("ok") is not True:
            raise RuntimeError(f"simulator reset was not confirmed: {response}")
        if errors := client.safe_stop():
            raise RuntimeError("post-reset safe stop failed: " + "; ".join(errors))
        deadline = time.monotonic() + config.driver.reset_wait_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return obstacle_structural_preflight(client, config, plan)
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(f"reset did not reach valid one-cone spawn: {last_error}")
    except BaseException:
        client.safe_stop()
        raise


def full_preflight(
    client: SimClient,
    config: ExpertConfig,
    plan: BypassPlan,
    sim_root: Path,
    geometry: dict[str, Any],
) -> tuple[Preflight, dict[str, Any]]:
    if geometry.get("result") != "PASS":
        raise RuntimeError("offline geometry gate is not PASS")
    environment = verify_derived_environment(config.environment, share_path(sim_root))
    initial = wait_after_obstacle_reset(client, config, plan)
    clock = clock_health_preflight(client)
    if clock.get("result") != "PASS":
        raise RuntimeError(str(clock.get("failure_reason", "clock health failed")))
    camera_started = time.monotonic()
    jpeg = client.camera_jpeg()
    with Image.open(BytesIO(jpeg)) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    if dimensions != [480, 360]:
        raise RuntimeError(f"camera dimensions differ from 480x360: {dimensions}")
    return initial, {
        "result": "PASS",
        "environment": environment,
        "offline_geometry_gate": "PASS",
        "world": initial.world,
        "switching": False,
        "route_points": initial.route_points,
        "route_length_m": initial.route.length,
        "cone_count": initial.cone_count,
        "pose": initial.pose,
        "bounds": initial.bounds,
        "clock_health": clock,
        "camera": {
            "result": "PASS", "dimensions": dimensions, "mode": mode,
            "acquisition_ms": (time.monotonic() - camera_started) * 1000.0,
        },
        "control_api": "PASS",
        "fixed_contract": {
            "speed_mps": config.driver.fixed_speed_mps,
            "lookahead_m": config.driver.lookahead_m,
            "control_frequency_hz": config.driver.control_frequency_hz,
            "steering_limit_rad": config.driver.max_steering_rad,
            "wheelbase_m": config.driver.wheelbase_m,
        },
    }


class ClearanceObserver:
    """Add privileged cone/footprint telemetry without changing commands."""

    def __init__(self, client: SimClient, nominal: ClosedRoute, plan: BypassPlan, config: ExpertConfig) -> None:
        self.client = client
        self.nominal = nominal
        self.plan = plan
        self.config = config
        self.samples: list[dict[str, Any]] = []
        self.intersection_occurred = False
        self.recovery_candidate_at: float | None = None
        self.recovery_candidate_ctes: list[float] = []
        self.recovery_success = False
        self.recovery_time_s: float | None = None
        self.recovery_cte_m: float | None = None
        self.first_after_return_at: float | None = None

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def pose(self) -> dict[str, Any]:
        pose = self.client.pose()
        objects = self.client.objects()
        cones = [
            item for item in objects.get("objects", [])
            if isinstance(item, dict) and item.get("name") == self.config.environment.derived_cone_model
        ]
        if len(cones) != 1:
            raise RuntimeError("cone GT telemetry lost the unique intended cone")
        cone_pose = cones[0].get("current") or cones[0].get("origin") or {}
        cone_x, cone_y = float(cone_pose["x"]), float(cone_pose["y"])
        cone_yaw = float(cone_pose.get("yaw", self.plan.site.yaw_rad))
        projection = self.nominal.project((pose["x"], pose["y"]))
        clearance, intersects = polygon_clearance(
            footprint_polygon(self.plan.footprint, float(pose["x"]), float(pose["y"]), float(pose["yaw"])),
            box_polygon(cone_x, cone_y, cone_yaw, self.plan.cone.half_length_m, self.plan.cone.half_width_m),
        )
        now = time.monotonic()
        row = {
            "wall_time_s": now,
            "route_s_m": projection.s,
            "nominal_cte_m": projection.distance,
            "nominal_signed_cte_m": projection.signed_error,
            "cone_clearance_m": clearance,
            "intersection": intersects,
            "steering_rad": None,
        }
        self.samples.append(row)
        self.intersection_occurred = self.intersection_occurred or intersects
        if projection.s >= self.plan.return_end_s_m:
            self.first_after_return_at = now if self.first_after_return_at is None else self.first_after_return_at
            threshold = self.config.return_to_route["maximum_absolute_nominal_cte_m"]
            if projection.distance <= threshold:
                if self.recovery_candidate_at is None:
                    self.recovery_candidate_at = now
                    self.recovery_candidate_ctes = []
                self.recovery_candidate_ctes.append(projection.distance)
                if (not self.recovery_success and
                        now - self.recovery_candidate_at >= self.config.return_to_route["minimum_stable_duration_s"]):
                    self.recovery_success = True
                    self.recovery_time_s = now - self.first_after_return_at
                    self.recovery_cte_m = max(self.recovery_candidate_ctes)
            elif not self.recovery_success:
                self.recovery_candidate_at = None
                self.recovery_candidate_ctes = []
        if intersects:
            raise RuntimeError("vehicle-footprint/cone collision intersection detected")
        if clearance + 1e-9 < self.config.environment.required_cone_clearance_m:
            raise RuntimeError(
                f"cone clearance {clearance:.6f}m is below {self.config.environment.required_cone_clearance_m:.3f}m"
            )
        return pose

    def command_steering(self, value: float) -> dict[str, Any]:
        response = self.client.command_steering(value)
        if self.samples:
            self.samples[-1]["steering_rad"] = float(value)
        return response


def run_obstacle_expert(
    client: SimClient,
    config: ExpertConfig,
    initial: Preflight,
    plan: BypassPlan,
) -> dict[str, Any]:
    observer = ClearanceObserver(client, initial.route, plan, config)
    control_route = ObstacleAwareRoute(initial.route, plan)
    control_preflight = Preflight(
        initial.world, control_route, initial.route_points, initial.cone_count,
        initial.bounds, initial.pose,
    )
    metrics = run_driver(observer, config.driver, control_preflight)
    rows = observer.samples
    commanded = [row for row in rows if row["steering_rad"] is not None]
    steering = [float(row["steering_rad"]) for row in commanded]
    deltas = [abs(steering[index] - steering[index - 1]) for index in range(1, len(steering))]
    minimum = min(rows, key=lambda row: float(row["cone_clearance_m"]), default=None)
    avoidance_rows = [
        row for row in rows
        if plan.departure_start_s_m <= float(row["route_s_m"]) <= plan.return_end_s_m
    ]
    directional_offsets = [float(row["nominal_signed_cte_m"]) * plan.side_sign for row in avoidance_rows]
    failure = str(metrics.get("failure") or "").lower()
    metrics.update({
        "classification": None,
        "lap_time_s": metrics["elapsed_s"],
        "route_completion_fraction": metrics["total_unwrapped_progress_m"] / metrics["route_length_m"],
        "nominal_route_used_for_progress": True,
        "control_reference": "nominal route outside frozen local bypass; bypass within avoidance span",
        "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.0,
        "minimum_footprint_to_cone_clearance_m": None if minimum is None else minimum["cone_clearance_m"],
        "minimum_cone_clearance_route_s_m": None if minimum is None else minimum["route_s_m"],
        "footprint_cone_intersection_occurred": observer.intersection_occurred,
        "maximum_lateral_avoidance_offset_reached_m": max(directional_offsets, default=0.0),
        "recovery_cte_m": observer.recovery_cte_m,
        "recovery_success": observer.recovery_success,
        "recovery_time_s": observer.recovery_time_s,
        "return_contract": dict(config.return_to_route),
        "control_loop_frequency_hz": 1.0 / metrics["mean_loop_period_s"] if metrics["mean_loop_period_s"] else 0.0,
        "timing_slips": metrics["period_slip_count"],
        "api_failures": int(any(token in failure for token in ("get ", "post ", "unavailable", "control rejected"))),
        "pose_failures": int("pose did not change meaningfully" in failure),
        "clock_failures": int("clock did not advance" in failure or "clock moved backward" in failure),
    })
    metrics["classification"] = classify_obstacle_run(metrics, config)
    return metrics


def classify_obstacle_run(metrics: dict[str, Any], config: ExpertConfig) -> str:
    failure = str(metrics.get("failure") or "").lower()
    if (
        not metrics.get("safe_stop_success", False)
        or metrics.get("api_failures", 0)
        or metrics.get("pose_failures", 0)
        or metrics.get("clock_failures", 0)
        or any(token in failure for token in ("simulator state changed", "unexpected world", "invalid track boundary"))
    ):
        return "INFRA_FAIL"
    clearance = metrics.get("minimum_footprint_to_cone_clearance_m")
    if (
        metrics.get("result") == "PASS"
        and clearance is not None
        and float(clearance) + 1e-9 >= config.environment.required_cone_clearance_m
        and not metrics.get("footprint_cone_intersection_occurred", True)
        and metrics.get("recovery_success") is True
    ):
        return "OBSTACLE_EXPERT_PASS"
    return "OBSTACLE_EXPERT_FAIL"


def run_bounded(
    client: SimClient,
    config: ExpertConfig,
    plan: BypassPlan,
    sim_root: Path,
    result_dir: Path,
    geometry: dict[str, Any],
    *,
    preflight_one: Callable[..., tuple[Preflight, dict[str, Any]]] = full_preflight,
    run_one: Callable[..., dict[str, Any]] = run_obstacle_expert,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    valid = 0
    for number in range(1, MAXIMUM_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight_result = preflight_one(client, config, plan, sim_root, geometry)
            metrics = run_one(client, config, initial, plan)
            classification = str(metrics["classification"])
            if classification != "INFRA_FAIL":
                valid += 1
            attempt = {
                "attempt_number": number,
                "valid_evaluation_number": valid if classification != "INFRA_FAIL" else None,
                "classification": classification,
                "preflight": preflight_result,
                "metrics": metrics,
            }
        except Exception as exc:
            stop_errors = client.safe_stop()
            attempt = {
                "attempt_number": number,
                "valid_evaluation_number": None,
                "classification": "INFRA_FAIL",
                "preflight": {
                    "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                    "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors,
                },
                "metrics": None,
            }
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "OBSTACLE_EXPERT_FAIL":
            return attempts, "FAIL"
        if attempt["classification"] == "OBSTACLE_EXPERT_PASS" and valid == TARGET_VALID_EVALUATIONS:
            return attempts, "PASS"
    return attempts, "INCONCLUSIVE"


def aggregate(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [item["metrics"] for item in attempts if item["classification"] == "OBSTACLE_EXPERT_PASS"]
    if len(runs) != TARGET_VALID_EVALUATIONS:
        raise ExpertError("aggregate requires exactly three obstacle Expert passes")
    lap_times = [float(run["lap_time_s"]) for run in runs]
    clearances = [float(run["minimum_footprint_to_cone_clearance_m"]) for run in runs]
    saturations = [float(run["steering_saturation_fraction"]) for run in runs]
    mean_ctes = [float(run["mean_centerline_error_m"]) for run in runs]
    return {
        "success": "3/3",
        "lap_time_mean_s": statistics.fmean(lap_times),
        "lap_time_sample_std_s": statistics.stdev(lap_times),
        "nominal_cte_mean_m": statistics.fmean(mean_ctes),
        "worst_max_nominal_cte_m": max(float(run["max_centerline_error_m"]) for run in runs),
        "minimum_cone_clearance_across_runs_m": min(clearances),
        "cone_clearance_mean_m": statistics.fmean(clearances),
        "cone_clearance_range_m": [min(clearances), max(clearances)],
        "steering_saturation_mean": statistics.fmean(saturations),
        "steering_saturation_range": [min(saturations), max(saturations)],
        "recovery_success": "3/3",
        "safe_stop_success": "3/3",
    }


def activate_world(client: SimClient, world: str, timeout_s: float = 40.0) -> dict[str, Any]:
    status = client.status()
    if status.get("running") is True and status.get("switching") is False and status.get("current") == world:
        return {"result": "PASS", "action": "already active", "status": status}
    if errors := client.safe_stop():
        raise RuntimeError("pre-switch safe stop failed: " + "; ".join(errors))
    response = client.switch_world(world)
    deadline = time.monotonic() + timeout_s
    last = status
    while time.monotonic() < deadline:
        try:
            last = client.status()
            if last.get("running") is True and last.get("switching") is False and last.get("current") == world:
                if errors := client.safe_stop():
                    raise RuntimeError("post-switch safe stop failed: " + "; ".join(errors))
                return {"result": "PASS", "action": "switched", "response": response, "status": last}
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError(f"derived world did not become ready before timeout: {last}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def audit_preserved_baselines(repo_root: Path) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, str]] = {}
    for name, (relative, expected_hash) in PRESERVED_EVIDENCE.items():
        path = repo_root / relative
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ExpertError(f"preserved evidence identity mismatch for {relative}")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        identities[name] = {"path": relative, "sha256": observed}
    v4 = loaded["pilotnet_v4"]
    v9_live = loaded["temporal_pilotnet_v9_live"]
    v9_training = loaded["temporal_pilotnet_v9_training"]
    expert = loaded["high_speed_expert"]
    if v4.get("result") != "PASS" or v4.get("aggregate", {}).get("policy_success") != "3/3":
        raise ExpertError("preserved PilotNet V4 evidence is not 3/3 PASS")
    if [item.get("classification") for item in v9_live.get("attempts", [])] != ["POLICY_PASS"] * 3:
        raise ExpertError("preserved Temporal PilotNet V9 evidence is not 3/3 PASS")
    architecture = v9_training.get("architecture", {})
    if architecture.get("parameter_count") != 255819 or architecture.get("input_shape") != [9, 66, 200]:
        raise ExpertError("preserved Temporal PilotNet V9 architecture identity mismatch")
    if expert.get("aggregate", {}).get("expert_success") != "3/3":
        raise ExpertError("preserved High-Speed Expert evidence is not 3/3 PASS")
    return {
        "result": "PASS",
        "identities": identities,
        "pilotnet_v4": {"speed_mps": 0.50, "observation": "single camera frame", "success": "3/3"},
        "temporal_pilotnet_v9": {
            "speed_mps": 1.80, "observation": "causal three-frame camera",
            "parameter_count": 255819, "success": "3/3 same-map/same-spawn",
        },
        "high_speed_expert_v1": {
            "speed_mps": 1.80, "lookahead_m": 0.90, "control_frequency_hz": 15.0,
            "max_steering_rad": 0.349066, "wheelbase_m": 0.18, "success": "3/3",
        },
        "preserved_files_unchanged": True,
    }


def _minimum_transition_for_curvature(offset: float, curvature_limit: float) -> float:
    # Quintic smoothstep has max |q''| = 10*sqrt(3)/3.  Ignoring the
    # denominator (1+y'^2)^(3/2) is conservative.
    maximum_second_derivative = 10.0 * math.sqrt(3.0) / 3.0
    return math.sqrt(offset * maximum_second_derivative / curvature_limit)


def _quintic(fraction: float) -> float:
    value = max(0.0, min(1.0, fraction))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _three_point_curvature(first: tuple[float, float], middle: tuple[float, float], last: tuple[float, float]) -> float:
    a, b, c = math.dist(first, middle), math.dist(middle, last), math.dist(first, last)
    if a * b * c <= 1e-15:
        return 0.0
    cross = (middle[0] - first[0]) * (last[1] - middle[1]) - (middle[1] - first[1]) * (last[0] - middle[0])
    return 2.0 * cross / (a * b * c)


def _wrapped(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline-geometry", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument(
        "--demo",
        action="store_true",
        help="run one non-evidentiary visual-check lap without changing frozen experiment results",
    )
    parser.add_argument("--activate-world", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    expected_result = repo_root / (
        EXPECTED_ENVIRONMENT_RESULT_DIRECTORY if args.offline_geometry else EXPECTED_RESULT_DIRECTORY
    )
    if args.result_dir.resolve() != expected_result.resolve():
        print(f"ERROR: result directory must be {expected_result.relative_to(repo_root)}", file=sys.stderr)
        return 2
    marker = args.result_dir / "experiment.started.json"
    summary_path = args.result_dir / "summary.json"
    if args.run and (marker.exists() or summary_path.exists()):
        print("ERROR: refusing to repeat the bounded Cone Avoidance Expert V1 experiment", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INCONCLUSIVE",
    }
    client: SimClient | None = None
    code = 2
    try:
        config = ExpertConfig.load(args.config, repo_root, args.sim_root)
        plan, route_data = build_bypass_plan(config, args.sim_root)
        geometry = validate_geometry(config, plan, route_data)
        report["baseline_audit"] = audit_preserved_baselines(repo_root)
        report["environment_verification"] = verify_derived_environment(
            config.environment, share_path(args.sim_root)
        )
        report["preserved_high_speed_expert"] = {
            "path": str(config.baseline_path),
            "sha256": sha256_file(config.baseline_path),
            "unchanged": True,
        }
        report["offline_geometry"] = geometry
        if args.offline_geometry:
            write_json(args.result_dir / "geometry.json", report)
            write_geometry_plot(args.result_dir / "geometry_plot.png", plan, route_data, geometry)
            report["geometry_plot"] = str(
                (args.result_dir / "geometry_plot.png").resolve().relative_to(repo_root)
            )
            report["result"] = "PASS"
            write_json(args.result_dir / "summary.json", report)
            code = 0
        else:
            client = SimClient(config.driver.base_url, config.driver.api_timeout_s)
            if args.activate_world:
                report["world_activation"] = activate_world(client, config.environment.derived_world)
            if errors := client.safe_stop():
                raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
            if args.preflight_only:
                _, preflight_result = full_preflight(client, config, plan, args.sim_root, geometry)
                report["preflight"] = preflight_result
                report["result"] = "PREFLIGHT_PASS"
                write_json(args.result_dir / "preflight.json", report)
                code = 0
            elif args.demo:
                initial, preflight_result = full_preflight(
                    client, config, plan, args.sim_root, geometry
                )
                metrics = run_obstacle_expert(client, config, initial, plan)
                report["mode"] = "NON_EVIDENTIARY_VISUAL_DEMO"
                report["notice"] = (
                    "This one-lap demo is not part of the frozen 3/3 experiment and is not "
                    "written to the official result directory."
                )
                report["preflight"] = preflight_result
                report["demo_metrics"] = metrics
                report["result"] = str(metrics["classification"])
                code = 0 if report["result"] == "OBSTACLE_EXPERT_PASS" else 1
            else:
                write_json(marker, {
                    "status": "CONE_AVOIDANCE_EXPERT_V1_STARTED_DO_NOT_REPEAT",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "maximum_live_attempts": MAXIMUM_LIVE_ATTEMPTS,
                    "target_valid_evaluations": TARGET_VALID_EVALUATIONS,
                    "frozen_bypass": config.frozen_bypass,
                })
                report["attempts"], report["result"] = run_bounded(
                    client, config, plan, args.sim_root, args.result_dir, geometry
                )
                report["total_live_attempts"] = len(report["attempts"])
                report["valid_evaluations"] = sum(
                    item["classification"] != "INFRA_FAIL" for item in report["attempts"]
                )
                report["aggregate"] = aggregate(report["attempts"]) if report["result"] == "PASS" else None
                report["expert_frozen"] = report["result"] == "PASS"
                report["obstacle_rosbag_collection_justified"] = report["result"] == "PASS"
                write_json(summary_path, report)
                code = 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.preflight_only:
            write_json(args.result_dir / "preflight.json", report)
        elif args.run:
            write_json(args.result_dir / "summary.json", report)
    finally:
        if client is not None:
            errors = client.safe_stop()
            report["final_safe_stop_success"] = not errors
            report["final_safe_stop_errors"] = errors
            if errors:
                report["result"] = "INCONCLUSIVE"
                code = 2
            if args.run:
                write_json(summary_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
