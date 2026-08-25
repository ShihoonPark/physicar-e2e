from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from physicar_e2e.cone_avoidance_environment import (
    asset_set,
    box_polygon,
    footprint_polygon,
    polygon_clearance,
    share_path,
)
from physicar_e2e.cone_avoidance_expert import (
    MAXIMUM_LIVE_ATTEMPTS,
    ObstacleAwareRoute,
    ExpertConfig,
    audit_preserved_baselines,
    build_bypass_plan,
    classify_obstacle_run,
    run_bounded,
    validate_geometry,
)
from physicar_e2e.expert_driver import Preflight


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")
CONFIG = REPO / "configs" / "cone_avoidance_expert_v1.json"


@pytest.fixture(scope="module")
def geometry_bundle():
    if not SIM_ROOT.is_dir():
        pytest.skip("read-only simulator asset checkout is unavailable")
    config = ExpertConfig.load(CONFIG, REPO, SIM_ROOT)
    plan, route_data = build_bypass_plan(config, SIM_ROOT)
    geometry = validate_geometry(config, plan, route_data)
    return config, plan, route_data, geometry


def test_fixed_high_speed_contract_and_privileged_boundary(geometry_bundle) -> None:
    config, _, _, _ = geometry_bundle
    assert config.driver.fixed_speed_mps == 1.80
    assert config.driver.lookahead_m == 0.90
    assert config.driver.control_frequency_hz == 15.0
    assert config.driver.max_steering_rad == 0.349066
    assert config.driver.wheelbase_m == 0.18
    assert config.driver.expected_world == config.environment.derived_world
    assert config.payload["future_policy_forbidden_inputs"] == [
        "cone_gt_coordinates", "route", "pose", "cte", "expert_command"
    ]


def test_preserved_baseline_evidence_audit() -> None:
    audit = audit_preserved_baselines(REPO)
    assert audit["result"] == "PASS"
    assert audit["pilotnet_v4"]["success"] == "3/3"
    assert audit["temporal_pilotnet_v9"]["parameter_count"] == 255819
    assert audit["high_speed_expert_v1"]["lookahead_m"] == 0.90


def test_geometric_offset_is_footprint_aware_and_has_frozen_margin(geometry_bundle) -> None:
    config, plan, _, _ = geometry_bundle
    expected = max(abs(plan.footprint.y_min_m), abs(plan.footprint.y_max_m)) + plan.cone.half_width_m + 0.05
    assert plan.required_center_offset_m == pytest.approx(expected)
    assert plan.maximum_lateral_offset_m == pytest.approx(expected + 0.005)
    cone = box_polygon(plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
                       plan.cone.half_length_m, plan.cone.half_width_m)
    point = plan.point_at(plan.cone_s_m)
    vehicle = footprint_polygon(plan.footprint, *point, plan.yaw_at(plan.cone_s_m))
    clearance, intersects = polygon_clearance(vehicle, cone)
    assert not intersects
    assert clearance >= config.environment.required_cone_clearance_m


def test_smooth_quintic_bypass_returns_exactly_to_nominal(geometry_bundle) -> None:
    _, plan, _, geometry = geometry_bundle
    assert plan.lateral_offset(plan.departure_start_s_m) == 0.0
    assert plan.lateral_offset(plan.return_end_s_m) == 0.0
    assert math.dist(plan.point_at(plan.departure_start_s_m), plan.nominal.point_at(plan.departure_start_s_m)) < 1e-12
    assert math.dist(plan.point_at(plan.return_end_s_m), plan.nominal.point_at(plan.return_end_s_m)) < 1e-12
    assert geometry["start_heading_error_rad"] < 1e-4
    assert geometry["end_heading_error_rad"] < 1e-4
    assert geometry["gates"]["smooth_start_end_position"]
    assert geometry["gates"]["smooth_start_end_heading"]


def test_reference_clearance_track_containment_and_curvature_gates(geometry_bundle) -> None:
    config, plan, _, geometry = geometry_bundle
    assert geometry["result"] == "PASS"
    assert geometry["minimum_planned_footprint_to_cone_clearance_m"] >= 0.05
    assert geometry["minimum_reference_center_track_clearance_m"] > 0.0
    assert geometry["maximum_reference_curvature_per_m"] < plan.physical_curvature_limit_per_m
    assert geometry["maximum_reference_equivalent_steering_rad"] < config.driver.max_steering_rad
    assert not geometry["footprint_intersection"]


def test_avoidance_span_is_geometry_derived_and_on_frozen_straight(geometry_bundle) -> None:
    config, plan, _, geometry = geometry_bundle
    assert plan.transition_length_m == max(
        plan.geometric_minimum_transition_m,
        config.driver.fixed_speed_mps * config.avoidance["transition_minimum_time_s"],
        config.driver.lookahead_m * config.avoidance["transition_minimum_lookaheads"],
    )
    assert plan.departure_start_s_m >= plan.site.straight_run_start_s_m
    assert plan.return_end_s_m <= plan.site.straight_run_end_s_m
    assert geometry["total_avoidance_span_m"] == pytest.approx(5.4)


def test_original_nominal_route_is_used_for_projection_and_progress(geometry_bundle) -> None:
    _, plan, _, _ = geometry_bundle
    adapter = ObstacleAwareRoute(plan.nominal, plan)
    probe = plan.nominal.point_at(plan.cone_s_m)
    assert adapter.project(probe) == plan.nominal.project(probe)
    assert adapter.length == plan.nominal.length
    assert adapter.points == plan.nominal.points
    assert math.dist(adapter.point_at(plan.cone_s_m), plan.nominal.point_at(plan.cone_s_m)) > 0.20


def test_return_to_route_criterion_is_preregistered(geometry_bundle) -> None:
    config, _, _, geometry = geometry_bundle
    assert config.return_to_route == {
        "maximum_absolute_nominal_cte_m": 0.05,
        "minimum_stable_duration_s": 0.50,
    }
    assert geometry["kinematic_preview"]["recovery_success"] is True


def test_classification_requires_clearance_recovery_and_safe_stop(geometry_bundle) -> None:
    config, _, _, _ = geometry_bundle
    passing = {
        "result": "PASS", "failure": None, "safe_stop_success": True,
        "api_failures": 0, "pose_failures": 0, "clock_failures": 0,
        "minimum_footprint_to_cone_clearance_m": 0.05,
        "footprint_cone_intersection_occurred": False, "recovery_success": True,
    }
    assert classify_obstacle_run(passing, config) == "OBSTACLE_EXPERT_PASS"
    for key, value in (
        ("minimum_footprint_to_cone_clearance_m", 0.049),
        ("footprint_cone_intersection_occurred", True),
        ("recovery_success", False),
        ("safe_stop_success", False),
    ):
        failed = {**passing, key: value}
        expected = "INFRA_FAIL" if key == "safe_stop_success" else "OBSTACLE_EXPERT_FAIL"
        assert classify_obstacle_run(failed, config) == expected


class _FakeClient:
    def __init__(self) -> None:
        self.safe_stop_calls = 0

    def safe_stop(self):
        self.safe_stop_calls += 1
        return []


def _initial(plan) -> Preflight:
    return Preflight("world", plan.nominal, len(plan.nominal.points), 1, {}, {"x": 0, "y": 0, "yaw": 0})


def test_primary_obstacle_failure_stops_repeatability(tmp_path: Path, geometry_bundle) -> None:
    config, plan, _, geometry = geometry_bundle
    client = _FakeClient()
    calls = []

    def preflight(*args):
        return _initial(plan), {"result": "PASS"}

    def run(*args):
        calls.append(1)
        return {"classification": "OBSTACLE_EXPERT_FAIL"}

    attempts, result = run_bounded(
        client, config, plan, SIM_ROOT, tmp_path, geometry,
        preflight_one=preflight, run_one=run,
    )
    assert result == "FAIL"
    assert len(attempts) == len(calls) == 1


def test_three_runs_are_conditional_and_identical_contract(tmp_path: Path, geometry_bundle) -> None:
    config, plan, _, geometry = geometry_bundle
    seen = []

    def preflight(*args):
        return _initial(plan), {"result": "PASS"}

    def run(_client, runtime_config, _initial_state, runtime_plan):
        seen.append((runtime_config.driver, runtime_plan))
        return {"classification": "OBSTACLE_EXPERT_PASS"}

    attempts, result = run_bounded(
        _FakeClient(), config, plan, SIM_ROOT, tmp_path, geometry,
        preflight_one=preflight, run_one=run,
    )
    assert result == "PASS"
    assert len(attempts) == len(seen) == 3
    assert all(driver == config.driver and runtime_plan is plan for driver, runtime_plan in seen)


def test_infrastructure_failures_have_bounded_replacement_semantics(tmp_path: Path, geometry_bundle) -> None:
    config, plan, _, geometry = geometry_bundle
    count = 0

    def preflight(*args):
        nonlocal count
        count += 1
        if count <= 2:
            raise RuntimeError("temporary infrastructure failure")
        return _initial(plan), {"result": "PASS"}

    def run(*args):
        return {"classification": "OBSTACLE_EXPERT_PASS"}

    attempts, result = run_bounded(
        _FakeClient(), config, plan, SIM_ROOT, tmp_path, geometry,
        preflight_one=preflight, run_one=run,
    )
    assert result == "PASS"
    assert len(attempts) == MAXIMUM_LIVE_ATTEMPTS
    assert [item["classification"] for item in attempts[:2]] == ["INFRA_FAIL", "INFRA_FAIL"]
