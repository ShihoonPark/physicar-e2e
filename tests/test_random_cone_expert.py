from __future__ import annotations

import copy
from pathlib import Path

import pytest

import physicar_e2e.random_cone_expert as random_cone
from physicar_e2e.cone_avoidance_environment import asset_set, load_route, share_path
from physicar_e2e.expert_driver import Preflight


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")
CONFIG = REPO / "configs" / "random_cone_expert_v1.json"


@pytest.fixture(scope="module")
def frozen_bundle():
    share = share_path(SIM_ROOT)
    if not asset_set(share, "custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1").route.is_file():
        pytest.skip("read-only simulator geometry checkout is unavailable")
    config = random_cone.RandomConeConfig.load(CONFIG, REPO, SIM_ROOT)
    bundles = random_cone.verify_frozen_scenarios(config, SIM_ROOT)
    route = load_route(asset_set(share, config.environment.canonical_cone_free_world).route)
    return config, bundles, route


def test_fixed_seed_exact_twelve_and_exact_roles_are_frozen(frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    assert config.random_seed == 20260825
    assert len(config.scenarios) == len(bundles) == 12
    assert [item.scenario.scenario_id for item in bundles] == [f"{number:02d}" for number in range(1, 13)]
    assert [item.scenario.role for item in bundles] == ["TRAIN"] * 8 + ["VALIDATION"] * 2 + ["UNSEEN_HOLDOUT"] * 2
    assert config.payload["scenario_roles"] == random_cone.ROLE_IDS
    assert all(item.scenario.provenance["random_seed"] == config.random_seed for item in bundles)
    assert all(item.scenario.provenance["algorithm"] == config.sampling["algorithm"] for item in bundles)


def test_same_original_map_route_spawn_vehicle_and_control(frozen_bundle) -> None:
    config, bundles, route_data = frozen_bundle
    assert config.payload["map_family"] == random_cone.MAP_FAMILY
    assert config.environment.canonical_cone_world == "custom_71e69ee938032295503bfed557fde18c"
    assert config.environment.canonical_cone_free_world.endswith("_e2e_lane_follow_v1")
    assert config.baseline.expected_world == config.environment.canonical_cone_free_world
    assert route_data.route.length == pytest.approx(30.50461103699958)
    assert (
        config.baseline.fixed_speed_mps,
        config.baseline.lookahead_m,
        config.baseline.control_frequency_hz,
        config.baseline.max_steering_rad,
        config.baseline.wheelbase_m,
    ) == (1.80, 0.90, 15.0, 0.349066, 0.18)
    assert all(config.driver_for(item.scenario).expected_world == config.world_name(item.scenario.scenario_id) for item in bundles)


def test_no_spawn_end_or_fixed_cone_placement_and_sufficient_span(frozen_bundle) -> None:
    config, bundles, route_data = frozen_bundle
    fixed_s = config.environment.frozen_cone["route_s_m"]
    for item in bundles:
        plan = item.plan
        assert plan.departure_start_s_m >= config.sampling["start_exclusion_m"]
        assert plan.return_end_s_m <= route_data.route.length - config.sampling["end_exclusion_m"]
        assert abs(plan.cone_s_m - fixed_s) >= config.sampling["fixed_cone_exclusion_m"]
        assert plan.cone_s_m - plan.departure_start_s_m > config.baseline.lookahead_m
        assert plan.return_end_s_m - plan.cone_s_m > config.baseline.lookahead_m


def test_curvature_and_side_diversity_are_geometry_selected(frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    counts = {
        name: sum(item.scenario.curvature_class == name for item in bundles)
        for name in random_cone.CLASS_ORDER
    }
    assert counts == config.sampling["target_count_by_class"] == {
        "low_curvature": 5,
        "moderate_left_curve": 4,
        "moderate_right_curve": 3,
    }
    assert {item.plan.side for item in bundles} == {"left", "right"}
    for item in bundles:
        feasible = {
            side: details for side, details in item.side_evaluations.items()
            if details["result"] == "PASS"
        }
        assert item.plan.side in feasible
        chosen = tuple(feasible[item.plan.side]["score"])
        assert chosen == max(tuple(details["score"]) for details in feasible.values())


def test_world_conflicts_nonlocal_ambiguity_and_track_clearance_are_gated(frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    for item in bundles:
        scenario = item.scenario
        geometry = item.geometry
        assert scenario.nearby_collision_clearance_m > 0.0
        assert scenario.nonlocal_route_clearance_m >= config.sampling["minimum_nonlocal_route_clearance_m"]
        assert scenario.left_logical_track_clearance_m > 0.0
        assert scenario.right_logical_track_clearance_m > 0.0
        assert geometry["minimum_reference_center_track_clearance_m"] >= 0.0
        assert geometry["gates"]["cone_clear_of_nearby_world_collisions"]
        assert geometry["gates"]["reference_center_track_feasible"]


def test_real_footprint_clearance_and_practical_collision_only_gate(frozen_bundle) -> None:
    _, bundles, _ = frozen_bundle
    footprint = bundles[0].plan.footprint
    cone = bundles[0].plan.cone
    assert footprint.length_m == pytest.approx(0.270000)
    assert footprint.width_m == pytest.approx(0.21854076122953137)
    assert cone.size_xyz_m[:2] == (0.18, 0.18)
    for item in bundles:
        geometry = item.geometry
        assert geometry["result"] == "PASS"
        assert geometry["minimum_planned_footprint_to_cone_clearance_m"] >= 0.055 - 1e-9
        assert geometry["kinematic_preview"]["minimum_footprint_to_cone_clearance_m"] > 0.0
        assert not geometry["planned_footprint_cone_intersection"]
        assert not geometry["kinematic_preview"]["footprint_cone_intersection"]
    passing = {
        "result": "PASS", "failure": None, "safe_stop_success": True,
        "api_failures": 0, "pose_failures": 0, "clock_failures": 0,
        "minimum_footprint_to_cone_clearance_m": 0.001,
        "cone_contact_or_intersection_occurred": False,
        "recovery_success": True, "off_track_event_count": 0,
    }
    assert random_cone.classify_random_cone_run(passing) == "RANDOM_CONE_EXPERT_PASS"
    assert random_cone.classify_random_cone_run({**passing, "minimum_footprint_to_cone_clearance_m": 0.0}) == "RANDOM_CONE_EXPERT_FAIL"
    assert random_cone.classify_random_cone_run({**passing, "cone_contact_or_intersection_occurred": True}) == "RANDOM_CONE_EXPERT_FAIL"
    assert random_cone.classify_random_cone_run({**passing, "safe_stop_success": False}) == "INFRA_FAIL"


def test_same_automatic_bypass_algorithm_and_steering_gate_for_every_location(frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    base_offset = bundles[0].plan.required_center_offset_m + config.avoidance["extra_planning_clearance_m"]
    for item in bundles:
        plan = item.plan
        geometry = item.geometry
        assert geometry["algorithm"] == "automatic_symmetric_quintic_route_normal_v1"
        assert geometry["profile"] == "symmetric_quintic_with_flat_pass"
        assert plan.required_center_offset_m == pytest.approx(bundles[0].plan.required_center_offset_m)
        increments = (plan.maximum_lateral_offset_m - base_offset) / config.avoidance["offset_search_step_m"]
        assert increments == pytest.approx(round(increments), abs=1e-8)
        assert plan.transition_length_m == pytest.approx(1.8)
        assert plan.plateau_half_length_m == pytest.approx(0.9)
        assert geometry["gates"]["added_profile_curvature_within_design_limit"]
        assert geometry["gates"]["local_combined_curvature_within_steering_limit"]
        assert geometry["gates"]["preview_steering_feasible"]
        assert geometry["kinematic_preview"]["maximum_applied_steering_rad"] <= config.baseline.max_steering_rad


def test_config_alone_reconstructs_exactly_one_cone_per_unique_world(frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    share = share_path(SIM_ROOT)
    assert len({config.world_name(item.scenario.scenario_id) for item in bundles}) == 12
    assert len({config.cone_model_name(item.scenario.scenario_id) for item in bundles}) == 12
    assert all(random_cone._expected_cone_count(config, item.plan, item.scenario.scenario_id, share) == 1 for item in bundles)


def test_frozen_position_mismatch_is_rejected_without_replacement(frozen_bundle) -> None:
    _, bundles, _ = frozen_bundle
    frozen = bundles[0].scenario.to_dict()
    changed = copy.deepcopy(frozen)
    changed["route_s_m"] += 0.05
    with pytest.raises(random_cone.RandomConeError, match="differs from frozen config"):
        random_cone._compare_frozen(changed, frozen, 1e-8, "01")


def test_no_training_bags_models_commits_or_pushes_are_permitted(frozen_bundle) -> None:
    config, _, _ = frozen_bundle
    assert config.permissions == {
        "neural_training_permitted": False,
        "training_bag_collection_permitted": False,
        "v9_model_changes_permitted": False,
        "c1_model_changes_permitted": False,
        "fixed_cone_evidence_changes_permitted": False,
        "tracked_simulator_source_changes_permitted": False,
        "commit_permitted": False,
        "push_permitted": False,
    }
    audit = random_cone.audit_preserved_state(REPO, SIM_ROOT)
    assert audit["result"] == "PASS"
    assert audit["temporal_pilotnet_v9"]["success"] == "3/3 cone-free"
    assert audit["fixed_cone_expert_v1"]["success"] == "3/3"
    assert audit["temporal_pilotnet_c1"]["cone_contact_or_intersection"] == "0/3"


class _FakeClient:
    def __init__(self) -> None:
        self.safe_stop_calls = 0

    def safe_stop(self):
        self.safe_stop_calls += 1
        return []


def test_live_protocol_runs_once_each_no_policy_retry_and_safe_stops(tmp_path: Path, frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    client = _FakeClient()
    calls: list[str] = []

    def activate(_client, world):
        return {"result": "PASS", "world": world}

    def preflight(_client, _config, bundle, _sim_root):
        return Preflight(
            _config.world_name(bundle.scenario.scenario_id), bundle.plan.nominal,
            len(bundle.plan.nominal.points), 1, {}, {"x": 0.0, "y": 0.0, "yaw": 0.0},
        ), {"result": "PASS"}

    def run(_client, _config, _initial, bundle):
        calls.append(bundle.scenario.scenario_id)
        classification = "RANDOM_CONE_EXPERT_FAIL" if bundle.scenario.scenario_id == "01" else "RANDOM_CONE_EXPERT_PASS"
        return {"classification": classification}

    records, result = random_cone.run_live_benchmark(
        client, config, bundles, SIM_ROOT, tmp_path,
        activate_one=activate, preflight_one=preflight, run_one=run,
    )
    assert result == "FAIL"
    assert calls == [f"{number:02d}" for number in range(1, 13)]
    assert len(records) == 12
    assert all(len(item["attempts"]) == 1 for item in records)
    assert sum(item["valid_policy_run_count"] for item in records) == 12
    assert client.safe_stop_calls >= 2 * 12


def test_one_fresh_infrastructure_replacement_is_bounded(tmp_path: Path, frozen_bundle) -> None:
    config, bundles, _ = frozen_bundle
    client = _FakeClient()
    counts: dict[str, int] = {}

    def activate(_client, world):
        return {"result": "PASS", "world": world}

    def preflight(_client, _config, bundle, _sim_root):
        scenario_id = bundle.scenario.scenario_id
        counts[scenario_id] = counts.get(scenario_id, 0) + 1
        if scenario_id == "01" and counts[scenario_id] == 1:
            raise RuntimeError("temporary infrastructure failure")
        return Preflight(
            _config.world_name(scenario_id), bundle.plan.nominal,
            len(bundle.plan.nominal.points), 1, {}, {"x": 0.0, "y": 0.0, "yaw": 0.0},
        ), {"result": "PASS"}

    def run(*_args):
        return {"classification": "RANDOM_CONE_EXPERT_PASS"}

    records, result = random_cone.run_live_benchmark(
        client, config, bundles, SIM_ROOT, tmp_path,
        activate_one=activate, preflight_one=preflight, run_one=run,
    )
    assert result == "PASS"
    assert len(records[0]["attempts"]) == 2
    assert records[0]["attempts"][0]["classification"] == "INFRA_FAIL"
    assert records[0]["attempts"][1]["classification"] == "RANDOM_CONE_EXPERT_PASS"
    assert sum(item["valid_policy_run_count"] for item in records) == 12
