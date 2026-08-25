from __future__ import annotations

from pathlib import Path

import pytest

import physicar_e2e.random_cone_expert as random_cone
from physicar_e2e.cone_avoidance_environment import asset_set, share_path


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")
STRESS_CONFIG = REPO / "configs" / "random_cone_expert_v1.json"
TARGET_CONFIG = REPO / "configs" / "random_cone_expert_1p0_v1.json"
TARGET_SUMMARY = REPO / "results" / "random_cone_expert_1p0_v1" / "summary.json"


@pytest.fixture(scope="module")
def operational_bundle():
    share = share_path(SIM_ROOT)
    if not asset_set(
        share, "custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1"
    ).route.is_file():
        pytest.skip("read-only simulator geometry checkout is unavailable")
    stress = random_cone.RandomConeConfig.load(STRESS_CONFIG, REPO, SIM_ROOT)
    target = random_cone.RandomConeConfig.load(TARGET_CONFIG, REPO, SIM_ROOT)
    bundles = random_cone.verify_frozen_scenarios(target, SIM_ROOT)
    return stress, target, bundles


def test_target_inherits_the_exact_stress_scenarios_and_roles(operational_bundle) -> None:
    stress, target, bundles = operational_bundle
    assert target.version == random_cone.OPERATIONAL_VERSION
    assert target.random_seed == stress.random_seed == 20260825
    assert target.scenarios == stress.scenarios
    assert tuple(item.scenario for item in bundles) == stress.scenarios
    assert [item.scenario.role for item in bundles] == (
        ["TRAIN"] * 8 + ["VALIDATION"] * 2 + ["UNSEEN_HOLDOUT"] * 2
    )
    assert target.worlds == stress.worlds


def test_only_speed_changes_in_the_operational_control_contract(operational_bundle) -> None:
    stress, target, _ = operational_bundle
    assert stress.baseline.fixed_speed_mps == 1.8
    assert target.baseline.fixed_speed_mps == 1.0
    assert (
        target.baseline.lookahead_m,
        target.baseline.control_frequency_hz,
        target.baseline.max_steering_rad,
        target.baseline.wheelbase_m,
    ) == (0.90, 15.0, 0.349066, 0.18)
    assert target.baseline == stress.baseline.__class__(
        **{**stress.baseline.__dict__, "fixed_speed_mps": 1.0}
    )


def test_same_bypass_algorithm_is_offline_feasible_for_all_twelve(operational_bundle) -> None:
    _, target, bundles = operational_bundle
    assert target.sampling["maximum_continuous_steering_saturation_s"] == 0.90
    assert all(item.geometry["result"] == "PASS" for item in bundles)
    assert all(
        item.geometry["algorithm"] == "automatic_symmetric_quintic_route_normal_v1"
        for item in bundles
    )
    assert all(item.plan.transition_length_m == pytest.approx(1.8) for item in bundles)
    assert all(item.plan.plateau_half_length_m == pytest.approx(0.9) for item in bundles)
    assert {item.plan.side for item in bundles} == {"left", "right"}
    assert all(
        item.geometry["kinematic_preview"]["minimum_footprint_to_cone_clearance_m"] > 0.0
        for item in bundles
    )
    assert all(
        not item.geometry["kinematic_preview"]["footprint_cone_intersection"]
        for item in bundles
    )
    scenario_01 = bundles[0].geometry["kinematic_preview"]
    assert 0.50 < scenario_01["maximum_continuous_steering_saturation_s"] <= 0.90


def test_stress_evidence_is_hash_gated_and_still_the_registered_failure(
    operational_bundle,
) -> None:
    _, target, _ = operational_bundle
    audit = random_cone.audit_preserved_state(REPO, SIM_ROOT, target)
    stress = audit["random_cone_expert_1p8_stress"]
    assert stress["result"] == "PASS"
    assert stress["success"] == "11/12"
    assert stress["cone_contact_or_intersection"] == "0/12"
    assert stress["failure_scenario_id"] == "01"
    assert stress["failure_mode"] == "return_off_track"
    assert stress["evidence_unchanged"]


def test_target_namespace_and_future_neural_boundary_are_separate(
    operational_bundle,
) -> None:
    _, target, _ = operational_bundle
    assert target.result_directory == "results/random_cone_expert_1p0_v1"
    assert target.result_directory != random_cone.EXPECTED_RESULT_DIRECTORY
    assert target.live_protocol["maximum_valid_policy_runs"] == 12
    assert target.live_protocol["retry_genuine_policy_failure"] is False
    assert target.permissions["neural_training_permitted"] is False
    assert target.permissions["training_bag_collection_permitted"] is False
    assert target.permissions["stress_baseline_changes_permitted"] is False
    assert target.payload["future_neural_split"] == {
        "TRAIN": ["01", "02", "03", "04", "05", "06", "07", "08"],
        "VALIDATION": ["09", "10"],
        "UNSEEN_HOLDOUT": ["11", "12"],
        "holdout_training_or_tuning_permitted": False,
    }


def test_official_target_evidence_is_twelve_of_twelve_collision_free() -> None:
    summary = random_cone._read_json(TARGET_SUMMARY)
    scenarios = summary["scenarios"]
    assert summary["version"] == random_cone.OPERATIONAL_VERSION
    assert summary["result"] == summary["live_result"] == "PASS"
    assert summary["aggregate"]["success"] == "12/12"
    assert summary["aggregate"]["valid_policy_runs"] == 12
    assert summary["aggregate"]["cone_contact_or_intersection_count"] == 0
    assert summary["aggregate"]["minimum_actual_clearance_m"] > 0.0
    assert summary["random_cone_expert_frozen"] is True
    assert summary["random_cone_bag_collection_justified"] is True
    assert summary["neural_training_performed"] is False
    assert summary["training_bags_collected"] is False
    assert len(scenarios) == 12
    assert all(item["valid_policy_run_count"] == 1 for item in scenarios)
    assert all(item["result"] == "RANDOM_CONE_EXPERT_PASS" for item in scenarios)
    assert all(
        item["valid_policy_run"]["metrics"]["cone_contact_or_intersection_occurred"]
        is False
        for item in scenarios
    )
    scenario_08 = scenarios[7]
    assert 0.0 < scenario_08["valid_policy_run"]["metrics"][
        "minimum_footprint_to_cone_clearance_m"
    ] < 0.05
    assert [attempt["classification"] for attempt in scenario_08["attempts"]] == [
        "INFRA_FAIL", "RANDOM_CONE_EXPERT_PASS"
    ]
