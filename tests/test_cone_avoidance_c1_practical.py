from __future__ import annotations

from pathlib import Path

import pytest

from physicar_e2e.cone_avoidance_c1_practical import (
    MAXIMUM_TOTAL_ATTEMPTS,
    RESULT_DIRECTORY,
    TARGET_VALID_PASSES,
    audit_practical_frozen,
    classify_practical_cone_run,
    load_practical_contract,
    run_practical_attempts,
)
from physicar_e2e.cone_avoidance_temporal_c1 import WORLD, load_c1_inference_config


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")


def _passing_run(**updates) -> dict:
    run = {
        "result": "PASS", "temporal_input_failure": False, "api_failures": 0,
        "liveness_failures": 0, "safe_stop_success": True,
        "minimum_footprint_to_cone_clearance_m": .043,
        "vehicle_cone_collision_or_intersection_occurred": False,
        "recovery_success": True,
    }
    run.update(updates)
    return run


def test_historical_result_and_frozen_c1_identities_are_unchanged() -> None:
    if not SIM_ROOT.is_dir():
        pytest.skip("simulator asset checkout unavailable")
    audit = audit_practical_frozen(REPO, SIM_ROOT)
    assert audit["result"] == "PASS"
    assert audit["historical_result"] == {
        "classification": "FAIL under the 5 cm experimental clearance contract",
        "minimum_clearance_m": pytest.approx(.043582666949213776),
        "intersection": False,
    }
    assert audit["checkpoint"]["sha256"] == "1e90002ca139b3cfb0f34074e013e52b6754df33ed0e3b438ca81809c9e2ee39"
    assert audit["onnx"]["sha256"] == "22440ad61f6e5136b33016eb0781d79ab71637e659478ac0c92cc04cffc98e5f"
    assert audit["architecture"]["input_shape"] == [9, 66, 200]
    assert audit["architecture"]["parameter_count"] == 255_819
    assert audit["world"] == WORLD
    assert audit["cone"] == {
        "route_s_m": pytest.approx(6.9),
        "x_m": pytest.approx(6.165700204349249),
        "y_m": pytest.approx(1.2298027858176892),
    }


def test_validation_only_contract_forbids_training_collection_dagger_and_parameter_changes() -> None:
    contract = load_practical_contract(REPO)
    assert contract["result_directory"] == RESULT_DIRECTORY
    for key in (
        "training_permitted", "data_collection_permitted", "dagger_permitted",
        "world_changes_permitted", "cone_pose_changes_permitted", "speed_changes_permitted",
        "architecture_changes_permitted", "steering_changes_permitted",
    ):
        assert contract[key] is False
    assert contract["speed_mps"] == 1.8
    assert contract["history_frames"] == 3
    assert contract["input_shape"] == [9, 66, 200]
    assert contract["control_frequency_hz"] == 15.0
    assert contract["maximum_steering_rad"] == .349066


def test_practical_contract_measures_but_does_not_enforce_five_cm_margin() -> None:
    run = _passing_run(minimum_footprint_to_cone_clearance_m=.000001)
    assert classify_practical_cone_run(run) == "PRACTICAL_CONE_PASS"
    assert run["minimum_footprint_to_cone_clearance_m"] < .05


def test_actual_intersection_is_policy_failure() -> None:
    assert classify_practical_cone_run(
        _passing_run(
            minimum_footprint_to_cone_clearance_m=0.0,
            vehicle_cone_collision_or_intersection_occurred=True,
        )
    ) == "PRACTICAL_CONE_FAIL"


def test_road_temporal_and_control_safety_remain_identical_to_c1() -> None:
    practical = load_practical_contract(REPO)
    inference = load_c1_inference_config(REPO).payload
    assert practical["expected_world"] == inference["expected_world"]
    assert practical["speed_mps"] == inference["smoke_speeds_mps"][0]
    assert practical["history_frames"] == inference["history_frames"]
    assert practical["input_shape"] == [inference["input_channels"], inference["model_height"], inference["model_width"]]
    assert practical["control_frequency_hz"] == inference["control_frequency_hz"]
    assert practical["maximum_steering_rad"] == inference["max_steering_rad"]
    assert inference["off_track_margin_m"] == .05
    assert inference["off_track_grace_s"] == .50
    assert inference["return_maximum_absolute_nominal_cte_m"] == .05
    assert inference["return_minimum_stable_duration_s"] == .50


class FakeClient:
    def __init__(self) -> None:
        self.safe_stop_calls = 0

    def safe_stop(self):
        self.safe_stop_calls += 1
        return []


def _attempt_run(classification: str) -> dict:
    return {"classification": classification}


def test_run_one_genuine_failure_stops_without_retry(tmp_path: Path) -> None:
    calls = 0

    def preflight(*args):
        return object(), {"result": "PASS"}

    def run(*args):
        nonlocal calls
        calls += 1
        return _attempt_run("PRACTICAL_CONE_FAIL")

    attempts, result = run_practical_attempts(
        FakeClient(), object(), object(), object(), object(), {}, Path("/sim"), tmp_path,
        preflight_one=preflight, run_one=run,
    )
    assert result == "FAIL"
    assert calls == len(attempts) == 1
    assert attempts[0]["valid_policy_run_number"] == 1


def test_conditional_three_run_logic_and_bounded_infrastructure_replacement(tmp_path: Path) -> None:
    preflight_calls = 0

    def preflight(*args):
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls <= 2:
            raise RuntimeError("temporary preflight failure")
        return object(), {"result": "PASS"}

    def run(*args):
        return _attempt_run("PRACTICAL_CONE_PASS")

    attempts, result = run_practical_attempts(
        FakeClient(), object(), object(), object(), object(), {}, Path("/sim"), tmp_path,
        preflight_one=preflight, run_one=run,
    )
    assert result == "PASS"
    assert TARGET_VALID_PASSES == 3
    assert len(attempts) == MAXIMUM_TOTAL_ATTEMPTS == 5
    assert [item["classification"] for item in attempts] == [
        "INFRA_FAIL", "INFRA_FAIL", "PRACTICAL_CONE_PASS", "PRACTICAL_CONE_PASS", "PRACTICAL_CONE_PASS",
    ]


def test_temporal_infrastructure_and_safe_stop_classification() -> None:
    assert classify_practical_cone_run(_passing_run(temporal_input_failure=True)) == "TEMPORAL_INPUT_FAIL"
    assert classify_practical_cone_run(_passing_run(api_failures=1)) == "INFRA_FAIL"
    assert classify_practical_cone_run(_passing_run(liveness_failures=1)) == "INFRA_FAIL"
    assert classify_practical_cone_run(_passing_run(safe_stop_success=False)) == "INFRA_FAIL"
    assert classify_practical_cone_run(_passing_run(result="FAIL", recovery_success=False)) == "PRACTICAL_CONE_FAIL"
