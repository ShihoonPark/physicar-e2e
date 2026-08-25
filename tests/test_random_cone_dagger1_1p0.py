from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from physicar_e2e.pilotnet_temporal import build_temporal_pilotnet
from physicar_e2e.random_cone_dagger1 import (
    DAGGER_EPISODES,
    HOLDOUT_SCENARIOS,
    TRAIN_SCENARIOS,
    VALIDATION_SCENARIOS,
    collection_gate,
    episode_specs,
    frozen_teacher_label,
    freeze_seal_contract,
    latest_causal_teacher_index,
    live_retry_decision,
    load_config,
    next_holdout,
    next_validation,
    teacher_row_usable,
    temporal_triplet_gaps,
    validation_allows_unseen,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/random_cone_dagger1_1p0_v1.json"


def _valid_record(scenario: str, *, outcome: str = "RANDOM_CONE_POLICY_PASS") -> dict:
    episode = f"dagger1_s{scenario}_r01"
    return {
        "episode_id": episode,
        "scenario_id": scenario,
        "classification": "DAGGER_EVIDENCE_PASS",
        "policy_outcome": outcome,
        "r1_controls_vehicle": True,
        "teacher_control_authority": False,
        "post_run_safe_stop_success": True,
        "final_safe_stop_success": True,
    }


def test_exact_train_only_dagger_collection_contract() -> None:
    config = load_config(CONFIG, REPO)
    assert tuple(item.scenario_id for item in episode_specs()) == TRAIN_SCENARIOS
    assert tuple(item.episode_id for item in episode_specs()) == DAGGER_EPISODES
    assert not set(TRAIN_SCENARIOS) & set(VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS)
    assert config.payload["permissions"]["dagger_scenarios"] == list(TRAIN_SCENARIOS)
    assert config.payload["permissions"]["validation_or_holdout_dagger_permitted"] is False
    assert config.payload["permissions"]["holdout_bag_collection_permitted"] is False


def test_r1_controls_and_teacher_stream_is_separate() -> None:
    records = [_valid_record(scenario) for scenario in TRAIN_SCENARIOS]
    gate = collection_gate(records)
    assert gate["result"] == "PASS"
    assert gate["gates"]["r1_controlled_all_rollouts"]
    assert gate["gates"]["teacher_never_controlled"]


def test_genuine_r1_policy_failure_is_valid_dagger_evidence() -> None:
    records = [_valid_record(scenario) for scenario in TRAIN_SCENARIOS]
    records[3] = _valid_record("04", outcome="RANDOM_CONE_POLICY_FAIL")
    assert collection_gate(records)["result"] == "PASS"
    assert live_retry_decision("RANDOM_CONE_POLICY_FAIL", 1) == "STOP_GENUINE_FAILURE"


class _Nominal:
    length = 30.0

    def project(self, position):
        return SimpleNamespace(s=float(position[0]), distance=abs(float(position[1])), signed_error=float(position[1]))

    def point_at(self, s):
        return (float(s), 0.0)


class _ControlRoute:
    def point_at(self, s):
        return (float(s), 0.25)


def test_teacher_is_computed_from_actual_learner_pose() -> None:
    bundle = SimpleNamespace(
        scenario=SimpleNamespace(x_m=8.0, y_m=0.0),
        plan=SimpleNamespace(departure_start_s_m=3.0, cone_s_m=8.0, return_end_s_m=11.0),
    )
    expert = SimpleNamespace(baseline=SimpleNamespace(lookahead_m=0.9, wheelbase_m=0.18, max_steering_rad=0.349066))
    first = frozen_teacher_label(_Nominal(), _ControlRoute(), {"x": 4.0, "y": 0.0, "yaw": 0.0}, bundle, expert)
    visited = frozen_teacher_label(_Nominal(), _ControlRoute(), {"x": 4.0, "y": 0.4, "yaw": 0.2}, bundle, expert)
    assert first["teacher_valid"] and visited["teacher_valid"]
    assert first["teacher_uses_actual_learner_pose"] is True
    assert visited["cte_m"] == pytest.approx(0.4)
    assert visited["heading_error_rad"] == pytest.approx(0.2)
    assert first["expert_steering_rad"] != visited["expert_steering_rad"]


def test_teacher_invalid_is_excluded() -> None:
    invalid = {"policy_status": "R1_CONTROL", "teacher_valid": False, "expert_steering_rad": None}
    valid = {"policy_status": "R1_CONTROL", "teacher_valid": True, "expert_steering_rad": 0.1}
    assert not teacher_row_usable(invalid, 1)
    assert not teacher_row_usable(valid, -1)
    assert not teacher_row_usable(valid, 120_000_001)
    assert teacher_row_usable(valid, 120_000_000)


def test_teacher_pairing_is_causal_zoh_and_never_future() -> None:
    times = [100, 200, 200, 300]
    assert latest_causal_teacher_index(times, 99) is None
    assert latest_causal_teacher_index(times, 100) == 0
    assert latest_causal_teacher_index(times, 200) == 2
    assert latest_causal_teacher_index(times, 250) == 2
    assert latest_causal_teacher_index(times, 300) == 3


def test_temporal_three_frame_order_gap_and_episode_boundary() -> None:
    assert temporal_triplet_gaps((100, 60_000_100, 120_000_100), ("e", "e", "e")) == pytest.approx((0.06, 0.06, 0.12))
    assert temporal_triplet_gaps((100, 60_000_100, 180_000_101), ("e", "e", "e")) is None
    with pytest.raises(Exception, match="strictly causal"):
        temporal_triplet_gaps((100, 100, 200), ("e", "e", "e"))
    with pytest.raises(Exception, match="episode boundary"):
        temporal_triplet_gaps((100, 200, 300), ("e1", "e1", "e2"))


def test_exact_architecture_and_scratch_training_contract() -> None:
    config = load_config(CONFIG, REPO)
    assert sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) == 255_819
    assert config.training["input_channels"] == 9
    assert config.training["history_frames"] == 3
    assert config.training["initialization"] == "from_scratch"
    assert config.training["sample_weighting"] is False
    assert config.training["augmentation"] is False


def test_validation_and_unseen_sequential_failure_gates() -> None:
    failed_s09 = {"result": "FAIL", "scenarios": [{"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_FAIL"}]}
    assert not validation_allows_unseen(failed_s09)
    assert next_validation([]) == "09"
    assert next_validation(failed_s09["scenarios"]) is None
    passed_validation = {
        "result": "PASS",
        "scenarios": [
            {"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_PASS"},
        ],
    }
    assert validation_allows_unseen(passed_validation)
    assert next_validation(passed_validation["scenarios"][:1]) == "10"
    assert next_validation(passed_validation["scenarios"]) is None
    assert next_holdout([]) == "11"
    assert next_holdout([{"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_FAIL"}]) is None
    assert next_holdout([{"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_PASS"}]) == "12"


def test_freeze_seal_is_required_before_new_s09_live() -> None:
    freeze = {
        "frozen_before_any_new_s09_live_run": True,
        "training_from_scratch": True,
        "single_logical_training_run": True,
    }
    seal = {
        "freeze_sha256": "f", "checkpoint_sha256": "c", "onnx_sha256": "o",
        "aggregate_manifest_sha256": "a", "validation_manifest_sha256": "v",
        "live_attempt_count_before_seal": 0,
        "retraining_or_tuning_after_seal_permitted": False,
    }
    assert freeze_seal_contract(
        freeze, seal, freeze_sha256="f", checkpoint_sha256="c", onnx_sha256="o",
        aggregate_sha256="a", validation_sha256="v",
    )
    seal["live_attempt_count_before_seal"] = 1
    assert not freeze_seal_contract(
        freeze, seal, freeze_sha256="f", checkpoint_sha256="c", onnx_sha256="o",
        aggregate_sha256="a", validation_sha256="v",
    )


def test_frozen_hashes_and_no_training_leakage_permissions() -> None:
    config = load_config(CONFIG, REPO)
    assert config.r1["train_manifest_sha256"] == "a9aaf25991cecbab3937deae545d392842007b228d8b8f571c519fba1772df73"
    assert config.r1["validation_manifest_sha256"] == "a1182170a5d853b599209e6ce31f7deaa27077a99bdd62603b92ed817349693b"
    assert config.payload["permissions"]["s09_s10_live_data_training_permitted"] is False
    assert config.payload["permissions"]["second_dagger_iteration_permitted"] is False
    assert config.payload["permissions"]["r1_evidence_changes_permitted"] is False
