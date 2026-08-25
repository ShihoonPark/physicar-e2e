from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image
import pytest

from physicar_e2e.pilotnet_temporal import build_temporal_pilotnet
from physicar_e2e.random_cone_dagger2_post_recovery import (
    DAGGER2_EPISODES,
    D2_FULL_PASS,
    D2_UNSEEN_FAIL,
    D2_VALIDATION_FAIL,
    HOLDOUT_SCENARIOS,
    HOST_CRASH,
    INFRASTRUCTURE_INTERRUPTION,
    INCONCLUSIVE,
    PROVENANCE,
    ROUTE_BINS,
    TRAIN_SCENARIOS,
    VALIDATION_SCENARIOS,
    _host_crash_attempt_record,
    _sequence_from_ring,
    aggregate_provenance_contract,
    classify_final_category,
    collect_dagger2_attempt,
    collection_gate,
    collection_stage,
    d2_coverage_gate,
    d2_training_authorized,
    dataset_stage,
    episode_specs,
    host_crash_replacement_attempt,
    live_retry_decision,
    load_config,
    next_holdout,
    next_validation,
    post_recovery_target_eligible,
    teacher_pair_is_causal,
    temporal_triplet_contract,
    validation_allows_unseen,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/random_cone_dagger2_post_recovery_1p0_v1.json"


def _valid_collection_record(scenario: str) -> dict:
    return {
        "episode_id": f"dagger2_s{scenario}_r01",
        "scenario_id": scenario,
        "policy_outcome": "RANDOM_CONE_POLICY_PASS",
        "d1_controls_vehicle": True,
        "teacher_control_authority": False,
        "rosbags_recorded": 0,
        "raw_full_resolution_images_persisted": 0,
        "future_teacher_label_violations": 0,
        "post_run_safe_stop_success": True,
    }


def _capture(index: int, *, recovery: bool, route_s: float) -> dict:
    camera = 1_000_000_000 + index * 66_000_000
    return {
        "capture_iteration": index,
        "camera_timestamp_ns": camera,
        "expert_label_timestamp_ns": camera - 5_000_000,
        "expert_label_age_ms": 5.0,
        "learner_x_m": route_s,
        "learner_y_m": 0.1,
        "learner_yaw_rad": 0.0,
        "route_progress_m": route_s,
        "route_s_m": route_s,
        "cte_m": 0.1,
        "signed_cte_m": 0.1,
        "heading_error_rad": 0.0,
        "cone_phase": "post_recovery" if recovery else "pass_return",
        "cone_passed": route_s >= 10.0,
        "recovery_state": "PASS" if recovery else "NOT_YET_PASS",
        "recovery_success_at_capture": recovery,
        "d1_steering_rad": 0.05,
        "expert_steering_rad": 0.10,
        "d1_minus_expert_rad": -0.05,
        "absolute_steering_error_rad": 0.05,
        "teacher_uses_actual_learner_pose": True,
        "teacher_valid": True,
        "_image": Image.new("RGB", (200, 66), (index, index, index)),
    }


def test_collection_is_exactly_one_train_rollout_per_s01_s08() -> None:
    config = load_config(CONFIG, REPO)
    assert tuple(item.scenario_id for item in episode_specs()) == TRAIN_SCENARIOS
    assert tuple(item.episode_id for item in episode_specs()) == DAGGER2_EPISODES
    assert tuple(config.collection["episode_order"]) == DAGGER2_EPISODES
    assert not set(TRAIN_SCENARIOS) & set(VALIDATION_SCENARIOS + HOLDOUT_SCENARIOS)


def test_d1_controls_and_expert_is_shadow_only() -> None:
    gate = collection_gate([_valid_collection_record(value) for value in TRAIN_SCENARIOS])
    assert gate["result"] == "PASS"
    assert gate["gates"]["d1_controlled_all"]
    assert gate["gates"]["expert_shadow_only"]


def test_no_rosbag_or_raw_full_resolution_persistence() -> None:
    config = load_config(CONFIG, REPO)
    assert config.collection["record_rosbag"] is False
    assert config.collection["persist_full_resolution_camera"] is False
    assert config.dataset["source_transport"] == "HTTP_JPEG_IN_MEMORY_ONLY"
    assert config.dataset["stored_format"] == "PNG"
    assert (config.dataset["output_width"], config.dataset["output_height"]) == (200, 66)


def test_persistence_starts_at_recovery_and_preserves_two_causal_history_frames(tmp_path: Path) -> None:
    config = load_config(CONFIG, REPO)
    episode = episode_specs()[0]
    ring = [
        _capture(0, recovery=False, route_s=9.95),
        _capture(1, recovery=False, route_s=10.02),
        _capture(2, recovery=True, route_s=10.10),
    ]
    frames: list[dict] = []
    sequences: list[dict] = []
    assert _sequence_from_ring(
        ring,
        episode=episode,
        staging=tmp_path,
        frame_rows=frames,
        sequence_rows=sequences,
        saved={},
        return_end_s_m=10.0,
    )
    assert len(frames) == 3 and len(sequences) == 1
    assert [row["history_context_only"] for row in frames] == [True, True, False]
    assert sequences[0]["provenance"] == PROVENANCE
    assert sequences[0]["post_recovery_target"] is True
    assert len({sequences[0][key] for key in (
        "frame_t_minus_2", "frame_t_minus_1", "frame_t",
    )}) == 3
    for capture in ring:
        capture["_image"].close()


def test_pre_recovery_target_contributes_no_sequence(tmp_path: Path) -> None:
    episode = episode_specs()[0]
    ring = [_capture(index, recovery=False, route_s=9.8 + 0.05 * index) for index in range(3)]
    assert not _sequence_from_ring(
        ring, episode=episode, staging=tmp_path, frame_rows=[], sequence_rows=[],
        saved={}, return_end_s_m=10.0,
    )
    assert not any(tmp_path.rglob("*.png"))
    for capture in ring:
        capture["_image"].close()


def test_teacher_label_is_actual_state_causal_and_never_future() -> None:
    capture = _capture(2, recovery=True, route_s=10.2)
    assert capture["teacher_uses_actual_learner_pose"] is True
    assert post_recovery_target_eligible(capture, 10.0)
    assert teacher_pair_is_causal(
        capture["expert_label_timestamp_ns"], capture["camera_timestamp_ns"],
    )
    assert not teacher_pair_is_causal(
        capture["camera_timestamp_ns"] + 1, capture["camera_timestamp_ns"],
    )
    capture["_image"].close()


def test_temporal_contract_rejects_future_order_gap_padding_and_boundary() -> None:
    assert temporal_triplet_contract(
        (100, 60_000_100, 120_000_100), ("e", "e", "e"),
    ) == pytest.approx((0.06, 0.06, 0.12))
    assert temporal_triplet_contract(
        (100, 60_000_100, 180_000_101), ("e", "e", "e"),
    ) is None
    with pytest.raises(Exception, match="strictly causal"):
        temporal_triplet_contract((100, 100, 200), ("e", "e", "e"))
    with pytest.raises(Exception, match="episode boundary"):
        temporal_triplet_contract((100, 200, 300), ("e", "e", "other"))


def test_route_bins_are_fixed_exactly() -> None:
    assert ROUTE_BINS == (
        (0.0, 10.0), (10.0, 20.0), (20.0, 26.0), (26.0, 30.50461070080936),
    )


def test_coverage_gate_requires_both_missing_late_regions() -> None:
    base = {"scenario_id": "01", "provenance": PROVENANCE}
    assert d2_coverage_gate([{**base, "route_s_m": 20.1}])["result"] == "FAIL"
    passed = d2_coverage_gate([
        {**base, "route_s_m": 20.1}, {**base, "route_s_m": 26.1},
    ])
    assert passed["result"] == "PASS"
    assert passed["sequences_after_20m"] == 2
    assert passed["sequences_after_26m"] == 1
    forbidden = d2_coverage_gate([{**base, "scenario_id": "09", "route_s_m": 27.0}])
    assert forbidden["result"] == "FAIL"


def test_exact_aggregate_provenance_contract() -> None:
    assert aggregate_provenance_contract(
        {"EXPERT_BASELINE": 6706, "DAGGER1": 1483, PROVENANCE: 400}, 400,
    )
    assert not aggregate_provenance_contract(
        {"EXPERT_BASELINE": 6706, "DAGGER1": 1482, PROVENANCE: 400}, 400,
    )


def test_training_is_blocked_until_coverage_and_integrity_pass() -> None:
    dataset = {
        "result": "PASS", "sequences_after_20m": 1, "sequences_after_26m": 1,
        "future_teacher_label_violations": 0, "temporal_corruption_count": 0,
        "gates": {"coverage": True, "causal": True},
    }
    assert d2_training_authorized(dataset)
    assert not d2_training_authorized({**dataset, "sequences_after_26m": 0})
    assert not d2_training_authorized({**dataset, "future_teacher_label_violations": 1})


def test_aggregate_is_not_built_before_coverage_gate() -> None:
    source = inspect.getsource(dataset_stage)
    authorization = source.index("if aggregate_authorized:")
    build = source.index("build_d2_aggregate", authorization)
    assert authorization < build
    assert '"NOT_BUILT_COVERAGE_OR_INTEGRITY_GATE_FAILED"' in source


def test_exact_architecture_and_scratch_single_run_contract() -> None:
    config = load_config(CONFIG, REPO)
    assert sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) == 255_819
    assert config.training["initialization"] == "from_scratch"
    assert config.training["max_epochs"] == 35
    assert config.training["learning_rate"] == pytest.approx(1e-3)
    assert config.training["batch_size"] == 64
    assert config.payload["permissions"]["d2_logical_training_runs_permitted"] == 1
    assert config.payload["permissions"]["retraining_after_freeze_permitted"] is False


def test_frozen_d1_and_validation_identities() -> None:
    config = load_config(CONFIG, REPO)
    assert config.inputs["d1"]["checkpoint_sha256"] == "b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434"
    assert config.inputs["d1"]["onnx_sha256"] == "3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c"
    assert config.inputs["validation_manifest"]["sha256"] == "a1182170a5d853b599209e6ce31f7deaa27077a99bdd62603b92ed817349693b"
    assert config.inputs["validation_manifest"]["sequence_count"] == 837


def test_freeze_is_required_before_s09_and_no_post_freeze_training() -> None:
    config = load_config(CONFIG, REPO)
    assert config.payload["live"]["freeze_required_before_s09"] is True
    assert config.payload["permissions"]["retraining_after_freeze_permitted"] is False
    assert config.payload["permissions"]["d3_permitted"] is False


def test_s09_failure_blocks_s10_and_s10_failure_blocks_holdout() -> None:
    s09_fail = [{"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert next_validation([]) == "09"
    assert next_validation(s09_fail) is None
    assert not validation_allows_unseen({"result": "FAIL", "scenarios": s09_fail})
    s09_pass = [{"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"}]
    assert next_validation(s09_pass) == "10"
    s10_fail = [*s09_pass, {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert not validation_allows_unseen({"result": "FAIL", "scenarios": s10_fail})


def test_s11_failure_blocks_s12() -> None:
    assert next_holdout([]) == "11"
    assert next_holdout([
        {"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_FAIL"},
    ]) is None
    assert next_holdout([
        {"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_PASS"},
    ]) == "12"


def test_infrastructure_replacement_is_bounded_and_policy_outcome_never_retries() -> None:
    assert live_retry_decision("INFRA_FAIL", 1) == "REPLACE_INFRA"
    assert live_retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"
    assert live_retry_decision("RANDOM_CONE_POLICY_PASS", 1) == "FINALIZE_PASS"
    assert live_retry_decision("RANDOM_CONE_POLICY_FAIL", 1) == "FINALIZE_GENUINE_FAILURE"


def test_incomplete_s08_host_crash_is_not_a_policy_failure(tmp_path: Path) -> None:
    state_path = tmp_path / "dagger2_s08_r01.json"
    state_path.write_text(
        '{"attempt_number":1,"episode_id":"dagger2_s08_r01",'
        '"scenario_id":"08","status":"STARTED_UNFINALIZED"}\n',
        encoding="utf-8",
    )
    state = {
        "attempt_number": 1,
        "episode_id": "dagger2_s08_r01",
        "scenario_id": "08",
        "status": "STARTED_UNFINALIZED",
    }
    record = _host_crash_attempt_record(
        state, episode_specs()[-1], state_path=state_path,
        staging=tmp_path / "empty_staging", archive=None,
    )
    assert record["classification"] == INFRASTRUCTURE_INTERRUPTION
    assert record["infrastructure_outcome"] == HOST_CRASH
    assert record["policy_outcome"] is None
    assert record["counts_as_genuine_policy_outcome"] is False
    assert record["do_not_reinterpret_as_policy_fail"] is True
    assert record["replacement_attempt_number"] == 2


def test_collection_host_crash_replacement_is_exactly_one() -> None:
    state = {"status": "STARTED_UNFINALIZED", "attempt_number": 1}
    assert host_crash_replacement_attempt(state, maximum_replacements=1) == 2
    exhausted = {"status": "STARTED_UNFINALIZED", "attempt_number": 2}
    assert host_crash_replacement_attempt(exhausted, maximum_replacements=1) is None
    finalized = {"status": "FINALIZED_VALID_POLICY_OUTCOME", "attempt_number": 1}
    assert host_crash_replacement_attempt(finalized, maximum_replacements=1) is None


def test_collection_resume_skips_finalized_and_zero_sample_evidence() -> None:
    source = inspect.getsource(collection_stage)
    assert "records.append(existing); skipped.append(episode.episode_id)" in source
    assert "continue" in source
    zero_sample = {
        **_valid_collection_record("01"),
        "policy_outcome": "RANDOM_CONE_POLICY_FAIL",
        "temporal_sequence_count": 0,
        "contributes_training_data": False,
        "zero_samples_due_to_pre_recovery_policy_failure": True,
    }
    assert zero_sample["temporal_sequence_count"] == 0
    assert zero_sample["policy_outcome"] == "RANDOM_CONE_POLICY_FAIL"


def test_resume_identity_guard_prevents_s01_s07_duplication() -> None:
    source = inspect.getsource(collection_stage)
    assert "preserved_after != preserved_before" in source
    assert "a finalized pre-crash DAgger2 episode changed during resume" in source


def test_final_classification_rules() -> None:
    validation_fail = {"result": "FAIL", "category": D2_VALIDATION_FAIL}
    assert classify_final_category(validation_fail, None) == D2_VALIDATION_FAIL
    validation_pass = {"result": "PASS", "category": "VALIDATION_PASS"}
    holdout_fail = {"result": "FAIL", "category": D2_UNSEEN_FAIL}
    assert classify_final_category(validation_pass, holdout_fail) == D2_UNSEEN_FAIL
    holdout_pass = {"result": "PASS", "category": "UNSEEN_PASS"}
    assert classify_final_category(validation_pass, holdout_pass) == D2_FULL_PASS
    assert classify_final_category(None, None) == INCONCLUSIVE


def test_holdout_permission_and_learning_boundaries_are_closed() -> None:
    config = load_config(CONFIG, REPO)
    assert config.payload["permissions"]["holdout_access_before_validation_pass_permitted"] is False
    assert config.payload["permissions"]["s09_s12_dagger_data_permitted"] is False
    assert config.payload["permissions"]["new_expert_nominal_data_permitted"] is False
    assert config.payload["live"]["collect_bags"] is False
    assert config.payload["live"]["generate_holdout_expert_labels"] is False


def test_collection_always_safe_stops_and_never_gives_teacher_authority() -> None:
    source = inspect.getsource(collect_dagger2_attempt)
    assert "client.safe_stop()" in source
    assert "finally:" in source
    assert '"teacher_control_authority": False' in source
    assert "observer.command_steering" not in source
