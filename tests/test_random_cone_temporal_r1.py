from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from physicar_e2e.pilotnet_temporal import CausalFrameBuffer, build_temporal_pilotnet
from physicar_e2e.random_cone_temporal_r1 import (
    HOLDOUT_SCENARIOS,
    MIN_COLLECTION_BYTES,
    MIN_LIVE_BYTES,
    MIN_TRAIN_BYTES,
    TRAIN_MANIFEST_SHA256,
    TRAIN_SCENARIOS,
    TRAIN_SEQUENCE_COUNT,
    VALIDATION_EPISODES,
    VALIDATION_SCENARIOS,
    ValidationSpec,
    build_validation_sequences,
    holdout_next_scenario,
    inference_config,
    live_retry_decision,
    load_config,
    run_live_once,
    summarize_neural_cone_run,
    validation_allows_unseen,
    validation_collection_gate,
    validation_retry_decision,
    validation_specs,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/random_cone_temporal_r1_v1.json"


def _config():
    return load_config(CONFIG_PATH, REPO)


def test_frozen_train_manifest_identity_and_exact_roles() -> None:
    config = _config()
    assert config.frozen_train["manifest_sha256"] == TRAIN_MANIFEST_SHA256
    assert config.frozen_train["sequence_count"] == TRAIN_SEQUENCE_COUNT == 6706
    assert tuple(config.frozen_train["scenario_ids"]) == TRAIN_SCENARIOS
    assert config.payload["scenario_roles"] == {
        "TRAIN": list(TRAIN_SCENARIOS),
        "VALIDATION": list(VALIDATION_SCENARIOS),
        "UNSEEN_HOLDOUT": list(HOLDOUT_SCENARIOS),
    }


def test_validation_collection_is_exactly_s09_s10_once_and_never_holdout() -> None:
    config = _config()
    assert tuple(config.collection["episode_order"]) == VALIDATION_EPISODES
    assert validation_specs() == (
        ValidationSpec("val_s09_r01", "09"), ValidationSpec("val_s10_r01", "10")
    )
    assert all(spec.role == "VALIDATION" and spec.repeat_id == "R01" for spec in validation_specs())
    assert not set(spec.scenario_id for spec in validation_specs()).intersection(HOLDOUT_SCENARIOS)
    assert config.payload["permissions"]["holdout_bag_collection_permitted"] is False
    assert config.payload["permissions"]["holdout_label_extraction_permitted"] is False


def test_disk_and_infrastructure_retry_contracts_are_frozen() -> None:
    config = _config()
    assert config.collection["minimum_free_bytes_before_collection"] == MIN_COLLECTION_BYTES
    assert config.dataset["minimum_free_bytes_before_training"] == MIN_TRAIN_BYTES
    assert config.live["minimum_free_bytes_before_live"] == MIN_LIVE_BYTES
    assert validation_retry_decision("GENUINE_EXPERT_FAIL", 1) == "STOP_GENUINE_FAILURE"
    assert validation_retry_decision("INFRA_FAIL", 1) == "RETRY_INFRA"
    assert validation_retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"
    assert live_retry_decision("RANDOM_CONE_POLICY_FAIL", 1) == "STOP_GENUINE_FAILURE"
    assert live_retry_decision("INFRA_FAIL", 1) == "RETRY_INFRA"
    assert live_retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"


def _frame(spec: ValidationSpec, index: int, timestamp: int) -> dict:
    return {
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "scenario_role": spec.role,
        "image_path": f"images/{spec.episode_id}/{index}.png", "image_sha256": f"{index:064x}",
        "camera_record_time_ns": timestamp, "steering_record_time_ns": timestamp - 1,
        "steering_age_ms": 0.000001, "steering_rad": index * 0.01,
        "speed_record_time_ns": timestamp - 1, "speed_age_ms": 0.000001,
        "speed_mps": 1.0, "route_s_m": float(index), "source_mcap_sha256": "a" * 64,
    }


def test_validation_temporal_manifest_is_causal_unpadded_and_gap_gated() -> None:
    spec = validation_specs()[0]
    rows = [_frame(spec, 0, 0), _frame(spec, 1, 60_000_000), _frame(spec, 2, 120_000_000), _frame(spec, 3, 300_000_000)]
    sequences, stats = build_validation_sequences(rows, spec, "b" * 64)
    assert len(sequences) == 1 and stats["gap_rejects"] == 1
    assert stats["boundary_rejects"] == 2 and stats["future_label_violations"] == 0
    row = sequences[0]
    assert row["scenario_role"] == "VALIDATION" and row["scenario_id"] == "09"
    assert row["camera_timestamp_t_minus_2_ns"] < row["camera_timestamp_t_minus_1_ns"] < row["camera_timestamp_t_ns"]
    assert max(row["adjacent_gap_1_s"], row["adjacent_gap_2_s"]) <= 0.120
    assert len({row["frame_t_minus_2"], row["frame_t_minus_1"], row["frame_t"]}) == 3
    assert row["steering_target_timestamp_ns"] <= row["camera_timestamp_t_ns"]


def test_architecture_input_and_parameter_count_are_exact() -> None:
    config = _config()
    model = build_temporal_pilotnet()
    assert config.training["input_channels"] == 9
    assert (config.training["image_height"], config.training["image_width"]) == (66, 200)
    assert sum(parameter.numel() for parameter in model.parameters()) == 255_819


def test_training_is_from_scratch_once_without_prior_or_balanced_sources() -> None:
    config = _config()
    training = config.training
    assert training["initialization"] == "from_scratch"
    assert training["optimizer"] == "Adam" and training["loss"] == "MSE"
    assert (training["learning_rate"], training["batch_size"], training["max_epochs"]) == (0.001, 64, 35)
    for key in ("augmentation", "sample_weighting", "scenario_weighting", "oversampling", "undersampling", "hyperparameter_sweep"):
        assert training[key] is False
    assert config.payload["permissions"]["fine_tuning_permitted"] is False
    assert config.payload["permissions"]["retraining_after_validation_or_holdout_permitted"] is False
    assert "train_dataset/temporal_manifests/train.csv" in config.frozen_train["manifest_path"]
    assert all(token not in config.frozen_train["manifest_path"].lower() for token in ("v9", "c1", "1p8", "dagger", "fixed_cone"))


def test_live_buffer_is_three_real_causal_frames_only() -> None:
    buffer = CausalFrameBuffer(0.120)
    frame = np.zeros((3, 66, 200), dtype=np.float32)
    buffer.append(1.00, frame)
    buffer.append(1.06, frame)
    assert not buffer.ready
    buffer.append(1.12, frame)
    assert buffer.ready and buffer.tensor().shape == (9, 66, 200)
    assert buffer.gaps() == (0.06000000000000005, 0.06000000000000005, 0.1200000000000001)


def _live_record(scenario: str, classification: str) -> dict:
    return {"scenario_id": scenario, "classification": classification}


def test_validation_failures_block_all_unseen() -> None:
    assert not validation_allows_unseen({"result": "FAIL", "scenarios": [_live_record("09", "RANDOM_CONE_POLICY_FAIL")]})
    assert not validation_allows_unseen({"result": "FAIL", "scenarios": [_live_record("09", "RANDOM_CONE_POLICY_PASS"), _live_record("10", "RANDOM_CONE_POLICY_FAIL")]})
    assert validation_allows_unseen({"result": "PASS", "scenarios": [_live_record("09", "RANDOM_CONE_POLICY_PASS"), _live_record("10", "RANDOM_CONE_POLICY_PASS")]})


def test_s11_failure_blocks_s12_and_s11_pass_unlocks_s12() -> None:
    assert holdout_next_scenario([]) == "11"
    assert holdout_next_scenario([_live_record("11", "RANDOM_CONE_POLICY_FAIL")]) is None
    assert holdout_next_scenario([_live_record("11", "RANDOM_CONE_POLICY_PASS")]) == "12"
    assert holdout_next_scenario([_live_record("11", "RANDOM_CONE_POLICY_PASS"), _live_record("12", "RANDOM_CONE_POLICY_PASS")]) is None


def test_collection_gate_rejects_duplicates_train_and_holdout() -> None:
    control = {"speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0, "steering_limit_rad": 0.349066, "wheelbase_m": 0.18}
    rows = [{
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": "R01", "scenario_role": "VALIDATION",
        "classification": "VALIDATION_EPISODE_PASS", "preflight": {"fixed_control": control},
        "post_run_safe_stop_success": True, "final_safe_stop_success": True,
        "expert_result_metrics": {"safe_stop_success": True},
    } for spec in validation_specs()]
    assert validation_collection_gate(rows)["result"] == "PASS"
    assert validation_collection_gate([rows[0], rows[0]])["result"] == "FAIL"
    contaminated = [rows[0], {**rows[1], "scenario_id": "11"}]
    assert validation_collection_gate(contaminated)["result"] == "FAIL"


def test_collision_only_practical_contract_accepts_positive_sub_5cm_clearance() -> None:
    observer = SimpleNamespace(
        samples=[{"cone_clearance_m": 0.001, "route_s_m": 9.2}],
        intersection_occurred=False, contact_or_movement_occurred=False,
        recovery_success=True, recovery_time_s=0.4, recovery_cte_m=0.03,
    )
    scenario = SimpleNamespace(
        scenario_id="09", role="VALIDATION", curvature_class="moderate_left",
        route_s_m=9.0, x_m=1.0, y_m=2.0,
    )
    bundle = SimpleNamespace(scenario=scenario, plan=SimpleNamespace(side="left"))
    run = {
        "result": "PASS", "failure": None, "temporal_input_failure": False,
        "api_failures": 0, "liveness_failures": 0, "safe_stop_success": True,
    }
    result = summarize_neural_cone_run(run, observer, bundle)
    assert result["classification"] == "RANDOM_CONE_POLICY_PASS"
    assert result["minimum_footprint_to_cone_clearance_m"] == 0.001
    assert result["clearance_0p05_m_not_required"] is True


def test_live_inference_is_camera_only_with_exact_operating_condition() -> None:
    config = _config()
    value = inference_config(config, "scenario_world")
    assert value.payload["expected_world"] == "scenario_world"
    assert value.payload["camera_only_model_observation"] is True
    assert (value.payload["history_frames"], value.payload["input_channels"]) == (3, 9)
    assert (value.payload["speed_mps"], value.payload["control_frequency_hz"]) == (1.0, 15.0)
    assert "speed" in value.payload["camera_only_forbidden_inputs"]


def test_live_lifecycle_starts_with_safe_stop() -> None:
    events: list[str] = []

    class Client:
        def safe_stop(self):
            events.append("safe_stop")
            return []

    def prepare(*args):
        events.append("preflight")
        route = SimpleNamespace()
        initial = SimpleNamespace(route=route)
        preflight = {"fixed_control": {"speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0, "steering_limit_rad": 0.349066, "wheelbase_m": 0.18}}
        return initial, {"result": "PASS"}, preflight

    def policy(*args):
        events.append("policy")
        return {"result": "FAIL", "failure": "test", "temporal_input_failure": False, "api_failures": 0, "liveness_failures": 0, "safe_stop_success": True}

    scenario = SimpleNamespace(scenario_id="09", role="VALIDATION", curvature_class="low", route_s_m=1.0, x_m=1.0, y_m=1.0)
    bundle = SimpleNamespace(scenario=scenario, plan=SimpleNamespace(side="left"))
    expert = SimpleNamespace(
        world_name=lambda scenario_id: f"world_{scenario_id}",
        cone_model_name=lambda scenario_id: f"cone_{scenario_id}",
        return_to_route={},
    )
    run_live_once(Client(), object(), _config(), expert, bundle, Path("/sim"), prepare=prepare, run_policy=policy)
    assert events[:3] == ["safe_stop", "preflight", "policy"]


def test_frozen_manifest_on_disk_has_no_holdout_rows_if_available() -> None:
    path = Path(_config().frozen_train["manifest_path"])
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == TRAIN_SEQUENCE_COUNT
    assert {row["scenario_id"] for row in rows} == set(TRAIN_SCENARIOS)
    assert not {row["scenario_id"] for row in rows}.intersection(HOLDOUT_SCENARIOS)
