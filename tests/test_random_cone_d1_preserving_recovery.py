from __future__ import annotations

import copy
import csv
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from physicar_e2e.dataset_extractor import sha256_file
from physicar_e2e.pilotnet_temporal import (
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
    preprocess_temporal_paths,
)
from physicar_e2e import random_cone_d1_preserving_recovery as recovery


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/random_cone_d1_preserving_recovery_1p0_v1.json"


@pytest.fixture(scope="module")
def config() -> recovery.AdaptationConfig:
    return recovery.load_config(CONFIG_PATH, REPO)


def test_exact_frozen_d1_initialization(config):
    adapted, teacher, contract = recovery.load_exact_d1_initialization(
        config, torch.device("cpu"),
    )
    assert contract["initialization_source_sha256"] == config.inputs["d1"]["checkpoint_sha256"]
    assert contract["all_initial_tensors_bitwise_equal"] is True
    assert all(
        torch.equal(adapted.state_dict()[name], teacher.state_dict()[name])
        for name in adapted.state_dict()
    )


def test_d2_fe_initialization_is_explicitly_forbidden(config):
    payload = copy.deepcopy(config.payload)
    payload["frozen_inputs"]["d1"]["checkpoint_path"] = payload["frozen_inputs"]["d2_fe_negative"]["checkpoint_path"]
    payload["frozen_inputs"]["d1"]["checkpoint_sha256"] = payload["frozen_inputs"]["d2_fe_negative"]["checkpoint_sha256"]
    invalid = recovery.AdaptationConfig(config.path, payload, config.dagger1, config.prior_frontier)
    with pytest.raises(recovery.AdaptationGateError, match="D2-FE initialization is forbidden"):
        recovery.load_exact_d1_initialization(invalid, torch.device("cpu"))
    assert config.payload["permissions"]["d2_fe_initialization_permitted"] is False


def test_convolutional_backbone_frozen_and_only_steering_head_trainable():
    model = build_temporal_pilotnet()
    contract = recovery.parameter_contract(model, freeze=True)
    assert contract["frozen_parameter_count"] == 134_948
    assert contract["trainable_parameter_count"] == 120_871
    assert {item["name"].split(".")[0] for item in contract["frozen_parameters"]} == {"features"}
    assert {item["name"].split(".")[0] for item in contract["trainable_parameters"]} == {"regressor"}


def test_post_recovery_source_is_exactly_109_existing_samples(config):
    reference = config.inputs["post_recovery_manifest"]
    path = Path(reference["path"])
    assert sha256_file(path) == reference["sha256"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 109
    assert {row["provenance"] for row in rows} == {"DAGGER2_POST_RECOVERY"}
    assert all(row["recovery_state"] == "PASS" for row in rows)
    assert all(row["post_recovery_target"].lower() == "true" for row in rows)


def test_retention_replay_uses_frozen_d1_outputs(config):
    adapted, teacher, _contract = recovery.load_exact_d1_initialization(
        config, torch.device("cpu"),
    )
    replay, _post = recovery._model_rows(REPO, config)
    features, targets = recovery._extract_frozen_features(
        teacher, replay[:1], torch.device("cpu"), 1,
        config.training["max_steering_rad"], target_kind="frozen_D1",
    )
    with torch.inference_mode():
        image = torch.from_numpy(np.stack([preprocess_temporal_paths(replay[0]["paths"])]))
        expected = teacher(image)
    assert torch.equal(targets, expected)
    assert torch.equal(features, teacher.features(image))
    assert adapted is not teacher


def test_s09_s10_and_s11_s12_are_excluded_from_all_training_sources(config):
    scenarios: set[str] = set()
    for key in ("retention_replay_manifest", "post_recovery_manifest"):
        with Path(config.inputs[key]["path"]).open(newline="", encoding="utf-8") as stream:
            scenarios.update(row["scenario_id"].zfill(2) for row in csv.DictReader(stream))
    assert scenarios == set(recovery.TRAIN_SCENARIOS)
    assert not scenarios.intersection({"09", "10", "11", "12"})
    assert config.payload["permissions"]["holdout_access_before_validation_pass_permitted"] is False


def test_no_collection_dagger3_or_scratch_training_surface(config):
    source = inspect.getsource(recovery)
    assert not hasattr(recovery, "collection_stage")
    assert "start_recorder(" not in source
    assert "run_dagger" not in source
    assert config.payload["permissions"]["new_data_collection_permitted"] is False
    assert config.payload["permissions"]["dagger3_permitted"] is False
    assert config.payload["permissions"]["train_from_scratch_permitted"] is False
    assert config.training["initialization"] == "exact_frozen_D1"


def test_exact_input_output_and_parameter_count():
    model = build_temporal_pilotnet().eval()
    value = model(torch.zeros((2, 9, 66, 200), dtype=torch.float32))
    assert value.shape == (2, 1)
    assert TEMPORAL_PARAMETER_COUNT == 255_819
    assert sum(parameter.numel() for parameter in model.parameters()) == 255_819


def test_preregistered_optimizer_loss_and_single_run(config):
    training = config.training
    assert training["optimizer"] == "Adam"
    assert training["learning_rate"] == 1e-4
    assert training["maximum_epochs"] == 5
    assert training["batch_size"] == 64
    assert training["post_recovery_coefficient"] == 1.0
    assert training["retention_coefficient"] == 1.0
    assert training["regularizers"] == []
    assert config.payload["permissions"]["adaptation_logical_runs_permitted"] == 1


def test_structural_retention_check_detects_registered_failures(config):
    definition = config.payload["offline"]["structural_check"]
    d1 = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
    assert recovery.structural_retention_check(
        {"nominal": (d1, d1.copy())}, definition,
    )["result"] == "PASS"
    reversed_output = -d1
    failed = recovery.structural_retention_check(
        {"nominal": (d1, reversed_output)}, definition,
    )
    assert failed["classification"] == recovery.OFFLINE_RETENTION_FAIL
    assert failed["failure_modes"]["broad_sign_reversal"] is True


def test_freeze_is_verified_before_every_live_attempt():
    source = inspect.getsource(recovery._live_group)
    assert source.index("verify_frozen_d1_r(repo, config)") < source.index("run_live_once(")
    assert '"model_frozen_before_attempt": True' in source


def test_s09_fail_blocks_s10_and_s10_fail_blocks_holdout():
    s09_fail = [{"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert recovery.next_validation(s09_fail) is None
    assert recovery.validation_allows_holdout({"result": "FAIL", "scenarios": s09_fail}) is False
    s10_fail = {
        "result": "FAIL",
        "scenarios": [
            {"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_FAIL"},
        ],
    }
    assert recovery.validation_allows_holdout(s10_fail) is False
    assert recovery.classify_final_category(
        {"structural_retention_check": {"result": "PASS"}}, s10_fail, None,
    ) == recovery.VALIDATION_FAIL


def test_s11_fail_blocks_s12():
    records = [{"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert recovery.next_holdout(records) is None


def test_policy_results_never_retry_and_infrastructure_replacement_is_bounded():
    assert recovery.live_retry_decision("RANDOM_CONE_POLICY_PASS", 1) == "FINALIZE_PASS"
    assert recovery.live_retry_decision("RANDOM_CONE_POLICY_FAIL", 1) == "FINALIZE_GENUINE_FAILURE"
    assert recovery.live_retry_decision("INFRA_FAIL", 1) == "REPLACE_INFRA"
    assert recovery.live_retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"


def test_live_group_has_unconditional_safe_stop():
    source = inspect.getsource(recovery._live_group)
    assert "finally:" in source
    assert "final_errors = client.safe_stop()" in source
    assert "errors = client.safe_stop()" in source


def test_final_category_requires_all_four_passes():
    offline = {"structural_retention_check": {"result": "PASS"}}
    validation = {
        "result": "PASS", "category": "VALIDATION_PASS",
        "scenarios": [
            {"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_PASS"},
        ],
    }
    holdout = {
        "result": "PASS", "category": "UNSEEN_PASS",
        "scenarios": [
            {"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "12", "classification": "RANDOM_CONE_POLICY_PASS"},
        ],
    }
    assert recovery.classify_final_category(offline, validation, holdout) == recovery.FULL_PASS
