from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import pytest

from physicar_e2e.dataset_extractor import sha256_file
from physicar_e2e.pilotnet_temporal import TEMPORAL_PARAMETER_COUNT, build_temporal_pilotnet
from physicar_e2e import random_cone_d2_frontier_expansion as fe


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/random_cone_d2_frontier_expansion_1p0_v1.json"


@pytest.fixture(scope="module")
def config() -> fe.FrontierConfig:
    return fe.load_config(CONFIG_PATH, REPO)


def test_previous_dagger2_coverage_failure_remains_unchanged(config):
    report = fe._audit_prior_failure(REPO, config)
    assert report["result"] == "PASS"
    assert report["historical_result"] == "FAIL"
    assert report["historical_training_authorized"] is False
    assert report["historical_requirement_after_26m"] == 1


def test_previous_dagger2_trees_have_preregistered_identities(config):
    identities = fe._prior_tree_gate(REPO, config)
    assert identities["collection"]["sha256"] == config.payload["prior_negative_result"]["collection_directory_sha256"]
    assert identities["dataset"]["sha256"] == config.payload["prior_negative_result"]["dataset_directory_sha256"]


def test_exact_aggregate_provenance_is_preregistered(config):
    assert config.payload["aggregate"]["sequence_count"] == 8298
    assert config.payload["aggregate"]["provenance_counts"] == {
        "EXPERT_BASELINE": 6706,
        "DAGGER1": 1483,
        "DAGGER2_POST_RECOVERY": 109,
    }


def test_no_new_data_collection_surface_or_permission(config):
    assert config.payload["permissions"]["new_data_collection_permitted"] is False
    assert config.payload["permissions"]["new_expert_rollouts_permitted"] is False
    assert config.payload["permissions"]["new_dagger_rollouts_permitted"] is False
    assert not hasattr(fe, "collection_stage")
    source = inspect.getsource(fe)
    assert "start_recorder(" not in source
    assert "run_dagger2_rollout(" not in source


def test_frozen_dagger2_manifest_has_no_validation_or_holdout_leakage(config):
    path = Path(config.inputs["dagger2_temporal_manifest"]["path"])
    assert sha256_file(path) == config.inputs["dagger2_temporal_manifest"]["sha256"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 109
    assert {row["provenance"] for row in rows} == {"DAGGER2_POST_RECOVERY"}
    assert {row["scenario_id"].zfill(2) for row in rows} <= set(fe.TRAIN_SCENARIOS)
    assert not ({"09", "10", "11", "12"} & {row["scenario_id"].zfill(2) for row in rows})


def test_frozen_validation_is_exact_and_not_training(config):
    reference = config.inputs["validation_manifest"]
    path = Path(reference["path"])
    assert sha256_file(path) == reference["sha256"]
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 837
    assert {row["scenario_id"].zfill(2) for row in rows} == {"09", "10"}


def test_training_is_scratch_unweighted_and_single_run(config):
    training = config.training
    assert training["initialization"] == "from_scratch"
    assert training["augmentation"] is False
    assert training["sample_weighting"] is False
    assert training["source_weighting"] is False
    assert training["oversampling"] is False
    assert training["undersampling"] is False
    assert config.payload["permissions"]["d2_fe_logical_training_runs_permitted"] == 1


def test_temporal_pilotnet_parameter_count_is_frozen():
    assert TEMPORAL_PARAMETER_COUNT == 255_819
    assert sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) == 255_819


def test_freeze_is_checked_before_each_live_attempt():
    source = inspect.getsource(fe._live_group)
    assert source.index("verify_frozen_d2_fe(repo, config)") < source.index("run_live_once(")
    assert '"model_frozen_before_attempt": True' in source


def test_one_genuine_s09_result_cannot_retry():
    assert fe.live_retry_decision("RANDOM_CONE_POLICY_PASS", 1) == "FINALIZE_PASS"
    assert fe.live_retry_decision("RANDOM_CONE_POLICY_FAIL", 1) == "FINALIZE_GENUINE_FAILURE"
    assert fe.live_retry_decision("INFRA_FAIL", 1) == "REPLACE_INFRA"
    assert fe.live_retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"


def test_s09_failure_blocks_s10_and_holdout():
    records = [{"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert fe.next_validation(records) is None
    report = {"result": "FAIL", "scenarios": records}
    assert fe.validation_allows_holdout(report) is False


def test_s10_failure_blocks_holdout():
    report = {
        "result": "FAIL",
        "scenarios": [
            {"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_FAIL"},
        ],
    }
    assert fe.validation_allows_holdout(report) is False
    assert fe.classify_final_category(report, None) == fe.D2_FE_VALIDATION_FAIL


def test_s11_failure_blocks_s12():
    records = [{"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_FAIL"}]
    assert fe.next_holdout(records) is None


def test_frontier_categories_are_decided_from_preregistered_live_metrics(config):
    baseline = config.payload["frontier"]
    partial = {
        "classification": "RANDOM_CONE_POLICY_FAIL",
        "run": {
            "total_unwrapped_progress_m": baseline["d1_s09_progress_m"] + 0.7,
            "final_route_s_m": baseline["d1_s09_final_route_s_m"] + 0.6,
            "route_completion_fraction": baseline["d1_s09_completion_fraction"] + 0.021,
            "recovery_success": True,
        },
    }
    assert fe.classify_s09_frontier(partial, baseline)["classification"] == fe.FRONTIER_EXPANSION_PARTIAL_SUPPORT
    regression = {
        "classification": "RANDOM_CONE_POLICY_FAIL",
        "run": {
            "total_unwrapped_progress_m": baseline["d1_s09_progress_m"] - 2.0,
            "final_route_s_m": baseline["d1_s09_final_route_s_m"] - 2.0,
            "route_completion_fraction": baseline["d1_s09_completion_fraction"] - 0.06,
            "recovery_success": False,
        },
    }
    assert fe.classify_s09_frontier(regression, baseline)["classification"] == fe.REGRESSION


def test_no_automatic_dagger3_and_safe_stop(config):
    assert config.payload["permissions"]["dagger3_permitted"] is False
    source = inspect.getsource(fe._live_group)
    assert "client.safe_stop()" in source
    assert "finally:" in source
    assert "dagger3" not in source.lower()


def test_final_category_requires_all_four_passes():
    validation = {
        "result": "PASS", "category": "VALIDATION_PASS",
        "scenarios": [
            {"scenario_id": "09", "classification": "RANDOM_CONE_POLICY_PASS"},
            {"scenario_id": "10", "classification": "RANDOM_CONE_POLICY_PASS"},
        ],
    }
    holdout = {"result": "PASS", "category": "UNSEEN_PASS", "scenarios": [
        {"scenario_id": "11", "classification": "RANDOM_CONE_POLICY_PASS"},
        {"scenario_id": "12", "classification": "RANDOM_CONE_POLICY_PASS"},
    ]}
    assert fe.classify_final_category(validation, holdout) == fe.D2_FE_FULL_PASS
