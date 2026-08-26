from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import torch

from physicar_e2e.random_cone_post_recovery_conflict_fast import (
    BOTH_SUPPORTED,
    EXPECTED_BASE_COMMIT,
    GRADIENT_CONFLICT_SUPPORTED,
    MIXED_OR_INCONCLUSIVE,
    NO_STRONG_CONFLICT_FOUND,
    VISUAL_STATE_ALIASING_SUPPORTED,
    _gradient_vector,
    classify_evidence,
    clone_state,
    construct_subsets,
    gradient_cosine,
    load_config,
    load_frozen_model,
    nearest_neighbor_mapping,
    states_exactly_equal,
)


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src/physicar_e2e/random_cone_post_recovery_conflict_fast.py"


def test_config_is_diagnostic_only_and_has_no_holdout_inputs():
    config = load_config(REPO)
    permissions = config.payload["permissions"]
    assert permissions["optimizer_steps_permitted"] == 0
    assert permissions["training_permitted"] is False
    assert permissions["new_data_collection_permitted"] is False
    assert permissions["simulator_invocation_permitted"] is False
    assert permissions["s11_s12_access_permitted"] is False
    assert all("s11" not in name.lower() and "s12" not in name.lower()
               for name in config.inputs)
    assert all("s11" not in str(spec["path"]).lower() and "s12" not in str(spec["path"]).lower()
               for spec in config.inputs.values())


def test_source_exposes_no_training_collection_or_simulator_dependency():
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(token in module.lower() for module in imported
                   for token in ("sim_client", "rosbag", "collector", "docker"))
    assert "torch.optim" not in text
    assert ".backward(" not in text
    assert "optimizer.step" not in text
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert not any(name.startswith(("train", "collect", "run_live")) for name in function_names)


def test_exact_preserved_d1_weights_and_parameter_contract_load():
    config = load_config(REPO)
    path = Path(config.inputs["d1_checkpoint"]["path"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, record = load_frozen_model(path)
    assert record["identity"]["task_config_sha256"] == config.inputs["dagger1_config"]["sha256"]
    assert record["identity"]["train_manifest_sha256"] == config.inputs["d1_aggregate_manifest"]["sha256"]
    assert sum(parameter.numel() for parameter in model.parameters()) == 255_819
    assert states_exactly_equal(model.state_dict(), checkpoint["model_state_dict"])
    assert EXPECTED_BASE_COMMIT == config.payload["expected_base_commit"]


def test_subsets_have_exact_source_provenance_and_validation_exclusion():
    config = load_config(REPO)
    subsets, audit = construct_subsets(config)
    assert len(subsets["EXPERT_BASELINE"]) == 6706
    assert len(subsets["DAGGER1_ALL"]) == 1483
    assert len(subsets["DAGGER1_AVOIDANCE_ONLY"]) == 155
    assert len(subsets["DAGGER2_POST_RECOVERY"]) == 109
    assert {row.provenance for row in subsets["DAGGER2_POST_RECOVERY"]} == {"DAGGER2_POST_RECOVERY"}
    assert audit["training_scenarios"] == [f"{value:02d}" for value in range(1, 9)]
    assert audit["validation_gradient_use"] == {"S09": False, "S10": False}
    assert audit["s11_s12_accessed"] is False


def test_nearest_neighbor_mapping_uses_cosine_and_euclidean():
    query = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    reference = np.asarray([[0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    result = nearest_neighbor_mapping(query, reference)
    assert result["cosine_index"].tolist() == [0, 1]
    assert result["euclidean_index"].tolist() == [0, 1]
    assert np.all(result["cosine_distance"] >= 0.0)


def test_gradient_cosine_contract():
    assert np.isclose(gradient_cosine(np.asarray([1.0, 0.0]), np.asarray([-1.0, 0.0])), -1.0)
    assert np.isclose(gradient_cosine(np.asarray([1.0, 1.0]), np.asarray([1.0, 1.0])), 1.0)


def test_diagnostic_gradient_leaves_every_model_tensor_identical():
    config = load_config(REPO)
    subsets, _ = construct_subsets(config)
    model, _ = load_frozen_model(Path(config.inputs["d1_checkpoint"]["path"]))
    before = clone_state(model.state_dict())
    value = _gradient_vector(model, subsets["DAGGER2_POST_RECOVERY"][:2])
    assert value["full"].size == 255_819
    assert value["head"].size == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("regressor.")
    )
    assert states_exactly_equal(before, model.state_dict())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_classification_logic_uses_exact_categories():
    assert classify_evidence(gradient_supported=True, aliasing_supported=False) == GRADIENT_CONFLICT_SUPPORTED
    assert classify_evidence(gradient_supported=False, aliasing_supported=True) == VISUAL_STATE_ALIASING_SUPPORTED
    assert classify_evidence(gradient_supported=True, aliasing_supported=True) == BOTH_SUPPORTED
    assert classify_evidence(gradient_supported=False, aliasing_supported=False) == NO_STRONG_CONFLICT_FOUND
    assert classify_evidence(
        gradient_supported=False,
        aliasing_supported=False,
        mixed_or_required_evidence_missing=True,
    ) == MIXED_OR_INCONCLUSIVE


def test_config_does_not_authorize_checkpoint_or_historical_result_changes():
    payload = json.loads((REPO / "configs/random_cone_post_recovery_conflict_fast_v1.json").read_text())
    assert payload["permissions"]["checkpoint_writes_permitted"] is False
    assert payload["prior_dagger2_negative"]["historical_result"] == "FAIL"
    assert payload["prior_dagger2_negative"]["training_authorized"] is False
