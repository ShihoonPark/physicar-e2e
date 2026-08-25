from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from physicar_e2e.random_cone_d1_cone_free_recheck import (
    DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED,
    LATE_ROUTE_REGIONS,
    MIXED_OR_INCONCLUSIVE,
    POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED,
    SHARED_1P0_LANE_WEAKNESS_SUPPORTED,
    classify_recheck,
    load_config,
    r1_run_authorized,
    run_with_infrastructure_replacement,
    safe_stop_guard,
)
from physicar_e2e.random_cone_d1_late_lap_diagnosis import (
    CANONICAL_WORLD,
    DiagnosisGateError,
    issue_policy_commands,
    shadow_expert_steering,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/random_cone_d1_cone_free_recheck_v1.json"
MODULE_PATH = REPO / "src/physicar_e2e/random_cone_d1_cone_free_recheck.py"


def _attempt(classification: str) -> dict:
    return {"classification": classification}


def test_config_is_diagnostic_only_and_uses_exact_frozen_contract() -> None:
    config = load_config(CONFIG_PATH, REPO)
    assert config.payload["canonical_cone_free_world"] == CANONICAL_WORLD
    assert LATE_ROUTE_REGIONS == ((20.0, 26.0), (26.0, 30.50461070080936))
    assert config.payload["fixed_control"] == {
        "speed_mps": 1.0,
        "control_frequency_hz": 15.0,
        "lookahead_m": 0.9,
        "steering_limit_rad": 0.349066,
        "wheelbase_m": 0.18,
        "history_frames": 3,
        "maximum_adjacent_gap_s": 0.12,
        "off_track_margin_m": 0.05,
        "off_track_grace_s": 0.5,
    }
    assert all(value is False for value in config.payload["permissions"].values())


def test_no_training_bag_or_image_persistence_invocation_exists() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    prohibited_invocations = (
        "train_temporal(", "training_stage(", "start_recorder(", "DockerRosBackend(",
        "export_temporal_onnx(", ".save(", "write_images(",
    )
    assert all(token not in source for token in prohibited_invocations)
    config = load_config(CONFIG_PATH, REPO)
    assert config.payload["attempt_policy"]["record_bags"] is False
    assert config.payload["attempt_policy"]["record_images"] is False
    assert config.payload["permissions"]["training_permitted"] is False
    assert config.payload["permissions"]["bag_collection_permitted"] is False
    assert config.payload["permissions"]["image_persistence_permitted"] is False


def test_frozen_d1_identity_is_exact() -> None:
    config = load_config(CONFIG_PATH, REPO)
    assert config.payload["frozen_models"]["D1"] == {
        "checkpoint_sha256": "b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434",
        "onnx_sha256": "3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c",
        "freeze_sha256": "66dbf7762ab089f111e2c02d22240d861e575730dcb416692bf6fac4e1e3fdc8",
        "freeze_seal_sha256": "7781423c7ba69f381e91120687d07d93d006393ff3c0c74af751085ce6ea1840",
    }


def test_d1_pass_is_one_attempt_and_blocks_replacement_and_r1() -> None:
    calls: list[tuple[str, int]] = []

    def run_one(policy: str, number: int) -> dict:
        calls.append((policy, number))
        return _attempt("FULL_LAP_PASS")

    result = run_with_infrastructure_replacement("D1", run_one)
    assert calls == [("D1", 1)]
    assert result["physical_attempt_count"] == 1
    assert result["policy_valid_result_count"] == 1
    assert r1_run_authorized(result["valid_result"]) is False


def test_d1_policy_failure_cannot_retry_and_authorizes_r1() -> None:
    calls: list[tuple[str, int]] = []

    def run_one(policy: str, number: int) -> dict:
        calls.append((policy, number))
        return _attempt("POLICY_FAIL")

    result = run_with_infrastructure_replacement("D1", run_one)
    assert calls == [("D1", 1)]
    assert result["policy_valid_result_count"] == 1
    assert r1_run_authorized(result["valid_result"]) is True


def test_one_infrastructure_replacement_maximum_and_only_one_valid_result() -> None:
    calls: list[tuple[str, int]] = []

    def run_one(policy: str, number: int) -> dict:
        calls.append((policy, number))
        return _attempt("INFRA_FAIL" if number == 1 else "FULL_LAP_PASS")

    result = run_with_infrastructure_replacement("D1", run_one)
    assert calls == [("D1", 1), ("D1", 2)]
    assert result["infrastructure_replacement_count"] == 1
    assert result["policy_valid_result_count"] == 1
    assert result["valid_result"]["classification"] == "FULL_LAP_PASS"


def test_two_infrastructure_failures_stop_without_policy_evidence() -> None:
    calls: list[tuple[str, int]] = []

    def run_one(policy: str, number: int) -> dict:
        calls.append((policy, number))
        return _attempt("INFRA_FAIL")

    result = run_with_infrastructure_replacement("D1", run_one)
    assert calls == [("D1", 1), ("D1", 2)]
    assert result["valid_result"] is None
    assert result["policy_valid_result_count"] == 0
    assert result["stop_reason"] == "TWO_INFRASTRUCTURE_INVALID_ATTEMPTS"


def test_r1_has_the_same_bounded_infrastructure_replacement() -> None:
    calls: list[tuple[str, int]] = []

    def run_one(policy: str, number: int) -> dict:
        calls.append((policy, number))
        return _attempt("INFRA_FAIL" if number == 1 else "POLICY_FAIL")

    result = run_with_infrastructure_replacement("R1", run_one)
    assert calls == [("R1", 1), ("R1", 2)]
    assert result["physical_attempt_count"] == 2
    assert result["valid_result"]["classification"] == "POLICY_FAIL"


def test_registered_recheck_classification_rules() -> None:
    assert classify_recheck(_attempt("FULL_LAP_PASS"), None)["classification"] == (
        POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED
    )
    assert classify_recheck(
        _attempt("POLICY_FAIL"), _attempt("FULL_LAP_PASS"),
    )["classification"] == DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED
    assert classify_recheck(
        _attempt("POLICY_FAIL"), _attempt("POLICY_FAIL"),
    )["classification"] == SHARED_1P0_LANE_WEAKNESS_SUPPORTED
    assert classify_recheck(None, None)["classification"] == MIXED_OR_INCONCLUSIVE
    assert classify_recheck(_attempt("POLICY_FAIL"), None)["classification"] == MIXED_OR_INCONCLUSIVE


def test_s11_s12_are_protected_and_absent_from_sources() -> None:
    config = load_config(CONFIG_PATH, REPO)
    assert tuple(config.payload["protected_scenarios"]) == ("11", "12")
    assert config.payload["permissions"]["holdout_access_permitted"] is False
    for source in config.base.sources.values():
        assert "scenario_11" not in source["path"]
        assert "scenario_12" not in source["path"]


class _Route:
    length = 30.50461070080936
    points = ((0.0, 0.0),)

    def project(self, position):
        return SimpleNamespace(
            s=float(position[0]), distance=abs(float(position[1])), signed_error=float(position[1]),
        )

    def point_at(self, route_s):
        return (float(route_s), 0.0)


class _CommandClient:
    def __init__(self) -> None:
        self.commands: list[tuple[str, float]] = []

    def command_steering(self, value: float) -> None:
        self.commands.append(("steering", float(value)))

    def command_speed(self, value: float) -> None:
        self.commands.append(("speed", float(value)))


def test_shadow_expert_never_commands_vehicle() -> None:
    shadow, _ = shadow_expert_steering(_Route(), {"x": 1.0, "y": 0.2, "yaw": 0.1})
    client = _CommandClient()
    policy = -0.123
    issue_policy_commands(client, policy, 1.0)
    assert client.commands == [("steering", policy), ("speed", 1.0)]
    assert shadow != policy


def test_safe_stop_guard_stops_after_success_and_failure() -> None:
    calls: list[str] = []

    def stop() -> list[str]:
        calls.append("stop")
        return []

    assert safe_stop_guard(lambda: "ok", stop) == "ok"
    assert calls == ["stop"]
    with pytest.raises(RuntimeError, match="forced"):
        safe_stop_guard(lambda: (_ for _ in ()).throw(RuntimeError("forced")), stop)
    assert calls == ["stop", "stop"]


def test_safe_stop_failure_is_a_gate_failure() -> None:
    with pytest.raises(DiagnosisGateError, match="safe stop failed"):
        safe_stop_guard(lambda: "ok", lambda: ["speed stop failed"])
