from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from physicar_e2e.random_cone_d1_late_lap_diagnosis import (
    DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED,
    MINIMUM_FREE_BYTES,
    MIXED_OR_INCONCLUSIVE,
    POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED,
    PROTECTED_SCENARIOS,
    ROUTE_BINS,
    ROUTE_LENGTH_M,
    SHARED_LANE_WEAKNESS_SUPPORTED,
    DiagnosisGateError,
    classify_evidence,
    disk_gate,
    issue_policy_commands,
    load_config,
    route_bin_index,
    run_compact_live_loop,
    run_conditional_live,
    shadow_expert_steering,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/random_cone_d1_late_lap_diagnosis_v1.json"


def _preserved_s09() -> dict:
    return {
        "D1": {
            "cone_avoidance": "PASS",
            "route_recovery": "PASS",
            "failure_route_s_m": 29.307113445990467,
        }
    }


def _offline(*, late_regression: bool = True, available: bool = True) -> dict:
    if not available:
        return {"result": "UNAVAILABLE"}
    return {
        "result": "PASS",
        "late_bin_assessment": {"d1_late_bin_regression": late_regression},
    }


def _run(classification: str, route_s: float = 29.0) -> dict:
    return {"classification": classification, "metrics": {"final_route_s_m": route_s}}


def test_config_is_diagnostic_only_and_preserves_fixed_contract() -> None:
    config = load_config(CONFIG_PATH, REPO)
    assert tuple(tuple(item) for item in config.payload["route_bins_m"]) == ROUTE_BINS
    assert config.payload["live_run_policy"]["record_images"] is False
    assert config.payload["live_run_policy"]["record_bags"] is False
    assert config.payload["live_run_policy"]["shadow_expert_control_authority"] is False
    assert all(value is False for value in config.payload["permissions"].values())
    assert config.payload["permissions"]["training_permitted"] is False
    assert config.payload["permissions"]["bag_collection_permitted"] is False
    assert config.payload["permissions"]["dagger_iteration2_permitted"] is False
    assert config.payload["permissions"]["model_artifact_changes_permitted"] is False


def test_no_training_or_bag_framework_is_invoked_by_diagnosis_module() -> None:
    source = (REPO / "src/physicar_e2e/random_cone_d1_late_lap_diagnosis.py").read_text(encoding="utf-8")
    assert "train_temporal_resumable" not in source
    assert "training_stage(" not in source
    assert "DockerRosBackend" not in source
    assert "start_recorder" not in source
    assert "create DAgger" not in source


def test_s09_s10_remain_validation_only_and_s11_s12_are_protected() -> None:
    config = load_config(CONFIG_PATH, REPO)
    assert PROTECTED_SCENARIOS == ("11", "12")
    assert tuple(config.payload["protected_scenarios"]) == PROTECTED_SCENARIOS
    assert config.payload["permissions"]["holdout_access_permitted"] is False
    assert config.payload["permissions"]["dagger_collection_permitted"] is False
    aggregate = config.sources["dagger1_aggregate_manifest"]
    assert aggregate["expert_sequence_count"] == 6706
    assert aggregate["dagger1_sequence_count"] == 1483
    assert aggregate["sequence_count"] == 8189


def test_route_bins_are_fixed_exactly_and_cover_endpoint_once() -> None:
    assert ROUTE_BINS == (
        (0.0, 10.0),
        (10.0, 20.0),
        (20.0, 26.0),
        (26.0, 30.50461070080936),
    )
    assert route_bin_index(0.0) == 0
    assert route_bin_index(9.999999) == 0
    assert route_bin_index(10.0) == 1
    assert route_bin_index(20.0) == 2
    assert route_bin_index(26.0) == 3
    assert route_bin_index(ROUTE_LENGTH_M) == 3
    assert route_bin_index(-0.01) is None
    assert route_bin_index(ROUTE_LENGTH_M + 0.01) is None


def test_disk_gate_requires_at_least_5p5_gib() -> None:
    passing = lambda _path: SimpleNamespace(total=20 * 1024**3, used=10 * 1024**3, free=MINIMUM_FREE_BYTES)
    failing = lambda _path: SimpleNamespace(total=20 * 1024**3, used=15 * 1024**3, free=MINIMUM_FREE_BYTES - 1)
    assert disk_gate("/", disk_usage=passing)["result"] == "PASS"
    with pytest.raises(DiagnosisGateError, match="at least 5.500 GiB"):
        disk_gate("/", disk_usage=failing)


def test_d1_runs_exactly_once_and_pass_prevents_r1() -> None:
    calls: list[str] = []

    def run_one(name: str) -> dict:
        calls.append(name)
        return _run("FULL_LAP_PASS")

    result = run_conditional_live(run_one)
    assert calls == ["D1"]
    assert result["run_counts"] == {"D1": 1, "R1": 0}
    assert result["R1"] is None
    assert result["r1_gate_reason"] == "NOT_RUN_D1_FULL_LAP_PASS"


def test_r1_runs_once_only_after_d1_policy_fail_and_no_retry_occurs() -> None:
    calls: list[str] = []

    def run_one(name: str) -> dict:
        calls.append(name)
        return _run("POLICY_FAIL" if name == "D1" else "FULL_LAP_PASS")

    result = run_conditional_live(run_one)
    assert calls == ["D1", "R1"]
    assert result["run_counts"] == {"D1": 1, "R1": 1}
    assert result["r1_gate_reason"] == "RUN_ONCE_AFTER_D1_POLICY_FAIL"


def test_d1_infrastructure_failure_does_not_authorize_r1() -> None:
    calls: list[str] = []

    def run_one(name: str) -> dict:
        calls.append(name)
        return _run("INFRA_FAIL")

    result = run_conditional_live(run_one)
    assert calls == ["D1"]
    assert result["R1"] is None
    assert result["r1_gate_reason"] == "NOT_RUN_D1_INFRASTRUCTURE_INVALID"


class _Route:
    length = ROUTE_LENGTH_M
    points = ((0.0, 0.0),)

    def project(self, position):
        return SimpleNamespace(s=float(position[0]), distance=abs(float(position[1])), signed_error=float(position[1]))

    def point_at(self, route_s):
        return (float(route_s), 0.0)


class _CommandClient:
    def __init__(self):
        self.commands: list[tuple[str, float]] = []

    def command_steering(self, value):
        self.commands.append(("steering", float(value)))

    def command_speed(self, value):
        self.commands.append(("speed", float(value)))


def test_compact_shadow_expert_never_commands_vehicle() -> None:
    route = _Route()
    pose = {"x": 1.0, "y": 0.2, "yaw": 0.1}
    shadow, geometry = shadow_expert_steering(route, pose)
    assert geometry["cte_m"] == pytest.approx(0.2)
    client = _CommandClient()
    policy = -0.123
    assert shadow != policy
    issue_policy_commands(client, policy, 1.0)
    assert client.commands == [("steering", policy), ("speed", 1.0)]
    assert all(value != shadow for name, value in client.commands if name == "steering")


class _StopClient:
    def __init__(self):
        self.stop_calls = 0

    def safe_stop(self):
        self.stop_calls += 1
        return []


def test_live_loop_always_safe_stops_after_failure() -> None:
    config = load_config(CONFIG_PATH, REPO)
    client = _StopClient()
    initial = SimpleNamespace(route=_Route(), pose={"x": 0.0, "y": 0.0, "yaw": 0.0})

    def fail_warmup(_client, _config):
        raise RuntimeError("forced warmup failure")

    metrics, telemetry = run_compact_live_loop(
        client, object(), config, initial, policy_name="D1", warm_buffer=fail_warmup,
    )
    assert telemetry == []
    assert metrics["result"] == "FAIL"
    assert metrics["safe_stop_success"] is True
    assert client.stop_calls == 1


def test_registered_classification_rules() -> None:
    post = classify_evidence(
        d1_run=_run("FULL_LAP_PASS"), r1_run=None,
        offline_route_bins=_offline(), preserved_s09=_preserved_s09(),
    )
    assert post["classification"] == POST_AVOIDANCE_RESIDUAL_SHIFT_SUPPORTED

    regression = classify_evidence(
        d1_run=_run("POLICY_FAIL", 29.1), r1_run=_run("FULL_LAP_PASS"),
        offline_route_bins=_offline(late_regression=True), preserved_s09=_preserved_s09(),
    )
    assert regression["classification"] == DAGGER_INDUCED_LATE_LAP_REGRESSION_SUPPORTED

    shared = classify_evidence(
        d1_run=_run("POLICY_FAIL", 29.1), r1_run=_run("POLICY_FAIL", 28.0),
        offline_route_bins=_offline(), preserved_s09=_preserved_s09(),
    )
    assert shared["classification"] == SHARED_LANE_WEAKNESS_SUPPORTED

    inconclusive = classify_evidence(
        d1_run=_run("INFRA_FAIL"), r1_run=None,
        offline_route_bins=_offline(available=False), preserved_s09=_preserved_s09(),
    )
    assert inconclusive["classification"] == MIXED_OR_INCONCLUSIVE


def test_dagger_regression_requires_late_failure_and_offline_support_when_available() -> None:
    early = classify_evidence(
        d1_run=_run("POLICY_FAIL", 20.0), r1_run=_run("FULL_LAP_PASS"),
        offline_route_bins=_offline(), preserved_s09=_preserved_s09(),
    )
    assert early["classification"] == MIXED_OR_INCONCLUSIVE
    conflicting = classify_evidence(
        d1_run=_run("POLICY_FAIL", 29.0), r1_run=_run("FULL_LAP_PASS"),
        offline_route_bins=_offline(late_regression=False), preserved_s09=_preserved_s09(),
    )
    assert conflicting["classification"] == MIXED_OR_INCONCLUSIVE
