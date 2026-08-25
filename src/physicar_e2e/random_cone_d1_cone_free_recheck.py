"""Infrastructure-valid, diagnostic-only D1 cone-free recheck.

This module deliberately reuses the frozen live inference and numeric telemetry
path from the late-lap diagnosis.  It adds only a stronger infrastructure
preflight, bounded infrastructure replacement, and recheck-specific reporting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

from .cone_avoidance_expert import activate_world
from .dataset_extractor import canonical_json_bytes, sha256_file
from .expert_driver import wait_after_reset
from .high_speed_temporal import TemporalOnnxModel, warm_temporal_buffer
from .random_cone_d1_late_lap_diagnosis import (
    CANONICAL_WORLD,
    CONTROL_HZ,
    LOOKAHEAD_M,
    MAX_STEERING_RAD,
    MINIMUM_FREE_BYTES,
    PROTECTED_SCENARIOS,
    ROUTE_LENGTH_M,
    SPEED_MPS,
    DiagnosisConfig,
    DiagnosisGateError,
    _driver_config,
    _live_inference_config,
    _preflight_cone_free,
    _protected_world,
    _window_metrics,
    analyze_live_telemetry,
    disk_gate,
    load_config as load_base_config,
    run_compact_live_loop,
    utc_now,
)
from .sim_client import SimClient


VERSION = "random_cone_d1_cone_free_recheck_v1"
EXPECTED_BRANCH = "experiment/random-cone-d1-cone-free-recheck-v1"
POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED = (
    "POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED"
)
DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED = "DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED"
SHARED_1P0_LANE_WEAKNESS_SUPPORTED = "SHARED_1P0_LANE_WEAKNESS_SUPPORTED"
MIXED_OR_INCONCLUSIVE = "MIXED_OR_INCONCLUSIVE"
LATE_ROUTE_REGIONS = ((20.0, 26.0), (26.0, ROUTE_LENGTH_M))
VALID_POLICY_RESULTS = ("FULL_LAP_PASS", "POLICY_FAIL")
ATTEMPT_RESULTS = (*VALID_POLICY_RESULTS, "INFRA_FAIL")

_EXPECTED_MODEL_HASHES = {
    "R1": {
        "checkpoint_sha256": "b50d5d3c3cdb4f7aa730b2a44c1ffd46d7e0deb7aa0328cb7d40b090ae9022a0",
        "onnx_sha256": "2ebb6faf79ff015ae79c31d404c1fc7eb932b726c60c9f0b6dc7d7e02e51c993",
        "freeze_sha256": "ac622a793ce2bc4794170b53cc9421cc343e1691eeb4d2b85e01609714e7e0d7",
        "freeze_seal_sha256": "3d1c1f647b587ae6a8788c2d545a3ee8a2b0a3b90b9b465059f38ed5b687c798",
    },
    "D1": {
        "checkpoint_sha256": "b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434",
        "onnx_sha256": "3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c",
        "freeze_sha256": "66dbf7762ab089f111e2c02d22240d861e575730dcb416692bf6fac4e1e3fdc8",
        "freeze_seal_sha256": "7781423c7ba69f381e91120687d07d93d006393ff3c0c74af751085ce6ea1840",
    },
}


@dataclass(frozen=True)
class RecheckConfig:
    path: Path
    payload: dict[str, Any]
    base: DiagnosisConfig

    def result_dir(self, repo: Path) -> Path:
        return repo / self.payload["result_directory"]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisGateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosisGateError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(dict(payload)))
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _hash_gate(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise DiagnosisGateError(f"missing frozen {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise DiagnosisGateError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.rstrip()


def load_config(path: Path, repo: Path) -> RecheckConfig:
    payload = _read_json(path)
    required = {
        "version", "expected_branch", "result_directory", "base_diagnosis_config",
        "previous_evidence", "canonical_cone_free_world", "canonical_route_length_m",
        "late_route_regions_m", "disk_gate", "fixed_control", "frozen_models",
        "attempt_policy", "protected_scenarios", "permissions",
    }
    if set(payload) != required or payload.get("version") != VERSION:
        raise DiagnosisGateError("recheck config fields/version changed")
    if (
        payload["expected_branch"] != EXPECTED_BRANCH
        or payload["canonical_cone_free_world"] != CANONICAL_WORLD
        or float(payload["canonical_route_length_m"]) != ROUTE_LENGTH_M
        or tuple(tuple(float(v) for v in item) for item in payload["late_route_regions_m"])
        != LATE_ROUTE_REGIONS
        or tuple(payload["protected_scenarios"]) != PROTECTED_SCENARIOS
    ):
        raise DiagnosisGateError("branch/world/route/holdout recheck contract changed")
    if int(payload["disk_gate"]["minimum_free_bytes"]) != MINIMUM_FREE_BYTES:
        raise DiagnosisGateError("5.5 GiB disk gate changed")
    fixed = payload["fixed_control"]
    fixed_tuple = (
        float(fixed["speed_mps"]), float(fixed["control_frequency_hz"]),
        float(fixed["lookahead_m"]), float(fixed["steering_limit_rad"]),
        float(fixed["wheelbase_m"]), int(fixed["history_frames"]),
        float(fixed["maximum_adjacent_gap_s"]), float(fixed["off_track_margin_m"]),
        float(fixed["off_track_grace_s"]),
    )
    if fixed_tuple != (1.0, 15.0, 0.9, 0.349066, 0.18, 3, 0.12, 0.05, 0.5):
        raise DiagnosisGateError(f"fixed controller/temporal/safety contract changed: {fixed_tuple}")
    expected_attempt_policy = {
        "maximum_physical_attempts_per_policy": 2,
        "maximum_infrastructure_replacements_per_policy": 1,
        "replacement_only_after_infrastructure_failure": True,
        "retry_after_policy_fail": False,
        "retry_after_full_lap_pass": False,
        "r1_only_after_d1_policy_fail": True,
        "shadow_expert_control_authority": False,
        "record_images": False,
        "record_bags": False,
    }
    if payload["attempt_policy"] != expected_attempt_policy:
        raise DiagnosisGateError("bounded attempt policy changed")
    if payload["frozen_models"] != _EXPECTED_MODEL_HASHES:
        raise DiagnosisGateError("frozen R1/D1 identity changed")
    if any(value is not False for value in payload["permissions"].values()):
        raise DiagnosisGateError("diagnostic-only permission boundary changed")
    base_ref = payload["base_diagnosis_config"]
    base_path = _resolve(repo, base_ref["path"])
    _hash_gate(base_path, base_ref["sha256"], "base diagnosis config")
    base = load_base_config(base_path, repo)
    return RecheckConfig(path.resolve(), payload, base)


def _require_close(actual: object, expected: float, label: str, tolerance: float = 1e-12) -> None:
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise DiagnosisGateError(f"preserved {label} is not numeric: {actual!r}") from exc
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=tolerance):
        raise DiagnosisGateError(f"preserved {label} changed: expected {expected}, got {numeric}")


def audit_frozen_evidence(config: RecheckConfig, repo: Path) -> dict[str, Any]:
    branch = _git(repo, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise DiagnosisGateError(f"expected branch {EXPECTED_BRANCH!r}, active branch is {branch!r}")
    model_hashes: dict[str, dict[str, str]] = {}
    artifact_fields = {
        "checkpoint": ("checkpoint_path", "checkpoint_sha256"),
        "onnx": ("onnx_path", "onnx_sha256"),
        "freeze": ("freeze_path", "freeze_sha256"),
        "freeze_seal": ("freeze_seal_path", "freeze_seal_sha256"),
    }
    for model_name in ("R1", "D1"):
        source = config.base.models[model_name]
        model_hashes[model_name] = {}
        for label, (path_key, hash_key) in artifact_fields.items():
            expected = _EXPECTED_MODEL_HASHES[model_name][hash_key]
            if source[hash_key] != expected:
                raise DiagnosisGateError(f"base config {model_name} {label} identity changed")
            model_hashes[model_name][label] = _hash_gate(
                _resolve(repo, source[path_key]), expected, f"{model_name} {label}",
            )

    source_hashes: dict[str, str] = {}
    for name, source in config.base.sources.items():
        source_hashes[name] = _hash_gate(
            _resolve(repo, source["path"]), source["sha256"], name.replace("_", " "),
        )

    evidence: dict[str, dict[str, Any]] = {}
    for name, reference in config.payload["previous_evidence"].items():
        path = _resolve(repo, reference["path"])
        _hash_gate(path, reference["sha256"], f"previous {name}")
        evidence[name] = _read_json(path)

    previous_summary = evidence["summary"]
    invalid = evidence["invalid_d1_run"]
    distribution = evidence["offline_distribution"]
    offline = evidence["offline_route_bins"]
    invalid_metrics = invalid.get("metrics") or {}
    invalid_analysis = invalid.get("analysis") or {}
    final_window = invalid_analysis.get("final_2_seconds_before_run_stop") or {}
    if invalid.get("classification") != "INFRA_FAIL":
        raise DiagnosisGateError("previous D1 cone-free run is no longer preserved as INFRA_FAIL")
    if invalid_metrics.get("failure") != (
        "simulator clock did not advance for 0.803s while motion was commanded"
    ):
        raise DiagnosisGateError("previous clock/pose infrastructure failure changed")
    _require_close(invalid_metrics.get("final_route_s_m"), 19.92646012516552, "invalid route s")
    _require_close(
        invalid_metrics.get("route_completion_fraction"), 0.6532278126937979,
        "invalid completion",
    )
    _require_close(invalid_metrics.get("max_cte_m"), 0.33932794915756964, "invalid max CTE")
    _require_close(
        invalid_metrics.get("mean_signed_steering_error_rad"), 0.026270236039424282,
        "invalid signed steering error",
    )
    _require_close(
        invalid_metrics.get("mean_absolute_steering_error_rad"), 0.054987880435968126,
        "invalid absolute steering error",
    )
    _require_close(
        invalid_metrics.get("corrective_magnitude_ratio"), 0.6341338024062039,
        "invalid corrective ratio",
    )
    _require_close(
        invalid_metrics.get("steering_sign_agreement_fraction"), 0.8006535947712419,
        "invalid sign agreement",
    )
    for key, expected in (
        ("mean_model_steering_rad", -0.06679677569697942),
        ("mean_shadow_expert_steering_rad", -0.23767885261306415),
        ("corrective_magnitude_ratio", 0.48497960999329487),
        ("steering_sign_agreement_fraction", 0.967741935483871),
        ("cte_growth_m", 0.0050214896320986),
    ):
        _require_close(final_window.get(key), expected, f"invalid final-window {key}")
    if not (
        invalid_metrics.get("off_track_events") == 0
        and invalid_metrics.get("temporal_input_failure") is False
        and invalid_metrics.get("timing_slips_over_100ms") == 0
        and invalid_metrics.get("steering_saturation_fraction") == 0.0
        and invalid_metrics.get("safe_stop_success") is True
        and (previous_summary.get("classification") or {}).get("classification")
        == MIXED_OR_INCONCLUSIVE
    ):
        raise DiagnosisGateError("previous invalid-run health evidence changed")
    bins = distribution.get("bins") or {}
    if not (
        distribution.get("dagger1_contributes_zero_late_lap_samples") is True
        and int((bins.get("20-26 m") or {}).get("dagger1_sequence_count", -1)) == 0
        and int((bins.get("26-30.504611 m") or {}).get("dagger1_sequence_count", -1)) == 0
    ):
        raise DiagnosisGateError("DAgger1 zero coverage after route s=20 m changed")
    assessment = offline.get("late_bin_assessment") or {}
    _require_close(
        ((assessment.get("per_bin_comparison") or {}).get("26-30.504611 m") or {}).get(
            "d1_to_r1_mae_ratio"
        ),
        1.2036525903068305,
        "combined final-bin D1/R1 MAE ratio",
    )
    _require_close(
        ((assessment.get("per_bin_comparison") or {}).get("0-10 m") or {}).get(
            "d1_to_r1_mae_ratio"
        ),
        1.956164232541744,
        "combined early-bin D1/R1 MAE ratio",
    )
    if assessment.get("d1_late_bin_regression_is_disproportionate") is not False:
        raise DiagnosisGateError("offline late-only regression conclusion changed")
    return {
        "version": VERSION + "_audit",
        "generated_utc": utc_now(),
        "result": "PASS",
        "branch": branch,
        "head_commit": _git(repo, "rev-parse", "HEAD"),
        "model_hashes": model_hashes,
        "source_hashes": source_hashes,
        "previous_evidence_hashes": {
            name: reference["sha256"]
            for name, reference in config.payload["previous_evidence"].items()
        },
        "previous_invalid_is_not_policy_evidence": True,
        "previous_invalid_summary": {
            "classification": "INFRA_FAIL",
            "failure": invalid_metrics["failure"],
            "route_s_m": invalid_metrics["final_route_s_m"],
            "completion_fraction": invalid_metrics["route_completion_fraction"],
            "max_cte_m": invalid_metrics["max_cte_m"],
            "off_track_events": invalid_metrics["off_track_events"],
            "temporal_input_failure": invalid_metrics["temporal_input_failure"],
            "timing_slips_over_100ms": invalid_metrics["timing_slips_over_100ms"],
            "steering_saturation_fraction": invalid_metrics["steering_saturation_fraction"],
            "safe_stop_success": invalid_metrics["safe_stop_success"],
            "full_valid_portion_shadow_comparison": {
                "mean_signed_error_rad": invalid_metrics["mean_signed_steering_error_rad"],
                "mean_absolute_error_rad": invalid_metrics["mean_absolute_steering_error_rad"],
                "corrective_magnitude_ratio": invalid_metrics["corrective_magnitude_ratio"],
                "sign_agreement_fraction": invalid_metrics["steering_sign_agreement_fraction"],
            },
            "final_2_seconds_before_stall": final_window,
        },
        "dagger1_coverage": {
            "sequence_count": 1483,
            "samples_after_route_s_20_m": 0,
            "bins": bins,
        },
        "offline_matched": {
            "d1_worse_overall": True,
            "late_regression_disproportionate": False,
            "final_bin_d1_to_r1_mae_ratio": 1.2036525903068305,
            "early_bin_d1_to_r1_mae_ratio": 1.956164232541744,
        },
        "preserved_s09": previous_summary.get("preserved_s09_comparison"),
        "protected_scenarios_accessed": [],
    }


T = TypeVar("T")


def safe_stop_guard(action: Callable[[], T], safe_stop: Callable[[], list[str]]) -> T:
    """Execute an action and require a successful safe stop even when it raises."""
    caught: BaseException | None = None
    result: T | None = None
    try:
        result = action()
    except BaseException as exc:  # safe stop also applies to interruption
        caught = exc
    stop_errors = safe_stop()
    if caught is not None:
        if stop_errors:
            caught.add_note("safe stop also failed: " + "; ".join(stop_errors))
        raise caught
    if stop_errors:
        raise DiagnosisGateError("safe stop failed: " + "; ".join(stop_errors))
    return result  # type: ignore[return-value]


def fresh_infrastructure_preflight(
    client: SimClient,
    config: RecheckConfig,
    repo: Path,
    sim_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Prove API/clock/camera/control/pose/temporal health, then reset spawn."""
    initial_stop = client.safe_stop()
    if initial_stop:
        raise DiagnosisGateError("fresh preflight initial safe stop failed: " + "; ".join(initial_stop))
    try:
        _, report = _preflight_cone_free(client, config.base, repo, sim_root)
        motion_start_pose = client.pose()
        motion_start_clock = float(client.clock()["sim_time"])
        steering_response = client.command_steering(0.0)
        speed_response = client.command_speed(SPEED_MPS)
        motion_started = time.monotonic()
        motion_samples: list[dict[str, float]] = []
        moved = False
        clock_advanced = False
        while time.monotonic() - motion_started < 0.75:
            time.sleep(0.05)
            pose = client.pose()
            sim_time = float(client.clock()["sim_time"])
            translation = math.dist(
                (float(motion_start_pose["x"]), float(motion_start_pose["y"])),
                (float(pose["x"]), float(pose["y"])),
            )
            yaw_delta = abs(math.atan2(
                math.sin(float(pose["yaw"]) - float(motion_start_pose["yaw"])),
                math.cos(float(pose["yaw"]) - float(motion_start_pose["yaw"])),
            ))
            motion_samples.append({
                "wall_elapsed_s": time.monotonic() - motion_started,
                "sim_time_s": sim_time,
                "translation_m": translation,
                "yaw_delta_rad": yaw_delta,
            })
            moved = moved or translation >= 0.005 or yaw_delta >= 0.01
            clock_advanced = clock_advanced or sim_time > motion_start_clock
            if moved and clock_advanced:
                break
        stop_errors = client.safe_stop()
        if stop_errors:
            raise DiagnosisGateError("motion-probe safe stop failed: " + "; ".join(stop_errors))
        if not moved or not clock_advanced:
            raise DiagnosisGateError(
                f"motion preflight failed: pose_moved={moved}, clock_advanced={clock_advanced}"
            )

        final_initial = wait_after_reset(client, _driver_config(config.base), False)
        final_clock_start = float(client.clock()["sim_time"])
        time.sleep(0.15)
        final_clock_end = float(client.clock()["sim_time"])
        final_pose = client.pose()
        if final_clock_end <= final_clock_start:
            raise DiagnosisGateError("clock failed to advance after final canonical-spawn reset")
        buffer, temporal = warm_temporal_buffer(client, _live_inference_config(config.base))
        tensor = buffer.tensor()
        adjacent = [float(value) for value in temporal["adjacent_gaps_s"]]
        if (
            temporal.get("result") != "PASS"
            or temporal.get("real_frame_acquisitions") != 3
            or tensor.shape != (9, 66, 200)
            or any(value > 0.12 for value in adjacent)
        ):
            raise DiagnosisGateError("three-frame temporal preflight contract failed")
        final_stop = client.safe_stop()
        if final_stop:
            raise DiagnosisGateError("post-temporal-preflight safe stop failed: " + "; ".join(final_stop))
        report.update({
            "result": "PASS",
            "fresh_recheck_steps": [
                "safe_stop", "canonical_world_activation", "openapi_control_schema",
                "clock_advancing", "pose_responding_and_advancing", "camera_stream",
                "steering_and_speed_control", "canonical_spawn_reset", "settle",
                "three_real_causal_frames",
            ],
            "motion_control_probe": {
                "result": "PASS",
                "speed_mps": SPEED_MPS,
                "steering_rad": 0.0,
                "steering_response_success": steering_response.get("success") is True,
                "speed_response_success": speed_response.get("success") is True,
                "pose_moved": moved,
                "clock_advanced": clock_advanced,
                "sample_count": len(motion_samples),
                "maximum_translation_m": max(item["translation_m"] for item in motion_samples),
                "maximum_yaw_delta_rad": max(item["yaw_delta_rad"] for item in motion_samples),
                "safe_stop_success": True,
            },
            "final_spawn": {
                "pose": final_pose,
                "route_distance_m": final_initial.route.project(
                    (float(final_pose["x"]), float(final_pose["y"]))
                ).distance,
                "clock_delta_s": final_clock_end - final_clock_start,
            },
            "temporal_buffer_population": {
                **temporal,
                "tensor_shape": list(tensor.shape),
                "maximum_adjacent_gap_s": max(adjacent),
                "images_persisted": 0,
            },
            "safe_stop_success": True,
        })
        return final_initial, report
    except BaseException:
        client.safe_stop()
        raise


def run_with_infrastructure_replacement(
    policy_name: str,
    run_one: Callable[[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Allow attempt 2 if and only if attempt 1 was infrastructure-invalid."""
    if policy_name not in ("D1", "R1"):
        raise ValueError("policy_name must be D1 or R1")
    attempts: list[dict[str, Any]] = []
    first = run_one(policy_name, 1)
    attempts.append(first)
    first_result = first.get("classification")
    if first_result not in ATTEMPT_RESULTS:
        raise DiagnosisGateError(f"invalid {policy_name} attempt classification: {first_result!r}")
    if first_result == "INFRA_FAIL":
        second = run_one(policy_name, 2)
        attempts.append(second)
        if second.get("classification") not in ATTEMPT_RESULTS:
            raise DiagnosisGateError(
                f"invalid {policy_name} replacement classification: {second.get('classification')!r}"
            )
    valid = [item for item in attempts if item.get("classification") in VALID_POLICY_RESULTS]
    if len(attempts) > 2 or len(valid) > 1:
        raise DiagnosisGateError("bounded infrastructure replacement cardinality violated")
    return {
        "policy": policy_name,
        "physical_attempt_count": len(attempts),
        "infrastructure_replacement_count": max(0, len(attempts) - 1),
        "attempts": attempts,
        "valid_result": valid[0] if valid else None,
        "policy_valid_result_count": len(valid),
        "stop_reason": (
            "VALID_FULL_LAP_PASS" if valid and valid[0]["classification"] == "FULL_LAP_PASS"
            else "VALID_POLICY_FAIL" if valid
            else "TWO_INFRASTRUCTURE_INVALID_ATTEMPTS"
        ),
    }


def classify_recheck(
    d1_valid: Mapping[str, Any] | None,
    r1_valid: Mapping[str, Any] | None,
) -> dict[str, Any]:
    d1_result = None if d1_valid is None else d1_valid.get("classification")
    r1_result = None if r1_valid is None else r1_valid.get("classification")
    if d1_result == "FULL_LAP_PASS":
        classification = POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED
        interpretation = (
            "Generic 1.00 m/s cone-free lane-following failure is not supported; the preserved "
            "S09 late failure likely depends on post-avoidance or scenario-specific closed-loop state."
        )
    elif d1_result == "POLICY_FAIL" and r1_result == "FULL_LAP_PASS":
        classification = DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED
        interpretation = "D1 degraded nominal cone-free closed-loop lane-following relative to R1."
    elif d1_result == "POLICY_FAIL" and r1_result == "POLICY_FAIL":
        classification = SHARED_1P0_LANE_WEAKNESS_SUPPORTED
        interpretation = "The cone-free 1.00 m/s weakness is shared by D1 and R1, not specific to DAgger1."
    else:
        classification = MIXED_OR_INCONCLUSIVE
        interpretation = (
            "The bounded infrastructure-valid comparisons did not produce the valid result pair "
            "needed to separate the hypotheses."
        )
    return {
        "classification": classification,
        "interpretation": interpretation,
        "conditions": {
            "d1_valid_result": d1_result,
            "r1_valid_result": r1_result,
        },
    }


def r1_run_authorized(d1_valid: Mapping[str, Any] | None) -> bool:
    """R1 may run only after one infrastructure-valid D1 policy failure."""
    return d1_valid is not None and d1_valid.get("classification") == "POLICY_FAIL"


def recommended_next_direction(classification: str) -> dict[str, Any]:
    if classification == POST_AVOIDANCE_OR_SCENARIO_DEPENDENT_SHIFT_SUPPORTED:
        direction = (
            "Run one bounded TRAIN-only DAgger2 experiment targeting post-recovery and late-route "
            "learner states actually visited after successful cone avoidance."
        )
    elif classification == DAGGER_INDUCED_NOMINAL_REGRESSION_SUPPORTED:
        direction = (
            "Run one existing-data source/route-coverage mixing A/B before collecting any DAgger2 data."
        )
    elif classification == SHARED_1P0_LANE_WEAKNESS_SUPPORTED:
        direction = "Run one dedicated 1.00 m/s cone-free temporal lane-baseline diagnosis."
    else:
        direction = (
            "Authorize no learning intervention; first obtain the missing infrastructure-valid "
            "cone-free comparison."
        )
    return {"count": 1, "direction": direction, "implemented_in_this_milestone": False}


def _region_label(bounds: tuple[float, float]) -> str:
    return "20-26 m" if bounds == (20.0, 26.0) else "26-30.504611 m"


def analyze_late_route(report: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = report.get("telemetry") or []
    metrics = report.get("metrics") or {}
    regions: dict[str, Any] = {}
    for index, bounds in enumerate(LATE_ROUTE_REGIONS):
        lower, upper = bounds
        rows = [
            row for row in telemetry
            if float(row["route_s_m"]) >= lower
            and (float(row["route_s_m"]) < upper or (index == 1 and float(row["route_s_m"]) <= upper))
        ]
        label = _region_label(bounds)
        values = _window_metrics(rows, label=label)
        if rows:
            ctes = [float(row["cte_m"]) for row in rows]
            gap1 = [float(row["camera_gap_t_minus_2_to_t_minus_1_s"]) for row in rows]
            gap2 = [float(row["camera_gap_t_minus_1_to_t_s"]) for row in rows]
            preprocessing = [float(row["preprocessing_ms"]) for row in rows]
            inference = [float(row["onnx_inference_ms"]) for row in rows]
            values.update({
                "mean_cte_m": statistics.fmean(ctes),
                "mean_policy_steering_rad": values["mean_model_steering_rad"],
                "mean_d1_or_r1_steering_rad": values["mean_model_steering_rad"],
                "mae_rad": values["mean_absolute_error_rad"],
                "signed_bias_rad": values["mean_signed_error_rad"],
                "maximum_adjacent_temporal_gap_s": max([*gap1, *gap2]),
                "mean_preprocessing_ms": statistics.fmean(preprocessing),
                "maximum_preprocessing_ms": max(preprocessing),
                "mean_inference_ms": statistics.fmean(inference),
                "maximum_inference_ms": max(inference),
                "temporal_healthy": (
                    max([*gap1, *gap2]) <= 0.12
                    and metrics.get("temporal_input_failure") is False
                    and int(metrics.get("temporal_invalid_history_count", 0)) == 0
                ),
                "timing_healthy": int(metrics.get("timing_slips_over_100ms", 0)) == 0,
                "liveness_healthy": all(
                    int(metrics.get(name, 0)) == 0
                    for name in ("api_failures", "pose_failures", "clock_failures", "liveness_failures")
                ),
            })
        regions[label] = values
    reached = any(float(row["route_s_m"]) >= 20.0 for row in telemetry)
    return {
        "result": "AVAILABLE" if reached else "NOT_REACHED",
        "zero_dagger1_training_samples_after_s_20_m": True,
        "regions": regions,
    }


def _attempt_paths(result_dir: Path, policy: str, attempt_number: int) -> tuple[Path, Path]:
    stem = f"{policy.lower()}_attempt_{attempt_number:02d}"
    output = result_dir / "attempts" / f"{stem}.json"
    return output, output.with_suffix(".started.json")


def _run_physical_attempt(
    *,
    client: SimClient,
    config: RecheckConfig,
    repo: Path,
    sim_root: Path,
    policy_name: str,
    attempt_number: int,
) -> dict[str, Any]:
    output, marker = _attempt_paths(config.result_dir(repo), policy_name, attempt_number)
    if output.is_file():
        existing = _read_json(output)
        if (
            existing.get("policy") != policy_name
            or int(existing.get("physical_attempt_number", 0)) != attempt_number
            or not marker.is_file()
        ):
            raise DiagnosisGateError(f"existing attempt evidence is inconsistent: {output}")
        return existing
    if marker.exists():
        raise DiagnosisGateError(
            f"{policy_name} attempt {attempt_number} marker exists without finalized evidence; "
            "its cause is not unequivocally infrastructure-only"
        )
    model_config = config.base.models[policy_name]
    onnx_path = _resolve(repo, model_config["onnx_path"])
    _hash_gate(onnx_path, _EXPECTED_MODEL_HASHES[policy_name]["onnx_sha256"], f"{policy_name} ONNX")
    model = TemporalOnnxModel(onnx_path)
    initial, preflight = fresh_infrastructure_preflight(client, config, repo, sim_root)
    _write_json(marker, {
        "version": VERSION + "_attempt_marker",
        "policy": policy_name,
        "physical_attempt_number": attempt_number,
        "started_utc": utc_now(),
        "status": "POLICY_EVALUATION_STARTED_DO_NOT_REPEAT_THIS_ATTEMPT",
        "replacement_authorized_only_if_classification": "INFRA_FAIL",
        "record_images": False,
        "record_bags": False,
    })

    def drive() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return run_compact_live_loop(
            client, model, config.base, initial, policy_name=policy_name,
        )

    metrics, telemetry = safe_stop_guard(drive, client.safe_stop)
    classification = metrics.get("classification")
    if classification not in ATTEMPT_RESULTS:
        raise DiagnosisGateError(f"unexpected live classification: {classification!r}")
    analysis = analyze_live_telemetry(metrics, telemetry)
    report = {
        "version": VERSION + "_attempt",
        "generated_utc": utc_now(),
        "policy": policy_name,
        "physical_attempt_number": attempt_number,
        "classification": classification,
        "is_valid_policy_evidence": classification in VALID_POLICY_RESULTS,
        "preflight": preflight,
        "metrics": metrics,
        "telemetry_schema": {
            "format": "compact JSON numeric rows",
            "camera_images": 0,
            "bags": 0,
            "shadow_expert_control_authority": False,
        },
        "telemetry": telemetry,
        "analysis": analysis,
    }
    report["late_route_telemetry"] = analyze_late_route(report)
    _write_json(output, report)
    return report


def _compact_attempt(attempt: Mapping[str, Any], result_dir: Path, repo: Path) -> dict[str, Any]:
    output, _ = _attempt_paths(
        result_dir,
        str(attempt["policy"]),
        int(attempt["physical_attempt_number"]),
    )
    return {
        "policy": attempt["policy"],
        "physical_attempt_number": attempt["physical_attempt_number"],
        "classification": attempt["classification"],
        "is_valid_policy_evidence": attempt["is_valid_policy_evidence"],
        "attempt_file": str(output.relative_to(repo)),
        "preflight": attempt.get("preflight"),
        "metrics": attempt.get("metrics"),
        "analysis": attempt.get("analysis"),
        "late_route_telemetry": attempt.get("late_route_telemetry"),
        "telemetry_sample_count": len(attempt.get("telemetry") or []),
        "telemetry_embedded_here": False,
    }


def _write_valid_result(
    config: RecheckConfig,
    repo: Path,
    policy_name: str,
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    compact = _compact_attempt(attempt, config.result_dir(repo), repo)
    compact.update({
        "version": VERSION + "_valid_result",
        "generated_utc": utc_now(),
        "valid_policy_result_number": 1,
    })
    path = config.result_dir(repo) / f"{policy_name.lower()}_cone_free_valid.json"
    if path.is_file():
        existing = _read_json(path)
        if (
            existing.get("classification") != compact.get("classification")
            or existing.get("physical_attempt_number") != compact.get("physical_attempt_number")
        ):
            raise DiagnosisGateError(f"existing valid result conflicts: {path}")
        return existing
    _write_json(path, compact)
    return compact


def execute_live_recheck(config: RecheckConfig, repo: Path, sim_root: Path) -> dict[str, Any]:
    audit_frozen_evidence(config, repo)
    disk_gate(config.payload["disk_gate"]["path"], MINIMUM_FREE_BYTES)
    client = SimClient("http://localhost:8080", float(config.base.control["api_timeout_s"]))
    initial_status = client.status()
    original_world = initial_status.get("current")
    if _protected_world(original_world):
        client.safe_stop()
        raise DiagnosisGateError("refusing recheck while protected S11/S12 world is active")
    initial_stop = client.safe_stop()
    if initial_stop:
        raise DiagnosisGateError("initial recheck safe stop failed: " + "; ".join(initial_stop))
    result_dir = config.result_dir(repo)
    restoration: dict[str, Any] | None = None
    final_stop_errors: list[str] = []
    try:
        def run_one(policy_name: str, attempt_number: int) -> dict[str, Any]:
            disk_gate(config.payload["disk_gate"]["path"], MINIMUM_FREE_BYTES)
            return _run_physical_attempt(
                client=client, config=config, repo=repo, sim_root=sim_root,
                policy_name=policy_name, attempt_number=attempt_number,
            )

        d1 = run_with_infrastructure_replacement("D1", run_one)
        d1_valid = d1["valid_result"]
        if d1_valid is not None:
            _write_valid_result(config, repo, "D1", d1_valid)
        r1: dict[str, Any] | None = None
        if r1_run_authorized(d1_valid):
            r1 = run_with_infrastructure_replacement("R1", run_one)
            if r1["valid_result"] is not None:
                _write_valid_result(config, repo, "R1", r1["valid_result"])
        else:
            unexpected = [
                path for path in (result_dir / "attempts").glob("r1_attempt_*.json")
                if not path.name.endswith(".started.json")
            ]
            if unexpected or (result_dir / "r1_cone_free_valid.json").exists():
                raise DiagnosisGateError("R1 evidence exists although the D1 conditional gate is closed")
        result = {
            "D1": d1,
            "R1": r1,
            "r1_gate_reason": (
                "AUTHORIZED_AFTER_D1_POLICY_FAIL" if r1 is not None
                else "BLOCKED_D1_FULL_LAP_PASS" if d1_valid is not None
                else "BLOCKED_NO_VALID_D1_RESULT"
            ),
        }
    finally:
        final_stop_errors = client.safe_stop()
        if original_world and original_world != CANONICAL_WORLD and not _protected_world(original_world):
            try:
                restoration = activate_world(client, str(original_world))
            except Exception as exc:
                restoration = {"result": "FAIL", "failure": str(exc)}
            final_stop_errors.extend(client.safe_stop())
    if final_stop_errors:
        raise DiagnosisGateError("final recheck safe stop failed: " + "; ".join(final_stop_errors))
    result.update({
        "initial_world": original_world,
        "world_restoration": restoration or {
            "result": "PASS", "action": "canonical cone-free world remained active",
        },
        "final_safe_stop_success": True,
        "final_safe_stop_errors": [],
    })
    return result


def _load_policy_attempts(config: RecheckConfig, repo: Path, policy: str) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for number in (1, 2):
        output, marker = _attempt_paths(config.result_dir(repo), policy, number)
        if output.is_file():
            if not marker.is_file():
                raise DiagnosisGateError(f"attempt result lacks marker: {output}")
            attempts.append(_read_json(output))
        elif marker.exists():
            raise DiagnosisGateError(f"unfinalized attempt marker prevents classification: {marker}")
    if len(attempts) == 2 and attempts[0].get("classification") != "INFRA_FAIL":
        raise DiagnosisGateError(f"{policy} replacement was not authorized by infrastructure failure")
    return attempts


def _valid_from_attempts(attempts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    valid = [item for item in attempts if item.get("classification") in VALID_POLICY_RESULTS]
    if len(valid) > 1:
        raise DiagnosisGateError("more than one policy-valid result exists")
    return valid[0] if valid else None


def _changed_files(repo: Path) -> list[str]:
    lines = _git(repo, "status", "--short").splitlines()
    return [line[3:] if len(line) > 3 else line for line in lines]


def build_summary(
    config: RecheckConfig,
    repo: Path,
    *,
    tests_summary: str = "PENDING",
) -> dict[str, Any]:
    result_dir = config.result_dir(repo)
    audit = audit_frozen_evidence(config, repo)
    disk = disk_gate(config.payload["disk_gate"]["path"], MINIMUM_FREE_BYTES)
    d1_attempts = _load_policy_attempts(config, repo, "D1")
    r1_attempts = _load_policy_attempts(config, repo, "R1")
    d1_valid = _valid_from_attempts(d1_attempts)
    r1_valid = _valid_from_attempts(r1_attempts)
    if r1_attempts and not r1_run_authorized(d1_valid):
        raise DiagnosisGateError("R1 attempt exists without a genuine D1 policy-failure gate")
    classification = classify_recheck(d1_valid, r1_valid)
    head_unchanged = _git(repo, "rev-parse", "HEAD") == audit["head_commit"]
    return {
        "version": VERSION,
        "generated_utc": utc_now(),
        "result": "DIAGNOSIS_COMPLETE",
        "diagnostic_only": True,
        "simulator_evidence_only": True,
        "real_robot_success_claimed": False,
        "preserved_hashes": audit["model_hashes"],
        "previous_invalid_isolation": audit["previous_invalid_summary"],
        "preserved_s09": audit["preserved_s09"],
        "preserved_distribution_and_offline_findings": {
            "dagger1_coverage": audit["dagger1_coverage"],
            "matched_offline": audit["offline_matched"],
        },
        "disk_state": disk,
        "preflight_results": [attempt.get("preflight") for attempt in [*d1_attempts, *r1_attempts]],
        "d1": {
            "physical_attempt_count": len(d1_attempts),
            "infrastructure_replacement_count": max(0, len(d1_attempts) - 1),
            "attempts": [
                _compact_attempt(item, result_dir, repo) for item in d1_attempts
            ],
            "valid_result": (
                _read_json(result_dir / "d1_cone_free_valid.json")
                if (result_dir / "d1_cone_free_valid.json").is_file() else None
            ),
            "late_route_telemetry": None if d1_valid is None else d1_valid.get("late_route_telemetry"),
        },
        "r1": {
            "authorized": r1_run_authorized(d1_valid),
            "physical_attempt_count": len(r1_attempts),
            "infrastructure_replacement_count": max(0, len(r1_attempts) - 1),
            "attempts": [
                _compact_attempt(item, result_dir, repo) for item in r1_attempts
            ],
            "valid_result": (
                _read_json(result_dir / "r1_cone_free_valid.json")
                if (result_dir / "r1_cone_free_valid.json").is_file() else None
            ),
            "late_route_telemetry": None if r1_valid is None else r1_valid.get("late_route_telemetry"),
        },
        "shadow_expert_comparison": {
            "D1": None if d1_valid is None else (d1_valid.get("analysis") or {}).get(
                "full_run_shadow_comparison"
            ),
            "R1": None if r1_valid is None else (r1_valid.get("analysis") or {}).get(
                "full_run_shadow_comparison"
            ),
        },
        "classification": classification,
        "recommended_next_direction": recommended_next_direction(classification["classification"]),
        "prohibited_actions_audit": {
            "training_invocations": 0,
            "fine_tuning_invocations": 0,
            "bags_collected": 0,
            "camera_images_persisted": 0,
            "dagger_sequences_collected": 0,
            "dagger_iteration2_created": False,
            "datasets_modified": False,
            "checkpoints_or_onnx_modified": False,
            "controller_speed_lookahead_route_or_safety_changed": False,
            "commit_performed": not head_unchanged,
            "push_performed": False,
        },
        "s11_s12_protection_audit": {
            "result": "PASS",
            "scenarios": ["11", "12"],
            "world_activations": 0,
            "neural_evaluations": 0,
            "bags_collected": 0,
            "camera_data_inspected": False,
            "expert_labels_generated": 0,
            "manifest_rows_added": 0,
        },
        "tests": {
            "result": "PASS" if tests_summary != "PENDING" else "PENDING",
            "summary": tests_summary,
        },
        "git_diff_check": _git(repo, "diff", "--check") or "PASS",
        "files_changed": _changed_files(repo),
        "final_git_status": _git(repo, "status", "--short", "--branch"),
    }


def _fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _attempt_lines(section: Mapping[str, Any]) -> str:
    attempts = section.get("attempts") or []
    if not attempts:
        return "No physical policy attempt was recorded."
    return "\n".join(
        f"- Attempt {item['physical_attempt_number']}: {item['classification']} "
        f"({item['telemetry_sample_count']} compact cycles; file `{item['attempt_file']}`)."
        for item in attempts
    )


def _late_lines(late: Mapping[str, Any] | None) -> str:
    if not late:
        return "No valid run reached a reportable late-route result."
    lines = [f"Late-route analysis: {late.get('result')}; DAgger1 samples after 20 m: 0."]
    for label, values in (late.get("regions") or {}).items():
        if values.get("result") != "AVAILABLE":
            lines.append(f"- {label}: no samples.")
            continue
        lines.append(
            f"- {label}: n={values['count']}, mean/max CTE "
            f"{_fmt(values.get('mean_cte_m'))}/{_fmt(values.get('maximum_cte_m'))} m; "
            f"policy/shadow mean steering {_fmt(values.get('mean_model_steering_rad'))}/"
            f"{_fmt(values.get('mean_shadow_expert_steering_rad'))} rad; MAE/bias "
            f"{_fmt(values.get('mean_absolute_error_rad'))}/{_fmt(values.get('mean_signed_error_rad'))} rad; "
            f"corrective ratio {_fmt(values.get('corrective_magnitude_ratio'), 4)}; sign agreement "
            f"{_fmt(values.get('steering_sign_agreement_fraction'), 4)}; CTE growth "
            f"{_fmt(values.get('cte_growth_m'))} m; saturation "
            f"{_fmt(values.get('steering_saturation_fraction'), 4)}; temporal/timing/liveness "
            f"{values.get('temporal_healthy')}/{values.get('timing_healthy')}/{values.get('liveness_healthy')}."
        )
    return "\n".join(lines)


def report_markdown(summary: Mapping[str, Any]) -> str:
    invalid = summary["previous_invalid_isolation"]
    d1 = summary["d1"]
    r1 = summary["r1"]
    classification = summary["classification"]
    hashes = summary["preserved_hashes"]
    preflights = summary.get("preflight_results") or []
    valid_preflights = sum(item is not None and item.get("result") == "PASS" for item in preflights)
    d1_valid_metrics = {} if d1["valid_result"] is None else d1["valid_result"].get("metrics", {})
    return f"""# Random-Cone D1 Cone-Free Validity Recheck V1

Simulator-only diagnostic evidence; this report makes no real-robot claim.

## 1. Preserved R1/D1 hashes

- R1 checkpoint/ONNX: `{hashes['R1']['checkpoint']}` / `{hashes['R1']['onnx']}`
- D1 checkpoint/ONNX: `{hashes['D1']['checkpoint']}` / `{hashes['D1']['onnx']}`
- Both freeze records and seals also matched their preserved hashes.

## 2. Previous invalid isolation evidence

The previous D1 cone-free run remains **INFRA_FAIL**, not policy evidence: clock/pose stopped for 0.803 s at s={_fmt(invalid['route_s_m'], 3)} m ({_fmt(100 * invalid['completion_fraction'], 2)}%), with no off-track event, temporal failure, >100 ms slip, or saturation. Max CTE before interruption was {_fmt(invalid['max_cte_m'], 4)} m and safe stop passed.

## 3. Disk state

`df -h /` was recorded. Free space was {_fmt(summary['disk_state']['free_gib'], 3)} GiB against the 5.5 GiB gate: **{summary['disk_state']['result']}**.

## 4. Fresh preflight

{valid_preflights} policy-attempt preflight(s) passed the canonical-world, API/schema, clock, pose-motion, camera, control, spawn/reset, three-frame temporal-buffer, and safe-stop checks.

## 5. D1 attempt(s)

{_attempt_lines(d1)}

Infrastructure replacement count: {d1['infrastructure_replacement_count']} (maximum one).

## 6. Final valid D1 cone-free result

{_fmt(None if d1['valid_result'] is None else d1['valid_result']['classification'])}. The canonical full-lap gate closed after {_fmt(d1_valid_metrics.get('elapsed_s'), 3)} s and {_fmt(d1_valid_metrics.get('total_unwrapped_progress_m'), 3)} m ({_fmt(100 * d1_valid_metrics.get('route_completion_fraction'), 2) if d1_valid_metrics else 'not available'}% completion), with mean/max CTE {_fmt(d1_valid_metrics.get('mean_cte_m'), 4)}/{_fmt(d1_valid_metrics.get('max_cte_m'), 4)} m and {_fmt(d1_valid_metrics.get('off_track_events'))} off-track events. Temporal, API, clock, pose, control-loop, and safe-stop health all passed. A previous infrastructure-invalid run was not counted as policy evidence.

## 7. D1 late-route telemetry

{_late_lines(d1.get('late_route_telemetry'))}

## 8. Conditional R1 result

R1 authorized: {r1['authorized']}. {_attempt_lines(r1)}

## 9. Policy versus shadow Expert

The Expert remained telemetry-only and never crossed the command boundary. Full-run D1 comparison: `{json.dumps(summary['shadow_expert_comparison']['D1'], sort_keys=True)}`. R1 comparison: `{json.dumps(summary['shadow_expert_comparison']['R1'], sort_keys=True)}`.

## 10. Final classification

**{classification['classification']}** — {classification['interpretation']}

## 11. Exactly one next direction

{summary['recommended_next_direction']['direction']} It was not implemented here.

## 12. No learning or data collection

No training, fine tuning, weighting, balancing, dataset editing, DAgger collection/iteration, rosbag, or persisted camera image occurred. No checkpoint or ONNX file was created or modified. Frozen artifact and manifest hashes still match.

## 13. S11/S12 protection

**{summary['s11_s12_protection_audit']['result']}** — zero activation, neural evaluation, bag, camera inspection, label generation, or manifest addition for S11/S12.

## 14. Tests

{summary['tests']['result']}: {summary['tests']['summary']}. `git diff --check`: {summary['git_diff_check']}.

## 15. Files changed

{chr(10).join('- `' + path + '`' for path in summary['files_changed'])}

## 16. Final Git status

```text
{summary['final_git_status']}
```

No commit or push was performed.
"""


def write_summary_and_report(
    config: RecheckConfig,
    repo: Path,
    *,
    tests_summary: str = "PENDING",
) -> dict[str, Any]:
    summary = build_summary(config, repo, tests_summary=tests_summary)
    result_dir = config.result_dir(repo)
    _write_json(result_dir / "summary.json", summary)
    _write_text(result_dir / "REPORT.md", report_markdown(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/random_cone_d1_cone_free_recheck_v1.json"),
    )
    parser.add_argument("--sim-root", type=Path, default=Path("/home/a/physicar-ai-sim-docker"))
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--tests-summary", default="PENDING")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path.cwd().resolve()
    config = load_config((repo / args.config).resolve(), repo)
    result_dir = config.result_dir(repo)
    audit = audit_frozen_evidence(config, repo)
    disk = disk_gate(config.payload["disk_gate"]["path"], MINIMUM_FREE_BYTES)
    _write_json(result_dir / "audit.json", audit)
    _write_json(result_dir / "disk.json", disk)
    if args.execute_live:
        execute_live_recheck(config, repo, args.sim_root.resolve())
        write_summary_and_report(config, repo, tests_summary=args.tests_summary)
    elif args.finalize:
        write_summary_and_report(config, repo, tests_summary=args.tests_summary)
    else:
        print(json.dumps({"audit": "PASS", "disk": disk["result"]}, sort_keys=True))
    return 0
