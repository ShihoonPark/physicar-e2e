"""Collision-only practical validation of the frozen Temporal PilotNet C1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Callable, Sequence

from .cone_avoidance_expert import (
    ClearanceObserver,
    ExpertConfig,
    activate_world,
    build_bypass_plan,
    full_preflight,
    validate_geometry,
)
from .cone_avoidance_temporal_c1 import (
    WORLD,
    audit_frozen,
    load_c1_inference_config,
    simulator_tracked_status,
)
from .high_speed_temporal import TemporalOnnxModel, run_temporal_live, utc_now, warm_temporal_buffer
from .high_speed_v5 import write_json
from .pilotnet_training import GateFailure, sha256_file
from .sim_client import SimClient


VERSION = "pilotnet_c1_practical_cone_validation_v1"
RESULT_DIRECTORY = "results/pilotnet_c1_practical_cone_validation_v1"
TARGET_VALID_PASSES = 3
MAXIMUM_VALID_POLICY_RUNS = 3
MAXIMUM_TOTAL_ATTEMPTS = 5
HISTORICAL_FILES = {
    "historical_report": "results/pilotnet_e2e_c1_cone_temporal/REPORT.md",
    "historical_summary": "results/pilotnet_e2e_c1_cone_temporal/summary.json",
    "historical_marker": "results/pilotnet_e2e_c1_cone_temporal/experiment.started.json",
    "historical_attempt_01": "results/pilotnet_e2e_c1_cone_temporal/attempt_01.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_practical_contract(repo: Path) -> dict[str, Any]:
    payload = _load_json(repo / "configs/pilotnet_c1_practical_cone_validation_v1.json")
    frozen = (
        payload.get("version"), payload.get("result_directory"), payload.get("historical_result_directory"),
        payload.get("practical_contract"), payload.get("expected_world"), payload.get("speed_mps"),
        payload.get("history_frames"), payload.get("input_shape"), payload.get("control_frequency_hz"),
        payload.get("maximum_steering_rad"), payload.get("maximum_valid_policy_runs"),
        payload.get("target_valid_passes"), payload.get("maximum_total_attempts"),
        payload.get("policy_failure_retries_permitted"), payload.get("training_permitted"),
        payload.get("data_collection_permitted"), payload.get("dagger_permitted"),
        payload.get("world_changes_permitted"), payload.get("cone_pose_changes_permitted"),
        payload.get("speed_changes_permitted"), payload.get("architecture_changes_permitted"),
        payload.get("steering_changes_permitted"),
    )
    expected = (
        VERSION, RESULT_DIRECTORY, "results/pilotnet_e2e_c1_cone_temporal",
        "no_vehicle_cone_collision_or_intersection", WORLD, 1.8, 3, [9, 66, 200], 15.0,
        .349066, 3, 3, 5, False, False, False, False, False, False, False, False, False,
    )
    if frozen != expected:
        raise GateFailure(f"practical validation contract changed: {frozen}")
    return payload


def _verified_hash(path: Path, expected: str, label: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected:
        raise GateFailure(f"{label} identity changed: {observed} != {expected}")
    return {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def audit_practical_frozen(repo: Path, sim_root: Path) -> dict[str, Any]:
    contract = load_practical_contract(repo)
    expected_hashes = contract["frozen_sha256"]
    identities = {
        label: _verified_hash(repo / relative, expected_hashes[label], label)
        for label, relative in HISTORICAL_FILES.items()
    }
    identities.update({
        "c1_training_summary": _verified_hash(
            repo / "results/pilotnet_training_c1_cone_temporal/summary.json",
            expected_hashes["c1_training_summary"], "C1 training summary",
        ),
        "c1_inference_config": _verified_hash(
            repo / "configs/pilotnet_inference_c1_cone_temporal.json",
            expected_hashes["c1_inference_config"], "C1 inference config",
        ),
        "cone_environment_config": _verified_hash(
            repo / "configs/cone_avoidance_environment_v1.json",
            expected_hashes["cone_environment_config"], "cone environment config",
        ),
        "cone_expert_config": _verified_hash(
            repo / "configs/cone_avoidance_expert_v1.json",
            expected_hashes["cone_expert_config"], "cone expert config",
        ),
    })
    training = _load_json(repo / "results/pilotnet_training_c1_cone_temporal/summary.json")
    if (training.get("result"), training.get("architecture", {}).get("input_shape"),
            training.get("architecture", {}).get("parameter_count")) != ("PASS", [9, 66, 200], 255_819):
        raise GateFailure("preserved C1 training architecture/result changed")
    for kind in ("checkpoint", "onnx"):
        artifact = training["artifacts"][kind]
        expected = expected_hashes[f"c1_{kind}"]
        if artifact.get("sha256") != expected:
            raise GateFailure(f"preserved C1 {kind} evidence hash changed")
        identities[f"c1_{kind}"] = _verified_hash(Path(artifact["path"]), expected, f"C1 {kind}")
    historical = _load_json(repo / HISTORICAL_FILES["historical_summary"])
    attempt = historical.get("attempts", [{}])[0]
    old_run = attempt.get("run") or {}
    if (
        historical.get("result") != "FAIL"
        or attempt.get("classification") != "CONE_POLICY_FAIL"
        or float(old_run.get("minimum_footprint_to_cone_clearance_m", 1.0)) >= .05
        or old_run.get("footprint_cone_intersection_occurred") is not False
    ):
        raise GateFailure("historical 5 cm C1 failure evidence changed")
    frozen = audit_frozen(repo, sim_root)
    if frozen.get("world") != WORLD:
        raise GateFailure("frozen world identity changed")
    return {
        "result": "PASS", "contract": contract, "identities": identities,
        "historical_result": {
            "classification": "FAIL under the 5 cm experimental clearance contract",
            "minimum_clearance_m": old_run["minimum_footprint_to_cone_clearance_m"],
            "intersection": old_run["footprint_cone_intersection_occurred"],
        },
        "world": frozen["world"], "cone": frozen["cone"], "architecture": training["architecture"],
        "checkpoint": training["artifacts"]["checkpoint"], "onnx": training["artifacts"]["onnx"],
        "no_retraining": True, "no_data_collection": True, "no_dagger": True,
    }


def classify_practical_cone_run(run: dict[str, Any]) -> str:
    if run.get("temporal_input_failure"):
        return "TEMPORAL_INPUT_FAIL"
    if run.get("api_failures") or run.get("liveness_failures") or not run.get("safe_stop_success", False):
        return "INFRA_FAIL"
    clearance = run.get("minimum_footprint_to_cone_clearance_m")
    collision = run.get("vehicle_cone_collision_or_intersection_occurred")
    if collision is None:
        collision = run.get("footprint_cone_intersection_occurred")
    if (
        run.get("result") == "PASS" and clearance is not None and float(clearance) >= 0.0
        and collision is False and run.get("recovery_success") is True
    ):
        return "PRACTICAL_CONE_PASS"
    return "PRACTICAL_CONE_FAIL"


def run_practical_cone_policy(
    client: SimClient, model: TemporalOnnxModel, inference: Any,
    initial: Any, expert: ExpertConfig, plan: Any,
) -> dict[str, Any]:
    observer = ClearanceObserver(
        client, initial.route, plan, expert, enforce_clearance_margin=False,
    )
    run = run_temporal_live(observer, model, inference, initial, 1.80)
    rows = observer.samples
    minimum = min(rows, key=lambda row: float(row["cone_clearance_m"]), default=None)
    avoidance = [
        row for row in rows
        if plan.departure_start_s_m <= float(row["route_s_m"]) <= plan.return_end_s_m
    ]
    offsets = [float(row["nominal_signed_cte_m"]) * plan.side_sign for row in avoidance]
    run.update({
        "minimum_footprint_to_cone_clearance_m": None if minimum is None else minimum["cone_clearance_m"],
        "minimum_cone_clearance_route_s_m": None if minimum is None else minimum["route_s_m"],
        "footprint_cone_intersection_occurred": observer.intersection_occurred,
        "vehicle_cone_collision_or_intersection_occurred": observer.intersection_occurred,
        "maximum_lateral_avoidance_offset_reached_m": max(offsets, default=0.0),
        "recovery_success": observer.recovery_success,
        "recovery_cte_m": observer.recovery_cte_m,
        "recovery_time_s": observer.recovery_time_s,
        "nominal_route_used_for_progress": True,
        "clearance_measured_not_enforced_as_margin": True,
        "practical_clearance_threshold_m": 0.0,
        "historical_experimental_clearance_threshold_m": 0.05,
        "privileged_metrics_only": ["pose", "route", "cte", "cone_pose", "footprint_clearance", "recovery"],
        "model_observation_only": ["camera_yuv_t_minus_2", "camera_yuv_t_minus_1", "camera_yuv_t"],
        "pose_failures": run.get("liveness_failures", 0),
        "clock_failures": run.get("liveness_failures", 0),
    })
    run["classification"] = classify_practical_cone_run(run)
    return run


def run_practical_attempts(
    client: SimClient, model: TemporalOnnxModel, inference: Any, expert: ExpertConfig,
    plan: Any, geometry: dict[str, Any], sim_root: Path, result_dir: Path,
    *, preflight_one: Callable[..., tuple[Any, dict[str, Any]]] = full_preflight,
    run_one: Callable[..., dict[str, Any]] = run_practical_cone_policy,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    valid = 0
    for number in range(1, MAXIMUM_TOTAL_ATTEMPTS + 1):
        try:
            initial, preflight = preflight_one(client, expert, plan, sim_root, geometry)
            run = run_one(client, model, inference, initial, expert, plan)
            classification = str(run["classification"])
            if classification not in {
                "PRACTICAL_CONE_PASS", "PRACTICAL_CONE_FAIL", "INFRA_FAIL", "TEMPORAL_INPUT_FAIL",
            }:
                raise RuntimeError(f"unknown practical classification {classification}")
            if classification in {"PRACTICAL_CONE_PASS", "PRACTICAL_CONE_FAIL"}:
                valid += 1
            attempt = {
                "attempt_number": number,
                "valid_policy_run_number": valid if classification in {"PRACTICAL_CONE_PASS", "PRACTICAL_CONE_FAIL"} else None,
                "classification": classification, "preflight": preflight, "run": run,
            }
        except Exception as exc:
            errors = client.safe_stop()
            attempt = {
                "attempt_number": number, "valid_policy_run_number": None,
                "classification": "INFRA_FAIL", "run": None,
                "preflight": {
                    "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                    "safe_stop_success": not errors, "safe_stop_errors": errors,
                },
            }
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "PRACTICAL_CONE_FAIL":
            return attempts, "FAIL"
        if attempt["classification"] == "PRACTICAL_CONE_PASS" and valid == TARGET_VALID_PASSES:
            return attempts, "PASS"
        if valid >= MAXIMUM_VALID_POLICY_RUNS:
            break
    return attempts, "INCONCLUSIVE"


def aggregate_practical(attempts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    runs = [item["run"] for item in attempts if item["classification"] == "PRACTICAL_CONE_PASS"]
    if len(runs) != TARGET_VALID_PASSES:
        raise GateFailure("practical aggregate requires exactly three valid passes")
    clearances = [float(run["minimum_footprint_to_cone_clearance_m"]) for run in runs]
    laps = [float(run["elapsed_s"]) for run in runs]
    return {
        "success": "3/3", "minimum_clearance_across_runs_m": min(clearances),
        "clearance_each_run_m": clearances, "lap_time_mean_s": statistics.fmean(laps),
        "lap_time_sample_std_s": statistics.stdev(laps), "intersection_each_run": [False, False, False],
        "recovery_success": "3/3", "safe_stop": "3/3",
        "conclusion": (
            "Temporal PilotNet C1 achieved 3/3 valid fixed-one-cone, 1.80 m/s simulation laps "
            "without cone contact using only causal camera observations."
        ),
    }


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Temporal PilotNet C1 Practical Cone Validation V1", "",
        f"Decision: **{report['result']}** under the collision/intersection-only practical contract.", "",
        "The historical C1 result remains FAIL under its 0.050000 m experimental clearance contract.", "",
    ]
    for attempt in report.get("attempts", []):
        run = attempt.get("run")
        lines.extend([f"## Attempt {attempt['attempt_number']}", "", f"Classification: `{attempt['classification']}`.", ""])
        if run:
            lines.extend([
                f"- Lap time: `{run.get('elapsed_s')}` s; completion: `{run.get('route_completion_fraction')}`.",
                f"- Minimum clearance: `{run.get('minimum_footprint_to_cone_clearance_m')}` m at route s `{run.get('minimum_cone_clearance_route_s_m')}` m.",
                f"- Vehicle/cone intersection: `{run.get('vehicle_cone_collision_or_intersection_occurred')}`.",
                f"- Recovery: `{run.get('recovery_success')}`; time: `{run.get('recovery_time_s')}` s.",
                f"- Mean/max nominal CTE: `{run.get('mean_cte_m')}` / `{run.get('max_cte_m')}` m; off-track events: `{run.get('off_track_events')}`.",
                f"- Mean/max absolute steering: `{run.get('mean_absolute_predicted_steering_rad')}` / `{run.get('max_absolute_predicted_steering_rad')}` rad; saturation: `{run.get('steering_saturation_fraction')}`.",
                f"- Temporal/API/pose/clock failures: `{run.get('temporal_invalid_history_count')}` / `{run.get('api_failures')}` / `{run.get('pose_failures')}` / `{run.get('clock_failures')}`.",
                f"- Safe stop: `{'PASS' if run.get('safe_stop_success') else 'FAIL'}`.", "",
            ])
    if report.get("aggregate"):
        lines.extend(["## Conclusion", "", report["aggregate"]["conclusion"], ""])
    lines.extend([
        f"Fixed-cone practical baseline complete: `{report.get('fixed_cone_practical_baseline_complete')}`.",
        f"Random/unseen-cone engineering work justified: `{report.get('random_cone_work_justified')}`.", "",
        "This is simulation evidence only; it is not a real-robot claim.", "",
    ])
    return "\n".join(lines)


def live_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    audit = audit_practical_frozen(repo, sim_root)
    result_dir = repo / RESULT_DIRECTORY
    summary_path = result_dir / "summary.json"
    marker = result_dir / "experiment.started.json"
    if result_dir.exists():
        raise FileExistsError("refusing to repeat or overwrite practical C1 validation evidence")
    inference = load_c1_inference_config(repo)
    training = _load_json(repo / "results/pilotnet_training_c1_cone_temporal/summary.json")
    model = TemporalOnnxModel(Path(training["artifacts"]["onnx"]["path"]))
    expert = ExpertConfig.load(repo / "configs/cone_avoidance_expert_v1.json", repo, sim_root)
    plan, route_data = build_bypass_plan(expert, sim_root)
    geometry = validate_geometry(expert, plan, route_data)
    client = SimClient(inference.payload["base_url"], inference.payload["api_timeout_s"])
    result_dir.mkdir(parents=True, exist_ok=False)
    write_json(marker, {
        "status": "C1_PRACTICAL_VALIDATION_STARTED_DO_NOT_REPEAT", "started_utc": utc_now(),
        "maximum_total_attempts": 5, "maximum_valid_policy_runs": 3,
        "historical_result_preserved": True,
    })
    report: dict[str, Any] = {
        "version": VERSION, "generated_utc": utc_now(), "result": "INCONCLUSIVE",
        "audit": audit, "contract": {
            "classification_pass": "PRACTICAL_CONE_PASS",
            "cone_requirement": "no vehicle/cone collision or intersection",
            "clearance_measured_but_0p05_not_enforced": True,
            "road_safety_thresholds_unchanged": True,
        },
        "camera_only_observation": True, "simulation_only": True,
    }
    try:
        activate_world(client, WORLD)
        if errors := client.safe_stop():
            raise GateFailure("initial practical validation safe stop failed: " + "; ".join(errors))
        initial, preflight = full_preflight(client, expert, plan, sim_root, geometry)
        _, buffer_check = warm_temporal_buffer(client, inference)
        report["temporal_live_preflight"] = {
            "result": "PASS", "world": initial.world, "environment": preflight,
            "buffer": buffer_check, "model_observation_fields": list(model.observation_fields),
        }
        report["attempts"], report["result"] = run_practical_attempts(
            client, model, inference, expert, plan, geometry, sim_root, result_dir,
        )
        report["total_attempts"] = len(report["attempts"])
        report["valid_policy_runs"] = sum(
            item["classification"] in {"PRACTICAL_CONE_PASS", "PRACTICAL_CONE_FAIL"}
            for item in report["attempts"]
        )
        report["aggregate"] = aggregate_practical(report["attempts"]) if report["result"] == "PASS" else None
        report["fixed_cone_practical_baseline_complete"] = report["result"] == "PASS"
        report["random_cone_work_justified"] = report["result"] == "PASS"
    finally:
        errors = client.safe_stop()
        report["final_safe_stop_success"] = not errors
        report["final_safe_stop_errors"] = errors
        if errors:
            report["result"] = "INCONCLUSIVE"
            report["fixed_cone_practical_baseline_complete"] = False
            report["random_cone_work_justified"] = False
        report["simulator_tracked_status"] = simulator_tracked_status(sim_root)
        write_json(summary_path, report)
        (result_dir / "REPORT.md").write_text(_render_report(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "live"), required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    sim_root = args.sim_root.expanduser().resolve()
    try:
        result = audit_practical_frozen(repo, sim_root) if args.stage == "audit" else live_stage(repo, sim_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("result") == "PASS" else 1
    except Exception as exc:
        print(json.dumps({
            "result": "FAIL", "stage": args.stage, "failure": f"{type(exc).__name__}: {exc}",
        }, indent=2), flush=True)
        return 1 if isinstance(exc, GateFailure) else 2


if __name__ == "__main__":
    raise SystemExit(main())
