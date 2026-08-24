"""Bounded validation-only repeatability runner for the preserved PilotNet V4."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from .expert_driver import wait_after_reset
from .pilotnet_dagger_iteration2_inference import load_v4_config
from .pilotnet_inference import CameraOnlyOnnxModel, live_camera_preflight, run_smoke, sha256_file
from .rosbag_collector import verify_environment
from .sim_client import SimClient


EXPECTED_CHECKPOINT_SHA256 = "a581c1a6cb13643a0af0ee2d568244291e1eb858516f685ef492a3016501d1d9"
EXPECTED_ONNX_SHA256 = "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
EXPECTED_ONNX_SIZE = 1_012_518
EXPECTED_ROUTE_POINTS = 388
EXPECTED_ROUTE_LENGTH_M = 30.50461070080936
EXPECTED_BOUNDS_M = (12.0, 7.0)
MAX_ATTEMPTS = 4
TARGET_NEW_VALID_RUNS = 2
CLOCK_SAMPLE_DURATION_S = 2.0
CLOCK_SAMPLE_INTERVAL_S = 0.05
CLOCK_MAX_STALL_S = 0.50


def verify_v4_identity(checkpoint: Path, onnx: Path) -> dict[str, Any]:
    result = {
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "onnx_path": str(onnx), "onnx_sha256": sha256_file(onnx), "onnx_size_bytes": onnx.stat().st_size,
    }
    if result["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("V4 checkpoint identity mismatch")
    if result["onnx_sha256"] != EXPECTED_ONNX_SHA256 or result["onnx_size_bytes"] != EXPECTED_ONNX_SIZE:
        raise RuntimeError("V4 ONNX identity mismatch")
    result["result"] = "PASS"
    return result


def verify_static_environment(initial) -> dict[str, Any]:
    bounds = initial.bounds
    width = float(bounds["maxX"]) - float(bounds["minX"])
    height = float(bounds["maxY"]) - float(bounds["minY"])
    if initial.route_points != EXPECTED_ROUTE_POINTS:
        raise RuntimeError(f"expected {EXPECTED_ROUTE_POINTS} route points, got {initial.route_points}")
    if not math.isclose(initial.route.length, EXPECTED_ROUTE_LENGTH_M, abs_tol=1e-3):
        raise RuntimeError(f"unexpected route length {initial.route.length}")
    if not (math.isclose(width, EXPECTED_BOUNDS_M[0], abs_tol=1e-6) and math.isclose(height, EXPECTED_BOUNDS_M[1], abs_tol=1e-6)):
        raise RuntimeError(f"expected 12x7 m bounds, got {width}x{height}")
    return {
        "result": "PASS", "world": initial.world, "switching": False,
        "route_points": initial.route_points, "route_length_m": initial.route.length,
        "bounds": bounds, "bounds_width_m": width, "bounds_height_m": height,
        "cone_count": initial.cone_count, "pose": initial.pose,
    }


def clock_health_preflight(
    client, *, duration_s: float = CLOCK_SAMPLE_DURATION_S,
    interval_s: float = CLOCK_SAMPLE_INTERVAL_S, max_stall_s: float = CLOCK_MAX_STALL_S,
    monotonic: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = monotonic(); wall_times: list[float] = []; sim_times: list[float] = []; payloads: list[dict[str, Any]] = []
    while True:
        now = monotonic()
        payload = client.clock(); wall_times.append(now); sim_times.append(float(payload["sim_time"])); payloads.append(payload)
        if now - started >= duration_s:
            break
        sleep(interval_s)
    wall_deltas = [wall_times[index] - wall_times[index - 1] for index in range(1, len(wall_times))]
    sim_deltas = [sim_times[index] - sim_times[index - 1] for index in range(1, len(sim_times))]
    stall_started = wall_times[0]; max_stall = 0.0
    for index in range(1, len(sim_times)):
        if sim_times[index] > sim_times[index - 1]:
            max_stall = max(max_stall, wall_times[index] - stall_started)
            stall_started = wall_times[index]
        else:
            max_stall = max(max_stall, wall_times[index] - stall_started)
    total_wall = wall_times[-1] - wall_times[0]; total_sim = sim_times[-1] - sim_times[0]
    paused_values = [payload.get("paused") for payload in payloads if "paused" in payload]
    healthy = total_sim > 0 and all(delta >= 0 for delta in sim_deltas) and max_stall < max_stall_s and not any(value is True for value in paused_values)
    result = {
        "result": "PASS" if healthy else "FAIL", "sample_count": len(sim_times),
        "requested_duration_s": duration_s, "wall_elapsed_s": total_wall, "sim_elapsed_s": total_sim,
        "wall_interval_s": {"mean": statistics.fmean(wall_deltas) if wall_deltas else 0.0,
                            "min": min(wall_deltas, default=0.0), "max": max(wall_deltas, default=0.0)},
        "sim_delta_s": {"mean": statistics.fmean(sim_deltas) if sim_deltas else 0.0,
                        "min": min(sim_deltas, default=0.0), "max": max(sim_deltas, default=0.0)},
        "observed_real_time_factor": total_sim / total_wall if total_wall > 0 else 0.0,
        "maximum_observed_clock_stall_s": max_stall, "maximum_allowed_stall_s": max_stall_s,
        "paused_exposed": bool(paused_values), "paused_observed": any(value is True for value in paused_values),
    }
    if not healthy:
        result["failure_reason"] = "simulator clock health gate did not show continuous forward progress"
    return result


def classify_policy_run(run: dict[str, Any]) -> str:
    if run.get("result") == "PASS":
        return "POLICY_PASS"
    failure = str(run.get("failure") or "").lower()
    if run.get("api_failures", 0) or run.get("liveness_failures", 0) or not run.get("safe_stop_success", False):
        return "INFRA_FAIL"
    if any(token in failure for token in ("clock", "simulator state changed", "get ", "post ", "unavailable")):
        return "INFRA_FAIL"
    return "POLICY_FAIL"


def decorate_attempt(number: int, run: dict[str, Any], clock: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    classification = classify_policy_run(run)
    failure = str(run.get("failure") or "").lower()
    return {
        "attempt_number": number, "classification": classification, "neural_policy_drove": True,
        "clock_health_preflight": clock, "environment_preflight": environment, "run": run,
        "pose_liveness_failures": int(run.get("liveness_failures", 0) if "pose" in failure else 0),
        "clock_liveness_failures": int(run.get("liveness_failures", 0) if "clock" in failure else 0),
        "policy_metrics_diagnostic_only": classification == "INFRA_FAIL",
    }


def infra_not_ready_attempt(number: int, clock: dict[str, Any], environment: dict[str, Any], stop_errors: list[str]) -> dict[str, Any]:
    return {
        "attempt_number": number, "classification": "INFRA_FAIL", "setup_status": "INFRA_NOT_READY",
        "neural_policy_drove": False, "clock_health_preflight": clock, "environment_preflight": environment,
        "exact_failure_reason": clock.get("failure_reason"), "safe_stop_success": not stop_errors,
        "safe_stop_errors": stop_errors, "policy_metrics_diagnostic_only": True,
    }


def aggregate_three_valid(historical: dict[str, Any], new_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    new_runs = [item["run"] for item in new_attempts if item["classification"] == "POLICY_PASS"]
    if len(new_runs) != TARGET_NEW_VALID_RUNS or historical.get("result") != "PASS":
        raise ValueError("aggregate requires one historical and exactly two new policy passes")
    runs = [historical, *new_runs]
    lap_times = [float(run["elapsed_s"]) for run in runs]; mean_ctes = [float(run["mean_cte_m"]) for run in runs]
    saturations = [float(run["steering_saturation_fraction"]) for run in runs]
    return {
        "policy_success": "3/3", "historical_full_lap_included_count": 1,
        "lap_time_mean_s": statistics.fmean(lap_times), "lap_time_sample_std_s": statistics.stdev(lap_times),
        "lap_time_range_s": [min(lap_times), max(lap_times)], "mean_cte_per_run_m": mean_ctes,
        "mean_of_mean_cte_m": statistics.fmean(mean_ctes), "worst_max_cte_m": max(float(run["max_cte_m"]) for run in runs),
        "steering_saturation_mean": statistics.fmean(saturations), "steering_saturation_range": [min(saturations), max(saturations)],
        "worst_loop_p95_ms": max(float(run["control_loop_period"]["p95_ms"]) for run in runs),
        "worst_loop_max_ms": max(float(run["control_loop_period"]["max_ms"]) for run in runs),
        "api_failures": sum(int(run.get("api_failures", 0)) for run in runs),
        "liveness_failures": sum(int(run.get("liveness_failures", 0)) for run in runs),
        "safe_stop_success_count": sum(bool(run.get("safe_stop_success")) for run in runs),
    }


def run_bounded(client, model, config, result_dir: Path, historical: dict[str, Any]) -> tuple[list[dict[str, Any]], str, dict[str, Any] | None]:
    attempts: list[dict[str, Any]] = []; valid_passes = 0
    for number in range(1, MAX_ATTEMPTS + 1):
        initial = wait_after_reset(client, config.safety_config(0.5), False)
        environment = verify_static_environment(initial)
        clock = clock_health_preflight(client)
        if clock["result"] != "PASS":
            attempt = infra_not_ready_attempt(number, clock, environment, client.safe_stop())
        else:
            run = run_smoke(client, model, config, initial, 0.5)
            attempt = decorate_attempt(number, run, clock, environment)
        attempts.append(attempt)
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"attempt_{number:02d}.json").write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if attempt["classification"] == "POLICY_FAIL":
            return attempts, "FAIL", None
        if attempt["classification"] == "POLICY_PASS":
            valid_passes += 1
            if valid_passes == TARGET_NEW_VALID_RUNS:
                return attempts, "PASS", aggregate_three_valid(historical, attempts)
    return attempts, "INCONCLUSIVE", None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True); parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True); parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only == args.run:
        print("ERROR: choose exactly one mode", file=sys.stderr); return 2
    marker = args.result_dir / "experiment.started.json"
    if args.run and (marker.exists() or (args.result_dir / "summary.json").exists()):
        print("ERROR: refusing to repeat bounded repeatability experiment", file=sys.stderr); return 2
    report: dict[str, Any] = {"version": "pilotnet_v4_repeatability_v1", "generated_utc": datetime.now(timezone.utc).isoformat(), "result": "FAIL", "attempts": []}
    client = None; code = 2
    try:
        config = load_v4_config(args.config); identity = verify_v4_identity(args.checkpoint, args.onnx)
        historical_payload = json.loads(args.historical.read_text(encoding="utf-8")); historical = historical_payload["runs"][0]
        if historical.get("result") != "PASS": raise RuntimeError("preserved historical V4 run #1 is not a PASS")
        report["identity"] = identity
        report["immutable_config"] = {"path": str(args.config), "sha256": sha256_file(args.config), "speed_mps": 0.5}
        report["external_environment_verifier"] = verify_environment(Path(__file__).resolve().parents[2], args.sim_root)
        client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
        if errors := client.safe_stop(): raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
        static = wait_after_reset(client, config.safety_config(0.5), False)
        report["environment"] = verify_static_environment(static); report["camera"] = live_camera_preflight(client, config)
        report["clock_health"] = clock_health_preflight(client)
        if report["clock_health"]["result"] != "PASS":
            report["result"] = "INFRA_NOT_READY"; code = 1
        elif args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            args.result_dir.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), "maximum_attempts": MAX_ATTEMPTS}, indent=2) + "\n", encoding="utf-8")
            report["attempts"], report["result"], report["aggregate"] = run_bounded(client, model=CameraOnlyOnnxModel(args.onnx), config=config, result_dir=args.result_dir, historical=historical)
            report["valid_new_policy_passes"] = sum(item["classification"] == "POLICY_PASS" for item in report["attempts"])
            report["infra_failures"] = sum(item["classification"] == "INFRA_FAIL" for item in report["attempts"])
            report["policy_failures"] = sum(item["classification"] == "POLICY_FAIL" for item in report["attempts"])
            code = 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if client is not None:
            errors = client.safe_stop(); report["final_safe_stop_success"] = not errors; report["final_safe_stop_errors"] = errors
            if errors: report["result"] = "FAIL"; code = 2
        args.result_dir.mkdir(parents=True, exist_ok=True)
        name = "preflight.json" if args.preflight_only else "summary.json"
        (args.result_dir / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__": raise SystemExit(main())
