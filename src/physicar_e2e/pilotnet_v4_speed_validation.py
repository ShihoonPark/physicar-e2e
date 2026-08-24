"""Bounded PilotNet V4 same-route high-speed validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from .expert_driver import Preflight, wait_after_reset
from .pilotnet_dagger_iteration2_inference import load_v4_config
from .pilotnet_inference import CameraOnlyOnnxModel, InferenceConfig, live_camera_preflight, run_smoke, sha256_file
from .pilotnet_v4_repeatability import clock_health_preflight, verify_static_environment
from .rosbag_collector import verify_environment
from .sim_client import SimClient


VERSION = "pilotnet_v4_speed_1p8_v1"
SPEED_MPS = 1.80
TARGET_VALID_RUNS = 3
MAX_LIVE_ATTEMPTS = 5
INITIAL_INFRA_RETRY_LIMIT = 1
EXPECTED_RESULT_DIRECTORY = "results/pilotnet_v4_speed_1p8_v1"
EXPECTED_CANONICAL_CONFIG_SHA256 = "5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45"
EXPECTED_ONNX_SHA256 = "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a"
EXPECTED_ONNX_SIZE = 1_012_518
BASELINE_WORST_MAX_CTE_M = 0.11645700000000048


@dataclass(frozen=True)
class SpeedValidationConfig:
    payload: dict[str, Any]
    canonical: InferenceConfig

    @classmethod
    def load(cls, experiment_path: Path, canonical_path: Path) -> "SpeedValidationConfig":
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected = {
            "version": VERSION,
            "canonical_config": "configs/pilotnet_inference_v4_dagger.json",
            "fixed_speed_mps": SPEED_MPS,
            "target_valid_policy_runs": TARGET_VALID_RUNS,
            "maximum_live_attempts": MAX_LIVE_ATTEMPTS,
            "initial_infrastructure_retry_limit": INITIAL_INFRA_RETRY_LIMIT,
            "automatic_policy_retry": False,
            "result_directory": EXPECTED_RESULT_DIRECTORY,
        }
        if payload != expected:
            raise ValueError("1.80 m/s experiment config does not match the bounded protocol")
        if sha256_file(canonical_path) != EXPECTED_CANONICAL_CONFIG_SHA256:
            raise RuntimeError("canonical V4 inference config identity mismatch")
        canonical = load_v4_config(canonical_path)
        safety = canonical.safety_config(SPEED_MPS)
        if safety.fixed_speed_mps != SPEED_MPS:
            raise RuntimeError("fixed speed is not exactly 1.80 m/s")
        if safety.control_frequency_hz != 15.0:
            raise RuntimeError("control rate differs from canonical 15 Hz")
        if safety.max_steering_rad != 0.349066:
            raise RuntimeError("steering clamp differs from canonical V4")
        return cls(payload, canonical)

    def safety_config(self):
        return self.canonical.safety_config(SPEED_MPS)


def verify_model_identity(path: Path) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size, "parameter_count": 252_219,
    }
    if result["sha256"] != EXPECTED_ONNX_SHA256 or result["size_bytes"] != EXPECTED_ONNX_SIZE:
        raise RuntimeError("canonical V4 ONNX identity mismatch")
    result["result"] = "PASS"
    return result


def classify_run(run: dict[str, Any]) -> str:
    if run.get("result") == "PASS":
        return "POLICY_PASS"
    failure = str(run.get("failure") or "").lower()
    if run.get("api_failures", 0) or run.get("liveness_failures", 0) or not run.get("safe_stop_success", False):
        return "INFRA_FAIL"
    if any(token in failure for token in ("clock", "simulator state changed", "get ", "post ", "unavailable")):
        return "INFRA_FAIL"
    return "POLICY_FAIL"


class ObservingClient:
    """Forward the normal API unchanged while retaining compact failure telemetry."""

    def __init__(self, client: SimClient, initial: Preflight, off_track_margin_m: float) -> None:
        self.client = client
        self.route = initial.route
        self.off_track_margin_m = off_track_margin_m
        self.samples: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def pose(self) -> dict[str, Any]:
        pose = self.client.pose()
        projection = self.route.project((pose["x"], pose["y"]))
        boundary = self.route.track_boundary_distance((pose["x"], pose["y"]))
        self.samples.append({
            "wall_time_s": time.monotonic(), "route_s_m": projection.s,
            "cte_m": projection.distance, "boundary_distance_m": boundary,
            "off_track": boundary is not None and boundary > self.off_track_margin_m,
            "steering_rad": None,
        })
        return pose

    def command_steering(self, value: float) -> dict[str, Any]:
        response = self.client.command_steering(value)
        if self.samples:
            self.samples[-1]["steering_rad"] = float(value)
        return response


def failure_analysis(samples: list[dict[str, Any]], max_steering_rad: float) -> dict[str, Any]:
    if not samples:
        return {"result": "UNAVAILABLE", "reason": "no pose telemetry"}
    divergence = next((row for row in samples if row["cte_m"] > BASELINE_WORST_MAX_CTE_M), None)
    first_off_track = next((row for row in samples if row["off_track"]), None)
    ctes = [float(row["cte_m"]) for row in samples]
    steering = [float(row["steering_rad"]) for row in samples if row["steering_rad"] is not None]
    tail_count = min(15, len(ctes))
    tail = ctes[-tail_count:]
    half = max(1, tail_count // 2)
    early_tail_mean = statistics.fmean(tail[:half])
    late_tail_mean = statistics.fmean(tail[half:]) if tail[half:] else tail[-1]
    increase = late_tail_mean - early_tail_mean
    if increase > 0.20:
        pattern = "abrupt terminal increase"
    elif increase > 0.03:
        pattern = "gradual increase over the final approximately one second"
    else:
        pattern = "no clear terminal increase"
    deltas = [abs(steering[index] - steering[index - 1]) for index in range(1, len(steering))]
    tail_steering = steering[-tail_count:] if steering else []
    return {
        "result": "AVAILABLE",
        "divergence_definition": f"first CTE above canonical 0.50 m/s worst max CTE ({BASELINE_WORST_MAX_CTE_M:.6f} m)",
        "divergence_route_s_m": None if divergence is None else divergence["route_s_m"],
        "divergence_cte_m": None if divergence is None else divergence["cte_m"],
        "first_off_track_route_s_m": None if first_off_track is None else first_off_track["route_s_m"],
        "first_off_track_cte_m": None if first_off_track is None else first_off_track["cte_m"],
        "cte_growth_pattern": pattern,
        "tail_sample_count": tail_count,
        "tail_cte_early_mean_m": early_tail_mean,
        "tail_cte_late_mean_m": late_tail_mean,
        "tail_cte_increase_m": increase,
        "steering_reached_physical_limit": any(math.isclose(abs(value), max_steering_rad, abs_tol=1e-8) for value in steering),
        "tail_steering_saturation_fraction": (sum(math.isclose(abs(value), max_steering_rad, abs_tol=1e-8) for value in tail_steering) / len(tail_steering)) if tail_steering else 0.0,
        "tail_mean_absolute_command_delta_rad": statistics.fmean(deltas[-tail_count:]) if deltas else 0.0,
    }


def run_observed_smoke(client, model, config: SpeedValidationConfig, initial: Preflight) -> dict[str, Any]:
    safety = config.safety_config()
    observer = ObservingClient(client, initial, safety.off_track_margin_m)
    run = run_smoke(observer, model, config.canonical, initial, SPEED_MPS)
    if run.get("result") != "PASS":
        run["high_speed_failure_analysis"] = failure_analysis(observer.samples, safety.max_steering_rad)
    return run


def attempt_preflight(client, config: SpeedValidationConfig) -> tuple[Preflight, dict[str, Any]]:
    safety = config.safety_config()
    initial = wait_after_reset(client, safety, False)
    environment = verify_static_environment(initial)
    camera = live_camera_preflight(client, config.canonical)
    clock = clock_health_preflight(client)
    if clock["result"] != "PASS":
        raise RuntimeError(str(clock.get("failure_reason", "simulator clock health failed")))
    return initial, {"result": "PASS", "environment": environment, "camera": camera, "clock_health": clock,
                     "control_api": "PASS", "speed_mps": SPEED_MPS, "control_frequency_hz": 15.0,
                     "max_steering_rad": 0.349066}


def decorate_attempt(number: int, classification: str, run: dict[str, Any] | None,
                     preflight: dict[str, Any], neural_policy_drove: bool) -> dict[str, Any]:
    failure = str((run or {}).get("failure") or preflight.get("failure") or "").lower()
    return {
        "attempt_number": number, "classification": classification,
        "neural_policy_drove": neural_policy_drove, "preflight": preflight, "run": run,
        "pose_liveness_failures": int((run or {}).get("liveness_failures", 0) if "pose" in failure else 0),
        "clock_liveness_failures": int((run or {}).get("liveness_failures", 0) if "clock" in failure else 0),
    }


def bounded_runs(client, model, config: SpeedValidationConfig, result_dir: Path,
                 run_one: Callable[..., dict[str, Any]] = run_observed_smoke) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    valid_passes = 0
    initial_infra_failures = 0
    for number in range(1, MAX_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight_result = attempt_preflight(client, config)
            run = run_one(client, model, config, initial)
            classification = classify_run(run)
            attempt = decorate_attempt(number, classification, run, preflight_result, True)
        except Exception as exc:
            stop_errors = client.safe_stop()
            preflight_result = {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                                "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors}
            attempt = decorate_attempt(number, "INFRA_FAIL", None, preflight_result, False)
        attempts.append(attempt)
        result_dir.mkdir(parents=True, exist_ok=True)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "POLICY_FAIL":
            return attempts, "FAIL" if valid_passes == 0 else "PARTIAL_PASS"
        if attempt["classification"] == "POLICY_PASS":
            valid_passes += 1
            if valid_passes == TARGET_VALID_RUNS:
                return attempts, "PASS"
        elif valid_passes == 0:
            initial_infra_failures += 1
            if initial_infra_failures > INITIAL_INFRA_RETRY_LIMIT:
                return attempts, "INCONCLUSIVE"
    return attempts, "INCONCLUSIVE"


def aggregate(attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
    runs = [item["run"] for item in attempts if item["classification"] == "POLICY_PASS"]
    if len(runs) != TARGET_VALID_RUNS:
        return None
    times = [float(run["elapsed_s"]) for run in runs]
    mean_ctes = [float(run["mean_cte_m"]) for run in runs]
    saturations = [float(run["steering_saturation_fraction"]) for run in runs]
    return {
        "policy_success": "3/3", "lap_time_mean_s": statistics.fmean(times),
        "lap_time_sample_std_s": statistics.stdev(times),
        "mean_cte_mean_m": statistics.fmean(mean_ctes), "mean_cte_sample_std_m": statistics.stdev(mean_ctes),
        "worst_max_cte_m": max(float(run["max_cte_m"]) for run in runs),
        "saturation_mean": statistics.fmean(saturations), "saturation_range": [min(saturations), max(saturations)],
        "worst_loop_p95_ms": max(float(run["control_loop_period"]["p95_ms"]) for run in runs),
        "worst_loop_max_ms": max(float(run["control_loop_period"]["max_ms"]) for run in runs),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True); parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only == args.run:
        print("ERROR: choose exactly one mode", file=sys.stderr); return 2
    repo_root = Path(__file__).resolve().parents[2]
    if args.result_dir.resolve() != (repo_root / EXPECTED_RESULT_DIRECTORY).resolve():
        print(f"ERROR: result directory must be {EXPECTED_RESULT_DIRECTORY}", file=sys.stderr); return 2
    marker = args.result_dir / "experiment.started.json"
    summary_path = args.result_dir / "summary.json"
    if args.run and (marker.exists() or summary_path.exists()):
        print("ERROR: refusing to repeat the bounded 1.80 m/s experiment", file=sys.stderr); return 2
    report: dict[str, Any] = {"version": VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
                              "result": "INCONCLUSIVE", "attempts": []}
    client: SimClient | None = None; code = 2
    try:
        config = SpeedValidationConfig.load(args.config, args.canonical_config)
        report["canonical_config"] = {"path": str(args.canonical_config.resolve()),
                                      "sha256": sha256_file(args.canonical_config), "unchanged": True}
        report["model_identity"] = verify_model_identity(args.onnx)
        report["protocol"] = {"speed_mps": SPEED_MPS, "control_frequency_hz": 15.0,
                              "distance_per_control_interval_m": SPEED_MPS / 15.0,
                              "baseline_interval_distance_ratio": SPEED_MPS / 0.5,
                              "target_valid_runs": TARGET_VALID_RUNS, "maximum_live_attempts": MAX_LIVE_ATTEMPTS,
                              "automatic_policy_retry": False}
        report["source_environment"] = verify_environment(repo_root, args.sim_root)
        model = CameraOnlyOnnxModel(args.onnx)
        client = SimClient(config.canonical.payload["base_url"], config.canonical.payload["api_timeout_s"])
        if errors := client.safe_stop():
            raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
        initial, preflight_result = attempt_preflight(client, config)
        report["preflight"] = preflight_result
        if args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            write_json(marker, {"status": "BOUNDED_LIVE_STARTED", "started_utc": datetime.now(timezone.utc).isoformat(),
                                "maximum_live_attempts": MAX_LIVE_ATTEMPTS, "policy_failure_retry": False})
            report["attempts"], report["result"] = bounded_runs(client, model, config, args.result_dir)
            report["aggregate"] = aggregate(report["attempts"])
            report["attempt_count"] = len(report["attempts"])
            report["valid_policy_evaluations"] = sum(item["classification"] in ("POLICY_PASS", "POLICY_FAIL") for item in report["attempts"])
            report["infrastructure_attempts"] = sum(item["classification"] == "INFRA_FAIL" for item in report["attempts"])
            code = 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        if client is not None:
            errors = client.safe_stop(); report["final_safe_stop_success"] = not errors; report["final_safe_stop_errors"] = errors
            if errors: report["result"] = "INCONCLUSIVE"; code = 2
        write_json(args.result_dir / ("preflight.json" if args.preflight_only else "summary.json"), report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
