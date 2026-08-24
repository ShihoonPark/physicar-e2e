"""Bounded repeatability validation for the frozen 1.80 m/s Expert."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

from .expert_driver import DriverConfig, Preflight
from .expert_speed_lookahead_validation import CandidateRuntime, run_candidate
from .expert_speed_validation import EXPECTED_CANONICAL_CONFIG_SHA256, classify, full_preflight, write_json
from .pilotnet_inference import sha256_file
from .sim_client import SimClient


VERSION = "expert_speed_1p8_repeatability_v1"
SPEED_MPS = 1.80
LOOKAHEAD_M = 0.90
TARGET_NEW_VALID_RUNS = 2
MAXIMUM_NEW_LIVE_ATTEMPTS = 4
EXPECTED_RESULT_DIRECTORY = "results/expert_speed_1p8_repeatability_v1"
EXPECTED_HISTORICAL_PATH = "results/expert_speed_1p8_lookahead_v1/attempt_03.json"
EXPECTED_HISTORICAL_SHA256 = "7e1b5e60fbceac6167c70c6643d4c6563a71ef92e13960c907527ddb67a1b345"


@dataclass(frozen=True)
class RepeatabilityConfig:
    payload: dict[str, Any]
    canonical: DriverConfig
    runtime: CandidateRuntime

    @classmethod
    def load(cls, experiment_path: Path, canonical_path: Path) -> "RepeatabilityConfig":
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected = {
            "version": VERSION, "canonical_config": "configs/expert_driver_v1.json",
            "historical_pass": EXPECTED_HISTORICAL_PATH, "fixed_speed_mps": SPEED_MPS,
            "fixed_lookahead_m": LOOKAHEAD_M, "target_new_valid_runs": TARGET_NEW_VALID_RUNS,
            "maximum_new_live_attempts": MAXIMUM_NEW_LIVE_ATTEMPTS, "retry_expert_failure": False,
            "result_directory": EXPECTED_RESULT_DIRECTORY,
        }
        if payload != expected:
            raise ValueError("repeatability config does not match the frozen protocol")
        if sha256_file(canonical_path) != EXPECTED_CANONICAL_CONFIG_SHA256:
            raise RuntimeError("canonical Expert config identity mismatch")
        canonical = DriverConfig.load(canonical_path)
        driver = replace(canonical, fixed_speed_mps=SPEED_MPS, lookahead_m=LOOKAHEAD_M)
        driver.validate()
        changed = [name for name in canonical.__dataclass_fields__
                   if getattr(canonical, name) != getattr(driver, name)]
        if changed != ["fixed_speed_mps", "lookahead_m"]:
            raise RuntimeError(f"unexpected runtime differences: {changed}")
        if driver.control_frequency_hz != 15.0 or driver.max_steering_rad != 0.349066 or driver.wheelbase_m != 0.18:
            raise RuntimeError("frozen high-speed Expert contract changed")
        return cls(payload, canonical, CandidateRuntime(driver))


def load_historical(path: Path) -> dict[str, Any]:
    if str(path) != EXPECTED_HISTORICAL_PATH and path.resolve() != (Path(__file__).resolve().parents[2] / EXPECTED_HISTORICAL_PATH).resolve():
        raise RuntimeError("historical pass path differs from the frozen provenance")
    if sha256_file(path) != EXPECTED_HISTORICAL_SHA256:
        raise RuntimeError("historical 0.90 m pass identity mismatch")
    attempt = json.loads(path.read_text(encoding="utf-8"))
    metrics = attempt.get("metrics", {})
    preflight = attempt.get("preflight", {})
    if attempt.get("classification") != "EXPERT_PASS" or metrics.get("result") != "PASS":
        raise RuntimeError("historical evidence is not an Expert pass")
    if attempt.get("lookahead_m") != LOOKAHEAD_M or preflight.get("speed_mps") != SPEED_MPS:
        raise RuntimeError("historical pass does not match the frozen speed/lookahead")
    if preflight.get("control_frequency_hz") != 15.0 or preflight.get("max_steering_rad") != 0.349066:
        raise RuntimeError("historical pass control contract mismatch")
    if not metrics.get("safe_stop_success", False):
        raise RuntimeError("historical pass lacks successful safe stop")
    return attempt


def run_bounded(
    client: SimClient,
    config: RepeatabilityConfig,
    sim_root: Path,
    result_dir: Path,
    *,
    preflight_one: Callable[[SimClient, CandidateRuntime, Path], tuple[Preflight, dict[str, Any]]] = full_preflight,
    run_one: Callable[[SimClient, CandidateRuntime, Preflight], dict[str, Any]] = run_candidate,
) -> tuple[list[dict[str, Any]], str]:
    attempts: list[dict[str, Any]] = []
    valid_passes = 0
    for number in range(1, MAXIMUM_NEW_LIVE_ATTEMPTS + 1):
        try:
            initial, preflight_result = preflight_one(client, config.runtime, sim_root)
            metrics = run_one(client, config.runtime, initial)
            classification = classify(metrics)
            attempt = {"attempt_number": number, "classification": classification,
                       "preflight": preflight_result, "metrics": metrics}
        except Exception as exc:
            stop_errors = client.safe_stop()
            attempt = {"attempt_number": number, "classification": "INFRA_FAIL", "metrics": None,
                       "preflight": {"result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                                     "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors}}
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "EXPERT_FAIL":
            return attempts, "FAIL"
        if attempt["classification"] == "EXPERT_PASS":
            valid_passes += 1
            if valid_passes == TARGET_NEW_VALID_RUNS:
                return attempts, "PASS"
    return attempts, "INCONCLUSIVE"


def aggregate(historical: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    new_passes = [item for item in attempts if item["classification"] == "EXPERT_PASS"]
    if len(new_passes) != TARGET_NEW_VALID_RUNS:
        raise ValueError("aggregate requires exactly two new valid Expert passes")
    runs = [historical["metrics"], *(item["metrics"] for item in new_passes)]
    lap_times = [float(run["elapsed_s"]) for run in runs]
    mean_ctes = [float(run["mean_centerline_error_m"]) for run in runs]
    saturations = [float(run["steering_saturation_fraction"]) for run in runs]
    return {
        "expert_success": "3/3", "historical_pass_counted": 1, "new_valid_pass_count": 2,
        "lap_time_mean_s": statistics.fmean(lap_times), "lap_time_sample_std_s": statistics.stdev(lap_times),
        "lap_time_range_s": [min(lap_times), max(lap_times)], "mean_cte_per_run_m": mean_ctes,
        "mean_of_mean_cte_m": statistics.fmean(mean_ctes),
        "worst_max_cte_m": max(float(run["max_centerline_error_m"]) for run in runs),
        "steering_saturation_mean": statistics.fmean(saturations),
        "steering_saturation_range": [min(saturations), max(saturations)],
        "safe_stop_success_count": sum(bool(run["safe_stop_success"]) for run in runs),
        "infrastructure_failure_count": sum(item["classification"] == "INFRA_FAIL" for item in attempts),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True); parser.add_argument("--sim-root", type=Path, required=True)
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
        print("ERROR: refusing to repeat the bounded Expert repeatability experiment", file=sys.stderr); return 2
    report: dict[str, Any] = {"version": VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
                              "result": "INCONCLUSIVE", "attempts": [],
                              "maximum_new_live_attempts": MAXIMUM_NEW_LIVE_ATTEMPTS}
    client: SimClient | None = None; code = 2
    try:
        config = RepeatabilityConfig.load(args.config, args.canonical_config)
        historical = load_historical(args.historical)
        report["canonical_config"] = {"path": str(args.canonical_config.resolve()),
                                      "sha256": sha256_file(args.canonical_config), "unchanged": True}
        report["historical_provenance"] = {"path": str(args.historical), "sha256": sha256_file(args.historical),
                                           "classification": historical["classification"], "counted_exactly_once": True}
        report["frozen_contract"] = {"speed_mps": SPEED_MPS, "lookahead_m": LOOKAHEAD_M,
                                     "control_frequency_hz": config.runtime.driver.control_frequency_hz,
                                     "max_steering_rad": config.runtime.driver.max_steering_rad,
                                     "wheelbase_m": config.runtime.driver.wheelbase_m}
        client = SimClient(config.runtime.driver.base_url, config.runtime.driver.api_timeout_s)
        if errors := client.safe_stop(): raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
        if args.preflight_only:
            _, preflight_result = full_preflight(client, config.runtime, args.sim_root)
            report["preflight"] = preflight_result; report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            write_json(marker, {"status": "BOUNDED_REPEATABILITY_STARTED_DO_NOT_REPEAT",
                                "started_utc": datetime.now(timezone.utc).isoformat(),
                                "maximum_new_live_attempts": MAXIMUM_NEW_LIVE_ATTEMPTS,
                                "target_new_valid_runs": TARGET_NEW_VALID_RUNS})
            report["attempts"], report["result"] = run_bounded(client, config, args.sim_root, args.result_dir)
            report["new_live_attempt_count"] = len(report["attempts"])
            report["new_valid_pass_count"] = sum(item["classification"] == "EXPERT_PASS" for item in report["attempts"])
            report["infrastructure_failure_count"] = sum(item["classification"] == "INFRA_FAIL" for item in report["attempts"])
            report["aggregate"] = aggregate(historical, report["attempts"]) if report["result"] == "PASS" else None
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


if __name__ == "__main__": raise SystemExit(main())
