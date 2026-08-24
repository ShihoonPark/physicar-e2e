"""One-run canonical Pure Pursuit Expert feasibility gate at 1.80 m/s."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

from .expert_driver import DriverConfig, Preflight, run_driver, wait_after_reset
from .pilotnet_inference import sha256_file
from .pilotnet_v4_repeatability import clock_health_preflight, verify_static_environment
from .rosbag_collector import verify_environment
from .sim_client import SimClient


VERSION = "expert_driver_speed_1p8_v1"
SPEED_MPS = 1.80
MAXIMUM_LIVE_RUNS = 1
EXPECTED_RESULT_DIRECTORY = "results/expert_speed_1p8_v1"
EXPECTED_CANONICAL_CONFIG_SHA256 = "63814d3a30f8753092cd33fc53d44414cfb343e39caf805e624dbaf33a4bd050"
TERMINAL_WINDOW_SAMPLES = 15


@dataclass(frozen=True)
class ExpertSpeedConfig:
    payload: dict[str, Any]
    canonical: DriverConfig
    driver: DriverConfig

    @classmethod
    def load(cls, experiment_path: Path, canonical_path: Path) -> "ExpertSpeedConfig":
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected = {
            "version": VERSION,
            "canonical_config": "configs/expert_driver_v1.json",
            "fixed_speed_mps": SPEED_MPS,
            "maximum_live_runs": MAXIMUM_LIVE_RUNS,
            "automatic_retry": False,
            "result_directory": EXPECTED_RESULT_DIRECTORY,
        }
        if payload != expected:
            raise ValueError("Expert 1.80 m/s config does not match the one-run protocol")
        if sha256_file(canonical_path) != EXPECTED_CANONICAL_CONFIG_SHA256:
            raise RuntimeError("canonical Expert config identity mismatch")
        canonical = DriverConfig.load(canonical_path)
        driver = replace(canonical, fixed_speed_mps=SPEED_MPS)
        driver.validate()
        changed = [name for name in canonical.__dataclass_fields__ if getattr(canonical, name) != getattr(driver, name)]
        if changed != ["fixed_speed_mps"]:
            raise RuntimeError(f"unexpected Expert runtime differences: {changed}")
        if driver.control_frequency_hz != 15.0 or driver.lookahead_m != 0.45:
            raise RuntimeError("canonical frequency or lookahead changed")
        if driver.max_steering_rad != 0.349066 or driver.wheelbase_m != 0.18:
            raise RuntimeError("canonical steering authority or wheelbase changed")
        return cls(payload, canonical, driver)


class ObservingExpertClient:
    """Forward the unchanged Expert commands while retaining compact telemetry."""

    def __init__(self, client: SimClient, initial: Preflight) -> None:
        self.client = client
        self.route = initial.route
        self.samples: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def pose(self) -> dict[str, Any]:
        pose = self.client.pose()
        projection = self.route.project((pose["x"], pose["y"]))
        self.samples.append({"wall_time_s": time.monotonic(), "route_s_m": projection.s,
                             "cte_m": projection.distance, "steering_rad": None})
        return pose

    def command_steering(self, value: float) -> dict[str, Any]:
        response = self.client.command_steering(value)
        if self.samples:
            self.samples[-1]["steering_rad"] = float(value)
        return response


def add_observability(metrics: dict[str, Any], samples: list[dict[str, Any]], config: DriverConfig) -> dict[str, Any]:
    commanded = [row for row in samples if row["steering_rad"] is not None]
    steerings = [float(row["steering_rad"]) for row in commanded]
    deltas = [abs(steerings[index] - steerings[index - 1]) for index in range(1, len(steerings))]
    terminal = steerings[-TERMINAL_WINDOW_SAMPLES:]
    max_cte = max(samples, key=lambda row: float(row["cte_m"]), default=None)
    max_steering = max(commanded, key=lambda row: abs(float(row["steering_rad"])), default=None)
    saturated = [row for row in commanded if math.isclose(abs(float(row["steering_rad"])), config.max_steering_rad, abs_tol=1e-9)]
    cte_growth = None
    if len(saturated) >= 2:
        cte_growth = float(saturated[-1]["cte_m"]) - float(saturated[0]["cte_m"])
    failure = str(metrics.get("failure") or "").lower()
    metrics.update({
        "route_completion_fraction": metrics["total_unwrapped_progress_m"] / metrics["route_length_m"],
        "control_loop_frequency_hz": 1.0 / metrics["mean_loop_period_s"] if metrics["mean_loop_period_s"] else 0.0,
        "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.0,
        "terminal_window_samples": len(terminal),
        "terminal_steering_saturation_fraction": (sum(math.isclose(abs(value), config.max_steering_rad, abs_tol=1e-9) for value in terminal) / len(terminal)) if terminal else 0.0,
        "max_cte_route_s_m": None if max_cte is None else max_cte["route_s_m"],
        "max_abs_steering_route_s_m": None if max_steering is None else max_steering["route_s_m"],
        "cte_change_across_saturated_samples_m": cte_growth,
        "cte_grew_while_saturated": None if cte_growth is None else cte_growth > 0.05,
        "api_failures": int(any(token in failure for token in ("get ", "post ", "unavailable", "control rejected"))),
        "pose_liveness_failures": int("pose did not change meaningfully" in failure),
        "clock_liveness_failures": int("clock did not advance" in failure or "clock moved backward" in failure),
    })
    return metrics


def run_observed_expert(client: SimClient, config: ExpertSpeedConfig, initial: Preflight) -> dict[str, Any]:
    observer = ObservingExpertClient(client, initial)
    return add_observability(run_driver(observer, config.driver, initial), observer.samples, config.driver)


def classify(metrics: dict[str, Any]) -> str:
    if metrics.get("result") == "PASS":
        return "EXPERT_PASS"
    failure = str(metrics.get("failure") or "").lower()
    if not metrics.get("safe_stop_success", False) or metrics.get("api_failures", 0):
        return "INFRA_FAIL"
    if metrics.get("pose_liveness_failures", 0) or metrics.get("clock_liveness_failures", 0):
        return "INFRA_FAIL"
    if any(token in failure for token in ("simulator state changed", "unexpected world", "invalid track boundary")):
        return "INFRA_FAIL"
    return "EXPERT_FAIL"


def execute_one(client: SimClient, config: ExpertSpeedConfig, initial: Preflight,
                run_one: Callable[[SimClient, ExpertSpeedConfig, Preflight], dict[str, Any]] = run_observed_expert) -> dict[str, Any]:
    metrics = run_one(client, config, initial)
    return {"attempt_number": 1, "classification": classify(metrics), "metrics": metrics}


def full_preflight(client: SimClient, config: ExpertSpeedConfig, sim_root: Path) -> tuple[Preflight, dict[str, Any]]:
    initial = wait_after_reset(client, config.driver, False)
    environment = verify_static_environment(initial)
    clock = clock_health_preflight(client)
    if clock["result"] != "PASS":
        raise RuntimeError(str(clock.get("failure_reason", "simulator clock health failed")))
    return initial, {
        "result": "PASS", "environment": environment, "clock_health": clock, "control_api": "PASS",
        "source_environment": verify_environment(Path(__file__).resolve().parents[2], sim_root),
        "speed_mps": config.driver.fixed_speed_mps, "control_frequency_hz": config.driver.control_frequency_hz,
        "lookahead_m": config.driver.lookahead_m, "wheelbase_m": config.driver.wheelbase_m,
        "max_steering_rad": config.driver.max_steering_rad,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True); parser.add_argument("--result-dir", type=Path, required=True)
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
        print("ERROR: refusing a second Expert 1.80 m/s run", file=sys.stderr); return 2
    report: dict[str, Any] = {"version": VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
                              "result": "INCONCLUSIVE", "maximum_live_runs": MAXIMUM_LIVE_RUNS,
                              "automatic_retry": False}
    client: SimClient | None = None; code = 2
    try:
        config = ExpertSpeedConfig.load(args.config, args.canonical_config)
        report["canonical_config"] = {"path": str(args.canonical_config.resolve()),
                                      "sha256": sha256_file(args.canonical_config), "unchanged": True}
        report["experimental_contract"] = {
            "speed_mps": config.driver.fixed_speed_mps, "control_frequency_hz": config.driver.control_frequency_hz,
            "lookahead_m": config.driver.lookahead_m, "wheelbase_m": config.driver.wheelbase_m,
            "max_steering_rad": config.driver.max_steering_rad,
            "runtime_difference_from_canonical": ["fixed_speed_mps"],
        }
        client = SimClient(config.driver.base_url, config.driver.api_timeout_s)
        if errors := client.safe_stop(): raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
        initial, preflight_result = full_preflight(client, config, args.sim_root)
        report["preflight"] = preflight_result
        if args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            write_json(marker, {"status": "LIVE_STARTED_DO_NOT_RETRY",
                                "started_utc": datetime.now(timezone.utc).isoformat(), "maximum_live_runs": 1})
            attempt = execute_one(client, config, initial)
            write_json(args.result_dir / "attempt_01.json", attempt)
            report["attempt"] = attempt
            report["live_runs_executed"] = 1
            report["result"] = {"EXPERT_PASS": "PASS", "EXPERT_FAIL": "FAIL", "INFRA_FAIL": "INCONCLUSIVE"}[attempt["classification"]]
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
