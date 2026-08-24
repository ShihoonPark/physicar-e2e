"""Bounded 1.80 m/s Expert lookahead characterization sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable

from .expert_driver import DriverConfig, Preflight, run_driver
from .expert_speed_validation import (
    EXPECTED_CANONICAL_CONFIG_SHA256,
    ObservingExpertClient,
    add_observability,
    classify,
    full_preflight,
    write_json,
)
from .pilotnet_inference import sha256_file
from .sim_client import SimClient


VERSION = "expert_speed_1p8_lookahead_v1"
SPEED_MPS = 1.80
LOOKAHEAD_CANDIDATES_M = [0.60, 0.75, 0.90]
MAXIMUM_LIVE_RUNS = 3
EXPECTED_RESULT_DIRECTORY = "results/expert_speed_1p8_lookahead_v1"


@dataclass(frozen=True)
class CandidateRuntime:
    driver: DriverConfig


@dataclass(frozen=True)
class LookaheadSweepConfig:
    payload: dict[str, Any]
    canonical: DriverConfig
    high_speed_base: DriverConfig

    @classmethod
    def load(cls, experiment_path: Path, canonical_path: Path) -> "LookaheadSweepConfig":
        payload = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected = {
            "version": VERSION,
            "canonical_config": "configs/expert_driver_v1.json",
            "fixed_speed_mps": SPEED_MPS,
            "lookahead_candidates_m": LOOKAHEAD_CANDIDATES_M,
            "maximum_live_runs": MAXIMUM_LIVE_RUNS,
            "stop_after_first_pass": True,
            "infrastructure_retry": False,
            "result_directory": EXPECTED_RESULT_DIRECTORY,
        }
        if payload != expected:
            raise ValueError("lookahead sweep config does not match the pre-registered protocol")
        if sha256_file(canonical_path) != EXPECTED_CANONICAL_CONFIG_SHA256:
            raise RuntimeError("canonical Expert config identity mismatch")
        canonical = DriverConfig.load(canonical_path)
        high_speed_base = replace(canonical, fixed_speed_mps=SPEED_MPS)
        high_speed_base.validate()
        return cls(payload, canonical, high_speed_base)

    def candidate(self, lookahead_m: float) -> CandidateRuntime:
        if lookahead_m not in LOOKAHEAD_CANDIDATES_M:
            raise ValueError("lookahead is not a pre-registered candidate")
        driver = replace(self.high_speed_base, lookahead_m=float(lookahead_m))
        driver.validate()
        changed = [name for name in self.canonical.__dataclass_fields__
                   if getattr(self.canonical, name) != getattr(driver, name)]
        if changed != ["fixed_speed_mps", "lookahead_m"]:
            raise RuntimeError(f"unexpected runtime differences: {changed}")
        if driver.control_frequency_hz != 15.0 or driver.max_steering_rad != 0.349066 or driver.wheelbase_m != 0.18:
            raise RuntimeError("a fixed high-speed contract value changed")
        return CandidateRuntime(driver)


def run_candidate(client: SimClient, runtime: CandidateRuntime, initial: Preflight) -> dict[str, Any]:
    observer = ObservingExpertClient(client, initial)
    metrics = run_driver(observer, runtime.driver, initial)
    return add_observability(metrics, observer.samples, runtime.driver)


def run_sweep(
    client: SimClient,
    config: LookaheadSweepConfig,
    sim_root: Path,
    result_dir: Path,
    *,
    preflight_one: Callable[[SimClient, CandidateRuntime, Path], tuple[Preflight, dict[str, Any]]] = full_preflight,
    run_one: Callable[[SimClient, CandidateRuntime, Preflight], dict[str, Any]] = run_candidate,
) -> tuple[list[dict[str, Any]], str, float | None]:
    attempts: list[dict[str, Any]] = []
    for number, lookahead_m in enumerate(LOOKAHEAD_CANDIDATES_M, start=1):
        runtime = config.candidate(lookahead_m)
        try:
            initial, preflight_result = preflight_one(client, runtime, sim_root)
            metrics = run_one(client, runtime, initial)
            classification = classify(metrics)
            attempt = {"attempt_number": number, "lookahead_m": lookahead_m,
                       "classification": classification, "preflight": preflight_result, "metrics": metrics}
        except Exception as exc:
            stop_errors = client.safe_stop()
            attempt = {"attempt_number": number, "lookahead_m": lookahead_m,
                       "classification": "INFRA_FAIL", "preflight": {
                           "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                           "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors,
                       }, "metrics": None}
        attempts.append(attempt)
        write_json(result_dir / f"attempt_{number:02d}.json", attempt)
        if attempt["classification"] == "EXPERT_PASS":
            return attempts, "PASS", lookahead_m
        if attempt["classification"] == "INFRA_FAIL":
            return attempts, "INCONCLUSIVE", None
    return attempts, "FAIL", None


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
        print("ERROR: refusing to repeat the bounded lookahead sweep", file=sys.stderr); return 2
    report: dict[str, Any] = {
        "version": VERSION, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INCONCLUSIVE", "attempts": [], "maximum_live_runs": MAXIMUM_LIVE_RUNS,
    }
    client: SimClient | None = None; code = 2
    try:
        config = LookaheadSweepConfig.load(args.config, args.canonical_config)
        report["canonical_config"] = {"path": str(args.canonical_config.resolve()),
                                      "sha256": sha256_file(args.canonical_config), "unchanged": True}
        report["fixed_contract"] = {
            "speed_mps": SPEED_MPS, "control_frequency_hz": config.high_speed_base.control_frequency_hz,
            "max_steering_rad": config.high_speed_base.max_steering_rad,
            "wheelbase_m": config.high_speed_base.wheelbase_m,
            "lookahead_candidates_m": LOOKAHEAD_CANDIDATES_M,
            "only_swept_controller_parameter": "lookahead_m",
        }
        client = SimClient(config.high_speed_base.base_url, config.high_speed_base.api_timeout_s)
        if errors := client.safe_stop(): raise RuntimeError("initial safe stop failed: " + "; ".join(errors))
        if args.preflight_only:
            _, preflight_result = full_preflight(client, config.candidate(LOOKAHEAD_CANDIDATES_M[0]), args.sim_root)
            report["preflight"] = preflight_result; report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            write_json(marker, {"status": "BOUNDED_SWEEP_STARTED_DO_NOT_REPEAT",
                                "started_utc": datetime.now(timezone.utc).isoformat(),
                                "lookahead_candidates_m": LOOKAHEAD_CANDIDATES_M,
                                "maximum_live_runs": MAXIMUM_LIVE_RUNS})
            report["attempts"], report["result"], report["first_passing_lookahead_m"] = run_sweep(
                client, config, args.sim_root, args.result_dir)
            report["live_runs_executed"] = len(report["attempts"])
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
