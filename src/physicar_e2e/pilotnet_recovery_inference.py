"""Gated PilotNet V2 recovery-data 0.50 m/s repeatability runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

from .expert_driver import wait_after_reset
from .pilotnet_inference import (
    CameraOnlyOnnxModel,
    InferenceConfig,
    live_camera_preflight,
    run_smoke,
    sha256_file,
)
from .sim_client import SimClient


def load_v2_config(path: Path) -> InferenceConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke_speeds_mps") != [0.5, 0.5, 0.5] or payload.get("maximum_smoke_runs") != 3:
        raise ValueError("V2 requires exactly three conditional 0.50 m/s runs")
    if payload.get("camera_only_model_observation") is not True:
        raise ValueError("V2 neural observation must remain camera-only")
    if (payload.get("source_width"), payload.get("source_height")) != (480, 360):
        raise ValueError("V2 source camera contract changed")
    if (payload.get("model_width"), payload.get("model_height")) != (200, 66):
        raise ValueError("V2 model input contract changed")
    return InferenceConfig(payload)


def run_conditional_v2(client, model, config: InferenceConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for speed in config.payload["smoke_speeds_mps"]:
        if results and results[-1]["result"] != "PASS":
            break
        initial = wait_after_reset(client, config.safety_config(float(speed)), False)
        result = run_smoke(client, model, config, initial, float(speed))
        results.append(result)
        if result["result"] != "PASS":
            break
    return results


def aggregate_repeatability(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if len(results) != 3 or any(item["result"] != "PASS" for item in results):
        return None
    lap_times = [float(item["elapsed_s"]) for item in results]
    mean_ctes = [float(item["mean_cte_m"]) for item in results]
    saturations = [float(item["steering_saturation_fraction"]) for item in results]
    return {
        "result": "3/3 PASS", "lap_time_mean_s": statistics.fmean(lap_times),
        "lap_time_std_s": statistics.pstdev(lap_times),
        "mean_cte_mean_m": statistics.fmean(mean_ctes), "mean_cte_std_m": statistics.pstdev(mean_ctes),
        "worst_max_cte_m": max(float(item["max_cte_m"]) for item in results),
        "saturation_fraction_mean": statistics.fmean(saturations),
        "saturation_fraction_range": [min(saturations), max(saturations)],
        "failure_count": 0, "safe_stop_count": sum(item["safe_stop_success"] for item in results),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only == args.run:
        print("ERROR: choose exactly one of --preflight-only or --run", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "version": "pilotnet_e2e_v2_recovery", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL", "runs": [],
    }
    client = None
    code = 2
    try:
        config = load_v2_config(args.config)
        model = CameraOnlyOnnxModel(args.onnx)
        client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
        stop = client.safe_stop()
        if stop:
            raise RuntimeError("initial safe-stop failed: " + "; ".join(stop))
        static = wait_after_reset(client, config.safety_config(0.5), False)
        report["provenance"] = {
            "inference_config_sha256": sha256_file(args.config), "onnx_sha256": sha256_file(args.onnx),
            "onnx_size_bytes": args.onnx.stat().st_size,
        }
        report["simulator_preflight"] = {
            "result": "PASS", "world": static.world, "route_length_m": static.route.length,
            "route_points": static.route_points, "cone_count": static.cone_count,
        }
        report["camera_preflight"] = live_camera_preflight(client, config)
        report["camera_only_model_observation"] = True
        if args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"
            code = 0
        else:
            report["runs"] = run_conditional_v2(client, model, config)
            report["repeatability"] = aggregate_repeatability(report["runs"])
            report["result"] = "PASS" if len(report["runs"]) == 3 and all(
                item["result"] == "PASS" for item in report["runs"]
            ) else "FAIL"
            code = 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        if client is not None:
            final_stop = client.safe_stop()
            report["final_safe_stop_success"] = not final_stop
            report["final_safe_stop_errors"] = final_stop
            if final_stop:
                report["result"] = "FAIL"
                code = 2
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
