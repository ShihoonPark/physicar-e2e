"""Conditional V4 DAgger Iteration-2 0.50 m/s closed-loop gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from .expert_driver import wait_after_reset
from .pilotnet_inference import CameraOnlyOnnxModel, InferenceConfig, live_camera_preflight, run_smoke, sha256_file
from .sim_client import SimClient


def load_v4_config(path: Path) -> InferenceConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "pilotnet_inference_v4_dagger":
        raise ValueError("unexpected V4 inference config")
    if payload.get("smoke_speeds_mps") != [0.5, 0.5, 0.5] or payload.get("maximum_smoke_runs") != 3:
        raise ValueError("V4 permits exactly three conditional 0.50 m/s slots")
    if payload.get("camera_only_model_observation") is not True:
        raise ValueError("V4 observation must remain camera-only")
    return InferenceConfig(payload)


def run_conditional_v4(client, model, config: InferenceConfig, persist=None) -> list[dict]:
    results: list[dict] = []
    for speed in config.payload["smoke_speeds_mps"]:
        if results and results[-1]["result"] != "PASS":
            break
        initial = wait_after_reset(client, config.safety_config(float(speed)), False)
        results.append(run_smoke(client, model, config, initial, float(speed)))
        if persist is not None:
            persist(results)
        if results[-1]["result"] != "PASS":
            break
    return results


def aggregate(results: list[dict]) -> dict | None:
    if len(results) != 3 or any(item["result"] != "PASS" for item in results):
        return None
    times = [item["elapsed_s"] for item in results]; ctes = [item["mean_cte_m"] for item in results]
    saturation = [item["steering_saturation_fraction"] for item in results]
    return {"result": "3/3 PASS", "lap_time_mean_s": statistics.fmean(times), "lap_time_std_s": statistics.pstdev(times),
            "lap_time_range_s": [min(times), max(times)], "mean_cte_mean_m": statistics.fmean(ctes),
            "mean_cte_std_m": statistics.pstdev(ctes), "worst_max_cte_m": max(item["max_cte_m"] for item in results),
            "saturation_mean": statistics.fmean(saturation), "saturation_range": [min(saturation), max(saturation)],
            "failure_count": 0, "safe_stop_count": sum(item["safe_stop_success"] for item in results)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true"); parser.add_argument("--run", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.preflight_only == args.run:
        print("ERROR: choose exactly one mode", file=__import__("sys").stderr); return 2
    marker = args.result.with_suffix(".started.json")
    if args.run and (args.result.exists() or marker.exists()):
        print("ERROR: refusing another V4 live experiment", file=__import__("sys").stderr); return 2
    report = {"version": "pilotnet_e2e_v4_dagger", "generated_utc": datetime.now(timezone.utc).isoformat(), "result": "FAIL", "runs": []}
    client = None; code = 2
    try:
        config = load_v4_config(args.config); model = CameraOnlyOnnxModel(args.onnx)
        client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
        if errors := client.safe_stop(): raise RuntimeError("initial safe-stop failed: " + "; ".join(errors))
        static = wait_after_reset(client, config.safety_config(0.5), False)
        report["provenance"] = {"config_sha256": sha256_file(args.config), "onnx_sha256": sha256_file(args.onnx), "onnx_size_bytes": args.onnx.stat().st_size}
        report["preflight"] = {"result": "PASS", "world": static.world, "route_length_m": static.route.length,
            "route_points": static.route_points, "cone_count": static.cone_count, "camera": live_camera_preflight(client, config)}
        if args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"; code = 0
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"status": "V4_LIVE_STARTED_DO_NOT_RETRY", "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
            def persist(results):
                report["runs"] = results; args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            report["runs"] = run_conditional_v4(client, model, config, persist)
            report["repeatability"] = aggregate(report["runs"])
            report["result"] = "PASS" if len(report["runs"]) == 3 and all(item["result"] == "PASS" for item in report["runs"]) else "FAIL"
            code = 0 if report["result"] == "PASS" else 1
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if client is not None:
            errors = client.safe_stop(); report["final_safe_stop_success"] = not errors; report["final_safe_stop_errors"] = errors
            if errors: report["result"] = "FAIL"; code = 2
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__": raise SystemExit(main())
