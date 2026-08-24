"""Camera-only ONNX PilotNet V1 simulator runner with privileged safety evaluation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image

from .expert_driver import DriverConfig, PoseLivenessMonitor, Preflight, preflight, wait_after_reset
from .pilotnet import clamp_steering_rad, preprocess_live_jpeg, steering_normalized_to_rad
from .route_geometry import OffTrackMonitor, ProgressTracker
from .sim_client import SimClient


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class InferenceConfig:
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "InferenceConfig":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        required = {
            "base_url", "expected_world", "camera_path", "camera_only_model_observation",
            "source_width", "source_height", "roi", "model_width", "model_height",
            "max_steering_rad", "smoke_speeds_mps", "maximum_smoke_runs",
            "control_frequency_hz", "maximum_runtime_s",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"inference config missing fields: {sorted(missing)}")
        if payload["camera_only_model_observation"] is not True:
            raise ValueError("camera_only_model_observation must remain true")
        if payload["smoke_speeds_mps"] != [0.30, 0.50] or payload["maximum_smoke_runs"] != 2:
            raise ValueError("V1 permits exactly Smoke A 0.30 then conditional Smoke B 0.50")
        if (payload["source_width"], payload["source_height"]) != (480, 360):
            raise ValueError("V1 live camera contract requires 480x360")
        if (payload["model_width"], payload["model_height"]) != (200, 66):
            raise ValueError("V1 model input contract requires 200x66")
        return cls(payload)

    def safety_config(self, speed_mps: float) -> DriverConfig:
        p = self.payload
        config = DriverConfig(
            base_url=p["base_url"], expected_world=p["expected_world"], wheelbase_m=0.18,
            max_steering_rad=p["max_steering_rad"], fixed_speed_mps=speed_mps,
            control_frequency_hz=p["control_frequency_hz"], lookahead_m=0.45,
            start_gate_radius_m=p["start_gate_radius_m"],
            minimum_lap_progress_fraction=p["minimum_lap_progress_fraction"],
            off_track_margin_m=p["off_track_margin_m"], off_track_grace_s=p["off_track_grace_s"],
            api_timeout_s=p["api_timeout_s"], pose_stale_timeout_s=p["pose_stale_timeout_s"],
            pose_motion_translation_threshold_m=p["pose_motion_translation_threshold_m"],
            pose_motion_yaw_threshold_rad=p["pose_motion_yaw_threshold_rad"],
            maximum_runtime_s=p["maximum_runtime_s"], closed_route_tolerance_m=p["closed_route_tolerance_m"],
            spawn_route_tolerance_m=p["spawn_route_tolerance_m"], minimum_route_points=p["minimum_route_points"],
            maximum_progress_jump_m=p["maximum_progress_jump_m"],
            world_check_interval_s=p["world_check_interval_s"], reset_wait_timeout_s=p["reset_wait_timeout_s"],
        )
        config.validate()
        return config

    @property
    def roi(self) -> tuple[int, int, int, int]:
        roi = self.payload["roi"]
        return (roi["x_start"], roi["y_start"], roi["x_end"], roi["y_end"])


class CameraClientLike(Protocol):
    def camera_jpeg(self, path: str = "/camera") -> bytes: ...
    def status(self) -> dict[str, Any]: ...
    def pose(self) -> dict[str, Any]: ...
    def clock(self) -> dict[str, Any]: ...
    def command_steering(self, value: float) -> dict[str, Any]: ...
    def command_speed(self, value: float) -> dict[str, Any]: ...
    def safe_stop(self) -> list[str]: ...


class CameraOnlyOnnxModel:
    """The neural boundary accepts one camera tensor and no privileged values."""

    observation_fields = ("camera_yuv",)

    def __init__(self, onnx_path: str | Path) -> None:
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        if input_meta.name != "camera_yuv" or input_meta.shape[1:] != [3, 66, 200]:
            raise ValueError(f"unexpected ONNX input contract: {input_meta.name} {input_meta.shape}")
        if output_meta.name != "steering_normalized" or output_meta.shape[-1] != 1:
            raise ValueError(f"unexpected ONNX output contract: {output_meta.name} {output_meta.shape}")

    def predict(self, camera_yuv: np.ndarray) -> float:
        if camera_yuv.shape != (3, 66, 200) or camera_yuv.dtype != np.float32:
            raise ValueError(f"model observation must be float32 camera CHW, got {camera_yuv.shape} {camera_yuv.dtype}")
        output = self.session.run(
            ["steering_normalized"], {"camera_yuv": np.expand_dims(camera_yuv, 0)}
        )[0]
        value = float(output.reshape(-1)[0])
        if not math.isfinite(value):
            raise RuntimeError("ONNX produced non-finite steering")
        return value


def fixed_speed_commands(speed_mps: float, steering_rad: float) -> tuple[float, float]:
    if not math.isfinite(speed_mps) or speed_mps <= 0:
        raise ValueError("fixed speed must be finite and positive")
    return clamp_steering_rad(steering_rad), float(speed_mps)


def _summary_ms(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "count": len(values), "mean_ms": float(np.mean(milliseconds)),
        "median_ms": float(np.median(milliseconds)), "p95_ms": float(np.percentile(milliseconds, 95)),
        "max_ms": float(np.max(milliseconds)),
    }


def classify_failure(message: str | None, steerings: Sequence[float], saturation_fraction: float) -> str | None:
    if message is None:
        return None
    lowered = message.lower()
    if "camera" in lowered or "jpeg" in lowered or "480x360" in lowered:
        return "JPEG-vs-raw input domain shift or camera acquisition/preprocessing failure"
    if "onnx" in lowered or "shape" in lowered or "non-finite steering" in lowered:
        return "preprocessing mismatch or inference implementation error"
    if "timeout" in lowered or "period" in lowered:
        return "inference/control latency"
    if "off-track" in lowered:
        if saturation_fraction > 0.25:
            return "output saturation"
        if len(steerings) > 4 and statistics.fmean(abs(steerings[i] - steerings[i - 1]) for i in range(1, len(steerings))) > 0.12:
            return "oscillation"
        return "accumulation/distribution shift or inability to recover after deviation"
    if "runtime" in lowered:
        return "insufficient route progress or accumulation/distribution shift"
    return "other implementation/runtime error"


def live_camera_preflight(client: CameraClientLike, config: InferenceConfig) -> dict[str, Any]:
    started = time.perf_counter()
    jpeg = client.camera_jpeg(config.payload["camera_path"])
    acquisition = time.perf_counter() - started
    with Image.open(io.BytesIO(jpeg)) as image:
        dimensions = list(image.size)
        fmt = image.format
    started = time.perf_counter()
    tensor = preprocess_live_jpeg(jpeg, roi=config.roi)
    preprocessing = time.perf_counter() - started
    if dimensions != [480, 360] or fmt != "JPEG" or tensor.shape != (3, 66, 200):
        raise RuntimeError(f"live camera contract mismatch: {fmt} {dimensions}, tensor {tensor.shape}")
    return {
        "result": "PASS", "transport": "HTTP JPEG", "source_dimensions": dimensions,
        "model_input_shape": list(tensor.shape), "camera_acquisition_ms": acquisition * 1000.0,
        "preprocessing_ms": preprocessing * 1000.0,
        "domain_difference": "Training uses raw ROS RGB-derived PNG; live input uses HTTP JPEG.",
    }


def run_smoke(
    client: CameraClientLike,
    model: CameraOnlyOnnxModel,
    config: InferenceConfig,
    initial: Preflight,
    speed_mps: float,
) -> dict[str, Any]:
    safety = config.safety_config(speed_mps)
    route = initial.route
    tracker = ProgressTracker(route.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s)
    liveness = PoseLivenessMonitor(
        safety.pose_stale_timeout_s, safety.pose_motion_translation_threshold_m,
        safety.pose_motion_yaw_threshold_rad,
    )
    periods: list[float] = []
    ctes: list[float] = []
    steerings: list[float] = []
    camera_latencies: list[float] = []
    preprocessing_latencies: list[float] = []
    inference_latencies: list[float] = []
    previous_tick: float | None = None
    next_tick = time.monotonic()
    next_world_check = next_tick
    started = time.monotonic()
    final_pose = initial.pose
    final_projection = route.project((final_pose["x"], final_pose["y"]))
    failure: str | None = None
    result = "FAIL"
    api_failures = 0
    liveness_failures = 0
    saturation_count = 0
    motion_commanded = False
    stop_errors: list[str] = []
    try:
        while True:
            now = time.monotonic()
            if now - started >= safety.maximum_runtime_s:
                raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick:
                time.sleep(next_tick - now)
            tick = time.monotonic()
            if previous_tick is not None:
                periods.append(tick - previous_tick)
            previous_tick = tick
            if tick >= next_world_check:
                status = client.status()
                if status.get("running") is not True or status.get("switching") is not False or status.get("current") != initial.world:
                    raise RuntimeError(f"simulator state changed while driving: {status}")
                next_world_check = tick + safety.world_check_interval_s

            camera_started = time.perf_counter()
            jpeg = client.camera_jpeg(config.payload["camera_path"])
            camera_latencies.append(time.perf_counter() - camera_started)
            preprocessing_started = time.perf_counter()
            camera_yuv = preprocess_live_jpeg(jpeg, roi=config.roi)
            preprocessing_latencies.append(time.perf_counter() - preprocessing_started)
            inference_started = time.perf_counter()
            normalized_steering = model.predict(camera_yuv)
            inference_latencies.append(time.perf_counter() - inference_started)
            physical_unclamped = float(steering_normalized_to_rad(normalized_steering, safety.max_steering_rad))
            steering, speed = fixed_speed_commands(speed_mps, physical_unclamped)

            # Privileged values begin here and are used only by safety/metrics.
            pose = client.pose()
            clock = client.clock()
            try:
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=motion_commanded)
            except RuntimeError:
                liveness_failures += 1
                raise
            final_pose = pose
            final_projection = route.project((pose["x"], pose["y"]))
            tracker.update(final_projection.s)
            boundary_distance = route.track_boundary_distance((pose["x"], pose["y"]))
            if boundary_distance is None or not math.isfinite(boundary_distance):
                raise RuntimeError("invalid track boundary geometry")
            if off_track.update(boundary_distance > safety.off_track_margin_m, time.monotonic()):
                raise RuntimeError(f"sustained off-track: boundary distance {boundary_distance:.3f}m")

            client.command_steering(steering)
            client.command_speed(speed)
            if not motion_commanded:
                motion_commanded = True
                liveness.update(pose, float(clock["sim_time"]), time.monotonic(), motion_commanded=True)
            ctes.append(final_projection.distance)
            steerings.append(steering)
            saturation_count += math.isclose(abs(steering), safety.max_steering_rad, abs_tol=1e-8)
            distance_to_start = math.dist((pose["x"], pose["y"]), route.points[0])
            if tracker.lap_complete(distance_to_start, safety.start_gate_radius_m, safety.minimum_lap_progress_fraction):
                result = "PASS"
                break
            next_tick += 1.0 / safety.control_frequency_hz
            if next_tick < time.monotonic() - 1.0 / safety.control_frequency_hz:
                next_tick = time.monotonic()
    except Exception as exc:
        failure = str(exc)
        if any(word in failure.lower() for word in ("get ", "post ", "control rejected", "unavailable")):
            api_failures += 1
    finally:
        ended = time.monotonic()
        off_track.finalize(ended)
        stop_errors = client.safe_stop()
        if stop_errors:
            result = "FAIL"
            failure = (failure + "; " if failure else "") + "; ".join(stop_errors)
            api_failures += len(stop_errors)
    elapsed = time.monotonic() - started
    final_distance = math.dist((final_pose["x"], final_pose["y"]), route.points[0])
    saturation_fraction = saturation_count / len(steerings) if steerings else 0.0
    deltas = [abs(steerings[i] - steerings[i - 1]) for i in range(1, len(steerings))]
    target_period_s = 1.0 / safety.control_frequency_hz
    return {
        "result": result, "failure": failure,
        "failure_category": classify_failure(failure, steerings, saturation_fraction),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "speed_mps": speed_mps,
        "elapsed_s": elapsed, "route_length_m": route.length,
        "route_completion_fraction": tracker.unwrapped / route.length,
        "total_unwrapped_progress_m": tracker.unwrapped, "final_route_s_m": final_projection.s,
        "final_distance_to_start_m": final_distance,
        "mean_cte_m": statistics.fmean(ctes) if ctes else 0.0,
        "max_cte_m": max(ctes, default=0.0), "off_track_events": off_track.event_count,
        "off_track_total_duration_s": off_track.total_duration_s,
        "mean_absolute_predicted_steering_rad": statistics.fmean(abs(v) for v in steerings) if steerings else 0.0,
        "max_absolute_predicted_steering_rad": max((abs(v) for v in steerings), default=0.0),
        "steering_saturation_fraction": saturation_fraction,
        "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.0,
        "camera_acquisition_latency": _summary_ms(camera_latencies),
        "preprocessing_latency": _summary_ms(preprocessing_latencies),
        "onnx_inference_latency": _summary_ms(inference_latencies),
        "control_loop_period": _summary_ms(periods),
        "control_loop_frequency_hz": 1.0 / statistics.fmean(periods) if periods else 0.0,
        "timing_slips": sum(period > 1.5 * target_period_s for period in periods),
        "control_iterations": len(steerings), "api_failures": api_failures,
        "liveness_failures": liveness_failures, "safe_stop_success": not stop_errors,
        "safe_stop_errors": stop_errors, "neural_observation_fields": list(model.observation_fields),
        "privileged_safety_and_metrics_fields": ["GT pose", "route", "track boundaries", "simulator clock", "world status"],
    }


def run_gated_smokes(client, model, config: InferenceConfig) -> list[dict[str, Any]]:
    """Run at most A then B; B is forbidden unless A passes."""
    results: list[dict[str, Any]] = []
    for index, speed in enumerate(config.payload["smoke_speeds_mps"]):
        if index >= config.payload["maximum_smoke_runs"]:
            break
        if index == 1 and results[0]["result"] != "PASS":
            break
        safety = config.safety_config(float(speed))
        initial = wait_after_reset(client, safety, False)
        results.append(run_smoke(client, model, config, initial, float(speed)))
        if results[-1]["result"] != "PASS":
            break
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--run-smokes", action="store_true")
    parser.add_argument("--result", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.preflight_only == args.run_smokes:
        print("ERROR: choose exactly one of --preflight-only or --run-smokes", file=sys.stderr)
        return 2
    client: SimClient | None = None
    exit_code = 2
    report: dict[str, Any] = {
        "version": "pilotnet_e2e_smoke_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "FAIL", "smokes": [],
    }
    try:
        config = InferenceConfig.load(args.config)
        model = CameraOnlyOnnxModel(args.onnx)
        report["provenance"] = {
            "inference_config_sha256": sha256_file(args.config),
            "onnx_sha256": sha256_file(args.onnx),
            "onnx_size_bytes": args.onnx.stat().st_size,
        }
        client = SimClient(config.payload["base_url"], config.payload["api_timeout_s"])
        initial_stop = client.safe_stop()
        if initial_stop:
            raise RuntimeError("initial safe-stop failed: " + "; ".join(initial_stop))
        safety = config.safety_config(0.30)
        static = preflight(client, safety, False)
        report["simulator_preflight"] = {
            "result": "PASS", "world": static.world, "route_length_m": static.route.length,
            "route_points": static.route_points, "cone_count": static.cone_count,
        }
        report["camera_preflight"] = live_camera_preflight(client, config)
        report["camera_only_model_observation"] = True
        report["transport_domain_limitation"] = "Training PNGs derive from raw ROS RGB; HTTP camera frames are JPEG."
        if args.preflight_only:
            report["result"] = "PREFLIGHT_PASS"
            exit_code = 0
        else:
            report["smokes"] = run_gated_smokes(client, model, config)
            report["smoke_a"] = report["smokes"][0] if report["smokes"] else {"result": "NOT_EXECUTED"}
            report["smoke_b"] = report["smokes"][1] if len(report["smokes"]) > 1 else {"result": "NOT_EXECUTED", "reason": "Smoke A did not pass"}
            report["result"] = "PASS" if report["smokes"] and all(item["result"] == "PASS" for item in report["smokes"]) else "FAIL"
            exit_code = 0 if report["result"] == "PASS" else 1
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
                exit_code = 2
        if args.result:
            args.result.parent.mkdir(parents=True, exist_ok=True)
            args.result.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
