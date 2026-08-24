import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import Preflight
from physicar_e2e.pilotnet_inference import (
    CameraOnlyOnnxModel,
    InferenceConfig,
    fixed_speed_commands,
    live_camera_preflight,
    run_gated_smokes,
    run_smoke,
)
from physicar_e2e.route_geometry import ClosedRoute


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilotnet_inference_v1.json"


def initial_state():
    center = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    inner = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
    outer = [(-0.2, -0.2), (1.2, -0.2), (1.2, 1.2), (-0.2, 1.2), (-0.2, -0.2)]
    return Preflight("world", ClosedRoute(center, inner, outer), 5, 0, {}, {"x": 0, "y": 0, "yaw": 0})


class InferenceContractTests(unittest.TestCase):
    def test_fixed_speed_is_separate_and_steering_is_clamped(self):
        steering, speed = fixed_speed_commands(0.30, 2.0)
        self.assertAlmostEqual(steering, 0.349066)
        self.assertEqual(speed, 0.30)

    def test_model_observation_is_camera_only(self):
        self.assertEqual(CameraOnlyOnnxModel.observation_fields, ("camera_yuv",))

    def test_live_preflight_reports_http_jpeg_shape(self):
        image = Image.new("RGB", (480, 360), (1, 2, 3))
        data = io.BytesIO()
        image.save(data, format="JPEG")

        class Client:
            def camera_jpeg(self, path="/camera"):
                return data.getvalue()

        result = live_camera_preflight(Client(), InferenceConfig.load(CONFIG))
        self.assertEqual(result["model_input_shape"], [3, 66, 200])
        self.assertEqual(result["transport"], "HTTP JPEG")

    def test_runtime_failure_safe_stops_and_privileged_data_stays_outside_model(self):
        class Client:
            def status(self):
                return {"running": True, "switching": False, "current": "world"}

            def camera_jpeg(self, path="/camera"):
                raise RuntimeError("camera unavailable")

            def safe_stop(self):
                self.stopped = True
                return []

        class Model:
            observation_fields = ("camera_yuv",)

        client = Client()
        metrics = run_smoke(client, Model(), InferenceConfig.load(CONFIG), initial_state(), 0.30)
        self.assertEqual(metrics["result"], "FAIL")
        self.assertTrue(metrics["safe_stop_success"])
        self.assertTrue(client.stopped)
        self.assertEqual(metrics["neural_observation_fields"], ["camera_yuv"])
        self.assertIn("GT pose", metrics["privileged_safety_and_metrics_fields"])

    def test_no_smoke_b_or_third_run_after_smoke_a_failure(self):
        config = InferenceConfig.load(CONFIG)
        with (
            patch("physicar_e2e.pilotnet_inference.wait_after_reset", return_value=initial_state()),
            patch("physicar_e2e.pilotnet_inference.run_smoke", return_value={"result": "FAIL"}) as run,
        ):
            results = run_gated_smokes(object(), object(), config)
        self.assertEqual(results, [{"result": "FAIL"}])
        self.assertEqual(run.call_count, 1)

    def test_exactly_two_runs_when_both_pass(self):
        config = InferenceConfig.load(CONFIG)
        with (
            patch("physicar_e2e.pilotnet_inference.wait_after_reset", return_value=initial_state()),
            patch("physicar_e2e.pilotnet_inference.run_smoke", return_value={"result": "PASS"}) as run,
        ):
            results = run_gated_smokes(object(), object(), config)
        self.assertEqual(len(results), 2)
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
