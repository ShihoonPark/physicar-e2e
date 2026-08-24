import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import Preflight
from physicar_e2e.pilotnet import build_pilotnet
from physicar_e2e.pilotnet_failure_diagnosis import (
    associate_frames,
    audit_preprocessing,
    compare_transport_predictions,
    classify_hypotheses,
    detect_divergence_windows,
    extract_features,
    issue_neural_commands,
    load_config,
    nearest_cosine_distances,
    run_count_guard,
    run_live_loop,
    steering_calibration,
    temporal_shift_diagnostic,
)
from physicar_e2e.route_geometry import ClosedRoute


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pilotnet_failure_diagnosis_v1.json"


def initial_state():
    center = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    inner = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
    outer = [(-0.2, -0.2), (1.2, -0.2), (1.2, 1.2), (-0.2, 1.2), (-0.2, -0.2)]
    return Preflight("world", ClosedRoute(center, inner, outer), 5, 0, {}, {"x": 0, "y": 0, "yaw": 0})


class FailureDiagnosisTests(unittest.TestCase):
    def test_steering_magnitude_bins_and_calibration(self):
        labels = np.asarray([-0.30, -0.20, -0.10, -0.01, 0.02, 0.10, 0.20, 0.30])
        predictions = labels * 0.8 + 0.01
        result = steering_calibration(predictions, labels)
        self.assertAlmostEqual(result["regression_slope"], 0.8)
        self.assertEqual(sum(item["count"] for item in result["magnitude_bins"].values()), 8)
        self.assertLess(result["magnitude_bins"]["abs_ge_0.25"]["absolute_magnitude_ratio"], 1.0)

    def test_temporal_shift_uses_actual_timestamps(self):
        t = np.asarray([0, 40, 95, 150, 215, 280, 350], dtype=np.int64) * 1_000_000
        predictions = np.sin(t / 1e8)
        labels = np.interp(t - 100_000_000, t, predictions)
        result = temporal_shift_diagnostic(t, predictions, labels, [-100, 0, 100])
        self.assertEqual(result["best_shift"]["shift_ms"], 100.0)
        self.assertGreater(result["mae_improvement_fraction"], 0.5)

    def test_feature_extraction_is_deterministic(self):
        torch.manual_seed(4)
        model = build_pilotnet()
        tensors = np.zeros((2, 3, 66, 200), dtype=np.float32)
        first = extract_features(model, tensors, torch.device("cpu"))
        second = extract_features(model, tensors, torch.device("cpu"))
        self.assertEqual(first.shape, (2, 1152))
        np.testing.assert_array_equal(first, second)

    def test_nearest_neighbor_cosine_distance(self):
        reference = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        query = np.asarray([[1, 0], [1, 1]], dtype=np.float32)
        result = nearest_cosine_distances(query, reference)
        self.assertAlmostEqual(float(result[0]), 0.0)
        self.assertAlmostEqual(float(result[1]), 1 - 1 / np.sqrt(2), places=6)

    def test_divergence_window_is_objective_and_precedes_growth(self):
        times = np.arange(12) * 0.1
        ctes = [0.01] * 5 + [0.031, 0.04, 0.05, 0.07, 0.10, 0.12, 0.14]
        result = detect_divergence_windows(times, ctes, stable_window_s=0.3, persistence_samples=4)
        self.assertEqual(result["divergence_index"], 5)
        self.assertIn(4, result["critical_pre_onset_indices"])

    def test_shadow_expert_has_no_command_authority(self):
        class Client:
            def command_steering(self, value):
                self.steering = value

            def command_speed(self, value):
                self.speed = value

        client = Client()
        issue_neural_commands(client, 0.12, 0.5)
        self.assertEqual(client.steering, 0.12)
        self.assertEqual(client.speed, 0.5)
        with self.assertRaises(TypeError):
            issue_neural_commands(client, 0.12, 0.5, 0.2)

    def test_transport_comparison_reports_prediction_and_pixel_difference(self):
        raw = np.full((360, 480, 3), [80, 120, 160], dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(raw).save(buffer, format="JPEG", quality=90)
        result = compare_transport_predictions(raw, buffer.getvalue(), lambda tensor: float(tensor.mean()))
        self.assertGreaterEqual(result["absolute_prediction_difference_rad"], 0.0)
        self.assertGreaterEqual(result["mean_absolute_pixel_difference_0_255"], 0.0)

    def test_frame_association_is_one_to_one_and_tolerance_bounded(self):
        matches = associate_frames([100, 210, 900], [95, 220, 300], tolerance_ms=0.00002)
        self.assertEqual(matches, [(0, 0, -0.000005), (1, 1, 0.00001)])

    def test_preprocessing_consistency_audit(self):
        result = audit_preprocessing(
            ROOT / "configs/dataset_extractor_v1.json",
            ROOT / "configs/pilotnet_training_v1.json",
            ROOT / "configs/pilotnet_inference_v1.json",
        )
        self.assertEqual(result["result"], "PASS")
        self.assertTrue(all(result["checks"].values()))

    def test_live_run_count_is_hard_limited_to_one(self):
        self.assertEqual(load_config(CONFIG)["maximum_live_runs"], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.json"
            run_count_guard(path)
            path.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "second diagnostic run"):
                run_count_guard(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.json"
            marker = path.with_suffix(".started.json")
            marker.write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "second diagnostic run"):
                run_count_guard(path, marker)

    def test_hypothesis_classification_prioritizes_on_policy_data(self):
        offline = {
            "preprocessing_audit": {"result": "PASS"},
            "offline": {"v1_calibration": {"mae_rad": 0.005, "regression_slope": 1.0,
                "magnitude_bins": {"abs_ge_0.25": {"absolute_magnitude_ratio": 1.0}}}},
        }
        analysis = {
            "network_shadow_expert_temporal_shift": {"best_shift": {"shift_ms": 0}, "mae_improvement_fraction": 0.0},
            "steering_windows": {"critical_pre_onset": {"corrective_magnitude_ratio": 0.9}},
            "raw_rgb_vs_http_jpeg": {"matched_pairs": 10,
                "absolute_prediction_difference_rad": {"mean": 0.001},
                "association_absolute_error_ms": {"median": 5.0},
                "windows": {"late_failure": {"absolute_prediction_difference_rad": {"median": 0.001}}}},
            "feature_distance": {"windows": {"stable_initial": {"median": 0.01}, "late_failure": {"median": 0.2}},
                                 "correlation_with_cte": 0.8},
        }
        result = classify_hypotheses(offline, analysis)
        self.assertEqual(result["hypotheses"]["H4_on_policy_distribution_shift"]["classification"], "SUPPORTED")
        self.assertIn("on-policy", result["exact_next_intervention"])

    def test_runtime_failure_always_safe_stops(self):
        class Client:
            def camera_jpeg(self, path="/camera"):
                raise RuntimeError("camera unavailable")

            def safe_stop(self):
                self.stopped = True
                return []

        class Model:
            pass

        config = load_config(CONFIG)
        config["expected_world"] = "world"
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            result, rows = run_live_loop(client, Model(), config, initial_state(), Path(directory) / "frames")
        self.assertEqual(result["result"], "FAIL")
        self.assertEqual(rows, [])
        self.assertTrue(result["safe_stop_success"])
        self.assertTrue(client.stopped)


if __name__ == "__main__":
    unittest.main()
