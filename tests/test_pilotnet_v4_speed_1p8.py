import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet_inference import sha256_file
from physicar_e2e.pilotnet_v4_speed_validation import (
    EXPECTED_CANONICAL_CONFIG_SHA256, EXPECTED_ONNX_SHA256, EXPECTED_RESULT_DIRECTORY,
    MAX_LIVE_ATTEMPTS, SPEED_MPS, SpeedValidationConfig, bounded_runs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pilotnet_v4_speed_1p8_v1.json"
CANONICAL = ROOT / "configs/pilotnet_inference_v4_dagger.json"


def run_result(result="PASS", failure=None):
    return {"result": result, "failure": failure, "api_failures": 0, "liveness_failures": 0,
            "safe_stop_success": True, "elapsed_s": 20.0, "mean_cte_m": 0.02, "max_cte_m": 0.1,
            "steering_saturation_fraction": 0.05, "control_loop_period": {"p95_ms": 67.0, "max_ms": 70.0}}


class FakeClient:
    def safe_stop(self): return []


class SpeedContractTests(unittest.TestCase):
    def setUp(self): self.config = SpeedValidationConfig.load(CONFIG, CANONICAL)

    def test_exact_speed_rate_clamp_and_separate_path(self):
        safety = self.config.safety_config()
        self.assertEqual((SPEED_MPS, safety.fixed_speed_mps), (1.8, 1.8))
        self.assertEqual(safety.control_frequency_hz, 15.0)
        self.assertEqual(safety.max_steering_rad, 0.349066)
        self.assertEqual(self.config.payload["result_directory"], EXPECTED_RESULT_DIRECTORY)

    def test_canonical_config_and_model_hash_constants_unchanged(self):
        self.assertEqual(sha256_file(CANONICAL), EXPECTED_CANONICAL_CONFIG_SHA256)
        self.assertEqual(self.config.canonical.payload["smoke_speeds_mps"], [0.5, 0.5, 0.5])
        self.assertEqual(EXPECTED_ONNX_SHA256, "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a")

    def execute(self, outcomes):
        with tempfile.TemporaryDirectory() as directory, \
             patch("physicar_e2e.pilotnet_v4_speed_validation.attempt_preflight", return_value=(object(), {"result": "PASS"})):
            iterator = iter(outcomes)
            return bounded_runs(FakeClient(), object(), self.config, Path(directory), run_one=lambda *_: next(iterator))

    def test_first_policy_failure_stops_without_retry(self):
        attempts, result = self.execute([run_result("FAIL", "sustained off-track"), run_result()])
        self.assertEqual((len(attempts), result), (1, "FAIL"))

    def test_repeatability_is_conditional_on_first_pass(self):
        attempts, result = self.execute([run_result(), run_result(), run_result()])
        self.assertEqual((len(attempts), result), (3, "PASS"))

    def test_later_policy_failure_is_partial_pass(self):
        attempts, result = self.execute([run_result(), run_result("FAIL", "sustained off-track"), run_result()])
        self.assertEqual((len(attempts), result), (2, "PARTIAL_PASS"))

    def test_initial_infrastructure_failure_gets_only_one_retry(self):
        infra = run_result("FAIL", "GET /camera unavailable"); infra["api_failures"] = 1
        attempts, result = self.execute([infra, infra, run_result()])
        self.assertEqual((len(attempts), result), (2, "INCONCLUSIVE"))

    def test_repeatability_infrastructure_replacements_are_bounded_at_five(self):
        infra = run_result("FAIL", "simulator clock did not advance"); infra["liveness_failures"] = 1
        attempts, result = self.execute([run_result(), infra, infra, infra, infra])
        self.assertEqual((len(attempts), result), (MAX_LIVE_ATTEMPTS, "INCONCLUSIVE"))

    def test_no_training_import_and_safe_stop_on_preflight_exception(self):
        source = (ROOT / "src/physicar_e2e/pilotnet_v4_speed_validation.py").read_text()
        self.assertNotIn("train_pilotnet", source)
        client = FakeClient(); client.safe_stop = unittest.mock.Mock(return_value=[])
        with tempfile.TemporaryDirectory() as directory, \
             patch("physicar_e2e.pilotnet_v4_speed_validation.attempt_preflight", side_effect=RuntimeError("clock")):
            attempts, result = bounded_runs(client, object(), self.config, Path(directory))
        self.assertEqual(result, "INCONCLUSIVE")
        self.assertEqual(client.safe_stop.call_count, 2)


if __name__ == "__main__": unittest.main()
