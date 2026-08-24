from pathlib import Path
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet_inference import sha256_file
from physicar_e2e.expert_speed_validation import (
    EXPECTED_CANONICAL_CONFIG_SHA256, EXPECTED_RESULT_DIRECTORY, MAXIMUM_LIVE_RUNS,
    SPEED_MPS, ExpertSpeedConfig, execute_one,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/expert_driver_speed_1p8_v1.json"
CANONICAL = ROOT / "configs/expert_driver_v1.json"


class ExpertSpeedTests(unittest.TestCase):
    def setUp(self): self.config = ExpertSpeedConfig.load(EXPERIMENT, CANONICAL)

    def test_only_runtime_difference_is_exact_speed(self):
        changed = [name for name in self.config.canonical.__dataclass_fields__
                   if getattr(self.config.canonical, name) != getattr(self.config.driver, name)]
        self.assertEqual(changed, ["fixed_speed_mps"])
        self.assertEqual((SPEED_MPS, self.config.driver.fixed_speed_mps), (1.8, 1.8))

    def test_canonical_config_lookahead_frequency_and_steering_are_unchanged(self):
        self.assertEqual(sha256_file(CANONICAL), EXPECTED_CANONICAL_CONFIG_SHA256)
        self.assertEqual(self.config.canonical.fixed_speed_mps, 0.5)
        self.assertEqual(self.config.driver.lookahead_m, 0.45)
        self.assertEqual(self.config.driver.control_frequency_hz, 15.0)
        self.assertEqual(self.config.driver.max_steering_rad, 0.349066)

    def test_one_run_no_retry_after_expert_failure_and_safe_stop_evidence(self):
        failure = {"result": "FAIL", "failure": "sustained off-track", "safe_stop_success": True,
                   "api_failures": 0, "pose_liveness_failures": 0, "clock_liveness_failures": 0}
        run = Mock(return_value=failure)
        attempt = execute_one(object(), self.config, object(), run_one=run)
        self.assertEqual(attempt["classification"], "EXPERT_FAIL")
        self.assertEqual(run.call_count, 1)
        self.assertTrue(attempt["metrics"]["safe_stop_success"])

    def test_infrastructure_failure_is_not_expert_failure(self):
        failure = {"result": "FAIL", "failure": "pose did not change meaningfully", "safe_stop_success": True,
                   "api_failures": 0, "pose_liveness_failures": 1, "clock_liveness_failures": 0}
        attempt = execute_one(object(), self.config, object(), run_one=Mock(return_value=failure))
        self.assertEqual(attempt["classification"], "INFRA_FAIL")

    def test_result_isolation_and_no_training_dependency(self):
        self.assertEqual(self.config.payload["result_directory"], EXPECTED_RESULT_DIRECTORY)
        self.assertEqual(MAXIMUM_LIVE_RUNS, 1)
        source = (ROOT / "src/physicar_e2e/expert_speed_validation.py").read_text()
        self.assertNotIn("train_pilotnet", source)
        self.assertNotIn("dagger", source.lower())


if __name__ == "__main__": unittest.main()
