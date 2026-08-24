from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.expert_speed_repeatability import (
    LOOKAHEAD_M, MAXIMUM_NEW_LIVE_ATTEMPTS, SPEED_MPS, TARGET_NEW_VALID_RUNS,
    RepeatabilityConfig, aggregate, load_historical, run_bounded,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/expert_speed_1p8_repeatability_v1.json"
CANONICAL = ROOT / "configs/expert_driver_v1.json"
HISTORICAL = ROOT / "results/expert_speed_1p8_lookahead_v1/attempt_03.json"


def metrics(result="PASS", failure=None):
    return {"result": result, "failure": failure, "safe_stop_success": True,
            "api_failures": 0, "pose_liveness_failures": 0, "clock_liveness_failures": 0,
            "elapsed_s": 16.0, "mean_centerline_error_m": 0.05,
            "max_centerline_error_m": 0.2, "steering_saturation_fraction": 0.06}


class Client:
    def __init__(self): self.stop_calls = 0
    def safe_stop(self): self.stop_calls += 1; return []


class RepeatabilityTests(unittest.TestCase):
    def setUp(self): self.config = RepeatabilityConfig.load(CONFIG, CANONICAL)

    def test_frozen_contract(self):
        driver = self.config.runtime.driver
        self.assertEqual((SPEED_MPS, driver.fixed_speed_mps), (1.8, 1.8))
        self.assertEqual((LOOKAHEAD_M, driver.lookahead_m), (0.9, 0.9))
        self.assertEqual(driver.control_frequency_hz, 15.0)
        self.assertEqual(driver.max_steering_rad, 0.349066)

    def execute(self, outcomes):
        run = Mock(side_effect=outcomes); preflight = Mock(return_value=(object(), {"result": "PASS"}))
        with tempfile.TemporaryDirectory() as directory:
            result = run_bounded(Client(), self.config, Path("/sim"), Path(directory),
                                 preflight_one=preflight, run_one=run)
        return result, run

    def test_target_two_new_valid_passes(self):
        (attempts, result), run = self.execute([metrics(), metrics(), metrics()])
        self.assertEqual((len(attempts), result, run.call_count), (TARGET_NEW_VALID_RUNS, "PASS", 2))

    def test_expert_failure_stops_immediately(self):
        (attempts, result), run = self.execute([metrics("FAIL", "sustained off-track"), metrics()])
        self.assertEqual((len(attempts), result, run.call_count), (1, "FAIL", 1))

    def test_infrastructure_failure_is_excluded_and_replaced(self):
        infra = metrics("FAIL", "clock did not advance"); infra["clock_liveness_failures"] = 1
        (attempts, result), _ = self.execute([infra, metrics(), metrics()])
        self.assertEqual((len(attempts), result), (3, "PASS"))
        result_aggregate = aggregate(load_historical(HISTORICAL), attempts)
        self.assertEqual(result_aggregate["expert_success"], "3/3")
        self.assertEqual(result_aggregate["historical_pass_counted"], 1)
        self.assertEqual(result_aggregate["infrastructure_failure_count"], 1)

    def test_maximum_four_new_attempts_and_safe_stop_on_infra_exception(self):
        client = Client(); preflight = Mock(side_effect=RuntimeError("clock unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            attempts, result = run_bounded(client, self.config, Path("/sim"), Path(directory),
                                           preflight_one=preflight, run_one=Mock())
        self.assertEqual((len(attempts), result), (MAXIMUM_NEW_LIVE_ATTEMPTS, "INCONCLUSIVE"))
        self.assertEqual(client.stop_calls, MAXIMUM_NEW_LIVE_ATTEMPTS)

    def test_no_bag_or_data_collection_dependency(self):
        source = (ROOT / "src/physicar_e2e/expert_speed_repeatability.py").read_text().lower()
        self.assertNotIn("rosbag", source)
        self.assertNotIn("mcap", source)
        self.assertNotIn("dataset", source)


if __name__ == "__main__": unittest.main()
