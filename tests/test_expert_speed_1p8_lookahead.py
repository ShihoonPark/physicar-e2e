from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet_inference import sha256_file
from physicar_e2e.expert_speed_lookahead_validation import (
    EXPECTED_CANONICAL_CONFIG_SHA256, EXPECTED_RESULT_DIRECTORY,
    LOOKAHEAD_CANDIDATES_M, MAXIMUM_LIVE_RUNS, SPEED_MPS,
    LookaheadSweepConfig, run_sweep,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/expert_speed_1p8_lookahead_v1.json"
CANONICAL = ROOT / "configs/expert_driver_v1.json"


def metrics(result="FAIL", failure="sustained off-track"):
    return {"result": result, "failure": None if result == "PASS" else failure,
            "safe_stop_success": True, "api_failures": 0,
            "pose_liveness_failures": 0, "clock_liveness_failures": 0}


class Client:
    def __init__(self): self.stop_calls = 0
    def safe_stop(self): self.stop_calls += 1; return []


class LookaheadSweepTests(unittest.TestCase):
    def setUp(self): self.config = LookaheadSweepConfig.load(CONFIG, CANONICAL)

    def test_candidates_are_exact_and_ascending(self):
        self.assertEqual(LOOKAHEAD_CANDIDATES_M, [0.60, 0.75, 0.90])
        self.assertEqual(LOOKAHEAD_CANDIDATES_M, sorted(LOOKAHEAD_CANDIDATES_M))

    def test_fixed_contract_and_canonical_lookahead_untouched(self):
        self.assertEqual(sha256_file(CANONICAL), EXPECTED_CANONICAL_CONFIG_SHA256)
        self.assertEqual(self.config.canonical.lookahead_m, 0.45)
        for candidate in LOOKAHEAD_CANDIDATES_M:
            driver = self.config.candidate(candidate).driver
            self.assertEqual(driver.fixed_speed_mps, SPEED_MPS)
            self.assertEqual(driver.control_frequency_hz, 15.0)
            self.assertEqual(driver.max_steering_rad, 0.349066)

    def execute(self, outcomes):
        run = Mock(side_effect=outcomes)
        preflight = Mock(return_value=(object(), {"result": "PASS"}))
        with tempfile.TemporaryDirectory() as directory:
            result = run_sweep(Client(), self.config, Path("/sim"), Path(directory),
                               preflight_one=preflight, run_one=run)
        return result, run, preflight

    def test_stops_after_first_pass(self):
        (attempts, result, passing), run, _ = self.execute([metrics("PASS"), metrics("PASS")])
        self.assertEqual((len(attempts), result, passing), (1, "PASS", 0.60))
        self.assertEqual(run.call_count, 1)

    def test_failures_continue_but_maximum_is_three(self):
        (attempts, result, passing), run, _ = self.execute([metrics(), metrics(), metrics(), metrics()])
        self.assertEqual((len(attempts), result, passing), (MAXIMUM_LIVE_RUNS, "FAIL", None))
        self.assertEqual(run.call_count, 3)
        self.assertEqual([item["lookahead_m"] for item in attempts], LOOKAHEAD_CANDIDATES_M)

    def test_infrastructure_stops_without_retry_and_safe_stops(self):
        client = Client()
        preflight = Mock(side_effect=RuntimeError("clock unavailable"))
        with tempfile.TemporaryDirectory() as directory:
            attempts, result, passing = run_sweep(client, self.config, Path("/sim"), Path(directory),
                                                  preflight_one=preflight, run_one=Mock())
        self.assertEqual((len(attempts), result, passing), (1, "INCONCLUSIVE", None))
        self.assertEqual(client.stop_calls, 1)

    def test_result_isolation(self):
        self.assertEqual(self.config.payload["result_directory"], EXPECTED_RESULT_DIRECTORY)


if __name__ == "__main__": unittest.main()
