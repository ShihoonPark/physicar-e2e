import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet_v4_repeatability import (
    aggregate_three_valid, classify_policy_run, clock_health_preflight, run_bounded,
)


def policy_run(result="PASS", failure=None):
    return {
        "result": result, "failure": failure, "api_failures": 0, "liveness_failures": 0,
        "safe_stop_success": True, "elapsed_s": 60.0, "mean_cte_m": 0.02, "max_cte_m": 0.10,
        "steering_saturation_fraction": 0.05, "control_loop_period": {"p95_ms": 67.0, "max_ms": 70.0},
    }


class Clock:
    def __init__(self, values): self.values = iter(values)
    def clock(self): return {"sim_time": next(self.values)}


class FakeTime:
    def __init__(self): self.now = 0.0
    def monotonic(self): return self.now
    def sleep(self, duration): self.now += duration


class ClassificationTests(unittest.TestCase):
    def test_policy_pass(self): self.assertEqual(classify_policy_run(policy_run()), "POLICY_PASS")
    def test_policy_fail(self): self.assertEqual(classify_policy_run(policy_run("FAIL", "sustained off-track")), "POLICY_FAIL")
    def test_infra_fail(self):
        run = policy_run("FAIL", "simulator clock did not advance"); run["liveness_failures"] = 1
        self.assertEqual(classify_policy_run(run), "INFRA_FAIL")

    def test_clock_health_requires_advancing_clock(self):
        fake = FakeTime(); values = [index * 0.05 for index in range(5)]
        result = clock_health_preflight(Clock(values), duration_s=0.2, interval_s=0.05, monotonic=fake.monotonic, sleep=fake.sleep)
        self.assertEqual(result["result"], "PASS")
        fake = FakeTime(); result = clock_health_preflight(Clock([1.0] * 5), duration_s=0.2, interval_s=0.05, max_stall_s=0.15, monotonic=fake.monotonic, sleep=fake.sleep)
        self.assertEqual(result["result"], "FAIL")


class BoundedTests(unittest.TestCase):
    def setUp(self):
        self.historical = policy_run()
        self.clock = {"result": "PASS"}; self.environment = {"result": "PASS"}

    def execute(self, outcomes):
        with (tempfile.TemporaryDirectory() as directory,
              patch("physicar_e2e.pilotnet_v4_repeatability.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_v4_repeatability.verify_static_environment", return_value=self.environment),
              patch("physicar_e2e.pilotnet_v4_repeatability.clock_health_preflight", return_value=self.clock),
              patch("physicar_e2e.pilotnet_v4_repeatability.run_smoke", side_effect=outcomes) as smoke):
            attempts, result, aggregate = run_bounded(object(), object(), type("C", (), {"safety_config": lambda self, speed: object()})(), Path(directory), self.historical)
        return attempts, result, aggregate, smoke.call_count

    def test_target_is_two_new_valid_runs_and_historical_once(self):
        attempts, result, aggregate, count = self.execute([policy_run(), policy_run()])
        self.assertEqual((result, count, len(attempts)), ("PASS", 2, 2)); self.assertEqual(aggregate["policy_success"], "3/3")
        self.assertEqual(aggregate["historical_full_lap_included_count"], 1)

    def test_policy_failure_stops_retries(self):
        attempts, result, _, count = self.execute([policy_run("FAIL", "sustained off-track"), policy_run()])
        self.assertEqual((result, count, len(attempts)), ("FAIL", 1, 1))

    def test_infra_failure_may_retry_and_is_excluded_from_aggregate(self):
        infra = policy_run("FAIL", "simulator clock did not advance"); infra["liveness_failures"] = 1
        attempts, result, aggregate, count = self.execute([infra, policy_run(), policy_run()])
        self.assertEqual((result, count, len(attempts)), ("PASS", 3, 3)); self.assertEqual(aggregate["policy_success"], "3/3")

    def test_maximum_four_attempts(self):
        infra = policy_run("FAIL", "simulator clock did not advance"); infra["liveness_failures"] = 1
        attempts, result, aggregate, count = self.execute([infra] * 4)
        self.assertEqual((result, count, len(attempts)), ("INCONCLUSIVE", 4, 4)); self.assertIsNone(aggregate)

    def test_safe_stop_failure_is_infra(self):
        run = policy_run("FAIL", "safe stop failed"); run["safe_stop_success"] = False
        self.assertEqual(classify_policy_run(run), "INFRA_FAIL")


if __name__ == "__main__": unittest.main()
