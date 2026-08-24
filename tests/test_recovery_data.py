import copy
import json
import math
import unittest
from pathlib import Path

import _bootstrap  # noqa: F401
from physicar_e2e.recovery_data import (
    RecoveryCompletionGate,
    RecoveryGateFailure,
    build_plan,
    check_recovery_limits,
    episode_matrix,
    run_fail_fast_sequence,
    select_anchors,
    tangent_at,
)
from physicar_e2e.route_geometry import ClosedRoute


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs" / "recovery_data_v1.json").read_text())


def square_route():
    center = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    inner = [(1, 1), (9, 1), (9, 9), (1, 9), (1, 1)]
    outer = [(-1, -1), (11, -1), (11, 11), (-1, 11), (-1, -1)]
    return ClosedRoute(center, inner, outer)


class RecoveryGeometryTests(unittest.TestCase):
    def test_tangent_and_left_lateral_sign(self):
        route = square_route()
        cfg = config()
        cfg["minimum_anchor_separation_m"] = 5.0
        anchors = select_anchors(route, cfg)
        failure = anchors[0]
        self.assertAlmostEqual(tangent_at(route, 2.95), 0.0, places=6)
        episodes = episode_matrix(route, anchors, cfg)
        positive = next(item for item in episodes if item.episode_id == "recovery_failure_lat_p10")
        negative = next(item for item in episodes if item.episode_id == "recovery_failure_lat_m10")
        self.assertGreater(positive.y, failure.y)
        self.assertLess(negative.y, failure.y)

    def test_yaw_offset_sign(self):
        route = square_route()
        anchors = select_anchors(route, config())
        episodes = episode_matrix(route, anchors, config())
        plus = next(item for item in episodes if item.episode_id == "recovery_failure_yaw_p06")
        minus = next(item for item in episodes if item.episode_id == "recovery_failure_yaw_m06")
        self.assertAlmostEqual(plus.yaw_rad - anchors[0].yaw_rad, math.radians(6), places=6)
        self.assertAlmostEqual(minus.yaw_rad - anchors[0].yaw_rad, -math.radians(6), places=6)

    def test_anchor_selection_is_three_and_separated(self):
        route = square_route()
        cfg = config()
        anchors = select_anchors(route, cfg)
        self.assertEqual([item.role for item in anchors], ["failure", "curvature_near", "curvature_far"])
        for left in range(3):
            for right in range(left + 1, 3):
                direct = abs(anchors[left].s_m - anchors[right].s_m)
                distance = min(direct, route.length - direct)
                self.assertGreaterEqual(distance, cfg["minimum_anchor_separation_m"])

    def test_exact_episode_matrix_and_metadata(self):
        route = square_route()
        plan = build_plan(route, config(), "unit-test")
        self.assertEqual(len(plan["episodes"]), 12)
        self.assertEqual(len({item["episode_id"] for item in plan["episodes"]}), 12)
        self.assertEqual(sum(item["anchor_role"] == "failure" for item in plan["episodes"]), 4)
        self.assertIn("sign_convention", plan)

    def test_unsafe_exact_perturbation_is_rejected(self):
        route = square_route()
        cfg = config()
        anchors = select_anchors(route, cfg)
        cfg["minimum_boundary_clearance_m"] = 2.0
        with self.assertRaisesRegex(RecoveryGateFailure, "unsafe perturbation"):
            episode_matrix(route, anchors, cfg)


class RecoveryCompletionTests(unittest.TestCase):
    def test_continuous_timing_and_minimum_progress(self):
        gate = RecoveryCompletionGate(0.03, 0.05, 0.75, 1.0)
        self.assertFalse(gate.update(abs_cte_m=0.02, abs_heading_rad=0.02, progress_m=0.5, now=0.0))
        self.assertFalse(gate.update(abs_cte_m=0.02, abs_heading_rad=0.02, progress_m=0.9, now=1.0))
        self.assertTrue(gate.update(abs_cte_m=0.02, abs_heading_rad=0.02, progress_m=1.0, now=1.01))

    def test_convergence_timer_resets(self):
        gate = RecoveryCompletionGate(0.03, 0.05, 0.75, 1.0)
        gate.update(abs_cte_m=0.02, abs_heading_rad=0.02, progress_m=0.2, now=0.0)
        self.assertFalse(gate.update(abs_cte_m=0.04, abs_heading_rad=0.02, progress_m=0.5, now=0.8))
        self.assertFalse(gate.update(abs_cte_m=0.02, abs_heading_rad=0.02, progress_m=1.0, now=0.9))

    def test_timeout_and_progress_limits(self):
        with self.assertRaisesRegex(RecoveryGateFailure, "timeout"):
            check_recovery_limits(10.0, 1.0, 10.0, 4.0)
        with self.assertRaisesRegex(RecoveryGateFailure, "progress"):
            check_recovery_limits(2.0, 4.01, 10.0, 4.0)

    def test_fail_fast_sequence_has_no_retry(self):
        calls = []

        def operation(item):
            calls.append(item)
            return {"result": "FAIL" if item == 2 else "PASS"}

        results = run_fail_fast_sequence([1, 2, 3], operation)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
