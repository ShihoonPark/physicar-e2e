import math
import unittest

import _bootstrap  # noqa: F401
from physicar_e2e.route_geometry import ClosedRoute, ProgressTracker


class ClosedRouteTests(unittest.TestCase):
    def setUp(self):
        self.route = ClosedRoute([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

    def test_cumulative_route_length(self):
        self.assertAlmostEqual(self.route.length, 4.0)
        self.assertEqual(self.route.cumulative, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_closed_loop_target_wraparound(self):
        self.assertEqual(self.route.point_at(4.25), self.route.point_at(0.25))
        self.assertAlmostEqual(self.route.point_at(3.5)[1], 0.5)

    def test_projection_onto_route(self):
        projection = self.route.project((0.4, -0.2))
        self.assertEqual(projection.segment_index, 0)
        self.assertAlmostEqual(projection.s, 0.4)
        self.assertAlmostEqual(projection.distance, 0.2)
        self.assertLess(projection.signed_error, 0.0)

    def test_target_lookup_by_arc_length(self):
        self.assertEqual(self.route.point_at(1.5), (1.0, 0.5))


class ProgressTests(unittest.TestCase):
    def test_start_does_not_complete_lap(self):
        tracker = ProgressTracker(10.0, 2.0)
        tracker.update(0.0)
        self.assertFalse(tracker.lap_complete(0.0, 0.3, 0.9))

    def test_forward_wrap_increases_unwrapped_progress(self):
        tracker = ProgressTracker(10.0, 2.0)
        tracker.update(9.0)
        tracker.update(9.8)
        tracker.update(0.4)
        self.assertAlmostEqual(tracker.unwrapped, 1.4)

    def test_lap_requires_progress_and_gate(self):
        tracker = ProgressTracker(10.0, 5.0)
        tracker.update(0.0)
        tracker.unwrapped = 9.2
        self.assertFalse(tracker.lap_complete(0.4, 0.3, 0.9))
        self.assertTrue(tracker.lap_complete(0.2, 0.3, 0.9))


if __name__ == "__main__":
    unittest.main()
