import unittest

import _bootstrap  # noqa: F401
from physicar_e2e.route_geometry import ClosedRoute, OffTrackMonitor


def square_track() -> ClosedRoute:
    center = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5), (-0.5, -0.5)]
    inner = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]  # explicitly closed
    outer = [(-1, -1), (3, -1), (3, 3), (-1, 3)]  # implicitly closed
    return ClosedRoute(center, inner, outer)


class TrackBandTests(unittest.TestCase):
    def setUp(self):
        self.route = square_track()

    def test_straight_track_interior(self):
        self.assertEqual(self.route.track_boundary_distance((1.0, -0.5)), 0.0)

    def test_corner_track_interior(self):
        self.assertEqual(self.route.track_boundary_distance((2.5, 2.5)), 0.0)

    def test_beyond_outer_boundary_at_corner(self):
        self.assertAlmostEqual(
            self.route.track_boundary_distance((3.1, 3.1)),
            2 ** 0.5 * 0.1,
        )
        self.assertTrue(self.route.is_off_track((3.1, 3.1), 0.05))

    def test_beyond_inner_boundary_at_corner(self):
        self.assertAlmostEqual(self.route.track_boundary_distance((1.9, 1.9)), 0.1)
        self.assertTrue(self.route.is_off_track((1.9, 1.9), 0.05))

    def test_outside_within_margin_is_accepted(self):
        self.assertAlmostEqual(self.route.track_boundary_distance((3.03, 1.0)), 0.03)
        self.assertFalse(self.route.is_off_track((3.03, 1.0), 0.05))

    def test_outside_beyond_margin_is_off_track(self):
        self.assertAlmostEqual(self.route.track_boundary_distance((3.06, 1.0)), 0.06)
        self.assertTrue(self.route.is_off_track((3.06, 1.0), 0.05))

    def test_unusable_track_ring_is_rejected(self):
        center = [(0, 0), (1, 0), (1, 1), (0, 1)]
        bow_tie = [(-1, -1), (2, 2), (-1, 2), (2, -1)]
        inner = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
        with self.assertRaisesRegex(ValueError, "self-intersects"):
            ClosedRoute(center, inner, bow_tie)


class OffTrackGraceTests(unittest.TestCase):
    def test_transient_outside_does_not_fail(self):
        monitor = OffTrackMonitor(0.5)
        self.assertFalse(monitor.update(True, 1.0))
        self.assertFalse(monitor.update(False, 1.2))
        self.assertEqual(monitor.event_count, 1)
        self.assertAlmostEqual(monitor.total_duration_s, 0.2)

    def test_sustained_outside_fails_after_grace(self):
        monitor = OffTrackMonitor(0.5)
        self.assertFalse(monitor.update(True, 2.0))
        self.assertTrue(monitor.update(True, 2.5))


if __name__ == "__main__":
    unittest.main()
