import math
import unittest

import _bootstrap  # noqa: F401
from physicar_e2e.route_geometry import pure_pursuit_steering


class PurePursuitTests(unittest.TestCase):
    def test_straight_target_is_zero(self):
        steering, curvature, distance = pure_pursuit_steering((0, 0), 0, (1, 0), 0.18, 0.349066)
        self.assertAlmostEqual(steering, 0.0)
        self.assertAlmostEqual(curvature, 0.0)
        self.assertAlmostEqual(distance, 1.0)

    def test_left_and_right_targets_have_correct_sign(self):
        left = pure_pursuit_steering((0, 0), 0, (1, 1), 0.18, 0.349066)[0]
        right = pure_pursuit_steering((0, 0), 0, (1, -1), 0.18, 0.349066)[0]
        self.assertGreater(left, 0)
        self.assertLess(right, 0)

    def test_steering_clamps(self):
        steering, curvature, distance = pure_pursuit_steering((0, 0), 0, (0.01, 1), 2.0, 0.349066)
        self.assertAlmostEqual(steering, 0.349066)
        self.assertTrue(all(math.isfinite(v) for v in (steering, curvature, distance)))


if __name__ == "__main__":
    unittest.main()
