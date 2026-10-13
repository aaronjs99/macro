import unittest

import numpy as np

from macro.attitude import tracking_operator, tracking_update


class AttitudeTrackingTests(unittest.TestCase):
    def gains(self):
        gains = np.full((2, 3, 3), 0.2)
        gains[:, -1] = 0.1
        return gains

    def test_each_axis_mixes_vector_errors_and_tracks_its_own_target(self):
        operator = tracking_operator({0: [1], 1: [0]}, self.gains(), 1)
        previous = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        result = tracking_update(previous, np.zeros((2, 3)), operator)
        np.testing.assert_allclose(result, [[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]])

    def test_desired_formation_is_a_fixed_point(self):
        desired = np.array([[1.0, -2.0, 0.5], [-0.5, 3.0, 2.0]])
        operator = tracking_operator({0: [1], 1: [0]}, self.gains(), 1)
        np.testing.assert_array_equal(
            tracking_update(desired, desired, operator), desired
        )

    def test_large_timesteps_are_subdivided_and_do_not_amplify_errors(self):
        operator = tracking_operator({0: [1], 1: [0]}, self.gains(), 100)
        previous = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        result = tracking_update(previous, np.zeros((2, 3)), operator)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((operator >= 0).all())
        self.assertLess(np.abs(result).max(), 3)

    def test_an_isolated_spacecraft_still_tracks_its_reference(self):
        operator = tracking_operator({0: [], 1: []}, self.gains(), 1)
        result = tracking_update(np.ones((2, 3)), np.zeros((2, 3)), operator)
        np.testing.assert_allclose(result, np.full((2, 3), 0.9))


if __name__ == "__main__":
    unittest.main()
