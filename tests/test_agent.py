import unittest

import numpy as np

from macro.agent import Agent


class HoldingManeuverTests(unittest.TestCase):
    def test_an_agent_at_its_target_still_cancels_residual_velocity(self):
        agent = Agent(
            0,
            1,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, 0.0, 0.0]),
            target=np.array([1.0, 2.0, 3.0]),
        )
        agent.update_direction()
        burn = agent.compute_delta_v(0, 1, np.eye(3), 10 * np.eye(3))
        np.testing.assert_allclose(burn, [-0.1, 0, 0])
        np.testing.assert_allclose(agent.velocity, [0, 0, 0], atol=1e-12)

    def test_burn_does_not_mutate_caller_owned_velocity(self):
        velocity = np.zeros(3)
        agent = Agent(0, 1, np.zeros(3), velocity, target=np.ones(3))
        agent.update_direction()
        agent.compute_delta_v(0, 1, np.eye(3), np.eye(3))
        np.testing.assert_array_equal(velocity, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
