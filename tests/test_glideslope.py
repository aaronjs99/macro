"""Regression tests for prescribed maneuvers and phase sample times."""

import math
import unittest
from unittest.mock import patch

import numpy as np

from macro.agent import Agent, OrbitalParams, SimulationParams
from macro.glideslope import GlideslopeSimulator


class ReverseAssignment:
    """Give scheduling tests one predictable assignment, independent of strategy."""

    def __init__(self, initial_positions, target_positions, graph):
        self.target_positions = target_positions

    def assign(self):
        return self.target_positions[::-1].copy()


class GlideslopeTests(unittest.TestCase):
    def make_simulator(
        self,
        *,
        positions=None,
        targets=None,
        num_jumps=1,
        num_frames=4,
        timestep=10.0,
        auction_type=None,
        auction_final_only=False,
    ):
        if positions is None:
            positions = np.array([[0.0, 100.0, 0.0]])
        if targets is None:
            targets = np.array([[0.0, 20.0, 0.0]])
        agents = np.array(
            [
                Agent(i, 0, position.copy(), np.zeros(3))
                for i, position in enumerate(positions)
            ],
            dtype=object,
        )
        count = len(agents)
        params = SimulationParams(
            num_jumps=num_jumps,
            num_frames=num_frames,
            timestep=timestep,
            auction_type=auction_type,
            layer_config=[list(range(count))],
            flat_config=list(range(count)),
            auction_final_only=auction_final_only,
        )
        orbit = OrbitalParams(
            radius=7_000_000.0,
            eccentricity=0.0,
            angular_momentum=0.0,
            angular_velocity=0.01,
        )
        return GlideslopeSimulator(
            agents=agents,
            neighbor_matrix=np.zeros((count, count)),
            neighbor_radius=1.0,
            target_positions=targets.copy(),
            orbit=orbit,
            sim=params,
        )

    def test_no_auction_preserves_prescribed_target_and_applies_burn(self):
        simulator = self.make_simulator()
        _, _, deltavs, energy = simulator.run()

        np.testing.assert_array_equal(simulator.target_positions, [[0.0, 20.0, 0.0]])
        # Cross-track circular motion obeys y(t) = y0*cos(n*t) + vy0*sin(n*t)/n.
        expected_burn = 0.01 * (20.0 - 100.0 * math.cos(0.4)) / math.sin(0.4)
        np.testing.assert_allclose(deltavs, [0.0, expected_burn, 0.0], atol=1e-12)
        self.assertAlmostEqual(energy, abs(expected_burn), places=12)

    def test_single_jump_samples_interval_endpoints_and_reaches_target(self):
        simulator = self.make_simulator(auction_type="Standard")
        trajectory, graphs, _, _ = simulator.run()

        burn_velocity = 0.01 * (20.0 - 100.0 * math.cos(0.4)) / math.sin(0.4)
        sample_times = [10.0, 20.0, 30.0, 40.0]
        expected_y = [
            100.0 * math.cos(0.01 * t) + burn_velocity * math.sin(0.01 * t) / 0.01
            for t in sample_times
        ]
        expected_vy = [
            -100.0 * 0.01 * math.sin(0.01 * t) + burn_velocity * math.cos(0.01 * t)
            for t in sample_times
        ]
        self.assertEqual(trajectory.shape, (1, 4, 6))
        self.assertEqual(len(graphs), 1)
        np.testing.assert_allclose(trajectory[0, :, 1], expected_y, atol=1e-12)
        np.testing.assert_allclose(trajectory[0, :, 4], expected_vy, atol=1e-12)
        np.testing.assert_allclose(trajectory[0, -1, :3], [0.0, 20.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(
            simulator.agents[0].position, trajectory[0, -1, :3]
        )
        np.testing.assert_array_equal(
            simulator.agents[0].velocity, trajectory[0, -1, 3:]
        )

    def test_each_jump_starts_from_the_previous_full_endpoint(self):
        simulator = self.make_simulator(num_jumps=2, auction_type="Standard")
        trajectory, graphs, _, _ = simulator.run()

        # Two jumps first reach the prescribed midpoint (60 m), then 20 m.
        np.testing.assert_allclose(trajectory[0, [1, 3], 1], [60.0, 20.0], atol=1e-12)
        second_burn_velocity = 0.01 * (20.0 - 60.0 * math.cos(0.2)) / math.sin(0.2)
        expected_second_jump_first_y = (
            60.0 * math.cos(0.1) + second_burn_velocity * math.sin(0.1) / 0.01
        )
        self.assertAlmostEqual(
            trajectory[0, 2, 1], expected_second_jump_first_y, places=12
        )
        self.assertEqual(len(graphs), 2)

    def test_one_frame_per_jump_still_advances_to_the_target(self):
        simulator = self.make_simulator(
            num_jumps=2, num_frames=2, auction_type="Standard"
        )
        trajectory, _, _, _ = simulator.run()

        np.testing.assert_allclose(trajectory[0, :, 1], [60.0, 20.0], atol=1e-12)

    def test_final_only_auction_preserves_targets_before_the_final_jump(self):
        positions = np.array([[0.0, 100.0, 0.0], [0.0, 200.0, 0.0]])
        targets = np.array([[0.0, 20.0, 0.0], [0.0, 40.0, 0.0]])
        simulator = self.make_simulator(
            positions=positions,
            targets=targets,
            num_jumps=2,
            auction_type="TestReverse",
            auction_final_only=True,
        )
        with patch.dict(
            "macro.glideslope.auction_map", {"TestReverse": ReverseAssignment}
        ):
            trajectory, _, _, _ = simulator.run()

        np.testing.assert_allclose(
            trajectory[:, 1, :3],
            [[0.0, 60.0, 0.0], [0.0, 120.0, 0.0]],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            trajectory[:, -1, :3], [[0.0, 40.0, 0.0], [0.0, 20.0, 0.0]], atol=1e-10
        )

    def test_final_only_auction_runs_when_the_only_jump_is_final(self):
        positions = np.array([[0.0, 100.0, 0.0], [0.0, 200.0, 0.0]])
        targets = np.array([[0.0, 20.0, 0.0], [0.0, 40.0, 0.0]])
        simulator = self.make_simulator(
            positions=positions,
            targets=targets,
            auction_type="TestReverse",
            auction_final_only=True,
        )
        with patch.dict(
            "macro.glideslope.auction_map", {"TestReverse": ReverseAssignment}
        ):
            trajectory, _, _, _ = simulator.run()

        np.testing.assert_allclose(
            trajectory[:, -1, :3], [[0.0, 40.0, 0.0], [0.0, 20.0, 0.0]], atol=1e-10
        )

    def test_rejects_invalid_jump_counts(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(num_jumps=value):
                with self.assertRaisesRegex(ValueError, "num_jumps"):
                    self.make_simulator(num_jumps=value)

    def test_rejects_invalid_frame_counts(self):
        for value in (0, -1, 2.5, True):
            with self.subTest(num_frames=value):
                with self.assertRaisesRegex(ValueError, "num_frames"):
                    self.make_simulator(num_frames=value)

    def test_rejects_nonpositive_or_nonfinite_timestep(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), "10", True):
            with self.subTest(timestep=value):
                with self.assertRaisesRegex(ValueError, "timestep"):
                    self.make_simulator(timestep=value)

    def test_rejects_frame_counts_not_divisible_by_jump_count(self):
        with self.assertRaisesRegex(ValueError, "num_frames.*num_jumps"):
            self.make_simulator(num_jumps=3, num_frames=4)

    def test_rejected_reconfiguration_keeps_the_existing_simulation_parameters(self):
        simulator = self.make_simulator()
        with self.assertRaisesRegex(ValueError, "num_jumps"):
            simulator.set_simulation_params(
                SimulationParams(num_jumps=0, timestep=1.0, num_frames=4)
            )

        self.assertEqual(simulator.num_jumps, 1)
        self.assertEqual(simulator.timestep, 10.0)
        self.assertEqual(simulator.steps_per_jump, 4)


if __name__ == "__main__":
    unittest.main()
