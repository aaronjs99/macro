"""Exercise the real reduced simulation and its saved results."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from macro import Macro
from test_config import scenario


class SimulationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "config" / "scenario.yaml"
        self.path.parent.mkdir()
        self.config = scenario()
        self.config["formation"].update(num_agents=6, num_jumps=2)
        self.config["simulation"].update(
            num_frames=8, output_dir=str(self.root / "results")
        )
        self.config["auction"]["enable"] = False
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def test_constructing_a_simulation_does_not_create_output(self):
        Macro(self.path)
        self.assertFalse((self.root / "results").exists())

    def test_seeded_initialization_leaves_global_random_generator_alone(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        np.random.seed(1234)
        expected = np.random.RandomState(1234).random_sample(4)
        Macro(self.path)
        np.testing.assert_array_equal(np.random.random_sample(4), expected)

    def test_no_exchange_case_has_full_attitude_history_and_correct_boundaries(self):
        sim = Macro(self.path)
        sim.run()
        self.assertEqual(
            sim.stage_changes, {"1_rendezvous": 0, "2_formation": 8, "3_ifmea": 16}
        )
        self.assertEqual(sim.trajectory.shape, (6, 24, 6))
        self.assertEqual(sim.attitudes.shape, (6, 8, 3))
        self.assertTrue(np.isfinite(sim.trajectory).all())
        np.testing.assert_allclose(
            sim.trajectory[:, -1, :3], sim.target_positions, atol=1e-8
        )
        summary = json.loads(
            (sim.output_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["num_agents"], 6)
        with np.load(sim.output_dir / "simulation.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["trajectory"], sim.trajectory)
            np.testing.assert_array_equal(saved["flat_config"], np.arange(6))

    def test_rerun_is_reproducible_and_preserves_the_previous_run(self):
        sim = Macro(self.path)
        sim.run()
        first_dir = sim.output_dir
        first_trajectory = sim.trajectory.copy()
        sim.run()
        self.assertNotEqual(sim.output_dir, first_dir)
        self.assertTrue((first_dir / "simulation.npz").exists())
        np.testing.assert_array_equal(sim.trajectory, first_trajectory)

    def test_ifmea_receives_types_and_exchanges_preserve_agent_identities(self):
        self.config["formation"]["num_agents"] = 18
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        sim = Macro(self.path)
        slots = np.arange(18)
        slots[0], slots[6] = slots[6], slots[0]
        sim.trajectory = np.zeros((18, 1, 6))
        sim.trajectory[slots, 0, :3] = sim.unordered_target_positions
        sim.stage_changes["3_ifmea"] = 1
        sim._generate_ifmea_config()
        self.assertEqual(sim.pre_ifmea_config[0], [2, 1, 1, 1, 1, 1])
        sim.output_dir.mkdir(parents=True)
        sim._intra_formation_exchange()
        self.assertEqual(sorted(sim.flat_config.tolist()), list(range(18)))
        self.assertTrue(all(agent < 6 for agent in sim.flat_config[:6]))
        self.assertTrue(all(agent >= 6 for agent in sim.flat_config[6:]))
        np.testing.assert_allclose(
            sim.trajectory[:, -1, :3], sim.target_positions, atol=1e-8
        )

    def test_tracking_controller_reduces_error_through_real_exchange_history(self):
        self.config["formation"].update(num_agents=18, num_jumps=4)
        self.config["simulation"]["num_frames"] = 40
        self.config["auction"]["enable"] = True
        self.config["attitude"] = {"mode": "tracking"}
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        sim = Macro(self.path)
        summary = sim.run(plots=False)
        self.assertLess(
            summary["final_max_attitude_error_rad"],
            summary["initial_max_attitude_error_rad"],
        )


if __name__ == "__main__":
    unittest.main()
