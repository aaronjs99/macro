import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from macro import Macro
from test_config import scenario


class RunStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = scenario()
        config["formation"].update(num_agents=6, num_jumps=2)
        config["simulation"].update(num_frames=8, output_dir=str(self.root / "runs"))
        config["auction"]["enable"] = False
        path = self.root / "scenario.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.simulation = Macro(path)

    def status(self):
        return json.loads(
            (self.simulation.output_dir / "summary.json").read_text(encoding="utf-8")
        )

    def test_keyboard_interruption_records_interruption_and_no_complete_archive(self):
        with patch.object(
            self.simulation, "_run_stages", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.simulation.run(plots=False)
        self.assertEqual(self.status()["status"], "interrupted")
        self.assertFalse((self.simulation.output_dir / "simulation.npz").exists())
        self.assertIsNone(self.simulation.summary)

    def test_export_failure_is_not_reported_as_a_completed_run(self):
        with patch("macro.main.save_results", side_effect=OSError("Disk full")):
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.simulation.run(plots=False)
        summary = self.status()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["phases_completed"], 3)
        self.assertIn("Disk full", summary["error"])
        self.assertIsNone(self.simulation.summary)


if __name__ == "__main__":
    unittest.main()
