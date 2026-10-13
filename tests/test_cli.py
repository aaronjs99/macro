import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from test_config import scenario


RUNNER = Path(__file__).resolve().parents[1] / "run.py"


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(RUNNER), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=45,
        )

    def test_validation_from_another_directory_does_not_create_output(self):
        result = self.run_cli("--quick", "--validate", "--output", "checked")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Valid scenario", result.stdout)
        self.assertFalse((self.root / "checked").exists())

    def test_bad_config_reports_an_actionable_error_without_traceback(self):
        config = scenario()
        config["formation"]["num_agents"] = 7
        path = self.root / "bad.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = self.run_cli("--config", str(path), "--validate")
        self.assertEqual(result.returncode, 2)
        self.assertIn("formation.num_agents", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse((self.root / "results").exists())

    def test_missing_file_reports_an_error_without_traceback(self):
        result = self.run_cli("--config", "missing.yaml")
        self.assertEqual(result.returncode, 2)
        self.assertIn("missing.yaml", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_headless_run_persists_arrays_and_honors_seed_and_output_overrides(self):
        result = self.run_cli(
            "--quick", "--no-plots", "--seed", "7", "--output", "runs"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        runs = list((self.root / "runs").glob("run-*"))
        self.assertEqual(len(runs), 1)
        summary = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["seed"], 7)
        self.assertEqual(summary["num_agents"], 18)
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["plots"])
        self.assertTrue((runs[0] / "simulation.npz").exists())
        self.assertEqual(list(runs[0].glob("*.html")), [])
        self.assertEqual(list(runs[0].glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
