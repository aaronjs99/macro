"""Scenario errors must be caught before a simulation or output is started."""

import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from macro.config import load_config, resolve_output_dir


def scenario():
    return {
        "orbit": {"radius": 7275, "gravpar": 398600.5, "eccentricity": 0.05},
        "simulation": {"seed": 42, "num_frames": 40, "timestep": 5.0},
        "formation": {
            "num_agents": 18,
            "num_jumps": 4,
            "neighbor_radius": 0.4,
            "clear_aperture": 0.002,
            "parab_radius": 0.003,
            "expansion": 3,
        },
        "auction": {"enable": True, "mode": "Hybrid", "final_only": False},
    }


class ScenarioValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config" / "scenario.yaml"
        self.path.parent.mkdir()

    def write(self, config):
        self.path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return self.path

    def test_valid_scenario_preserves_explicit_values(self):
        original = scenario()
        self.assertEqual(
            load_config(self.write(original))["formation"], original["formation"]
        )

    def test_invalid_fields_report_the_field_before_creating_output(self):
        cases = [
            ("orbit", "radius", 0),
            ("orbit", "gravpar", -1),
            ("orbit", "eccentricity", -0.1),
            ("orbit", "eccentricity", 1),
            ("orbit", "radius", float("nan")),
            ("simulation", "num_frames", 0),
            ("simulation", "num_frames", 41),
            ("simulation", "timestep", float("inf")),
            ("simulation", "timestep", True),
            ("simulation", "seed", -1),
            ("formation", "num_agents", 7),
            ("formation", "num_agents", 0),
            ("formation", "num_agents", True),
            ("formation", "num_jumps", 0),
            ("formation", "neighbor_radius", -1),
            ("formation", "clear_aperture", 0),
            ("formation", "parab_radius", 0),
            ("formation", "expansion", 0),
            ("auction", "enable", "false"),
            ("auction", "mode", "typo"),
            ("simulation", "output_dir", ""),
        ]
        for section, key, value in cases:
            with self.subTest(field=f"{section}.{key}", value=value):
                config = scenario()
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, rf"{section}\.{key}"):
                    load_config(self.write(config))
                self.assertFalse((Path(self.temp.name) / "results").exists())

    def test_missing_and_malformed_sections_are_actionable(self):
        for section in ("orbit", "simulation", "formation"):
            for value in (None, [], "invalid"):
                with self.subTest(section=section, value=value):
                    config = scenario()
                    config[section] = value
                    with self.assertRaisesRegex(ValueError, section):
                        load_config(self.write(config))
            config = scenario()
            del config[section]
            with self.assertRaisesRegex(ValueError, section):
                load_config(self.write(config))

    def test_optional_defaults_do_not_require_an_auction_section(self):
        config = scenario()
        del config["auction"]
        del config["simulation"]["seed"]
        loaded = load_config(self.write(config))
        self.assertEqual(loaded["simulation"]["seed"], 0)
        self.assertEqual(loaded["auction"]["mode"], "Hybrid")

    def test_malformed_yaml_is_reported_as_a_configuration_error(self):
        self.path.write_text("orbit: [", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "YAML"):
            load_config(self.path)

    def test_output_resolution_can_be_checked_without_creating_files(self):
        config = copy.deepcopy(scenario())
        config["simulation"]["output_dir"] = "results"
        output = resolve_output_dir(config, self.path, create=False)
        self.assertEqual(output, Path(self.temp.name) / "results")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
