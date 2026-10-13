"""Shared scenario loading and output paths for simulations and benchmarks."""

import copy
import math
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
QUICK_CONFIG_PATH = DEFAULT_CONFIG_PATH.with_name("quick.yaml")


def resolve_config_path(config_path=None):
    return (
        Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    ).resolve()


def load_config(config_path=None):
    """Load a YAML scenario; relative input paths use the caller's working directory."""
    path = resolve_config_path(config_path)
    try:
        with path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"Invalid YAML in {path}{location}") from exc
    return validate_config(config)


def validate_config(config):
    """Return a validated scenario with optional defaults, without side effects."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    config = copy.deepcopy(config)
    required = {
        "orbit": ("radius", "gravpar", "eccentricity"),
        "simulation": ("num_frames", "timestep"),
        "formation": (
            "num_agents",
            "num_jumps",
            "neighbor_radius",
            "clear_aperture",
            "parab_radius",
            "expansion",
        ),
    }
    for section, keys in required.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a YAML mapping")
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"{section}.{key} is required")

    def number(section, key, *, positive=True, integer=False):
        value = config[section][key]
        expected = int if integer else (int, float)
        if (
            isinstance(value, bool)
            or not isinstance(value, expected)
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"{section}.{key} must be a finite {'integer' if integer else 'number'}"
            )
        if positive and value <= 0:
            raise ValueError(f"{section}.{key} must be greater than zero")
        return value

    for section, keys in {
        "orbit": ("radius", "gravpar"),
        "simulation": ("timestep",),
        "formation": ("clear_aperture", "parab_radius", "expansion"),
    }.items():
        for key in keys:
            number(section, key)
    eccentricity = number("orbit", "eccentricity", positive=False)
    if not 0 <= eccentricity < 1:
        raise ValueError("orbit.eccentricity must satisfy 0 <= eccentricity < 1")
    if number("formation", "neighbor_radius", positive=False) < 0:
        raise ValueError("formation.neighbor_radius must be nonnegative")
    frames = number("simulation", "num_frames", integer=True)
    jumps = number("formation", "num_jumps", integer=True)
    if frames % jumps:
        raise ValueError(
            "simulation.num_frames must be divisible by formation.num_jumps"
        )
    agents = number("formation", "num_agents", integer=True)
    layers = (math.isqrt(1 + 4 * (agents // 3)) - 1) // 2
    if 3 * layers * (layers + 1) != agents:
        raise ValueError(
            "formation.num_agents must fill complete layers: 6, 18, 36, 60, 90, ..."
        )
    simulation = config["simulation"]
    simulation.setdefault("seed", 0)
    seed = number("simulation", "seed", positive=False, integer=True)
    if not 0 <= seed < 2**32:
        raise ValueError("simulation.seed must be between 0 and 4294967295")
    simulation.setdefault("output_dir", "results")
    if (
        not isinstance(simulation["output_dir"], str)
        or not simulation["output_dir"].strip()
    ):
        raise ValueError("simulation.output_dir must be a nonempty path")
    auction = config.setdefault("auction", {})
    if not isinstance(auction, dict):
        raise ValueError("auction must be a YAML mapping")
    auction.setdefault("enable", True)
    auction.setdefault("mode", "Hybrid")
    auction.setdefault("final_only", False)
    for key in ("enable", "final_only"):
        if not isinstance(auction[key], bool):
            raise ValueError(f"auction.{key} must be true or false")
    # Import here to keep the accepted strategies in sync with the simulator.
    from macro.auction import auction_map

    if not isinstance(auction["mode"], str) or auction["mode"] not in auction_map:
        raise ValueError(f"auction.mode must be one of: {', '.join(auction_map)}")
    attitude = config.setdefault("attitude", {})
    if not isinstance(attitude, dict):
        raise ValueError("attitude must be a YAML mapping")
    # Existing external scenarios retain their historical controller unless selected explicitly.
    attitude.setdefault("mode", "legacy")
    if attitude["mode"] not in ("legacy", "tracking"):
        raise ValueError("attitude.mode must be legacy or tracking")
    return config


def resolve_output_dir(config, config_path=None, output_dir=None, *, create=True):
    """Resolve relative output paths against the scenario's parent project directory."""
    configured = config["simulation"].get("output_dir", "results")
    path = Path(configured if output_dir is None else output_dir)
    if not path.is_absolute():
        path = resolve_config_path(config_path).parent.parent / path
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
