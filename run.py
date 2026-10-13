"""Run a MACRO orbital-assembly simulation from a chosen configuration."""

import argparse
import logging
import sys
from pathlib import Path

# Support a source checkout as well as an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from macro import Macro
from macro.config import (
    DEFAULT_CONFIG_PATH,
    QUICK_CONFIG_PATH,
    load_config,
    validate_config,
)


def build_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("macro")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a MACRO orbital-assembly simulation"
    )
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML configuration file",
    )
    scenario.add_argument(
        "--quick",
        dest="config",
        action="store_const",
        const=QUICK_CONFIG_PATH,
        help="Run the supplied reduced 18-spacecraft scenario",
    )
    parser.set_defaults(config=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output", type=Path, help="Results root (relative to the current directory)"
    )
    parser.add_argument("--seed", type=int, help="Override the scenario's random seed")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Check the scenario without running or writing files",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Save numerical results and summary without HTML/PNG plots",
    )
    parser.add_argument("--verbose", action="store_true", help="Show diagnostic output")
    args = parser.parse_args(argv)
    logger = build_logger(args.verbose)
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["simulation"]["seed"] = args.seed
            config = validate_config(config)
        if args.validate:
            print(
                f"Valid scenario: {config['formation']['num_agents']} spacecraft, "
                f"{config['simulation']['num_frames']} frames per maneuver, "
                f"{config['formation']['num_jumps']} jumps, seed {config['simulation']['seed']}."
            )
            return 0
        simulation = Macro(
            config_path=args.config,
            logger=logger,
            output_dir=args.output.resolve() if args.output is not None else None,
        )
        simulation.config = config
        summary = simulation.run(plots=not args.no_plots)
        print(
            f"Run complete: {summary['num_agents']} spacecraft, {summary['trajectory_frames']} trajectory samples.\n"
            f"Results: {simulation.output_dir}"
        )
        return 0
    except KeyboardInterrupt:
        logger.error("Simulation interrupted.")
        return 130
    except Exception as exc:
        logger.error(str(exc), exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
