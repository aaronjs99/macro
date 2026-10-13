# MACRO: Megastructure Assembly via Collaborative Robots in Orbits

MACRO is the simulation code accompanying *Scalable Techniques for Autonomous Construction of a Paraboloidal Space Telescope in an Elliptic Orbit* by Aaron John Sabu and Dwaipayan Mukherjee. It models cooperative spacecraft maneuvers, formation assignment, intra-formation exchange, and attitude consensus for a distributed parabolic reflector.

The current runner builds on the paper's orbital equations and reflector geometry, with repaired assignment, maneuver timing, phase handoffs and exchange planning. It is a desktop simulation, not a flight-control system. Numerical results differ from earlier versions that dropped jump endpoints or lost assignments.

## Architecture

| Module | Role |
| --- | --- |
| `macro.agent` | Spacecraft, orbit, and simulation parameter models |
| `macro.maneuvers` | Orbital state-transition and maneuver calculations |
| `macro.glideslope` | Multi-agent maneuver simulation |
| `macro.auction` | Assignment strategies for target allocation |
| `macro.ifmea` | Intra-formation motion/exchange planning |
| `macro.mirror` | Parabolic reflector geometry and desired attitudes |
| `macro.graph` | Communication-graph representation |
| `macro.main` | Top-level MACRO simulation composition |
| `macro.config` | Shared YAML loading and scenario/output path resolution |
| `macro.attitude` | Bounded, per-axis attitude error tracking |
| `macro.results` | Unique run directories, numerical exports and offline reports |

`config/default.yaml` contains the 36-spacecraft, 800-frame scenario. `config/quick.yaml` uses 18 spacecraft and 40 frames per maneuver for a smaller complete run. Generated results are intentionally not versioned.

## Setup

Use either of these minimal environments. The pip requirements file installs this
package and uses the dependencies declared in `pyproject.toml`:

```bash
conda env create -f environment.yml
conda activate macro
```

or:

```bash
pip install -r requirements.txt
```

These instructions use the editable source checkout. The wheel currently contains the Python library; supply an explicit scenario path when using it without the repository's `config/` directory.

## Run

Start with the reduced scenario:

```bash
python run.py --quick
```

The runner prints a results directory. Open its `index.html` for the stage summary, interactive 3D trajectories and attitude histories. Plots are usable offline.

Run the full supplied scenario:

```bash
python run.py
```

Use another configuration file when needed:

```bash
python run.py --config path/to/scenario.yaml
```

Add `--verbose` to show diagnostic output.

Validate without creating files, choose a reproducible seed, or save numerical results without plots:

```bash
python run.py --quick --validate
python run.py --quick --seed 7 --output path/to/runs
python run.py --quick --no-plots
```

Every run creates a new `run-<UTC timestamp>-<unique suffix>` directory, preserving previous results. It contains `scenario.yaml`, `simulation.npz` and `summary.json`; plotted runs also include `index.html`, three stage HTML plots and `attitude_consensus.png`. A failed or interrupted run records that status rather than claiming completion. Constructing `Macro` and using `--validate` do not create outputs.

Load saved arrays without pickle:

```python
import numpy as np

with np.load("path/to/run/simulation.npz", allow_pickle=False) as data:
    positions = data["trajectory"][:, :, :3]  # spacecraft, sample, X/Y/Z in km
    times = data["trajectory_time_s"]
```

The archive also saves velocities, attitudes, desired states, phase initial states, layer assignments and communication history. The summary records endpoint errors, delta-v, controller mode, frame boundaries, configuration checksum and runtime versions.

Relative config paths and `--output` paths use the current working directory. Relative YAML output paths use
`config_path.parent.parent`, preserving the supplied `config/default.yaml` layout;
set an absolute `simulation.output_dir` when a scenario lives elsewhere. The
`BenchmarkRunner` also accepts `config_path` and an optional `output_dir` override.

## Scenario behavior

Spacecraft counts must fill complete concentric layers: 6, 18, 36, 60, 90, and so on. Frame counts must be positive and divisible by the positive number of jumps. Configuration validation reports the exact offending field for invalid values.

`auction.enable: false` keeps prescribed targets. Supported modes are `Standard`, `Distributed`, `Consensus`, `Greedy` and `Hybrid`. `auction.final_only: true` auctions targets only on the final jump of each maneuver; otherwise auctions run at every jump. Iterative strategies fail explicitly on nonconvergence. Distributed and Consensus require a connected graph; Hybrid uses its Greedy fallback when disconnected. Assignment does not establish collision-free trajectories or global optimality.

The supplied scenarios explicitly select `attitude.mode: tracking`. This controller mixes per-axis attitude errors with neighbors and anchors each spacecraft to its own desired attitude. Bounded integration substeps prevent a large timestep from amplifying errors through negative mixing weights. The historical norm-error update is available with `attitude.mode: legacy`; external scenarios without an attitude section retain that mode. The legacy law can diverge or overflow during long runs and reports an actionable failure. Reports show measured residuals for either controller; completing a run does not imply the attitude targets have been reached.

Positions use kilometers, velocities kilometers per second, time seconds, and Euler angles radians. Exactly one state is recorded at each interval endpoint, starting at one timestep. Initial states are saved separately. Stage frame ranges include their start and exclude their end. Positions and velocities carry across phases, while the historical orbital reference epoch restarts for each maneuver. The full runner uses ring-topology communication; `neighbor_radius` remains a legacy benchmark input.

## Verification

After the editable installation, run:

```bash
python -m unittest discover -s tests -v
python run.py --quick
python run.py --no-plots
```

To test an uninstalled Windows checkout, set `$env:PYTHONPATH = "$PWD/src"` before running unittest. The tests exercise assignments, endpoint timing, exchange saturation, controller stability, CLI failures, saved data and reproducible reruns. Optional development checks are `python -m ruff check src run.py tests` and `git diff --check`. Build a wheel with `python -m build --wheel`; the normal isolated build installs its own build tools.

## Citation

```bibtex
@inproceedings{sabu2021scalable,
  author    = {John-Sabu, Aaron and Mukherjee, Dwaipayan},
  title     = {Scalable Techniques for Autonomous Construction of a Paraboloidal Space Telescope in an Elliptic Orbit},
  booktitle = {2021 Seventh Indian Control Conference (ICC)},
  year      = {2021},
  pages     = {329-334},
  doi       = {10.1109/ICC54714.2021.9703189},
  url       = {https://ieeexplore.ieee.org/document/9703189}
}
```

The machine-readable citation is in [CITATION.cff](CITATION.cff).

## License

GNU General Public License v2. See [LICENSE](LICENSE).
