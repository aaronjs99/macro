"""Persist inspectable simulation runs without replacing earlier results."""

from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import yaml


def write_status(directory, summary):
    """Replace the status atomically, including on an interrupted or failed run."""
    path = Path(directory) / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def create_run_directory(root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-")
    directory = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    (directory / "scenario.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    write_status(directory, {"status": "running"})
    return directory


def save_results(sim):
    """Save finite arrays and a factual summary; completion never implies convergence."""
    n = sim.num_agents
    expected_types = np.repeat(np.arange(1, sim.num_layers + 1), sim.layer_lengths)
    if not np.array_equal(np.sort(sim.flat_config), np.arange(n)):
        raise RuntimeError("Final formation lost or duplicated spacecraft identities")
    if not np.array_equal(expected_types[sim.flat_config], expected_types):
        raise RuntimeError(
            "IFMEA did not put every spacecraft in its destination layer"
        )
    final_error = np.linalg.norm(
        sim.trajectory[:, -1, :3] - sim.target_positions, axis=1
    )
    if final_error.max() > 1e-7:
        raise RuntimeError("Final formation did not reach its prescribed targets")
    arrays = {
        "trajectory": sim.trajectory,
        "attitudes": sim.attitudes,
        "trajectory_time_s": np.arange(1, sim.trajectory.shape[1] + 1) * sim.timestep,
        "attitude_time_s": (
            sim.stage_changes["3_ifmea"] + np.arange(1, sim.attitudes.shape[1] + 1)
        )
        * sim.timestep,
        "initial_positions": sim.initial_positions,
        "initial_attitudes": sim.initial_attitudes,
        "target_positions": sim.target_positions,
        "desired_attitudes": sim.desired_attitudes,
        "flat_config": sim.flat_config,
        "layer_lengths": np.asarray(sim.layer_lengths),
        "phase_initial_states": np.asarray(sim.phase_initial_states),
        "phase_start_frames": np.asarray(
            [phase["start_frame"] for phase in sim.phase_summaries]
        ),
        "phase_end_frames": np.asarray(
            [phase["end_frame"] for phase in sim.phase_summaries]
        ),
        "ifmea_flat_configs": np.asarray(sim.all_flat_config),
    }
    adjacency = np.zeros((len(sim.all_neighbors), n, n), dtype=int)
    for phase, neighbors in enumerate(sim.all_neighbors):
        if set(neighbors) != set(range(n)):
            raise RuntimeError("Communication history does not cover every spacecraft")
        for agent, adjacent in neighbors.items():
            adjacency[phase, agent, adjacent] = 1
    arrays["ifmea_neighbor_adjacency"] = adjacency
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Cannot export nonfinite simulation data")
    np.savez_compressed(sim.output_dir / "simulation.npz", **arrays)
    scenario_text = (sim.output_dir / "scenario.yaml").read_bytes()
    summary = {
        "status": "complete",
        "run_id": sim.output_dir.name,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_agents": n,
        "seed": sim.config["simulation"]["seed"],
        "auction": sim.config["auction"],
        "attitude_controller": sim.config["attitude"]["mode"],
        "trajectory_frames": sim.trajectory.shape[1],
        "attitude_frames": sim.attitudes.shape[1],
        "timestep_s": sim.timestep,
        "sample_convention": "Interval endpoints; initial state at t=0 is saved separately",
        "orbital_epoch_convention": "The historical orbital reference epoch restarts at each maneuver phase",
        "stage_start_frames": sim.stage_changes,
        "phases": sim.phase_summaries,
        "ifmea_commands": sim.ifmea_commands,
        "final_max_target_error_km": float(final_error.max()),
        "initial_max_attitude_error_rad": float(
            np.linalg.norm(sim.initial_attitudes - sim.desired_attitudes, axis=1).max()
        ),
        "final_max_attitude_error_rad": float(
            np.linalg.norm(sim.attitudes[:, -1] - sim.desired_attitudes, axis=1).max()
        ),
        "total_delta_v_km_s": sum(
            phase["total_delta_v_km_s"] for phase in sim.phase_summaries
        ),
        "config_sha256": sha256(scenario_text).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "plots": sim._plots,
        "notes": [
            f"Attitude controller: {sim.config['attitude']['mode']}. Inspect residuals; finishing integration does not establish convergence.",
            "The communication graph uses the configured ring topology; neighbor_radius is retained for legacy benchmark inputs.",
        ],
    }
    if sim._plots:
        (sim.output_dir / "index.html").write_text(
            _render_report(summary), encoding="utf-8"
        )
    write_status(sim.output_dir, summary)
    return summary


def _render_report(summary):
    names = {
        "1_rendezvous": "Rendezvous",
        "2_formation": "Reflector formation",
        "3_ifmea": "Intra-formation exchange",
    }
    rows = []
    for stage, name in names.items():
        phases = [phase for phase in summary["phases"] if phase["stage"] == stage]
        rows.append(
            f'<tr><th scope="row"><a href="{stage}.html">{name}</a></th>'
            f"<td>{len(phases)}</td><td>{sum(p['end_frame'] - p['start_frame'] for p in phases):,}</td>"
            f"<td>{sum(p['total_delta_v_km_s'] for p in phases):.6g}</td>"
            f"<td>{1000 * max(p['max_target_error_km'] for p in phases):.3g}</td></tr>"
        )
    notes = "".join(f"<li>{escape(note)}</li>" for note in summary["notes"])
    auction = (
        summary["auction"]["mode"]
        if summary["auction"]["enable"]
        else "Disabled (fixed targets)"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>MACRO assembly run</title><style>
:root{{color-scheme:dark;--bg:#0a1420;--panel:#142333;--ink:#e7eef5;--muted:#adbdcd;--accent:#8de2d0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:40px 24px 56px}}h1{{font-size:clamp(2rem,5vw,3rem);line-height:1.15;margin:12px 0}}
h2{{font-size:1.35rem;margin:0 0 16px}}p{{margin:8px 0 20px;color:var(--muted)}}a{{color:var(--accent);text-underline-offset:4px}}
a:focus-visible{{outline:3px solid white;outline-offset:4px}}.eyebrow{{color:var(--accent);font-weight:650;letter-spacing:.08em;font-size:.8rem}}
.badge{{display:inline-block;color:var(--accent);border:1px solid #38685f;padding:4px 12px;border-radius:99px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:30px 0}}
.card,section{{background:var(--panel);border:1px solid #293d50;border-radius:12px;padding:22px;margin-bottom:24px}}
.card{{margin:0}}.card span{{display:block;color:var(--muted);font-size:.9rem}}.card strong{{font-size:1.8rem;font-weight:650}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}}
th,td{{padding:12px 16px;text-align:left;border-bottom:1px solid #304355}}thead th{{font-size:.8rem;color:var(--muted)}}
img{{max-width:100%;height:auto;background:white;border-radius:8px}}.files{{display:flex;flex-wrap:wrap;gap:12px 26px}}
code{{overflow-wrap:anywhere;font-size:.85rem;color:var(--muted)}}footer{{font-size:.9rem;color:var(--muted)}}
@media(max-width:600px){{main{{padding:24px 16px}}section,.card{{padding:16px}}th,td{{padding:10px}}}}
</style></head><body><main>
<div class="eyebrow">MACRO / ORBITAL ASSEMBLY SIMULATION</div><h1>Assembly run</h1>
<p>Cooperative spacecraft maneuvers, reflector formation and intra-formation exchange.</p>
<span class="badge">Simulation completed</span>
<div class="cards"><div class="card"><span>Spacecraft</span><strong>{summary["num_agents"]}</strong></div>
<div class="card"><span>Trajectory samples</span><strong>{summary["trajectory_frames"]:,}</strong></div>
<div class="card"><span>Exchange commands</span><strong>{len(summary["ifmea_commands"])}</strong></div>
<div class="card"><span>Final target error [m]</span><strong>{summary["final_max_target_error_km"] * 1000:.3g}</strong></div></div>
<section><h2>Maneuver stages</h2><p>Select a stage to rotate and inspect its 3D trajectories. Auction: {escape(auction)}. Seed: {summary["seed"]}.</p>
<div class="table-wrap"><table><thead><tr><th scope="col">Stage</th><th scope="col">Maneuvers</th><th scope="col">Samples</th>
<th scope="col">Total delta-v [km/s]</th><th scope="col">Max endpoint error [m]</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>
<section><h2>Attitude residual</h2><p>Largest spacecraft error: {summary["initial_max_attitude_error_rad"]:.4g} rad initially &rarr;
{summary["final_max_attitude_error_rad"]:.4g} rad finally. This measures the distance to the desired Euler-angle vector.</p>
<img src="attitude_consensus.png" alt="Roll, pitch and yaw histories for every spacecraft during the exchange stage">
<ul>{notes}</ul></section>
<section><h2>Inspect or reproduce this run</h2><p>Numerical arrays include trajectories, attitudes, times, phase initial states, final assignments and communication history.</p>
<div class="files"><a href="simulation.npz" download>Numerical results (.npz)</a><a href="summary.json">Run summary (.json)</a>
<a href="scenario.yaml" download>Resolved scenario (.yaml)</a></div>
<p>Samples occur at interval endpoints, starting at one timestep. Stage frame ranges use an inclusive start and exclusive end.</p>
<code>Configuration SHA-256: {summary["config_sha256"]}</code></section>
<footer>Run {escape(summary["run_id"])} &middot; {escape(summary["completed_at_utc"])} &middot; timestep {summary["timestep_s"]:g} s</footer>
</main></body></html>"""
