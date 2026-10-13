# /***********************************************************
# * Copyright (c) 2025                                      *
# * Indian Institute of Technology, Bombay                  *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# ***********************************************************/

from macro.config import load_config, resolve_config_path, resolve_output_dir
from macro.results import create_run_directory, save_results, write_status
from macro.attitude import tracking_operator, tracking_update

import numpy as np
import matplotlib.pyplot as mplplt
import plotly.graph_objects as go
from scipy.optimize import linear_sum_assignment

from macro.ifmea import IFMEAEngine
from macro.mirror import (
    generate_target_positions,
    calc_number_of_layers,
    compute_desired_attitudes,
)
from macro.utils import (
    initialize_poses,
    consensus_step,
    to_layer_index,
    to_flat_index,
    NullLogger,
)
from macro.agent import Agent, OrbitalParams, SimulationParams
from macro.glideslope import GlideslopeSimulator


class Macro:
    def __init__(self, config_path=None, logger=None, *, output_dir=None):
        self.config_path = resolve_config_path(config_path)
        self.config = load_config(self.config_path)
        self.output_root = resolve_output_dir(
            self.config, self.config_path, output_dir, create=False
        )
        self.output_dir = self.output_root
        self.logger = logger if logger is not None else NullLogger()
        self._init_params()
        self.ifmea_engine = IFMEAEngine(logger=self.logger)
        self._plots = True

    def _init_params(self):
        rng = np.random.RandomState(self.config["simulation"]["seed"])

        self.num_agents = self.config["formation"]["num_agents"]
        self.num_jumps = self.config["formation"]["num_jumps"]
        self.num_frames = self.config["simulation"]["num_frames"]
        self.timestep = self.config["simulation"]["timestep"]

        self.radius = self.config["orbit"]["radius"]
        self.gravpar = self.config["orbit"]["gravpar"]
        self.eccentricity = self.config["orbit"]["eccentricity"]
        self.ang_vel = np.sqrt(self.gravpar / self.radius**3)
        self.ang_mom = self.radius**2 * self.ang_vel

        self.clear_aperture = self.config["formation"]["clear_aperture"]
        self.parab_radius = self.config["formation"]["parab_radius"]
        self.expansion = self.config["formation"]["expansion"]

        self.neighbors = np.zeros((self.num_agents, self.num_agents), dtype=int)
        self.gains = np.full(
            (self.num_agents + 1, self.num_agents + 1, 3),
            [0.5 * 4e-6, 0.5 * 8e-6, 0.5 * 8e-6],
        )
        self.num_layers, self.layer_lengths = calc_number_of_layers(self.num_agents)

        self.initial_velocities = np.zeros((self.num_agents, 3))
        self.initial_positions, self.initial_attitudes = initialize_poses(
            self.num_agents, 5.00, rng=rng
        )
        self.formation_positions, _ = initialize_poses(self.num_agents, 0.01, rng=rng)
        self.unordered_target_positions = generate_target_positions(
            self.clear_aperture,
            self.parab_radius,
            self.num_agents,
            self.layer_lengths,
            self.expansion,
        )
        self.target_positions = self.unordered_target_positions.copy()

        self.stage_changes = {
            "1_rendezvous": None,
            "2_formation": None,
            "3_ifmea": None,
        }
        self.trajectory = np.empty((self.num_agents, 0, 6))
        self.attitudes = np.empty((self.num_agents, 0, 3))
        self.all_neighbors, self.all_flat_config = [], []
        self.phase_summaries = []
        self.phase_initial_states = []
        self.ifmea_commands = []
        self.summary = None

        self.true_mapping = [
            list(np.full(length, i + 1)) for i, length in enumerate(self.layer_lengths)
        ]
        self.flat_config = np.arange(self.num_agents)
        self.pre_ifmea_config = [[] for _ in self.layer_lengths]

    def _plot_trajectory(self, title, stage_name):
        if not self._plots:
            return
        fig = go.Figure()
        stage_keys = list(self.stage_changes.keys())
        stage_index = stage_keys.index(stage_name)

        # Get start index for this stage
        start_idx = self.stage_changes[stage_name]

        # Get end index: next stage start or end of trajectory
        if stage_index + 1 < len(stage_keys):
            next_stage = stage_keys[stage_index + 1]
            end_idx = self.stage_changes[next_stage]
        else:
            end_idx = self.trajectory.shape[1]

        for agent in range(self.num_agents):
            fig.add_trace(
                go.Scatter3d(
                    x=self.trajectory[agent, start_idx:end_idx, 0],
                    y=self.trajectory[agent, start_idx:end_idx, 1],
                    z=self.trajectory[agent, start_idx:end_idx, 2],
                    mode="lines",
                    line=dict(width=3),
                    name=f"S/c {agent + 1}",
                )
            )
        fig.update_layout(
            scene=dict(
                xaxis_title="X [km]",
                yaxis_title="Y [km]",
                zaxis_title="Z [km]",
                aspectmode="data",
            ),
            legend_title_text="Spacecraft",
            margin=dict(l=0, r=0, b=0, t=30),
            title_text=f"3D Trajectories: {stage_name.replace('_', ' ').title()}",
            template="plotly_dark",
        )
        fig.write_html(str(title), auto_open=False)

    def _plot_attitude(self):
        if not self._plots:
            return
        labels = ["Roll", "Pitch", "Yaw"]
        fig, axes = mplplt.subplots(
            3, 1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        for dim, ax in enumerate(axes):
            for agent in range(self.num_agents):
                ax.plot(
                    np.arange(1, self.attitudes.shape[1] + 1) * self.timestep,
                    self.attitudes[agent, :, dim],
                    label=f"Agent {agent + 1}",
                )
            ax.set_ylabel(f"{labels[dim]} [rad]")
            ax.grid(True, linestyle="--", alpha=0.6)
        axes[-1].set_xlabel("Time since IFMEA began [s]")
        fig.suptitle(
            f"Attitude histories ({self.config['attitude']['mode']} controller)"
        )
        fig.savefig(self.output_dir / "attitude_consensus.png", dpi=300)
        mplplt.close(fig)

    def _simulate_phase(self, start, end, stage_name, auction_type, plot=True):
        velocities = (
            self.trajectory[:, -1, 3:].copy()
            if self.trajectory.shape[1]
            else self.initial_velocities.copy()
        )
        self.phase_initial_states.append(np.concatenate((start, velocities), axis=1))
        agents = [
            Agent(
                agent_id=agent,
                agent_type=1,
                position=start[agent],
                velocity=velocities[agent],
                target=end[agent],
            )
            for agent in range(self.num_agents)
        ]
        orbit_params = OrbitalParams(
            radius=self.radius,
            eccentricity=self.eccentricity,
            angular_momentum=self.ang_mom,
            angular_velocity=self.ang_vel,
        )
        sim_params = SimulationParams(
            num_jumps=self.num_jumps,
            timestep=self.timestep,
            num_frames=self.num_frames,
            auction_type=auction_type,
            layer_config=self.true_mapping,
            flat_config=self.flat_config,
            auction_final_only=self.config["auction"]["final_only"],
        )
        sim = GlideslopeSimulator(
            agents=agents,
            neighbor_matrix=self.neighbors,
            neighbor_radius=self.config["formation"]["neighbor_radius"],
            target_positions=end.copy(),
            orbit=orbit_params,
            sim=sim_params,
        )
        traj, _, _, total_delta_v = sim.run()
        traj = np.array(traj)
        if not np.isfinite(traj).all():
            raise RuntimeError(f"{stage_name} produced nonfinite trajectory values")
        start_frame = self.trajectory.shape[1]
        self.trajectory = np.concatenate((self.trajectory, traj), axis=1)
        self.phase_summaries.append(
            {
                "stage": stage_name,
                "start_frame": start_frame,
                "end_frame": self.trajectory.shape[1],
                "total_delta_v_km_s": float(total_delta_v),
                "max_target_error_km": float(
                    np.linalg.norm(traj[:, -1, :3] - sim.target_positions, axis=1).max()
                ),
            }
        )

        if plot:
            self._plot_trajectory(self.output_dir / f"{stage_name}.html", stage_name)
        return sim.get_last_system_graph()

    def _generate_ifmea_config(self):
        distances = np.linalg.norm(
            self.trajectory[:, -1, None, :3]
            - self.unordered_target_positions[None, :, :],
            axis=2,
        )
        agents, slots = linear_sum_assignment(distances)
        if distances[agents, slots].max() > 1e-7:
            raise RuntimeError(
                "Formation did not reach a unique reflector position for every spacecraft"
            )
        self.flat_config = np.empty(self.num_agents, dtype=int)
        self.flat_config[slots] = agents
        self.pre_ifmea_config = [[] for _ in self.layer_lengths]
        for slot, agent in enumerate(self.flat_config):
            layer_idx, _ = to_layer_index(self.layer_lengths, slot)
            true_layer, _ = to_layer_index(self.layer_lengths, int(agent))
            self.pre_ifmea_config[layer_idx].append(true_layer + 1)

    def _intra_formation_exchange(self):
        def apply_rotation(layer_idx, shift):
            start_idx = sum(self.layer_lengths[:layer_idx])
            end_idx = start_idx + self.layer_lengths[layer_idx]
            block = self.flat_config[start_idx:end_idx]
            self.flat_config[start_idx:end_idx] = np.roll(block, shift)

        def apply_exchange(index_triplet):
            agents = [
                to_flat_index(self.layer_lengths, idx[0], idx[1])
                for idx in index_triplet
            ]
            (
                self.flat_config[agents[0]],
                self.flat_config[agents[1]],
                self.flat_config[agents[2]],
            ) = (
                self.flat_config[agents[2]],
                self.flat_config[agents[0]],
                self.flat_config[agents[1]],
            )

        target_positions = self.trajectory[:, -1, :3].copy()

        self.ifmea_engine.set_layer_config(self.pre_ifmea_config)
        commands = self.ifmea_engine.run()
        self.ifmea_commands = list(commands)

        for command in commands:
            cmd_type = command[0]

            if cmd_type == "R":
                layer_idx, shift = command[1], command[2]
                apply_rotation(layer_idx, shift)
            elif cmd_type == "E":
                index_triplet = command[1:4]
                apply_exchange(index_triplet)
            self.all_flat_config.append(self.flat_config.copy())

            intermediate_ifmea_positions = self.trajectory[:, -1, :3].copy()
            for idx, agent in enumerate(self.flat_config):
                target_positions[agent] = self.unordered_target_positions[idx]

            system_graph = self._simulate_phase(
                start=intermediate_ifmea_positions,
                end=target_positions,
                stage_name="3_ifmea",
                auction_type=None,
                plot=False,
            )
            self.all_neighbors.append(system_graph.get_all_neighbors())

        if not commands:
            # A correctly assembled formation still needs a full holding/attitude interval.
            for slot, agent in enumerate(self.flat_config):
                target_positions[agent] = self.unordered_target_positions[slot]
            self.all_flat_config.append(self.flat_config.copy())
            system_graph = self._simulate_phase(
                self.trajectory[:, -1, :3].copy(),
                target_positions,
                "3_ifmea",
                None,
                plot=False,
            )
            self.all_neighbors.append(system_graph.get_all_neighbors())

        self.target_positions = target_positions.copy()
        self._plot_trajectory(self.output_dir / "3_ifmea.html", "3_ifmea")
        return self.all_neighbors, self.all_flat_config

    def _attitude_consensus(self):
        num_steps = self.trajectory.shape[1] - self.stage_changes["3_ifmea"]
        self.attitudes = np.zeros((self.num_agents, num_steps, 3), dtype=float)

        # Compute the target/desired attitude
        desired_attitudes = compute_desired_attitudes(
            self.clear_aperture,
            self.parab_radius,
            self.layer_lengths,
            self.flat_config,
        )

        self.desired_attitudes = desired_attitudes[: self.num_agents].copy()
        tracking = self.config["attitude"]["mode"] == "tracking"
        operators = (
            [
                tracking_operator(
                    neighbors, self.gains[: self.num_agents], self.timestep
                )
                for neighbors in self.all_neighbors
            ]
            if tracking
            else []
        )
        # Record the same interval endpoints as the trajectory history.
        for frame in range(num_steps):
            previous = (
                self.initial_attitudes
                if frame == 0
                else self.attitudes[:, frame - 1, :]
            )
            neighbors = self.all_neighbors[
                min(frame // self.num_frames, len(self.all_neighbors) - 1)
            ]
            if tracking:
                operator = operators[min(frame // self.num_frames, len(operators) - 1)]
                self.attitudes[:, frame] = tracking_update(
                    previous, self.desired_attitudes, operator
                )
                continue
            for agent_id in range(self.num_agents):
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        self.attitudes[agent_id, frame, :] = consensus_step(
                            agent_id=agent_id,
                            neighbors=neighbors.get(agent_id, []),
                            previous=previous,
                            desired=desired_attitudes,
                            gains=self.gains[agent_id],
                            timestep=self.timestep,
                            count=frame + 1,
                        )
                except FloatingPointError as exc:
                    raise RuntimeError(
                        "Legacy attitude update became unstable; select attitude.mode: tracking"
                    ) from exc
        if not np.isfinite(self.attitudes).all():
            raise RuntimeError("Attitude update produced nonfinite values")
        self._plot_attitude()

    def run(self, *, plots=True):
        """Run from the configured initial state and save a separate result directory."""
        self._init_params()
        self._plots = plots
        self.output_dir = create_run_directory(self.output_root, self.config)
        try:
            self._run_stages()
            self.summary = save_results(self)
        except (Exception, KeyboardInterrupt) as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            write_status(
                self.output_dir,
                {
                    "status": status,
                    "error": str(exc),
                    "phases_completed": len(self.phase_summaries),
                },
            )
            raise
        self.logger.info(f"Results saved to {self.output_dir}")
        return self.summary

    def _run_stages(self):
        auction_type = (
            self.config["auction"]["mode"] if self.config["auction"]["enable"] else None
        )
        self.stage_changes["1_rendezvous"] = 0
        self._simulate_phase(
            start=self.initial_positions,
            end=self.formation_positions,
            stage_name="1_rendezvous",
            auction_type=auction_type,
        )
        self.logger.info(
            f"Rendezvous completed. Trajectory shape: {self.trajectory.shape}"
        )

        self.stage_changes["2_formation"] = self.trajectory.shape[1]
        self._simulate_phase(
            start=self.trajectory[:, -1, :3].copy(),
            end=self.unordered_target_positions,
            stage_name="2_formation",
            auction_type=auction_type,
        )
        self.logger.info(
            f"Formation completed. Trajectory shape: {self.trajectory.shape}"
        )

        self.stage_changes["3_ifmea"] = self.trajectory.shape[1]
        self._generate_ifmea_config()
        self._intra_formation_exchange()
        self.logger.info(f"IFMEA completed. Trajectory shape: {self.trajectory.shape}")

        self._attitude_consensus()
        self.logger.info(
            "Attitude integration completed; see the report for residual errors."
        )


if __name__ == "__main__":
    raise NotImplementedError(
        "This script is not intended to be run directly. Use the run.py script instead."
    )
