# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * Indian Institute of Technology, Bombay                  *
# *                                                         *
# * Author(s): Aaron John Sabu, Dwaipayan Mukherjee         *
# * Contact  : aaronjs@g.ucla.edu, dm@ee.iitb.ac.in         *
# *                                                         *
# ***********************************************************/

import numpy as np
from dataclasses import dataclass, field


@dataclass
class OrbitalParams:
    radius: float
    eccentricity: float
    angular_momentum: float
    angular_velocity: float


@dataclass
class SimulationParams:
    num_jumps: int
    timestep: float
    num_frames: int
    auction_type: str = "Distributed"
    layer_config: list = field(default_factory=list)
    flat_config: list = field(default_factory=list)
    auction_final_only: bool = False


class Agent:
    """Represents a spacecraft agent in the glideslope simulation."""

    def __init__(
        self,
        agent_id: int,
        agent_type: int,
        position: np.ndarray,
        velocity: np.ndarray = None,
        target: np.ndarray = None,
        glideslope_dir: np.ndarray = None,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.position = np.array(position, dtype=float, copy=True)
        self.velocity = (
            np.zeros(3)
            if velocity is None
            else np.array(velocity, dtype=float, copy=True)
        )
        self.target = (
            np.zeros(3) if target is None else np.array(target, dtype=float, copy=True)
        )
        self.glideslope_dir = glideslope_dir

    def update_direction(self):
        delta = self.position - self.target
        norm = np.linalg.norm(delta)
        self.glideslope_dir = delta / norm if norm > 0 else np.zeros_like(delta)

    def compute_delta_v(
        self, jump: int, num_jumps: int, phi_r: np.ndarray, phi_v: np.ndarray
    ) -> np.ndarray:
        if self.glideslope_dir is None:
            self.update_direction()

        scale = 1.0 - (jump + 1) / num_jumps
        offset = (
            scale * np.linalg.norm(self.position - self.target) * self.glideslope_dir
        )
        r_next = self.target + offset

        delta_v = np.linalg.solve(phi_v, r_next - phi_r @ self.position) - self.velocity
        self.velocity += delta_v
        return delta_v
