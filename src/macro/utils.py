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

# Keep these imports available for existing callers of macro.utils.
from macro.config import (
    DEFAULT_CONFIG_PATH as DEFAULT_CONFIG_PATH,
    load_config as load_config,
)


def random_sign(rng=None):
    """Return either +1 or -1 randomly."""
    rng = np.random if rng is None else rng
    return 1 if rng.randint(0, 2) == 0 else -1


def random_vector(scale=1.0, *, rng=None):
    """Generate a 3D vector with random direction and uniform magnitude up to `scale`."""
    rng = np.random if rng is None else rng
    return scale * np.array([random_sign(rng) * rng.rand() for _ in range(3)])


def initialize_poses(num_agents: int, radius: float, *, seed=None, rng=None):
    """
    Generate random initial positions and attitudes for spacecraft agents.

    Parameters:
    ----------
    num_agents : int
      Number of spacecraft agents
    radius : float
      Maximum position magnitude from the origin
    seed : int, optional
      Random seed for reproducibility

    Returns:
    -------
    initial_positions : np.ndarray
      Initial position vectors for each agent
    initial_attitudes : np.ndarray
      Initial attitude vectors (Euler angles in radians)
    """
    if seed is not None and rng is not None:
        raise ValueError("Provide seed or rng, not both")
    rng = np.random.RandomState(seed) if seed is not None else rng

    initial_positions = np.array(
        [random_vector(radius, rng=rng) for _ in range(num_agents)]
    )
    initial_attitudes = np.array(
        [random_vector(np.pi, rng=rng) for _ in range(num_agents)]
    )

    return initial_positions, initial_attitudes


def check_intersection(ego_initial, ego_target, other_initial, other_target, tol=1e-3):
    """
    Check if the trajectory of the ego agent intersects with another agent's trajectory in 3D space.

    Parameters
    ----------
    ego_initial : np.ndarray
        Initial position of the ego agent.
    ego_target : np.ndarray
        Target position of the ego agent.
    other_initial : np.ndarray
        Initial position of the other agent.
    other_target : np.ndarray
        Target position of the other agent.
    tol : float, optional
        Tolerance for considering an intersection (default is 1e-3).

    Returns
    -------
    bool
        True if the line segments intersect (within `tol`), False otherwise.
    """
    dir_e = ego_target - ego_initial
    dir_o = other_target - other_initial
    sep = ego_initial - other_initial

    dot_ee = np.dot(dir_e, dir_e)
    dot_eo = np.dot(dir_e, dir_o)
    dot_oo = np.dot(dir_o, dir_o)
    dot_es = np.dot(dir_e, sep)
    dot_os = np.dot(dir_o, sep)

    denom = dot_ee * dot_oo - dot_eo * dot_eo
    if abs(denom) < 1e-8:
        return False  # Segments are nearly parallel

    param_a = (dot_eo * dot_os - dot_oo * dot_es) / denom
    param_b = (dot_ee * dot_os - dot_eo * dot_es) / denom

    if not (0 <= param_a <= 1 and 0 <= param_b <= 1):
        return False  # Closest points are outside the segments

    closest_point_e = ego_initial + param_a * dir_e
    closest_point_o = other_initial + param_b * dir_o

    return np.linalg.norm(closest_point_e - closest_point_o) < tol


def generate_cost(ego_location, target_locations, min_value=True, func="euclidean"):
    """
    Generate cost based on Euclidean distance between an agent and all target positions.

    Args:
        ego_location (np.ndarray): Current position of the agent.
        target_locations (np.ndarray): Positions of all targets.
        min_value (bool or int): If True or non-zero, apply negative scaling.

    Returns:
        list[float]: List of cost values.
    """
    ego_location = np.array(ego_location)
    target_locations = np.array(target_locations)
    scale = -(1 ** (int(min_value) - 1))
    if func == "euclidean":
        return [scale * np.linalg.norm(ego_location - t) for t in target_locations]
    elif func == "manhattan":
        return [scale * np.sum(np.abs(ego_location - t)) for t in target_locations]
    else:
        raise ValueError(f"Unknown cost function: {func}")


def time_to_true_anomaly(
    ang_vel: float | np.ndarray,
    eccentricity: float | np.ndarray,
    time_since_periapsis: float | np.ndarray,
    calc_derivative: bool = False,
) -> tuple[np.ndarray, ...]:
    """
    Compute true anomaly (and optionally its derivatives) from time since periapsis
    using an approximation to Kepler's Equation.

    Parameters:
    ----------
    ang_vel : float or np.ndarray
      Mean motion (rad/s)
    eccentricity : float or np.ndarray
      Orbital eccentricity
    time_since_periapsis : float or np.ndarray
      Time since periapsis passage (s)
    calc_derivative : bool, default=False
      If True, return first and second time derivatives of the anomaly

    Returns:
    -------
    F : np.ndarray
      True anomaly (radians)
    F_dot : np.ndarray (optional)
      First derivative of F with respect to time
    F_ddot : np.ndarray (optional)
      Second derivative of F with respect to time
    """

    ang_vel = np.asarray(ang_vel, dtype=float)
    eccentricity = np.asarray(eccentricity, dtype=float)
    time_since_periapsis = np.asarray(time_since_periapsis, dtype=float)

    # Mean anomaly
    mean_anomaly = ang_vel * time_since_periapsis

    # Fourier expansion approximation
    F = (
        mean_anomaly
        + (2.0 * eccentricity - (eccentricity**3.0) / 4.0) * np.sin(mean_anomaly)
        + (5.0 / 4) * (eccentricity**2.0) * np.sin(2.0 * mean_anomaly)
        + (13.0 / 12) * (eccentricity**3.0) * np.sin(3.0 * mean_anomaly)
    )
    if not calc_derivative:
        return (F,)

    # Derivatives of true anomaly
    F_dot = (
        (1.0 - (eccentricity**2.0)) ** (3.0 / 2)
        / (1.0 + (eccentricity * np.cos(F))) ** 2
    ) * ang_vel
    F_ddot = (
        -2.0
        * eccentricity
        * (F_dot**2.0)
        * np.sin(F)
        / (1 + (eccentricity * np.cos(F)))
    )

    return F, F_dot, F_ddot


def consensus_step(
    agent_id: int,
    neighbors: np.ndarray,
    previous: np.ndarray,
    desired: np.ndarray,
    gains: np.ndarray,
    timestep: float,
    count: int,
):
    """
    Perform one value update step for a given agent using consensus-based control.

    Parameters:
      agent_id (int): Index of the ego agent performing the update.
      neighbors (np.ndarray): Indices of neighbors of the ego agent.
      previous (np.ndarray): Previous value vectors for each agent.
      desired (np.ndarray): Desired value vectors for each agent.
      gains (np.ndarray): Per-agent control gain vectors (one per agent).
      timestep (float): Base timestep (scalar multiplier for update magnitude).
      count (int): Current iteration (used to scale the update).

    Returns:
      np.ndarray: Updated value vector for the ego agent.
    """
    updated = previous[agent_id, :].copy()
    ego_error = np.linalg.norm(previous[agent_id, :] - desired[agent_id, :])

    for neighbor_id in neighbors:
        neighbor_error = np.linalg.norm(
            previous[neighbor_id, :] - desired[neighbor_id, :]
        )
        updated -= timestep * count * gains[neighbor_id] * (ego_error - neighbor_error)

    return updated


def to_flat_index(layer_lengths, layer_idx, pos_idx):
    """
    Convert a 2D index (layer, position) to a flat list index.

    Parameters
    ----------
    layer_lengths : list[int]
        Lengths of each layer.
    layer_idx : int
        Index of the layer.
    pos_idx : int
        Index within the layer.

    Returns
    -------
    int
        Flat index in a 1D configuration list.
    """
    offset = sum(layer_lengths[:layer_idx])
    return offset + (pos_idx % layer_lengths[layer_idx])


def to_layer_index(layer_lengths, flat_idx):
    """
    Convert a flat list index to a 2D index (layer, position).

    Parameters
    ----------
    layer_lengths : list[int]
        Lengths of each layer.
    flat_idx : int
        Flat index in a 1D configuration list.

    Returns
    -------
    tuple[int, int]
        Tuple of (layer index, position within layer).
    """
    for layer_idx, layer_len in enumerate(layer_lengths):
        if flat_idx < layer_len:
            return layer_idx, flat_idx
        flat_idx -= layer_len


class NullLogger:
    """
    A logger that does nothing. Used to disable logging in certain contexts.
    """

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def setLevel(self, level):
        pass

    def addHandler(self, handler):
        pass

    def removeHandler(self, handler):
        pass

    def handlers(self):
        return []

    def hasHandlers(self):
        return False

    def getEffectiveLevel(self):
        return 0
