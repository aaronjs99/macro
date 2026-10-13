"""Bounded Euler-angle error tracking on a spacecraft communication graph."""

import numpy as np


def tracking_operator(neighbors, gains, timestep):
    """Build one interval of neighbor error mixing with a desired-state anchor.

    ``gains[i, j, axis]`` couples spacecraft i to j. The last column anchors
    each spacecraft's error to zero. Euler substeps keep every weight
    nonnegative; matrix powers combine them into one recorded interval.
    """
    gains = np.asarray(gains, dtype=float)
    n = len(neighbors)
    if gains.shape != (n, n + 1, 3) or set(neighbors) != set(range(n)):
        raise ValueError(
            "Tracking requires one graph entry and gain row for every spacecraft"
        )
    if not np.isfinite(gains).all() or (gains < 0).any() or (gains[:, -1] <= 0).any():
        raise ValueError(
            "Tracking gains must be finite and nonnegative with positive anchors"
        )
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError("Tracking timestep must be finite and positive")
    weights = np.zeros((n, n, 3))
    for agent, adjacent in neighbors.items():
        for neighbor in adjacent:
            if not 0 <= neighbor < n:
                raise ValueError("Tracking neighbor IDs must identify a spacecraft")
            if neighbor != agent:
                weights[agent, neighbor] = gains[agent, neighbor]
    rates = weights.sum(axis=1) + gains[:, -1]
    substeps = max(1, int(np.ceil(float(rates.max()) * timestep)))
    step = timestep / substeps
    operator = step * weights
    operator[np.arange(n), np.arange(n)] = np.maximum(0, 1 - step * rates)
    for axis in range(3):
        operator[:, :, axis] = np.linalg.matrix_power(operator[:, :, axis], substeps)
    return operator


def tracking_update(previous, desired, operator):
    """Mix per-axis errors; a spacecraft already at its desired state stays there."""
    return desired + np.einsum("ija,ja->ia", operator, previous - desired)
