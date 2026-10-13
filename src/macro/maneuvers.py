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

from macro.config import load_config
from macro.utils import time_to_true_anomaly


def compute_state_transition_matrix(
    times: tuple[float, float] = (0, 1),
    eccentricity: float = 0,
    angular_velocity: float = 1.0,
    *args,
    **kwargs,
):
    """
    Compute the 6x6 state transition matrix for two-body orbital dynamics.

    This matrix maps an initial state vector at time `t0` to a final state at `t1`,
    taking into account the orbit's eccentricity and angular velocity. It handles both
    circular (eccentricity = 0) and elliptical orbits.

    Parameters
    ----------
    times : tuple of float, default=(0, 1)
        Tuple of two time points `(t0, t1)` over which to compute the transition.
    eccentricity : float, default=0.0
        Orbital eccentricity. Must be in the range [0, 1).
    angular_velocity : float, default=1.0
        Mean angular velocity of the orbit.
    args : tuple
        Additional positional arguments (ignored).
    kwargs : dict
        Additional keyword arguments:
        - orbital_radius (float): Orbital radius at the initial point. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        A 6x6 numpy array representing the state transition matrix.

    Raises
    ------
    ValueError
        If eccentricity is not in [0, 1), or if `times` is not of length 2, or if
        `angular_velocity` is non-positive.
    TypeError
        If `times` is not a tuple.
    """
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError("Eccentricity must be in the range [0, 1).")
    if len(times) != 2:
        raise ValueError("Times must be a tuple of two elements (t0, t1).")
    if angular_velocity <= 0:
        raise ValueError("Angular velocity must be a positive value.")
    if not isinstance(times, tuple):
        raise TypeError("Times must be a tuple of two elements (t0, t1).")

    t0, t1 = times  # Corresponding times for the true anomalies
    delta_t = t1 - t0  # Time difference

    f0 = time_to_true_anomaly(
        angular_velocity, eccentricity, t0, calc_derivative=False
    )[0]
    f1 = time_to_true_anomaly(
        angular_velocity, eccentricity, t1, calc_derivative=False
    )[0]

    if eccentricity == 0:
        angular_distance = (
            angular_velocity * delta_t
        )  # Angular distance for circular orbits
        ct = np.cos(angular_distance)
        st = np.sin(angular_distance)
        return np.array(
            [
                [
                    1.0,
                    0.0,
                    6.0 * ct - 6.0 * st,
                    4.0 * st / angular_velocity - 3.0 * delta_t,
                    0.0,
                    2.0 * (1.0 - ct) / angular_velocity,
                ],
                [0.0, ct, 0.0, 0.0, st / angular_velocity, 0.0],
                [
                    0.0,
                    0.0,
                    4.0 - 3.0 * ct,
                    2.0 * (ct - 1.0) / angular_velocity,
                    0.0,
                    st / angular_velocity,
                ],
                [
                    0.0,
                    0.0,
                    6 * angular_velocity * (1.0 - ct),
                    4.0 * ct - 3.0,
                    0.0,
                    2.0 * st,
                ],
                [0.0, -angular_velocity * st, 0.0, 0.0, ct, 0.0],
                [0.0, 0.0, 3.0 * st * angular_velocity, -2.0 * st, 0.0, ct],
            ]
        )

    orbital_radius = kwargs.get("orbital_radius")
    if orbital_radius is None:
        orbital_radius = load_config()["orbit"]["radius"]
    angular_momentum = (
        orbital_radius**2 * angular_velocity
    )  # Angular momentum of the orbit

    sf0, sf1 = np.sin(f0), np.sin(f1)  # Sine and cosine of true anomalies
    cf0, cf1 = np.cos(f0), np.cos(f1)  # Sine and cosine of true anomalies

    rho0 = 1 + eccentricity * cf0  # Effective radius for f0
    rho1 = 1 + eccentricity * cf1  # Effective radius for f1
    p0 = orbital_radius * rho0  # Semi-latus rectum
    p1 = orbital_radius * rho1  # Semi-latus rectum

    K1 = 1.0 / (1.0 - (eccentricity**2.0))  # K1 for the orbit
    K2_0 = angular_momentum / (p0**2)  # K2 for f0
    K2_1 = angular_momentum / (p1**2)  # K2 for f1
    K3_0 = 1.0 + (1.0 / rho0)  # K3 for f0
    K3_1 = 1.0 + (1.0 / rho1)  # K3 for f1

    J = K2_0 * delta_t  # J for f0

    c0, c1 = rho0 * cf0, rho1 * cf1  # Cosine factors for f0 and f1
    s0, s1 = rho0 * sf0, rho1 * sf1  # Sine factors for f0 and f1
    cd0, _cd1 = -(sf0 + eccentricity * np.sin(2 * f0)), -(
        sf1 + eccentricity * np.sin(2 * f1)
    )  # Derivatives of cosine factors
    sd0, _sd1 = cf0 + eccentricity * np.cos(2 * f0), cf1 + eccentricity * np.cos(
        2 * f1
    )  # Derivatives of sine factors

    delta_f = f0 - f1  # Difference in true anomalies
    cY, sY = np.cos(delta_f), np.sin(delta_f)
    rhoY = (
        1 + eccentricity * cY
    )  # Effective radius for the difference in true anomalies
    pY = orbital_radius * rhoY  # Semi-latus rectum for the difference
    K2Y = angular_momentum / (pY**2)  # K2 for the difference in true anomalies

    T1 = np.array(
        [
            [1 / rho0, 0, 0, 0, 0, 0],
            [0, 1 / rhoY, 0, 0, 0, 0],
            [0, 0, 1 / rho0, 0, 0, 0],
            [K2_0 * eccentricity * sf0, 0, 0, rho0 * K2_0, 0, 0],
            [0, K2Y * eccentricity * sY, 0, 0, rhoY * K2Y, 0],
            [0, 0, K2_0 * eccentricity * sf0, 0, 0, rho0 * K2_0],
        ]
    )

    A1 = np.array(
        [
            [1, 0, -K3_0 * c0, K3_0 * s0, 0, 3 * J * rho0**2],
            [0, cY, 0, 0, sY, 0],
            [0, 0, s0, c0, 0, 2 - 3 * eccentricity * s0 * J],
            [
                0,
                0,
                2 * s0,
                2 * c0 - eccentricity,
                0,
                3 * (1 - 2 * eccentricity * s0 * J),
            ],
            [0, -sY, 0, 0, cY, 0],
            [0, 0, sd0, cd0, 0, -3 * eccentricity * (sd0 * J + s0 / rho0**2)],
        ]
    )

    A2 = np.array(
        [
            [
                1,
                0,
                3 * K1 * K3_1 * eccentricity * s1 / rho1,
                -K1 * K3_1 * eccentricity * s1,
                0,
                K1 * (-eccentricity * c1 + 2),
            ],
            [0, 1, 0, 0, 0, 0],
            [
                0,
                0,
                -3 * K1 * s1 * (1 + eccentricity**2 / rho1) / rho1,
                K1 * K3_1 * s1,
                0,
                K1 * (c1 - 2 * eccentricity),
            ],
            [
                0,
                0,
                -3 * K1 * (c1 / rho1 + eccentricity),
                K1 * (K3_1 * c1 + eccentricity),
                0,
                -K1 * s1,
            ],
            [0, 0, 0, 0, 1, 0],
            [
                0,
                0,
                K1 * (3 * rho1 + eccentricity**2 - 1),
                -K1 * rho1**2,
                0,
                K1 * eccentricity * s1,
            ],
        ]
    )

    T2 = np.array(
        [
            [rho1, 0, 0, 0, 0, 0],
            [0, rhoY, 0, 0, 0, 0],
            [0, 0, rho1, 0, 0, 0],
            [-eccentricity * sf1, 0, 0, 1 / (rho1 * K2_1), 0, 0],
            [0, -eccentricity * sY, 0, 0, 1 / (rhoY * K2Y), 0],
            [0, 0, -eccentricity * sf1, 0, 0, 1 / (rho1 * K2_1)],
        ]
    )

    return T1 @ A1 @ A2 @ T2


def compute_maneuver_points(
    r_0: np.ndarray, v_0: np.ndarray, times: tuple[float, float], **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the future position and velocity from an initial state using
    the state transition matrix for orbital motion.

    Parameters
    ----------
    r_0 : array_like
        Initial position vector (3D).
    v_0 : array_like
        Initial velocity vector (3D).
    times : tuple of float
        Tuple of time points `(t0, t1)` over which to propagate the state.
    args : tuple
        Additional positional arguments (ignored).
    kwargs : dict
        Additional keyword arguments:
        - eccentricity (float): Orbital eccentricity. Default is 0.0.
        - orbital_radius (float): Orbital radius. Default is 1.0.
        - angular_velocity (float): Mean angular velocity. Default is computed from `gravpar`.

    Returns
    -------
    tuple of numpy.ndarray
        - Position vector at `t1` (3D)
        - Velocity vector at `t1` (3D)

    Raises
    ------
    ValueError
        If eccentricity is outside the range [0, 1).
    """
    eccentricity = kwargs.get("eccentricity", 0.0)
    if eccentricity < 0 or eccentricity >= 1:
        raise ValueError("Eccentricity must be in the range [0, 1).")
    orbital_radius = kwargs.get("orbital_radius")
    angular_velocity = kwargs.get("angular_velocity")
    if orbital_radius is None or angular_velocity is None:
        orbit = load_config()["orbit"]
        if orbital_radius is None:
            orbital_radius = orbit["radius"]
        if angular_velocity is None:
            angular_velocity = np.sqrt(orbit["gravpar"] / orbital_radius**3)
    P = compute_state_transition_matrix(
        times=times,
        eccentricity=eccentricity,
        orbital_radius=orbital_radius,
        angular_velocity=angular_velocity,
    )
    at = P @ np.array([r_0[0], r_0[1], r_0[2], v_0[0], v_0[1], v_0[2]])
    return at[0:3], at[3:6]
