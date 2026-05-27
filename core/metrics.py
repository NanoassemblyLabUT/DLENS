"""Analysis metrics and parameter-distribution helpers."""

from __future__ import annotations

from math import sqrt

import numpy as np


def mean_log_squared_error(I_true: np.ndarray, I_pred: np.ndarray) -> float:
    """Return the GUI's mean squared log intensity error."""
    return float(np.mean(np.square(np.log(I_true + 1) - np.log(I_pred + 1))))


def gaussian_probability_grid(mean: float, sigma: float, grid: np.ndarray | None=None) -> np.ndarray:
    """Return the unnormalized Gaussian curve used by the probability plots."""
    if grid is None:
        grid = np.linspace(0, 2, 257)[:-1]
    return np.exp(-np.square((grid - mean)/(2*sigma)))/sigma/np.sqrt(2*np.pi)


def guinier_radius(
    q_arr:        np.ndarray,
    I_arr:        np.ndarray,
    radius_guess: float,
    class_id:     int,
) -> float:
    """
    Task: Estimate Rg from the low-q Guinier region.
    
    Args:
        - q_arr:        1D array of q-values from the SAXS profile.
        - I_arr:        1D array of intensity values from the SAXS profile.
        - radius_guess: Initial guess for the radius used to scale q-values.
        - class_id:     Identifier for the class of the particle (0 for sphere, 1 for cylinder).

    Returns:
        - Estimated radius of gyration (Rg) based on the Guinier approximation.
    """
    qr = q_arr*radius_guess

    if class_id == 0:
        cut = 1.3
    elif class_id == 1:
        cut = 1.0
    else:
        # The plan is to eventually support more classes, but for now we only have spheres and 
        # cylinders. If an invalid class_id is provided, raise an error.
        raise ValueError(f"Invalid class_id: {class_id}.")

    q_new = q_arr[qr <= cut]
    i_new = I_arr[qr <= cut]

    # If there are too few points in the Guinier region, use the first 8 points of the profile 
    # as a fallback.
    if q_new.size < 8:
        q_new = q_arr[:8]
        i_new = I_arr[:8]

    slope, _ = np.polyfit(x=np.square(q_new), y=np.log(i_new), deg=1)
    return sqrt(-3*slope) if slope < 0 else 0.0
