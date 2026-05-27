"""Preprocessing helpers for classifier and parameter models."""

from __future__ import annotations

import numpy as np


def prepare_model_input(I_arr: np.ndarray) -> np.ndarray:
    """
    Task: Transform normalized intensity into the trained CNN input tensor.
    
    Args:
        - I_arr: 1D array of intensity values from the SAXS profile, normalized to a maximum of 1.
    
    Returns:
        - A 3D NumPy array of shape (1, N, 1) where N is the number of q-values, containing the 
          transformed intensity values suitable for input into the trained CNN models. The 
          transformation includes a logarithmic scaling followed by a hyperbolic tangent function.
    """
    x = np.array(I_arr, dtype=float, copy=True)

    # Raise an error if the intensity array contains no positive values, as the logarithm of 
    # non-positive values is undefined. If there are non-positive values, replace them with the 
    # smallest positive value found in the array. If no positive values are found, use a small 
    # default value (e.g., 1e-10) to avoid issues.
    positive = x > 0
    if not np.any(positive):
        raise ValueError("Intensity must contain at least one positive value.")

    x[~positive] = np.min(x[positive])

    x = 1 + np.log10(x)/2
    x = np.tanh(x)

    return x[np.newaxis, :, np.newaxis]


def interpolate_for_classifier(
    q_log_arr: np.ndarray,
    I_arr:     np.ndarray,
    qr: float,
) -> np.ndarray:
    """Interpolate log intensity onto the classifier's qr-scaled grid."""
    x_int = np.linspace(np.log10(0.64), np.log10(10.0), 64)
    x_ref = q_log_arr + np.log10(qr)
    y_ref = np.log10(I_arr)
    return np.interp(x=x_int, xp=x_ref, fp=y_ref)[np.newaxis, :]
