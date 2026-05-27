"""Background-subtraction calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BackgroundSubtractionResult:
    q:            np.ndarray
    intensity:    np.ndarray
    sigma:        np.ndarray
    scale_factor: float

    """
    Dataclass to hold the results of background subtraction.
        - q:            1D array of q-values for the background-subtracted data.
        - intensity:    1D array of background-subtracted intensities.
        - sigma:        1D array of standard deviations for the background-subtracted data.
        - scale_factor: The scale factor applied to the background data during subtraction.
    """


def auto_subtract_background(
    raw_q:        np.ndarray,
    raw_i:        np.ndarray,
    raw_s:        np.ndarray,
    background_q: np.ndarray,
    background_i: np.ndarray,
    background_s: np.ndarray,
    q_crit:       float,
) -> BackgroundSubtractionResult:
    
    """
    Task: Scale background by least squares above q_crit and subtract it.
    
    Parameters:
        - raw_q:        1D array of q-values for the raw data.
        - raw_i:        1D array of intensities for the raw data.
        - raw_s:        1D array of standard deviations for the raw data.
        - background_q: 1D array of q-values for the background data.
        - background_i: 1D array of intensities for the background data.
        - background_s: 1D array of standard deviations for the background data.
        - q_crit:       Critical q-value above which to scale the background.
    
    Returns:
        - BackgroundSubtractionResult: Dataclass containing the background-subtracted data and scale factor.
    """

    # Extract the tails of the raw and background data above q = q_crit.
    raw_tail        = raw_i[raw_q > q_crit]
    background_tail = background_i[background_q > q_crit]

    # Validate that the tails are non-empty and have matching lengths.
    if raw_tail.size == 0 or background_tail.size == 0:
        raise ValueError("q_crit leaves no data for background scaling.")
    if raw_tail.size != background_tail.size:
        raise ValueError("Raw and background tails must have matching lengths.")

    # Compute the least-squares scale factor to match the background tail to the raw tail.
    denom = np.sum(np.square(background_tail))
    # Guard against division by zero in case the background tail has zero norm.
    if denom == 0:
        raise ValueError("Background tail has zero norm.")

    # The scale factor is calculated using the least square fit.
    # It minimizes the difference between raw_tail and scale_factor*background_tail.
    scale_factor = float(np.sum(raw_tail*background_tail)/denom)

    # Subtract the scaled background from the raw data and propagate the uncertainties.
    intensity = raw_i - scale_factor*background_i
    sigma     = np.sqrt(np.square(raw_s) + np.square(scale_factor*background_s))

    return BackgroundSubtractionResult(raw_q, intensity, sigma, scale_factor)
