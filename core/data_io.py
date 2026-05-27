"""File and run-directory helpers for SAXS profiles."""

from __future__ import annotations

import os
import shutil
import re

import numpy as np

from datetime import datetime
from pathlib import Path

from .config import LOG_HEADER


def create_run_directory(cwd: str | os.PathLike[str], username: str | None=None) -> dict[str, str]:
    """
    Task: Create the D-LENS run directory and log file used by the GUI.
    
    Args:
        - cwd:      The current working directory where the D-LENS folder will be created.
        - username: Optional username to include in the run name. If None, the system login name 
                    will be used.

    Returns:
        - A dictionary containing paths for the base D-LENS folder, the specific run directory, 
          and the log file.

    """

    # Create the base D-LENS directory if it doesn't exist.
    cwd_path  = Path(cwd)
    base_path = cwd_path/"D-LENS"
    base_path.mkdir(exist_ok=True)

    # Determine the username to use in the run name.
    # If not provided, get the system login name.
    if username is None:
        username = os.getlogin()

    # Generate a unique run name using the username and current date.
    # If a run with the same name already exists, append a counter to ensure uniqueness.
    # Format: {username}_{YYYYMMDD}_{counter}
    current = datetime.now().strftime("%Y%m%d")
    count = 0
    while True:
        run_name = f"{username}_{current}_{count}"
        if not (base_path / run_name).exists():
            break
        count += 1

    # Create the run directory and log file.
    working_dir = base_path/run_name
    working_dir.mkdir(exist_ok=True)

    log_file = f"{run_name}.csv"
    log_path = base_path/log_file
    log_path.write_text(LOG_HEADER, encoding="utf-8")

    return {
        "base_path":   str(base_path),
        "working_dir": str(working_dir),
        "log_file":    log_file,
        "log_path":    str(log_path),
    }


def read_saxs_columns(file_path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Task: Read a SAXS file formatted as q/I or q/I/s columns.
    
    Args:
        - file_path: The path to the SAXS data file. The file can be either a CSV (comma-separated)
                     or a whitespace-separated text file. The function will automatically detect 
                     the format based on the file extension.
    
    Returns:
        - A tuple containing three elements:
            1. A NumPy array of q values (scattering vector magnitudes).
            2. A NumPy array of intensity values corresponding to each q.
            3. A NumPy array of sigma values (uncertainties) if present in the file; 
               otherwise, None. The function will check if all rows contain a sigma value 
               and return None if any row is missing a sigma value.
    """

    # Determine the file format based on the extension and read the data accordingly.
    path   = Path(file_path)
    is_csv = path.suffix.lower() == ".csv"

    rows: list[tuple[float, float, float | None]] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip empty lines and lines that don't start with a number (to ignore headers 
            # or comments).
            if not line or line[0] not in "0123456789.-+":
                continue

            # If the file is a CSV, split by commas; otherwise, split by whitespace.
            parts = line.strip().split(",") if is_csv else line.split()

            # We expect at least two columns (q and intensity). If there are fewer than 2 columns, 
            # skip the line.
            if len(parts) < 2:
                continue

            q = float(parts[0])
            I = float(parts[1])
            s = float(parts[2]) if len(parts) > 2 and parts[2] else None
            rows.append((q, I, s))

    if not rows:
        raise ValueError(f"No SAXS data rows found in {path}")

    data   = np.array([(q, I) for q, I, _ in rows], dtype=float)

    q_arr = data[:, 0]
    I_arr = data[:, 1]

    # Ensure all intensity values are positive by replacing non-positive values with the smallest 
    # positive value found in the array. If no positive values are found, use a small default value 
    # (e.g., 1e-10) to avoid issues with logarithmic scaling or model fitting.
    I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0]) if np.any(I_arr > 0) else 1e-10

    sigmas = [s for _, _, s in rows]
    s_arr  = None if any(s is None for s in sigmas) else np.array(sigmas, dtype=float)

    if s_arr is None:
        # Use sqrt(I) as a fallback for sigma if not provided.
        s_arr = np.sqrt(I_arr)

    return q_arr, I_arr, s_arr


def interpolate_normalized_intensity(
    q_arr: np.ndarray,
    I_arr: np.ndarray,
    q_ref: np.ndarray,
) -> np.ndarray:
    """
    Task: Interpolate intensity onto the model q-grid and normalize to max 1.
    Args:
        - q_arr: The array of q values from the SAXS profile.
        - I_arr: The array of intensity values corresponding to each q in q_arr.
        - q_ref: The reference q-grid onto which the intensity should be interpolated.
    Returns:
        - A NumPy array of interpolated and normalized intensity values corresponding to the 
          reference q-grid.
    """

    # Convert intensity to a safe floating-point array and ensure all values are positive by 
    # replacing non-positive values with the smallest positive value found in the array. If no 
    # positive values are found, use a small default value (e.g., 1e-10) to avoid issues with 
    # logarithmic scaling or model fitting.
    safe_intensity = np.array(I_arr, dtype=float, copy=True)
    positive       = safe_intensity > 0

    if not np.any(positive):
        raise ValueError("Intensity must contain at least one positive value.")

    safe_intensity[~positive] = np.min(safe_intensity[positive])
    interpolated = np.interp(q_ref, q_arr, safe_intensity)

    return interpolated/np.max(interpolated)


def load_saxs_profile(
    file_path:   str | os.PathLike[str],
    q_ref: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Task: Load and normalize a SAXS profile for model inference.
    
    Args:
        - file_path: The path to the SAXS data file. The file can be either a CSV (comma-separated)
                     or a whitespace-separated text file. The function will automatically detect 
                     the format based on the file extension.
        - q_ref:     The reference q-grid onto which the intensity should be interpolated and 
                     normalized. This grid is typically defined by the model and should cover 
                     the same q-range as the input data. The function will interpolate the 
                     intensity values from the input data onto this grid.
    
    Returns:
        - A tuple containing two elements:
            1. A NumPy array of q values corresponding to the reference q-grid (same as q_ref).
            2. A NumPy array of intensity values interpolated and normalized to the reference 
               q-grid, with a maximum value of 1.
    """
    q_arr, I_arr, _ = read_saxs_columns(file_path)
    return q_ref, interpolate_normalized_intensity(q_arr=q_arr, I_arr=I_arr, q_ref=q_ref)


def append_export_log(log_path: str | os.PathLike[str], values: list[object]) -> None:
    """
    Task: Append one CSV row to the analysis log.

    Args:
        - log_path: The path to the log file where the values should be appended. The log file is 
                    expected to be a CSV file with a header defined by LOG_HEADER.
        - values:   A list of values to append as a new row in the log file. The values will be 
                    converted to strings and joined with commas before being written to the file.
    """
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(",".join(str(value) for value in values) + "\n")

    return None


def copy_profile_to_run(file_path: str | os.PathLike[str], working_dir: str | os.PathLike[str]) -> str:
    """
    Task: Copy the analyzed SAXS file into the active run directory.

    Args:
        - file_path:   The path to the SAXS data file to be copied.
        - working_dir: The directory where the file should be copied.

    Returns:
        - The path to the copied file in the working directory.
    """
    target = Path(working_dir)/Path(file_path).name
    shutil.copy(file_path, target)
    return str(target)
