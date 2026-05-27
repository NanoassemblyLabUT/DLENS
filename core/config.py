"""Application constants shared by the GUI and inference pipeline."""

from __future__ import annotations

import numpy as np


LOG_HEADER = (
    "CWD,From,File,Shape,Param 1,Param 2,Param 3,Param 4,Param 5,Param 6,"
    "Param 7,mu 1,mu 2,sigma 1,sigma 2,Error,R_g,R_g,Comment\n"
)

MODEL_DIR_NAME = "models"

MODEL_FILENAMES = {
    "spheroid": {
        "radius": "2025_01_21_sphere_CPNN_Radius_0.keras",
        "shape":  "2025_01_21_sphere_CPNN_AspectRatio_0.keras",
        "pdi":    "2025_03_09_sphere_CPNN_PDI_0.keras",
        "rg":     "2025_01_21_sphere_CPNN_GyrationRadius_0.keras",
    },
    "cylinder": {
        "radius": "2025_01_21_cylinder_CPNN_Radius_0.keras",
        "shape":  "2025_01_21_cylinder_CPNN_AspectRatio_0.keras",
        "pdi":    "2025_03_09_cylinder_CPNN_PDI_0.keras",
        "rg":     "2025_01_21_cylinder_CPNN_GyrationRadius_0.keras",
    },
    "disk": {
        "radius": "disk_regression_0_cpnn.keras",
        "shape":  "disk_regression_1_cpnn.keras",
        "pdi":    "disk_regression_2_cpnn.keras",
        "rg":     None,
    },
    "worm": {
        "radius": "worm_regression_0_cpnn.keras",
        "shape":  "worm_regression_1_cpnn.keras",
        "pdi":    "worm_regression_2_cpnn.keras",
        "rg":     None,
    },
    "empty_shell": {
        "radius": "empty_regression_0_cpnn.keras",
        "shape":  "empty_regression_1_cpnn.keras",
        "pdi":    "empty_regression_2_cpnn.keras",
        "rg":     None,
    },
    "inverse_shell": {
        "radius": "inverse_regression_0_cpnn.keras",
        "shape":  "inverse_regression_1_cpnn.keras",
        "pdi":    "inverse_regression_2_cpnn.keras",
        "rg":     None,
    },
}

QR_MODEL_FILENAME = "2025_01_26_SCNN_qr_0.keras"

# Stage 1 classifier: broad family prediction, currently spheroid vs. cylinder.
FAMILY_CLASSIFIER_FILENAME = "binary_classification.json"

# Backward-compatible alias used by older code and notebooks.
CLASSIFIER_FILENAME = FAMILY_CLASSIFIER_FILENAME

# Stage 2 classifiers: optional family-specific subclass prediction.
# Fill these in when the trained subclass classifiers are ready.
SUBCLASSIFIER_FILENAMES = {
    "spheroid": "spheroid_subclassification.json",
    "cylinder": "cylinder_subclassification.json",
}


def reference_q_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the q-grid used by the trained models."""
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    return q_log_arr, q_arr



