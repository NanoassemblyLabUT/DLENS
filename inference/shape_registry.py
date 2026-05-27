"""Shape metadata used to remove class-specific branching from the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scattering.scaled_Debye import Disperse_Cylinder_Shell, Disperse_Spheroid_Shell
from scattering.scaled_Debye_addon import Disperse_Disk, Disperse_Empty_Shell, Disperse_Worm


@dataclass(frozen=True)
class PredictedParameters:
    p_0: float
    p_1: float
    p_2: float
    p_3: float
    p_4: float
    p_5: float
    p_6: float
    p_7: float
    std_0: float
    std_1: float
    std_2: float
    r_g_0: float


@dataclass(frozen=True)
class ParameterDisplay:
    """Display and plot formatting for one physical parameter."""

    label: str
    unit: str = ""
    entry_scale: float = 1.0
    probability_title: str | None = None
    probability_xlabel: str | None = None
    model_to_plot: Callable[[np.ndarray, PredictedParameters], np.ndarray] | None = None
    log_x: bool = False


@dataclass(frozen=True)
class ShapeSpec:
    """Metadata and model translation functions for one shape class."""

    key: str
    display_name: str
    class_id: int
    family_key: str
    subclass_id: int | None
    scattering_class: type | None
    model_key: str | None
    primary_label: str
    primary_unit: str
    translate: Callable[[float, float, float, float, float, float, float], PredictedParameters] | None
    simulation_kwargs: Callable[[PredictedParameters], dict[str, float]] | None
    deviance: Callable[[PredictedParameters, float, float, float, float, float, float], tuple[float, float, float]] | None
    parameter_displays: tuple[ParameterDisplay, ...]


def _pdi_from_raw(raw: float | np.ndarray, scale: float) -> float | np.ndarray:
    return np.power(10, -scale * raw) / 2


def _pdi_raw(pdi: float, scale: float) -> float:
    return float(-np.log10(max(2 * pdi, 1e-12)) / scale)


def _pdi_std(pdi: float, raw_sigma: float, scale: float) -> float:
    return float(abs(scale * np.log(10) * pdi * raw_sigma))


def translate_spheroid(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    p_0 = 256 * m_0
    p_1 = 2 * m_1
    p_2 = _pdi_from_raw(m_2, 4)
    return PredictedParameters(p_0, p_1, p_2, 0.75, 0.025, 2 * p_0, 2.0, 0.0, 256 * s_0, 2 * s_1, _pdi_std(p_2, s_2, 4), 256 * m_3)


def translate_cylinder(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    p_0 = 256 * m_0
    p_1 = 16 * m_1 * p_0
    p_2 = _pdi_from_raw(m_2, 4)
    r_g_0 = 1024 * m_3
    if np.square(r_g_0) > np.square(p_0) / 2:
        p_1 = float(np.sqrt(12 * (np.square(r_g_0) - np.square(p_0) / 2)))
    return PredictedParameters(p_0, p_1, p_2, 0.75, 0.025, 2 * p_0, 1.0, 0.0, 256 * s_0, 16 * s_1 * p_0, _pdi_std(p_2, s_2, 4), r_g_0)


def translate_disk(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    radius = 500.0 * m_0
    height = radius * m_1 / 4.0
    pdi = _pdi_from_raw(m_2, 3)
    return PredictedParameters(radius, height, pdi, 0.75, 0.10, max(height, 1.0), 1.0, 0.0, 500.0 * s_0, radius * s_1 / 4.0, _pdi_std(pdi, s_2, 3), m_3)


def translate_empty_shell(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    radius = 500.0 * m_0
    epsilon = 2.0 * m_1
    pdi = _pdi_from_raw(m_2, 3)
    return PredictedParameters(radius, epsilon, pdi, 0.01, 0.10, radius, 2.0, 2.0, 500.0 * s_0, 2.0 * s_1, _pdi_std(pdi, s_2, 3), m_3)


def translate_inverse_shell(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    radius = 500.0 * m_0
    epsilon = 2.0 * m_1
    pdi = _pdi_from_raw(m_2, 3)
    return PredictedParameters(radius, epsilon, pdi, 0.50, -0.25, radius, 2.0, 0.0, 500.0 * s_0, 2.0 * s_1, _pdi_std(pdi, s_2, 3), m_3)


def translate_worm(m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float, m_3: float) -> PredictedParameters:
    radius = 128.0 * m_0
    contour_length = radius * 160.0 * m_1
    pdi = _pdi_from_raw(m_2, 3)
    return PredictedParameters(radius, contour_length, pdi, 0.70, 0.10, radius, 1.0, 0.0, 128.0 * s_0, radius * 160.0 * s_1, _pdi_std(pdi, s_2, 3), m_3)


def retranslate_spheroid(params: PredictedParameters) -> tuple[float, float, float]:
    return params.p_0 / 256.0, params.p_1 / 2.0, _pdi_raw(params.p_2, 4)


def retranslate_cylinder(params: PredictedParameters) -> tuple[float, float, float]:
    return params.p_0 / 256.0, params.p_1 / (16.0 * params.p_0), _pdi_raw(params.p_2, 4)


def retranslate_disk(params: PredictedParameters) -> tuple[float, float, float]:
    return params.p_0 / 500.0, 4.0 * params.p_1 / params.p_0, _pdi_raw(params.p_2, 3)


def retranslate_empty_shell(params: PredictedParameters) -> tuple[float, float, float]:
    return params.p_0 / 500.0, params.p_1 / 2.0, _pdi_raw(params.p_2, 3)


def retranslate_inverse_shell(params: PredictedParameters) -> tuple[float, float, float]:
    return retranslate_empty_shell(params)


def retranslate_worm(params: PredictedParameters) -> tuple[float, float, float]:
    return params.p_0 / 128.0, params.p_1 / (160.0 * params.p_0), _pdi_raw(params.p_2, 3)


def _deviance(raw_values: tuple[float, float, float], means: tuple[float, float, float], sigmas: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple((raw - mean) / sigma for raw, mean, sigma in zip(raw_values, means, sigmas))


def spheroid_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_spheroid(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def cylinder_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_cylinder(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def disk_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_disk(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def empty_shell_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_empty_shell(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def inverse_shell_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_inverse_shell(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def worm_deviance(params: PredictedParameters, m_0: float, s_0: float, m_1: float, s_1: float, m_2: float, s_2: float) -> tuple[float, float, float]:
    return _deviance(retranslate_worm(params), (m_0, m_1, m_2), (s_0, s_1, s_2))


def spheroid_kwargs(params: PredictedParameters) -> dict[str, float]:
    return {"R": params.p_0, "epsilon": params.p_1, "PDI": params.p_2, "f_core": params.p_3, "rho_delta": params.p_4, "t": params.p_5, "p": params.p_6, "q": params.p_7}


def cylinder_kwargs(params: PredictedParameters) -> dict[str, float]:
    return {"R": params.p_0, "epsilon": params.p_1 / params.p_0, "PDI": params.p_2, "f_core": params.p_3, "rho_delta": params.p_4, "t": params.p_5, "p": params.p_6, "q": params.p_7}


def disk_kwargs(params: PredictedParameters) -> dict[str, float]:
    return {"R": params.p_0, "h": params.p_1, "PDI": params.p_2, "t_shell": params.p_5, "f_core": params.p_3, "rho_delta": params.p_4}


def empty_shell_kwargs(params: PredictedParameters) -> dict[str, float]:
    return {"R": params.p_0, "epsilon": params.p_1, "PDI": params.p_2, "t": params.p_5, "p": params.p_6, "rho_delta": params.p_4}


def inverse_shell_kwargs(params: PredictedParameters) -> dict[str, float]:
    return {"R": params.p_0, "epsilon": params.p_1, "PDI": params.p_2, "f_core": params.p_3, "rho_delta": params.p_4, "t": params.p_5, "p": params.p_6, "q": params.p_7}


def worm_kwargs(params: PredictedParameters) -> dict[str, float]:
    n_seg = 10
    return {"R": params.p_0, "L_seg": max(params.p_1 / n_seg, 1.0), "n_seg": n_seg, "PDI": params.p_2, "t_shell": params.p_5, "p_core": params.p_6, "p_shell": params.p_7, "f_core": params.p_3, "rho_delta": params.p_4}


def radius_256_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 256 * raw


def radius_500_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 500 * raw


def radius_128_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 128 * raw


def spheroid_aspect_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 100 * (2 * raw)


def cylinder_length_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 16 * raw * params.p_0


def disk_height_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return params.p_0 * raw / 4


def worm_contour_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return 160 * raw * params.p_0


def pdi_4_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return _pdi_from_raw(raw, 4)


def pdi_3_plot(raw: np.ndarray, params: PredictedParameters) -> np.ndarray:
    return _pdi_from_raw(raw, 3)


COMMON_TRAILING_DISPLAYS = (
    ParameterDisplay("Core Fraction", "%", 100.0),
    ParameterDisplay("Scattering Fraction", "", 1000.0),
    ParameterDisplay("Corona Length", "angstrom"),
    ParameterDisplay("Core Density"),
    ParameterDisplay("Corona Density"),
)

ADDON_TRAILING_DISPLAYS = (
    ParameterDisplay("Core Fraction", "%", 100.0),
    ParameterDisplay("Scattering Fraction", "", 1000.0),
    ParameterDisplay("Shell Thickness", "angstrom"),
    ParameterDisplay("Core Density"),
    ParameterDisplay("Shell Density"),
)

SPHEROID_PARAMETER_DISPLAYS = (
    ParameterDisplay("Radius", "angstrom", probability_title="Radius Probability Function", probability_xlabel=r"Radius ($\AA$)", model_to_plot=radius_256_plot),
    ParameterDisplay("Aspect Ratio", "%", 100.0, "Aspect Ratio Probability Function", r"Aspect Ratio (%)", spheroid_aspect_plot),
    ParameterDisplay("PDI", "", probability_title="PDI Probability Function", probability_xlabel="PDI", model_to_plot=pdi_4_plot, log_x=True),
    *COMMON_TRAILING_DISPLAYS,
)

CYLINDER_PARAMETER_DISPLAYS = (
    ParameterDisplay("Radius", "angstrom", probability_title="Radius Probability Function", probability_xlabel=r"Radius ($\AA$)", model_to_plot=radius_256_plot),
    ParameterDisplay("Axial Length", "angstrom", 1.0, "Axial Length Probability Function", r"Axial Length ($\AA$)", cylinder_length_plot),
    ParameterDisplay("PDI", "", probability_title="PDI Probability Function", probability_xlabel="PDI", model_to_plot=pdi_4_plot, log_x=True),
    *COMMON_TRAILING_DISPLAYS,
)

DISK_PARAMETER_DISPLAYS = (
    ParameterDisplay("Radius", "angstrom", probability_title="Radius Probability Function", probability_xlabel=r"Radius ($\AA$)", model_to_plot=radius_500_plot),
    ParameterDisplay("Height", "angstrom", 1.0, "Height Probability Function", r"Height ($\AA$)", disk_height_plot),
    ParameterDisplay("PDI", "", probability_title="PDI Probability Function", probability_xlabel="PDI", model_to_plot=pdi_3_plot, log_x=True),
    *ADDON_TRAILING_DISPLAYS,
)

WORM_PARAMETER_DISPLAYS = (
    ParameterDisplay("Radius", "angstrom", probability_title="Radius Probability Function", probability_xlabel=r"Radius ($\AA$)", model_to_plot=radius_128_plot),
    ParameterDisplay("Contour Length", "angstrom", 1.0, "Contour Length Probability Function", r"Contour Length ($\AA$)", worm_contour_plot),
    ParameterDisplay("PDI", "", probability_title="PDI Probability Function", probability_xlabel="PDI", model_to_plot=pdi_3_plot, log_x=True),
    *ADDON_TRAILING_DISPLAYS,
)

EMPTY_SHELL_PARAMETER_DISPLAYS = (
    ParameterDisplay("Radius", "angstrom", probability_title="Radius Probability Function", probability_xlabel=r"Radius ($\AA$)", model_to_plot=radius_500_plot),
    ParameterDisplay("Aspect Ratio", "%", 100.0, "Aspect Ratio Probability Function", r"Aspect Ratio (%)", spheroid_aspect_plot),
    ParameterDisplay("PDI", "", probability_title="PDI Probability Function", probability_xlabel="PDI", model_to_plot=pdi_3_plot, log_x=True),
    *ADDON_TRAILING_DISPLAYS,
)

SHAPES: dict[str, ShapeSpec] = {
    "spheroid": ShapeSpec("spheroid", "Spheroid", 0, "spheroid", 0, Disperse_Spheroid_Shell, "spheroid", "Aspect Ratio", "%", translate_spheroid, spheroid_kwargs, spheroid_deviance, SPHEROID_PARAMETER_DISPLAYS),
    "cylinder": ShapeSpec("cylinder", "Cylinder", 1, "cylinder", 0, Disperse_Cylinder_Shell, "cylinder", "Axial Length", "angstrom", translate_cylinder, cylinder_kwargs, cylinder_deviance, CYLINDER_PARAMETER_DISPLAYS),
    "disk": ShapeSpec("disk", "Disk", 2, "cylinder", 1, Disperse_Disk, "disk", "Height", "angstrom", translate_disk, disk_kwargs, disk_deviance, DISK_PARAMETER_DISPLAYS),
    "worm": ShapeSpec("worm", "Long Worm", 3, "cylinder", 2, Disperse_Worm, "worm", "Contour Length", "angstrom", translate_worm, worm_kwargs, worm_deviance, WORM_PARAMETER_DISPLAYS),
    "empty_shell": ShapeSpec("empty_shell", "Empty Shell", 4, "spheroid", 1, Disperse_Empty_Shell, "empty_shell", "Aspect Ratio", "%", translate_empty_shell, empty_shell_kwargs, empty_shell_deviance, EMPTY_SHELL_PARAMETER_DISPLAYS),
    "inverse_shell": ShapeSpec("inverse_shell", "Sign-Inverting Core Shell", 5, "spheroid", 2, Disperse_Spheroid_Shell, "inverse_shell", "Aspect Ratio", "%", translate_inverse_shell, inverse_shell_kwargs, inverse_shell_deviance, EMPTY_SHELL_PARAMETER_DISPLAYS),
}

SHAPES_BY_CLASS = {spec.class_id: spec for spec in SHAPES.values()}
SHAPES_BY_DISPLAY_NAME = {spec.display_name: spec for spec in SHAPES.values()}

FAMILY_CLASS_TO_KEY = {
    0: "spheroid",
    1: "cylinder",
}

DEFAULT_SHAPE_BY_FAMILY = {
    "spheroid": SHAPES["spheroid"],
    "cylinder": SHAPES["cylinder"],
}

SHAPES_BY_FAMILY_AND_SUBCLASS = {
    (spec.family_key, spec.subclass_id): spec
    for spec in SHAPES.values()
    if spec.subclass_id is not None
}
