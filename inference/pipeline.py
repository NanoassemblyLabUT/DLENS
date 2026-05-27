"""Inference pipeline functions decoupled from Tkinter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.preprocessing import prepare_model_input
from .shape_registry import (
    DEFAULT_SHAPE_BY_FAMILY,
    FAMILY_CLASS_TO_KEY,
    SHAPES_BY_CLASS,
    SHAPES_BY_FAMILY_AND_SUBCLASS,
    PredictedParameters,
    ShapeSpec,
)


@dataclass(frozen=True)
class ClassificationResult:
    """Results from the hierarchical shape classification workflow."""

    spec: ShapeSpec
    class_id: int
    class_score: float
    qr: float
    family_key: str
    family_class_id: int
    family_score: float
    subclass_id: int | None = None
    subclass_score: float | None = None
    used_subclassifier: bool = False


@dataclass(frozen=True)
class ParameterPrediction:
    """Parameter prediction results for a classified shape."""

    params: PredictedParameters
    means: tuple[float, float, float, float]
    sigmas: tuple[float, float, float]


def _as_scalar_prediction(raw_pred: Any) -> float:
    """Return a scalar numeric prediction from sklearn-like outputs."""
    arr = np.asarray(raw_pred)
    return float(arr.reshape(-1)[0])


def predict_qr(I_arr: np.ndarray, qr_model: Any) -> float:
    """Predict the radius-of-gyration scale used by the downstream regression workflow."""
    x = prepare_model_input(I_arr)
    qr_raw = qr_model.predict(x)[0]
    return float(np.power(10, 3 * np.ravel(qr_raw)[0]))

def prepare_classifier_input(I_arr: np.ndarray) -> np.ndarray:
    """Return the log-intensity vector used by the retrained XGBoost classifiers."""
    intensities = np.asarray(I_arr, dtype=float).reshape(-1)
    positive = intensities[intensities > 0]
    floor = float(np.min(positive)) if positive.size else 1e-12
    clipped = np.where(intensities > 0, intensities, floor)
    return np.log10(clipped)[np.newaxis, :]


def _predict_class_id(model: Any, classifier_input: np.ndarray) -> int:
    raw_pred = _as_scalar_prediction(model.predict(classifier_input))
    return int(round(raw_pred))


def _class_probability(model: Any, classifier_input: np.ndarray, class_id: int) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(classifier_input))
    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]
    if probabilities.size == 0:
        return None
    row = probabilities[0]
    if 0 <= class_id < row.shape[0]:
        return float(row[class_id])
    return None


def classify_family(classifier_input: np.ndarray, family_classifier: Any) -> tuple[str, int, float]:
    """Run the broad family classifier and map its class id to a family key."""
    family_class_id = _predict_class_id(family_classifier, classifier_input)
    family_key = FAMILY_CLASS_TO_KEY.get(family_class_id)
    if family_key is None:
        raise KeyError(f"No shape family is registered for class id {family_class_id}.")

    cylinder_probability = _class_probability(family_classifier, classifier_input, 1)
    if cylinder_probability is None:
        cylinder_probability = float(family_class_id)
    return family_key, family_class_id, cylinder_probability


def classify_subclass(
    classifier_input: np.ndarray,
    family_key: str,
    subclassifiers: dict[str, Any] | None,
) -> tuple[int | None, float | None, bool]:
    """Run the optional family-specific subclass classifier."""
    if not subclassifiers:
        return None, None, False

    subclassifier = subclassifiers.get(family_key)
    if subclassifier is None:
        return None, None, False

    subclass_id = _predict_class_id(subclassifier, classifier_input)
    subclass_score = _class_probability(subclassifier, classifier_input, subclass_id)
    if subclass_score is None:
        subclass_score = float(subclass_id)
    return subclass_id, subclass_score, True


def classify_profile_hierarchical(
    I_arr: np.ndarray,
    q_log_arr: np.ndarray,
    qr_model: Any,
    family_classifier: Any,
    subclassifiers: dict[str, Any] | None = None,
) -> ClassificationResult:
    """
    Predict Rg scale, broad shape family, optional subclass, and final shape spec.

    Classifiers use normalized log-intensity only, matching the retrained XGBoost workflow.
    Rg is still predicted for display and downstream parameter regression.

    If no subclassifier is configured for the predicted family, the pipeline falls back to the
    existing family default: spheroid for spheroid-family data and cylinder for cylinder-family data.
    """
    qr = predict_qr(I_arr, qr_model)
    classifier_input = prepare_classifier_input(I_arr)

    family_key, family_class_id, family_score = classify_family(classifier_input, family_classifier)
    subclass_id, subclass_score, used_subclassifier = classify_subclass(
        classifier_input,
        family_key,
        subclassifiers,
    )

    spec = DEFAULT_SHAPE_BY_FAMILY[family_key]
    if subclass_id is not None:
        spec = SHAPES_BY_FAMILY_AND_SUBCLASS.get((family_key, subclass_id), spec)

    return ClassificationResult(
        spec=spec,
        class_id=spec.class_id,
        class_score=subclass_score if subclass_score is not None else family_score,
        qr=qr,
        family_key=family_key,
        family_class_id=family_class_id,
        family_score=family_score,
        subclass_id=subclass_id,
        subclass_score=subclass_score,
        used_subclassifier=used_subclassifier,
    )


def classify_profile(
    I_arr: np.ndarray,
    q_log_arr: np.ndarray,
    qr_model: Any,
    classifier: Any,
) -> ClassificationResult:
    """Backward-compatible wrapper for the original one-stage classifier workflow."""
    return classify_profile_hierarchical(I_arr, q_log_arr, qr_model, classifier, {})


def predict_parameters(I_arr: np.ndarray, spec: ShapeSpec, shape_models: dict[str, Any]) -> ParameterPrediction:
    """Predict and translate physical parameters for a classified shape."""
    if spec.model_key is None or spec.translate is None:
        raise NotImplementedError(f"No parameter model mapping has been configured for {spec.display_name}.")

    models = shape_models[spec.model_key]
    x = prepare_model_input(I_arr)
    pred_0 = models["radius"].predict(x)
    pred_1 = models["shape"].predict(x)
    pred_2 = models["pdi"].predict(x)
    pred_3 = models["rg"].predict(x)

    m_0, s_0 = float(pred_0[0, 0]), float(pred_0[0, 1])
    m_1, s_1 = float(pred_1[0, 0]), float(pred_1[0, 1])
    m_2, s_2 = float(pred_2[0, 0]), float(pred_2[0, 1])
    m_3 = float(pred_3[0, 0])

    return ParameterPrediction(
        params=spec.translate(m_0, s_0, m_1, s_1, m_2, s_2, m_3),
        means=(m_0, m_1, m_2, m_3),
        sigmas=(s_0, s_1, s_2),
    )


def simulate_profile(q_arr: np.ndarray, spec: ShapeSpec, params: PredictedParameters) -> np.ndarray:
    """Run Debye scattering for a shape spec and parameter set."""
    if spec.scattering_class is None or spec.simulation_kwargs is None:
        raise NotImplementedError(f"No simulation mapping has been configured for {spec.display_name}.")

    scattering = spec.scattering_class(**spec.simulation_kwargs(params))
    return scattering.Debye_scattering(q_arr=q_arr)

