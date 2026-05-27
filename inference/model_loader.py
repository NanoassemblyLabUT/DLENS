"""Model loading utilities."""

from __future__ import annotations

import os
import pickle as pk

from pathlib import Path
from typing import Any

import tensorflow as tf

from core.config import (
    FAMILY_CLASSIFIER_FILENAME,
    MODEL_DIR_NAME,
    MODEL_FILENAMES,
    QR_MODEL_FILENAME,
    SUBCLASSIFIER_FILENAMES,
)


def load_shape_models(model_dir: str | os.PathLike[str], shape_key: str) -> dict[str, Any]:
    """
    Task: Load all parameter models for one shape.
    
    Args:
        - model_dir: Directory containing the model files.
        - shape_key: Key identifying the shape for which to load models.
    
    Returns:
        - A dictionary mapping model names to loaded TensorFlow models.
    """
    filenames = MODEL_FILENAMES[shape_key]
    loaded: dict[str, Any] = {}
    for name, filename in filenames.items():
        loaded[name] = None if filename is None else tf.keras.models.load_model(Path(model_dir) / filename, compile=False)
    return loaded


def load_pickle_model(path: Path) -> Any:
    """Load a pickle-based sklearn/SVM-style model."""
    with path.open("rb") as f:
        return pk.load(f)


def load_classifier_model(path: Path) -> Any:
    """Load a classifier from either pickle or XGBoost JSON format."""
    if path.suffix.lower() == ".json":
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(path)
        return model
    return load_pickle_model(path)


def load_subclassifiers(model_dir: Path) -> dict[str, Any]:
    """Load configured family-specific subclass classifiers."""
    classifiers: dict[str, Any] = {}
    for family_key, filename in SUBCLASSIFIER_FILENAMES.items():
        if filename:
            classifiers[family_key] = load_classifier_model(model_dir / filename)
    return classifiers


def load_all_models(cwd: str | os.PathLike[str]) -> dict[str, Any]:
    """
    Tasks: Load QR, classifier, optional subclass classifiers, and all configured shape-parameter models.
    Args:
        - cwd: Current working directory where the model directory is located.
    
    Returns:
        - A dictionary containing the loaded models.
    """
    model_dir = Path(cwd) / MODEL_DIR_NAME
    family_classifier = load_classifier_model(model_dir / FAMILY_CLASSIFIER_FILENAME)
    models: dict[str, Any] = {
        "shape_models": {},
        "subclassifiers": load_subclassifiers(model_dir),
        "qr": tf.keras.models.load_model(model_dir / QR_MODEL_FILENAME, compile=False),
        "family_classifier": family_classifier,
        # Backward-compatible alias.
        "classifier": family_classifier,
    }

    for shape_key in MODEL_FILENAMES:
        models["shape_models"][shape_key] = load_shape_models(model_dir, shape_key)

    return models

