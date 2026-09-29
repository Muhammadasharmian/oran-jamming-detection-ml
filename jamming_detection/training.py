"""Shared training workflow and model persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from jamming_detection import __version__
from jamming_detection.data.loader import JAMMING_LABEL, NORMAL_LABEL, Split, oversample_training_set
from jamming_detection.models import build_model, is_binary_only


def binarise(y: np.ndarray, normal_label: str = NORMAL_LABEL) -> np.ndarray:
    return np.where(np.asarray(y) == normal_label, NORMAL_LABEL, JAMMING_LABEL)


def prepare_labels(split: Split, binary: bool) -> Split:
    """Return a copy of ``split`` with labels collapsed to normal / jamming if requested."""
    if not binary:
        return split
    return Split(
        split.X_train,
        binarise(split.y_train),
        split.X_val,
        binarise(split.y_val),
        split.X_test,
        binarise(split.y_test),
        split.feature_names,
    )


def train_model(
    name: str,
    split: Split,
    binary: bool = False,
    oversample: bool = False,
    tune_ensemble: bool = True,
    model_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build, fit and (optionally) tune a registered model.

    Returns a dictionary with the fitted model, the (possibly binarised)
    split, the training time and any tuned ensemble parameters.
    """
    binary = binary or is_binary_only(name)
    split = prepare_labels(split, binary)

    X_train, y_train = split.X_train, split.y_train
    if oversample:
        X_train, y_train = oversample_training_set(X_train, y_train)
        if verbose:
            values, counts = np.unique(y_train, return_counts=True)
            print(f"  oversampled training set: {dict(zip(values.tolist(), counts.tolist()))}")

    model = build_model(name)
    if model_overrides:
        model.set_params(**model_overrides)
    if name == "ddpg":
        model.set_params(model__feature_names=split.feature_names, model__verbose=verbose)

    t0 = time.perf_counter()
    if hasattr(model, "fit_weights"):
        model.fit(X_train, y_train, verbose=verbose)
    else:
        model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    tuned: Dict[str, Any] = {}
    if len(split.y_val):
        if tune_ensemble and hasattr(model, "fit_weights"):
            tuned = model.fit_weights(split.X_val, split.y_val)
        elif name == "isolation_forest":
            final = model[-1]
            tuned = {"threshold": final.fit_threshold(model[:-1].transform(split.X_val), split.y_val)}
        if verbose and tuned:
            print(f"  tuned on validation set: {tuned}")

    return {
        "model": model,
        "split": split,
        "binary": binary,
        "train_seconds": train_seconds,
        "tuned": tuned,
    }


def save_artifact(
    path: str | Path,
    name: str,
    model: Any,
    feature_names,
    binary: bool,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist a fitted model together with the metadata needed for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "name": name,
            "model": model,
            "feature_names": list(feature_names),
            "binary": binary,
            "package_version": __version__,
        },
        path,
    )
    if metrics is not None:
        serialisable = {k: v for k, v in metrics.items() if k != "report"}
        path.with_suffix(".metrics.json").write_text(json.dumps(serialisable, indent=2))
    return path


def load_artifact(path: str | Path) -> Dict[str, Any]:
    return joblib.load(Path(path))
