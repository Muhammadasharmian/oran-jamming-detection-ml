"""Gradient boosting and randomised tree ensembles.

CatBoost and LightGBM are optional dependencies. If either is missing the
builder falls back to scikit-learn's ``HistGradientBoostingClassifier`` and
emits a warning, so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from jamming_detection import config


def _fallback(name: str) -> HistGradientBoostingClassifier:
    warnings.warn(
        f"{name} is not installed; using HistGradientBoostingClassifier instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )


try:
    from catboost import CatBoostClassifier as _CatBoostBase
except ImportError:  # pragma: no cover - optional dependency
    _CatBoostBase = None

if _CatBoostBase is not None:

    class CatBoostJammingClassifier(_CatBoostBase):
        """CatBoost with a 1-D ``predict`` output, like scikit-learn estimators."""

        def predict(self, data, *args, **kwargs):
            return np.asarray(super().predict(data, *args, **kwargs)).ravel()


def build_catboost(**overrides: Any):
    if _CatBoostBase is None:
        return _fallback("CatBoost")
    return CatBoostJammingClassifier(**{**config.CATBOOST, **overrides})


def build_lightgbm(**overrides: Any):
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return _fallback("LightGBM")
    return LGBMClassifier(**{**config.LIGHTGBM, **overrides})


def build_extra_trees(**overrides: Any) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**{**config.EXTRA_TREES, **overrides})
