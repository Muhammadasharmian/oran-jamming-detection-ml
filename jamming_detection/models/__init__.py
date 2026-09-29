"""Model registry.

Every model is returned as a scikit-learn ``Pipeline`` of ``StandardScaler``
followed by the estimator, so all models share the same ``fit`` /
``predict`` / ``predict_proba`` interface and the same preprocessing.

Binary-only models (Isolation Forest, DDPG and the jamming-score
ensembles) predict ``normal`` vs ``jamming``; the training script converts
the labels automatically when one of them is selected.
"""

from __future__ import annotations

from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from jamming_detection.models.boosting import build_catboost, build_extra_trees, build_lightgbm
from jamming_detection.models.classical import IsolationForestDetector, build_random_forest, build_svm
from jamming_detection.models.ensemble import JammingScoreEnsemble, WeightedVotingEnsemble
from jamming_detection.models.neural import build_bi_cnn_lstm, build_cnn_lstm, build_deep_mlp, build_mlp


def _scaled(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _ddpg() -> BaseEstimator:
    from jamming_detection.models.drl import DDPGDetector  # torch-heavy import

    return DDPGDetector(actor_type="mlp")


_BASE: Dict[str, Callable[[], BaseEstimator]] = {
    "random_forest": build_random_forest,
    "svm": build_svm,
    "isolation_forest": IsolationForestDetector,
    "catboost": build_catboost,
    "lightgbm": build_lightgbm,
    "extra_trees": build_extra_trees,
    "mlp": build_mlp,
    "deep_mlp": build_deep_mlp,
    "cnn_lstm": build_cnn_lstm,
    "bi_cnn_lstm": build_bi_cnn_lstm,
    "ddpg": _ddpg,
}


def _boosting_ensemble() -> WeightedVotingEnsemble:
    return WeightedVotingEnsemble(
        {
            "catboost": _scaled(build_catboost()),
            "lightgbm": _scaled(build_lightgbm()),
            "extra_trees": _scaled(build_extra_trees()),
        }
    )


def _classical_ensemble() -> JammingScoreEnsemble:
    return JammingScoreEnsemble(
        {
            "random_forest": _scaled(build_random_forest()),
            "svm": _scaled(build_svm()),
            "isolation_forest": _scaled(IsolationForestDetector()),
        }
    )


def _catboost_iforest_ensemble() -> JammingScoreEnsemble:
    return JammingScoreEnsemble(
        {
            "catboost": _scaled(build_catboost()),
            "isolation_forest": _scaled(IsolationForestDetector()),
        }
    )


_ENSEMBLES: Dict[str, Callable[[], BaseEstimator]] = {
    "ensemble_boosting": _boosting_ensemble,
    "ensemble_classical": _classical_ensemble,
    "ensemble_catboost_iforest": _catboost_iforest_ensemble,
}

MODEL_NAMES = sorted([*_BASE, *_ENSEMBLES])

BINARY_ONLY = {"isolation_forest", "ddpg", "ensemble_classical", "ensemble_catboost_iforest"}


def build_model(name: str) -> BaseEstimator:
    """Return an unfitted model by registry name."""
    if name in _BASE:
        return _scaled(_BASE[name]())
    if name in _ENSEMBLES:
        return _ENSEMBLES[name]()
    raise KeyError(f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}")


def is_binary_only(name: str) -> bool:
    return name in BINARY_ONLY


__all__ = ["MODEL_NAMES", "build_model", "is_binary_only"]
