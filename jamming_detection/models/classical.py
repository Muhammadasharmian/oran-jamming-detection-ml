"""Classical machine learning detectors: Random Forest, SVM and Isolation Forest."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from jamming_detection import config
from jamming_detection.data.loader import JAMMING_LABEL, NORMAL_LABEL


def build_random_forest(**overrides: Any) -> RandomForestClassifier:
    return RandomForestClassifier(**{**config.RANDOM_FOREST, **overrides})


def build_svm(**overrides: Any) -> SVC:
    return SVC(**{**config.SVM, **overrides})


class IsolationForestDetector(BaseEstimator, ClassifierMixin):
    """Isolation Forest wrapped as a binary ``normal`` / ``jamming`` classifier.

    The forest is fitted on normal traffic only (novelty detection), so the
    model learns what an unjammed link looks like and flags deviations. If
    no labels are provided it is fitted on all samples.

    ``decision_function`` follows scikit-learn: larger values are more
    normal. A sample is classified as jamming when its score falls below
    ``threshold`` (0 by default, the forest's own boundary). The threshold
    can be tuned on held-out data with :meth:`fit_threshold`.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        fit_on_normal_only: bool = True,
        threshold: float = 0.0,
        normal_label: str = NORMAL_LABEL,
    ) -> None:
        self.params = params
        self.fit_on_normal_only = fit_on_normal_only
        self.threshold = threshold
        self.normal_label = normal_label

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> IsolationForestDetector:
        X = np.asarray(X, dtype=np.float64)
        X_fit = X
        if y is not None and self.fit_on_normal_only:
            mask = np.asarray(y) == self.normal_label
            if mask.any():
                X_fit = X[mask]
        self.model_ = IsolationForest(**{**config.ISOLATION_FOREST, **(self.params or {})})
        self.model_.fit(X_fit)
        scores = self.model_.decision_function(X_fit)
        self.score_scale_ = float(np.std(scores)) or 1.0
        self.classes_ = np.array([JAMMING_LABEL, self.normal_label])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(np.asarray(X, dtype=np.float64))

    def jamming_probability(self, X: np.ndarray) -> np.ndarray:
        """Logistic mapping of the anomaly score to P(jamming)."""
        z = (self.decision_function(X) - self.threshold) / self.score_scale_
        return 1.0 / (1.0 + np.exp(z))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_jam = self.jamming_probability(X)
        return np.column_stack([p_jam, 1.0 - p_jam])  # order matches classes_

    def predict(self, X: np.ndarray) -> np.ndarray:
        is_jam = self.decision_function(X) < self.threshold
        return np.where(is_jam, JAMMING_LABEL, self.normal_label)

    def fit_threshold(self, X_val: np.ndarray, y_val: np.ndarray, n_candidates: int = 99) -> float:
        """Choose the score threshold that maximises jamming F1 on validation data."""
        scores = self.decision_function(X_val)
        y_true = (np.asarray(y_val) != self.normal_label).astype(int)
        candidates = np.percentile(scores, np.linspace(1, 99, n_candidates))
        best_f1, best_t = -1.0, self.threshold
        for t in candidates:
            f1 = f1_score(y_true, (scores < t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t
        return best_t
