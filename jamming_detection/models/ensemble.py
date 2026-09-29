"""Weighted ensembles.

``WeightedVotingEnsemble``
    Soft-voting ensemble of supervised classifiers over the full label set
    (binary or multi-class). Used for the CatBoost + LightGBM + ExtraTrees
    ensemble.

``JammingScoreEnsemble``
    Binary detector that fuses each member's probability of jamming. Members
    may be supervised classifiers (their non-normal class probabilities are
    summed) or the unsupervised :class:`IsolationForestDetector`. Used for the
    Random Forest + SVM + Isolation Forest ensemble and the CatBoost +
    Isolation Forest ensemble.

Ensemble weights start uniform. Call ``fit_weights`` with a validation set
to search the weight simplex (and, for ``JammingScoreEnsemble``, the
decision threshold) instead of hand-picking values.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from jamming_detection.data.loader import JAMMING_LABEL, NORMAL_LABEL


def _simplex_grid(n: int, step: float) -> Iterable[np.ndarray]:
    """All weight vectors of length ``n`` on a grid of ``step`` that sum to 1."""
    ticks = int(round(1.0 / step))
    for combo in itertools.product(range(ticks + 1), repeat=n - 1):
        rest = ticks - sum(combo)
        if rest >= 0:
            yield np.array(list(combo) + [rest], dtype=float) / ticks


def _normalise(weights: Dict[str, float], names: List[str]) -> Dict[str, float]:
    total = sum(weights[n] for n in names)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    return {n: weights[n] / total for n in names}


class WeightedVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted soft-voting ensemble of supervised classifiers."""

    def __init__(self, estimators: Dict[str, BaseEstimator], weights: Optional[Dict[str, float]] = None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> WeightedVotingEnsemble:
        self.classes_ = np.unique(y)
        self.estimators_: Dict[str, BaseEstimator] = {}
        for name, est in self.estimators.items():
            if verbose:
                print(f"  fitting ensemble member: {name}")
            self.estimators_[name] = clone(est).fit(X, y)
        names = list(self.estimators_)
        self.weights_ = _normalise(self.weights or {n: 1.0 for n in names}, names)
        return self

    def _member_proba(self, name: str, X: np.ndarray) -> np.ndarray:
        """Member probabilities re-ordered to ``self.classes_``."""
        est = self.estimators_[name]
        proba = est.predict_proba(X)
        order = [list(np.asarray(est.classes_).astype(str)).index(str(c)) for c in self.classes_]
        return proba[:, order]

    def _combine(self, probas: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        return sum(weights[n] * probas[n] for n in probas)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = {n: self._member_proba(n, X) for n in self.estimators_}
        return self._combine(probas, self.weights_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def fit_weights(
        self, X_val: np.ndarray, y_val: np.ndarray, step: float = 0.05, average: str = "macro"
    ) -> Dict[str, float]:
        """Grid-search member weights that maximise F1 on validation data."""
        names = list(self.estimators_)
        probas = {n: self._member_proba(n, X_val) for n in names}
        best_score, best_w = -1.0, self.weights_
        for vec in _simplex_grid(len(names), step):
            w = dict(zip(names, vec))
            pred = self.classes_[np.argmax(self._combine(probas, w), axis=1)]
            score = f1_score(y_val, pred, average=average, zero_division=0)
            if score > best_score:
                best_score, best_w = score, w
        self.weights_ = best_w
        return dict(best_w)


class JammingScoreEnsemble(BaseEstimator, ClassifierMixin):
    """Binary jamming detector that fuses members' P(jamming)."""

    def __init__(
        self,
        estimators: Dict[str, BaseEstimator],
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        normal_label: str = NORMAL_LABEL,
    ):
        self.estimators = estimators
        self.weights = weights
        self.threshold = threshold
        self.normal_label = normal_label

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> JammingScoreEnsemble:
        self.classes_ = np.array([JAMMING_LABEL, self.normal_label])
        self.estimators_: Dict[str, BaseEstimator] = {}
        for name, est in self.estimators.items():
            if verbose:
                print(f"  fitting ensemble member: {name}")
            self.estimators_[name] = clone(est).fit(X, y)
        names = list(self.estimators_)
        self.weights_ = _normalise(self.weights or {n: 1.0 for n in names}, names)
        self.threshold_ = float(self.threshold)
        return self

    def _member_jamming_prob(self, name: str, X: np.ndarray) -> np.ndarray:
        est = self.estimators_[name]
        final = est[-1] if isinstance(est, Pipeline) else est
        if hasattr(final, "jamming_probability"):
            X_t = est[:-1].transform(X) if isinstance(est, Pipeline) else X
            return final.jamming_probability(X_t)
        proba = est.predict_proba(X)
        classes = np.asarray(est.classes_).astype(str)
        normal_mask = classes == self.normal_label
        return proba[:, ~normal_mask].sum(axis=1)

    def jamming_probability(self, X: np.ndarray) -> np.ndarray:
        return sum(self.weights_[n] * self._member_jamming_prob(n, X) for n in self.estimators_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.jamming_probability(X)
        return np.column_stack([p, 1.0 - p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.jamming_probability(X) > self.threshold_, JAMMING_LABEL, self.normal_label)

    def fit_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        step: float = 0.05,
        thresholds: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Jointly search member weights and decision threshold for best jamming F1."""
        if thresholds is None:
            thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
        names = list(self.estimators_)
        y_true = (np.asarray(y_val) != self.normal_label).astype(int)
        member = np.column_stack([self._member_jamming_prob(n, X_val) for n in names])
        best = (-1.0, self.weights_, self.threshold_)
        for vec in _simplex_grid(len(names), step):
            score = member @ vec
            for t in thresholds:
                f1 = f1_score(y_true, (score > t).astype(int), zero_division=0)
                if f1 > best[0]:
                    best = (f1, dict(zip(names, vec)), float(t))
        _, self.weights_, self.threshold_ = best
        return {**self.weights_, "threshold": self.threshold_}
