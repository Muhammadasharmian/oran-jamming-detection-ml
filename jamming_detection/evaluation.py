"""Evaluation metrics and inference latency measurement."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from jamming_detection.data.loader import NORMAL_LABEL


def _jamming_score(model: Any, X: np.ndarray, normal_label: str) -> Optional[np.ndarray]:
    """Probability that each sample is jammed, if the model exposes one."""
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
    except (AttributeError, NotImplementedError):
        return None
    classes = np.asarray(model.classes_).astype(str)
    return proba[:, classes != normal_label].sum(axis=1)


def evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    normal_label: str = NORMAL_LABEL,
) -> Dict[str, Any]:
    """Compute multi-class and binary (jamming vs normal) detection metrics."""
    y = np.asarray(y).astype(str)
    y_pred = np.asarray(model.predict(X)).astype(str)
    labels = sorted(set(y) | set(y_pred))

    results: Dict[str, Any] = {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, y_pred)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "report": classification_report(y, y_pred, labels=labels, digits=4, zero_division=0),
    }

    # Binary detection view: any jamming class counts as "jamming".
    y_bin = (y != normal_label).astype(int)
    pred_bin = (y_pred != normal_label).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()
    binary = {
        "accuracy": float(accuracy_score(y_bin, pred_bin)),
        "precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "miss_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }
    score = _jamming_score(model, X, normal_label)
    if score is not None and len(np.unique(y_bin)) == 2:
        binary["roc_auc"] = float(roc_auc_score(y_bin, score))
    results["binary"] = binary
    return results


def measure_latency(model: Any, X: np.ndarray, n_samples: int = 200, repeats: int = 3) -> Dict[str, float]:
    """Single-sample and batch inference latency in milliseconds."""
    X = np.asarray(X)
    idx = np.random.default_rng(0).choice(len(X), size=min(n_samples, len(X)), replace=False)
    model.predict(X[idx[:1]])  # warm-up

    single = []
    for i in idx:
        t0 = time.perf_counter()
        model.predict(X[i : i + 1])
        single.append((time.perf_counter() - t0) * 1e3)

    batch = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(X[idx])
        batch.append((time.perf_counter() - t0) * 1e3 / len(idx))

    return {
        "single_sample_ms_mean": float(np.mean(single)),
        "single_sample_ms_p95": float(np.percentile(single, 95)),
        "batched_ms_per_sample": float(np.median(batch)),
    }


def format_summary(name: str, metrics: Dict[str, Any]) -> str:
    b = metrics["binary"]
    lines = [
        f"Model: {name}",
        f"  samples           : {metrics['n_samples']}",
        f"  accuracy          : {metrics['accuracy']:.4f}",
        f"  F1 (macro)        : {metrics['f1_macro']:.4f}",
        f"  F1 (weighted)     : {metrics['f1_weighted']:.4f}",
        f"  MCC               : {metrics['mcc']:.4f}",
        "  Binary detection (jamming vs normal)",
        f"    accuracy        : {b['accuracy']:.4f}",
        f"    precision       : {b['precision']:.4f}",
        f"    recall          : {b['recall']:.4f}",
        f"    F1              : {b['f1']:.4f}",
        f"    false alarm rate: {b['false_alarm_rate']:.4f}",
        f"    miss rate       : {b['miss_rate']:.4f}",
    ]
    if "roc_auc" in b:
        lines.append(f"    ROC AUC         : {b['roc_auc']:.4f}")
    lines += ["", metrics["report"]]
    return "\n".join(lines)
