#!/usr/bin/env python3
"""Bayesian hyperparameter search with Optuna.

Searches the hyperparameters of CatBoost, LightGBM, Random Forest or the
Isolation Forest using stratified k-fold cross-validation on the training
partition only. The test partition is never touched. The best parameters
are written to JSON and can be passed back to ``train.py`` through
``--set model__<param>=<value>``.

Example
-------
::

    python scripts/tune.py --model catboost --data data/srsran-gnb-kpm-dataset --trials 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jamming_detection import config
from jamming_detection.cli_utils import add_data_args, load_from_args, set_seed
from jamming_detection.data import split_dataset
from jamming_detection.training import binarise

TUNABLE = ("catboost", "lightgbm", "random_forest", "isolation_forest")


def search_space(trial, name: str) -> dict:
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 40),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        }
    if name == "isolation_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    raise ValueError(name)


def build(name: str, params: dict):
    from jamming_detection.models import build_model

    model = build_model(name)
    if name == "isolation_forest":
        model.set_params(model__params=params)
    else:
        model.set_params(**{f"model__{k}": v for k, v in params.items()})
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=TUNABLE)
    add_data_args(parser)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None, help="JSON file for the best parameters.")
    args = parser.parse_args()

    import optuna
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    set_seed(args.seed)
    dataset = load_from_args(args)
    split = split_dataset(dataset, args.test_size, val_size=0.0, random_state=args.seed)
    X, y = split.X_train, split.y_train
    if args.binary or args.model == "isolation_forest":
        y = binarise(y)
    average = "binary" if len(np.unique(y)) == 2 else "macro"
    pos_label = "jamming" if average == "binary" else 1

    folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(X, y))

    def objective(trial) -> float:
        params = search_space(trial, args.model)
        scores = []
        for train_idx, val_idx in folds:
            model = build(args.model, params).fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            scores.append(f1_score(y[val_idx], pred, average=average, pos_label=pos_label, zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print(f"\nBest cross-validated F1 ({average}): {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    out = Path(args.output or f"results/{args.model}_best_params.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"model": args.model, "cv_f1": study.best_value, "params": study.best_params}, indent=2)
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
