"""Default hyperparameters for every model in the package.

All values here are plain model settings. None of them are tied to a
particular dataset or tuned towards a specific score; use
``scripts/tune.py`` to search hyperparameters and ensemble weights for
your own data.
"""

from __future__ import annotations

from typing import Any, Dict

RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Classical models
# ---------------------------------------------------------------------------

RANDOM_FOREST: Dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

SVM: Dict[str, Any] = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "probability": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

ISOLATION_FOREST: Dict[str, Any] = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "max_features": 1.0,
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Gradient boosting / tree ensembles
# ---------------------------------------------------------------------------

CATBOOST: Dict[str, Any] = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_STATE,
    "thread_count": -1,
    "verbose": False,
}

LIGHTGBM: Dict[str, Any] = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 10,
    "num_leaves": 128,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbosity": -1,
}

EXTRA_TREES: Dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 15,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# Neural networks (PyTorch)
# ---------------------------------------------------------------------------

MLP: Dict[str, Any] = {
    "hidden_dims": (64, 32),
    "dropout": 0.2,
    "batch_norm": False,
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
}

DEEP_MLP: Dict[str, Any] = {
    "hidden_dims": (128, 64, 32),
    "dropout": (0.3, 0.2),
    "batch_norm": True,
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
}

CNN_LSTM: Dict[str, Any] = {
    "conv_channels": (16,),
    "lstm_hidden": 32,
    "lstm_layers": 1,
    "bidirectional": False,
    "epochs": 20,
    "batch_size": 32,
    "lr": 1e-3,
}

BI_CNN_LSTM: Dict[str, Any] = {
    "conv_channels": (32, 64),
    "lstm_hidden": 64,
    "lstm_layers": 2,
    "bidirectional": True,
    "epochs": 25,
    "batch_size": 32,
    "lr": 5e-4,
}

# ---------------------------------------------------------------------------
# Deep reinforcement learning (DDPG)
# ---------------------------------------------------------------------------

DDPG: Dict[str, Any] = {
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.0,  # samples are i.i.d. in the dataset environment (contextual bandit)
    "tau": 0.005,
    "batch_size": 128,
    "replay_buffer_size": 300_000,
    "hidden_dims": (256, 256),
    "noise_theta": 0.15,
    "noise_sigma": 0.2,
    "warmup_steps": 2000,
    "reward_clip": 2.0,
    "total_steps": 50_000,
    "episode_length": 1000,
}

# ---------------------------------------------------------------------------
# Data handling
# ---------------------------------------------------------------------------

DATA: Dict[str, Any] = {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": RANDOM_STATE,
}
