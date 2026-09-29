"""Fast smoke tests on a small synthetic dataset.

The synthetic data only exercises the code paths (fit, predict, persist);
it says nothing about detection performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jamming_detection.data import load_csvs, split_dataset
from jamming_detection.data.srsran_parser import parse_line, parse_suffixed
from jamming_detection.evaluation import evaluate
from jamming_detection.models import MODEL_NAMES
from jamming_detection.training import load_artifact, save_artifact, train_model

FAST_OVERRIDES = {
    "catboost": {"model__iterations": 30},
    "lightgbm": {"model__n_estimators": 30},
    "extra_trees": {"model__n_estimators": 20},
    "mlp": {"model__epochs": 2},
    "deep_mlp": {"model__epochs": 2},
    "cnn_lstm": {"model__epochs": 2},
    "bi_cnn_lstm": {"model__epochs": 2},
    "ddpg": {
        "model__total_steps": 400,
        "model__params": {"warmup_steps": 100, "batch_size": 32, "episode_length": 100},
    },
}


@pytest.fixture(scope="module")
def csv_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("kpm")
    specs = {"normal_data": (0.0, 300), "power_jamming": (2.0, 120), "sweep_jamming": (-2.0, 120)}
    for name, (shift, n) in specs.items():
        frame = pd.DataFrame(rng.normal(shift, 1.0, size=(n, 6)), columns=[f"f{i}" for i in range(6)])
        frame["rnti"] = "4601"
        frame["source_file"] = name
        frame.to_csv(root / f"{name}.csv", index=False)
    return root


def test_loader_labels_from_filenames(csv_dir):
    ds = load_csvs([csv_dir])
    assert set(ds.classes) == {"normal", "power_jamming", "sweep_jamming"}
    assert "rnti" not in ds.feature_names and "source_file" not in ds.feature_names
    assert ds.X.shape == (540, 6)


def test_parser_handles_suffixes_and_rows():
    assert parse_suffixed("47M") == 47e6
    assert parse_suffixed("685k") == 685e3
    assert parse_suffixed("ovl") is None
    row = parse_line(
        "   1 4601 |  15 1.0   27    47M 1196    1   0%   272k |  32.3 -19.5   1   27   685k"
        "   68    0   0%      0   170n   24"
    )
    assert row is not None and row["dl_cqi"] == 15 and row["phr"] == 24


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_trains_and_predicts(name, csv_dir, tmp_path):
    split = split_dataset(load_csvs([csv_dir]), test_size=0.25, val_size=0.15)
    result = train_model(name, split, model_overrides=FAST_OVERRIDES.get(name), verbose=False)
    model, model_split = result["model"], result["split"]

    metrics = evaluate(model, model_split.X_test, model_split.y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    path = save_artifact(tmp_path / f"{name}.joblib", name, model, split.feature_names, result["binary"])
    reloaded = load_artifact(path)["model"]
    np.testing.assert_array_equal(reloaded.predict(model_split.X_test), model.predict(model_split.X_test))
