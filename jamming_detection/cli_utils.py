"""Argument parsing helpers shared by the command-line scripts."""

from __future__ import annotations

import argparse
import random

import numpy as np

from jamming_detection import config
from jamming_detection.data.loader import Dataset, load_csvs


def add_data_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="CSV files or directories. Without --label-column, each file is one class "
        "named after the file (files starting with 'normal' map to 'normal').",
    )
    group.add_argument(
        "--label-column", default=None, help="Column that holds the class label (single-file layout)."
    )
    group.add_argument(
        "--features",
        nargs="+",
        default=None,
        help="Explicit feature columns. Defaults to all numeric non-identifier columns.",
    )
    group.add_argument(
        "--binary", action="store_true", help="Collapse all jamming classes into a single 'jamming' class."
    )
    group.add_argument("--test-size", type=float, default=config.DATA["test_size"])
    group.add_argument("--val-size", type=float, default=config.DATA["val_size"])
    group.add_argument("--seed", type=int, default=config.RANDOM_STATE)


def load_from_args(args: argparse.Namespace) -> Dataset:
    return load_csvs(args.data, label_column=args.label_column, feature_columns=args.features)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
