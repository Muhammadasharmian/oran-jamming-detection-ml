#!/usr/bin/env python3
"""Train and evaluate a single jamming detection model.

Examples
--------
Train the CatBoost + LightGBM + ExtraTrees ensemble on the srsRAN KPM
dataset (one CSV per class)::

    python scripts/train.py --model ensemble_boosting --data data/srsran-gnb-kpm-dataset

Train a Random Forest on a single CSV with a label column::

    python scripts/train.py --model random_forest --data traffic.csv --label-column scenario
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jamming_detection.cli_utils import add_data_args, load_from_args, set_seed
from jamming_detection.data import split_dataset
from jamming_detection.evaluation import evaluate, format_summary, measure_latency
from jamming_detection.models import MODEL_NAMES
from jamming_detection.training import save_artifact, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    add_data_args(parser)
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Balance the training partition with SMOTE (never applied to val/test).",
    )
    parser.add_argument(
        "--no-tune-ensemble",
        action="store_true",
        help="Keep uniform ensemble weights instead of fitting them on the validation set.",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override estimator parameters, e.g. model__iterations=500.",
    )
    parser.add_argument("--output", default="artifacts", help="Directory for the saved model and metrics.")
    parser.add_argument("--latency", action="store_true", help="Also measure inference latency.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    verbose = not args.quiet

    dataset = load_from_args(args)
    if verbose:
        print(f"Loaded {len(dataset)} samples, {len(dataset.feature_names)} features")
        print(f"Class counts: {dataset.class_counts()}")

    split = split_dataset(dataset, args.test_size, args.val_size, args.seed)
    result = train_model(
        args.model,
        split,
        binary=args.binary,
        oversample=args.oversample,
        tune_ensemble=not args.no_tune_ensemble,
        model_overrides=parse_overrides(args.set),
        verbose=verbose,
    )
    model, split = result["model"], result["split"]

    metrics = evaluate(model, split.X_test, split.y_test)
    metrics["train_seconds"] = result["train_seconds"]
    metrics["tuned"] = result["tuned"]
    if args.latency:
        metrics["latency"] = measure_latency(model, split.X_test)

    print()
    print(format_summary(args.model, metrics))
    print(f"Training time: {result['train_seconds']:.1f} s")
    if "latency" in metrics:
        print(f"Latency: {metrics['latency']}")

    path = save_artifact(
        Path(args.output) / f"{args.model}.joblib",
        args.model,
        model,
        split.feature_names,
        result["binary"],
        metrics,
    )
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()
