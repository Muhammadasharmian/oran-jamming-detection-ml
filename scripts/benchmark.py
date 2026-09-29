#!/usr/bin/env python3
"""Train every model (or a chosen subset) on the same split and compare them.

All models are evaluated on the same held-out test set. Multi-class models
are additionally scored on the binary jamming-vs-normal task so that they
can be compared with the binary-only detectors.

Example
-------
::

    python scripts/benchmark.py --data data/srsran-gnb-kpm-dataset --binary --output results/benchmark.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from jamming_detection.cli_utils import add_data_args, load_from_args, set_seed
from jamming_detection.data import split_dataset
from jamming_detection.evaluation import evaluate, measure_latency
from jamming_detection.models import MODEL_NAMES
from jamming_detection.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    add_data_args(parser)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--output", default="results/benchmark.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset = load_from_args(args)
    split = split_dataset(dataset, args.test_size, args.val_size, args.seed)
    print(
        f"Loaded {len(dataset)} samples, {len(dataset.feature_names)} features, classes {dataset.class_counts()}"
    )

    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        result = train_model(name, split, binary=args.binary, oversample=args.oversample, verbose=False)
        model, model_split = result["model"], result["split"]
        metrics = evaluate(model, model_split.X_test, model_split.y_test)
        latency = measure_latency(model, model_split.X_test, n_samples=100)
        row = {
            "model": name,
            "task": "binary" if result["binary"] else "multiclass",
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "mcc": metrics["mcc"],
            "binary_f1": metrics["binary"]["f1"],
            "false_alarm_rate": metrics["binary"]["false_alarm_rate"],
            "miss_rate": metrics["binary"]["miss_rate"],
            "roc_auc": metrics["binary"].get("roc_auc"),
            "train_seconds": result["train_seconds"],
            "latency_ms": latency["single_sample_ms_mean"],
        }
        rows.append(row)
        print(
            f"  accuracy {row['accuracy']:.4f} | F1 macro {row['f1_macro']:.4f} | "
            f"binary F1 {row['binary_f1']:.4f} | train {row['train_seconds']:.1f}s | "
            f"latency {row['latency_ms']:.2f} ms"
        )

    table = pd.DataFrame(rows).sort_values("binary_f1", ascending=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
