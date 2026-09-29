#!/usr/bin/env python3
"""Run a trained model on new KPM data.

The input may be a CSV produced by ``parse_srsran_logs.py`` (or any CSV
with the model's feature columns) or a raw srsRAN gNB console log.

Examples
--------
::

    python scripts/predict.py --model artifacts/ensemble_boosting.joblib --input new_capture.csv
    python scripts/predict.py --model artifacts/catboost.joblib --input gnb_console.log --output preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from jamming_detection.data.srsran_parser import parse_log
from jamming_detection.training import load_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to a .joblib artifact from train.py.")
    parser.add_argument("--input", required=True, help="CSV file or raw srsRAN gNB log.")
    parser.add_argument("--output", default=None, help="Optional CSV to write predictions to.")
    return parser.parse_args()


def load_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return parse_log(path)


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.model)
    model, features = artifact["model"], artifact["feature_names"]

    frame = load_input(Path(args.input))
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise SystemExit(f"Input is missing feature columns required by the model: {missing}")

    X = frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0).to_numpy(dtype=np.float64)

    frame["prediction"] = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_).astype(str)
        frame["jamming_probability"] = proba[:, classes != "normal"].sum(axis=1)

    counts = frame["prediction"].value_counts().to_dict()
    print(f"Model: {artifact['name']}  |  {len(frame)} samples  |  predictions: {counts}")

    if args.output:
        frame.to_csv(args.output, index=False)
        print(f"Wrote predictions to {args.output}")
    else:
        cols = [c for c in ("rnti", "prediction", "jamming_probability") if c in frame.columns]
        print(frame[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
