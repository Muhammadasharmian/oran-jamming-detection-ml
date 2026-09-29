#!/usr/bin/env python3
"""Convert srsRAN gNB console logs into KPM CSV files.

Examples
--------
Convert every log in a directory into one combined CSV::

    python scripts/parse_srsran_logs.py raw_logs/*.txt --output data/kpm_combined.csv

Group logs by scenario, producing one CSV per class that ``train.py`` can
consume directly::

    python scripts/parse_srsran_logs.py \\
        --group normal=raw/normal_*.txt \\
        --group power_jamming=raw/power_*.txt \\
        --group sweep_jamming=raw/sweep_*.txt \\
        --output-dir data/kpm
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from jamming_detection.data.srsran_parser import parse_logs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("logs", nargs="*", help="Log files to combine into a single CSV.")
    parser.add_argument("--output", default="kpm_combined.csv", help="Output CSV for positional logs.")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=GLOB",
        help="Write the logs matching GLOB to <output-dir>/NAME.csv. Repeatable.",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for grouped CSVs.")
    return parser.parse_args()


def _report(frame, path: Path) -> None:
    print(f"  {path}: {len(frame)} rows from {frame['source_file'].nunique() if len(frame) else 0} file(s)")


def main() -> None:
    args = parse_args()
    if not args.logs and not args.group:
        raise SystemExit("Provide log files and/or --group NAME=GLOB arguments.")

    if args.logs:
        frame = parse_logs(sorted(args.logs))
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        _report(frame, out)

    out_dir = Path(args.output_dir)
    for spec in args.group:
        name, _, pattern = spec.partition("=")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  warning: no files match '{pattern}' for group '{name}'")
            continue
        frame = parse_logs(files)
        out = out_dir / f"{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        _report(frame, out)


if __name__ == "__main__":
    main()
