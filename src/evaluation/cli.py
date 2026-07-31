"""Shared CLI for running a goal over a subset of configs.

Goal 1A and Goal 2 default to their full config lists and write the canonical result CSVs.
Restricting them to a subset (e.g. the PCA-256 variants that the forecasting main set runs
on) must not silently land in those same files, so --configs requires an explicit --out.
"""

from __future__ import annotations

import argparse


def parse_config_subset_args(description: str, default_out: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--configs", nargs="+", metavar="KEY", default=None,
        help='Config keys to evaluate (default: the goal\'s full list). Accepts the '
             'reduced variants, e.g. --configs te3_atc_path_d256. Requires --out.')
    parser.add_argument(
        "--out", default=default_out, metavar="NAME",
        help=f"Output CSV name under results/ (default: {default_out}).")
    args = parser.parse_args()

    if args.configs and args.out == default_out:
        parser.error(f"--configs requires --out so the canonical {default_out} is not "
                     f"overwritten with a partial run, e.g. --out {default_out.replace('.csv', '_pca256.csv')}")
    return args
