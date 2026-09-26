"""Command-line entry point for the end-to-end experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the California Housing ML portfolio project")
    parser.add_argument("--output", default="artifacts", help="Directory for metrics, models, and charts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip the unsupervised analysis")
    args = parser.parse_args()
    result = run_experiment(Path(args.output), seed=args.seed, clustering=not args.skip_clustering)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
