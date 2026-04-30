from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.open_task_benchmark.core import create_task_packages, load_benchmark_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate open-task benchmark packages.")
    parser.add_argument("--config", default=Path(__file__).with_name("benchmark_config.json").as_posix())
    parser.add_argument("--output-dir", default="runtime/reports/open_task_benchmark")
    parser.add_argument("--mode", choices=["package-only"], default="package-only")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_benchmark_config(args.config)
    if args.mode == "package-only":
        summary = create_task_packages(config, args.output_dir, seed=args.seed, limit=args.limit)
    else:  # pragma: no cover - argparse prevents this.
        raise ValueError(f"unsupported mode: {args.mode}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

