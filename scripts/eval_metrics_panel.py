#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.metrics_panel import (  # noqa: E402
    build_eval_metrics_panel_from_paths,
    render_eval_metrics_panel,
)


def _default_input_paths() -> list[Path]:
    return [ROOT / "runtime", ROOT / "reports", ROOT / "audit"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Cognitive OS evaluation metrics panel from audit JSON files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Audit JSON/JSONL files or directories. Defaults to runtime/, reports/, and audit/.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the panel JSON.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "both"),
        default="text",
        help="Console output format.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_paths = [Path(path) for path in args.paths] if args.paths else _default_input_paths()
    panel = build_eval_metrics_panel_from_paths(input_paths)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(panel, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if args.format in {"text", "both"}:
        print(render_eval_metrics_panel(panel))
    if args.format in {"json", "both"}:
        print(json.dumps(panel, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
