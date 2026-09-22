"""Small, dependency-free validator for the AGI research packet.

The validator deliberately checks structure and provenance hygiene, not whether
an architecture is correct.  It is useful in CI when replacing reproductions
with authenticated captures.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

EXPECTED_IDS = {
    "openai_gpt", "anthropic_claude", "google_gemini", "xai_grok",
    "deepseek", "qwen", "meta_llama", "mistral",
}
DIMENSIONS = {
    "memory_architecture", "reasoning_planning_loop",
    "learning_self_improvement", "tool_use_action_execution",
    "world_model_representation", "safety",
}
LABELS = {"explicit", "partial", "absent", "unclear"}
SECRET_PATTERNS = (re.compile(r"sk-[A-Za-z0-9]{20,}"), re.compile(r"api_key\s*="))


def validate_packet(root: Path) -> list[str]:
    """Return human-readable validation errors; return an empty list if valid."""
    errors: list[str] = []
    required = {"README.md", "prompts.md", "summary.md", "synthesis.md", "sources.md", "comparison.csv"}
    errors.extend(f"missing file: {name}" for name in sorted(required - {p.name for p in root.iterdir()}))
    raw_dir = root / "raw_outputs"
    if not raw_dir.is_dir():
        return errors + ["missing directory: raw_outputs"]

    raw_ids = {p.stem for p in raw_dir.glob("*.md")}
    if raw_ids != EXPECTED_IDS:
        errors.append(f"raw output IDs differ: {sorted(raw_ids ^ EXPECTED_IDS)}")

    comparison = root / "comparison.csv"
    if comparison.is_file():
        with comparison.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if set(rows[0]) - (DIMENSIONS | {"system_id", "model_family"}) if rows else True:
            errors.append("comparison.csv has no usable header")
        if len(rows) != len(EXPECTED_IDS):
            errors.append("comparison.csv must have one row per expected model family")
        if {row.get("system_id") for row in rows} != {i.replace("_", "-") for i in EXPECTED_IDS}:
            errors.append("comparison.csv system IDs do not match the packet")
        for row in rows:
            for dimension in DIMENSIONS:
                if row.get(dimension) not in LABELS:
                    errors.append(f"invalid {dimension} label for {row.get('system_id')}")

    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".csv", ".toml"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                errors.append(f"possible secret material: {path.relative_to(root)}")
    return errors


if __name__ == "__main__":
    packet_root = Path(__file__).parent
    problems = validate_packet(packet_root)
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"valid: {packet_root}")
