#!/usr/bin/env python3
"""Deterministic acceptance checks for the issue #5 research packet."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "prompts.md",
    "comparison.csv",
    "summary.md",
    "synthesis.md",
    "sources.md",
    "PROVENANCE_MANIFEST.json",
]

DIMENSIONS = [
    "memory_architecture",
    "reasoning_planning_loop",
    "learning_self_improvement",
    "tool_use_action",
    "world_model",
    "safety_governance",
    "evaluation_benchmarks",
    "persistence_runtime",
    "multi_agent_orchestration",
    "engineering_feasibility",
    "originality_non_obvious",
]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"][A-Za-z0-9_./+-]{16,}['\"]"),
]


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    for name in REQUIRED_FILES:
        path = ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"missing or empty required file: {name}")

    with (ROOT / "comparison.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if len(rows) < 8:
        fail(f"comparison.csv has {len(rows)} systems; need at least 8")

    fields = set(rows[0].keys()) if rows else set()
    missing_dimensions = [name for name in DIMENSIONS if name not in fields]
    if missing_dimensions:
        fail(f"comparison.csv missing dimensions: {', '.join(missing_dimensions)}")

    sources = (ROOT / "sources.md").read_text(encoding="utf-8")
    provenance = json.loads((ROOT / "PROVENANCE_MANIFEST.json").read_text(encoding="utf-8"))
    prompt_path = ROOT / provenance.get("shared_prompt_file", "")
    if not prompt_path.is_file():
        fail("provenance manifest shared prompt is missing")
    prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if prompt_hash != provenance.get("shared_prompt_sha256"):
        fail("shared prompt hash does not match provenance manifest")
    entries = {entry.get("raw_file"): entry for entry in provenance.get("entries", [])}
    if provenance.get("counted_systems") != 8:
        fail("provenance manifest must declare exactly 8 counted systems")
    referenced_raw: list[Path] = []

    for index, row in enumerate(rows, start=1):
        system = (row.get("system") or "").strip()
        raw_rel = (row.get("file") or "").strip()
        if not system or not raw_rel:
            fail(f"comparison row {index} is missing system/file")
        for dimension in DIMENSIONS:
            if not (row.get(dimension) or "").strip():
                fail(f"{system}: empty comparison dimension {dimension}")

        raw_path = ROOT / raw_rel
        if not raw_path.is_file():
            fail(f"{system}: referenced raw output is missing: {raw_rel}")
        if raw_path.stat().st_size < 500:
            fail(f"{system}: raw output is unexpectedly small: {raw_rel}")
        if Path(raw_rel).name not in sources:
            fail(f"{system}: sources.md does not reference {Path(raw_rel).name}")
        entry = entries.get(raw_rel)
        if not entry:
            fail(f"{system}: provenance manifest does not reference {raw_rel}")
        if entry.get("counted") is not True or entry.get("status") != "usable":
            fail(f"{system}: provenance entry is not marked counted/usable")
        if entry.get("system") != system:
            fail(f"{system}: provenance system label mismatch")
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if digest != entry.get("sha256"):
            fail(f"{system}: raw output hash does not match provenance manifest")
        if raw_path.stat().st_size != entry.get("bytes"):
            fail(f"{system}: raw output byte count does not match provenance manifest")
        referenced_raw.append(raw_path)

    if len({p.resolve() for p in referenced_raw}) != len(referenced_raw):
        fail("comparison.csv reuses a raw output for multiple systems")

    counted_entries = [e for e in provenance.get("entries", []) if e.get("counted") is True]
    if {e.get("raw_file") for e in counted_entries} != {str(p.relative_to(ROOT)) for p in referenced_raw}:
        fail("provenance counted set does not exactly match comparison.csv")
    failed = entries.get("raw_outputs/deepseek_r1_8b.md")
    if not failed or failed.get("counted") is not False or failed.get("status") != "empty_no_final_answer":
        fail("DeepSeek failed attempt is not transparently marked non-counted")

    prompt_text = (ROOT / "prompts.md").read_text(encoding="utf-8").lower()
    if "prompt" not in prompt_text:
        fail("prompts.md does not document the collection prompt")

    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [*(ROOT / n for n in REQUIRED_FILES), *referenced_raw]
    )
    for pattern in SECRET_PATTERNS:
        if pattern.search(combined):
            fail(f"possible secret/private key material matched pattern: {pattern.pattern}")

    print("PASS: issue #5 research packet acceptance checks")
    print(f"  systems compared: {len(rows)}")
    print(f"  required dimensions per system: {len(DIMENSIONS)}")
    print(f"  distinct referenced raw outputs: {len(referenced_raw)}")
    print("  required files: present and non-empty")
    print("  provenance references + SHA-256 hashes: verified for every compared raw output")
    print("  failed DeepSeek attempt: explicitly non-counted")
    print("  secret-pattern scan: no matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
