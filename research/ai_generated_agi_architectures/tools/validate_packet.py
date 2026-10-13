#!/usr/bin/env python3
"""Mechanical consistency checks for the research/ai_generated_agi_architectures
packet (Cognitive-OS issue #5).

Checks structure and cross-references only; it does not judge content quality.
Run from anywhere:

    python3 research/ai_generated_agi_architectures/tools/validate_packet.py

Exit code 0 if every check passes, 1 otherwise. With --report PATH, writes
the same output to a file.
"""
import csv
import re
import statistics
import sys
from pathlib import Path

PACKET = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "README.md", "prompts.md", "comparison.csv", "comparison_scores.csv",
    "scoring.md", "summary.md", "synthesis.md", "sources.md",
]
REQUIRED_DIMS = [
    "memory_architecture", "reasoning_planning", "learning_mechanism",
    "tool_use", "world_model", "safety_governance", "evaluation_strategy",
    "persistence_runtime", "multi_agent_design", "engineering_feasibility",
    "originality_insight",
]
# crude secret-pattern scan (acceptance criteria: no private keys/tokens)
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}", r"ghp_[A-Za-z0-9]{20,}", r"gho_[A-Za-z0-9]{20,}",
    r"xox[bap]-[A-Za-z0-9-]{10,}", r"AKIA[0-9A-Z]{16}",
    r"api[_-]?key\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
]


def main() -> int:
    out_lines = []
    failures = []

    def check(name, ok, detail=""):
        out_lines.append(f"{'PASS' if ok else 'FAIL'} - {name}"
                         + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    # 1. required files
    for fname in REQUIRED_FILES:
        check(f"required file present: {fname}", (PACKET / fname).is_file())

    # 2. raw outputs: at least 8, one per system
    raw_dir = PACKET / "raw_outputs"
    raw_files = sorted(raw_dir.glob("*.md")) if raw_dir.is_dir() else []
    check("raw_outputs/ has >= 8 proposals", len(raw_files) >= 8,
          f"found {len(raw_files)}")

    # 3. comparison.csv structure
    csv_path = PACKET / "comparison.csv"
    rows = []
    header = []
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            rows = list(reader)
    check("comparison.csv covers all 11 required dimensions",
          all(d in header for d in REQUIRED_DIMS),
          f"missing={[d for d in REQUIRED_DIMS if d not in header]}")
    check("comparison.csv has >= 8 systems", len(rows) >= 8,
          f"found {len(rows)}")
    empty = [(r.get("system"), c) for r in rows for c in header
             if not (r.get(c) or "").strip()]
    check("comparison.csv has no empty cells", not empty,
          f"empty={empty[:3]}" if empty else "")

    # 4. every CSV system has a matching raw output file
    systems = [r.get("system", "") for r in rows]
    missing_raw = [s for s in systems if not (raw_dir / f"{s}.md").is_file()]
    check("every comparison row has a raw_outputs file", not missing_raw,
          f"missing={missing_raw}" if missing_raw else "")

    # 5. sources.md mentions every model family
    sources = (PACKET / "sources.md").read_text(encoding="utf-8") \
        if (PACKET / "sources.md").is_file() else ""
    families = [r.get("model_family", "") for r in rows]
    missing_src = [f for f in families if f and f.split(" / ")[0].split(" (")[0]
                   not in sources]
    check("sources.md mentions every model family", not missing_src,
          f"missing={missing_src}" if missing_src else "")

    # 6. scores file: same systems, integers 1-5, means
    scores_path = PACKET / "comparison_scores.csv"
    srows = []
    if scores_path.is_file():
        with open(scores_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            srows = list(reader)
            sheader = reader.fieldnames or []
        check("comparison_scores.csv systems match comparison.csv",
              [r.get("system") for r in srows] == systems)
        check("comparison_scores.csv covers all 11 dimensions",
              all(d in sheader for d in REQUIRED_DIMS))
        bad = [(r.get("system"), c, r.get(c)) for r in srows
               for c in REQUIRED_DIMS
               if not re.fullmatch(r"[1-5]", (r.get(c) or "").strip())]
        check("comparison_scores.csv values are integers 1-5", not bad,
              f"bad={bad[:3]}" if bad else "")
    else:
        check("comparison_scores.csv present", False)

    if srows:
        out_lines.append("")
        out_lines.append("Per-system mean score (11 dims):")
        for r in sorted(srows,
                        key=lambda x: statistics.mean(int(x[d]) for d in REQUIRED_DIMS),
                        reverse=True):
            mean = statistics.mean(int(r[d]) for d in REQUIRED_DIMS)
            out_lines.append(f"  {r['system']:18s} {mean:.2f}")
        out_lines.append("Per-dimension mean score (10 systems):")
        for d in sorted(REQUIRED_DIMS,
                        key=lambda x: statistics.mean(int(r[x]) for r in srows),
                        reverse=True):
            mean = statistics.mean(int(r[d]) for r in srows)
            out_lines.append(f"  {d:22s} {mean:.2f}")

    # 7. no private secrets in the packet
    hits = []
    for path in list(PACKET.rglob("*.md")) + list(PACKET.rglob("*.csv")) \
            + list(PACKET.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                hits.append(f"{path.relative_to(PACKET)}:{pat[:12]}")
                break
    check("no private API keys/tokens in packet", not hits,
          f"hits={hits}" if hits else "")

    out_lines.append("")
    if failures:
        out_lines.append(f"RESULT: {len(failures)} check(s) FAILED: "
                         + ", ".join(failures))
    else:
        out_lines.append("RESULT: all checks passed")
    report = "\n".join(out_lines) + "\n"
    print(report, end="")

    if "--report" in sys.argv:
        idx = sys.argv.index("--report")
        Path(sys.argv[idx + 1]).write_text(report, encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
