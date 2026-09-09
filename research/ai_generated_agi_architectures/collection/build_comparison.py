"""Build a consistent comparison and locate every quoted source in raw output."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIMENSIONS = ["memory", "reasoning_planning", "learning", "tools_actions", "world_self",
              "safety_governance", "evaluation", "persistence_recovery", "orchestration",
              "engineering_feasibility", "originality"]
analysis = json.loads((ROOT / "collection/analysis.json").read_text(encoding="utf-8"))
models = json.loads((ROOT / "collection/models.json").read_text(encoding="utf-8"))
rows = []
for model in models:
    slug = model["slug"]
    raw_path = ROOT / f"raw_outputs/{slug}.txt"
    raw = raw_path.read_text(encoding="utf-8")
    response = json.loads((ROOT / f"collection/runs/{slug}/response.json").read_text(encoding="utf-8"))
    entries = analysis[slug]
    assert len(entries) == len(DIMENSIONS), slug
    for dimension, (quote, assessment) in zip(DIMENSIONS, entries):
        assert quote and quote in raw, (slug, dimension, quote)
        line = raw[:raw.index(quote)].count("\n") + 1
        rows.append(dict(system=slug, model=model["model_id"], dimension=dimension,
                         output_quote=quote, analysis=assessment,
                         evidence=f"raw_outputs/{slug}.txt#L{line}",
                         termination=response["termination"]))
with (ROOT / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} source-checked comparison rows.")
