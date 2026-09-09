"""Check packet completeness and internal evidence consistency without inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ("README.md", "prompts.md", "comparison.csv", "summary.md", "synthesis.md", "sources.md", "validation.md")
DIMENSIONS = {"memory", "reasoning_planning", "learning", "tools_actions", "world_self",
              "safety_governance", "evaluation", "persistence_recovery", "orchestration",
              "engineering_feasibility", "originality"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packet_files():
    return [p for p in sorted(ROOT.rglob("*")) if p.is_file()
            and p.name != "manifest.json" and "__pycache__" not in p.parts
            and not p.is_relative_to(ROOT / "collection/sources")]


def validate():
    for name in REQUIRED:
        assert (ROOT / name).stat().st_size > 0, name
    models = json.loads((ROOT / "collection/models.json").read_text())
    assert len(models) == 8 and len({m["model_id"] for m in models}) == 8
    raw = {}
    for model in models:
        slug = model["slug"]
        folder = ROOT / "collection/runs" / slug
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
        text = (ROOT / f"raw_outputs/{slug}.txt").read_text(encoding="utf-8")
        raw[slug] = text
        assert request["model"] == response["model"] == model
        prompt = (ROOT / "collection/prompt.txt").read_text(encoding="utf-8")
        assert request["messages"] == [{"role": "user", "content": prompt}]
        assert request["prompt_sha256"] == sha(prompt.encode())
        assert response["request_sha256"] == sha((folder / "request.json").read_bytes())
        assert response["collector_sha256"] == sha((ROOT / "collection/collect.py").read_bytes())
        assert text == response["raw_output"]
        assert response["output_sha256"] == sha(text.encode())
        assert response["generated_tokens"] == len(response["output_ids"]) > 0
        config = request["generation_config"]
        assert config["do_sample"] is False and config["max_new_tokens"] == 1400
        eos = config["eos_token_id"]
        eos = [eos] if isinstance(eos, int) else eos or []
        if response["termination"] == "eos":
            assert response["output_ids"][-1] in eos
        else:
            assert response["termination"] == "length" and response["generated_tokens"] == 1400
        files = json.loads((folder / "model-files.json").read_text())
        assert files["model"] == model
        assert any(f["file"].endswith(".safetensors") for f in files["files"])
        assert all(len(f["sha256"]) == 64 and f["bytes"] > 0 for f in files["files"])
    with (ROOT / "comparison.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 88
    for slug in raw:
        selected = [r for r in rows if r["system"] == slug]
        assert len(selected) == 11 and {r["dimension"] for r in selected} == DIMENSIONS
        for row in selected:
            assert row["output_quote"] in raw[slug] and len(row["analysis"]) >= 40
            line = raw[slug][:raw[slug].index(row["output_quote"])].count("\n") + 1
            assert row["evidence"] == f"raw_outputs/{slug}.txt#L{line}"
    experiment = json.loads((ROOT / "recovery-results-windows.json").read_text())
    assert experiment["script_sha256"] == sha((ROOT / "recovery_experiment.py").read_bytes())
    assert len(experiment["cases"]) == 14
    robust = [c for c in experiment["cases"] if c["strategy"] == "reconcile"]
    assert len(robust) == 7 and all(c["invariant_passed"] and c["second_recovery_unchanged"] for c in robust)
    baseline = [c for c in experiment["cases"] if c["strategy"] == "naive"]
    assert sum(c["invariant_passed"] for c in baseline) == 4
    print("Verified 8 attributed outputs, 88 source-linked rows, collection records and 14 recorded experiment cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    validate()
    current = {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in packet_files()}
    if args.write_manifest:
        (ROOT / "manifest.json").write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print("Wrote", len(current), "file checksums.")
    else:
        saved = json.loads((ROOT / "manifest.json").read_text())
        assert current == saved, "Packet bytes differ from manifest; inspect changes before regenerating."
        print("All", len(current), "packet file checksums match.")
