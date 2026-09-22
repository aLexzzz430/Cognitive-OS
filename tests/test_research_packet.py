from pathlib import Path
import csv
import re
import sys

ROOT = Path(__file__).parents[1] / "research" / "ai_generated_agi_architectures"
sys.path.insert(0, str(ROOT))
from validate_packet import validate_packet
EXPECTED = {"openai_gpt", "anthropic_claude", "google_gemini", "xai_grok", "deepseek", "qwen", "meta_llama", "mistral"}
DIMS = {"memory_architecture", "reasoning_planning_loop", "learning_self_improvement", "tool_use_action_execution", "world_model_representation", "safety"}

def test_packet_files_and_raw_outputs_exist():
    for name in ("README.md", "prompts.md", "summary.md", "synthesis.md", "sources.md", "comparison.csv"):
        assert (ROOT / name).is_file()
    assert {p.stem for p in (ROOT / "raw_outputs").glob("*.md")} == EXPECTED

def test_comparison_is_rectangular_and_controlled():
    with (ROOT / "comparison.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and DIMS <= set(rows[0])
    assert len({r["system_id"] for r in rows}) == 8
    for row in rows:
        assert all(row[d] in {"explicit", "partial", "absent", "unclear"} for d in DIMS)

def test_raw_outputs_cover_topics():
    markers = ("Memory", "Reasoning", "Learning", "Tools", "World model", "safety")
    for path in (ROOT / "raw_outputs").glob("*.md"):
        text = path.read_text().lower()
        assert all(marker.lower() in text for marker in markers), path

def test_no_obvious_secret_material():
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".csv", ".toml"}:
            text = path.read_text().lower()
            assert not re.search(r"sk-[A-Za-z0-9]{20,}", text)
            assert "api_key=" not in text


def test_validator_accepts_current_packet():
    assert validate_packet(ROOT) == []


def test_validator_rejects_missing_raw_output(tmp_path):
    import shutil

    shutil.copytree(ROOT, tmp_path / "packet")
    (tmp_path / "packet" / "raw_outputs" / "qwen.md").unlink()
    errors = validate_packet(tmp_path / "packet")
    assert any("raw output IDs differ" in error for error in errors)


def test_validator_rejects_invalid_dimension_label(tmp_path):
    import shutil

    shutil.copytree(ROOT, tmp_path / "packet")
    comparison = tmp_path / "packet" / "comparison.csv"
    text = comparison.read_text().replace('"explicit"', '"invented"', 1)
    comparison.write_text(text)
    errors = validate_packet(tmp_path / "packet")
    assert any("invalid memory_architecture" in error for error in errors)
