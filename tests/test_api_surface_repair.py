from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.runtime.api_surface import (
    extract_attribute_errors,
    extract_python_api_surface,
    plan_repair_targets_for_validation,
)


def test_extract_python_api_surface_indexes_public_class_methods(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "sentiment_analyzer"
    source.mkdir(parents=True)
    (source / "analyzer.py").write_text(
        "\n".join(
            [
                "class SentimentAnalyzer:",
                "    def analyze_sentiment(self, text: str) -> dict:",
                "        return {'label': 'positive'}",
                "",
                "    def _internal(self) -> None:",
                "        pass",
                "",
                "def build_analyzer() -> SentimentAnalyzer:",
                "    return SentimentAnalyzer()",
            ]
        ),
        encoding="utf-8",
    )

    surface = extract_python_api_surface(project)
    class_rows = surface["class_index"]["SentimentAnalyzer"]
    methods = {method["name"] for method in class_rows[0]["methods"]}

    assert class_rows[0]["path"] == "src/sentiment_analyzer/analyzer.py"
    assert "analyze_sentiment" in methods
    assert "_internal" not in methods
    assert "build_analyzer" in surface["function_index"]


def test_repair_plan_expands_attribute_error_to_source_and_test_targets(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "sentiment_analyzer"
    tests = project / "tests"
    source.mkdir(parents=True)
    tests.mkdir(parents=True)
    (source / "analyzer.py").write_text(
        "\n".join(
            [
                "class SentimentAnalyzer:",
                "    def analyze_sentiment(self, text: str) -> dict:",
                "        return {'label': 'positive'}",
            ]
        ),
        encoding="utf-8",
    )
    (tests / "test_analyzer.py").write_text(
        "\n".join(
            [
                "from src.sentiment_analyzer.analyzer import SentimentAnalyzer",
                "",
                "def test_analyze():",
                "    assert SentimentAnalyzer().analyze('good')['label'] == 'positive'",
            ]
        ),
        encoding="utf-8",
    )
    validation = {
        "ok": False,
        "stderr": (
            "tests/test_analyzer.py:4: AttributeError: 'SentimentAnalyzer' object "
            "has no attribute 'analyze'"
        ),
    }

    plan = plan_repair_targets_for_validation(
        project,
        validation,
        file_specs=[
            {"relative_path": "src/sentiment_analyzer/analyzer.py"},
            {"relative_path": "tests/test_analyzer.py"},
        ],
    )
    target_paths = {target["path"] for target in plan["targets"]}
    diagnostic = plan["diagnostics"][0]

    assert extract_attribute_errors(validation["stderr"]) == [
        {"kind": "object", "class_name": "SentimentAnalyzer", "attribute": "analyze"}
    ]
    assert "src/sentiment_analyzer/analyzer.py" in target_paths
    assert "tests/test_analyzer.py" in target_paths
    assert diagnostic["kind"] == "missing_object_attribute"
    assert diagnostic["missing_attribute"] == "analyze"
    assert diagnostic["available_methods"] == ["analyze_sentiment"]
