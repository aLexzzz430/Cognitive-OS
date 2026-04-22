from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.adapter_registry import find_adapter_registry_violations
from core.conos_repository_layout import (
    LAYER_ADAPTER,
    LAYER_CONOS_CORE,
    LAYER_PRIVATE_COGNITIVE_CORE,
    LAYER_RUNTIME,
    classify_repo_path,
    describe_repo_layers,
)


def test_core_path_classification() -> None:
    assert classify_repo_path("core/adapter_registry.py") == LAYER_CONOS_CORE


def test_adapter_path_classification() -> None:
    candidate = Path("integrations/arc_agi3/perception_bridge.py")
    if not candidate.exists():
        adapter_paths = sorted(Path("integrations").glob("**/*.py"))
        assert adapter_paths, "No adapter paths found in integrations/."
        candidate = adapter_paths[0]
    assert classify_repo_path(candidate.as_posix()) == LAYER_ADAPTER


def test_adapter_registry_has_no_boundary_violations() -> None:
    assert find_adapter_registry_violations() == []


def test_repo_layer_summary_is_non_empty_and_contains_key_layers() -> None:
    summaries = describe_repo_layers()
    assert summaries
    layer_names = {summary.layer_name for summary in summaries}
    assert LAYER_CONOS_CORE in layer_names
    assert LAYER_ADAPTER in layer_names
    assert LAYER_PRIVATE_COGNITIVE_CORE in layer_names
    assert LAYER_RUNTIME in layer_names
