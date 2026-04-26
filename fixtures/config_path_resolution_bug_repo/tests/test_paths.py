from __future__ import annotations

from app_core.paths import resolve_config_path


def test_resolve_config_path_preserves_relative_subdirectories(tmp_path):
    expected = tmp_path / "profiles" / "default.json"

    assert resolve_config_path(tmp_path, "profiles/default.json") == expected


def test_resolve_config_path_keeps_absolute_paths(tmp_path):
    absolute = tmp_path / "profiles" / "default.json"

    assert resolve_config_path(tmp_path / "other", absolute) == absolute
