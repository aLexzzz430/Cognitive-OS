from __future__ import annotations

import json

from app_core.loader import load_config


def test_load_config_resolves_nested_relative_path(tmp_path):
    config_dir = tmp_path / "profiles"
    config_dir.mkdir()
    (config_dir / "default.json").write_text(
        json.dumps({"name": "nested", "retries": 3}),
        encoding="utf-8",
    )

    config = load_config(tmp_path, "profiles/default.json")

    assert config.name == "nested"
    assert config.retries == 3


def test_load_config_accepts_plain_filename(tmp_path):
    (tmp_path / "default.json").write_text(
        json.dumps({"name": "plain", "retries": 1}),
        encoding="utf-8",
    )

    config = load_config(tmp_path, "default.json")

    assert config.name == "plain"
    assert config.retries == 1
