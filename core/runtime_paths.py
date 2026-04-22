"""Shared runtime output paths for mutable artifacts.

Phase 0 keeps mutable runtime state out of tracked source paths by default.
Callers may override the root directory or individual files via environment
variables when a harness needs isolated output locations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]

_RUNTIME_ROOT_ENV = "THE_AGI_RUNTIME_ROOT"
_EVAL_ROOT_ENV = "THE_AGI_EVAL_ROOT"
_MODEL_ROOT_ENV = "THE_AGI_MODEL_ARTIFACTS_ROOT"
_EVENT_LOG_ENV = "THE_AGI_EVENT_LOG_PATH"
_STATE_PATH_ENV = "THE_AGI_STATE_PATH"
_REPRESENTATION_UPDATES_ENV = "THE_AGI_REPRESENTATION_UPDATES_PATH"


def repo_root() -> Path:
    return _REPO_ROOT


def ensure_runtime_tree(root: Path | None = None) -> Path:
    runtime_root = Path(root) if root is not None else runtime_root_path()
    runtime_root.mkdir(parents=True, exist_ok=True)
    for relative_dir in ("logs", "state", "representations", "reports", "models", "evals"):
        (runtime_root / relative_dir).mkdir(parents=True, exist_ok=True)
    return runtime_root


def ensure_eval_tree(root: Path | None = None) -> Path:
    eval_root = Path(root) if root is not None else eval_root_path()
    eval_root.mkdir(parents=True, exist_ok=True)
    for relative_dir in ("reports", "runs", "logs", "audits", "datasets"):
        (eval_root / relative_dir).mkdir(parents=True, exist_ok=True)
    return eval_root


def ensure_model_tree(root: Path | None = None) -> Path:
    model_root = Path(root) if root is not None else model_artifacts_root_path()
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root


def runtime_root_path(configured_root: str | os.PathLike[str] | None = None) -> Path:
    if configured_root is not None:
        return ensure_runtime_tree(Path(configured_root))

    env_root = os.getenv(_RUNTIME_ROOT_ENV, "").strip()
    if env_root:
        return ensure_runtime_tree(Path(env_root))

    return ensure_runtime_tree(_REPO_ROOT / "runtime")


def eval_root_path(configured_root: str | os.PathLike[str] | None = None) -> Path:
    if configured_root is not None:
        return ensure_eval_tree(Path(configured_root))

    env_root = os.getenv(_EVAL_ROOT_ENV, "").strip()
    if env_root:
        return ensure_eval_tree(Path(env_root))

    return ensure_eval_tree(runtime_root_path() / "evals")


def model_artifacts_root_path(configured_root: str | os.PathLike[str] | None = None) -> Path:
    if configured_root is not None:
        return ensure_model_tree(Path(configured_root))

    env_root = os.getenv(_MODEL_ROOT_ENV, "").strip()
    if env_root:
        return ensure_model_tree(Path(env_root))

    return ensure_model_tree(runtime_root_path() / "models")


def _normalized_artifact_name(name: str, *, default_suffix: str = ".json") -> str:
    raw = str(name or "").strip()
    if not raw:
        raw = "artifact"
    candidate = Path(raw)
    clean_stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", candidate.stem).strip("-._") or "artifact"
    clean_suffix = candidate.suffix or default_suffix
    return f"{clean_stem}{clean_suffix}"


def resolve_runtime_file(
    *,
    env_var: str,
    default_relative_path: str,
    configured_path: str | os.PathLike[str] | None = None,
) -> Path:
    if configured_path is not None:
        return Path(configured_path)

    env_path = os.getenv(env_var, "").strip()
    if env_path:
        return Path(env_path)

    return runtime_root_path() / default_relative_path


def default_event_log_path(configured_path: str | os.PathLike[str] | None = None) -> Path:
    return resolve_runtime_file(
        env_var=_EVENT_LOG_ENV,
        default_relative_path="logs/event_log.jsonl",
        configured_path=configured_path,
    )


def default_state_path(configured_path: str | os.PathLike[str] | None = None) -> Path:
    return resolve_runtime_file(
        env_var=_STATE_PATH_ENV,
        default_relative_path="state/state.json",
        configured_path=configured_path,
    )


def default_representation_updates_path(
    configured_path: str | os.PathLike[str] | None = None,
) -> Path:
    return resolve_runtime_file(
        env_var=_REPRESENTATION_UPDATES_ENV,
        default_relative_path="representations/runtime_updates.jsonl",
        configured_path=configured_path,
    )


def default_model_artifact_path(
    artifact_name: str,
    configured_path: str | os.PathLike[str] | None = None,
) -> Path:
    if configured_path is not None:
        return Path(configured_path)
    return model_artifacts_root_path() / _normalized_artifact_name(artifact_name, default_suffix=".json")


def default_eval_report_path(
    report_name: str,
    configured_path: str | os.PathLike[str] | None = None,
) -> Path:
    if configured_path is not None:
        return Path(configured_path)
    return eval_root_path() / "reports" / _normalized_artifact_name(report_name, default_suffix=".json")


def default_eval_run_dir(
    run_name: str,
    configured_path: str | os.PathLike[str] | None = None,
) -> Path:
    if configured_path is not None:
        run_dir = Path(configured_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(run_name or "").strip()).strip("-._") or "run"
    run_dir = eval_root_path() / "runs" / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def runtime_layout_snapshot() -> dict[str, str | list[str]]:
    runtime_root = runtime_root_path()
    eval_root = eval_root_path()
    model_root = model_artifacts_root_path()
    return {
        "repo_root": str(repo_root()),
        "source_tree": str(repo_root()),
        "runtime_root": str(runtime_root),
        "eval_root": str(eval_root),
        "model_root": str(model_root),
        "tracked_runtime_paths": [
            str(runtime_root / "logs"),
            str(runtime_root / "state"),
            str(runtime_root / "representations"),
            str(runtime_root / "reports"),
            str(runtime_root / "models"),
        ],
        "tracked_eval_paths": [
            str(eval_root / "reports"),
            str(eval_root / "runs"),
            str(eval_root / "logs"),
            str(eval_root / "audits"),
            str(eval_root / "datasets"),
        ],
        "tracked_model_paths": [
            str(model_root),
        ],
    }
