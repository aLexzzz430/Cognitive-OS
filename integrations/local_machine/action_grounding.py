from __future__ import annotations

from dataclasses import asdict, dataclass
import difflib
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ACTION_GROUNDING_VERSION = "conos.local_machine.action_grounding/v1"
OPEN_TASK_EVIDENCE_GATE_VERSION = "conos.local_machine.open_task_evidence_gate/v1"
PYTEST_CONTEXT_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
LOW_PRIORITY_SOURCE_PARTS = frozenset(
    {
        "bench",
        "benches",
        "benchmark",
        "benchmarks",
        "docs",
        "doc",
        "examples",
        "example",
        "fuzz",
        "scripts",
    }
)


@dataclass(frozen=True)
class LocalMachineActionSchema:
    action_name: str
    required_kwargs: tuple[str, ...]
    optional_kwargs: tuple[str, ...]
    allow_empty_kwargs: bool
    repair_strategy: str
    side_effect_class: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_kwargs"] = list(self.required_kwargs)
        payload["optional_kwargs"] = list(self.optional_kwargs)
        return payload


LOCAL_MACHINE_ACTION_SCHEMA_REGISTRY: dict[str, LocalMachineActionSchema] = {
    "repo_tree": LocalMachineActionSchema(
        "repo_tree",
        (),
        ("path", "root", "depth", "max_entries", "exclude", "excludes"),
        True,
        "default_repo_tree",
        "mirror_control_write",
    ),
    "repo_find": LocalMachineActionSchema(
        "repo_find",
        (),
        ("root", "path", "name_pattern", "pattern", "kind", "max_results"),
        True,
        "default_repo_find",
        "mirror_control_write",
    ),
    "repo_grep": LocalMachineActionSchema(
        "repo_grep",
        ("query",),
        ("root", "path", "pattern", "globs", "max_matches", "case_sensitive", "regex"),
        False,
        "infer_query",
        "mirror_control_write",
    ),
    "file_read": LocalMachineActionSchema(
        "file_read",
        ("path",),
        ("start_line", "end_line", "max_bytes"),
        False,
        "infer_path",
        "mirror_control_write",
    ),
    "run_test": LocalMachineActionSchema(
        "run_test",
        ("target",),
        ("timeout_seconds",),
        False,
        "infer_test_target",
        "mirror_workspace_read",
    ),
    "read_test_failure": LocalMachineActionSchema(
        "read_test_failure",
        (),
        ("run_ref", "max_chars"),
        True,
        "latest_test_failure",
        "mirror_control_read",
    ),
    "apply_patch": LocalMachineActionSchema(
        "apply_patch",
        ("patch",),
        ("path", "max_files", "max_hunks", "evidence_refs"),
        False,
        "infer_patch_if_grounded",
        "mirror_workspace_write",
    ),
    "propose_patch": LocalMachineActionSchema(
        "propose_patch",
        (),
        ("target_file", "max_changed_lines", "expected_tests"),
        True,
        "target_binding_patch_proposal",
        "mirror_workspace_write",
    ),
    "run_typecheck": LocalMachineActionSchema(
        "run_typecheck",
        ("target",),
        ("timeout_seconds",),
        False,
        "infer_source_target",
        "mirror_workspace_read",
    ),
    "mirror_plan": LocalMachineActionSchema(
        "mirror_plan",
        (),
        (),
        True,
        "no_repair_needed",
        "mirror_control_write",
    ),
    "no_op_complete": LocalMachineActionSchema(
        "no_op_complete",
        (),
        ("reason",),
        True,
        "no_repair_needed",
        "none",
    ),
}


def action_schema_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": ACTION_GROUNDING_VERSION,
        "actions": {
            name: schema.to_dict()
            for name, schema in sorted(LOCAL_MACHINE_ACTION_SCHEMA_REGISTRY.items())
        },
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _instruction_without_default_template(context: Mapping[str, Any]) -> str:
    text = _text(context.get("instruction"))
    marker = "CONOS_DEFAULT_OPEN_PROJECT_TEMPLATE:"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text


def _probe_variant(context: Mapping[str, Any]) -> str:
    instruction = _text(context.get("instruction"))
    match = re.search(r"\[closed_loop_probe_variant=([a-zA-Z0-9_\-]+)\]", instruction)
    return match.group(1) if match else "full"


def _nowish_event_base(action_name: str) -> dict[str, Any]:
    return {
        "schema_version": ACTION_GROUNDING_VERSION,
        "action": action_name,
    }


def _state(context: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(context.get("investigation_state"))


def _source_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("source_root")) or ".").resolve()


def _workspace_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("workspace_root")) or ".").resolve()


def _run_output_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("run_output_root")) or ".").resolve()


def _path_exists_in_source(context: Mapping[str, Any], relative: str) -> bool:
    if not relative or relative.startswith("<"):
        return False
    root = _source_root(context)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path.exists() and path.is_file()


def _path_exists_in_workspace(context: Mapping[str, Any], relative: str) -> bool:
    if not relative or relative.startswith("<"):
        return False
    root = _workspace_root(context)
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path.exists() and path.is_file()


def _load_run_output(context: Mapping[str, Any], run_ref: str) -> dict[str, Any]:
    if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", _text(run_ref)):
        return {}
    path = _run_output_root(context) / f"{run_ref}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _latest_run_output(context: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(context)
    last_ref = _text(state.get("last_run_ref"))
    if last_ref:
        payload = _load_run_output(context, last_ref)
        if payload:
            return payload
    for row in reversed(_as_list(state.get("validation_runs"))):
        run_ref = _text(_as_dict(row).get("run_ref"))
        payload = _load_run_output(context, run_ref)
        if payload:
            return payload
    return {}


def _latest_failed_run_output(context: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(context)
    for row in reversed(_as_list(state.get("validation_runs"))):
        item = _as_dict(row)
        if bool(item.get("success", False)):
            continue
        payload = _load_run_output(context, _text(item.get("run_ref")))
        if payload:
            return payload
    payload = _latest_run_output(context)
    return payload if int(payload.get("returncode", 0) or 0) != 0 else {}


def latest_failure_text(context: Mapping[str, Any]) -> str:
    payload = _latest_failed_run_output(context)
    return f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}".strip()


def _tree_entries(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _as_list(_as_dict(_state(context).get("last_tree")).get("entries"))
    return [dict(row) for row in entries if isinstance(row, Mapping)]


def _looks_like_test_file_path(path: str) -> bool:
    clean = _text(path).replace("\\", "/")
    if not clean:
        return False
    parts = set(Path(clean).parts)
    name = Path(clean).name
    return (
        bool(parts.intersection({"tests", "testing"}))
        or name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _in_test_directory_or_conftest(path: str) -> bool:
    clean = _text(path).replace("\\", "/")
    if not clean:
        return False
    parts = set(Path(clean).parts)
    name = Path(clean).name
    return bool(parts.intersection({"tests", "testing"})) or name == "conftest.py"


def _source_files_from_tree(context: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []

    def add(path: str) -> None:
        clean = _text(path).replace("\\", "/")
        if not clean or clean in paths or not clean.endswith(".py"):
            return
        name = Path(clean).name
        if _in_test_directory_or_conftest(clean):
            return
        if (name.startswith("test_") or name.endswith("_test.py")) and _source_file_looks_like_pytest(context, clean):
            return
        paths.append(clean)

    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        add(path)
    root = _source_root(context)
    if root.exists():
        for path in sorted(root.rglob("*.py")):
            if len(paths) >= 80:
                break
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            parts = set(Path(relative).parts)
            if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                continue
            add(relative)
    return paths


def _test_files_from_tree(context: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        if path.endswith(".py") and _source_file_looks_like_pytest(context, path):
            if path not in paths:
                paths.append(path)
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", _text(context.get("instruction")).lower()))

    def score(path: str) -> tuple[int, str]:
        lowered = path.lower()
        return (-sum(1 for token in tokens if token in lowered), path)

    return sorted(paths, key=score)


def _overview_files_from_tree(context: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        if not path or "/" in path:
            continue
        name = Path(path).name.lower()
        if name.startswith("readme") or name in {"pyproject.toml", "setup.cfg", "setup.py", "requirements.txt"}:
            paths.append(path)

    def score(path: str) -> tuple[int, str]:
        name = Path(path).name.lower()
        if name.startswith("readme"):
            return (0, path)
        if name == "pyproject.toml":
            return (1, path)
        return (2, path)

    return sorted(dict.fromkeys(paths), key=score)


def pytest_context_paths_from_tree(context: Mapping[str, Any], *, limit: int = 200) -> list[str]:
    config_paths: list[str] = []
    test_paths: list[str] = []
    source_paths: list[str] = []
    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        if not path:
            continue
        name = Path(path).name
        if name in {"pyproject.toml", "pytest.ini", "setup.cfg", "setup.py", "requirements.txt"} and "/" not in path:
            config_paths.append(path)
        elif path.endswith(".py") and (path.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py")):
            test_paths.append(path)
        elif path.endswith(".py"):
            source_paths.append(path)
    root = _source_root(context)
    if not config_paths and root.exists():
        for name in ("pyproject.toml", "pytest.ini", "setup.cfg", "setup.py", "requirements.txt"):
            if (root / name).is_file():
                config_paths.append(name)
    if root.exists():
        root_test_paths: list[str] = []
        for test_dir_name in ("tests", "testing"):
            for tests_root in sorted(root.rglob(test_dir_name)):
                if not tests_root.exists() or not tests_root.is_dir():
                    continue
                try:
                    tests_root.resolve().relative_to(root)
                except ValueError:
                    continue
                parts = set(tests_root.relative_to(root).parts)
                if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                    continue
                for path in sorted(tests_root.rglob("*")):
                    try:
                        relative = path.resolve().relative_to(root).as_posix()
                    except ValueError:
                        continue
                    if not path.is_file() or path.is_symlink():
                        continue
                    parts = set(Path(relative).parts)
                    if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                        continue
                    if relative not in root_test_paths:
                        root_test_paths.append(relative)
                    if len(root_test_paths) >= max(10, min(80, int(limit))):
                        break
                if len(root_test_paths) >= max(10, min(80, int(limit))):
                    break
            if len(root_test_paths) >= max(10, min(80, int(limit))):
                break
        for path in sorted(root.rglob("*.py")):
            if len(root_test_paths) >= max(10, min(80, int(limit))):
                break
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            if not path.is_file() or path.is_symlink():
                continue
            parts = set(Path(relative).parts)
            if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                continue
            name = Path(relative).name
            if not (name.startswith("test_") or name.endswith("_test.py")):
                continue
            if relative not in root_test_paths:
                root_test_paths.append(relative)
        if root_test_paths:
            test_paths = [*root_test_paths, *[path for path in test_paths if path not in root_test_paths]]
        else:
            tests_root = root / "tests"
            if tests_root.exists() and tests_root.is_dir():
                root_test_paths = []
                for path in sorted(tests_root.rglob("*")):
                    try:
                        relative = path.resolve().relative_to(root).as_posix()
                    except ValueError:
                        continue
                    if not path.is_file() or path.is_symlink():
                        continue
                    parts = set(Path(relative).parts)
                    if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                        continue
                    if relative not in root_test_paths:
                        root_test_paths.append(relative)
                    if len(root_test_paths) >= max(10, min(80, int(limit))):
                        break
                test_paths = [*root_test_paths, *[path for path in test_paths if path not in root_test_paths]]
        root_source_paths: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if len(root_source_paths) >= max(20, min(120, int(limit))):
                break
            try:
                relative = path.resolve().relative_to(root).as_posix()
            except ValueError:
                continue
            parts = set(Path(relative).parts)
            if parts & PYTEST_CONTEXT_EXCLUDED_PARTS:
                continue
            if relative.startswith("tests/") or "/tests/" in relative:
                continue
            if relative not in root_source_paths:
                root_source_paths.append(relative)
        source_paths = [*source_paths, *[path for path in root_source_paths if path not in source_paths]]
    ordered = list(dict.fromkeys([*config_paths, *test_paths, *source_paths]))
    return ordered[: max(1, int(limit))]


def _extract_traceback_files(context: Mapping[str, Any], text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r'(?:File "([^"]+?\.py)"|([A-Za-z0-9_./-]+\.py):\d+)', text):
        raw = _text(match.group(1) or match.group(2))
        if not raw or raw.startswith("<") or "site-packages" in raw:
            continue
        path = Path(raw)
        if path.is_absolute():
            try:
                raw = path.resolve().relative_to(_source_root(context)).as_posix()
            except ValueError:
                try:
                    raw = path.resolve().relative_to(_workspace_root(context)).as_posix()
                except ValueError:
                    continue
        raw = raw.replace("\\", "/").lstrip("./")
        if raw and raw not in candidates and _path_exists_in_source(context, raw):
            candidates.append(raw)
    return candidates


def target_file_from_failure(context: Mapping[str, Any]) -> str:
    text = latest_failure_text(context)
    files = _extract_traceback_files(context, text)
    for path in files:
        if not (path.startswith("tests/") or "/tests/" in path):
            return path
    return ""


def _traceback_source_files(context: Mapping[str, Any]) -> list[str]:
    files = _extract_traceback_files(context, latest_failure_text(context))
    return [
        path
        for path in files
        if path and not (path.startswith("tests/") or "/tests/" in path)
    ]


def _deeper_traceback_source_file(
    context: Mapping[str, Any],
    *,
    unread_only: bool = False,
    extra_read_paths: set[str] | None = None,
) -> str:
    files = _traceback_source_files(context)
    for path in files[1:]:
        if unread_only and (_file_has_been_read(context, path) or path in set(extra_read_paths or set())):
            continue
        return path
    return ""


def _latest_failed_test_target(context: Mapping[str, Any]) -> str:
    payload = _latest_failed_run_output(context)
    text = f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}"
    for match in re.finditer(r"FAILED\s+([A-Za-z0-9_./-]+\.py(?:::[^\s]+)?)", text):
        target = _text(match.group(1)).split("::", 1)[0]
        if target and _path_exists_in_source(context, target):
            return target
    command = [str(part) for part in _as_list(payload.get("command"))]
    if command:
        target = _text(command[-1])
        if target and target != "." and _path_exists_in_source(context, target.split("::", 1)[0]):
            return target
    return ""


def _last_grep_file(context: Mapping[str, Any]) -> str:
    search = _as_dict(_state(context).get("last_search"))
    if search.get("action") != "repo_grep":
        return ""
    rows = _as_list(search.get("matches")) or _as_list(search.get("results"))
    fallback = ""
    for row in rows:
        path = _text(_as_dict(row).get("path"))
        if not path or not path.endswith(".py") or not _path_exists_in_source(context, path):
            continue
        if not (path.startswith("tests/") or "/tests/" in path):
            return path
        fallback = fallback or path
    return fallback


def _last_grep_match(context: Mapping[str, Any], query: str = "") -> dict[str, Any]:
    search = _as_dict(_state(context).get("last_search"))
    if search.get("action") != "repo_grep":
        return {}
    if query and _text(search.get("query")).lower() != _text(query).lower():
        return {}
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row in _as_list(search.get("matches")) or _as_list(search.get("results")):
        payload = _as_dict(row)
        path = _text(payload.get("path"))
        if path and path.endswith(".py") and _path_exists_in_source(context, path):
            penalty, _ = _source_file_rank(context, path)
            text = _text(payload.get("text")).lower()
            if "self." in text or "cls." in text:
                penalty -= 30
            if "[" in text and "]" in text:
                penalty -= 20
            try:
                line = int(payload.get("line", 0) or 0)
            except (TypeError, ValueError):
                line = 0
            if line and _line_has_been_read(context, path, line):
                penalty += 25
            candidates.append((penalty, payload))
    if not candidates:
        return {}
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _read_ranges_for_path(context: Mapping[str, Any], path: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    state = _state(context)
    for row in _as_list(state.get("read_files")):
        payload = _as_dict(row)
        if _text(payload.get("path")) != path:
            continue
        try:
            start = int(payload.get("start_line", 1) or 1)
            end = int(payload.get("end_line", start) or start)
        except (TypeError, ValueError):
            continue
        ranges.append((start, end))
    last_read = _as_dict(state.get("last_read"))
    if _text(last_read.get("path")) == path:
        try:
            start = int(last_read.get("start_line", 1) or 1)
            end = int(last_read.get("end_line", start) or start)
            ranges.append((start, end))
        except (TypeError, ValueError):
            pass
    return ranges


def _line_has_been_read(context: Mapping[str, Any], path: str, line: int) -> bool:
    if line <= 0:
        return _file_has_been_read(context, path)
    return any(start <= line <= end for start, end in _read_ranges_for_path(context, path))


def _file_read_kwargs_for_last_grep_match(context: Mapping[str, Any], query: str) -> dict[str, Any]:
    match = _last_grep_match(context, query)
    path = _text(match.get("path"))
    if not path:
        return {}
    try:
        line = int(match.get("line", 1) or 1)
    except (TypeError, ValueError):
        line = 1
    if _line_has_been_read(context, path, line):
        return {}
    return {"path": path, "start_line": max(1, line - 40), "end_line": line + 80}


def _top_hypothesis_target(context: Mapping[str, Any]) -> str:
    state = _state(context)
    hypotheses = [_as_dict(row) for row in _as_list(state.get("hypotheses")) if isinstance(row, Mapping)]

    def score(row: Mapping[str, Any]) -> float:
        for key in ("posterior", "confidence", "belief", "score"):
            try:
                return float(row.get(key))
            except (TypeError, ValueError):
                continue
        return 0.0

    hypotheses.sort(key=score, reverse=True)
    for row in hypotheses:
        meta = _as_dict(row.get("metadata"))
        for value in (
            row.get("target_file"),
            row.get("candidate_file"),
            meta.get("target_file"),
            meta.get("candidate_file"),
        ):
            path = _text(value)
            if path and _path_exists_in_source(context, path):
                return path
    return ""


def _experiment_candidate_path(context: Mapping[str, Any]) -> str:
    state = _state(context)
    for row in _as_list(state.get("discriminating_tests")):
        test = _as_dict(row)
        action = _as_dict(test.get("candidate_action"))
        kwargs = _as_dict(action.get("kwargs")) or _as_dict(action.get("args"))
        path = _text(kwargs.get("path") or action.get("path"))
        if path and _path_exists_in_source(context, path):
            return path
    return ""


def _target_binding_file(context: Mapping[str, Any]) -> str:
    binding = _target_binding(context)
    path = _text(binding.get("top_target_file"))
    if path and _path_exists_in_source(context, path):
        return path
    return ""


def _goal_tokens(context: Mapping[str, Any]) -> set[str]:
    text = " ".join(
        [
            _instruction_without_default_template(context),
            latest_failure_text(context),
            json.dumps(_as_dict(context.get("posterior_summary")), ensure_ascii=False, default=str),
        ]
    ).lower()
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text))
    return {
        token
        for token in tokens
        if token not in {"the", "and", "this", "that", "with", "from", "true", "false", "none", "test"}
    }


def _path_goal_aligned(context: Mapping[str, Any], path: str) -> bool:
    tokens = _goal_tokens(context)
    parts = {part.lower().replace("-", "_") for part in Path(_text(path)).parts}
    return bool(tokens & parts)


def _source_file_rank(context: Mapping[str, Any], path: str) -> tuple[int, str]:
    clean = _text(path).replace("\\", "/")
    parts = [part.lower() for part in Path(clean).parts]
    penalty = 0
    if any(part in LOW_PRIORITY_SOURCE_PARTS for part in parts) and not _path_goal_aligned(context, clean):
        penalty += 80
    if Path(clean).name == "__init__.py":
        penalty += 20
        try:
            content = (_source_root(context) / clean).read_text(encoding="utf-8")
        except OSError:
            content = ""
        if not content.strip():
            penalty += 50
    if len(parts) <= 1:
        penalty += 10
    return (penalty, clean)


def _tree_relevant_source_file(context: Mapping[str, Any]) -> str:
    tokens = _goal_tokens(context)
    best: tuple[int, int, str] | None = None
    for path in _source_files_from_tree(context):
        lowered = path.lower()
        score = sum(1 for token in tokens if token in lowered)
        penalty, ranked_path = _source_file_rank(context, path)
        candidate = (-score, penalty, ranked_path)
        if best is None or candidate < best:
            best = candidate
    return best[2] if best else ""


def choose_file_read_path(context: Mapping[str, Any]) -> tuple[str, str]:
    sources = [
        ("latest test failure traceback file", target_file_from_failure(context)),
        ("deeper traceback implementation file", _deeper_traceback_source_file(context, unread_only=True)),
        ("target binding top candidate", _target_binding_file(context)),
        ("top hypothesis target_file", _top_hypothesis_target(context)),
        ("top discriminating experiment candidate_action path", _experiment_candidate_path(context)),
        ("latest repo_grep result top file", _last_grep_file(context)),
        ("repo_tree relevant source file", _tree_relevant_source_file(context)),
    ]
    for source, path in sources:
        if path and _path_exists_in_source(context, path) and not _file_has_been_read(context, path):
            return path, source
    for source, path in sources:
        if path and _path_exists_in_source(context, path):
            return path, source
    return "", ""


LOW_VALUE_GREP_QUERY_TOKENS = frozenset(
    {
        "actual",
        "bootstrap",
        "condition",
        "error",
        "expected",
        "failed",
        "github",
        "importlib",
        "investigate",
        "module",
        "python",
        "repository",
        "runtime",
        "source",
        "through",
        "validation",
    }
)


def _extract_code_query_from_text(text: str) -> str:
    raw = str(text or "")
    slice_matches = re.findall(
        r"\b((?:self|cls|[A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_][A-Za-z0-9_]{2,})\s*\[",
        raw,
    )
    for match in slice_matches:
        name = match.rsplit(".", 1)[-1]
        if name.lower() not in LOW_VALUE_GREP_QUERY_TOKENS:
            return name
    member_matches = re.findall(
        r"\b(?:self|cls|[A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]{2,})",
        raw,
    )
    for match in member_matches:
        if match.lower() not in LOW_VALUE_GREP_QUERY_TOKENS:
            return match
    dotted_call_matches = re.findall(
        r"\b(?:self|cls|[A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]{2,})\s*\(",
        raw,
    )
    for match in dotted_call_matches:
        if match.lower() not in LOW_VALUE_GREP_QUERY_TOKENS:
            return match
    return ""


def _extract_query_from_text(text: str) -> str:
    code_query = _extract_code_query_from_text(text)
    if code_query:
        return code_query
    function_names = [
        match.group(1)
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", text)
        if not match.group(1).startswith("_")
        and match.group(1) not in {"assert", "print", "str", "int", "float", "Decimal", "ValueError"}
        and match.group(1).lower() not in LOW_VALUE_GREP_QUERY_TOKENS
    ]
    if function_names:
        return function_names[0]
    quoted = re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]{3,})['\"]", text)
    for token in quoted:
        if token.lower() not in LOW_VALUE_GREP_QUERY_TOKENS:
            return token
    tokens = [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", text)
        if not token.startswith("_") and token.lower() not in LOW_VALUE_GREP_QUERY_TOKENS
    ]
    return tokens[0] if tokens else ""


def choose_repo_grep_query(context: Mapping[str, Any]) -> tuple[str, str]:
    goal_text = _instruction_without_default_template(context)
    goal_code_query = _extract_code_query_from_text(goal_text)
    if goal_code_query:
        return goal_code_query, "goal code token"
    candidates = [
        ("latest failure text", latest_failure_text(context)),
        ("top hypothesis summary", json.dumps(_state(context).get("hypotheses", [])[:1], ensure_ascii=False, default=str)),
        ("posterior_summary", json.dumps(_as_dict(context.get("posterior_summary")), ensure_ascii=False, default=str)),
        ("goal", goal_text),
    ]
    for source, text in candidates:
        query = _extract_query_from_text(text)
        if query:
            return query, source
    return "", ""


def choose_run_test_target(context: Mapping[str, Any]) -> tuple[str, str]:
    target = _latest_failed_test_target(context)
    if target:
        return target, "recent failed test"
    return ".", "full pytest"


def _is_pytest_target_path(path: str) -> bool:
    clean = _text(path).split("::", 1)[0]
    if not clean:
        return False
    name = Path(clean).name
    if name == "__init__.py":
        return False
    return clean.startswith("tests/") or "/tests/" in clean or name.startswith("test_") or name.endswith("_test.py")


def _source_file_looks_like_pytest(context: Mapping[str, Any], path: str) -> bool:
    clean = _text(path).split("::", 1)[0]
    if not clean:
        return False
    name = Path(clean).name
    if name in {"__init__.py", "conftest.py"}:
        return False
    if not _path_exists_in_source(context, clean):
        parts = set(Path(clean).parts)
        return bool(parts.intersection({"tests", "testing"})) and (
            name.startswith("test_") or name.endswith("_test.py")
        )
    try:
        text = (_source_root(context) / clean).read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*(?:def\s+test_|class\s+Test)", text, flags=re.MULTILINE))


def repair_run_test_source_target(context: Mapping[str, Any], target: str) -> tuple[str, str]:
    clean = _text(target).split("::", 1)[0]
    if not clean or clean == ".":
        return "", ""
    if _is_pytest_target_path(clean) and _source_file_looks_like_pytest(context, clean):
        return "", ""
    if _in_test_directory_or_conftest(clean) and clean.endswith(".py"):
        tests = _test_files_from_tree(context)
        if tests:
            return tests[0], "selected test helper is not a runnable pytest target"
        return ".", "selected test helper is not a runnable pytest target; use full pytest"
    if not clean.endswith(".py") or not _path_exists_in_source(context, clean):
        return "", ""
    direct = _direct_test_file_for_source(context, clean)
    if direct:
        return direct, "direct pytest file for selected source target"
    return ".", "source file is not a pytest target; use full pytest"


def choose_verification_target(context: Mapping[str, Any]) -> tuple[str, str]:
    grounding = _as_dict(_state(context).get("grounding"))
    if _text(grounding.get("last_successful_test_target")):
        return ".", "full pytest after repaired test passed"
    target = _latest_failed_test_target(context)
    if target:
        return target, "failed test after patch"
    return ".", "full pytest after patch"


def choose_typecheck_target(context: Mapping[str, Any]) -> tuple[str, str]:
    target = target_file_from_failure(context) or _top_hypothesis_target(context) or _last_grep_file(context) or _tree_relevant_source_file(context)
    if target:
        return target, "source file from failure/hypothesis/search/tree"
    return ".", "full compileall"


def _line_ref_for_last_read(context: Mapping[str, Any], path: str) -> str:
    read = _as_dict(_state(context).get("last_read"))
    if _text(read.get("path")) == path:
        start = int(read.get("start_line", 1) or 1)
        end = int(read.get("end_line", start) or start)
        return f"file:{path}:{start}-{end}"
    return f"file:{path}:1"


def _has_failure_evidence(context: Mapping[str, Any]) -> bool:
    return bool(latest_failure_text(context) or _latest_failed_run_output(context))


def _file_has_been_read(context: Mapping[str, Any], path: str) -> bool:
    state = _state(context)
    if path in {str(item) for item in _as_list(context.get("episode_read_paths"))}:
        return True
    if _text(_as_dict(state.get("last_read")).get("path")) == path:
        return True
    for row in _as_list(state.get("read_files")):
        if _as_dict(row).get("path") == path:
            return True
    return False


def _source_evidence_candidates(context: Mapping[str, Any], target_file: str = "") -> list[str]:
    paths: list[str] = []

    def add(path: str) -> None:
        clean = _text(path)
        if not clean or clean in paths:
            return
        if not clean.endswith(".py"):
            return
        name = Path(clean).name
        if _in_test_directory_or_conftest(clean):
            return
        if (name.startswith("test_") or name.endswith("_test.py")) and _source_file_looks_like_pytest(context, clean):
            return
        if not _path_exists_in_source(context, clean):
            return
        paths.append(clean)

    add(target_file)
    add(_target_binding_file(context))
    add(_top_hypothesis_target(context))
    binding = _target_binding(context)
    for row in _as_list(binding.get("target_file_candidates")):
        payload = _as_dict(row)
        add(_text(payload.get("target_file") or payload.get("path")))
    add(_last_grep_file(context))
    add(_tree_relevant_source_file(context))
    for path in sorted(_source_files_from_tree(context), key=lambda item: _source_file_rank(context, item)):
        add(path)
        if len(paths) >= 4:
            break
    return paths


def open_task_patch_evidence_gap(
    context: Mapping[str, Any],
    *,
    target_file: str = "",
) -> dict[str, Any]:
    """Return the next evidence action needed before open-task patch proposal.

    This gate is intentionally about open-ended improvement work. Bug-fix runs
    with concrete failure evidence keep the existing closed-loop path.
    """

    if not _open_improvement_goal(context):
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": True,
            "reason": "goal_is_not_open_improvement",
        }
    if _has_failure_evidence(context):
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": True,
            "reason": "failure_evidence_present",
        }
    if _diff_entry_count(context) > 0:
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": True,
            "reason": "workspace_already_has_changes",
        }
    if bool(context.get("default_command_present", False)) and not _run_test_targets(context):
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": True,
            "reason": "default_command_should_run_before_open_task_evidence_gate",
        }

    tree_entries = _tree_entries(context)
    if not tree_entries:
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": False,
            "reason": "repo_inventory_required_before_open_task_patch",
            "suggested_action": "repo_tree",
            "suggested_kwargs": {"path": ".", "depth": 3, "max_entries": 200},
            "repair_source": "open_task_evidence_gate",
        }

    overview_files = _overview_files_from_tree(context)
    for path in overview_files[:1]:
        if not _file_has_been_read(context, path):
            return {
                "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
                "sufficient": False,
                "reason": "project_overview_required_before_open_task_patch",
                "missing_evidence": "project_overview",
                "suggested_action": "file_read",
                "suggested_kwargs": {"path": path, "start_line": 1, "end_line": 220},
                "repair_source": "open_task_evidence_gate",
                "overview_files": overview_files,
            }

    test_files = _test_files_from_tree(context)
    for path in test_files[:1]:
        if not _file_has_been_read(context, path):
            return {
                "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
                "sufficient": False,
                "reason": "test_source_required_before_open_task_patch",
                "missing_evidence": "test_source",
                "suggested_action": "file_read",
                "suggested_kwargs": {"path": path, "start_line": 1, "end_line": 240},
                "repair_source": "open_task_evidence_gate",
                "test_files": test_files[:3],
            }
    if test_files and not any(_test_has_been_run(context, path) for path in test_files):
        return {
            "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
            "sufficient": False,
            "reason": "test_execution_required_before_open_task_patch",
            "missing_evidence": "test_execution",
            "suggested_action": "run_test",
            "suggested_kwargs": {"target": test_files[0], "timeout_seconds": 30},
            "repair_source": "open_task_evidence_gate",
            "test_files": test_files[:3],
        }

    source_candidates = _source_evidence_candidates(context, target_file=target_file)
    required_source_count = min(2, len(source_candidates))
    for path in source_candidates[:required_source_count]:
        if not _file_has_been_read(context, path):
            return {
                "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
                "sufficient": False,
                "reason": "related_source_required_before_open_task_patch",
                "missing_evidence": "related_source",
                "suggested_action": "file_read",
                "suggested_kwargs": {"path": path, "start_line": 1, "end_line": 240},
                "repair_source": "open_task_evidence_gate",
                "target_file": target_file,
                "source_candidates": source_candidates[:4],
            }
    if not source_candidates:
        query, query_source = choose_repo_grep_query(context)
        if query:
            return {
                "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
                "sufficient": False,
                "reason": "source_candidate_search_required_before_open_task_patch",
                "missing_evidence": "source_candidate",
                "suggested_action": "repo_grep",
                "suggested_kwargs": {"root": ".", "query": query, "globs": ["*.py", "*.md", "*.toml"], "max_matches": 50},
                "repair_source": query_source or "open_task_evidence_gate",
            }

    return {
        "schema_version": OPEN_TASK_EVIDENCE_GATE_VERSION,
        "sufficient": True,
        "reason": "open_task_patch_evidence_sufficient",
        "overview_files": overview_files[:3],
        "test_files": test_files[:3],
        "source_candidates": source_candidates[:4],
        "required_source_count": required_source_count,
    }


def _run_test_targets(context: Mapping[str, Any]) -> set[str]:
    targets = {str(item) for item in _as_list(context.get("episode_run_test_targets")) if str(item)}
    for row in _as_list(_state(context).get("validation_runs")):
        payload = _load_run_output(context, _text(_as_dict(row).get("run_ref")))
        command = [str(part) for part in _as_list(payload.get("command"))]
        if len(command) >= 4 and command[-2:] and command[-2] == "pytest":
            targets.add(command[-1])
        elif command:
            tail = command[-1]
            if tail and tail != "pytest":
                targets.add(tail)
    return targets


def _successful_run_test_targets(context: Mapping[str, Any]) -> set[str]:
    targets: set[str] = set()
    for row in _as_list(_state(context).get("action_history")):
        payload = _as_dict(row)
        if _text(payload.get("function_name")) != "run_test" or not bool(payload.get("success", False)):
            continue
        target = _text(payload.get("target"))
        if target:
            targets.add(target.split("::", 1)[0])
    for row in _as_list(_state(context).get("validation_runs")):
        payload = _as_dict(row)
        if not bool(payload.get("success", False)):
            continue
        run_output = _load_run_output(context, _text(payload.get("run_ref")))
        command = [str(part) for part in _as_list(run_output.get("command"))]
        if command:
            target = _text(command[-1])
            if target and target != "pytest":
                targets.add(target.split("::", 1)[0])
    return targets


def _test_has_been_run(context: Mapping[str, Any], target: str) -> bool:
    wanted = _text(target).split("::", 1)[0]
    if not wanted:
        return False
    for run_target in _run_test_targets(context):
        normalized = _text(run_target).split("::", 1)[0]
        if normalized == wanted:
            return True
    return False


def _successful_test_has_been_run(context: Mapping[str, Any], target: str) -> bool:
    wanted = _text(target).split("::", 1)[0]
    if not wanted:
        return False
    return any(_text(item).split("::", 1)[0] == wanted for item in _successful_run_test_targets(context))


def _recent_same_empty_repo_grep(context: Mapping[str, Any], query: str) -> bool:
    wanted = _text(query).lower()
    if not wanted:
        return False
    recent = _recent_action_history(context, limit=6)
    count = 0
    for row in reversed(recent):
        payload = _as_dict(row)
        if _text(payload.get("function_name")) != "repo_grep":
            continue
        if _text(payload.get("query")).lower() != wanted:
            continue
        try:
            match_count = int(payload.get("match_count", 0) or 0)
        except (TypeError, ValueError):
            match_count = 0
        if match_count == 0:
            count += 1
    if count >= 1:
        return True
    last_search = _as_dict(_state(context).get("last_search"))
    try:
        last_count = int(last_search.get("match_count", 0) or 0)
    except (TypeError, ValueError):
        last_count = 0
    return (
        _text(last_search.get("action")) == "repo_grep"
        and _text(last_search.get("query")).lower() == wanted
        and last_count == 0
    )


def _direct_test_file_for_source(context: Mapping[str, Any], source_path: str) -> str:
    if not source_path or source_path.startswith("tests/"):
        return ""
    stem = Path(source_path).stem.lower()
    if not stem:
        return ""
    tokens = {stem}
    if stem.endswith("s") and len(stem) > 3:
        tokens.add(stem[:-1])
    best: tuple[int, str] | None = None
    for test_path in _test_files_from_tree(context):
        lowered = test_path.lower()
        score = sum(1 for token in tokens if token and token in lowered)
        if score <= 0:
            continue
        candidate = (-score, test_path)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best else ""


def _direct_test_needed_before_patch(context: Mapping[str, Any], target: str) -> str:
    direct = _direct_test_file_for_source(context, target)
    if not direct:
        return ""
    if _test_has_been_run(context, direct) or _file_has_been_read(context, direct):
        return ""
    return direct


def _downstream_direct_test_needed_before_patch(context: Mapping[str, Any], target: str) -> str:
    if not target or target.startswith("tests/") or not _path_exists_in_source(context, target):
        return ""
    try:
        text = (_source_root(context) / target).read_text(encoding="utf-8")
    except OSError:
        return ""
    package_dir = Path(target).parent
    source_files = set(_source_files_from_tree(context))
    for match in re.finditer(r"^\s*from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\b", text, flags=re.MULTILINE):
        candidate = (package_dir / f"{match.group(1)}.py").as_posix()
        if candidate not in source_files:
            continue
        direct = _direct_test_needed_before_patch(context, candidate)
        if direct:
            return direct
    return ""


def _normalization_patch_for_file(context: Mapping[str, Any], path: str) -> tuple[str, str]:
    if not path or not _path_exists_in_source(context, path):
        return "", ""
    workspace_path = (_workspace_root(context) / path).resolve()
    try:
        if workspace_path.exists() and '.replace(",", "")' in workspace_path.read_text(encoding="utf-8"):
            return "", "target file already contains separator normalization in the mirror"
    except OSError:
        pass
    source_path = (_source_root(context) / path).resolve()
    try:
        original = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return "", ""
    failure = latest_failure_text(context)
    has_grouped_number_failure = bool(re.search(r"\d{1,3},\d{3}", failure))
    if not has_grouped_number_failure:
        return "", ""
    text = "".join(original)
    replacements = [
        ('.replace("$", "")', '.replace("$", "").replace(",", "")'),
        (".replace('$', '')", ".replace('$', '').replace(',', '')"),
    ]
    updated_text = text
    reason = ""
    for old, new in replacements:
        if old in updated_text and new not in updated_text:
            updated_text = updated_text.replace(old, new, 1)
            reason = "normalization failure contains grouped numeric input and parser removes currency but not separators"
            break
    if not reason:
        assignment = re.search(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?\.strip\(\).*)$", text, flags=re.MULTILINE)
        if not assignment or '.replace(",", "")' in text:
            return "", ""
        indent, variable, line = assignment.group(1), assignment.group(2), assignment.group(0)
        insertion = f"{indent}{variable} = {variable}.replace(\",\", \"\")\n"
        updated_text = text.replace(line + "\n", line + "\n" + insertion, 1)
        reason = "normalization failure contains grouped numeric input and stripped value is parsed without separator removal"
    if updated_text == text:
        return "", ""
    diff = "".join(
        difflib.unified_diff(
            original,
            updated_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    return diff, reason


def _semantic_boundary_patch_for_file(context: Mapping[str, Any], path: str) -> tuple[str, str]:
    if not path or not _path_exists_in_source(context, path):
        return "", ""
    failure = f"{latest_failure_text(context)}\n{_text(context.get('instruction'))}".lower()
    if not any(token in failure for token in ("threshold", "boundary", "inclusive", "equal", "discount")):
        return "", ""
    source_path = (_source_root(context) / path).resolve()
    try:
        original = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError:
        return "", ""
    updated = list(original)
    reason = ""
    for index, line in enumerate(original):
        stripped = line.strip()
        if not stripped.startswith("if ") or ">=" in line or "!=" in line:
            continue
        match = re.match(r"^(\s*if\s+[^:\n<>!=]+?)\s*>\s*([^:\n]+:\s*(?:#.*)?\n?)$", line)
        if not match:
            continue
        right = match.group(2)
        if "threshold" not in right.lower() and not re.search(r"\b[A-Z][A-Z0-9_]*\b", right):
            continue
        updated[index] = f"{match.group(1)} >= {right}"
        reason = "semantic boundary failure indicates threshold comparison should include equality"
        break
    if not reason or updated == original:
        return "", ""
    diff = "".join(
        difflib.unified_diff(
            original,
            updated,
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    return diff, reason


def _bounded_patch_for_file(context: Mapping[str, Any], path: str) -> tuple[str, str]:
    for patcher in (_normalization_patch_for_file, _semantic_boundary_patch_for_file):
        patch, reason = patcher(context, path)
        if patch:
            return patch, reason
    return "", ""


def _patch_header_target(patch: str) -> str:
    for line in str(patch or "").splitlines():
        if not line.startswith("+++ "):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        for prefix in ("a/", "b/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
        return raw if raw not in {"/dev/null", "dev/null"} else ""
    return ""


def patch_fingerprint(patch: str) -> str:
    text = str(patch or "")
    if not text.strip():
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _patch_already_applied_or_verifying(context: Mapping[str, Any], kwargs: Mapping[str, Any]) -> tuple[bool, str]:
    state = _state(context)
    phase = _text(state.get("investigation_phase"))
    if phase == "verify":
        return True, "investigation is already in verify phase after a patch"
    target = _text(kwargs.get("path")) or _patch_header_target(str(kwargs.get("patch") or "")) or _text(_as_dict(state.get("grounding")).get("target_file"))
    if not target:
        return False, ""
    workspace_path = (_workspace_root(context) / target).resolve()
    try:
        if workspace_path.exists() and '.replace(",", "")' in workspace_path.read_text(encoding="utf-8"):
            return True, "patch effect is already present in mirror workspace"
    except OSError:
        return False, ""
    return False, ""


def _apply_patch_kwargs_from_action(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    kwargs = action.get("kwargs") if isinstance(action.get("kwargs"), Mapping) else tool_args.get("kwargs", {})
    return dict(kwargs or {}) if isinstance(kwargs, Mapping) else {}


def _action_function_name(action: Mapping[str, Any]) -> str:
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    return _text(
        action.get("function_name")
        or action.get("action")
        or tool_args.get("function_name")
        or tool_args.get("action")
        or payload.get("function_name")
        or payload.get("action")
    )


def _set_action_meta(action: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(action)
    meta = dict(updated.get("_candidate_meta", {}) if isinstance(updated.get("_candidate_meta", {}), Mapping) else {})
    meta.update(dict(updates))
    if "risk" in updates:
        updated["risk"] = updates["risk"]
    if "opportunity_estimate" in updates:
        updated["opportunity_estimate"] = updates["opportunity_estimate"]
    if "final_score" in updates:
        updated["final_score"] = updates["final_score"]
    updated["_candidate_meta"] = meta
    return updated


def _candidate_signature_path_patch(action: Mapping[str, Any]) -> tuple[str, str]:
    kwargs = _apply_patch_kwargs_from_action(action)
    patch = str(kwargs.get("patch") or "")
    path = _text(kwargs.get("path")) or _patch_header_target(patch)
    return path, patch_fingerprint(patch)


def _candidate_kwargs(action: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(action.get("payload"))
    tool_args = _as_dict(payload.get("tool_args"))
    for candidate in (action.get("kwargs"), action.get("args"), tool_args.get("kwargs"), tool_args.get("args")):
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


SIDE_EFFECT_ACTIONS_AFTER_COMPLETION = {
    "apply_patch",
    "propose_patch",
    "edit_replace_range",
    "edit_insert_after",
    "create_file",
    "delete_file",
    "mirror_exec",
    "mirror_plan",
    "mirror_apply",
    "internet_fetch",
    "internet_fetch_project",
}

COMPLETION_NOOP_ACTIONS = {"no_op_complete", "emit_final_report", "task_done", "wait"}


def _terminal_state(context: Mapping[str, Any]) -> str:
    return _text(_state(context).get("terminal_state"))


def is_local_machine_side_effect_action(action_name: str) -> bool:
    return str(action_name or "").strip() in SIDE_EFFECT_ACTIONS_AFTER_COMPLETION


def is_completed_verified_context(context: Mapping[str, Any]) -> bool:
    return _terminal_state(context) == "completed_verified"


def side_effect_after_completion_event(action_name: str, kwargs: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(context)
    return {
        **_nowish_event_base(action_name),
        "event_type": "side_effect_after_verified_completion",
        "requested_action": {"function_name": str(action_name or ""), "kwargs": dict(kwargs or {})},
        "terminal_state": _text(state.get("terminal_state")),
        "completion_reason": _text(state.get("completion_reason")),
        "terminal_tick": state.get("terminal_tick"),
        "suggested_replan_reason": "task is already completed_verified; emit a final report or no_op_complete instead",
    }


def _completion_noop_candidate(context: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(context)
    reason = _text(state.get("completion_reason")) or "verified completion already reached"
    terminal_state = _text(state.get("terminal_state")) or "completed_verified"
    return _make_call_action(
        "no_op_complete",
        {"reason": reason},
        {
            "terminal_completion_gate": True,
            "terminal_state": terminal_state,
            "completion_reason": reason,
            "terminal_tick": state.get("terminal_tick"),
            "risk": 0.0,
            "opportunity_estimate": 1.0,
            "final_score": 2.0,
        },
    )


def _target_binding(context: Mapping[str, Any]) -> dict[str, Any]:
    binding = _as_dict(_state(context).get("target_binding"))
    if binding:
        return binding
    try:
        from integrations.local_machine.target_binding import bind_target
    except Exception:
        return {}
    try:
        return _as_dict(bind_target(context))
    except Exception:
        return {}


def _budget_policy(context: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(context.get("llm_budget"))


def _prefers_llm_patch_proposals(context: Mapping[str, Any]) -> bool:
    return bool(context.get("prefer_llm_patch_proposals", False))


def _open_improvement_goal(context: Mapping[str, Any]) -> bool:
    text = _text(context.get("instruction")).lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "improve",
            "improvement",
            "enhance",
            "harden",
            "reliability",
            "quickstart",
            "docs",
            "documentation",
            "refactor",
            "cleanup",
            "low-risk",
            "small useful",
            "改进",
            "增强",
            "优化",
            "文档",
            "可靠性",
        )
    )


def _diff_entry_count(context: Mapping[str, Any]) -> int:
    summary = _as_dict(context.get("diff_summary"))
    for key in ("entry_count", "changed_file_count", "changed_paths_count"):
        try:
            value = int(summary.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _recent_action_history(context: Mapping[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    history = [
        dict(row)
        for row in _as_list(_state(context).get("action_history"))
        if isinstance(row, Mapping)
    ]
    return history[-max(1, int(limit)) :]


def _latest_stalled_event(context: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in _as_list(_state(context).get("stalled_events"))
        if isinstance(row, Mapping)
    ]
    return rows[-1] if rows else {}


def _stalled_loop_state(context: Mapping[str, Any]) -> tuple[bool, str, list[str]]:
    stalled_event = _latest_stalled_event(context)
    if stalled_event:
        recent_actions = [str(item) for item in _as_list(stalled_event.get("recent_actions")) if str(item)]
        return True, "investigation_stalled event was emitted by local-machine state", recent_actions
    if _diff_entry_count(context) > 0:
        return False, "", []
    recent = _recent_action_history(context, limit=4)
    if len(recent) < 3:
        return False, "", [str(row.get("function_name") or "") for row in recent]
    names = [str(row.get("function_name") or "") for row in recent]
    recent_names = names[-3:]
    progress_actions = {
        "file_read",
        "apply_patch",
        "propose_patch",
        "edit_replace_range",
        "candidate_files_set",
        "candidate_files_update",
        "note_write",
    }
    if any(name in progress_actions for name in recent_names):
        return False, "", names
    repeated_validation_actions = {"run_test", "repo_grep", "read_test_failure"}
    repeated_validation_count = sum(1 for name in recent_names if name in repeated_validation_actions)
    if repeated_validation_count < 3:
        return False, "", names
    if recent_names.count("run_test") >= 2:
        targets = [
            str(row.get("target") or row.get("path") or "")
            for row in recent[-3:]
            if str(row.get("function_name") or "") == "run_test"
        ]
        if len(set(targets)) <= 1:
            return True, "repeated run_test target produced no diff or new file evidence", names
    if len(set(recent_names)) <= 2:
        return True, "repeated validation/search actions produced no diff or new file evidence", names
    return False, "", names


def _stalled_recovery_query(context: Mapping[str, Any]) -> tuple[str, str]:
    preferred = (
        "runtime",
        "quickstart",
        "status",
        "watchdog",
        "approval",
        "mirror",
        "verification",
        "reliability",
        "config",
        "path",
    )
    tokens = _goal_tokens(context)
    for token in preferred:
        if token in tokens:
            return token, "goal token matched common local-machine improvement axis"
    query, source = choose_repo_grep_query(context)
    if query:
        return query, source
    for token in sorted(tokens):
        if len(token) >= 4:
            return token, "goal token fallback"
    return "", ""


def _stalled_loop_recovery_candidate(
    context: Mapping[str, Any],
    available: set[str],
    *,
    episode_trace: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    stalled, reason, recent_actions = _stalled_loop_state(context)
    if not stalled:
        return None
    binding = _target_binding(context)
    target_file = _text(binding.get("top_target_file")) or _text(_as_dict(_state(context).get("grounding")).get("target_file"))
    try:
        target_confidence = float(binding.get("target_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_confidence = 0.0
    function_name = ""
    kwargs: dict[str, Any] = {}
    recovery_reason = reason
    path, path_source = choose_file_read_path(context)
    recommended_action = _text(_latest_stalled_event(context).get("recommended_action"))
    evidence_gate = open_task_patch_evidence_gap(context, target_file=target_file)
    suggested_action = _text(evidence_gate.get("suggested_action"))
    suggested_kwargs = _as_dict(evidence_gate.get("suggested_kwargs"))
    if (
        (recommended_action == "repo_tree" or not _tree_entries(context))
        and "repo_tree" in available
    ):
        function_name = "repo_tree"
        kwargs = {"path": ".", "depth": 3, "max_entries": 200}
        recovery_reason = f"{reason}; take a bounded repository inventory before replanning"
    elif not bool(evidence_gate.get("sufficient", True)) and suggested_action in available:
        function_name = suggested_action
        kwargs = suggested_kwargs
        recovery_reason = f"{reason}; {evidence_gate.get('reason')}"
    elif path and "file_read" in available and not _file_has_been_read(context, path):
        function_name = "file_read"
        kwargs = {"path": path, "start_line": 1, "end_line": 240}
        recovery_reason = f"{reason}; inspect unread source from {path_source}"
        target_file = target_file or path
    elif (
        target_file
        and (target_confidence >= 0.55 or recommended_action == "propose_patch")
        and _file_has_been_read(context, target_file)
        and (_has_failure_evidence(context) or (_prefers_llm_patch_proposals(context) and _open_improvement_goal(context)))
        and "propose_patch" in available
    ):
        function_name = "propose_patch"
        kwargs = {"target_file": target_file, "max_changed_lines": 20}
        recovery_reason = f"{reason}; target is read and grounded, request verifier-gated bounded patch proposal"
    elif "repo_grep" in available:
        query, query_source = _stalled_recovery_query(context)
        if query:
            function_name = "repo_grep"
            kwargs = {"root": ".", "query": query, "globs": ["*.py", "*.md", "*.toml"], "max_matches": 50}
            recovery_reason = f"{reason}; search for a new evidence path from {query_source}"
    elif "investigation_status" in available:
        function_name = "investigation_status"
        kwargs = {}
        recovery_reason = f"{reason}; surface current investigation state before replanning"
    if not function_name:
        return None
    if available and function_name not in available:
        return None
    risk = 0.18 if function_name == "propose_patch" else 0.05
    return _make_call_action(
        function_name,
        kwargs,
        {
            "stalled_loop_recovery_bonus": 0.62,
            "progress_recovery_bonus": 0.62,
            "stalled_loop_recovery_reason": recovery_reason,
            "recent_actions": recent_actions,
            "target_file": target_file or _text(kwargs.get("path")) or _text(kwargs.get("target_file")),
            "target_confidence": target_confidence,
            "episode_trace_seen": bool(episode_trace),
            "risk": risk,
            "opportunity_estimate": 0.99,
            "final_score": 1.36,
        },
    )


def _fast_path_bridge_candidate(
    context: Mapping[str, Any],
    available: set[str],
    *,
    episode_trace: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    budget = _budget_policy(context)
    fast_path = _as_dict(budget.get("fast_path"))
    escalation = _as_dict(budget.get("escalation_path"))
    if str(budget.get("selected_path_hint") or "") != "fast_path":
        return None
    if not bool(fast_path.get("eligible", False)):
        return None
    target_file = _text(fast_path.get("target_file"))
    target_confidence = 0.0
    try:
        target_confidence = float(fast_path.get("target_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_confidence = 0.0
    state = _state(context)
    phase = _text(state.get("investigation_phase")) or "discover"
    grounding = _as_dict(state.get("grounding"))
    function_name = ""
    kwargs: dict[str, Any] = {}
    reason = ""
    if phase == "verify" and "run_test" in available:
        target, source = choose_verification_target(context)
        function_name = "run_test"
        kwargs = {"target": target, "timeout_seconds": 30}
        reason = f"fast path keeps verification deterministic after patch; target from {source}"
    elif phase == "complete-ready" and "mirror_plan" in available:
        function_name = "mirror_plan"
        kwargs = {}
        reason = "fast path reached verified completion; sync planning is now allowed"
    elif target_file and "file_read" in available and not _file_has_been_read(context, target_file):
        function_name = "file_read"
        kwargs = {"path": target_file, "start_line": 1, "end_line": 240}
        reason = "fast path has high-confidence target binding; read target before patch"
    elif target_file and "apply_patch" in available and _file_has_been_read(context, target_file):
        patch_kwargs, patch_reason, patched_target = choose_apply_patch_kwargs(context)
        if patch_kwargs:
            function_name = "apply_patch"
            kwargs = patch_kwargs
            if patched_target:
                kwargs.setdefault("path", patched_target)
            reason = f"fast path uses deterministic bounded patch before LLM escalation; {patch_reason}"
    if not function_name:
        return None
    if available and function_name not in available:
        return None
    return _make_call_action(
        function_name,
        kwargs,
        {
            "budget_path_hint": "fast_path",
            "fast_path_bonus": 0.55,
            "budget_action_reason": reason,
            "budget_fast_path_reasons": list(fast_path.get("reasons", []) or []),
            "budget_escalation_triggers": list(escalation.get("triggers", []) or []),
            "target_file": target_file or _text(kwargs.get("path")),
            "target_confidence": target_confidence,
            "llm_layer_preference": "deterministic",
            "episode_trace_seen": bool(episode_trace),
            "verification_pending": bool(_as_dict(grounding.get("last_patch")) and phase == "verify"),
            "risk": 0.06 if function_name in {"file_read", "run_test"} else 0.14,
            "opportunity_estimate": 0.99,
            "final_score": 1.32,
        },
    )


def annotate_local_machine_patch_ranking(
    candidate_actions: Sequence[Any],
    obs: Mapping[str, Any],
    *,
    episode_trace: Sequence[Mapping[str, Any]] | None = None,
) -> list[Any]:
    mirror = _local_mirror(obs)
    if not mirror:
        return list(candidate_actions or [])
    context = _bridge_context_from_obs(obs, episode_trace=episode_trace)
    state = _state(context)
    budget = _budget_policy(context)
    budget_hint = _text(budget.get("selected_path_hint"))
    fast_path = _as_dict(budget.get("fast_path"))
    progress_stalled, stalled_reason, stalled_recent_actions = _stalled_loop_state(context)
    if _text(state.get("terminal_state")) in {"completed_verified", "needs_human_review"}:
        annotated_terminal: list[Any] = []
        for action in list(candidate_actions or []):
            if not isinstance(action, Mapping):
                annotated_terminal.append(action)
                continue
            fn = _action_function_name(action)
            if fn in SIDE_EFFECT_ACTIONS_AFTER_COMPLETION:
                annotated_terminal.append(
                    _set_action_meta(
                        action,
                        {
                            "completion_gate_penalty": 1.0,
                            "terminal_state": _text(state.get("terminal_state")),
                            "completion_reason": _text(state.get("completion_reason")),
                            "terminal_tick": state.get("terminal_tick"),
                            "risk": 1.0,
                            "opportunity_estimate": 0.0,
                            "final_score": -1.0,
                        },
                    )
                )
                continue
            if fn in COMPLETION_NOOP_ACTIONS:
                annotated_terminal.append(
                    _set_action_meta(
                        action,
                        {
                            "terminal_completion_gate": True,
                            "terminal_state": "completed_verified",
                            "risk": 0.0,
                            "opportunity_estimate": 1.0,
                            "final_score": 2.0,
                        },
                    )
                )
                continue
            annotated_terminal.append(action)
        return annotated_terminal
    grounding = _as_dict(state.get("grounding"))
    last_patch = _as_dict(grounding.get("last_patch"))
    phase = _text(state.get("investigation_phase"))
    verification_pending = bool(last_patch and phase == "verify")
    open_task_gate = open_task_patch_evidence_gap(
        context,
        target_file=_text(grounding.get("target_file")) or _target_binding_file(context),
    )
    open_task_gate_active = not bool(open_task_gate.get("sufficient", True))
    if not verification_pending and budget_hint != "fast_path" and not progress_stalled and not open_task_gate_active:
        return list(candidate_actions or [])
    applied_paths = set(str(path) for path in _as_list(last_patch.get("touched_files")) if str(path))
    applied_fingerprint = _text(last_patch.get("patch_sha256"))
    annotated: list[Any] = []
    for action in list(candidate_actions or []):
        if not isinstance(action, Mapping):
            annotated.append(action)
            continue
        fn = _action_function_name(action)
        if open_task_gate_active:
            suggested_action = _text(open_task_gate.get("suggested_action"))
            suggested_kwargs = _as_dict(open_task_gate.get("suggested_kwargs"))
            action_kwargs = _candidate_kwargs(action)
            if fn == "propose_patch":
                annotated.append(
                    _set_action_meta(
                        action,
                        {
                            "open_task_evidence_gate_penalty": 0.72,
                            "open_task_evidence_gate": dict(open_task_gate),
                            "posterior_action_reason": "open-ended improvement patch proposal needs project overview, tests, and related source evidence first",
                            "risk": 0.78,
                            "opportunity_estimate": 0.12,
                            "final_score": -0.35,
                        },
                    )
                )
                continue
            if fn == suggested_action:
                expected_path = _text(suggested_kwargs.get("path") or suggested_kwargs.get("target"))
                candidate_path = _text(action_kwargs.get("path") or action_kwargs.get("target"))
                if not expected_path or not candidate_path or expected_path == candidate_path:
                    annotated.append(
                        _set_action_meta(
                            action,
                            {
                                "open_task_evidence_gate_bonus": 0.46,
                                "open_task_evidence_gate": dict(open_task_gate),
                                "posterior_action_reason": str(open_task_gate.get("reason") or ""),
                                "risk": 0.05 if fn in {"file_read", "repo_tree", "run_test"} else 0.12,
                                "opportunity_estimate": 0.98,
                                "final_score": 1.18,
                            },
                        )
                    )
                    continue
        if progress_stalled:
            if fn in {"run_test", "read_test_failure"}:
                annotated.append(
                    _set_action_meta(
                        action,
                        {
                            "stalled_loop_penalty": 0.78,
                            "action_cooldown_recommended": True,
                            "recent_action_feedback": {
                                "consecutive_no_progress_count": len(stalled_recent_actions),
                                "positive_progress_count": 0,
                                "action_cooldown_recommended": True,
                                "reason": stalled_reason,
                            },
                            "stalled_loop_recovery_reason": stalled_reason,
                            "recent_actions": stalled_recent_actions,
                            "risk": 0.74,
                            "opportunity_estimate": 0.08,
                            "final_score": -0.45,
                        },
                    )
                )
                continue
            if fn == "repo_grep" and "repo_grep" in stalled_recent_actions:
                annotated.append(
                    _set_action_meta(
                        action,
                        {
                            "stalled_loop_penalty": 0.35,
                            "action_cooldown_recommended": True,
                            "stalled_loop_recovery_reason": stalled_reason,
                            "recent_actions": stalled_recent_actions,
                            "risk": 0.35,
                            "opportunity_estimate": 0.35,
                            "final_score": 0.12,
                        },
                    )
                )
                continue
            if fn in {"file_read", "propose_patch", "candidate_files_set", "candidate_files_update", "investigation_status"}:
                annotated.append(
                    _set_action_meta(
                        action,
                        {
                            "stalled_loop_recovery_bonus": 0.34,
                            "progress_recovery_bonus": 0.34,
                            "stalled_loop_recovery_reason": stalled_reason,
                            "recent_actions": stalled_recent_actions,
                            "risk": 0.2 if fn == "propose_patch" else 0.06,
                            "opportunity_estimate": 0.94,
                            "final_score": 1.08,
                        },
                    )
                )
                continue
        if budget_hint == "fast_path":
            meta_updates = {
                "budget_path_hint": "fast_path",
                "budget_fast_path_eligible": bool(fast_path.get("eligible", False)),
                "budget_action_reason": "budget policy prefers deterministic fast path before LLM escalation",
            }
            if fn in {"repo_tree", "repo_find", "repo_grep", "file_read", "run_test", "run_lint", "run_typecheck"}:
                meta_updates.update(
                    {
                        "fast_path_candidate_bonus": 0.18,
                        "risk": 0.08,
                        "opportunity_estimate": 0.96,
                    }
                )
                annotated.append(_set_action_meta(action, meta_updates))
                continue
            if fn in {"propose_patch", "mirror_exec"}:
                meta_updates.update(
                    {
                        "fast_path_llm_escalation_penalty": 0.28,
                        "risk": 0.38,
                        "opportunity_estimate": 0.55,
                    }
                )
                annotated.append(_set_action_meta(action, meta_updates))
                continue
        if fn == "apply_patch":
            patch_path, fingerprint = _candidate_signature_path_patch(action)
            same_path = bool(patch_path and patch_path in applied_paths)
            same_patch = bool(fingerprint and fingerprint == applied_fingerprint)
            if same_path or same_patch:
                annotated.append(
                    _set_action_meta(
                        action,
                        {
                            "stale_patch_penalty": 0.85,
                            "patch_fingerprint": fingerprint or applied_fingerprint,
                            "verification_pending": True,
                            "posterior_action_reason": "same patch/path already applied; verify before any further patch",
                            "risk": 0.95,
                            "opportunity_estimate": 0.05,
                            "final_score": -0.8,
                        },
                    )
                )
                continue
        if fn == "run_test":
            meta_updates = {
                "verify_after_patch_bonus": 0.45,
                "patch_fingerprint": applied_fingerprint,
                "verification_pending": True,
                "posterior_action_reason": "patch applied and verification is pending",
                "risk": 0.05,
                "opportunity_estimate": 0.99,
                "final_score": 1.25,
            }
            annotated.append(_set_action_meta(action, meta_updates))
            continue
        annotated.append(action)
    return annotated


def choose_apply_patch_kwargs(context: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    state = _state(context)
    target = _text(_as_dict(state.get("grounding")).get("target_file"))
    target = target or target_file_from_failure(context) or _top_hypothesis_target(context)
    if not target:
        return {}, "no target_file is bound to the leading hypothesis or failure evidence", ""
    if not _file_has_been_read(context, target):
        return {}, "target_file is known but file content has not been read", target
    if not _has_failure_evidence(context):
        return {}, "target_file is read but no failure evidence is available", target
    patch, reason = _bounded_patch_for_file(context, target)
    if not patch:
        for alternate in _traceback_source_files(context):
            if alternate == target or not _file_has_been_read(context, alternate):
                continue
            patch, reason = _bounded_patch_for_file(context, alternate)
            if patch:
                target = alternate
                reason = f"{reason}; shifted from primary traceback file to deeper failing implementation"
                break
    if not patch:
        return {}, "no bounded patch could be inferred from failure evidence and read file content", target
    run_ref = _text(_latest_failed_run_output(context).get("run_ref"))
    evidence_refs = [_line_ref_for_last_read(context, target)]
    if run_ref:
        evidence_refs.insert(0, f"run:{run_ref}")
    return {
        "patch": patch,
        "max_files": 1,
        "max_hunks": 3,
        "evidence_refs": evidence_refs,
    }, reason, target


def _missing_fields(schema: LocalMachineActionSchema, kwargs: Mapping[str, Any]) -> list[str]:
    missing = []
    for field in schema.required_kwargs:
        value = kwargs.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def _repaired_payload(
    *,
    action_name: str,
    original_kwargs: Mapping[str, Any],
    repaired_kwargs: Mapping[str, Any],
    repair_reason: str,
    repair_source: str,
) -> dict[str, Any]:
    return {
        **_nowish_event_base(action_name),
        "event_type": "local_machine_action_kwargs_repaired",
        "original_action": {"function_name": action_name, "kwargs": dict(original_kwargs)},
        "repaired_action": {"function_name": action_name, "kwargs": dict(repaired_kwargs)},
        "repair_reason": repair_reason,
        "repair_source": repair_source,
    }


def _invalid_payload(
    *,
    action_name: str,
    kwargs: Mapping[str, Any],
    missing_fields: Sequence[str],
    reason: str,
) -> dict[str, Any]:
    return {
        **_nowish_event_base(action_name),
        "event_type": "invalid_action_kwargs",
        "requested_action": {"function_name": action_name, "kwargs": dict(kwargs)},
        "missing_fields": [str(item) for item in missing_fields],
        "suggested_replan_reason": reason,
    }


def validate_local_machine_action(
    action_name: str,
    kwargs: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    schema = LOCAL_MACHINE_ACTION_SCHEMA_REGISTRY.get(str(action_name or ""))
    original_kwargs = dict(kwargs or {})
    if schema is None:
        return {"status": "valid", "function_name": action_name, "kwargs": original_kwargs}
    missing = _missing_fields(schema, original_kwargs)
    if action_name == "apply_patch" and not missing:
        already_applied, stale_reason = _patch_already_applied_or_verifying(context, original_kwargs)
        if already_applied:
            target, repair_source = choose_verification_target(context)
            repaired = {"target": target, "timeout_seconds": 30}
            event = _repaired_payload(
                action_name=action_name,
                original_kwargs=original_kwargs,
                repaired_kwargs=repaired,
                repair_reason=f"{stale_reason}; verify instead of reapplying patch",
                repair_source=repair_source,
            )
            event["repaired_action"]["function_name"] = "run_test"
            event["stale_apply_patch_repaired"] = True
            event["stale_patch_penalty"] = 0.85
            event["verify_after_patch_bonus"] = 0.45
            event["verification_pending"] = True
            event["patch_fingerprint"] = patch_fingerprint(str(original_kwargs.get("patch") or ""))
            return {
                "status": "repaired",
                "function_name": "run_test",
                "kwargs": repaired,
                "schema": schema.to_dict(),
                "event": event,
            }
    if not missing:
        if action_name == "file_read":
            original_path = _text(original_kwargs.get("path"))
            evidence_gate = open_task_patch_evidence_gap(context, target_file=original_path)
            suggested_action = _text(evidence_gate.get("suggested_action"))
            suggested_kwargs = _as_dict(evidence_gate.get("suggested_kwargs"))
            suggested_path = _text(suggested_kwargs.get("path") or suggested_kwargs.get("target"))
            if (
                original_path
                and _file_has_been_read(context, original_path)
                and not bool(evidence_gate.get("sufficient", True))
                and suggested_action
                and (suggested_action != "file_read" or suggested_path != original_path)
            ):
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=suggested_kwargs,
                    repair_reason="file_read target was already read; continue open-task evidence collection instead of repeating it",
                    repair_source=str(evidence_gate.get("reason") or "open_task_evidence_gate"),
                )
                event["repaired_action"]["function_name"] = suggested_action
                event["stale_file_read_repaired"] = True
                event["open_task_evidence_gate"] = dict(evidence_gate)
                return {
                    "status": "repaired",
                    "function_name": suggested_action,
                    "kwargs": suggested_kwargs,
                    "schema": schema.to_dict(),
                    "event": event,
                }
        if action_name == "propose_patch":
            target_file = _text(original_kwargs.get("target_file")) or _target_binding_file(context)
            evidence_gate = open_task_patch_evidence_gap(context, target_file=target_file)
            suggested_action = _text(evidence_gate.get("suggested_action"))
            suggested_kwargs = _as_dict(evidence_gate.get("suggested_kwargs"))
            if (
                not bool(evidence_gate.get("sufficient", True))
                and suggested_action
                and suggested_action != "propose_patch"
            ):
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=suggested_kwargs,
                    repair_reason="patch proposal lacked enough open-task evidence; execute the next evidence action first",
                    repair_source=str(evidence_gate.get("reason") or "open_task_evidence_gate"),
                )
                event["repaired_action"]["function_name"] = suggested_action
                event["premature_propose_patch_repaired"] = True
                event["open_task_evidence_gate"] = dict(evidence_gate)
                return {
                    "status": "repaired",
                    "function_name": suggested_action,
                    "kwargs": suggested_kwargs,
                    "schema": schema.to_dict(),
                    "event": event,
                }
        if action_name == "repo_grep":
            query = _text(original_kwargs.get("query") or original_kwargs.get("pattern"))
            better_query, better_query_source = choose_repo_grep_query(context)
            if (
                query
                and better_query
                and better_query_source == "goal code token"
                and query.lower() in LOW_VALUE_GREP_QUERY_TOKENS
                and better_query.lower() != query.lower()
            ):
                grep_read_kwargs = _file_read_kwargs_for_last_grep_match(context, better_query)
                if grep_read_kwargs:
                    event = _repaired_payload(
                        action_name=action_name,
                        original_kwargs=original_kwargs,
                        repaired_kwargs=grep_read_kwargs,
                        repair_reason="repo_grep query was a low-value generic token and the better code-token search already found an unread match",
                        repair_source=better_query_source,
                    )
                    event["repaired_action"]["function_name"] = "file_read"
                    event["low_value_repo_grep_query_repaired"] = True
                    event["replacement_query"] = better_query
                    return {
                        "status": "repaired",
                        "function_name": "file_read",
                        "kwargs": grep_read_kwargs,
                        "schema": schema.to_dict(),
                        "event": event,
                    }
                evidence_gate = open_task_patch_evidence_gap(context)
                target_file = _target_binding_file(context) or _top_hypothesis_target(context) or _last_grep_file(context)
                if (
                    target_file
                    and _file_has_been_read(context, target_file)
                    and bool(evidence_gate.get("sufficient", False))
                    and float(_target_binding(context).get("target_confidence", 0.0) or 0.0) >= 0.65
                ):
                    repaired_kwargs = {"target_file": target_file, "max_changed_lines": 20}
                    event = _repaired_payload(
                        action_name=action_name,
                        original_kwargs=original_kwargs,
                        repaired_kwargs=repaired_kwargs,
                        repair_reason="issue code-token evidence is sufficient; stop repeating low-value search and propose a bounded patch",
                        repair_source=better_query_source,
                    )
                    event["repaired_action"]["function_name"] = "propose_patch"
                    event["low_value_repo_grep_query_repaired"] = True
                    event["replacement_query"] = better_query
                    event["issue_evidence_to_patch_bridge"] = True
                    event["open_task_evidence_gate"] = dict(evidence_gate)
                    return {
                        "status": "repaired",
                        "function_name": "propose_patch",
                        "kwargs": repaired_kwargs,
                        "schema": schema.to_dict(),
                        "event": event,
                    }
                repaired_kwargs = dict(original_kwargs)
                repaired_kwargs["query"] = better_query
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=repaired_kwargs,
                    repair_reason="repo_grep query was a low-value generic token; use code token from task evidence",
                    repair_source=better_query_source,
                )
                event["low_value_repo_grep_query_repaired"] = True
                event["replacement_query"] = better_query
                return {
                    "status": "repaired",
                    "function_name": "repo_grep",
                    "kwargs": repaired_kwargs,
                    "schema": schema.to_dict(),
                    "event": event,
                }
            evidence_gate = open_task_patch_evidence_gap(context)
            suggested_action = _text(evidence_gate.get("suggested_action"))
            suggested_kwargs = _as_dict(evidence_gate.get("suggested_kwargs"))
            if (
                query
                and _recent_same_empty_repo_grep(context, query)
                and not bool(evidence_gate.get("sufficient", True))
                and suggested_action
                and suggested_action != "repo_grep"
            ):
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=suggested_kwargs,
                    repair_reason="repo_grep repeated the same empty query; continue open-task evidence collection instead",
                    repair_source=str(evidence_gate.get("reason") or "open_task_evidence_gate"),
                )
                event["repaired_action"]["function_name"] = suggested_action
                event["stale_repo_grep_repaired"] = True
                event["open_task_evidence_gate"] = dict(evidence_gate)
                return {
                    "status": "repaired",
                    "function_name": suggested_action,
                    "kwargs": suggested_kwargs,
                    "schema": schema.to_dict(),
                    "event": event,
                }
        if action_name == "run_test":
            original_target = _text(original_kwargs.get("target"))
            evidence_gate = open_task_patch_evidence_gap(context)
            suggested_action = _text(evidence_gate.get("suggested_action"))
            suggested_kwargs = _as_dict(evidence_gate.get("suggested_kwargs"))
            if (
                original_target
                and _successful_test_has_been_run(context, original_target)
                and not bool(evidence_gate.get("sufficient", True))
                and suggested_action
                and (suggested_action != "run_test" or _text(suggested_kwargs.get("target")) != original_target)
            ):
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=suggested_kwargs,
                    repair_reason="run_test target already passed; continue open-task evidence collection instead of rerunning it",
                    repair_source=str(evidence_gate.get("reason") or "open_task_evidence_gate"),
                )
                event["repaired_action"]["function_name"] = suggested_action
                event["repeated_successful_run_test_repaired"] = True
                event["open_task_evidence_gate"] = dict(evidence_gate)
                return {
                    "status": "repaired",
                    "function_name": suggested_action,
                    "kwargs": suggested_kwargs,
                    "schema": schema.to_dict(),
                    "event": event,
                }
            if (
                original_target
                and _successful_test_has_been_run(context, original_target)
                and _open_improvement_goal(context)
                and not _has_failure_evidence(context)
                and _diff_entry_count(context) <= 0
            ):
                target_file = _target_binding_file(context) or _top_hypothesis_target(context) or _tree_relevant_source_file(context)
                if target_file and _file_has_been_read(context, target_file):
                    repaired_kwargs = {"target_file": target_file, "max_changed_lines": 20}
                    repaired_action = "propose_patch"
                    repair_reason = "run_test target already passed and open-task evidence is complete; request bounded patch proposal instead of rerunning"
                else:
                    path, path_source = choose_file_read_path(context)
                    repaired_kwargs = {"path": path, "start_line": 1, "end_line": 240} if path else {}
                    repaired_action = "file_read" if path else ""
                    repair_reason = f"run_test target already passed; inspect source evidence from {path_source} instead of rerunning"
                if repaired_action:
                    event = _repaired_payload(
                        action_name=action_name,
                        original_kwargs=original_kwargs,
                        repaired_kwargs=repaired_kwargs,
                        repair_reason=repair_reason,
                        repair_source="successful_test_cooldown",
                    )
                    event["repaired_action"]["function_name"] = repaired_action
                    event["repeated_successful_run_test_repaired"] = True
                    event["successful_test_cooldown"] = True
                    event["open_task_evidence_gate"] = dict(evidence_gate)
                    return {
                        "status": "repaired",
                        "function_name": repaired_action,
                        "kwargs": repaired_kwargs,
                        "schema": schema.to_dict(),
                        "event": event,
                    }
            repaired_target, repair_source = repair_run_test_source_target(
                context,
                str(original_kwargs.get("target") or ""),
            )
            if repaired_target:
                repaired = dict(original_kwargs)
                repaired["target"] = repaired_target
                repaired.setdefault("timeout_seconds", 30)
                event = _repaired_payload(
                    action_name=action_name,
                    original_kwargs=original_kwargs,
                    repaired_kwargs=repaired,
                    repair_reason="run_test target was a source file, not a pytest test target",
                    repair_source=repair_source,
                )
                return {
                    "status": "repaired",
                    "function_name": action_name,
                    "kwargs": repaired,
                    "schema": schema.to_dict(),
                    "event": event,
                }
        return {
            "status": "valid",
            "function_name": action_name,
            "kwargs": original_kwargs,
            "schema": schema.to_dict(),
        }
    if schema.allow_empty_kwargs and not original_kwargs:
        return {
            "status": "valid",
            "function_name": action_name,
            "kwargs": original_kwargs,
            "schema": schema.to_dict(),
        }

    repaired = dict(original_kwargs)
    repair_source = ""
    repair_reason = ""
    if action_name == "file_read" and "path" in missing:
        path, repair_source = choose_file_read_path(context)
        if path:
            repaired["path"] = path
            repaired.setdefault("start_line", 1)
            repaired.setdefault("end_line", 240)
            repair_reason = "file_read path inferred from investigation evidence"
    elif action_name == "repo_grep" and "query" in missing:
        query, repair_source = choose_repo_grep_query(context)
        if query:
            repaired["query"] = query
            repaired.setdefault("root", ".")
            repaired.setdefault("globs", ["*.py"])
            repaired.setdefault("max_matches", 50)
            repair_reason = "repo_grep query inferred from goal/failure/hypothesis text"
    elif action_name == "run_test" and "target" in missing:
        target, repair_source = choose_run_test_target(context)
        repaired["target"] = target
        repaired.setdefault("timeout_seconds", 30)
        repair_reason = "run_test target inferred from failed test or full pytest fallback"
    elif action_name == "run_typecheck" and "target" in missing:
        target, repair_source = choose_typecheck_target(context)
        repaired["target"] = target
        repaired.setdefault("timeout_seconds", 30)
        repair_reason = "run_typecheck target inferred from source investigation state"
    elif action_name == "apply_patch" and "patch" in missing:
        patch_kwargs, repair_reason, target = choose_apply_patch_kwargs(context)
        if patch_kwargs:
            repaired.update(patch_kwargs)
            repaired.setdefault("path", target)
            repair_source = "leading target file with read content and failure evidence"

    still_missing = _missing_fields(schema, repaired)
    if still_missing:
        invalid = _invalid_payload(
            action_name=action_name,
            kwargs=original_kwargs,
            missing_fields=still_missing,
            reason=repair_reason or f"{action_name} kwargs could not be repaired by {schema.repair_strategy}",
        )
        return {
            "status": "invalid",
            "function_name": action_name,
            "kwargs": original_kwargs,
            "schema": schema.to_dict(),
            "missing_fields": still_missing,
            "event": invalid,
        }
    event = _repaired_payload(
        action_name=action_name,
        original_kwargs=original_kwargs,
        repaired_kwargs=repaired,
        repair_reason=repair_reason,
        repair_source=repair_source or schema.repair_strategy,
    )
    return {
        "status": "repaired",
        "function_name": action_name,
        "kwargs": repaired,
        "schema": schema.to_dict(),
        "event": event,
    }


def _local_mirror(obs: Mapping[str, Any]) -> dict[str, Any]:
    mirror = _as_dict(obs.get("local_mirror"))
    if mirror:
        return mirror
    raw = _as_dict(obs.get("raw"))
    return _as_dict(raw.get("local_mirror"))


def _latest_posterior_summary(episode_trace: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    for row in reversed(list(episode_trace or [])):
        summary = _as_dict(row.get("posterior_summary")) if isinstance(row, Mapping) else {}
        if summary:
            return summary
    return {}


def _bridge_context_from_obs(
    obs: Mapping[str, Any],
    *,
    episode_trace: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    mirror = _local_mirror(obs)
    manifest_control = _text(mirror.get("control_root"))
    return {
        "instruction": _text(mirror.get("instruction") or obs.get("text")),
        "source_root": _text(mirror.get("source_root")),
        "mirror_root": _text(mirror.get("mirror_root")),
        "workspace_root": _text(mirror.get("workspace_root")),
        "run_output_root": str(Path(manifest_control) / "run_outputs") if manifest_control else "",
        "investigation_state": _as_dict(mirror.get("investigation")),
        "diff_summary": _as_dict(mirror.get("diff_summary")),
        "prefer_llm_patch_proposals": bool(mirror.get("prefer_llm_patch_proposals", False)),
        "default_command_present": bool(mirror.get("default_command_present", False)),
        "posterior_summary": _latest_posterior_summary(episode_trace),
        "llm_budget": _as_dict(mirror.get("llm_budget")),
        "episode_read_paths": sorted(_episode_file_read_paths(episode_trace)),
        "episode_run_test_targets": sorted(_episode_run_test_targets(episode_trace)),
    }


def _episode_file_read_paths(episode_trace: Sequence[Mapping[str, Any]] | None) -> set[str]:
    paths: set[str] = set()
    for row in list(episode_trace or []):
        if not isinstance(row, Mapping):
            continue
        outcome = _as_dict(row.get("outcome"))
        if str(outcome.get("function_name") or "") == "file_read":
            path = _text(outcome.get("path"))
            if path:
                paths.add(path)
        action = _as_dict(row.get("action"))
        snapshot = _as_dict(row.get("action_snapshot"))
        payload = _as_dict(action.get("payload"))
        tool_args = _as_dict(payload.get("tool_args"))
        function_name = _text(
            snapshot.get("function_name")
            or action.get("function_name")
            or tool_args.get("function_name")
        )
        kwargs = _as_dict(snapshot.get("kwargs")) or _as_dict(action.get("kwargs")) or _as_dict(tool_args.get("kwargs"))
        if function_name == "file_read":
            path = _text(kwargs.get("path"))
            if path:
                paths.add(path)
    return paths


def _episode_run_test_targets(episode_trace: Sequence[Mapping[str, Any]] | None) -> set[str]:
    targets: set[str] = set()
    for row in list(episode_trace or []):
        if not isinstance(row, Mapping):
            continue
        outcome = _as_dict(row.get("outcome"))
        if str(outcome.get("function_name") or "") == "run_test":
            command = [str(part) for part in _as_list(outcome.get("command"))]
            if command:
                targets.add(command[-1])
        action = _as_dict(row.get("action"))
        snapshot = _as_dict(row.get("action_snapshot"))
        payload = _as_dict(action.get("payload"))
        tool_args = _as_dict(payload.get("tool_args"))
        function_name = _text(
            snapshot.get("function_name")
            or action.get("function_name")
            or tool_args.get("function_name")
        )
        kwargs = _as_dict(snapshot.get("kwargs")) or _as_dict(action.get("kwargs")) or _as_dict(tool_args.get("kwargs"))
        if function_name == "run_test":
            target = _text(kwargs.get("target"))
            if target:
                targets.add(target)
    return targets


def _make_call_action(function_name: str, kwargs: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "call_tool",
        "function_name": function_name,
        "kwargs": dict(kwargs),
        "payload": {
            "tool_name": "call_hidden_function",
            "tool_args": {"function_name": function_name, "kwargs": dict(kwargs)},
        },
        "_source": "local_machine_action_grounding_bridge",
        "_candidate_meta": dict(meta),
        "risk": float(meta.get("risk", 0.12)),
        "opportunity_estimate": float(meta.get("opportunity_estimate", 0.92)),
        "final_score": float(meta.get("final_score", 0.88)),
    }


def build_local_machine_posterior_action_bridge_candidate(
    obs: Mapping[str, Any],
    *,
    episode_trace: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    mirror = _local_mirror(obs)
    if not mirror:
        return None
    available = set(str(item) for item in _as_list(obs.get("available_functions")))
    novel_api = _as_dict(obs.get("novel_api"))
    available.update(str(item) for item in _as_list(novel_api.get("visible_functions")))
    available.update(str(item) for item in _as_list(novel_api.get("available_functions")))
    available.update(str(key) for key in _as_dict(obs.get("function_signatures")).keys())
    registry = _as_dict(_as_dict(mirror.get("action_schema_registry")).get("actions"))
    available.update(str(key) for key in registry.keys())
    context = _bridge_context_from_obs(obs, episode_trace=episode_trace)
    variant = _probe_variant(context)
    state = _state(context)
    phase = _text(state.get("investigation_phase")) or "discover"
    prefer_llm_patch_proposals = bool(mirror.get("prefer_llm_patch_proposals", False))
    if _text(state.get("terminal_state")) in {"completed_verified", "needs_human_review"}:
        return _completion_noop_candidate(context)
    fast_path_candidate = _fast_path_bridge_candidate(
        context,
        available,
        episode_trace=episode_trace,
    )
    if isinstance(fast_path_candidate, dict):
        return fast_path_candidate
    is_stalled, _, _ = _stalled_loop_state(context)
    open_task_gate = open_task_patch_evidence_gap(context)
    if not is_stalled and not bool(open_task_gate.get("sufficient", True)):
        suggested_action = _text(open_task_gate.get("suggested_action"))
        suggested_kwargs = _as_dict(open_task_gate.get("suggested_kwargs"))
        if suggested_action in available:
            return _make_call_action(
                suggested_action,
                suggested_kwargs,
                {
                    "open_task_evidence_gate_bonus": 0.58,
                    "open_task_evidence_gate": dict(open_task_gate),
                    "posterior_action_reason": str(open_task_gate.get("reason") or ""),
                    "target_file": _text(suggested_kwargs.get("path") or suggested_kwargs.get("target")),
                    "risk": 0.05 if suggested_action in {"repo_tree", "file_read", "run_test"} else 0.12,
                    "opportunity_estimate": 0.98,
                    "final_score": 1.34,
                },
            )
    stalled_recovery_candidate = _stalled_loop_recovery_candidate(
        context,
        available,
        episode_trace=episode_trace,
    )
    if isinstance(stalled_recovery_candidate, dict):
        return stalled_recovery_candidate
    posterior = _latest_posterior_summary(episode_trace)
    leading_posterior = 0.0
    try:
        leading_posterior = float(posterior.get("leading_posterior", 0.0) or 0.0)
    except (TypeError, ValueError):
        leading_posterior = 0.0
    leading_id = _text(posterior.get("leading_hypothesis_id") or posterior.get("leading_hypothesis_object_id"))
    binding = _target_binding(context)
    bound_target_file = _text(binding.get("top_target_file"))
    target_confidence = 0.0
    try:
        target_confidence = float(binding.get("target_confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_confidence = 0.0
    grounding_target_file = _text(_as_dict(state.get("grounding")).get("target_file"))
    if bound_target_file and target_confidence >= 0.55:
        target_file = bound_target_file
    else:
        target_file = grounding_target_file or bound_target_file or target_file_from_failure(context)

    bonus = 0.0
    reason = ""
    function_name = ""
    kwargs: dict[str, Any] = {}
    grounding = _as_dict(state.get("grounding"))
    if phase == "verify" and "run_test" in available:
        verify_target = "." if _text(grounding.get("last_successful_test_target")) else (_latest_failed_test_target(context) or ".")
        function_name = "run_test"
        kwargs = {"target": verify_target, "timeout_seconds": 30}
        bonus = 0.25 if leading_posterior >= 0.75 else 0.0
        reason = "patch succeeded; verify the repaired behavior before completion"
    elif phase == "complete-ready" and "mirror_plan" in available:
        function_name = "mirror_plan"
        kwargs = {}
        bonus = 0.2 if leading_posterior >= 0.75 else 0.0
        reason = "full verification passed; sync plan is now completion-ready"
    elif (
        variant not in {"no_posterior", "no_discriminating_experiment"}
        and phase == "patch"
        and target_file
        and _file_has_been_read(context, target_file)
    ):
        direct_test = _direct_test_needed_before_patch(context, target_file)
        downstream_direct_test = _downstream_direct_test_needed_before_patch(context, target_file)
        if direct_test and "run_test" in available:
            function_name = "run_test"
            kwargs = {"target": direct_test, "timeout_seconds": 30}
            bonus = 0.32
            reason = "localized implementation has a matching direct unit test; run it before patching"
        elif downstream_direct_test and "run_test" in available:
            function_name = "run_test"
            kwargs = {"target": downstream_direct_test, "timeout_seconds": 30}
            bonus = 0.33
            reason = "localized wrapper imports downstream implementation with a direct test; run it before patching"
        else:
            patch_kwargs, patch_reason, _ = choose_apply_patch_kwargs(context)
            if patch_kwargs and prefer_llm_patch_proposals and "propose_patch" in available:
                function_name = "propose_patch"
                kwargs = {"target_file": target_file, "max_changed_lines": 20}
                bonus = 0.36
                reason = f"patch phase has read target evidence; request verifier-gated LLM patch proposal without fallback patch; {patch_reason}"
            elif patch_kwargs:
                function_name = "apply_patch"
                kwargs = patch_kwargs
                bonus = 0.35
                reason = f"patch phase has read target evidence; {patch_reason}"
            else:
                alternate_traceback_file = _deeper_traceback_source_file(
                    context,
                    unread_only=True,
                    extra_read_paths=_episode_file_read_paths(episode_trace),
                )
                if alternate_traceback_file:
                    function_name = "file_read"
                    kwargs = {"path": alternate_traceback_file, "start_line": 1, "end_line": 240}
                    bonus = 0.25
                    reason = "primary traceback file yielded no bounded patch; inspect deeper traceback implementation"
                elif target_confidence >= 0.55 and "propose_patch" in available:
                    function_name = "propose_patch"
                    kwargs = {"target_file": target_file, "max_changed_lines": 20}
                    bonus = 0.34
                    reason = "target binding is confident but heuristic patch inference has no bounded diff; request verifier-gated patch proposal"
    elif variant != "no_posterior" and leading_posterior >= 0.75 and target_file:
        bonus = 0.35
        if not _file_has_been_read(context, target_file):
            function_name = "file_read"
            kwargs = {"path": target_file, "start_line": 1, "end_line": 240}
            reason = "leading posterior is high and target_file has not been read"
        else:
            direct_test = (
                _direct_test_needed_before_patch(context, target_file)
                if variant != "no_discriminating_experiment"
                else ""
            )
            downstream_direct_test = (
                _downstream_direct_test_needed_before_patch(context, target_file)
                if variant != "no_discriminating_experiment"
                else ""
            )
            if direct_test and "run_test" in available:
                function_name = "run_test"
                kwargs = {"target": direct_test, "timeout_seconds": 30}
                bonus = 0.32
                reason = "leading posterior is high; run matching direct unit test before patching"
            elif downstream_direct_test and "run_test" in available:
                function_name = "run_test"
                kwargs = {"target": downstream_direct_test, "timeout_seconds": 30}
                bonus = 0.33
                reason = "leading target wraps a downstream implementation with a direct test; run it before patching"
            else:
                patch_kwargs, patch_reason, _ = choose_apply_patch_kwargs(context)
                alternate_traceback_file = _deeper_traceback_source_file(
                    context,
                    unread_only=True,
                    extra_read_paths=_episode_file_read_paths(episode_trace),
                )
                if patch_kwargs and prefer_llm_patch_proposals and "propose_patch" in available:
                    function_name = "propose_patch"
                    kwargs = {"target_file": target_file, "max_changed_lines": 20}
                    reason = f"leading posterior is high and target_file is read; request verifier-gated LLM patch proposal without fallback patch; {patch_reason}"
                elif patch_kwargs:
                    function_name = "apply_patch"
                    kwargs = patch_kwargs
                    reason = f"leading posterior is high and target_file is read; {patch_reason}"
                elif alternate_traceback_file:
                    function_name = "file_read"
                    kwargs = {"path": alternate_traceback_file, "start_line": 1, "end_line": 240}
                    reason = "primary traceback file was inspected; read deeper traceback implementation before patching"
                elif target_confidence >= 0.55 and "propose_patch" in available:
                    function_name = "propose_patch"
                    kwargs = {"target_file": target_file, "max_changed_lines": 20}
                    bonus = 0.34
                    reason = "leading target is read but no heuristic patch applies; request verifier-gated patch proposal"
    elif variant != "no_discriminating_experiment" and phase in {"inspect", "test"} and "run_test" in available:
        tests = _test_files_from_tree(context)
        tests = [test for test in tests if not _test_has_been_run(context, test)]
        if tests:
            function_name = "run_test"
            kwargs = {"target": tests[0], "timeout_seconds": 30}
            reason = "investigation phase needs a concrete failing test target"
    elif phase == "localize":
        if target_file and "file_read" in available and not _file_has_been_read(context, target_file):
            function_name = "file_read"
            kwargs = {"path": target_file, "start_line": 1, "end_line": 240}
            bonus = 0.28 if target_confidence >= 0.55 else 0.0
            reason = "failure evidence localized a target_file that has not been read"
        elif "repo_grep" in available:
            query, query_source = choose_repo_grep_query(context)
            if query:
                function_name = "repo_grep"
                kwargs = {"root": ".", "query": query, "globs": ["*.py"], "max_matches": 50}
                reason = f"failure evidence has no source target_file yet; grep query inferred from {query_source}"

    if not function_name:
        return None
    if available and function_name not in available:
        return None
    meta = {
        "posterior_action_bonus": bonus,
        "posterior_action_reason": reason,
        "leading_hypothesis_id": leading_id,
        "leading_posterior": leading_posterior,
        "target_file": target_file,
        "closed_loop_probe_variant": variant,
        "verify_after_patch_bonus": bonus if function_name == "run_test" and phase == "verify" else 0.0,
        "verification_pending": bool(phase == "verify"),
        "risk": 0.08 if function_name in {"file_read", "run_test"} else 0.18,
        "opportunity_estimate": 0.98 if bonus else 0.9,
        "final_score": 1.2 if bonus else 0.95,
    }
    return _make_call_action(function_name, kwargs, meta)


def extract_grounding_target_file(context: Mapping[str, Any]) -> str:
    return target_file_from_failure(context) or _top_hypothesis_target(context) or _last_grep_file(context)
