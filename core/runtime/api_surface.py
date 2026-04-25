from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


API_SURFACE_SCHEMA_VERSION = "conos.api_surface/v1"
REPAIR_TARGET_PLAN_SCHEMA_VERSION = "conos.repair_target_plan/v1"

_IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}

_OBJECT_ATTRIBUTE_ERROR_RE = re.compile(
    r"AttributeError:\s*'(?P<class_name>[^']+)'\s+object\s+has\s+no\s+attribute\s+'(?P<attribute>[^']+)'"
)
_MODULE_ATTRIBUTE_ERROR_RE = re.compile(
    r"AttributeError:\s*module\s+'(?P<module>[^']+)'\s+has\s+no\s+attribute\s+'(?P<attribute>[^']+)'"
)
_PYTHON_FILE_REF_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py|[A-Za-z0-9_.-]+\.py)"
)


def extract_attribute_errors(text: str) -> list[dict[str, str]]:
    """Extract Python AttributeError facts from validation output."""
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _OBJECT_ATTRIBUTE_ERROR_RE.finditer(str(text or "")):
        item = {
            "kind": "object",
            "class_name": match.group("class_name"),
            "attribute": match.group("attribute"),
        }
        key = (item["kind"], item["class_name"], item["attribute"])
        if key not in seen:
            seen.add(key)
            errors.append(item)
    for match in _MODULE_ATTRIBUTE_ERROR_RE.finditer(str(text or "")):
        item = {
            "kind": "module",
            "module": match.group("module"),
            "attribute": match.group("attribute"),
        }
        key = (item["kind"], item["module"], item["attribute"])
        if key not in seen:
            seen.add(key)
            errors.append(item)
    return errors


def extract_python_api_surface(
    project_root: str | Path,
    *,
    python_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Build a compact public Python API manifest for a generated project."""
    root = Path(project_root).expanduser().resolve()
    files: list[dict[str, Any]] = []
    class_index: dict[str, list[dict[str, Any]]] = {}
    function_index: dict[str, list[dict[str, Any]]] = {}

    for path in _iter_python_paths(root, python_paths):
        file_surface = _extract_file_surface(root, path)
        files.append(file_surface)
        for function in file_surface.get("functions", []):
            name = str(function.get("name", ""))
            function_index.setdefault(name, []).append(
                {
                    "path": file_surface["path"],
                    "module": file_surface["module"],
                    "lineno": function.get("lineno"),
                }
            )
        for class_surface in file_surface.get("classes", []):
            name = str(class_surface.get("name", ""))
            class_index.setdefault(name, []).append(
                {
                    "path": file_surface["path"],
                    "module": file_surface["module"],
                    "lineno": class_surface.get("lineno"),
                    "methods": list(class_surface.get("methods", [])),
                }
            )

    return {
        "schema_version": API_SURFACE_SCHEMA_VERSION,
        "project_root": str(root),
        "files": files,
        "class_index": class_index,
        "function_index": function_index,
    }


def plan_repair_targets_for_validation(
    project_root: str | Path,
    validation: Mapping[str, Any] | str,
    *,
    file_specs: Sequence[Mapping[str, Any]] | None = None,
    python_paths: Iterable[str | Path] | None = None,
    max_targets: int = 8,
) -> dict[str, Any]:
    """
    Expand a failed validation trace into source-plus-test repair targets.

    The important behavior is conservative: if tests call a missing method,
    include the class source file and the tests that likely encode the bad
    contract, instead of letting repair mutate only the last file mentioned.
    """
    root = Path(project_root).expanduser().resolve()
    validation_text = _validation_text(validation)
    surface = extract_python_api_surface(root, python_paths=python_paths)
    file_refs = _extract_python_file_refs(validation_text)
    file_spec_paths = _file_spec_paths(file_specs or ())
    test_specs = [path for path in file_spec_paths if _looks_like_test_path(path)]
    source_specs = [path for path in file_spec_paths if not _looks_like_test_path(path)]
    targets: dict[str, set[str]] = {}
    diagnostics: list[dict[str, Any]] = []

    for path in file_refs:
        if _is_known_relative_file(root, path):
            _add_target(targets, path, "validation_trace_reference")

    for error in extract_attribute_errors(validation_text):
        if error.get("kind") == "object":
            class_name = str(error.get("class_name", ""))
            attribute = str(error.get("attribute", ""))
            class_rows = list(surface.get("class_index", {}).get(class_name, []))
            class_paths = [str(row.get("path", "")) for row in class_rows if str(row.get("path", ""))]
            available_methods = sorted(
                {
                    str(method.get("name", ""))
                    for row in class_rows
                    for method in list(row.get("methods", []) or [])
                    if str(method.get("name", ""))
                }
            )
            for path in class_paths:
                _add_target(targets, path, "missing_attribute_source")
            for path in test_specs:
                _add_target(targets, path, "missing_attribute_test_contract")
            if not class_paths:
                for path in source_specs:
                    _add_target(targets, path, "missing_attribute_candidate_source")
            diagnostics.append(
                {
                    "kind": "missing_object_attribute",
                    "class_name": class_name,
                    "missing_attribute": attribute,
                    "class_source_paths": class_paths,
                    "available_methods": available_methods,
                }
            )
        elif error.get("kind") == "module":
            module_name = str(error.get("module", ""))
            module_paths = _module_paths(surface, module_name)
            for path in module_paths:
                _add_target(targets, path, "missing_module_attribute_source")
            for path in test_specs:
                _add_target(targets, path, "missing_module_attribute_test_contract")
            diagnostics.append(
                {
                    "kind": "missing_module_attribute",
                    "module": module_name,
                    "missing_attribute": str(error.get("attribute", "")),
                    "module_paths": module_paths,
                }
            )

    if not targets and _validation_failed(validation):
        for path in file_refs:
            if _is_known_relative_file(root, path):
                _add_target(targets, path, "failed_validation_reference")
        for path in file_spec_paths[: max(1, max_targets)]:
            _add_target(targets, path, "failed_validation_file_spec")

    ordered_targets = [
        {"path": path, "reasons": sorted(reasons)}
        for path, reasons in sorted(targets.items(), key=lambda item: _target_sort_key(item[0]))
    ][: max(1, int(max_targets or 8))]
    return {
        "schema_version": REPAIR_TARGET_PLAN_SCHEMA_VERSION,
        "project_root": str(root),
        "targets": ordered_targets,
        "diagnostics": diagnostics,
        "attribute_errors": extract_attribute_errors(validation_text),
        "file_references": file_refs,
        "api_surface": {
            "schema_version": surface["schema_version"],
            "class_index": surface["class_index"],
            "function_index": surface["function_index"],
        },
    }


def _iter_python_paths(root: Path, python_paths: Iterable[str | Path] | None) -> list[Path]:
    if python_paths is None:
        candidates = list(root.rglob("*.py"))
    else:
        candidates = []
        for raw in python_paths:
            path = Path(raw)
            candidates.append(path if path.is_absolute() else root / path)
    resolved: list[Path] = []
    for path in candidates:
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        resolved.append(path.resolve())
    return sorted(set(resolved), key=lambda item: item.as_posix())


def _extract_file_surface(root: Path, path: Path) -> dict[str, Any]:
    relative = _relative_posix(root, path)
    surface: dict[str, Any] = {
        "path": relative,
        "module": _module_name(relative),
        "functions": [],
        "classes": [],
    }
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        surface["syntax_error"] = str(exc)
        return surface

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(node.name):
            surface["functions"].append(_function_surface(node))
        elif isinstance(node, ast.ClassDef) and _is_public(node.name):
            methods = [
                _function_surface(child)
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(child.name)
            ]
            surface["classes"].append(
                {
                    "name": node.name,
                    "lineno": int(getattr(node, "lineno", 0) or 0),
                    "methods": methods,
                }
            )
    return surface


def _function_surface(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "name": node.name,
        "lineno": int(getattr(node, "lineno", 0) or 0),
        "async": isinstance(node, ast.AsyncFunctionDef),
        "args": [arg.arg for arg in list(node.args.args)],
    }


def _module_name(relative_path: str) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_public(name: str) -> bool:
    return bool(name) and not name.startswith("_")


def _relative_posix(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _validation_text(value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        return value
    chunks: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            chunks.append(item)
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, (int, float, bool)):
            chunks.append(str(item))

    visit(value)
    return "\n".join(chunks)


def _validation_failed(value: Mapping[str, Any] | str) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in ("failed", "error", "traceback", "attributeerror"))
    ok = value.get("ok")
    if ok is not None:
        return not bool(ok)
    return bool(extract_attribute_errors(_validation_text(value)))


def _extract_python_file_refs(text: str) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for match in _PYTHON_FILE_REF_RE.finditer(str(text or "")):
        path = _normalize_relative_path(match.group("path"))
        if not path or any(part in _IGNORED_PARTS for part in Path(path).parts):
            continue
        if path not in seen:
            seen.add(path)
            refs.append(path)
    return refs


def _file_spec_paths(file_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for spec in file_specs:
        raw = spec.get("relative_path") or spec.get("path") or spec.get("target_path") or spec.get("file")
        path = _normalize_relative_path(str(raw or ""))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _normalize_relative_path(path: str) -> str:
    text = str(path or "").strip().strip('"').strip("'")
    if not text:
        return ""
    text = text.replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    while "../" in text:
        text = text.replace("../", "")
    return text.lstrip("/")


def _looks_like_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def _is_known_relative_file(root: Path, path: str) -> bool:
    normalized = _normalize_relative_path(path)
    if not normalized:
        return False
    try:
        return (root / normalized).resolve().relative_to(root) is not None
    except ValueError:
        return False


def _module_paths(surface: Mapping[str, Any], module_name: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for file_surface in list(surface.get("files", []) or []):
        if not isinstance(file_surface, Mapping):
            continue
        if str(file_surface.get("module", "")) != module_name:
            continue
        path = str(file_surface.get("path", ""))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _add_target(targets: dict[str, set[str]], path: str, reason: str) -> None:
    normalized = _normalize_relative_path(path)
    if not normalized:
        return
    targets.setdefault(normalized, set()).add(str(reason or "candidate"))


def _target_sort_key(path: str) -> tuple[int, str]:
    if _looks_like_test_path(path):
        return (1, path)
    return (0, path)
