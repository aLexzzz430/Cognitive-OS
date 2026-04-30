from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


TARGET_BINDING_VERSION = "conos.local_machine.target_binding/v1"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value or []) if isinstance(value, list) else []


def _text(value: Any) -> str:
    return str(value or "").strip()


def _state(context: Mapping[str, Any]) -> dict[str, Any]:
    return _as_dict(context.get("investigation_state"))


def _source_root(context: Mapping[str, Any]) -> Path:
    return Path(_text(context.get("source_root")) or ".").resolve()


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


def _load_run_output(context: Mapping[str, Any], run_ref: str) -> dict[str, Any]:
    if not re.fullmatch(r"run_[A-Za-z0-9_.-]+", _text(run_ref)):
        return {}
    path = _run_output_root(context) / f"{run_ref}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def latest_failed_run_output(context: Mapping[str, Any]) -> dict[str, Any]:
    state = _state(context)
    for row in reversed(_as_list(state.get("validation_runs"))):
        item = _as_dict(row)
        if bool(item.get("success", False)):
            continue
        payload = _load_run_output(context, _text(item.get("run_ref")))
        if payload:
            return payload
    return {}


def latest_failure_text(context: Mapping[str, Any]) -> str:
    payload = latest_failed_run_output(context)
    return f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}".strip()


def _tree_entries(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = _as_list(_as_dict(_state(context).get("last_tree")).get("entries"))
    return [dict(row) for row in entries if isinstance(row, Mapping)]


def _is_test_path(path: str) -> bool:
    clean = _text(path).replace("\\", "/")
    if not clean:
        return False
    parts = set(Path(clean).parts)
    name = Path(clean).name
    return bool(parts.intersection({"tests", "testing"})) or name == "conftest.py"


def _is_pytest_named_source(context: Mapping[str, Any], path: str) -> bool:
    name = Path(path).name
    if not (name.startswith("test_") or name.endswith("_test.py")):
        return False
    try:
        text = (_source_root(context) / path).read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(re.search(r"^\s*(?:def\s+test_|class\s+Test)", text, flags=re.MULTILINE))


def _source_files(context: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        if path.endswith(".py") and not _is_test_path(path) and not _is_pytest_named_source(context, path):
            paths.append(path)
    if paths:
        return sorted(dict.fromkeys(paths))
    root = _source_root(context)
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if _is_test_path(relative) or _is_pytest_named_source(context, relative) or "__pycache__" in Path(relative).parts:
            continue
        found.append(relative)
    return sorted(dict.fromkeys(found))


def _test_files(context: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for row in _tree_entries(context):
        if row.get("kind") != "file":
            continue
        path = _text(row.get("path"))
        if path.endswith(".py") and (_is_test_path(path) or _is_pytest_named_source(context, path)):
            paths.append(path)
    return sorted(dict.fromkeys(paths))


def _latest_failed_test_target(context: Mapping[str, Any]) -> str:
    payload = latest_failed_run_output(context)
    command = [str(part) for part in _as_list(payload.get("command"))]
    if command:
        target = _text(command[-1])
        if target and target != ".":
            return target.split("::", 1)[0]
    text = f"{payload.get('stdout', '')}\n{payload.get('stderr', '')}"
    match = re.search(r"FAILED\s+([A-Za-z0-9_./-]+\.py(?:::[^\s]+)?)", text)
    return _text(match.group(1)).split("::", 1)[0] if match else ""


def _traceback_source_files(context: Mapping[str, Any]) -> list[str]:
    files: list[str] = []
    for match in re.finditer(r'(?:File "([^"]+?\.py)"|([A-Za-z0-9_./-]+\.py):\d+)', latest_failure_text(context)):
        raw = _text(match.group(1) or match.group(2))
        if not raw or "site-packages" in raw or raw.startswith("<"):
            continue
        path = Path(raw)
        if path.is_absolute():
            try:
                raw = path.resolve().relative_to(_source_root(context)).as_posix()
            except ValueError:
                continue
        raw = raw.replace("\\", "/").lstrip("./")
        if raw and not _is_test_path(raw) and not _is_pytest_named_source(context, raw) and _path_exists_in_source(context, raw):
            files.append(raw)
    return list(dict.fromkeys(files))


def _symbols_from_failure(text: str) -> list[str]:
    symbols: list[str] = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", text):
        token = match.group(1)
        if token in {"assert", "where", "PosixPath", "FileNotFoundError", "ValueError"}:
            continue
        symbols.append(token)
    for match in re.finditer(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]{2,})\b", text):
        symbols.append(match.group(1))
    return list(dict.fromkeys(symbols))[:12]


def _file_text(context: Mapping[str, Any], relative: str) -> str:
    try:
        return (_source_root(context) / relative).read_text(encoding="utf-8")
    except OSError:
        return ""


def _stem_tokens(path: str) -> set[str]:
    stem = Path(path).stem.lower()
    if stem.startswith("test_"):
        stem = stem[5:]
    tokens = {token for token in re.split(r"[_\W]+", stem) if len(token) >= 3}
    if stem:
        tokens.add(stem)
    for token in list(tokens):
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _path_tokens(path: str) -> set[str]:
    tokens = set(_stem_tokens(path))
    for part in Path(path).parts:
        lowered = str(part).lower()
        if lowered in {"core", "src", "lib", "app", "tests", "test"}:
            continue
        tokens.update(token for token in re.split(r"[_\W]+", lowered) if len(token) >= 3)
    return tokens


def _goal_tokens(context: Mapping[str, Any]) -> set[str]:
    text = _text(context.get("instruction") or context.get("goal")).lower()
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text))
    stopwords = {
        "the",
        "and",
        "this",
        "that",
        "with",
        "from",
        "into",
        "make",
        "made",
        "small",
        "source",
        "change",
        "changes",
        "file",
        "files",
        "task",
        "open",
        "ended",
        "smoke",
        "inspect",
        "identify",
        "improvement",
        "opportunity",
        "justified",
        "relevant",
        "verification",
        "report",
    }
    return {token for token in tokens if token not in stopwords}


def _definition_files(context: Mapping[str, Any], symbols: Sequence[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for source_file in _source_files(context):
        text = _file_text(context, source_file)
        if not text:
            continue
        for symbol in symbols:
            if re.search(rf"^\s*(?:def|class)\s+{re.escape(symbol)}\b", text, flags=re.MULTILINE):
                result.setdefault(symbol, []).append(source_file)
    return result


def _imported_downstream_files(context: Mapping[str, Any]) -> list[str]:
    state = _state(context)
    source_files = set(_source_files(context))
    downstream: list[str] = []
    for row in _as_list(state.get("read_files")):
        source_path = _text(_as_dict(row).get("path"))
        if not source_path or source_path.startswith("tests/"):
            continue
        text = _file_text(context, source_path)
        package_dir = Path(source_path).parent
        for match in re.finditer(r"^\s*from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\b", text, flags=re.MULTILINE):
            candidate = (package_dir / f"{match.group(1)}.py").as_posix()
            if candidate in source_files:
                downstream.append(candidate)
    return list(dict.fromkeys(downstream))


def _imported_symbol_downstream_files(context: Mapping[str, Any], symbols: Sequence[str]) -> list[tuple[str, str]]:
    state = _state(context)
    wanted = {str(symbol) for symbol in symbols if str(symbol)}
    if not wanted:
        return []
    source_files = set(_source_files(context))
    downstream: list[tuple[str, str]] = []
    for row in _as_list(state.get("read_files")):
        source_path = _text(_as_dict(row).get("path"))
        if not source_path or source_path.startswith("tests/"):
            continue
        text = _file_text(context, source_path)
        package_dir = Path(source_path).parent
        for match in re.finditer(r"^\s*from\s+\.([A-Za-z_][A-Za-z0-9_]*)\s+import\s+([^\n]+)", text, flags=re.MULTILINE):
            candidate = (package_dir / f"{match.group(1)}.py").as_posix()
            if candidate not in source_files:
                continue
            imported_symbols = [
                token.strip().split(" as ", 1)[0].strip()
                for token in match.group(2).split(",")
            ]
            for symbol in imported_symbols:
                if symbol in wanted:
                    downstream.append((candidate, symbol))
    return list(dict.fromkeys(downstream))


def _state_leak_signal(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("mutat", "alias", "defensive copy", "internal state", "state leak", "leak"))


def _looks_like_mutable_state_owner(text: str) -> bool:
    return bool(
        re.search(r"self\._[A-Za-z0-9_]+\s*=\s*(?:list|dict|set)\s*\(", text)
        and re.search(r"return\s+self\._[A-Za-z0-9_]+", text)
    )


def bind_target(context: Mapping[str, Any]) -> dict[str, Any]:
    failure = latest_failure_text(context)
    failed_test = _latest_failed_test_target(context)
    traceback_files = _traceback_source_files(context)
    symbols = _symbols_from_failure(failure)
    definitions = _definition_files(context, symbols)
    test_tokens = _stem_tokens(failed_test)
    downstream = _imported_downstream_files(context)
    symbol_downstream = _imported_symbol_downstream_files(context, symbols)
    last_search = _as_dict(_state(context).get("last_search"))
    search_paths = [
        _text(_as_dict(row).get("path"))
        for row in _as_list(last_search.get("matches") or last_search.get("results"))
        if _text(_as_dict(row).get("path"))
    ]
    search_paths = list(dict.fromkeys(search_paths))
    failure_present = bool(failure or failed_test or traceback_files or symbols)
    goal_tokens = _goal_tokens(context)

    candidates: dict[str, dict[str, Any]] = {}

    def add(path: str, score: float, reason: str) -> None:
        if not path or _is_test_path(path) or _is_pytest_named_source(context, path) or not path.endswith(".py"):
            return
        if not _path_exists_in_source(context, path):
            return
        row = candidates.setdefault(path, {"target_file": path, "score": 0.0, "reasons": []})
        row["score"] = min(1.0, float(row.get("score", 0.0) or 0.0) + score)
        if reason not in row["reasons"]:
            row["reasons"].append(reason)

    for index, path in enumerate(traceback_files):
        add(path, 0.22 if index == 0 else 0.32, "source file appears in traceback")
    for source_file in _source_files(context):
        file_tokens = _stem_tokens(source_file)
        overlap = test_tokens.intersection(file_tokens)
        if overlap:
            add(source_file, 0.42 + min(0.12, 0.03 * len(overlap)), "failed test filename matches source filename")
    for symbol, paths in definitions.items():
        for path in paths:
            add(path, 0.36, f"failure symbol is defined here: {symbol}")
    for path in search_paths:
        add(path, 0.12 if failure_present else 0.1, "latest repo search matched this file")
    for path in downstream:
        add(path, 0.2, "already-read file imports this downstream module")
    for path, symbol in symbol_downstream:
        add(path, 0.22, f"failure symbol is imported from this downstream module: {symbol}")
    if not failure_present and goal_tokens:
        for source_file in _source_files(context):
            overlap = goal_tokens.intersection(_path_tokens(source_file))
            if overlap:
                add(
                    source_file,
                    0.28 + min(0.16, 0.04 * len(overlap)),
                    "source path matches open-task goal tokens",
                )
    if _state_leak_signal(failure):
        for path in _source_files(context):
            if _looks_like_mutable_state_owner(_file_text(context, path)):
                add(path, 0.8, "failure suggests mutable state exposure and this file owns a returned mutable field")
    for row in _as_list(_state(context).get("read_files")):
        read_path = _text(_as_dict(row).get("path"))
        add(read_path, 0.04 if failure_present else 0.16, "file has already been read")
        if not failure_present and goal_tokens and goal_tokens.intersection(_path_tokens(read_path)):
            add(read_path, 0.12, "read file also matches open-task goal tokens")

    ordered = sorted(
        (
            {
                "target_file": str(row["target_file"]),
                "score": round(float(row["score"]), 6),
                "reasons": list(row.get("reasons", []) or []),
            }
            for row in candidates.values()
        ),
        key=lambda row: (-float(row["score"]), str(row["target_file"])),
    )
    top = ordered[0] if ordered else {}
    confidence = min(0.95, float(top.get("score", 0.0) or 0.0)) if top else 0.0
    if top and not failure_present:
        reasons = " ".join(str(item) for item in list(top.get("reasons", []) or [])).lower()
        if "file has already been read" not in reasons:
            confidence = min(confidence, 0.48)
        elif "goal tokens" not in reasons and "repo search" not in reasons:
            confidence = min(confidence, 0.5)
        else:
            confidence = min(confidence, 0.72)
    return {
        "schema_version": TARGET_BINDING_VERSION,
        "target_file_candidates": ordered[:10],
        "top_target_file": str(top.get("target_file") or ""),
        "target_confidence": round(confidence, 6),
        "binding_reasons": list(top.get("reasons", []) or []),
        "latest_failed_test_target": failed_test,
        "traceback_files": traceback_files,
        "failure_symbols": symbols,
    }
