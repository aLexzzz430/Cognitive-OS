from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

LAYER_CONOS_CORE = "conos-core"
LAYER_CONOS_EVALS = "conos-evals"
LAYER_PRIVATE_COGNITIVE_CORE = "private-cognitive-core"
LAYER_ADAPTER = "adapter"
LAYER_RUNTIME = "runtime"
LAYER_UNCLASSIFIED = "unclassified"

CONOS_CORE_PATH_PREFIXES = (
    "core/",
    "decision/",
    "evolution/",
    "memory/",
    "modules/",
    "planner/",
    "self_model/",
    "state/",
    "trace/",
)

CONOS_EVAL_PATH_PREFIXES = (
    "conos-evals/",
    "conos_evals/",
    "eval/",
    "scripts/arc_agi2_compare_modes.py",
    "scripts/arc_agi2_curriculum_eval.py",
    "scripts/arc_agi2_eval.py",
    "scripts/capability_smoke_eval.py",
    "scripts/cognitive_curriculum_eval.py",
    "scripts/object_relation_blind_eval.py",
    "scripts/regression_runtime_report.py",
    "scripts/run_arc_agi3_all_games_llm_batch.py",
)

PRIVATE_COGNITIVE_CORE_PATH_PREFIXES = (
    "private-cognitive-core/",
    "private_cognitive_core/",
    "core/orchestration/structured_answer.py",
    "modules/hypothesis/mechanism_posterior_updater.py",
    "scripts/latent_",
    "scripts/multi_domain_",
)

ADAPTER_PATH_PREFIXES = (
    "integrations/arc_agi3/",
    "integrations/local_machine/",
    "integrations/survival_world/",
    "integrations/webarena/",
)

RUNTIME_PATH_PREFIXES = (
    "runtime/",
    "audit/",
    "reports/",
)

BOUNDARY_IMPORT_SCAN_ROOTS = (
    "core",
    "decision",
    "evolution",
    "memory",
    "modules",
    "planner",
    "self_model",
    "state",
    "trace",
)

BOUNDARY_IMPORT_SCAN_LAYERS = (
    LAYER_CONOS_CORE,
    LAYER_PRIVATE_COGNITIVE_CORE,
)

FORBIDDEN_PUBLIC_CORE_IMPORT_PREFIXES = (
    "integrations",
    "eval",
    "scripts",
)


@dataclass(frozen=True)
class LayerSummary:
    layer_name: str
    path_prefixes: List[str]


def _normalize_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    text = str(path).strip().replace("\\", "/")
    if not text:
        return ""
    resolved_root = Path(repo_root).resolve()
    if text.startswith(str(resolved_root).replace("\\", "/")):
        text = str(Path(text).resolve().relative_to(resolved_root)).replace("\\", "/")
    return text.lstrip("./")


def _matches_any_prefix(path: str, prefixes: Iterable[str]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def classify_repo_path(path: str | Path, repo_root: Path = REPO_ROOT) -> str:
    normalized = _normalize_repo_path(path, repo_root=repo_root)
    if not normalized:
        return LAYER_UNCLASSIFIED
    if _matches_any_prefix(normalized, RUNTIME_PATH_PREFIXES):
        return LAYER_RUNTIME
    if _matches_any_prefix(normalized, ADAPTER_PATH_PREFIXES):
        return LAYER_ADAPTER
    if _matches_any_prefix(normalized, PRIVATE_COGNITIVE_CORE_PATH_PREFIXES):
        return LAYER_PRIVATE_COGNITIVE_CORE
    if _matches_any_prefix(normalized, CONOS_EVAL_PATH_PREFIXES):
        return LAYER_CONOS_EVALS
    if _matches_any_prefix(normalized, CONOS_CORE_PATH_PREFIXES):
        return LAYER_CONOS_CORE
    return LAYER_UNCLASSIFIED


def describe_repo_layers() -> List[LayerSummary]:
    return [
        LayerSummary(LAYER_CONOS_CORE, list(CONOS_CORE_PATH_PREFIXES)),
        LayerSummary(LAYER_ADAPTER, list(ADAPTER_PATH_PREFIXES)),
        LayerSummary(LAYER_CONOS_EVALS, list(CONOS_EVAL_PATH_PREFIXES)),
        LayerSummary(LAYER_PRIVATE_COGNITIVE_CORE, list(PRIVATE_COGNITIVE_CORE_PATH_PREFIXES)),
        LayerSummary(LAYER_RUNTIME, list(RUNTIME_PATH_PREFIXES)),
    ]


def load_optional_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def boundary_checked_python_files(repo_root: Path = REPO_ROOT) -> List[Path]:
    files: List[Path] = []
    for root_name in BOUNDARY_IMPORT_SCAN_ROOTS:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative = path.relative_to(repo_root)
            if classify_repo_path(relative, repo_root=repo_root) not in BOUNDARY_IMPORT_SCAN_LAYERS:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            if "__pycache__" in relative.parts:
                continue
            files.append(path)
    return sorted(files)


def public_core_python_files(repo_root: Path = REPO_ROOT) -> List[Path]:
    return [
        path
        for path in boundary_checked_python_files(repo_root)
        if classify_repo_path(path.relative_to(repo_root), repo_root=repo_root) == LAYER_CONOS_CORE
    ]


def find_forbidden_public_core_imports(repo_root: Path = REPO_ROOT) -> List[dict]:
    findings: List[dict] = []
    for path in boundary_checked_python_files(repo_root):
        relative = path.relative_to(repo_root)
        importer_layer = classify_repo_path(relative, repo_root=repo_root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            findings.append(
                {
                    "path": str(relative).replace("\\", "/"),
                    "layer": importer_layer,
                    "line": 1,
                    "import": "<syntax-error>",
                }
            )
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = str(alias.name or "").split(".", 1)[0]
                    if root_name in FORBIDDEN_PUBLIC_CORE_IMPORT_PREFIXES:
                        findings.append(
                            {
                                "path": str(relative).replace("\\", "/"),
                                "layer": importer_layer,
                                "line": int(getattr(node, "lineno", 1) or 1),
                                "import": str(alias.name or ""),
                            }
                        )
            elif isinstance(node, ast.ImportFrom):
                module_name = str(node.module or "")
                root_name = module_name.split(".", 1)[0]
                if root_name in FORBIDDEN_PUBLIC_CORE_IMPORT_PREFIXES:
                    findings.append(
                        {
                            "path": str(relative).replace("\\", "/"),
                            "layer": importer_layer,
                            "line": int(getattr(node, "lineno", 1) or 1),
                            "import": module_name,
                        }
                    )
    return findings
