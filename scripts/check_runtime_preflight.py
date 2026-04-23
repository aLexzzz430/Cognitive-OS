from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 10)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True

    @property
    def status(self) -> str:
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "WARN"


def _check_python_version() -> CheckResult:
    current = sys.version_info[:3]
    ok = current >= MIN_PYTHON
    return CheckResult(
        name="python_version",
        ok=ok,
        detail=(
            f"current={current[0]}.{current[1]}.{current[2]}, "
            f"required>={MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        ),
    )


def _check_core_import() -> CheckResult:
    try:
        from core.main_loop import CoreMainLoop  # noqa: F401
    except Exception as exc:  # pragma: no cover - diagnostics path
        return CheckResult(
            name="core_import",
            ok=False,
            detail=f"from core.main_loop import CoreMainLoop failed: {exc!r}",
        )
    return CheckResult(
        name="core_import",
        ok=True,
        detail="CoreMainLoop imports successfully",
    )


def _check_repo_layout() -> CheckResult:
    try:
        from core.adapter_registry import find_adapter_registry_violations
        from core.conos_repository_layout import find_forbidden_public_core_imports
    except Exception as exc:  # pragma: no cover - diagnostics path
        return CheckResult(
            name="repo_layout",
            ok=False,
            detail=f"layout checker imports failed: {exc!r}",
        )

    forbidden = find_forbidden_public_core_imports(REPO_ROOT)
    adapter = find_adapter_registry_violations()
    problems = []
    if forbidden:
        problems.append(f"forbidden_public_core_imports={len(forbidden)}")
    if adapter:
        problems.append(f"adapter_registry_violations={len(adapter)}")
    if problems:
        return CheckResult(
            name="repo_layout",
            ok=False,
            detail=", ".join(problems),
        )
    return CheckResult(
        name="repo_layout",
        ok=True,
        detail="public-core imports and adapter registry boundaries are clean",
    )


def _check_entrypoint_help(script_name: str) -> CheckResult:
    path = REPO_ROOT / "scripts" / script_name
    if not path.exists():
        return CheckResult(
            name=f"entrypoint:{script_name}",
            ok=False,
            detail=f"missing {path.relative_to(REPO_ROOT)}",
        )
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return CheckResult(
            name=f"entrypoint:{script_name}",
            ok=False,
            detail=f"cannot read script: {exc!r}",
        )
    if "main" not in source:
        return CheckResult(
            name=f"entrypoint:{script_name}",
            ok=False,
            detail="script does not expose a visible main bridge",
        )
    return CheckResult(
        name=f"entrypoint:{script_name}",
        ok=True,
        detail="script is present and bridges to a runner main",
    )


def _check_pytest(strict_dev: bool) -> CheckResult:
    ok = importlib.util.find_spec("pytest") is not None
    return CheckResult(
        name="dev_dependency:pytest",
        ok=ok,
        detail="pytest is installed" if ok else "pytest is missing; install requirements-dev.txt before running tests",
        required=bool(strict_dev),
    )


def _check_public_smoke_test_present(strict_dev: bool) -> CheckResult:
    path = REPO_ROOT / "tests" / "test_public_repo_smoke.py"
    return CheckResult(
        name="public_smoke_test",
        ok=path.exists(),
        detail=(
            "tests/test_public_repo_smoke.py is present"
            if path.exists()
            else "tests/test_public_repo_smoke.py is missing"
        ),
        required=bool(strict_dev),
    )


def _run_checks(strict_dev: bool) -> List[CheckResult]:
    checks: List[Callable[[], CheckResult]] = [
        _check_python_version,
        _check_core_import,
        _check_repo_layout,
        lambda: _check_entrypoint_help("run_arc_agi3.py"),
        lambda: _check_entrypoint_help("run_webarena.py"),
        lambda: _check_pytest(strict_dev),
        lambda: _check_public_smoke_test_present(strict_dev),
    ]
    return [check() for check in checks]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether this Cognitive OS source snapshot is runnable in the current environment."
    )
    parser.add_argument(
        "--strict-dev",
        action="store_true",
        help="Treat development-only test dependencies and public smoke tests as required.",
    )
    args = parser.parse_args()

    print(f"Repository root: {REPO_ROOT}")
    results = _run_checks(strict_dev=bool(args.strict_dev))
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    failed = [result for result in results if result.required and not result.ok]
    if failed:
        print(f"Preflight failed: {len(failed)} required check(s) did not pass.")
        return 1
    print("Preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
