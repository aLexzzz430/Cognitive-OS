from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.conos_repository_layout import (
    describe_repo_layers,
    find_forbidden_public_core_imports,
)
from core.adapter_registry import find_adapter_registry_violations


def main() -> int:
    print(f"Repository root: {REPO_ROOT}")
    print("Logical layers:")
    for summary in describe_repo_layers():
        joined = ", ".join(summary.path_prefixes)
        print(f"- {summary.layer_name}: {joined}")

    findings = find_forbidden_public_core_imports(REPO_ROOT)
    if findings:
        print("Forbidden public/private boundary imports detected:")
        for finding in findings:
            print(
                f"  - {finding['path']}:{finding['line']} [{finding.get('layer', 'unknown')}] imports {finding['import']}"
            )
        return 1

    adapter_findings = find_adapter_registry_violations()
    if adapter_findings:
        print("Adapter registry violations detected:")
        for finding in adapter_findings:
            print(
                f"  - {finding['adapter_key']} points to {finding['repo_path']} in layer {finding['layer']}"
            )
        return 1

    print("No forbidden public/private boundary imports detected.")
    print("Adapter registry points only to adapter-layer paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
