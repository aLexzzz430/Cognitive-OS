from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


PRODUCT_CLI_VERSION = "conos.product_cli/v1"
RUNTIME_COMMANDS = {
    "setup",
    "install-service",
    "uninstall-service",
    "start",
    "stop",
    "status",
    "validate-install",
    "logs",
    "approvals",
    "approve",
    "pause",
    "resume",
    "soak",
    "doctor",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos",
        description="Unified product entrypoint for Cognitive OS.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run Cognitive OS against a supported environment adapter.",
    )
    run_parser.add_argument("target", choices=("local-machine",))
    run_parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected runner.",
    )

    auth_parser = subparsers.add_parser(
        "auth",
        help="Manage product authentication providers.",
    )
    auth_parser.add_argument("auth_args", nargs=argparse.REMAINDER)

    mirror_parser = subparsers.add_parser(
        "mirror",
        help="Manage an empty-first local mirror workspace.",
    )
    mirror_parser.add_argument("mirror_args", nargs=argparse.REMAINDER)

    llm_parser = subparsers.add_parser(
        "llm",
        help="Check and use local-first LLM providers.",
    )
    llm_parser.add_argument("llm_args", nargs=argparse.REMAINDER)

    vm_parser = subparsers.add_parser(
        "vm",
        help="Manage the built-in Con OS managed VM provider.",
    )
    vm_parser.add_argument("vm_args", nargs=argparse.REMAINDER)

    discover_parser = subparsers.add_parser(
        "discover-tasks",
        help="Discover, score, and queue autonomous task candidates.",
    )
    discover_parser.add_argument("discover_args", nargs=argparse.REMAINDER)

    supervisor_parser = subparsers.add_parser(
        "supervisor",
        help="Manage resumable long-running Cognitive OS runs.",
    )
    supervisor_parser.add_argument("supervisor_args", nargs=argparse.REMAINDER)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Check local runtime readiness.",
    )
    preflight_parser.add_argument(
        "--strict-dev",
        action="store_true",
        help="Treat test dependencies and smoke tests as required.",
    )

    subparsers.add_parser(
        "layout",
        help="Check repository layer boundaries.",
    )
    subparsers.add_parser(
        "version",
        help="Print product entrypoint version metadata.",
    )
    for runtime_command in sorted(RUNTIME_COMMANDS):
        runtime_parser = subparsers.add_parser(
            runtime_command,
            help=f"Runtime service command: {runtime_command}.",
        )
        runtime_parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    return parser


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos run",
        description="Run Cognitive OS against a supported environment adapter.",
    )
    parser.add_argument("target", choices=("local-machine",))
    return parser


def _run_target(target: str, target_args: Sequence[str]) -> int:
    if target == "local-machine":
        from integrations.local_machine.runner import main as local_machine_main

        return int(local_machine_main(list(target_args)))
    raise ValueError(f"Unsupported run target: {target}")


def _dispatch_run(args: Sequence[str]) -> int:
    run_args = list(args)
    if not run_args or run_args[0] in {"-h", "--help"}:
        _build_run_parser().parse_args(run_args)
        return 0
    target = str(run_args[0] or "")
    if target != "local-machine":
        _build_run_parser().error(f"invalid choice: {target!r} (choose from 'local-machine')")
    return _run_target(target, run_args[1:])


def _auth(args: Sequence[str]) -> int:
    auth_args = list(args)
    provider = str(auth_args[0] if auth_args else "openai")
    if provider == "openai":
        from core.auth.openai_oauth import main as openai_oauth_main

        return int(openai_oauth_main(auth_args[1:]))
    if provider in {"codex", "codex-cli"}:
        from core.auth.codex_cli_oauth import main as codex_oauth_main

        return int(codex_oauth_main(auth_args[1:]))
    if provider != "openai":
        print(f"Unsupported auth provider: {provider}", file=sys.stderr)
        return 2


def _mirror(args: Sequence[str]) -> int:
    from modules.local_mirror.mirror import main as mirror_main

    return int(mirror_main(list(args)))


def _supervisor(args: Sequence[str]) -> int:
    from core.runtime.supervisor_cli import main as supervisor_main

    return int(supervisor_main(list(args)))


def _llm(args: Sequence[str]) -> int:
    from modules.llm.cli import main as llm_main

    return int(llm_main(list(args)))


def _vm(args: Sequence[str]) -> int:
    from modules.local_mirror.managed_vm import main as managed_vm_main

    return int(managed_vm_main(list(args)))


def _discover_tasks(args: Sequence[str]) -> int:
    from core.task_discovery.cli import main as task_discovery_main

    return int(task_discovery_main(list(args)))


def _runtime(args: Sequence[str]) -> int:
    from core.runtime.runtime_service import main as runtime_main

    return int(runtime_main(list(args)))


def _preflight(*, strict_dev: bool = False) -> int:
    from scripts.check_runtime_preflight import main as preflight_main

    args = ["--strict-dev"] if strict_dev else []
    return int(preflight_main(args))


def _layout() -> int:
    from scripts.check_conos_repo_layout import main as layout_main

    return int(layout_main())


def _version() -> int:
    payload = {
        "product": "Cognitive OS",
        "entrypoint": "conos",
        "schema_version": PRODUCT_CLI_VERSION,
        "commands": [
            "run",
            "auth",
            "mirror",
            "llm",
            "vm",
            "discover-tasks",
            "supervisor",
            "preflight",
            "layout",
            "version",
            *sorted(RUNTIME_COMMANDS),
        ],
        "run_targets": ["local-machine"],
        "auth_providers": ["openai", "codex"],
        "llm_providers": ["ollama", "openai", "codex-cli"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if raw_args:
        command = str(raw_args[0] or "")
        if command == "run":
            return _dispatch_run(raw_args[1:])
        if command == "auth":
            return _auth(raw_args[1:])
        if command == "mirror":
            return _mirror(raw_args[1:])
        if command == "llm":
            return _llm(raw_args[1:])
        if command == "vm":
            return _vm(raw_args[1:])
        if command == "discover-tasks":
            return _discover_tasks(raw_args[1:])
        if command == "supervisor":
            return _supervisor(raw_args[1:])
        if command in RUNTIME_COMMANDS:
            return _runtime(raw_args)

    parser = _build_parser()
    args = parser.parse_args(raw_args)
    command = str(args.command or "")
    if not command:
        parser.print_help()
        return 0
    if command == "run":
        return _run_target(str(args.target), list(args.target_args or []))
    if command == "auth":
        return _auth(list(args.auth_args or []))
    if command == "mirror":
        return _mirror(list(args.mirror_args or []))
    if command == "llm":
        return _llm(list(args.llm_args or []))
    if command == "vm":
        return _vm(list(args.vm_args or []))
    if command == "discover-tasks":
        return _discover_tasks(list(args.discover_args or []))
    if command == "supervisor":
        return _supervisor(list(args.supervisor_args or []))
    if command in RUNTIME_COMMANDS:
        return _runtime([command, *list(getattr(args, "runtime_args", []) or [])])
    if command == "preflight":
        return _preflight(strict_dev=bool(args.strict_dev))
    if command == "layout":
        return _layout()
    if command == "version":
        return _version()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
