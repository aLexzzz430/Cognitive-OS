from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


PRODUCT_CLI_VERSION = "conos.product_cli/v1"
RUNTIME_COMMANDS = {
    "install-service",
    "uninstall-service",
    "start",
    "stop",
    "status",
    "logs",
    "approvals",
    "approve",
    "pause",
    "resume",
    "soak",
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
    run_parser.add_argument("target", choices=("arc-agi3", "local-machine", "webarena"))
    run_parser.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected runner.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Build the evaluation metrics panel from audit JSON/JSONL files.",
    )
    eval_parser.add_argument("eval_args", nargs=argparse.REMAINDER)

    ui_parser = subparsers.add_parser(
        "ui",
        help="Render or serve the local product dashboard.",
    )
    ui_parser.add_argument("ui_args", nargs=argparse.REMAINDER)

    app_parser = subparsers.add_parser(
        "app",
        help="Launch the local desktop app shell.",
    )
    app_parser.add_argument("app_args", nargs=argparse.REMAINDER)

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

    supervisor_parser = subparsers.add_parser(
        "supervisor",
        help="Manage resumable long-running Cognitive OS runs.",
    )
    supervisor_parser.add_argument("supervisor_args", nargs=argparse.REMAINDER)

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Render the current evaluation dashboard from audit outputs.",
    )
    dashboard_parser.add_argument("dashboard_args", nargs=argparse.REMAINDER)

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
    parser.add_argument("target", choices=("arc-agi3", "local-machine", "webarena"))
    return parser


def _run_target(target: str, target_args: Sequence[str]) -> int:
    if target == "arc-agi3":
        from integrations.arc_agi3.runner import main as arc_agi3_main

        return int(arc_agi3_main(list(target_args)))
    if target == "local-machine":
        from integrations.local_machine.runner import main as local_machine_main

        return int(local_machine_main(list(target_args)))
    if target == "webarena":
        from integrations.webarena.runner import main as webarena_main

        return int(webarena_main(list(target_args)))
    raise ValueError(f"Unsupported run target: {target}")


def _dispatch_run(args: Sequence[str]) -> int:
    run_args = list(args)
    if not run_args or run_args[0] in {"-h", "--help"}:
        _build_run_parser().parse_args(run_args)
        return 0
    target = str(run_args[0] or "")
    if target not in {"arc-agi3", "local-machine", "webarena"}:
        _build_run_parser().error(f"invalid choice: {target!r} (choose from 'arc-agi3', 'local-machine', 'webarena')")
    return _run_target(target, run_args[1:])


def _eval_panel(args: Sequence[str]) -> int:
    from scripts.eval_metrics_panel import main as eval_metrics_main

    return int(eval_metrics_main(list(args)))


def _dashboard(args: Sequence[str]) -> int:
    dashboard_args = list(args)
    if "--format" not in dashboard_args:
        dashboard_args.extend(["--format", "text"])
    return _eval_panel(dashboard_args)


def _ui(args: Sequence[str]) -> int:
    from core.evaluation.dashboard_app import main as dashboard_ui_main

    return int(dashboard_ui_main(list(args)))


def _app(args: Sequence[str]) -> int:
    from core.app.desktop_client import main as desktop_app_main

    return int(desktop_app_main(list(args)))


def _auth(args: Sequence[str]) -> int:
    auth_args = list(args)
    provider = str(auth_args[0] if auth_args else "openai")
    if provider != "openai":
        print(f"Unsupported auth provider: {provider}", file=sys.stderr)
        return 2
    from core.auth.openai_oauth import main as openai_oauth_main

    return int(openai_oauth_main(auth_args[1:]))


def _mirror(args: Sequence[str]) -> int:
    from modules.local_mirror.mirror import main as mirror_main

    return int(mirror_main(list(args)))


def _supervisor(args: Sequence[str]) -> int:
    from core.runtime.supervisor_cli import main as supervisor_main

    return int(supervisor_main(list(args)))


def _llm(args: Sequence[str]) -> int:
    from modules.llm.cli import main as llm_main

    return int(llm_main(list(args)))


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
            "eval",
            "ui",
            "app",
            "auth",
            "mirror",
            "llm",
            "supervisor",
            "dashboard",
            "preflight",
            "layout",
            "version",
            *sorted(RUNTIME_COMMANDS),
        ],
        "run_targets": ["arc-agi3", "local-machine", "webarena"],
        "auth_providers": ["openai"],
        "llm_providers": ["ollama"],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else list(sys.argv[1:])
    if raw_args:
        command = str(raw_args[0] or "")
        if command == "run":
            return _dispatch_run(raw_args[1:])
        if command == "eval":
            return _eval_panel(raw_args[1:])
        if command == "ui":
            return _ui(raw_args[1:])
        if command == "app":
            return _app(raw_args[1:])
        if command == "auth":
            return _auth(raw_args[1:])
        if command == "mirror":
            return _mirror(raw_args[1:])
        if command == "llm":
            return _llm(raw_args[1:])
        if command == "supervisor":
            return _supervisor(raw_args[1:])
        if command == "dashboard":
            return _dashboard(raw_args[1:])
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
    if command == "eval":
        return _eval_panel(list(args.eval_args or []))
    if command == "ui":
        return _ui(list(args.ui_args or []))
    if command == "app":
        return _app(list(args.app_args or []))
    if command == "auth":
        return _auth(list(args.auth_args or []))
    if command == "mirror":
        return _mirror(list(args.mirror_args or []))
    if command == "llm":
        return _llm(list(args.llm_args or []))
    if command == "supervisor":
        return _supervisor(list(args.supervisor_args or []))
    if command == "dashboard":
        return _dashboard(list(args.dashboard_args or []))
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
