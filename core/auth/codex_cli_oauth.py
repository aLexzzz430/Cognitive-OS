from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Any, Callable, Mapping, Sequence


CODEX_CLI_AUTH_VERSION = "conos.codex_cli_auth/v1"


def _command(binary: str | None = None) -> str:
    return str(binary or os.getenv("CONOS_CODEX_COMMAND") or "codex").strip() or "codex"


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str))


def _tail(text: str, limit: int = 4000) -> str:
    payload = str(text or "")
    if len(payload) <= limit:
        return payload
    return payload[-limit:]


def codex_login_status(
    *,
    command: str | None = None,
    timeout_seconds: int = 20,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    binary = _command(command)
    cmd = [binary, "login", "status"]
    try:
        completed = runner(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except Exception as exc:
        return {
            "schema_version": CODEX_CLI_AUTH_VERSION,
            "provider": "codex-cli",
            "auth_type": "chatgpt_oauth_delegate",
            "connected": False,
            "command": cmd,
            "error": str(exc),
            "token_access": "delegated_to_codex_cli",
        }
    output = str(getattr(completed, "stdout", "") or "")
    returncode = int(getattr(completed, "returncode", 1) or 0)
    connected = returncode == 0 and "not logged in" not in output.lower()
    return {
        "schema_version": CODEX_CLI_AUTH_VERSION,
        "provider": "codex-cli",
        "auth_type": "chatgpt_oauth_delegate",
        "connected": connected,
        "command": cmd,
        "returncode": returncode,
        "status_output": _tail(output),
        "token_access": "delegated_to_codex_cli",
        "quota_scope": "chatgpt_codex_plan_or_api_org_via_codex_cli",
        "error": "" if connected else _tail(output),
    }


def run_codex_login(
    *,
    command: str | None = None,
    device_auth: bool = False,
    interactive: bool = True,
    timeout_seconds: int = 900,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    binary = _command(command)
    cmd = [binary, "login"]
    if device_auth:
        cmd.append("--device-auth")
    try:
        completed = runner(
            cmd,
            text=True,
            stdout=None if interactive else subprocess.PIPE,
            stderr=None if interactive else subprocess.STDOUT,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except Exception as exc:
        return {
            "schema_version": CODEX_CLI_AUTH_VERSION,
            "provider": "codex-cli",
            "auth_type": "chatgpt_oauth_delegate",
            "ok": False,
            "command": cmd,
            "error": str(exc),
            "token_access": "delegated_to_codex_cli",
        }
    output = str(getattr(completed, "stdout", "") or "")
    returncode = int(getattr(completed, "returncode", 1) or 0)
    status = codex_login_status(command=binary, runner=runner)
    return {
        "schema_version": CODEX_CLI_AUTH_VERSION,
        "provider": "codex-cli",
        "auth_type": "chatgpt_oauth_delegate",
        "ok": returncode == 0,
        "command": cmd,
        "returncode": returncode,
        "output_tail": _tail(output),
        "status": status,
        "token_access": "delegated_to_codex_cli",
        "quota_scope": "chatgpt_codex_plan_or_api_org_via_codex_cli",
    }


def run_codex_logout(
    *,
    command: str | None = None,
    timeout_seconds: int = 60,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    binary = _command(command)
    cmd = [binary, "logout"]
    try:
        completed = runner(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
    except Exception as exc:
        return {
            "schema_version": CODEX_CLI_AUTH_VERSION,
            "provider": "codex-cli",
            "ok": False,
            "command": cmd,
            "error": str(exc),
        }
    output = str(getattr(completed, "stdout", "") or "")
    return {
        "schema_version": CODEX_CLI_AUTH_VERSION,
        "provider": "codex-cli",
        "ok": int(getattr(completed, "returncode", 1) or 0) == 0,
        "command": cmd,
        "returncode": int(getattr(completed, "returncode", 1) or 0),
        "output_tail": _tail(output),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos auth codex",
        description="Delegate ChatGPT OAuth login to the official Codex CLI.",
    )
    parser.add_argument("--command", default="", help="Codex CLI binary. Defaults to CONOS_CODEX_COMMAND or codex.")
    subparsers = parser.add_subparsers(dest="command_name")
    login = subparsers.add_parser("login", help="Run codex login for ChatGPT/Codex access.")
    login.add_argument("--device-auth", action="store_true", help="Ask Codex CLI to use device auth when available.")
    login.add_argument("--no-interactive", action="store_true", help="Capture Codex login output instead of attaching to this terminal.")
    login.add_argument("--timeout", type=int, default=900)
    status = subparsers.add_parser("status", help="Show Codex CLI login status.")
    status.add_argument("--timeout", type=int, default=20)
    logout = subparsers.add_parser("logout", help="Run codex logout.")
    logout.add_argument("--timeout", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command_name = str(args.command_name or "status")
    binary = str(args.command or "").strip() or None
    if command_name == "login":
        payload = run_codex_login(
            command=binary,
            device_auth=bool(args.device_auth),
            interactive=not bool(args.no_interactive),
            timeout_seconds=int(args.timeout),
        )
        _print_json(payload)
        return 0 if bool(payload.get("ok", False)) else 1
    if command_name == "logout":
        payload = run_codex_logout(command=binary, timeout_seconds=int(args.timeout))
        _print_json(payload)
        return 0 if bool(payload.get("ok", False)) else 1
    payload = codex_login_status(command=binary, timeout_seconds=int(args.timeout))
    _print_json(payload)
    return 0 if bool(payload.get("connected", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
