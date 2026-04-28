#!/usr/bin/env python3
"""Minimal Con OS guest agent for managed Apple Virtualization guests."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from typing import Any


AGENT_VERSION = "conos.guest_agent/v0.1"
PROTOCOL_VERSION = "conos.guest_agent.protocol/v0.1"
HOST_CID = 2
DEFAULT_PORT = 48080


def _handshake(*, port: int, execution_ready: bool = True) -> dict[str, Any]:
    return {
        "event_type": "guest_agent_ready",
        "agent_version": AGENT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "pid": os.getpid(),
        "port": int(port),
        "execution_ready": bool(execution_ready),
        "capabilities": ["ready", "heartbeat", "exec"],
        "no_host_fallback": True,
    }


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _read_json_line(stream) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        return {}
    if isinstance(line, bytes):
        line = line.decode("utf-8", errors="replace")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return {"event_type": "invalid_json", "error": str(exc)}
    return payload if isinstance(payload, dict) else {"event_type": "invalid_json", "error": "payload is not an object"}


def _run_exec_command(request: dict[str, Any]) -> dict[str, Any]:
    command = request.get("command")
    if not isinstance(command, list) or not all(isinstance(part, str) for part in command) or not command:
        return {"event_type": "exec_result", "status": "INVALID_REQUEST", "reason": "command must be a non-empty string list"}
    timeout = int(request.get("timeout_seconds") or 30)
    cwd = request.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        return {"event_type": "exec_result", "status": "INVALID_REQUEST", "reason": "cwd must be a string"}
    completed = subprocess.run(
        command,
        cwd=cwd or None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "event_type": "exec_result",
        "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _serve(sock: socket.socket, *, port: int) -> int:
    with sock.makefile("rwb", buffering=0) as stream:
        stream.write(_json_line(_handshake(port=port)))
        while True:
            request = _read_json_line(stream)
            event_type = str(request.get("event_type") or request.get("action") or "")
            if not request:
                return 0
            if event_type in {"host_ack", "ack"}:
                continue
            if event_type == "ping":
                stream.write(_json_line({"event_type": "pong", "agent_version": AGENT_VERSION}))
            elif event_type == "exec":
                stream.write(_json_line(_run_exec_command(request)))
            elif event_type in {"shutdown", "stop"}:
                stream.write(_json_line({"event_type": "stopped", "status": "STOPPING"}))
                return 0
            else:
                stream.write(_json_line({"event_type": "error", "status": "UNKNOWN_EVENT", "request": request}))


def connect_vsock(port: int) -> socket.socket:
    if not hasattr(socket, "AF_VSOCK"):
        raise RuntimeError("Python socket.AF_VSOCK is not available in this guest")
    sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    sock.connect((HOST_CID, int(port)))
    return sock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Con OS managed-VM guest agent.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("CONOS_GUEST_AGENT_PORT", DEFAULT_PORT)))
    parser.add_argument("--print-handshake", action="store_true")
    args = parser.parse_args(argv)

    if args.print_handshake:
        print(json.dumps(_handshake(port=int(args.port)), indent=2, sort_keys=True))
        return 0

    try:
        sock = connect_vsock(int(args.port))
    except Exception as exc:  # pragma: no cover - depends on guest kernel vsock support.
        print(json.dumps({"event_type": "guest_agent_failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 78
    with sock:
        return _serve(sock, port=int(args.port))


if __name__ == "__main__":
    raise SystemExit(main())
