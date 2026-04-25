from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Sequence

from modules.llm.model_profile import profile_ollama_models
from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient


LLM_CLI_VERSION = "conos.llm_cli/v1"


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True, default=str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos llm",
        description="Check and use local-first LLM providers.",
    )
    parser.add_argument("--provider", default="ollama", choices=("ollama",), help="Local LLM provider.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"Ollama base URL. Use the LAN host URL, e.g. http://192.168.1.23:11434. Default: OLLAMA_BASE_URL or {DEFAULT_OLLAMA_BASE_URL}.",
    )
    parser.add_argument("--model", default=None, help="Model name, e.g. qwen3:8b.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("check", help="Check provider connectivity and list available models.")
    subparsers.add_parser("list", help="List provider models.")

    prompt_parser = subparsers.add_parser("prompt", help="Send one prompt to the selected model.")
    prompt_parser.add_argument("prompt")
    prompt_parser.add_argument("--max-tokens", type=int, default=128)
    prompt_parser.add_argument("--temperature", type=float, default=0.0)
    prompt_parser.add_argument("--raw", action="store_true", help="Keep model thinking fields when the backend returns them.")

    profile_parser = subparsers.add_parser("profile", help="Profile available Ollama models and emit route policies.")
    profile_parser.add_argument("--store", default="", help="Optional model profile store path. Defaults to ~/.conos/runtime/model_profiles.json.")
    profile_parser.add_argument("--force", action="store_true", help="Regenerate profiles even when cached profiles exist.")
    profile_parser.add_argument(
        "--route-policy-output",
        default="",
        help="Optional JSON path that receives only generated llm_route_policies.",
    )
    return parser


def _ollama(args: argparse.Namespace, *, auto_select_model: bool = False) -> OllamaClient:
    return OllamaClient(
        base_url=args.base_url,
        model=args.model,
        timeout_sec=float(args.timeout),
        auto_select_model=auto_select_model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0
    if args.provider != "ollama":
        parser.error(f"unsupported provider: {args.provider}")

    if args.command == "check":
        client = _ollama(args, auto_select_model=False)
        health = client.health()
        health["schema_version"] = LLM_CLI_VERSION
        print(_json_dumps(health))
        return 0 if bool(health.get("connected", False)) else 1

    if args.command == "list":
        client = _ollama(args, auto_select_model=False)
        try:
            models = client.list_models()
        except Exception as exc:
            print(_json_dumps({"provider": "ollama", "connected": False, "base_url": client.base_url, "models": [], "error": str(exc)}))
            return 1
        print(_json_dumps({"provider": "ollama", "connected": True, "base_url": client.base_url, "models": models, "error": ""}))
        return 0

    if args.command == "prompt":
        client = _ollama(args, auto_select_model=True)
        if args.raw:
            text = client.complete_raw(args.prompt, max_tokens=int(args.max_tokens), temperature=float(args.temperature))
        else:
            text = client.complete(args.prompt, max_tokens=int(args.max_tokens), temperature=float(args.temperature))
        print(_json_dumps({"provider": "ollama", "base_url": client.base_url, "model": client.model, "response": text}))
        return 0

    if args.command == "profile":
        report = profile_ollama_models(
            base_url=args.base_url,
            models=[args.model] if args.model else None,
            timeout_sec=float(args.timeout),
            store_path=args.store or None,
            force=bool(args.force),
        )
        route_policy_output = str(args.route_policy_output or "").strip()
        if route_policy_output:
            from pathlib import Path

            path = Path(route_policy_output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(report.get("route_policies", {}), indent=2, ensure_ascii=False, sort_keys=True, default=str),
                encoding="utf-8",
            )
        print(_json_dumps(report))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
