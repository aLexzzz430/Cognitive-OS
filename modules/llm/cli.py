from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Mapping, Sequence

from modules.llm.model_profile import (
    build_model_route_summary,
    load_profile_backed_route_policies,
    list_openai_models,
    profile_all_configured_models,
    profile_provider_models,
    render_model_route_summary,
)
from modules.llm.provider_inventory import list_visible_provider_models
from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient
from modules.llm.openai_client import DEFAULT_OPENAI_BASE_URL, OpenAIClient
from modules.llm.codex_cli_client import CodexCliClient
from modules.llm.runtime_contracts import build_llm_runtime_plan
from modules.control_plane import (
    AGENT_CONTROL_PLANE_VERSION,
    AgentControlPlane,
    AgentControlRequest,
    agent_specs_from_model_route_policies,
    load_agent_registry,
    render_agent_control_decision,
)


LLM_CLI_VERSION = "conos.llm_cli/v1"


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True, default=str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conos llm",
        description="Check and use local-first LLM providers.",
    )
    parser.add_argument("--provider", default="ollama", choices=("ollama", "openai", "codex", "codex-cli", "all"), help="LLM provider to inspect/profile.")
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Provider base URL. For Ollama use the LAN host URL, e.g. http://192.168.1.23:11434. "
            f"Defaults: OLLAMA_BASE_URL or {DEFAULT_OLLAMA_BASE_URL}; OPENAI_BASE_URL or {DEFAULT_OPENAI_BASE_URL}."
        ),
    )
    parser.add_argument("--openai-base-url", default=None, help="OpenAI-compatible base URL used when --provider all.")
    parser.add_argument("--model", default=None, help="Model name, e.g. qwen3:8b.")
    parser.add_argument("--models", default="", help="Comma-separated model names. Overrides provider inventory when set.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("check", help="Check provider connectivity and list available models.")
    subparsers.add_parser("list", help="List provider models.")
    subparsers.add_parser("runtime-plan", help="Show separated provider/auth/runtime/tool/cost/context/verifier contracts.")

    prompt_parser = subparsers.add_parser("prompt", help="Send one prompt to the selected model.")
    prompt_parser.add_argument("prompt")
    prompt_parser.add_argument("--max-tokens", type=int, default=128)
    prompt_parser.add_argument("--temperature", type=float, default=0.0)
    prompt_parser.add_argument("--raw", action="store_true", help="Keep model thinking fields when the backend returns them.")

    profile_parser = subparsers.add_parser("profile", help="Profile provider models and emit route policies.")
    profile_parser.add_argument("--store", default="", help="Optional model profile store path. Defaults to ~/.conos/runtime/model_profiles.json.")
    profile_parser.add_argument("--force", action="store_true", help="Regenerate profiles even when cached profiles exist.")
    profile_parser.add_argument(
        "--all-cloud-models",
        action="store_true",
        help="For OpenAI/API providers, list all text-capable cloud models from /models before profiling.",
    )
    profile_parser.add_argument("--max-cloud-models", type=int, default=0, help="Optional safety cap for cloud model profiling.")
    profile_parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Create profiles from provider-visible model catalogs without live probe prompts.",
    )
    profile_parser.add_argument(
        "--discover-visible",
        action="store_true",
        help="Pull all visible models from the provider before creating profiles.",
    )
    profile_parser.add_argument("--include-hidden", action="store_true", help="Include hidden provider models when the provider exposes them.")
    profile_parser.add_argument(
        "--route-policy-output",
        default="",
        help="Optional JSON path that receives only generated llm_route_policies.",
    )

    routes_parser = subparsers.add_parser("routes", help="Show profile-backed model routing decisions.")
    routes_parser.add_argument("--store", default="", help="Optional model profile store path.")
    routes_parser.add_argument("--route-policy-file", default="", help="Optional llm_route_policies JSON path.")
    routes_parser.add_argument("--explain", action="store_true", help="Include candidate route scoring details.")
    routes_parser.add_argument("--format", choices=("text", "json", "both"), default="text")

    control_parser = subparsers.add_parser("control-plane", help="Resolve a model-agnostic agent/control-plane decision.")
    control_parser.add_argument("--store", default="", help="Optional model profile store path.")
    control_parser.add_argument("--route-policy-file", default="", help="Optional llm_route_policies JSON path.")
    control_parser.add_argument("--agent-registry", default="", help="Optional JSON registry of non-model agents/tools.")
    control_parser.add_argument("--task-type", default="general")
    control_parser.add_argument("--route", default="", help="Requested route, e.g. structured_answer or coding.")
    control_parser.add_argument("--capability", default="", help="Concrete capability request name.")
    control_parser.add_argument("--required-capability", action="append", default=[])
    control_parser.add_argument("--permission", action="append", default=[])
    control_parser.add_argument("--risk-level", default="low", choices=("none", "low", "medium", "high", "critical"))
    control_parser.add_argument("--prefer-low-cost", type=float, default=0.0)
    control_parser.add_argument("--prefer-low-latency", type=float, default=0.0)
    control_parser.add_argument("--prefer-high-trust", type=float, default=0.0)
    control_parser.add_argument("--prefer-local", type=float, default=0.0)
    control_parser.add_argument("--format", choices=("text", "json", "both"), default="text")
    return parser


def _ollama(args: argparse.Namespace, *, auto_select_model: bool = False) -> OllamaClient:
    return OllamaClient(
        base_url=args.base_url,
        model=args.model,
        timeout_sec=float(args.timeout),
        auto_select_model=auto_select_model,
    )


def _openai(args: argparse.Namespace) -> OpenAIClient:
    return OpenAIClient(
        base_url=args.base_url,
        model=args.model,
        timeout_sec=float(args.timeout),
    )


def _codex(args: argparse.Namespace) -> CodexCliClient:
    return CodexCliClient(
        model=args.model or None,
        timeout_sec=max(300.0, float(args.timeout)),
    )


def _selected_models(args: argparse.Namespace) -> list[str]:
    models: list[str] = []
    if str(getattr(args, "models", "") or "").strip():
        models.extend(part.strip() for part in str(args.models).replace("\n", ",").split(",") if part.strip())
    if str(getattr(args, "model", "") or "").strip():
        models.append(str(args.model).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)
    return deduped


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "check":
        if args.provider == "ollama":
            client = _ollama(args, auto_select_model=False)
            health = client.health()
            health["schema_version"] = LLM_CLI_VERSION
            print(_json_dumps(health))
            return 0 if bool(health.get("connected", False)) else 1
        if args.provider == "openai":
            try:
                client = OpenAIClient(base_url=args.base_url, model=args.model or "", timeout_sec=float(args.timeout), require_model=False)
                models = client.list_models()
                health = {
                    "schema_version": LLM_CLI_VERSION,
                    "provider": "openai",
                    "connected": True,
                    "base_url": client.base_url,
                    "selected_model": args.model or "",
                    "model_count": len(models),
                    "models": models,
                    "error": "",
                }
                print(_json_dumps(health))
                return 0
            except Exception as exc:
                print(_json_dumps({"schema_version": LLM_CLI_VERSION, "provider": "openai", "connected": False, "base_url": args.base_url or DEFAULT_OPENAI_BASE_URL, "models": [], "error": str(exc)}))
                return 1
        if args.provider in {"codex", "codex-cli"}:
            client = _codex(args)
            health = client.health()
            health["schema_version"] = LLM_CLI_VERSION
            print(_json_dumps(health))
            return 0 if bool(health.get("connected", False)) else 1
        statuses = []
        exit_code = 1
        for provider in ("ollama", "openai", "codex"):
            nested = argparse.Namespace(**vars(args))
            nested.provider = provider
            if provider == "openai" and args.openai_base_url:
                nested.base_url = args.openai_base_url
            try:
                if provider == "ollama":
                    client = _ollama(nested, auto_select_model=False)
                    health = client.health()
                elif provider == "openai":
                    client = OpenAIClient(base_url=nested.base_url, model=args.model or "", timeout_sec=float(args.timeout), require_model=False)
                    models = client.list_models()
                    health = {"provider": "openai", "connected": True, "base_url": client.base_url, "selected_model": args.model or "", "models": models, "error": ""}
                else:
                    client = _codex(nested)
                    health = client.health()
            except Exception as exc:
                health = {"provider": provider, "connected": False, "base_url": str(nested.base_url or ""), "models": [], "error": str(exc)}
            statuses.append(health)
            if bool(health.get("connected", False)):
                exit_code = 0
        print(_json_dumps({"schema_version": LLM_CLI_VERSION, "provider": "all", "providers": statuses}))
        return exit_code

    if args.command == "list":
        if args.provider == "ollama":
            client = _ollama(args, auto_select_model=False)
            try:
                models = client.list_models()
            except Exception as exc:
                print(_json_dumps({"provider": "ollama", "connected": False, "base_url": client.base_url, "models": [], "error": str(exc)}))
                return 1
            print(_json_dumps({"provider": "ollama", "connected": True, "base_url": client.base_url, "models": models, "error": ""}))
            return 0
        if args.provider == "openai":
            try:
                models = list_openai_models(base_url=args.base_url, timeout_sec=float(args.timeout))
            except Exception as exc:
                print(_json_dumps({"provider": "openai", "connected": False, "base_url": args.base_url or DEFAULT_OPENAI_BASE_URL, "models": [], "error": str(exc)}))
                return 1
            print(_json_dumps({"provider": "openai", "connected": True, "base_url": args.base_url or DEFAULT_OPENAI_BASE_URL, "models": models, "error": ""}))
            return 0
        if args.provider in {"codex", "codex-cli"}:
            try:
                visible = list_visible_provider_models(
                    provider="codex-cli",
                    timeout_sec=float(args.timeout),
                )
            except Exception as exc:
                print(_json_dumps({
                    "provider": "codex-cli",
                    "connected": False,
                    "base_url": "codex-cli://chatgpt-oauth",
                    "models": [],
                    "visible_models": [],
                    "error": str(exc),
                }))
                return 1
            print(_json_dumps({
                "provider": "codex-cli",
                "connected": True,
                "base_url": "codex-cli://chatgpt-oauth",
                "models": [row.model for row in visible],
                "visible_models": [row.to_dict() for row in visible],
                "model_inventory": "codex_debug_models",
                "error": "",
            }))
            return 0
        payload = {"provider": "all", "providers": []}
        exit_code = 1
        for provider in ("ollama", "openai", "codex"):
            nested = argparse.Namespace(**vars(args))
            nested.provider = provider
            if provider == "openai" and args.openai_base_url:
                nested.base_url = args.openai_base_url
            try:
                if provider == "ollama":
                    client = _ollama(nested, auto_select_model=False)
                    models = client.list_models()
                    base_url = client.base_url
                elif provider == "openai":
                    models = list_openai_models(base_url=nested.base_url, timeout_sec=float(args.timeout))
                    base_url = nested.base_url or DEFAULT_OPENAI_BASE_URL
                else:
                    visible = list_visible_provider_models(provider="codex-cli", timeout_sec=float(args.timeout))
                    models = [row.model for row in visible]
                    base_url = "codex-cli://chatgpt-oauth"
                payload["providers"].append({"provider": provider, "connected": True, "base_url": base_url, "models": models, "error": ""})
                exit_code = 0
            except Exception as exc:
                payload["providers"].append({"provider": provider, "connected": False, "base_url": str(nested.base_url or ""), "models": [], "error": str(exc)})
        print(_json_dumps(payload))
        return exit_code

    if args.command == "runtime-plan":
        provider = "codex-cli" if args.provider == "codex" else args.provider
        if provider == "all":
            plans = {
                name: build_llm_runtime_plan(
                    name,
                    model=args.model or "",
                    base_url=(args.openai_base_url if name == "openai" else args.base_url) or "",
                    timeout_sec=float(args.timeout),
                ).to_dict()
                for name in ("ollama", "openai", "codex-cli")
            }
            print(_json_dumps({"schema_version": LLM_CLI_VERSION, "provider": "all", "runtime_plans": plans}))
            return 0
        plan = build_llm_runtime_plan(
            provider,
            model=args.model or "",
            base_url=args.base_url or "",
            timeout_sec=float(args.timeout),
        )
        print(_json_dumps(plan.to_dict()))
        return 0

    if args.command == "prompt":
        if args.provider == "all":
            parser.error("prompt requires --provider ollama, openai, or codex")
        if args.provider == "ollama":
            client = _ollama(args, auto_select_model=True)
        elif args.provider == "openai":
            client = _openai(args)
        else:
            client = _codex(args)
        if args.raw:
            text = client.complete_raw(args.prompt, max_tokens=int(args.max_tokens), temperature=float(args.temperature))
        else:
            text = client.complete(args.prompt, max_tokens=int(args.max_tokens), temperature=float(args.temperature))
        print(_json_dumps({"provider": args.provider, "base_url": client.base_url, "model": client.model, "response": text}))
        return 0

    if args.command == "profile":
        selected_models = _selected_models(args) or None
        if args.provider == "all":
            report = profile_all_configured_models(
                ollama_base_url=args.base_url,
                openai_base_url=args.openai_base_url,
                ollama_models=selected_models,
                openai_models=selected_models,
                timeout_sec=float(args.timeout),
                store_path=args.store or None,
                force=bool(args.force),
                all_cloud_models=bool(args.all_cloud_models),
                max_cloud_models=int(args.max_cloud_models or 0) or None,
                catalog_only=bool(args.catalog_only or args.discover_visible),
                include_codex=True,
            )
        else:
            report = profile_provider_models(
                provider=args.provider,
                base_url=args.base_url,
                models=selected_models,
                timeout_sec=float(args.timeout),
                store_path=args.store or None,
                force=bool(args.force),
                all_cloud_models=bool(args.all_cloud_models),
                max_cloud_models=int(args.max_cloud_models or 0) or None,
                catalog_only=bool(args.catalog_only),
                discover_visible=bool(args.discover_visible),
                include_hidden=bool(args.include_hidden),
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

    if args.command == "routes":
        policies = load_profile_backed_route_policies(
            store_path=args.store or None,
            route_policy_path=args.route_policy_file or None,
            base_url=args.base_url,
        )
        summary = build_model_route_summary(policies, explain=bool(args.explain))
        if args.format in {"text", "both"}:
            print(render_model_route_summary(summary))
        if args.format in {"json", "both"}:
            print(_json_dumps(summary))
        return 0 if policies else 1

    if args.command == "control-plane":
        policies = load_profile_backed_route_policies(
            store_path=args.store or None,
            route_policy_path=args.route_policy_file or None,
            base_url=args.base_url,
        )
        agents = agent_specs_from_model_route_policies(policies)
        agents.extend(load_agent_registry(args.agent_registry or None))
        decision = AgentControlPlane(agents).decide(
            AgentControlRequest(
                task_type=args.task_type,
                route_name=args.route,
                capability_request=args.capability,
                required_capabilities=list(args.required_capability or []),
                permissions_required=list(args.permission or []),
                risk_level=args.risk_level,
                prefer_low_cost=float(args.prefer_low_cost),
                prefer_low_latency=float(args.prefer_low_latency),
                prefer_high_trust=float(args.prefer_high_trust),
                prefer_local=float(args.prefer_local),
            )
        )
        payload = {
            "schema_version": AGENT_CONTROL_PLANE_VERSION,
            "agent_count": len(agents),
            "route_policy_count": len(policies),
            "decision": decision.to_dict(),
        }
        if args.format in {"text", "both"}:
            print(render_agent_control_decision(decision))
        if args.format in {"json", "both"}:
            print(_json_dumps(payload))
        return 0 if decision.status in {"SELECTED", "WAITING_APPROVAL"} else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
