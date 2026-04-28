from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
import subprocess
from typing import Any, Dict, Iterable, Mapping, Sequence

from modules.llm.ollama_client import DEFAULT_OLLAMA_BASE_URL, OllamaClient
from modules.llm.openai_client import DEFAULT_OPENAI_BASE_URL, OpenAIClient


PROVIDER_INVENTORY_VERSION = "conos.llm.provider_inventory/v1"


@dataclass(frozen=True)
class VisibleModel:
    provider: str
    model: str
    base_url: str = ""
    display_name: str = ""
    visibility: str = "list"
    supported_in_api: bool | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = PROVIDER_INVENTORY_VERSION
        return payload


def _split_model_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        chunks = value.replace("\n", ",").split(",")
    elif isinstance(value, Sequence):
        chunks = [str(item) for item in value]
    else:
        chunks = [str(value)]
    seen: set[str] = set()
    names: list[str] = []
    for chunk in chunks:
        name = str(chunk or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _looks_like_text_model(model_id: str) -> bool:
    name = str(model_id or "").strip().lower()
    if not name:
        return False
    excluded = (
        "embedding",
        "moderation",
        "whisper",
        "tts",
        "audio",
        "realtime",
        "transcribe",
        "dall-e",
        "image",
        "speech",
    )
    if any(fragment in name for fragment in excluded):
        return False
    return name.startswith(("gpt-", "chatgpt-", "o1", "o2", "o3", "o4", "o5", "codex"))


def _json_from_codex_debug_output(output: str) -> Mapping[str, Any]:
    for line in reversed(str(output or "").splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    return {}


def _sanitize_provider_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    blocked = {
        "base_instructions",
        "instructions",
        "instructions_template",
        "instructions_variables",
        "model_messages",
        "personality",
        "prompt",
        "system_prompt",
    }
    metadata: Dict[str, Any] = {}
    for key, value in dict(row or {}).items():
        if str(key) in blocked:
            continue
        metadata[str(key)] = value
    return metadata


def list_codex_visible_models(
    *,
    command: str | None = None,
    timeout_sec: float = 30.0,
    include_hidden: bool = False,
    runner: Any = None,
) -> list[VisibleModel]:
    binary = str(command or os.getenv("CONOS_CODEX_COMMAND") or "codex").strip() or "codex"
    run = runner or subprocess.run
    completed = run(
        [binary, "debug", "models"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max(1.0, float(timeout_sec or 30.0)),
        check=False,
    )
    if int(getattr(completed, "returncode", 1) or 0) != 0:
        raise RuntimeError(str(getattr(completed, "stdout", "") or "codex debug models failed"))
    payload = _json_from_codex_debug_output(str(getattr(completed, "stdout", "") or ""))
    models: list[VisibleModel] = []
    for row in list(payload.get("models", []) or []):
        if not isinstance(row, Mapping):
            continue
        slug = str(row.get("slug", "") or "").strip()
        if not slug:
            continue
        visibility = str(row.get("visibility", "") or "list").strip() or "list"
        if visibility != "list" and not include_hidden:
            continue
        models.append(
            VisibleModel(
                provider="codex-cli",
                model=slug,
                base_url="codex-cli://chatgpt-oauth",
                display_name=str(row.get("display_name", "") or slug),
                visibility=visibility,
                supported_in_api=bool(row.get("supported_in_api")) if row.get("supported_in_api") is not None else None,
                metadata=_sanitize_provider_metadata(row),
            )
        )
    return models


def list_visible_provider_models(
    *,
    provider: str,
    base_url: str | None = None,
    models: Iterable[str] | None = None,
    timeout_sec: float = 30.0,
    include_hidden: bool = False,
    text_only: bool = True,
) -> list[VisibleModel]:
    normalized = str(provider or "").strip().lower()
    requested = _split_model_names(models)
    requested_set = set(requested)
    if normalized in {"ollama", "local", "local-http"}:
        client = OllamaClient(base_url=base_url, auto_select_model=False, timeout_sec=timeout_sec)
        names = requested or client.list_models()
        return [
            VisibleModel(provider="ollama", model=name, base_url=client.base_url, display_name=name)
            for name in names
        ]
    if normalized in {"openai", "responses"}:
        client = OpenAIClient(base_url=base_url, model="", timeout_sec=timeout_sec, require_model=False)
        names = requested or client.list_models()
        if text_only:
            names = [name for name in names if _looks_like_text_model(name)]
        return [
            VisibleModel(
                provider="openai",
                model=name,
                base_url=client.base_url or DEFAULT_OPENAI_BASE_URL,
                display_name=name,
                supported_in_api=True,
            )
            for name in names
        ]
    if normalized in {"codex", "codex-cli", "openai-oauth-codex"}:
        visible = list_codex_visible_models(timeout_sec=timeout_sec, include_hidden=include_hidden)
        if requested_set:
            visible = [row for row in visible if row.model in requested_set]
        return visible
    raise ValueError(f"Unsupported provider inventory: {provider}")


def inventory_report(
    *,
    provider: str,
    base_url: str | None = None,
    models: Iterable[str] | None = None,
    timeout_sec: float = 30.0,
    include_hidden: bool = False,
) -> Dict[str, Any]:
    visible = list_visible_provider_models(
        provider=provider,
        base_url=base_url,
        models=models,
        timeout_sec=timeout_sec,
        include_hidden=include_hidden,
    )
    return {
        "schema_version": PROVIDER_INVENTORY_VERSION,
        "provider": str(provider or ""),
        "base_url": str(base_url or ""),
        "visible_model_count": len(visible),
        "models": [row.to_dict() for row in visible],
    }
