from __future__ import annotations

from typing import Any, Optional

from modules.llm.minimax_client import MinimaxClient
from modules.llm.ollama_client import OllamaClient


def build_llm_client(
    provider: str = "none",
    *,
    token_file: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 60.0,
) -> Any:
    normalized = (provider or "none").strip().lower()
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized == "minimax":
        return MinimaxClient(token_file=token_file)
    if normalized in {"ollama", "local", "local-http"}:
        return OllamaClient(base_url=base_url, model=model, timeout_sec=timeout_sec)
    raise ValueError(f"Unsupported llm provider: {provider}")
