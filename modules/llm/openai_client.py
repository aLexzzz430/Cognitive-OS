from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIClient:
    """Small Responses API client with the same surface used by Con OS LLM adapters."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 60.0,
        require_model: bool = True,
    ) -> None:
        self._api_key = str(api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self._base_url = str(base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
        self._model = str(model or os.getenv("OPENAI_MODEL") or "").strip()
        self._timeout_sec = float(timeout_sec or 60.0)
        self._request_count = 0
        self._request_wall_sec = 0.0
        if not self._api_key:
            raise ValueError("OpenAI provider requires OPENAI_API_KEY.")
        if require_model and not self._model:
            raise ValueError("OpenAI provider requires --llm-model or OPENAI_MODEL.")

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        **_: Any,
    ) -> str:
        return self.complete_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout_sec=timeout_sec,
        )

    def complete_raw(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        **_: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self._model,
            "input": str(prompt or ""),
            "max_output_tokens": int(max_tokens or 512),
            "temperature": float(temperature),
        }
        if system_prompt:
            payload["instructions"] = str(system_prompt)
        started_at = time.perf_counter()
        try:
            response = requests.post(
                f"{self._base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=float(timeout_sec if timeout_sec is not None else self._timeout_sec),
            )
        finally:
            self._request_count += 1
            self._request_wall_sec += max(0.0, time.perf_counter() - started_at)
        response.raise_for_status()
        return self._extract_text(response.json())

    def _extract_text(self, payload: Any) -> str:
        if not isinstance(payload, dict):
            return ""
        output_text = str(payload.get("output_text", "") or "")
        if output_text:
            return output_text
        chunks: list[str] = []
        for item in list(payload.get("output", []) or []):
            if not isinstance(item, dict):
                continue
            for content in list(item.get("content", []) or []):
                if not isinstance(content, dict):
                    continue
                text = str(content.get("text", "") or "")
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def health(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "connected": bool(self._api_key and self._model),
            "base_url": self._base_url,
            "selected_model": self._model,
            "error": "" if self._api_key and self._model else "missing_api_key_or_model",
        }

    def list_models(self) -> list[str]:
        response = requests.get(
            f"{self._base_url}/models",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        models: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("id", "") or "").strip()
            if model_id:
                models.append(model_id)
        return models

    def request_count(self) -> int:
        return int(self._request_count)

    def request_wall_sec(self) -> float:
        return float(self._request_wall_sec)
