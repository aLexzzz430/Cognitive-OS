from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


class OllamaClient:
    """
    Lightweight client for an Ollama-compatible local chat endpoint.

    Expected default API shape:
    - GET  {base_url}/api/tags
    - POST {base_url}/api/chat
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 60.0,
        seed: Optional[int] = None,
        auto_select_model: bool = True,
    ) -> None:
        self._base_url = self._resolve_base_url(base_url)
        self._timeout_sec = float(timeout_sec or 60.0)
        self._model = str(model or "").strip()
        if not self._model and auto_select_model:
            self._model = self._resolve_default_model()
        self._seed = self._resolve_seed(seed)
        self._request_count = 0
        self._request_wall_sec = 0.0

    def _resolve_base_url(self, base_url: Optional[str]) -> str:
        resolved = str(base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip()
        return resolved.rstrip("/")

    def _resolve_default_model(self) -> str:
        env_model = str(os.getenv("OLLAMA_MODEL", "") or "").strip()
        if env_model:
            return env_model
        try:
            models = self.list_models()
        except Exception as exc:  # pragma: no cover - network path
            raise ValueError(
                "Ollama model is required. Provide llm_model/OLLAMA_MODEL, "
                f"or make {self._base_url}/api/tags reachable. Last error: {exc}"
            ) from exc
        if not models:
            raise ValueError(
                f"No Ollama models were reported by {self._base_url}/api/tags. "
                "Provide llm_model explicitly."
            )
        return models[0]

    def _resolve_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        env_seed = str(os.getenv("OLLAMA_SEED", "") or "").strip()
        if env_seed:
            try:
                return int(env_seed)
            except ValueError:
                pass
        return 7

    def list_models(self) -> List[str]:
        response = requests.get(f"{self._base_url}/api/tags", timeout=self._timeout_sec)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", []) if isinstance(payload, dict) else []
        names: List[str] = []
        for row in models:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "") or "").strip()
            if name:
                names.append(name)
        return names

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> str:
        raw_content = self.complete_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            think=think,
        )
        return self._strip_thinking(raw_content)

    def complete_raw(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        if not self._model:
            self._model = self._resolve_default_model()
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
                "seed": int(self._seed),
            },
        }
        if think is not None:
            payload["think"] = bool(think)
        started_at = time.perf_counter()
        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout_sec,
            )
        finally:
            self._request_count += 1
            self._request_wall_sec += max(0.0, time.perf_counter() - started_at)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {}) if isinstance(data, dict) else {}
        content = str(message.get("content", "") or "")
        thinking = str(message.get("thinking", "") or "")
        if content and thinking:
            return f"<think>{thinking}</think>\n{content}"
        if content:
            return content
        if thinking:
            return f"<think>{thinking}</think>"
        return ""

    def health(self) -> Dict[str, Any]:
        try:
            models = self.list_models()
        except Exception as exc:
            return {
                "provider": "ollama",
                "connected": False,
                "base_url": self._base_url,
                "selected_model": self._model,
                "models": [],
                "error": str(exc),
            }
        selected = self._model or (models[0] if models else "")
        return {
            "provider": "ollama",
            "connected": True,
            "base_url": self._base_url,
            "selected_model": selected,
            "models": models,
            "error": "",
        }

    def complete_json(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature, think=think).strip()
        if not text:
            return {}
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end] if start >= 0 and end > start else "{}"
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            fixed = self._fix_single_quoted_json(candidate)
            try:
                payload = json.loads(fixed)
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                return {}

    def _fix_single_quoted_json(self, text: str) -> str:
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            if c == "'":
                prev_char = text[i - 1] if i > 0 else ""
                next_char = text[i + 1] if i + 1 < len(text) else ""
                if prev_char.isalpha() and next_char.isalpha():
                    result.append(c)
                else:
                    result.append('"')
            else:
                result.append(c)
            i += 1
        return "".join(result)

    def _strip_thinking(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL)
        cleaned = cleaned.replace("</think>", "")
        return cleaned.strip()

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self._base_url!r}, model={self._model!r}, seed={self._seed!r})"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def request_count(self) -> int:
        return int(self._request_count)

    @property
    def request_wall_sec(self) -> float:
        return float(self._request_wall_sec)
