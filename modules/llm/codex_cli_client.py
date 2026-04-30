from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional


DEFAULT_CODEX_MODEL = "gpt-5.3-codex-spark"


class CodexCliClient:
    """LLM client backed by the locally authenticated Codex CLI.

    This intentionally uses the user's existing Codex/OpenAI OAuth session
    instead of pretending that the OpenAI API supports the same OAuth token.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        command: Optional[str] = None,
        cwd: Optional[str] = None,
        sandbox: Optional[str] = None,
        timeout_sec: float = 300.0,
        runtime_plan: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._model = str(model or os.getenv("CODEX_MODEL") or DEFAULT_CODEX_MODEL).strip()
        self._command = str(command or os.getenv("CONOS_CODEX_COMMAND") or "codex").strip()
        self._cwd = Path(cwd or os.getenv("CONOS_CODEX_CWD") or os.getcwd()).resolve()
        self._sandbox = str(sandbox or os.getenv("CONOS_CODEX_SANDBOX") or "read-only").strip()
        self._timeout_sec = float(timeout_sec or 300.0)
        self._min_timeout_sec = float(os.getenv("CONOS_CODEX_MIN_TIMEOUT_SEC") or 20.0)
        self._request_count = 0
        self._request_wall_sec = 0.0
        self._last_usage: Dict[str, Any] = {}
        self._conos_llm_runtime_plan = dict(runtime_plan or {})
        if not self._model:
            raise ValueError("Codex CLI provider requires --llm-model or CODEX_MODEL.")
        if not self._command:
            raise ValueError("Codex CLI provider requires a codex command.")

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return "codex-cli://chatgpt-oauth"

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        return self.complete_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            timeout_sec=timeout_sec,
            **kwargs,
        )

    def complete_raw(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        del max_tokens, temperature
        full_prompt = self._compose_prompt(prompt, system_prompt=system_prompt)
        timeout = max(float(timeout_sec if timeout_sec is not None else self._timeout_sec), self._min_timeout_sec)
        with tempfile.TemporaryDirectory(prefix="conos_codex_llm_") as tmp:
            output_path = Path(tmp) / "last_message.txt"
            command = self._build_command(output_path, kwargs)
            started_at = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self._cwd),
                    input=full_prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    check=False,
                    env=self._subprocess_env(),
                )
            except subprocess.TimeoutExpired as exc:
                self._record_request(started_at)
                raise TimeoutError(f"Codex CLI model timed out after {timeout:.1f}s") from exc
            self._record_request(started_at)
            self._last_usage = self._extract_usage(completed.stdout)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Codex CLI model call failed with returncode "
                    f"{completed.returncode}: {self._tail(completed.stdout)}"
                )
            text = ""
            try:
                text = output_path.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
            return self._extract_stdout_message(completed.stdout)

    def health(self) -> Dict[str, Any]:
        try:
            completed = subprocess.run(
                [self._command, "login", "status"],
                cwd=str(self._cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=min(10.0, self._timeout_sec),
                check=False,
                env=self._subprocess_env(),
            )
        except Exception as exc:
            return {
                "provider": "codex-cli",
                "connected": False,
                "base_url": self.base_url,
                "selected_model": self._model,
                "error": str(exc),
            }
        output = str(completed.stdout or "")
        connected = completed.returncode == 0 and "Logged in" in output
        return {
            "provider": "codex-cli",
            "connected": connected,
            "base_url": self.base_url,
            "selected_model": self._model,
            "auth": "chatgpt_oauth_delegate",
            "auth_profile": self.auth_profile(),
            "execution_runtime": self.execution_runtime(),
            "error": "" if connected else self._tail(output),
        }

    def is_available(self) -> bool:
        return bool(self.health().get("connected", False))

    def request_count(self) -> int:
        return int(self._request_count)

    def request_wall_sec(self) -> float:
        return float(self._request_wall_sec)

    def last_usage(self) -> Dict[str, Any]:
        return dict(self._last_usage)

    def auth_profile(self) -> Dict[str, Any]:
        plan = dict(getattr(self, "_conos_llm_runtime_plan", {}) or {})
        profile = plan.get("auth_profile", {}) if isinstance(plan.get("auth_profile", {}), dict) else {}
        if profile:
            return dict(profile)
        return {
            "provider": "codex-cli",
            "auth_type": "chatgpt_oauth_delegate",
            "credential_source": "codex_cli_local_credentials",
            "requires_user_login": True,
            "login_command": [self._command, "login"],
            "status_command": [self._command, "login", "status"],
            "token_storage": "managed_by_codex_cli",
            "direct_token_access": False,
            "quota_scope": "chatgpt_codex_plan_or_api_org_via_codex_cli",
        }

    def execution_runtime(self) -> Dict[str, Any]:
        plan = dict(getattr(self, "_conos_llm_runtime_plan", {}) or {})
        runtime = plan.get("execution_runtime", {}) if isinstance(plan.get("execution_runtime", {}), dict) else {}
        if runtime:
            return dict(runtime)
        return {
            "runtime_id": "codex_cli_exec",
            "runtime_type": "local_cli_agent",
            "command": self._command,
            "cwd": str(self._cwd),
            "sandbox": self._sandbox,
            "timeout_sec": self._timeout_sec,
            "local_credentials_allowed": True,
        }

    def _build_command(self, output_path: Path, kwargs: Dict[str, Any]) -> list[str]:
        command = [
            self._command,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            self._sandbox,
            "--json",
            "-m",
            self._model,
            "-o",
            str(output_path),
        ]
        effort = self._reasoning_effort(kwargs)
        if effort:
            command.extend(["-c", f'model_reasoning_effort="{effort}"'])
        command.append("-")
        return command

    def _reasoning_effort(self, kwargs: Dict[str, Any]) -> str:
        think = kwargs.get("think")
        budget = kwargs.get("thinking_budget")
        if think is False or budget == 0:
            return "low"
        if think is True and budget is None:
            return "high"
        try:
            budget_int = int(budget)
        except (TypeError, ValueError):
            return "medium"
        if budget_int >= 2048:
            return "high"
        if budget_int <= 256:
            return "low"
        return "medium"

    def _compose_prompt(self, prompt: str, *, system_prompt: Optional[str]) -> str:
        pieces: list[str] = []
        if system_prompt:
            pieces.append("System instructions:\n" + str(system_prompt).strip())
        pieces.append(
            "You are being used as a bounded LLM backend for Cognitive OS. "
            "Do not edit files or run commands unless the prompt explicitly asks you to; "
            "return only the requested model output."
        )
        pieces.append(str(prompt or ""))
        return "\n\n".join(piece for piece in pieces if piece).strip() + "\n"

    def _subprocess_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.setdefault("CODEX_CI", "1")
        return env

    def _record_request(self, started_at: float) -> None:
        self._request_count += 1
        self._request_wall_sec += max(0.0, time.perf_counter() - started_at)

    def _extract_stdout_message(self, stdout: str) -> str:
        lines = [line.rstrip() for line in str(stdout or "").splitlines()]
        if "tokens used" in lines:
            lines = lines[: lines.index("tokens used")]
        if "codex" in lines:
            lines = lines[lines.index("codex") + 1 :]
        filtered = [
            line
            for line in lines
            if line
            and not line.startswith("OpenAI Codex ")
            and not line.startswith("--------")
            and not line.startswith("workdir:")
            and not line.startswith("model:")
            and not line.startswith("provider:")
            and not line.startswith("approval:")
            and not line.startswith("sandbox:")
            and not line.startswith("reasoning ")
            and not line.startswith("session id:")
            and not line.startswith("user")
            and "WARN " not in line
            and "Reading additional input" not in line
        ]
        return "\n".join(filtered).strip()

    def _extract_usage(self, stdout: str) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}
        for line in str(stdout or "").splitlines():
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("type") == "turn.completed":
                raw_usage = payload.get("usage", {})
                if isinstance(raw_usage, dict):
                    usage = dict(raw_usage)
        return usage

    def _tail(self, text: str, limit: int = 2000) -> str:
        value = str(text or "").strip()
        if len(value) <= limit:
            return value
        return value[-limit:]
