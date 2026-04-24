from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import requests


OPENAI_OAUTH_VERSION = "conos.openai_oauth/v1"
DEFAULT_REDIRECT_HOST = "127.0.0.1"
DEFAULT_REDIRECT_PORT = 8767
DEFAULT_REDIRECT_PATH = "/oauth/openai/callback"
DEFAULT_TOKEN_STORE = Path("runtime/auth/openai_oauth_token.json")
DEFAULT_SCOPES = ("openid", "profile", "email")


def _env_first(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = str(env.get(name, "") or "").strip()
        if value:
            return value
    return default


def _utc_now_ts() -> int:
    return int(time.time())


def _iso_utc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(timespec="seconds")


def _pkce_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _redact(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 10:
        return "<present>"
    return f"{text[:4]}...{text[-4:]}"


@dataclass(frozen=True)
class OpenAIOAuthConfig:
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    scopes: tuple[str, ...]
    redirect_host: str
    redirect_port: int
    redirect_path: str
    token_store_path: Path

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        cwd: str | Path | None = None,
    ) -> "OpenAIOAuthConfig":
        source = dict(os.environ if env is None else env)
        scope_text = _env_first(source, "OPENAI_OAUTH_SCOPES", "OPENAI_OAUTH_SCOPE", default=" ".join(DEFAULT_SCOPES))
        raw_path = _env_first(source, "OPENAI_OAUTH_TOKEN_STORE", "OPENAI_OAUTH_STORE", default=str(DEFAULT_TOKEN_STORE))
        store_path = Path(raw_path)
        if not store_path.is_absolute():
            store_path = (Path(cwd) if cwd is not None else Path.cwd()) / store_path
        raw_redirect_path = _env_first(source, "OPENAI_OAUTH_REDIRECT_PATH", default=DEFAULT_REDIRECT_PATH)
        redirect_path = raw_redirect_path if raw_redirect_path.startswith("/") else f"/{raw_redirect_path}"
        try:
            redirect_port = int(_env_first(source, "OPENAI_OAUTH_REDIRECT_PORT", default=str(DEFAULT_REDIRECT_PORT)))
        except ValueError:
            redirect_port = DEFAULT_REDIRECT_PORT
        return cls(
            client_id=_env_first(source, "OPENAI_OAUTH_CLIENT_ID"),
            client_secret=_env_first(source, "OPENAI_OAUTH_CLIENT_SECRET"),
            authorization_url=_env_first(source, "OPENAI_OAUTH_AUTHORIZATION_URL", "OPENAI_OAUTH_AUTH_URL"),
            token_url=_env_first(source, "OPENAI_OAUTH_TOKEN_URL"),
            scopes=tuple(item for item in scope_text.split() if item),
            redirect_host=_env_first(source, "OPENAI_OAUTH_REDIRECT_HOST", default=DEFAULT_REDIRECT_HOST),
            redirect_port=redirect_port,
            redirect_path=redirect_path,
            token_store_path=store_path,
        )

    @property
    def redirect_uri(self) -> str:
        return f"http://{self.redirect_host}:{self.redirect_port}{self.redirect_path}"

    def missing_required_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.client_id:
            missing.append("OPENAI_OAUTH_CLIENT_ID")
        if not self.authorization_url:
            missing.append("OPENAI_OAUTH_AUTHORIZATION_URL")
        if not self.token_url:
            missing.append("OPENAI_OAUTH_TOKEN_URL")
        return missing

    def is_configured(self) -> bool:
        return not self.missing_required_fields()


@dataclass(frozen=True)
class OAuthLoginSession:
    authorization_url: str
    state: str
    code_verifier: str
    code_challenge: str
    redirect_uri: str


@dataclass(frozen=True)
class OAuthCallbackResult:
    code: str
    state: str
    error: str = ""
    error_description: str = ""


def build_login_session(
    config: OpenAIOAuthConfig,
    *,
    state: str | None = None,
    code_verifier: str | None = None,
) -> OAuthLoginSession:
    verifier = code_verifier or secrets.token_urlsafe(64)
    challenge = _pkce_challenge(verifier)
    state_value = state or secrets.token_urlsafe(32)
    params = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": config.redirect_uri,
        "scope": " ".join(config.scopes),
        "state": state_value,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    separator = "&" if "?" in config.authorization_url else "?"
    return OAuthLoginSession(
        authorization_url=f"{config.authorization_url}{separator}{urlencode(params)}",
        state=state_value,
        code_verifier=verifier,
        code_challenge=challenge,
        redirect_uri=config.redirect_uri,
    )


def wait_for_callback(
    config: OpenAIOAuthConfig,
    *,
    expected_state: str,
    timeout_seconds: int = 300,
) -> OAuthCallbackResult:
    parsed_redirect = urlparse(config.redirect_uri)
    callback_path = parsed_redirect.path or DEFAULT_REDIRECT_PATH
    result: Dict[str, OAuthCallbackResult] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != callback_path:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return
            query = parse_qs(parsed.query)
            error = str((query.get("error") or [""])[0] or "")
            description = str((query.get("error_description") or [""])[0] or "")
            code = str((query.get("code") or [""])[0] or "")
            state = str((query.get("state") or [""])[0] or "")
            if state != expected_state:
                result["value"] = OAuthCallbackResult(
                    code="",
                    state=state,
                    error="invalid_state",
                    error_description="OAuth state did not match the login request.",
                )
                body = "OAuth login failed: invalid state."
            elif error:
                result["value"] = OAuthCallbackResult(code="", state=state, error=error, error_description=description)
                body = f"OAuth login failed: {error}."
            else:
                result["value"] = OAuthCallbackResult(code=code, state=state)
                body = "OAuth login received. You can return to Cognitive OS."
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(f"<html><body><p>{body}</p></body></html>".encode("utf-8"))

    server = ThreadingHTTPServer((config.redirect_host, config.redirect_port), Handler)
    server.timeout = 0.25
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        while "value" not in result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if "value" not in result:
        raise TimeoutError(f"Timed out waiting for OAuth callback on {config.redirect_uri}")
    return result["value"]


def exchange_code_for_token(
    config: OpenAIOAuthConfig,
    *,
    code: str,
    code_verifier: str,
    timeout_seconds: int = 30,
    http_post: Callable[..., Any] = requests.post,
) -> Dict[str, Any]:
    form = {
        "grant_type": "authorization_code",
        "client_id": config.client_id,
        "code": code,
        "redirect_uri": config.redirect_uri,
        "code_verifier": code_verifier,
    }
    if config.client_secret:
        form["client_secret"] = config.client_secret
    response = http_post(
        config.token_url,
        data=form,
        headers={"Accept": "application/json"},
        timeout=timeout_seconds,
    )
    status_code = int(getattr(response, "status_code", 0) or 0)
    try:
        payload = response.json()
    except Exception:
        payload = {"raw": str(getattr(response, "text", "") or "")}
    if status_code < 200 or status_code >= 300:
        raise RuntimeError(f"OAuth token exchange failed with status {status_code}: {payload}")
    if not isinstance(payload, dict) or not payload.get("access_token"):
        raise RuntimeError("OAuth token exchange did not return an access_token.")
    return dict(payload)


def save_token_response(config: OpenAIOAuthConfig, token_response: Mapping[str, Any]) -> Path:
    now = _utc_now_ts()
    payload = {
        "schema_version": OPENAI_OAUTH_VERSION,
        "provider": "openai_oauth",
        "saved_at": _iso_utc(now),
        "token_response": dict(token_response),
    }
    expires_in = token_response.get("expires_in")
    try:
        payload["expires_at"] = _iso_utc(now + int(expires_in))
    except (TypeError, ValueError):
        pass
    path = config.token_store_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    tmp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_token_payload(config: OpenAIOAuthConfig) -> Dict[str, Any]:
    path = config.token_store_path
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"invalid": True}
    return dict(payload) if isinstance(payload, dict) else {"invalid": True}


def delete_token(config: OpenAIOAuthConfig) -> bool:
    try:
        config.token_store_path.unlink()
        return True
    except FileNotFoundError:
        return False


def token_status(config: OpenAIOAuthConfig) -> Dict[str, Any]:
    payload = load_token_payload(config)
    token_response = payload.get("token_response", {}) if isinstance(payload.get("token_response", {}), dict) else {}
    expires_at = str(payload.get("expires_at", "") or "")
    expired: Optional[bool] = None
    if expires_at:
        try:
            expired = datetime.fromisoformat(expires_at).timestamp() <= time.time()
        except ValueError:
            expired = None
    return {
        "schema_version": OPENAI_OAUTH_VERSION,
        "configured": config.is_configured(),
        "missing_fields": config.missing_required_fields(),
        "redirect_uri": config.redirect_uri,
        "token_store_path": str(config.token_store_path),
        "token_present": bool(token_response.get("access_token")),
        "token_type": str(token_response.get("token_type", "") or ""),
        "scope": str(token_response.get("scope", "") or ""),
        "expires_at": expires_at,
        "expired": expired,
        "access_token": _redact(token_response.get("access_token")),
        "refresh_token": _redact(token_response.get("refresh_token")),
        "invalid_store": bool(payload.get("invalid", False)),
    }


def _print_json(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str))


def _login(args: argparse.Namespace) -> int:
    config = OpenAIOAuthConfig.from_env()
    missing = config.missing_required_fields()
    if missing:
        _print_json(
            {
                "ok": False,
                "error": "openai_oauth_not_configured",
                "missing_fields": missing,
                "note": "OpenAI API access uses API keys; OAuth login requires explicit OAuth provider endpoints configured by this app.",
            }
        )
        return 2
    session = build_login_session(config)
    print(f"Open this URL to sign in:\n{session.authorization_url}\n")
    if not bool(args.no_browser):
        webbrowser.open(session.authorization_url)
    try:
        callback = wait_for_callback(config, expected_state=session.state, timeout_seconds=int(args.timeout))
    except TimeoutError as exc:
        _print_json({"ok": False, "error": "oauth_callback_timeout", "detail": str(exc)})
        return 1
    if callback.error:
        _print_json({"ok": False, "error": callback.error, "detail": callback.error_description})
        return 1
    try:
        token_response = exchange_code_for_token(
            config,
            code=callback.code,
            code_verifier=session.code_verifier,
            timeout_seconds=int(args.token_timeout),
        )
    except Exception as exc:
        _print_json({"ok": False, "error": "oauth_token_exchange_failed", "detail": str(exc)})
        return 1
    path = save_token_response(config, token_response)
    status = token_status(config)
    status.update({"ok": True, "saved": str(path)})
    _print_json(status)
    return 0


def _status(_args: argparse.Namespace) -> int:
    _print_json(token_status(OpenAIOAuthConfig.from_env()))
    return 0


def _logout(_args: argparse.Namespace) -> int:
    config = OpenAIOAuthConfig.from_env()
    deleted = delete_token(config)
    _print_json({"ok": True, "deleted": deleted, "token_store_path": str(config.token_store_path)})
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conos auth openai", description="OpenAI-compatible OAuth login for Cognitive OS.")
    subparsers = parser.add_subparsers(dest="command")
    login_parser = subparsers.add_parser("login", help="Start Authorization Code + PKCE login.")
    login_parser.add_argument("--no-browser", action="store_true", help="Print the login URL without opening a browser.")
    login_parser.add_argument("--timeout", type=int, default=300, help="Seconds to wait for the local OAuth callback.")
    login_parser.add_argument("--token-timeout", type=int, default=30, help="Seconds to wait for token exchange.")
    subparsers.add_parser("status", help="Show local OAuth configuration and token status.")
    subparsers.add_parser("logout", help="Delete the stored local OAuth token.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = str(args.command or "status")
    if command == "login":
        return _login(args)
    if command == "logout":
        return _logout(args)
    return _status(args)
