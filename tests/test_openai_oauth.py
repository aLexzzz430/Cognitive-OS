import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.auth.openai_oauth import (
    OpenAIOAuthConfig,
    build_login_session,
    exchange_code_for_token,
    save_token_response,
    token_status,
)


def _env(tmp_path: Path, **overrides):
    env = {
        "OPENAI_OAUTH_CLIENT_ID": "client-1",
        "OPENAI_OAUTH_CLIENT_SECRET": "secret-1",
        "OPENAI_OAUTH_AUTHORIZATION_URL": "https://auth.example.test/authorize",
        "OPENAI_OAUTH_TOKEN_URL": "https://auth.example.test/token",
        "OPENAI_OAUTH_SCOPES": "openid profile",
        "OPENAI_OAUTH_TOKEN_STORE": str(tmp_path / "token.json"),
    }
    env.update(overrides)
    return env


def test_openai_oauth_config_requires_explicit_provider_urls(tmp_path):
    config = OpenAIOAuthConfig.from_env(
        {
            "OPENAI_OAUTH_CLIENT_ID": "client-1",
            "OPENAI_OAUTH_TOKEN_STORE": str(tmp_path / "token.json"),
        }
    )

    assert config.is_configured() is False
    assert config.missing_required_fields() == [
        "OPENAI_OAUTH_AUTHORIZATION_URL",
        "OPENAI_OAUTH_TOKEN_URL",
    ]
    assert config.redirect_uri == "http://127.0.0.1:8767/oauth/openai/callback"


def test_openai_oauth_login_session_uses_pkce_state_and_redirect(tmp_path):
    config = OpenAIOAuthConfig.from_env(_env(tmp_path))

    session = build_login_session(
        config,
        state="state-123",
        code_verifier="verifier-123",
    )

    parsed = urlparse(session.authorization_url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "auth.example.test"
    assert parsed.path == "/authorize"
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-1"]
    assert query["redirect_uri"] == ["http://127.0.0.1:8767/oauth/openai/callback"]
    assert query["scope"] == ["openid profile"]
    assert query["state"] == ["state-123"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == [session.code_challenge]
    assert session.code_challenge != "verifier-123"


def test_openai_oauth_exchange_posts_form_and_stores_redacted_status(tmp_path):
    config = OpenAIOAuthConfig.from_env(_env(tmp_path))
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {
                "access_token": "access-token-secret",
                "refresh_token": "refresh-token-secret",
                "token_type": "bearer",
                "scope": "openid profile",
                "expires_in": 3600,
            }

    def fake_post(url, *, data, headers, timeout):
        captured["url"] = url
        captured["data"] = dict(data)
        captured["headers"] = dict(headers)
        captured["timeout"] = timeout
        return Response()

    token = exchange_code_for_token(
        config,
        code="code-1",
        code_verifier="verifier-1",
        timeout_seconds=9,
        http_post=fake_post,
    )
    path = save_token_response(config, token)
    status = token_status(config)

    assert captured == {
        "url": "https://auth.example.test/token",
        "data": {
            "grant_type": "authorization_code",
            "client_id": "client-1",
            "client_secret": "secret-1",
            "code": "code-1",
            "redirect_uri": "http://127.0.0.1:8767/oauth/openai/callback",
            "code_verifier": "verifier-1",
        },
        "headers": {"Accept": "application/json"},
        "timeout": 9,
    }
    assert path.exists()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["token_response"]["access_token"] == "access-token-secret"
    assert status["token_present"] is True
    assert status["access_token"] == "acce...cret"
    assert status["refresh_token"] == "refr...cret"
    assert status["expired"] is False
