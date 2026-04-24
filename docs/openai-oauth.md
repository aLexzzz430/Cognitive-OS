# OpenAI OAuth Login

Cognitive OS exposes an OpenAI-compatible OAuth entrypoint through:

```bash
conos auth openai status
conos auth openai login
conos auth openai logout
```

The implementation uses Authorization Code + PKCE and a local callback server.
Tokens are stored locally under `runtime/auth/openai_oauth_token.json` with file
mode `0600` when the host operating system allows it.

## Important Boundary

OpenAI API calls are authenticated with API keys. This OAuth entrypoint is for
OpenAI-compatible OAuth providers and ChatGPT/OpenAI integration flows where a
registered OAuth client, authorization URL, and token URL are available. Cognitive
OS does not hard-code or pretend there is a public "Sign in with OpenAI" identity
provider endpoint.

## Environment

Required:

- `OPENAI_OAUTH_CLIENT_ID`
- `OPENAI_OAUTH_AUTHORIZATION_URL`
- `OPENAI_OAUTH_TOKEN_URL`

Optional:

- `OPENAI_OAUTH_CLIENT_SECRET`
- `OPENAI_OAUTH_SCOPES` defaults to `openid profile email`
- `OPENAI_OAUTH_REDIRECT_HOST` defaults to `127.0.0.1`
- `OPENAI_OAUTH_REDIRECT_PORT` defaults to `8767`
- `OPENAI_OAUTH_REDIRECT_PATH` defaults to `/oauth/openai/callback`
- `OPENAI_OAUTH_TOKEN_STORE` defaults to `runtime/auth/openai_oauth_token.json`

The redirect URI registered with the OAuth provider must match:

```text
http://127.0.0.1:8767/oauth/openai/callback
```

or the customized host, port, and path from the environment.
