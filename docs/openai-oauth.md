# OpenAI OAuth Login

Cognitive OS exposes two distinct authentication paths:

1. OpenAI-compatible OAuth for integrations where you explicitly provide OAuth
   client/provider endpoints.
2. Codex CLI delegated ChatGPT OAuth for using a locally authenticated Codex
   account and its Codex quota.

## Codex CLI ChatGPT Login

For ChatGPT/Codex quota, Cognitive OS does not read or store ChatGPT OAuth
tokens. It delegates login to the official Codex CLI:

```bash
conos auth codex status
conos auth codex login
conos auth codex logout
```

After login, select the Codex-backed LLM runtime with:

```bash
conos llm --provider codex --model gpt-5.3-codex check
conos llm --provider codex --model gpt-5.3-codex runtime-plan
```

To catalog every Codex model visible to the logged-in ChatGPT account and build
route policies without live probe prompts:

```bash
conos llm --provider codex profile \
  --discover-visible \
  --catalog-only \
  --route-policy-output runtime/models/codex_route_policies.json
```

The runtime contract reports:

- `Provider`: `codex-cli`
- `AuthProfile`: `chatgpt_oauth_delegate`
- `ExecutionRuntime`: `local_cli_agent`
- `ToolAdapter`: `codex_exec`
- `CostPolicy`, `ContextPolicy`, and `VerifierPolicy` as separate objects

The quota boundary is `chatgpt_codex_plan_or_api_org_via_codex_cli`; exact
usage and rate limits remain managed by the Codex CLI/OpenAI account, not by a
Con OS token store.

## Generic OpenAI-Compatible OAuth

The OpenAI-compatible OAuth entrypoint remains:

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
