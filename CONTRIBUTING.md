# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

## Run minimal checks

```bash
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

## Contribution rules

- Place adapter-specific code in `integrations/`.
- Do not commit runtime artifacts into the source tree (`runtime/`, `reports/`, `audit/`).
- Public core layers must not directly import eval/scripts/integrations/private areas.

## Pull request requirements

Each PR should include:

- the purpose of the change;
- whether it affects the public surface;
- whether it introduces a new adapter dependency.
