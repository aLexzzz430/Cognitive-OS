# Cognitive OS

A model-agnostic cognitive control plane for AI agents.

Cognitive OS is **not** a new foundation model. It is **not** a replacement for Windows, macOS, or Linux. It is a cognitive control plane that runs on top of an existing host operating system.

## Current status

**Public alpha**.

This repository is published as an alpha-grade entry point: the core boundary model is available, but APIs outside the declared public surface can still evolve.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

## What this repo contains

### Stable-enough public surface

- `core/`, `decision/`, `evolution/`, `modules/`, `planner/`, `self_model/`
- repository boundary tooling and checks

### Public but not stability promise

- `integrations/` adapter implementations
- selected scripts and evaluation helpers
- compatibility shims that may change during alpha

### Not part of the public runtime contract

- `private_cognitive_core/` or `private-cognitive-core/` research/staging areas
- generated runtime outputs under `runtime/`, `reports/`, `audit/`

## Repository boundary rules

The boundary model is enforced through three key files:

- `core/conos_repository_layout.py`: declares layer classification rules (core, adapter, eval, private, runtime).
- `scripts/check_conos_repo_layout.py`: runs repository-level enforcement checks for boundary violations.
- `core/adapter_registry.py`: keeps adapter bindings centralized so adapter imports do not leak into public core paths.

## Runtime layout

The project separates:

- **Source tree**: versioned code and docs in this repository.
- **Runtime tree**: mutable local artifacts under `runtime/`.
- **Eval outputs**: generated outputs under `reports/` and `audit/`.

This separation keeps runtime artifacts from polluting source-layer contracts.

## Development

- Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Public boundary guide: [docs/public-boundary.md](docs/public-boundary.md)
- Repository structure map: [docs/repo-map.md](docs/repo-map.md)
