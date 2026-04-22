# Cognitive OS

A model-agnostic cognitive control plane for AI agents.

Cognitive OS is not a new foundation model and not a generic chat agent wrapper. It provides task contracts, execution authority, verifier gates, structured runtime state, and cognitive feedback loops that help AI workers become more delegable, auditable, and safer to operate.

## Current status

- **Public alpha**: the repository exposes a public runtime/kernel surface, selected adapters, and selected evaluation utilities.
- **Supported direction**: model-agnostic control-plane development, repository boundary checks, task-contract and verifier-oriented runtime work.
- **Unstable / research-only areas**: research staging directories and compatibility shims may exist in the repository, but they are not stability promises.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

### Optional adapter extras

```bash
pip install -e ".[arc3]"
```

The core repository should still install without optional adapter dependencies. Adapter-specific integrations remain optional.

## What this repo contains

### Supported public surface

- `core/`, `decision/`, `evolution/`, `modules/`, `planner/`, `self_model/`, `state/`, `trace/`  
  Public kernel and runtime-facing control-plane code.
- `integrations/`  
  Adapter layer for supported environments and external surfaces.
- `eval/`, `conos_evals/`, selected `scripts/`  
  Public evaluation utilities and compatibility entry points where available.
- `scripts/check_conos_repo_layout.py`  
  Repository boundary enforcement entry point.
- `tests/test_public_repo_smoke.py`  
  Minimal public smoke coverage for repo-layer and adapter-registry integrity.

### Present in the repository, but not a stability promise

- `private_cognitive_core/`
- `private-cognitive-core/`
- selected research-oriented shims under `scripts/`
- experimental or adapter-specific staging code

These areas may move faster, change shape, or be excluded from runtime-only packaging.

## Repository map

A fuller map lives in [docs/repo-map.md](docs/repo-map.md).

- **Public kernel**: `core/`, `decision/`, `evolution/`, `modules/`, `planner/`, `self_model/`, `state/`, `trace/`
- **Adapters**: `integrations/`
- **Evaluation utilities**: `eval/`, `conos_evals/`, selected `scripts/`
- **Research staging**: `private_cognitive_core/`, `private-cognitive-core/`, selected research shims
- **Runtime artifacts**: `runtime/`, `reports/`, `audit/` (generated state, reports, or local outputs)

## Repository boundary rules

This repository already contains explicit boundary tooling:

- `core/conos_repository_layout.py` classifies repo paths into core / adapter / eval / private / runtime layers.
- `scripts/check_conos_repo_layout.py` enforces that public core code does not drift into adapter/eval/private imports.
- `core/adapter_registry.py` centralizes adapter resolution instead of scattering concrete adapter imports through runtime code.

Run the boundary check before opening a PR:

```bash
python scripts/check_conos_repo_layout.py
```

## Runtime layout

The repository distinguishes three kinds of paths:

- **source tree**: version-controlled code, docs, and tests
- **runtime tree**: local mutable runtime state, logs, caches, and state snapshots
- **eval outputs**: reports, run artifacts, and audit outputs

By default, runtime artifacts are expected under `runtime/` and `runtime/evals/`, not inside the public source tree.

## Development

- Install dev dependencies with `pip install -r requirements-dev.txt`
- Run the repo boundary check
- Run the public smoke test
- Keep adapter-specific code inside `integrations/`
- Keep runtime artifacts out of the source tree

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution rules and [docs/public-boundary.md](docs/public-boundary.md) for the supported public surface.

## Packaging

The public package metadata currently lives in `pyproject.toml`.
A runtime-only package manifest is documented in [PACKAGE_MANIFEST.md](PACKAGE_MANIFEST.md).

## Project positioning

Cognitive OS is best understood as a cognitive control plane that runs on top of an existing operating system. It is not a replacement for Windows, macOS, or Linux, and it is not limited to a single assistant persona. The focus of this repository is the control/runtime layer that helps AI workers execute tasks under stronger contracts, boundaries, and verification.
