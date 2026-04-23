# Cognitive OS

A model-agnostic cognitive control plane for AI agents.

Cognitive OS is **not** a new foundation model. It is **not** a replacement for Windows, macOS, or Linux. It is a cognitive control plane that runs on top of an existing host operating system.

## Current status

**Public alpha**.

This repository is published as an alpha-grade entry point: the core boundary model is available, but APIs outside the declared public surface can still evolve.

## Quick start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_runtime_preflight.py --strict-dev
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

The project requires Python 3.10 or newer. If you only need a runtime import
check and not the public smoke test dependencies, run:

```bash
python scripts/check_runtime_preflight.py
```

## Evaluation metrics panel

Saved ARC-AGI-3/WebArena audit JSON can be summarized into the document-level
success metrics panel:

```bash
python scripts/eval_metrics_panel.py runtime reports audit --output runtime/evals/eval_metrics_panel.json
```

The panel reports:

- `verified_success_rate`: successful runs that also carry a passing verifier signal.
- `human_intervention_rate`: runs with teacher, manual, user, or human intervention evidence.
- `recovery_rate`: recovery-attempted runs that resolved or later succeeded.
- `verifier_coverage`: success, completion, or verification-required runs covered by verifier authority or verifier-result evidence.

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
- `scripts/check_runtime_preflight.py`: checks Python version, core imports, entry points, and optional dev test readiness.
- `core/adapter_registry.py`: keeps adapter bindings centralized so adapter imports do not leak into public core paths.

## Runtime layout

The project separates:

- **Source tree**: versioned code and docs in this repository.
- **Runtime tree**: mutable local artifacts under `runtime/`.
- **Eval outputs**: generated outputs under `reports/` and `audit/`.

This separation keeps runtime artifacts from polluting source-layer contracts.

## Execution safety boundary

Execution governance is explicitly **best-effort**, not an OS security sandbox.
The runtime issues policy tickets, records audit events, checks approval and
secret-lease requirements, and now annotates each execution ticket with
`sandbox_audit` data for file paths, write paths, network targets, and
credential-like arguments. These controls are meant to make side effects
visible and reviewable; they do not guarantee kernel-level filesystem or
network isolation.

## Development

- Contributor guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Public boundary guide: [docs/public-boundary.md](docs/public-boundary.md)
- Repository structure map: [docs/repo-map.md](docs/repo-map.md)
