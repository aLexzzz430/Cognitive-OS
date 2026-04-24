# Repository Map

This document describes the public-facing repository structure for Cognitive OS.

## Public kernel

Primary control-plane code:

- `core/`
- `decision/`
- `evolution/`
- `memory/`
- `modules/`
- `planner/`
- `self_model/`
- `state/`
- `trace/`

Product-facing public entry points:

- `conos_cli.py`
- `scripts/conos.py`
- `core/auth/`
- `core/app/`
- `core/evaluation/dashboard_app.py`

## Adapters

Environment and surface adapters:

- `integrations/arc_agi3/`
- `integrations/local_machine/`
- `integrations/survival_world/`
- `integrations/webarena/`

`integrations/` is the adapter layer, not the kernel itself.

The local-machine adapter is backed by `modules/local_mirror/`. The mirror
runtime is public core plumbing: it starts with an empty workspace, materializes
files on demand, and writes source files only through a reviewed sync plan.

## Public eval utilities

Evaluation and check utilities that can be run publicly:

- `scripts/check_conos_repo_layout.py`
- `scripts/check_runtime_preflight.py`
- selected evaluation scripts under `scripts/`

## Research / private staging

Research and staging zones that may change without compatibility promises:

- `private_cognitive_core/`
- `private-cognitive-core/`
- `core/orchestration/structured_answer.py`
- `modules/hypothesis/mechanism_posterior_updater.py`
- selected staging-oriented scripts

## Runtime artifacts

Generated outputs and mutable runtime data:

- `runtime/`
- `reports/`
- `audit/`

`runtime/`, `reports/`, and `audit/` are artifact/output layers, not source-code layers.
