# Repository Map

This document describes the public-facing repository structure for Cognitive OS.

## Public kernel

Primary control-plane code:

- `core/`
- `decision/`
- `evolution/`
- `modules/`
- `planner/`
- `self_model/`

## Adapters

Environment and surface adapters:

- `integrations/arc_agi3/`
- `integrations/webarena/`

`integrations/` is the adapter layer, not the kernel itself.

## Public eval utilities

Evaluation and check utilities that can be run publicly:

- `scripts/check_conos_repo_layout.py`
- selected evaluation scripts under `scripts/`

## Research / private staging

Research and staging zones that may change without compatibility promises:

- `private_cognitive_core/`
- `private-cognitive-core/`
- selected staging-oriented scripts

## Runtime artifacts

Generated outputs and mutable runtime data:

- `runtime/`
- `reports/`
- `audit/`

`runtime/`, `reports/`, and `audit/` are artifact/output layers, not source-code layers.
