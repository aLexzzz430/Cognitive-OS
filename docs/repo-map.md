# Repository Map

This document describes the public-facing repository structure for Cognitive OS.

## Public kernel

Primary control-plane code:

- `core/`
- `decision/`
- `evolution/`
- `memory/`
- `modules/`
- `modules/control_plane/`
- `planner/`
- `self_model/`
- `state/`
- `trace/`

Product-facing public entry points:

- `conos_cli.py`
- `scripts/conos.py`
- `core/auth/`

## Adapters

Environment and surface adapters:

- `integrations/local_machine/`

`integrations/` is the adapter layer, not the kernel itself. The distilled
runtime currently keeps only the local-machine adapter.

The local-machine adapter is backed by `modules/local_mirror/`. The mirror
runtime is public core plumbing: it starts with an empty workspace, materializes
files on demand, and writes source files only through a reviewed sync plan.
`modules/local_mirror/managed_vm.py` defines the built-in Con OS managed-VM
provider contract, state root, blank disk artifacts, base-image manifests, and
per-instance manifests, including the managed helper boot/lifecycle contract
wrappers, Apple Virtualization runner launcher, guest-agent initrd bundle
builder, cloud-init NoCloud seed builder, sha256/sha512-verified artifact
recipe/cache resolver, digest-pinned recipe activation, built-in recipe registry,
managed Linux base-image builder, verified base-image bundle exporter/importer,
bootstrap verification flow, runtime manifest, request-spool bridge, and guest-agent
execution gate.
`modules/local_mirror/vm_backend.py`,
`modules/local_mirror/vm_manager.py`, and
`modules/local_mirror/vm_workspace_sync.py` hold the real-VM execution,
workspace lifecycle, and explicit workspace sync bridge for the managed helper
plus advanced Lima or SSH VMs; local execution remains explicitly best-effort.

Action authority is governed by `modules/control_plane/action_governance.py`.
This policy layer decides whether an agent may read, propose a patch, write the
mirror, run validation, or sync source changes. It is deliberately separate from
tool routing so "tool exists" does not mean "agent is authorized to use it."

Failure learning is handled by `core/runtime/failure_learning.py` plus the
runtime SQLite store. It keeps failed actions as object-layer evidence with
violated assumptions, regression-test suggestions, governance-rule suggestions,
and future retrieval keys.

## Public checks

Check utilities that can be run publicly:

- `scripts/check_conos_repo_layout.py`
- `scripts/check_runtime_preflight.py`

## Research / private staging

Research and staging zones that may change without compatibility promises:

- `core/orchestration/structured_answer.py`
- `modules/hypothesis/mechanism_posterior_updater.py`

## Runtime artifacts

Generated outputs and mutable runtime data:

- `runtime/`
- `reports/`
- `audit/`

`runtime/`, `reports/`, and `audit/` are artifact/output layers, not source-code layers.
