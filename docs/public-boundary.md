# Public Boundary

## Stable-enough public surface

The alpha public surface is centered on core control-plane layers and boundary tooling:

- `core/`, `decision/`, `evolution/`, `modules/`, `planner/`, `self_model/`
- `modules/control_plane/` for model-agnostic agent capability routing and action authority governance
- `core/conos_repository_layout.py`
- `scripts/check_conos_repo_layout.py`
- `core/adapter_registry.py`

These areas are the baseline for external understanding and contribution.

## Public but not stability promise

Some areas are visible in the public repository but can evolve during alpha:

- adapter implementations in `integrations/`
- compatibility shims
- selected evaluation and script utilities

Compatibility shim code being present does **not** mean it is a stable API contract.

## Not part of the public runtime contract

The following are not part of the stable runtime contract:

- `private_cognitive_core/` and `private-cognitive-core/` (research/staging)
- local runtime outputs under `runtime/`, `reports/`, `audit/`
- adapter-specific details that should not back-propagate into public core imports

Adapter-specific code must remain in adapter layers and cannot reverse-pollute the public core.

## Execution Safety Boundary

Execution policy is a best-effort audit and governance layer. It is not a
claim of OS-level sandboxing. The public contract is that execution tickets and
policy-block audit events expose the declared boundary, approval state,
secret-lease state, file/path signals, write-path signals, and network target
signals. Callers that require hard isolation must provide it outside this
runtime.

Action governance is part of this boundary: write/exec authority can require
prior evidence, passing validation, an approved source-sync plan, and can be
downgraded after repeated failed actions.

Failure learning is also part of the runtime boundary. Failed actions should be
stored as structured object-layer evidence with retrieval keys and governance or
regression suggestions; raw failure records should not be expanded wholesale
into prompt context.
