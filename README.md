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
conos preflight --strict-dev
conos layout
pytest -q
```

The project requires Python 3.10 or newer. If you only need a runtime import
check and not the public smoke test dependencies, run:

```bash
conos preflight
```

The legacy script entry points remain available:

```bash
python scripts/check_runtime_preflight.py --strict-dev
python scripts/check_conos_repo_layout.py
pytest -q tests/test_public_repo_smoke.py
```

## Product entrypoint

`conos` is the unified product-facing command:

```bash
conos run arc-agi3 --game vc33 --max-ticks 1 --save-audit runtime/evals/vc33.json
conos run local-machine --instruction "inspect README" --candidate README.md --max-ticks 2
conos run webarena --task-id smoke --instruction "open the target page" --max-ticks 1
conos app
conos app --summary-json
conos mirror init --mirror-root runtime/mirrors/session-1
conos mirror acquire --mirror-root runtime/mirrors/session-1 --instruction "inspect README" --candidate README.md
conos ui runtime reports audit
conos ui runtime reports audit --output runtime/ui/dashboard.html
conos eval runtime reports audit --output runtime/evals/eval_metrics_panel.json
conos dashboard runtime reports audit
conos llm control-plane --route structured_answer --required-capability structured_output --permission generate_text
conos preflight --strict-dev
conos layout
```

## Model-agnostic agent control plane

Con OS treats models and agents as governed execution resources, not as one
hard-coded coding assistant. Profile-backed LLM route policies can be combined
with a generic agent registry for coding agents, browsers, CI runners, VM
backends, or other tool executors:

```bash
conos llm profile \
  --base-url http://127.0.0.1:11434 \
  --route-policy-output runtime/models/llm_route_policies.json
conos llm control-plane \
  --route-policy-file runtime/models/llm_route_policies.json \
  --agent-registry runtime/models/agent_registry.json \
  --task-type code_review \
  --route coding \
  --required-capability coding \
  --permission propose_patch \
  --risk-level medium
```

The control decision records selected agent id, capability match, permission
gate, approval-required permissions, blocked candidates, and an audit event id.

## Action capability governance

Con OS separates tool availability from action authority. `modules/control_plane/action_governance.py`
checks whether the selected agent may read, propose a patch, write the mirror,
run validation, or sync back to source. The local-machine adapter now enforces
that mirror writes have evidence references first, code syncs have passing
validation first, source syncs carry an approved plan, and repeatedly failing
agents are downgraded away from write/exec authority.

## Failure learning ledger

Failures are stored as structured runtime objects, not chat memory. `core/runtime/failure_learning.py`
normalizes failed actions into failure mode, violated assumption, evidence refs,
missing tool or bad policy, suggested regression test, suggested governance
rule, and a future retrieval object. These objects are persisted in SQLite by
`RuntimeStateStore` and are injected into later unified context as object-layer
evidence; the local-machine runner exposes them as JSON artifacts instead of
dumping raw failure objects into the prompt.

## Desktop app

The local desktop app is the non-terminal product shell for this alpha. It
summarizes runtime/evaluation health, runs strict preflight, exports the HTML
dashboard, and can open the live dashboard service:

```bash
conos app
```

Headless app checks are available for automation:

```bash
conos app --summary-json
conos app --write-dashboard runtime/ui/dashboard.html
conos app --preflight
```

On macOS, build a lightweight `.app` launcher:

```bash
python scripts/build_macos_app.py --output-dir dist
open "dist/Cognitive OS.app"
```

## Empty-first local mirror

The local mirror runtime starts with an empty data workspace. It does not copy
the source tree up front. Files are materialized only after an explicit fetch
request, or after an instruction-scoped acquisition selects them from supplied
candidate paths:

```bash
conos mirror init --mirror-root runtime/mirrors/session-1
conos mirror fetch --mirror-root runtime/mirrors/session-1 --path README.md
conos mirror acquire \
  --mirror-root runtime/mirrors/session-1 \
  --instruction "inspect pyproject before changing dependencies" \
  --candidate README.md \
  --candidate pyproject.toml
```

The mirror keeps control metadata under `control/` and user-materialized files
under `workspace/`, so the workspace can be audited independently from runtime
metadata.

The execution/sync path is also gated. Commands run from the mirror workspace,
diffs are converted into a patch-gated sync plan, and source files are updated
only when an approved plan id is supplied:

```bash
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --allow-command python3 \
  -- python3 -c "from pathlib import Path; print(Path('pyproject.toml').exists())"
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend docker \
  --allow-command python3 \
  -- python3 -c "print('runs in docker with network disabled')"
conos mirror plan --mirror-root runtime/mirrors/session-1
conos mirror apply \
  --mirror-root runtime/mirrors/session-1 \
  --plan-id <plan_id> \
  --approved-by machine
conos mirror rollback \
  --mirror-root runtime/mirrors/session-1 \
  --plan-id <plan_id>
```

Machine approval is limited to added/modified text-like files with clean mirror
command results. Failed commands, deletions, unsupported file types, or missing
text patches require human review. Apply is a patch gate: it verifies the
current source hash and mirror hash from the plan, applies the unified text
patch, writes a rollback checkpoint with original sha256 plus reverse patch, and
records the checkpoint/apply event in the mirror audit log. The Docker execution
backend is available as `--backend docker`; VM execution is declared as a future
backend and is rejected until a VM provider is wired.

The same mirror can also be used as a CoreMainLoop environment adapter:

```bash
conos run local-machine \
  --instruction "inspect README before planning changes" \
  --candidate README.md \
  --max-ticks 2
```

In this mode the first observation still has zero user files. The system can
only see files after a mirror acquisition/fetch action, and source-tree writes
remain behind the sync-plan approval gate.

## Evaluation metrics panel

Saved ARC-AGI-3/WebArena audit JSON can be summarized into the document-level
success metrics panel:

```bash
conos eval runtime reports audit --output runtime/evals/eval_metrics_panel.json
```

The panel reports:

- `verified_success_rate`: successful runs that also carry a passing verifier signal.
- `human_intervention_rate`: runs with teacher, manual, user, or human intervention evidence.
- `recovery_rate`: recovery-attempted runs that resolved or later succeeded.
- `verifier_coverage`: success, completion, or verification-required runs covered by verifier authority or verifier-result evidence.

## Cognitive loop anti-cheat benchmark

The controlled cognitive-loop benchmark is a negative-control suite, not a
success demo. It compares the full loop against component ablations and
anti-cheat controls:

```bash
python scripts/cognitive_loop_ablation.py --task-count 25 --seed 17
```

Core controls include `NoPosteriorUpdate`, `NoDiscriminatingExperiment`,
`NoHypothesisCompetition`, `NoSemanticStrictness`, `RandomProbe`, and
`BaselineLLM`. Anti-cheat controls include `NoMemory`, `NoEvidence`, `NoStep9`,
`NoStep10`, `ShuffledEvidence`, `WrongBinding`, and `FreshBaseline`.

The report fails unless `Full` clears the hard thresholds for success margin,
wrong commits, false rejections, posterior accuracy, semantic mismatch, evidence
binding integrity, and formal commit completeness.

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
