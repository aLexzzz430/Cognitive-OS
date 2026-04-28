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
conos auth codex login
conos llm --provider codex --model gpt-5.3-codex runtime-plan
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
  --discover-visible \
  --catalog-only \
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

When a new provider is connected, `conos llm ... profile --discover-visible`
pulls the provider-visible model catalog first, creates catalog-backed model
profiles for every visible model, and emits route policies without requiring a
probe prompt. This works for Ollama, OpenAI-compatible model catalogs, and
Codex CLI ChatGPT OAuth catalogs:

```bash
conos llm --provider codex profile \
  --discover-visible \
  --catalog-only \
  --route-policy-output runtime/models/codex_route_policies.json
```

Local-machine runs can then opt into the same profile-backed router:

```bash
python -m integrations.local_machine.runner \
  --llm-provider all \
  --llm-auto-route-models \
  --llm-route-policy-file ~/.conos/runtime/llm_route_policies.json
```

Each generated route policy now carries a per-route runtime policy that binds
the selected model to its thinking mode, thinking budget, timeout, max response
tokens, and per-tick route budget. Cheap routes such as retrieval and structured
tool kwargs default to no-thinking, while planning and patch design routes get
longer timeouts and stronger-model budgets.

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
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend vm \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace \
  --allow-command python3 \
  -- python3 -c "print('runs in a configured Lima VM')"
conos vm init
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend managed-vm \
  --allow-command python3 \
  -- python3 -c "print('runs in the Con OS managed VM provider')"
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
backend is available as `--backend docker`. The product VM path is
`--backend managed-vm`, which uses the bundled Con OS managed-VM helper when
installed and defaults to explicit `push-pull` workspace sync. Advanced
developer bridges remain available through `--backend vm` with configured Lima
instances (`--vm-provider lima --vm-name <instance>`) or SSH-managed VMs
(`--vm-provider ssh --vm-host <host>` or `CONOS_VM_SSH_HOST`). None of these
paths silently fall back to host execution.

VM-backed mirrors also expose a small lifecycle manager. For the managed
provider, Con OS owns the state root under `~/.conos/vm` and looks for a
`conos-managed-vm` helper plus, for real boot, a clang-built `conos-vz-runner`
Apple Virtualization launcher signed with the
`com.apple.security.virtualization` entitlement. It can
create a Con OS-owned blank disk artifact, register a caller-provided base disk
image into that state root, prepare per-task instance manifests with overlay
paths, and run a helper-level boot contract that stops at
`BOOT_CONTRACT_READY_EXEC_UNAVAILABLE` until a guest agent exists. `start`
prefers the bundled Apple Virtualization runner when it has been built; that
runner is a long-lived process that owns the `VZVirtualMachine` object and writes
its live pid into `runtime.json` only after the Virtualization start callback
succeeds. Linux direct-boot images can be registered with a disk, kernel,
optional initrd, kernel command line, and declared virtio-vsock guest-agent
port; the runner maps these manifests to `VZLinuxBootLoader` and a virtio socket
device. The host runner listens on that port for the bundled guest agent's JSON
ready handshake and records guest-agent protocol/capability state in
`runtime.json`. Once the guest agent is ready, `agent-exec` can use the runner's
Con OS-owned request spool under the instance directory to send bounded exec
requests over virtio-vsock and wait for audited result JSON; it still never
falls back to host execution. `conos vm build-guest-initrd` creates a Con OS
guest-agent initrd bundle containing the guest agent, systemd unit, installer
script, and optional init wrapper; image registration records this as
`guest_agent_autostart_configured`, while actual execution readiness remains
strictly gated by the runtime handshake. For EFI cloud images,
`conos vm build-cloud-init-seed` creates a NoCloud CIDATA seed disk and
`conos vm register-cloud-init-image` marks the image so `start-instance`
generates an instance-specific seed and attaches it as a second read-only
virtio block device. `conos vm build-base-image` wraps this
into a managed Linux base-image builder: when given a Linux root disk and kernel
artifact, it creates the guest initrd bundle and registers a `linux_direct`
image; without those boot artifacts it returns
`BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS` instead of claiming a bootable OS exists.
`conos vm bootstrap-image` is the product-level path that builds the image,
starts a smoke instance, waits for `guest_agent_ready=true`, runs
`agent-exec -- echo ok`, and marks the image verified only after that check
passes. If a `linux_direct` VM process starts but produces no guest console
output or initramfs trace marker, bootstrap now reports
`BOOTSTRAP_GUEST_BOOT_UNOBSERVABLE` with a boot-path recommendation instead of
treating the failure as a generic guest-agent timeout; the usual next path is
an EFI/cloud-init image or a known-good direct Linux kernel/initrd set that
emits `console=hvc0`. It can also resolve a trusted artifact recipe with `--recipe-path`;
each recipe artifact is cached under the Con OS VM state root and must pass
sha256/sha512 digest verification before being used. Built-in recipes are
listed with `conos vm recipe-report`; the current
`builtin:debian-nocloud-arm64` candidate is intentionally blocked until it pins
a concrete source disk artifact and digest. `conos vm pin-artifact-recipe`
turns a blocked recipe into a `READY` recipe only after its source disk is
digest-pinned; local files are hashed before the recipe is written, and remote
URLs without an explicit digest are refused.
EFI disk images use a per-instance EFI variable store. Without a built
runner, start still returns the explicit
`START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING` contract instead of pretending
to boot. Managed `agent-exec` is additionally gated on a live VM process plus
`guest_agent_ready=true` and `execution_ready=true`; otherwise it returns a
structured refusal instead of falling back to the host. For
advanced external providers, the same lifecycle commands can target Lima or SSH
VMs. The manager can preflight the boundary, prepare the VM workdir, create an
in-VM checkpoint, restore that checkpoint, or clean the workdir while preserving
checkpoints. If the VM workdir is not mounted, use `vm-sync` or
`--vm-sync-mode push-pull` to explicitly copy the mirror workspace into the VM
and copy results back:

```bash
conos vm report
conos vm init
conos vm build-helper
conos vm build-runner
conos vm build-guest-initrd --output-path runtime/conos-guest-agent-initrd.img
conos vm build-cloud-init-seed --instance-id session-1 --output-path runtime/cloud-init-seed.img
conos vm build-base-image --image-id linux-base --source-disk /path/to/rootfs.img --kernel-path /path/to/vmlinuz --base-initrd-path /path/to/initrd.img
conos vm bootstrap-image --image-id linux-base --source-disk /path/to/rootfs.img --kernel-path /path/to/vmlinuz --base-initrd-path /path/to/initrd.img
conos vm bootstrap-image --image-id linux-base --recipe-path /path/to/conos-image-recipe.json
conos vm recipe-report
conos vm pin-artifact-recipe --base-recipe builtin:debian-nocloud-arm64 --source-disk /path/to/cloud.img
conos vm resolve-artifact-recipe --recipe-path builtin:debian-nocloud-arm64
conos vm resolve-artifact-recipe --recipe-path /path/to/conos-image-recipe.json
conos vm create-blank-image --image-id conos-base --size-mb 8192
conos vm register-image --image-id conos-base --source-disk /path/to/disk.img
conos vm register-cloud-init-image --image-id cloud-base --source-disk /path/to/cloud.img
conos vm register-linux-boot-image --image-id linux-base --source-disk /path/to/rootfs.img --kernel-path /path/to/vmlinuz --initrd-path runtime/conos-guest-agent-initrd.img
conos vm prepare-instance --image-id conos-base --instance-id session-1
conos vm boot-instance --image-id conos-base --instance-id session-1
conos vm start-instance --image-id conos-base --instance-id session-1
conos vm runtime-status --image-id conos-base --instance-id session-1
conos vm agent-status --image-id conos-base --instance-id session-1
conos vm agent-exec --image-id conos-base --instance-id session-1 -- python3 -m pytest -q
conos vm stop-instance --image-id conos-base --instance-id session-1
conos mirror boundary --backend managed-vm
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend managed-vm \
  --allow-command python3 \
  -- python3 -m pytest -q
conos mirror vm --operation report --vm-provider lima --vm-name conos-vm
conos mirror vm \
  --mirror-root runtime/mirrors/session-1 \
  --operation prepare \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace
conos mirror vm \
  --mirror-root runtime/mirrors/session-1 \
  --operation checkpoint \
  --checkpoint-id before-risky-run \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace
conos mirror vm \
  --mirror-root runtime/mirrors/session-1 \
  --operation restore \
  --checkpoint-id before-risky-run \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace
conos mirror vm-sync \
  --mirror-root runtime/mirrors/session-1 \
  --direction push \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace
conos mirror exec \
  --mirror-root runtime/mirrors/session-1 \
  --backend vm \
  --vm-provider lima \
  --vm-name conos-vm \
  --vm-workdir /workspace \
  --vm-sync-mode push-pull \
  --allow-command python3 \
  -- python3 -m pytest -q
```

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
