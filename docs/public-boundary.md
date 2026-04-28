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

- the local-machine adapter in `integrations/local_machine/`
- compatibility shims
- selected script utilities

Compatibility shim code being present does **not** mean it is a stable API contract.

## Not part of the public runtime contract

The following are not part of the stable runtime contract:

- private-classified implementation files listed by `core/conos_repository_layout.py`
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

The local mirror VM path is the exception to the local-process best-effort
boundary: it is marked as a real VM boundary only when a real provider is
available. The product default is the Con OS managed-VM provider, which requires
a `conos-managed-vm` helper and manages state under `~/.conos/vm`, including
blank disk artifacts, registered base-image manifests, and per-instance
manifests. The managed helper exposes a boot contract that creates the overlay
artifact and lifecycle commands that write `runtime.json`. A separately built
`conos-vz-runner` owns the real Apple Virtualization process; `start-instance`
only reports `STARTED` after that runner writes a live pid and
`virtual_machine_started=true`. Execution still remains unavailable until the
guest agent marks itself ready. After readiness, Con OS can send `agent-exec`
requests through the instance-local request spool, which the runner forwards to
the guest agent over virtio-vsock. `build-guest-initrd` creates an auditable
guest-agent autostart bundle, and `build-cloud-init-seed` creates a NoCloud
CIDATA seed for EFI cloud images. Cloud-init is only a compatibility path:
registered cloud images expose `cloud_init_guest_capability`, and images without
cloud-init service markers are marked blocked for guest-agent installation
instead of being treated as configured. `start-instance` attaches that seed as a
second read-only virtio block device when an image is registered with
`register-cloud-init-image`, but the public boundary still treats runtime
readiness as unproven until the guest sends the vsock handshake. `build-base-image`
can register a Linux root disk and kernel as a Con OS managed boot image with a
verified initrd guest-agent bundle; if those boot artifacts are absent, it
returns a structured blocked result rather than advertising a fake bundled OS.
`bootstrap-image` is the verification path: it may mark an image verified only
after start, guest-agent readiness, and an `agent-exec` smoke command succeed.
`bundle-base-image` then freezes a verified linux_direct image into a
self-contained directory with disk, kernel, final Con OS initrd, initrd sidecar,
image manifest, and a relative-path digest-pinned recipe. `install-base-image-bundle`
is the inverse public install path: it validates the bundle manifest, all pinned
artifact digests, the relative recipe, and the initrd sidecar before registering
the image under the local VM state root. When direct Linux boot
reaches a live VM process but exposes no early guest signal, bootstrap surfaces
`BOOTSTRAP_GUEST_BOOT_UNOBSERVABLE` plus a boot-path recommendation instead of
promoting the image to verified. Recipe-based artifact acquisition is
allowed only through sha256/sha512-verified cache entries under the managed VM
state root. Recipes can use absolute, file URL, remote, or bundle-relative
artifact paths, but every artifact must match its pinned digest. Built-in
recipes are explicit metadata entries; blocked recipes can be activated only
through `pin-artifact-recipe`, which writes a `READY` recipe after local artifact
hashing or explicit remote digest pinning. Recipes remain non-executable until
their missing artifact or readiness requirements are implemented.
Managed guest-agent execution is gated by `runtime.json` readiness and never
falls back to host execution; advanced
developer paths can still use Lima or SSH. The VM manager can prepare,
checkpoint, restore, and clean the VM workdir, and the VM sync manager can
push/pull the mirror workspace over that real provider. It does not silently
fall back to host execution.

Action governance is part of this boundary: write/exec authority can require
prior evidence, passing validation, an approved source-sync plan, and can be
downgraded after repeated failed actions.

Failure learning is also part of the runtime boundary. Failed actions should be
stored as structured object-layer evidence with retrieval keys and governance or
regression suggestions; raw failure records should not be expanded wholesale
into prompt context.
