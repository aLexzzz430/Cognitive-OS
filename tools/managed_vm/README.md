# Con OS Managed VM Helper Contract

`managed-vm` is the product-facing VM provider. Users should not have to
configure Lima, SSH, or Docker for this path. The Python runtime owns policy,
state directories, workspace sync, audit, and patch gates; a platform helper
owns the final call into the host virtualization API.

The helper binary is named `conos-managed-vm` and is discovered in this order:

1. `CONOS_MANAGED_VM_HELPER`
2. `bin/conos-managed-vm`
3. `tools/managed_vm/conos-managed-vm`
4. `PATH`

Build the bundled macOS helper and Apple Virtualization runner with:

```bash
conos vm build-helper
conos vm build-runner
conos vm build-guest-initrd --output-path /path/to/conos-guest-agent-initrd.img
conos vm build-cloud-init-seed --instance-id task-001 --output-path /path/to/cloud-init-seed.img
conos vm build-base-image --image-id linux-base --source-disk /path/to/rootfs.img --kernel-path /path/to/vmlinuz --base-initrd-path /path/to/initrd.img
conos vm bootstrap-image --image-id linux-base --source-disk /path/to/rootfs.img --kernel-path /path/to/vmlinuz --base-initrd-path /path/to/initrd.img
conos vm bootstrap-image --image-id linux-base --recipe-path /path/to/conos-image-recipe.json
conos vm recipe-report
conos vm pin-artifact-recipe --base-recipe builtin:debian-nocloud-arm64 --source-disk /path/to/cloud.img
conos vm resolve-artifact-recipe --recipe-path builtin:debian-nocloud-arm64
conos vm resolve-artifact-recipe --recipe-path /path/to/conos-image-recipe.json
```

Register a base disk image and prepare an instance manifest with:

```bash
conos vm create-blank-image --image-id conos-base --size-mb 8192
conos vm register-image --image-id conos-base --source-disk /path/to/disk.img
conos vm register-cloud-init-image --image-id cloud-base --source-disk /path/to/cloud.img
conos vm register-linux-boot-image \
  --image-id linux-base \
  --source-disk /path/to/rootfs.img \
  --kernel-path /path/to/vmlinuz \
  --initrd-path /path/to/conos-guest-agent-initrd.img
conos vm prepare-instance --image-id conos-base --instance-id task-001
conos vm boot-instance --image-id conos-base --instance-id task-001
conos vm start-instance --image-id conos-base --instance-id task-001
conos vm runtime-status --image-id conos-base --instance-id task-001
conos vm agent-status --image-id conos-base --instance-id task-001
conos vm agent-exec --image-id conos-base --instance-id task-001 -- python3 -m pytest -q
conos vm stop-instance --image-id conos-base --instance-id task-001
conos vm image-report --image-id conos-base
conos vm instance-report --instance-id task-001
```

For direct development checks:

```bash
python scripts/build_managed_vm_helper.py --output-path /tmp/conos-managed-vm
/tmp/conos-managed-vm report
```

Minimum helper contract:

```bash
conos-managed-vm exec \
  --state-root ~/.conos/vm \
  --instance-id default \
  --image-id conos-base \
  --network-mode provider_default \
  -- \
  bash -lc "cd /workspace && exec python3 -m pytest -q"
```

The helper must:

- create or reuse a real VM instance for `instance-id`
- use a Con OS managed base image plus per-task overlay disk
- keep host directories unmounted unless explicitly requested by the Python sync layer
- preserve stdin/stdout byte streams for workspace tar push/pull
- return the VM process exit code
- never fall back to host process execution

Current v0.1 helper status:

- builds on macOS without requiring Foundation or a preconfigured external provider
- reports whether Apple Virtualization.framework is present
- initializes Con OS VM state directories
- supports product-side blank disk creation, base image registration, and instance manifests
- marks blank disks as not bootable and not execution-ready until an OS and guest agent are installed
- supports Linux direct-boot image registration with disk, kernel, optional
  initrd, kernel command line, and a declared virtio-vsock guest-agent port.
- supports a `boot` contract that validates the host Virtualization.framework,
  verifies the base image exists, prepares instance directories, creates the
  overlay artifact, and reports `BOOT_CONTRACT_READY_EXEC_UNAVAILABLE`
- supports `start`, `runtime-status`, and `stop` lifecycle contracts with a
  `runtime.json` manifest, but reports
  `START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING` until the helper owns a real
  Apple Virtualization process and guest agent
- adds `conos-vz-runner`, a separate Apple Virtualization.framework process
  launcher built with clang/Objective-C so it does not depend on Swift module
  compatibility. `conos vm build-runner` signs the binary with the
  `com.apple.security.virtualization` entitlement by default. When
  `start-instance` can find that binary, Python launches it as a long-lived
  child, waits for its
  Virtualization start callback, and only reports `STARTED` when `runtime.json`
  contains `virtual_machine_started=true` plus a live runner pid. Until a guest
  agent marks itself ready, `agent-exec` still remains blocked.
- configures a `VZLinuxBootLoader` when the image manifest declares
  `boot_mode=linux_direct`, otherwise uses EFI disk boot with a per-instance EFI
  variable store.
- installs a host-side virtio-vsock listener on the declared guest-agent port.
  The bundled guest agent at `tools/managed_vm/guest_agent/conos_guest_agent.py`
  sends a JSON `guest_agent_ready` handshake, after which `runtime.json` records
  `guest_agent_ready=true`, `execution_ready=true`, protocol version, and
  advertised capabilities.
- builds a Con OS guest-agent initrd bundle with `conos vm build-guest-initrd`.
  The bundle contains the guest agent, installer script, systemd unit, and
  optional init wrapper. Registering a Linux boot image with that initrd records
  `guest_agent_autostart_configured=true` only when the bundle sidecar validates
  the required guest-agent files and init/local-bottom activation path. Runtime
  readiness still requires the actual vsock handshake.
- builds a cloud-init NoCloud CIDATA seed disk with
  `conos vm build-cloud-init-seed`. The seed is a Con OS-owned VFAT artifact
  containing `user-data`, `meta-data`, and `network-config`; `user-data`
  installs and starts the bundled guest agent.
- registers EFI cloud images with `conos vm register-cloud-init-image`. On
  `start-instance`, Con OS creates an instance-specific NoCloud seed and the
  Apple Virtualization runner attaches it as a second read-only virtio block
  device with `--cloud-init-seed`. Cloud-init is a compatibility path: if raw
  disk preflight cannot find cloud-init service markers, Con OS records
  `guest_agent_installation_status=BLOCKED_CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE`
  and recommends the verified initrd or preinstalled-agent path instead.
- builds and registers a managed Linux base image with `conos vm build-base-image`
  when a root disk and Linux kernel are available. If those boot artifacts are
  missing, the command returns `BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS` with the
  missing inputs instead of pretending Con OS has a bootable OS.
- verifies that image with `conos vm bootstrap-image`: build/register, start a
  smoke instance, wait for guest-agent readiness, run `agent-exec -- echo ok`,
  and mark the image verified only if the smoke passes.
- packages a verified linux_direct image with `conos vm bundle-base-image`.
  The bundle is a self-contained directory containing disk, kernel, final Con
  OS initrd, the initrd sidecar, the image manifest, and a relative-path
  digest-pinned recipe. By default it refuses unverified images; pass
  `--allow-unverified` only for development bundles.
- installs a bundled image with `conos vm install-base-image-bundle
  --bundle-dir <bundle>`. The installer validates the bundle manifest, every
  artifact sha256, the relative recipe, and the initrd sidecar before copying
  disk/kernel/initrd into the local Con OS VM state root. `import-base-image-bundle`
  is an alias for the same path.
- reports built-in image recipes with `recipe-report`. The current
  `builtin:debian-nocloud-arm64` recipe is intentionally blocked until Con OS
  pins a concrete source disk artifact and digest.
- creates an enabled recipe with `pin-artifact-recipe`. Local artifact paths are
  hashed and embedded as digest-pinned file URLs; remote URLs must provide
  `--source-disk-sha256` or `--source-disk-sha512` and are refused otherwise.
- resolves recipe-defined boot artifacts with `resolve-artifact-recipe` or
  `bootstrap-image --recipe-path`; every artifact is cached under the Con OS VM
  state root and must match the recipe sha256/sha512 digest before use. Bundle
  recipes may use relative artifact paths, which are resolved relative to the
  recipe file location.
- supports `agent-status` and `agent-exec` contracts, but the Python runtime
  blocks `agent-exec` unless `runtime.json` proves a live VM process plus
  `guest_agent_ready=true` and `execution_ready=true`
- when no legacy helper is configured and the Apple runner has a ready guest
  agent connection, `agent-exec` writes bounded request JSON under
  `instances/<id>/agent-requests/`; the runner forwards those requests over
  virtio-vsock and writes audited result JSON back to the same directory
- refuses `exec` until a managed base image and guest execution path exist
- never falls back to host process execution

The Python layer marks `managed-vm` as unavailable until this helper exists.
