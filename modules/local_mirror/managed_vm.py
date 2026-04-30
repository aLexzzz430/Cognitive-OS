"""Con OS managed VM provider contract.

This module owns the product-level VM provider surface. It does not emulate a
VM in Python and it does not require users to preconfigure Lima or SSH. Instead
it locates a Con OS managed-VM helper that is responsible for talking to the
host virtualization API and for managing images, instances, overlays, and
snapshots under a Con OS state root.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from core.runtime.recovery_guidance import guidance_for_vm_report


MANAGED_VM_PROVIDER_VERSION = "conos.managed_vm_provider/v1"
DEFAULT_MANAGED_VM_STATE_ROOT = "~/.conos/vm"
DEFAULT_MANAGED_VM_IMAGE_ID = "conos-base"
DEFAULT_MANAGED_VM_INSTANCE_ID = "default"
DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT = 48080
MANAGED_VM_HELPER_NAME = "conos-managed-vm"
MANAGED_VM_HELPER_SOURCE = "tools/managed_vm/macos/conos-managed-vm.swift"
MANAGED_VM_RUNNER_NAME = "conos-vz-runner"
MANAGED_VM_RUNNER_SOURCE = "tools/managed_vm/macos/conos-vz-runner.m"
MANAGED_VM_RUNNER_ENTITLEMENTS = "tools/managed_vm/macos/conos-vz-runner.entitlements.plist"
MANAGED_VM_GUEST_AGENT_SOURCE = "tools/managed_vm/guest_agent/conos_guest_agent.py"
MANAGED_VM_RECIPE_DIR = "tools/managed_vm/recipes"
MANAGED_VM_RECIPE_REGISTRY = "tools/managed_vm/recipes/registry.json"
DEFAULT_MANAGED_VM_RECIPE_REFERENCE = "builtin:debian-genericcloud-arm64"
MANAGED_VM_BASE_IMAGE_MANIFEST = "image.json"
MANAGED_VM_INSTANCE_MANIFEST = "instance.json"
MANAGED_VM_RUNTIME_MANIFEST = "runtime.json"
MANAGED_VM_AGENT_REQUESTS_DIR = "agent-requests"
MANAGED_VM_GUEST_BUNDLE_MANIFEST = "guest-initrd.manifest.json"
MANAGED_VM_CLOUD_INIT_SEED_NAME = "cloud-init-seed.img"
MANAGED_VM_CLOUD_INIT_SEED_MANIFEST = "cloud-init-seed.manifest.json"
MANAGED_VM_ARTIFACT_RECIPE_VERSION = "conos.managed_vm_artifact_recipe/v1"
MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION = "conos.managed_vm_base_image_bundle/v1"
MANAGED_VM_BASE_IMAGE_BUNDLE_MANIFEST = "conos-base-image.manifest.json"
MANAGED_VM_BASE_IMAGE_BUNDLE_RECIPE = "conos-base-image.recipe.json"
MANAGED_VM_SHARED_DIR_TAG = "conos_host"
DEFAULT_MANAGED_VM_BLANK_IMAGE_SIZE_MB = 1024
DEFAULT_MANAGED_VM_CLOUD_INIT_SEED_SIZE_MB = 4
DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS = 5.0
MANAGED_VM_CLOUD_INIT_MARKER_FILES = (
    "cloud-init-bootcmd.txt",
    "cloud-init-runcmd.txt",
    "cloud-init-agent-enable.txt",
    "cloud-init-virtiofs.err",
)
EFI_SYSTEM_PARTITION_GUID = uuid.UUID("c12a7328-f81f-11d2-ba4b-00a0c93ec93b")
LINUX_ARM64_ROOT_PARTITION_GUID = uuid.UUID("b921b045-1df0-41c3-af44-4c6f280d3fae")
MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS = (
    "EFI/debian/grub.cfg",
    "EFI/BOOT/grub.cfg",
)
MANAGED_VM_EFI_AGENT_INITRD_PATH = "CONOSAGT.IMG"


@dataclass(frozen=True)
class ManagedVMConfig:
    state_root: str
    helper_path: str
    image_id: str = DEFAULT_MANAGED_VM_IMAGE_ID
    instance_id: str = DEFAULT_MANAGED_VM_INSTANCE_ID
    provider: str = "managed"
    schema_version: str = MANAGED_VM_PROVIDER_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_id(value: str, *, default: str, label: str) -> str:
    selected = _clean(value) or default
    if selected in {".", ".."} or "/" in selected or "\\" in selected:
        raise ValueError(f"{label} must be a simple identifier, not a path")
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_digest(path: Path, algorithm: str) -> tuple[str, str]:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    raw = digest.digest()
    return digest.hexdigest(), base64.b64encode(raw).decode("ascii").rstrip("=")


def _copy_file_efficient(source: Path, destination: Path) -> Dict[str, Any]:
    """Copy a VM artifact, using APFS clone-on-write when available."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    method = "shutil.copy2"
    try:
        cp = shutil.which("cp")
        if sys.platform == "darwin" and cp:
            completed = subprocess.run(
                [cp, "-c", str(source), str(tmp)],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if int(completed.returncode) == 0:
                method = "apfs_clone_cp_c"
            else:
                tmp.unlink(missing_ok=True)
                shutil.copy2(source, tmp)
        else:
            shutil.copy2(source, tmp)
        os.replace(tmp, destination)
        return {
            "method": method,
            "source": str(source),
            "destination": str(destination),
            "status": "COPIED",
        }
    except Exception:
        tmp.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise


def _uuid_from_gpt_bytes(raw: bytes) -> str:
    try:
        return str(uuid.UUID(bytes_le=raw))
    except (TypeError, ValueError):
        return ""


def _uuid_from_ext4_bytes(raw: bytes) -> str:
    try:
        return str(uuid.UUID(bytes=raw))
    except (TypeError, ValueError):
        return ""


def _detect_linux_root_partition_from_disk(source_disk: Path, *, disk_device: str = "/dev/vda") -> Dict[str, str]:
    """Best-effort root partition inference for partitioned Linux raw disks."""

    result = {
        "root_device": "",
        "root_partition_uuid": "",
        "root_filesystem_uuid": "",
        "root_boot_spec": "",
    }

    try:
        with source_disk.open("rb") as handle:
            handle.seek(512)
            header = handle.read(92)
            if len(header) >= 92 and header[:8] == b"EFI PART":
                entries_lba = int.from_bytes(header[72:80], "little")
                entry_count = int.from_bytes(header[80:84], "little")
                entry_size = int.from_bytes(header[84:88], "little")
                for index in range(max(0, min(entry_count, 256))):
                    handle.seek(entries_lba * 512 + index * entry_size)
                    entry = handle.read(entry_size)
                    if len(entry) < 48 or entry[:16] == b"\0" * 16:
                        continue
                    first_lba = int.from_bytes(entry[32:40], "little")
                    partition_uuid = _uuid_from_gpt_bytes(entry[16:32])
                    handle.seek(first_lba * 512 + 1024 + 56)
                    if handle.read(2) == b"\x53\xef":
                        handle.seek(first_lba * 512 + 1024 + 104)
                        filesystem_uuid = _uuid_from_ext4_bytes(handle.read(16))
                        result.update(
                            {
                                "root_device": f"{disk_device}{index + 1}",
                                "root_partition_uuid": partition_uuid,
                                "root_filesystem_uuid": filesystem_uuid,
                                "root_boot_spec": f"PARTUUID={partition_uuid}" if partition_uuid else f"{disk_device}{index + 1}",
                            }
                        )
                        return result
            handle.seek(446)
            for index in range(4):
                entry = handle.read(16)
                if len(entry) < 16:
                    break
                partition_type = entry[4]
                first_lba = int.from_bytes(entry[8:12], "little")
                if partition_type and first_lba:
                    handle.seek(first_lba * 512 + 1024 + 56)
                    if handle.read(2) == b"\x53\xef":
                        handle.seek(first_lba * 512 + 1024 + 104)
                        filesystem_uuid = _uuid_from_ext4_bytes(handle.read(16))
                        root_device = f"{disk_device}{index + 1}"
                        result.update(
                            {
                                "root_device": root_device,
                                "root_filesystem_uuid": filesystem_uuid,
                                "root_boot_spec": root_device,
                            }
                        )
                        return result
        return result
    except OSError:
        return result


def _detect_linux_root_device_from_disk(source_disk: Path, *, disk_device: str = "/dev/vda") -> str:
    """Best-effort root-device inference for partitioned Linux raw disks."""

    return _detect_linux_root_partition_from_disk(source_disk, disk_device=disk_device).get("root_device", "")


def _scan_disk_for_tokens(disk_path: Path, tokens: Sequence[bytes]) -> Dict[str, bool]:
    """Best-effort raw disk token scan used for VM boot capability preflight."""

    selected = [bytes(token) for token in tokens if token]
    found = {token.decode("ascii", "ignore"): False for token in selected}
    if not selected:
        return found
    overlap_size = max(len(token) for token in selected) - 1
    previous = b""
    try:
        with disk_path.open("rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                data = previous + chunk
                for token in selected:
                    key = token.decode("ascii", "ignore")
                    if not found[key] and token in data:
                        found[key] = True
                if all(found.values()):
                    break
                previous = data[-overlap_size:] if overlap_size > 0 else b""
    except OSError:
        return found
    return found


def _managed_vm_cloud_init_guest_capability(disk_path: Path) -> Dict[str, Any]:
    """Classify whether a disk image appears able to consume a NoCloud seed."""

    token_hits = _scan_disk_for_tokens(
        disk_path,
        (
            b"/usr/bin/cloud-init",
            b"cloud-init-local.service",
            b"cloud-init.service",
            b"cloud-config.service",
            b"cloud-final.service",
            b"cloud-init.target",
        ),
    )
    service_tokens = {
        "cloud-init-local.service",
        "cloud-init.service",
        "cloud-config.service",
        "cloud-final.service",
        "cloud-init.target",
    }
    binary_present = bool(token_hits.get("/usr/bin/cloud-init", False))
    service_present = any(bool(token_hits.get(token, False)) for token in service_tokens)
    available = bool(binary_present and service_present)
    return {
        "status": "AVAILABLE" if available else "UNAVAILABLE",
        "cloud_init_likely_available": available,
        "binary_present": binary_present,
        "service_present": service_present,
        "token_hits": token_hits,
        "method": "raw_disk_token_scan",
        "disk_path": str(disk_path),
    }


def _managed_vm_guest_initrd_bundle_capability(
    sidecar: Dict[str, Any],
    *,
    expected_port: int | None = None,
) -> Dict[str, Any]:
    """Validate that an initrd sidecar exposes the bounded Con OS execution path."""

    files = list(sidecar.get("files") or []) if isinstance(sidecar.get("files"), list) else []
    file_set = {str(name) for name in files}
    required_files = {
        "conos/conos_guest_agent.py",
        "conos/conos_guest_agent_launcher.sh",
        "etc/systemd/system/conos-guest-agent.service",
    }
    activation_files = {"init", "scripts/local-bottom/conos-guest-agent"}
    missing_required = sorted(required_files.difference(file_set))
    activation_present = bool(activation_files.intersection(file_set))
    transport = str(sidecar.get("guest_agent_transport") or "")
    port = sidecar.get("guest_agent_port")
    port_matches = expected_port is None or port is None or int(port) == int(expected_port)
    verified = (
        str(sidecar.get("status") or "") == "BUILT"
        and str(sidecar.get("artifact_type") or "") == "managed_vm_guest_initrd_bundle"
        and bool(sidecar.get("guest_agent_autostart_configured", False))
        and transport == "virtio-vsock"
        and not missing_required
        and activation_present
        and port_matches
    )
    return {
        "status": "VERIFIED" if verified else "INVALID",
        "verified": verified,
        "execution_path": "linux_direct_initrd_guest_agent_bundle",
        "artifact_type": str(sidecar.get("artifact_type") or ""),
        "bundle_status": str(sidecar.get("status") or ""),
        "guest_agent_autostart_configured": bool(sidecar.get("guest_agent_autostart_configured", False)),
        "guest_agent_transport": transport,
        "guest_agent_port": port,
        "expected_guest_agent_port": expected_port,
        "port_matches": port_matches,
        "missing_required_files": missing_required,
        "activation_present": activation_present,
        "activation_files_present": sorted(activation_files.intersection(file_set)),
        "readiness_contract": "runtime.json must prove guest_agent_ready=true and execution_ready=true after vsock handshake",
    }


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _read_json_from_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_sidecar_manifest(path: Path) -> Dict[str, Any]:
    return _read_json(Path(f"{path}.manifest.json"))


def _tail_file(path: Path, max_chars: int = 4000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-int(max_chars) :]


def managed_vm_state_root(explicit: str = "") -> str:
    raw = _clean(explicit) or _clean(os.environ.get("CONOS_MANAGED_VM_STATE_ROOT")) or _clean(
        os.environ.get("CONOS_VM_STATE_ROOT")
    ) or DEFAULT_MANAGED_VM_STATE_ROOT
    return str(Path(raw).expanduser().resolve())


def managed_vm_image_id(explicit: str = "") -> str:
    selected = _clean(explicit) or _clean(os.environ.get("CONOS_MANAGED_VM_IMAGE_ID")) or DEFAULT_MANAGED_VM_IMAGE_ID
    return _safe_id(selected, default=DEFAULT_MANAGED_VM_IMAGE_ID, label="managed VM image_id")


def managed_vm_instance_id(explicit: str = "") -> str:
    selected = _clean(explicit) or _clean(os.environ.get("CONOS_MANAGED_VM_INSTANCE_ID")) or DEFAULT_MANAGED_VM_INSTANCE_ID
    return _safe_id(selected, default=DEFAULT_MANAGED_VM_INSTANCE_ID, label="managed VM instance_id")


def managed_vm_helper_path(explicit: str = "") -> str:
    raw = _clean(explicit) or _clean(os.environ.get("CONOS_MANAGED_VM_HELPER"))
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            _repo_root() / "bin" / MANAGED_VM_HELPER_NAME,
            _repo_root() / "tools" / "managed_vm" / MANAGED_VM_HELPER_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    path_candidate = shutil.which(MANAGED_VM_HELPER_NAME)
    return str(path_candidate or "")


def managed_vm_helper_source_path() -> str:
    return str((_repo_root() / MANAGED_VM_HELPER_SOURCE).resolve())


def managed_vm_build_output_path(state_root: str = "") -> str:
    return str(Path(managed_vm_state_root(state_root)) / "bin" / MANAGED_VM_HELPER_NAME)


def managed_vm_runner_source_path() -> str:
    return str((_repo_root() / MANAGED_VM_RUNNER_SOURCE).resolve())


def managed_vm_runner_entitlements_path() -> str:
    return str((_repo_root() / MANAGED_VM_RUNNER_ENTITLEMENTS).resolve())


def managed_vm_guest_agent_source_path() -> str:
    return str((_repo_root() / MANAGED_VM_GUEST_AGENT_SOURCE).resolve())


def managed_vm_recipe_dir_path() -> str:
    return str((_repo_root() / MANAGED_VM_RECIPE_DIR).resolve())


def managed_vm_recipe_registry_path() -> str:
    return str((_repo_root() / MANAGED_VM_RECIPE_REGISTRY).resolve())


def managed_vm_runner_build_output_path(state_root: str = "") -> str:
    return str(Path(managed_vm_state_root(state_root)) / "bin" / MANAGED_VM_RUNNER_NAME)


def managed_vm_runner_path(explicit: str = "", state_root: str = "") -> str:
    raw = _clean(explicit) or _clean(os.environ.get("CONOS_MANAGED_VM_RUNNER"))
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            Path(managed_vm_runner_build_output_path(state_root)),
            _repo_root() / "bin" / MANAGED_VM_RUNNER_NAME,
            _repo_root() / "tools" / "managed_vm" / MANAGED_VM_RUNNER_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    path_candidate = shutil.which(MANAGED_VM_RUNNER_NAME)
    return str(path_candidate or "")


def managed_vm_image_root(state_root: str = "", image_id: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "images" / managed_vm_image_id(image_id)


def managed_vm_base_image_path(state_root: str = "", image_id: str = "") -> str:
    return str(managed_vm_image_root(state_root, image_id) / "disk.img")


def managed_vm_kernel_path(state_root: str = "", image_id: str = "") -> Path:
    return managed_vm_image_root(state_root, image_id) / "vmlinuz"


def managed_vm_initrd_path(state_root: str = "", image_id: str = "") -> Path:
    return managed_vm_image_root(state_root, image_id) / "initrd.img"


def managed_vm_guest_bundle_root(state_root: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "guest-bundles"


def managed_vm_artifact_cache_root(state_root: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "artifacts"


def managed_vm_pinned_recipe_root(state_root: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "recipes"


def managed_vm_base_image_bundle_root(state_root: str = "", image_id: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "image-bundles" / managed_vm_image_id(image_id)


def managed_vm_guest_initrd_bundle_path(state_root: str = "") -> Path:
    return managed_vm_guest_bundle_root(state_root) / "conos-guest-agent-initrd.img"


def managed_vm_image_guest_initrd_path(state_root: str = "", image_id: str = "") -> Path:
    return managed_vm_image_root(state_root, image_id) / "conos-guest-agent-initrd.img"


def managed_vm_instance_cloud_init_seed_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / MANAGED_VM_CLOUD_INIT_SEED_NAME


def managed_vm_instance_cloud_init_seed_manifest_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / MANAGED_VM_CLOUD_INIT_SEED_MANIFEST


def managed_vm_image_manifest_path(state_root: str = "", image_id: str = "") -> Path:
    return managed_vm_image_root(state_root, image_id) / MANAGED_VM_BASE_IMAGE_MANIFEST


def managed_vm_instance_root(state_root: str = "", instance_id: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "instances" / managed_vm_instance_id(instance_id)


def managed_vm_instance_manifest_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / MANAGED_VM_INSTANCE_MANIFEST


def managed_vm_runtime_manifest_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / MANAGED_VM_RUNTIME_MANIFEST


def managed_vm_agent_request_root(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / MANAGED_VM_AGENT_REQUESTS_DIR


def managed_vm_overlay_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / "overlay.img"


def managed_vm_writable_disk_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / "runtime-disk.img"


def managed_vm_efi_variable_store_path(state_root: str = "", instance_id: str = "") -> Path:
    return managed_vm_instance_root(state_root, instance_id) / "efi-variable-store.bin"


def managed_vm_config(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
) -> ManagedVMConfig:
    return ManagedVMConfig(
        state_root=managed_vm_state_root(state_root),
        helper_path=managed_vm_helper_path(helper_path),
        image_id=managed_vm_image_id(image_id),
        instance_id=managed_vm_instance_id(instance_id),
    )


def managed_vm_manifest_path(state_root: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "manifest.json"


def init_managed_vm_state(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    root = Path(config.state_root)
    for child in ("images", "instances", "snapshots", "overlays", "logs"):
        (root / child).mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "helper_available": bool(config.helper_path),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "created_at": _now_iso(),
        "directories": {
            "images": str(root / "images"),
            "instances": str(root / "instances"),
            "snapshots": str(root / "snapshots"),
            "overlays": str(root / "overlays"),
            "logs": str(root / "logs"),
        },
        "provider_contract": {
            "helper_command": MANAGED_VM_HELPER_NAME,
            "helper_source": managed_vm_helper_source_path(),
            "virtualization_runner_command": MANAGED_VM_RUNNER_NAME,
            "virtualization_runner_source": managed_vm_runner_source_path(),
            "exec_contract": (
                "helper exec --state-root <root> --instance-id <id> --image-id <image> "
                "--network-mode <mode> -- <argv...>"
            ),
            "start_contract": (
                "runner run --state-root <root> --instance-id <id> --image-id <image> "
                "--disk-path <instance-runtime-disk> --runtime-manifest <runtime.json>"
            ),
            "no_host_fallback": True,
        },
    }
    manifest = managed_vm_manifest_path(config.state_root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_managed_vm_manifest(state_root: str = "") -> Dict[str, Any]:
    return _read_json(managed_vm_manifest_path(state_root))


def load_managed_vm_image_manifest(state_root: str = "", image_id: str = "") -> Dict[str, Any]:
    return _read_json(managed_vm_image_manifest_path(state_root, image_id))


def load_managed_vm_instance_manifest(state_root: str = "", instance_id: str = "") -> Dict[str, Any]:
    return _read_json(managed_vm_instance_manifest_path(state_root, instance_id))


def load_managed_vm_runtime_manifest(state_root: str = "", instance_id: str = "") -> Dict[str, Any]:
    return _read_json(managed_vm_runtime_manifest_path(state_root, instance_id))


def _process_alive(pid_value: object) -> bool:
    try:
        pid = int(str(pid_value or "").strip())
    except ValueError:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        return False
    return True


def _process_finished_after_signal(pid: int) -> bool:
    try:
        waited_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        waited_pid = 0
    except OSError:
        waited_pid = 0
    if waited_pid == pid:
        return True
    return not _process_alive(pid)


def _managed_vm_host_virtualization_capability() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "platform": sys.platform,
        "apple_virtualization_framework_expected": sys.platform == "darwin",
        "kern_hv_support": "",
        "probe_status": "NOT_RUN",
    }
    if sys.platform != "darwin":
        payload["probe_status"] = "UNSUPPORTED_PLATFORM"
        return payload
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "kern.hv_support"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        payload.update({"probe_status": "PROBE_FAILED", "probe_error": str(exc)})
        return payload
    payload.update(
        {
            "probe_status": "PROBED",
            "kern_hv_support": str(completed.stdout or "").strip(),
            "probe_returncode": int(completed.returncode),
            "probe_stderr": str(completed.stderr or "").strip(),
        }
    )
    return payload


def _managed_vm_start_blocker_payload(reason: object) -> Dict[str, Any]:
    text = str(reason or "")
    lowered = text.lower()
    unsupported_kernel_markers = (
        "kernel artifact is efi/pe-wrapped",
        "not a raw linux kernel image",
        "linux_direct kernel artifact",
    )
    if any(marker in lowered for marker in unsupported_kernel_markers):
        return {
            "status": "START_BLOCKED_UNSUPPORTED_BOOT_ARTIFACT",
            "lifecycle_state": "blocked",
            "blocker_type": "unsupported_boot_artifact",
            "reason": text or "linux_direct boot artifact is unsupported",
            "host_virtualization_capability": _managed_vm_host_virtualization_capability(),
            "next_required_step": (
                "Use a raw Linux kernel Image artifact for linux_direct boot, or register this image through "
                "an EFI boot path instead of VZLinuxBootLoader."
            ),
            "virtual_machine_started": False,
            "process_alive": False,
            "execution_ready": False,
            "guest_agent_ready": False,
            "no_host_fallback": True,
        }
    unavailable_markers = (
        "virtualization is not available on this hardware",
        "virtualization framework is not available",
        "virtualization not available",
    )
    if not any(marker in lowered for marker in unavailable_markers):
        return {}
    return {
        "status": "START_BLOCKED_HOST_VIRTUALIZATION_UNAVAILABLE",
        "lifecycle_state": "blocked",
        "blocker_type": "host_virtualization_unavailable",
        "reason": text or "host Apple Virtualization capability is unavailable",
        "host_virtualization_capability": _managed_vm_host_virtualization_capability(),
        "next_required_step": (
            "Run on an Apple Silicon macOS host where Virtualization.framework can create VMs; "
            "nested or restricted desktop environments may report kern.hv_support=1 but still block VM creation."
        ),
        "virtual_machine_started": False,
        "process_alive": False,
        "execution_ready": False,
        "guest_agent_ready": False,
        "no_host_fallback": True,
    }


def _helper_lifecycle_command(
    helper_command: str,
    *,
    config: ManagedVMConfig,
    network_mode: str,
) -> list[str]:
    return [
        config.helper_path,
        helper_command,
        "--state-root",
        config.state_root,
        "--instance-id",
        config.instance_id,
        "--image-id",
        config.image_id,
        "--network-mode",
        _clean(network_mode) or "provider_default",
    ]


def _helper_agent_exec_command(
    command: Sequence[str],
    *,
    config: ManagedVMConfig,
    network_mode: str,
) -> list[str]:
    return [
        config.helper_path,
        "agent-exec",
        "--state-root",
        config.state_root,
        "--instance-id",
        config.instance_id,
        "--image-id",
        config.image_id,
        "--network-mode",
        _clean(network_mode) or "provider_default",
        "--",
        *[str(part) for part in command],
    ]


def _managed_vm_agent_request_paths(*, state_root: str, instance_id: str, request_id: str) -> tuple[Path, Path]:
    request_root = managed_vm_agent_request_root(state_root, instance_id)
    return request_root / f"{request_id}.request.json", request_root / f"{request_id}.result.json"


def _write_managed_vm_agent_request(
    command: Sequence[str],
    *,
    state_root: str,
    image_id: str,
    instance_id: str,
    timeout_seconds: int,
    cwd: str = "",
    stdin_bytes: bytes | None = None,
) -> Dict[str, Any]:
    request_id = uuid.uuid4().hex
    request_path, result_path = _managed_vm_agent_request_paths(
        state_root=state_root,
        instance_id=instance_id,
        request_id=request_id,
    )
    request_payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "event_type": "exec",
        "request_id": request_id,
        "status": "PENDING",
        "state_root": state_root,
        "image_id": image_id,
        "instance_id": instance_id,
        "command": [str(part) for part in command],
        "timeout_seconds": int(timeout_seconds),
        "created_at": _now_iso(),
        "no_host_fallback": True,
    }
    if _clean(cwd):
        request_payload["cwd"] = _clean(cwd)
    if stdin_bytes is not None:
        request_payload["stdin_b64"] = base64.b64encode(bytes(stdin_bytes)).decode("ascii")
    request_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = request_path.with_suffix(request_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(request_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(request_path)
    return {
        "request_id": request_id,
        "request_path": str(request_path),
        "result_path": str(result_path),
        "request_payload": request_payload,
    }


def _wait_for_managed_vm_agent_result(result_path: Path, *, timeout_seconds: int) -> Dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds) + 1.0)
    while time.monotonic() < deadline:
        payload = _read_json(result_path)
        if payload:
            return payload
        time.sleep(0.05)
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "event_type": "exec_result",
        "status": "TIMEOUT",
        "reason": f"timed out waiting for guest-agent result at {result_path}",
        "returncode": 124,
        "stdout": "",
        "stderr": "",
    }


def _cpio_pad(size: int) -> bytes:
    return b"\0" * ((4 - (size % 4)) % 4)


def _newc_entry(name: str, data: bytes, *, mode: int, ino: int) -> bytes:
    clean_name = name.strip("/")
    name_bytes = clean_name.encode("utf-8") + b"\0"
    fields = [
        "070701",
        f"{ino & 0xFFFFFFFF:08x}",
        f"{mode & 0xFFFFFFFF:08x}",
        "00000000",
        "00000000",
        "00000001",
        "00000000",
        f"{len(data) & 0xFFFFFFFF:08x}",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
        f"{len(name_bytes) & 0xFFFFFFFF:08x}",
        "00000000",
    ]
    header = "".join(fields).encode("ascii")
    return header + name_bytes + _cpio_pad(len(header) + len(name_bytes)) + data + _cpio_pad(len(data))


def _build_newc_archive(entries: Sequence[tuple[str, bytes, int]]) -> bytes:
    archive = bytearray()
    ino = 1
    directory_names: set[str] = set()
    for name, _, _ in entries:
        parts = name.strip("/").split("/")[:-1]
        current: list[str] = []
        for part in parts:
            current.append(part)
            directory_names.add("/".join(current))
    for directory in sorted(directory_names):
        archive.extend(_newc_entry(directory, b"", mode=0o040755, ino=ino))
        ino += 1
    for name, data, mode in entries:
        archive.extend(_newc_entry(name, data, mode=mode, ino=ino))
        ino += 1
    archive.extend(_newc_entry("TRAILER!!!", b"", mode=0, ino=ino))
    return bytes(archive)


def _read_newc_file(archive: bytes, target_name: str) -> bytes:
    clean_target = target_name.strip("/")
    position = 0
    while position + 110 <= len(archive) and archive[position : position + 6] in {b"070701", b"070702"}:
        try:
            file_size = int(archive[position + 54 : position + 62], 16)
            name_size = int(archive[position + 94 : position + 102], 16)
        except ValueError:
            break
        name_start = position + 110
        name_end = name_start + name_size
        if name_end > len(archive):
            break
        name = archive[name_start : max(name_start, name_end - 1)].decode("utf-8", errors="replace")
        data_start = (name_end + 3) & ~3
        data_end = (data_start + file_size + 3) & ~3
        if data_end > len(archive):
            break
        if name.strip("/") == clean_target:
            return archive[data_start : data_start + file_size]
        position = data_end
    return b""


def _remove_newc_entries(archive: bytes, remove_names: set[str]) -> tuple[bytes, list[str]]:
    clean_remove_names = {name.strip("/") for name in remove_names if name.strip("/")}
    if not clean_remove_names:
        return archive, []
    output = bytearray()
    removed: list[str] = []
    position = 0
    while position + 110 <= len(archive) and archive[position : position + 6] in {b"070701", b"070702"}:
        entry_start = position
        try:
            file_size = int(archive[position + 54 : position + 62], 16)
            name_size = int(archive[position + 94 : position + 102], 16)
        except ValueError:
            return archive, []
        name_start = position + 110
        name_end = name_start + name_size
        if name_end > len(archive):
            return archive, []
        name = archive[name_start : max(name_start, name_end - 1)].decode("utf-8", errors="replace").strip("/")
        data_start = (name_end + 3) & ~3
        data_end = (data_start + file_size + 3) & ~3
        if data_end > len(archive):
            return archive, []
        if name in clean_remove_names:
            removed.append(name)
        else:
            output.extend(archive[entry_start:data_end])
        position = data_end
    output.extend(archive[position:])
    return bytes(output), removed


def _extend_initramfs_stage_order(existing: bytes, *, stage: str, hook_name: str) -> tuple[bytes, bool]:
    hook_line = f'/scripts/{stage}/{hook_name} "$@"'
    text = existing.decode("utf-8", errors="replace") if existing else ""
    if hook_line in text:
        return existing, bool(existing)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    lines.append(hook_line)
    if not any("/conf/param.conf" in line for line in lines):
        lines.append("[ -e /conf/param.conf ] && . /conf/param.conf")
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8"), bool(existing)


def _extend_initramfs_local_bottom_order(existing: bytes) -> tuple[bytes, bool]:
    return _extend_initramfs_stage_order(existing, stage="local-bottom", hook_name="conos-guest-agent")


def _extend_initramfs_modules(existing: bytes) -> tuple[bytes, bool, list[str]]:
    required_modules = [
        "virtio_pci",
        "virtio_mmio",
        "virtio_blk",
        "virtio_net",
        "virtio_console",
        "virtio_rng",
        "virtiofs",
        "ext4",
    ]
    text = existing.decode("utf-8", errors="replace") if existing else ""
    lines = [line.rstrip() for line in text.splitlines()]
    present = {line.strip().split()[0] for line in lines if line.strip() and not line.lstrip().startswith("#")}
    added: list[str] = []
    for module in required_modules:
        if module in present:
            continue
        lines.append(module)
        added.append(module)
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8"), bool(existing), added


def _strip_final_newc_trailer(archive: bytes) -> tuple[bytes, bool]:
    """Remove the final newc TRAILER!!! entry so another archive can be merged."""

    position = 0
    last_trailer_start = -1
    last_trailer_end = -1
    while position + 110 <= len(archive) and archive[position : position + 6] in {b"070701", b"070702"}:
        try:
            file_size = int(archive[position + 54 : position + 62], 16)
            name_size = int(archive[position + 94 : position + 102], 16)
        except ValueError:
            break
        name_start = position + 110
        name_end = name_start + name_size
        if name_end > len(archive):
            break
        name = archive[name_start : max(name_start, name_end - 1)]
        data_start = (name_end + 3) & ~3
        data_end = (data_start + file_size + 3) & ~3
        if data_end > len(archive):
            break
        if name == b"TRAILER!!!":
            last_trailer_start = position
            last_trailer_end = data_end
        position = data_end
    if last_trailer_start < 0:
        return archive, False
    if any(byte != 0 for byte in archive[last_trailer_end:]):
        return archive, False
    return archive[:last_trailer_start], True


def _initrd_compression_for_bytes(data: bytes) -> str:
    if data.startswith(b"\x28\xb5\x2f\xfd"):
        return "zstd"
    if data.startswith(b"\x1f\x8b"):
        return "gzip"
    if data.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if data.startswith(b"\x04\x22\x4d\x18"):
        return "lz4"
    return "unknown"


def _zstd_path() -> str:
    return _clean(shutil.which("zstd"))


def _decompress_initrd_payload(path: Path, *, compression: str) -> tuple[bytes, str]:
    if compression == "gzip":
        return gzip.decompress(path.read_bytes()), "python_gzip"
    if compression == "zstd":
        zstd = _zstd_path()
        if not zstd:
            raise RuntimeError("zstd initrd merge requires the zstd command")
        completed = subprocess.run(
            [zstd, "-q", "-d", "-c", str(path)],
            capture_output=True,
            check=False,
        )
        if int(completed.returncode) != 0:
            raise RuntimeError((completed.stderr or b"zstd decompression failed").decode("utf-8", "replace"))
        return bytes(completed.stdout), "zstd"
    raise RuntimeError(f"unsupported compressed initrd format for merge: {compression}")


def _compress_initrd_payload(payload: bytes, *, compression: str) -> tuple[bytes, str]:
    if compression == "gzip":
        return gzip.compress(payload, compresslevel=9, mtime=0), "python_gzip"
    if compression == "zstd":
        zstd = _zstd_path()
        if not zstd:
            raise RuntimeError("zstd initrd merge requires the zstd command")
        completed = subprocess.run(
            [zstd, "-q", "-19", "-c"],
            input=payload,
            capture_output=True,
            check=False,
        )
        if int(completed.returncode) != 0:
            raise RuntimeError((completed.stderr or b"zstd compression failed").decode("utf-8", "replace"))
        return bytes(completed.stdout), "zstd"
    raise RuntimeError(f"unsupported compressed initrd format for merge: {compression}")


def _sanitize_cloud_init_identifier(value: str, *, default: str) -> str:
    raw = _clean(value) or default
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in raw)
    cleaned = cleaned.strip(".-_")
    return cleaned or default


def _fat_lfn_checksum(short_name: bytes) -> int:
    checksum = 0
    for byte in short_name:
        checksum = (((checksum & 1) << 7) + (checksum >> 1) + byte) & 0xFF
    return checksum


def _fat_lfn_entry(sequence: int, text: str, checksum: int, *, last: bool) -> bytes:
    chars = [ord(char) for char in text]
    if len(chars) < 13:
        chars.append(0)
    while len(chars) < 13:
        chars.append(0xFFFF)
    entry = bytearray(32)
    entry[0] = sequence | (0x40 if last else 0)
    entry[11] = 0x0F
    entry[12] = 0
    entry[13] = checksum
    entry[26:28] = (0).to_bytes(2, "little")
    positions = [1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30]
    for position, codepoint in zip(positions, chars):
        entry[position : position + 2] = int(codepoint).to_bytes(2, "little")
    return bytes(entry)


def _fat_short_name(filename: str, index: int) -> bytes:
    stem = "".join(char for char in filename.upper() if char.isalnum()) or "FILE"
    alias = f"{stem[:6]}~{int(index)}"[:8].ljust(8)
    return (alias + "   ").encode("ascii")


def _fat_directory_entries(files: Sequence[tuple[str, bytes, int]]) -> bytes:
    entries = bytearray()
    label = "CIDATA".ljust(11).encode("ascii")
    volume = bytearray(32)
    volume[0:11] = label
    volume[11] = 0x08
    entries.extend(volume)
    for index, (filename, data, first_cluster) in enumerate(files, start=1):
        short_name = _fat_short_name(filename, index)
        checksum = _fat_lfn_checksum(short_name)
        chunks = [filename[offset : offset + 13] for offset in range(0, len(filename), 13)] or [filename]
        for reverse_index, chunk in enumerate(reversed(chunks), start=1):
            sequence = len(chunks) - reverse_index + 1
            entries.extend(_fat_lfn_entry(sequence, chunk, checksum, last=sequence == len(chunks)))
        short_entry = bytearray(32)
        short_entry[0:11] = short_name
        short_entry[11] = 0x20
        short_entry[26:28] = int(first_cluster).to_bytes(2, "little")
        short_entry[28:32] = int(len(data)).to_bytes(4, "little")
        entries.extend(short_entry)
    return bytes(entries)


def _build_vfat_nocloud_seed_image(files: Dict[str, bytes], *, size_mb: int = DEFAULT_MANAGED_VM_CLOUD_INIT_SEED_SIZE_MB) -> bytes:
    """Build a small VFAT CIDATA image for cloud-init NoCloud.

    The implementation is intentionally tiny and deterministic: one sector per
    cluster, FAT16, a fixed CIDATA label, and VFAT long filename entries for the
    cloud-init filenames.
    """

    sector_size = 512
    total_sectors = max(2048, int(size_mb) * 1024 * 1024 // sector_size)
    reserved_sectors = 1
    fat_count = 2
    root_entries = 512
    root_dir_sectors = (root_entries * 32 + sector_size - 1) // sector_size
    sectors_per_cluster = 1
    fat_sectors = 1
    while True:
        data_sectors = total_sectors - reserved_sectors - root_dir_sectors - fat_count * fat_sectors
        cluster_count = data_sectors // sectors_per_cluster
        required_fat_sectors = ((cluster_count + 2) * 2 + sector_size - 1) // sector_size
        if required_fat_sectors == fat_sectors:
            break
        fat_sectors = required_fat_sectors
    if cluster_count < 4085:
        raise ValueError("cloud-init seed image is too small for FAT16")

    ordered_files: list[tuple[str, bytes, int]] = []
    next_cluster = 2
    fat_entries = [0x0000] * (cluster_count + 2)
    fat_entries[0] = 0xFFF8
    fat_entries[1] = 0xFFFF
    for name in ("user-data", "meta-data", "network-config"):
        if name not in files:
            continue
        payload = bytes(files[name])
        clusters_needed = max(1, (len(payload) + sector_size * sectors_per_cluster - 1) // (sector_size * sectors_per_cluster))
        first_cluster = next_cluster
        for offset in range(clusters_needed):
            cluster = first_cluster + offset
            fat_entries[cluster] = 0xFFFF if offset == clusters_needed - 1 else cluster + 1
        ordered_files.append((name, payload, first_cluster))
        next_cluster += clusters_needed
    if next_cluster >= len(fat_entries):
        raise ValueError("cloud-init seed files exceed seed image capacity")

    image = bytearray(total_sectors * sector_size)
    boot = bytearray(sector_size)
    boot[0:3] = b"\xEB\x3C\x90"
    boot[3:11] = b"CONOS   "
    boot[11:13] = sector_size.to_bytes(2, "little")
    boot[13] = sectors_per_cluster
    boot[14:16] = reserved_sectors.to_bytes(2, "little")
    boot[16] = fat_count
    boot[17:19] = root_entries.to_bytes(2, "little")
    boot[19:21] = (total_sectors if total_sectors < 65536 else 0).to_bytes(2, "little")
    boot[21] = 0xF8
    boot[22:24] = fat_sectors.to_bytes(2, "little")
    boot[24:26] = (32).to_bytes(2, "little")
    boot[26:28] = (64).to_bytes(2, "little")
    boot[28:32] = (0).to_bytes(4, "little")
    boot[32:36] = (total_sectors if total_sectors >= 65536 else 0).to_bytes(4, "little")
    boot[36] = 0x80
    boot[38] = 0x29
    boot[39:43] = (0xC0050001).to_bytes(4, "little")
    boot[43:54] = b"CIDATA     "
    boot[54:62] = b"FAT16   "
    boot[510:512] = b"\x55\xAA"
    image[0:sector_size] = boot

    fat_bytes = bytearray(fat_sectors * sector_size)
    for index, entry in enumerate(fat_entries):
        if index * 2 + 2 > len(fat_bytes):
            break
        fat_bytes[index * 2 : index * 2 + 2] = int(entry).to_bytes(2, "little")
    fat_start = reserved_sectors * sector_size
    for fat_index in range(fat_count):
        start = fat_start + fat_index * fat_sectors * sector_size
        image[start : start + len(fat_bytes)] = fat_bytes

    root_start_sector = reserved_sectors + fat_count * fat_sectors
    root_start = root_start_sector * sector_size
    root_payload = _fat_directory_entries(ordered_files)
    image[root_start : root_start + len(root_payload)] = root_payload[: root_dir_sectors * sector_size]

    data_start_sector = root_start_sector + root_dir_sectors
    for _, data, first_cluster in ordered_files:
        offset = 0
        cluster = first_cluster
        while cluster < len(fat_entries):
            sector = data_start_sector + (cluster - 2) * sectors_per_cluster
            start = sector * sector_size
            chunk = data[offset : offset + sector_size * sectors_per_cluster]
            image[start : start + len(chunk)] = chunk
            offset += len(chunk)
            if fat_entries[cluster] >= 0xFFF8 or offset >= len(data):
                break
            cluster = fat_entries[cluster]
    return bytes(image)


def _build_partitioned_vfat_nocloud_seed_image(
    files: Dict[str, bytes], *, size_mb: int = DEFAULT_MANAGED_VM_CLOUD_INIT_SEED_SIZE_MB
) -> tuple[bytes, Dict[str, Any]]:
    sector_size = 512
    total_sectors = max(8192, int(size_mb) * 1024 * 1024 // sector_size)
    partition_start_lba = 2048
    partition_sectors = total_sectors - partition_start_lba
    partition_size_mb = max(3, (partition_sectors * sector_size) // (1024 * 1024))
    fat_image = _build_vfat_nocloud_seed_image(files, size_mb=partition_size_mb)
    fat_sectors = len(fat_image) // sector_size
    required_total_sectors = partition_start_lba + fat_sectors
    if required_total_sectors > total_sectors:
        total_sectors = required_total_sectors
    image = bytearray(total_sectors * sector_size)
    mbr = bytearray(sector_size)
    entry = bytearray(16)
    entry[4] = 0x0E
    entry[8:12] = int(partition_start_lba).to_bytes(4, "little")
    entry[12:16] = int(fat_sectors).to_bytes(4, "little")
    mbr[446:462] = entry
    mbr[510:512] = b"\x55\xaa"
    image[0:sector_size] = mbr
    start = partition_start_lba * sector_size
    image[start : start + len(fat_image)] = fat_image
    return bytes(image), {
        "seed_layout": "mbr_partitioned_vfat",
        "partition_table": "mbr",
        "partition_type": "0x0e",
        "partition_start_lba": partition_start_lba,
        "partition_sector_count": fat_sectors,
        "partition_byte_offset": start,
        "partition_byte_size": len(fat_image),
    }


def _cloud_init_guest_agent_user_data(
    *,
    guest_agent_path: Path,
    python_path: str,
    port: int,
    shared_dir_tag: str = MANAGED_VM_SHARED_DIR_TAG,
) -> str:
    agent_b64 = base64.b64encode(guest_agent_path.read_bytes()).decode("ascii")
    launcher = _guest_agent_launcher_script(python_path=python_path, port=port).rstrip()
    launcher_block = "\n".join(f"      {line}" for line in launcher.splitlines())
    unit = _guest_agent_systemd_unit(python_path=python_path, port=port).rstrip()
    unit_block = "\n".join(f"      {line}" for line in unit.splitlines())
    marker_command = (
        "mkdir -p /mnt/conos-host && "
        f"mount -t virtiofs {shared_dir_tag} /mnt/conos-host 2>/tmp/conos-virtiofs.err || true"
    )
    return f"""#cloud-config
write_files:
  - path: /opt/conos/conos_guest_agent.py
    permissions: '0755'
    owner: root:root
    encoding: b64
    content: {agent_b64}
  - path: /opt/conos/conos_guest_agent_launcher.sh
    permissions: '0755'
    owner: root:root
    content: |
{launcher_block}
  - path: /etc/systemd/system/conos-guest-agent.service
    permissions: '0644'
    owner: root:root
    content: |
{unit_block}
  - path: /etc/systemd/system/systemd-networkd-wait-online.service.d/conos-fast-boot.conf
    permissions: '0644'
    owner: root:root
    content: |
      [Service]
      ExecStart=
      ExecStart=/bin/true
bootcmd:
  - [/bin/sh, -c, "mkdir -p /etc/systemd/system/systemd-networkd-wait-online.service.d; printf '[Service]\\nExecStart=\\nExecStart=/bin/true\\n' > /etc/systemd/system/systemd-networkd-wait-online.service.d/conos-fast-boot.conf; systemctl daemon-reload 2>/dev/null || true"]
  - [/bin/sh, -c, "echo CONOS_CLOUD_INIT_BOOTCMD > /dev/hvc0 2>/dev/null || echo CONOS_CLOUD_INIT_BOOTCMD > /dev/console 2>/dev/null || true"]
  - [/bin/sh, -c, "{marker_command}; if grep -qs ' /mnt/conos-host ' /proc/mounts; then {{ echo CONOS_CLOUD_INIT_BOOTCMD; date -u 2>/dev/null || true; uname -a 2>/dev/null || true; }} > /mnt/conos-host/cloud-init-bootcmd.txt; fi; true"]
runcmd:
  - [/bin/sh, -c, "echo CONOS_CLOUD_INIT_RUNCMD > /dev/hvc0 2>/dev/null || echo CONOS_CLOUD_INIT_RUNCMD > /dev/console 2>/dev/null || true"]
  - [/bin/sh, -c, "{marker_command}; if grep -qs ' /mnt/conos-host ' /proc/mounts; then {{ echo CONOS_CLOUD_INIT_RUNCMD; date -u 2>/dev/null || true; cloud-init status --long 2>/dev/null || true; }} > /mnt/conos-host/cloud-init-runcmd.txt; cp /tmp/conos-virtiofs.err /mnt/conos-host/cloud-init-virtiofs.err 2>/dev/null || true; fi; true"]
  - [systemctl, daemon-reload]
  - [/bin/sh, -c, "systemctl enable --now conos-guest-agent.service; RC=$?; {marker_command}; if grep -qs ' /mnt/conos-host ' /proc/mounts; then {{ echo CONOS_CLOUD_INIT_AGENT_ENABLE rc=$RC; systemctl status --no-pager conos-guest-agent.service 2>&1 || true; }} > /mnt/conos-host/cloud-init-agent-enable.txt; fi; true"]
"""


def build_managed_vm_cloud_init_seed(
    *,
    state_root: str = "",
    instance_id: str = "",
    output_path: str = "",
    guest_agent_path: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    guest_python_path: str = "/usr/bin/python3",
    hostname: str = "",
    network_config: str = "",
    overwrite: bool = False,
    size_mb: int = DEFAULT_MANAGED_VM_CLOUD_INIT_SEED_SIZE_MB,
) -> Dict[str, Any]:
    """Build a NoCloud CIDATA seed disk that enables the Con OS guest agent."""

    instance = managed_vm_instance_id(instance_id)
    output = (
        Path(output_path).expanduser()
        if _clean(output_path)
        else managed_vm_instance_cloud_init_seed_path(state_root, instance)
    ).resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"managed VM cloud-init seed already exists: {output}")
    agent = Path(_clean(guest_agent_path) or managed_vm_guest_agent_source_path()).expanduser().resolve()
    if not agent.exists() or not agent.is_file():
        raise FileNotFoundError(f"managed VM guest agent source does not exist: {agent}")
    selected_hostname = _sanitize_cloud_init_identifier(hostname, default=f"conos-{instance}")
    selected_python = _clean(guest_python_path) or "/usr/bin/python3"
    selected_port = int(guest_agent_port)
    user_data = _cloud_init_guest_agent_user_data(
        guest_agent_path=agent,
        python_path=selected_python,
        port=selected_port,
    )
    meta_data = (
        f"instance-id: {_sanitize_cloud_init_identifier(instance, default='conos-instance')}\n"
        f"local-hostname: {selected_hostname}\n"
    )
    selected_network_config = _clean(network_config) or "version: 2\n"
    files = {
        "user-data": user_data.encode("utf-8"),
        "meta-data": meta_data.encode("utf-8"),
        "network-config": selected_network_config.encode("utf-8"),
    }
    image, seed_layout = _build_partitioned_vfat_nocloud_seed_image(files, size_mb=int(size_mb))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image)
    manifest_path = (
        Path(f"{output}.manifest.json")
        if _clean(output_path)
        else managed_vm_instance_cloud_init_seed_manifest_path(state_root, instance)
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_cloud_init_nocloud_seed",
        "status": "BUILT",
        "state_root": managed_vm_state_root(state_root),
        "instance_id": instance,
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "output_byte_size": output.stat().st_size,
        "filesystem": "vfat",
        "volume_label": "CIDATA",
        "seed_format": "cloud-init_nocloud",
        **seed_layout,
        "files": list(files.keys()),
        "cloud_init_marker_files": list(MANAGED_VM_CLOUD_INIT_MARKER_FILES),
        "cloud_init_diagnostic_stages": ["bootcmd", "runcmd", "agent_enable"],
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "guest_agent_path": str(agent),
        "guest_agent_sha256": _sha256_file(agent),
        "guest_agent_port": selected_port,
        "guest_agent_transport": "virtio-vsock",
        "guest_agent_autostart_configured": True,
        "guest_agent_start_target": "sysinit.target",
        "network_wait_online_override": True,
        "guest_python_path": selected_python,
        "hostname": selected_hostname,
        "created_at": _now_iso(),
        "owned_by_conos": True,
        "no_host_fallback": True,
        "readiness_contract": "runtime.json guest_agent_ready=true and execution_ready=true after vsock handshake",
        "runtime_verified": False,
    }
    _write_json(manifest_path, payload)
    return payload


def _guest_agent_systemd_unit(*, python_path: str, port: int) -> str:
    return f"""[Unit]
Description=Con OS managed VM guest agent
DefaultDependencies=no
Wants=local-fs.target
After=local-fs.target
Before=cloud-init.service cloud-config.service cloud-final.service network-online.target multi-user.target

[Service]
Type=simple
Environment=CONOS_GUEST_AGENT_PORT={int(port)}
Environment=CONOS_GUEST_AGENT_PYTHON={python_path}
ExecStart=/opt/conos/conos_guest_agent_launcher.sh
Restart=always
RestartSec=1

[Install]
WantedBy=sysinit.target
"""


def _guest_agent_launcher_script(*, python_path: str, port: int) -> str:
    return f"""#!/bin/sh
set -u
PORT="${{CONOS_GUEST_AGENT_PORT:-{int(port)}}}"
PYTHON="${{CONOS_GUEST_AGENT_PYTHON:-{python_path}}}"
LOG="/var/log/conos-guest-agent.log"
CONSOLE="/dev/hvc0"
if [ ! -w "$CONSOLE" ]; then
  CONSOLE="/dev/console"
fi
mkdir -p /var/log
echo "CONOS_GUEST_AGENT_START port=$PORT python=$PYTHON" >>"$LOG"
echo "CONOS_GUEST_AGENT_START port=$PORT" >"$CONSOLE" 2>/dev/null || true
"$PYTHON" /opt/conos/conos_guest_agent.py --port "$PORT" >>"$LOG" 2>&1
RC=$?
echo "CONOS_GUEST_AGENT_EXIT rc=$RC" >>"$LOG"
echo "CONOS_GUEST_AGENT_EXIT rc=$RC" >"$CONSOLE" 2>/dev/null || true
tail -20 "$LOG" >"$CONSOLE" 2>/dev/null || true
exit "$RC"
"""


def _guest_agent_install_script(*, python_path: str, port: int) -> str:
    return f"""#!/bin/sh
set -eu
install -d /opt/conos /etc/systemd/system
install -m 0755 /conos/conos_guest_agent.py /opt/conos/conos_guest_agent.py
install -m 0755 /conos/conos_guest_agent_launcher.sh /opt/conos/conos_guest_agent_launcher.sh
cat >/etc/systemd/system/conos-guest-agent.service <<'UNIT'
{_guest_agent_systemd_unit(python_path=python_path, port=port).rstrip()}
UNIT
if command -v systemctl >/dev/null 2>&1; then
  systemctl disable conos-guest-agent.service 2>/dev/null || true
  systemctl enable conos-guest-agent.service || true
fi
"""


def _guest_agent_initramfs_trace_snippet(*, stage: str) -> str:
    marker = f"conos-initramfs-{stage}.txt"
    message = f"CONOS_INITRAMFS_{stage.replace('-', '_').upper()}"
    return f"""
CONOS_TRACE_STAGE="{stage}"
CONOS_TRACE_MARKER="{marker}"
conos_trace() {{
  MSG="${{1:-{message}}}"
  echo "$MSG" >"$CONSOLE" 2>/dev/null || true
  mkdir -p /conos-host 2>/dev/null || true
  if ! grep -qs ' /conos-host ' /proc/mounts 2>/dev/null; then
    modprobe virtiofs 2>/dev/null || true
    mount -t virtiofs conos_host /conos-host 2>/tmp/conos-virtiofs-"$CONOS_TRACE_STAGE".err || true
  fi
  if grep -qs ' /conos-host ' /proc/mounts 2>/dev/null; then
    {{
      echo "$MSG"
      echo "stage=$CONOS_TRACE_STAGE"
      echo "cmdline=$(cat /proc/cmdline 2>/dev/null || true)"
      echo "rootmnt=${{rootmnt:-}}"
    }} >>"/conos-host/$CONOS_TRACE_MARKER" 2>/dev/null || true
  fi
}}
"""


def _guest_agent_initramfs_trace_hook(*, stage: str) -> str:
    return f"""#!/bin/sh
set -u
CONSOLE="/dev/hvc0"
if [ ! -w "$CONSOLE" ]; then
  CONSOLE="/dev/console"
fi
{_guest_agent_initramfs_trace_snippet(stage=stage).strip()}
conos_trace
exit 0
"""


def _guest_agent_initramfs_local_bottom_hook() -> str:
    return """#!/bin/sh
set -u
ROOTMNT="${rootmnt:-/root}"
CONSOLE="/dev/hvc0"
if [ ! -w "$CONSOLE" ]; then
  CONSOLE="/dev/console"
fi
""" + _guest_agent_initramfs_trace_snippet(stage="local-bottom") + """
echo "CONOS_INITRAMFS_LOCAL_BOTTOM rootmnt=$ROOTMNT" >"$CONSOLE" 2>/dev/null || true
conos_trace "CONOS_INITRAMFS_LOCAL_BOTTOM rootmnt=$ROOTMNT"
if [ ! -d "$ROOTMNT" ]; then
  exit 0
fi
mkdir -p "$ROOTMNT/opt/conos" "$ROOTMNT/etc/systemd/system" "$ROOTMNT/etc/systemd/system/sysinit.target.wants" 2>/dev/null || true
cp /conos/conos_guest_agent.py "$ROOTMNT/opt/conos/conos_guest_agent.py" 2>/dev/null || true
cp /conos/conos_guest_agent_launcher.sh "$ROOTMNT/opt/conos/conos_guest_agent_launcher.sh" 2>/dev/null || true
cp /etc/systemd/system/conos-guest-agent.service "$ROOTMNT/etc/systemd/system/conos-guest-agent.service" 2>/dev/null || true
chmod 0755 "$ROOTMNT/opt/conos/conos_guest_agent.py" "$ROOTMNT/opt/conos/conos_guest_agent_launcher.sh" 2>/dev/null || true
rm -f "$ROOTMNT/etc/systemd/system/multi-user.target.wants/conos-guest-agent.service" 2>/dev/null || true
ln -sf /etc/systemd/system/conos-guest-agent.service "$ROOTMNT/etc/systemd/system/sysinit.target.wants/conos-guest-agent.service" 2>/dev/null || true
echo "CONOS_INITRAMFS_AGENT_INSTALLED" >"$CONSOLE" 2>/dev/null || true
conos_trace "CONOS_INITRAMFS_AGENT_INSTALLED rootmnt=$ROOTMNT"
exit 0
"""


def _guest_agent_init_wrapper(*, root_device: str, python_path: str, port: int) -> str:
    return f"""#!/bin/sh
set -u
ROOT_DEVICE="{root_device}"
KERNEL_ROOT=""
CONSOLE="/dev/hvc0"
if [ ! -w "$CONSOLE" ]; then
  CONSOLE="/dev/console"
fi
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
modprobe virtio_pci 2>/dev/null || true
modprobe virtio_mmio 2>/dev/null || true
modprobe virtio_blk 2>/dev/null || true
modprobe virtio_console 2>/dev/null || true
modprobe virtiofs 2>/dev/null || true
modprobe ext4 2>/dev/null || true
""" + _guest_agent_initramfs_trace_snippet(stage="init-wrapper") + f"""
conos_trace "CONOS_INIT_WRAPPER_START root_device=$ROOT_DEVICE"
for arg in $(cat /proc/cmdline 2>/dev/null || true); do
  case "$arg" in
    conos.root=*) ROOT_DEVICE="${{arg#conos.root=}}" ;;
    root=*) KERNEL_ROOT="${{arg#root=}}" ;;
  esac
done
root_candidate_paths() {{
  printf '%s\\n' "$ROOT_DEVICE"
  case "$KERNEL_ROOT" in
    /dev/*) printf '%s\\n' "$KERNEL_ROOT" ;;
    PARTUUID=*) printf '%s\\n' "/dev/disk/by-partuuid/${{KERNEL_ROOT#PARTUUID=}}" ;;
    partuuid=*) printf '%s\\n' "/dev/disk/by-partuuid/${{KERNEL_ROOT#partuuid=}}" ;;
    UUID=*) printf '%s\\n' "/dev/disk/by-uuid/${{KERNEL_ROOT#UUID=}}" ;;
    uuid=*) printf '%s\\n' "/dev/disk/by-uuid/${{KERNEL_ROOT#uuid=}}" ;;
  esac
  if command -v findfs >/dev/null 2>&1 && [ -n "$KERNEL_ROOT" ]; then
    findfs "$KERNEL_ROOT" 2>/dev/null || true
  fi
  for dev in /dev/vd[a-z][0-9]* /dev/sd[a-z][0-9]* /dev/xvd[a-z][0-9]* /dev/nvme[0-9]n[0-9]p[0-9]* /dev/vd[a-z] /dev/sd[a-z] /dev/xvd[a-z]; do
    [ -e "$dev" ] && printf '%s\\n' "$dev"
  done
}}
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24; do
  for candidate in $(root_candidate_paths); do
    [ -b "$candidate" ] && ROOT_DEVICE="$candidate" && break 2
  done
  sleep 0.25 2>/dev/null || true
done
mkdir -p /newroot
ROOT_MOUNTED=0
for candidate in $(root_candidate_paths); do
  [ -b "$candidate" ] || continue
  if mount -o rw "$candidate" /newroot 2>/tmp/conos-root-mount.err; then
    ROOT_DEVICE="$candidate"
    ROOT_MOUNTED=1
    break
  fi
done
if [ "$ROOT_MOUNTED" != "1" ]; then
  conos_trace "CONOS_INIT_WRAPPER_ROOT_MOUNT_FAILED root_device=$ROOT_DEVICE kernel_root=$KERNEL_ROOT candidates=$(root_candidate_paths | tr '\\n' ',')"
  cat /tmp/conos-root-mount.err >"$CONSOLE" 2>/dev/null || true
  exec /bin/sh
fi
conos_trace "CONOS_INIT_WRAPPER_ROOT_MOUNTED root_device=$ROOT_DEVICE"
mkdir -p /newroot/opt/conos /newroot/etc/systemd/system /newroot/etc/systemd/system/sysinit.target.wants
cp /conos/conos_guest_agent.py /newroot/opt/conos/conos_guest_agent.py
chmod 0755 /newroot/opt/conos/conos_guest_agent.py
cat >/newroot/opt/conos/conos_guest_agent_launcher.sh <<'LAUNCHER'
{_guest_agent_launcher_script(python_path=python_path, port=port).rstrip()}
LAUNCHER
chmod 0755 /newroot/opt/conos/conos_guest_agent_launcher.sh
cat >/newroot/etc/systemd/system/conos-guest-agent.service <<'UNIT'
{_guest_agent_systemd_unit(python_path=python_path, port=port).rstrip()}
UNIT
rm -f /newroot/etc/systemd/system/multi-user.target.wants/conos-guest-agent.service 2>/dev/null || true
ln -sf /etc/systemd/system/conos-guest-agent.service /newroot/etc/systemd/system/sysinit.target.wants/conos-guest-agent.service 2>/dev/null || true
conos_trace "CONOS_INIT_WRAPPER_AGENT_INSTALLED root_device=$ROOT_DEVICE"
if command -v switch_root >/dev/null 2>&1; then
  exec switch_root /newroot /sbin/init
fi
if command -v run-init >/dev/null 2>&1; then
  exec run-init /newroot /sbin/init
fi
exec chroot /newroot /sbin/init
"""


def build_managed_vm_guest_initrd_bundle(
    *,
    state_root: str = "",
    output_path: str = "",
    base_initrd_path: str = "",
    guest_agent_path: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    root_device: str = "/dev/vda",
    guest_python_path: str = "/usr/bin/python3",
    include_init_wrapper: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build a Con OS guest-agent initrd overlay artifact.

    The artifact is an auditable boot bundle. It configures guest-agent
    autostart, but runtime readiness is still proven only by the vsock
    handshake written to runtime.json.
    """

    output = Path(output_path).expanduser() if _clean(output_path) else managed_vm_guest_initrd_bundle_path(state_root)
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"managed VM guest initrd bundle already exists: {output}")
    agent = Path(_clean(guest_agent_path) or managed_vm_guest_agent_source_path()).expanduser().resolve()
    if not agent.exists() or not agent.is_file():
        raise FileNotFoundError(f"managed VM guest agent source does not exist: {agent}")
    base_initrd = Path(base_initrd_path).expanduser().resolve() if _clean(base_initrd_path) else None
    if base_initrd is not None and (not base_initrd.exists() or not base_initrd.is_file()):
        raise FileNotFoundError(f"managed VM base initrd source does not exist: {base_initrd}")
    port = int(guest_agent_port)
    entries: list[tuple[str, bytes, int]] = [
        ("conos/conos_guest_agent.py", agent.read_bytes(), 0o100755),
        (
            "conos/conos_guest_agent_launcher.sh",
            _guest_agent_launcher_script(python_path=_clean(guest_python_path) or "/usr/bin/python3", port=port).encode(
                "utf-8"
            ),
            0o100755,
        ),
        (
            "conos/install_guest_agent.sh",
            _guest_agent_install_script(python_path=_clean(guest_python_path) or "/usr/bin/python3", port=port).encode(
                "utf-8"
            ),
            0o100755,
        ),
        (
            "etc/systemd/system/conos-guest-agent.service",
            _guest_agent_systemd_unit(python_path=_clean(guest_python_path) or "/usr/bin/python3", port=port).encode(
                "utf-8"
            ),
            0o100644,
        ),
        (
            "conos/README.txt",
            (
                "Con OS guest-agent boot bundle.\n"
                "Runtime execution is trusted only after the host runner receives "
                "the guest_agent_ready vsock handshake.\n"
            ).encode("utf-8"),
            0o100644,
        ),
    ]
    initramfs_integration = "standalone_init_wrapper"
    if base_initrd is not None:
        for trace_stage in ("init-top", "local-top"):
            entries.append(
                (
                    f"scripts/{trace_stage}/conos-trace",
                    _guest_agent_initramfs_trace_hook(stage=trace_stage).encode("utf-8"),
                    0o100755,
                )
            )
        entries.append(
            (
                "scripts/local-bottom/conos-guest-agent",
                _guest_agent_initramfs_local_bottom_hook().encode("utf-8"),
                0o100755,
            )
        )
        initramfs_integration = "initramfs_tools_local_bottom_hook"
    if include_init_wrapper:
        entries.append(
            (
                "init",
                _guest_agent_init_wrapper(
                    root_device=_clean(root_device) or "/dev/vda",
                    python_path=_clean(guest_python_path) or "/usr/bin/python3",
                    port=port,
                ).encode("utf-8"),
                0o100755,
            )
        )
    base_initrd_compression = "none"
    initrd_merge_strategy = "standalone_gzip_overlay"
    compression_tool = "python_gzip"
    base_initrd_trailer_stripped = False
    base_initrd_entries_replaced: list[str] = []
    initramfs_local_bottom_order_installed = False
    initramfs_local_bottom_order_preserved = False
    initramfs_modules_installed = False
    initramfs_modules_preserved = False
    initramfs_modules_added: list[str] = []
    initramfs_trace_hooks_installed = False
    initramfs_trace_stages: list[str] = []
    overlay_archive = b""
    output_bytes = b""
    output.parent.mkdir(parents=True, exist_ok=True)
    if base_initrd is not None:
        base_bytes = base_initrd.read_bytes()
        base_initrd_compression = _initrd_compression_for_bytes(base_bytes)
        if base_initrd_compression in {"gzip", "zstd"}:
            base_archive, decompress_tool = _decompress_initrd_payload(
                base_initrd,
                compression=base_initrd_compression,
            )
            for trace_stage in ("init-top", "local-top"):
                trace_order, _ = _extend_initramfs_stage_order(
                    _read_newc_file(base_archive, f"scripts/{trace_stage}/ORDER"),
                    stage=trace_stage,
                    hook_name="conos-trace",
                )
                entries.append((f"scripts/{trace_stage}/ORDER", trace_order, 0o100644))
            initramfs_trace_hooks_installed = True
            initramfs_trace_stages = ["init-top", "local-top", "local-bottom"]
            local_bottom_order, initramfs_local_bottom_order_preserved = _extend_initramfs_local_bottom_order(
                _read_newc_file(base_archive, "scripts/local-bottom/ORDER")
            )
            entries.append(("scripts/local-bottom/ORDER", local_bottom_order, 0o100644))
            initramfs_local_bottom_order_installed = True
            modules_conf, initramfs_modules_preserved, initramfs_modules_added = _extend_initramfs_modules(
                _read_newc_file(base_archive, "conf/modules")
            )
            entries.append(("conf/modules", modules_conf, 0o100644))
            initramfs_modules_installed = True
            overlay_archive = _build_newc_archive(entries)
            base_archive, base_initrd_entries_replaced = _remove_newc_entries(
                base_archive,
                {name for name, _, _ in entries},
            )
            base_archive, base_initrd_trailer_stripped = _strip_final_newc_trailer(base_archive)
            output_bytes, compression_tool = _compress_initrd_payload(
                base_archive + overlay_archive,
                compression=base_initrd_compression,
            )
            initrd_merge_strategy = "decompress_append_cpio_recompress"
            compression_tool = f"{decompress_tool}/{compression_tool}"
        else:
            for trace_stage in ("init-top", "local-top"):
                trace_order, _ = _extend_initramfs_stage_order(b"", stage=trace_stage, hook_name="conos-trace")
                entries.append((f"scripts/{trace_stage}/ORDER", trace_order, 0o100644))
            initramfs_trace_hooks_installed = True
            initramfs_trace_stages = ["init-top", "local-top", "local-bottom"]
            entries.append(
                (
                    "scripts/local-bottom/ORDER",
                    _extend_initramfs_local_bottom_order(b"")[0],
                    0o100644,
                )
            )
            initramfs_local_bottom_order_installed = True
            modules_conf, initramfs_modules_preserved, initramfs_modules_added = _extend_initramfs_modules(b"")
            entries.append(("conf/modules", modules_conf, 0o100644))
            initramfs_modules_installed = True
            overlay_archive = _build_newc_archive(entries)
            output_bytes = base_bytes + gzip.compress(overlay_archive, compresslevel=9, mtime=0)
            initrd_merge_strategy = "append_compressed_overlay_fallback"
            compression_tool = "python_gzip"
        output.write_bytes(output_bytes)
    else:
        overlay_archive = _build_newc_archive(entries)
        output_bytes = gzip.compress(overlay_archive, compresslevel=9, mtime=0)
        output.write_bytes(output_bytes)
    if include_init_wrapper and "init-wrapper" not in initramfs_trace_stages:
        initramfs_trace_stages.append("init-wrapper")
    limitations = [
        "requires a Linux kernel supplied by the image registration path",
        "runtime readiness is not inferred from this artifact; it is proven by guest-agent handshake",
    ]
    if base_initrd is not None and include_init_wrapper:
        limitations.append(
            "when include_init_wrapper is enabled, Con OS replaces /init with a bounded wrapper and preserves the original initramfs hooks where possible"
        )
    elif base_initrd is not None:
        limitations.append(
            "when base_initrd_path is present, Con OS uses initramfs-tools hooks without replacing the guest's original /init"
        )
    if include_init_wrapper:
        limitations.append(
            "standalone init wrapper assumes the guest initrd/rootfs provides mount, cp, mkdir, switch_root or run-init, and a compatible root block device"
        )
    manifest_path = Path(f"{output}.manifest.json")
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_guest_initrd_bundle",
        "status": "BUILT",
        "state_root": managed_vm_state_root(state_root),
        "output_path": str(output),
        "output_sha256": _sha256_file(output),
        "output_byte_size": output.stat().st_size,
        "base_initrd_path": str(base_initrd or ""),
        "base_initrd_present": bool(base_initrd),
        "base_initrd_sha256": _sha256_file(base_initrd) if base_initrd is not None else "",
        "overlay_format": "newc_cpio",
        "overlay_byte_size": len(overlay_archive),
        "overlay_archive_byte_size": len(overlay_archive),
        "base_initrd_compression": base_initrd_compression,
        "base_initrd_trailer_stripped": base_initrd_trailer_stripped,
        "base_initrd_entries_replaced": base_initrd_entries_replaced,
        "initrd_merge_strategy": initrd_merge_strategy,
        "initrd_compression_tool": compression_tool,
        "guest_agent_path": str(agent),
        "guest_agent_sha256": _sha256_file(agent),
        "guest_agent_port": port,
        "guest_agent_transport": "virtio-vsock",
        "guest_agent_autostart_configured": True,
        "include_init_wrapper": bool(include_init_wrapper),
        "initramfs_integration": initramfs_integration,
        "initramfs_local_bottom_hook": base_initrd is not None,
        "initramfs_local_bottom_order_installed": initramfs_local_bottom_order_installed,
        "initramfs_local_bottom_order_preserved": initramfs_local_bottom_order_preserved,
        "initramfs_modules_installed": initramfs_modules_installed,
        "initramfs_modules_preserved": initramfs_modules_preserved,
        "initramfs_modules_added": initramfs_modules_added,
        "initramfs_trace_hooks_installed": initramfs_trace_hooks_installed,
        "initramfs_trace_stages": initramfs_trace_stages,
        "root_device": _clean(root_device) or "/dev/vda",
        "guest_python_path": _clean(guest_python_path) or "/usr/bin/python3",
        "files": [name for name, _, _ in entries],
        "created_at": _now_iso(),
        "owned_by_conos": True,
        "no_host_fallback": True,
        "readiness_contract": "runtime.json guest_agent_ready=true and execution_ready=true after vsock handshake",
        "runtime_verified": False,
        "limitations": limitations,
    }
    _write_json(manifest_path, payload)
    return payload


def build_managed_vm_linux_base_image(
    *,
    state_root: str = "",
    image_id: str = "",
    source_disk_path: str = "",
    kernel_path: str = "",
    base_initrd_path: str = "",
    guest_agent_path: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    kernel_command_line: str = "",
    root_device: str = "/dev/vda",
    guest_python_path: str = "/usr/bin/python3",
    include_init_wrapper: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Build and register a Con OS managed Linux base image when boot artifacts exist."""

    image = managed_vm_image_id(image_id)
    root = managed_vm_state_root(state_root)
    required_inputs = {
        "source_disk_path": source_disk_path,
        "kernel_path": kernel_path,
    }
    missing_fields = [name for name, value in required_inputs.items() if not _clean(value)]
    missing_paths: list[str] = []
    source_disk = Path(source_disk_path).expanduser().resolve() if _clean(source_disk_path) else None
    kernel = Path(kernel_path).expanduser().resolve() if _clean(kernel_path) else None
    base_initrd = Path(base_initrd_path).expanduser().resolve() if _clean(base_initrd_path) else None
    for candidate in (source_disk, kernel, base_initrd):
        if candidate is not None and (not candidate.exists() or not candidate.is_file()):
            missing_paths.append(str(candidate))
    blocked_payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "build_linux_base_image",
        "status": "BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS",
        "state_root": root,
        "image_id": image,
        "source_disk_path": str(source_disk or ""),
        "kernel_path": str(kernel or ""),
        "base_initrd_path": str(base_initrd or ""),
        "guest_agent_port": int(guest_agent_port),
        "guest_agent_transport": "virtio-vsock",
        "guest_agent_autostart_configured": False,
        "requires_user_configured_vm": False,
        "requires_boot_artifacts": True,
        "required_inputs": [
            "source_disk_path: bootable Linux root disk image",
            "kernel_path: Linux kernel usable by Apple Virtualization VZLinuxBootLoader",
            "base_initrd_path: optional existing initrd to append the Con OS guest-agent bundle",
        ],
        "missing_fields": missing_fields,
        "missing_paths": missing_paths,
        "no_host_fallback": True,
        "reason": "source disk and kernel are required before Con OS can build a bootable managed base image",
    }
    if missing_fields or missing_paths:
        return blocked_payload
    assert source_disk is not None
    assert kernel is not None
    detected_root = _detect_linux_root_partition_from_disk(source_disk)
    detected_root_device = str(detected_root.get("root_device") or "")
    detected_root_boot_spec = str(detected_root.get("root_boot_spec") or "")
    requested_root_device = _clean(root_device)
    selected_root_device = (
        detected_root_device
        if requested_root_device in {"", "/dev/vda"} and detected_root_device
        else requested_root_device or "/dev/vda"
    )
    selected_root_boot_spec = (
        detected_root_boot_spec
        if requested_root_device in {"", "/dev/vda"} and detected_root_boot_spec
        else selected_root_device
    )
    initrd_output = managed_vm_image_guest_initrd_path(root, image)
    guest_bundle = build_managed_vm_guest_initrd_bundle(
        state_root=root,
        output_path=str(initrd_output),
        base_initrd_path=str(base_initrd or ""),
        guest_agent_path=guest_agent_path,
        guest_agent_port=int(guest_agent_port),
        root_device=selected_root_device,
        guest_python_path=guest_python_path,
        include_init_wrapper=bool(include_init_wrapper),
        overwrite=True,
    )
    command_line = _clean(kernel_command_line) or (
        f"console=hvc0 root={selected_root_boot_spec} rw rootwait "
        f"conos.agent=vsock:{int(guest_agent_port)} conos.root={selected_root_device}"
    )
    registered = register_managed_vm_linux_boot_image(
        state_root=root,
        image_id=image,
        source_disk_path=str(source_disk),
        kernel_path=str(kernel),
        initrd_path=str(initrd_output),
        kernel_command_line=command_line,
        guest_agent_port=int(guest_agent_port),
    )
    registered.update(
        {
            "built_by_conos_base_image_builder": True,
            "base_image_builder_status": "BUILT",
            "guest_initrd_bundle": guest_bundle,
            "guest_agent_autostart_configured": bool(guest_bundle.get("guest_agent_autostart_configured", False)),
            "guest_agent_installation_mode": "initrd_autostart_bundle",
            "guest_agent_installation_status": "INITRD_AUTOSTART_BUNDLE_CONFIGURED",
            "verified_execution_path": "linux_direct_initrd_guest_agent_bundle",
            "next_required_step": "start-instance then wait for guest-agent runtime handshake",
        }
    )
    _write_json(managed_vm_image_manifest_path(root, image), registered)
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "build_linux_base_image",
        "status": "BUILT",
        "state_root": root,
        "image_id": image,
        "source_disk_path": str(source_disk),
        "kernel_path": str(kernel),
        "base_initrd_path": str(base_initrd or ""),
        "requested_root_device": requested_root_device or "/dev/vda",
        "detected_root_device": detected_root_device,
        "detected_root_partition_uuid": str(detected_root.get("root_partition_uuid") or ""),
        "detected_root_filesystem_uuid": str(detected_root.get("root_filesystem_uuid") or ""),
        "root_boot_spec": selected_root_boot_spec,
        "root_device": selected_root_device,
        "initrd_path": str(initrd_output),
        "guest_initrd_bundle": guest_bundle,
        "image_manifest": registered,
        "image_manifest_path": str(managed_vm_image_manifest_path(root, image)),
        "guest_agent_autostart_configured": bool(guest_bundle.get("guest_agent_autostart_configured", False)),
        "guest_agent_installation_mode": "initrd_autostart_bundle",
        "guest_agent_installation_status": "INITRD_AUTOSTART_BUNDLE_CONFIGURED",
        "verified_execution_path": "linux_direct_initrd_guest_agent_bundle",
        "guest_agent_verified": False,
        "execution_ready": False,
        "requires_user_configured_vm": False,
        "no_host_fallback": True,
        "readiness_contract": "start-instance must observe guest_agent_ready=true and execution_ready=true in runtime.json",
    }


def _bundle_artifact_entry(source: Path, destination: Path, *, relative_path: str) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": relative_path,
        "sha256": _sha256_file(destination),
        "byte_size": destination.stat().st_size,
        "source_path": str(source),
    }


def create_managed_vm_base_image_bundle(
    *,
    state_root: str = "",
    image_id: str = "",
    output_dir: str = "",
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a self-contained, digest-pinned Con OS base-image bundle directory."""

    image = managed_vm_image_id(image_id)
    root = managed_vm_state_root(state_root)
    image_manifest = load_managed_vm_image_manifest(root, image)
    if not image_manifest:
        return {
            "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
            "operation": "bundle_base_image",
            "status": "BUNDLE_BLOCKED_IMAGE_UNAVAILABLE",
            "state_root": root,
            "image_id": image,
            "reason": "managed VM base image is not registered",
            "no_host_fallback": True,
        }
    if str(image_manifest.get("boot_mode") or "") != "linux_direct":
        return {
            "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
            "operation": "bundle_base_image",
            "status": "BUNDLE_BLOCKED_UNSUPPORTED_BOOT_MODE",
            "state_root": root,
            "image_id": image,
            "boot_mode": str(image_manifest.get("boot_mode") or ""),
            "reason": "base-image bundles currently require linux_direct boot artifacts",
            "no_host_fallback": True,
        }
    bundle_capability = (
        image_manifest.get("guest_initrd_bundle_capability")
        if isinstance(image_manifest.get("guest_initrd_bundle_capability"), dict)
        else {}
    )
    if str(bundle_capability.get("status") or "") != "VERIFIED":
        return {
            "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
            "operation": "bundle_base_image",
            "status": "BUNDLE_BLOCKED_UNVERIFIED_INITRD_BUNDLE",
            "state_root": root,
            "image_id": image,
            "guest_initrd_bundle_capability": bundle_capability,
            "reason": "image initrd does not expose a verified Con OS guest-agent bundle",
            "no_host_fallback": True,
        }
    release_eligible = bool(
        image_manifest.get("boot_verified", False)
        and image_manifest.get("guest_agent_verified", False)
        and image_manifest.get("bootstrap_verified", False)
    )
    if not release_eligible and not allow_unverified:
        return {
            "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
            "operation": "bundle_base_image",
            "status": "BUNDLE_BLOCKED_UNVERIFIED_IMAGE",
            "state_root": root,
            "image_id": image,
            "release_eligible": False,
            "reason": "image must pass bootstrap verification before it can be bundled for release",
            "required_verified_fields": ["boot_verified", "guest_agent_verified", "bootstrap_verified"],
            "no_host_fallback": True,
        }
    disk = Path(str(image_manifest.get("disk_path") or managed_vm_base_image_path(root, image))).expanduser()
    kernel = Path(str(image_manifest.get("kernel_path") or managed_vm_kernel_path(root, image))).expanduser()
    initrd = Path(str(image_manifest.get("initrd_path") or managed_vm_initrd_path(root, image))).expanduser()
    missing = [str(path) for path in (disk, kernel, initrd) if not path.exists() or not path.is_file()]
    if missing:
        return {
            "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
            "operation": "bundle_base_image",
            "status": "BUNDLE_BLOCKED_MISSING_ARTIFACTS",
            "state_root": root,
            "image_id": image,
            "missing_paths": missing,
            "reason": "one or more boot artifacts are missing",
            "no_host_fallback": True,
        }
    output = Path(output_dir).expanduser() if _clean(output_dir) else managed_vm_base_image_bundle_root(root, image)
    output = output.resolve()
    if output.exists():
        if not overwrite:
            return {
                "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
                "operation": "bundle_base_image",
                "status": "BUNDLE_BLOCKED_OUTPUT_EXISTS",
                "state_root": root,
                "image_id": image,
                "output_dir": str(output),
                "reason": "bundle output directory already exists; pass overwrite to replace it",
                "no_host_fallback": True,
            }
        shutil.rmtree(output)
    artifacts_dir = output / "artifacts"
    artifacts = {
        "source_disk": _bundle_artifact_entry(disk, artifacts_dir / "disk.img", relative_path="artifacts/disk.img"),
        "kernel": _bundle_artifact_entry(kernel, artifacts_dir / "vmlinuz", relative_path="artifacts/vmlinuz"),
        "initrd": _bundle_artifact_entry(initrd, artifacts_dir / "initrd.img", relative_path="artifacts/initrd.img"),
    }
    initrd_sidecar = Path(f"{initrd}.manifest.json")
    initrd_sidecar_payload: Dict[str, Any] = {}
    if initrd_sidecar.exists() and initrd_sidecar.is_file():
        sidecar_destination = artifacts_dir / "initrd.img.manifest.json"
        shutil.copy2(initrd_sidecar, sidecar_destination)
        initrd_sidecar_payload = _read_json(sidecar_destination)
        artifacts["initrd_manifest"] = {
            "path": "artifacts/initrd.img.manifest.json",
            "sha256": _sha256_file(sidecar_destination),
            "byte_size": sidecar_destination.stat().st_size,
            "source_path": str(initrd_sidecar),
        }
    image_manifest_destination = output / MANAGED_VM_BASE_IMAGE_MANIFEST
    _write_json(image_manifest_destination, image_manifest)
    artifacts["image_manifest"] = {
        "path": MANAGED_VM_BASE_IMAGE_MANIFEST,
        "sha256": _sha256_file(image_manifest_destination),
        "byte_size": image_manifest_destination.stat().st_size,
        "source_path": str(managed_vm_image_manifest_path(root, image)),
    }
    recipe = {
        "schema_version": MANAGED_VM_ARTIFACT_RECIPE_VERSION,
        "recipe_id": f"{image}-bundle",
        "status": "READY",
        "image_id": image,
        "boot_mode": "linux_direct",
        "guest_agent_installation_mode": "initrd_autostart_bundle",
        "guest_agent_installation_status": str(image_manifest.get("guest_agent_installation_status") or ""),
        "verified_execution_path": str(image_manifest.get("verified_execution_path") or ""),
        "guest_agent_port": image_manifest.get("guest_agent_port"),
        "root_device": str(image_manifest.get("root_device") or ""),
        "kernel_command_line": str(image_manifest.get("kernel_command_line") or ""),
        "guest_python_path": str(image_manifest.get("guest_python_path") or "/usr/bin/python3"),
        "artifacts": {
            "source_disk": {"path": artifacts["source_disk"]["path"], "sha256": artifacts["source_disk"]["sha256"]},
            "kernel": {"path": artifacts["kernel"]["path"], "sha256": artifacts["kernel"]["sha256"]},
            "initrd": {"path": artifacts["initrd"]["path"], "sha256": artifacts["initrd"]["sha256"]},
        },
        "bundle_manifest": MANAGED_VM_BASE_IMAGE_BUNDLE_MANIFEST,
        "created_at": _now_iso(),
        "no_host_fallback": True,
    }
    recipe_path = output / MANAGED_VM_BASE_IMAGE_BUNDLE_RECIPE
    _write_json(recipe_path, recipe)
    artifacts["recipe"] = {
        "path": MANAGED_VM_BASE_IMAGE_BUNDLE_RECIPE,
        "sha256": _sha256_file(recipe_path),
        "byte_size": recipe_path.stat().st_size,
    }
    manifest_payload = {
        "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
        "artifact_type": "conos_managed_vm_base_image_bundle",
        "operation": "bundle_base_image",
        "status": "BUNDLED" if release_eligible else "BUNDLED_UNVERIFIED",
        "state_root": root,
        "image_id": image,
        "output_dir": str(output),
        "manifest_path": str(output / MANAGED_VM_BASE_IMAGE_BUNDLE_MANIFEST),
        "recipe_path": str(recipe_path),
        "recipe_relative_path": MANAGED_VM_BASE_IMAGE_BUNDLE_RECIPE,
        "boot_mode": "linux_direct",
        "verified_execution_path": str(image_manifest.get("verified_execution_path") or ""),
        "guest_agent_installation_mode": str(image_manifest.get("guest_agent_installation_mode") or ""),
        "guest_initrd_bundle_capability": bundle_capability,
        "initrd_sidecar_manifest": initrd_sidecar_payload,
        "release_eligible": release_eligible,
        "allow_unverified": bool(allow_unverified),
        "artifacts": artifacts,
        "created_at": _now_iso(),
        "readiness_contract": "bundle recipe must bootstrap to guest_agent_ready=true and execution_ready=true before release use",
        "no_host_fallback": True,
    }
    _write_json(output / MANAGED_VM_BASE_IMAGE_BUNDLE_MANIFEST, manifest_payload)
    return manifest_payload


def _managed_vm_bundle_status(
    status: str,
    *,
    state_root: str,
    bundle_dir: str,
    image_id: str = "",
    reason: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
        "operation": "install_base_image_bundle",
        "status": status,
        "state_root": state_root,
        "bundle_dir": bundle_dir,
        "image_id": image_id,
        "no_host_fallback": True,
    }
    if reason:
        payload["reason"] = reason
    payload.update(extra)
    return payload


def _resolve_bundle_member(bundle_root: Path, relative_path: str) -> Path | None:
    selected = _clean(relative_path)
    if not selected:
        return None
    candidate = Path(selected)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        return None
    resolved = (bundle_root / candidate).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError:
        return None
    return resolved


def _validate_bundle_artifacts(
    *,
    bundle_root: Path,
    artifacts: Dict[str, Any],
    required_keys: Sequence[str],
) -> tuple[list[Dict[str, Any]], list[str], list[Dict[str, Any]], Dict[str, Path]]:
    verified: list[Dict[str, Any]] = []
    missing: list[str] = []
    mismatches: list[Dict[str, Any]] = []
    paths: Dict[str, Path] = {}
    for key in required_keys:
        entry = artifacts.get(key)
        if not isinstance(entry, dict):
            missing.append(key)
            continue
        path = _resolve_bundle_member(bundle_root, str(entry.get("path") or ""))
        expected_sha = _clean(entry.get("sha256"))
        expected_size = entry.get("byte_size")
        if path is None:
            mismatches.append({"artifact": key, "reason": "unsafe_or_empty_path", "path": str(entry.get("path") or "")})
            continue
        paths[key] = path
        if not path.exists() or not path.is_file():
            missing.append(str(entry.get("path") or key))
            continue
        actual_sha = _sha256_file(path)
        actual_size = path.stat().st_size
        size_matches = True
        if expected_size is not None:
            try:
                size_matches = actual_size == int(expected_size)
            except (TypeError, ValueError):
                size_matches = False
        if expected_sha and actual_sha != expected_sha:
            mismatches.append(
                {
                    "artifact": key,
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )
            continue
        if not size_matches:
            mismatches.append(
                {
                    "artifact": key,
                    "path": str(path),
                    "expected_byte_size": expected_size,
                    "actual_byte_size": actual_size,
                }
            )
            continue
        verified.append({"artifact": key, "path": str(path), "sha256": actual_sha, "byte_size": actual_size})
    return verified, missing, mismatches, paths


def install_managed_vm_base_image_bundle(
    *,
    state_root: str = "",
    bundle_dir: str = "",
    image_id: str = "",
    allow_unverified: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Install a self-contained Con OS base-image bundle into local VM state."""

    root = managed_vm_state_root(state_root)
    bundle = Path(bundle_dir).expanduser().resolve() if _clean(bundle_dir) else Path()
    if not _clean(bundle_dir) or not bundle.exists() or not bundle.is_dir():
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_BUNDLE_UNAVAILABLE",
            state_root=root,
            bundle_dir=str(bundle) if _clean(bundle_dir) else "",
            reason="base-image bundle directory does not exist",
        )
    manifest_path = bundle / MANAGED_VM_BASE_IMAGE_BUNDLE_MANIFEST
    manifest = _read_json(manifest_path)
    if not manifest:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_MANIFEST_MISSING",
            state_root=root,
            bundle_dir=str(bundle),
            reason="bundle manifest is missing or invalid JSON",
        )
    if str(manifest.get("schema_version") or "") != MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_SCHEMA_MISMATCH",
            state_root=root,
            bundle_dir=str(bundle),
            reason="bundle manifest schema_version is not supported",
            observed_schema_version=str(manifest.get("schema_version") or ""),
            expected_schema_version=MANAGED_VM_BASE_IMAGE_BUNDLE_VERSION,
        )
    release_eligible = bool(manifest.get("release_eligible", False))
    if not release_eligible and not allow_unverified:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_UNVERIFIED_BUNDLE",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            release_eligible=False,
            reason="bundle is not release eligible; pass allow_unverified only for development installs",
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_MANIFEST_MISSING",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            reason="bundle manifest does not contain an artifacts map",
        )
    required = ["source_disk", "kernel", "initrd", "initrd_manifest", "image_manifest", "recipe"]
    verified_artifacts, missing, mismatches, artifact_paths = _validate_bundle_artifacts(
        bundle_root=bundle,
        artifacts=artifacts,
        required_keys=required,
    )
    if missing:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_MISSING_ARTIFACTS",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            missing_artifacts=missing,
            verified_artifacts=verified_artifacts,
            reason="one or more bundle artifacts are missing",
        )
    if mismatches:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_DIGEST_MISMATCH",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            digest_mismatches=mismatches,
            verified_artifacts=verified_artifacts,
            reason="one or more bundle artifacts failed sha256 or byte-size validation",
        )
    recipe = _read_json(artifact_paths["recipe"])
    if str(recipe.get("schema_version") or "") != MANAGED_VM_ARTIFACT_RECIPE_VERSION:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_SCHEMA_MISMATCH",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            reason="bundle recipe schema_version is not supported",
            observed_recipe_schema_version=str(recipe.get("schema_version") or ""),
            expected_recipe_schema_version=MANAGED_VM_ARTIFACT_RECIPE_VERSION,
        )
    recipe_artifacts = _recipe_artifacts(recipe)
    _, recipe_missing, recipe_mismatches, _ = _validate_bundle_artifacts(
        bundle_root=artifact_paths["recipe"].parent.resolve(),
        artifacts=recipe_artifacts,
        required_keys=["source_disk", "kernel", "initrd"],
    )
    if recipe_missing or recipe_mismatches:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_DIGEST_MISMATCH",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            recipe_missing_artifacts=recipe_missing,
            recipe_digest_mismatches=recipe_mismatches,
            reason="bundle recipe artifacts do not resolve to digest-matching files",
        )
    source_image_manifest = _read_json(artifact_paths["image_manifest"])
    if not source_image_manifest:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_MANIFEST_MISSING",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            reason="bundle image manifest is missing or invalid JSON",
        )
    try:
        expected_port = int(
            source_image_manifest.get("guest_agent_port")
            or recipe.get("guest_agent_port")
            or DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT
        )
    except (TypeError, ValueError):
        expected_port = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT
    initrd_sidecar = _read_json(artifact_paths["initrd_manifest"])
    bundle_capability = _managed_vm_guest_initrd_bundle_capability(initrd_sidecar, expected_port=expected_port)
    bundle_verified = bool(bundle_capability.get("verified", False))
    if not bundle_verified and not allow_unverified:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_UNVERIFIED_BUNDLE",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=str(manifest.get("image_id") or ""),
            guest_initrd_bundle_capability=bundle_capability,
            reason="bundle initrd sidecar does not validate the Con OS guest-agent bundle",
        )
    selected_image_id = _clean(image_id) or _clean(manifest.get("image_id")) or _clean(source_image_manifest.get("image_id"))
    image = managed_vm_image_id(selected_image_id)
    image_root = managed_vm_image_root(root, image)
    disk_dest = Path(managed_vm_base_image_path(root, image))
    kernel_dest = managed_vm_kernel_path(root, image)
    initrd_dest = managed_vm_initrd_path(root, image)
    initrd_sidecar_dest = Path(f"{initrd_dest}.manifest.json")
    image_manifest_dest = managed_vm_image_manifest_path(root, image)
    destinations = [disk_dest, kernel_dest, initrd_dest, initrd_sidecar_dest, image_manifest_dest]
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not overwrite:
        return _managed_vm_bundle_status(
            "INSTALL_BLOCKED_OUTPUT_EXISTS",
            state_root=root,
            bundle_dir=str(bundle),
            image_id=image,
            existing_paths=existing,
            reason="managed VM image artifacts already exist; pass overwrite to replace them",
        )
    image_root.mkdir(parents=True, exist_ok=True)
    copied = {
        "disk": _copy_file_efficient(artifact_paths["source_disk"], disk_dest),
        "kernel": _copy_file_efficient(artifact_paths["kernel"], kernel_dest),
        "initrd": _copy_file_efficient(artifact_paths["initrd"], initrd_dest),
        "initrd_manifest": _copy_file_efficient(artifact_paths["initrd_manifest"], initrd_sidecar_dest),
    }
    installed_manifest = dict(source_image_manifest)
    installed_manifest.update(
        {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "artifact_type": "managed_vm_base_image",
            "status": "REGISTERED",
            "image_id": image,
            "state_root": root,
            "source_disk_path": str(artifact_paths["source_disk"]),
            "disk_path": str(disk_dest),
            "sha256": _sha256_file(disk_dest),
            "byte_size": disk_dest.stat().st_size,
            "kernel_source_path": str(artifact_paths["kernel"]),
            "kernel_path": str(kernel_dest),
            "kernel_sha256": _sha256_file(kernel_dest),
            "kernel_byte_size": kernel_dest.stat().st_size,
            "initrd_path": str(initrd_dest),
            "initrd_sha256": _sha256_file(initrd_dest),
            "initrd_byte_size": initrd_dest.stat().st_size,
            "initrd_present": True,
            "guest_initrd_bundle_manifest_path": str(initrd_sidecar_dest),
            "guest_initrd_bundle_manifest_present": True,
            "guest_initrd_bundle_status": str(initrd_sidecar.get("status") or ""),
            "guest_initrd_bundle_capability": bundle_capability,
            "guest_agent_autostart_configured": bundle_verified,
            "guest_agent_bundle_files": list(initrd_sidecar.get("files") or []) if isinstance(initrd_sidecar.get("files"), list) else [],
            "boot_mode": "linux_direct",
            "bootable": True,
            "owned_by_conos": True,
            "guest_agent_transport": "virtio-vsock",
            "guest_agent_port": expected_port,
            "guest_agent_installation_mode": "initrd_autostart_bundle" if bundle_verified else "external_or_preinstalled",
            "guest_agent_installation_status": (
                "INITRD_AUTOSTART_BUNDLE_CONFIGURED" if bundle_verified else "INITRD_BUNDLE_UNVERIFIED"
            ),
            "verified_execution_path": "linux_direct_initrd_guest_agent_bundle" if bundle_verified else "",
            "installed_from_bundle": True,
            "bundle_manifest_path": str(manifest_path),
            "bundle_source_dir": str(bundle),
            "bundle_release_eligible": release_eligible,
            "installed_at": _now_iso(),
            "no_host_fallback": True,
            "next_required_step": "start-instance then wait for guest agent readiness",
        }
    )
    _write_json(image_manifest_dest, installed_manifest)
    return _managed_vm_bundle_status(
        "INSTALLED",
        state_root=root,
        bundle_dir=str(bundle),
        image_id=image,
        release_eligible=release_eligible,
        allow_unverified=bool(allow_unverified),
        image_manifest_path=str(image_manifest_dest),
        installed_artifacts={
            "disk": {"path": str(disk_dest), "sha256": _sha256_file(disk_dest), "byte_size": disk_dest.stat().st_size},
            "kernel": {"path": str(kernel_dest), "sha256": _sha256_file(kernel_dest), "byte_size": kernel_dest.stat().st_size},
            "initrd": {"path": str(initrd_dest), "sha256": _sha256_file(initrd_dest), "byte_size": initrd_dest.stat().st_size},
            "initrd_manifest": {
                "path": str(initrd_sidecar_dest),
                "sha256": _sha256_file(initrd_sidecar_dest),
                "byte_size": initrd_sidecar_dest.stat().st_size,
            },
        },
        copied_artifacts=copied,
        guest_initrd_bundle_capability=bundle_capability,
        verified_execution_path=str(installed_manifest.get("verified_execution_path") or ""),
        readiness_contract="start-instance must observe guest_agent_ready=true and execution_ready=true before agent-exec is allowed",
    )


def _recipe_artifacts(recipe: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = recipe.get("artifacts")
    return dict(artifacts) if isinstance(artifacts, dict) else {}


def _load_managed_vm_recipe_registry() -> Dict[str, Any]:
    return _read_json(Path(managed_vm_recipe_registry_path()))


def _builtin_recipe_entries() -> list[Dict[str, Any]]:
    registry = _load_managed_vm_recipe_registry()
    recipes = registry.get("recipes")
    return [dict(item) for item in recipes if isinstance(item, dict)] if isinstance(recipes, list) else []


def managed_vm_recipe_report(recipe_id: str = "") -> Dict[str, Any]:
    registry_path = Path(managed_vm_recipe_registry_path())
    registry = _load_managed_vm_recipe_registry()
    entries = _builtin_recipe_entries()
    selected = _clean(recipe_id)
    if selected:
        entries = [entry for entry in entries if _clean(entry.get("id")) == selected]
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "recipe_report",
        "status": "AVAILABLE" if registry else "UNAVAILABLE",
        "recipe_registry_path": str(registry_path),
        "recipe_registry_present": registry_path.exists(),
        "default_recipe_id": _clean(registry.get("default_recipe_id")) if registry else "",
        "recipes": entries,
        "selected_recipe_id": selected,
        "no_host_fallback": True,
    }


def _resolve_managed_vm_recipe_reference(recipe_path: str) -> tuple[str, Dict[str, Any]]:
    raw = _clean(recipe_path)
    if raw.startswith("builtin:") or raw == "builtin":
        recipe_id = raw.split(":", 1)[1] if ":" in raw else ""
        registry = _load_managed_vm_recipe_registry()
        if not recipe_id:
            recipe_id = _clean(registry.get("default_recipe_id"))
        for entry in _builtin_recipe_entries():
            if _clean(entry.get("id")) != recipe_id:
                continue
            relative = _clean(entry.get("path"))
            path = Path(relative)
            if not path.is_absolute():
                path = Path(managed_vm_recipe_dir_path()) / relative
            metadata = dict(entry)
            metadata["recipe_id"] = recipe_id
            metadata["recipe_reference"] = raw
            return str(path.resolve()), metadata
        return raw, {
            "recipe_id": recipe_id,
            "recipe_reference": raw,
            "status": "RECIPE_UNKNOWN",
            "reason": f"builtin recipe was not found: {recipe_id}",
        }
    return raw, {}


def _artifact_spec_value(spec: Dict[str, Any], *names: str) -> str:
    for name in names:
        value = _clean(spec.get(name))
        if value:
            return value
    return ""


def _artifact_expected_digest(spec: Dict[str, Any]) -> tuple[str, str]:
    for algorithm in ("sha256", "sha512"):
        value = _clean(spec.get(algorithm))
        if value:
            return algorithm, value
    digest = _clean(spec.get("digest"))
    if ":" in digest:
        algorithm, value = digest.split(":", 1)
        algorithm = algorithm.lower()
        if algorithm in {"sha256", "sha512"} and value:
            return algorithm, value
    return "", ""


def _digest_matches(path: Path, algorithm: str, expected: str) -> tuple[bool, str, str]:
    actual_hex, actual_b64 = _file_digest(path, algorithm)
    normalized = _clean(expected).removeprefix(f"{algorithm}:").strip()
    if normalized.lower() == actual_hex.lower():
        return True, actual_hex, actual_b64
    if normalized.rstrip("=") == actual_b64.rstrip("="):
        return True, actual_hex, actual_b64
    return False, actual_hex, actual_b64


def _digest_cache_key(algorithm: str, expected: str) -> str:
    normalized = _clean(expected).removeprefix(f"{algorithm}:").strip()
    if normalized and all(char in "0123456789abcdefABCDEF" for char in normalized):
        return normalized.lower()
    return hashlib.sha256(f"{algorithm}:{normalized}".encode("utf-8")).hexdigest()


def _artifact_filename(name: str, spec: Dict[str, Any], source: str) -> str:
    explicit = _clean(spec.get("filename"))
    if explicit:
        return Path(explicit).name
    parsed = urlparse(source)
    if parsed.scheme == "file":
        selected = Path(unquote(parsed.path)).name
    elif parsed.scheme:
        selected = Path(unquote(parsed.path)).name
    else:
        selected = Path(source).name
    return selected or f"{name}.artifact"


def _artifact_spec_with_recipe_base(spec: Dict[str, Any], recipe_dir: Path) -> Dict[str, Any]:
    selected = dict(spec)
    for key in ("url", "uri", "path"):
        source = _clean(selected.get(key))
        if not source:
            continue
        parsed = urlparse(source)
        if parsed.scheme:
            return selected
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            source_path = recipe_dir / source_path
        selected[key] = str(source_path.resolve())
        return selected
    return selected


def _artifact_cache_path(*, state_root: str, sha256: str, filename: str) -> Path:
    digest = _clean(sha256).lower()
    safe_name = Path(filename).name or "artifact.bin"
    return managed_vm_artifact_cache_root(state_root) / digest[:2] / digest / safe_name


def _artifact_cache_path_for_digest(*, state_root: str, algorithm: str, expected: str, filename: str) -> Path:
    digest = _digest_cache_key(algorithm, expected)
    safe_name = Path(filename).name or "artifact.bin"
    return managed_vm_artifact_cache_root(state_root) / algorithm / digest[:2] / digest / safe_name


def _artifact_source_is_local(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"", "file"}


def _artifact_source_local_path(source: str) -> Path:
    parsed = urlparse(source)
    raw = unquote(parsed.path) if parsed.scheme == "file" else source
    return Path(raw).expanduser().resolve()


def _pinned_artifact_spec(
    *,
    name: str,
    source: str,
    sha256: str = "",
    sha512: str = "",
    filename: str = "",
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    selected_source = _clean(source)
    if not selected_source:
        return {}, {
            "name": name,
            "status": "MISSING_SOURCE",
            "reason": f"{name} source was not provided",
        }
    selected_sha256 = _clean(sha256)
    selected_sha512 = _clean(sha512)
    parsed = urlparse(selected_source)
    local_path = _artifact_source_local_path(selected_source) if _artifact_source_is_local(selected_source) else None
    if local_path is not None:
        if not local_path.exists() or not local_path.is_file():
            return {}, {
                "name": name,
                "status": "SOURCE_MISSING",
                "reason": f"{name} source does not exist: {local_path}",
                "source": selected_source,
            }
        actual_sha256 = _sha256_file(local_path)
        actual_sha512, _ = _file_digest(local_path, "sha512")
        if selected_sha256 and selected_sha256.lower() != actual_sha256.lower():
            return {}, {
                "name": name,
                "status": "DIGEST_MISMATCH",
                "reason": f"{name} sha256 does not match local source",
                "source": str(local_path),
                "expected_sha256": selected_sha256,
                "actual_sha256": actual_sha256,
            }
        if selected_sha512 and selected_sha512.lower() != actual_sha512.lower():
            return {}, {
                "name": name,
                "status": "DIGEST_MISMATCH",
                "reason": f"{name} sha512 does not match local source",
                "source": str(local_path),
                "expected_sha512": selected_sha512,
                "actual_sha512": actual_sha512,
            }
        selected_sha256 = selected_sha256 or actual_sha256
        selected_source = local_path.as_uri()
        selected_filename = Path(filename).name if _clean(filename) else local_path.name
        return {
            "url": selected_source,
            "sha256": selected_sha256,
            "filename": selected_filename,
        }, {
            "name": name,
            "status": "PINNED",
            "source": str(local_path),
            "url": selected_source,
            "sha256": selected_sha256,
            "sha512": selected_sha512 or actual_sha512,
            "filename": selected_filename,
            "digest_source": "computed_from_local_source" if not _clean(sha256) else "provided_and_verified",
        }
    if parsed.scheme not in {"http", "https"}:
        return {}, {
            "name": name,
            "status": "UNSUPPORTED_SOURCE",
            "reason": f"unsupported artifact source scheme: {parsed.scheme}",
            "source": selected_source,
        }
    if not selected_sha256 and not selected_sha512:
        return {}, {
            "name": name,
            "status": "MISSING_DIGEST",
            "reason": f"{name} remote source requires explicit sha256 or sha512",
            "source": selected_source,
        }
    selected_filename = Path(filename).name if _clean(filename) else _artifact_filename(name, {}, selected_source)
    spec: Dict[str, Any] = {"url": selected_source, "filename": selected_filename}
    if selected_sha256:
        spec["sha256"] = selected_sha256
    if selected_sha512:
        spec["sha512"] = selected_sha512
    return spec, {
        "name": name,
        "status": "PINNED",
        "source": selected_source,
        "sha256": selected_sha256,
        "sha512": selected_sha512,
        "filename": selected_filename,
        "digest_source": "provided",
    }


def create_managed_vm_pinned_artifact_recipe(
    *,
    base_recipe_path: str,
    output_path: str = "",
    state_root: str = "",
    recipe_id: str = "",
    image_id: str = "",
    source_disk: str = "",
    source_disk_sha256: str = "",
    source_disk_sha512: str = "",
    source_disk_filename: str = "",
    kernel: str = "",
    kernel_sha256: str = "",
    kernel_sha512: str = "",
    kernel_filename: str = "",
    base_initrd: str = "",
    base_initrd_sha256: str = "",
    base_initrd_sha512: str = "",
    base_initrd_filename: str = "",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a READY artifact recipe only from digest-pinned boot artifacts."""

    root = managed_vm_state_root(state_root)
    resolved_base_path, builtin_metadata = _resolve_managed_vm_recipe_reference(base_recipe_path)
    base_file = Path(resolved_base_path).expanduser().resolve() if _clean(resolved_base_path) else Path("")
    if not _clean(base_recipe_path) or not base_file.exists() or not base_file.is_file():
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "pin_artifact_recipe",
            "status": "PIN_BLOCKED_BASE_RECIPE_UNAVAILABLE",
            "state_root": root,
            "base_recipe_path": str(base_file) if _clean(base_recipe_path) else "",
            "builtin_recipe": builtin_metadata,
            "reason": "base recipe path was not provided or does not exist",
            "no_host_fallback": True,
        }
    base_recipe = _read_json(base_file)
    base_recipe_id = _clean(base_recipe.get("recipe_id")) or _clean(builtin_metadata.get("recipe_id")) or base_file.stem
    selected_recipe_id = _clean(recipe_id) or f"{base_recipe_id}-pinned"
    selected_image_id = _clean(image_id) or _clean(base_recipe.get("image_id")) or selected_recipe_id
    boot_mode = _clean(base_recipe.get("boot_mode")) or "linux_direct"
    cloud_init_seed_enabled = bool(base_recipe.get("cloud_init_seed_enabled", False)) or str(
        base_recipe.get("guest_agent_installation_mode") or ""
    ) == "cloud_init_nocloud_seed"
    required = (
        ("source_disk",)
        if boot_mode == "efi_disk" and cloud_init_seed_enabled
        else ("source_disk", "kernel")
    )
    artifacts: Dict[str, Any] = {}
    pin_results: Dict[str, Dict[str, Any]] = {}
    source_specs = {
        "source_disk": (source_disk, source_disk_sha256, source_disk_sha512, source_disk_filename),
        "kernel": (kernel, kernel_sha256, kernel_sha512, kernel_filename),
        "base_initrd": (base_initrd, base_initrd_sha256, base_initrd_sha512, base_initrd_filename),
    }
    for name, (source, sha256, sha512, filename) in source_specs.items():
        if name not in required and not _clean(source):
            continue
        spec, result = _pinned_artifact_spec(
            name=name,
            source=source,
            sha256=sha256,
            sha512=sha512,
            filename=filename,
        )
        pin_results[name] = result
        if spec:
            artifacts[name] = spec
    failed = {name: result for name, result in pin_results.items() if result.get("status") != "PINNED"}
    if failed:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "pin_artifact_recipe",
            "status": "PIN_BLOCKED",
            "state_root": root,
            "base_recipe_path": str(base_file),
            "builtin_recipe": builtin_metadata,
            "base_recipe": base_recipe,
            "required_artifacts": list(required),
            "pin_results": pin_results,
            "failed_artifacts": failed,
            "reason": "one or more required artifacts could not be pinned",
            "no_host_fallback": True,
        }
    output = Path(output_path).expanduser() if _clean(output_path) else managed_vm_pinned_recipe_root(root) / f"{selected_recipe_id}.recipe.json"
    output = output.resolve()
    if output.exists() and not overwrite:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "pin_artifact_recipe",
            "status": "PIN_BLOCKED_OUTPUT_EXISTS",
            "state_root": root,
            "output_path": str(output),
            "reason": "output recipe already exists; pass overwrite to replace it",
            "no_host_fallback": True,
        }
    recipe: Dict[str, Any] = {
        "schema_version": MANAGED_VM_ARTIFACT_RECIPE_VERSION,
        "recipe_id": selected_recipe_id,
        "status": "READY",
        "reason": "digest-pinned managed VM artifact recipe",
        "base_recipe_id": base_recipe_id,
        "base_recipe_path": str(base_file),
        "image_id": selected_image_id,
        "boot_mode": boot_mode,
        "artifacts": artifacts,
        "guest_agent_port": base_recipe.get("guest_agent_port", DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT),
        "guest_python_path": _clean(base_recipe.get("guest_python_path")) or "/usr/bin/python3",
        "root_device": _clean(base_recipe.get("root_device")) or "/dev/vda",
        "kernel_command_line": _clean(base_recipe.get("kernel_command_line")),
        "cloud_init_seed_enabled": cloud_init_seed_enabled,
        "guest_agent_installation_mode": (
            "cloud_init_nocloud_seed"
            if cloud_init_seed_enabled
            else _clean(base_recipe.get("guest_agent_installation_mode"))
        ),
        "created_at": _now_iso(),
        "created_by": "conos vm pin-artifact-recipe",
        "no_host_fallback": True,
    }
    if not recipe["kernel_command_line"]:
        recipe.pop("kernel_command_line", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, recipe)
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "pin_artifact_recipe",
        "status": "PINNED",
        "state_root": root,
        "base_recipe_path": str(base_file),
        "builtin_recipe": builtin_metadata,
        "recipe": recipe,
        "recipe_path": str(output),
        "pin_results": pin_results,
        "required_artifacts": list(required),
        "next_step": f"conos vm bootstrap-image --recipe-path {output}",
        "no_host_fallback": True,
    }


def _copy_or_download_vm_artifact(
    *,
    state_root: str,
    name: str,
    spec: Dict[str, Any],
    allow_download: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    source = _artifact_spec_value(spec, "url", "uri", "path")
    digest_algorithm, expected_digest = _artifact_expected_digest(spec)
    if not source:
        return {"name": name, "status": "INVALID_ARTIFACT_SPEC", "reason": "missing url/path"}
    if not expected_digest:
        return {"name": name, "status": "INVALID_ARTIFACT_SPEC", "reason": "missing sha256/sha512 digest"}
    filename = _artifact_filename(name, spec, source)
    cache_path = _artifact_cache_path_for_digest(
        state_root=state_root,
        algorithm=digest_algorithm,
        expected=expected_digest,
        filename=filename,
    )
    if cache_path.exists():
        matches, actual_hex, actual_b64 = _digest_matches(cache_path, digest_algorithm, expected_digest)
        if matches:
            sidecar_path = Path(f"{cache_path}.manifest.json")
            return {
                "name": name,
                "status": "CACHED",
                "path": str(cache_path),
                "digest_algorithm": digest_algorithm,
                "digest": actual_hex,
                "digest_base64": actual_b64,
                "expected_digest": expected_digest,
                "source": source,
                "from_cache": True,
                "sidecar_manifest_path": str(sidecar_path) if sidecar_path.exists() else "",
                "sidecar_manifest_present": sidecar_path.exists(),
            }
        cache_path.unlink()
    parsed = urlparse(source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cache_manifest_path = Path(f"{cache_path}.cache.json")
    try:
        source_path: Path | None = None
        if parsed.scheme in {"", "file"}:
            source_path = Path(unquote(parsed.path) if parsed.scheme == "file" else source).expanduser().resolve()
            if not source_path.exists() or not source_path.is_file():
                return {
                    "name": name,
                    "status": "ARTIFACT_SOURCE_MISSING",
                    "reason": f"artifact source does not exist: {source_path}",
                    "source": source,
                }
            shutil.copy2(source_path, tmp_path)
        elif parsed.scheme in {"http", "https"}:
            if not allow_download:
                return {
                    "name": name,
                    "status": "DOWNLOAD_DISABLED",
                    "reason": "network artifact download is disabled",
                    "source": source,
                }
            with urlopen(source, timeout=int(timeout_seconds)) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        else:
            return {
                "name": name,
                "status": "UNSUPPORTED_ARTIFACT_URL",
                "reason": f"unsupported artifact URL scheme: {parsed.scheme}",
                "source": source,
            }
        matches, actual_hex, actual_b64 = _digest_matches(tmp_path, digest_algorithm, expected_digest)
        if not matches:
            tmp_path.unlink(missing_ok=True)
            return {
                "name": name,
                "status": "DIGEST_MISMATCH",
                "reason": "artifact digest did not match recipe",
                "source": source,
                "digest_algorithm": digest_algorithm,
                "expected_digest": expected_digest,
                "actual_digest": actual_hex,
                "actual_digest_base64": actual_b64,
            }
        tmp_path.replace(cache_path)
        sidecar_copy_path = Path(f"{cache_path}.manifest.json")
        if source_path is not None:
            source_sidecar = Path(f"{source_path}.manifest.json")
            if source_sidecar.exists() and source_sidecar.is_file():
                shutil.copy2(source_sidecar, sidecar_copy_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return {
            "name": name,
            "status": "ARTIFACT_FETCH_FAILED",
            "reason": str(exc),
            "source": source,
        }
    manifest = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_cached_artifact",
        "name": name,
        "status": "CACHED",
        "path": str(cache_path),
        "source": source,
        "digest_algorithm": digest_algorithm,
        "expected_digest": expected_digest,
        "cached_at": _now_iso(),
        "recipe_spec": spec,
        "sidecar_manifest_path": str(Path(f"{cache_path}.manifest.json"))
        if Path(f"{cache_path}.manifest.json").exists()
        else "",
        "sidecar_manifest_present": Path(f"{cache_path}.manifest.json").exists(),
        "no_host_fallback": True,
    }
    _write_json(cache_manifest_path, manifest)
    return {
        "name": name,
        "status": "CACHED",
        "path": str(cache_path),
        "digest_algorithm": digest_algorithm,
        "expected_digest": expected_digest,
        "source": source,
        "from_cache": False,
        "manifest_path": str(cache_manifest_path),
        "sidecar_manifest_path": str(Path(f"{cache_path}.manifest.json"))
        if Path(f"{cache_path}.manifest.json").exists()
        else "",
        "sidecar_manifest_present": Path(f"{cache_path}.manifest.json").exists(),
    }


def resolve_managed_vm_artifact_recipe(
    *,
    recipe_path: str,
    state_root: str = "",
    allow_download: bool = True,
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Resolve a trusted artifact recipe into local cached boot artifact paths."""

    root = managed_vm_state_root(state_root)
    resolved_recipe_path, builtin_metadata = _resolve_managed_vm_recipe_reference(recipe_path)
    if str(builtin_metadata.get("status") or "") == "RECIPE_UNKNOWN":
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "resolve_artifact_recipe",
            "status": "RECIPE_UNKNOWN",
            "recipe_path": resolved_recipe_path,
            "state_root": root,
            "builtin_recipe": builtin_metadata,
            "reason": str(builtin_metadata.get("reason") or "builtin recipe was not found"),
            "no_host_fallback": True,
        }
    recipe_file = Path(resolved_recipe_path).expanduser().resolve() if _clean(resolved_recipe_path) else Path("")
    if not _clean(recipe_path) or not recipe_file.exists() or not recipe_file.is_file():
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "resolve_artifact_recipe",
            "status": "RECIPE_UNAVAILABLE",
            "recipe_path": str(recipe_file) if _clean(recipe_path) else "",
            "builtin_recipe": builtin_metadata,
            "reason": "artifact recipe path was not provided or does not exist",
            "state_root": root,
            "no_host_fallback": True,
        }
    recipe = _read_json(recipe_file)
    recipe_status = _clean(recipe.get("status")) or "READY"
    if recipe_status not in {"READY", "AVAILABLE"}:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "resolve_artifact_recipe",
            "status": "RECIPE_BLOCKED",
            "recipe_path": str(recipe_file),
            "recipe_schema_version": str(recipe.get("schema_version") or ""),
            "recipe_status": recipe_status,
            "state_root": root,
            "recipe": recipe,
            "builtin_recipe": builtin_metadata,
            "reason": str(recipe.get("reason") or "recipe is not enabled for bootstrap"),
            "no_host_fallback": True,
        }
    artifacts = _recipe_artifacts(recipe)
    boot_mode = _clean(recipe.get("boot_mode")) or "linux_direct"
    cloud_init_seed_enabled = bool(recipe.get("cloud_init_seed_enabled", False)) or str(
        recipe.get("guest_agent_installation_mode") or ""
    ) == "cloud_init_nocloud_seed"
    required_artifacts = (
        ("source_disk",)
        if boot_mode == "efi_disk" and cloud_init_seed_enabled
        else ("source_disk", "kernel")
    )
    missing_artifacts = [name for name in required_artifacts if not isinstance(artifacts.get(name), dict)]
    if missing_artifacts:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "resolve_artifact_recipe",
            "status": "RECIPE_INVALID",
            "recipe_path": str(recipe_file),
            "state_root": root,
            "recipe": recipe,
            "builtin_recipe": builtin_metadata,
            "missing_artifacts": missing_artifacts,
            "required_artifacts": list(required_artifacts),
            "reason": "recipe does not define the boot artifacts required by its boot mode",
            "no_host_fallback": True,
        }
    artifact_results: Dict[str, Dict[str, Any]] = {}
    for name in ("source_disk", "kernel", "base_initrd", "initrd"):
        spec = artifacts.get(name)
        if not isinstance(spec, dict):
            continue
        artifact_results[name] = _copy_or_download_vm_artifact(
            state_root=root,
            name=name,
            spec=_artifact_spec_with_recipe_base(dict(spec), recipe_file.parent),
            allow_download=allow_download,
            timeout_seconds=int(timeout_seconds),
        )
    failed = {name: result for name, result in artifact_results.items() if result.get("status") != "CACHED"}
    resolved_paths = {
        "source_disk_path": str(artifact_results.get("source_disk", {}).get("path") or ""),
        "kernel_path": str(artifact_results.get("kernel", {}).get("path") or ""),
        "base_initrd_path": str(artifact_results.get("base_initrd", {}).get("path") or ""),
        "initrd_path": str(artifact_results.get("initrd", {}).get("path") or ""),
    }
    status = "RESOLVED" if not failed else "ARTIFACT_RESOLUTION_FAILED"
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "resolve_artifact_recipe",
        "status": status,
        "recipe_path": str(recipe_file),
        "recipe_schema_version": str(recipe.get("schema_version") or ""),
        "recipe_status": recipe_status,
        "builtin_recipe": builtin_metadata,
        "state_root": root,
        "artifact_cache_root": str(managed_vm_artifact_cache_root(root)),
        "artifact_results": artifact_results,
        "failed_artifacts": failed,
        "resolved_paths": resolved_paths,
        "image_id": _clean(recipe.get("image_id")),
        "boot_mode": boot_mode,
        "cloud_init_seed_enabled": cloud_init_seed_enabled,
        "guest_agent_installation_mode": _clean(recipe.get("guest_agent_installation_mode")),
        "guest_agent_port": recipe.get("guest_agent_port"),
        "kernel_command_line": _clean(recipe.get("kernel_command_line")),
        "root_device": _clean(recipe.get("root_device")),
        "guest_python_path": _clean(recipe.get("guest_python_path")),
        "allow_download": bool(allow_download),
        "no_host_fallback": True,
        "reason": "" if status == "RESOLVED" else "one or more recipe artifacts could not be resolved",
    }


def wait_managed_vm_guest_agent_ready(
    *,
    state_root: str = "",
    image_id: str = "",
    instance_id: str = "",
    wait_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
) -> Dict[str, Any]:
    """Wait until runtime.json proves the managed guest agent is execution-ready."""

    config = managed_vm_config(state_root=state_root, image_id=image_id, instance_id=instance_id)
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    last_gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    attempts = 1
    while not bool(last_gate.get("ready")) and time.monotonic() < deadline:
        time.sleep(max(0.05, float(poll_interval_seconds)))
        last_gate = managed_vm_guest_agent_gate(
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
        )
        attempts += 1
    ready = bool(last_gate.get("ready"))
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "wait_guest_agent_ready",
        "status": "GUEST_AGENT_READY" if ready else "GUEST_AGENT_TIMEOUT",
        "ready": ready,
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "wait_seconds": float(wait_seconds),
        "poll_interval_seconds": float(poll_interval_seconds),
        "attempts": attempts,
        "gate": last_gate,
        "blocked_reasons": list(last_gate.get("blocked_reasons") or []),
        "reason": "" if ready else str(last_gate.get("reason") or "guest agent did not become ready before timeout"),
        "no_host_fallback": True,
    }


def _mark_managed_vm_image_verified(
    *,
    state_root: str,
    image_id: str,
    instance_id: str,
    agent_exec_result: Dict[str, Any],
) -> Dict[str, Any]:
    manifest = load_managed_vm_image_manifest(state_root, image_id)
    if not manifest:
        return {}
    manifest.update(
        {
            "boot_verified": True,
            "guest_agent_verified": True,
            "bootstrap_verified": True,
            "bootstrap_verified_at": _now_iso(),
            "bootstrap_verified_instance_id": instance_id,
            "bootstrap_verification_command": ["echo", "ok"],
            "bootstrap_agent_exec_returncode": int(agent_exec_result.get("returncode", 0) or 0),
            "bootstrap_agent_exec_stdout": str(agent_exec_result.get("stdout") or ""),
            "bootstrap_agent_exec_status": str(agent_exec_result.get("status") or ""),
            "execution_ready": False,
            "readiness_contract": "image verified by start-instance guest-agent handshake and agent-exec smoke command",
        }
    )
    _write_json(managed_vm_image_manifest_path(state_root, image_id), manifest)
    return manifest


def bootstrap_managed_vm_image(
    *,
    state_root: str = "",
    image_id: str = "",
    instance_id: str = "bootstrap-smoke",
    source_disk_path: str = "",
    kernel_path: str = "",
    base_initrd_path: str = "",
    recipe_path: str = "",
    allow_artifact_download: bool = True,
    artifact_timeout_seconds: int = 120,
    guest_agent_path: str = "",
    runner_path: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    kernel_command_line: str = "",
    root_device: str = "/dev/vda",
    guest_python_path: str = "/usr/bin/python3",
    network_mode: str = "provider_default",
    build_runner: bool = True,
    start_instance: bool = True,
    verify_agent_exec: bool = True,
    keep_running: bool = False,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 180.0,
    agent_timeout_seconds: int = 30,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Bootstrap a managed VM image through build, start, guest-agent, and exec verification."""

    image = managed_vm_image_id(image_id)
    instance = managed_vm_instance_id(instance_id)
    root = managed_vm_state_root(state_root)
    init_report = init_managed_vm_state(state_root=root, image_id=image, instance_id=instance)
    steps: list[Dict[str, Any]] = [{"step": "init_state", "status": str(init_report.get("status") or "READY")}]
    selected_runner = managed_vm_runner_path(runner_path, state_root=root)
    runner_report: Dict[str, Any] = {}
    if not selected_runner and build_runner:
        runner_report = build_managed_vm_virtualization_runner(state_root=root)
        steps.append({"step": "build_runner", "status": str(runner_report.get("status") or "")})
        selected_runner = managed_vm_runner_path(state_root=root)
    elif selected_runner:
        steps.append({"step": "runner_available", "status": "AVAILABLE", "runner_path": selected_runner})

    recipe_report: Dict[str, Any] = {}
    resolved_source_disk_path = source_disk_path
    resolved_kernel_path = kernel_path
    resolved_base_initrd_path = base_initrd_path
    resolved_initrd_path = ""
    resolved_guest_agent_port = int(guest_agent_port)
    resolved_kernel_command_line = kernel_command_line
    resolved_root_device = root_device
    resolved_guest_python_path = guest_python_path
    resolved_boot_mode = "linux_direct"
    resolved_cloud_init_seed_enabled = False
    if _clean(recipe_path) and (not _clean(resolved_source_disk_path) or not _clean(resolved_kernel_path)):
        recipe_report = resolve_managed_vm_artifact_recipe(
            recipe_path=recipe_path,
            state_root=root,
            allow_download=allow_artifact_download,
            timeout_seconds=int(artifact_timeout_seconds),
        )
        steps.append({"step": "resolve_artifact_recipe", "status": str(recipe_report.get("status") or "")})
        resolved_paths = recipe_report.get("resolved_paths") if isinstance(recipe_report.get("resolved_paths"), dict) else {}
        if str(recipe_report.get("status") or "") == "RESOLVED":
            resolved_source_disk_path = _clean(resolved_source_disk_path) or str(resolved_paths.get("source_disk_path") or "")
            resolved_kernel_path = _clean(resolved_kernel_path) or str(resolved_paths.get("kernel_path") or "")
            resolved_base_initrd_path = _clean(resolved_base_initrd_path) or str(resolved_paths.get("base_initrd_path") or "")
            resolved_initrd_path = str(resolved_paths.get("initrd_path") or "")
            recipe_port = recipe_report.get("guest_agent_port")
            if recipe_port is not None:
                resolved_guest_agent_port = int(recipe_port)
            resolved_kernel_command_line = _clean(resolved_kernel_command_line) or str(
                recipe_report.get("kernel_command_line") or ""
            )
            recipe_root_device = _clean(recipe_report.get("root_device"))
            if recipe_root_device and _clean(root_device) == "/dev/vda":
                resolved_root_device = recipe_root_device
            recipe_python_path = _clean(recipe_report.get("guest_python_path"))
            if recipe_python_path and _clean(guest_python_path) == "/usr/bin/python3":
                resolved_guest_python_path = recipe_python_path
            resolved_boot_mode = _clean(recipe_report.get("boot_mode")) or resolved_boot_mode
            resolved_cloud_init_seed_enabled = bool(recipe_report.get("cloud_init_seed_enabled", False))
        else:
            return {
                "schema_version": MANAGED_VM_PROVIDER_VERSION,
                "operation": "bootstrap_image",
                "status": "BOOTSTRAP_BLOCKED_ARTIFACT_RESOLUTION_FAILED",
                "state_root": root,
                "image_id": image,
                "instance_id": instance,
                "runner_path": selected_runner,
                "runner_available": bool(selected_runner),
                "runner_report": runner_report,
                "recipe_report": recipe_report,
                "build_report": {},
                "steps": steps,
                "failed_artifacts": recipe_report.get("failed_artifacts") if isinstance(recipe_report, dict) else {},
                "reason": str(recipe_report.get("reason") or "one or more recipe artifacts could not be resolved"),
                "next_required_step": (
                    "enable artifact download, provide a digest-matching cached artifact, "
                    "or bootstrap with an explicit source disk"
                ),
                "requires_user_configured_vm": False,
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
                "no_host_fallback": True,
            }

    if resolved_boot_mode == "efi_disk" and resolved_cloud_init_seed_enabled:
        if not _clean(resolved_source_disk_path):
            build_report = {
                "schema_version": MANAGED_VM_PROVIDER_VERSION,
                "operation": "register_cloud_init_image",
                "status": "BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS",
                "state_root": root,
                "image_id": image,
                "source_disk_path": "",
                "missing_fields": ["source_disk_path"],
                "missing_paths": [],
                "reason": "source disk is required before Con OS can register an EFI cloud-init managed image",
                "cloud_init_seed_enabled": True,
                "no_host_fallback": True,
            }
        else:
            build_report = register_managed_vm_cloud_init_image(
                state_root=root,
                image_id=image,
                source_disk_path=resolved_source_disk_path,
                guest_agent_path=guest_agent_path,
                guest_agent_port=int(resolved_guest_agent_port),
                guest_python_path=resolved_guest_python_path,
            )
            build_report["operation"] = "register_cloud_init_image"
            build_report["base_image_builder_status"] = "BUILT"
        steps.append({"step": "register_cloud_init_image", "status": str(build_report.get("status") or "")})
    elif _clean(resolved_initrd_path):
        if not _clean(resolved_source_disk_path) or not _clean(resolved_kernel_path):
            build_report = {
                "schema_version": MANAGED_VM_PROVIDER_VERSION,
                "operation": "register_bundled_linux_image",
                "status": "BUILD_BLOCKED_MISSING_BOOT_ARTIFACTS",
                "state_root": root,
                "image_id": image,
                "source_disk_path": resolved_source_disk_path,
                "kernel_path": resolved_kernel_path,
                "initrd_path": resolved_initrd_path,
                "missing_fields": [
                    name
                    for name, value in {
                        "source_disk_path": resolved_source_disk_path,
                        "kernel_path": resolved_kernel_path,
                    }.items()
                    if not _clean(value)
                ],
                "reason": "source disk and kernel are required before Con OS can register bundled linux_direct artifacts",
                "no_host_fallback": True,
            }
        else:
            build_report = register_managed_vm_linux_boot_image(
                state_root=root,
                image_id=image,
                source_disk_path=resolved_source_disk_path,
                kernel_path=resolved_kernel_path,
                initrd_path=resolved_initrd_path,
                kernel_command_line=resolved_kernel_command_line,
                guest_agent_port=int(resolved_guest_agent_port),
            )
            build_report["operation"] = "register_bundled_linux_image"
            build_report["base_image_builder_status"] = "REGISTERED_BUNDLED_ARTIFACTS"
        steps.append({"step": "register_bundled_linux_image", "status": str(build_report.get("status") or "")})
    else:
        build_report = build_managed_vm_linux_base_image(
            state_root=root,
            image_id=image,
            source_disk_path=resolved_source_disk_path,
            kernel_path=resolved_kernel_path,
            base_initrd_path=resolved_base_initrd_path,
            guest_agent_path=guest_agent_path,
            guest_agent_port=int(resolved_guest_agent_port),
            kernel_command_line=resolved_kernel_command_line,
            root_device=resolved_root_device,
            guest_python_path=resolved_guest_python_path,
            overwrite=overwrite,
        )
        steps.append({"step": "build_base_image", "status": str(build_report.get("status") or "")})
    base_payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "bootstrap_image",
        "state_root": root,
        "image_id": image,
        "instance_id": instance,
        "runner_path": selected_runner,
        "runner_available": bool(selected_runner),
        "runner_report": runner_report,
        "recipe_report": recipe_report,
        "build_report": build_report,
        "steps": steps,
        "requires_user_configured_vm": False,
        "no_host_fallback": True,
    }
    if str(build_report.get("status") or "") not in {"BUILT", "REGISTERED"}:
        base_payload.update(
            {
                "status": "BOOTSTRAP_BLOCKED_MISSING_BOOT_ARTIFACTS",
                "reason": str(build_report.get("reason") or "boot artifacts are required"),
                "missing_fields": list(build_report.get("missing_fields") or []),
                "missing_paths": list(build_report.get("missing_paths") or []),
                "next_required_step": "provide source-disk and kernel-path, or add an artifact downloader recipe",
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
            }
        )
        return base_payload

    prepare_report = prepare_managed_vm_instance(
        state_root=root,
        image_id=image,
        instance_id=instance,
        network_mode=network_mode,
    )
    steps.append({"step": "prepare_instance", "status": str(prepare_report.get("status") or "")})
    base_payload["prepare_report"] = prepare_report
    if not start_instance:
        base_payload.update(
            {
                "status": "BOOTSTRAP_IMAGE_BUILT",
                "reason": "image built and instance prepared; start verification was not requested",
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
            }
        )
        return base_payload
    if not selected_runner:
        base_payload.update(
            {
                "status": "BOOTSTRAP_BLOCKED_RUNNER_UNAVAILABLE",
                "reason": "Apple Virtualization runner is required to start the managed image",
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
                "next_required_step": "run conos vm build-runner on macOS with Apple Virtualization support",
            }
        )
        return base_payload

    start_report = start_managed_vm_instance(
        state_root=root,
        runner_path=selected_runner,
        image_id=image,
        instance_id=instance,
        network_mode=network_mode,
        startup_wait_seconds=float(startup_wait_seconds),
    )
    steps.append({"step": "start_instance", "status": str(start_report.get("status") or "")})
    base_payload["start_report"] = start_report
    if str(start_report.get("status") or "") != "STARTED":
        start_status = str(start_report.get("status") or "")
        bootstrap_status = "BOOTSTRAP_START_FAILED"
        if start_status.startswith("START_BLOCKED_"):
            bootstrap_status = "BOOTSTRAP_" + start_status[len("START_") :]
        start_runtime = start_report.get("runtime_manifest") if isinstance(start_report.get("runtime_manifest"), dict) else {}
        start_diagnostic = (
            start_runtime.get("last_guest_boot_diagnostic")
            if isinstance(start_runtime.get("last_guest_boot_diagnostic"), dict)
            else {}
        )
        boot_path_recommendation = _managed_vm_boot_path_recommendation(
            boot_mode=resolved_boot_mode,
            guest_boot_diagnostic=start_diagnostic,
            start_status=start_status,
        )
        base_payload.update(
            {
                "status": bootstrap_status,
                "reason": str(start_report.get("reason") or "managed VM did not start"),
                "blocker_type": str(start_report.get("blocker_type") or ""),
                "host_virtualization_capability": start_report.get("host_virtualization_capability") or {},
                "boot_path_recommendation": boot_path_recommendation,
                "next_required_step": str(start_report.get("next_required_step") or ""),
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
            }
        )
        return base_payload

    wait_report = wait_managed_vm_guest_agent_ready(
        state_root=root,
        image_id=image,
        instance_id=instance,
        wait_seconds=float(guest_wait_seconds),
    )
    steps.append({"step": "wait_guest_agent", "status": str(wait_report.get("status") or "")})
    base_payload["wait_report"] = wait_report
    if not bool(wait_report.get("ready")):
        wait_gate = wait_report.get("gate") if isinstance(wait_report.get("gate"), dict) else {}
        boot_diagnostic = (
            wait_gate.get("guest_boot_diagnostic")
            if isinstance(wait_gate.get("guest_boot_diagnostic"), dict)
            else {}
        )
        diagnostic_status = str(boot_diagnostic.get("diagnosis_status") or "")
        boot_path_recommendation = _managed_vm_boot_path_recommendation(
            boot_mode=resolved_boot_mode,
            guest_boot_diagnostic=boot_diagnostic,
            start_status=str(start_report.get("status") or ""),
        )
        stop_report = (
            stop_managed_vm_instance(state_root=root, image_id=image, instance_id=instance)
            if not keep_running
            else {}
        )
        if stop_report:
            steps.append({"step": "stop_instance", "status": str(stop_report.get("status") or "")})
            base_payload["stop_report"] = stop_report
        if diagnostic_status == "CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE":
            bootstrap_status = "BOOTSTRAP_GUEST_AGENT_INSTALLATION_BLOCKED"
            blocker_type = "cloud_init_unavailable_in_guest_image"
        elif diagnostic_status == "LINUX_DIRECT_NO_EARLY_GUEST_SIGNAL":
            bootstrap_status = "BOOTSTRAP_GUEST_BOOT_UNOBSERVABLE"
            blocker_type = "linux_direct_no_early_guest_signal"
        else:
            bootstrap_status = "BOOTSTRAP_GUEST_AGENT_TIMEOUT"
            blocker_type = (
                "guest_boot_or_cloud_init_not_observed"
                if diagnostic_status in {"GUEST_BOOT_NO_OBSERVABILITY", "CLOUD_INIT_RUNCMD_NOT_OBSERVED"}
                else "guest_agent_not_ready"
            )
        base_payload.update(
            {
                "status": bootstrap_status,
                "reason": str(wait_report.get("reason") or "guest agent did not become ready"),
                "blocker_type": blocker_type,
                "guest_boot_diagnostic": boot_diagnostic,
                "boot_path_recommendation": boot_path_recommendation,
                "next_required_step": (
                    str((boot_path_recommendation.get("recommended_next_steps") or [""])[0])
                    if isinstance(boot_path_recommendation.get("recommended_next_steps"), list)
                    and boot_path_recommendation.get("recommended_next_steps")
                    else str((boot_diagnostic.get("recommended_next_steps") or [""])[0])
                    if isinstance(boot_diagnostic.get("recommended_next_steps"), list)
                    and boot_diagnostic.get("recommended_next_steps")
                    else ""
                ),
                "guest_agent_ready": False,
                "execution_ready": False,
                "verified": False,
            }
        )
        return base_payload

    if not verify_agent_exec:
        base_payload.update(
            {
                "status": "BOOTSTRAP_GUEST_AGENT_READY",
                "reason": "guest agent is ready; agent-exec smoke verification was not requested",
                "guest_agent_ready": True,
                "execution_ready": True,
                "verified": False,
            }
        )
        return base_payload

    exec_report = run_managed_vm_agent_command(
        ["echo", "ok"],
        state_root=root,
        image_id=image,
        instance_id=instance,
        timeout_seconds=int(agent_timeout_seconds),
    )
    steps.append({"step": "agent_exec_smoke", "status": str(exec_report.get("status") or "")})
    base_payload["agent_exec_report"] = exec_report
    returncode_value = exec_report.get("returncode")
    smoke_returncode = int(returncode_value) if returncode_value is not None else 1
    smoke_ok = smoke_returncode == 0 and str(exec_report.get("stdout") or "").strip() == "ok"
    if not smoke_ok:
        stop_report = (
            stop_managed_vm_instance(state_root=root, image_id=image, instance_id=instance)
            if not keep_running
            else {}
        )
        if stop_report:
            steps.append({"step": "stop_instance", "status": str(stop_report.get("status") or "")})
            base_payload["stop_report"] = stop_report
        base_payload.update(
            {
                "status": "BOOTSTRAP_AGENT_EXEC_FAILED",
                "reason": "guest agent became ready but agent-exec smoke command failed",
                "guest_agent_ready": True,
                "execution_ready": True,
                "verified": False,
            }
        )
        return base_payload

    verified_manifest = _mark_managed_vm_image_verified(
        state_root=root,
        image_id=image,
        instance_id=instance,
        agent_exec_result=exec_report,
    )
    stop_report = (
        stop_managed_vm_instance(state_root=root, image_id=image, instance_id=instance)
        if not keep_running
        else {}
    )
    if stop_report:
        steps.append({"step": "stop_instance", "status": str(stop_report.get("status") or "")})
        base_payload["stop_report"] = stop_report
    base_payload.update(
        {
            "status": "BOOTSTRAP_VERIFIED",
            "reason": "managed VM image booted, guest agent became ready, and agent-exec smoke passed",
            "guest_agent_ready": True,
            "execution_ready": True,
            "verified": True,
            "image_manifest": verified_manifest,
        }
    )
    return base_payload


def install_default_managed_vm_image(
    *,
    state_root: str = "",
    image_id: str = "",
    instance_id: str = "bootstrap-smoke",
    recipe_path: str = "",
    allow_artifact_download: bool = True,
    artifact_timeout_seconds: int = 120,
    guest_agent_path: str = "",
    runner_path: str = "",
    network_mode: str = "provider_default",
    build_runner: bool = True,
    start_instance: bool = True,
    verify_agent_exec: bool = True,
    keep_running: bool = False,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 60.0,
    agent_timeout_seconds: int = 30,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Install the Con OS default managed VM image from the built-in recipe."""

    selected_recipe = _clean(recipe_path) or DEFAULT_MANAGED_VM_RECIPE_REFERENCE
    image = managed_vm_image_id(image_id)
    instance = managed_vm_instance_id(instance_id)
    root = managed_vm_state_root(state_root)
    bootstrap_report = bootstrap_managed_vm_image(
        state_root=root,
        image_id=image,
        instance_id=instance,
        recipe_path=selected_recipe,
        allow_artifact_download=allow_artifact_download,
        artifact_timeout_seconds=int(artifact_timeout_seconds),
        guest_agent_path=guest_agent_path,
        runner_path=runner_path,
        network_mode=network_mode,
        build_runner=build_runner,
        start_instance=start_instance,
        verify_agent_exec=verify_agent_exec,
        keep_running=keep_running,
        startup_wait_seconds=float(startup_wait_seconds),
        guest_wait_seconds=float(guest_wait_seconds),
        agent_timeout_seconds=int(agent_timeout_seconds),
        overwrite=overwrite,
    )
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "install_default_image",
        "status": str(bootstrap_report.get("status") or ""),
        "state_root": root,
        "image_id": image,
        "instance_id": instance,
        "recipe_path": selected_recipe,
        "allow_artifact_download": bool(allow_artifact_download),
        "bootstrap_report": bootstrap_report,
        "runner_available": bool(bootstrap_report.get("runner_available")),
        "guest_agent_ready": bool(bootstrap_report.get("guest_agent_ready")),
        "execution_ready": bool(bootstrap_report.get("execution_ready")),
        "verified": bool(bootstrap_report.get("verified")),
        "reason": str(bootstrap_report.get("reason") or ""),
        "next_required_step": str(bootstrap_report.get("next_required_step") or ""),
        "requires_user_configured_vm": False,
        "no_host_fallback": True,
    }


def _run_helper_lifecycle_command(
    helper_command: str,
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
) -> tuple[ManagedVMConfig, list[str], subprocess.CompletedProcess[str], Dict[str, Any]]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    if not config.helper_path:
        raise FileNotFoundError("managed VM helper was not found")
    command = _helper_lifecycle_command(helper_command, config=config, network_mode=network_mode)
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    return config, command, completed, _read_json_from_text(str(completed.stdout or ""))


def _runner_start_command(
    *,
    runner_path: str,
    state_root: str,
    image_id: str,
    instance_id: str,
    disk_path: Path,
    base_image_path: Path,
    efi_variable_store_path: Path,
    runtime_manifest_path: Path,
    network_mode: str,
    image_manifest: Dict[str, Any],
    console_log_path: Path | None = None,
    shared_dir_path: Path | None = None,
    shared_dir_tag: str = MANAGED_VM_SHARED_DIR_TAG,
) -> list[str]:
    command = [
        str(runner_path),
        "run",
        "--state-root",
        str(state_root),
        "--instance-id",
        str(instance_id),
        "--image-id",
        str(image_id),
        "--disk-path",
        str(disk_path),
        "--base-image",
        str(base_image_path),
        "--efi-variable-store",
        str(efi_variable_store_path),
        "--runtime-manifest",
        str(runtime_manifest_path),
        "--network-mode",
        _clean(network_mode) or "provider_default",
    ]
    if console_log_path is not None:
        command.extend(["--console-log", str(console_log_path)])
    if shared_dir_path is not None:
        command.extend(["--shared-dir", str(shared_dir_path), "--shared-tag", _clean(shared_dir_tag) or MANAGED_VM_SHARED_DIR_TAG])
    boot_mode = str(image_manifest.get("boot_mode") or "efi_disk")
    command.extend(["--boot-mode", boot_mode])
    kernel_path = _clean(image_manifest.get("kernel_path"))
    initrd_path = _clean(image_manifest.get("initrd_path"))
    kernel_command_line = _clean(image_manifest.get("kernel_command_line"))
    guest_agent_port = image_manifest.get("guest_agent_port")
    if kernel_path:
        command.extend(["--kernel-path", kernel_path])
    if initrd_path:
        command.extend(["--initrd-path", initrd_path])
    if kernel_command_line:
        command.extend(["--kernel-command-line", kernel_command_line])
    if guest_agent_port is not None:
        command.extend(["--guest-agent-port", str(guest_agent_port)])
    cloud_init_seed_path = _clean(image_manifest.get("cloud_init_seed_path"))
    if cloud_init_seed_path:
        command.extend(["--cloud-init-seed", cloud_init_seed_path])
    return command


def _ensure_instance_cloud_init_seed(
    *,
    state_root: str,
    image_id: str,
    instance_id: str,
    image_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(image_manifest.get("cloud_init_seed_enabled", False)):
        return {"enabled": False, "status": "NOT_REQUIRED"}
    seed_path = managed_vm_instance_cloud_init_seed_path(state_root, instance_id)
    payload = build_managed_vm_cloud_init_seed(
        state_root=state_root,
        instance_id=instance_id,
        guest_agent_path=str(image_manifest.get("guest_agent_path") or ""),
        guest_agent_port=int(image_manifest.get("guest_agent_port") or DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT),
        guest_python_path=str(image_manifest.get("guest_python_path") or "/usr/bin/python3"),
        hostname=f"conos-{image_id}-{instance_id}",
        overwrite=True,
    )
    return {
        "enabled": True,
        "status": str(payload.get("status") or ""),
        "seed_path": str(seed_path),
        "seed_manifest_path": str(managed_vm_instance_cloud_init_seed_manifest_path(state_root, instance_id)),
        "seed_manifest": payload,
    }


def _gpt_partitions(disk_path: Path) -> list[Dict[str, Any]]:
    if not disk_path.exists():
        return []
    with disk_path.open("rb") as handle:
        handle.seek(512)
        header = handle.read(512)
        if header[:8] != b"EFI PART":
            return []
        entries_lba = int.from_bytes(header[72:80], "little")
        entry_count = int.from_bytes(header[80:84], "little")
        entry_size = int.from_bytes(header[84:88], "little")
        if entries_lba <= 0 or entry_count <= 0 or entry_size < 128:
            return []
        handle.seek(entries_lba * 512)
        raw_entries = handle.read(entry_count * entry_size)
    partitions: list[Dict[str, Any]] = []
    for index in range(entry_count):
        entry = raw_entries[index * entry_size : (index + 1) * entry_size]
        if len(entry) < 128 or entry[:16] == b"\x00" * 16:
            continue
        first_lba = int.from_bytes(entry[32:40], "little")
        last_lba = int.from_bytes(entry[40:48], "little")
        if first_lba <= 0 or last_lba < first_lba:
            continue
        try:
            name = entry[56:128].decode("utf-16le", "ignore").rstrip("\x00")
            type_guid = uuid.UUID(bytes_le=entry[:16])
            partition_guid = uuid.UUID(bytes_le=entry[16:32])
        except (TypeError, ValueError):
            continue
        partitions.append(
            {
                "index": index + 1,
                "type_guid": str(type_guid),
                "partition_guid": str(partition_guid),
                "first_lba": first_lba,
                "last_lba": last_lba,
                "byte_offset": first_lba * 512,
                "byte_size": (last_lba - first_lba + 1) * 512,
                "name": name,
            }
        )
    return partitions


def _ext4_uuid_at_offset(disk_path: Path, byte_offset: int) -> str:
    try:
        with disk_path.open("rb") as handle:
            handle.seek(int(byte_offset) + 1024)
            superblock = handle.read(1024)
    except OSError:
        return ""
    if len(superblock) < 136 or superblock[56:58] != b"\x53\xef":
        return ""
    try:
        return str(uuid.UUID(bytes=superblock[104:120]))
    except ValueError:
        return ""


def _managed_vm_root_ext_uuid(disk_path: Path, partitions: Sequence[Dict[str, Any]]) -> str:
    preferred: list[Dict[str, Any]] = []
    fallback: list[Dict[str, Any]] = []
    for partition in partitions:
        type_guid = str(partition.get("type_guid") or "").lower()
        if type_guid == str(EFI_SYSTEM_PARTITION_GUID):
            continue
        if type_guid == str(LINUX_ARM64_ROOT_PARTITION_GUID):
            preferred.append(partition)
        else:
            fallback.append(partition)
    for partition in preferred + fallback:
        candidate = _ext4_uuid_at_offset(disk_path, int(partition.get("byte_offset") or 0))
        if candidate:
            return candidate
    return ""


def _managed_vm_root_ext_partition(disk_path: Path, partitions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    preferred: list[Dict[str, Any]] = []
    fallback: list[Dict[str, Any]] = []
    for partition in partitions:
        type_guid = str(partition.get("type_guid") or "").lower()
        if type_guid == str(EFI_SYSTEM_PARTITION_GUID):
            continue
        if type_guid == str(LINUX_ARM64_ROOT_PARTITION_GUID):
            preferred.append(partition)
        else:
            fallback.append(partition)
    for partition in preferred + fallback:
        if _ext4_uuid_at_offset(disk_path, int(partition.get("byte_offset") or 0)):
            return dict(partition)
    return {}


def _ext4_filesystem(handle, *, partition_offset: int) -> Dict[str, Any]:
    handle.seek(int(partition_offset) + 1024)
    superblock = handle.read(1024)
    if len(superblock) < 256 or superblock[56:58] != b"\x53\xef":
        raise ValueError("ext4 superblock was not found")
    log_block_size = int.from_bytes(superblock[24:28], "little")
    block_size = 1024 << log_block_size
    inodes_per_group = int.from_bytes(superblock[40:44], "little")
    blocks_per_group = int.from_bytes(superblock[32:36], "little")
    inode_size = int.from_bytes(superblock[88:90], "little") or 128
    descriptor_size = int.from_bytes(superblock[254:256], "little") if len(superblock) >= 256 else 32
    descriptor_size = max(32, descriptor_size or 32)
    group_descriptor_offset = int(partition_offset) + (2 * block_size if block_size == 1024 else block_size)
    return {
        "partition_offset": int(partition_offset),
        "block_size": block_size,
        "inodes_per_group": inodes_per_group,
        "blocks_per_group": blocks_per_group,
        "inode_size": inode_size,
        "descriptor_size": descriptor_size,
        "group_descriptor_offset": group_descriptor_offset,
    }


def _ext4_read_block(handle, fs: Dict[str, Any], block: int) -> bytes:
    handle.seek(int(fs["partition_offset"]) + int(block) * int(fs["block_size"]))
    return handle.read(int(fs["block_size"]))


def _ext4_inode(handle, fs: Dict[str, Any], inode_number: int) -> Dict[str, Any]:
    inode_index = int(inode_number) - 1
    group = inode_index // int(fs["inodes_per_group"])
    index = inode_index % int(fs["inodes_per_group"])
    descriptor_offset = int(fs["group_descriptor_offset"]) + group * int(fs["descriptor_size"])
    handle.seek(descriptor_offset)
    descriptor = handle.read(int(fs["descriptor_size"]))
    if len(descriptor) < 32:
        raise ValueError("ext4 group descriptor is truncated")
    inode_table = int.from_bytes(descriptor[8:12], "little")
    if len(descriptor) >= 44:
        inode_table |= int.from_bytes(descriptor[40:44], "little") << 32
    inode_offset = int(fs["partition_offset"]) + inode_table * int(fs["block_size"]) + index * int(fs["inode_size"])
    handle.seek(inode_offset)
    raw = handle.read(int(fs["inode_size"]))
    if len(raw) < 100:
        raise ValueError("ext4 inode is truncated")
    size = int.from_bytes(raw[4:8], "little")
    if len(raw) >= 112:
        size |= int.from_bytes(raw[108:112], "little") << 32
    return {
        "inode_number": int(inode_number),
        "mode": int.from_bytes(raw[0:2], "little"),
        "size": size,
        "flags": int.from_bytes(raw[32:36], "little"),
        "i_block": raw[40:100],
    }


def _ext4_extent_records(handle, fs: Dict[str, Any], node: bytes) -> list[tuple[int, int, int]]:
    if len(node) < 12 or node[0:2] != b"\x0a\xf3":
        raise ValueError("ext4 inode does not use an extent tree")
    entries = int.from_bytes(node[2:4], "little")
    depth = int.from_bytes(node[6:8], "little")
    records: list[tuple[int, int, int]] = []
    if depth == 0:
        for index in range(entries):
            offset = 12 + index * 12
            entry = node[offset : offset + 12]
            if len(entry) < 12:
                break
            logical_block = int.from_bytes(entry[0:4], "little")
            length = int.from_bytes(entry[4:6], "little") & 0x7FFF
            physical = (int.from_bytes(entry[6:8], "little") << 32) | int.from_bytes(entry[8:12], "little")
            if length and physical:
                records.append((logical_block, physical, length))
        return records
    for index in range(entries):
        offset = 12 + index * 12
        entry = node[offset : offset + 12]
        if len(entry) < 12:
            break
        leaf = (int.from_bytes(entry[8:10], "little") << 32) | int.from_bytes(entry[4:8], "little")
        if leaf:
            records.extend(_ext4_extent_records(handle, fs, _ext4_read_block(handle, fs, leaf)))
    return records


def _ext4_inode_data(handle, fs: Dict[str, Any], inode: Dict[str, Any], *, max_bytes: int = 256 * 1024 * 1024) -> bytes:
    size = int(inode.get("size") or 0)
    if size > int(max_bytes):
        raise ValueError(f"ext4 file is too large to extract safely: {size} bytes")
    records = sorted(_ext4_extent_records(handle, fs, bytes(inode.get("i_block") or b"")), key=lambda item: item[0])
    output = bytearray()
    block_size = int(fs["block_size"])
    for logical_block, physical_block, length in records:
        wanted_offset = logical_block * block_size
        if len(output) < wanted_offset:
            output.extend(b"\x00" * (wanted_offset - len(output)))
        handle.seek(int(fs["partition_offset"]) + physical_block * block_size)
        output.extend(handle.read(length * block_size))
        if len(output) >= size:
            break
    return bytes(output[:size])


def _ext4_lookup_path(handle, fs: Dict[str, Any], path: str) -> Dict[str, Any]:
    inode = _ext4_inode(handle, fs, 2)
    for part in [item for item in path.strip("/").split("/") if item]:
        directory = _ext4_inode_data(handle, fs, inode)
        found_inode = 0
        offset = 0
        while offset + 8 <= len(directory):
            inode_number = int.from_bytes(directory[offset : offset + 4], "little")
            rec_len = int.from_bytes(directory[offset + 4 : offset + 6], "little")
            name_len = directory[offset + 6]
            if rec_len < 8:
                break
            name = directory[offset + 8 : offset + 8 + name_len].decode("utf-8", "ignore")
            if inode_number and name == part:
                found_inode = inode_number
                break
            offset += rec_len
        if not found_inode:
            raise FileNotFoundError(path)
        inode = _ext4_inode(handle, fs, found_inode)
    return inode


def _extract_ext4_file_from_image(disk_path: Path, *, partition_offset: int, file_path: str) -> bytes:
    with disk_path.open("rb") as handle:
        fs = _ext4_filesystem(handle, partition_offset=int(partition_offset))
        inode = _ext4_lookup_path(handle, fs, file_path)
        return _ext4_inode_data(handle, fs, inode)


def _fat_parameters(handle, partition_offset: int) -> Dict[str, Any]:
    handle.seek(int(partition_offset))
    boot = handle.read(512)
    if len(boot) < 512 or boot[510:512] != b"\x55\xaa":
        raise ValueError("partition does not look like a FAT filesystem")
    bytes_per_sector = int.from_bytes(boot[11:13], "little")
    sectors_per_cluster = int(boot[13])
    reserved_sectors = int.from_bytes(boot[14:16], "little")
    fat_count = int(boot[16])
    root_entry_count = int.from_bytes(boot[17:19], "little")
    fat16_size = int.from_bytes(boot[22:24], "little")
    fat32_size = int.from_bytes(boot[36:40], "little")
    fat_size = fat16_size or fat32_size
    if bytes_per_sector <= 0 or sectors_per_cluster <= 0 or reserved_sectors <= 0 or fat_count <= 0 or fat_size <= 0:
        raise ValueError("FAT filesystem has invalid geometry")
    root_dir_sectors = ((root_entry_count * 32) + (bytes_per_sector - 1)) // bytes_per_sector
    first_data_sector = reserved_sectors + fat_count * fat_size + root_dir_sectors
    root_cluster = int.from_bytes(boot[44:48], "little") if fat16_size == 0 else 0
    total_sectors_16 = int.from_bytes(boot[19:21], "little")
    total_sectors_32 = int.from_bytes(boot[32:36], "little")
    total_sectors = total_sectors_16 or total_sectors_32
    data_sectors = max(0, total_sectors - first_data_sector)
    cluster_count = data_sectors // sectors_per_cluster if sectors_per_cluster else 0
    return {
        "partition_offset": int(partition_offset),
        "bytes_per_sector": bytes_per_sector,
        "sectors_per_cluster": sectors_per_cluster,
        "reserved_sectors": reserved_sectors,
        "fat_count": fat_count,
        "root_entry_count": root_entry_count,
        "root_dir_sectors": root_dir_sectors,
        "fat_size": fat_size,
        "first_data_sector": first_data_sector,
        "root_cluster": root_cluster,
        "fat_bits": 32 if fat16_size == 0 else 16,
        "cluster_size": bytes_per_sector * sectors_per_cluster,
        "total_sectors": total_sectors,
        "cluster_count": cluster_count,
    }


def _fat_cluster_offset(params: Dict[str, Any], cluster: int) -> int:
    sector = int(params["first_data_sector"]) + (int(cluster) - 2) * int(params["sectors_per_cluster"])
    return int(params["partition_offset"]) + sector * int(params["bytes_per_sector"])


def _fat_entry_value(handle, params: Dict[str, Any], cluster: int) -> int:
    if int(params["fat_bits"]) == 32:
        offset = int(params["partition_offset"]) + int(params["reserved_sectors"]) * int(params["bytes_per_sector"]) + int(cluster) * 4
        handle.seek(offset)
        return int.from_bytes(handle.read(4), "little") & 0x0FFFFFFF
    offset = int(params["partition_offset"]) + int(params["reserved_sectors"]) * int(params["bytes_per_sector"]) + int(cluster) * 2
    handle.seek(offset)
    return int.from_bytes(handle.read(2), "little")


def _set_fat_entry_value(handle, params: Dict[str, Any], cluster: int, value: int) -> None:
    entry_size = 4 if int(params["fat_bits"]) == 32 else 2
    mask = 0x0FFFFFFF if int(params["fat_bits"]) == 32 else 0xFFFF
    clean_value = int(value) & mask
    for fat_index in range(int(params["fat_count"])):
        offset = (
            int(params["partition_offset"])
            + (int(params["reserved_sectors"]) + fat_index * int(params["fat_size"])) * int(params["bytes_per_sector"])
            + int(cluster) * entry_size
        )
        handle.seek(offset)
        if int(params["fat_bits"]) == 32:
            existing = int.from_bytes(handle.read(4), "little")
            encoded = (existing & 0xF0000000) | clean_value
            handle.seek(offset)
            handle.write(encoded.to_bytes(4, "little"))
        else:
            handle.write(clean_value.to_bytes(2, "little"))


def _find_free_fat_cluster(handle, params: Dict[str, Any]) -> int:
    for cluster in range(2, int(params["cluster_count"]) + 2):
        if _fat_entry_value(handle, params, cluster) == 0:
            return cluster
    return 0


def _fat_cluster_chain(handle, params: Dict[str, Any], first_cluster: int) -> list[int]:
    if first_cluster < 2:
        return []
    chain: list[int] = []
    seen: set[int] = set()
    cluster = int(first_cluster)
    eof_marker = 0x0FFFFFF8 if int(params["fat_bits"]) == 32 else 0xFFF8
    while cluster >= 2 and cluster not in seen and len(chain) < 4096:
        seen.add(cluster)
        chain.append(cluster)
        value = _fat_entry_value(handle, params, cluster)
        if value >= eof_marker or value == 0:
            break
        cluster = value
    return chain


def _fat_short_entry_name(entry: bytes) -> str:
    raw_name = entry[0:8].decode("ascii", "ignore").strip()
    raw_ext = entry[8:11].decode("ascii", "ignore").strip()
    if raw_ext:
        return f"{raw_name}.{raw_ext}"
    return raw_name


def _fat_lfn_entry_text(entry: bytes) -> str:
    chars: list[str] = []
    for position in (1, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30):
        codepoint = int.from_bytes(entry[position : position + 2], "little")
        if codepoint in {0x0000, 0xFFFF}:
            break
        chars.append(chr(codepoint))
    return "".join(chars)


def _fat_lfn_name(entries: Sequence[bytes]) -> str:
    if not entries:
        return ""
    chunks: list[tuple[int, str]] = []
    for entry in entries:
        sequence = int(entry[0]) & 0x1F
        if sequence <= 0:
            continue
        chunks.append((sequence, _fat_lfn_entry_text(entry)))
    return "".join(text for _, text in sorted(chunks, key=lambda item: item[0]))


def _fat_first_cluster(entry: bytes) -> int:
    high = int.from_bytes(entry[20:22], "little")
    low = int.from_bytes(entry[26:28], "little")
    return (high << 16) | low


def _fat_directory_regions(handle, params: Dict[str, Any], directory_cluster: int) -> list[tuple[int, int]]:
    if directory_cluster == 0 and int(params["root_entry_count"]) > 0:
        start_sector = int(params["reserved_sectors"]) + int(params["fat_count"]) * int(params["fat_size"])
        return [
            (
                int(params["partition_offset"]) + start_sector * int(params["bytes_per_sector"]),
                int(params["root_dir_sectors"]) * int(params["bytes_per_sector"]),
            )
        ]
    root_cluster = int(params["root_cluster"]) or directory_cluster
    selected_cluster = int(directory_cluster or root_cluster)
    return [(_fat_cluster_offset(params, cluster), int(params["cluster_size"])) for cluster in _fat_cluster_chain(handle, params, selected_cluster)]


def _find_fat_path_entry(handle, params: Dict[str, Any], path: str) -> Dict[str, Any]:
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    if not parts:
        raise FileNotFoundError(path)
    directory_cluster = 0 if int(params["root_entry_count"]) > 0 else int(params["root_cluster"])
    selected: Dict[str, Any] = {}
    for part_index, part in enumerate(parts):
        wanted = part.upper()
        found: Dict[str, Any] = {}
        for region_offset, region_size in _fat_directory_regions(handle, params, directory_cluster):
            handle.seek(region_offset)
            region = handle.read(region_size)
            lfn_entries: list[bytes] = []
            for offset in range(0, len(region), 32):
                entry = region[offset : offset + 32]
                if len(entry) < 32:
                    continue
                first = entry[0]
                if first == 0x00:
                    break
                if first == 0xE5:
                    lfn_entries = []
                    continue
                if entry[11] == 0x0F:
                    lfn_entries.append(bytes(entry))
                    continue
                if entry[11] & 0x08:
                    lfn_entries = []
                    continue
                short_name = _fat_short_entry_name(entry).upper()
                long_name = _fat_lfn_name(lfn_entries).upper()
                lfn_entries = []
                if short_name != wanted and long_name != wanted:
                    continue
                found = {
                    "entry_offset": region_offset + offset,
                    "entry": entry,
                    "attr": int(entry[11]),
                    "first_cluster": _fat_first_cluster(entry),
                    "file_size": int.from_bytes(entry[28:32], "little"),
                    "name": long_name or short_name,
                }
                break
            if found:
                break
        if not found:
            raise FileNotFoundError(path)
        selected = found
        if part_index < len(parts) - 1:
            if not (int(found["attr"]) & 0x10):
                raise NotADirectoryError(part)
            directory_cluster = int(found["first_cluster"])
    return selected


def _find_fat_directory_cluster(handle, params: Dict[str, Any], directory_path: str) -> int:
    parts = [part for part in directory_path.replace("\\", "/").split("/") if part]
    if not parts:
        return 0 if int(params["root_entry_count"]) > 0 else int(params["root_cluster"])
    current = 0 if int(params["root_entry_count"]) > 0 else int(params["root_cluster"])
    current_path: list[str] = []
    for part in parts:
        current_path.append(part)
        entry = _find_fat_path_entry(handle, params, "/".join(current_path))
        if not (int(entry["attr"]) & 0x10):
            raise NotADirectoryError(directory_path)
        current = int(entry["first_cluster"])
    return current


def _fat_83_name(filename: str) -> bytes:
    if "/" in filename or "\\" in filename:
        raise ValueError("FAT filename must be a single path component")
    stem, dot, ext = filename.partition(".")
    if not stem or (dot and not ext):
        raise ValueError("FAT filename must be representable as 8.3")
    clean_stem = "".join(char for char in stem.upper() if char.isalnum() or char in {"_", "$", "~", "-"})
    clean_ext = "".join(char for char in ext.upper() if char.isalnum() or char in {"_", "$", "~", "-"})
    if not clean_stem or len(clean_stem) > 8 or len(clean_ext) > 3:
        raise ValueError("FAT filename must be representable as 8.3")
    return (clean_stem.ljust(8) + clean_ext.ljust(3)).encode("ascii")


def _create_fat_file_in_image(image_path: Path, *, partition_offset: int, file_path: str, content: bytes) -> Dict[str, Any]:
    normalized = file_path.replace("\\", "/").strip("/")
    parent_path, _, filename = normalized.rpartition("/")
    if not filename:
        return {"status": "SKIPPED", "reason": "target file name is empty", "path": file_path}
    with image_path.open("r+b") as handle:
        params = _fat_parameters(handle, int(partition_offset))
        try:
            _find_fat_path_entry(handle, params, normalized)
            return {"status": "SKIPPED", "reason": "target file already exists", "path": file_path}
        except FileNotFoundError:
            pass
        directory_cluster = _find_fat_directory_cluster(handle, params, parent_path)
        free_entry_offset = 0
        for region_offset, region_size in _fat_directory_regions(handle, params, directory_cluster):
            handle.seek(region_offset)
            region = handle.read(region_size)
            for offset in range(0, len(region), 32):
                entry = region[offset : offset + 32]
                if len(entry) < 32:
                    continue
                if entry[0] in {0x00, 0xE5}:
                    free_entry_offset = region_offset + offset
                    break
            if free_entry_offset:
                break
        if not free_entry_offset:
            return {"status": "SKIPPED", "reason": "no free FAT directory entry is available", "path": file_path}
        clusters_needed = max(1, (len(content) + int(params["cluster_size"]) - 1) // int(params["cluster_size"]))
        clusters: list[int] = []
        for _ in range(clusters_needed):
            cluster = _find_free_fat_cluster(handle, params)
            if cluster < 2:
                break
            clusters.append(cluster)
            _set_fat_entry_value(handle, params, cluster, 0x0FFFFFFF if int(params["fat_bits"]) == 32 else 0xFFFF)
        if len(clusters) != clusters_needed:
            return {"status": "SKIPPED", "reason": "no free FAT cluster is available", "path": file_path}
        eof = 0x0FFFFFFF if int(params["fat_bits"]) == 32 else 0xFFFF
        for index, cluster in enumerate(clusters):
            next_value = clusters[index + 1] if index + 1 < len(clusters) else eof
            _set_fat_entry_value(handle, params, cluster, next_value)
        remaining = bytes(content)
        for cluster in clusters:
            cluster_offset = _fat_cluster_offset(params, cluster)
            chunk = remaining[: int(params["cluster_size"])]
            remaining = remaining[int(params["cluster_size"]) :]
            handle.seek(cluster_offset)
            handle.write(chunk)
            if len(chunk) < int(params["cluster_size"]):
                handle.write(b"\x00" * (int(params["cluster_size"]) - len(chunk)))
        entry = bytearray(32)
        entry[0:11] = _fat_83_name(filename)
        entry[11] = 0x20
        first_cluster = clusters[0]
        entry[20:22] = ((first_cluster >> 16) & 0xFFFF).to_bytes(2, "little")
        entry[26:28] = (first_cluster & 0xFFFF).to_bytes(2, "little")
        entry[28:32] = int(len(content)).to_bytes(4, "little")
        handle.seek(free_entry_offset)
        handle.write(entry)
    return {
        "status": "CREATED",
        "path": file_path,
        "replacement_byte_size": len(content),
        "capacity": len(clusters) * int(params["cluster_size"]),
        "first_cluster": first_cluster,
        "cluster_count": len(clusters),
    }


def _replace_fat_file_in_image(image_path: Path, *, partition_offset: int, file_path: str, content: bytes) -> Dict[str, Any]:
    with image_path.open("r+b") as handle:
        params = _fat_parameters(handle, int(partition_offset))
        entry = _find_fat_path_entry(handle, params, file_path)
        if int(entry["attr"]) & 0x10:
            return {"status": "SKIPPED", "reason": "target path is a directory", "path": file_path}
        chain = _fat_cluster_chain(handle, params, int(entry["first_cluster"]))
        capacity = len(chain) * int(params["cluster_size"])
        if not chain:
            return {"status": "SKIPPED", "reason": "target file has no allocated clusters", "path": file_path}
        if len(content) > capacity:
            return {
                "status": "SKIPPED",
                "reason": "replacement exceeds existing FAT cluster chain capacity",
                "path": file_path,
                "replacement_byte_size": len(content),
                "capacity": capacity,
            }
        remaining = bytes(content)
        for cluster in chain:
            cluster_offset = _fat_cluster_offset(params, cluster)
            chunk = remaining[: int(params["cluster_size"])]
            remaining = remaining[int(params["cluster_size"]) :]
            handle.seek(cluster_offset)
            handle.write(chunk)
            if len(chunk) < int(params["cluster_size"]):
                handle.write(b"\x00" * (int(params["cluster_size"]) - len(chunk)))
        handle.seek(int(entry["entry_offset"]) + 28)
        handle.write(int(len(content)).to_bytes(4, "little"))
    return {
        "status": "PATCHED",
        "path": file_path,
        "replacement_byte_size": len(content),
        "capacity": capacity,
        "first_cluster": int(entry["first_cluster"]),
    }


def _read_fat_file_in_image(image_path: Path, *, partition_offset: int, file_path: str) -> bytes:
    with image_path.open("rb") as handle:
        params = _fat_parameters(handle, int(partition_offset))
        entry = _find_fat_path_entry(handle, params, file_path)
        if int(entry["attr"]) & 0x10:
            raise IsADirectoryError(file_path)
        payload = bytearray()
        for cluster in _fat_cluster_chain(handle, params, int(entry["first_cluster"])):
            handle.seek(_fat_cluster_offset(params, cluster))
            payload.extend(handle.read(int(params["cluster_size"])))
        return bytes(payload[: int(entry["file_size"])])


def _ensure_efi_boot_fallback_loaders(disk_path: Path, *, partition_offset: int) -> Dict[str, Any]:
    copies: list[Dict[str, Any]] = []
    targets = [
        ("EFI/BOOT/grubaa64.efi", ("EFI/debian/grubaa64.efi",)),
        ("EFI/BOOT/mmaa64.efi", ("EFI/debian/mmaa64.efi",)),
    ]
    for target, sources in targets:
        with disk_path.open("rb") as handle:
            params = _fat_parameters(handle, int(partition_offset))
            try:
                existing = _find_fat_path_entry(handle, params, target)
            except FileNotFoundError:
                existing = {}
        if existing:
            copies.append({"target": target, "status": "ALREADY_PRESENT", "file_size": existing.get("file_size")})
            continue
        selected_source = ""
        source_bytes = b""
        for source in sources:
            try:
                source_bytes = _read_fat_file_in_image(disk_path, partition_offset=int(partition_offset), file_path=source)
                selected_source = source
                break
            except (FileNotFoundError, IsADirectoryError, OSError, ValueError):
                continue
        if not selected_source:
            copies.append({"target": target, "status": "SKIPPED", "reason": "fallback source loader was not found"})
            continue
        try:
            result = _create_fat_file_in_image(
                disk_path,
                partition_offset=int(partition_offset),
                file_path=target,
                content=source_bytes,
            )
        except (OSError, ValueError, FileNotFoundError, NotADirectoryError) as exc:
            result = {"status": "SKIPPED", "reason": str(exc), "path": target}
        result["source"] = selected_source
        result["target"] = target
        result["sha256"] = hashlib.sha256(source_bytes).hexdigest()
        copies.append(result)
    copied = [copy for copy in copies if str(copy.get("status") or "") == "CREATED"]
    skipped = [copy for copy in copies if str(copy.get("status") or "") == "SKIPPED"]
    return {
        "status": "READY" if not skipped else ("PARTIAL" if copied else "SKIPPED"),
        "copies": copies,
        "created_count": len(copied),
        "skipped_count": len(skipped),
        "purpose": "ensure ARM64 EFI fallback shim can find grubaa64.efi in EFI/BOOT",
    }


def _efi_boot_fallback_loaders_present(disk_path: Path, *, partition_offset: int) -> Dict[str, Any]:
    targets = ("EFI/BOOT/grubaa64.efi", "EFI/BOOT/mmaa64.efi")
    copies: list[Dict[str, Any]] = []
    try:
        with disk_path.open("rb") as handle:
            params = _fat_parameters(handle, int(partition_offset))
            for target in targets:
                try:
                    entry = _find_fat_path_entry(handle, params, target)
                    if int(entry["attr"]) & 0x10:
                        copies.append({"target": target, "status": "MISSING", "reason": "fallback loader path is a directory"})
                    else:
                        copies.append({"target": target, "status": "ALREADY_PRESENT", "file_size": entry.get("file_size")})
                except (FileNotFoundError, IsADirectoryError, OSError, ValueError) as exc:
                    copies.append({"target": target, "status": "MISSING", "reason": str(exc)})
    except (OSError, ValueError) as exc:
        return {
            "status": "SKIPPED",
            "copies": [],
            "created_count": 0,
            "skipped_count": len(targets),
            "reason": str(exc),
            "purpose": "check ARM64 EFI fallback shim availability without mutating the ESP",
        }
    missing = [copy for copy in copies if str(copy.get("status") or "") != "ALREADY_PRESENT"]
    return {
        "status": "READY" if not missing else "MISSING",
        "copies": copies,
        "created_count": 0,
        "skipped_count": len(missing),
        "purpose": "check ARM64 EFI fallback shim availability without mutating the ESP",
    }


def _scan_linux_boot_artifact_paths(disk_path: Path) -> Dict[str, str]:
    kernels: set[str] = set()
    initrds: set[str] = set()
    kernel_pattern = re.compile(rb"vmlinuz-[A-Za-z0-9._+~:-]+")
    initrd_pattern = re.compile(rb"initrd\.img-[A-Za-z0-9._+~:-]+")
    overlap = 128
    previous = b""
    try:
        with disk_path.open("rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                data = previous + chunk
                for match in kernel_pattern.findall(data):
                    kernels.add(match.decode("ascii", "ignore"))
                for match in initrd_pattern.findall(data):
                    initrds.add(match.decode("ascii", "ignore"))
                previous = data[-overlap:]
    except OSError:
        return {}
    kernel_versions = {name.removeprefix("vmlinuz-"): name for name in kernels}
    initrd_versions = {name.removeprefix("initrd.img-"): name for name in initrds}
    common_versions = sorted(set(kernel_versions).intersection(initrd_versions))
    if not common_versions:
        return {}
    selected = common_versions[-1]
    return {
        "kernel_path": f"/boot/{kernel_versions[selected]}",
        "initrd_path": f"/boot/{initrd_versions[selected]}",
        "kernel_version": selected,
    }


def _observable_grub_config_tokens_present(config: bytes) -> bool:
    required_tokens = (
        b"search.fs_uuid ",
        b" root\n",
        b"linux ($root)",
        b" root=UUID=",
        b"console=hvc0",
        b"systemd.show_status=1",
        b"systemd.journald.forward_to_console=1",
        b"cloud-init=enabled ds=nocloud",
        b"initrd ($root)",
        b"boot\n",
    )
    return all(token in config for token in required_tokens)


def _boot_artifacts_from_observable_grub_config(config: bytes) -> Dict[str, str]:
    try:
        text = config.decode("utf-8", "ignore")
    except UnicodeDecodeError:
        return {}
    artifacts: Dict[str, str] = {}
    linux_match = re.search(r"(?m)^linux \(\$root\)(\S+)", text)
    initrd_match = re.search(r"(?m)^initrd \(\$root\)(\S+)", text)
    root_match = re.search(r"(?m)^search\.fs_uuid\s+([0-9A-Fa-f-]+)\s+root", text)
    if linux_match:
        artifacts["kernel_path"] = linux_match.group(1)
        kernel_name = Path(linux_match.group(1)).name
        if kernel_name.startswith("vmlinuz-"):
            artifacts["kernel_version"] = kernel_name.removeprefix("vmlinuz-")
    if initrd_match:
        artifacts["initrd_path"] = initrd_match.group(1)
    if root_match:
        artifacts["root_uuid"] = root_match.group(1)
    return artifacts


def _existing_observable_efi_boot_patch_report(disk_path: Path, *, partition_offset: int) -> Dict[str, Any]:
    configs: list[bytes] = []
    attempts: list[Dict[str, Any]] = []
    for grub_path in MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS:
        try:
            config = _read_fat_file_in_image(disk_path, partition_offset=int(partition_offset), file_path=grub_path)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError):
            return {}
        if not _observable_grub_config_tokens_present(config):
            return {}
        configs.append(config)
        attempts.append(
            {
                "status": "UNCHANGED",
                "path": grub_path,
                "reason": "existing grub config already matches Con OS observable boot markers",
                "config_sha256": hashlib.sha256(config).hexdigest(),
                "config_byte_size": len(config),
            }
        )
    if not configs or any(config != configs[0] for config in configs[1:]):
        return {}
    fallback_loader_report = _efi_boot_fallback_loaders_present(disk_path, partition_offset=int(partition_offset))
    if str(fallback_loader_report.get("status") or "") != "READY":
        return {}
    boot_artifacts = _boot_artifacts_from_observable_grub_config(configs[0])
    return {
        "status": "UNCHANGED",
        "disk_path": str(disk_path),
        "patched_paths": [],
        "unchanged_paths": list(MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS),
        "attempts": attempts,
        "config_sha256": hashlib.sha256(configs[0]).hexdigest(),
        "config_byte_size": len(configs[0]),
        "boot_artifacts": boot_artifacts,
        "root_uuid": str(boot_artifacts.get("root_uuid") or ""),
        "fallback_loaders": fallback_loader_report,
        "agent_initrd": {
            "status": "UNCHANGED_OR_DISABLED",
            "reason": "existing observable GRUB config was reused without mutating the ESP",
        },
        "agent_initrd_injection_enabled": False,
        "idempotent_skip_count": len(MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS),
        "write_avoided_count": len(MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS),
        "boot_artifacts_source": "existing_observable_grub_config",
        "purpose": "reuse existing observable EFI Linux boot config without rescanning the root disk",
    }


def _standalone_guest_agent_initrd_payload(*, guest_agent_path: Path, python_path: str, port: int) -> bytes:
    launcher = _guest_agent_launcher_script(python_path=python_path, port=port)
    unit = _guest_agent_systemd_unit(python_path=python_path, port=port)
    entries = [
        ("conos/conos_guest_agent.py", guest_agent_path.read_bytes(), 0o100755),
        ("conos/conos_guest_agent_launcher.sh", launcher.encode("utf-8"), 0o100755),
        ("etc/systemd/system/conos-guest-agent.service", unit.encode("utf-8"), 0o100644),
        ("scripts/init-top/conos-trace", _guest_agent_initramfs_trace_hook(stage="init-top").encode("utf-8"), 0o100755),
        ("scripts/local-top/conos-trace", _guest_agent_initramfs_trace_hook(stage="local-top").encode("utf-8"), 0o100755),
        ("scripts/local-bottom/conos-guest-agent", _guest_agent_initramfs_local_bottom_hook().encode("utf-8"), 0o100755),
        ("scripts/init-top/ORDER", b"/scripts/init-top/conos-trace \"$@\"\n", 0o100644),
        ("scripts/local-top/ORDER", b"/scripts/local-top/conos-trace \"$@\"\n", 0o100644),
        ("scripts/local-bottom/ORDER", b"/scripts/local-bottom/conos-guest-agent \"$@\"\n", 0o100644),
        ("conf/modules", b"virtio_pci\nvirtio_blk\nvirtio_console\nvirtiofs\n", 0o100644),
    ]
    return _build_newc_archive(entries)


def _observable_efi_grub_config(
    *,
    root_uuid: str,
    kernel_path: str = "",
    initrd_path: str = "",
    agent_initrd_path: str = "",
) -> bytes:
    selected_kernel = _clean(kernel_path) or "/vmlinuz"
    selected_initrd = _clean(initrd_path) or "/initrd.img"
    agent_initrd = _clean(agent_initrd_path)
    initrd_line = f"initrd ($root){selected_initrd}"
    if agent_initrd:
        initrd_line = f"{initrd_line} ($esp)/{agent_initrd}"
    return (
        "set esp=$root\n"
        f"search.fs_uuid {root_uuid} root\n"
        "set timeout=0\n"
        "set default=0\n"
        f"linux ($root){selected_kernel} root=UUID={root_uuid} ro console=hvc0 console=tty0 "
        "systemd.show_status=1 systemd.journald.forward_to_console=1 cloud-init=enabled ds=nocloud\n"
        f"{initrd_line}\n"
        "boot\n"
    ).encode("utf-8")


def _apply_efi_observable_boot_patch(
    disk_path: Path,
    *,
    guest_agent_path: str = "",
    guest_python_path: str = "/usr/bin/python3",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    inject_agent_initrd: bool = False,
) -> Dict[str, Any]:
    partitions = _gpt_partitions(disk_path)
    esp = next((partition for partition in partitions if str(partition.get("type_guid") or "").lower() == str(EFI_SYSTEM_PARTITION_GUID)), None)
    if not esp:
        return {"status": "SKIPPED", "reason": "EFI system partition was not found", "disk_path": str(disk_path)}
    if not inject_agent_initrd:
        existing_report = _existing_observable_efi_boot_patch_report(disk_path, partition_offset=int(esp["byte_offset"]))
        if existing_report:
            existing_report["esp_partition_index"] = esp.get("index")
            existing_report["esp_partition_offset"] = esp.get("byte_offset")
            return existing_report
    root_uuid = _managed_vm_root_ext_uuid(disk_path, partitions)
    if not root_uuid:
        return {"status": "SKIPPED", "reason": "Linux root ext4 UUID was not found", "disk_path": str(disk_path)}
    boot_artifacts = _scan_linux_boot_artifact_paths(disk_path)
    fallback_loader_report = _ensure_efi_boot_fallback_loaders(disk_path, partition_offset=int(esp["byte_offset"]))
    agent_initrd_report: Dict[str, Any] = {
        "status": "DISABLED",
        "reason": "agent initrd injection is experimental and must be explicitly enabled by the image manifest",
    }
    agent_initrd_path = ""
    agent = Path(_clean(guest_agent_path) or managed_vm_guest_agent_source_path()).expanduser().resolve()
    if inject_agent_initrd and agent.exists() and agent.is_file():
        agent_initrd = _standalone_guest_agent_initrd_payload(
            guest_agent_path=agent,
            python_path=_clean(guest_python_path) or "/usr/bin/python3",
            port=int(guest_agent_port),
        )
        try:
            agent_initrd_report = _replace_fat_file_in_image(
                disk_path,
                partition_offset=int(esp["byte_offset"]),
                file_path=MANAGED_VM_EFI_AGENT_INITRD_PATH,
                content=agent_initrd,
            )
        except FileNotFoundError:
            agent_initrd_report = _create_fat_file_in_image(
                disk_path,
                partition_offset=int(esp["byte_offset"]),
                file_path=MANAGED_VM_EFI_AGENT_INITRD_PATH,
                content=agent_initrd,
            )
        if str(agent_initrd_report.get("status") or "") in {"PATCHED", "CREATED"}:
            agent_initrd_path = MANAGED_VM_EFI_AGENT_INITRD_PATH
            agent_initrd_report["sha256"] = hashlib.sha256(agent_initrd).hexdigest()
            agent_initrd_report["byte_size"] = len(agent_initrd)
    elif inject_agent_initrd:
        agent_initrd_report = {
            "status": "SKIPPED",
            "reason": "guest agent source file was not available for initrd injection",
            "guest_agent_path": str(agent),
        }
    grub_config = _observable_efi_grub_config(
        root_uuid=root_uuid,
        kernel_path=str(boot_artifacts.get("kernel_path") or ""),
        initrd_path=str(boot_artifacts.get("initrd_path") or ""),
        agent_initrd_path=agent_initrd_path,
    )
    attempts: list[Dict[str, Any]] = []
    patched_paths: list[str] = []
    for grub_path in MANAGED_VM_OBSERVABLE_GRUB_CONFIG_PATHS:
        try:
            result = _replace_fat_file_in_image(
                disk_path,
                partition_offset=int(esp["byte_offset"]),
                file_path=grub_path,
                content=grub_config,
            )
        except FileNotFoundError:
            result = _create_fat_file_in_image(
                disk_path,
                partition_offset=int(esp["byte_offset"]),
                file_path=grub_path,
                content=grub_config,
            )
            attempts.append(result)
            if str(result.get("status") or "") == "CREATED":
                patched_paths.append(grub_path)
            continue
        except (OSError, ValueError) as exc:
            attempts.append({"status": "SKIPPED", "path": grub_path, "reason": str(exc)})
            continue
        attempts.append(result)
        if str(result.get("status") or "") in {"PATCHED", "CREATED"}:
            patched_paths.append(grub_path)
    status = "PATCHED" if patched_paths else "SKIPPED"
    return {
        "status": status,
        "disk_path": str(disk_path),
        "esp_partition_index": esp.get("index"),
        "esp_partition_offset": esp.get("byte_offset"),
        "root_uuid": root_uuid,
        "patched_paths": patched_paths,
        "attempts": attempts,
        "config_sha256": hashlib.sha256(grub_config).hexdigest(),
        "config_byte_size": len(grub_config),
        "boot_artifacts": boot_artifacts,
        "fallback_loaders": fallback_loader_report,
        "agent_initrd": agent_initrd_report,
        "agent_initrd_injection_enabled": bool(inject_agent_initrd),
        "purpose": "force observable EFI Linux boot with console=hvc0 and cloud-init enabled",
    }


def _cloud_init_marker_paths(shared_dir: Path | None) -> list[Path]:
    if shared_dir is None or not shared_dir.exists():
        return []
    return [shared_dir / name for name in MANAGED_VM_CLOUD_INIT_MARKER_FILES if (shared_dir / name).exists()]


def _update_instance_runtime_fields(
    *,
    state_root: str,
    instance_id: str,
    runtime_payload: Dict[str, Any],
) -> Dict[str, Any]:
    instance_manifest = load_managed_vm_instance_manifest(state_root, instance_id)
    if not instance_manifest:
        return {}
    console_log_raw = str(runtime_payload.get("guest_console_log_path") or "")
    shared_dir_raw = str(runtime_payload.get("guest_shared_dir_path") or "")
    shared_dir = Path(shared_dir_raw) if shared_dir_raw else None
    shared_marker = shared_dir / "cloud-init-runcmd.txt" if shared_dir is not None else None
    cloud_init_marker_names = [path.name for path in _cloud_init_marker_paths(shared_dir)]
    initramfs_marker_names = (
        [path.name for path in sorted(shared_dir.glob("conos-initramfs-*.txt"))]
        if shared_dir is not None and shared_dir.exists()
        else []
    )
    last_guest_boot_diagnostic = _managed_vm_guest_boot_diagnostic(
        runtime_manifest=runtime_payload,
        process_alive=_process_alive(runtime_payload.get("process_pid")),
        guest_console_tail=_tail_file(Path(console_log_raw)) if console_log_raw else "",
        guest_shared_runcmd_marker_present=bool(shared_marker and shared_marker.exists()),
        guest_cloud_init_markers=cloud_init_marker_names,
        guest_initramfs_trace_markers=initramfs_marker_names,
    )
    instance_manifest.update(
        {
            "status": runtime_payload.get("status", ""),
            "lifecycle_state": runtime_payload.get("lifecycle_state", ""),
            "runtime_manifest_path": str(managed_vm_runtime_manifest_path(state_root, instance_id)),
            "runtime_manifest_present": True,
            "process_pid": str(runtime_payload.get("process_pid") or ""),
            "process_alive": bool(runtime_payload.get("process_alive", False)),
            "overlay_path": str(runtime_payload.get("overlay_path") or managed_vm_overlay_path(state_root, instance_id)),
            "overlay_present": bool(runtime_payload.get("overlay_present", False)),
            "writable_disk_path": str(runtime_payload.get("writable_disk_path") or ""),
            "writable_disk_present": bool(runtime_payload.get("writable_disk_present", False)),
            "cloud_init_seed_enabled": bool(runtime_payload.get("cloud_init_seed_enabled", False)),
            "cloud_init_seed_path": str(runtime_payload.get("cloud_init_seed_path") or ""),
            "cloud_init_seed_present": bool(runtime_payload.get("cloud_init_seed_present", False)),
            "cloud_init_seed_read_only": bool(runtime_payload.get("cloud_init_seed_read_only", False)),
            "cloud_init_guest_capability": (
                runtime_payload.get("cloud_init_guest_capability")
                if isinstance(runtime_payload.get("cloud_init_guest_capability"), dict)
                else {}
            ),
            "guest_cloud_init_marker_present": bool(cloud_init_marker_names),
            "guest_cloud_init_markers": cloud_init_marker_names,
            "virtual_machine_started": bool(runtime_payload.get("virtual_machine_started", False)),
            "execution_ready": bool(runtime_payload.get("execution_ready", False)),
            "guest_agent_ready": bool(runtime_payload.get("guest_agent_ready", False)),
            "guest_agent_transport": str(runtime_payload.get("guest_agent_transport") or ""),
            "guest_agent_port": runtime_payload.get("guest_agent_port"),
            "boot_mode": str(runtime_payload.get("boot_mode") or ""),
            "last_start_attempt_at": runtime_payload.get("started_at", _now_iso()),
            "last_start_returncode": runtime_payload.get("last_runner_returncode"),
            "launcher_kind": runtime_payload.get("launcher_kind", ""),
            "last_guest_boot_diagnostic": last_guest_boot_diagnostic,
            "no_host_fallback": True,
        }
    )
    _write_json(managed_vm_instance_manifest_path(state_root, instance_id), instance_manifest)
    return instance_manifest


def _terminate_runner_process(pid_value: object, *, timeout_seconds: int = 5) -> Dict[str, Any]:
    try:
        pid = int(str(pid_value or "").strip())
    except ValueError:
        return {"attempted": False, "reason": "invalid_pid"}
    if pid <= 0:
        return {"attempted": False, "reason": "empty_pid"}
    if pid == os.getpid():
        return {"attempted": False, "reason": "refusing_to_signal_current_process", "pid": str(pid)}
    if not _process_alive(pid):
        return {"attempted": False, "reason": "process_not_alive", "pid": str(pid)}
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        if not _process_alive(pid):
            return {"attempted": True, "pid": str(pid), "terminated": True, "signal": "SIGTERM", "reason": "process_exited"}
        return {"attempted": True, "pid": str(pid), "terminated": False, "reason": str(exc)}
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if _process_finished_after_signal(pid):
            return {"attempted": True, "pid": str(pid), "terminated": True, "signal": "SIGTERM"}
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError as exc:
        if not _process_alive(pid):
            return {
                "attempted": True,
                "pid": str(pid),
                "terminated": True,
                "signal": "SIGTERM",
                "reason": "process_exited_after_sigterm_timeout",
            }
        return {
            "attempted": True,
            "pid": str(pid),
            "terminated": False,
            "signal": "SIGTERM",
            "force_signal_attempted": "SIGKILL",
            "reason": str(exc),
        }
    force_deadline = time.monotonic() + 2.0
    while time.monotonic() < force_deadline:
        if _process_finished_after_signal(pid):
            return {
                "attempted": True,
                "pid": str(pid),
                "terminated": True,
                "signal": "SIGTERM",
                "force_signal": "SIGKILL",
                "reason": "sigterm_timeout_forced",
            }
        time.sleep(0.05)
    return {
        "attempted": True,
        "pid": str(pid),
        "terminated": False,
        "signal": "SIGTERM",
        "force_signal_attempted": "SIGKILL",
        "reason": "timeout",
    }


def _crash_runner_process(
    pid_value: object,
    *,
    signal_name: str = "SIGKILL",
    timeout_seconds: int = 5,
) -> Dict[str, Any]:
    try:
        pid = int(str(pid_value or "").strip())
    except ValueError:
        return {"attempted": False, "reason": "invalid_pid"}
    if pid <= 0:
        return {"attempted": False, "reason": "empty_pid"}
    if pid == os.getpid():
        return {"attempted": False, "reason": "refusing_to_signal_current_process", "pid": str(pid)}
    if not _process_alive(pid):
        return {"attempted": False, "reason": "process_not_alive", "pid": str(pid)}
    normalized_signal = str(signal_name or "SIGKILL").strip().upper()
    signal_map = {"SIGKILL": signal.SIGKILL, "SIGTERM": signal.SIGTERM}
    if normalized_signal not in signal_map:
        return {"attempted": False, "pid": str(pid), "reason": f"unsupported_signal:{normalized_signal}"}
    try:
        os.kill(pid, signal_map[normalized_signal])
    except OSError as exc:
        if not _process_alive(pid):
            return {
                "attempted": True,
                "pid": str(pid),
                "terminated": True,
                "signal": normalized_signal,
                "reason": "process_exited",
            }
        return {
            "attempted": True,
            "pid": str(pid),
            "terminated": False,
            "signal": normalized_signal,
            "reason": str(exc),
        }
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if _process_finished_after_signal(pid):
            return {
                "attempted": True,
                "pid": str(pid),
                "terminated": True,
                "signal": normalized_signal,
                "reason": "failure_injection_signal_observed",
            }
        time.sleep(0.05)
    return {
        "attempted": True,
        "pid": str(pid),
        "terminated": False,
        "signal": normalized_signal,
        "reason": "timeout",
    }


def _managed_vm_guest_boot_diagnostic(
    *,
    runtime_manifest: Dict[str, Any],
    process_alive: bool,
    guest_console_tail: str = "",
    guest_shared_runcmd_marker_present: bool = False,
    guest_cloud_init_markers: Sequence[str] | None = None,
    guest_initramfs_trace_markers: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Classify the guest-side boot/agent stage without trusting timeout alone."""

    if not runtime_manifest:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "NO_RUNTIME_MANIFEST",
            "blocked_stage": "host_start",
            "reason": "runtime manifest is missing",
            "observed_signals": {},
            "likely_causes": ["VM runner did not write runtime state"],
            "recommended_next_steps": ["inspect runner stdout/stderr and host virtualization capability"],
        }

    vm_started = bool(runtime_manifest.get("virtual_machine_started", False))
    guest_ready = bool(runtime_manifest.get("guest_agent_ready", False))
    execution_ready = bool(runtime_manifest.get("execution_ready", False))
    listener_ready = bool(runtime_manifest.get("guest_agent_listener_ready", False))
    seed_enabled = bool(runtime_manifest.get("cloud_init_seed_enabled", False))
    seed_present = bool(runtime_manifest.get("cloud_init_seed_present", False))
    shared_dir_present = bool(runtime_manifest.get("guest_shared_dir_present", False))
    console_observed = bool(_clean(guest_console_tail))
    boot_mode = str(runtime_manifest.get("boot_mode") or "")
    cloud_init_guest_capability = (
        runtime_manifest.get("cloud_init_guest_capability")
        if isinstance(runtime_manifest.get("cloud_init_guest_capability"), dict)
        else {}
    )
    cloud_init_markers = list(guest_cloud_init_markers or [])
    initramfs_markers = list(guest_initramfs_trace_markers or [])
    signals = {
        "virtual_machine_started": vm_started,
        "process_alive": bool(process_alive),
        "guest_agent_listener_ready": listener_ready,
        "guest_agent_ready": guest_ready,
        "execution_ready": execution_ready,
        "boot_mode": boot_mode,
        "cloud_init_seed_enabled": seed_enabled,
        "cloud_init_seed_present": seed_present,
        "guest_console_output_observed": console_observed,
        "guest_shared_dir_present": shared_dir_present,
        "guest_shared_runcmd_marker_present": bool(guest_shared_runcmd_marker_present),
        "guest_cloud_init_marker_present": bool(cloud_init_markers),
        "guest_cloud_init_markers": cloud_init_markers,
        "guest_initramfs_trace_marker_present": bool(initramfs_markers),
        "guest_initramfs_trace_markers": initramfs_markers,
        "cloud_init_guest_capability": cloud_init_guest_capability,
    }
    if str(runtime_manifest.get("blocker_type") or "") == "unsupported_boot_artifact" or str(
        runtime_manifest.get("status") or ""
    ) == "START_BLOCKED_UNSUPPORTED_BOOT_ARTIFACT":
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "UNSUPPORTED_BOOT_ARTIFACT",
            "blocked_stage": "host_start_preflight",
            "reason": str(runtime_manifest.get("reason") or "linux_direct boot artifact is unsupported"),
            "observed_signals": signals,
            "likely_causes": ["kernel artifact is not suitable for VZLinuxBootLoader"],
            "recommended_next_steps": [
                "use a raw Linux kernel Image for linux_direct boot",
                "or register this image through an EFI boot path instead of direct Linux boot",
            ],
        }
    if guest_ready and execution_ready:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "GUEST_AGENT_READY",
            "blocked_stage": "",
            "reason": "guest agent handshake completed",
            "observed_signals": signals,
            "likely_causes": [],
            "recommended_next_steps": [],
        }
    if not vm_started:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "HOST_VM_NOT_STARTED",
            "blocked_stage": "host_start",
            "reason": "Apple Virtualization did not report a started VM",
            "observed_signals": signals,
            "likely_causes": ["host virtualization failure", "invalid VM configuration", "runner exited before start callback"],
            "recommended_next_steps": ["inspect start_report blocker_type and runner stderr"],
        }
    if not process_alive:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "VM_PROCESS_EXITED_BEFORE_AGENT",
            "blocked_stage": "guest_boot",
            "reason": "VM process exited before guest agent readiness",
            "observed_signals": signals,
            "likely_causes": ["guest kernel panic or boot loader failure", "unsupported disk/kernel boot path"],
            "recommended_next_steps": ["inspect runner stderr and guest console log"],
        }
    if seed_enabled and seed_present and "cloud-init-bootcmd.txt" in cloud_init_markers and not guest_shared_runcmd_marker_present:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "CLOUD_INIT_BOOTCMD_OBSERVED_RUNCMD_NOT_OBSERVED",
            "blocked_stage": "cloud_init",
            "reason": "cloud-init bootcmd marker was observed, but runcmd marker was not observed",
            "observed_signals": signals,
            "likely_causes": [
                "cloud-init config/final stage did not complete",
                "runcmd failed before guest-agent enablement",
                "guest systemd or cloud-init modules are not progressing to final stage",
            ],
            "recommended_next_steps": [
                "inspect cloud-init status and logs from the guest disk",
                "verify systemd and cloud-init final modules in the base image",
            ],
        }
    if seed_enabled and seed_present and not guest_shared_runcmd_marker_present and not console_observed:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "GUEST_BOOT_NO_OBSERVABILITY",
            "blocked_stage": "guest_boot_or_cloud_init",
            "reason": "VM is running, but neither guest console output nor cloud-init runcmd marker was observed",
            "observed_signals": signals,
            "likely_causes": [
                "EFI cloud image may not be emitting to the Apple virtio console",
                "NoCloud seed may not be detected by the guest",
                "guest may not have reached cloud-init runcmd",
            ],
            "recommended_next_steps": [
                "inspect the guest boot configuration or use a direct Linux boot recipe with explicit console=hvc0",
                "prefer a digest-pinned kernel/initrd path for managed VM bootstrap",
            ],
        }
    if seed_enabled and seed_present and not guest_shared_runcmd_marker_present:
        if str(cloud_init_guest_capability.get("status") or "") == "UNAVAILABLE":
            return {
                "schema_version": MANAGED_VM_PROVIDER_VERSION,
                "diagnosis_status": "CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE",
                "blocked_stage": "guest_agent_installation",
                "reason": "NoCloud seed was attached, but the guest disk does not appear to contain cloud-init services",
                "observed_signals": signals,
                "likely_causes": [
                    "base image is an EFI Linux image without cloud-init installed",
                    "guest-agent installation mode does not match the base image capability",
                ],
                "recommended_next_steps": [
                    "use a base image with cloud-init installed and enabled",
                    "or register a linux_direct image with a verified initrd guest-agent bundle",
                    "or preinstall the Con OS guest agent in the base image before registration",
                ],
            }
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "CLOUD_INIT_RUNCMD_NOT_OBSERVED",
            "blocked_stage": "cloud_init",
            "reason": "guest console produced output, but cloud-init runcmd marker was not observed",
            "observed_signals": signals,
            "likely_causes": ["NoCloud seed was not detected", "cloud-init failed before runcmd", "virtiofs mount failed inside guest"],
            "recommended_next_steps": ["inspect cloud-init logs from the guest disk or switch to initrd guest-agent injection"],
        }
    if "cloud-init-agent-enable.txt" in cloud_init_markers and not guest_ready:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "CLOUD_INIT_AGENT_ENABLE_OBSERVED_HANDSHAKE_NOT_OBSERVED",
            "blocked_stage": "guest_agent",
            "reason": "cloud-init attempted to enable the guest agent, but the host did not receive the vsock handshake",
            "observed_signals": signals,
            "likely_causes": ["guest agent failed after systemd enablement", "AF_VSOCK unavailable in guest", "host/guest vsock port mismatch"],
            "recommended_next_steps": ["inspect cloud-init-agent-enable.txt and guest agent logs from the shared directory or guest disk"],
        }
    if guest_shared_runcmd_marker_present and not guest_ready:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "GUEST_AGENT_HANDSHAKE_NOT_OBSERVED",
            "blocked_stage": "guest_agent",
            "reason": "cloud-init runcmd marker was observed, but the guest agent did not connect to host vsock",
            "observed_signals": signals,
            "likely_causes": ["guest agent failed to start", "AF_VSOCK unavailable in guest Python/kernel", "host/guest vsock port mismatch"],
            "recommended_next_steps": ["capture guest agent logs through the shared directory and verify vsock support inside guest"],
        }
    if initramfs_markers and not guest_ready:
        reached_local_bottom = any("local-bottom" in marker for marker in initramfs_markers)
        reached_install = any("agent-installed" in marker or "local-bottom" in marker for marker in initramfs_markers)
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "INITRAMFS_TRACE_OBSERVED_AGENT_NOT_READY",
            "blocked_stage": "guest_agent" if reached_install else "guest_root_mount_or_initramfs",
            "reason": "initramfs trace markers reached the host share, but guest agent readiness was not proven",
            "observed_signals": signals,
            "likely_causes": (
                ["systemd did not start the guest agent", "guest Python or AF_VSOCK failed", "host/guest vsock port mismatch"]
                if reached_local_bottom
                else ["root filesystem was not mounted", "local-bottom did not run", "initramfs stopped before root handoff"]
            ),
            "recommended_next_steps": [
                "inspect initramfs trace marker contents from the guest share",
                "if local-bottom was reached, inspect offline rootfs for /opt/conos and the systemd unit",
            ],
        }
    if boot_mode == "linux_direct" and not console_observed and not initramfs_markers:
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "diagnosis_status": "LINUX_DIRECT_NO_EARLY_GUEST_SIGNAL",
            "blocked_stage": "guest_boot_or_initramfs",
            "reason": "VM is running, but no guest console output or initramfs trace marker has been observed",
            "observed_signals": signals,
            "likely_causes": [
                "kernel/initrd does not expose a usable virtio console during early boot",
                "initramfs cannot mount the host shared directory for trace markers",
                "direct Linux boot artifacts may not match the guest disk root device or module set",
            ],
            "recommended_next_steps": [
                "verify the initrd contains the Con OS init wrapper and only one /init entry",
                "try a direct Linux boot artifact set known to emit console=hvc0 under Apple Virtualization",
                "or use an EFI/cloud-init path with guest-visible diagnostics",
            ],
        }
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "diagnosis_status": "GUEST_AGENT_NOT_READY",
        "blocked_stage": "guest_agent",
        "reason": "VM is running but guest agent readiness has not been proven",
        "observed_signals": signals,
        "likely_causes": ["guest agent autostart did not complete", "guest transport not connected"],
        "recommended_next_steps": ["inspect guest diagnostics and retry with longer guest wait only if progress signals are present"],
    }


def _managed_vm_boot_path_recommendation(
    *,
    boot_mode: str = "",
    guest_boot_diagnostic: Dict[str, Any] | None = None,
    start_status: str = "",
) -> Dict[str, Any]:
    """Suggest the next boot path without pretending readiness was proven."""

    diagnostic = guest_boot_diagnostic if isinstance(guest_boot_diagnostic, dict) else {}
    diagnostic_status = str(diagnostic.get("diagnosis_status") or "")
    current_boot_mode = _clean(boot_mode) or str(
        (diagnostic.get("observed_signals") or {}).get("boot_mode") or ""
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "status": "NO_BOOT_PATH_RECOMMENDATION",
        "current_boot_mode": current_boot_mode,
        "diagnosis_status": diagnostic_status,
        "start_status": _clean(start_status),
        "recommended_boot_mode": current_boot_mode,
        "reason": "",
        "recommended_next_steps": [],
        "retry_same_path": True,
        "no_host_fallback": True,
    }
    if diagnostic_status == "UNSUPPORTED_BOOT_ARTIFACT":
        payload.update(
            {
                "status": "BOOT_PATH_ARTIFACT_INCOMPATIBLE",
                "recommended_boot_mode": "efi_disk_or_raw_linux_direct",
                "reason": "linux_direct requires a raw Linux kernel Image; this artifact is not suitable for that path",
                "recommended_next_steps": [
                    "use a raw Linux kernel Image for linux_direct",
                    "or register the image through the EFI/cloud-init boot path",
                ],
                "retry_same_path": False,
            }
        )
    elif diagnostic_status == "LINUX_DIRECT_NO_EARLY_GUEST_SIGNAL":
        payload.update(
            {
                "status": "BOOT_PATH_NO_EARLY_GUEST_SIGNAL",
                "recommended_boot_mode": "efi_disk_cloud_init_or_known_good_linux_direct",
                "reason": "linux_direct reached a live VM process but produced no guest console or initramfs trace signal",
                "recommended_next_steps": [
                    "prefer the EFI/cloud-init managed image path for this disk when available",
                    "otherwise retry linux_direct only with a known-good kernel/initrd artifact set that emits console=hvc0",
                    "do not mark the image boot_verified until guest-agent handshake succeeds",
                ],
                "retry_same_path": False,
            }
        )
    elif diagnostic_status in {"GUEST_BOOT_NO_OBSERVABILITY", "CLOUD_INIT_RUNCMD_NOT_OBSERVED"}:
        payload.update(
            {
                "status": "BOOT_PATH_GUEST_OBSERVABILITY_MISSING",
                "recommended_boot_mode": "linux_direct_with_initrd_trace_or_verified_cloud_init",
                "reason": "guest boot was not observable enough to prove agent installation",
                "recommended_next_steps": list(diagnostic.get("recommended_next_steps") or []),
                "retry_same_path": False,
            }
        )
    elif diagnostic_status == "CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE":
        payload.update(
            {
                "status": "BOOT_PATH_CLOUD_INIT_UNAVAILABLE",
                "recommended_boot_mode": "linux_direct_with_verified_initrd_bundle_or_preinstalled_agent",
                "reason": "the attached NoCloud seed cannot install the guest agent because the base image lacks cloud-init",
                "recommended_next_steps": [
                    "build a linux_direct managed image with build-linux-base-image and a verified Con OS initrd bundle",
                    "or use a Con OS base image with the guest agent preinstalled",
                    "retry cloud-init only with a base image whose cloud_init_guest_capability is AVAILABLE",
                ],
                "retry_same_path": False,
            }
        )
    return payload


def _refresh_apple_runner_runtime_exit(
    *,
    state_root: str,
    instance_id: str,
    runtime_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    if not runtime_manifest:
        return runtime_manifest
    refreshed = dict(runtime_manifest)
    process_alive = _process_alive(refreshed.get("process_pid"))
    refreshed["process_alive"] = process_alive
    status = str(refreshed.get("status") or "")
    if (
        str(refreshed.get("launcher_kind") or "") == "apple_virtualization_runner"
        and status == "STOPPED"
        and process_alive
        and bool(refreshed.get("virtual_machine_started", False))
    ):
        console_log_raw = str(refreshed.get("guest_console_log_path") or "")
        shared_dir_raw = str(refreshed.get("guest_shared_dir_path") or "")
        shared_dir = Path(shared_dir_raw) if shared_dir_raw else None
        shared_marker = shared_dir / "cloud-init-runcmd.txt" if shared_dir is not None else None
        cloud_init_marker_names = [path.name for path in _cloud_init_marker_paths(shared_dir)]
        initramfs_marker_names = (
            [path.name for path in sorted(shared_dir.glob("conos-initramfs-*.txt"))]
            if shared_dir is not None and shared_dir.exists()
            else []
        )
        refreshed.update(
            {
                "status": "STARTED",
                "lifecycle_state": "started",
                "reason": "apple virtualization runner process is alive; restored stale stopped runtime status",
                "stale_stop_status_repaired_at": _now_iso(),
                "guest_console_tail": _tail_file(Path(console_log_raw)) if console_log_raw else "",
            }
        )
        refreshed.pop("runner_exit_observed_at", None)
        refreshed["last_guest_boot_diagnostic"] = _managed_vm_guest_boot_diagnostic(
            runtime_manifest=refreshed,
            process_alive=True,
            guest_console_tail=str(refreshed.get("guest_console_tail") or ""),
            guest_shared_runcmd_marker_present=bool(shared_marker and shared_marker.exists()),
            guest_cloud_init_markers=cloud_init_marker_names,
            guest_initramfs_trace_markers=initramfs_marker_names,
        )
        _write_json(managed_vm_runtime_manifest_path(state_root, instance_id), refreshed)
        _update_instance_runtime_fields(
            state_root=state_root,
            instance_id=instance_id,
            runtime_payload=refreshed,
        )
        return refreshed
    if (
        str(refreshed.get("launcher_kind") or "") == "apple_virtualization_runner"
        and status in {"STARTING", "STARTED"}
        and not process_alive
    ):
        console_log_raw = str(refreshed.get("guest_console_log_path") or "")
        shared_dir_raw = str(refreshed.get("guest_shared_dir_path") or "")
        shared_dir = Path(shared_dir_raw) if shared_dir_raw else None
        shared_marker = shared_dir / "cloud-init-runcmd.txt" if shared_dir is not None else None
        cloud_init_marker_names = [path.name for path in _cloud_init_marker_paths(shared_dir)]
        initramfs_marker_names = (
            [path.name for path in sorted(shared_dir.glob("conos-initramfs-*.txt"))]
            if shared_dir is not None and shared_dir.exists()
            else []
        )
        refreshed.update(
            {
                "status": "STOPPED",
                "lifecycle_state": "stopped",
                "reason": str(
                    refreshed.get("reason")
                    or "apple virtualization runner process exited before guest agent readiness"
                ),
                "runner_exit_observed_at": _now_iso(),
                "guest_console_tail": _tail_file(Path(console_log_raw)) if console_log_raw else "",
            }
        )
        refreshed["last_guest_boot_diagnostic"] = _managed_vm_guest_boot_diagnostic(
            runtime_manifest=refreshed,
            process_alive=False,
            guest_console_tail=str(refreshed.get("guest_console_tail") or ""),
            guest_shared_runcmd_marker_present=bool(shared_marker and shared_marker.exists()),
            guest_cloud_init_markers=cloud_init_marker_names,
            guest_initramfs_trace_markers=initramfs_marker_names,
        )
        _write_json(managed_vm_runtime_manifest_path(state_root, instance_id), refreshed)
        _update_instance_runtime_fields(
            state_root=state_root,
            instance_id=instance_id,
            runtime_payload=refreshed,
        )
    return refreshed


def managed_vm_guest_agent_gate(
    *,
    state_root: str = "",
    image_id: str = "",
    instance_id: str = "",
) -> Dict[str, Any]:
    config = managed_vm_config(state_root=state_root, image_id=image_id, instance_id=instance_id)
    runtime_manifest = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    runtime_manifest = _refresh_apple_runner_runtime_exit(
        state_root=config.state_root,
        instance_id=config.instance_id,
        runtime_manifest=runtime_manifest,
    )
    process_alive = _process_alive(runtime_manifest.get("process_pid"))
    console_log_raw = str(runtime_manifest.get("guest_console_log_path") or "")
    shared_dir_raw = str(runtime_manifest.get("guest_shared_dir_path") or "")
    shared_dir = Path(shared_dir_raw) if shared_dir_raw else None
    shared_marker = shared_dir / "cloud-init-runcmd.txt" if shared_dir is not None else None
    cloud_init_marker_paths = _cloud_init_marker_paths(shared_dir)
    cloud_init_marker_names = [path.name for path in cloud_init_marker_paths]
    initramfs_marker_paths = (
        sorted(shared_dir.glob("conos-initramfs-*.txt")) if shared_dir is not None and shared_dir.exists() else []
    )
    initramfs_marker_names = [path.name for path in initramfs_marker_paths]
    console_tail = _tail_file(Path(console_log_raw)) if console_log_raw else ""
    shared_marker_present = bool(shared_marker and shared_marker.exists())
    shared_marker_tail = _tail_file(shared_marker) if shared_marker else ""
    diagnostic = _managed_vm_guest_boot_diagnostic(
        runtime_manifest=runtime_manifest,
        process_alive=process_alive,
        guest_console_tail=console_tail,
        guest_shared_runcmd_marker_present=shared_marker_present,
        guest_cloud_init_markers=cloud_init_marker_names,
        guest_initramfs_trace_markers=initramfs_marker_names,
    )
    reasons: list[str] = []
    if not runtime_manifest:
        reasons.append("runtime_manifest_missing")
    if runtime_manifest and not runtime_manifest.get("virtual_machine_started"):
        reasons.append("virtual_machine_not_started")
    if runtime_manifest and not runtime_manifest.get("guest_agent_ready"):
        reasons.append("guest_agent_not_ready")
    if runtime_manifest and not runtime_manifest.get("execution_ready"):
        reasons.append("execution_not_ready")
    if runtime_manifest and not process_alive:
        reasons.append("vm_process_not_alive")
    ready = bool(runtime_manifest) and not reasons
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "guest_agent_gate",
        "status": "GUEST_AGENT_READY" if ready else "GUEST_AGENT_NOT_READY",
        "ready": ready,
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "runtime_manifest_present": bool(runtime_manifest),
        "runtime_manifest": runtime_manifest,
        "process_pid": str(runtime_manifest.get("process_pid") or ""),
        "process_alive": process_alive,
        "virtual_machine_started": bool(runtime_manifest.get("virtual_machine_started", False)),
        "guest_agent_ready": bool(runtime_manifest.get("guest_agent_ready", False)),
        "execution_ready": bool(runtime_manifest.get("execution_ready", False)),
        "guest_console_log_path": console_log_raw,
        "guest_console_tail": console_tail,
        "guest_shared_dir_path": shared_dir_raw,
        "guest_shared_dir_present": bool(shared_dir and shared_dir.exists()),
        "guest_shared_runcmd_marker_present": shared_marker_present,
        "guest_shared_runcmd_marker": shared_marker_tail,
        "guest_cloud_init_marker_present": bool(cloud_init_marker_names),
        "guest_cloud_init_markers": cloud_init_marker_names,
        "guest_cloud_init_marker_tails": {path.name: _tail_file(path) for path in cloud_init_marker_paths},
        "guest_initramfs_trace_marker_present": bool(initramfs_marker_names),
        "guest_initramfs_trace_markers": initramfs_marker_names,
        "guest_initramfs_trace_marker_tails": {path.name: _tail_file(path) for path in initramfs_marker_paths},
        "guest_boot_diagnostic": diagnostic,
        "blocked_reasons": reasons,
        "reason": ", ".join(reasons),
        "no_host_fallback": True,
    }


def register_managed_vm_base_image(
    *,
    source_disk_path: str,
    state_root: str = "",
    image_id: str = "",
) -> Dict[str, Any]:
    """Copy a caller-provided boot disk into Con OS managed VM state."""

    source = Path(source_disk_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"managed VM base image source does not exist: {source}")
    image = managed_vm_image_id(image_id)
    image_root = managed_vm_image_root(state_root, image)
    disk_path = Path(managed_vm_base_image_path(state_root, image))
    manifest_path = managed_vm_image_manifest_path(state_root, image)
    image_root.mkdir(parents=True, exist_ok=True)
    disk_copy: Dict[str, Any] = {"method": "already_in_place", "status": "SKIPPED"}
    if source != disk_path:
        disk_copy = _copy_file_efficient(source, disk_path)
    digest = _sha256_file(disk_path)
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_base_image",
        "status": "REGISTERED",
        "image_id": image,
        "state_root": managed_vm_state_root(state_root),
        "source_disk_path": str(source),
        "disk_path": str(disk_path),
        "sha256": digest,
        "byte_size": disk_path.stat().st_size,
        "disk_copy": disk_copy,
        "registered_at": _now_iso(),
        "owned_by_conos": True,
        "boot_mode": "efi_disk",
        "bootable": False,
        "boot_verified": False,
        "guest_agent_verified": False,
        "guest_agent_transport": "",
        "guest_agent_port": None,
        "no_host_fallback": True,
        "next_required_step": "prepare-instance then boot via managed helper",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def register_managed_vm_cloud_init_image(
    *,
    source_disk_path: str,
    state_root: str = "",
    image_id: str = "",
    guest_agent_path: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
    guest_python_path: str = "/usr/bin/python3",
) -> Dict[str, Any]:
    """Register an EFI disk image that will receive a Con OS NoCloud seed at start."""

    source = Path(source_disk_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"managed VM cloud-init image source does not exist: {source}")
    agent = Path(_clean(guest_agent_path) or managed_vm_guest_agent_source_path()).expanduser().resolve()
    if not agent.exists() or not agent.is_file():
        raise FileNotFoundError(f"managed VM guest agent source does not exist: {agent}")
    image = managed_vm_image_id(image_id)
    image_root = managed_vm_image_root(state_root, image)
    disk_path = Path(managed_vm_base_image_path(state_root, image))
    manifest_path = managed_vm_image_manifest_path(state_root, image)
    image_root.mkdir(parents=True, exist_ok=True)
    disk_copy: Dict[str, Any] = {"method": "already_in_place", "status": "SKIPPED"}
    if source != disk_path:
        disk_copy = _copy_file_efficient(source, disk_path)
    cloud_init_capability = _managed_vm_cloud_init_guest_capability(disk_path)
    cloud_init_available = bool(cloud_init_capability.get("cloud_init_likely_available", False))
    efi_boot_artifacts = _scan_linux_boot_artifact_paths(disk_path)
    linux_direct_fallback_report: Dict[str, Any] = {"status": "NOT_REQUIRED"}
    if not cloud_init_available and efi_boot_artifacts:
        try:
            partitions = _gpt_partitions(disk_path)
            root_partition = _managed_vm_root_ext_partition(disk_path, partitions)
            if not root_partition:
                raise ValueError("Linux root ext4 partition was not found")
            extracted_kernel = image_root / "extracted-vmlinuz"
            extracted_initrd = image_root / "extracted-initrd.img"
            extracted_kernel.write_bytes(
                _extract_ext4_file_from_image(
                    disk_path,
                    partition_offset=int(root_partition["byte_offset"]),
                    file_path=str(efi_boot_artifacts.get("kernel_path") or ""),
                )
            )
            extracted_initrd.write_bytes(
                _extract_ext4_file_from_image(
                    disk_path,
                    partition_offset=int(root_partition["byte_offset"]),
                    file_path=str(efi_boot_artifacts.get("initrd_path") or ""),
                )
            )
            linux_direct_fallback_report = build_managed_vm_linux_base_image(
                state_root=state_root,
                image_id=image,
                source_disk_path=str(disk_path),
                kernel_path=str(extracted_kernel),
                base_initrd_path=str(extracted_initrd),
                guest_agent_path=str(agent),
                guest_agent_port=int(guest_agent_port),
                guest_python_path=_clean(guest_python_path) or "/usr/bin/python3",
                include_init_wrapper=True,
                overwrite=True,
            )
            if str(linux_direct_fallback_report.get("status") or "") == "BUILT":
                registered = load_managed_vm_image_manifest(state_root=state_root, image_id=image)
                registered.update(
                    {
                        "operation": "register_cloud_init_image",
                        "efi_linux_direct_fallback": linux_direct_fallback_report,
                        "cloud_init_guest_capability": cloud_init_capability,
                        "cloud_init_capability_warning": "cloud-init unavailable; registered linux_direct guest-agent initrd fallback",
                        "source_recipe_boot_mode": "efi_disk",
                        "source_recipe_guest_agent_installation_mode": "cloud_init_nocloud_seed",
                    }
                )
                _write_json(manifest_path, registered)
                return registered
        except Exception as exc:
            linux_direct_fallback_report = {
                "status": "FAILED",
                "reason": str(exc),
                "boot_artifacts": efi_boot_artifacts,
            }
    efi_agent_initrd_injection_enabled = (not cloud_init_available) and bool(efi_boot_artifacts)
    guest_agent_autostart_configured = cloud_init_available or efi_agent_initrd_injection_enabled
    if cloud_init_available:
        installation_mode = "cloud_init_nocloud_seed"
        installation_status = "CLOUD_INIT_READY_FOR_NOCLOUD_SEED"
        verified_execution_path = ""
        next_required_step = "start-instance will attach a Con OS NoCloud seed and wait for guest agent readiness"
    elif efi_agent_initrd_injection_enabled:
        installation_mode = "efi_initrd_guest_agent_bundle"
        installation_status = "EFI_INITRD_AGENT_INJECTION_CONFIGURED"
        verified_execution_path = "efi_disk_observable_boot_agent_initrd"
        next_required_step = "start-instance will patch EFI boot config with a Con OS guest-agent initrd and wait for readiness"
    else:
        installation_mode = "cloud_init_nocloud_seed"
        installation_status = "BLOCKED_CLOUD_INIT_UNAVAILABLE_IN_GUEST_IMAGE"
        verified_execution_path = ""
        next_required_step = "use linux_direct with a verified Con OS initrd bundle, or preinstall the Con OS guest agent in the base image"
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_base_image",
        "status": "REGISTERED",
        "image_id": image,
        "state_root": managed_vm_state_root(state_root),
        "source_disk_path": str(source),
        "disk_path": str(disk_path),
        "sha256": _sha256_file(disk_path),
        "byte_size": disk_path.stat().st_size,
        "disk_copy": disk_copy,
        "registered_at": _now_iso(),
        "owned_by_conos": True,
        "boot_mode": "efi_disk",
        "bootable": True,
        "boot_verified": False,
        "guest_agent_transport": "virtio-vsock",
        "guest_agent_port": int(guest_agent_port),
        "guest_agent_verified": False,
        "guest_agent_autostart_planned": True,
        "guest_agent_autostart_configured": guest_agent_autostart_configured,
        "guest_agent_installation_mode": installation_mode,
        "guest_agent_installation_status": installation_status,
        "guest_agent_path": str(agent),
        "guest_agent_sha256": _sha256_file(agent),
        "guest_python_path": _clean(guest_python_path) or "/usr/bin/python3",
        "cloud_init_seed_enabled": True,
        "cloud_init_seed_format": "nocloud_vfat_cidata",
        "cloud_init_seed_read_only": True,
        "cloud_init_guest_capability": cloud_init_capability,
        "efi_linux_direct_fallback": linux_direct_fallback_report,
        "efi_agent_initrd_injection_enabled": efi_agent_initrd_injection_enabled,
        "efi_agent_initrd_boot_artifacts": efi_boot_artifacts,
        "cloud_init_capability_warning": (
            ""
            if bool(cloud_init_capability.get("cloud_init_likely_available", False))
            else "registered image did not expose cloud-init service markers during raw disk preflight"
        ),
        "verified_execution_path": verified_execution_path,
        "execution_ready": False,
        "no_host_fallback": True,
        "next_required_step": next_required_step,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def register_managed_vm_linux_boot_image(
    *,
    source_disk_path: str,
    kernel_path: str,
    initrd_path: str = "",
    kernel_command_line: str = "",
    state_root: str = "",
    image_id: str = "",
    guest_agent_port: int = DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT,
) -> Dict[str, Any]:
    """Register a Linux boot image with explicit kernel/initrd boot artifacts."""

    source = Path(source_disk_path).expanduser().resolve()
    kernel_source = Path(kernel_path).expanduser().resolve()
    initrd_source = Path(initrd_path).expanduser().resolve() if _clean(initrd_path) else None
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"managed VM base image source does not exist: {source}")
    if not kernel_source.exists() or not kernel_source.is_file():
        raise FileNotFoundError(f"managed VM kernel source does not exist: {kernel_source}")
    if initrd_source is not None and (not initrd_source.exists() or not initrd_source.is_file()):
        raise FileNotFoundError(f"managed VM initrd source does not exist: {initrd_source}")
    image = managed_vm_image_id(image_id)
    image_root = managed_vm_image_root(state_root, image)
    disk_path = Path(managed_vm_base_image_path(state_root, image))
    kernel_dest = managed_vm_kernel_path(state_root, image)
    initrd_dest = managed_vm_initrd_path(state_root, image)
    manifest_path = managed_vm_image_manifest_path(state_root, image)
    image_root.mkdir(parents=True, exist_ok=True)
    disk_copy: Dict[str, Any] = {"method": "already_in_place", "status": "SKIPPED"}
    kernel_copy: Dict[str, Any] = {"method": "already_in_place", "status": "SKIPPED"}
    initrd_copy: Dict[str, Any] = {"method": "not_present", "status": "SKIPPED"}
    if source != disk_path:
        disk_copy = _copy_file_efficient(source, disk_path)
    if kernel_source != kernel_dest:
        kernel_copy = _copy_file_efficient(kernel_source, kernel_dest)
    initrd_payload: Dict[str, Any] = {
        "initrd_path": "",
        "initrd_sha256": "",
        "initrd_byte_size": 0,
        "initrd_present": False,
        "guest_initrd_bundle_manifest_path": "",
        "guest_initrd_bundle_manifest_present": False,
        "guest_initrd_bundle_status": "",
        "guest_initrd_bundle_capability": {"status": "NOT_PRESENT", "verified": False},
        "guest_agent_autostart_configured": False,
        "guest_agent_bundle_files": [],
        "verified_execution_path": "",
        "guest_agent_installation_status": "NO_INITRD_BUNDLE",
    }
    if initrd_source is not None:
        if initrd_source != initrd_dest:
            initrd_copy = _copy_file_efficient(initrd_source, initrd_dest)
        else:
            initrd_copy = {"method": "already_in_place", "status": "SKIPPED"}
        sidecar = _read_sidecar_manifest(initrd_source)
        sidecar_dest = Path(f"{initrd_dest}.manifest.json")
        if sidecar:
            _write_json(sidecar_dest, sidecar)
        bundle_capability = (
            _managed_vm_guest_initrd_bundle_capability(sidecar, expected_port=int(guest_agent_port))
            if sidecar
            else {"status": "NOT_PRESENT", "verified": False}
        )
        bundle_verified = bool(bundle_capability.get("verified", False))
        initrd_payload = {
            "initrd_path": str(initrd_dest),
            "initrd_sha256": _sha256_file(initrd_dest),
            "initrd_byte_size": initrd_dest.stat().st_size,
            "initrd_present": True,
            "guest_initrd_bundle_manifest_path": str(sidecar_dest if sidecar else ""),
            "guest_initrd_bundle_manifest_present": bool(sidecar),
            "guest_initrd_bundle_status": str(sidecar.get("status") or ""),
            "guest_initrd_bundle_capability": bundle_capability,
            "guest_agent_autostart_configured": bundle_verified,
            "guest_agent_bundle_files": list(sidecar.get("files") or []) if isinstance(sidecar.get("files"), list) else [],
            "verified_execution_path": (
                "linux_direct_initrd_guest_agent_bundle"
                if bundle_verified
                else ""
            ),
            "guest_agent_installation_status": (
                "INITRD_AUTOSTART_BUNDLE_CONFIGURED"
                if bundle_verified
                else "INITRD_BUNDLE_UNVERIFIED"
            ),
        }
    detected_root = _detect_linux_root_partition_from_disk(source)
    selected_root_device = str(detected_root.get("root_device") or "") or "/dev/vda"
    selected_root_boot_spec = str(detected_root.get("root_boot_spec") or "") or selected_root_device
    command_line = _clean(kernel_command_line) or (
        f"console=hvc0 root={selected_root_boot_spec} rw rootwait "
        f"conos.agent=vsock:{int(guest_agent_port)} conos.root={selected_root_device}"
    )
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_base_image",
        "status": "REGISTERED",
        "image_id": image,
        "state_root": managed_vm_state_root(state_root),
        "source_disk_path": str(source),
        "disk_path": str(disk_path),
        "sha256": _sha256_file(disk_path),
        "byte_size": disk_path.stat().st_size,
        "kernel_source_path": str(kernel_source),
        "kernel_path": str(kernel_dest),
        "kernel_sha256": _sha256_file(kernel_dest),
        "kernel_byte_size": kernel_dest.stat().st_size,
        "disk_copy": disk_copy,
        "kernel_copy": kernel_copy,
        "initrd_copy": initrd_copy,
        "kernel_command_line": command_line,
        "detected_root_device": str(detected_root.get("root_device") or ""),
        "detected_root_partition_uuid": str(detected_root.get("root_partition_uuid") or ""),
        "detected_root_filesystem_uuid": str(detected_root.get("root_filesystem_uuid") or ""),
        "root_boot_spec": selected_root_boot_spec,
        "root_device": selected_root_device,
        **initrd_payload,
        "registered_at": _now_iso(),
        "owned_by_conos": True,
        "boot_mode": "linux_direct",
        "bootable": True,
        "boot_verified": False,
        "guest_agent_transport": "virtio-vsock",
        "guest_agent_port": int(guest_agent_port),
        "guest_agent_verified": False,
        "guest_agent_installation_mode": (
            "initrd_autostart_bundle"
            if bool(initrd_payload.get("guest_agent_autostart_configured"))
            else "external_or_preinstalled"
        ),
        "guest_agent_installation_status": str(initrd_payload.get("guest_agent_installation_status") or ""),
        "verified_execution_path": str(initrd_payload.get("verified_execution_path") or ""),
        "execution_ready": False,
        "no_host_fallback": True,
        "next_required_step": "start-instance then wait for guest agent readiness",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def create_managed_vm_blank_image(
    *,
    state_root: str = "",
    image_id: str = "",
    size_mb: int = DEFAULT_MANAGED_VM_BLANK_IMAGE_SIZE_MB,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a Con OS-owned sparse disk artifact.

    The resulting disk is intentionally not marked bootable. It is a managed
    artifact for installer/restore flows, not a guest execution-ready image.
    """

    image = managed_vm_image_id(image_id)
    size = int(size_mb)
    if size <= 0:
        raise ValueError("managed VM blank image size must be positive")
    image_root = managed_vm_image_root(state_root, image)
    disk_path = Path(managed_vm_base_image_path(state_root, image))
    manifest_path = managed_vm_image_manifest_path(state_root, image)
    if disk_path.exists() and not overwrite:
        raise FileExistsError(f"managed VM image already exists: {disk_path}")
    image_root.mkdir(parents=True, exist_ok=True)
    with disk_path.open("wb") as handle:
        handle.truncate(size * 1024 * 1024)
    digest = _sha256_file(disk_path)
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_base_image",
        "status": "BLANK_CREATED",
        "image_id": image,
        "state_root": managed_vm_state_root(state_root),
        "disk_path": str(disk_path),
        "sha256": digest,
        "byte_size": disk_path.stat().st_size,
        "size_mb": size,
        "build_method": "blank_sparse_disk",
        "created_at": _now_iso(),
        "created_by_conos": True,
        "owned_by_conos": True,
        "boot_mode": "efi_disk",
        "bootable": False,
        "boot_verified": False,
        "guest_agent_transport": "",
        "guest_agent_port": None,
        "guest_agent_verified": False,
        "execution_ready": False,
        "no_host_fallback": True,
        "next_required_step": "install guest OS and guest agent before exec",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def prepare_managed_vm_instance(
    *,
    state_root: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
) -> Dict[str, Any]:
    image = managed_vm_image_id(image_id)
    instance = managed_vm_instance_id(instance_id)
    image_manifest = load_managed_vm_image_manifest(state_root, image)
    base_disk = Path(managed_vm_base_image_path(state_root, image))
    if not image_manifest or not base_disk.exists():
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "artifact_type": "managed_vm_instance",
            "status": "UNAVAILABLE",
            "reason": "managed VM base image is not registered",
            "image_id": image,
            "instance_id": instance,
            "base_image_path": str(base_disk),
            "base_image_present": base_disk.exists(),
        }
    root = managed_vm_instance_root(state_root, instance)
    overlay = managed_vm_overlay_path(state_root, instance)
    for child in ("logs", "workspace", "snapshots"):
        (root / child).mkdir(parents=True, exist_ok=True)
    manifest_path = managed_vm_instance_manifest_path(state_root, instance)
    payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_instance",
        "status": "PREPARED",
        "state_root": managed_vm_state_root(state_root),
        "image_id": image,
        "instance_id": instance,
        "network_mode": _clean(network_mode) or "provider_default",
        "base_image_path": str(base_disk),
        "base_image_sha256": str(image_manifest.get("sha256", "")),
        "base_image_status": str(image_manifest.get("status", "")),
        "boot_mode": str(image_manifest.get("boot_mode") or "efi_disk"),
        "base_image_boot_verified": bool(image_manifest.get("boot_verified", False)),
        "base_image_guest_agent_verified": bool(image_manifest.get("guest_agent_verified", False)),
        "guest_agent_transport": str(image_manifest.get("guest_agent_transport") or ""),
        "guest_agent_port": image_manifest.get("guest_agent_port"),
        "guest_agent_installation_mode": str(image_manifest.get("guest_agent_installation_mode") or ""),
        "guest_agent_installation_status": str(image_manifest.get("guest_agent_installation_status") or ""),
        "guest_agent_autostart_configured": bool(image_manifest.get("guest_agent_autostart_configured", False)),
        "guest_agent_autostart_planned": bool(image_manifest.get("guest_agent_autostart_planned", False)),
        "verified_execution_path": str(image_manifest.get("verified_execution_path") or ""),
        "cloud_init_seed_enabled": bool(image_manifest.get("cloud_init_seed_enabled", False)),
        "cloud_init_seed_format": str(image_manifest.get("cloud_init_seed_format") or ""),
        "cloud_init_seed_path": "",
        "cloud_init_seed_present": False,
        "cloud_init_guest_capability": (
            image_manifest.get("cloud_init_guest_capability")
            if isinstance(image_manifest.get("cloud_init_guest_capability"), dict)
            else {}
        ),
        "kernel_path": str(image_manifest.get("kernel_path") or ""),
        "initrd_path": str(image_manifest.get("initrd_path") or ""),
        "overlay_path": str(overlay),
        "overlay_present": overlay.exists(),
        "prepared_at": _now_iso(),
        "booted": False,
        "execution_ready": False,
        "guest_agent_ready": False,
        "no_host_fallback": True,
        "next_required_step": "helper boot/exec must create overlay and start guest agent",
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def boot_managed_vm_instance(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
) -> Dict[str, Any]:
    """Ask the bundled helper to prepare the managed VM boot boundary."""

    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    image_manifest = load_managed_vm_image_manifest(config.state_root, config.image_id)
    instance_manifest = load_managed_vm_instance_manifest(config.state_root, config.instance_id)
    base_disk = Path(managed_vm_base_image_path(config.state_root, config.image_id))
    manifest_path = managed_vm_instance_manifest_path(config.state_root, config.instance_id)
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "boot_instance",
        "status": "NOT_RUN",
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "base_image_path": str(base_disk),
        "base_image_present": base_disk.exists(),
        "image_manifest_present": bool(image_manifest),
        "instance_manifest_present": bool(instance_manifest),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "no_host_fallback": True,
    }
    if not config.helper_path:
        payload.update({"status": "UNAVAILABLE", "reason": "managed VM helper was not found"})
        return payload
    if not image_manifest or not base_disk.exists():
        payload.update({"status": "UNAVAILABLE", "reason": "managed VM base image is not registered"})
        return payload
    if not instance_manifest:
        instance_manifest = prepare_managed_vm_instance(
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=str(payload["network_mode"]),
        )
        payload["instance_manifest_present"] = bool(instance_manifest)
        payload["instance_prepared_by_boot"] = True
    command = [
        config.helper_path,
        "boot",
        "--state-root",
        config.state_root,
        "--instance-id",
        config.instance_id,
        "--image-id",
        config.image_id,
        "--network-mode",
        str(payload["network_mode"]),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    helper_payload = _read_json_from_text(str(completed.stdout or ""))
    payload.update(
        {
            "status": str(helper_payload.get("status") or ("BOOT_FAILED" if int(completed.returncode) else "BOOT_CONTRACT_READY")),
            "reason": str(helper_payload.get("reason") or ""),
            "helper_payload": helper_payload,
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "overlay_path": str(helper_payload.get("overlay_path") or managed_vm_overlay_path(config.state_root, config.instance_id)),
            "overlay_present": bool(helper_payload.get("overlay_present", managed_vm_overlay_path(config.state_root, config.instance_id).exists())),
            "virtual_machine_started": bool(helper_payload.get("virtual_machine_started", False)),
            "guest_agent_ready": bool(helper_payload.get("guest_agent_ready", False)),
            "execution_ready": bool(helper_payload.get("execution_ready", False)),
            "boot_attempted_at": _now_iso(),
        }
    )
    updated_manifest = dict(instance_manifest)
    updated_manifest.update(
        {
            "status": payload["status"],
            "boot_contract_status": payload["status"],
            "boot_reason": payload["reason"],
            "booted": bool(payload["virtual_machine_started"]),
            "virtual_machine_started": bool(payload["virtual_machine_started"]),
            "execution_ready": bool(payload["execution_ready"]),
            "guest_agent_ready": bool(payload["guest_agent_ready"]),
            "overlay_path": str(payload["overlay_path"]),
            "overlay_present": bool(payload["overlay_present"]),
            "last_boot_attempt_at": payload["boot_attempted_at"],
            "last_boot_returncode": int(completed.returncode),
            "helper_payload": helper_payload,
            "no_host_fallback": True,
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(updated_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["instance_manifest"] = updated_manifest
    return payload


def _start_managed_vm_instance_with_runner(
    *,
    config: ManagedVMConfig,
    runner: str,
    image_manifest: Dict[str, Any],
    base_disk: Path,
    network_mode: str,
    timeout_seconds: int,
    startup_wait_seconds: float,
) -> Dict[str, Any]:
    started_monotonic = time.monotonic()
    stage_timing: list[Dict[str, Any]] = []
    if not load_managed_vm_instance_manifest(config.state_root, config.instance_id):
        stage_started = time.monotonic()
        prepare_managed_vm_instance(
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
        )
        stage_timing.append(
            {
                "stage": "prepare_instance",
                "duration_seconds": max(0.0, time.monotonic() - stage_started),
                "status": "PREPARED",
            }
        )
    else:
        stage_timing.append({"stage": "prepare_instance", "duration_seconds": 0.0, "status": "SKIPPED_ALREADY_PRESENT"})

    instance_root = managed_vm_instance_root(config.state_root, config.instance_id)
    logs_root = instance_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    runtime_path = managed_vm_runtime_manifest_path(config.state_root, config.instance_id)
    overlay = managed_vm_overlay_path(config.state_root, config.instance_id)
    writable_disk = managed_vm_writable_disk_path(config.state_root, config.instance_id)
    efi_variable_store = managed_vm_efi_variable_store_path(config.state_root, config.instance_id)
    writable_disk_created = False
    writable_disk_copy: Dict[str, Any] = {"method": "already_present", "status": "SKIPPED"}
    if not writable_disk.exists():
        stage_started = time.monotonic()
        writable_disk_copy = _copy_file_efficient(base_disk, writable_disk)
        writable_disk_created = True
        stage_timing.append(_stage_timing_entry("prepare_writable_disk", stage_started, writable_disk_copy))
    else:
        stage_timing.append({"stage": "prepare_writable_disk", "duration_seconds": 0.0, "status": "SKIPPED_ALREADY_PRESENT"})
    stage_started = time.monotonic()
    cloud_init_seed = _ensure_instance_cloud_init_seed(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
        image_manifest=image_manifest,
    )
    stage_timing.append(_stage_timing_entry("ensure_cloud_init_seed", stage_started, cloud_init_seed))
    cloud_init_guest_capability = (
        image_manifest.get("cloud_init_guest_capability")
        if isinstance(image_manifest.get("cloud_init_guest_capability"), dict)
        else {}
    )
    if bool(cloud_init_seed.get("enabled")) and not cloud_init_guest_capability:
        stage_started = time.monotonic()
        cloud_init_guest_capability = _managed_vm_cloud_init_guest_capability(writable_disk)
        stage_timing.append(_stage_timing_entry("scan_cloud_init_capability", stage_started, cloud_init_guest_capability))
    else:
        stage_timing.append({"stage": "scan_cloud_init_capability", "duration_seconds": 0.0, "status": "SKIPPED_CACHED"})
    runner_image_manifest = dict(image_manifest)
    if bool(cloud_init_seed.get("enabled")):
        runner_image_manifest["cloud_init_seed_path"] = str(cloud_init_seed.get("seed_path") or "")
        runner_image_manifest["cloud_init_seed_present"] = Path(str(cloud_init_seed.get("seed_path") or "")).exists()
        runner_image_manifest["cloud_init_seed_read_only"] = True
    efi_observable_boot_patch: Dict[str, Any] = {"status": "NOT_REQUIRED"}
    if str(runner_image_manifest.get("boot_mode") or image_manifest.get("boot_mode") or "efi_disk") == "efi_disk" and bool(
        cloud_init_seed.get("enabled", False)
    ):
        stage_started = time.monotonic()
        efi_observable_boot_patch = _apply_efi_observable_boot_patch(
            writable_disk,
            guest_agent_path=str(image_manifest.get("guest_agent_path") or ""),
            guest_python_path=str(image_manifest.get("guest_python_path") or "/usr/bin/python3"),
            guest_agent_port=int(image_manifest.get("guest_agent_port") or DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT),
            inject_agent_initrd=bool(image_manifest.get("efi_agent_initrd_injection_enabled", False)),
        )
        stage_timing.append(_stage_timing_entry("apply_efi_observable_boot_patch", stage_started, efi_observable_boot_patch))
    else:
        stage_timing.append({"stage": "apply_efi_observable_boot_patch", "duration_seconds": 0.0, "status": "SKIPPED_NOT_REQUIRED"})
    stdout_path = logs_root / "vz-runner.stdout.log"
    stderr_path = logs_root / "vz-runner.stderr.log"
    console_log_path = logs_root / "guest-console.log"
    shared_dir_path = instance_root / "guest-share"
    shared_dir_path.mkdir(parents=True, exist_ok=True)
    command = _runner_start_command(
        runner_path=runner,
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
        disk_path=writable_disk,
        base_image_path=base_disk,
        efi_variable_store_path=efi_variable_store,
        runtime_manifest_path=runtime_path,
        network_mode=network_mode,
        image_manifest=runner_image_manifest,
        console_log_path=console_log_path,
        shared_dir_path=shared_dir_path,
        shared_dir_tag=MANAGED_VM_SHARED_DIR_TAG,
    )
    prelaunch_payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": "STARTING",
        "lifecycle_state": "starting",
        "reason": "apple virtualization runner launched and awaiting VM start callback",
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": network_mode,
        "base_image_path": str(base_disk),
        "base_image_sha256": str(image_manifest.get("sha256", "")),
        "base_image_boot_verified": bool(image_manifest.get("boot_verified", False)),
        "boot_mode": str(image_manifest.get("boot_mode") or "efi_disk"),
        "kernel_path": str(image_manifest.get("kernel_path") or ""),
        "initrd_path": str(image_manifest.get("initrd_path") or ""),
        "overlay_path": str(overlay),
        "overlay_present": overlay.exists(),
        "writable_disk_path": str(writable_disk),
        "writable_disk_present": writable_disk.exists(),
        "efi_variable_store_path": str(efi_variable_store),
        "efi_variable_store_present": efi_variable_store.exists(),
        "cloud_init_seed_enabled": bool(cloud_init_seed.get("enabled", False)),
        "cloud_init_seed_path": str(cloud_init_seed.get("seed_path") or ""),
        "cloud_init_seed_present": Path(str(cloud_init_seed.get("seed_path") or "")).exists()
        if cloud_init_seed.get("seed_path")
        else False,
        "cloud_init_seed_read_only": bool(cloud_init_seed.get("enabled", False)),
        "cloud_init_seed_report": cloud_init_seed,
        "cloud_init_guest_capability": cloud_init_guest_capability,
        "efi_observable_boot_patch": efi_observable_boot_patch,
        "writable_disk_created": writable_disk_created,
        "writable_disk_copy": writable_disk_copy,
        "efi_variable_store_path": str(efi_variable_store),
        "efi_variable_store_present": efi_variable_store.exists(),
        "process_pid": "",
        "process_alive": False,
        "virtual_machine_started": False,
        "guest_agent_ready": False,
        "execution_ready": False,
        "guest_agent_transport": str(image_manifest.get("guest_agent_transport") or ""),
        "guest_agent_port": image_manifest.get("guest_agent_port"),
        "launcher_kind": "apple_virtualization_runner",
        "runner_path": str(runner),
        "runner_stdout_path": str(stdout_path),
        "runner_stderr_path": str(stderr_path),
        "guest_console_log_path": str(console_log_path),
        "guest_console_tail": _tail_file(console_log_path),
        "guest_shared_dir_path": str(shared_dir_path),
        "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
        "guest_shared_dir_present": shared_dir_path.exists(),
        "started_at": _now_iso(),
        "last_runner_command": command,
        "last_runner_returncode": None,
        "no_host_fallback": True,
    }
    _write_json(runtime_path, prelaunch_payload)
    stage_started = time.monotonic()
    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
    stage_timing.append(
        {
            "stage": "launch_runner_process",
            "duration_seconds": max(0.0, time.monotonic() - stage_started),
            "status": "LAUNCHED",
            "process_pid": str(process.pid),
        }
    )

    current_runtime = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    if str(current_runtime.get("status") or "") != "STARTED":
        current_runtime.update({"process_pid": str(process.pid), "process_alive": _process_alive(process.pid)})
        _write_json(runtime_path, current_runtime)

    deadline = time.monotonic() + max(0.0, float(startup_wait_seconds))
    final_runtime = current_runtime
    stage_started = time.monotonic()
    while time.monotonic() <= deadline:
        final_runtime = load_managed_vm_runtime_manifest(config.state_root, config.instance_id) or current_runtime
        runner_pid = str(final_runtime.get("process_pid") or process.pid)
        started = str(final_runtime.get("status") or "") == "STARTED" and bool(final_runtime.get("virtual_machine_started"))
        ready = bool(final_runtime.get("guest_agent_ready", False)) and bool(final_runtime.get("execution_ready", False))
        if started and ready and _process_alive(runner_pid):
            break
        returncode = process.poll()
        if returncode is not None:
            final_runtime.update(
                {
                    "status": "START_FAILED",
                    "lifecycle_state": "failed",
                    "reason": str(final_runtime.get("reason") or "apple virtualization runner exited before STARTED"),
                    "process_pid": str(process.pid),
                    "process_alive": False,
                    "virtual_machine_started": False,
                    "execution_ready": False,
                    "guest_agent_ready": False,
                    "last_runner_returncode": int(returncode),
                    "runner_stdout_tail": _tail_file(stdout_path),
                    "runner_stderr_tail": _tail_file(stderr_path),
                    "guest_console_log_path": str(console_log_path),
                    "guest_console_tail": _tail_file(console_log_path),
                    "guest_shared_dir_path": str(shared_dir_path),
                    "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
                    "guest_shared_dir_present": shared_dir_path.exists(),
                }
            )
            _write_json(runtime_path, final_runtime)
            break
        time.sleep(0.05)
    stage_timing.append(
        _stage_timing_entry(
            "wait_runner_ready",
            stage_started,
            final_runtime,
            extra={
                "process_pid": str(final_runtime.get("process_pid") or process.pid),
                "process_alive": _process_alive(final_runtime.get("process_pid") or process.pid),
                "virtual_machine_started": bool(final_runtime.get("virtual_machine_started", False)),
                "guest_agent_ready": bool(final_runtime.get("guest_agent_ready", False)),
                "execution_ready": bool(final_runtime.get("execution_ready", False)),
            },
        )
    )

    stage_started = time.monotonic()
    final_runtime = load_managed_vm_runtime_manifest(config.state_root, config.instance_id) or current_runtime
    runner_pid = str(final_runtime.get("process_pid") or process.pid)
    process_alive = _process_alive(runner_pid)
    if str(final_runtime.get("status") or "") == "STARTED" and bool(final_runtime.get("virtual_machine_started")) and process_alive:
        final_runtime.update(
            {
                "status": "STARTED",
                "lifecycle_state": "started",
                "process_pid": runner_pid,
                "process_alive": True,
                "launcher_kind": "apple_virtualization_runner",
                "runner_path": str(runner),
                "runner_stdout_path": str(stdout_path),
                "runner_stderr_path": str(stderr_path),
                "guest_console_log_path": str(console_log_path),
                "guest_console_tail": _tail_file(console_log_path),
                "guest_shared_dir_path": str(shared_dir_path),
                "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
                "guest_shared_dir_present": shared_dir_path.exists(),
                "no_host_fallback": True,
            }
        )
    elif process.poll() is None:
        final_runtime.update(
            {
                "status": "STARTING",
                "lifecycle_state": "starting",
                "reason": str(final_runtime.get("reason") or "runner alive but VM STARTED marker was not observed before timeout"),
                "process_pid": str(process.pid),
                "process_alive": True,
                "virtual_machine_started": bool(final_runtime.get("virtual_machine_started", False)),
                "boot_mode": str(image_manifest.get("boot_mode") or final_runtime.get("boot_mode") or "efi_disk"),
                "guest_agent_transport": str(image_manifest.get("guest_agent_transport") or final_runtime.get("guest_agent_transport") or ""),
                "guest_agent_port": image_manifest.get("guest_agent_port", final_runtime.get("guest_agent_port")),
                "cloud_init_seed_enabled": bool(cloud_init_seed.get("enabled", False)),
                "cloud_init_seed_path": str(cloud_init_seed.get("seed_path") or final_runtime.get("cloud_init_seed_path") or ""),
                "cloud_init_seed_present": Path(str(cloud_init_seed.get("seed_path") or "")).exists()
                if cloud_init_seed.get("seed_path")
                else bool(final_runtime.get("cloud_init_seed_present", False)),
                "cloud_init_seed_read_only": bool(cloud_init_seed.get("enabled", False)),
                "cloud_init_guest_capability": cloud_init_guest_capability,
                "efi_observable_boot_patch": efi_observable_boot_patch,
                "launcher_kind": "apple_virtualization_runner",
                "runner_path": str(runner),
                "runner_stdout_path": str(stdout_path),
                "runner_stderr_path": str(stderr_path),
                "runner_stdout_tail": _tail_file(stdout_path),
                "runner_stderr_tail": _tail_file(stderr_path),
                "guest_console_log_path": str(console_log_path),
                "guest_console_tail": _tail_file(console_log_path),
                "guest_shared_dir_path": str(shared_dir_path),
                "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
                "guest_shared_dir_present": shared_dir_path.exists(),
                "no_host_fallback": True,
            }
        )
    else:
        final_runtime.update(
            {
                "status": "START_FAILED",
                "lifecycle_state": "failed",
                "reason": str(final_runtime.get("reason") or "apple virtualization runner exited before STARTED"),
                "process_pid": str(process.pid),
                "process_alive": False,
                "virtual_machine_started": False,
                "execution_ready": False,
                "guest_agent_ready": False,
                "boot_mode": str(image_manifest.get("boot_mode") or final_runtime.get("boot_mode") or "efi_disk"),
                "guest_agent_transport": str(image_manifest.get("guest_agent_transport") or final_runtime.get("guest_agent_transport") or ""),
                "guest_agent_port": image_manifest.get("guest_agent_port", final_runtime.get("guest_agent_port")),
                "cloud_init_seed_enabled": bool(cloud_init_seed.get("enabled", False)),
                "cloud_init_seed_path": str(cloud_init_seed.get("seed_path") or final_runtime.get("cloud_init_seed_path") or ""),
                "cloud_init_seed_present": Path(str(cloud_init_seed.get("seed_path") or "")).exists()
                if cloud_init_seed.get("seed_path")
                else bool(final_runtime.get("cloud_init_seed_present", False)),
                "cloud_init_seed_read_only": bool(cloud_init_seed.get("enabled", False)),
                "cloud_init_guest_capability": cloud_init_guest_capability,
                "efi_observable_boot_patch": efi_observable_boot_patch,
                "launcher_kind": "apple_virtualization_runner",
                "runner_path": str(runner),
                "runner_stdout_path": str(stdout_path),
                "runner_stderr_path": str(stderr_path),
                "last_runner_returncode": process.poll(),
                "runner_stdout_tail": _tail_file(stdout_path),
                "runner_stderr_tail": _tail_file(stderr_path),
                "guest_console_log_path": str(console_log_path),
                "guest_console_tail": _tail_file(console_log_path),
                "guest_shared_dir_path": str(shared_dir_path),
                "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
                "guest_shared_dir_present": shared_dir_path.exists(),
                "no_host_fallback": True,
            }
        )
    final_runtime.update(
        {
            "base_image_path": str(base_disk),
            "boot_mode": str(image_manifest.get("boot_mode") or final_runtime.get("boot_mode") or "efi_disk"),
            "kernel_path": str(image_manifest.get("kernel_path") or final_runtime.get("kernel_path") or ""),
            "initrd_path": str(image_manifest.get("initrd_path") or final_runtime.get("initrd_path") or ""),
            "guest_agent_transport": str(image_manifest.get("guest_agent_transport") or final_runtime.get("guest_agent_transport") or ""),
            "guest_agent_port": image_manifest.get("guest_agent_port", final_runtime.get("guest_agent_port")),
            "overlay_path": str(overlay),
            "overlay_present": overlay.exists(),
            "writable_disk_path": str(writable_disk),
            "writable_disk_present": writable_disk.exists(),
            "efi_variable_store_path": str(efi_variable_store),
            "efi_variable_store_present": efi_variable_store.exists(),
            "cloud_init_seed_enabled": bool(cloud_init_seed.get("enabled", False)),
            "cloud_init_seed_path": str(cloud_init_seed.get("seed_path") or final_runtime.get("cloud_init_seed_path") or ""),
            "cloud_init_seed_present": Path(str(cloud_init_seed.get("seed_path") or "")).exists()
            if cloud_init_seed.get("seed_path")
            else bool(final_runtime.get("cloud_init_seed_present", False)),
            "cloud_init_seed_read_only": bool(cloud_init_seed.get("enabled", False)),
            "cloud_init_seed_report": cloud_init_seed,
            "cloud_init_guest_capability": cloud_init_guest_capability,
            "efi_observable_boot_patch": efi_observable_boot_patch,
            "runner_stdout_path": str(stdout_path),
            "runner_stderr_path": str(stderr_path),
            "guest_console_log_path": str(console_log_path),
            "guest_console_tail": _tail_file(console_log_path),
            "guest_shared_dir_path": str(shared_dir_path),
            "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
            "guest_shared_dir_present": shared_dir_path.exists(),
        }
    )
    if str(final_runtime.get("status") or "") == "START_FAILED":
        final_runtime.update(_managed_vm_start_blocker_payload(final_runtime.get("reason")))
    _write_json(runtime_path, final_runtime)
    stage_timing.append(_stage_timing_entry("write_final_runtime", stage_started, final_runtime))
    stage_started = time.monotonic()
    instance_manifest = _update_instance_runtime_fields(
        state_root=config.state_root,
        instance_id=config.instance_id,
        runtime_payload=final_runtime,
    )
    stage_timing.append(_stage_timing_entry("sync_instance_manifest", stage_started, instance_manifest))
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "start_instance",
        "status": str(final_runtime.get("status") or "START_FAILED"),
        "lifecycle_state": str(final_runtime.get("lifecycle_state") or ""),
        "reason": str(final_runtime.get("reason") or ""),
        "blocker_type": str(final_runtime.get("blocker_type") or ""),
        "host_virtualization_capability": final_runtime.get("host_virtualization_capability") or {},
        "next_required_step": str(final_runtime.get("next_required_step") or ""),
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": str(runner),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": network_mode,
        "base_image_path": str(base_disk),
        "base_image_present": base_disk.exists(),
        "image_manifest_present": bool(image_manifest),
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest": final_runtime,
        "instance_manifest": instance_manifest,
        "cloud_init_seed_report": cloud_init_seed,
        "command": command,
        "returncode": process.poll(),
        "stdout": "",
        "stderr": "",
        "process_pid": str(final_runtime.get("process_pid") or ""),
        "process_alive": bool(final_runtime.get("process_alive", False)),
        "virtual_machine_started": bool(final_runtime.get("virtual_machine_started", False)),
        "guest_agent_ready": bool(final_runtime.get("guest_agent_ready", False)),
        "execution_ready": bool(final_runtime.get("execution_ready", False)),
        "cloud_init_seed_enabled": bool(final_runtime.get("cloud_init_seed_enabled", False)),
        "cloud_init_seed_path": str(final_runtime.get("cloud_init_seed_path") or ""),
        "cloud_init_seed_present": bool(final_runtime.get("cloud_init_seed_present", False)),
        "cloud_init_seed_read_only": bool(final_runtime.get("cloud_init_seed_read_only", False)),
        "launcher_kind": "apple_virtualization_runner",
        "runner_stdout_path": str(stdout_path),
        "runner_stderr_path": str(stderr_path),
        "runner_stdout_tail": _tail_file(stdout_path),
        "runner_stderr_tail": _tail_file(stderr_path),
        "guest_console_log_path": str(console_log_path),
        "guest_console_tail": _tail_file(console_log_path),
        "guest_shared_dir_path": str(shared_dir_path),
        "guest_shared_dir_tag": MANAGED_VM_SHARED_DIR_TAG,
        "guest_shared_dir_present": shared_dir_path.exists(),
        "stage_timing": stage_timing,
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "no_host_fallback": True,
    }


def start_managed_vm_instance(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    auto_build_runner: bool = True,
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    runner = managed_vm_runner_path(runner_path, config.state_root)
    image_manifest = load_managed_vm_image_manifest(config.state_root, config.image_id)
    base_disk = Path(managed_vm_base_image_path(config.state_root, config.image_id))
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "start_instance",
        "status": "NOT_RUN",
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": runner,
        "runner_available": bool(runner),
        "auto_build_runner": bool(auto_build_runner),
        "runner_build_report": {},
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "base_image_path": str(base_disk),
        "base_image_present": base_disk.exists(),
        "image_manifest_present": bool(image_manifest),
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "no_host_fallback": True,
    }
    if not image_manifest or not base_disk.exists():
        payload.update({"status": "UNAVAILABLE", "lifecycle_state": "unavailable", "reason": "managed VM base image is not registered"})
        return payload
    if not runner and bool(auto_build_runner) and not config.helper_path:
        runner_build_report = build_managed_vm_virtualization_runner(state_root=config.state_root)
        runner = managed_vm_runner_path(state_root=config.state_root)
        payload.update(
            {
                "runner_build_report": runner_build_report,
                "runner_path": runner,
                "runner_available": bool(runner),
            }
        )
    if runner:
        return _start_managed_vm_instance_with_runner(
            config=config,
            runner=runner,
            image_manifest=image_manifest,
            base_disk=base_disk,
            network_mode=str(payload["network_mode"]),
            timeout_seconds=timeout_seconds,
            startup_wait_seconds=startup_wait_seconds,
        )
    if not config.helper_path:
        payload.update(
            {
                "status": "START_BLOCKED_RUNNER_UNAVAILABLE",
                "lifecycle_state": "blocked",
                "blocker_type": "runner_unavailable",
                "reason": "Apple Virtualization runner was not found or could not be built",
                "next_required_step": "run conos vm build-runner, then retry start-instance",
            }
        )
        return payload
    if not load_managed_vm_instance_manifest(config.state_root, config.instance_id):
        prepare_managed_vm_instance(
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=str(payload["network_mode"]),
        )
        payload["instance_prepared_by_start"] = True
    config, command, completed, helper_payload = _run_helper_lifecycle_command(
        "start",
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=str(payload["network_mode"]),
        timeout_seconds=timeout_seconds,
    )
    runtime_payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": str(helper_payload.get("status") or ("START_FAILED" if int(completed.returncode) else "STARTED")),
        "lifecycle_state": str(helper_payload.get("lifecycle_state") or "unknown"),
        "reason": str(helper_payload.get("reason") or ""),
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": str(payload["network_mode"]),
        "base_image_path": str(base_disk),
        "overlay_path": str(helper_payload.get("overlay_path") or managed_vm_overlay_path(config.state_root, config.instance_id)),
        "overlay_present": bool(helper_payload.get("overlay_present", managed_vm_overlay_path(config.state_root, config.instance_id).exists())),
        "process_pid": str(helper_payload.get("process_pid") or ""),
        "process_alive": _process_alive(helper_payload.get("process_pid")),
        "virtual_machine_started": bool(helper_payload.get("virtual_machine_started", False)),
        "guest_agent_ready": bool(helper_payload.get("guest_agent_ready", False)),
        "execution_ready": bool(helper_payload.get("execution_ready", False)),
        "started_at": _now_iso(),
        "last_helper_command": command,
        "last_helper_returncode": int(completed.returncode),
        "helper_payload": helper_payload,
        "no_host_fallback": True,
    }
    runtime_path = managed_vm_runtime_manifest_path(config.state_root, config.instance_id)
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    instance_manifest = load_managed_vm_instance_manifest(config.state_root, config.instance_id)
    instance_manifest.update(
        {
            "status": runtime_payload["status"],
            "lifecycle_state": runtime_payload["lifecycle_state"],
            "runtime_manifest_path": str(runtime_path),
            "runtime_manifest_present": True,
            "process_pid": runtime_payload["process_pid"],
            "process_alive": runtime_payload["process_alive"],
            "overlay_path": runtime_payload["overlay_path"],
            "overlay_present": runtime_payload["overlay_present"],
            "virtual_machine_started": runtime_payload["virtual_machine_started"],
            "execution_ready": runtime_payload["execution_ready"],
            "guest_agent_ready": runtime_payload["guest_agent_ready"],
            "last_start_attempt_at": runtime_payload["started_at"],
            "last_start_returncode": int(completed.returncode),
            "no_host_fallback": True,
        }
    )
    managed_vm_instance_manifest_path(config.state_root, config.instance_id).write_text(
        json.dumps(instance_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload.update(runtime_payload)
    payload.update(
        {
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "runtime_manifest": runtime_payload,
            "instance_manifest": instance_manifest,
        }
    )
    return payload


def managed_vm_runtime_status(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    runtime_manifest = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    runtime_manifest = _refresh_apple_runner_runtime_exit(
        state_root=config.state_root,
        instance_id=config.instance_id,
        runtime_manifest=runtime_manifest,
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "runtime_status",
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "runtime_manifest_present": bool(runtime_manifest),
        "runtime_manifest": runtime_manifest,
        "process_alive": _process_alive(runtime_manifest.get("process_pid")),
        "no_host_fallback": True,
    }
    payload["status"] = str(runtime_manifest.get("status") or "STOPPED")
    payload["lifecycle_state"] = str(runtime_manifest.get("lifecycle_state") or "stopped")
    payload["process_alive"] = _process_alive(runtime_manifest.get("process_pid"))
    if config.helper_path and str(runtime_manifest.get("launcher_kind") or "") != "apple_virtualization_runner":
        try:
            _, command, completed, helper_payload = _run_helper_lifecycle_command(
                "runtime-status",
                state_root=config.state_root,
                helper_path=config.helper_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=str(payload["network_mode"]),
                timeout_seconds=timeout_seconds,
            )
            payload.update(
                {
                    "helper_payload": helper_payload,
                    "command": command,
                    "returncode": int(completed.returncode),
                    "stdout": str(completed.stdout or ""),
                    "stderr": str(completed.stderr or ""),
                }
            )
        except FileNotFoundError:
            pass
    return payload


def managed_vm_guest_agent_status(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "guest_agent_status",
        "status": gate["status"],
        "ready": bool(gate["ready"]),
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "gate": gate,
        "no_host_fallback": True,
    }
    if config.helper_path:
        try:
            _, command, completed, helper_payload = _run_helper_lifecycle_command(
                "agent-status",
                state_root=config.state_root,
                helper_path=config.helper_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=str(payload["network_mode"]),
                timeout_seconds=timeout_seconds,
            )
            payload.update(
                {
                    "helper_payload": helper_payload,
                    "command": command,
                    "returncode": int(completed.returncode),
                    "stdout": str(completed.stdout or ""),
                    "stderr": str(completed.stderr or ""),
                }
            )
        except FileNotFoundError:
            pass
    return payload


def _stage_timing_entry(
    stage: str,
    started_at: float,
    payload: Dict[str, Any] | None = None,
    *,
    status: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a compact stage-timing record without embedding large payloads."""

    source = payload if isinstance(payload, dict) else {}
    entry: Dict[str, Any] = {
        "stage": str(stage),
        "duration_seconds": max(0.0, time.monotonic() - started_at),
    }
    selected_status = _clean(status) or _clean(source.get("status")) or _clean(source.get("operation"))
    if selected_status:
        entry["status"] = selected_status
    for key in (
        "ready",
        "healthy",
        "recovered",
        "passed",
        "start_attempted",
        "already_ready",
        "process_alive",
        "guest_agent_ready",
        "execution_ready",
        "pid_changed",
    ):
        if key in source:
            entry[key] = bool(source.get(key))
    for key in ("process_pid", "initial_pid", "final_pid", "reason"):
        if key in source and _clean(source.get(key)):
            entry[key] = str(source.get(key))
    if extra:
        entry.update(extra)
    return entry


def ensure_managed_vm_instance_running(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 60.0,
    auto_build_runner: bool = True,
) -> Dict[str, Any]:
    """Idempotently start a managed VM instance and wait for guest execution readiness."""

    started_monotonic = time.monotonic()
    stage_timing: list[Dict[str, Any]] = []
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    stage_started = time.monotonic()
    before_status = managed_vm_runtime_status(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("before_runtime_status", stage_started, before_status))
    stage_started = time.monotonic()
    before_gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    stage_timing.append(_stage_timing_entry("before_guest_agent_gate", stage_started, before_gate))
    before_runtime = before_status.get("runtime_manifest") if isinstance(before_status.get("runtime_manifest"), dict) else {}
    process_alive = bool(before_status.get("process_alive", False))
    already_ready = bool(before_gate.get("ready")) and process_alive
    start_report: Dict[str, Any] = {}
    wait_report: Dict[str, Any] = {}
    start_attempted = False
    if not already_ready:
        lifecycle = str(before_status.get("lifecycle_state") or "").lower()
        status = str(before_status.get("status") or "")
        runtime_missing = not bool(before_status.get("runtime_manifest_present", False))
        runner_dead = bool(before_runtime) and not process_alive
        startable = runtime_missing or runner_dead or lifecycle in {
            "",
            "stopped",
            "failed",
            "blocked",
            "start_blocked",
            "unavailable",
        } or status in {
            "",
            "STOPPED",
            "START_FAILED",
            "START_BLOCKED_RUNNER_UNAVAILABLE",
            "START_BLOCKED_GUEST_AGENT_OR_BOOT_IMPL_MISSING",
            "UNAVAILABLE",
        }
        if startable:
            start_attempted = True
            stage_started = time.monotonic()
            start_report = start_managed_vm_instance(
                state_root=config.state_root,
                helper_path=config.helper_path,
                runner_path=runner_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=network_mode,
                timeout_seconds=timeout_seconds,
                startup_wait_seconds=startup_wait_seconds,
                auto_build_runner=auto_build_runner,
            )
            stage_timing.append(_stage_timing_entry("start_instance", stage_started, start_report))
        else:
            stage_timing.append(
                {
                    "stage": "start_instance",
                    "duration_seconds": 0.0,
                    "status": "SKIPPED_NOT_STARTABLE",
                }
            )
    else:
        stage_timing.append(
            {
                "stage": "start_instance",
                "duration_seconds": 0.0,
                "status": "SKIPPED_ALREADY_READY",
            }
        )
    stage_started = time.monotonic()
    wait_report = wait_managed_vm_guest_agent_ready(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
        wait_seconds=guest_wait_seconds,
    )
    stage_timing.append(
        _stage_timing_entry(
            "wait_guest_agent_ready",
            stage_started,
            wait_report,
            extra={
                "attempts": int(wait_report.get("attempts", 0) or 0),
                "wait_seconds": float(wait_report.get("wait_seconds", 0.0) or 0.0),
                "poll_interval_seconds": float(wait_report.get("poll_interval_seconds", 0.0) or 0.0),
            },
        )
    )
    stage_started = time.monotonic()
    after_status = managed_vm_runtime_status(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("after_runtime_status", stage_started, after_status))
    stage_started = time.monotonic()
    after_gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    stage_timing.append(_stage_timing_entry("after_guest_agent_gate", stage_started, after_gate))
    ready = bool(after_gate.get("ready")) and bool(after_status.get("process_alive", False))
    final_runtime = after_status.get("runtime_manifest") if isinstance(after_status.get("runtime_manifest"), dict) else {}
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "ensure_instance_running",
        "status": "READY" if ready else ("STARTING" if bool(after_status.get("process_alive", False)) else "NOT_READY"),
        "ready": ready,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": managed_vm_runner_path(runner_path, config.state_root),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "start_attempted": start_attempted,
        "already_ready": already_ready,
        "before_status": before_status,
        "before_gate": before_gate,
        "start_report": start_report,
        "wait_report": wait_report,
        "runtime_status": after_status,
        "guest_agent_gate": after_gate,
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "runtime_manifest_present": bool(after_status.get("runtime_manifest_present", False)),
        "runtime_manifest": final_runtime,
        "process_pid": str(final_runtime.get("process_pid") or ""),
        "process_alive": bool(after_status.get("process_alive", False)),
        "virtual_machine_started": bool(final_runtime.get("virtual_machine_started", False)),
        "guest_agent_ready": bool(final_runtime.get("guest_agent_ready", False)),
        "execution_ready": bool(final_runtime.get("execution_ready", False)),
        "stage_timing": stage_timing,
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "reason": "" if ready else str(after_gate.get("reason") or wait_report.get("reason") or "managed VM instance is not ready"),
        "no_host_fallback": True,
    }


def managed_vm_health_check(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Report managed VM lifecycle health and repair stale manifest fields."""

    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    before_runtime = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    before_instance = load_managed_vm_instance_manifest(config.state_root, config.instance_id)
    runtime_status = managed_vm_runtime_status(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
    )
    gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    runtime_manifest = gate.get("runtime_manifest") if isinstance(gate.get("runtime_manifest"), dict) else {}
    if not runtime_manifest:
        runtime_manifest = runtime_status.get("runtime_manifest") if isinstance(runtime_status.get("runtime_manifest"), dict) else {}

    synced_instance: Dict[str, Any] = {}
    if runtime_manifest:
        synced_instance = _update_instance_runtime_fields(
            state_root=config.state_root,
            instance_id=config.instance_id,
            runtime_payload=runtime_manifest,
        )

    process_alive = bool(gate.get("process_alive", False)) or bool(runtime_status.get("process_alive", False))
    ready = bool(gate.get("ready", False)) and process_alive
    runtime_present = bool(runtime_status.get("runtime_manifest_present", False))
    if ready:
        health_status = "HEALTHY"
        lifecycle_state = "ready"
    elif process_alive:
        health_status = "DEGRADED"
        lifecycle_state = "started_not_ready"
    elif runtime_present:
        health_status = "STOPPED"
        lifecycle_state = "stopped"
    else:
        health_status = "NOT_PREPARED"
        lifecycle_state = "missing_runtime"

    repairs: list[str] = []
    before_status = str(before_runtime.get("status") or "")
    after_status = str(runtime_manifest.get("status") or runtime_status.get("status") or "")
    if before_runtime and before_status == "STOPPED" and after_status == "STARTED" and process_alive:
        repairs.append("stale_stopped_runtime_restored_started")
    if before_runtime and before_status in {"STARTING", "STARTED"} and after_status == "STOPPED" and not process_alive:
        repairs.append("dead_runner_runtime_marked_stopped")
    if before_instance and synced_instance:
        for key in ("status", "lifecycle_state", "process_pid", "process_alive", "virtual_machine_started", "guest_agent_ready", "execution_ready"):
            if before_instance.get(key) != synced_instance.get(key):
                repairs.append("instance_manifest_synced")
                break
    elif runtime_manifest and not before_instance:
        repairs.append("instance_manifest_missing")

    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "health_check",
        "status": health_status,
        "healthy": ready,
        "lifecycle_state": lifecycle_state,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "runtime_manifest_present": runtime_present,
        "runtime_status": runtime_status,
        "guest_agent_gate": gate,
        "instance_manifest_path": str(managed_vm_instance_manifest_path(config.state_root, config.instance_id)),
        "instance_manifest_present": bool(load_managed_vm_instance_manifest(config.state_root, config.instance_id)),
        "instance_manifest_synced": bool(synced_instance),
        "repairs": repairs,
        "process_pid": str(runtime_manifest.get("process_pid") or ""),
        "process_alive": process_alive,
        "virtual_machine_started": bool(runtime_manifest.get("virtual_machine_started", False)),
        "guest_agent_ready": bool(runtime_manifest.get("guest_agent_ready", False)),
        "execution_ready": bool(runtime_manifest.get("execution_ready", False)),
        "reason": "" if ready else str(gate.get("reason") or runtime_status.get("reason") or health_status.lower()),
        "recommended_action": "none" if ready else "recover-instance",
        "no_host_fallback": True,
    }


def recover_managed_vm_instance(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 180.0,
    auto_build_runner: bool = True,
    restart_unready: bool = False,
) -> Dict[str, Any]:
    """Recover a managed VM instance from missing/stopped/stale lifecycle state."""

    started_monotonic = time.monotonic()
    stage_timing: list[Dict[str, Any]] = []
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    stage_started = time.monotonic()
    initial = managed_vm_health_check(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("initial_health_check", stage_started, initial))
    steps: list[Dict[str, Any]] = [{"step": "initial_health_check", "status": str(initial.get("status") or "")}]
    stop_report: Dict[str, Any] = {}
    ensure_report: Dict[str, Any] = {}
    if bool(initial.get("healthy")):
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "recover_instance",
            "status": "READY",
            "recovered": False,
            "state_root": config.state_root,
            "helper_path": config.helper_path,
            "image_id": config.image_id,
            "instance_id": config.instance_id,
            "network_mode": _clean(network_mode) or "provider_default",
            "initial_health": initial,
            "steps": steps,
            "recovery_stage_timing": stage_timing,
            "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
            "final_health": initial,
            "reason": "instance already healthy",
            "no_host_fallback": True,
        }

    if bool(initial.get("process_alive")) and not bool(initial.get("healthy")) and bool(restart_unready):
        stage_started = time.monotonic()
        stop_report = stop_managed_vm_instance(
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=min(max(1, int(timeout_seconds)), 30),
        )
        stage_timing.append(_stage_timing_entry("stop_unready_instance", stage_started, stop_report))
        steps.append({"step": "stop_unready_instance", "status": str(stop_report.get("status") or "")})

    stage_started = time.monotonic()
    ensure_report = ensure_managed_vm_instance_running(
        state_root=config.state_root,
        helper_path=config.helper_path,
        runner_path=runner_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=timeout_seconds,
        startup_wait_seconds=startup_wait_seconds,
        guest_wait_seconds=guest_wait_seconds,
        auto_build_runner=auto_build_runner,
    )
    stage_timing.append(_stage_timing_entry("ensure_running", stage_started, ensure_report))
    steps.append({"step": "ensure_running", "status": str(ensure_report.get("status") or "")})
    stage_started = time.monotonic()
    final_health = managed_vm_health_check(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("final_health_check", stage_started, final_health))
    steps.append({"step": "final_health_check", "status": str(final_health.get("status") or "")})
    recovered = bool(final_health.get("healthy"))
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "recover_instance",
        "status": "RECOVERED" if recovered else "RECOVERY_FAILED",
        "recovered": recovered,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": managed_vm_runner_path(runner_path, config.state_root),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "initial_health": initial,
        "stop_report": stop_report,
        "ensure_report": ensure_report,
        "final_health": final_health,
        "steps": steps,
        "recovery_stage_timing": stage_timing,
        "process_pid": str(final_health.get("process_pid") or ""),
        "process_alive": bool(final_health.get("process_alive", False)),
        "guest_agent_ready": bool(final_health.get("guest_agent_ready", False)),
        "execution_ready": bool(final_health.get("execution_ready", False)),
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "reason": "" if recovered else str(final_health.get("reason") or ensure_report.get("reason") or "recovery failed"),
        "no_host_fallback": True,
    }


def managed_vm_recovery_drill(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 180.0,
    auto_build_runner: bool = True,
    ensure_ready_first: bool = True,
    crash_signal: str = "SIGKILL",
    verify_agent_exec: bool = True,
    agent_timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Inject a runner failure, recover the VM, and verify guest execution."""

    started_monotonic = time.monotonic()
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    steps: list[Dict[str, Any]] = []
    stage_timing: list[Dict[str, Any]] = []
    baseline_ensure: Dict[str, Any] = {}
    stage_started = time.monotonic()
    initial_health = managed_vm_health_check(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("initial_health_check", stage_started, initial_health))
    steps.append({"step": "initial_health_check", "status": str(initial_health.get("status") or "")})
    if not bool(initial_health.get("healthy")) and bool(ensure_ready_first):
        stage_started = time.monotonic()
        baseline_ensure = ensure_managed_vm_instance_running(
            state_root=config.state_root,
            helper_path=config.helper_path,
            runner_path=runner_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=int(timeout_seconds),
            startup_wait_seconds=float(startup_wait_seconds),
            guest_wait_seconds=float(guest_wait_seconds),
            auto_build_runner=bool(auto_build_runner),
        )
        stage_timing.append(_stage_timing_entry("ensure_ready_before_drill", stage_started, baseline_ensure))
        steps.append({"step": "ensure_ready_before_drill", "status": str(baseline_ensure.get("status") or "")})
        stage_started = time.monotonic()
        initial_health = managed_vm_health_check(
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=min(max(1, int(timeout_seconds)), 30),
        )
        stage_timing.append(_stage_timing_entry("post_ensure_health_check", stage_started, initial_health))
        steps.append({"step": "post_ensure_health_check", "status": str(initial_health.get("status") or "")})

    if not bool(initial_health.get("healthy")):
        return {
            "schema_version": MANAGED_VM_PROVIDER_VERSION,
            "operation": "recovery_drill",
            "status": "DRILL_BLOCKED_VM_NOT_READY",
            "passed": False,
            "state_root": config.state_root,
            "helper_path": config.helper_path,
            "runner_path": managed_vm_runner_path(runner_path, config.state_root),
            "image_id": config.image_id,
            "instance_id": config.instance_id,
            "network_mode": _clean(network_mode) or "provider_default",
            "ensure_ready_first": bool(ensure_ready_first),
            "baseline_ensure": baseline_ensure,
            "initial_health": initial_health,
            "steps": steps,
            "recovery_stage_timing": stage_timing,
            "reason": str(initial_health.get("reason") or "VM was not ready before recovery drill"),
            "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
            "no_host_fallback": True,
        }

    initial_pid = str(initial_health.get("process_pid") or "")
    stage_started = time.monotonic()
    crash_report = _crash_runner_process(
        initial_pid,
        signal_name=str(crash_signal or "SIGKILL"),
        timeout_seconds=min(max(1, int(timeout_seconds)), 10),
    )
    stage_timing.append(
        _stage_timing_entry(
            "inject_runner_failure",
            stage_started,
            crash_report,
            status="CRASHED" if crash_report.get("terminated") else "FAILED",
        )
    )
    steps.append({"step": "inject_runner_failure", "status": "CRASHED" if crash_report.get("terminated") else "FAILED"})
    stage_started = time.monotonic()
    post_crash_health = managed_vm_health_check(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    stage_timing.append(_stage_timing_entry("post_crash_health_check", stage_started, post_crash_health))
    steps.append({"step": "post_crash_health_check", "status": str(post_crash_health.get("status") or "")})
    recovery_started = time.monotonic()
    recovery = recover_managed_vm_instance(
        state_root=config.state_root,
        helper_path=config.helper_path,
        runner_path=runner_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=int(timeout_seconds),
        startup_wait_seconds=float(startup_wait_seconds),
        guest_wait_seconds=float(guest_wait_seconds),
        auto_build_runner=bool(auto_build_runner),
        restart_unready=True,
    )
    recovery_seconds = max(0.0, time.monotonic() - recovery_started)
    stage_timing.append(_stage_timing_entry("recover_instance", recovery_started, recovery))
    steps.append({"step": "recover_instance", "status": str(recovery.get("status") or "")})
    final_health = recovery.get("final_health") if isinstance(recovery.get("final_health"), dict) else {}
    if not final_health:
        stage_started = time.monotonic()
        final_health = managed_vm_health_check(
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=min(max(1, int(timeout_seconds)), 30),
        )
        stage_timing.append(_stage_timing_entry("final_health_check", stage_started, final_health))
    else:
        stage_timing.append(
            {
                "stage": "final_health_check",
                "duration_seconds": 0.0,
                "status": str(final_health.get("status") or "REUSED_FROM_RECOVERY"),
                "healthy": bool(final_health.get("healthy", False)),
                "process_pid": str(final_health.get("process_pid") or ""),
            }
        )
    steps.append({"step": "final_health_check", "status": str(final_health.get("status") or "")})

    agent_exec_probe: Dict[str, Any] = {}
    if bool(verify_agent_exec) and bool(final_health.get("healthy")):
        stage_started = time.monotonic()
        agent_exec_probe = run_managed_vm_agent_command(
            ["echo", "ok"],
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=int(agent_timeout_seconds),
        )
        stage_timing.append(_stage_timing_entry("agent_exec_probe", stage_started, agent_exec_probe))
        steps.append({"step": "agent_exec_probe", "status": str(agent_exec_probe.get("status") or "")})
    agent_returncode = agent_exec_probe.get("returncode", 1)
    try:
        agent_returncode_int = int(agent_returncode)
    except (TypeError, ValueError):
        agent_returncode_int = 1
    agent_ok = (not bool(verify_agent_exec)) or (
        str(agent_exec_probe.get("status") or "") == "COMPLETED"
        and agent_returncode_int == 0
        and "ok" in str(agent_exec_probe.get("stdout") or "")
    )
    final_pid = str(final_health.get("process_pid") or "")
    pid_changed = bool(initial_pid and final_pid and initial_pid != final_pid)
    passed = bool(crash_report.get("terminated")) and bool(final_health.get("healthy")) and bool(agent_ok)
    return {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "recovery_drill",
        "status": "DRILL_PASSED" if passed else "DRILL_FAILED",
        "passed": passed,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": managed_vm_runner_path(runner_path, config.state_root),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "ensure_ready_first": bool(ensure_ready_first),
        "crash_signal": str(crash_signal or "SIGKILL").strip().upper(),
        "baseline_ensure": baseline_ensure,
        "initial_health": initial_health,
        "initial_pid": initial_pid,
        "crash_report": crash_report,
        "post_crash_health": post_crash_health,
        "recovery": recovery,
        "final_health": final_health,
        "final_pid": final_pid,
        "pid_changed": pid_changed,
        "agent_exec_probe": agent_exec_probe,
        "agent_exec_verified": bool(agent_ok),
        "recovery_seconds": recovery_seconds,
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "recovery_stage_timing": stage_timing,
        "steps": steps,
        "reason": "" if passed else str(final_health.get("reason") or recovery.get("reason") or crash_report.get("reason") or "recovery drill failed"),
        "no_host_fallback": True,
    }


def _float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        return None
    return result


def _duration_distribution(values: Sequence[object]) -> Dict[str, Any]:
    selected = sorted(value for value in (_float_or_none(item) for item in values) if value is not None)
    if not selected:
        return {
            "values": [],
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    midpoint = len(selected) // 2
    if len(selected) % 2:
        median = selected[midpoint]
    else:
        median = (selected[midpoint - 1] + selected[midpoint]) / 2.0
    # Nearest-rank percentile keeps small recovery drills easy to audit.
    p95_index = max(0, min(len(selected) - 1, ((95 * len(selected) + 99) // 100) - 1))
    return {
        "values": selected,
        "min": selected[0],
        "max": selected[-1],
        "mean": sum(selected) / float(len(selected)),
        "median": median,
        "p95": selected[p95_index],
    }


def _returncode_zero(payload: Dict[str, Any]) -> bool:
    try:
        return int(payload.get("returncode", 1)) == 0
    except (TypeError, ValueError):
        return False


def managed_vm_recovery_soak(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 180.0,
    auto_build_runner: bool = True,
    rounds: int = 3,
    cooldown_seconds: float = 0.0,
    crash_signal: str = "SIGKILL",
    verify_agent_exec: bool = True,
    agent_timeout_seconds: int = 30,
    disk_probe: bool = True,
    report_path: str = "",
) -> Dict[str, Any]:
    """Run repeated crash/recovery drills and summarize VM recovery stability."""

    started_monotonic = time.monotonic()
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    requested_rounds = max(1, int(rounds))
    cooldown = max(0.0, float(cooldown_seconds))
    round_reports: list[Dict[str, Any]] = []

    for round_index in range(1, requested_rounds + 1):
        drill = managed_vm_recovery_drill(
            state_root=config.state_root,
            helper_path=config.helper_path,
            runner_path=runner_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=network_mode,
            timeout_seconds=int(timeout_seconds),
            startup_wait_seconds=float(startup_wait_seconds),
            guest_wait_seconds=float(guest_wait_seconds),
            auto_build_runner=bool(auto_build_runner),
            ensure_ready_first=True,
            crash_signal=str(crash_signal or "SIGKILL"),
            verify_agent_exec=bool(verify_agent_exec),
            agent_timeout_seconds=int(agent_timeout_seconds),
        )
        disk_probe_report: Dict[str, Any] = {}
        disk_probe_ok = not bool(disk_probe)
        if bool(disk_probe) and bool(drill.get("passed")):
            disk_probe_started = time.monotonic()
            disk_probe_report = run_managed_vm_agent_command(
                [
                    "sh",
                    "-lc",
                    "printf conos-recovery-soak-ok > /tmp/conos-recovery-soak-probe && sync && cat /tmp/conos-recovery-soak-probe",
                ],
                state_root=config.state_root,
                helper_path=config.helper_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=network_mode,
                timeout_seconds=int(agent_timeout_seconds),
            )
            disk_probe_seconds = max(0.0, time.monotonic() - disk_probe_started)
            disk_probe_ok = (
                str(disk_probe_report.get("status") or "") == "COMPLETED"
                and _returncode_zero(disk_probe_report)
                and "conos-recovery-soak-ok" in str(disk_probe_report.get("stdout") or "")
            )
        else:
            disk_probe_seconds = 0.0
        round_passed = bool(drill.get("passed")) and bool(disk_probe_ok)
        round_reports.append(
            {
                "round_index": round_index,
                "status": "ROUND_PASSED" if round_passed else "ROUND_FAILED",
                "passed": round_passed,
                "drill_status": str(drill.get("status") or ""),
                "reason": "" if round_passed else str(drill.get("reason") or disk_probe_report.get("reason") or "recovery soak round failed"),
                "recovery_seconds": _float_or_none(drill.get("recovery_seconds")),
                "duration_seconds": _float_or_none(drill.get("duration_seconds")),
                "initial_pid": str(drill.get("initial_pid") or ""),
                "final_pid": str(drill.get("final_pid") or ""),
                "pid_changed": bool(drill.get("pid_changed", False)),
                "agent_exec_verified": bool(drill.get("agent_exec_verified", False)),
                "disk_probe_enabled": bool(disk_probe),
                "disk_probe_ok": bool(disk_probe_ok),
                "disk_probe_seconds": disk_probe_seconds,
                "disk_probe": disk_probe_report,
                "recovery_stage_timing": list(drill.get("recovery_stage_timing") or []),
                "drill": drill,
            }
        )
        if round_index < requested_rounds and cooldown > 0:
            time.sleep(cooldown)

    final_health = managed_vm_health_check(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
        network_mode=network_mode,
        timeout_seconds=min(max(1, int(timeout_seconds)), 30),
    )
    success_count = sum(1 for item in round_reports if bool(item.get("passed")))
    failure_count = max(0, requested_rounds - success_count)
    final_healthy = bool(final_health.get("healthy"))
    passed = failure_count == 0 and len(round_reports) == requested_rounds and final_healthy
    failure_reasons = [
        f"round {item.get('round_index')}: {item.get('reason')}"
        for item in round_reports
        if not bool(item.get("passed"))
    ]
    if not final_healthy:
        failure_reasons.append(str(final_health.get("reason") or "final VM health check failed"))
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "recovery_soak",
        "status": "SOAK_PASSED" if passed else "SOAK_FAILED",
        "passed": passed,
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "runner_path": managed_vm_runner_path(runner_path, config.state_root),
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "rounds_requested": requested_rounds,
        "rounds_completed": len(round_reports),
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": success_count / float(requested_rounds),
        "agent_exec_success_count": sum(1 for item in round_reports if bool(item.get("agent_exec_verified"))),
        "disk_probe_enabled": bool(disk_probe),
        "disk_probe_success_count": sum(1 for item in round_reports if bool(item.get("disk_probe_ok"))),
        "pid_changed_count": sum(1 for item in round_reports if bool(item.get("pid_changed"))),
        "recovery_seconds": _duration_distribution([item.get("recovery_seconds") for item in round_reports]),
        "rounds": round_reports,
        "final_health": final_health,
        "final_pid": str(final_health.get("process_pid") or ""),
        "duration_seconds": max(0.0, time.monotonic() - started_monotonic),
        "failure_reasons": failure_reasons,
        "reason": "" if passed else "; ".join(reason for reason in failure_reasons if reason),
        "report_path": str(Path(report_path).expanduser()) if _clean(report_path) else "",
        "no_host_fallback": True,
    }
    if _clean(report_path):
        _write_json(Path(report_path).expanduser(), payload)
    return payload


def run_managed_vm_agent_command(
    command: Sequence[str],
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 30,
    cwd: str = "",
    stdin_bytes: bytes | None = None,
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "agent_exec",
        "status": "NOT_RUN",
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "command": [str(part) for part in command],
        "cwd": _clean(cwd),
        "stdin_bytes_present": stdin_bytes is not None,
        "stdin_size_bytes": len(stdin_bytes) if stdin_bytes is not None else 0,
        "gate": gate,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "no_host_fallback": True,
    }
    if not gate["ready"]:
        payload.update(
            {
                "status": "EXEC_BLOCKED_GUEST_AGENT_NOT_READY",
                "reason": str(gate.get("reason") or "guest agent not ready"),
                "returncode": 78,
            }
        )
        return payload
    if not command:
        payload.update({"status": "INVALID_REQUEST", "reason": "agent-exec requires a command", "returncode": 64})
        return payload
    if not config.helper_path:
        request = _write_managed_vm_agent_request(
            command,
            state_root=config.state_root,
            image_id=config.image_id,
            instance_id=config.instance_id,
            timeout_seconds=int(timeout_seconds),
            cwd=_clean(cwd),
            stdin_bytes=stdin_bytes,
        )
        result_path = Path(str(request["result_path"]))
        agent_payload = _wait_for_managed_vm_agent_result(result_path, timeout_seconds=int(timeout_seconds))
        status = str(agent_payload.get("status") or "FAILED")
        payload.update(
            {
                "status": status,
                "reason": str(agent_payload.get("reason") or ""),
                "agent_transport": "apple_virtualization_request_spool",
                "request_id": request["request_id"],
                "request_path": request["request_path"],
                "result_path": request["result_path"],
                "returncode": int(agent_payload.get("returncode", 0 if status == "COMPLETED" else 1) or 0),
                "stdout": str(agent_payload.get("stdout") or ""),
                "stderr": str(agent_payload.get("stderr") or ""),
                "agent_payload": agent_payload,
            }
        )
        if "stdout_b64" in agent_payload:
            payload["stdout_b64"] = str(agent_payload.get("stdout_b64") or "")
        if "stderr_b64" in agent_payload:
            payload["stderr_b64"] = str(agent_payload.get("stderr_b64") or "")
        return payload
    agent_command = _helper_agent_exec_command(command, config=config, network_mode=str(payload["network_mode"]))
    completed = subprocess.run(
        agent_command,
        capture_output=True,
        text=True,
        timeout=int(timeout_seconds),
        check=False,
    )
    helper_payload = _read_json_from_text(str(completed.stdout or ""))
    payload.update(
        {
            "status": str(helper_payload.get("status") or ("COMPLETED" if int(completed.returncode) == 0 else "FAILED")),
            "reason": str(helper_payload.get("reason") or ""),
            "agent_command": agent_command,
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "helper_payload": helper_payload,
        }
    )
    return payload


def stop_managed_vm_instance(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    runtime_path = managed_vm_runtime_manifest_path(config.state_root, config.instance_id)
    runtime_manifest = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "stop_instance",
        "state_root": config.state_root,
        "helper_path": config.helper_path,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "runtime_manifest_path": str(runtime_path),
        "runtime_manifest_present_before_stop": bool(runtime_manifest),
        "no_host_fallback": True,
    }
    helper_payload: Dict[str, Any] = {}
    command: list[str] = []
    completed_returncode = 0
    stdout = ""
    stderr = ""
    runner_termination: Dict[str, Any] = {}
    last_guest_boot_diagnostic: Dict[str, Any] = {}
    if runtime_manifest:
        console_log_raw = str(runtime_manifest.get("guest_console_log_path") or "")
        shared_dir_raw = str(runtime_manifest.get("guest_shared_dir_path") or "")
        shared_dir = Path(shared_dir_raw) if shared_dir_raw else None
        shared_marker = shared_dir / "cloud-init-runcmd.txt" if shared_dir is not None else None
        cloud_init_marker_names = [path.name for path in _cloud_init_marker_paths(shared_dir)]
        initramfs_marker_names = (
            [path.name for path in sorted(shared_dir.glob("conos-initramfs-*.txt"))]
            if shared_dir is not None and shared_dir.exists()
            else []
        )
        last_guest_boot_diagnostic = _managed_vm_guest_boot_diagnostic(
            runtime_manifest=runtime_manifest,
            process_alive=_process_alive(runtime_manifest.get("process_pid")),
            guest_console_tail=_tail_file(Path(console_log_raw)) if console_log_raw else "",
            guest_shared_runcmd_marker_present=bool(shared_marker and shared_marker.exists()),
            guest_cloud_init_markers=cloud_init_marker_names,
            guest_initramfs_trace_markers=initramfs_marker_names,
        )
    if str(runtime_manifest.get("launcher_kind") or "") == "apple_virtualization_runner":
        runner_termination = _terminate_runner_process(
            runtime_manifest.get("process_pid"),
            timeout_seconds=max(1, int(timeout_seconds)),
        )
        helper_payload = {
            "status": "STOPPED",
            "lifecycle_state": "stopped",
            "reason": str(runner_termination.get("reason") or ""),
            "runner_termination": runner_termination,
        }
    elif config.helper_path:
        try:
            _, command, completed, helper_payload = _run_helper_lifecycle_command(
                "stop",
                state_root=config.state_root,
                helper_path=config.helper_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=str(payload["network_mode"]),
                timeout_seconds=timeout_seconds,
            )
            completed_returncode = int(completed.returncode)
            stdout = str(completed.stdout or "")
            stderr = str(completed.stderr or "")
        except FileNotFoundError:
            helper_payload = {"status": "STOPPED", "reason": "managed VM helper was not found during stop"}
    stopped_payload = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "artifact_type": "managed_vm_runtime",
        "status": str(helper_payload.get("status") or "STOPPED"),
        "lifecycle_state": str(helper_payload.get("lifecycle_state") or "stopped"),
        "reason": str(helper_payload.get("reason") or ""),
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "process_pid": "",
        "process_alive": False,
        "virtual_machine_started": False,
        "guest_agent_ready": False,
        "execution_ready": False,
        "stopped_at": _now_iso(),
        "helper_payload": helper_payload,
        "runner_termination": runner_termination,
        "last_guest_boot_diagnostic": last_guest_boot_diagnostic,
        "no_host_fallback": True,
    }
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(json.dumps(stopped_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    instance_manifest = load_managed_vm_instance_manifest(config.state_root, config.instance_id)
    if instance_manifest:
        instance_manifest.update(
            {
                "status": stopped_payload["status"],
                "lifecycle_state": stopped_payload["lifecycle_state"],
                "process_pid": "",
                "process_alive": False,
                "virtual_machine_started": False,
                "guest_agent_ready": False,
                "execution_ready": False,
                "last_guest_boot_diagnostic": last_guest_boot_diagnostic,
                "last_stop_at": stopped_payload["stopped_at"],
            }
        )
        managed_vm_instance_manifest_path(config.state_root, config.instance_id).write_text(
            json.dumps(instance_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    payload.update(
        stopped_payload,
        runtime_manifest=stopped_payload,
        instance_manifest=instance_manifest,
        command=command,
        returncode=completed_returncode,
        stdout=stdout,
        stderr=stderr,
    )
    return payload


def managed_vm_report(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
) -> Dict[str, Any]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    manifest = load_managed_vm_manifest(config.state_root)
    image_manifest = load_managed_vm_image_manifest(config.state_root, config.image_id)
    instance_manifest = load_managed_vm_instance_manifest(config.state_root, config.instance_id)
    runtime_manifest = load_managed_vm_runtime_manifest(config.state_root, config.instance_id)
    guest_agent_gate = managed_vm_guest_agent_gate(
        state_root=config.state_root,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    helper_available = bool(config.helper_path)
    source_path = Path(managed_vm_helper_source_path())
    build_output = managed_vm_build_output_path(config.state_root)
    runner_source_path = Path(managed_vm_runner_source_path())
    runner_entitlements_path = Path(managed_vm_runner_entitlements_path())
    runner_build_output = managed_vm_runner_build_output_path(config.state_root)
    runner = managed_vm_runner_path(state_root=config.state_root)
    guest_agent_source = Path(managed_vm_guest_agent_source_path())
    guest_bundle = managed_vm_guest_initrd_bundle_path(config.state_root)
    base_image = managed_vm_base_image_path(config.state_root, config.image_id)
    kernel = managed_vm_kernel_path(config.state_root, config.image_id)
    initrd = managed_vm_initrd_path(config.state_root, config.image_id)
    overlay = managed_vm_overlay_path(config.state_root, config.instance_id)
    writable_disk = managed_vm_writable_disk_path(config.state_root, config.instance_id)
    swiftc = shutil.which("swiftc")
    clang = shutil.which("clang")
    provider_status = "AVAILABLE" if helper_available or runner else "UNAVAILABLE"
    provider_reason = "" if helper_available or runner else "managed VM helper/Apple Virtualization runner was not found"
    if runner and not helper_available:
        provider_reason = "Apple Virtualization runner available; legacy helper is not required for runner-spool execution"
    report = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "status": provider_status,
        "provider": "managed",
        "real_vm_boundary": bool(runner),
        "requires_user_configured_vm": False,
        "requires_helper": False,
        "requires_virtualization_runner_for_start": True,
        "helper_required_for_start": False,
        "helper_required_for_agent_exec": False,
        "helper_path": config.helper_path,
        "helper_name": MANAGED_VM_HELPER_NAME,
        "helper_source_path": str(source_path),
        "helper_source_present": source_path.exists(),
        "helper_build_output": build_output,
        "helper_build_output_present": Path(build_output).exists(),
        "virtualization_runner_path": runner,
        "virtualization_runner_available": bool(runner),
        "virtualization_runner_name": MANAGED_VM_RUNNER_NAME,
        "virtualization_runner_source_path": str(runner_source_path),
        "virtualization_runner_source_present": runner_source_path.exists(),
        "virtualization_runner_entitlements_path": str(runner_entitlements_path),
        "virtualization_runner_entitlements_present": runner_entitlements_path.exists(),
        "virtualization_runner_build_output": runner_build_output,
        "virtualization_runner_build_output_present": Path(runner_build_output).exists(),
        "guest_agent_source_path": str(guest_agent_source),
        "guest_agent_source_present": guest_agent_source.exists(),
        "guest_initrd_bundle_path": str(guest_bundle),
        "guest_initrd_bundle_present": guest_bundle.exists(),
        "guest_agent_autostart_configured": bool(image_manifest.get("guest_agent_autostart_configured", False)),
        "guest_agent_autostart_planned": bool(image_manifest.get("guest_agent_autostart_planned", False)),
        "guest_agent_installation_mode": str(image_manifest.get("guest_agent_installation_mode") or ""),
        "guest_agent_installation_status": str(image_manifest.get("guest_agent_installation_status") or ""),
        "verified_execution_path": str(image_manifest.get("verified_execution_path") or ""),
        "cloud_init_guest_capability": (
            image_manifest.get("cloud_init_guest_capability")
            if isinstance(image_manifest.get("cloud_init_guest_capability"), dict)
            else {}
        ),
        "swiftc_path": str(swiftc or ""),
        "clang_path": str(clang or ""),
        "state_root": config.state_root,
        "image_id": config.image_id,
        "base_image_path": base_image,
        "base_image_present": Path(base_image).exists(),
        "boot_mode": str(image_manifest.get("boot_mode") or ""),
        "kernel_path": str(image_manifest.get("kernel_path") or kernel),
        "kernel_present": Path(str(image_manifest.get("kernel_path") or kernel)).exists(),
        "initrd_path": str(image_manifest.get("initrd_path") or initrd),
        "initrd_present": Path(str(image_manifest.get("initrd_path") or initrd)).exists(),
        "image_manifest_present": bool(image_manifest),
        "image_manifest": image_manifest,
        "instance_id": config.instance_id,
        "instance_manifest_path": str(managed_vm_instance_manifest_path(config.state_root, config.instance_id)),
        "instance_manifest_present": bool(instance_manifest),
        "instance_manifest": instance_manifest,
        "runtime_manifest_path": str(managed_vm_runtime_manifest_path(config.state_root, config.instance_id)),
        "runtime_manifest_present": bool(runtime_manifest),
        "runtime_manifest": runtime_manifest,
        "runtime_process_alive": _process_alive(runtime_manifest.get("process_pid")),
        "guest_agent_gate": guest_agent_gate,
        "overlay_path": str(overlay),
        "overlay_present": overlay.exists(),
        "writable_disk_path": str(writable_disk),
        "writable_disk_present": writable_disk.exists(),
        "manifest_present": bool(manifest),
        "manifest": manifest,
        "reason": provider_reason,
        "default_directories": {
            "images": str(Path(config.state_root) / "images"),
            "instances": str(Path(config.state_root) / "instances"),
            "snapshots": str(Path(config.state_root) / "snapshots"),
            "overlays": str(Path(config.state_root) / "overlays"),
            "logs": str(Path(config.state_root) / "logs"),
        },
        "limitations": [
            "requires_apple_virtualization_runner",
            "requires_guest_agent_ready_for_exec",
            "does_not_fall_back_to_host_process",
        ],
    }
    report["operator_guidance"] = guidance_for_vm_report(report)
    report["operator_summary"] = (
        "VM provider 已就绪"
        if provider_status == "AVAILABLE" and bool(report.get("base_image_present")) and bool(report.get("runtime_process_alive"))
        else "VM 还需要 setup 或 recovery，暂时不能作为稳定默认执行边界"
    )
    return report


MANAGED_VM_SETUP_PLAN_VERSION = "conos.managed_vm_setup_plan/v1"
MANAGED_VM_DEFAULT_BOUNDARY_SETUP_VERSION = "conos.managed_vm_default_boundary_setup/v1"


def _managed_vm_command(command: str, *, state_root: str, image_id: str, instance_id: str, **kwargs: Any) -> Dict[str, Any]:
    args = ["conos", "vm", str(command)]
    if state_root:
        args.extend(["--state-root", str(state_root)])
    if image_id:
        args.extend(["--image-id", str(image_id)])
    if instance_id:
        args.extend(["--instance-id", str(instance_id)])
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(option)
            continue
        args.extend([option, str(value)])
    return {
        "command": args,
        "display": " ".join(args),
    }


def _setup_stage(
    name: str,
    *,
    ready: bool,
    status: str,
    reason: str = "",
    next_actions: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "name": str(name),
        "ready": bool(ready),
        "status": str(status),
        "reason": str(reason or ""),
        "next_actions": list(next_actions or []),
    }


def _guest_agent_ready_from_gate(gate: Mapping[str, Any]) -> bool:
    return bool(
        gate.get("ready")
        or gate.get("guest_agent_ready")
        or gate.get("execution_ready")
        or str(gate.get("status") or "").upper() in {"READY", "OK"}
    )


def _managed_vm_setup_plan_from_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    state_root = str(report.get("state_root") or "")
    image_id = str(report.get("image_id") or "")
    instance_id = str(report.get("instance_id") or "")
    runner_ready = bool(report.get("virtualization_runner_available"))
    image_ready = bool(report.get("image_manifest_present")) and bool(report.get("base_image_present"))
    instance_ready = bool(report.get("instance_manifest_present"))
    runtime_manifest_present = bool(report.get("runtime_manifest_present"))
    runtime_alive = bool(report.get("runtime_process_alive"))
    real_vm_boundary = bool(report.get("real_vm_boundary"))
    gate = report.get("guest_agent_gate") if isinstance(report.get("guest_agent_gate"), Mapping) else {}
    guest_agent_ready = _guest_agent_ready_from_gate(gate)
    execution_ready = real_vm_boundary and runtime_alive and guest_agent_ready

    runner_actions = [
        _managed_vm_command("build-runner", state_root=state_root, image_id="", instance_id=""),
    ]
    image_actions = [
        _managed_vm_command("recipe-report", state_root="", image_id="", instance_id=""),
        _managed_vm_command("bootstrap-image", state_root=state_root, image_id=image_id, instance_id=instance_id),
        {
            **_managed_vm_command("install-base-image-bundle", state_root=state_root, image_id=image_id, instance_id=""),
            "requires": "bundle-dir",
        },
    ]
    instance_actions = [
        _managed_vm_command("prepare-instance", state_root=state_root, image_id=image_id, instance_id=instance_id),
    ]
    start_actions = [
        _managed_vm_command("start-instance", state_root=state_root, image_id=image_id, instance_id=instance_id),
        _managed_vm_command("ensure-running", state_root=state_root, image_id=image_id, instance_id=instance_id),
    ]
    recovery_actions = [
        _managed_vm_command("recover-instance", state_root=state_root, image_id=image_id, instance_id=instance_id),
        _managed_vm_command("recovery-drill", state_root=state_root, image_id=image_id, instance_id=instance_id),
    ]
    agent_actions = [
        _managed_vm_command("agent-status", state_root=state_root, image_id=image_id, instance_id=instance_id),
        _managed_vm_command("recover-instance", state_root=state_root, image_id=image_id, instance_id=instance_id, restart_unready=True),
    ]

    stages = [
        _setup_stage(
            "virtualization_runner",
            ready=runner_ready,
            status="READY" if runner_ready else "MISSING",
            reason="" if runner_ready else "Apple Virtualization runner is required before Con OS can start its own VM boundary.",
            next_actions=[] if runner_ready else runner_actions,
        ),
        _setup_stage(
            "base_image",
            ready=image_ready,
            status="READY" if image_ready else "MISSING",
            reason="" if image_ready else "A Con OS base image manifest and disk artifact are required.",
            next_actions=[] if image_ready else image_actions,
        ),
        _setup_stage(
            "instance_manifest",
            ready=instance_ready,
            status="READY" if instance_ready else "MISSING",
            reason="" if instance_ready else "The default VM instance has not been prepared from the base image.",
            next_actions=[] if instance_ready else instance_actions,
        ),
        _setup_stage(
            "runtime_process",
            ready=runtime_alive,
            status="READY" if runtime_alive else ("STALE" if runtime_manifest_present else "STOPPED"),
            reason=(
                ""
                if runtime_alive
                else (
                    "A runtime manifest exists but the VM process is not alive."
                    if runtime_manifest_present
                    else "The VM runtime process has not been started."
                )
            ),
            next_actions=[] if runtime_alive else (recovery_actions if runtime_manifest_present else start_actions),
        ),
        _setup_stage(
            "guest_agent",
            ready=guest_agent_ready,
            status="READY" if guest_agent_ready else "NOT_READY",
            reason="" if guest_agent_ready else str(gate.get("reason") or "Guest execution is not observable yet."),
            next_actions=[] if guest_agent_ready else agent_actions,
        ),
        _setup_stage(
            "execution_boundary",
            ready=execution_ready,
            status="READY" if execution_ready else "BLOCKED",
            reason=(
                ""
                if execution_ready
                else "Default execution is blocked until runner, image, instance, live VM process, and guest agent are all ready."
            ),
            next_actions=[],
        ),
    ]

    blocking_reasons = [stage["reason"] for stage in stages if not stage["ready"] and stage.get("reason")]
    ordered_next_actions: List[Dict[str, Any]] = []
    seen_actions: set[tuple[str, ...]] = set()
    for stage in stages:
        if stage["ready"]:
            continue
        for action in stage.get("next_actions", []):
            command = tuple(str(part) for part in action.get("command", []))
            if command in seen_actions:
                continue
            seen_actions.add(command)
            ordered_next_actions.append(dict(action, stage=stage["name"]))
        if ordered_next_actions:
            break

    if execution_ready:
        status = "READY"
    elif not runner_ready or not image_ready:
        status = "NEEDS_SETUP"
    elif not instance_ready:
        status = "NEEDS_INSTANCE"
    elif runtime_manifest_present and not runtime_alive:
        status = "NEEDS_RECOVERY"
    elif not runtime_alive:
        status = "NEEDS_START"
    else:
        status = "WAITING_GUEST_AGENT"

    return {
        "schema_version": MANAGED_VM_SETUP_PLAN_VERSION,
        "status": status,
        "safe_to_run_tasks": bool(execution_ready),
        "default_execution_boundary_ready": bool(execution_ready),
        "no_host_fallback": True,
        "requires_user_configured_vm": False,
        "state_root": state_root,
        "image_id": image_id,
        "instance_id": instance_id,
        "stages": stages,
        "blocking_reasons": blocking_reasons,
        "next_actions": ordered_next_actions,
        "operator_summary": (
            "VM 默认执行边界已就绪"
            if execution_ready
            else "VM 默认执行边界未就绪，请先完成 next_actions 中的第一组步骤"
        ),
        "provider_snapshot": {
            "status": str(report.get("status") or ""),
            "real_vm_boundary": real_vm_boundary,
            "virtualization_runner_available": runner_ready,
            "base_image_present": bool(report.get("base_image_present")),
            "image_manifest_present": bool(report.get("image_manifest_present")),
            "instance_manifest_present": instance_ready,
            "runtime_manifest_present": runtime_manifest_present,
            "runtime_process_alive": runtime_alive,
            "guest_agent_ready": guest_agent_ready,
            "guest_agent_gate_status": str(gate.get("status") or ""),
            "guest_agent_gate_reason": str(gate.get("reason") or ""),
        },
        "source_report_schema_version": str(report.get("schema_version") or ""),
    }


def managed_vm_setup_plan(
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
) -> Dict[str, Any]:
    report = managed_vm_report(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    return _managed_vm_setup_plan_from_report(report)


def managed_vm_setup_audit_path(state_root: str = "") -> Path:
    return Path(managed_vm_state_root(state_root)) / "logs" / "setup-audit.jsonl"


def _append_managed_vm_setup_audit_event(state_root: str, event: Mapping[str, Any]) -> None:
    path = managed_vm_setup_audit_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANAGED_VM_DEFAULT_BOUNDARY_SETUP_VERSION,
        "event_type": "managed_vm_setup_stage",
        "recorded_at": _now_iso(),
        **dict(event),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _stage_execution_ok(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status") or "").upper()
    if status in {
        "AVAILABLE",
        "BOOTSTRAPPED",
        "BUILT",
        "OK",
        "PREPARED",
        "READY",
        "RECOVERED",
        "RUNNING",
        "STARTED",
        "VERIFIED",
    }:
        return True
    return bool(result.get("execution_ready") or result.get("verified") or result.get("guest_agent_ready"))


def _agent_exec_ok(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status") or "").upper()
    returncode = result.get("returncode")
    return status in {"OK", "READY"} or returncode == 0


def _first_unready_stage(plan: Mapping[str, Any]) -> Dict[str, Any] | None:
    for stage in list(plan.get("stages") or []):
        if isinstance(stage, dict) and not bool(stage.get("ready")):
            return stage
    return None


def managed_vm_prepare_default_boundary(
    *,
    state_root: str = "",
    helper_path: str = "",
    runner_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
    execute: bool = False,
    allow_artifact_download: bool = False,
    artifact_timeout_seconds: int = 120,
    timeout_seconds: int = 120,
    startup_wait_seconds: float = DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS,
    guest_wait_seconds: float = 180.0,
    agent_timeout_seconds: int = 30,
    verify_agent_exec: bool = True,
    overwrite: bool = False,
    write_audit: bool = True,
    max_stage_attempts: int = 8,
) -> Dict[str, Any]:
    """Prepare the built-in VM as the default execution boundary.

    The default mode is a dry-run plan. Side-effect stages only run when
    ``execute`` is true, and large artifact download remains opt-in.
    """

    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    initial_plan = managed_vm_setup_plan(
        state_root=config.state_root,
        helper_path=config.helper_path,
        image_id=config.image_id,
        instance_id=config.instance_id,
    )
    payload: Dict[str, Any] = {
        "schema_version": MANAGED_VM_DEFAULT_BOUNDARY_SETUP_VERSION,
        "operation": "prepare_default_boundary",
        "status": "READY" if initial_plan.get("safe_to_run_tasks") else ("DRY_RUN" if not execute else "RUNNING"),
        "execute": bool(execute),
        "dry_run": not bool(execute),
        "state_root": config.state_root,
        "image_id": config.image_id,
        "instance_id": config.instance_id,
        "network_mode": _clean(network_mode) or "provider_default",
        "allow_artifact_download": bool(allow_artifact_download),
        "no_host_fallback": True,
        "initial_plan": initial_plan,
        "stage_results": [],
        "audit_path": str(managed_vm_setup_audit_path(config.state_root)),
        "next_actions": list(initial_plan.get("next_actions") or []),
    }
    if initial_plan.get("safe_to_run_tasks"):
        payload["final_plan"] = initial_plan
        payload["operator_summary"] = "VM 默认执行边界已就绪"
        return payload
    if not execute:
        payload["operator_summary"] = "dry-run only; add --execute to prepare the VM boundary"
        return payload

    current_plan = initial_plan
    for _attempt in range(max(1, int(max_stage_attempts))):
        if current_plan.get("safe_to_run_tasks"):
            break
        stage = _first_unready_stage(current_plan)
        if not stage:
            break
        stage_name = str(stage.get("name") or "")
        started_at = _now_iso()
        result: Dict[str, Any]
        action = ""
        side_effect = True
        if stage_name == "virtualization_runner":
            action = "build-runner"
            result = build_managed_vm_virtualization_runner(
                state_root=config.state_root,
                output_path=runner_path,
            )
        elif stage_name == "base_image":
            action = "install-default-image"
            result = install_default_managed_vm_image(
                state_root=config.state_root,
                image_id=config.image_id,
                instance_id=config.instance_id,
                allow_artifact_download=bool(allow_artifact_download),
                artifact_timeout_seconds=int(artifact_timeout_seconds),
                runner_path=runner_path,
                network_mode=_clean(network_mode) or "provider_default",
                build_runner=False,
                start_instance=False,
                verify_agent_exec=False,
                keep_running=False,
                startup_wait_seconds=float(startup_wait_seconds),
                guest_wait_seconds=float(guest_wait_seconds),
                agent_timeout_seconds=int(agent_timeout_seconds),
                overwrite=bool(overwrite),
            )
        elif stage_name == "instance_manifest":
            action = "prepare-instance"
            result = prepare_managed_vm_instance(
                state_root=config.state_root,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=_clean(network_mode) or "provider_default",
            )
        elif stage_name == "runtime_process":
            action = "ensure-running"
            result = ensure_managed_vm_instance_running(
                state_root=config.state_root,
                helper_path=config.helper_path,
                runner_path=runner_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=_clean(network_mode) or "provider_default",
                timeout_seconds=int(timeout_seconds),
                startup_wait_seconds=float(startup_wait_seconds),
                guest_wait_seconds=float(guest_wait_seconds),
                auto_build_runner=False,
            )
        elif stage_name == "guest_agent":
            action = "ensure-running"
            result = ensure_managed_vm_instance_running(
                state_root=config.state_root,
                helper_path=config.helper_path,
                runner_path=runner_path,
                image_id=config.image_id,
                instance_id=config.instance_id,
                network_mode=_clean(network_mode) or "provider_default",
                timeout_seconds=int(timeout_seconds),
                startup_wait_seconds=float(startup_wait_seconds),
                guest_wait_seconds=float(guest_wait_seconds),
                auto_build_runner=False,
            )
        else:
            action = "blocked"
            side_effect = False
            result = {
                "status": "BLOCKED",
                "reason": f"no executable setup action for stage {stage_name!r}",
            }

        event = {
            "stage": stage_name,
            "action": action,
            "side_effect": bool(side_effect),
            "started_at": started_at,
            "finished_at": _now_iso(),
            "status": str(result.get("status") or ""),
            "ok": _stage_execution_ok(result),
            "reason": str(result.get("reason") or result.get("next_required_step") or ""),
        }
        payload["stage_results"].append({**event, "result": result})
        if write_audit:
            _append_managed_vm_setup_audit_event(config.state_root, event)
        current_plan = managed_vm_setup_plan(
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
        )
        if not event["ok"] and not current_plan.get("safe_to_run_tasks"):
            break

    final_plan = current_plan
    verification_result: Dict[str, Any] | None = None
    if final_plan.get("safe_to_run_tasks") and verify_agent_exec:
        verification_result = run_managed_vm_agent_command(
            ["/bin/sh", "-lc", "printf conos-vm-ready"],
            state_root=config.state_root,
            helper_path=config.helper_path,
            image_id=config.image_id,
            instance_id=config.instance_id,
            network_mode=_clean(network_mode) or "provider_default",
            timeout_seconds=int(agent_timeout_seconds),
        )
        event = {
            "stage": "agent_exec_smoke",
            "action": "agent-exec",
            "side_effect": False,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
            "status": str(verification_result.get("status") or ""),
            "ok": _agent_exec_ok(verification_result),
            "reason": str(verification_result.get("reason") or verification_result.get("stderr") or ""),
        }
        payload["stage_results"].append({**event, "result": verification_result})
        if write_audit:
            _append_managed_vm_setup_audit_event(config.state_root, event)

    payload["final_plan"] = final_plan
    payload["verification_result"] = verification_result or {}
    payload["safe_to_run_tasks"] = bool(final_plan.get("safe_to_run_tasks")) and (
        not verify_agent_exec
        or not verification_result
        or _agent_exec_ok(verification_result)
    )
    payload["status"] = "READY" if payload["safe_to_run_tasks"] else "BLOCKED"
    payload["next_actions"] = [] if payload["safe_to_run_tasks"] else list(final_plan.get("next_actions") or [])
    payload["operator_summary"] = (
        "VM 默认执行边界准备完成"
        if payload["safe_to_run_tasks"]
        else "VM 默认执行边界仍未就绪，请查看 stage_results 与 next_actions"
    )
    return payload


def build_managed_vm_helper(
    *,
    state_root: str = "",
    source_path: str = "",
    output_path: str = "",
    swiftc_path: str = "",
) -> Dict[str, Any]:
    """Build the macOS managed-VM helper when Swift tooling is available."""

    source = Path(source_path or managed_vm_helper_source_path()).expanduser().resolve()
    output = Path(output_path or managed_vm_build_output_path(state_root)).expanduser().resolve()
    swiftc = _clean(swiftc_path) or _clean(shutil.which("swiftc"))
    report: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "build_helper",
        "status": "NOT_RUN",
        "platform": sys.platform,
        "source_path": str(source),
        "source_present": source.exists(),
        "output_path": str(output),
        "swiftc_path": swiftc,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    if sys.platform != "darwin":
        report.update({"status": "UNAVAILABLE", "reason": "managed VM helper build is macOS-only"})
        return report
    if not source.exists():
        report.update({"status": "UNAVAILABLE", "reason": "managed VM helper Swift source is missing"})
        return report
    if not swiftc:
        report.update({"status": "UNAVAILABLE", "reason": "swiftc was not found"})
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [swiftc, str(source), "-O", "-o", str(output)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    report.update(
        {
            "status": "BUILT" if int(completed.returncode) == 0 else "FAILED",
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "output_present": output.exists(),
        }
    )
    if int(completed.returncode) == 0:
        try:
            output.chmod(0o755)
        except OSError:
            pass
    return report


def build_managed_vm_virtualization_runner(
    *,
    state_root: str = "",
    source_path: str = "",
    output_path: str = "",
    swiftc_path: str = "",
    clang_path: str = "",
    codesign_path: str = "",
    entitlements_path: str = "",
    skip_codesign: bool = False,
) -> Dict[str, Any]:
    """Build the Apple Virtualization long-running VM process launcher."""

    source = Path(source_path or managed_vm_runner_source_path()).expanduser().resolve()
    output = Path(output_path or managed_vm_runner_build_output_path(state_root)).expanduser().resolve()
    entitlements = Path(entitlements_path or managed_vm_runner_entitlements_path()).expanduser().resolve()
    swiftc = _clean(swiftc_path) or _clean(shutil.which("swiftc"))
    clang = _clean(clang_path) or _clean(shutil.which("clang"))
    codesign = _clean(codesign_path) or _clean(shutil.which("codesign"))
    use_objc = source.suffix == ".m"
    compiler = clang if use_objc else swiftc
    report: Dict[str, Any] = {
        "schema_version": MANAGED_VM_PROVIDER_VERSION,
        "operation": "build_virtualization_runner",
        "status": "NOT_RUN",
        "platform": sys.platform,
        "source_path": str(source),
        "source_present": source.exists(),
        "output_path": str(output),
        "swiftc_path": swiftc,
        "clang_path": clang,
        "codesign_path": codesign,
        "entitlements_path": str(entitlements),
        "entitlements_present": entitlements.exists(),
        "skip_codesign": bool(skip_codesign),
        "source_language": "objective-c" if use_objc else "swift",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "codesign_returncode": None,
        "codesign_stdout": "",
        "codesign_stderr": "",
        "codesigned": False,
    }
    if sys.platform != "darwin":
        report.update({"status": "UNAVAILABLE", "reason": "Apple Virtualization runner build is macOS-only"})
        return report
    if not source.exists():
        report.update({"status": "UNAVAILABLE", "reason": "Apple Virtualization runner source is missing"})
        return report
    if not compiler:
        report.update({"status": "UNAVAILABLE", "reason": "clang was not found" if use_objc else "swiftc was not found"})
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    if use_objc:
        command = [
            compiler,
            "-fobjc-arc",
            "-framework",
            "Foundation",
            "-framework",
            "Virtualization",
            str(source),
            "-o",
            str(output),
        ]
    else:
        command = [compiler, str(source), "-O", "-o", str(output)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    report.update(
        {
            "status": "COMPILED" if int(completed.returncode) == 0 else "FAILED",
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": str(completed.stdout or ""),
            "stderr": str(completed.stderr or ""),
            "output_present": output.exists(),
        }
    )
    if int(completed.returncode) != 0:
        return report
    try:
        output.chmod(0o755)
    except OSError:
        pass
    if skip_codesign:
        report.update({"status": "BUILT_UNSIGNED", "reason": "codesign skipped by caller"})
        return report
    if not entitlements.exists():
        report.update({"status": "BUILT_UNSIGNED", "reason": "runner entitlements file is missing"})
        return report
    if not codesign:
        report.update({"status": "BUILT_UNSIGNED", "reason": "codesign was not found"})
        return report
    codesign_command = [
        codesign,
        "--force",
        "--sign",
        "-",
        "--entitlements",
        str(entitlements),
        str(output),
    ]
    signed = subprocess.run(
        codesign_command,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    report.update(
        {
            "codesign_command": codesign_command,
            "codesign_returncode": int(signed.returncode),
            "codesign_stdout": str(signed.stdout or ""),
            "codesign_stderr": str(signed.stderr or ""),
            "codesigned": int(signed.returncode) == 0,
            "status": "BUILT" if int(signed.returncode) == 0 else "FAILED_SIGNING",
        }
    )
    return report


def build_managed_vm_exec_command(
    command: Sequence[str],
    *,
    state_root: str = "",
    helper_path: str = "",
    image_id: str = "",
    instance_id: str = "",
    network_mode: str = "provider_default",
) -> tuple[list[str], list[str], ManagedVMConfig]:
    config = managed_vm_config(
        state_root=state_root,
        helper_path=helper_path,
        image_id=image_id,
        instance_id=instance_id,
    )
    if not config.helper_path:
        raise FileNotFoundError(
            "managed VM helper was not found; install or bundle conos-managed-vm before using managed-vm"
        )
    cmd = [str(part) for part in command]
    actual = [
        config.helper_path,
        "exec",
        "--state-root",
        config.state_root,
        "--instance-id",
        config.instance_id,
        "--image-id",
        config.image_id,
        "--network-mode",
        str(network_mode or "provider_default"),
        "--",
        *cmd,
    ]
    return actual, list(actual), config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conos vm", description="Manage the built-in Con OS managed VM provider.")
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser("report", help="Report managed VM provider availability.")
    init_parser = subparsers.add_parser("init", help="Create the managed VM state directory and manifest.")
    status_parser = subparsers.add_parser("status", help="Alias for report.")
    setup_plan_parser = subparsers.add_parser("setup-plan", help="Plan the steps required to make the VM execution boundary ready.")
    doctor_parser = subparsers.add_parser("doctor", help="Alias for setup-plan.")
    setup_default_parser = subparsers.add_parser(
        "setup-default",
        help="Prepare the built-in VM as the default execution boundary; dry-run unless --execute is set.",
    )
    build_parser = subparsers.add_parser("build-helper", help="Build the bundled macOS managed VM helper.")
    build_runner_parser = subparsers.add_parser("build-runner", help="Build the Apple Virtualization VM process launcher.")
    build_guest_initrd_parser = subparsers.add_parser(
        "build-guest-initrd",
        help="Build a Con OS guest-agent initrd boot bundle.",
    )
    build_cloud_init_seed_parser = subparsers.add_parser(
        "build-cloud-init-seed",
        help="Build a Con OS cloud-init NoCloud seed disk.",
    )
    build_base_image_parser = subparsers.add_parser(
        "build-base-image",
        help="Build and register a Con OS managed Linux base image from boot artifacts.",
    )
    bootstrap_image_parser = subparsers.add_parser(
        "bootstrap-image",
        help="Build, start, and verify a Con OS managed Linux image when boot artifacts exist.",
    )
    install_default_image_parser = subparsers.add_parser(
        "install-default-image",
        help="Install the default Con OS managed VM image from the built-in digest-pinned recipe.",
    )
    bundle_base_image_parser = subparsers.add_parser(
        "bundle-base-image",
        help="Create a self-contained digest-pinned Con OS base-image bundle.",
    )
    install_bundle_parser = subparsers.add_parser(
        "install-base-image-bundle",
        help="Install a self-contained Con OS base-image bundle into local VM state.",
    )
    import_bundle_parser = subparsers.add_parser(
        "import-base-image-bundle",
        help="Alias for install-base-image-bundle.",
    )
    resolve_recipe_parser = subparsers.add_parser(
        "resolve-artifact-recipe",
        help="Resolve a managed VM artifact recipe into the local cache.",
    )
    pin_recipe_parser = subparsers.add_parser(
        "pin-artifact-recipe",
        help="Create a READY managed VM artifact recipe from digest-pinned artifacts.",
    )
    recipe_report_parser = subparsers.add_parser("recipe-report", help="List built-in managed VM artifact recipes.")
    blank_image_parser = subparsers.add_parser("create-blank-image", help="Create a Con OS-owned blank disk artifact.")
    register_parser = subparsers.add_parser("register-image", help="Copy a base disk image into Con OS VM state.")
    register_cloud_init_parser = subparsers.add_parser(
        "register-cloud-init-image",
        help="Register an EFI cloud image that receives a Con OS NoCloud seed at start.",
    )
    register_linux_parser = subparsers.add_parser("register-linux-boot-image", help="Register a Linux disk plus kernel/initrd boot artifacts.")
    prepare_parser = subparsers.add_parser("prepare-instance", help="Prepare a managed VM instance manifest.")
    boot_parser = subparsers.add_parser("boot-instance", help="Prepare the managed VM boot boundary with the helper.")
    start_parser = subparsers.add_parser("start-instance", help="Start or attempt to start a managed VM instance.")
    ensure_parser = subparsers.add_parser(
        "ensure-running",
        help="Idempotently start a managed VM instance and wait for guest execution readiness.",
    )
    health_parser = subparsers.add_parser("health-check", help="Check and repair managed VM lifecycle health metadata.")
    recover_parser = subparsers.add_parser("recover-instance", help="Recover a managed VM instance from stopped or stale state.")
    recovery_drill_parser = subparsers.add_parser(
        "recovery-drill",
        help="Inject a runner crash, recover the instance, and verify guest execution.",
    )
    recovery_soak_parser = subparsers.add_parser(
        "recovery-soak",
        help="Run repeated VM recovery drills and summarize recovery stability.",
    )
    runtime_status_parser = subparsers.add_parser("runtime-status", help="Report managed VM runtime lifecycle status.")
    stop_parser = subparsers.add_parser("stop-instance", help="Stop a managed VM instance.")
    agent_status_parser = subparsers.add_parser("agent-status", help="Report managed VM guest-agent readiness.")
    agent_exec_parser = subparsers.add_parser("agent-exec", help="Execute through the managed VM guest-agent gate.")
    image_report_parser = subparsers.add_parser("image-report", help="Report a managed VM base image manifest.")
    instance_report_parser = subparsers.add_parser("instance-report", help="Report a managed VM instance manifest.")
    for child in (report_parser, init_parser, status_parser, setup_plan_parser, doctor_parser):
        child.add_argument("--state-root", default="")
        child.add_argument("--helper-path", default="")
        child.add_argument("--image-id", default="")
        child.add_argument("--instance-id", default="")
    setup_default_parser.add_argument("--state-root", default="")
    setup_default_parser.add_argument("--helper-path", default="")
    setup_default_parser.add_argument("--runner-path", default="")
    setup_default_parser.add_argument("--image-id", default="")
    setup_default_parser.add_argument("--instance-id", default="")
    setup_default_parser.add_argument("--network-mode", default="provider_default")
    setup_default_parser.add_argument("--execute", action="store_true")
    setup_default_parser.add_argument("--allow-artifact-download", action="store_true")
    setup_default_parser.add_argument("--artifact-timeout-seconds", type=int, default=120)
    setup_default_parser.add_argument("--timeout-seconds", type=int, default=120)
    setup_default_parser.add_argument("--startup-wait-seconds", type=float, default=DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS)
    setup_default_parser.add_argument("--guest-wait-seconds", type=float, default=180.0)
    setup_default_parser.add_argument("--agent-timeout-seconds", type=int, default=30)
    setup_default_parser.add_argument("--no-verify-agent-exec", action="store_true")
    setup_default_parser.add_argument("--overwrite", action="store_true")
    setup_default_parser.add_argument("--no-write-audit", action="store_true")
    build_parser.add_argument("--state-root", default="")
    build_parser.add_argument("--source-path", default="")
    build_parser.add_argument("--output-path", default="")
    build_parser.add_argument("--swiftc-path", default="")
    build_runner_parser.add_argument("--state-root", default="")
    build_runner_parser.add_argument("--source-path", default="")
    build_runner_parser.add_argument("--output-path", default="")
    build_runner_parser.add_argument("--swiftc-path", default="")
    build_runner_parser.add_argument("--clang-path", default="")
    build_runner_parser.add_argument("--codesign-path", default="")
    build_runner_parser.add_argument("--entitlements-path", default="")
    build_runner_parser.add_argument("--skip-codesign", action="store_true")
    build_guest_initrd_parser.add_argument("--state-root", default="")
    build_guest_initrd_parser.add_argument("--output-path", default="")
    build_guest_initrd_parser.add_argument("--base-initrd-path", default="")
    build_guest_initrd_parser.add_argument("--guest-agent-path", default="")
    build_guest_initrd_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    build_guest_initrd_parser.add_argument("--root-device", default="/dev/vda")
    build_guest_initrd_parser.add_argument("--guest-python-path", default="/usr/bin/python3")
    build_guest_initrd_parser.add_argument("--include-init-wrapper", action=argparse.BooleanOptionalAction, default=True)
    build_guest_initrd_parser.add_argument("--overwrite", action="store_true")
    build_cloud_init_seed_parser.add_argument("--state-root", default="")
    build_cloud_init_seed_parser.add_argument("--instance-id", default="")
    build_cloud_init_seed_parser.add_argument("--output-path", default="")
    build_cloud_init_seed_parser.add_argument("--guest-agent-path", default="")
    build_cloud_init_seed_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    build_cloud_init_seed_parser.add_argument("--guest-python-path", default="/usr/bin/python3")
    build_cloud_init_seed_parser.add_argument("--hostname", default="")
    build_cloud_init_seed_parser.add_argument("--network-config", default="")
    build_cloud_init_seed_parser.add_argument("--size-mb", type=int, default=DEFAULT_MANAGED_VM_CLOUD_INIT_SEED_SIZE_MB)
    build_cloud_init_seed_parser.add_argument("--overwrite", action="store_true")
    build_base_image_parser.add_argument("--state-root", default="")
    build_base_image_parser.add_argument("--image-id", default="")
    build_base_image_parser.add_argument("--source-disk", default="")
    build_base_image_parser.add_argument("--kernel-path", default="")
    build_base_image_parser.add_argument("--base-initrd-path", default="")
    build_base_image_parser.add_argument("--guest-agent-path", default="")
    build_base_image_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    build_base_image_parser.add_argument("--kernel-command-line", default="")
    build_base_image_parser.add_argument("--root-device", default="/dev/vda")
    build_base_image_parser.add_argument("--guest-python-path", default="/usr/bin/python3")
    build_base_image_parser.add_argument("--include-init-wrapper", action=argparse.BooleanOptionalAction, default=True)
    build_base_image_parser.add_argument("--overwrite", action="store_true")
    bootstrap_image_parser.add_argument("--state-root", default="")
    bootstrap_image_parser.add_argument("--image-id", default="")
    bootstrap_image_parser.add_argument("--instance-id", default="bootstrap-smoke")
    bootstrap_image_parser.add_argument("--source-disk", default="")
    bootstrap_image_parser.add_argument("--kernel-path", default="")
    bootstrap_image_parser.add_argument("--base-initrd-path", default="")
    bootstrap_image_parser.add_argument("--recipe-path", default="")
    bootstrap_image_parser.add_argument("--allow-artifact-download", action=argparse.BooleanOptionalAction, default=True)
    bootstrap_image_parser.add_argument("--artifact-timeout-seconds", type=int, default=120)
    bootstrap_image_parser.add_argument("--guest-agent-path", default="")
    bootstrap_image_parser.add_argument("--runner-path", default="")
    bootstrap_image_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    bootstrap_image_parser.add_argument("--kernel-command-line", default="")
    bootstrap_image_parser.add_argument("--root-device", default="/dev/vda")
    bootstrap_image_parser.add_argument("--guest-python-path", default="/usr/bin/python3")
    bootstrap_image_parser.add_argument("--network-mode", default="provider_default")
    bootstrap_image_parser.add_argument("--build-runner", action=argparse.BooleanOptionalAction, default=True)
    bootstrap_image_parser.add_argument("--start-instance", action=argparse.BooleanOptionalAction, default=True)
    bootstrap_image_parser.add_argument("--verify-agent-exec", action=argparse.BooleanOptionalAction, default=True)
    bootstrap_image_parser.add_argument("--keep-running", action="store_true")
    bootstrap_image_parser.add_argument("--startup-wait-seconds", type=float, default=DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS)
    bootstrap_image_parser.add_argument("--guest-wait-seconds", type=float, default=60.0)
    bootstrap_image_parser.add_argument("--agent-timeout-seconds", type=int, default=30)
    bootstrap_image_parser.add_argument("--overwrite", action="store_true")
    install_default_image_parser.add_argument("--state-root", default="")
    install_default_image_parser.add_argument("--image-id", default="")
    install_default_image_parser.add_argument("--instance-id", default="bootstrap-smoke")
    install_default_image_parser.add_argument("--recipe-path", default=DEFAULT_MANAGED_VM_RECIPE_REFERENCE)
    install_default_image_parser.add_argument("--allow-artifact-download", action=argparse.BooleanOptionalAction, default=True)
    install_default_image_parser.add_argument("--artifact-timeout-seconds", type=int, default=120)
    install_default_image_parser.add_argument("--guest-agent-path", default="")
    install_default_image_parser.add_argument("--runner-path", default="")
    install_default_image_parser.add_argument("--network-mode", default="provider_default")
    install_default_image_parser.add_argument("--build-runner", action=argparse.BooleanOptionalAction, default=True)
    install_default_image_parser.add_argument("--start-instance", action=argparse.BooleanOptionalAction, default=True)
    install_default_image_parser.add_argument("--verify-agent-exec", action=argparse.BooleanOptionalAction, default=True)
    install_default_image_parser.add_argument("--keep-running", action="store_true")
    install_default_image_parser.add_argument("--startup-wait-seconds", type=float, default=DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS)
    install_default_image_parser.add_argument("--guest-wait-seconds", type=float, default=60.0)
    install_default_image_parser.add_argument("--agent-timeout-seconds", type=int, default=30)
    install_default_image_parser.add_argument("--overwrite", action="store_true")
    bundle_base_image_parser.add_argument("--state-root", default="")
    bundle_base_image_parser.add_argument("--image-id", default="")
    bundle_base_image_parser.add_argument("--output-dir", default="")
    bundle_base_image_parser.add_argument("--allow-unverified", action="store_true")
    bundle_base_image_parser.add_argument("--overwrite", action="store_true")
    for child in (install_bundle_parser, import_bundle_parser):
        child.add_argument("--state-root", default="")
        child.add_argument("--image-id", default="")
        child.add_argument("--bundle-dir", required=True)
        child.add_argument("--allow-unverified", action="store_true")
        child.add_argument("--overwrite", action="store_true")
    resolve_recipe_parser.add_argument("--state-root", default="")
    resolve_recipe_parser.add_argument("--recipe-path", required=True)
    resolve_recipe_parser.add_argument("--allow-download", action=argparse.BooleanOptionalAction, default=True)
    resolve_recipe_parser.add_argument("--timeout-seconds", type=int, default=120)
    pin_recipe_parser.add_argument("--state-root", default="")
    pin_recipe_parser.add_argument("--base-recipe", required=True)
    pin_recipe_parser.add_argument("--output-path", default="")
    pin_recipe_parser.add_argument("--recipe-id", default="")
    pin_recipe_parser.add_argument("--image-id", default="")
    pin_recipe_parser.add_argument("--source-disk", required=True)
    pin_recipe_parser.add_argument("--source-disk-sha256", default="")
    pin_recipe_parser.add_argument("--source-disk-sha512", default="")
    pin_recipe_parser.add_argument("--source-disk-filename", default="")
    pin_recipe_parser.add_argument("--kernel", default="")
    pin_recipe_parser.add_argument("--kernel-sha256", default="")
    pin_recipe_parser.add_argument("--kernel-sha512", default="")
    pin_recipe_parser.add_argument("--kernel-filename", default="")
    pin_recipe_parser.add_argument("--base-initrd", default="")
    pin_recipe_parser.add_argument("--base-initrd-sha256", default="")
    pin_recipe_parser.add_argument("--base-initrd-sha512", default="")
    pin_recipe_parser.add_argument("--base-initrd-filename", default="")
    pin_recipe_parser.add_argument("--overwrite", action="store_true")
    recipe_report_parser.add_argument("--recipe-id", default="")
    blank_image_parser.add_argument("--state-root", default="")
    blank_image_parser.add_argument("--image-id", default="")
    blank_image_parser.add_argument("--size-mb", type=int, default=DEFAULT_MANAGED_VM_BLANK_IMAGE_SIZE_MB)
    blank_image_parser.add_argument("--overwrite", action="store_true")
    register_parser.add_argument("--state-root", default="")
    register_parser.add_argument("--image-id", default="")
    register_parser.add_argument("--source-disk", required=True)
    register_cloud_init_parser.add_argument("--state-root", default="")
    register_cloud_init_parser.add_argument("--image-id", default="")
    register_cloud_init_parser.add_argument("--source-disk", required=True)
    register_cloud_init_parser.add_argument("--guest-agent-path", default="")
    register_cloud_init_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    register_cloud_init_parser.add_argument("--guest-python-path", default="/usr/bin/python3")
    register_linux_parser.add_argument("--state-root", default="")
    register_linux_parser.add_argument("--image-id", default="")
    register_linux_parser.add_argument("--source-disk", required=True)
    register_linux_parser.add_argument("--kernel-path", required=True)
    register_linux_parser.add_argument("--initrd-path", default="")
    register_linux_parser.add_argument("--kernel-command-line", default="")
    register_linux_parser.add_argument("--guest-agent-port", type=int, default=DEFAULT_MANAGED_VM_AGENT_VSOCK_PORT)
    prepare_parser.add_argument("--state-root", default="")
    prepare_parser.add_argument("--image-id", default="")
    prepare_parser.add_argument("--instance-id", default="")
    prepare_parser.add_argument("--network-mode", default="provider_default")
    boot_parser.add_argument("--state-root", default="")
    boot_parser.add_argument("--helper-path", default="")
    boot_parser.add_argument("--image-id", default="")
    boot_parser.add_argument("--instance-id", default="")
    boot_parser.add_argument("--network-mode", default="provider_default")
    boot_parser.add_argument("--timeout-seconds", type=int, default=120)
    for child in (start_parser, ensure_parser, health_parser, recover_parser, recovery_drill_parser, recovery_soak_parser, runtime_status_parser, stop_parser):
        child.add_argument("--state-root", default="")
        child.add_argument("--helper-path", default="")
        child.add_argument("--image-id", default="")
        child.add_argument("--instance-id", default="")
        child.add_argument("--network-mode", default="provider_default")
        child.add_argument("--timeout-seconds", type=int, default=120)
    for child in (start_parser, ensure_parser, recover_parser, recovery_drill_parser, recovery_soak_parser):
        child.add_argument("--runner-path", default="")
        child.add_argument("--startup-wait-seconds", type=float, default=DEFAULT_MANAGED_VM_STARTUP_WAIT_SECONDS)
        child.add_argument("--no-build-runner", action="store_true")
        child.add_argument("--guest-wait-seconds", type=float, default=180.0)
    recover_parser.add_argument("--restart-unready", action="store_true")
    recovery_drill_parser.add_argument("--skip-initial-ensure", action="store_true")
    recovery_drill_parser.add_argument("--crash-signal", choices=("SIGKILL", "SIGTERM"), default="SIGKILL")
    recovery_drill_parser.add_argument("--no-verify-agent-exec", action="store_true")
    recovery_drill_parser.add_argument("--agent-timeout-seconds", type=int, default=30)
    recovery_soak_parser.add_argument("--rounds", type=int, default=3)
    recovery_soak_parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    recovery_soak_parser.add_argument("--crash-signal", choices=("SIGKILL", "SIGTERM"), default="SIGKILL")
    recovery_soak_parser.add_argument("--no-verify-agent-exec", action="store_true")
    recovery_soak_parser.add_argument("--agent-timeout-seconds", type=int, default=30)
    recovery_soak_parser.add_argument("--no-disk-probe", action="store_true")
    recovery_soak_parser.add_argument("--report-path", default="")
    for child in (agent_status_parser, agent_exec_parser):
        child.add_argument("--state-root", default="")
        child.add_argument("--helper-path", default="")
        child.add_argument("--image-id", default="")
        child.add_argument("--instance-id", default="")
        child.add_argument("--network-mode", default="provider_default")
        child.add_argument("--timeout-seconds", type=int, default=30)
    agent_exec_parser.add_argument("agent_args", nargs=argparse.REMAINDER)
    image_report_parser.add_argument("--state-root", default="")
    image_report_parser.add_argument("--image-id", default="")
    instance_report_parser.add_argument("--state-root", default="")
    instance_report_parser.add_argument("--instance-id", default="")

    args = parser.parse_args(list(argv) if argv is not None else None)
    command = str(args.command or "report")
    if command == "init":
        payload = init_managed_vm_state(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
        )
    elif command in {"report", "status"}:
        payload = managed_vm_report(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
        )
    elif command in {"setup-plan", "doctor"}:
        payload = managed_vm_setup_plan(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
        )
    elif command == "setup-default":
        payload = managed_vm_prepare_default_boundary(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            execute=bool(args.execute),
            allow_artifact_download=bool(args.allow_artifact_download),
            artifact_timeout_seconds=int(args.artifact_timeout_seconds),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            agent_timeout_seconds=int(args.agent_timeout_seconds),
            verify_agent_exec=not bool(args.no_verify_agent_exec),
            overwrite=bool(args.overwrite),
            write_audit=not bool(args.no_write_audit),
        )
    elif command == "build-helper":
        payload = build_managed_vm_helper(
            state_root=str(args.state_root),
            source_path=str(args.source_path),
            output_path=str(args.output_path),
            swiftc_path=str(args.swiftc_path),
        )
    elif command == "build-runner":
        payload = build_managed_vm_virtualization_runner(
            state_root=str(args.state_root),
            source_path=str(args.source_path),
            output_path=str(args.output_path),
            swiftc_path=str(args.swiftc_path),
            clang_path=str(args.clang_path),
            codesign_path=str(args.codesign_path),
            entitlements_path=str(args.entitlements_path),
            skip_codesign=bool(args.skip_codesign),
        )
    elif command == "build-guest-initrd":
        payload = build_managed_vm_guest_initrd_bundle(
            state_root=str(args.state_root),
            output_path=str(args.output_path),
            base_initrd_path=str(args.base_initrd_path),
            guest_agent_path=str(args.guest_agent_path),
            guest_agent_port=int(args.guest_agent_port),
            root_device=str(args.root_device),
            guest_python_path=str(args.guest_python_path),
            include_init_wrapper=bool(args.include_init_wrapper),
            overwrite=bool(args.overwrite),
        )
    elif command == "build-cloud-init-seed":
        payload = build_managed_vm_cloud_init_seed(
            state_root=str(args.state_root),
            instance_id=str(args.instance_id),
            output_path=str(args.output_path),
            guest_agent_path=str(args.guest_agent_path),
            guest_agent_port=int(args.guest_agent_port),
            guest_python_path=str(args.guest_python_path),
            hostname=str(args.hostname),
            network_config=str(args.network_config),
            overwrite=bool(args.overwrite),
            size_mb=int(args.size_mb),
        )
    elif command == "build-base-image":
        payload = build_managed_vm_linux_base_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            source_disk_path=str(args.source_disk),
            kernel_path=str(args.kernel_path),
            base_initrd_path=str(args.base_initrd_path),
            guest_agent_path=str(args.guest_agent_path),
            guest_agent_port=int(args.guest_agent_port),
            kernel_command_line=str(args.kernel_command_line),
            root_device=str(args.root_device),
            guest_python_path=str(args.guest_python_path),
            include_init_wrapper=bool(args.include_init_wrapper),
            overwrite=bool(args.overwrite),
        )
    elif command == "bootstrap-image":
        payload = bootstrap_managed_vm_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            source_disk_path=str(args.source_disk),
            kernel_path=str(args.kernel_path),
            base_initrd_path=str(args.base_initrd_path),
            recipe_path=str(args.recipe_path),
            allow_artifact_download=bool(args.allow_artifact_download),
            artifact_timeout_seconds=int(args.artifact_timeout_seconds),
            guest_agent_path=str(args.guest_agent_path),
            runner_path=str(args.runner_path),
            guest_agent_port=int(args.guest_agent_port),
            kernel_command_line=str(args.kernel_command_line),
            root_device=str(args.root_device),
            guest_python_path=str(args.guest_python_path),
            network_mode=str(args.network_mode),
            build_runner=bool(args.build_runner),
            start_instance=bool(args.start_instance),
            verify_agent_exec=bool(args.verify_agent_exec),
            keep_running=bool(args.keep_running),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            agent_timeout_seconds=int(args.agent_timeout_seconds),
            overwrite=bool(args.overwrite),
        )
    elif command == "install-default-image":
        payload = install_default_managed_vm_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            recipe_path=str(args.recipe_path),
            allow_artifact_download=bool(args.allow_artifact_download),
            artifact_timeout_seconds=int(args.artifact_timeout_seconds),
            guest_agent_path=str(args.guest_agent_path),
            runner_path=str(args.runner_path),
            network_mode=str(args.network_mode),
            build_runner=bool(args.build_runner),
            start_instance=bool(args.start_instance),
            verify_agent_exec=bool(args.verify_agent_exec),
            keep_running=bool(args.keep_running),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            agent_timeout_seconds=int(args.agent_timeout_seconds),
            overwrite=bool(args.overwrite),
        )
    elif command == "bundle-base-image":
        payload = create_managed_vm_base_image_bundle(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            output_dir=str(args.output_dir),
            allow_unverified=bool(args.allow_unverified),
            overwrite=bool(args.overwrite),
        )
    elif command in {"install-base-image-bundle", "import-base-image-bundle"}:
        payload = install_managed_vm_base_image_bundle(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            bundle_dir=str(args.bundle_dir),
            allow_unverified=bool(args.allow_unverified),
            overwrite=bool(args.overwrite),
        )
    elif command == "resolve-artifact-recipe":
        payload = resolve_managed_vm_artifact_recipe(
            recipe_path=str(args.recipe_path),
            state_root=str(args.state_root),
            allow_download=bool(args.allow_download),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "pin-artifact-recipe":
        payload = create_managed_vm_pinned_artifact_recipe(
            base_recipe_path=str(args.base_recipe),
            output_path=str(args.output_path),
            state_root=str(args.state_root),
            recipe_id=str(args.recipe_id),
            image_id=str(args.image_id),
            source_disk=str(args.source_disk),
            source_disk_sha256=str(args.source_disk_sha256),
            source_disk_sha512=str(args.source_disk_sha512),
            source_disk_filename=str(args.source_disk_filename),
            kernel=str(args.kernel),
            kernel_sha256=str(args.kernel_sha256),
            kernel_sha512=str(args.kernel_sha512),
            kernel_filename=str(args.kernel_filename),
            base_initrd=str(args.base_initrd),
            base_initrd_sha256=str(args.base_initrd_sha256),
            base_initrd_sha512=str(args.base_initrd_sha512),
            base_initrd_filename=str(args.base_initrd_filename),
            overwrite=bool(args.overwrite),
        )
    elif command == "recipe-report":
        payload = managed_vm_recipe_report(recipe_id=str(args.recipe_id))
    elif command == "create-blank-image":
        payload = create_managed_vm_blank_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            size_mb=int(args.size_mb),
            overwrite=bool(args.overwrite),
        )
    elif command == "register-image":
        payload = register_managed_vm_base_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            source_disk_path=str(args.source_disk),
        )
    elif command == "register-cloud-init-image":
        payload = register_managed_vm_cloud_init_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            source_disk_path=str(args.source_disk),
            guest_agent_path=str(args.guest_agent_path),
            guest_agent_port=int(args.guest_agent_port),
            guest_python_path=str(args.guest_python_path),
        )
    elif command == "register-linux-boot-image":
        payload = register_managed_vm_linux_boot_image(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            source_disk_path=str(args.source_disk),
            kernel_path=str(args.kernel_path),
            initrd_path=str(args.initrd_path),
            kernel_command_line=str(args.kernel_command_line),
            guest_agent_port=int(args.guest_agent_port),
        )
    elif command == "prepare-instance":
        payload = prepare_managed_vm_instance(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
        )
    elif command == "boot-instance":
        payload = boot_managed_vm_instance(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "start-instance":
        payload = start_managed_vm_instance(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            auto_build_runner=not bool(args.no_build_runner),
        )
    elif command == "ensure-running":
        payload = ensure_managed_vm_instance_running(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            auto_build_runner=not bool(args.no_build_runner),
        )
    elif command == "health-check":
        payload = managed_vm_health_check(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "recover-instance":
        payload = recover_managed_vm_instance(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            auto_build_runner=not bool(args.no_build_runner),
            restart_unready=bool(args.restart_unready),
        )
    elif command == "recovery-drill":
        payload = managed_vm_recovery_drill(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            auto_build_runner=not bool(args.no_build_runner),
            ensure_ready_first=not bool(args.skip_initial_ensure),
            crash_signal=str(args.crash_signal),
            verify_agent_exec=not bool(args.no_verify_agent_exec),
            agent_timeout_seconds=int(args.agent_timeout_seconds),
        )
    elif command == "recovery-soak":
        payload = managed_vm_recovery_soak(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            runner_path=str(args.runner_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
            startup_wait_seconds=float(args.startup_wait_seconds),
            guest_wait_seconds=float(args.guest_wait_seconds),
            auto_build_runner=not bool(args.no_build_runner),
            rounds=int(args.rounds),
            cooldown_seconds=float(args.cooldown_seconds),
            crash_signal=str(args.crash_signal),
            verify_agent_exec=not bool(args.no_verify_agent_exec),
            agent_timeout_seconds=int(args.agent_timeout_seconds),
            disk_probe=not bool(args.no_disk_probe),
            report_path=str(args.report_path),
        )
    elif command == "runtime-status":
        payload = managed_vm_runtime_status(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "stop-instance":
        payload = stop_managed_vm_instance(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "agent-status":
        payload = managed_vm_guest_agent_status(
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "agent-exec":
        agent_args = list(getattr(args, "agent_args", []))
        if agent_args and agent_args[0] == "--":
            agent_args = agent_args[1:]
        payload = run_managed_vm_agent_command(
            agent_args,
            state_root=str(args.state_root),
            helper_path=str(args.helper_path),
            image_id=str(args.image_id),
            instance_id=str(args.instance_id),
            network_mode=str(args.network_mode),
            timeout_seconds=int(args.timeout_seconds),
        )
    elif command == "image-report":
        payload = load_managed_vm_image_manifest(
            state_root=str(args.state_root),
            image_id=str(args.image_id),
        )
    elif command == "instance-report":
        payload = load_managed_vm_instance_manifest(
            state_root=str(args.state_root),
            instance_id=str(args.instance_id),
        )
    else:
        parser.print_help()
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
