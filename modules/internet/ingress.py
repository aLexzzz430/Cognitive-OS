from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen
import zipfile


INTERNET_INGRESS_VERSION = "conos.internet_ingress/v1"
INTERNET_INGRESS_MANIFEST_VERSION = "conos.internet_ingress_manifest/v1"


class InternetIngressError(ValueError):
    """Raised when a URL or fetch violates the configured ingress policy."""


@dataclass(frozen=True)
class InternetIngressPolicy:
    allowed_schemes: tuple[str, ...] = ("http", "https")
    allowed_hosts: tuple[str, ...] = ()
    blocked_hosts: tuple[str, ...] = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    blocked_host_suffixes: tuple[str, ...] = (".localhost", ".local")
    allow_private_networks: bool = False
    max_bytes: int = 2 * 1024 * 1024
    timeout_seconds: float = 20.0
    user_agent: str = "ConOS-InternetIngress/0.1"


@dataclass
class InternetArtifact:
    artifact_id: str
    source_url: str
    normalized_url: str
    fetch_kind: str
    local_path: str
    bytes_written: int
    sha256: str
    content_type: str
    status: str
    error: str = ""
    fetched_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = INTERNET_INGRESS_VERSION
        return payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_host(host: str) -> str:
    return str(host or "").strip().lower().rstrip(".")


def _host_for_netloc(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def normalize_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise InternetIngressError("url must include scheme and host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise InternetIngressError(f"invalid url port: {exc}") from exc
    host = _canonical_host(parsed.hostname or "")
    if not host:
        raise InternetIngressError("url must include host")
    netloc = _host_for_netloc(host)
    if port is not None:
        netloc = f"{netloc}:{port}"
    path = parsed.path or "/"
    return urlunparse((parsed.scheme.lower(), netloc, path, "", parsed.query, ""))


def validate_url(url: str, policy: InternetIngressPolicy | None = None) -> str:
    policy = policy or InternetIngressPolicy()
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    scheme = parsed.scheme.lower()
    if scheme not in {item.lower() for item in policy.allowed_schemes}:
        raise InternetIngressError(f"url scheme is not allowed: {scheme}")
    original = urlparse(str(url or "").strip())
    if original.username or original.password:
        raise InternetIngressError("url credentials are not allowed")
    host = _canonical_host(parsed.hostname or "")
    if not host:
        raise InternetIngressError("url must include host")
    allowed_hosts = {_canonical_host(item) for item in policy.allowed_hosts if str(item).strip()}
    if allowed_hosts and host not in allowed_hosts:
        raise InternetIngressError(f"url host is not in allowed_hosts: {host}")
    blocked_hosts = {_canonical_host(item) for item in policy.blocked_hosts if str(item).strip()}
    if host in blocked_hosts:
        raise InternetIngressError(f"url host is blocked: {host}")
    for suffix in policy.blocked_host_suffixes:
        normalized_suffix = str(suffix or "").strip().lower()
        if normalized_suffix and host.endswith(normalized_suffix):
            raise InternetIngressError(f"url host suffix is blocked: {normalized_suffix}")
    if not policy.allow_private_networks:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            if not ip.is_global:
                raise InternetIngressError(f"url host is not public/global: {host}")
    return normalized


def _safe_filename(url: str, content_type: str, filename: str | None) -> str:
    requested = str(filename or "").strip()
    if requested:
        candidate = Path(requested).name
    else:
        parsed = urlparse(url)
        candidate = Path(parsed.path).name or "index"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-") or "artifact"
    if "." not in candidate:
        if "html" in content_type:
            candidate = f"{candidate}.html"
        elif "json" in content_type:
            candidate = f"{candidate}.json"
        elif content_type.startswith("text/"):
            candidate = f"{candidate}.txt"
        else:
            candidate = f"{candidate}.bin"
    return candidate[:120]


def _safe_directory_name(url: str, directory_name: str | None) -> str:
    requested = str(directory_name or "").strip()
    if requested:
        candidate = Path(requested).name
    else:
        parsed = urlparse(url)
        candidate = Path(parsed.path.rstrip("/")).name or _canonical_host(parsed.hostname or "project")
    if candidate.endswith(".git"):
        candidate = candidate[:-4]
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".zip", ".tar"):
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            break
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-") or "project"
    return candidate[:100]


def _is_within_directory(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _directory_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(root).parts:
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        total_bytes += len(data)
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def _manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.json"


def _events_path(output_root: Path) -> Path:
    return output_root / "events.jsonl"


def load_manifest(output_root: str | Path) -> dict[str, Any]:
    path = _manifest_path(Path(output_root))
    if not path.exists():
        return {
            "schema_version": INTERNET_INGRESS_MANIFEST_VERSION,
            "artifacts": [],
            "artifact_count": 0,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": INTERNET_INGRESS_MANIFEST_VERSION,
            "artifacts": [],
            "artifact_count": 0,
            "load_error": "manifest_unreadable",
        }
    return dict(payload) if isinstance(payload, dict) else {}


def _write_manifest(output_root: Path, artifact: InternetArtifact) -> None:
    manifest = load_manifest(output_root)
    artifacts = [dict(item) for item in list(manifest.get("artifacts", []) or []) if isinstance(item, Mapping)]
    artifacts.append(artifact.to_dict())
    payload = {
        "schema_version": INTERNET_INGRESS_MANIFEST_VERSION,
        "updated_at": _now(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _manifest_path(output_root).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _append_event(output_root: Path, event_type: str, artifact: InternetArtifact) -> None:
    event = {
        "schema_version": INTERNET_INGRESS_VERSION,
        "event_type": event_type,
        "created_at": _now(),
        "artifact": artifact.to_dict(),
    }
    with _events_path(output_root).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def _response_content_type(response: Any) -> str:
    headers = getattr(response, "headers", None)
    if headers is not None:
        get_content_type = getattr(headers, "get_content_type", None)
        if callable(get_content_type):
            value = str(get_content_type() or "").strip()
            if value:
                return value
        get = getattr(headers, "get", None)
        if callable(get):
            value = str(get("content-type", "") or get("Content-Type", "") or "").strip()
            if value:
                return value.split(";", 1)[0].strip()
    return "application/octet-stream"


def fetch_url(
    url: str,
    output_root: str | Path,
    *,
    policy: InternetIngressPolicy | None = None,
    filename: str | None = None,
) -> InternetArtifact:
    policy = policy or InternetIngressPolicy()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    normalized = validate_url(url, policy)
    fetched_at = _now()
    request = Request(
        normalized,
        headers={
            "User-Agent": policy.user_agent,
            "Accept": "*/*",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=float(policy.timeout_seconds)) as response:
            content_type = _response_content_type(response)
            content = response.read(int(policy.max_bytes) + 1)
            status_code = int(response.getcode() or 0) if hasattr(response, "getcode") else 0
    except HTTPError as exc:
        raise InternetIngressError(f"http fetch failed: {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise InternetIngressError(f"url fetch failed: {exc.reason}") from exc
    if len(content) > int(policy.max_bytes):
        raise InternetIngressError(f"download exceeds max_bytes={int(policy.max_bytes)}")
    digest = hashlib.sha256(content).hexdigest()
    safe_name = _safe_filename(normalized, content_type, filename)
    artifact_id = hashlib.sha256(f"{normalized}\n{digest}".encode("utf-8")).hexdigest()[:16]
    destination = artifacts_dir / f"{artifact_id}-{safe_name}"
    destination.write_bytes(content)
    artifact = InternetArtifact(
        artifact_id=artifact_id,
        source_url=str(url),
        normalized_url=normalized,
        fetch_kind="http",
        local_path=str(destination.resolve()),
        bytes_written=len(content),
        sha256=digest,
        content_type=content_type,
        status="fetched",
        fetched_at=fetched_at,
        metadata={
            "status_code": status_code,
            "max_bytes": int(policy.max_bytes),
            "timeout_seconds": float(policy.timeout_seconds),
            "allow_private_networks": bool(policy.allow_private_networks),
        },
    )
    _write_manifest(root, artifact)
    _append_event(root, "internet_artifact_fetched", artifact)
    return artifact


def clone_git_repository(
    url: str,
    output_root: str | Path,
    *,
    policy: InternetIngressPolicy | None = None,
    ref: str | None = None,
    depth: int = 1,
    directory_name: str | None = None,
    timeout_seconds: float | None = None,
) -> InternetArtifact:
    policy = policy or InternetIngressPolicy()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    repositories_dir = root / "repositories"
    repositories_dir.mkdir(parents=True, exist_ok=True)
    normalized = validate_url(url, policy)
    fetched_at = _now()
    safe_name = _safe_directory_name(normalized, directory_name)
    artifact_id = hashlib.sha256(f"{normalized}\n{ref or ''}\n{fetched_at}".encode("utf-8")).hexdigest()[:16]
    destination = repositories_dir / f"{artifact_id}-{safe_name}"
    command = ["git", "clone", "--depth", str(max(1, int(depth or 1)))]
    if ref:
        command.extend(["--branch", str(ref)])
    command.extend([normalized, str(destination)])
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.setdefault("GIT_ASKPASS", "echo")
    timeout = float(timeout_seconds if timeout_seconds is not None else max(float(policy.timeout_seconds), 60.0))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise InternetIngressError("git executable is not available") from exc
    except subprocess.TimeoutExpired as exc:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise InternetIngressError(f"git clone timed out after {timeout:.1f}s") from exc
    if completed.returncode != 0:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        stderr = str(completed.stderr or completed.stdout or "").strip().replace("\n", " ")[:500]
        raise InternetIngressError(f"git clone failed with returncode={completed.returncode}: {stderr}")
    tree_sha256, total_bytes, file_count = _directory_digest(destination)
    artifact = InternetArtifact(
        artifact_id=artifact_id,
        source_url=str(url),
        normalized_url=normalized,
        fetch_kind="git_clone",
        local_path=str(destination.resolve()),
        bytes_written=total_bytes,
        sha256=tree_sha256,
        content_type="application/x-git-repository",
        status="fetched",
        fetched_at=fetched_at,
        metadata={
            "ref": str(ref or ""),
            "depth": max(1, int(depth or 1)),
            "file_count": file_count,
            "timeout_seconds": timeout,
            "allow_private_networks": bool(policy.allow_private_networks),
        },
    )
    _write_manifest(root, artifact)
    _append_event(root, "internet_project_cloned", artifact)
    return artifact


def _extract_zip_archive(archive_path: Path, destination: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = destination / info.filename
            if not _is_within_directory(destination, target):
                raise InternetIngressError(f"archive member escapes destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    return extracted


def _extract_tar_archive(archive_path: Path, destination: Path) -> int:
    extracted = 0
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = destination / member.name
            if not _is_within_directory(destination, target):
                raise InternetIngressError(f"archive member escapes destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted += 1
    return extracted


def fetch_archive_project(
    url: str,
    output_root: str | Path,
    *,
    policy: InternetIngressPolicy | None = None,
    directory_name: str | None = None,
) -> InternetArtifact:
    policy = policy or InternetIngressPolicy()
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    archive_artifact = fetch_url(url, root, policy=policy)
    archive_path = Path(archive_artifact.local_path)
    projects_dir = root / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_directory_name(archive_artifact.normalized_url, directory_name)
    project_artifact_id = hashlib.sha256(f"{archive_artifact.artifact_id}\n{_now()}".encode("utf-8")).hexdigest()[:16]
    destination = projects_dir / f"{project_artifact_id}-{safe_name}"
    destination.mkdir(parents=True, exist_ok=False)
    try:
        if zipfile.is_zipfile(archive_path):
            extracted_count = _extract_zip_archive(archive_path, destination)
        elif tarfile.is_tarfile(archive_path):
            extracted_count = _extract_tar_archive(archive_path, destination)
        else:
            shutil.rmtree(destination, ignore_errors=True)
            raise InternetIngressError("archive project fetch requires zip or tar-compatible content")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    tree_sha256, total_bytes, file_count = _directory_digest(destination)
    artifact = InternetArtifact(
        artifact_id=project_artifact_id,
        source_url=archive_artifact.source_url,
        normalized_url=archive_artifact.normalized_url,
        fetch_kind="archive_extract",
        local_path=str(destination.resolve()),
        bytes_written=total_bytes,
        sha256=tree_sha256,
        content_type=archive_artifact.content_type,
        status="fetched",
        fetched_at=_now(),
        metadata={
            "archive_artifact": archive_artifact.to_dict(),
            "archive_artifact_id": archive_artifact.artifact_id,
            "extracted_file_count": extracted_count,
            "file_count": file_count,
            "allow_private_networks": bool(policy.allow_private_networks),
        },
    )
    _write_manifest(root, artifact)
    _append_event(root, "internet_project_archive_extracted", artifact)
    return artifact


def fetch_project(
    url: str,
    output_root: str | Path,
    *,
    policy: InternetIngressPolicy | None = None,
    source_type: str = "auto",
    ref: str | None = None,
    depth: int = 1,
    directory_name: str | None = None,
    timeout_seconds: float | None = None,
) -> InternetArtifact:
    normalized = validate_url(url, policy or InternetIngressPolicy())
    kind = str(source_type or "auto").strip().lower()
    if kind not in {"auto", "git", "archive"}:
        raise InternetIngressError("source_type must be one of: auto, git, archive")
    path = urlparse(normalized).path.lower()
    archive_suffixes = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")
    if kind == "archive" or (kind == "auto" and path.endswith(archive_suffixes)):
        return fetch_archive_project(
            normalized,
            output_root,
            policy=policy,
            directory_name=directory_name,
        )
    return clone_git_repository(
        normalized,
        output_root,
        policy=policy,
        ref=ref,
        depth=depth,
        directory_name=directory_name,
        timeout_seconds=timeout_seconds,
    )
