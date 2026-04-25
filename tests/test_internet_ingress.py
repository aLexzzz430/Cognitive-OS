from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from modules.internet import (
    InternetIngressError,
    InternetIngressPolicy,
    clone_git_repository,
    fetch_project,
    fetch_url,
    load_manifest,
    validate_url,
)


class _FakeHeaders:
    def __init__(self, content_type: str) -> None:
        self._content_type = content_type

    def get_content_type(self) -> str:
        return self._content_type

    def get(self, name: str, default: str = "") -> str:
        if name.lower() == "content-type":
            return self._content_type
        return default


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str = "text/html", status: int = 200) -> None:
        self._body = body
        self.headers = _FakeHeaders(content_type)
        self._status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, _: int) -> bytes:
        return self._body

    def getcode(self) -> int:
        return self._status


def test_validate_url_rejects_non_http_and_private_hosts() -> None:
    with pytest.raises(InternetIngressError):
        validate_url("file:///etc/passwd")
    with pytest.raises(InternetIngressError):
        validate_url("http://127.0.0.1:8000")
    with pytest.raises(InternetIngressError):
        validate_url("https://user:secret@example.com/path")

    assert validate_url("HTTPS://Example.COM/docs#fragment") == "https://example.com/docs"


def test_fetch_url_writes_artifact_manifest_and_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"<html>ok</html>"

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        assert timeout == 3.0
        assert getattr(request, "full_url") == "https://example.com/index.html"
        return _FakeResponse(body, content_type="text/html")

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)

    artifact = fetch_url(
        "https://example.com/index.html",
        tmp_path,
        policy=InternetIngressPolicy(max_bytes=1024, timeout_seconds=3),
    )

    assert artifact.status == "fetched"
    assert artifact.bytes_written == len(body)
    assert artifact.sha256 == hashlib.sha256(body).hexdigest()
    assert Path(artifact.local_path).read_bytes() == body
    manifest = load_manifest(tmp_path)
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["normalized_url"] == "https://example.com/index.html"
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").count("internet_artifact_fetched") == 1


def test_fetch_url_enforces_max_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(b"abcdef", content_type="text/plain")

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)

    with pytest.raises(InternetIngressError):
        fetch_url(
            "https://example.com/large.txt",
            tmp_path,
            policy=InternetIngressPolicy(max_bytes=5, timeout_seconds=1),
        )
    assert load_manifest(tmp_path)["artifact_count"] == 0


def test_clone_git_repository_records_project_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        destination = Path(command[-1])
        destination.mkdir(parents=True)
        (destination / "README.md").write_text("hello\n", encoding="utf-8")
        assert command[:3] == ["git", "clone", "--depth"]
        assert command[-2] == "https://example.com/project.git"
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("modules.internet.ingress.subprocess.run", fake_run)

    artifact = clone_git_repository(
        "https://example.com/project.git",
        tmp_path,
        policy=InternetIngressPolicy(timeout_seconds=1),
        ref="main",
        depth=1,
    )

    assert artifact.fetch_kind == "git_clone"
    assert Path(artifact.local_path, "README.md").read_text(encoding="utf-8") == "hello\n"
    manifest = load_manifest(tmp_path)
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["metadata"]["ref"] == "main"


def test_fetch_project_extracts_zip_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("demo/README.md", "ok\n")
        archive.writestr("demo/src/app.py", "print('ok')\n")

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        return _FakeResponse(buffer.getvalue(), content_type="application/zip")

    monkeypatch.setattr("modules.internet.ingress.urlopen", fake_urlopen)

    artifact = fetch_project(
        "https://example.com/demo.zip",
        tmp_path,
        policy=InternetIngressPolicy(max_bytes=4096, timeout_seconds=1),
        source_type="auto",
    )

    assert artifact.fetch_kind == "archive_extract"
    project_root = Path(artifact.local_path)
    assert (project_root / "demo" / "README.md").read_text(encoding="utf-8") == "ok\n"
    assert artifact.metadata["extracted_file_count"] == 2
    manifest = load_manifest(tmp_path)
    assert manifest["artifact_count"] == 2
    assert {row["fetch_kind"] for row in manifest["artifacts"]} == {"http", "archive_extract"}
