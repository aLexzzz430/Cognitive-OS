"""Audited, policy-gated internet ingress for local-first Con OS runs."""

from __future__ import annotations

from modules.internet.ingress import (
    INTERNET_INGRESS_MANIFEST_VERSION,
    INTERNET_INGRESS_VERSION,
    InternetArtifact,
    InternetIngressError,
    InternetIngressPolicy,
    clone_git_repository,
    fetch_archive_project,
    fetch_project,
    fetch_url,
    load_manifest,
    normalize_url,
    validate_url,
)

__all__ = [
    "INTERNET_INGRESS_MANIFEST_VERSION",
    "INTERNET_INGRESS_VERSION",
    "InternetArtifact",
    "InternetIngressError",
    "InternetIngressPolicy",
    "clone_git_repository",
    "fetch_archive_project",
    "fetch_project",
    "fetch_url",
    "load_manifest",
    "normalize_url",
    "validate_url",
]
