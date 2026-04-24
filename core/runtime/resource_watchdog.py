from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import time
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RESOURCE_WATCHDOG_VERSION = "conos.resource_watchdog/v1"


def _now() -> float:
    return float(time.time())


def _memory_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return 0.0
    rss = float(getattr(usage, "ru_maxrss", 0.0) or 0.0)
    if platform.system() == "Darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _load_average() -> List[float]:
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return []


def _http_probe(url: str, *, timeout: float) -> Dict[str, Any]:
    started_at = _now()
    request = Request(url, headers={"User-Agent": "ConOS-Runtime-Watchdog/0.1"})
    try:
        with urlopen(request, timeout=max(0.1, float(timeout))) as response:
            status_code = int(getattr(response, "status", 0) or 0)
            ok = 200 <= status_code < 500
            return {
                "status": "OK" if ok else "DEGRADED",
                "status_code": status_code,
                "latency_seconds": max(0.0, _now() - started_at),
            }
    except socket.timeout as exc:
        return {"status": "DEGRADED", "reason": "timeout", "error": str(exc), "latency_seconds": max(0.0, _now() - started_at)}
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, socket.timeout):
            reason_name = "timeout"
        else:
            reason_name = "disconnect"
        return {"status": "DEGRADED", "reason": reason_name, "error": str(exc), "latency_seconds": max(0.0, _now() - started_at)}
    except Exception as exc:
        return {"status": "DEGRADED", "reason": "probe_error", "error": str(exc), "latency_seconds": max(0.0, _now() - started_at)}


@dataclass(frozen=True)
class WatchdogThresholds:
    max_memory_mb: float = 4096.0
    max_load_1m: Optional[float] = None
    min_disk_free_mb: float = 256.0
    min_disk_free_ratio: float = 0.02
    network_url: Optional[str] = None
    network_timeout_seconds: float = 3.0
    ollama_base_url: Optional[str] = None
    ollama_timeout_seconds: float = 10.0
    ollama_required: bool = False


class ResourceWatchdog:
    """Best-effort local resource and model-endpoint health monitor."""

    def __init__(self, *, runtime_home: str | Path, thresholds: Optional[WatchdogThresholds] = None) -> None:
        self.runtime_home = Path(runtime_home).expanduser()
        self.thresholds = thresholds or WatchdogThresholds(
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL") or os.environ.get("CONOS_OLLAMA_BASE_URL") or None,
        )

    def evaluate(self) -> Dict[str, Any]:
        checks: Dict[str, Any] = {
            "memory": self._check_memory(),
            "cpu": self._check_cpu(),
            "disk": self._check_disk(),
            "network": self._check_network(),
            "ollama": self._check_ollama(),
        }
        degraded_reasons = [
            f"{name}:{check.get('reason') or check.get('status')}"
            for name, check in checks.items()
            if str(check.get("status", "")) == "DEGRADED"
        ]
        return {
            "schema_version": RESOURCE_WATCHDOG_VERSION,
            "status": "DEGRADED" if degraded_reasons else "OK",
            "created_at": _now(),
            "runtime_home": str(self.runtime_home),
            "checks": checks,
            "degraded_reasons": degraded_reasons,
        }

    def _check_memory(self) -> Dict[str, Any]:
        value = _memory_mb()
        limit = float(self.thresholds.max_memory_mb)
        status = "DEGRADED" if limit > 0 and value > limit else "OK"
        return {"status": status, "rss_mb": value, "max_memory_mb": limit, "reason": "memory_limit_exceeded" if status == "DEGRADED" else ""}

    def _check_cpu(self) -> Dict[str, Any]:
        loads = _load_average()
        threshold = self.thresholds.max_load_1m
        degraded = threshold is not None and bool(loads) and loads[0] > float(threshold)
        return {
            "status": "DEGRADED" if degraded else "OK",
            "load_average": loads,
            "max_load_1m": threshold,
            "reason": "load_average_exceeded" if degraded else "",
        }

    def _check_disk(self) -> Dict[str, Any]:
        self.runtime_home.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.runtime_home)
        free_mb = float(usage.free) / (1024.0 * 1024.0)
        free_ratio = float(usage.free) / float(usage.total) if usage.total else 0.0
        degraded = free_mb < float(self.thresholds.min_disk_free_mb) or free_ratio < float(self.thresholds.min_disk_free_ratio)
        return {
            "status": "DEGRADED" if degraded else "OK",
            "free_mb": free_mb,
            "free_ratio": free_ratio,
            "min_disk_free_mb": float(self.thresholds.min_disk_free_mb),
            "min_disk_free_ratio": float(self.thresholds.min_disk_free_ratio),
            "reason": "disk_space_low" if degraded else "",
        }

    def _check_network(self) -> Dict[str, Any]:
        if not self.thresholds.network_url:
            return {"status": "SKIPPED", "reason": "network_check_not_configured"}
        return _http_probe(str(self.thresholds.network_url), timeout=float(self.thresholds.network_timeout_seconds))

    def _check_ollama(self) -> Dict[str, Any]:
        base_url = (self.thresholds.ollama_base_url or "").strip().rstrip("/")
        if not base_url:
            if self.thresholds.ollama_required:
                return {"status": "DEGRADED", "reason": "ollama_not_configured"}
            return {"status": "SKIPPED", "reason": "ollama_not_configured"}
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return {"status": "DEGRADED", "reason": "invalid_ollama_base_url", "base_url": base_url}
        result = _http_probe(f"{base_url}/api/tags", timeout=float(self.thresholds.ollama_timeout_seconds))
        result["base_url"] = base_url
        return result
