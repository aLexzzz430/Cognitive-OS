"""Local machine adapter backed by an empty-first mirror workspace."""

from __future__ import annotations

from integrations.local_machine.task_adapter import (
    LOCAL_MACHINE_ADAPTER_VERSION,
    LocalMachineSurfaceAdapter,
)

__all__ = [
    "LOCAL_MACHINE_ADAPTER_VERSION",
    "LocalMachineSurfaceAdapter",
]
