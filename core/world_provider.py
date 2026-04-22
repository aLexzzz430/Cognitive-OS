"""World provider contract for CoreMainLoop startup."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


class MissingWorldAdapterError(RuntimeError):
    """Raised when runtime world adapter contract is not explicitly satisfied."""


@dataclass
class WorldProviderConfig:
    runtime_env: Optional[str]
    world_adapter: Any
    world_provider_source: Optional[str] = None


def resolve_world_provider(config: WorldProviderConfig) -> Tuple[Any, Dict[str, str]]:
    """
    Resolve world provider with strict runtime contract.

    Rules:
    - runtime_env is required.
    - world_adapter is required.
    - world_provider_source is recorded for audit.
    """
    if not config.runtime_env:
        raise MissingWorldAdapterError(
            "WORLD_PROVIDER_MISSING_RUNTIME_ENV: runtime_env is required and must be explicit."
        )

    if config.world_adapter is None:
        raise MissingWorldAdapterError(
            "WORLD_PROVIDER_MISSING_ADAPTER: world_adapter is required. "
            f"runtime_env={config.runtime_env}. source={config.world_provider_source or 'unknown'}"
        )

    source = config.world_provider_source or config.runtime_env
    return config.world_adapter, {
        'runtime_env': str(config.runtime_env),
        'world_provider_source': str(source),
    }
