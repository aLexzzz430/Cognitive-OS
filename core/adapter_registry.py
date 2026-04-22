from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from core.conos_repository_layout import LAYER_ADAPTER, classify_repo_path


@dataclass(frozen=True)
class AdapterSpec:
    adapter_key: str
    module_name: str
    symbol_name: str
    description: str = ""

    @property
    def repo_path(self) -> str:
        return f"{self.module_name.replace('.', '/')}.py"


_DEFAULT_ADAPTER_SPECS: Dict[str, AdapterSpec] = {
    "arc_agi3.perception_bridge": AdapterSpec(
        adapter_key="arc_agi3.perception_bridge",
        module_name="integrations.arc_agi3.perception_bridge",
        symbol_name="PerceptionBridge",
        description="ARC-AGI-3 visual/perception adapter bridge.",
    ),
    "arc_agi3.intervention_execution_compiler": AdapterSpec(
        adapter_key="arc_agi3.intervention_execution_compiler",
        module_name="integrations.arc_agi3.intervention_execution_compiler",
        symbol_name="ARCAGI3InterventionExecutionCompiler",
        description="ARC-AGI-3 execution compiler for generic intervention targets.",
    ),
    "webarena.task_adapter": AdapterSpec(
        adapter_key="webarena.task_adapter",
        module_name="integrations.webarena.task_adapter",
        symbol_name="WebArenaSurfaceAdapter",
        description="WebArena environment adapter.",
    ),
}


def _default_symbol_loader(module_name: str, symbol_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


class AdapterRegistry:
    def __init__(
        self,
        *,
        specs: Optional[Mapping[str, AdapterSpec]] = None,
        symbol_loader: Optional[Callable[[str, str], Any]] = None,
    ) -> None:
        self._specs = dict(specs or _DEFAULT_ADAPTER_SPECS)
        self._symbol_loader = symbol_loader or _default_symbol_loader
        self._symbol_cache: Dict[str, Any] = {}

    def specs(self) -> Dict[str, AdapterSpec]:
        return dict(self._specs)

    def get_spec(self, adapter_key: str) -> Optional[AdapterSpec]:
        return self._specs.get(str(adapter_key or "").strip())

    def load_symbol(self, adapter_key: str) -> Any:
        key = str(adapter_key or "").strip()
        if not key:
            return None
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        spec = self.get_spec(key)
        if spec is None:
            return None
        try:
            symbol = self._symbol_loader(spec.module_name, spec.symbol_name)
        except Exception:
            return None
        self._symbol_cache[key] = symbol
        return symbol

    def build(self, adapter_key: str, *args: Any, **kwargs: Any) -> Any:
        symbol = self.load_symbol(adapter_key)
        if symbol is None:
            return None
        try:
            return symbol(*args, **kwargs)
        except Exception:
            return None


_DEFAULT_REGISTRY: Optional[AdapterRegistry] = None


def get_adapter_registry() -> AdapterRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = AdapterRegistry()
    return _DEFAULT_REGISTRY


def build_optional_adapter(adapter_key: str, *args: Any, registry: Optional[AdapterRegistry] = None, **kwargs: Any) -> Any:
    adapter_registry = registry or get_adapter_registry()
    return adapter_registry.build(adapter_key, *args, **kwargs)


def find_adapter_registry_violations(registry: Optional[AdapterRegistry] = None) -> list[dict]:
    adapter_registry = registry or get_adapter_registry()
    findings: list[dict] = []
    for key, spec in sorted(adapter_registry.specs().items()):
        layer = classify_repo_path(spec.repo_path)
        if layer != LAYER_ADAPTER:
            findings.append(
                {
                    "adapter_key": key,
                    "repo_path": spec.repo_path,
                    "layer": layer,
                }
            )
    return findings
