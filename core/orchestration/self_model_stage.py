from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from modules.continuity.estimator import estimate_continuity_confidence


@dataclass(frozen=True)
class SelfModelRefreshInput:
    continuity_snapshot: Optional[Dict[str, Any]]
    resource_state: Any
    self_model_facade: Any
    agent_id: str
    arm_mode: str
    teacher_present: Optional[bool] = None
    runtime_source: str = 'unknown'
    world_provider: str = 'unknown'
    ablation_mode: str = 'unknown'
    enabled_modules: Optional[List[str]] = None
    capability_switches: Optional[Dict[str, bool]] = None
    module_registry: Optional[Dict[str, str]] = None


class SelfModelStage:
    """Stage-specific self-model refresh orchestration."""

    @staticmethod
    def _build_external_dependencies(input_obj: SelfModelRefreshInput) -> List[str]:
        capability_switches = input_obj.capability_switches or {}
        module_registry = input_obj.module_registry or {}
        enabled_modules = set(input_obj.enabled_modules or [])

        for module_name, is_enabled in capability_switches.items():
            if is_enabled:
                enabled_modules.add(str(module_name))

        dependencies: List[str] = []
        for module_name in sorted(enabled_modules):
            dependency = str(module_registry.get(module_name, module_name) or '').strip()
            if dependency:
                dependencies.append(dependency)

        return dependencies or ['unknown']

    @staticmethod
    def refresh(input_obj: SelfModelRefreshInput) -> bool:
        if input_obj.self_model_facade is None:
            return False

        continuity_confidence = estimate_continuity_confidence(
            input_obj.continuity_snapshot,
            fallback=0.5,
        )
        external_dependencies = SelfModelStage._build_external_dependencies(input_obj)

        input_obj.self_model_facade.refresh(
            resource_state=input_obj.resource_state,
            continuity_confidence=continuity_confidence,
            continuity_snapshot=input_obj.continuity_snapshot,
            teacher_present=input_obj.teacher_present,
            identity_markers={
                'agent_id': input_obj.agent_id,
                'arm_mode': input_obj.arm_mode,
                'runtime_source': input_obj.runtime_source,
                'world_provider': input_obj.world_provider,
                'ablation_mode': input_obj.ablation_mode,
                'enabled_modules': sorted(input_obj.enabled_modules or []),
            },
            external_dependencies=external_dependencies,
        )
        return True
