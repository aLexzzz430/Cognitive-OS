"""
self_model/__init__.py

Sprint 5: self_model/ 自我认知

提供自我认知查询.

文件:
- capability_profile.py: CapabilityProfile, FunctionCapability
- reliability_tracker.py: ReliabilityTracker, ModuleReliability
- resource_state.py: ResourceState, ResourceBudget, StateIndicators

Rules:
- 第一版只提供可查询先验
- 不直接控制行为
- 输出给 decision/arbiter 和 planner
"""

from self_model.capability_profile import (
    CapabilityLevel,
    FunctionCapability,
    CapabilityProfile,
)

from self_model.reliability_tracker import (
    ModuleReliability,
    FailureStrategyProfile,
    ReliabilityTracker,
)

from self_model.resource_state import (
    ResourceBudget,
    StateIndicators,
    ResourceState,
)
from self_model.identity_ledger import DurableIdentityLedger, DurableIdentitySnapshot
from self_model.capability_envelope import CapabilityEnvelope, build_capability_envelope
from self_model.autobiographical_summary import build_autobiographical_summary
from self_model.state import SelfModelState
from self_model.facade import SelfModelFacade
from self_model.context_adapter import SelfModelContextAdapter

__all__ = [
    # Capability Profile
    'CapabilityLevel',
    'FunctionCapability',
    'CapabilityProfile',
    # Reliability Tracker
    'ModuleReliability',
    'FailureStrategyProfile',
    'ReliabilityTracker',
    # Resource State
    'ResourceBudget',
    'StateIndicators',
    'ResourceState',
    'DurableIdentityLedger',
    'DurableIdentitySnapshot',
    'CapabilityEnvelope',
    'build_capability_envelope',
    'build_autobiographical_summary',
    # High-level Self Model
    'SelfModelState',
    'SelfModelFacade',
    'SelfModelContextAdapter',
]
