"""
modules/world_model/__init__.py

World Model Omega — Phases A through E

Stage A: Event Schema
- EventType, WorldModelEvent, WorldModelEventBus

Stage B: Belief State
- BeliefStatus, BeliefValue, Belief, BeliefLedger, BeliefUpdater

Stage D: Counterfactual Engine
- CounterfactualConfidence, CounterfactualOutcome, StateSlice, CounterfactualEngine

Stage E: Distillation Compiler
- DistillationTarget, DistillationDecision, DistillationCandidate, DistillationCompiler

Rules:
- CoreMainLoop emits events via WorldModelEventBus
- World Model Omega reads events to update belief state
- NO direct writes to formal truth (must go through validator/committer)
- NO new control authority
- Counterfactual engine and compiler are advisory only
"""

from modules.world_model.events import (
    EventType,
    WorldModelEvent,
    WorldModelEventBus,
)

from modules.world_model.belief import (
    BeliefStatus,
    BeliefValue,
    Belief,
    BeliefLedger,
    BeliefUpdater,
)

from modules.world_model.counterfactual import (
    CounterfactualConfidence,
    CounterfactualOutcome,
    StateSlice,
    CounterfactualEngine,
)
from modules.world_model.mechanism import (
    MechanismHypothesis,
)

from modules.world_model.protocol import (
    WorldModelControlProtocol,
)
from modules.world_model.context_adapter import WorldModelContextAdapter
from modules.world_model.object_graph import build_object_graph
from modules.world_model.affordance_graph import build_affordance_graph
from modules.world_model.mechanism_graph import build_mechanism_graph
from modules.world_model.latent_dynamics import summarize_latent_dynamics
from modules.world_model.rollout import (
    build_rollout_support,
    compare_function_rollouts,
    simulate_function_rollout,
)
from modules.world_model.test_proposal import propose_discriminating_tests

from modules.world_model.compiler import (
    DistillationTarget,
    DistillationDecision,
    DistillationCandidate,
    DistillationCompiler,
)
from modules.world_model.prediction_feedback import (
    PredictionMissFeedbackRuntime,
)
from modules.world_model.hidden_state import (
    HiddenStateSnapshot,
    HiddenStateTracker,
)

__all__ = [
    # Stage A
    'EventType',
    'WorldModelEvent',
    'WorldModelEventBus',
    # Stage B
    'BeliefStatus',
    'BeliefValue',
    'Belief',
    'BeliefLedger',
    'BeliefUpdater',
    # Stage D
    'CounterfactualConfidence',
    'CounterfactualOutcome',
    'StateSlice',
    'CounterfactualEngine',
    'MechanismHypothesis',
    # Control protocol
    'WorldModelControlProtocol',
    'WorldModelContextAdapter',
    'build_object_graph',
    'build_affordance_graph',
    'build_mechanism_graph',
    'summarize_latent_dynamics',
    'simulate_function_rollout',
    'compare_function_rollouts',
    'build_rollout_support',
    'propose_discriminating_tests',
    # Stage E
    'DistillationTarget',
    'DistillationDecision',
    'DistillationCandidate',
    'DistillationCompiler',
    'PredictionMissFeedbackRuntime',
    'HiddenStateSnapshot',
    'HiddenStateTracker',
]
