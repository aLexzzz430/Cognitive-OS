"""
core/main_loop.py - 唯一真主循环

使用分离后的模块构建。CoreMainLoop 作为唯一正式主体调度器。

Architecture:
  CoreMainLoop
    ├── _world: ISurfaceAdapter (EUltimateWorld via adapter)
    ├── _store: IObjectStore (ObjectStore)
    ├── _retriever: IRetriever (EpisodicRetriever with arm modes)
    ├── _hypotheses: HypothesisTracker
    ├── _test_engine: DiscriminatingTestEngine
    ├── _skill_rewriter: SkillRewriter
    ├── _extractor: NovelAPIRawEvidenceExtractor (Step 9)
    ├── _committer: NovelAPICommitter (Step 10)
    ├── _validator: ProposalValidator
    └── _state_mgr: IStateManager (StateManager)

All capabilities are injected as interfaces. CoreMainLoop does not
directly implement P0-1/P0-2/P0-3 - it coordinates them via well-defined interfaces.
"""

import random
import uuid
import time
import hashlib
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

# Import separated modules
from modules.governance.object_store import ObjectStore, GovernanceDecision, ProposalValidator, REJECT
from modules.evidence.extractor import NovelAPIRawEvidenceExtractor
from modules.hypothesis import (
    HypothesisStatus,
    Hypothesis,
    HypothesisTracker,
    DiscriminatingTest,
    DiscriminatingTestEngine,
    LLMHypothesisInterface,
    LLMProbeDesigner,
)
from modules.hypothesis.mechanism_posterior_updater import MechanismPosteriorUpdater
from modules.skills import SkillRewriter, LLMSkillFrontend
from modules.state import StateManager, get_state_manager, init_state
from modules.episodic import LLMRetrievalInterface, RetrievalRuntimeTier
from modules.recovery import LLMErrorRecoveryInterface, RecoveryType
from modules.representations import LLMRepresentationProposer
from modules.representations.store import get_warehouse, get_runtime_store
from modules.representations.proposer_llm import TrajectoryEntry
from modules.strategies import (
    RetrievalGateStrategy,
    RerankStrategy,
    QueryRewriteStrategy,
)
from modules.continuity import ContinuityManager
from modules.teacher import TeacherProtocol
from modules.governance.gate import get_governance
from modules.governance.family_registry import FamilyCard, FamilyRegistry, FamilyState, init_registry
from modules.world_model.events import WorldModelEventBus, EventType, WorldModelEvent
from modules.world_model.belief import BeliefLedger, BeliefUpdater
from modules.world_model.counterfactual import CounterfactualEngine, StateSlice
from modules.world_model.mechanism import MechanismHypothesis, MechanismExtractor, MechanismFormalWriter
from modules.world_model.compiler import DistillationCompiler, DistillationDecision
from modules.world_model.protocol import WorldModelControlProtocol
from modules.world_model.prediction_feedback import PredictionMissFeedbackRuntime
from modules.world_model.hidden_state import HiddenStateTracker
from modules.world_model.object_binding import build_object_bindings
from modules.world_model.task_frame import infer_task_frame
from core.adapter_registry import build_optional_adapter
from core.runtime_budget import RuntimeBudgetConfig
from core.runtime_paths import default_event_log_path
from core.prediction_runtime import PredictionAdjudicator, PredictionEngine, PredictionRegistry
from core.world_provider import WorldProviderConfig, resolve_world_provider
from core.retrieval_control import RetrievalControlState, RetrievalSignals, RetrievalDecisionRecord, RetrievalAuxPolicy
from core.cognition.ablation_config import CausalLayerAblationConfig
from core.cognition.unified_context import UnifiedCognitiveContext
from core.learning import CreditAssignment, OutcomeSignal, aggregate_learning_updates
from core.orchestration.context_stage import (
    build_tick_context,
    build_world_model_context,
    build_world_model_transition_priors,
    build_unified_cognitive_context,
    build_legacy_decision_context,
)
from core.orchestration.self_model_stage import SelfModelRefreshInput, SelfModelStage
from core.orchestration.state_sync import StateSyncInput, StateSyncOrchestrator
from core.orchestration.continuity_persistence_adapter import ContinuityPersistenceAdapter
from core.orchestration.episode_lifecycle import EpisodeLifecycle
from core.orchestration.episode_consolidation_runtime import commit_consolidation_candidates
from core.orchestration.formal_memory_persistence import persist_durable_object_records
from core.orchestration.main_loop_bootstrap import (
    MainLoopBootstrapConfig,
    bootstrap_core_capabilities,
    bootstrap_learning_prediction_and_procedure,
    bootstrap_memory_trace_and_self_model,
    bootstrap_state_and_stage_runtimes,
    bootstrap_tracking_state,
)
from core.orchestration.retrieval_stage import RetrievalStage
from core.orchestration.staged_prediction_bridge_runtime import run_stage2_prediction_bridge
from core.orchestration.staged_retrieval_runtime import run_stage1_retrieval
from core.orchestration.staged_execution_runtime import run_stage3_execution
from core.orchestration.stage3_execution_support_runtime import (
    collect_executable_function_names,
    is_trackable_executable_function,
    record_memory_consumption_proof,
    resolve_action_for_execution,
)
from core.orchestration.stage5_evidence_commit_runtime import run_stage5_evidence_commit
from core.orchestration.stage6_post_commit_runtime import run_stage6_post_commit
from core.orchestration.planner_stage import PlannerStage
from core.orchestration.planner_runtime import PlannerPorts, PlannerRuntime
from core.orchestration.governance_stage import GovernanceStage
from core.orchestration.state_abstraction import summarize_cognitive_object_records
from core.orchestration.state_sync_stage import StateSyncStage
from core.orchestration.goal_progress_runtime import (
    derive_action_effect_signature,
    derive_goal_progress_assessment,
    recent_goal_progress_state,
)
from core.orchestration.learning_runtime import (
    annotate_candidates_with_learning_updates,
    collect_outcome_learning_signal,
    commit_learning_updates,
    run_apply_learning_updates,
    run_apply_learning_policy_updates,
)
from core.orchestration.staged_tick_runtime import (
    apply_execution_outcome,
    begin_staged_tick,
    finalize_staged_tick,
    record_candidate_frontier,
    record_surfaced_candidates,
    sync_tick_state,
)
from core.orchestration.stage2_candidate_generation_runtime import run_stage2_candidate_generation
from core.orchestration.stage2_action_runtime import run_stage2_action_generation
from core.orchestration.stage2_candidate_filter_runtime import (
    materialize_stage2_prediction_fallback,
    rank_counterfactual_candidates,
    run_stage2_plan_constraints,
    run_stage2_self_model_suppression,
)
from core.orchestration.stage2_governance_runtime import run_stage2_governance
from core.orchestration.audit_report import build_main_loop_audit
from core.orchestration.audit_utils import json_safe, record_continuity_tick, compute_observation_signature, cooldown_ready, record_llm_tick_summary
from core.orchestration.llm_shadow_runtime import (
    finalize_llm_analyst_post_execution,
    finalize_llm_shadow_post_execution,
    prepare_llm_analyst_initial_goal,
    run_llm_shadow_pre_execution,
)
from core.orchestration.action_utils import extract_available_functions, extract_action_function_name, extract_action_identity, extract_action_xy, candidate_counts, repair_action_function_name, build_kwargs_from_context
from decision.candidate_cooldown_gate import CandidateCooldownGate
from decision.discriminating_action_selector import DiscriminatingActionSelector
from core.orchestration.structured_answer import StructuredAnswerSynthesizer
from core.orchestration.testing_recovery_runtime import TestingRecoveryRuntime, TeacherInterventionPolicy, TestingRecoveryResult
from core.orchestration.stage_types import (
    RetrievalStageInput,
    PlannerStageInput,
    GovernanceStageInput,
)
from core.orchestration.runtime_stage_contracts import (
    ApplyLearningUpdatesInput,
    PostCommitIntegrationInput,
    ProcessGraduationCandidatesInput,
    Stage1RetrievalInput,
    Stage2CandidateGenerationInput,
    Stage2GovernanceInput as Stage2GovernanceSubstageInput,
    Stage2PlanConstraintsInput,
    Stage2PredictionBridgeInput,
    Stage2SelfModelSuppressionInput,
    Stage3ExecutionInput,
    Stage5EvidenceCommitInput,
    Stage6PostCommitInput,
)
from core.orchestration.runtime_stage_modules import (
    ApplyLearningUpdatesRuntime,
    PostCommitIntegrationRuntime,
    ProcessGraduationCandidatesRuntime,
    Stage1RetrievalRuntime,
    Stage2CandidateGenerationRuntime,
    Stage2GovernanceRuntime as Stage2GovernanceSubstageRuntime,
    Stage2PlanConstraintsRuntime,
    Stage2PredictionBridgeRuntime,
    Stage2SelfModelSuppressionRuntime,
    Stage3ExecutionRuntime,
    Stage5EvidenceCommitRuntime,
    Stage6PostCommitRuntime,
)
from core.orchestration.prediction_feedback import (
    PredictionFeedbackInput,
    PredictionFeedbackPipeline,
    apply_prediction_error_feedback,
    prediction_bundle_to_dict,
    record_prediction_trace,
)
from core.orchestration.prediction_context_runtime import (
    build_recovery_prediction_context,
    build_self_model_prediction_summary,
)
from core.orchestration.probe_ranking_runtime import rank_probe_candidates_by_prediction
from core.orchestration.post_commit_integration import integrate_committed_objects
from core.orchestration.llm_route_runtime import (
    ensure_llm_capability_registry,
    ensure_model_router,
    initialize_model_router,
    llm_route_budget_status,
    llm_route_state,
    llm_route_usage_bucket,
    llm_route_usage_summary,
    record_llm_route_blocked,
    record_llm_route_usage,
    resolve_llm_capability_spec,
    resolve_llm_client,
    resolve_llm_gateway,
)
from core.orchestration.llm_route_policy_runtime import (
    build_llm_route_context,
    goal_task_binding_for_llm_policy,
    goal_task_capability_specs,
    goal_task_route_specs,
    resolved_llm_capability_specs,
    resolved_llm_route_specs,
    route_capability_requirements,
    runtime_budget_capability_specs,
    runtime_budget_route_specs,
)
from core.orchestration.procedure_memory_runtime import (
    load_procedure_objects,
    maybe_commit_procedure_chain,
    procedure_observed_functions,
    procedure_task_signature,
    procedure_text_tokens,
)
from core.orchestration.plan_feedback_runtime import (
    apply_step_transitions_with_feedback as run_step_transitions_with_feedback,
    plan_step_feedback_reference,
    record_verification_feedback_for_transition,
    recent_llm_route_usage_for_task,
    should_auto_consume_verifier_authority,
    verification_feedback_from_transition,
)
from core.orchestration.main_loop_ports import MainLoopContextProvider, MainLoopGovernancePorts
from core.reasoning import DeliberationEngine
from trace_runtime import resolve_trace_runtime

CausalTraceLogger, EventTimeline, _ = resolve_trace_runtime()

from core.main_loop_components import (
    ARM_MODE_FULL,
    ARM_MODE_WRONG_BINDING,
    ARM_MODE_LOCAL_ONLY,
    ARM_MODE_SHUFFLED,
    ARM_MODE_FRESH,
    ARM_MODE_NO_TRANSFER,
    ARM_MODE_NO_PREDICTION,
    ARM_MODE_NO_PROCEDURE_LEARNING,
    CAPABILITY_ADVISORY,
    CAPABILITY_CONSTRAINED_CONTROL,
    CAPABILITY_PRIMARY_CONTROL,
    LOW_RISK_CONTROL_FUNCTIONS,
    ORGAN_CAPABILITY_KEYS,
    RetrievedCandidate,
    RetrieveResult,
    TickContextFrame,
    PlannerStageOutput,
    DecisionBridgeInput,
    GovernanceStageOutput,
    _NoopMetaSnapshot,
    _AdaptiveMetaControl,
    _NoopUpdateEngine,
    _FormalUpdateEngine,
    _NoopMetaControl,
    _NoopPromotionEngine,
    TransferTraceEvent,
    _extract_candidate_function_name,
    TransferTraceLogger,
    TestOpportunityEntry,
    TestOpportunityAudit,
    EpisodicRetriever,
    _BaseArmEpisodicRetriever,
    _WrongBindingEpisodicRetriever,
    _LocalOnlyEpisodicRetriever,
    _ShuffledEpisodicRetriever,
    _FreshEpisodicRetriever,
    _NoTransferEpisodicRetriever,
    NovelAPICommitter,
)


# =============================================================================
# CoreMainLoop - 唯一真主循环
# =============================================================================


class CoreMainLoop:
    """
    唯一真主循环 - The single authoritative main loop.

    All P0 capabilities are coordinated via well-defined interfaces.
    The world adapter is injected, not hardcoded.

    Per tick pipeline:
      1. observe()        - get environment observation
      2. build_query()    - P0-1: build retrieval query
      3. retrieve()       - P0-1: search object store
      4. surface()        - P0-1: present candidates
      5. arm_evaluate()   - P0-1: apply arm distortion (if any)
      6. execute()        - act on world
      7. consume()        - P0-1: trace retrieval influence
      8. extract_evidence - Step 9: extract evidence
      9. validate()       - Step 9: run through ProposalValidator
      10. commit()        - Step 10: commit to ObjectStore
    """

    def __init__(
        self,
        agent_id: str,
        run_id: str,
        seed: int = 0,
        max_episodes: int = 3,
        max_ticks_per_episode: int = 50,
        arm_mode: str = ARM_MODE_FULL,
        causal_ablation: Optional[CausalLayerAblationConfig] = None,
        verbose: bool = False,
        world_adapter=None,
        world_provider_config: Optional[WorldProviderConfig] = None,
        world_provider_source: Optional[str] = None,
        llm_client=None,
        llm_mode: str = "integrated",
        runtime_budget: Optional[RuntimeBudgetConfig] = None,
        causal_layer_ablation_config: Optional[CausalLayerAblationConfig] = None,
        prediction_enabled: Optional[bool] = None,
    ):
        """
        Initialize CoreMainLoop with all required capability modules.

        Args:
            agent_id: Unique agent identifier
            run_id: Unique run identifier
            seed: Random seed
            max_episodes: Number of episodes to run
            max_ticks_per_episode: Max ticks per episode
            arm_mode: Legacy retrieval-arm mode ('full', 'fresh', 'wrong_binding', etc.)
            causal_ablation: Unified causal-layer ablation config. If omitted, uses defaults.
            causal_layer_ablation_config: Deprecated alias of causal_ablation; merged immediately when provided.
            verbose: Print verbose output
            world_adapter: ISurfaceAdapter implementation. Must be explicitly provided.
            world_provider_config: Runtime config requiring runtime_env/world_adapter.
            world_provider_source: Source label for audit trace (config/test_harness/injected).
        """
        self.agent_id = agent_id
        self.run_id = run_id
        self.seed = seed
        self.max_episodes = max_episodes
        self.max_ticks = max_ticks_per_episode
        self.verbose = verbose
        self.arm_mode = arm_mode
        # Migration: causal-layer experiments should use CausalLayerAblationConfig;
        # arm_mode remains for historical retrieval-arm behavior only.
        resolved_ablation = causal_ablation
        if causal_layer_ablation_config is not None:
            if resolved_ablation is None:
                resolved_ablation = causal_layer_ablation_config
            if verbose:
                print('[CoreMainLoop] Deprecated parameter causal_layer_ablation_config received; merged into causal_ablation.')
        self._causal_ablation = resolved_ablation or CausalLayerAblationConfig()
        self._llm_client = llm_client
        self._llm_mode = str(llm_mode or "integrated").strip().lower() or "integrated"
        self._llm_shadow_client = llm_client if self._llm_mode == "shadow" else None
        self._llm_analyst_client = llm_client if self._llm_mode == "analyst" else None
        self._runtime_budget = runtime_budget or RuntimeBudgetConfig()
        self._llm_route_specs: Dict[str, Dict[str, Any]] = self._runtime_budget_route_specs()
        self._model_router = initialize_model_router(self)
        self._learning_enabled_by_config = bool(getattr(self._runtime_budget, 'enable_learning', True))
        self._prediction_enabled_by_config = (
            bool(prediction_enabled)
            if prediction_enabled is not None
            else bool(getattr(self._runtime_budget, 'enable_prediction', True))
        )

        if world_provider_config is None and world_adapter is None:
            raise ValueError("world_adapter is required for CoreMainLoop")

        # World adapter must be explicitly configured, no implicit experiments fallback.
        provider_config = world_provider_config
        if provider_config is None:
            provider_config = WorldProviderConfig(
                runtime_env='injected' if world_adapter is not None else None,
                world_adapter=world_adapter,
                world_provider_source=world_provider_source or ('injected' if world_adapter is not None else None),
            )
        self._world, self._world_provider_meta = resolve_world_provider(provider_config)
        bootstrap_config = MainLoopBootstrapConfig(
            agent_id=agent_id,
            run_id=run_id,
            seed=seed,
            arm_mode=arm_mode,
            llm_client=llm_client,
        )
        bootstrap_core_capabilities(self, bootstrap_config)
        bootstrap_state_and_stage_runtimes(self, bootstrap_config)
        bootstrap_tracking_state(self)
        bootstrap_memory_trace_and_self_model(self, bootstrap_config)
        self._context_provider = MainLoopContextProvider(self)
        self._governance_ports = MainLoopGovernancePorts(self)
        bootstrap_learning_prediction_and_procedure(self)
        self._sync_llm_clients()
        self._bootstrap_policy_profile_object(episode=0)

    def _on_belief_event(self, event: WorldModelEvent) -> None:
        """
        Passive belief update handler - subscribes to event bus.

        This is purely advisory. It updates belief state but does NOT
        mutate formal truth or take control actions.

        P3-B: Belief updates from events (Stage B1).
        Issue 2 fix: Added OBJECT_CREATED for belief self-bootstrapping.
        """
        if event.event_type == EventType.OBSERVATION_RECEIVED:
            self._belief_updater.on_observation_received(event.data)
        elif event.event_type == EventType.ACTION_EXECUTED:
            self._belief_updater.on_action_executed(event.data)
        elif event.event_type == EventType.HYPOTHESIS_UPDATED:
            self._belief_updater.on_hypothesis_updated(event.data)
        elif event.event_type == EventType.TEST_EXECUTED:
            self._belief_updater.on_test_executed(event.data)
        elif event.event_type == EventType.OBJECT_CREATED:
            # Issue 2 fix: Create belief when object is committed
            self._belief_updater.on_object_created(event.data)
        elif event.event_type == EventType.RECOVERY_EXECUTED:
            self._belief_updater.on_recovery_executed(event.data)
        elif event.event_type == EventType.RECOVERY_OUTCOME_OBSERVED:
            self._belief_updater.on_recovery_outcome_observed(event.data)
        elif event.event_type == EventType.MECHANISM_EVIDENCE_ADDED:
            self._belief_updater.on_mechanism_evidence_added(event.data)

    def _write_world_model_state(self) -> None:
        """
        P3-B2: Write world_model state summaries to StateManager.

        Writes JSON-safe summaries only. No rich Python objects.

        Paths:
        - world_model.belief_state: summary of active beliefs
        - world_model.active_mechanisms: established beliefs
        - world_model.boundary_flags: high-uncertainty beliefs
        """
        # Collect belief summaries (JSON-safe only)
        active = self._belief_ledger.get_active_beliefs()
        established = self._belief_ledger.get_established_beliefs()
        high_uncertainty = [b for b in active if b.uncertainty > 0.6]

        belief_state_summary = {
            'total_beliefs': self._belief_ledger.belief_count(),
            'active_count': len(active),
            'established_count': len(established),
            'uncertain_count': len(high_uncertainty),
        }
        hidden_state_summary = self._hidden_state_tracker.summary() if self._hidden_state_tracker is not None else {}

        # Write to state manager (allowed in world_model namespace)
        self._state_sync.sync(StateSyncInput(
            updates={
                'world_model.belief_state': belief_state_summary,
                'world_model.belief_state.confidence': {
                    b.belief_id: b.confidence for b in active
                },
                'world_model.active_mechanisms': [
                    {'id': b.belief_id, 'variable': b.variable_name, 'posterior': b.posterior}
                    for b in established
                ],
                'world_model.boundary_flags': [
                    {'id': b.belief_id, 'variable': b.variable_name, 'uncertainty': b.uncertainty}
                    for b in high_uncertainty
                ],
            },
            reason='P3-B2 world_model belief state sync',
        ))
        self._state_sync.sync(StateSyncInput(
            updates={
                'world_model.hidden_state': hidden_state_summary,
            },
            reason='P3-B2 hidden state sync',
            module='world_model',
        ))

    def _build_world_model_context(self, perception_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return build_world_model_context(self._context_provider, perception_summary)

    def _prediction_runtime_active(self) -> bool:
        return bool(
            self._prediction_enabled
            and self._prediction_engine is not None
            and self._prediction_adjudicator is not None
        )

    def _build_world_model_transition_priors(
        self,
        perception_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_world_model_transition_priors(self._context_provider, perception_summary)

    def _build_unified_cognitive_context(
        self,
        obs: Optional[Dict[str, Any]],
        continuity_snapshot: Optional[Dict[str, Any]] = None,
        *,
        world_model_summary: Optional[Dict[str, Any]] = None,
        self_model_summary: Optional[Dict[str, Any]] = None,
        recent_failures: Optional[int] = None,
        world_shift_risk: Optional[float] = None,
    ) -> UnifiedCognitiveContext:
        ablation_cfg = getattr(self, '_causal_ablation', None)
        unified_enabled = bool(getattr(ablation_cfg, 'enable_unified_context', True))
        ablation_mode = str(getattr(ablation_cfg, 'unified_context_ablation_mode', 'stripped') or 'stripped')
        return build_unified_cognitive_context(
            self._context_provider,
            episode=self._episode,
            tick=self._tick,
            obs=obs,
            continuity_snapshot=continuity_snapshot,
            unified_enabled=unified_enabled,
            ablation_mode=ablation_mode,
            world_model_summary=world_model_summary,
            self_model_summary=self_model_summary,
            recent_failures=recent_failures,
            world_shift_risk=world_shift_risk,
        )

    def _build_tick_context_frame(
        self,
        obs_before: Optional[Dict[str, Any]],
        continuity_snapshot: Optional[Dict[str, Any]],
    ) -> TickContextFrame:
        return self.build_tick_context(obs_before, continuity_snapshot)

    def build_tick_context(
        self,
        obs_before: Optional[Dict[str, Any]],
        continuity_snapshot: Optional[Dict[str, Any]],
    ) -> TickContextFrame:
        frame_key = (self._episode, self._tick)
        build_count = self._tick_context_frame_build_counts.get(frame_key, 0) + 1
        self._tick_context_frame_build_counts[frame_key] = build_count
        if self.verbose:
            assert build_count <= 1, f"TickContextFrame rebuilt in same tick: {frame_key}, count={build_count}"

        if self._active_tick_context_frame and (
            self._active_tick_context_frame.episode == self._episode
            and self._active_tick_context_frame.tick == self._tick
        ):
            return self._active_tick_context_frame

        ablation_cfg = getattr(self, '_causal_ablation', None)
        unified_enabled = bool(getattr(ablation_cfg, 'enable_unified_context', True))
        ablation_mode = str(getattr(ablation_cfg, 'unified_context_ablation_mode', 'stripped') or 'stripped')
        frame = build_tick_context(
            self._context_provider,
            episode=self._episode,
            tick=self._tick,
            obs_before=obs_before,
            continuity_snapshot=continuity_snapshot,
            unified_enabled=unified_enabled,
            ablation_mode=ablation_mode,
        )
        mechanism_rows = []
        if isinstance(getattr(frame, 'unified_context', None), UnifiedCognitiveContext):
            mechanism_rows = list(frame.unified_context.mechanism_hypotheses_summary or [])
        runtime_view = self._mechanism_posterior_updater.build_runtime_view(
            self._mechanism_runtime_state,
            mechanism_rows,
            tick=self._tick,
            obs_before=obs_before,
        )
        base_world_model_summary = dict(frame.world_model_summary or {}) if isinstance(frame.world_model_summary, dict) else {}
        task_frame_summary = {}
        mechanism_prior_usage = {}
        if isinstance(getattr(frame, 'unified_context', None), UnifiedCognitiveContext):
            task_frame_summary = (
                dict(getattr(frame.unified_context, 'task_frame_summary', {}) or {})
                if isinstance(getattr(frame.unified_context, 'task_frame_summary', {}), dict)
                else {}
            )
        if task_frame_summary:
            base_world_model_summary['task_frame_summary'] = dict(task_frame_summary)
            mechanism_prior_rows = [
                dict(item)
                for item in list(task_frame_summary.get('mechanism_priors', []) or [])
                if isinstance(item, dict)
            ]
            if mechanism_prior_rows:
                base_world_model_summary['mechanism_priors'] = mechanism_prior_rows
            inferred_goal = (
                dict(task_frame_summary.get('inferred_level_goal', {}) or {})
                if isinstance(task_frame_summary.get('inferred_level_goal', {}), dict)
                else {}
            )
            mechanism_prior_usage = {
                'count': int(inferred_goal.get('mechanism_prior_count', 0) or 0),
                'confidence': float(inferred_goal.get('mechanism_prior_confidence', 0.0) or 0.0),
                'object_ids': [
                    str(item or '')
                    for item in list(inferred_goal.get('mechanism_prior_object_ids', []) or [])
                    if str(item or '')
                ],
                'supporting_functions': [
                    str(item or '')
                    for item in list(inferred_goal.get('mechanism_prior_supporting_functions', []) or [])
                    if str(item or '')
                ],
                'supported_goal_anchor_refs': [
                    str(item or '')
                    for item in list(inferred_goal.get('mechanism_prior_supported_goal_anchor_refs', []) or [])
                    if str(item or '')
                ],
                'controller_anchor_refs': [
                    str(item or '')
                    for item in list(inferred_goal.get('mechanism_prior_controller_anchor_refs', []) or [])
                    if str(item or '')
                ],
                'supported_goal_colors': [
                    int(item)
                    for item in list(inferred_goal.get('mechanism_prior_supported_goal_colors', []) or [])
                    if isinstance(item, int)
                ],
                'preferred_progress_mode': str(
                    inferred_goal.get('preferred_progress_mode', '') or ''
                ),
            }
            if (
                mechanism_prior_usage['count'] > 0
                or mechanism_prior_usage['object_ids']
                or mechanism_prior_usage['supporting_functions']
            ):
                base_world_model_summary['mechanism_prior_usage'] = dict(mechanism_prior_usage)
        static_control = base_world_model_summary.get('mechanism_control_summary', {}) if isinstance(base_world_model_summary.get('mechanism_control_summary', {}), dict) else {}
        merged_control = dict(static_control)
        merged_control.update(dict(runtime_view.control_summary or {}))
        base_world_model_summary['mechanism_hypotheses_summary'] = list(runtime_view.mechanisms or mechanism_rows)
        base_world_model_summary['mechanism_hypothesis_objects'] = list(runtime_view.mechanisms or mechanism_rows)
        base_world_model_summary['mechanism_control_summary'] = merged_control
        frame.world_model_summary = base_world_model_summary
        if isinstance(getattr(frame, 'unified_context', None), UnifiedCognitiveContext):
            frame.unified_context.active_beliefs_summary = dict(base_world_model_summary)
            frame.unified_context.mechanism_hypotheses_summary = list(runtime_view.mechanisms or mechanism_rows)
            frame.unified_context.mechanism_control_summary = dict(merged_control)
        self._last_task_frame_summary = dict(task_frame_summary)
        self._last_mechanism_prior_usage = dict(mechanism_prior_usage)
        self._last_mechanism_runtime_view = {
            'mechanism_hypotheses_summary': list(runtime_view.mechanisms or mechanism_rows),
            'mechanism_control_summary': dict(merged_control),
            'mechanism_prior_usage': dict(mechanism_prior_usage),
            'task_frame_summary': dict(task_frame_summary),
        }
        self._active_tick_context_frame = frame
        return frame

    def commit_objects(self, proposals: List[Dict[str, Any]], top_k: int = 5) -> List[str]:
        """Public commit facade for scripts and harnesses."""
        return self._committer.commit(proposals, top_k=top_k)

    def get_committed_objects(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """Public retrieval facade for recently committed durable objects."""
        return self._committer.get_committed_objects(top_k=top_k)

    def run_post_commit_integration(
        self,
        committed_ids: List[str],
        *,
        obs_before: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Public post-commit integration facade for harnesses and tests."""
        return self._post_commit_integration(
            committed_ids,
            obs_before or {},
            result or {},
        )

    def _available_functions_for_deliberation(self, obs_before: Dict[str, Any]) -> List[str]:
        api_raw = obs_before.get('novel_api', {}) if isinstance(obs_before.get('novel_api', {}), dict) else {}
        visible = [str(fn) for fn in list(api_raw.get('visible_functions', []) or []) if str(fn or '')]
        discovered = [str(fn) for fn in list(api_raw.get('discovered_functions', []) or []) if str(fn or '')]
        merged: List[str] = []
        for fn in visible + discovered:
            if fn and fn not in merged:
                merged.append(fn)
        return merged

    def _apply_deliberation_to_unified_context(self, frame: TickContextFrame, deliberation_result: Dict[str, Any]) -> None:
        unified = getattr(frame, 'unified_context', None)
        if not isinstance(unified, UnifiedCognitiveContext) or not isinstance(deliberation_result, dict):
            return
        workspace_patch: Dict[str, Any] = {}
        if isinstance(deliberation_result.get('ranked_candidate_hypothesis_objects'), list):
            competing_hypothesis_objects = [
                dict(item)
                for item in deliberation_result.get('ranked_candidate_hypothesis_objects', [])
                if isinstance(item, dict)
            ]
            if competing_hypothesis_objects:
                active_hypotheses_summary = summarize_cognitive_object_records(
                    competing_hypothesis_objects,
                    limit=8,
                )
                unified.active_hypotheses_summary = active_hypotheses_summary
                workspace_patch['object_workspace.competing_hypothesis_objects'] = competing_hypothesis_objects
                workspace_patch['object_workspace.active_hypotheses_summary'] = active_hypotheses_summary
        if isinstance(deliberation_result.get('ranked_candidate_hypotheses'), list):
            competing_hypotheses = [
                dict(item)
                for item in deliberation_result.get('ranked_candidate_hypotheses', [])
                if isinstance(item, dict)
            ]
            unified.competing_hypotheses = competing_hypotheses
            workspace_patch['object_workspace.competing_hypotheses'] = competing_hypotheses
        if isinstance(deliberation_result.get('ranked_candidate_tests'), list):
            candidate_tests = [dict(item) for item in deliberation_result.get('ranked_candidate_tests', []) if isinstance(item, dict)]
            unified.candidate_tests = candidate_tests
            workspace_patch['object_workspace.candidate_tests'] = candidate_tests
        if isinstance(deliberation_result.get('active_test_ids'), list):
            workspace_patch['object_workspace.active_tests'] = [
                str(item or '').strip()
                for item in deliberation_result.get('active_test_ids', [])
                if str(item or '').strip()
            ]
        if isinstance(deliberation_result.get('ranked_candidate_programs'), list):
            candidate_programs = [dict(item) for item in deliberation_result.get('ranked_candidate_programs', []) if isinstance(item, dict)]
            unified.candidate_programs = candidate_programs
            workspace_patch['object_workspace.candidate_programs'] = candidate_programs
        if isinstance(deliberation_result.get('ranked_candidate_outputs'), list):
            candidate_outputs = [dict(item) for item in deliberation_result.get('ranked_candidate_outputs', []) if isinstance(item, dict)]
            unified.candidate_outputs = candidate_outputs
            workspace_patch['object_workspace.candidate_outputs'] = candidate_outputs
        if isinstance(deliberation_result.get('ranked_discriminating_experiments'), list):
            experiments = [dict(item) for item in deliberation_result.get('ranked_discriminating_experiments', []) if isinstance(item, dict)]
            unified.ranked_discriminating_experiments = experiments
            workspace_patch['object_workspace.ranked_discriminating_experiments'] = experiments
        if isinstance(deliberation_result.get('posterior_summary'), dict):
            posterior_summary = dict(deliberation_result.get('posterior_summary', {}) or {})
            unified.posterior_summary = posterior_summary
            workspace_patch['object_workspace.posterior_summary'] = posterior_summary
        if isinstance(deliberation_result.get('budget'), dict):
            unified.deliberation_budget = dict(deliberation_result.get('budget', {}))
        if deliberation_result.get('mode'):
            unified.deliberation_mode = str(deliberation_result.get('mode') or unified.deliberation_mode or 'reactive')
        if isinstance(deliberation_result.get('deliberation_trace'), list):
            workspace_provenance = dict(unified.workspace_provenance or {})
            workspace_provenance['deliberation_trace_length'] = len(deliberation_result.get('deliberation_trace', []))
            workspace_provenance['deliberation_backend'] = str(deliberation_result.get('backend', '') or '')
            if isinstance(deliberation_result.get('control_policy'), dict):
                workspace_provenance['deliberation_control_strategy'] = str(
                    deliberation_result.get('control_policy', {}).get('strategy', '') or ''
                )
            unified.workspace_provenance = workspace_provenance
        if workspace_patch and hasattr(self, '_state_mgr'):
            self._state_mgr.update_state(
                workspace_patch,
                reason='reasoning:deliberation_context_update',
                module='core.reasoning',
            )

    def _run_deliberation_engine(
        self,
        *,
        obs_before: Dict[str, Any],
        surfaced: List[Any],
        continuity_snapshot: Dict[str, Any],
        frame: TickContextFrame,
        candidate_actions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        workspace = frame.unified_context.to_dict() if isinstance(getattr(frame, 'unified_context', None), UnifiedCognitiveContext) else {}
        available_functions = self._available_functions_for_deliberation(obs_before)
        result = self._deliberation_engine.deliberate(
            workspace=workspace,
            obs=obs_before,
            surfaced=surfaced,
            candidate_actions=candidate_actions,
            continuity_snapshot=continuity_snapshot,
            task_family=self._infer_task_family(obs_before),
            available_functions=available_functions,
            llm_client=self._resolve_llm_client("deliberation"),
            structured_answer_synthesizer=self._structured_answer_synthesizer,
        )
        self._last_deliberation_result = dict(result)
        self._apply_deliberation_to_unified_context(frame, result)
        return result

    def _build_legacy_decision_context(
        self,
        *,
        frame: TickContextFrame,
        unified_context: Optional[UnifiedCognitiveContext],
        obs_before: Optional[Dict[str, Any]],
        continuity_snapshot: Optional[Dict[str, Any]],
        surfaced: List[Any],
        plan_tick_meta: Dict[str, Any],
        recent_failures: int,
    ) -> Dict[str, Any]:
        return build_legacy_decision_context(
            self,
            frame=frame,
            unified_context=unified_context,
            obs_before=obs_before,
            continuity_snapshot=continuity_snapshot,
            surfaced=surfaced,
            plan_tick_meta=plan_tick_meta,
            recent_failures=recent_failures,
        )

    def _maybe_enrich_observation(self, obs: Any) -> Any:
        """Attach perception and world-model summaries to frame-bearing observations."""
        from core.orchestration.execution_control import infer_available_tools_from_observation, serialize_tool_spec

        enriched: Optional[Dict[str, Any]] = None
        available_tools_payload: Optional[List[Dict[str, Any]]] = None
        if isinstance(obs, dict):
            enriched = dict(obs)
        else:
            raw_payload = getattr(obs, 'raw', None)
            observation = getattr(obs, 'observation', None)
            observation_raw = getattr(observation, 'raw', None) if observation is not None else None
            direct_tools = getattr(obs, 'available_tools', None)
            nested_tools = getattr(observation, 'available_tools', None) if observation is not None else None
            candidate_tools = direct_tools if isinstance(direct_tools, list) else nested_tools
            if isinstance(candidate_tools, list):
                available_tools_payload = [serialize_tool_spec(item) for item in candidate_tools]
            if isinstance(observation_raw, dict) or isinstance(raw_payload, dict):
                enriched = dict(observation_raw or {})
                if isinstance(raw_payload, dict):
                    enriched.update(raw_payload)
        if enriched is not None:
            inference_source = dict(enriched)
            if available_tools_payload:
                inference_source['available_tools'] = [dict(item) for item in available_tools_payload]
            inferred_tools = infer_available_tools_from_observation(inference_source)
            if inferred_tools:
                available_tools_payload = inferred_tools
        if enriched is None:
            return obs
        if available_tools_payload:
            enriched['available_tools'] = available_tools_payload
            self._last_available_tools = [dict(item) for item in available_tools_payload]
        elif isinstance(enriched.get('available_tools'), list):
            serialized_tools = [serialize_tool_spec(item) for item in enriched.get('available_tools', [])]
            enriched['available_tools'] = serialized_tools
            self._last_available_tools = [dict(item) for item in serialized_tools]
        elif getattr(self, '_last_available_tools', None):
            enriched['available_tools'] = [dict(item) for item in list(self._last_available_tools or [])]
        perception_summary: Dict[str, Any] = (
            dict(enriched.get('perception', {}))
            if isinstance(enriched.get('perception', {}), dict)
            else {}
        )
        frame = enriched.get('frame')
        if isinstance(frame, list) and frame:
            try:
                if self._perception_bridge is None:
                    self._perception_bridge = build_optional_adapter("arc_agi3.perception_bridge")
                if self._perception_bridge is not None:
                    observed_perception = self._perception_bridge.observe(enriched)
                    if isinstance(observed_perception, dict) and observed_perception:
                        perception_summary = observed_perception
            except Exception:
                pass

        if perception_summary:
            enriched['perception'] = dict(perception_summary)
            self._last_perception_summary = dict(perception_summary)
        elif self._last_perception_summary and 'perception' not in enriched:
            enriched['perception'] = dict(self._last_perception_summary)

        self._surface_visual_feedback_fields(enriched, enriched.get('perception'))
        enriched['world_model'] = self._build_world_model_context(enriched.get('perception'))
        return enriched

    def _surface_visual_feedback_fields(
        self,
        obs: Dict[str, Any],
        perception_summary: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(obs, dict) or not isinstance(perception_summary, dict):
            return

        changed_pixels = self._safe_float(perception_summary.get('changed_pixel_count'), default=None)
        if changed_pixels is not None and 'changed_pixel_count' not in obs:
            obs['changed_pixel_count'] = changed_pixels

        changed_bbox = perception_summary.get('changed_bbox')
        if isinstance(changed_bbox, dict) and changed_bbox and 'changed_bbox' not in obs:
            obs['changed_bbox'] = dict(changed_bbox)

        hotspot = perception_summary.get('suggested_hotspot')
        if isinstance(hotspot, dict) and hotspot and 'suggested_hotspot' not in obs:
            obs['suggested_hotspot'] = dict(hotspot)

        if (self._safe_float(obs.get('changed_pixel_count'), default=0.0) or 0.0) > 0.0:
            obs['observation_changed'] = bool(obs.get('observation_changed', False) or True)

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bbox_area(bbox: Any) -> float:
        if not isinstance(bbox, dict):
            return 0.0
        width = CoreMainLoop._safe_float(bbox.get('width'), default=0.0) or 0.0
        height = CoreMainLoop._safe_float(bbox.get('height'), default=0.0) or 0.0
        if width <= 0.0 or height <= 0.0:
            return 0.0
        return float(width * height)

    def _extract_visual_feedback(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {
                'changed_pixel_count': 0.0,
                'grid_area': 0.0,
                'changed_ratio': 0.0,
                'changed_bbox_area': 0.0,
                'changed_bbox_ratio': 0.0,
                'hotspot_source': '',
            }

        perception = result.get('perception', {}) if isinstance(result.get('perception', {}), dict) else {}
        changed_pixels = self._safe_float(
            result.get('changed_pixel_count', perception.get('changed_pixel_count')),
            default=0.0,
        ) or 0.0
        changed_bbox = (
            result.get('changed_bbox')
            if isinstance(result.get('changed_bbox'), dict)
            else perception.get('changed_bbox')
        )
        changed_bbox_area = self._bbox_area(changed_bbox)
        grid_shape = perception.get('grid_shape', {}) if isinstance(perception.get('grid_shape', {}), dict) else {}
        grid_width = self._safe_float(grid_shape.get('width'), default=0.0) or 0.0
        grid_height = self._safe_float(grid_shape.get('height'), default=0.0) or 0.0
        grid_area = float(grid_width * grid_height) if grid_width > 0.0 and grid_height > 0.0 else 0.0
        changed_ratio = (changed_pixels / grid_area) if grid_area > 0.0 else 0.0
        changed_bbox_ratio = (changed_bbox_area / grid_area) if grid_area > 0.0 else 0.0
        hotspot = result.get('suggested_hotspot') if isinstance(result.get('suggested_hotspot'), dict) else perception.get('suggested_hotspot')
        hotspot_source = str((hotspot or {}).get('source', '') or '') if isinstance(hotspot, dict) else ''
        return {
            'changed_pixel_count': float(max(0.0, changed_pixels)),
            'grid_area': float(max(0.0, grid_area)),
            'changed_ratio': float(max(0.0, changed_ratio)),
            'changed_bbox_area': float(max(0.0, changed_bbox_area)),
            'changed_bbox_ratio': float(max(0.0, changed_bbox_ratio)),
            'hotspot_source': hotspot_source,
        }

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _surface_object_descriptors_from_obs(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(obs, dict):
            return []
        perception = obs.get('perception', {}) if isinstance(obs.get('perception', {}), dict) else {}
        salient_objects = list(perception.get('salient_objects', []) or []) if isinstance(perception.get('salient_objects', []), list) else []
        descriptors: List[Dict[str, Any]] = []
        for obj in salient_objects:
            if not isinstance(obj, dict):
                continue
            bbox = obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {}
            centroid = obj.get('centroid', {}) if isinstance(obj.get('centroid', {}), dict) else {}
            x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
            x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
            y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
            y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
            descriptors.append({
                'anchor_ref': str(obj.get('object_id', '') or ''),
                'color': self._safe_int(obj.get('color')),
                'bbox': {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max,
                    'width': int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1)),
                    'height': int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1)),
                },
                'centroid': {
                    'x': float(centroid.get('x', (x_min + x_max) / 2.0) or 0.0),
                    'y': float(centroid.get('y', (y_min + y_max) / 2.0) or 0.0),
                },
                'shape_labels': self._surface_shape_labels(obj),
                'boundary_contact': bool(obj.get('boundary_contact', False)),
                'salience_score': float(obj.get('salience_score', 0.0) or 0.0),
                'actionable_score': float(obj.get('actionable_score', 0.0) or 0.0),
            })
        return descriptors

    def _surface_shape_labels(self, obj: Dict[str, Any]) -> List[str]:
        semantic_rows = list(obj.get('semantic_candidates', []) or []) if isinstance(obj.get('semantic_candidates', []), list) else []
        labels: List[str] = []
        for row in semantic_rows:
            if not isinstance(row, dict):
                continue
            label = str(row.get('label', '') or '').strip()
            if label and label not in labels:
                labels.append(label)
        if labels:
            return labels

        bbox = obj.get('bbox', {}) if isinstance(obj.get('bbox', {}), dict) else {}
        x_min = int(bbox.get('x_min', bbox.get('col_min', 0)) or 0)
        x_max = int(bbox.get('x_max', bbox.get('col_max', x_min)) or x_min)
        y_min = int(bbox.get('y_min', bbox.get('row_min', 0)) or 0)
        y_max = int(bbox.get('y_max', bbox.get('row_max', y_min)) or y_min)
        width = int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1))
        height = int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1))
        area = int(obj.get('area', width * height) or width * height)
        fill_ratio = float(area / float(max(width * height, 1)))
        aspect_ratio = float(width / float(max(height, 1))) if height > 0 else 1.0
        boundary_contact = bool(obj.get('boundary_contact', False))
        if area <= 4:
            labels.append('token_like')
        if width > 0 and height > 0 and fill_ratio >= 0.70 and abs(width - height) <= 1:
            labels.append('block_like')
        if width > 0 and height > 0 and max(aspect_ratio, 1.0 / max(aspect_ratio, 1e-6)) >= 2.0 and fill_ratio >= 0.45:
            labels.append('bar_like')
        if boundary_contact:
            labels.append('boundary_structure')
        if not labels:
            labels.append('generic_object')
        return labels

    def _match_click_to_descriptor(
        self,
        descriptors: List[Dict[str, Any]],
        point: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if not descriptors:
            return None
        px, py = int(point[0]), int(point[1])
        containing: List[Tuple[int, float, Dict[str, Any]]] = []
        nearest: List[Tuple[float, int, Dict[str, Any]]] = []
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            bbox = descriptor.get('bbox', {}) if isinstance(descriptor.get('bbox', {}), dict) else {}
            x_min = int(bbox.get('x_min', 0) or 0)
            x_max = int(bbox.get('x_max', x_min) or x_min)
            y_min = int(bbox.get('y_min', 0) or 0)
            y_max = int(bbox.get('y_max', y_min) or y_min)
            area = max(1, int(bbox.get('width', max(1, x_max - x_min + 1)) or max(1, x_max - x_min + 1)) * int(bbox.get('height', max(1, y_max - y_min + 1)) or max(1, y_max - y_min + 1)))
            centroid = descriptor.get('centroid', {}) if isinstance(descriptor.get('centroid', {}), dict) else {}
            distance = abs(float(centroid.get('x', 0.0) or 0.0) - float(px)) + abs(float(centroid.get('y', 0.0) or 0.0) - float(py))
            if x_min <= px <= x_max and y_min <= py <= y_max:
                containing.append((area, distance, descriptor))
            else:
                nearest.append((distance, area, descriptor))
        if containing:
            containing.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            return containing[0][2]
        if nearest:
            nearest.sort(key=lambda item: (item[0], item[1], str(item[2].get('anchor_ref', '') or '')))
            if nearest[0][0] <= 2.5:
                return nearest[0][2]
        return None

    def _family_summary_from_descriptor(
        self,
        descriptor: Optional[Dict[str, Any]],
        action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = action.get('_candidate_meta', {}) if isinstance(getattr(action, 'get', None), type(dict.get)) and isinstance(action.get('_candidate_meta', {}), dict) else {}
        family = {
            'anchor_ref': '',
            'color': None,
            'shape_labels': [],
            'boundary_contact': False,
            'target_family': str(meta.get('target_family', '') or ''),
            'surface_click_role': str(meta.get('surface_click_role', '') or ''),
        }
        if isinstance(descriptor, dict):
            family['anchor_ref'] = str(descriptor.get('anchor_ref', '') or family['anchor_ref'])
            family['color'] = descriptor.get('color') if descriptor.get('color') is not None else family['color']
            family['shape_labels'] = [str(label) for label in list(descriptor.get('shape_labels', []) or []) if str(label)]
            family['boundary_contact'] = bool(descriptor.get('boundary_contact', family['boundary_contact']))
        if family['color'] is None and self._safe_int(meta.get('object_color')) is not None:
            family['color'] = int(self._safe_int(meta.get('object_color')) or 0)
        return family

    def _family_match_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0
        score = 0.0
        left_anchor = str(left.get('anchor_ref', '') or '')
        right_anchor = str(right.get('anchor_ref', '') or '')
        if left_anchor and right_anchor and left_anchor == right_anchor:
            score += 1.15
        left_color = self._safe_int(left.get('color'))
        right_color = self._safe_int(right.get('color'))
        if left_color is not None and right_color is not None:
            if left_color == right_color:
                score += 0.90
            else:
                score -= 0.20
        left_shapes = {str(item) for item in list(left.get('shape_labels', []) or []) if str(item)}
        right_shapes = {str(item) for item in list(right.get('shape_labels', []) or []) if str(item)}
        if left_shapes and right_shapes:
            overlap = len(left_shapes & right_shapes) / float(max(len(left_shapes | right_shapes), 1))
            score += 0.42 * overlap
        if str(left.get('target_family', '') or '') and str(left.get('target_family', '') or '') == str(right.get('target_family', '') or ''):
            score += 0.18
        if bool(left.get('boundary_contact', False)) and bool(right.get('boundary_contact', False)):
            score += 0.08
        return max(0.0, float(score))

    def _descriptor_affected_by_visual_change(
        self,
        descriptor: Dict[str, Any],
        changed_bbox: Optional[Dict[str, Any]],
        hotspot: Optional[Dict[str, Any]],
    ) -> bool:
        if not isinstance(descriptor, dict):
            return False
        bbox = descriptor.get('bbox', {}) if isinstance(descriptor.get('bbox', {}), dict) else {}
        x_min = int(bbox.get('x_min', 0) or 0)
        x_max = int(bbox.get('x_max', x_min) or x_min)
        y_min = int(bbox.get('y_min', 0) or 0)
        y_max = int(bbox.get('y_max', y_min) or y_min)
        if isinstance(changed_bbox, dict):
            cx_min = int(changed_bbox.get('x_min', 0) or 0)
            cx_max = int(changed_bbox.get('x_max', cx_min) or cx_min)
            cy_min = int(changed_bbox.get('y_min', 0) or 0)
            cy_max = int(changed_bbox.get('y_max', cy_min) or cy_min)
            if not (x_max < cx_min or cx_max < x_min or y_max < cy_min or cy_max < y_min):
                return True
        if isinstance(hotspot, dict):
            hx = self._safe_int(hotspot.get('x'))
            hy = self._safe_int(hotspot.get('y'))
            if hx is not None and hy is not None:
                if x_min <= hx <= x_max and y_min <= hy <= y_max:
                    return True
                centroid = descriptor.get('centroid', {}) if isinstance(descriptor.get('centroid', {}), dict) else {}
                distance = abs(float(centroid.get('x', 0.0) or 0.0) - float(hx)) + abs(float(centroid.get('y', 0.0) or 0.0) - float(hy))
                if distance <= 3.0:
                    return True
        return False

    def _progress_markers_show_positive_progress(self, progress_markers: List[Dict[str, Any]]) -> bool:
        task_progress_seen = False
        goal_stalled = False
        local_only_reaction = False
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            name = str(marker.get('name', '') or '')
            if name in {'goal_progressed', 'positive_reward'}:
                return True
            if name == 'task_progressed':
                task_progress_seen = True
            if name == 'goal_stalled':
                goal_stalled = True
            if name == 'local_only_reaction':
                local_only_reaction = True
            if name == 'terminal_reached' and bool(marker.get('success', False)):
                return True
        return bool(task_progress_seen and not goal_stalled and not local_only_reaction)

    def _derive_action_effect_signature(
        self,
        *,
        obs_before: Dict[str, Any],
        result: Dict[str, Any],
        action: Dict[str, Any],
        information_gain: float,
        progress_markers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return derive_action_effect_signature(
            self,
            obs_before=obs_before,
            result=result,
            action=action,
            information_gain=information_gain,
            progress_markers=progress_markers,
        )

    def _infer_level_goal_summary(
        self,
        *,
        obs_before: Dict[str, Any],
        world_model_summary: Optional[Dict[str, Any]] = None,
        task_frame_summary: Optional[Dict[str, Any]] = None,
        object_bindings_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = dict(world_model_summary or {})
        frame = dict(task_frame_summary or {})
        bindings = dict(object_bindings_summary or {})
        inferred = frame.get('inferred_level_goal', {}) if isinstance(frame.get('inferred_level_goal', {}), dict) else {}
        if inferred:
            return dict(inferred)
        try:
            if not bindings:
                bindings = dict(build_object_bindings(obs_before, summary) or {})
            if not frame:
                frame = dict(infer_task_frame(obs_before, summary, bindings, list(self._episode_trace[-8:])) or {})
            inferred = frame.get('inferred_level_goal', {}) if isinstance(frame.get('inferred_level_goal', {}), dict) else {}
            return dict(inferred)
        except Exception:
            return {}

    def _entry_has_local_anchor_signal(self, entry: Dict[str, Any]) -> bool:
        if not isinstance(entry, dict):
            return False
        if float(entry.get('information_gain', 0.0) or 0.0) >= 0.10:
            return True
        task_progress = entry.get('task_progress', {}) if isinstance(entry.get('task_progress', {}), dict) else {}
        if bool(task_progress.get('progressed', False)):
            return True
        if bool(entry.get('state_changed', False) or entry.get('observation_changed', False)):
            return True
        progress_markers = entry.get('progress_markers', []) if isinstance(entry.get('progress_markers', []), list) else []
        for marker in progress_markers:
            if not isinstance(marker, dict):
                continue
            if str(marker.get('name', '') or '') in {'task_progressed', 'goal_progressed', 'visual_change_detected', 'positive_reward'}:
                return True
        return False

    def _recent_goal_progress_state(
        self,
        episode_trace: List[Dict[str, Any]],
        *,
        limit: int = 12,
    ) -> Dict[str, Any]:
        return recent_goal_progress_state(self, episode_trace, limit=limit)

    def _recent_same_goal_anchor_streak(
        self,
        episode_trace: List[Dict[str, Any]],
        clicked_anchor_ref: str,
    ) -> int:
        anchor = str(clicked_anchor_ref or '').strip()
        if not anchor:
            return 0
        streak = 0
        for entry in reversed(list(episode_trace or [])):
            if not isinstance(entry, dict):
                continue
            assessment = entry.get('goal_progress_assessment', {}) if isinstance(entry.get('goal_progress_assessment', {}), dict) else {}
            prior_anchor = str(assessment.get('clicked_anchor_ref', '') or '')
            if not prior_anchor:
                clicked_family = entry.get('clicked_family', {}) if isinstance(entry.get('clicked_family', {}), dict) else {}
                prior_anchor = str(clicked_family.get('anchor_ref', '') or '')
            if prior_anchor != anchor:
                break
            streak += 1
        return streak

    def _derive_goal_bundle_state(
        self,
        *,
        goal_summary: Dict[str, Any],
        goal_progress_assessment: Dict[str, Any],
        recent_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(goal_summary, dict) or not goal_summary:
            return {}
        assessment = goal_progress_assessment if isinstance(goal_progress_assessment, dict) else {}
        recent = recent_state if isinstance(recent_state, dict) else {}
        goal_anchor_refs = {
            str(ref or '').strip()
            for ref in list(goal_summary.get('goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        engaged_anchor_refs = set(recent.get('engaged_goal_anchor_refs', set()) or set())
        engaged_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(assessment.get('engaged_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        necessary_anchor_refs = set(recent.get('necessary_anchor_refs', set()) or set())
        clicked_anchor_ref = str(assessment.get('clicked_anchor_ref', '') or '')
        if bool(assessment.get('necessary_signal', False)) and clicked_anchor_ref:
            necessary_anchor_refs.add(clicked_anchor_ref)
        necessary_but_insufficient_anchor_refs = set(recent.get('necessary_but_insufficient_anchor_refs', set()) or set())
        if bool(assessment.get('necessary_but_insufficient', False)) and clicked_anchor_ref:
            necessary_but_insufficient_anchor_refs.add(clicked_anchor_ref)
        local_only_anchor_refs = set(recent.get('local_only_anchor_refs', set()) or set())
        if bool(assessment.get('local_only_signal', False)) and clicked_anchor_ref:
            local_only_anchor_refs.add(clicked_anchor_ref)
        controller_anchor_refs = set(recent.get('controller_anchor_refs', set()) or set())
        controller_supported_goal_anchor_refs = set(
            recent.get('controller_supported_goal_anchor_refs', set()) or set()
        )
        controller_supported_goal_colors = set(
            recent.get('controller_supported_goal_colors', set()) or set()
        )
        controller_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(goal_summary.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_colors |= {
            self._safe_int(color)
            for color in list(goal_summary.get('controller_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        controller_anchor_ref = str(assessment.get('controller_anchor_ref', '') or '')
        if bool(assessment.get('controller_effect', False)) and controller_anchor_ref:
            controller_anchor_refs.add(controller_anchor_ref)
        controller_supported_goal_anchor_refs |= {
            str(ref or '').strip()
            for ref in list(assessment.get('controller_supported_goal_anchor_refs', []) or [])
            if str(ref or '').strip()
        }
        controller_supported_goal_anchor_refs &= goal_anchor_refs
        controller_supported_goal_colors |= {
            self._safe_int(color)
            for color in list(assessment.get('controller_supported_goal_colors', []) or [])
            if self._safe_int(color) is not None
        }
        coverage_target = int(goal_summary.get('coverage_target', 0) or 0)
        if coverage_target <= 0:
            coverage_target = max(1, len(goal_anchor_refs) or 1)
        requires_multi_anchor_coordination = bool(
            goal_summary.get('requires_multi_anchor_coordination', False)
            or coverage_target > 1
            or len(necessary_but_insufficient_anchor_refs) > 0
            or len(necessary_anchor_refs) >= 2
            or len(controller_supported_goal_anchor_refs) > 0
        )
        complementary_goal_anchor_refs = set(goal_anchor_refs - engaged_anchor_refs)
        complementary_goal_anchor_refs |= {
            ref
            for ref in controller_supported_goal_anchor_refs
            if ref and ref != controller_anchor_ref
        }
        active_combo_seed_anchor = str(recent.get('active_combo_seed_anchor', '') or '')
        active_seed_is_controller = bool(
            active_combo_seed_anchor and active_combo_seed_anchor in controller_anchor_refs
        )
        clicked_anchor_is_controller_supported = bool(
            clicked_anchor_ref and clicked_anchor_ref in controller_supported_goal_anchor_refs
        )
        if (
            bool(assessment.get('controller_effect', False))
            and bool(assessment.get('progressed', False))
            and controller_anchor_ref
            and requires_multi_anchor_coordination
        ):
            active_combo_seed_anchor = controller_anchor_ref
        if not active_combo_seed_anchor and necessary_but_insufficient_anchor_refs:
            active_combo_seed_anchor = sorted(necessary_but_insufficient_anchor_refs)[0]
        if (
            bool(assessment.get('necessary_but_insufficient', False))
            and clicked_anchor_ref
            and not (
                active_seed_is_controller
                and clicked_anchor_is_controller_supported
            )
        ):
            active_combo_seed_anchor = clicked_anchor_ref
        if active_combo_seed_anchor:
            complementary_goal_anchor_refs.discard(active_combo_seed_anchor)
        bundle_progress_fraction = min(
            1.0,
            len(engaged_anchor_refs & goal_anchor_refs) / float(max(coverage_target, 1))
        ) if goal_anchor_refs else 0.0
        coordination_pressure = 0.0
        if requires_multi_anchor_coordination:
            coordination_pressure = min(
                1.0,
                (len(necessary_but_insufficient_anchor_refs) * 0.45)
                + (len(complementary_goal_anchor_refs) * 0.18)
                + (0.16 if active_combo_seed_anchor else 0.0),
            )
        return {
            'goal_family': str(goal_summary.get('goal_family', '') or ''),
            'coverage_target': int(coverage_target),
            'requires_multi_anchor_coordination': bool(requires_multi_anchor_coordination),
            'engaged_anchor_refs': sorted(engaged_anchor_refs),
            'necessary_anchor_refs': sorted(necessary_anchor_refs),
            'necessary_but_insufficient_anchor_refs': sorted(necessary_but_insufficient_anchor_refs),
            'local_only_anchor_refs': sorted(local_only_anchor_refs),
            'controller_anchor_refs': sorted(controller_anchor_refs),
            'controller_supported_goal_anchor_refs': sorted(controller_supported_goal_anchor_refs),
            'controller_supported_goal_colors': sorted(controller_supported_goal_colors),
            'complementary_goal_anchor_refs': sorted(complementary_goal_anchor_refs)[:6],
            'active_combo_seed_anchor': active_combo_seed_anchor,
            'bundle_progress_fraction': round(float(bundle_progress_fraction), 4),
            'coordination_pressure': round(float(coordination_pressure), 4),
        }

    def _derive_goal_progress_assessment(
        self,
        *,
        goal_summary: Dict[str, Any],
        effect_trace: Dict[str, Any],
        information_gain: float,
        progress_markers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return derive_goal_progress_assessment(
            self,
            goal_summary=goal_summary,
            effect_trace=effect_trace,
            information_gain=information_gain,
            progress_markers=progress_markers,
        )

    def _build_goal_progress_markers(
        self,
        goal_progress_assessment: Dict[str, Any],
        fn_name: str,
    ) -> List[Dict[str, Any]]:
        if not isinstance(goal_progress_assessment, dict) or not goal_progress_assessment:
            return []
        markers: List[Dict[str, Any]] = []
        if bool(goal_progress_assessment.get('progressed', False)):
            markers.append({
                'name': 'goal_progressed',
                'function_name': fn_name,
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
                'goal_progress_score': float(goal_progress_assessment.get('goal_progress_score', 0.0) or 0.0),
                'goal_coverage_delta': int(goal_progress_assessment.get('goal_coverage_delta', 0) or 0),
            })
        if bool(goal_progress_assessment.get('stalled', False)):
            markers.append({
                'name': 'goal_stalled',
                'function_name': fn_name,
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'repeat_anchor_overrun': int(goal_progress_assessment.get('repeat_anchor_overrun', 0) or 0),
            })
        if bool(goal_progress_assessment.get('necessary_but_insufficient', False)):
            markers.append({
                'name': 'necessary_but_insufficient_anchor',
                'function_name': fn_name,
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
            })
        if bool(goal_progress_assessment.get('local_only_signal', False)):
            markers.append({
                'name': 'local_only_reaction',
                'function_name': fn_name,
                'clicked_anchor_ref': str(goal_progress_assessment.get('clicked_anchor_ref', '') or ''),
                'goal_family': str(goal_progress_assessment.get('goal_family', '') or ''),
            })
        return markers
    
    def _build_state_slice(self, obs_before: Optional[Dict[str, Any]] = None) -> StateSlice:
        """
        P3-D: Build a StateSlice for counterfactual reasoning.

        Collects minimal state needed for bounded counterfactual reasoning.
        """
        # Get established beliefs
        established = self._belief_ledger.get_established_beliefs()
        belief_summaries = [
            {'belief_id': b.belief_id, 'variable': b.variable_name, 'posterior': b.posterior, 'confidence': b.confidence}
            for b in established
        ]

        # Get active hypotheses
        hyp_summaries = []
        for h in self._hypotheses.iter_hypotheses():
            if h.status.value not in ('refuted', 'superseded'):
                hyp_summaries.append({
                    'id': h.id,
                    'claim': h.claim[:50],
                    'type': h.type,
                    'confidence': h.confidence,
                    'trigger_condition': h.trigger_condition,
                    'expected_transition': h.expected_transition,
                })

        # Get recent actions (last 5)
        recent_actions = []
        for entry in self._episode_trace[-5:]:
            act = entry.get('action', {})
            fn = act.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown') if isinstance(act.get('payload'), dict) else 'unknown'
            recent_actions.append(fn)

        # Determine reward trend
        recent_rewards = [e.get('reward', 0) for e in self._episode_trace[-5:]]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        reward_trend = "positive" if avg_reward > 0 else ("negative" if avg_reward < 0 else "neutral")
        perception = dict(getattr(self, '_last_perception_summary', {}) or {})
        obs_before = obs_before if isinstance(obs_before, dict) else {}
        obs_state_features = dict(obs_before.get('state_features', {}) or {})
        world_state = dict(obs_before.get('world_state', {}) or {})
        high_motion = (
            float(perception.get('camera_motion_score', 0.0) or 0.0) >= 0.6
            or any(b.get('variable') == 'observation_camera_motion' and b.get('posterior') == 'high_motion' for b in belief_summaries)
        )
        state_features = {
            'observation_mode': perception.get('coordinate_type', 'unknown'),
            'high_motion': bool(high_motion),
            'reward_trend': reward_trend,
            'active_hypotheses_count': len(hyp_summaries),
        }
        hidden_state_summary = self._hidden_state_tracker.summary() if getattr(self, '_hidden_state_tracker', None) is not None else {}
        if isinstance(hidden_state_summary, dict) and hidden_state_summary:
            state_features['hidden_state'] = dict(hidden_state_summary)
            state_features.setdefault('world_phase', hidden_state_summary.get('phase'))
        if isinstance(obs_state_features, dict):
            state_features.update(obs_state_features)
        if world_state.get('phase') is not None:
            state_features.setdefault('world_phase', world_state.get('phase'))
        if world_state.get('task_family') is not None:
            state_features.setdefault('task_family', world_state.get('task_family'))

        return StateSlice(
            available_functions=list(self._consumed_fns),
            established_beliefs=belief_summaries,
            active_hypotheses=hyp_summaries,
            recent_actions=recent_actions,
            current_reward_trend=reward_trend,
            state_features=state_features,
        )

    def _is_mechanism_formal_path_enabled(self) -> bool:
        """
        Experiment gate for mechanism-matching features.

        Ownership boundary:
        - Ablation switchboard (`_causal_ablation.enable_mechanism_matching`) decides
          whether mechanism matching is enabled at all.
        - Runtime budget may degrade heavy sub-steps, but must not disable this
          experiment layer.
        """
        return bool(getattr(self._causal_ablation, 'enable_mechanism_matching', True))

    def _is_mechanism_commit_budget_available(self) -> bool:
        """Runtime budget gate for heavy formal-commit writes (degrade-only, not experiment-off)."""
        return bool(getattr(self._runtime_budget, 'enable_mechanism_formal_path', True))

    def _extract_mechanism_candidates(self, latest_test: Optional[Dict[str, Any]]) -> List[MechanismHypothesis]:
        """Extract mechanism hypotheses from confirmed hypotheses, latest test, and short recent trace."""
        if not self._is_mechanism_formal_path_enabled():
            return []
        confirmed = list(self._hypotheses.get_confirmed()) if hasattr(self._hypotheses, 'get_confirmed') else []
        episode_trace = self._episode_trace if isinstance(self._episode_trace, list) else []
        return self._mechanism_extractor.extract(
            confirmed_hypotheses=confirmed,
            latest_test=latest_test if isinstance(latest_test, dict) else {},
            episode_trace=episode_trace,
            episode=self._episode,
            tick=self._tick,
            action_name_fn=self._extract_action_function_name,
        )

    def _commit_mechanism_candidates_formal(self, mechanisms: List[MechanismHypothesis]) -> List[str]:
        """Commit mechanism candidates through validator+committer formal path only."""
        if not self._is_mechanism_formal_path_enabled():
            return []
        if not self._is_mechanism_commit_budget_available():
            return []
        safe_mechanisms = [m for m in (mechanisms or []) if isinstance(m, MechanismHypothesis)]
        if not safe_mechanisms:
            return []
        return self._mechanism_writer.commit(safe_mechanisms, validator=self._validator, committer=self._committer)

    def _get_recent_mechanism_candidates(self, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Retrieve recent committed mechanism hypotheses for counterfactual context.

        Guardrail: when mechanism formal path is disabled, always return empty.
        """
        if not self._is_mechanism_formal_path_enabled():
            return []
        records: List[Dict[str, Any]] = []
        for obj in self._shared_store.retrieve(sort_by='confidence', limit=max(16, int(limit) * 4)):
            if not isinstance(obj, dict):
                continue
            if str(obj.get('memory_type', '')) != 'mechanism_hypothesis':
                continue
            content = obj.get('content', {}) if isinstance(obj.get('content', {}), dict) else {}
            if not content:
                continue
            mechanism = dict(content)
            mechanism.setdefault('confidence', float(obj.get('confidence', mechanism.get('confidence', 0.0)) or 0.0))
            mechanism.setdefault('status', mechanism.get('status', 'candidate'))
            records.append(mechanism)
            if len(records) >= limit:
                break
        return records

    @staticmethod
    def _make_retriever(arm_mode: str, object_store, seed: int) -> EpisodicRetriever:
        """Factory: create appropriate retriever subclass for arm_mode."""
        if arm_mode == ARM_MODE_WRONG_BINDING:
            return _WrongBindingEpisodicRetriever(object_store, seed)
        elif arm_mode == ARM_MODE_LOCAL_ONLY:
            return _LocalOnlyEpisodicRetriever(object_store, seed)
        elif arm_mode == ARM_MODE_SHUFFLED:
            return _ShuffledEpisodicRetriever(object_store, seed)
        elif arm_mode == ARM_MODE_FRESH:
            return _FreshEpisodicRetriever(object_store, seed)
        elif arm_mode == ARM_MODE_NO_TRANSFER:
            return _NoTransferEpisodicRetriever(object_store, seed)
        else:
            return EpisodicRetriever(object_store, seed)

    # =====================================================================
    # Public API
    # =====================================================================

    def run(self) -> dict:
        """Run all episodes and return audit."""
        for episode in range(1, self.max_episodes + 1):
            self.run_episode(episode)
        return self.audit()

    def run_episode(self, episode: int):
        """Run one episode."""
        self._prepare_episode(episode)

        if self.verbose:
            plan_summary = self._plan_state.get_plan_summary()
            print(f"  Episode {episode}: starting "
                  f"(objects: {self._shared_store.count_objects()}, "
                  f"hypotheses: {len(self._hypotheses.get_active())}, "
                  f"plan: {plan_summary.get('plan_id', 'none')[:20] if plan_summary.get('plan_id') else 'none'})")

        obs = self.observe()
        
        # Phase 1: Emit observation event to raw event log
        self._event_log.append(self._event_log_builder.observation(
            episode=episode,
            tick=0,
            data={'observation_keys': list(obs.keys()) if isinstance(obs, dict) else []},
            source_stage='observe',
        ))

        for tick in range(self.max_ticks):
            if isinstance(obs, dict) and bool(obs.get('terminal') or obs.get('done')):
                if self.verbose:
                    print(f"  Episode {episode}: terminal before tick {tick}")
                break
            self._tick = tick
            tick_result = self._tick_loop(obs)
            self._tick_log.append(tick_result)

            if tick_result.get('terminal'):
                if self.verbose:
                    print(f"  Episode {episode}: terminal at tick {tick}")
                break

            obs = tick_result.get('next_obs', obs)

        self._total_reward += self._episode_reward

        self._episode_lifecycle.on_episode_end(self, episode)

        retrieval_bundle = self._build_episode_retrieval_bundle(episode)
        consolidation_report = self._run_episode_consolidation(episode)

        # Phase 1: Emit episode_end event to raw event log
        self._event_log.append({
            'event_type': 'episode_end',
            'episode': episode,
            'tick': self._tick,
            'data': {
                'episode_reward': self._episode_reward,
                'tick_count': self._tick + 1,
                'consolidation_summary': {
                    'merged': consolidation_report.merged_count,
                    'promoted': consolidation_report.promoted_count,
                    'weakened': consolidation_report.weakened_count,
                    'retired': consolidation_report.retired_count,
                },
                'retrieval_bundle_summary': retrieval_bundle.to_dict() if retrieval_bundle else {},
            },
            'source_module': 'core',
            'source_stage': 'episode_end',
        })

        if self.verbose:
            print(f"  Episode {episode}: reward={self._episode_reward:.1f}, "
                  f"ticks={self._tick+1}, "
                  f"objects={self._committer.object_store.count_objects()}, "
                  f"hyp={len(self._hypotheses.get_active())}/{len(self._hypotheses.get_confirmed())}, "
                  f"tests={len(self._test_engine.get_tests())}")
        
        self._run_episode_procedure_induction(episode)
        persist_durable_object_records(
            self,
            reason='episode_end_durable_formal_memory_persist',
        )

    def _prepare_episode(self, episode: int) -> None:
        self._episode = episode
        self._tick = 0
        self._episode_reward = 0.0
        self._episode_trace = []
        self._consumed_fns = set()
        self._mechanism_runtime_state = {}
        self._last_mechanism_runtime_view = {
            'mechanism_hypotheses_summary': [],
            'mechanism_control_summary': {},
        }
        if self._hidden_state_tracker is not None:
            self._hidden_state_tracker.reset(episode)
        if getattr(self, "_persistent_object_identity_tracker", None) is not None:
            try:
                self._persistent_object_identity_tracker.reset()
            except Exception:
                pass

        if self._perception_bridge is not None:
            try:
                self._perception_bridge.reset()
            except Exception:
                pass
        if episode > 1:
            try:
                self._world.next_episode()
            except (AttributeError, TypeError):
                pass

        self._propagate_hypotheses()
        self._continuity.cleanup_agenda()
        self._resource_state.new_episode()
        self._ensure_episode_plan(episode)

    def _ensure_episode_plan(self, episode: int) -> None:
        if self._plan_state.has_plan:
            return
        continuity_snapshot = self._continuity.tick()
        top_goal = continuity_snapshot.get('top_goal') if continuity_snapshot else None
        if not top_goal:
            return

        task_family = self._infer_task_family()
        domain = str(self._world_provider_meta.get('runtime_env', '') or task_family or 'unknown')
        new_plan = self._objective_decomposer.decompose(
            goal=top_goal,
            context={
                'episode': episode,
                'tick': 0,
                'discovered_functions': list(self._consumed_fns),
                'active_hypotheses': self._hypotheses.get_active(),
                'max_ticks': self.max_ticks,
                'task_family': task_family,
                'domain': domain,
                'environment_tags': [task_family, domain],
            },
        )
        self._plan_state.set_plan(new_plan)
        if self.verbose:
            print(f"  Sprint 3: Created plan {new_plan.plan_id} with {len(new_plan.steps)} steps")

    def _build_episode_retrieval_bundle(self, episode: int):
        continuity_snapshot = self._continuity.tick()
        top_goal = continuity_snapshot.get('top_goal') if continuity_snapshot else None
        goal_id = str(getattr(top_goal, 'goal_id', '') or '').lower()
        goal_type = (
            'exploitation' if 'exploit' in goal_id else
            'testing' if 'confirm' in goal_id else
            'exploration'
        )
        teacher_exit_episode = getattr(self._grad_tracker, 'teacher_exit_episode', -1)
        return self._retrieval_bundle_builder.from_goal_type(
            goal_type=goal_type,
            query=f"episode_{episode}_summary",
            episode=episode,
            teacher_present=episode <= teacher_exit_episode,
        )

    def _run_episode_consolidation(self, episode: int):
        should_heavy = episode - self._last_heavy_consolidation_episode >= self._consolidation_episode_interval
        consolidator = self._run_heavy_consolidation if should_heavy else self._run_light_consolidation
        report = consolidator(
            object_store=self._shared_store,
            event_log=self._event_log,
            current_episode=episode,
        )
        report = commit_consolidation_candidates(self, report, episode=episode)
        if should_heavy:
            self._last_heavy_consolidation_episode = episode
        if self.verbose and should_heavy and report.merged_count > 0:
            print(
                f"  Consolidation: merged={report.merged_count}, "
                f"promoted={report.promoted_count}, "
                f"weakened={report.weakened_count}, "
                f"retired={report.retired_count}"
            )
        return report

    def _run_episode_procedure_induction(self, episode: int) -> None:
        if not self._procedure_enabled:
            return
        last_tick = self._tick_log[-1] if self._tick_log else {}
        episode_terminal_obs = last_tick.get('next_obs', {}) if isinstance(last_tick, dict) else {}
        accepted = self._procedure_pipeline.run_episode_induction(
            episode=episode,
            trace=list(self._episode_trace),
            audit=self._build_procedure_success_audit(),
            task_metadata={'task_family': self._infer_task_family(episode_terminal_obs)},
        )
        if accepted:
            self._procedure_promotion_log.append({
                'episode': episode,
                'promoted_or_added': list(accepted),
            })
    
    def _apply_learning_updates(self, assignments: List[Any]) -> None:
        self._learning_updates_runtime.run(ApplyLearningUpdatesInput(assignments=assignments))

    def _apply_learning_updates_impl(self, stage_input: ApplyLearningUpdatesInput) -> Dict[str, Any]:
        return run_apply_learning_updates(self, stage_input)

    def _learning_budget_remaining(self) -> int:
        return max(0, int(self._max_learning_updates_per_episode) - int(self._learning_updates_sent_this_episode))


    def _get_policy_profile(self) -> Dict[str, Any]:
        return self._meta_control.get_snapshot(self._episode, self._tick, context={'phase': 'policy_profile_read'}).to_policy_profile()

    def _get_representation_profile(self) -> Dict[str, Any]:
        return self._meta_control.get_snapshot(
            self._episode,
            self._tick,
            context={'phase': 'representation_profile_read'},
        ).to_representation_profile()

    def _ablation_flags_snapshot(self) -> Dict[str, Any]:
        cfg = getattr(self, '_causal_ablation', CausalLayerAblationConfig())
        unified_enabled = bool(getattr(cfg, 'enable_unified_context', True))
        ablation_mode = str(getattr(cfg, 'unified_context_ablation_mode', 'stripped') or 'stripped')
        if ablation_mode not in {'stripped', 'hard_off'}:
            ablation_mode = 'stripped'
        return {
            'enable_unified_context': unified_enabled,
            'unified_context_ablation_mode': ablation_mode,
            'unified_context_mode': 'full' if unified_enabled else ablation_mode,
            'enable_high_level_self_model': bool(getattr(cfg, 'enable_high_level_self_model', True)),
            'enable_representation_adaptation': bool(getattr(cfg, 'enable_representation_adaptation', True)),
            'enable_mechanism_matching': bool(getattr(cfg, 'enable_mechanism_matching', True)),
            'freeze_retrieval_pressure': bool(getattr(cfg, 'freeze_retrieval_pressure', False)),
        }

    def _sync_learning_state_from_policy_object(self) -> None:
        # compatibility shim; coordinator owns sync lifecycle
        return None

    def _get_control_profile_value(self, key: str, default: float) -> float:
        context = {'phase': 'control_read', 'key': key, 'default': default}
        snapshot = self._meta_control.get_snapshot(self._episode, self._tick, context=context)
        mapping = {
            'planner_bias': snapshot.planner_bias,
            'retrieval_aggressiveness': snapshot.retrieval_aggressiveness,
            'retrieval_pressure': snapshot.retrieval_pressure,
            'probe_bias': snapshot.probe_bias,
            'last_episode_reward': snapshot.last_episode_reward,
        }
        return float(mapping.get(key, default))

    def _bootstrap_policy_profile_object(self, episode: int) -> None:
        self._meta_control.bootstrap_policy_profile_object(episode=episode, tick=self._tick)

    def _apply_learning_policy_updates(self, episode: int) -> None:
        run_apply_learning_policy_updates(self, episode)
    def observe(self) -> dict:
        """Get current observation from world."""
        return self._maybe_enrich_observation(self._world.observe())

    def audit(self) -> dict:
        """Return comprehensive audit results."""
        return build_main_loop_audit(self)

    # =====================================================================
    # Internal Pipeline
    # =====================================================================

    def _propagate_hypotheses(self):
        """P0-2: Create hypotheses from committed objects at episode start."""
        for obj in self._committer.get_committed_objects(top_k=20):
            obj_id = obj.get('object_id', obj.get('id', ''))
            if not obj_id or self._hypotheses.has_object_binding(obj_id):
                continue
            created = self._hypotheses.create_from_object(obj, obj_id, tick=0, episode=self._episode)

            # Graduation tracking: notify when hypotheses are created
            # Issue 1 fix: Pass object_id so provenance has canonical identity
            for h in created:
                self._grad_tracker.on_object_created(h.id, created_round=0, episode=self._episode, object_id=obj_id)

            # P3-A: Emit hypothesis_created events
            for h in created:
                self._event_bus.emit(WorldModelEvent(
                    event_type=EventType.HYPOTHESIS_CREATED,
                    episode=self._episode,
                    tick=0,
                    data={
                        'hypothesis_id': h.id,
                        'hypothesis_type': h.type,
                        'object_id': obj_id,
                    },
                    source_stage='hypothesis_propagation',
                ))

            # Issue 2 fix: Wire belief creation into hypothesis creation path
            # This is the primary path for self-bootstrapping the belief layer
            for h in created:
                # Create a belief for each hypothesis created from an object
                if hasattr(h, 'type') and hasattr(h, 'claim'):
                    variable_name = f"hyp_{h.type}_{h.id[:8]}"
                    proposed_value = h.claim if hasattr(h, 'claim') else "unknown"
                    initial_confidence = h.confidence if hasattr(h, 'confidence') else 0.3

                    # BeliefUpdater creates and adds the belief to the ledger
                    self._belief_updater.create_belief_from_hypothesis(
                        hypothesis_id=h.id,
                        variable_name=variable_name,
                        proposed_value=proposed_value[:100] if len(proposed_value) > 100 else proposed_value,
                        initial_confidence=initial_confidence,
                    )

                    # Emit belief created event for audit
                    self._event_bus.emit(WorldModelEvent(
                        event_type=EventType.HYPOTHESIS_CREATED,  # Reuse event type for belief creation
                        episode=self._episode,
                        tick=0,
                        data={
                            'hypothesis_id': h.id,
                            'belief_created': True,
                            'variable_name': variable_name,
                            'source_stage': 'hypothesis_propagation',
                        },
                        source_stage='hypothesis_propagation',
                    ))

        # Mark competing pairs
        all_fn = [h for h in self._hypotheses.iter_hypotheses() if h.type == 'function_existence']
        for i in range(len(all_fn) - 1):
            self._hypotheses.mark_competing(all_fn[i].id, all_fn[i+1].id)

    # =============================================================================
    # TASK 0.1: SHADOW REFACTOR - Stage Helper Methods
    # These methods extract _tick_loop logic into discrete, testable stages.
    # The original _tick_loop remains unchanged during the shadow phase.
    # After verification, _tick_loop will delegate to _tick_loop_staged().
    # =============================================================================

    def _record_budgeted_llm_call(self, kind: str, route_name: str = "general") -> None:
        route_key = str(route_name or "general").strip() or "general"
        client = self._resolve_llm_client(route_key)
        route_metadata = {}
        if client is not None:
            route_metadata = dict(getattr(client, "_route_metadata", {}) or {})
        else:
            route_metadata = dict(self._ensure_model_router().decide(route_key).metadata or {})
        budget_status = self._llm_route_budget_status(
            route_name=route_key,
            route_metadata=route_metadata,
            prompt_tokens=0,
            reserved_response_tokens=0,
        )
        self._append_state_entry(
            "llm_advice_log",
            {
                "episode": self._episode,
                "tick": self._tick,
                "kind": kind,
                "entry": "budgeted_llm_call_intent",
                "route_name": route_key,
                "llm_available": client is not None,
                "budget_status": budget_status,
            },
        )

    def _record_llm_tick_summary(self) -> None:
        record_llm_tick_summary(
            episode=self._episode,
            tick=self._tick,
            llm_calls_this_tick=self._llm_calls_this_tick,
            state_writer=self._append_state_entry,
        )

    def _cooldown_ready(self, last_tick: int, cooldown_ticks: int) -> bool:
        return cooldown_ready(self._tick, last_tick, cooldown_ticks)

    def _compute_observation_signature(self, obs_before: dict) -> str:
        return compute_observation_signature(obs_before)

    def _extract_phase_hint(self, continuity_snapshot: dict) -> str:
        top_goal = continuity_snapshot.get('top_goal') if continuity_snapshot else None
        next_task = continuity_snapshot.get('next_task') if continuity_snapshot else None
        return "|".join([
            getattr(top_goal, 'goal_id', '') or '',
            getattr(next_task, 'task_id', '') or '',
        ])

    def _append_state_entry(self, list_name: str, entry: Dict[str, Any]) -> None:
        target = getattr(self, f"_{list_name}", None)
        if hasattr(target, 'append'):
            target.append(entry)

    def _extract_action_function_name(self, action: Optional[Dict[str, Any]], default: str = "wait") -> str:
        return extract_action_function_name(action, default=default)

    def _repair_action_function_name(self, action: Optional[Dict[str, Any]], selected_name: str) -> Dict[str, Any]:
        return repair_action_function_name(action, selected_name)

    def _json_safe(self, value: Any) -> Any:
        return json_safe(value)

    def _snapshot_candidate_list(self, candidates: Any) -> List[Dict[str, Any]]:
        if not isinstance(candidates, list):
            return []
        snapshots: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                snapshots.append({'raw': self._json_safe(candidate)})
                continue
            fn_name = self._extract_action_function_name(candidate, default='wait')
            snapshots.append({
                'kind': str(candidate.get('kind', '')),
                'function_name': fn_name,
                'is_wait': fn_name == 'wait' or candidate.get('kind') == 'wait',
                'candidate': self._json_safe(candidate),
            })
        return snapshots

    def _candidate_counts(self, candidates: Any) -> Tuple[int, int]:
        return candidate_counts(candidates)

    def _estimate_information_gain(self, obs_before: Dict[str, Any], result: Dict[str, Any], fn_name: str) -> float:
        before_api = obs_before.get('novel_api', {}) if isinstance(obs_before.get('novel_api', {}), dict) else {}
        after_api = result.get('novel_api', {}) if isinstance(result.get('novel_api', {}), dict) else {}
        before_discovered = set(before_api.get('discovered_functions', []) or [])
        after_discovered = set(after_api.get('discovered_functions', []) or [])
        delta_discovered = max(0, len(after_discovered - before_discovered))
        novelty_bonus = 0.0
        recent_functions = {
            self._extract_action_function_name(entry.get('action', {}), default='')
            for entry in self._episode_trace[-5:]
            if isinstance(entry, dict)
        }
        if fn_name and fn_name not in recent_functions:
            novelty_bonus = 0.2
        visual_feedback = self._extract_visual_feedback(result)
        changed_pixels = float(visual_feedback.get('changed_pixel_count', 0.0) or 0.0)
        changed_ratio = float(visual_feedback.get('changed_ratio', 0.0) or 0.0)
        changed_bbox_ratio = float(visual_feedback.get('changed_bbox_ratio', 0.0) or 0.0)
        hotspot_source = str(visual_feedback.get('hotspot_source', '') or '')

        visual_bonus = 0.0
        if changed_pixels > 0.0:
            visual_bonus += 0.04
        if changed_pixels >= 4.0:
            visual_bonus += 0.06
        if changed_ratio > 0.0:
            visual_bonus += min(0.18, changed_ratio * 2.2)
        if changed_bbox_ratio > 0.0:
            visual_bonus += min(0.12, changed_bbox_ratio * 0.24)
        if hotspot_source == 'changed_pixels':
            visual_bonus += 0.04

        return float(min(1.0, delta_discovered * 0.5 + novelty_bonus + visual_bonus))

    def _build_progress_markers(self, result: Dict[str, Any], reward: float, fn_name: str) -> List[Dict[str, Any]]:
        markers: List[Dict[str, Any]] = []
        if bool(result.get('success')):
            markers.append({'name': 'action_success', 'function_name': fn_name})
        if reward > 0.0:
            markers.append({'name': 'positive_reward', 'value': float(reward), 'function_name': fn_name})
        if bool(result.get('terminal') or result.get('done')):
            markers.append({'name': 'terminal_reached', 'success': bool(result.get('success'))})
        failure_reason = result.get('failure_reason')
        if isinstance(failure_reason, str) and failure_reason and failure_reason != 'none':
            markers.append({'name': 'failure_reason', 'value': failure_reason})
        visual_feedback = self._extract_visual_feedback(result)
        changed_pixels = float(visual_feedback.get('changed_pixel_count', 0.0) or 0.0)
        changed_ratio = float(visual_feedback.get('changed_ratio', 0.0) or 0.0)
        changed_bbox_ratio = float(visual_feedback.get('changed_bbox_ratio', 0.0) or 0.0)
        hotspot_source = str(visual_feedback.get('hotspot_source', '') or '')
        if changed_pixels > 0.0:
            markers.append({
                'name': 'visual_change_detected',
                'function_name': fn_name,
                'changed_pixel_count': round(changed_pixels, 4),
                'changed_ratio': round(changed_ratio, 4),
                'changed_bbox_ratio': round(changed_bbox_ratio, 4),
                'hotspot_source': hotspot_source,
            })
        if changed_pixels >= 16.0 or changed_ratio >= 0.02 or changed_bbox_ratio >= 0.08:
            markers.append({
                'name': 'task_progressed',
                'function_name': fn_name,
                'source': 'visual_change',
                'changed_pixel_count': round(changed_pixels, 4),
                'changed_ratio': round(changed_ratio, 4),
                'changed_bbox_ratio': round(changed_bbox_ratio, 4),
            })
        return markers

    def _infer_task_family(self, obs: Optional[Dict[str, Any]] = None) -> str:
        """Infer task family from runtime context instead of hard-coded constants."""
        if isinstance(obs, dict):
            candidates = [
                obs.get('task_family'),
                (obs.get('task') or {}).get('family') if isinstance(obs.get('task'), dict) else None,
                (obs.get('world_state') or {}).get('task_family') if isinstance(obs.get('world_state'), dict) else None,
                (obs.get('metadata') or {}).get('task_family') if isinstance(obs.get('metadata'), dict) else None,
            ]
            for value in candidates:
                if isinstance(value, str) and value.strip():
                    return value.strip()
        runtime_env = str(self._world_provider_meta.get('runtime_env', '') or '').strip()
        if runtime_env and runtime_env != 'injected':
            return runtime_env
        world_module = str(getattr(self._world.__class__, '__module__', '') or '')
        world_name = str(getattr(self._world.__class__, '__name__', '') or '')
        if 'hard_partial_observable' in world_module or 'HardPartialObservable' in world_name:
            return 'hard_partial_observable'
        return 'unknown'

    def _build_procedure_success_audit(self) -> Dict[str, Any]:
        """Build richer end-of-episode success audit for procedure mining/scoring."""
        if not self._tick_log:
            return {'success': False, 'solved': False}
        rewards = [float(entry.get('reward', 0.0) or 0.0) for entry in self._tick_log]
        terminal_rows = [
            entry for entry in self._tick_log
            if bool(entry.get('terminal'))
            or bool((entry.get('next_obs') or {}).get('terminal'))
            or bool((entry.get('next_obs') or {}).get('done'))
        ]
        solved = any(
            bool((entry.get('next_obs') or {}).get('solved'))
            or (
                bool((entry.get('next_obs') or {}).get('success'))
                and bool((entry.get('next_obs') or {}).get('terminal') or (entry.get('next_obs') or {}).get('done'))
            )
            for entry in self._tick_log
        )
        terminal_success = any(
            (
                bool((entry.get('next_obs') or {}).get('solved'))
                or bool((entry.get('next_obs') or {}).get('success'))
            ) and float(entry.get('reward', 0.0) or 0.0) > 0.0
            for entry in terminal_rows
        )
        positive_ticks = sum(1 for reward in rewards if reward > 0.0)
        return {
            'success': bool(solved or terminal_success or self._episode_reward > 0.0),
            'solved': bool(solved),
            'terminal_success': bool(terminal_success),
            'episode_reward': float(self._episode_reward),
            'positive_tick_ratio': float(positive_ticks / max(1, len(rewards))),
            'terminal_ticks': len(terminal_rows),
        }

    def _recent_reward_stagnation(self, window: Optional[int] = None) -> bool:
        history = self._episode_trace[-max(1, window or self._runtime_budget.reward_stagnation_window):]
        if len(history) < max(1, window or self._runtime_budget.reward_stagnation_window):
            return False
        rewards = [float(entry.get('reward', 0.0) or 0.0) for entry in history]
        return max(rewards) <= 0.0

    def _recent_action_repetition(self, window: Optional[int] = None) -> bool:
        history = self._episode_trace[-max(1, window or self._runtime_budget.repeated_action_window):]
        if len(history) < max(1, window or self._runtime_budget.repeated_action_window):
            return False
        action_names = []
        for entry in history:
            action = entry.get('action', {})
            payload = action.get('payload', {}) if isinstance(action, dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            fn_name = tool_args.get('function_name', '')
            if not fn_name:
                return False
            action_names.append(fn_name)
        return len(set(action_names)) == 1

    def _retrieval_candidate_margin(self, surfaced: List[Any]) -> float:
        if len(surfaced) < 2:
            return 1.0

        def _score(candidate: Any) -> float:
            for attr_name in ('relevance_score', 'retrieval_score', 'score'):
                value = getattr(candidate, attr_name, None)
                if isinstance(value, (int, float)):
                    return float(value)
            obj = getattr(candidate, 'object', {})
            if isinstance(obj, dict):
                confidence = obj.get('confidence', None)
                if isinstance(confidence, (int, float)):
                    return float(confidence)
            return 0.0

        ordered = sorted((_score(candidate) for candidate in surfaced), reverse=True)
        return ordered[0] - ordered[1]

    def _stage1_retrieval(self, obs_before: dict, ctx: dict, continuity_snapshot: dict) -> dict:
        stage_out = self._stage1_runtime.run(
            Stage1RetrievalInput(obs_before=obs_before, context=ctx, continuity_snapshot=continuity_snapshot)
        )
        return {
            'query': stage_out.query,
            'retrieve_result': stage_out.retrieve_result,
            'surfaced': stage_out.surfaced,
            'surfacing_protocol': stage_out.surfacing_protocol,
            'llm_retrieval_ctx': stage_out.llm_retrieval_ctx,
            'budget': stage_out.budget,
        }

    def _stage1_retrieval_impl(self, stage_input: Stage1RetrievalInput) -> dict:
        """
        Stage 1 (Retrieval): Build query → retrieve → surface → augment hypotheses.

        Returns dict with keys:
            query, retrieve_result, surfaced, llm_retrieval_ctx
        """
        return run_stage1_retrieval(self, stage_input)

    def _annotate_candidates_with_counterfactual(
        self,
        candidate_actions: List[Dict[str, Any]],
        continuity_snapshot: Optional[Dict[str, Any]] = None,
        obs_before: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Inject bounded counterfactual signals into candidate metadata for downstream scoring."""
        if not candidate_actions:
            return

        state_slice = self._build_state_slice(obs_before=obs_before)
        mechanism_candidates = self._get_recent_mechanism_candidates(limit=8)
        if len(state_slice.established_beliefs) < 1 and not mechanism_candidates:
            return
        transition_priors = self._build_world_model_transition_priors(getattr(self, '_last_perception_summary', {}))

        wait_action = None
        for action in candidate_actions:
            if isinstance(action, dict) and action.get('kind') == 'wait':
                wait_action = action
                break
        if wait_action is None:
            wait_action = {'kind': 'wait', 'payload': {}}

        for action in candidate_actions[:6]:
            if not isinstance(action, dict):
                continue
            kind = action.get('kind', 'call_tool')
            if kind == 'wait':
                continue
            payload = action.get('payload', {}) if isinstance(action.get('payload'), dict) else {}
            tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
            fn_name = tool_args.get('function_name', 'unknown')
            if not fn_name:
                continue

            cf_outcome = self._counterfactual.simulate_action_difference(
                state_slice,
                action,
                wait_action,
                context={
                    'goal_id': getattr((continuity_snapshot or {}).get('top_goal'), 'goal_id', None),
                    'episode': self._episode,
                    'tick': self._tick,
                    'state_features': getattr(state_slice, 'state_features', {}),
                    'transition_priors': transition_priors,
                    'mechanism_candidates': mechanism_candidates,
                },
            )

            meta = action.setdefault('_candidate_meta', {})
            if not isinstance(meta, dict):
                meta = {}
                action['_candidate_meta'] = meta

            preferred = str(cf_outcome.preferred_action or '')
            meta['counterfactual_confidence'] = cf_outcome.confidence.value
            meta['counterfactual_advantage'] = preferred == fn_name
            meta['counterfactual_delta'] = float(cf_outcome.estimated_delta or 0.0)
            meta['counterfactual_decision_path'] = cf_outcome.decision_path
            meta['counterfactual_rollout_trace'] = [
                dict(item) for item in list(getattr(cf_outcome, 'rollout_trace', []) or [])[:8]
                if isinstance(item, dict)
            ]
            meta['counterfactual_rollout_summary'] = dict(getattr(cf_outcome, 'rollout_summary', {}) or {})
            if isinstance(cf_outcome.mechanism_match, dict):
                meta['counterfactual_mechanism_id'] = cf_outcome.mechanism_match.get('mechanism_id')

            self._governance_log.append({
                'tick': self._tick,
                'episode': self._episode,
                'entry': 'counterfactual_candidate_annotation',
                'function_name': fn_name,
                'preferred_action': preferred,
                'confidence': cf_outcome.confidence.value,
                'advantage': meta['counterfactual_advantage'],
            })

    def _stage2_candidate_generation_substage(self, obs_before: dict, surfaced: list, continuity_snapshot: dict, frame: TickContextFrame) -> PlannerStageOutput:
        return self._stage2_candidate_runtime.run(
            Stage2CandidateGenerationInput(
                obs_before=obs_before,
                surfaced=surfaced,
                continuity_snapshot=continuity_snapshot,
                frame=frame,
            )
        )

    def _stage2_candidate_generation_substage_impl(self, stage_input: Stage2CandidateGenerationInput) -> PlannerStageOutput:
        return run_stage2_candidate_generation(self, stage_input)

    def _stage2_plan_constraints_substage(self, obs_before: Dict[str, Any], candidate_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._stage2_plan_constraints_runtime.run(
            Stage2PlanConstraintsInput(obs_before=obs_before, candidate_actions=candidate_actions)
        )

    def _stage2_plan_constraints_substage_impl(self, stage_input: Stage2PlanConstraintsInput) -> List[Dict[str, Any]]:
        return run_stage2_plan_constraints(self, stage_input)

    def _stage2_self_model_suppression_substage(
        self,
        candidate_actions: List[Dict[str, Any]],
        continuity_snapshot: Dict[str, Any],
        obs_before: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self._stage2_self_model_runtime.run(
            Stage2SelfModelSuppressionInput(
                candidate_actions=candidate_actions,
                continuity_snapshot=continuity_snapshot,
                obs_before=obs_before,
            )
        )

    def _stage2_self_model_suppression_substage_impl(
        self,
        stage_input: Stage2SelfModelSuppressionInput,
    ) -> List[Dict[str, Any]]:
        return run_stage2_self_model_suppression(self, stage_input)

    def _stage2_prediction_context_bridge_substage(self, bridge: DecisionBridgeInput) -> Dict[str, Any]:
        return self._stage2_prediction_bridge_runtime.run(Stage2PredictionBridgeInput(bridge=bridge))

    def _stage2_prediction_runtime_substage(self, candidate_actions: List[Dict[str, Any]]) -> None:
        materialize_stage2_prediction_fallback(self, candidate_actions)

    def _stage2_prediction_context_bridge_substage_impl(self, stage_input: Stage2PredictionBridgeInput) -> Dict[str, Any]:
        return run_stage2_prediction_bridge(self, stage_input)

    def _extract_bridge_deliberation_result(self, bridge: Any) -> Dict[str, Any]:
        raw_result = getattr(bridge, 'deliberation_result', {})
        return dict(raw_result) if isinstance(raw_result, dict) else {}

    def _stage2_governance_substage(
        self,
        action_to_use: Dict[str, Any],
        candidate_actions: List[Dict[str, Any]],
        arm_meta: Dict[str, Any],
        continuity_snapshot: Dict[str, Any],
        obs_before: Dict[str, Any],
        decision_outcome: Any,
        frame: TickContextFrame,
    ) -> GovernanceStageOutput:
        return self._stage2_governance_runtime.run(
            Stage2GovernanceSubstageInput(
                action_to_use=action_to_use,
                candidate_actions=candidate_actions,
                arm_meta=arm_meta,
                continuity_snapshot=continuity_snapshot,
                obs_before=obs_before,
                decision_outcome=decision_outcome,
                frame=frame,
            )
        )

    def _stage2_governance_substage_impl(self, stage_input: Stage2GovernanceSubstageInput) -> GovernanceStageOutput:
        return run_stage2_governance(self, stage_input)

    def _stage2_action_generation(self, obs_before: dict, surfaced: list, continuity_snapshot: dict, frame: Optional[TickContextFrame] = None) -> dict:
        return run_stage2_action_generation(
            self,
            obs_before,
            surfaced,
            continuity_snapshot,
            frame=frame,
        )

    def _counterfactual_rank_candidates(self, candidate_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return rank_counterfactual_candidates(candidate_actions)

    def _stage3_execution(self, action_to_use: dict, query, obs_before: dict) -> dict:
        out = self._stage3_runtime.run(
            Stage3ExecutionInput(action_to_use=action_to_use, query=query, obs_before=obs_before)
        )
        return {'result': out.result, 'reward': out.reward}

    def _resolve_action_for_execution(self, action_to_use: Dict[str, Any], obs_before: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_action_for_execution(self, action_to_use, obs_before)

    def _stage3_execution_impl(self, stage_input: Stage3ExecutionInput) -> dict:
        return run_stage3_execution(self, stage_input)

    def _collect_executable_function_names(self, obs_before: dict) -> Set[str]:
        return collect_executable_function_names(obs_before)

    def _is_trackable_executable_function(self, function_name: str, executable_functions: Set[str]) -> bool:
        return is_trackable_executable_function(function_name, executable_functions)

    def _record_memory_consumption_proof(self, action_to_use: dict, query: Any, result: dict) -> None:
        record_memory_consumption_proof(self, action_to_use, query, result)

    def _stage4_testing(self, obs_before: dict, surfaced: list, action_to_use: dict, result: dict, frame: TickContextFrame) -> None:
        runtime_outcome: TestingRecoveryResult = self._testing_recovery_runtime.run_testing_and_recovery(
            obs_before=obs_before,
            surfaced=surfaced,
            action_to_use=action_to_use,
            result=result,
            frame=frame,
        )
        self._testing_recovery_runtime.apply_effects(runtime_outcome.effects, runtime_outcome.telemetry)
        probe_patch = runtime_outcome.state_patch.get('pending_probe_patch')
        if isinstance(probe_patch, dict) and probe_patch.get('clear_pending_probe'):
            self._pending_recovery_probe = None
        elif isinstance(probe_patch, dict):
            self._pending_recovery_probe = probe_patch
        replan_patch = runtime_outcome.state_patch.get('pending_replan_patch')
        if isinstance(replan_patch, dict):
            self._pending_replan = replan_patch

    def _execute_test_action(self, test: Any) -> dict:
        return self._testing_recovery_runtime._execute_test_action(test)

    def _teacher_allows_intervention(self) -> bool:
        policy = TeacherInterventionPolicy(
            experiment_mode=str(getattr(self, '_teacher_experiment_mode', 'full')),
            episode=int(self._episode),
            tick=int(self._tick),
        )
        return policy.allows_intervention()

    def _apply_probe_policy_bias(self, test_advice: Dict[str, Any]) -> Dict[str, Any]:
        return self._testing_recovery_runtime._apply_probe_policy_bias(test_advice)

    def _build_action_id(self, action: dict) -> str:
        payload = action.get('payload', {}) if isinstance(action, dict) else {}
        tool_args = payload.get('tool_args', {}) if isinstance(payload, dict) else {}
        fn_name = tool_args.get('function_name', 'wait') if isinstance(tool_args, dict) else 'wait'
        kwargs = tool_args.get('kwargs', {}) if isinstance(tool_args, dict) else {}
        source = action.get('_source', 'unknown') if isinstance(action, dict) else 'unknown'
        action_identity = extract_action_identity(action, include_function_fallback=True) or fn_name
        kwargs_hash = hashlib.sha1(repr(kwargs).encode('utf-8', errors='ignore')).hexdigest()[:10]
        action_id = f"{self._episode}:{self._tick}:{source}:{action_identity}:{kwargs_hash}"
        if isinstance(action, dict):
            action['_action_id'] = action_id
        return action_id

    def _prediction_bundle_to_dict(self, bundle) -> dict:
        return prediction_bundle_to_dict(bundle)

    def _record_prediction_trace(self, bundle: Any, outcome: Any, error: Any) -> None:
        record_prediction_trace(
            episode=int(self._episode),
            tick=int(self._tick),
            prediction_trace_log=self._prediction_trace_log,
            bundle=bundle,
            outcome=outcome,
            error=error,
        )

    def _apply_prediction_error_feedback(self, error: Any, *, bundle: Any = None, outcome: Any = None) -> None:
        feedback = apply_prediction_error_feedback(
            episode=int(self._episode),
            tick=int(self._tick),
            error=error,
            meta_control=self._meta_control,
            governance_log=self._governance_log,
            prediction_miss_feedback=self._prediction_miss_feedback,
            prediction_positive_miss_streak=self._prediction_positive_miss_streak,
            bundle=bundle,
            outcome=outcome,
        )
        self._prediction_positive_miss_streak = feedback.prediction_positive_miss_streak
        if feedback.pending_replan_patch is not None:
            self._pending_replan = feedback.pending_replan_patch

    def _build_self_model_prediction_summary(self) -> dict:
        return build_self_model_prediction_summary(
            causal_ablation=getattr(self, '_causal_ablation', None),
            self_model_facade=getattr(self, '_self_model_facade', None),
            reliability_tracker=getattr(self, '_reliability_tracker', None),
            resource_state=getattr(self, '_resource_state', None),
        )

    def _build_recovery_prediction_context(self) -> dict:
        return build_recovery_prediction_context(
            pending_recovery_probe=getattr(self, '_pending_recovery_probe', None),
            pending_replan=getattr(self, '_pending_replan', None),
            recovery_log=getattr(self, '_recovery_log', []),
        )

    def _rank_probe_candidates_by_prediction(self, probe_candidates, obs_before, surfaced, frame: TickContextFrame):
        return rank_probe_candidates_by_prediction(self, probe_candidates, obs_before, surfaced, frame)

    def _stage5_evidence_commit(self, action_to_use: dict, result: dict) -> dict:
        out = self._stage5_runtime.run(Stage5EvidenceCommitInput(action_to_use=action_to_use, result=result))
        return {'validated': out.validated, 'committed_ids': out.committed_ids}

    def _stage5_evidence_commit_impl(self, stage_input: Stage5EvidenceCommitInput) -> dict:
        """
        Stage 5 (Evidence & Commit): Extract evidence → validate → commit.

        Returns dict with keys:
            validated, committed_ids
        """
        return run_stage5_evidence_commit(self, stage_input)

    def _stage6_post_commit(self, committed_ids: list, obs_before: dict, result: dict, action_to_use: dict, reward: float) -> None:
        self._stage6_runtime.run(Stage6PostCommitInput(
            committed_ids=list(committed_ids),
            obs_before=obs_before,
            result=result,
            action_to_use=action_to_use,
            reward=reward,
        ))

    def _stage6_post_commit_impl(self, stage_input: Stage6PostCommitInput) -> Dict[str, Any]:
        """
        Stage 6 (Post-Commit): Post-commit integration, graduation tracking, state update.
        """
        return run_stage6_post_commit(self, stage_input)

    def _collect_outcome_learning_signal(self, action_to_use: Dict[str, Any], obs_before: Dict[str, Any], result: Dict[str, Any], reward: float) -> None:
        collect_outcome_learning_signal(
            self,
            action_to_use=action_to_use,
            obs_before=obs_before,
            result=result,
            reward=reward,
        )

    def _latest_governance_entry_for_tick(self) -> Dict[str, Any]:
        for row in reversed(list(getattr(self, '_governance_log', [])[-40:])):
            if not isinstance(row, dict):
                continue
            if int(row.get('episode', -1)) != int(self._episode) or int(row.get('tick', -1)) != int(self._tick):
                continue
            if row.get('reason') or row.get('selected_name') or row.get('selected') or row.get('entry') in {
                'counterfactual_adoption_metric',
                'prediction_high_error_retrieval_pressure',
                'prediction_replan_hint',
            }:
                return row
        return {}

    def _latest_plan_lookahead_telemetry(self) -> Dict[str, Any]:
        payload = self._last_planner_runtime_payload if isinstance(self._last_planner_runtime_payload, dict) else {}
        if (
            isinstance(payload, dict)
            and int(payload.get('episode', -1)) == int(self._episode)
            and int(payload.get('tick', -1)) == int(self._tick)
        ):
            telemetry = payload.get('telemetry', {}) if isinstance(payload.get('telemetry', {}), dict) else {}
            lookahead = telemetry.get('plan_lookahead', {}) if isinstance(telemetry.get('plan_lookahead', {}), dict) else {}
            if lookahead:
                return dict(lookahead)
        for row in reversed(list(getattr(self, '_planner_runtime_log', [])[-20:])):
            if not isinstance(row, dict):
                continue
            if int(row.get('episode', -1)) != int(self._episode) or int(row.get('tick', -1)) != int(self._tick):
                continue
            telemetry = row.get('telemetry', {}) if isinstance(row.get('telemetry', {}), dict) else {}
            lookahead = telemetry.get('plan_lookahead', {}) if isinstance(telemetry.get('plan_lookahead', {}), dict) else {}
            if lookahead:
                return dict(lookahead)
        return {}

    def _classify_retention_failure(
        self,
        *,
        action_to_use: Dict[str, Any],
        function_name: str,
        result: Dict[str, Any],
        reward: float,
        prediction_mismatch: float,
        task_family: str,
        phase: str,
        observation_mode: str,
        resource_band: str,
        action_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mismatch = max(0.0, min(1.0, float(prediction_mismatch or 0.0)))
        lookahead = self._latest_plan_lookahead_telemetry()
        governance_entry = self._latest_governance_entry_for_tick()
        meta = dict(action_meta or {})
        forced_replan_events = [
            str(event or '')
            for event in list(lookahead.get('forced_replan_events', []) or [])
            if str(event or '')
        ]
        forced_event_set = set(forced_replan_events)
        persistence_ratio = max(0.0, min(1.0, float(lookahead.get('rollout_branch_persistence_ratio', 0.0) or 0.0)))
        governance_reason = str(governance_entry.get('reason', '') or '')
        raw_counterfactual_confidence = meta.get('counterfactual_confidence', 0.0)
        if isinstance(raw_counterfactual_confidence, (int, float)):
            counterfactual_confidence = max(0.0, min(1.0, float(raw_counterfactual_confidence)))
        else:
            counterfactual_confidence = {
                'high': 0.9,
                'medium': 0.6,
                'low': 0.3,
            }.get(str(raw_counterfactual_confidence or '').strip().lower(), 0.0)
        counterfactual_advantage = bool(meta.get('counterfactual_advantage', False))
        action_name = str(function_name or self._extract_action_function_name(action_to_use, default='wait') or 'wait')

        failure_type = ''
        severity = 0.0
        if (
            (not counterfactual_advantage and counterfactual_confidence >= 0.72)
            or 'counterfactual_oppose' in governance_reason
        ) and (float(reward) < 0.0 or mismatch >= 0.45):
            failure_type = 'governance_overrule_misfire'
            severity = max(mismatch, 0.35 + counterfactual_confidence * 0.55)
        elif (
            'wm_branch_persistence_replan' in forced_event_set
            or persistence_ratio <= 0.30
            or str(lookahead.get('rollout_final_phase', '') or '') == 'disrupted'
        ):
            failure_type = 'branch_persistence_collapse'
            severity = max(mismatch * 0.55, 1.0 - persistence_ratio)
            if str(lookahead.get('rollout_final_phase', '') or '') == 'disrupted':
                severity = max(severity, 0.68)
        elif forced_event_set.intersection({'wm_branch_salvage_replan', 'wm_belief_branch_replan', 'wm_rollout_value_replan', 'wm_value_drop_replan'}):
            failure_type = 'planner_target_switch'
            severity = max(0.38, mismatch * 0.60 + min(0.3, len(forced_event_set) * 0.08))
        elif mismatch >= 0.35 or 'prediction_high_error_retrieval_pressure' == str(governance_entry.get('entry', '') or ''):
            failure_type = 'prediction_drift'
            severity = mismatch

        severity = max(0.0, min(1.0, severity))
        if not failure_type or (float(reward) >= 0.0 and mismatch < 0.35 and not forced_event_set):
            return {'failure_type': '', 'severity': 0.0, 'context': {}}

        base_context_key = (
            f"task_family={str(task_family or 'unknown')}|phase={str(phase or 'unknown')}"
            f"|observation_mode={str(observation_mode or 'unknown')}|resource_band={str(resource_band or 'normal')}"
        )
        strategy_mode_hint = 'recover'
        if failure_type in {'prediction_drift', 'governance_overrule_misfire'}:
            strategy_mode_hint = 'verify'
        elif failure_type == 'planner_target_switch':
            strategy_mode_hint = 'recover'
        context = {
            'context_key': f"{base_context_key}|failure={failure_type}",
            'base_context_key': base_context_key,
            'failure_type': failure_type,
            'severity': severity,
            'strategy_mode_hint': strategy_mode_hint,
            'branch_budget_hint': 2 if failure_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0,
            'verification_budget_hint': 2 if failure_type in {'prediction_drift', 'governance_overrule_misfire'} else (1 if failure_type in {'branch_persistence_collapse', 'planner_target_switch'} else 0),
            'forced_replan_events': list(forced_replan_events),
            'rollout_branch_persistence_ratio': persistence_ratio,
            'rollout_branch_id': str(lookahead.get('rollout_branch_id', '') or ''),
            'rollout_branch_target_phase': str(lookahead.get('rollout_branch_target_phase', '') or ''),
            'rollout_final_phase': str(lookahead.get('rollout_final_phase', '') or ''),
            'governance_reason': governance_reason,
            'selected_name': str(governance_entry.get('selected_name', governance_entry.get('selected', action_name)) or action_name),
            'counterfactual_confidence': counterfactual_confidence,
            'counterfactual_advantage': counterfactual_advantage,
        }
        return {'failure_type': failure_type, 'severity': severity, 'context': context}

    def _commit_learning_updates(self, updates: List[Any]) -> int:
        return commit_learning_updates(self, updates)

    def _refresh_learning_policy_snapshot(self) -> None:
        objects = []
        for obj in self._shared_store.iter_objects(limit=500):
            if not isinstance(obj, dict):
                continue
            if str(obj.get('memory_type', '')) != 'learning_update':
                continue
            if obj.get('status') == 'invalidated':
                continue
            objects.append(obj)
        self._learning_policy_snapshot = aggregate_learning_updates(objects)
        if hasattr(self, '_reliability_tracker') and hasattr(self._reliability_tracker, 'synchronize_failure_preference_learning'):
            self._reliability_tracker.synchronize_failure_preference_learning(
                self._learning_policy_snapshot.get('failure_preference_policy', {})
                if isinstance(self._learning_policy_snapshot.get('failure_preference_policy', {}), dict)
                else {}
            )

    @staticmethod
    def _clamp_learning_signal(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return max(minimum, min(maximum, float(default)))

    @staticmethod
    def _learning_merge_ordered_lists(*values: Any) -> List[str]:
        merged: List[str] = []
        seen = set()
        for value in values:
            pool = value if isinstance(value, list) else []
            for item in pool:
                text = str(item or '').strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                merged.append(text)
        return merged

    @staticmethod
    def _is_learning_verification_function(fn_name: str) -> bool:
        name = str(fn_name or '').strip().lower()
        return bool(name) and any(token in name for token in ('probe', 'inspect', 'verify', 'check', 'test'))

    def _learning_resource_band(self, decision_context: Dict[str, Any]) -> str:
        if not isinstance(decision_context, dict):
            return 'normal'
        explicit = str(decision_context.get('resource_band', '') or '').strip().lower()
        if explicit in {'tight', 'normal'}:
            return explicit
        pressure = str(decision_context.get('resource_pressure', '') or '').strip().lower()
        if pressure in {'high', 'tight', 'critical'}:
            return 'tight'
        if pressure in {'normal', 'low', 'relaxed'}:
            return 'normal'
        self_model_summary = decision_context.get('self_model_summary', {}) if isinstance(decision_context.get('self_model_summary', {}), dict) else {}
        fallback = str(self_model_summary.get('resource_tightness', '') or '').strip().lower()
        if fallback in {'high', 'tight', 'critical'}:
            return 'tight'
        return 'normal'

    def _learning_world_model_competition_profile(
        self,
        decision_context: Dict[str, Any],
        *,
        candidate_function_universe: List[str],
    ) -> Dict[str, Any]:
        wm_control = WorldModelControlProtocol.from_context(decision_context if isinstance(decision_context, dict) else {})
        latent_branches = list(wm_control.latent_branches or [])
        dominant_branch_id = str(wm_control.dominant_branch_id or '').strip()
        dominant_branch: Dict[str, Any] = {}
        for row in latent_branches:
            if not isinstance(row, dict):
                continue
            if dominant_branch_id and str(row.get('branch_id', '') or '').strip() == dominant_branch_id:
                dominant_branch = dict(row)
                break
        if not dominant_branch and latent_branches and isinstance(latent_branches[0], dict):
            dominant_branch = dict(latent_branches[0])

        dominant_branch_id = str(dominant_branch.get('branch_id', dominant_branch_id) or dominant_branch_id or '')
        dominant_branch_confidence = self._clamp_learning_signal(dominant_branch.get('confidence', 0.0), 0.0, 1.0, 0.0)
        required_probes = self._learning_merge_ordered_lists(list(wm_control.required_probes or []))
        dominant_anchor_functions = [
            fn_name
            for fn_name in self._learning_merge_ordered_lists(
                dominant_branch.get('anchor_functions', []),
                dominant_branch.get('anchored_functions', []),
            )
            if fn_name in candidate_function_universe
        ]
        dominant_risky_functions = [
            fn_name
            for fn_name in self._learning_merge_ordered_lists(dominant_branch.get('risky_functions', []))
            if fn_name in candidate_function_universe
        ]
        probe_pressure = min(1.0, len(required_probes) / 3.0)
        latent_instability = self._clamp_learning_signal(
            (1.0 - dominant_branch_confidence) * 0.38
            + self._clamp_learning_signal(wm_control.hidden_drift_score, 0.0, 1.0, 0.0) * 0.28
            + self._clamp_learning_signal(wm_control.hidden_uncertainty_score, 0.0, 1.0, 0.0) * 0.22
            + self._clamp_learning_signal(wm_control.state_shift_risk, 0.0, 1.0, 0.0) * 0.12,
            0.0,
            1.0,
            0.0,
        )
        probe_pressure_active = (
            probe_pressure >= 0.34
            and (
                self._clamp_learning_signal(wm_control.control_trust, 0.0, 1.0, 0.5) <= 0.52
                or self._clamp_learning_signal(wm_control.transition_confidence, 0.0, 1.0, 0.5) <= 0.48
                or self._clamp_learning_signal(wm_control.hidden_drift_score, 0.0, 1.0, 0.0) >= 0.55
                or self._clamp_learning_signal(wm_control.hidden_uncertainty_score, 0.0, 1.0, 0.0) >= 0.62
                or latent_instability >= 0.58
                or self._clamp_learning_signal(wm_control.state_shift_risk, 0.0, 1.0, 0.0) >= 0.58
            )
        )
        return {
            'required_probes': required_probes,
            'probe_pressure': float(probe_pressure),
            'probe_pressure_active': bool(probe_pressure_active),
            'latent_instability': float(latent_instability),
            'dominant_branch_id': dominant_branch_id,
            'dominant_anchor_functions': dominant_anchor_functions,
            'dominant_risky_functions': dominant_risky_functions,
        }

    def _merge_learned_failure_strategy_profile(
        self,
        existing: Dict[str, Any],
        learned: Dict[str, Any],
        *,
        competition: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(existing) if isinstance(existing, dict) else {}
        if not isinstance(learned, dict) or not learned:
            return merged

        merged['strategy_mode_hint'] = str(
            learned.get('strategy_mode_hint', merged.get('strategy_mode_hint', 'balanced')) or merged.get('strategy_mode_hint', 'balanced')
        )
        merged['branch_budget_hint'] = max(
            int(merged.get('branch_budget_hint', 0) or 0),
            int(learned.get('branch_budget_hint', 0) or 0),
            2 if float(competition.get('latent_instability', 0.0) or 0.0) >= 0.58 else 0,
        )
        merged['verification_budget_hint'] = max(
            int(merged.get('verification_budget_hint', 0) or 0),
            int(learned.get('verification_budget_hint', 0) or 0),
            1 if bool(competition.get('probe_pressure_active', False)) else 0,
        )
        merged['safe_fallback_class'] = str(
            learned.get('safe_fallback_class', merged.get('safe_fallback_class', 'wait')) or merged.get('safe_fallback_class', 'wait')
        )
        merged['preferred_verification_functions'] = self._learning_merge_ordered_lists(
            competition.get('required_probes', []) if bool(competition.get('probe_pressure_active', False)) else [],
            [
                fn_name
                for fn_name in list(competition.get('dominant_anchor_functions', []) or [])
                if self._is_learning_verification_function(fn_name)
            ],
            learned.get('preferred_verification_functions', []),
            merged.get('preferred_verification_functions', []),
        )
        merged['preferred_fallback_functions'] = self._learning_merge_ordered_lists(
            learned.get('preferred_fallback_functions', []),
            competition.get('dominant_anchor_functions', []),
            merged.get('preferred_fallback_functions', []),
        )
        merged['blocked_action_classes'] = self._learning_merge_ordered_lists(
            learned.get('blocked_action_classes', []),
            competition.get('dominant_risky_functions', []) if float(competition.get('latent_instability', 0.0) or 0.0) >= 0.55 else [],
            merged.get('blocked_action_classes', []),
        )
        merged['required_probes'] = self._learning_merge_ordered_lists(
            learned.get('required_probes', []),
            competition.get('required_probes', []),
        )
        merged['dominant_anchor_functions'] = self._learning_merge_ordered_lists(
            learned.get('dominant_anchor_functions', []),
            competition.get('dominant_anchor_functions', []),
        )
        merged['dominant_risky_functions'] = self._learning_merge_ordered_lists(
            learned.get('dominant_risky_functions', []),
            competition.get('dominant_risky_functions', []),
        )
        merged['probe_pressure'] = max(
            float(merged.get('probe_pressure', 0.0) or 0.0),
            float(learned.get('probe_pressure', 0.0) or 0.0),
            float(competition.get('probe_pressure', 0.0) or 0.0),
        )
        merged['latent_instability'] = max(
            float(merged.get('latent_instability', 0.0) or 0.0),
            float(learned.get('latent_instability', 0.0) or 0.0),
            float(competition.get('latent_instability', 0.0) or 0.0),
        )
        merged['dominant_branch_id'] = str(
            learned.get('dominant_branch_id', competition.get('dominant_branch_id', merged.get('dominant_branch_id', ''))) or merged.get('dominant_branch_id', '')
        )
        merged['source_action'] = str(learned.get('source_action', merged.get('source_action', '')) or merged.get('source_action', ''))
        return merged

    def _annotate_candidates_with_learning_updates(
        self,
        candidate_actions: List[Dict[str, Any]],
        decision_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        return annotate_candidates_with_learning_updates(
            self,
            candidate_actions,
            decision_context,
        )

    def _apply_mechanism_candidate_control(
        self,
        *,
        candidate_actions: List[Dict[str, Any]],
        decision_context: Optional[Dict[str, Any]],
        obs_before: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply hard mechanism control before the governance arbiter.

        This hook used to exist on CoreMainLoop and GovernanceStage still calls it. A refactor
        removed the method without migrating the call site, which left the runtime in a broken
        half-migrated state. Restore the loop hook and wire it to the already-present mechanism
        control modules: cooldown filtering + discriminating-action enforcement.
        """
        if not isinstance(candidate_actions, list) or not candidate_actions:
            return list(candidate_actions or [])

        context = dict(decision_context or {})
        world_model_summary = context.get('world_model_summary', {}) if isinstance(context.get('world_model_summary', {}), dict) else {}
        unified = context.get('unified_context') or context.get('unified_cognitive_context')
        unified_dict = unified.to_dict() if isinstance(unified, UnifiedCognitiveContext) else (dict(unified) if isinstance(unified, dict) else {})

        runtime_control = self._last_mechanism_runtime_view.get('mechanism_control_summary', {}) if isinstance(self._last_mechanism_runtime_view, dict) else {}
        if runtime_control:
            merged_wm = dict(world_model_summary)
            static_control = merged_wm.get('mechanism_control_summary', {}) if isinstance(merged_wm.get('mechanism_control_summary', {}), dict) else {}
            merged_control = dict(static_control)
            merged_control.update(dict(runtime_control))
            merged_wm['mechanism_control_summary'] = merged_control
            context['world_model_summary'] = merged_wm
            context['mechanism_control_summary'] = dict(merged_control)
            if unified_dict:
                unified_dict['mechanism_control_summary'] = dict(merged_control)
                if 'active_beliefs_summary' not in unified_dict or not isinstance(unified_dict.get('active_beliefs_summary', {}), dict):
                    unified_dict['active_beliefs_summary'] = dict(merged_wm)
                else:
                    active_beliefs = dict(unified_dict.get('active_beliefs_summary', {}))
                    active_beliefs['mechanism_control_summary'] = dict(merged_control)
                    unified_dict['active_beliefs_summary'] = active_beliefs
                context['unified_context'] = unified_dict
                context['unified_cognitive_context'] = unified_dict

        cooldown_report = self._candidate_cooldown_gate.filter_candidates(
            candidate_actions,
            decision_context=context,
            episode_trace=self._episode_trace,
            tick=self._tick,
            obs_before=obs_before,
        )
        working_candidates = list(cooldown_report.filtered_candidates or [])
        if not working_candidates:
            # Fail safe: never hand an empty frontier to governance. Preserve original candidates
            # but keep the diagnostics so traces reveal that cooldown would have removed all options.
            working_candidates = list(candidate_actions)

        selection_report = self._discriminating_action_selector.enforce(
            working_candidates,
            decision_context=context,
            obs_before=obs_before,
            tick=self._tick,
        )
        final_candidates = list(selection_report.candidates or working_candidates)

        self._mechanism_control_audit_log.append({
            'episode': self._episode,
            'tick': self._tick,
            'input_count': len(candidate_actions),
            'post_cooldown_count': len(cooldown_report.filtered_candidates or []),
            'post_selector_count': len(final_candidates),
            'cooldown_diagnostics': self._json_safe(cooldown_report.diagnostics),
            'selector_enforced': bool(selection_report.enforced),
            'selector_diagnostics': self._json_safe(selection_report.diagnostics),
        })
        return final_candidates

    def _learning_context_payload(self) -> Dict[str, Any]:
        snapshot = self._learning_policy_snapshot if isinstance(self._learning_policy_snapshot, dict) else {}
        return {
            'learning_enabled': bool(getattr(self, '_learning_enabled', False)),
            'selector_bias': dict(snapshot.get('selector_bias', {}) if isinstance(snapshot.get('selector_bias', {}), dict) else {}),
            'agenda_prior': dict(snapshot.get('agenda_prior', {}) if isinstance(snapshot.get('agenda_prior', {}), dict) else {}),
            'recovery_shortcut': dict(snapshot.get('recovery_shortcut', {}) if isinstance(snapshot.get('recovery_shortcut', {}), dict) else {}),
            'failure_preference_policy': dict(snapshot.get('failure_preference_policy', {}) if isinstance(snapshot.get('failure_preference_policy', {}), dict) else {}),
            'retention_failure_policy': dict(snapshot.get('retention_failure_policy', {}) if isinstance(snapshot.get('retention_failure_policy', {}), dict) else {}),
        }

    def _update_plan_progress(self, action_to_use: dict, result: dict, reward: float, obs_before: dict) -> None:
        """Compatibility wrapper: planner progress moved to PlannerRuntime."""
        runtime_out = self._planner_runtime.tick(
            phase='progress',
            obs=obs_before,
            selected_action=action_to_use,
            result=result,
            reward=reward,
        )
        self._consume_planner_runtime_result(runtime_out, fallback_action=action_to_use)

    def _tick_loop_staged(self, obs_before: dict) -> dict:
        """
        Shadow refactor of _tick_loop using stage helper methods.
        This method calls the discrete stage methods in order.
        After verification, _tick_loop will delegate to this method.
        
        Sprint 2-6 Integration:
        - Sprint 4: CausalTrace and EventTimeline at each stage
        - Sprint 5: Self-model tracking (capability, reliability, resource)
        """
        prepare_llm_analyst_initial_goal(self, obs_before=obs_before)
        tick_start = begin_staged_tick(self, obs_before)
        continuity_snapshot = tick_start.continuity_snapshot
        ctx = tick_start.context
        tick_frame = tick_start.tick_frame
        tick_trace = tick_start.tick_trace

        # Stage 1: Retrieval
        retrieval_out = self._retrieval_stage.run(
            self,
            RetrievalStageInput(
                obs_before=obs_before,
                context=ctx,
                continuity_snapshot=continuity_snapshot,
            ),
        )
        surfaced = retrieval_out.surfaced
        query = retrieval_out.query
        record_surfaced_candidates(self, tick_trace, surfaced)

        # Track hypothesis consumption for graduation (Task 1.2)
        # P3-B4: Will be recorded AFTER execution when reward is known

        # Stage 2: Planner + Governance
        planner_out = self._planner_stage.run(
            self,
            PlannerStageInput(
                obs_before=obs_before,
                surfaced=surfaced,
                continuity_snapshot=continuity_snapshot,
                frame=tick_frame,
            ),
        )
        action_to_use = planner_out.arm_action if planner_out.arm_meta.get('arm') != 'base' else planner_out.base_action
        governance_out = self._governance_stage.run(
            self,
            GovernanceStageInput(
                action_to_use=action_to_use,
                planner_output=planner_out,
                continuity_snapshot=continuity_snapshot,
                obs_before=obs_before,
                surfaced=surfaced,
                frame=tick_frame,
            ),
        )
        action_to_use = governance_out.action_to_use
        candidate_actions = governance_out.candidate_actions
        decision_outcome = governance_out.decision_outcome
        governance_result = governance_out.governance_result
        record_candidate_frontier(
            self,
            tick_trace,
            candidate_actions,
            decision_outcome,
            governance_result if isinstance(governance_result, dict) else {},
        )
        run_llm_shadow_pre_execution(
            self,
            obs_before=obs_before,
            planner_output=planner_out,
            governance_output=governance_out,
            action_to_use=action_to_use,
        )

        # Stage 3: Execution
        action_to_use = self._resolve_action_for_execution(action_to_use, obs_before)
        e = self._stage3_execution(action_to_use, query, obs_before)
        result = e['result']
        reward = e['reward']
        execution_out = apply_execution_outcome(
            self,
            tick_trace,
            action_to_use=action_to_use,
            obs_before=obs_before,
            surfaced=surfaced,
            result=result,
            reward=reward,
        )
        fn_name = execution_out.function_name
        terminal = execution_out.terminal

        # Stage 4: Testing
        self._event_timeline.emit_stage_enter(self._episode, self._tick, 'testing')
        self._stage4_testing(obs_before, surfaced, action_to_use, result, tick_frame)
        self._event_timeline.emit_stage_exit(self._episode, self._tick, 'testing', {'result': result.get('success', False)})

        # Stage 5: Evidence & Commit
        self._event_timeline.emit_stage_enter(self._episode, self._tick, 'evidence_commit')
        ec = self._stage5_evidence_commit(action_to_use, result)
        
        # Sprint 4: Record commit in timeline
        if ec.get('committed_ids'):
            self._event_timeline.emit_commit(self._episode, self._tick, ec['committed_ids'])
        self._event_timeline.emit_stage_exit(self._episode, self._tick, 'evidence_commit', {'committed': len(ec.get('committed_ids', []))})

        # Stage 6: Post-commit
        self._stage6_post_commit(ec['committed_ids'], obs_before, result, action_to_use, reward)
        finalize_llm_shadow_post_execution(
            self,
            obs_before=obs_before,
            action_to_use=action_to_use,
            result=result,
            reward=reward,
        )
        finalize_llm_analyst_post_execution(
            self,
            obs_before=obs_before,
            action_to_use=action_to_use,
            result=result,
            reward=reward,
        )
        finalize_staged_tick(
            self,
            tick_trace,
            continuity_snapshot=continuity_snapshot,
            surfaced=surfaced,
            function_name=fn_name,
            reward=reward,
            terminal=terminal,
        )
        return sync_tick_state(
            self,
            continuity_snapshot=continuity_snapshot,
            surfaced=surfaced,
            action_to_use=action_to_use,
            result=result,
            reward=reward,
            terminal=terminal,
        )

    def _tick_loop(self, obs_before: dict) -> dict:
        """
        Core tick loop - delegates to staged implementation.

        Task 0.1: Shadow refactor complete. The original inline logic has been
        extracted into discrete stage methods. _tick_loop now delegates to
        _tick_loop_staged() which calls these methods in order.
        """
        return self._tick_loop_staged(obs_before)

    def _extract_available_functions(self, obs: dict) -> List[str]:
        return extract_available_functions(obs)

    def _build_kwargs_from_context(self, fn: str, obs: dict, continuity_snapshot: Optional[dict]) -> dict:
        plan_summary = self._plan_state.get_plan_summary()
        return build_kwargs_from_context(
            fn,
            obs=obs,
            continuity_snapshot=continuity_snapshot,
            tick=self._tick,
            episode=self._episode,
            plan_step_intent=self._plan_state.get_intent_for_step(),
            plan_step_target=self._plan_state.get_target_function_for_step(),
            plan_revision_count=plan_summary.get('revision_count', 0),
            recent_reward=float(self._episode_trace[-1].get('reward', 0.0)) if self._episode_trace else 0.0,
        )

    def _run_planner_control_tick(self, obs: dict, continuity_snapshot: Optional[dict], frame: Optional[TickContextFrame] = None) -> dict:
        """Compatibility wrapper: planner control moved to PlannerRuntime."""
        runtime_out = self._planner_runtime.tick(
            phase='control',
            obs=obs,
            continuity_snapshot=continuity_snapshot or {},
            frame=frame,
        )
        runtime_payload = self._consume_planner_runtime_result(runtime_out)
        return {
            'events': runtime_payload['decision_flags'].get('events', []),
            'has_plan': self._plan_state.has_plan,
            'plan_summary': self._plan_state.get_plan_summary(),
            'policy_profile': runtime_payload['decision_flags'].get('policy_profile'),
            'representation_profile': runtime_payload['decision_flags'].get('representation_profile'),
            'meta_control_snapshot_id': runtime_payload['decision_flags'].get('meta_control_snapshot_id'),
            'meta_control_inputs_hash': runtime_payload['decision_flags'].get('meta_control_inputs_hash'),
        }

    def _make_wait_governance_candidate(self, intent: str = 'safe_fallback') -> Dict[str, Any]:
        return {
            'action': 'wait',
            'function_name': 'wait',
            'intent': intent,
            'risk': 0.05,
            'opportunity_estimate': 0.1,
            'final_score': 0.05,
            'estimated_cost': 0.1,
            'raw_action': {'kind': 'wait', 'payload': {}},
        }

    def _governance_wait_baseline_allowed(self, raw_candidates: List[Dict[str, Any]], selected_action: Optional[Dict[str, Any]]) -> bool:
        # Default allow wait safety baseline unless policy/constraints explicitly disable it.
        for candidate in [*raw_candidates, selected_action]:
            if not isinstance(candidate, dict):
                continue
            meta = candidate.get('_candidate_meta', {}) if isinstance(candidate.get('_candidate_meta', {}), dict) else {}
            for key in ('forbid_wait_baseline', 'disable_wait_baseline', 'planner_forbid_wait_baseline'):
                if bool(meta.get(key)):
                    return False
        return True

    def _generate_action(self, obs: dict, continuity_snapshot: dict = None) -> dict:
        """
        Generate base action from observation.

        Task 1.1: Continuity influences action selection.
        Exploration goals prioritize novel functions; exploitation goals prioritize confirming known patterns.
        """
        if isinstance(obs, dict) and bool(obs.get('terminal') or obs.get('done')):
            return {'kind': 'wait', 'payload': {}}

        available = self._extract_available_functions(obs)

        if self._plan_state.has_plan:
            target = self._plan_state.get_target_function_for_step()
            if target and target in available:
                available = [target] + [fn for fn in available if fn != target]

        if available:
            fn = available[0]
            return {
                'kind': 'call_tool',
                'payload': {
                    'tool_name': 'call_hidden_function',
                    'tool_args': {
                        'function_name': fn,
                        'kwargs': self._build_kwargs_from_context(fn, obs, continuity_snapshot),
                    },
                },
            }

        # Only use inspect fallback for non-terminal discovery surfaces.
        return {'kind': 'inspect', 'payload': {}}

    def _process_graduation_candidates(self):
        self._graduation_runtime.run(
            ProcessGraduationCandidatesInput(episode=int(self._episode), tick=int(self._tick))
        )

    def _process_graduation_candidates_impl(self, stage_input: ProcessGraduationCandidatesInput) -> Dict[str, Any]:
        """
        Process graduation candidates: format → validate → commit via Step10.

        This follows the rule: all writes go through validator + committer.
        GraduationTracker only proposes; validator decides; committer writes.

        P3-E: DistillationCompiler provides advisory decisions on eligibility,
        but actual promotions still go through the formal write path.
        """
        from modules.graduation import GraduationProposalFormatter

        formatter = GraduationProposalFormatter()

        # Get compile candidates
        committed_count = 0
        compile_candidates = self._grad_tracker.get_compile_candidates()
        for candidate in compile_candidates:
            proposal = formatter.format_compile_proposal(candidate)
            decision = self._validator.validate(proposal)
            if decision.decision == 'accept_new':
                self._committer.commit([proposal])
                committed_count += 1

        # Get distillation candidates
        distillation_candidates = self._grad_tracker.get_distillation_candidates()
        for candidate in distillation_candidates:
            hyp_id = candidate.get('hyp_id')
            if not hyp_id:
                continue

            # Issue 1 fix: Look up hypothesis to get its provenance and object_id
            hyp = self._hypotheses.get_hypothesis(hyp_id)
            prov = self._grad_tracker.get_provenance(hyp_id)

            # Get the canonical object_id for this hypothesis (from provenance or hypothesis store)
            object_id = None
            if prov and hasattr(prov, 'object_id') and prov.object_id:
                object_id = prov.object_id
            elif hyp:
                # Fallback: check if hypothesis was created from an object
                object_id = getattr(hyp, 'object_id', None) or self._hypotheses.get_object_id_for_hypothesis(hyp_id)

            # Get object data from store for compiler eligibility check
            compiled_obj = None
            if object_id:
                compiled_obj = self._shared_store.get(object_id) if hasattr(self._shared_store, 'get') else None

            # Use compiler to check eligibility (advisory)
            if compiled_obj:
                dist_candidate = self._distillation_compiler.check_distillation_eligibility(
                    object_id=object_id or hyp_id,
                    asset_status=compiled_obj.get('asset_status', 'compiled_asset'),
                    consumption_count=compiled_obj.get('consumption_count', 0),
                    reuse_history=compiled_obj.get('reuse_history', []),
                    trigger_source=compiled_obj.get('trigger_source', 'unknown'),
                    trigger_episode=compiled_obj.get('trigger_episode', 0),
                    confidence=compiled_obj.get('confidence', 0.5),
                    content=compiled_obj.get('content', {}),
                    current_episode=self._episode,
                )

                # Get compilation decision from compiler
                decision = self._distillation_compiler.compile(dist_candidate, self._episode)

                # P3-E: Log distillation decision for audit
                self._grad_tracker._grad_log.append({
                    'tick': self._tick,
                    'episode': self._episode,
                    'entry': 'distillation_decision',
                    'hyp_id': hyp_id,
                    'object_id': object_id,
                    'decision': decision.value,
                    'compilation_target': dist_candidate.compilation_target.value if dist_candidate.compilation_target else None,
                })

                # Issue 3 fix: COMPILE and DISTILL are separate decisions
                # Only propose if compiler says DISTILL (verified) or COMPILE
                if decision == DistillationDecision.DISTILL:
                    # DISTILL requires verification gate - propose as distillation
                    proposal = formatter.format_distillation_proposal(candidate)
                    write_decision = self._validator.validate(proposal)
                    if write_decision.decision == 'accept_new':
                        self._committer.commit([proposal])
                        committed_count += 1
                elif decision == DistillationDecision.COMPILE:
                    # COMPILE is separate - propose as compile only (not distillation)
                    compile_proposal = formatter.format_compile_proposal(candidate)
                    write_decision = self._validator.validate(compile_proposal)
                    if write_decision.decision == 'accept_new':
                        self._committer.commit([compile_proposal])
                        committed_count += 1
                # DEFER and RETIRE: do not promote
        return {'proposals_committed': committed_count}

    def _get_reward(self, result: dict) -> float:
        """Extract reward from world result."""
        if isinstance(result, dict):
            na = result.get('novel_api', {})
            if hasattr(na, '_data'):
                na = na._data
            rew = na.get('reward_signal', 0.0) if isinstance(na, dict) else 0.0
            rew += result.get('world', {}).get('reward_signal', 0.0) if isinstance(result.get('world'), dict) else 0.0
            return rew
        return 0.0

    def _mark_continuity_task_completed(self, task_id: Optional[str], reason: str = '') -> None:
        """Close continuity agenda task as completed when recovery action consumed."""
        task = str(task_id or '')
        if not task or not self._continuity.agenda.has_task(task):
            return
        self._continuity.agenda.mark_completed(task)
        if hasattr(self, '_continuity_log'):
            self._continuity_log.append({
                'event': 'agenda_task_completed',
                'task_id': task,
                'episode': self._episode,
                'tick': self._tick,
                'reason': reason or 'completed',
            })

    def _mark_continuity_task_cancelled(self, task_id: Optional[str], reason: str = '') -> None:
        """Close continuity agenda task as cancelled when recovery path is abandoned."""
        task = str(task_id or '')
        if not task or not self._continuity.agenda.has_task(task):
            return
        self._continuity.agenda.mark_cancelled(task)
        if hasattr(self, '_continuity_log'):
            self._continuity_log.append({
                'event': 'agenda_task_cancelled',
                'task_id': task,
                'episode': self._episode,
                'tick': self._tick,
                'reason': reason or 'cancelled',
            })

    def _sync_llm_clients(self) -> None:
        """Bind route-aware capability gateways into the integrated interfaces."""
        routed_bindings = (
            "_retrieval_llm",
            "_hypothesis_llm",
            "_probe_designer",
            "_skill_frontend",
            "_recovery",
            "_repr_proposer",
        )
        for attr_name in routed_bindings:
            target = getattr(self, attr_name, None)
            if target is None or not hasattr(target, "_llm"):
                continue
            route_name = str(
                getattr(target, "LLM_ROUTE_NAME", getattr(target, "llm_route_name", "general"))
                or "general"
            ).strip() or "general"
            capability_prefix = str(
                getattr(target, "LLM_CAPABILITY_NAMESPACE", getattr(target, "llm_capability_namespace", route_name))
                or route_name
            ).strip() or route_name
            gateway = self._resolve_llm_gateway(route_name, capability_prefix=capability_prefix)
            target._llm_gateway = gateway
            target._llm = gateway

    def _default_llm_client_fallback(self):
        return (
            getattr(self, "_llm_client", None)
            or getattr(getattr(self, "_retriever", None), "_llm", None)
            or getattr(getattr(self, "_skill_rewriter", None), "_llm", None)
            or getattr(getattr(self, "_hypotheses", None), "_llm", None)
            or getattr(getattr(self, "_test_engine", None), "_llm", None)
        )

    def _llm_route_state(self) -> Dict[str, Any]:
        return llm_route_state(self)

    def _llm_route_feedback_state(self) -> Dict[str, Any]:
        state = getattr(self, "_llm_route_feedback_state_store", None)
        if not isinstance(state, dict):
            state = {}
            self._llm_route_feedback_state_store = state
        return state

    def _record_llm_route_feedback(
        self,
        route_name: str,
        *,
        score: float,
        source: str = "verifier",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        route_key = str(route_name or "general").strip() or "general"
        state = self._llm_route_feedback_state()
        entry = state.setdefault(
            route_key,
            {
                "successes": 0,
                "failures": 0,
                "score_sum": 0.0,
                "samples": 0,
                "last_source": "",
                "last_reason": "",
                "last_metadata": {},
                "last_episode": None,
                "last_tick": None,
            },
        )
        normalized_score = max(-1.0, min(1.0, float(score or 0.0)))
        entry["samples"] = int(entry.get("samples", 0) or 0) + 1
        entry["score_sum"] = float(entry.get("score_sum", 0.0) or 0.0) + normalized_score
        if normalized_score >= 0.0:
            entry["successes"] = int(entry.get("successes", 0) or 0) + 1
        else:
            entry["failures"] = int(entry.get("failures", 0) or 0) + 1
        entry["last_source"] = str(source or "verifier")
        entry["last_reason"] = str(reason or "")
        entry["last_metadata"] = dict(metadata or {})
        entry["last_episode"] = int(getattr(self, "_episode", 0) or 0)
        entry["last_tick"] = int(getattr(self, "_tick", 0) or 0)
        feedback_row = {
            "episode": entry["last_episode"],
            "tick": entry["last_tick"],
            "route_name": route_key,
            "score": normalized_score,
            "source": str(source or "verifier"),
            "reason": str(reason or ""),
            "metadata": dict(metadata or {}),
        }
        usage_log = getattr(self, "_llm_route_usage_log", None)
        if isinstance(usage_log, list):
            usage_log.append({**feedback_row, "event": "feedback"})
        advice_log = getattr(self, "_llm_advice_log", None)
        if isinstance(advice_log, list):
            advice_log.append(
                {
                    "episode": feedback_row["episode"],
                    "tick": feedback_row["tick"],
                    "kind": f"route_feedback::{route_key}",
                    "entry": "llm_route_feedback",
                    "route_name": route_key,
                    "score": normalized_score,
                    "source": feedback_row["source"],
                    "reason": feedback_row["reason"],
                }
            )
        return self._llm_route_feedback_summary().get(route_key, {})

    def _llm_route_feedback_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for route_name, raw in dict(self._llm_route_feedback_state() or {}).items():
            entry = dict(raw or {}) if isinstance(raw, dict) else {}
            samples = int(entry.get("samples", 0) or 0)
            score_sum = float(entry.get("score_sum", 0.0) or 0.0)
            score = max(-1.0, min(1.0, (score_sum / float(samples)) if samples else 0.0))
            summary[str(route_name or "")] = {
                "score": round(score, 4),
                "samples": samples,
                "successes": int(entry.get("successes", 0) or 0),
                "failures": int(entry.get("failures", 0) or 0),
                "last_source": str(entry.get("last_source", "") or ""),
                "last_reason": str(entry.get("last_reason", "") or ""),
                "last_metadata": dict(entry.get("last_metadata", {}) or {}) if isinstance(entry.get("last_metadata", {}), dict) else {},
                "last_episode": entry.get("last_episode"),
                "last_tick": entry.get("last_tick"),
            }
        return summary

    def _plan_step_feedback_reference(self, step_id: Optional[str] = None) -> Dict[str, Any]:
        return plan_step_feedback_reference(getattr(self, "_plan_state", None), step_id=step_id)

    def _verification_feedback_from_transition(self, transition: Any) -> Optional[Dict[str, Any]]:
        return verification_feedback_from_transition(transition)

    def _should_auto_consume_verifier_authority(
        self,
        transition: Any,
        *,
        parsed_feedback: Optional[Dict[str, Any]],
        step_ref: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return should_auto_consume_verifier_authority(
            transition,
            parsed_feedback=parsed_feedback,
            step_ref=step_ref,
            plan_state=getattr(self, "_plan_state", None),
        )

    def _recent_llm_route_usage_for_task(
        self,
        *,
        task_node_id: str,
        goal_id: str,
        tick_window: int = 12,
    ) -> list[Dict[str, Any]]:
        return recent_llm_route_usage_for_task(
            getattr(self, "_llm_route_usage_log", None),
            task_node_id=task_node_id,
            goal_id=goal_id,
            current_episode=int(getattr(self, "_episode", 0) or 0),
            current_tick=int(getattr(self, "_tick", 0) or 0),
            tick_window=tick_window,
        )

    def _record_verification_feedback_for_transition(
        self,
        transition: Any,
        *,
        step_ref: Optional[Dict[str, Any]] = None,
    ) -> None:
        verification_result = record_verification_feedback_for_transition(
            transition,
            step_ref=step_ref,
            plan_state=getattr(self, "_plan_state", None),
            route_usage_log=getattr(self, "_llm_route_usage_log", None),
            current_episode=int(getattr(self, "_episode", 0) or 0),
            current_tick=int(getattr(self, "_tick", 0) or 0),
            record_route_feedback=self._record_llm_route_feedback,
        )
        if verification_result is not None:
            self._last_verification_result = verification_result

    def _apply_step_transitions_with_feedback(self, transitions: List[Dict[str, Any]]) -> int:
        result = run_step_transitions_with_feedback(
            plan_state=getattr(self, "_plan_state", None),
            transitions=list(transitions or []),
            route_usage_log=getattr(self, "_llm_route_usage_log", None),
            current_episode=int(getattr(self, "_episode", 0) or 0),
            current_tick=int(getattr(self, "_tick", 0) or 0),
            record_route_feedback=self._record_llm_route_feedback,
        )
        verification_result = result.get("last_verification_result")
        if verification_result is not None:
            self._last_verification_result = verification_result
        return int(result.get("applied", 0) or 0)

    def _route_capability_requirements(self, route_name: str) -> list[str]:
        return route_capability_requirements(route_name)

    def _build_llm_route_context(
        self,
        route_name: str,
        *,
        capability_request: str = "",
        capability_resolution: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_llm_route_context(
            self,
            route_name,
            capability_request=capability_request,
            capability_resolution=capability_resolution,
        )

    def _llm_route_usage_bucket(self, route_name: str) -> Dict[str, Any]:
        return llm_route_usage_bucket(self, route_name)

    def _llm_route_budget_status(
        self,
        *,
        route_name: str,
        route_metadata: Optional[Dict[str, Any]] = None,
        prompt_tokens: int = 0,
        reserved_response_tokens: int = 0,
    ) -> Dict[str, Any]:
        return llm_route_budget_status(
            self,
            route_name=route_name,
            route_metadata=route_metadata,
            prompt_tokens=prompt_tokens,
            reserved_response_tokens=reserved_response_tokens,
        )

    def _record_llm_route_blocked(
        self,
        *,
        route_name: str,
        method_name: str,
        route_metadata: Optional[Dict[str, Any]],
        budget_status: Dict[str, Any],
        entry_kind: str,
    ) -> None:
        record_llm_route_blocked(
            self,
            route_name=route_name,
            method_name=method_name,
            route_metadata=route_metadata,
            budget_status=budget_status,
            entry_kind=entry_kind,
        )

    def _record_llm_route_usage(
        self,
        *,
        route_name: str,
        method_name: str,
        prompt_tokens: int,
        response_tokens: int,
        reserved_response_tokens: int,
        route_metadata: Optional[Dict[str, Any]],
    ) -> None:
        record_llm_route_usage(
            self,
            route_name=route_name,
            method_name=method_name,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            reserved_response_tokens=reserved_response_tokens,
            route_metadata=route_metadata,
        )

    def _llm_route_usage_summary(self) -> Dict[str, Any]:
        return llm_route_usage_summary(self)

    def _runtime_budget_route_specs(self) -> Dict[str, Dict[str, Any]]:
        return runtime_budget_route_specs(self)

    def _runtime_budget_capability_specs(self) -> Dict[str, Dict[str, Any]]:
        return runtime_budget_capability_specs(self)

    def _goal_task_binding_for_llm_policy(self):
        return goal_task_binding_for_llm_policy(self)

    def _goal_task_route_specs(self) -> Dict[str, Dict[str, Any]]:
        return goal_task_route_specs(self)

    def _goal_task_capability_specs(self) -> Dict[str, Dict[str, Any]]:
        return goal_task_capability_specs(self)

    def _resolved_llm_route_specs(self) -> Dict[str, Any]:
        return resolved_llm_route_specs(self)

    def _resolved_llm_capability_specs(self) -> Dict[str, Dict[str, Any]]:
        return resolved_llm_capability_specs(self)

    def _ensure_llm_capability_registry(self):
        return ensure_llm_capability_registry(self)

    def _resolve_llm_capability_spec(self, capability_request: str, fallback_route: str = "general") -> Dict[str, Any]:
        return resolve_llm_capability_spec(
            self,
            capability_request,
            fallback_route=fallback_route,
        )

    def _ensure_model_router(self):
        return ensure_model_router(self)

    def _resolve_llm_client(
        self,
        route_name: str = "general",
        *,
        capability_request: str = "",
        capability_resolution: Optional[Dict[str, Any]] = None,
    ):
        return resolve_llm_client(
            self,
            route_name,
            capability_request=capability_request,
            capability_resolution=capability_resolution,
        )

    def _resolve_llm_gateway(self, route_name: str = "general", *, capability_prefix: str = ""):
        return resolve_llm_gateway(self, route_name, capability_prefix=capability_prefix)

    def _resolve_structured_answer_llm_client(self):
        return self._resolve_llm_client("structured_answer")

    def _augment_hypotheses_with_llm(self, obs: dict, continuity_snapshot: Dict[str, Any]) -> None:
        known_functions = self._extract_known_functions(obs)
        context = (
            f"{continuity_snapshot.get('identity_summary', '')}; "
            f"goals={continuity_snapshot.get('active_goal_count', 0)}; "
            f"entropy={self._hypotheses.entropy():.3f}"
        )
        candidates = self._hypothesis_llm.generate_hypothesis_candidates(obs, context, known_functions)
        if not candidates:
            return
        created = self._hypothesis_llm.add_llm_generated_hypotheses(
            candidates[:2],
            tick=self._tick,
            episode=self._episode,
        )
        for hyp in created:
            if self._teacher_allows_intervention():
                self._teacher.teacher_proposal(
                    target_id=hyp.id,
                    target_type='hypothesis',
                    content=hyp.to_dict(),
                    rationale='LLM-generated hypothesis candidate routed through CoreMainLoop',
                    actor='system_llm',
                )
                self._teacher_log.append({
                    'tick': self._tick,
                    'episode': self._episode,
                    'target_id': hyp.id,
                    'entry': 'teacher_proposal',
                })

    def _candidate_dedupe_signature(self, action: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        return candidate_dedupe_signature(action)

    def _handle_recovery_if_needed(self, action: dict, result: dict) -> None:
        recovery_event = self._testing_recovery_runtime._handle_recovery_if_needed(action, result)
        if isinstance(recovery_event, dict):
            if isinstance(recovery_event.get('pending_probe_patch'), dict):
                self._pending_recovery_probe = recovery_event['pending_probe_patch']
            if isinstance(recovery_event.get('pending_replan_patch'), dict):
                self._pending_replan = recovery_event['pending_replan_patch']

    def _post_commit_integration(self, committed_ids: List[str], obs_before: dict, result: dict) -> Dict[str, Any]:
        out = self._post_commit_runtime.run(
            PostCommitIntegrationInput(committed_ids=list(committed_ids), obs_before=obs_before, result=result)
        )
        return out.integration_summary

    def _post_commit_integration_impl(self, stage_input: PostCommitIntegrationInput) -> Dict[str, Any]:
        """Fan out Step 10 commits to the remaining modules."""
        committed_ids = stage_input.committed_ids
        obs_before = stage_input.obs_before
        result = stage_input.result
        if not committed_ids:
            return {'integration_summary': {'committed_count': 0}}
        integration_summary = integrate_committed_objects(
            committed_ids=committed_ids,
            processed_committed_ids=self._processed_committed_ids,
            shared_store=self._shared_store,
            runtime_store=self._runtime_store,
            family_registry=self._family_registry,
            confirmed_functions=self._confirmed_functions,
            commit_log=self._commit_log,
            teacher=self._teacher,
            teacher_log=self._teacher_log,
            teacher_allows_intervention=self._teacher_allows_intervention,
            tick=self._tick,
            episode=self._episode,
            obs_before=obs_before,
            result=result,
            reward=self._get_reward(result),
        )
        if hasattr(self, '_continuity'):
            autobiographical_summary = integration_summary.get('autobiographical_summary', {})
            if isinstance(autobiographical_summary, dict) and autobiographical_summary:
                self._continuity.record_autobiographical_summary(autobiographical_summary)
            self._continuity.record_memory_summary(
                semantic_memory={
                    'surfaced_object_ids': list(integration_summary.get('surfaced_object_ids', []) or []),
                },
                procedural_memory={
                    'planner_prior_object_ids': list(integration_summary.get('planner_prior_object_ids', []) or []),
                },
                transfer_memory={
                    'cross_domain_prior_object_ids': list(integration_summary.get('cross_domain_prior_object_ids', []) or []),
                },
            )
        self._maybe_commit_procedure_chain(committed_ids=committed_ids, obs_before=obs_before, result=result)
        self._write_object_workspace_state(integration_summary)
        integration_summary.setdefault('committed_count', len(committed_ids))
        return {'integration_summary': integration_summary}

    def _write_object_workspace_state(self, integration_summary: Dict[str, Any]) -> None:
        if not isinstance(integration_summary, dict) or not hasattr(self, '_state_mgr'):
            return
        patch = {
            'object_workspace.surfaced_object_ids': list(integration_summary.get('surfaced_object_ids', []) or []),
            'object_workspace.mechanism_object_ids': list(integration_summary.get('mechanism_object_ids', []) or []),
            'object_workspace.object_competitions': list(integration_summary.get('object_competitions', []) or []),
            'object_workspace.active_tests': list(integration_summary.get('active_tests', []) or []),
            'object_workspace.current_identity_snapshot': dict(integration_summary.get('current_identity_snapshot', {}) or {}),
            'object_workspace.autobiographical_summary': dict(integration_summary.get('autobiographical_summary', {}) or {}),
        }
        if 'candidate_tests' in integration_summary:
            patch['object_workspace.candidate_tests'] = list(integration_summary.get('candidate_tests', []) or [])
        if 'candidate_programs' in integration_summary:
            patch['object_workspace.candidate_programs'] = list(integration_summary.get('candidate_programs', []) or [])
        if 'candidate_outputs' in integration_summary:
            patch['object_workspace.candidate_outputs'] = list(integration_summary.get('candidate_outputs', []) or [])
        self._state_mgr.update_state(
            patch,
            reason='workflow:post_commit_object_workspace',
            module='core',
        )

    def _load_procedure_objects(self, obs_before: Dict[str, Any]) -> List[Dict[str, Any]]:
        return load_procedure_objects(self._shared_store, obs_before)

    def _procedure_task_signature(self, obs_before: Dict[str, Any]) -> str:
        return procedure_task_signature(obs_before)

    def _procedure_text_tokens(self, text: str) -> Set[str]:
        return procedure_text_tokens(text)

    def _procedure_observed_functions(self, obs_before: Dict[str, Any]) -> List[str]:
        return procedure_observed_functions(obs_before)

    def _maybe_commit_procedure_chain(self, committed_ids: List[str], obs_before: Dict[str, Any], result: Dict[str, Any]) -> None:
        maybe_commit_procedure_chain(
            committed_ids=committed_ids,
            obs_before=obs_before,
            reward=float(self._get_reward(result) or 0.0),
            shared_store=self._shared_store,
            validator=self._validator,
            committer=self._committer,
            procedure_proposal_log=self._procedure_proposal_log,
            episode=self._episode,
            tick=self._tick,
            reject_decision=REJECT,
        )

    def _register_dynamic_family(self, fn_name: str, obj: Dict[str, Any]) -> None:
        family_id = f'family_{fn_name}'
        card = self._family_registry.get(family_id)
        if card is None:
            card = FamilyCard(
                family_id=family_id,
                claim=f"Learned family for function '{fn_name}'",
                mechanism='CoreMainLoop dynamic object family',
                signal_type='runtime_evidence',
                firing_phase='step10_commit',
                protective_primitive='commit',
                scope_boundary='Dynamic family inferred from committed object evidence',
                state=FamilyState.QUALIFYING,
                variants=[fn_name],
                n_seeds_tested=1,
            )
            self._family_registry.register(card)
        evidence = {'confidence': obj.get('confidence', 0.5), 'object_id': obj.get('object_id')}
        if obj.get('confidence', 0.5) >= 0.7 and not card.is_graduated():
            self._family_registry.graduate(
                family_id=family_id,
                evidence=evidence,
                gates_passed=['G_dynamic_confidence'],
                gates_failed=[],
                variant_scope=fn_name,
                reason='Dynamic family graduated after high-confidence commit',
            )

    def _record_continuity_tick(self, continuity_snapshot: Dict[str, Any]) -> None:
        record_continuity_tick(
            continuity_snapshot,
            episode=self._episode,
            tick=self._tick,
            state_writer=self._append_state_entry,
        )

    def _extract_known_functions(self, obs: dict) -> List[str]:
        api_raw = obs.get('novel_api', {})
        if hasattr(api_raw, 'raw'):
            api_raw = api_raw.raw
        discovered = list(api_raw.get('discovered_functions', []) if isinstance(api_raw, dict) else [])
        visible = list(api_raw.get('visible_functions', []) if isinstance(api_raw, dict) else [])
        known = []
        for fn in discovered + visible:
            if fn and fn not in known:
                known.append(fn)
        return known
    def _build_planner_ports(self) -> PlannerPorts:
        return PlannerPorts(
            plan_state=self._plan_state,
            objective_decomposer=self._objective_decomposer,
            plan_reviser=self._plan_reviser,
            meta_control=self._meta_control,
            build_tick_context_frame=lambda obs, continuity: self._build_tick_context_frame(obs, continuity),
            extract_available_functions=self._extract_available_functions,
            infer_task_family=self._infer_task_family,
            ablation_flags_snapshot=self._ablation_flags_snapshot,
            mark_continuity_task_completed=self._mark_continuity_task_completed,
            mark_continuity_task_cancelled=self._mark_continuity_task_cancelled,
            build_world_model_context=self._build_world_model_context,
            build_world_model_transition_priors=self._build_world_model_transition_priors,
            get_active_hypotheses=lambda: self._hypotheses.get_active() if hasattr(self, '_hypotheses') else [],
            get_reliability_tracker=lambda: self._reliability_tracker if hasattr(self, '_reliability_tracker') else None,
            get_episode=lambda: self._episode,
            get_tick=lambda: self._tick,
            get_max_ticks=lambda: self.max_ticks,
            get_episode_reward=lambda: self._episode_reward,
            get_episode_trace=lambda: self._episode_trace,
            get_pending_replan=lambda: self._pending_replan,
            get_world_provider_meta=lambda: self._world_provider_meta,
            get_causal_ablation=lambda: self._causal_ablation,
            get_learned_dynamics_predictor=lambda: getattr(self, '_learned_dynamics_shadow_predictor', None),
            get_learned_dynamics_deployment_mode=lambda: getattr(self, '_learned_dynamics_deployment_mode', 'shadow'),
            get_persistent_object_identity_tracker=lambda: getattr(self, '_persistent_object_identity_tracker', None),
        )

    def _apply_planner_state_patch(self, patch: Dict[str, Any]) -> None:
        if not isinstance(patch, dict) or not patch:
            return
        update_context = patch.get('update_context')
        if isinstance(update_context, dict):
            self._plan_state.update_context(
                tick=update_context.get('tick', self._tick),
                reward=update_context.get('reward', self._episode_reward),
                discovered_functions=update_context.get('discovered_functions', []),
            )
        step_transitions = patch.get('step_transitions')
        if isinstance(step_transitions, list) and step_transitions:
            self._apply_step_transitions_with_feedback(step_transitions)
        else:
            if patch.get('advance_step'):
                self._plan_state.advance_step()
            mark_failed_reason = patch.get('mark_failed_reason')
            if mark_failed_reason and self._plan_state.current_step:
                self._plan_state.fail_current_step(reason=str(mark_failed_reason))
        if patch.get('clear_plan'):
            self._plan_state.clear_plan()
        if patch.get('set_plan') is not None:
            self._plan_state.set_plan(patch['set_plan'])
        if 'pending_replan' in patch:
            self._pending_replan = patch.get('pending_replan')

    def _consume_planner_runtime_result(self, runtime_out: Any, fallback_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state_patch = runtime_out.state_patch if isinstance(getattr(runtime_out, 'state_patch', None), dict) else {}
        decision_flags = runtime_out.decision_flags if isinstance(getattr(runtime_out, 'decision_flags', None), dict) else {}
        telemetry = runtime_out.telemetry if isinstance(getattr(runtime_out, 'telemetry', None), dict) else {}
        self._apply_planner_state_patch(state_patch)
        selected_action = runtime_out.selected_action if isinstance(runtime_out.selected_action, dict) else (fallback_action or {})
        planner_payload = {
            'episode': int(self._episode),
            'tick': int(self._tick),
            'state_patch': dict(state_patch),
            'decision_flags': dict(decision_flags),
            'telemetry': dict(telemetry),
        }
        self._last_planner_runtime_payload = planner_payload
        self._planner_runtime_log.append(planner_payload)
        del self._planner_runtime_log[:-60]
        return {
            'selected_action': selected_action,
            'state_patch': state_patch,
            'decision_flags': decision_flags,
            'telemetry': telemetry,
        }
