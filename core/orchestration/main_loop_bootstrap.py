from __future__ import annotations

import os

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from core.main_loop_components import (
    CAPABILITY_ADVISORY,
    CAPABILITY_CONSTRAINED_CONTROL,
    ORGAN_CAPABILITY_KEYS,
    DecisionBridgeInput,
    TestOpportunityAudit,
    TestOpportunityEntry,
    TransferTraceLogger,
    _AdaptiveMetaControl,
    _FormalUpdateEngine,
    _NoopPromotionEngine,
    _NoopUpdateEngine,
    NovelAPICommitter,
)
from core.orchestration.continuity_persistence_adapter import ContinuityPersistenceAdapter
from core.orchestration.execution_control import ApprovalPolicy, ToolCapabilityRegistry
from core.orchestration.goal_task_control import GoalTaskRuntime
from core.orchestration.episode_lifecycle import EpisodeLifecycle
from core.orchestration.formal_memory_persistence import (
    DURABLE_OBJECT_RECORDS_STATE_PATH,
    load_persisted_durable_object_records,
    restore_durable_object_records,
)
from core.orchestration.governance_stage import GovernanceStage
from core.orchestration.planner_runtime import PlannerRuntime
from core.orchestration.planner_stage import PlannerStage
from core.orchestration.prediction_feedback import PredictionFeedbackPipeline
from core.orchestration.retrieval_stage import RetrievalStage
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
from core.orchestration.state_sync import StateSyncInput, StateSyncOrchestrator
from core.orchestration.state_sync_stage import StateSyncStage
from core.prediction_runtime import PredictionAdjudicator, PredictionEngine, PredictionRegistry
from core.reasoning import DeliberationEngine
from core.runtime_paths import default_event_log_path
from decision.candidate_cooldown_gate import CandidateCooldownGate
from decision.discriminating_action_selector import DiscriminatingActionSelector
from modules.continuity import ContinuityManager
from modules.episodic import LLMRetrievalInterface, RetrievalRuntimeTier
from modules.evidence.extractor import NovelAPIRawEvidenceExtractor
from modules.governance.family_registry import FamilyRegistry, init_registry
from modules.governance.gate import get_governance
from modules.governance.object_store import ObjectStore, ProposalValidator
from modules.hypothesis import (
    DiscriminatingTestEngine,
    LLMHypothesisInterface,
    LLMProbeDesigner,
    HypothesisTracker,
)
from modules.hypothesis.mechanism_posterior_updater import MechanismPosteriorUpdater
from modules.recovery import LLMErrorRecoveryInterface
from modules.representations import LLMRepresentationProposer
from modules.representations.proposer_llm import TrajectoryEntry
from modules.representations.store import get_runtime_store, get_warehouse
from modules.skills import LLMSkillFrontend, SkillRewriter
from modules.state import reset_state_manager
from modules.teacher import TeacherProtocol
from modules.world_model.belief import BeliefLedger, BeliefUpdater
from modules.world_model.compiler import DistillationCompiler
from modules.world_model.counterfactual import CounterfactualEngine
from modules.world_model.events import WorldModelEventBus
from modules.world_model.hidden_state import HiddenStateTracker
from modules.world_model.object_identity import PersistentObjectIdentityTracker
from modules.world_model.learned_dynamics import (
    BucketedLearnedDynamicsModel,
    default_learned_dynamics_model_path,
)
from modules.world_model.mechanism import MechanismExtractor, MechanismFormalWriter
from modules.world_model.prediction_feedback import PredictionMissFeedbackRuntime
from trace_runtime import resolve_trace_runtime

CausalTraceLogger, EventTimeline, _ = resolve_trace_runtime()

_RESUME_STATE_ENV = "THE_AGI_RESUME_STATE"
_STATE_PATH_ENV = "THE_AGI_STATE_PATH"
_RUNTIME_ROOT_ENV = "THE_AGI_RUNTIME_ROOT"


def _truthy_env(name: str, *, default: bool = False) -> bool:
    raw = str(os.getenv(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _fresh_start_durable_restore_enabled() -> bool:
    return bool(str(os.getenv(_STATE_PATH_ENV, "") or "").strip() or str(os.getenv(_RUNTIME_ROOT_ENV, "") or "").strip())


@dataclass(frozen=True)
class MainLoopBootstrapConfig:
    agent_id: str
    run_id: str
    seed: int
    arm_mode: str
    llm_client: Any


def bootstrap_core_capabilities(loop: Any, config: MainLoopBootstrapConfig) -> None:
    loop._state_mgr = reset_state_manager()
    loop._shared_store = ObjectStore()
    loop._goal_task_runtime = GoalTaskRuntime()
    loop._tool_capability_registry = ToolCapabilityRegistry()
    loop._execution_approval_policy = ApprovalPolicy()
    loop._retriever = loop._make_retriever(config.arm_mode, loop._shared_store, config.seed)
    loop._consumed_fns = set()
    loop._skill_rewriter = SkillRewriter(loop._shared_store, seed=config.seed)
    loop._hypotheses = HypothesisTracker(
        seed=config.seed,
        skill_rewriter=loop._skill_rewriter,
    )
    loop._test_engine = DiscriminatingTestEngine(loop._hypotheses, seed=config.seed)
    loop._extractor = NovelAPIRawEvidenceExtractor()
    loop._validator = ProposalValidator(
        f"{config.agent_id}_{config.run_id}",
        object_store=loop._shared_store,
    )
    loop._committer = NovelAPICommitter(object_store=loop._shared_store)
    loop._continuity = ContinuityManager(config.agent_id)
    loop._teacher = TeacherProtocol(agent_id=config.agent_id)
    loop._governance = get_governance()
    loop._governance.reset()
    try:
        loop._family_registry = init_registry()
    except ValueError:
        loop._family_registry = FamilyRegistry.get_instance()
    loop._card_warehouse = get_warehouse()
    loop._runtime_store = get_runtime_store()
    loop._repr_proposer = LLMRepresentationProposer(
        llm_client=config.llm_client,
        card_store=loop._card_warehouse,
    )
    loop._trajectory_entry_cls = TrajectoryEntry
    loop._retrieval_llm = LLMRetrievalInterface(
        llm_client=config.llm_client,
        runtime_tier=getattr(
            loop._runtime_budget,
            "retrieval_runtime_tier",
            RetrievalRuntimeTier.LLM_ASSISTED,
        ),
    )
    loop._hypothesis_llm = LLMHypothesisInterface(
        loop._hypotheses,
        llm_client=config.llm_client,
    )
    loop._probe_designer = LLMProbeDesigner(
        loop._test_engine,
        llm_client=config.llm_client,
    )
    loop._skill_frontend = LLMSkillFrontend(
        loop._skill_rewriter,
        llm_client=config.llm_client,
    )
    loop._recovery = LLMErrorRecoveryInterface(
        state_manager=loop._state_mgr,
        object_store=loop._shared_store,
        llm_client=config.llm_client,
    )
    from core.orchestration.structured_answer import StructuredAnswerSynthesizer

    loop._structured_answer_synthesizer = StructuredAnswerSynthesizer()
    loop._deliberation_engine = DeliberationEngine()


def bootstrap_state_and_stage_runtimes(loop: Any, config: MainLoopBootstrapConfig) -> None:
    if getattr(loop, "_state_mgr", None) is None:
        loop._state_mgr = reset_state_manager()
    loop._state_sync = StateSyncOrchestrator(loop._state_mgr)
    loop._state_sync_input_cls = StateSyncInput
    loop._decision_bridge_input_cls = DecisionBridgeInput
    loop._retrieval_stage = RetrievalStage()
    loop._planner_stage = PlannerStage()
    from planner import ObjectiveDecomposer, PlanReviser, PlanState

    loop._plan_state = PlanState()
    loop._objective_decomposer = ObjectiveDecomposer()
    loop._plan_reviser = PlanReviser()
    loop._meta_control = _AdaptiveMetaControl(loop._shared_store)
    loop._planner_runtime = PlannerRuntime(ports=loop._build_planner_ports())
    loop._governance_stage = GovernanceStage()
    loop._state_sync_stage = StateSyncStage()
    loop._stage1_runtime = Stage1RetrievalRuntime(loop._stage1_retrieval_impl)
    loop._stage2_candidate_runtime = Stage2CandidateGenerationRuntime(
        loop._stage2_candidate_generation_substage_impl
    )
    loop._stage2_plan_constraints_runtime = Stage2PlanConstraintsRuntime(
        loop._stage2_plan_constraints_substage_impl
    )
    loop._stage2_self_model_runtime = Stage2SelfModelSuppressionRuntime(
        loop._stage2_self_model_suppression_substage_impl
    )
    loop._stage2_prediction_bridge_runtime = Stage2PredictionBridgeRuntime(
        loop._stage2_prediction_context_bridge_substage_impl
    )
    loop._stage2_governance_runtime = Stage2GovernanceSubstageRuntime(
        loop._stage2_governance_substage_impl
    )
    loop._stage3_runtime = Stage3ExecutionRuntime(loop._stage3_execution_impl)
    loop._stage5_runtime = Stage5EvidenceCommitRuntime(
        loop._stage5_evidence_commit_impl
    )
    loop._post_commit_runtime = PostCommitIntegrationRuntime(
        loop._post_commit_integration_impl
    )
    loop._graduation_runtime = ProcessGraduationCandidatesRuntime(
        loop._process_graduation_candidates_impl
    )
    loop._learning_updates_runtime = ApplyLearningUpdatesRuntime(
        loop._apply_learning_updates_impl
    )
    loop._stage6_runtime = Stage6PostCommitRuntime(loop._stage6_post_commit_impl)
    loop._episode_lifecycle = EpisodeLifecycle(
        continuity_persistence=ContinuityPersistenceAdapter(
            state_mgr=loop._state_mgr,
            state_sync=loop._state_sync,
            state_sync_input_cls=loop._state_sync_input_cls,
        )
    )
    resume_state = _truthy_env(_RESUME_STATE_ENV, default=False)
    durable_records = []
    if not resume_state and _fresh_start_durable_restore_enabled():
        durable_records = load_persisted_durable_object_records(loop._state_mgr)
    loaded_state = bool(resume_state and loop._state_mgr.load())
    if not loaded_state:
        loop._state_mgr.initialize(
            agent_id=config.agent_id,
            run_id=config.run_id,
            episode_id="1",
            top_goal="explore",
        )
        if durable_records:
            loop._shared_store.restore_records(durable_records, replace=False)
            loop._state_sync.sync(
                StateSyncInput(
                    updates={DURABLE_OBJECT_RECORDS_STATE_PATH: durable_records},
                    reason="restore_durable_object_records_fresh_start",
                )
            )
    else:
        loop._state_sync.sync(
            StateSyncInput(
                updates={
                    "identity.agent_id": config.agent_id,
                    "identity.run_id": config.run_id,
                },
                reason="resume_runtime_identity",
            )
        )
        restore_durable_object_records(loop)
    if loaded_state:
        loop._continuity_resume_verdict = {
            "accepted": True,
            "loaded": True,
            "resume_requested": True,
            "policy": "degraded_restore",
            "reasons": ["resume_state_loaded"],
        }
    elif resume_state:
        loop._continuity_resume_verdict = {
            "accepted": True,
            "loaded": False,
            "resume_requested": True,
            "policy": "fresh_start",
            "reasons": ["resume_requested_no_state"],
        }
    else:
        loop._continuity_resume_verdict = {
            "accepted": True,
            "loaded": False,
            "resume_requested": False,
            "policy": "fresh_start",
            "reasons": ["fresh_start"],
        }
    loop._episode_lifecycle.on_boot(loop)


def bootstrap_tracking_state(loop: Any) -> None:
    loop._episode = 0
    loop._tick = 0
    loop._episode_reward = 0.0
    loop._total_reward = 0.0
    loop._tick_log = []
    loop._commit_log = []
    loop._leak_reject_log = []
    loop._transfer_trace = TransferTraceLogger()
    loop._pre_action_ranking = []
    loop._last_action_ranking = []
    loop._test_audit = TestOpportunityAudit()
    loop._pending_test_entries = []
    loop._confirmed_functions = set()
    loop._commit_quality_log = []
    loop._pre_sat_test_count = 0
    loop._pre_sat_test_cost = 0.0
    loop._last_test_verdict = None
    loop._last_test_passed = None
    loop._recovery_log = []
    loop._continuity_log = []
    loop._teacher_log = []
    loop._processed_committed_ids = set()
    loop._continuity_log.append(
        {
            "event": "world_provider_initialized",
            "runtime_env": loop._world_provider_meta.get("runtime_env", "unknown"),
            "world_provider_source": loop._world_provider_meta.get(
                "world_provider_source",
                "unknown",
            ),
        }
    )
    loop._continuity_log.append(
        {
            "event": "continuity_resume_verdict",
            **loop._json_safe(loop._continuity_resume_verdict),
        }
    )
    loop._representation_log = []
    loop._governance_log = []
    loop._organ_control_audit_log = []
    loop._viability_audit_log = []
    loop._candidate_viability_log = deque(maxlen=100)
    loop._llm_advice_log = []
    loop._llm_calls_per_tick = []
    loop._llm_route_usage_log = []
    loop._llm_route_runtime_state = {}
    loop._llm_route_client_wrappers = {}
    loop._llm_capability_gateways = {}
    loop._llm_capability_registry = None
    loop._llm_shadow_log = []
    loop._llm_analyst_log = []
    loop._llm_analyst_hypothesis_candidates = []
    loop._llm_initial_goal_hypothesis_candidates = []
    loop._llm_shadow_last_observation_signature = ""
    loop._llm_shadow_last_failure_signature = ""
    loop._llm_analyst_last_observation_signature = ""
    loop._llm_analyst_last_failure_signature = ""
    loop._episode_trace = []
    loop._learned_dynamics_shadow_log = []
    loop._persistent_object_identity_tracker = PersistentObjectIdentityTracker()
    loop._last_retrieval_tick = -10_000
    loop._last_retrieval_decision = {}
    loop._last_retrieval_decision_record = None
    loop._last_retrieval_aux_decisions = {}
    loop._last_hypothesis_augment_tick = -10_000
    loop._active_tick_context_frame = None
    loop._last_deliberation_result = {}
    loop._tick_context_frame_build_counts = {}
    loop._hidden_state_tracker = HiddenStateTracker()
    loop._last_rerank_tick = -10_000
    loop._last_query_rewrite_tick = -10_000
    loop._last_observation_signature = ""
    loop._last_phase_hint = ""
    loop._llm_calls_this_tick = 0
    loop._perception_bridge = None
    loop._last_perception_summary = {}
    loop._pending_organ_control_audit = []
    loop._organ_failure_streaks = {
        key: 0 for key in ORGAN_CAPABILITY_KEYS
    }
    loop._organ_failure_threshold = 3
    loop._organ_capability_flags = {
        "world_model": CAPABILITY_ADVISORY,
        "planner": CAPABILITY_CONSTRAINED_CONTROL,
        "self_model": CAPABILITY_CONSTRAINED_CONTROL,
        "prediction": CAPABILITY_CONSTRAINED_CONTROL,
    }


def bootstrap_memory_trace_and_self_model(loop: Any, config: MainLoopBootstrapConfig) -> None:
    from decision import CandidateGenerator, DecisionArbiter
    from modules.graduation import GraduationTracker, TriggerSource
    from modules.memory.consolidation import (
        run_heavy_consolidation,
        run_light_consolidation,
    )
    from modules.memory.episode_summarizer import EpisodeSummarizer
    from modules.memory.event_log import EventLog, EventLogBuilder
    from modules.memory.retrieval_bundle import RetrievalBundleBuilder
    from self_model import (
        CapabilityProfile,
        ReliabilityTracker,
        ResourceState,
        SelfModelFacade,
        SelfModelState,
    )

    loop._grad_tracker = GraduationTracker(
        compile_threshold=0.8,
        teacher_exit_episode=loop.max_episodes,
    )
    loop._grad_trigger_source = TriggerSource
    loop._pending_recovery_probe = None
    loop._pending_replan = None
    from core.orchestration.testing_recovery_runtime import TestingRecoveryRuntime

    loop._testing_recovery_runtime = TestingRecoveryRuntime(loop)
    loop._event_bus = WorldModelEventBus()
    loop._belief_ledger = BeliefLedger()
    loop._belief_updater = BeliefUpdater(loop._belief_ledger)
    loop._event_bus.subscribe(loop._on_belief_event)
    loop._counterfactual = CounterfactualEngine(seed=config.seed)
    loop._mechanism_extractor = MechanismExtractor()
    loop._mechanism_writer = MechanismFormalWriter()
    loop._distillation_compiler = DistillationCompiler()
    loop._episode_summarizer = EpisodeSummarizer()
    loop._last_episode_record_id = None
    loop._event_log = EventLog(path=str(default_event_log_path()))
    loop._event_log_builder = EventLogBuilder
    loop._retrieval_bundle_builder = RetrievalBundleBuilder
    loop._run_light_consolidation = run_light_consolidation
    loop._run_heavy_consolidation = run_heavy_consolidation
    loop._consolidation_episode_interval = 3
    loop._last_heavy_consolidation_episode = 0
    loop._action_runtime = None
    loop._decision_arbiter = DecisionArbiter()
    loop._candidate_generator = CandidateGenerator()
    loop._causal_trace = CausalTraceLogger()
    loop._event_timeline = EventTimeline()
    loop._capability_profile = CapabilityProfile(agent_id=config.agent_id)
    loop._reliability_tracker = ReliabilityTracker()
    loop._resource_state = ResourceState()
    loop._self_model_state = SelfModelState()
    loop._self_model_facade = SelfModelFacade(
        reliability_tracker=loop._reliability_tracker,
        capability_profile=loop._capability_profile,
        state=loop._self_model_state,
    )


def bootstrap_learning_prediction_and_procedure(loop: Any) -> None:
    loop._learning_enabled = bool(loop._learning_enabled_by_config)
    from core.learning import CreditAssignment

    loop._credit_assignment = CreditAssignment()
    loop._update_engine = _FormalUpdateEngine(loop._shared_store)
    loop._promotion_engine = _NoopPromotionEngine()
    loop._learning_update_log = []
    loop._max_learning_updates_per_episode = 20
    loop._learning_updates_sent_this_episode = 0
    loop._learning_signal_log = []
    loop._learning_policy_snapshot = {
        "selector_bias": {},
        "agenda_prior": {},
        "recovery_shortcut": {},
        "failure_preference_policy": {},
        "retention_failure_policy": {},
    }
    loop._teacher_experiment_mode = "full"
    loop._mechanism_posterior_updater = MechanismPosteriorUpdater()
    loop._candidate_cooldown_gate = CandidateCooldownGate()
    loop._discriminating_action_selector = DiscriminatingActionSelector()
    loop._mechanism_runtime_state = {}
    loop._last_mechanism_runtime_view = {
        "mechanism_hypotheses_summary": [],
        "mechanism_control_summary": {},
    }
    loop._mechanism_control_audit_log = []
    loop._prediction_feature_builder = None
    loop._prediction_engine = PredictionEngine()
    loop._prediction_adjudicator = PredictionAdjudicator()
    loop._prediction_registry = PredictionRegistry()
    loop._prediction_enabled = bool(loop._prediction_enabled_by_config)
    loop._last_prediction_bundle_by_action_id = {}
    loop._prediction_trace_log = []
    loop._prediction_positive_miss_streak = 0
    loop._prediction_feedback = PredictionFeedbackPipeline()
    loop._prediction_miss_feedback = PredictionMissFeedbackRuntime(
        loop._belief_updater
    )
    loop._learned_dynamics_shadow_model_path = str(default_learned_dynamics_model_path())
    loop._learned_dynamics_shadow_predictor = BucketedLearnedDynamicsModel.load(
        loop._learned_dynamics_shadow_model_path
    )
    deployment_mode = str(os.getenv("LEARNED_DYNAMICS_MODE", "shadow") or "shadow").strip().lower()
    if deployment_mode not in {
        "shadow",
        "selective_routing",
        "limited_veto_promotion",
        "planner_rollout_dependence",
    }:
        deployment_mode = "shadow"
    loop._learned_dynamics_deployment_mode = deployment_mode
    loop._last_planner_runtime_payload = {}
    loop._planner_runtime_log = []
    loop._procedure_registry = _NoopUpdateEngine()
    loop._procedure_matcher = None
    loop._procedure_executor = None
    loop._procedure_pipeline = None
    loop._procedure_enabled = False
    loop._last_procedure_matches = []
    loop._procedure_proposal_log = []
    loop._procedure_execution_log = []
    loop._procedure_promotion_log = []
