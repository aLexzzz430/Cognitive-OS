from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.cognition.unified_context import UnifiedCognitiveContext
from core.objects import (
    OBJECT_TYPE_AUTOBIOGRAPHICAL,
    OBJECT_TYPE_DISCRIMINATING_TEST,
    OBJECT_TYPE_HYPOTHESIS,
    OBJECT_TYPE_IDENTITY,
    OBJECT_TYPE_REPRESENTATION,
    OBJECT_TYPE_SKILL,
    OBJECT_TYPE_TRANSFER,
)
from core.orchestration.state_abstraction import (
    summarize_cognitive_object_records,
    summarize_evidence_queue,
    summarize_goal_agenda,
    summarize_long_horizon_commitments,
    summarize_uncertainty_vector,
    summarize_workspace_budget_state,
)
from modules.world_model.protocol import WorldModelControlProtocol
from modules.world_model.task_frame import infer_task_frame
from modules.world_model.object_binding import build_object_bindings
from modules.world_model.goal_hypothesis import build_goal_hypotheses, summarize_solver_state
from modules.world_model.mechanism_hypothesis import build_mechanism_hypotheses, summarize_mechanism_control


@dataclass(frozen=True)
class UnifiedContextInput:
    unified_enabled: bool
    unified_ablation_mode: str
    obs: Optional[Dict[str, Any]]
    continuity_snapshot: Optional[Dict[str, Any]]
    world_model_summary: Optional[Dict[str, Any]]
    self_model_summary: Optional[Dict[str, Any]]
    plan_summary: Optional[Dict[str, Any]]
    current_task: str
    active_hypotheses: Optional[List[Dict[str, Any]]]
    episode_trace_tail: Optional[List[Dict[str, Any]]]
    retrieval_should_query: bool
    probe_pressure: float
    ablation_mode_validated: Optional[str] = None
    retrieval_pressure: Optional[float] = None
    recent_failures: Optional[int] = None
    world_shift_risk: Optional[float] = None
    workspace_state: Optional[Dict[str, Any]] = None
    cognitive_object_records: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _selector_id_for_row(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ''
    for key in (
        'selector_id',
        'object_id',
        'test_id',
        'experiment_id',
        'program_id',
        'output_id',
        'hypothesis_id',
        'function_name',
    ):
        text = str(row.get(key, '') or '').strip()
        if text:
            return text
    return ''


def _selector_ids(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    selected: List[str] = []
    for item in value:
        if isinstance(item, dict):
            text = _selector_id_for_row(item)
        else:
            text = str(item or '').strip()
        if text:
            selected.append(text)
    return selected


def _top_goal_description(continuity_snapshot: Optional[Dict[str, Any]]) -> str:
    if not isinstance(continuity_snapshot, dict):
        return ''
    top_goal = continuity_snapshot.get('top_goal', {})
    if isinstance(top_goal, dict):
        return str(top_goal.get('description', '') or '')
    return str(getattr(top_goal, 'description', '') or '')


def _records_for_type(
    records_by_type: Dict[str, List[Dict[str, Any]]],
    object_type: str,
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    return summarize_cognitive_object_records(
        records_by_type.get(object_type, []),
        limit=limit,
    )


def _prefer_selected_rows(
    rows: List[Dict[str, Any]],
    selected_ids: List[str],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if not selected_ids:
        return rows[:limit]
    selected = {str(item or '') for item in selected_ids if str(item or '')}
    ordered = [row for row in rows if _selector_id_for_row(row) in selected]
    ordered.extend(
        row for row in rows
        if _selector_id_for_row(row) not in selected
    )
    return ordered[:limit]


def _merge_runtime_rows(
    runtime_rows: List[Dict[str, Any]],
    formal_rows: List[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in list(runtime_rows or []) + list(formal_rows or []):
        if not isinstance(row, dict):
            continue
        selector_id = _selector_id_for_row(row)
        dedupe_key = selector_id or f"summary:{str(row.get('summary', '') or row.get('description', '') or '')}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(dict(row))
        if len(merged) >= limit:
            break
    return merged[:limit]


def _merge_competing_hypotheses(
    formal_rows: List[Dict[str, Any]],
    transient_rows: List[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in list(formal_rows or []) + list(transient_rows or []):
        if not isinstance(row, dict):
            continue
        object_id = str(
            row.get("object_id", "")
            or row.get("hypothesis_id", "")
            or ""
        )
        dedupe_key = object_id or f"summary:{str(row.get('summary', '') or row.get('description', '') or '')}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        merged.append(dict(row))
        if len(merged) >= limit:
            break
    return merged[:limit]


def _merge_runtime_hypothesis_views(
    runtime_rows: List[Dict[str, Any]],
    object_rows: List[Dict[str, Any]],
    *,
    limit: int = 8,
) -> List[Dict[str, Any]]:
    if not runtime_rows:
        return list(object_rows or [])[:limit]
    object_rows_by_id: Dict[str, Dict[str, Any]] = {}
    for row in list(object_rows or []):
        if not isinstance(row, dict):
            continue
        selector_id = _selector_id_for_row(row)
        if selector_id:
            object_rows_by_id[selector_id] = dict(row)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for row in list(runtime_rows or []):
        if not isinstance(row, dict):
            continue
        selector_id = _selector_id_for_row(row)
        merged_row = dict(object_rows_by_id.get(selector_id, {}))
        merged_row.update(dict(row))
        merged.append(merged_row)
        if selector_id:
            seen.add(selector_id)
        if len(merged) >= limit:
            return merged[:limit]
    for row in list(object_rows or []):
        if not isinstance(row, dict):
            continue
        selector_id = _selector_id_for_row(row)
        if selector_id and selector_id in seen:
            continue
        merged.append(dict(row))
        if len(merged) >= limit:
            break
    return merged[:limit]


def _promote_mechanism_hypotheses(
    rows: List[Dict[str, Any]],
    *,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    promoted: List[Dict[str, Any]] = []
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        predicted_action_effects = row.get('predicted_action_effects', {})
        predicted_observation_tokens = row.get('predicted_observation_tokens', [])
        if not isinstance(predicted_action_effects, dict) and not isinstance(predicted_observation_tokens, list):
            continue
        promoted_row = dict(row)
        hypothesis_id = str(promoted_row.get('hypothesis_id', '') or promoted_row.get('object_id', '') or '')
        if hypothesis_id and not promoted_row.get('object_id'):
            promoted_row['object_id'] = hypothesis_id
        promoted_row.setdefault('object_type', OBJECT_TYPE_HYPOTHESIS)
        promoted_row.setdefault('source', 'world_model_mechanism')
        metadata = _dict_or_empty(promoted_row.get('metadata', {}))
        metadata.setdefault('mechanism_hypothesis', True)
        promoted_row['metadata'] = metadata
        promoted.append(promoted_row)
        if len(promoted) >= limit:
            break
    return promoted[:limit]


def _is_mechanism_prior_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    memory_layer = str(row.get('memory_layer', '') or '').strip().lower()
    memory_type = str(row.get('memory_type', '') or '').strip().lower()
    content = row.get('content', {})
    content = content if isinstance(content, dict) else {}
    content_type = str(content.get('type', '') or '').strip().lower()
    return (
        memory_layer == 'mechanism'
        or memory_type == 'mechanism_summary'
        or content_type == 'mechanism_summary'
    )


def _is_mechanism_hypothesis_row(row: Dict[str, Any]) -> bool:
    if not isinstance(row, dict):
        return False
    content = row.get('content', {})
    content = content if isinstance(content, dict) else {}
    metadata = content.get('metadata', row.get('metadata', {}))
    metadata = metadata if isinstance(metadata, dict) else {}
    hypothesis_type = str(
        row.get('hypothesis_type')
        or content.get('hypothesis_type')
        or ''
    ).strip().lower()
    source = str(
        row.get('source')
        or content.get('source')
        or ''
    ).strip().lower()
    return (
        hypothesis_type == 'mechanism_hypothesis'
        or bool(metadata.get('mechanism_hypothesis', False))
        or source == 'world_model_mechanism'
    )


def _identity_state(
    *,
    object_workspace: Dict[str, Any],
    identity_rows: List[Dict[str, Any]],
    self_model_summary: Dict[str, Any],
) -> Dict[str, Any]:
    snapshot = _dict_or_empty(object_workspace.get('current_identity_snapshot', {}))
    if snapshot:
        return snapshot
    if identity_rows:
        return dict(identity_rows[0])
    self_model_state = _dict_or_empty(self_model_summary.get('self_model_state', {}))
    identity_markers = _dict_or_empty(self_model_state.get('identity_markers', {}))
    durable_identity = _dict_or_empty(self_model_state.get('durable_identity', {}))
    if not self_model_state and not identity_markers:
        return {}
    return {
        'summary': str(
            durable_identity.get('narrative', '')
            or self_model_state.get('value_commitments_summary', '')
            or ''
        ),
        'identity_markers': identity_markers,
        'durable_identity': durable_identity,
        'continuity_confidence': float(
            self_model_state.get(
                'continuity_confidence',
                self_model_summary.get('continuity_confidence', 0.5),
            ) or 0.5
        ),
    }


def _autobiographical_state(
    *,
    object_workspace: Dict[str, Any],
    autobiographical_rows: List[Dict[str, Any]],
    self_model_summary: Dict[str, Any],
) -> Dict[str, Any]:
    summary = _dict_or_empty(object_workspace.get('autobiographical_summary', {}))
    if summary:
        return summary
    if autobiographical_rows:
        return dict(autobiographical_rows[0])
    self_model_state = _dict_or_empty(self_model_summary.get('self_model_state', {}))
    return _dict_or_empty(self_model_state.get('autobiographical_summary', {}))


def _deliberation_mode(
    *,
    retrieval_should_query: bool,
    probe_pressure: float,
    plan_summary: Dict[str, Any],
    candidate_tests: List[Dict[str, Any]],
    competing_hypotheses: List[Dict[str, Any]],
) -> str:
    if candidate_tests and probe_pressure >= 0.4:
        return 'test_selection'
    if retrieval_should_query:
        return 'retrieval'
    if bool(plan_summary.get('has_plan', False)):
        return 'plan_execution'
    if competing_hypotheses:
        return 'hypothesis_refinement'
    return 'reactive'


class UnifiedContextBuilder:
    """Build UnifiedCognitiveContext from explicit typed input."""

    @staticmethod
    def build(input_obj: UnifiedContextInput) -> UnifiedCognitiveContext:
        wm_summary = dict(input_obj.world_model_summary or {})
        sm_summary = dict(input_obj.self_model_summary or {})
        plan_summary = dict(input_obj.plan_summary or {})
        active_hypotheses = list(input_obj.active_hypotheses or [])
        trace_tail = list(input_obj.episode_trace_tail or [])
        obs = dict(input_obj.obs or {})
        workspace_state = dict(input_obj.workspace_state or {})
        object_workspace = _dict_or_empty(workspace_state.get('object_workspace', {}))
        goal_stack = _dict_or_empty(workspace_state.get('goal_stack', {}))
        self_summary_state = _dict_or_empty(workspace_state.get('self_summary', {}))
        governance_context = _dict_or_empty(workspace_state.get('governance_context', {}))
        records_by_type = {
            str(object_type): _list_of_dicts(rows)
            for object_type, rows in (input_obj.cognitive_object_records or {}).items()
        }
        top_goal = _top_goal_description(input_obj.continuity_snapshot)

        ablation_mode = str(input_obj.ablation_mode_validated or input_obj.unified_ablation_mode or 'stripped')
        if ablation_mode not in {'stripped', 'hard_off'}:
            ablation_mode = 'stripped'

        object_bindings_summary = build_object_bindings(obs, wm_summary)
        representation_records = [dict(row) for row in list(records_by_type.get(OBJECT_TYPE_REPRESENTATION, []) or []) if isinstance(row, dict)]
        formal_hypothesis_records = _list_of_dicts(records_by_type.get(OBJECT_TYPE_HYPOTHESIS, []))
        formal_competing_hypotheses = _records_for_type(records_by_type, OBJECT_TYPE_HYPOTHESIS, limit=8)
        runtime_competing_hypothesis_objects = _list_of_dicts(
            object_workspace.get('competing_hypothesis_objects', [])
        )
        runtime_competing_hypothesis_object_summaries = summarize_cognitive_object_records(
            runtime_competing_hypothesis_objects,
            limit=8,
        )
        initial_goal_prior_rows = [
            dict(row)
            for row in _list_of_dicts(object_workspace.get('analyst_hypothesis_candidates', []))
            if str(row.get('memory_type', '') or '').strip().lower() == 'analyst_initial_goal_prior'
        ]
        mechanism_prior_rows = _prefer_selected_rows(
            [row for row in representation_records if _is_mechanism_prior_row(row)],
            [str(item) for item in list(object_workspace.get('mechanism_object_ids', []) or []) if str(item or '')],
            limit=6,
        )
        mechanism_hypothesis_rows = _prefer_selected_rows(
            [
                dict(row)
                for row in (runtime_competing_hypothesis_objects + formal_hypothesis_records)
                if _is_mechanism_hypothesis_row(row)
            ],
            [str(item) for item in list(object_workspace.get('mechanism_object_ids', []) or []) if str(item or '')],
            limit=6,
        )
        task_frame_world_summary = dict(wm_summary)
        if mechanism_prior_rows:
            task_frame_world_summary['mechanism_priors'] = [dict(row) for row in mechanism_prior_rows]
        if mechanism_hypothesis_rows:
            task_frame_world_summary['mechanism_hypothesis_objects'] = [dict(row) for row in mechanism_hypothesis_rows]
        if initial_goal_prior_rows:
            task_frame_world_summary['initial_goal_priors'] = [dict(row) for row in initial_goal_prior_rows[:4]]
        task_frame_summary = infer_task_frame(obs, task_frame_world_summary, object_bindings_summary, trace_tail)
        if mechanism_prior_rows:
            task_frame_summary['mechanism_priors'] = [dict(row) for row in mechanism_prior_rows]
        if mechanism_hypothesis_rows:
            task_frame_summary['mechanism_hypothesis_objects'] = [dict(row) for row in mechanism_hypothesis_rows]
        if initial_goal_prior_rows:
            task_frame_summary['initial_goal_priors'] = [dict(row) for row in initial_goal_prior_rows[:4]]
        goal_hypotheses_summary = build_goal_hypotheses(obs, task_frame_summary, object_bindings_summary, trace_tail)
        solver_state_summary = summarize_solver_state(task_frame_summary, object_bindings_summary, goal_hypotheses_summary)
        mechanism_hypotheses_summary = build_mechanism_hypotheses(
            obs,
            task_frame_summary,
            object_bindings_summary,
            goal_hypotheses_summary,
            trace_tail,
        )
        mechanism_control_summary = summarize_mechanism_control(mechanism_hypotheses_summary)
        active_beliefs_summary = dict(wm_summary)
        if mechanism_hypotheses_summary:
            active_beliefs_summary['mechanism_hypotheses'] = [
                dict(item) for item in mechanism_hypotheses_summary if isinstance(item, dict)
            ]
            active_beliefs_summary['mechanism_hypotheses_summary'] = [
                dict(item) for item in mechanism_hypotheses_summary if isinstance(item, dict)
            ]
            active_beliefs_summary['mechanism_control_summary'] = dict(mechanism_control_summary)

        surfaced_representations = _prefer_selected_rows(
            _records_for_type(records_by_type, OBJECT_TYPE_REPRESENTATION, limit=10),
            _selector_ids(object_workspace.get('surfaced_object_ids', [])),
            limit=10,
        )
        runtime_competing_hypotheses = _list_of_dicts(
            object_workspace.get('competing_hypotheses', [])
        )
        runtime_competing_hypotheses = _merge_runtime_hypothesis_views(
            runtime_competing_hypotheses,
            runtime_competing_hypothesis_object_summaries,
            limit=8,
        )
        runtime_active_hypotheses_summary = _list_of_dicts(
            object_workspace.get('active_hypotheses_summary', [])
        )
        if runtime_active_hypotheses_summary:
            active_hypotheses = _merge_runtime_rows(
                runtime_active_hypotheses_summary,
                active_hypotheses,
                limit=8,
            )
        elif runtime_competing_hypothesis_object_summaries:
            active_hypotheses = _merge_runtime_rows(
                runtime_competing_hypothesis_object_summaries,
                active_hypotheses,
                limit=8,
            )
        transient_analyst_hypotheses = _list_of_dicts(object_workspace.get('analyst_hypothesis_candidates', []))
        promoted_mechanism_hypotheses = _promote_mechanism_hypotheses(mechanism_hypotheses_summary, limit=3)
        competing_hypotheses = _merge_competing_hypotheses(
            runtime_competing_hypotheses + runtime_competing_hypothesis_object_summaries + formal_competing_hypotheses,
            transient_analyst_hypotheses + promoted_mechanism_hypotheses,
            limit=8,
        )
        if not competing_hypotheses:
            competing_hypotheses = _merge_competing_hypotheses(
                summarize_cognitive_object_records(active_hypotheses, limit=6),
                promoted_mechanism_hypotheses,
                limit=8,
            )
        runtime_candidate_tests = _list_of_dicts(object_workspace.get('candidate_tests', []))
        formal_candidate_tests = _records_for_type(records_by_type, OBJECT_TYPE_DISCRIMINATING_TEST, limit=8)
        candidate_tests = _prefer_selected_rows(
            _merge_runtime_rows(runtime_candidate_tests, formal_candidate_tests, limit=8),
            _selector_ids(object_workspace.get('active_tests', [])),
            limit=8,
        )
        active_skills = _records_for_type(records_by_type, OBJECT_TYPE_SKILL, limit=8)
        transfer_candidates = _records_for_type(records_by_type, OBJECT_TYPE_TRANSFER, limit=8)
        identity_rows = _records_for_type(records_by_type, OBJECT_TYPE_IDENTITY, limit=2)
        autobiographical_rows = _records_for_type(records_by_type, OBJECT_TYPE_AUTOBIOGRAPHICAL, limit=2)
        identity_state = _identity_state(
            object_workspace=object_workspace,
            identity_rows=identity_rows,
            self_model_summary=sm_summary,
        )
        autobiographical_state = _autobiographical_state(
            object_workspace=object_workspace,
            autobiographical_rows=autobiographical_rows,
            self_model_summary=sm_summary,
        )
        candidate_programs = _list_of_dicts(object_workspace.get('candidate_programs', []))
        candidate_outputs = _list_of_dicts(object_workspace.get('candidate_outputs', []))
        ranked_discriminating_experiments = _list_of_dicts(
            object_workspace.get('ranked_discriminating_experiments', [])
        )
        posterior_summary = _dict_or_empty(object_workspace.get('posterior_summary', {}))

        progress_markers: List[Dict[str, Any]] = []
        for row in trace_tail:
            if not isinstance(row, dict):
                continue
            reward = float(row.get('reward', 0.0) or 0.0)
            if reward > 0.0:
                progress_markers.append({
                    'tick': int(row.get('tick', 0) or 0),
                    'reward': reward,
                })

        recent_failures_count = input_obj.recent_failures
        if recent_failures_count is None:
            recent_failures_count = sum(
                1 for entry in trace_tail
                if float(entry.get('reward', 0.0) or 0.0) < 0.0
            )

        shift_risk = input_obj.world_shift_risk
        if shift_risk is None:
            wm_control = WorldModelControlProtocol.from_context({'world_model_summary': wm_summary})
            shift_risk = float(wm_control.state_shift_risk)
        retrieval_pressure = input_obj.retrieval_pressure
        if retrieval_pressure is None:
            retrieval_pressure = 1.0 if bool(input_obj.retrieval_should_query) else 0.0

        uncertainty_vector = summarize_uncertainty_vector(
            world_shift_risk=shift_risk,
            retrieval_pressure=retrieval_pressure,
            probe_pressure=input_obj.probe_pressure,
            active_hypotheses=competing_hypotheses or active_hypotheses,
            self_model_summary=sm_summary,
        )
        budget_state = summarize_workspace_budget_state(
            self_summary=self_summary_state,
            governance_context=governance_context,
            self_model_summary=sm_summary,
            plan_summary=plan_summary,
            retrieval_pressure=retrieval_pressure,
            probe_pressure=input_obj.probe_pressure,
            uncertainty_vector=uncertainty_vector,
        )
        goal_agenda = summarize_goal_agenda(
            goal_stack=goal_stack,
            continuity_snapshot=input_obj.continuity_snapshot,
            plan_summary=plan_summary,
        )
        long_horizon_commitments = summarize_long_horizon_commitments(
            goal_stack=goal_stack,
            continuity_snapshot=input_obj.continuity_snapshot,
            plan_summary=plan_summary,
            identity_state=identity_state,
        )
        evidence_queue = summarize_evidence_queue(trace_tail, limit=6)
        deliberation_mode = _deliberation_mode(
            retrieval_should_query=bool(input_obj.retrieval_should_query),
            probe_pressure=float(input_obj.probe_pressure or 0.0),
            plan_summary=plan_summary,
            candidate_tests=candidate_tests,
            competing_hypotheses=competing_hypotheses,
        )
        workspace_provenance = {
            'builder': 'UnifiedContextBuilder',
            'source_contract': 'ContextProvider',
            'workspace_state_keys': sorted(workspace_state.keys()),
            'transient_analyst_hypothesis_count': len(transient_analyst_hypotheses),
            'runtime_competing_hypothesis_object_count': len(runtime_competing_hypothesis_objects),
            'initial_goal_prior_count': len(initial_goal_prior_rows),
            'object_counts': {
                OBJECT_TYPE_REPRESENTATION: len(surfaced_representations),
                OBJECT_TYPE_HYPOTHESIS: len(competing_hypotheses),
                OBJECT_TYPE_DISCRIMINATING_TEST: len(candidate_tests),
                OBJECT_TYPE_SKILL: len(active_skills),
                OBJECT_TYPE_TRANSFER: len(transfer_candidates),
                OBJECT_TYPE_IDENTITY: len(identity_rows),
                OBJECT_TYPE_AUTOBIOGRAPHICAL: len(autobiographical_rows),
            },
            'trace_events': len(evidence_queue),
        }

        if not input_obj.unified_enabled:
            if ablation_mode == 'hard_off':
                return UnifiedCognitiveContext()
            minimal_sm = {
                'global_reliability': float(sm_summary.get('global_reliability', 0.5) or 0.5),
                'recovery_availability': float(sm_summary.get('recovery_availability', 0.5) or 0.5),
                'resource_tightness': str(sm_summary.get('resource_tightness', 'normal') or 'normal'),
                'recent_failure_modes': list(sm_summary.get('recent_failure_modes', []))
                if isinstance(sm_summary.get('recent_failure_modes', []), list)
                else [],
            }
            return UnifiedCognitiveContext.from_parts(
                current_goal=top_goal,
                current_task=str(input_obj.current_task or ''),
                self_model_summary=minimal_sm,
                plan_state_summary={'has_plan': bool(plan_summary.get('has_plan', False))},
                resource_pressure=str(minimal_sm.get('resource_tightness', 'normal') or 'normal'),
                task_frame_summary=task_frame_summary,
                object_bindings_summary=object_bindings_summary,
                goal_hypotheses_summary=goal_hypotheses_summary,
                solver_state_summary=solver_state_summary,
                mechanism_hypotheses_summary=mechanism_hypotheses_summary,
                mechanism_control_summary=mechanism_control_summary,
                surfaced_representations=surfaced_representations,
                competing_hypotheses=competing_hypotheses,
                candidate_tests=candidate_tests,
                active_skills=active_skills,
                transfer_candidates=transfer_candidates,
                identity_state=identity_state,
                autobiographical_state=autobiographical_state,
                candidate_programs=candidate_programs,
                candidate_outputs=candidate_outputs,
                ranked_discriminating_experiments=ranked_discriminating_experiments,
                posterior_summary=posterior_summary,
                deliberation_budget=budget_state.get('deliberation_budget', {}),
                deliberation_mode=deliberation_mode,
                uncertainty_vector=uncertainty_vector,
                evidence_queue=evidence_queue,
                workspace_provenance=workspace_provenance,
                safety_budget=budget_state.get('safety_budget', {}),
                compute_budget=budget_state.get('compute_budget', {}),
                goal_agenda=goal_agenda,
                long_horizon_commitments=long_horizon_commitments,
            )

        return UnifiedCognitiveContext.from_parts(
            current_goal=top_goal,
            current_task=str(input_obj.current_task or ''),
            active_beliefs_summary=active_beliefs_summary,
            active_hypotheses_summary=active_hypotheses,
            plan_state_summary=plan_summary,
            self_model_summary=sm_summary,
            recent_failure_profile=list(sm_summary.get('recent_failure_modes', []))
            if isinstance(sm_summary.get('recent_failure_modes', []), list)
            else [],
            recent_progress_markers=progress_markers,
            retrieval_pressure=float(retrieval_pressure or 0.0),
            retrieval_triggered=bool(input_obj.retrieval_should_query),
            probe_pressure=float(input_obj.probe_pressure or 0.0),
            resource_pressure=str(sm_summary.get('resource_tightness', 'normal') or 'normal'),
            world_shift_risk=float(shift_risk or 0.0),
            task_frame_summary=task_frame_summary,
            object_bindings_summary=object_bindings_summary,
            goal_hypotheses_summary=goal_hypotheses_summary,
            solver_state_summary=solver_state_summary,
            mechanism_hypotheses_summary=mechanism_hypotheses_summary,
            mechanism_control_summary=mechanism_control_summary,
            surfaced_representations=surfaced_representations,
            competing_hypotheses=competing_hypotheses,
            candidate_tests=candidate_tests,
            active_skills=active_skills,
            transfer_candidates=transfer_candidates,
            identity_state=identity_state,
            autobiographical_state=autobiographical_state,
            candidate_programs=candidate_programs,
            candidate_outputs=candidate_outputs,
            ranked_discriminating_experiments=ranked_discriminating_experiments,
            posterior_summary=posterior_summary,
            deliberation_budget=budget_state.get('deliberation_budget', {}),
            deliberation_mode=deliberation_mode,
            uncertainty_vector=uncertainty_vector,
            evidence_queue=evidence_queue,
            workspace_provenance=workspace_provenance,
            safety_budget=budget_state.get('safety_budget', {}),
            compute_budget=budget_state.get('compute_budget', {}),
            goal_agenda=goal_agenda,
            long_horizon_commitments=long_horizon_commitments,
        )
