"""
modules/memory/episode_summarizer.py

Phase 2: Typed Episodic Memory

Creates typed episode_record objects at episode end.
Episode records summarize key events, actions, and outcomes.

Rules:
- episode_record is a formal memory object
- Must go through validator + committer (Step10)
- Stored in object_store with memory_type=episode_record

Episode Record Fields:
- episode_id
- tick_count
- actions: list of action summaries
- key_discoveries: what was learned
- failures: what failed
- reward_trend: positive/neutral/negative
- committed_objects: object_ids committed during episode
- belief_state_summary: belief ledger snapshot
- graduation_candidates: candidates produced
- retention_competition_summary: why probe-preserving / risk-suppressing behavior was retained
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeActionSummary:
    """Summary of a single action in an episode."""
    tick: int
    function_name: str
    reward: float
    success: bool


@dataclass
class EpisodeRecord:
    """
    Typed episodic memory object.
    
    Created at episode end via formal write path.
    Summarizes episode for later retrieval.
    """
    episode_id: int
    
    # Episode stats
    tick_count: int
    total_reward: float
    
    # Content summaries
    actions: List[Dict] = field(default_factory=list)  # [EpisodeActionSummary]
    key_discoveries: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    
    # State
    reward_trend: str = "neutral"  # "positive", "neutral", "negative"
    terminal_reached: bool = False
    
    # Memory relationships
    committed_object_ids: List[str] = field(default_factory=list)
    hypothesis_ids_created: List[str] = field(default_factory=list)
    belief_delta: Dict[str, Any] = field(default_factory=dict)  # beliefs added/updated
    
    # Metadata
    teacher_present: bool = False
    teacher_exit_episode: Optional[int] = None
    retention_competition_summary: Dict[str, Any] = field(default_factory=dict)
    source_event_ids: List[str] = field(default_factory=list)
    callable_functions: List[str] = field(default_factory=list)
    identity_implications: List[str] = field(default_factory=list)
    mechanism_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dict for storage."""
        return {
            'episode_id': self.episode_id,
            'tick_count': self.tick_count,
            'total_reward': self.total_reward,
            'actions': self.actions,
            'key_discoveries': self.key_discoveries,
            'failures': self.failures,
            'reward_trend': self.reward_trend,
            'terminal_reached': self.terminal_reached,
            'committed_object_ids': self.committed_object_ids,
            'hypothesis_ids_created': self.hypothesis_ids_created,
            'belief_delta': self.belief_delta,
            'teacher_present': self.teacher_present,
            'teacher_exit_episode': self.teacher_exit_episode,
            'retention_competition_summary': self.retention_competition_summary,
            'source_event_ids': self.source_event_ids,
            'callable_functions': self.callable_functions,
            'identity_implications': self.identity_implications,
            'mechanism_signals': self.mechanism_signals,
        }


class EpisodeSummarizer:
    """
    Creates EpisodeRecord from episode data.
    
    Called at episode end before formal write.
    """
    
    def __init__(self):
        pass
    
    def summarize(
        self,
        episode_id: int,
        episode_trace: List[Dict],
        committed_object_ids: List[str],
        hypothesis_ids: List[str],
        belief_delta: Dict[str, Any],
        teacher_present: bool,
        teacher_exit_episode: Optional[int] = None,
        retention_competition_summary: Optional[Dict[str, Any]] = None,
    ) -> EpisodeRecord:
        """
        Create EpisodeRecord from episode data.
        
        Args:
            episode_id: Episode number
            episode_trace: List of action entries from main loop
            committed_object_ids: Object IDs committed this episode
            hypothesis_ids: Hypothesis IDs created this episode
            belief_delta: Changes in belief ledger this episode
            teacher_present: Whether teacher was active this episode
            teacher_exit_episode: Episode when teacher exits (if known)
            retention_competition_summary: Aggregated retention / governance rationale summary
        
        Returns:
            EpisodeRecord ready for validator + committer
        """
        # Calculate stats
        tick_count = len(episode_trace)
        total_reward = sum(e.get('reward', 0.0) for e in episode_trace)
        
        # Summarize actions
        actions = []
        callable_functions: List[str] = []
        source_event_ids: List[str] = []
        for entry in episode_trace:
            action = entry.get('action', {})
            fn = action.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown') if isinstance(action.get('payload'), dict) else 'unknown'
            reward = entry.get('reward', 0.0)
            success = reward > 0
            tick = entry.get('tick', 0)
            source_event_ids.append(str(entry.get('event_id') or f"episode:{episode_id}:tick:{tick}"))
            if fn and fn != 'unknown':
                callable_functions.append(fn)

            actions.append({
                'tick': tick,
                'function_name': fn,
                'reward': reward,
                'success': success,
            })
        
        # Identify discoveries and failures
        key_discoveries = []
        failures = []
        
        for entry in episode_trace:
            result = entry.get('outcome', {})
            novel_api = result.get('novel_api', {}) if isinstance(result, dict) else {}
            discovered = novel_api.get('discovered_functions', []) if isinstance(novel_api, dict) else []
            
            for fn in discovered:
                key_discoveries.append(f"discovered_function:{fn}")
            
            if entry.get('reward', 0.0) < 0:
                failures.append(f"negative_reward:{entry.get('action', {}).get('kind', 'unknown')}")
        
        # Determine reward trend
        if total_reward > tick_count * 0.5:
            reward_trend = "positive"
        elif total_reward < -tick_count * 0.2:
            reward_trend = "negative"
        else:
            reward_trend = "neutral"
        
        # Check terminal
        terminal_reached = any(entry.get('outcome', {}).get('terminal', False) for entry in episode_trace)

        identity_implications: List[str] = []
        if retention_competition_summary:
            dominant_failure = str((retention_competition_summary or {}).get('dominant_retention_failure_type', '') or '')
            if dominant_failure:
                identity_implications.append(f"retention_failure:{dominant_failure}")
            if int((retention_competition_summary or {}).get('required_probe_preserved_count', 0) or 0) > 0:
                identity_implications.append("probe_preserved")
        if not teacher_present and key_discoveries:
            identity_implications.append(f"agent_discovery:{key_discoveries[0]}")

        def _normalize_refs(values: Any) -> List[str]:
            out: List[str] = []
            for value in values if isinstance(values, list) else []:
                text = str(value or '').strip()
                if text and text not in out:
                    out.append(text)
            return out

        def _normalize_colors(values: Any) -> List[int]:
            out: List[int] = []
            for value in values if isinstance(values, list) else []:
                try:
                    color_int = int(value)
                except (TypeError, ValueError):
                    continue
                if color_int not in out:
                    out.append(color_int)
            return out

        def _select_goal_support_refs(
            *,
            explicit_refs: List[str],
            engaged_refs: List[str],
            affected_refs: List[str],
            goal_refs: List[str],
            controller_anchor_ref: str,
        ) -> List[str]:
            if explicit_refs:
                return explicit_refs
            if engaged_refs:
                return [ref for ref in engaged_refs if ref != controller_anchor_ref]
            if goal_refs:
                return [
                    ref
                    for ref in affected_refs
                    if ref in goal_refs and ref != controller_anchor_ref
                ]
            return [ref for ref in affected_refs if ref != controller_anchor_ref]

        def _select_goal_support_colors(
            *,
            explicit_colors: List[int],
            affected_colors: List[int],
            goal_colors: List[int],
            clicked_color: Optional[int],
        ) -> List[int]:
            if explicit_colors:
                return explicit_colors
            if goal_colors:
                return [color for color in affected_colors if color in goal_colors]
            return [
                color
                for color in affected_colors
                if clicked_color is None or color != clicked_color
            ]

        mechanism_signals_by_signature: Dict[str, Dict[str, Any]] = {}
        for entry in episode_trace:
            if not isinstance(entry, dict):
                continue
            assessment = entry.get('goal_progress_assessment', {})
            if not isinstance(assessment, dict):
                continue
            task_progress = entry.get('task_progress', {})
            task_progress = task_progress if isinstance(task_progress, dict) else {}
            inferred_goal = entry.get('inferred_level_goal', {})
            inferred_goal = inferred_goal if isinstance(inferred_goal, dict) else {}
            action_snapshot = entry.get('action_snapshot', {})
            action_snapshot = action_snapshot if isinstance(action_snapshot, dict) else {}
            effect_signature = entry.get('action_effect_signature', {})
            effect_signature = effect_signature if isinstance(effect_signature, dict) else {}
            family_effect_attribution = entry.get('family_effect_attribution', {})
            family_effect_attribution = (
                family_effect_attribution
                if isinstance(family_effect_attribution, dict)
                else {}
            )
            clicked_family = entry.get('clicked_family', {})
            clicked_family = clicked_family if isinstance(clicked_family, dict) else {}
            outcome = entry.get('outcome', {})
            outcome = outcome if isinstance(outcome, dict) else {}
            perception = outcome.get('perception', {})
            perception = perception if isinstance(perception, dict) else {}

            progressed = bool(
                assessment.get('progressed', False)
                or task_progress.get('progressed', False)
                or float(entry.get('reward', 0.0) or 0.0) > 0.0
                or float(entry.get('information_gain', 0.0) or 0.0) > 0.0
            )

            goal_family = str(
                assessment.get('goal_family')
                or inferred_goal.get('goal_family')
                or ''
            ).strip()
            if not goal_family:
                continue

            function_name = str(action_snapshot.get('function_name') or '').strip()
            controller_anchor_ref = str(
                assessment.get('controller_anchor_ref')
                or assessment.get('clicked_anchor_ref')
                or clicked_family.get('anchor_ref')
                or ''
            ).strip()
            explicit_supported_goal_anchor_refs = _normalize_refs(
                assessment.get('controller_supported_goal_anchor_refs', [])
            )
            explicit_supported_goal_colors = _normalize_colors(
                assessment.get('controller_supported_goal_colors', [])
            )
            engaged_goal_anchor_refs = _normalize_refs(
                assessment.get('engaged_goal_anchor_refs', [])
            )
            goal_anchor_refs = _normalize_refs(inferred_goal.get('goal_anchor_refs', []))
            goal_anchor_colors = _normalize_colors(
                inferred_goal.get('goal_anchor_colors', [])
            )
            affected_anchor_refs = _normalize_refs(
                effect_signature.get('affected_anchor_refs', [])
            )
            affected_colors = _normalize_colors(effect_signature.get('affected_colors', []))
            try:
                clicked_color = int(clicked_family.get('color'))
            except (TypeError, ValueError):
                clicked_color = None

            supported_goal_anchor_refs = _select_goal_support_refs(
                explicit_refs=explicit_supported_goal_anchor_refs,
                engaged_refs=engaged_goal_anchor_refs,
                affected_refs=affected_anchor_refs,
                goal_refs=goal_anchor_refs,
                controller_anchor_ref=controller_anchor_ref,
            )
            supported_goal_colors = _select_goal_support_colors(
                explicit_colors=explicit_supported_goal_colors,
                affected_colors=affected_colors,
                goal_colors=goal_anchor_colors,
                clicked_color=clicked_color,
            )

            changed_pixel_count = int(
                perception.get(
                    'changed_pixel_count',
                    effect_signature.get('changed_pixel_count', 0),
                )
                or 0
            )
            explicit_controller_effect = bool(assessment.get('controller_effect', False))
            controller_like_reaction = bool(
                not explicit_controller_effect
                and bool(assessment.get('goal_aligned_effect', False))
                and str(assessment.get('progress_class') or '').strip()
                == 'goal_aligned_reaction'
                and controller_anchor_ref
                and controller_anchor_ref not in goal_anchor_refs
                and bool(supported_goal_anchor_refs or supported_goal_colors)
                and (
                    progressed
                    or bool(assessment.get('positive_progress', False))
                    or changed_pixel_count >= 32
                    or float(entry.get('information_gain', 0.0) or 0.0) >= 0.08
                )
                and (
                    str(family_effect_attribution.get('preference') or '').strip()
                    == 'other_family'
                    or any(ref != controller_anchor_ref for ref in supported_goal_anchor_refs)
                    or bool(supported_goal_colors)
                )
            )
            if not explicit_controller_effect and not controller_like_reaction:
                continue
            if not progressed and changed_pixel_count < 32:
                continue
            if not controller_anchor_ref and not supported_goal_anchor_refs and not supported_goal_colors:
                continue

            signature = "|".join(
                [
                    "controller_support",
                    goal_family,
                    function_name or "any_function",
                    str(len(supported_goal_anchor_refs)),
                    str(len(supported_goal_colors)),
                ]
            )
            signal = mechanism_signals_by_signature.setdefault(
                signature,
                {
                    'mechanism_kind': 'controller_support',
                    'goal_family': goal_family,
                    'action_function': function_name,
                    'controller_anchor_refs': [],
                    'supported_goal_anchor_refs': [],
                    'supported_goal_colors': [],
                    'support_count': 0,
                    'max_goal_progress_score': 0.0,
                    'total_information_gain': 0.0,
                    'max_visual_change': 0,
                    'preferred_progress_mode': str(
                        inferred_goal.get('preferred_progress_mode') or 'expand_anchor_coverage'
                    ),
                    'requires_multi_anchor_coordination': bool(
                        inferred_goal.get('requires_multi_anchor_coordination', False)
                        or len(supported_goal_anchor_refs) > 1
                    ),
                },
            )
            signal['support_count'] = int(signal.get('support_count', 0) or 0) + 1
            signal['max_goal_progress_score'] = max(
                float(signal.get('max_goal_progress_score', 0.0) or 0.0),
                float(
                    assessment.get('goal_progress_score', task_progress.get('goal_progress_score', 0.0))
                    or 0.0
                ),
            )
            signal['total_information_gain'] = float(
                signal.get('total_information_gain', 0.0) or 0.0
            ) + float(entry.get('information_gain', 0.0) or 0.0)
            signal['max_visual_change'] = max(
                int(signal.get('max_visual_change', 0) or 0),
                changed_pixel_count,
            )
            for ref in [controller_anchor_ref]:
                if ref and ref not in signal['controller_anchor_refs']:
                    signal['controller_anchor_refs'].append(ref)
            for ref in supported_goal_anchor_refs:
                if ref not in signal['supported_goal_anchor_refs']:
                    signal['supported_goal_anchor_refs'].append(ref)
            for color_int in supported_goal_colors:
                if color_int not in signal['supported_goal_colors']:
                    signal['supported_goal_colors'].append(color_int)

        mechanism_signals = sorted(
            mechanism_signals_by_signature.values(),
            key=lambda row: (
                -int(row.get('support_count', 0) or 0),
                -float(row.get('max_goal_progress_score', 0.0) or 0.0),
                str(row.get('goal_family', '') or ''),
                str(row.get('action_function', '') or ''),
            ),
        )

        return EpisodeRecord(
            episode_id=episode_id,
            tick_count=tick_count,
            total_reward=total_reward,
            actions=actions,
            key_discoveries=key_discoveries,
            failures=failures,
            reward_trend=reward_trend,
            terminal_reached=terminal_reached,
            committed_object_ids=committed_object_ids,
            hypothesis_ids_created=hypothesis_ids,
            belief_delta=belief_delta,
            teacher_present=teacher_present,
            teacher_exit_episode=teacher_exit_episode,
            retention_competition_summary=dict(retention_competition_summary or {}),
            source_event_ids=source_event_ids,
            callable_functions=list(dict.fromkeys(callable_functions)),
            identity_implications=list(dict.fromkeys(identity_implications)),
            mechanism_signals=mechanism_signals,
        )
