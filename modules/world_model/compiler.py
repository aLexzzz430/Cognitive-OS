"""
modules/world_model/compiler.py

Stage E: Distillation Compiler

Turns high-utility world structure into lasting agent capability.

Core principle:
- "Distilled" is a VERIFIED state, not a label
- Asset must prove teacher-absent benefit before being called "distilled"
- Only goes through validator/committer for formal writes

Distillation targets:
- selector_bias: modify action selection priors
- skill_patch: update skill effectiveness
- recovery_shortcut: add to recovery decision tree
- agenda_prior: influence agenda prioritization
- representation_prior: affect representation card weighting

Asset lifecycle:
- NEW_ASSET -> LIVE_ASSET -> REUSABLE_ASSET -> COMPILED_ASSET -> DISTILLED_ASSET
- Any state -> GARBAGE (if contradicted or no utility)

Kill conditions:
- Never bypass validator/committer for formal writes
- Never call something "distilled" just because it was stored
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time


class DistillationTarget(Enum):
    """What type of lasting capability this distillation produces."""
    SELECTOR_BIAS = "selector_bias"         # Modify action selection priors
    SKILL_PATCH = "skill_patch"             # Update skill effectiveness
    RECOVERY_SHORTCUT = "recovery_shortcut" # Add to recovery decision tree
    AGENDA_PRIOR = "agenda_prior"          # Influence agenda prioritization
    REPRESENTATION_PRIOR = "representation_prior"  # Affect representation card weighting


class DistillationDecision(Enum):
    """Decision on whether to distill an asset."""
    DISTILL = "distill"                    # Promote to distilled
    COMPILE = "compile"                    # Promote to compiled only (needs more evidence)
    DEFER = "defer"                       # Not enough evidence yet
    RETIRE = "retire"                     # Move to garbage (no utility


@dataclass
class DistillationCandidate:
    """
    An asset that is being considered for distillation.
    
    Tracks eligibility criteria and compilation progress.
    """
    object_id: str
    asset_status: str                      # Current asset status
    consumption_count: int
    reuse_history: List[Dict[str, Any]]    # [(tick, episode, was_beneficial), ...]
    trigger_source: str                    # 'teacher' or 'agent'
    trigger_episode: int
    confidence: float
    content: Dict[str, Any]
    
    # Eligibility criteria
    reused_more_than_once: bool = False
    beneficial_after_reuse: bool = False
    not_teacher_only_trigger: bool = False
    survives_contradiction: bool = False
    changes_policy_behavior: bool = False
    
    # Compilation state
    compilation_target: Optional[DistillationTarget] = None
    compilation_episode: int = 0
    distillation_episode: int = 0
    
    def evaluate_eligibility(self, current_episode: int) -> Tuple[bool, List[str]]:
        """
        Check if this candidate meets distillation criteria.
        
        Returns: (is_eligible, reasons)
        """
        reasons = []
        
        # Criterion 1: Reused more than once
        self.reused_more_than_once = self.consumption_count >= 2
        if self.reused_more_than_once:
            reasons.append("✓ Reused >= 2 times")
        else:
            reasons.append(f"✗ Only reused {self.consumption_count} time(s)")
        
        # Criterion 2: Beneficial after reuse
        beneficial_count = sum(1 for r in self.reuse_history if r.get('was_beneficial', False))
        self.beneficial_after_reuse = beneficial_count >= 2
        if self.beneficial_after_reuse:
            reasons.append("✓ Beneficial on >= 2 uses")
        else:
            reasons.append(f"✗ Only beneficial on {beneficial_count} use(s)")
        
        # Criterion 3: Not teacher-only trigger
        # If teacher injected this in early episodes and it still works in agent episodes, it's not teacher-only
        teacher_episodes = [i for i in range(1, min(self.trigger_episode + 1, 4))]
        agent_episodes = [i for i in range(max(4, self.trigger_episode), current_episode + 1)]
        
        # Check if used successfully in agent episodes (episode >= 4)
        agent_uses = [r for r in self.reuse_history if r.get('episode', 0) >= 4]
        self.not_teacher_only_trigger = len(agent_uses) >= 1
        if self.not_teacher_only_trigger:
            reasons.append("✓ Used successfully in agent episodes (not teacher-only)")
        else:
            reasons.append("✗ Only used in teacher episodes or not used in agent episodes")
        
        # Criterion 4: Survives contradiction (no strong contradiction evidence)
        # For now, simplified: if confidence > 0.3, passes
        self.survives_contradiction = self.confidence > 0.3
        if self.survives_contradiction:
            reasons.append(f"✓ Confidence {self.confidence:.2f} > 0.3 (survives)")
        else:
            reasons.append(f"✗ Confidence {self.confidence:.2f} <= 0.3 (contradicted)")
        
        # Criterion 5: Changes policy/selection behavior
        # Simplified: if it influenced governance or recovery, it changed behavior
        self.changes_policy_behavior = beneficial_count >= 1 and self.consumption_count >= 2
        if self.changes_policy_behavior:
            reasons.append("✓ Beneficial + multiple uses = policy effect")
        else:
            reasons.append("✗ No clear policy effect")
        
        is_eligible = (
            self.reused_more_than_once
            and self.beneficial_after_reuse
            and self.not_teacher_only_trigger
            and self.survives_contradiction
            and self.changes_policy_behavior
        )
        
        return is_eligible, reasons


class DistillationCompiler:
    """
    Compiles eligible assets into lasting agent capabilities.
    
    Advisory only — decisions go through governance and formal write path.
    """
    
    def __init__(self):
        self._compilation_log: List[Dict[str, Any]] = []
        self._distillation_log: List[Dict[str, Any]] = []
    
    def check_distillation_eligibility(
        self,
        object_id: str,
        asset_status: str,
        consumption_count: int,
        reuse_history: List[Dict[str, Any]],
        trigger_source: str,
        trigger_episode: int,
        confidence: float,
        content: Dict[str, Any],
        current_episode: int,
    ) -> DistillationCandidate:
        """
        Check if an asset is eligible for distillation.
        
        Returns a DistillationCandidate with eligibility assessment.
        """
        candidate = DistillationCandidate(
            object_id=object_id,
            asset_status=asset_status,
            consumption_count=consumption_count,
            reuse_history=reuse_history,
            trigger_source=trigger_source,
            trigger_episode=trigger_episode,
            confidence=confidence,
            content=content,
        )
        
        is_eligible, reasons = candidate.evaluate_eligibility(current_episode)
        
        self._compilation_log.append({
            'object_id': object_id,
            'episode': current_episode,
            'is_eligible': is_eligible,
            'reasons': reasons,
            'candidate': candidate,
        })
        
        return candidate
    
    def determine_compilation_target(
        self,
        candidate: DistillationCandidate,
    ) -> Optional[DistillationTarget]:
        """
        Determine which distillation target is appropriate for this asset.
        
        Returns None if no appropriate target found.
        """
        content = candidate.content
        
        # Determine target based on content type
        content_type = content.get('type', '') if isinstance(content, dict) else ''
        
        if 'skill' in content_type.lower() or 'rewrite' in content_type.lower():
            return DistillationTarget.SKILL_PATCH
        elif 'recovery' in content_type.lower() or 'error' in content_type.lower():
            return DistillationTarget.RECOVERY_SHORTCUT
        elif 'representation' in content_type.lower() or 'pattern' in content_type.lower():
            return DistillationTarget.REPRESENTATION_PRIOR
        elif 'agenda' in content_type.lower() or 'goal' in content_type.lower():
            return DistillationTarget.AGENDA_PRIOR
        elif 'selector' in content_type.lower() or 'action' in content_type.lower():
            return DistillationTarget.SELECTOR_BIAS
        
        # Default based on asset status
        if candidate.asset_status == 'reusable_asset':
            return DistillationTarget.SELECTOR_BIAS
        elif candidate.asset_status == 'compiled_asset':
            return DistillationTarget.REPRESENTATION_PRIOR
        
        return None
    
    def compile(
        self,
        candidate: DistillationCandidate,
        current_episode: int,
    ) -> DistillationDecision:
        """
        Decide whether to distill, compile, defer, or retire this asset.
        
        Returns DistillationDecision with reasoning.
        """
        # Check eligibility
        is_eligible, _ = candidate.evaluate_eligibility(current_episode)
        
        if not is_eligible:
            # Check if it should be retired
            if not candidate.survives_contradiction or not candidate.not_teacher_only_trigger:
                return DistillationDecision.RETIRE
            return DistillationDecision.DEFER
        
        # Determine target
        target = self.determine_compilation_target(candidate)
        candidate.compilation_target = target
        candidate.compilation_episode = current_episode
        
        # Check if it meets distillation criteria (not just compilation)
        # Distillation requires teacher-absent verification
        agent_uses = [r for r in candidate.reuse_history if r.get('episode', 0) >= 4]
        has_agent_proven_benefit = any(r.get('was_beneficial', False) for r in agent_uses)
        
        if has_agent_proven_benefit:
            # Can be promoted to DISTILLED (teacher-absent benefit proven)
            return DistillationDecision.DISTILL
        else:
            # Only compiled, not yet distilled (needs more agent evidence)
            return DistillationDecision.COMPILE
    
    def verify_distillation(
        self,
        object_id: str,
        reuse_history: List[Dict[str, Any]],
        current_episode: int,
    ) -> Tuple[bool, str]:
        """
        Verify that an asset truly demonstrated distillation value.
        
        An asset is "distilled" only if:
        1. Teacher is absent (episode >= teacher_exit_episode)
        2. System reuses it later
        3. Measurable benefit remains
        
        Returns: (is_verified, reason)
        """
        # Must have uses in agent episodes (teacher absent)
        agent_uses = [r for r in reuse_history if r.get('episode', 0) >= 4]
        
        if not agent_uses:
            return False, "No uses in agent episodes (teacher may still be present)"
        
        # Check if beneficial in agent episodes
        beneficial_agent_uses = [r for r in agent_uses if r.get('was_beneficial', False)]
        
        if not beneficial_agent_uses:
            return False, "No beneficial uses in agent episodes"
        
        # Check trend: should be consistently beneficial, not just lucky
        if len(beneficial_agent_uses) < len(agent_uses) * 0.5:
            return False, f"Only {len(beneficial_agent_uses)}/{len(agent_uses)} agent uses beneficial"
        
        return True, f"Verified: {len(beneficial_agent_uses)}/{len(agent_uses)} agent uses beneficial"
    
    def get_compilation_log(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent compilation decisions."""
        return self._compilation_log[-n:]
    
    def get_distillation_log(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent distillation verifications."""
        return self._distillation_log[-n:]
