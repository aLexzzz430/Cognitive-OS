"""
modules/skills/skill_rewriter.py

P0-3: Skill Rewrite System

Separated from main_loop.py for clarity. CoreMainLoop imports this module.
"""

import re
import random
from typing import List, Dict, Optional

from modules.memory.skill_registry import SkillRegistry


class SkillRewriter:
    """
    P0-3: Rewrite actions using retrieved skill objects.

    C4: Skill invalidation — when a hypothesis is refuted, all skills
    derived from it are invalidated and no longer used for rewriting.
    """
    def __init__(self, store, seed: int = 0):
        self._store = store
        self._registry = SkillRegistry(store)
        self._rng = random.Random(seed)
        self._log: List[dict] = []
        self._invalidated_skills: set = set()  # C4: invalidated skill IDs

    def invalidate_skills_for_hyp(self, hyp_id: str, skill_ids: List[str]):
        """C4: Invalidate skills derived from a refuted hypothesis."""
        for sid in skill_ids:
            self._invalidated_skills.add(sid)

    def is_skill_valid(self, skill_id: str) -> bool:
        """C4: Check if a skill is still valid (not invalidated)."""
        return skill_id not in self._invalidated_skills

    def retrieve_skills(self, hyp, top_k: int = 3) -> List[dict]:
        """
        Retrieve valid skill objects from the store that match the given hypothesis.
        Only returns skills that have not been invalidated.
        """
        matched = self._registry.match_hypothesis(
            hyp,
            top_k=top_k,
            invalidated_skill_ids=set(self._invalidated_skills),
        )
        if matched:
            return matched

        objs = self._store.retrieve(sort_by='confidence')
        skills = []
        m = re.search(r"'([^']+)'", hyp.claim)
        hyp_fn = m.group(1) if m else ''

        for obj in objs[:top_k*2]:
            content = obj.get('content', {})
            obj_id = obj.get('object_id', obj.get('id', ''))
            skill_id = f"s_{obj_id[:8]}"
            if not self.is_skill_valid(skill_id):
                continue
            conds = []
            if hyp_fn and content.get('tool_args', {}).get('function_name') == hyp_fn:
                conds.append(f'applies:{hyp_fn}')
            if hyp.id[:8] in obj_id:
                conds.append('from_hyp')
            if conds or content.get('skill_type'):
                skills.append({
                    'skill_id': skill_id,
                    'object_id': obj_id,
                    'skill_type': content.get('skill_type', 'rewrite'),
                    'content': content,
                    'conditions': conds,
                    'hints': content.get('rewrite_hints', {}),
                })
            if len(skills) >= top_k:
                break

        return skills

    def rewrite(self, base: dict, skills: List[dict], hyp) -> dict:
        """
        Rewrite base action using skills.
        
        Applies hints from matched skills:
        - force_function: force a specific function name
        - parameter_overrides: override parameters with specific values
        - suppress_if_conflict: suppress action if hypothesis contains conflict markers
        """
        if not skills:
            return base

        rew = base.copy()
        for s in skills:
            if not s['conditions']:
                continue

            hints = s['hints']
            if hints:
                fn = base.get('payload', {}).get('tool_args', {}).get('function_name', '')
                if hints.get('force_function') and fn != hints['force_function']:
                    continue
                if 'parameter_overrides' in hints:
                    kw = rew.get('payload', {}).get('tool_args', {}).get('kwargs', {})
                    kw.update(hints['parameter_overrides'])
                    rew['payload']['tool_args']['kwargs'] = kw
                if 'suppress_if_conflict' in hints and hyp:
                    if any(c in hyp.claim for c in hints['suppress_if_conflict']):
                        rew = {'kind': 'wait', 'payload': {}}
                        break

        self._log.append({'action': str(base)[:50], 'rewritten': len(skills) > 0})
        return rew

    def get_log(self) -> List[dict]:
        return list(self._log)
