"""
core/surfaces/novel_api_governance.py — Extended with Merge/Update Lifecycle

Adds 4 gates for object lifecycle management:
  Gate 6: Merge legality (same content_hash → MERGE_UPDATE_EXISTING)
  Gate 7: Evidence strengthening (support set expansion)
  Gate 8: Contradiction handling (WEAKEN/REOPEN/SPLIT/INVALIDATE)
  Gate 9: Later utility (updated objects used in subsequent episodes)

Each merge produces a MergeRecord tracking:
  - original object
  - new evidence
  - support set expansion
  - confidence evolution
  - provenance chain
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import hashlib
import re


# =============================================================================
# Governance Decision Output
# =============================================================================

@dataclass
class SurfaceGovernanceDecision:
    """Explicit adjudication for every candidate entering Step 10."""
    candidate_id: str
    decision: str
    reason_code: str
    reason_detail: str
    target_object_id: Optional[str] = None
    evidence_ids_used: list[str] = field(default_factory=list)
    content_hash: str = ""

    def __repr__(self):
        return (f"SurfaceGovernanceDecision(cand={self.candidate_id}, "
                f"decision={self.decision}, code={self.reason_code})")


# =============================================================================
# Merge / Lifecycle Decision Codes
# =============================================================================

# Gate 6 — Merge legality
ACCEPT_NEW = "ACCEPT_NEW"
MERGE_UPDATE_EXISTING = "MERGE_UPDATE_EXISTING"
REJECT_DUPLICATE = "REJECT_DUPLICATE"

# Gate 8 — Contradiction handling
WEAKEN_EXISTING = "WEAKEN_EXISTING"
REOPEN_OBJECT = "REOPEN_OBJECT"
SPLIT_OBJECT = "SPLIT_OBJECT"
INVALIDATE_OBJECT = "INVALIDATE_OBJECT"

# Other reject codes
REJECT_MISSING_REQUIRED_FIELD = "REJECT_MISSING_REQUIRED_FIELD"
REJECT_EMPTY_REQUIRED_FIELD = "REJECT_EMPTY_REQUIRED_FIELD"
REJECT_NO_SUPPORTING_EVIDENCE = "REJECT_NO_SUPPORTING_EVIDENCE"
REJECT_EVIDENCE_ID_NOT_FOUND = "REJECT_EVIDENCE_ID_NOT_FOUND"
REJECT_UNSUPPORTED_PROPOSAL_TYPE = "REJECT_UNSUPPORTED_PROPOSAL_TYPE"
REJECT_CLAIM_EVIDENCE_MISMATCH = "REJECT_CLAIM_EVIDENCE_MISMATCH"
REJECT_EMPTY_CONTENT = "REJECT_EMPTY_CONTENT"
REJECT_TEMPLATE_STUB = "REJECT_TEMPLATE_STUB"
REJECT_NO_ACTIONABLE_NEXT_STEP = "REJECT_NO_ACTIONABLE_NEXT_STEP"
REJECT_LOW_INFORMATION_CONTENT = "REJECT_LOW_INFORMATION_CONTENT"
REJECT_STOPWORD_PLAINTEXT = "REJECT_STOPWORD_PLAINTEXT"

# Leak codes
REJECT_FORWARD_LOOKING_ACTION = "REJECT_FORWARD_LOOKING_ACTION"
REJECT_UNOBSERVED_VALUE = "REJECT_UNOBSERVED_VALUE"
REJECT_INTERNAL_STATE_LEAK = "REJECT_INTERNAL_STATE_LEAK"


# =============================================================================
# Content Hash Utilities
# =============================================================================

STOPWORD_TEMPLATES = {
    "need more research",
    "further investigation required",
    "something may be wrong",
    "explore this issue",
    "further research needed",
    "to be determined",
    "tbd",
    "undefined",
    "placeholder",
    "null",
}

GENERIC_STOPWORDS = {
    'function', 'functions', 'tool', 'tools', 'call', 'calls', 'called',
    'invoke', 'invokes', 'invoked', 'api', 'apis',
    'error', 'errors', 'data', 'datum', 'param', 'params', 'parameter', 'parameters',
    'value', 'values', 'result', 'results', 'resulted',
    'investigate', 'investigation', 'explore', 'explores', 'explored',
    'check', 'checks', 'test', 'tests', 'tested',
    'object', 'objects', 'item', 'items', 'entry', 'entries',
    'array', 'list', 'set', 'map', 'dict', 'string', 'number', 'int', 'float',
    'something', 'nothing', 'thing', 'things', 'stuff',
    'may', 'might', 'could', 'would', 'should', 'must', 'need', 'needs',
    'what', 'which', 'how', 'why', 'when', 'where', 'who',
    'does', 'doesn', 'is', 'are', 'was', 'were', 'be', 'been',
    'accept', 'accepted', 'accepts', 'reject', 'rejected', 'rejects',
    'null', 'none', 'nil', 'empty', 'zero', 'one', 'two', 'first', 'last',
    'before', 'after', 'during', 'through', 'then', 'than',
    'more', 'less', 'most', 'least', 'some', 'any', 'all', 'each', 'every',
    'valid', 'invalid', 'type', 'types', 'kind', 'kinds',
}


# =============================================================================
# Leak Detection Patterns (Gate 5)
# =============================================================================

LEAK_PATTERNS = {
    'L1_forward_action': [
        'next_best_action',
        'recommended_action',
        'optimal_action',
        'best_next',
        'action_to_take',
        'should_do_next',
        'must_call_next',
        'plan_next',
        'best_follow_up',
        'recommended_move',
        'next_thing_to_do',
        'preferred_action',
        'should_now',
        'proceed_by',
        'natural_language_next_action',
        'optimal_move',
        'best_next_step',
        'recommended_next',
        'suggested_action',
        'action_recommendation',
        'next_recommended',
        'should_do_this',
        'must_do_next',
        'call_next',
        'invoke_next',
    ],
    'L2_unobserved_value': [
        'hidden_value',
        'unobserved',
        'secret',
        'solution',
        'correct_answer',
        'ground_truth',
        'true_value',
        'real_value',
        'computed_value',
        'secret_code',
        'hidden_param',
    ],
    # Partial/stem patterns for L3 (matched against full all_text, not just content dict)
    # NOTE: Only use stems that are unambiguous. 'complete' removed — too broad
    # (catches 'analysis_complete', 'recording_complete' which are benign).
    # Rely on exact phrase matches in L3_internal_state for completion-related leaks.
    'L3_partial_stems': [
        'require',      # matches 'requirement', 'requirements', 'required'
        'prerequis',    # matches 'prerequisite', 'prerequisites', 'prerequisite_met'
        'satisf',       # matches 'satisfied', 'satisfies', 'satisfaction'
        'intern',       # matches 'internal', '_internal', 'internal_state'
        'hidden',       # matches 'hidden_preconditions', 'hidden_state'
        'kill',         # matches 'kill_condition', 'kill_flag'
        'solv',         # matches 'solved', 'solve', 'solution'
        'nothing left', # matches 'nothing left', 'nothing remaining'
        'no further',   # matches 'no further', 'nothing ahead'
        'all done',     # matches 'all done', 'all complete'
        'final state',  # matches 'final state', 'terminal state'
        'ready to terminate',  # matches 'ready to terminate'
    ],

    'L3_internal_state': [
        'solved',
        'hidden_preconditions',
        'internal_state',
        'precondition_met',
        'kill_condition',
        'success_metric',
        'system_state',
        'world_state',
        'system_completed',
        'prerequisites_set',
        # Implicit internal state descriptions (natural language)
        'requirements met',
        'nothing left',
        'all complete',
        'final state',
        'no further',
        'all conditions',
        'ready to terminate',
        'nothing remaining',
        'all done',
        'task complete',
        'work complete',
        # Additional natural language variants
        'all five steps are complete',
        'all five steps complete',
        'all steps are complete',
        'every step complete',
        'all requirements met',
        'requirements are met',
        'all prerequisites met',
        'all prerequisites satisfied',
    ],
}


# =============================================================================
# Semantic Property Categories for Binding Validation
# =============================================================================

SEMANTIC_PROPERTY_CATEGORIES = {
    'status': {'result', 'success', 'error', 'failure', 'status', 'state', 'ok', 'fail', 'complete', 'failed', 'passed'},
    'phase': {'phase', 'step', 'stage', 'level', 'round', 'iteration', 'sequence', 'order', 'action', 'invoke', 'call', 'execute', 'run', 'perform'},
    'progress': {'progress', 'percent', 'ratio', 'complete', 'done', 'remaining', 'ahead', 'behind'},
    'count': {'count', 'number', 'total', 'size', 'length', 'quantity', 'num', 'index', 'result', 'value', 'item'},
    'format': {'format', 'type', 'json', 'xml', 'csv', 'encoding', 'mime', 'schema'},
    'time': {'time', 'duration', 'elapsed', 'timeout', 'delay', 'ms', 'second', 'minute'},
    'size': {'size', 'length', 'width', 'height', 'dimension', 'bytes', 'kb', 'mb'},
    'code': {'code', 'return', 'exit', 'status', 'id', 'key', 'result'},
    'next': {'next', 'future', 'upcoming', 'following', 'then', 'invoke', 'call', 'action'},
    'last': {'last', 'previous', 'prior', 'past', 'before', 'earlier'},
    'version': {'version', 'rev', 'revision', 'v', 'build'},
    'memory': {'memory', 'ram', 'usage', 'alloc', 'heap', 'stack'},
    'cpu': {'cpu', 'core', 'thread', 'usage', 'load'},
    'disk': {'disk', 'storage', 'space', 'free', 'used', 'io'},
    'network': {'network', 'http', 'tcp', 'ip', 'connection', 'request'},
    'output': {'output', 'return', 'response', 'payload', 'body', 'result', 'value'},
    'input': {'input', 'argument', 'arg', 'payload', 'body'},
    'id': {'id', 'uuid', 'key', 'token', 'handle'},
    'name': {'name', 'label', 'title', 'id'},
    'type': {'type', 'kind', 'class', 'category'},
    'value': {'value', 'val', 'data', 'content'},
    'param': {'param', 'argument', 'arg', 'input', 'option'},
    'config': {'config', 'setting', 'option', 'preference'},
    'error': {'error', 'exception', 'fault', 'fail', 'crash'},
    'mode': {'mode', 'option', 'setting', 'type'},
    'flag': {'flag', 'option', 'toggle', 'switch', 'bit'},
    'ratio': {'ratio', 'rate', 'percent', 'fraction'},
    'threshold': {'threshold', 'limit', 'bound', 'max', 'min'},
    'priority': {'priority', 'weight', 'order', 'rank'},
    'reason': {'reason', 'cause', 'why', 'explanation'},
    'message': {'message', 'msg', 'text', 'string', 'log'},
}

def _get_semantic_category(word):
    for cat in SEMANTIC_PROPERTY_CATEGORIES:
        if word in SEMANTIC_PROPERTY_CATEGORIES[cat]:
            return cat
    return None

def _compute_semantic_binding_score(word, ev_str):
    cat = _get_semantic_category(word)
    if cat is None:
        return 0.0
    for member in SEMANTIC_PROPERTY_CATEGORIES[cat]:
        if member in ev_str:
            return 0.5
    return 0.0


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_claim(claim: str) -> str:
    return claim.lower().strip() if claim else ""


def compute_content_hash(proposal: dict, evidence_ids: list[str]) -> str:
    parts = [
        str(proposal.get('kind', '')),
        normalize_claim(proposal.get('title', '')),
        normalize_claim(proposal.get('question', '')),
        normalize_claim(proposal.get('rationale', '')),
        "|".join(sorted(evidence_ids)),
    ]
    return hashlib.sha256("||".join(parts).encode('utf-8')).hexdigest()[:16]


def compute_novelty_hash(proposal: dict) -> str:
    parts = [
        str(proposal.get('kind', '')),
        normalize_claim(proposal.get('title', '')),
        normalize_claim(proposal.get('question', '')),
    ]
    return hashlib.sha256("||".join(parts).encode('utf-8')).hexdigest()[:12]


def _strip_punctuation(word: str) -> str:
    for ch in '.,!?;:()[]{}':
        word = word.replace(ch, '')
    return word.replace('"', '').replace("'", "")


def extract_key_entities(claim: str) -> dict:
    entities = {'function_names': [], 'param_values': [], 'error_codes': [], 'constraint_terms': []}
    claim_lower = claim.lower()
    words = claim.split()
    for w in words:
        w_clean = _strip_punctuation(w)
        if '_' in w_clean and len(w_clean) > 3:
            parts = w_clean.split('_')
            if len(parts) >= 2 and all(p.isalpha() or p.isdigit() for p in parts if p):
                entities['function_names'].append(w_clean)
    camel = re.findall(r'\b([a-z]+(?:[A-Z][a-z0-9]*)+)\b', claim)
    for word in camel:
        if len(word) > 3:
            entities['function_names'].append(word.lower())
    err_pattern = re.findall(r'(e\d+|err[_\d]+|error[_\d]+|fault[_\d]+)', claim_lower)
    entities['error_codes'].extend(err_pattern)
    quoted = re.findall(r'["\']([^"\']{2,})["\']', claim)
    entities['param_values'].extend([v.lower() for v in quoted])
    constraint_kws = ['must', 'shall', 'required', 'not null', 'positive', 'negative',
                       'non-zero', 'integer', 'string', 'array', 'object', 'min', 'max', 'range']
    for kw in constraint_kws:
        if kw in claim_lower:
            entities['constraint_terms'].append(kw)
    return entities


# =============================================================================
# Gate 1: Schema Validation
# =============================================================================

def gate1_schema_validation(proposal: dict, candidate_id: str) -> Optional[SurfaceGovernanceDecision]:
    required_fields = ['kind', 'title', 'question']
    for field_name in required_fields:
        if field_name not in proposal or proposal[field_name] is None:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_MISSING_REQUIRED_FIELD,
                reason_detail=f"Missing required field: {field_name}",
            )
        val = proposal[field_name]
        if isinstance(val, str) and not val.strip():
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_EMPTY_REQUIRED_FIELD,
                reason_detail=f"Required field is empty: {field_name}",
            )
    return None


# =============================================================================
# Gate 2: Leak Detection (L1/L2/L3)
# =============================================================================

def _contains_forward_action_language(text: str) -> Optional[str]:
    """
    Detect natural language forward-looking action suggestions.
    These are phrases that recommend or predict actions without using explicit field names.
    """
    # "next is X", "next will be X" — explicit forward prediction
    import re
    next_patterns = [
        r'\bnext is\b',
        r'\bnext will be\b',
        r'\bcall next\b',
        r'\bcalled next\b',
        r'\binvoke next\b',
        r'\bbest move\b',
        r'\boptimal path\b',
        r'\bshould happen\b',
        r'\blogical next\b',
        r'\bafter this\b',
        r'\bwhen should\b',
        r'\bbased on history\b',
        r'\bnext call\b',
        r'\bnext step\b',
        r'\boptimal next\b',
        r'\bwill be called\b',
        # Compound word variants (underscore-separated)
        r'optimal_path',
        r'best_move',
        r'next_step',
        r'call_next',
        r'invoke_next',
    ]
    for pattern in next_patterns:
        if re.search(pattern, text):
            return pattern.strip('\\b')
    # Check for "action word + next" pattern ONLY as standalone phrase
    # Don't catch compound words like "run_time" or "next_phase"
    action_words = ['call', 'invoke']
    import re
    for aw in action_words:
        if re.search(rf'\b{aw} next\b', text):
            return f'{aw} next'
    return None


def gate2_leak_detection(proposal: dict, candidate_id: str,
                         evidence_store: dict[str, dict]) -> Optional[SurfaceGovernanceDecision]:
    title = proposal.get('title', '').lower()
    question = proposal.get('question', '').lower()
    rationale = proposal.get('rationale', '').lower()
    suggested = str(proposal.get('suggested_tests', [])).lower()
    content_str = str(proposal.get('content', {})).lower()
    all_text = title + ' ' + question + ' ' + rationale + ' ' + suggested + ' ' + content_str

    # L0: Semantic forward action language detection (natural language)
    # This catches forward-looking phrases that don't use explicit field names
    sem_hit = _contains_forward_action_language(all_text)
    if sem_hit:
        return SurfaceGovernanceDecision(
            candidate_id=candidate_id, decision="REJECT",
            reason_code=REJECT_FORWARD_LOOKING_ACTION,
            reason_detail=f"Natural language forward action detected: '{sem_hit}'",
        )

    # L1: forward-looking action fields check FIRST
    all_state_patterns = LEAK_PATTERNS['L2_unobserved_value'] + LEAK_PATTERNS['L3_internal_state']
    for pattern in all_state_patterns:
        if pattern.lower() in all_text:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_INTERNAL_STATE_LEAK,
                reason_detail=f"Content contains internal state field: '{pattern}'",
            )

    # L3 partial stem check — catch stem variants like "prerequisites satisfied" → "prerequis" + "satisf"
    # Use word-boundary-aware matching to avoid false positives (e.g., 'hidden' in 'NovelAPIDomain')
    import re
    l3_stems = LEAK_PATTERNS.get('L3_partial_stems', [])
    for stem in l3_stems:
        # Match stem as a whole word (with word boundaries)
        # But allow it as part of underscore-separated compound words
        if re.search(r'\b' + re.escape(stem) + r'\b', all_text.lower()):
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_INTERNAL_STATE_LEAK,
                reason_detail=f"Content contains internal state stem: '{stem}'",
            )

    # L1: forward-looking action fields
    for pattern in LEAK_PATTERNS['L1_forward_action']:
        if pattern.lower() in all_text:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_FORWARD_LOOKING_ACTION,
                reason_detail=f"Content contains forward-looking action field: '{pattern}'",
            )

    # L2: unobserved values — only numeric values in content dict
    content_dict = proposal.get('content', {})
    if isinstance(content_dict, dict):
        for field_name, field_value in content_dict.items():
            if isinstance(field_value, (int, float)) and field_value not in (0, 1, 2, 3, 4, 5):
                num_str = str(field_value)
                found_num = any(
                    num_str in str(ev.get('content', {})) + ' ' + str(ev.get('claim', ''))
                    for ev in evidence_store.values()
                )
                if not found_num:
                    return SurfaceGovernanceDecision(
                        candidate_id=candidate_id, decision="REJECT",
                        reason_code=REJECT_UNOBSERVED_VALUE,
                        reason_detail=f"Content field '{field_name}' has unobserved numeric value: {field_value}",
                    )

    return None


# =============================================================================
# Gate 3: Evidence-Proposal Binding Validation
# =============================================================================

def gate3_binding_validation(
    proposal: dict, candidate_id: str, evidence_store: dict[str, dict],
    allowed_evidence_kinds: dict[str, list[str]],
    min_binding_score: float = 0.3,
) -> Optional[SurfaceGovernanceDecision]:
    evidence_ids = proposal.get('supporting_evidence_ids', [])
    proposal_type = proposal.get('kind', '')

    if not evidence_ids:
        return SurfaceGovernanceDecision(
            candidate_id=candidate_id, decision="REJECT",
            reason_code=REJECT_NO_SUPPORTING_EVIDENCE,
            reason_detail="Proposal has no supporting evidence IDs",
            evidence_ids_used=[],
        )

    for eid in evidence_ids:
        if eid not in evidence_store:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_EVIDENCE_ID_NOT_FOUND,
                reason_detail=f"Evidence ID '{eid}' not found in evidence store",
                evidence_ids_used=evidence_ids,
            )

    allowed_kinds = allowed_evidence_kinds.get(proposal_type, [])
    if allowed_kinds:
        bound_kinds = set()
        for eid in evidence_ids:
            ekind = evidence_store[eid].get('kind', evidence_store[eid].get('type', ''))
            bound_kinds.add(ekind)
        if not bound_kinds.intersection(allowed_kinds):
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_UNSUPPORTED_PROPOSAL_TYPE,
                reason_detail=f"Evidence kinds {bound_kinds} don't match proposal type '{proposal_type}' (need {allowed_kinds})",
                evidence_ids_used=evidence_ids,
            )

    claim = normalize_claim(proposal.get('title', '') + ' ' + proposal.get('question', ''))
    entities = extract_key_entities(claim)
    non_stopword_words = [w for w in claim.split() if len(w) > 3 and w not in GENERIC_STOPWORDS and '_' not in w]

    # Collect all words to check from claim (split underscore compounds too)
    words_to_check = []
    for w in non_stopword_words:
        # Strip punctuation for matching
        w_clean = _strip_punctuation(w)
        words_to_check.append(w_clean)
        # Also add underscore-separated sub-words for partial matching
        if '_' in w_clean:
            for part in w_clean.split('_'):
                if len(part) > 2 and part not in GENERIC_STOPWORDS:
                    words_to_check.append(part)

    # Add ALL content key words to words_to_check (not just underscore-split)
    # Content keys represent what the proposal is asking about — they should
    # always be considered for binding, even if they're generic words
    content_keys = proposal.get('content', {}).keys()
    for ck in content_keys:
        ck_str = str(ck)
        if '_' in ck_str:
            # Underscore-separated compound key: add each part
            for part in ck_str.split('_'):
                if len(part) > 2 and part not in GENERIC_STOPWORDS:
                    words_to_check.append(part)
        else:
            # Simple key: add if meaningful (exclude single chars)
            if len(ck_str) > 2:
                words_to_check.append(ck_str)

    words_to_check = list(set(words_to_check))  # deduplicate

    if words_to_check or entities['function_names'] or entities['error_codes']:
        total_score = 0.0
        all_matched = []
        for eid in evidence_ids:
            ev = evidence_store[eid]
            ev_str = str(ev.get('content', {})).replace("'", "").replace('"', '') + " "
            ev_str += str(ev.get('claim', '')).lower()

            # Function names: +1.0
            for fn in entities.get('function_names', []):
                if fn in ev_str:
                    total_score += 1.0
                    all_matched.append(f'fn:{fn}')

            # Error codes: +1.0
            for ec in entities.get('error_codes', []):
                if ec in ev_str:
                    total_score += 1.0
                    all_matched.append(f'ec:{ec}')

            # Word matches (exact): +0.3
            for word in words_to_check:
                if word in ev_str:
                    total_score += 0.3
                    all_matched.append(f'cw:{word}')
                else:
                    # Fallback: semantic category match: +0.3
                    sem_score = _compute_semantic_binding_score(word, ev_str)
                    if sem_score > 0:
                        total_score += sem_score
                        all_matched.append(f'ss:{word}')
                    # No tier3 fallback - semantic categories must provide the connection

        if total_score < min_binding_score:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_CLAIM_EVIDENCE_MISMATCH,
                reason_detail=f"Binding score {total_score:.2f} < {min_binding_score}. Matched: {all_matched}",
                evidence_ids_used=evidence_ids,
            )

    return None


# =============================================================================
# Gate 4: Content Quality Check
# =============================================================================

def gate4_content_quality(proposal: dict, candidate_id: str) -> Optional[SurfaceGovernanceDecision]:
    title = normalize_claim(proposal.get('title', ''))
    question = normalize_claim(proposal.get('question', ''))
    rationale = normalize_claim(proposal.get('rationale', ''))
    suggested_tests = proposal.get('suggested_tests', [])

    for stopword in STOPWORD_TEMPLATES:
        if title == stopword or question == stopword or rationale == stopword:
            return SurfaceGovernanceDecision(
                candidate_id=candidate_id, decision="REJECT",
                reason_code=REJECT_STOPWORD_PLAINTEXT,
                reason_detail=f"Proposal is stopword template: '{stopword}'",
            )

    if not title and not question:
        return SurfaceGovernanceDecision(
            candidate_id=candidate_id, decision="REJECT",
            reason_code=REJECT_EMPTY_CONTENT,
            reason_detail="Both title and question are empty",
        )

    total_text = (title + " " + question + " " + rationale).strip()
    words = total_text.split()
    unique_words = set(words)
    info_score = len(unique_words) / max(len(words), 1)
    if len(words) < 5 or info_score < 0.5:
        return SurfaceGovernanceDecision(
            candidate_id=candidate_id, decision="REJECT",
            reason_code=REJECT_LOW_INFORMATION_CONTENT,
            reason_detail=f"Low information content: {len(words)} words, {info_score:.2f} unique ratio",
        )

    has_actionable = bool(suggested_tests) or proposal.get('next_action') or proposal.get('test')
    if not has_actionable:
        return SurfaceGovernanceDecision(
            candidate_id=candidate_id, decision="REJECT",
            reason_code=REJECT_NO_ACTIONABLE_NEXT_STEP,
            reason_detail="No suggested_tests, next_action, or test provided",
        )

    return None


# =============================================================================
# Merge Record (for lifecycle tracking)
# =============================================================================

@dataclass
class MergeRecord:
    """Tracks merge/update lifecycle of an object."""
    object_id: str
    content_hash: str
    original_evidence_ids: list[str]
    new_evidence_ids: list[str]
    support_set: set[str]  # all evidence IDs supporting this object
    confidence: float  # 0.0 to 1.0, grows with more supporting evidence
    merge_type: str  # MERGE_UPDATE | WEAKEN | REOPEN | SPLIT | INVALIDATE
    provenance: list[str]  # chain of content_hash values (for dedup)
    timestamp: int = 0


# =============================================================================
# Object Store (with merge/update lifecycle)
# =============================================================================

class ObjectStore:
    """
    Tracks committed objects with full lifecycle (merge/update/replace).

    Key operations:
      - add(proposal, decision, evidence_ids) → creates new object
      - merge_update(proposal, decision, new_evidence_ids) → updates existing
      - weaken(object_id, new_evidence_id) → reduces confidence
      - invalidate(object_id) → marks as invalid
      - get(object_id) → returns object + merge history
      - retrieve() → returns all non-invalidated objects
    """

    def __init__(self):
        self._objects: dict[str, dict] = {}  # object_id → object record
        self._by_content_hash: dict[str, list[str]] = {}  # content_hash → [object_ids]
        self._by_key: dict[str, str] = {}  # key → object_id (for keyed lookups)
        self._by_novelty_hash: dict[str, list[str]] = {}  # novelty_hash → [object_ids]
        self._merge_history: dict[str, list[MergeRecord]] = {}  # object_id → merge records
        self._counter = 0

    def add(self, proposal: dict, decision: SurfaceGovernanceDecision,
            evidence_ids: list[str]) -> str:
        """Add a new object."""
        self._counter += 1
        obj_id = f"obj_{self._counter}"

        self._objects[obj_id] = {
            'object_id': obj_id,
            'kind': proposal.get('kind', 'unknown'),
            'title': proposal.get('title', ''),
            'question': proposal.get('question', ''),
            'content': proposal.get('content', {}),
            'content_hash': decision.content_hash,
            'support_set': set(evidence_ids),
            'evidence_ids': list(evidence_ids),
            'confidence': min(len(evidence_ids) * 0.15, 0.9),
            'status': 'active',
            'created_at': self._counter,
            'merge_count': 0,
        }

        # Register by novelty_hash (primary for merge detection)
        _novelty_hash = compute_novelty_hash(proposal)
        if _novelty_hash not in self._by_novelty_hash:
            self._by_novelty_hash[_novelty_hash] = []
        self._by_novelty_hash[_novelty_hash].append(obj_id)

        # Register by content_hash (secondary)
        content_hash_val = decision.content_hash
        if content_hash_val not in self._by_content_hash:
            self._by_content_hash[content_hash_val] = []
        self._by_content_hash[content_hash_val].append(obj_id)

        self._merge_history[obj_id] = []
        return obj_id

    def find_existing(self, content_hash: str, novelty_hash: str = "") -> Optional[str]:
        """
        Find existing object for merge/update.

        Uses NOVELTY_HASH (title + question + kind) as primary key for finding
        the same conceptual object across different evidence expansions.

        _by_novelty_hash stores novelty_hash → [object_ids] for efficient lookup.
        """
        # Primary: match by novelty_hash (title + question + kind — same conceptual object)
        by_novelty = getattr(self, '_by_novelty_hash', {})
        if novelty_hash and novelty_hash in by_novelty:
            for obj_id in by_novelty[novelty_hash]:
                if obj_id in self._objects and self._objects[obj_id]['status'] == 'active':
                    return obj_id

        # Fallback: check by content_hash (same exact evidence set)
        if content_hash and content_hash in self._by_content_hash:
            for obj_id in self._by_content_hash[content_hash]:
                if obj_id in self._objects and self._objects[obj_id]['status'] == 'active':
                    return obj_id

        return None

    def merge_update(self, object_id: str, new_evidence_ids: list[str],
                     additional_content: dict = None) -> MergeRecord:
        """Merge new evidence into existing object."""
        obj = self._objects[object_id]
        old_support = set(obj['support_set'])
        old_confidence = obj['confidence']

        # Expand support set
        new_support = old_support | set(new_evidence_ids)
        obj['support_set'] = new_support
        obj['evidence_ids'] = list(new_support)

        # Increase confidence (diminishing returns)
        n = len(new_support)
        new_confidence = min(0.5 + (n * 0.08), 0.95)
        obj['confidence'] = new_confidence

        # Track merge
        record = MergeRecord(
            object_id=object_id,
            content_hash=obj['content_hash'],
            original_evidence_ids=list(old_support),
            new_evidence_ids=new_evidence_ids,
            support_set=new_support,
            confidence=new_confidence,
            merge_type=MERGE_UPDATE_EXISTING,
            provenance=[obj['content_hash']],
        )
        self._merge_history[object_id].append(record)
        obj['merge_count'] += 1

        # Update content if additional provided
        if additional_content:
            obj['content'] = {**obj['content'], **additional_content}

        return record

    def weaken(self, object_id: str, contradicting_evidence_id: str) -> MergeRecord:
        """Weaken an object due to contradicting evidence."""
        obj = self._objects[object_id]
        old_confidence = obj['confidence']

        new_confidence = max(old_confidence * 0.5, 0.1)
        obj['confidence'] = new_confidence

        record = MergeRecord(
            object_id=object_id,
            content_hash=obj['content_hash'],
            original_evidence_ids=list(obj['support_set']),
            new_evidence_ids=[contradicting_evidence_id],
            support_set=obj['support_set'],
            confidence=new_confidence,
            merge_type=WEAKEN_EXISTING,
            provenance=[obj['content_hash']],
        )
        self._merge_history[object_id].append(record)
        return record

    def invalidate(self, object_id: str) -> None:
        """Mark object as invalidated."""
        if object_id in self._objects:
            self._objects[object_id]['status'] = 'invalidated'

    def get(self, object_id: str) -> Optional[dict]:
        return self._objects.get(object_id)

    def retrieve(self, sort_by: str = 'confidence') -> list[dict]:
        """Return all active objects, optionally sorted by confidence."""
        active = [obj for obj in self._objects.values() if obj['status'] == 'active']
        if sort_by == 'confidence':
            active.sort(key=lambda x: -x['confidence'])
        elif sort_by == 'merge_count':
            active.sort(key=lambda x: -x['merge_count'])
        return active

    def get_merge_history(self, object_id: str) -> list[MergeRecord]:
        return self._merge_history.get(object_id, [])

    def summary(self) -> dict:
        by_status = {}
        for obj in self._objects.values():
            s = obj['status']
            by_status[s] = by_status.get(s, 0) + 1
        return {
            'total_objects': len(self._objects),
            'active': by_status.get('active', 0),
            'invalidated': by_status.get('invalidated', 0),
            'by_content_hash': {k: len(v) for k, v in self._by_content_hash.items()},
        }


# =============================================================================
# Proposal Validator (Gates 1-4 + Merge Lifecycle)
# =============================================================================

ALLOWED_EVIDENCE_KINDS = [
    'successful_tool_invocation',
    'failed_tool_invocation',
    'error_missing_required_argument',
    'error_type_mismatch',
    'error_hidden_precondition',
    'error_permission_or_scope_constraint',
    'state_change_after_call',
]


class ProposalValidator:
    """
    4-gate validator + merge lifecycle manager.

    Gate order: Schema → Leak → Binding → Quality
    Then: Merge check (Gate 6) — determines ACCEPT/MERGE/REJECT

    For MERGE:
      1. Find existing object by content_hash
      2. If found and new evidence is stronger → MERGE_UPDATE_EXISTING
      3. If evidence contradicts → WEAKEN/INVALIDATE
      4. If no existing → ACCEPT_NEW
    """

    def __init__(self, current_episode: str = "ep_0", object_store: ObjectStore = None):
        self.current_episode = current_episode
        self.evidence_store: dict[str, dict] = {}
        self.existing_index: dict[str, dict] = {}
        self.episode_history: dict[str, list[str]] = {}
        self.decisions: list[SurfaceGovernanceDecision] = []
        self.object_store = object_store if object_store is not None else ObjectStore()
        self.allowed_evidence_kinds = {
            'research': ALLOWED_EVIDENCE_KINDS,
            'abstraction': ['visible_function_signature', 'successful_tool_invocation',
                           'error_hidden_precondition', 'state_change_after_call'],
            'hypothesis': ['visible_function_signature', 'error_permission_or_scope_constraint',
                           'ordering_dependency_hint'],
        }

    def ingest_evidence(self, evidence_packets: list) -> list[str]:
        """Ingest evidence packets and return the assigned evidence IDs."""
        assigned_ids = []
        for pkt in evidence_packets:
            eid = f"ev_{len(self.evidence_store):04d}"
            self.evidence_store[eid] = {
                'kind': pkt.type if hasattr(pkt, 'type') else pkt.get('type', 'unknown'),
                'claim': pkt.claim if hasattr(pkt, 'claim') else pkt.get('claim', ''),
                'content': pkt.content if hasattr(pkt, 'content') else pkt.get('content', {}),
                'source_refs': pkt.source_refs if hasattr(pkt, 'source_refs') else pkt.get('source_refs', []),
                'confidence': pkt.confidence if hasattr(pkt, 'confidence') else pkt.get('confidence', 0.5),
            }
            assigned_ids.append(eid)
        return assigned_ids

    def validate(self, proposal: dict, candidate_id: str,
                 new_evidence_ids: list[str] = None,
                 use_existing_evidence: bool = True) -> SurfaceGovernanceDecision:
        """Full validation: gates 1-4 + merge lifecycle."""

        # Gate 1: Schema
        g1 = gate1_schema_validation(proposal, candidate_id)
        if g1:
            return self._record(g1)

        # Gate 2: Leak (L1/L2/L3)
        g2 = gate2_leak_detection(proposal, candidate_id, self.evidence_store)
        if g2:
            return self._record(g2)

        # Gate 3: Binding
        g3 = gate3_binding_validation(
            proposal, candidate_id, self.evidence_store, self.allowed_evidence_kinds
        )
        if g3:
            return self._record(g3)

        # Gate 4: Content quality
        g4 = gate4_content_quality(proposal, candidate_id)
        if g4:
            return self._record(g4)

        # All 4 gates passed — now determine lifecycle decision (Gate 5: Merge)
        evidence_ids = new_evidence_ids or proposal.get('supporting_evidence_ids', [])
        content_hash = compute_content_hash(proposal, evidence_ids)
        novelty_hash = compute_novelty_hash(proposal)

        # Check for existing object with same content_hash
        existing_id = self.object_store.find_existing(content_hash, novelty_hash)

        if existing_id:
            # Gate 5: Merge decision
            existing_obj = self.object_store.get(existing_id)

            # Check if new evidence contradicts old
            # Contradiction = new evidence has conflicting claim
            contradicts = self._check_contradiction(existing_obj, evidence_ids)

            if contradicts:
                # Handle contradiction
                action = self._handle_contradiction(existing_id, evidence_ids)
                return self._record(action)
            else:
                # Non-contradicting merge: expand support + confidence
                record = self.object_store.merge_update(
                    existing_id, evidence_ids, proposal.get('content', {})
                )
                # Emit MERGE_UPDATE_EXISTING decision
                decision = SurfaceGovernanceDecision(
                    candidate_id=candidate_id,
                    decision="MERGE",
                    reason_code=MERGE_UPDATE_EXISTING,
                    reason_detail=(f"Updated object {existing_id} with {len(evidence_ids)} new evidence. "
                                   f"Support set expanded to {len(record.support_set)} items. "
                                   f"Confidence: {existing_obj['confidence']:.2f} → {record.confidence:.2f}"),
                    target_object_id=existing_id,
                    evidence_ids_used=evidence_ids,
                    content_hash=content_hash,
                )
                self.decisions.append(decision)
                self._update_episode_history(content_hash)
                return decision
        else:
            # No existing object → ACCEPT_NEW
            obj_id = self.object_store.add(proposal, SurfaceGovernanceDecision(
                candidate_id=candidate_id,
                decision="ACCEPT",
                reason_code=ACCEPT_NEW,
                reason_detail="New object created",
                content_hash=content_hash,
            ), evidence_ids)

            decision = SurfaceGovernanceDecision(
                candidate_id=candidate_id,
                decision="ACCEPT",
                reason_code=ACCEPT_NEW,
                reason_detail=f"New object {obj_id} created with {len(evidence_ids)} supporting evidence",
                target_object_id=obj_id,
                evidence_ids_used=evidence_ids,
                content_hash=content_hash,
            )
            self.decisions.append(decision)
            self._update_episode_history(content_hash)
            return decision

    def _check_contradiction(self, existing_obj: dict, new_evidence_ids: list[str]) -> bool:
        """Check if new evidence contradicts the existing object."""
        existing_claims = set()
        for eid in existing_obj.get('evidence_ids', []):
            if eid in self.evidence_store:
                existing_claims.add(self.evidence_store[eid].get('claim', '').lower())

        for eid in new_evidence_ids:
            if eid in self.evidence_store:
                new_claim = self.evidence_store[eid].get('claim', '').lower()
                # Simple contradiction check: new claim says the opposite of existing
                # Example: existing says "function was invoked", new says "function not found"
                for existing in existing_claims:
                    # Check for negation patterns
                    if self._is_contradiction(existing, new_claim):
                        return True
        return False

    def _is_contradiction(self, claim_a: str, claim_b: str) -> bool:
        """Check if claim_b contradicts claim_a."""
        # Check for negation words
        neg_words = ['not', 'no', 'never', 'failed', 'error', 'wrong', 'missing', 'absent']
        has_neg_a = any(nw in claim_a.split() for nw in neg_words)
        has_neg_b = any(nw in claim_b.split() for nw in neg_words)

        # Direct contradiction: one has negation, other doesn't, but same subject
        # Very simple heuristic — in real system would use more sophisticated logic
        words_a = set(claim_a.split())
        words_b = set(claim_b.split())
        common = words_a & words_b

        if len(common) >= 2 and has_neg_a != has_neg_b:
            return True

        return False

    def _handle_contradiction(self, existing_id: str,
                               new_evidence_ids: list[str]) -> SurfaceGovernanceDecision:
        """Handle contradiction: WEAKEN, REOPEN, SPLIT, or INVALIDATE."""
        existing_obj = self.object_store.get(existing_id)
        old_conf = existing_obj['confidence']
        new_evidence_conf = sum(
            self.evidence_store.get(eid, {}).get('confidence', 0.5)
            for eid in new_evidence_ids
        ) / max(len(new_evidence_ids), 1)

        # If new evidence is significantly stronger, invalidate old
        if new_evidence_conf > old_conf + 0.3:
            self.object_store.invalidate(existing_id)
            return SurfaceGovernanceDecision(
                candidate_id=f"contradiction_{existing_id}",
                decision="REJECT",
                reason_code=INVALIDATE_OBJECT,
                reason_detail=(f"Object {existing_id} invalidated due to stronger contradicting evidence. "
                               f"Old confidence: {old_conf:.2f}, new evidence strength: {new_evidence_conf:.2f}"),
                target_object_id=existing_id,
                evidence_ids_used=new_evidence_ids,
            )
        else:
            # New evidence is weaker — weaken existing object
            record = self.object_store.weaken(existing_id, new_evidence_ids[0] if new_evidence_ids else "")
            return SurfaceGovernanceDecision(
                candidate_id=f"contradiction_{existing_id}",
                decision="REJECT",
                reason_code=WEAKEN_EXISTING,
                reason_detail=(f"Object {existing_id} weakened. "
                               f"Confidence: {old_conf:.2f} → {record.confidence:.2f}. "
                               f"New evidence contradicts but is weaker."),
                target_object_id=existing_id,
                evidence_ids_used=new_evidence_ids,
            )

    def _record(self, decision: SurfaceGovernanceDecision) -> SurfaceGovernanceDecision:
        self.decisions.append(decision)
        return decision

    def _update_episode_history(self, content_hash: str):
        ep = self.current_episode
        if ep not in self.episode_history:
            self.episode_history[ep] = []
        if content_hash not in self.episode_history[ep]:
            self.episode_history[ep].append(content_hash)

    def summary(self) -> dict:
        by_code = {}
        for d in self.decisions:
            by_code[d.reason_code] = by_code.get(d.reason_code, 0) + 1
        return {
            'total': len(self.decisions),
            'by_reason_code': by_code,
            'object_store_summary': self.object_store.summary(),
            'decisions': self.decisions,
        }


# =============================================================================
# EvidencePacket (for use in validator)
# =============================================================================

@dataclass
class EvidencePacket:
    type: str
    kind: str
    claim: str
    content: dict
    source_refs: list
    confidence: float
