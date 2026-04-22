"""
modules/evidence/extractor.py

NovelAPIRawEvidenceExtractor: extracts raw evidence from NovelAPI interactions.

Evidence kinds:
  - visible_function_signature: API reveals function name(s)
  - error_missing_required_argument: tool call failed due to missing arg
  - error_type_mismatch: tool call failed due to type error
  - error_hidden_precondition: tool call failed due to hidden precondition
  - error_permission_or_scope_constraint: tool call blocked by permission/scope
  - successful_tool_invocation: tool call succeeded
  - failed_tool_invocation: tool call failed (general)
  - ordering_dependency_hint: hint suggests ordering between operations
  - state_change_after_call: environment state changed after tool call

Adapted for AGI_WORLD_V2: uses local type definitions instead of core.surfaces.base.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Local type definitions (originally from core/surfaces/base.py)
@dataclass
class EvidencePacket:
    """Single piece of evidence extracted from an action trace."""
    evidence_id: str
    evidence_kind: str
    content: Dict[str, Any]
    confidence: float
    content_hash: Optional[str] = None  # Task 0.2: canonical shape field
    source_action: Optional[Dict] = None
    surface_result: Optional[Dict] = None
    tick: int = 0
    episode: int = 1


class RawEvidenceExtractor:
    """Base class for raw evidence extractors (interface only)."""
    def extract(self, action_trace: Dict, surface_result: Dict) -> List[EvidencePacket]:
        raise NotImplementedError


NOVEL_API_EVIDENCE_KINDS = [
    "visible_function_signature",
    "visible_parameter_constraint",
    "error_missing_required_argument",
    "error_type_mismatch",
    "error_hidden_precondition",
    "error_permission_or_scope_constraint",
    "successful_tool_invocation",
    "failed_tool_invocation",
    "ordering_dependency_hint",
    "state_change_after_call",
]


class NovelAPIRawEvidenceExtractor(RawEvidenceExtractor):
    """
    Extracts raw evidence from NovelAPI surface interactions.

    Maps NovelAPI action results to structured EvidencePackets.
    """

    def extract(self, action_trace: Dict, surface_result: Dict) -> List[EvidencePacket]:
        """
        Extract evidence packets from an action trace and surface result.

        Args:
            action_trace: dict with keys 'action' and 'result'
            surface_result: result from world.act()

        Returns:
            List of EvidencePackets
        """
        packets = []
        action = action_trace.get('action', {})
        result = surface_result

        # Extract from novel_api result
        na_result = result.get('novel_api', {})
        if hasattr(na_result, '_data'):
            na_result = na_result._data

        # 1. Successful function invocation
        if na_result.get('correct_function'):
            fn = na_result.get('correct_function')
            pkt = self._make_packet(
                evidence_kind='successful_tool_invocation',
                content={
                    'function_name': fn,
                    'is_discovery': na_result.get('discovery_event', False),
                    'unlocks_progress': na_result.get('unlocks_progress', False),
                    'kwargs': na_result.get('kwargs', {}),
                },
                confidence=0.9 if na_result.get('discovery_event') else 0.7,
                action=action,
                result=result,
            )
            packets.append(pkt)

        # 2. Failed invocation
        if na_result.get('error'):
            err = na_result.get('error', '')
            err_type = self._classify_error(err)
            pkt = self._make_packet(
                evidence_kind=err_type,
                content={
                    'error_message': err,
                    'function_name': action.get('payload', {}).get('tool_args', {}).get('function_name', 'unknown'),
                },
                confidence=0.8,
                action=action,
                result=result,
            )
            packets.append(pkt)

        # 3. Visible function signature (from inspect)
        visible = na_result.get('visible_functions', [])
        for fn in visible:
            pkt = self._make_packet(
                evidence_kind='visible_function_signature',
                content={'function_name': fn},
                confidence=0.95,
                action=action,
                result=result,
            )
            packets.append(pkt)

        # 4. Ordering dependency hint
        if na_result.get('prerequisite_triggered'):
            pkt = self._make_packet(
                evidence_kind='ordering_dependency_hint',
                content={
                    'prerequisite': na_result.get('prerequisite_triggered'),
                    'dependent': na_result.get('dependent_function', ''),
                },
                confidence=0.85,
                action=action,
                result=result,
            )
            packets.append(pkt)

        # 5. State change after call
        if na_result.get('state_change'):
            pkt = self._make_packet(
                evidence_kind='state_change_after_call',
                content=na_result.get('state_change', {}),
                confidence=0.75,
                action=action,
                result=result,
            )
            packets.append(pkt)

        return packets

    def _make_packet(
        self,
        evidence_kind: str,
        content: Dict[str, Any],
        confidence: float,
        action: Optional[Dict] = None,
        result: Optional[Dict] = None,
    ) -> EvidencePacket:
        """Create an EvidencePacket with auto-generated ID and content_hash."""
        import uuid
        import time
        import hashlib
        import json
        # Task 0.2: compute content_hash for deduplication
        content_str = json.dumps(content, sort_keys=True, default=str)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        return EvidencePacket(
            evidence_id=f"ev_{int(time.time()*1000)%100000}_{uuid.uuid4().hex[:6]}",
            evidence_kind=evidence_kind,
            content=content,
            confidence=confidence,
            content_hash=content_hash,
            source_action=action,
            surface_result=result,
        )

    def _classify_error(self, error_msg: str) -> str:
        """Classify error type from error message."""
        err_lower = error_msg.lower()
        if 'missing' in err_lower and 'argument' in err_lower:
            return 'error_missing_required_argument'
        if 'type' in err_lower or 'mismatch' in err_lower:
            return 'error_type_mismatch'
        if 'precondition' in err_lower or 'hidden' in err_lower:
            return 'error_hidden_precondition'
        if 'permission' in err_lower or 'scope' in err_lower or 'constraint' in err_lower:
            return 'error_permission_or_scope_constraint'
        return 'failed_tool_invocation'