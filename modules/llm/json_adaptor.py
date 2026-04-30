"""
modules/llm/json_adaptor.py

JSONAdaptor: Gate + Adaptor for LLM natural language → structured JSON.

Architecture:
1. Gate: Check if LLM output is already valid JSON → skip adaptation
2. Adaptor: Parse natural language to extract structured data
3. Validator: Ensure output conforms to expected schema

Usage:
    adaptor = JSONAdaptor(schema={'fn': float, 'score': float})
    result = adaptor.run(llm_output)
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List, Mapping, Sequence, Literal


LLM_OUTPUT_ADAPTER_VERSION = "conos.llm.output_adapter/v1"


def _strip_markdown_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _fix_single_quoted_json_text(text: str) -> str:
    result: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "'":
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < len(text) else ""
            if prev_char.isalpha() and next_char.isalpha():
                result.append(c)
            else:
                result.append('"')
        else:
            result.append(c)
        i += 1
    return "".join(result)


@dataclass
class LLMOutputContract:
    output_kind: str
    expected_type: Literal["dict", "list", "any"] = "dict"
    expected_prefixes: tuple[str, ...] = ()
    repair_strategy: str = "json_extract_single_quote_repair"
    required_fields: tuple[str, ...] = ()
    schema_version: str = LLM_OUTPUT_ADAPTER_VERSION

    @property
    def contract_id(self) -> str:
        return f"{self.schema_version}:{self.output_kind}"

    def to_trace(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "output_kind": self.output_kind,
            "expected_type": self.expected_type,
            "expected_prefixes": list(self.expected_prefixes),
            "repair_strategy": self.repair_strategy,
            "required_fields": list(self.required_fields),
        }


_OUTPUT_CONTRACTS: dict[str, LLMOutputContract] = {
    "action_kwargs": LLMOutputContract(
        output_kind="action_kwargs",
        expected_type="dict",
        expected_prefixes=("KWARGS_JSON:",),
    ),
    "reasoning_state": LLMOutputContract(
        output_kind="reasoning_state",
        expected_type="dict",
        expected_prefixes=("REASONING_STATE_JSON:",),
    ),
    "patch_proposal": LLMOutputContract(
        output_kind="patch_proposal",
        expected_type="dict",
        expected_prefixes=("PATCH_JSON:", "PROPOSAL_JSON:"),
    ),
    "hypothesis_generation": LLMOutputContract(output_kind="hypothesis_generation", expected_type="list"),
    "probe_design": LLMOutputContract(output_kind="probe_design", expected_type="list"),
    "probe_urgency": LLMOutputContract(output_kind="probe_urgency", expected_type="dict"),
    "representation_card": LLMOutputContract(output_kind="representation_card", expected_type="list"),
    "representation_card_proposal": LLMOutputContract(output_kind="representation_card_proposal", expected_type="list"),
    "recovery_diagnosis": LLMOutputContract(output_kind="recovery_diagnosis", expected_type="dict"),
    "recovery_error_diagnosis": LLMOutputContract(output_kind="recovery_error_diagnosis", expected_type="dict"),
    "recovery_plan": LLMOutputContract(output_kind="recovery_plan", expected_type="list"),
    "recovery_plan_synthesis": LLMOutputContract(output_kind="recovery_plan_synthesis", expected_type="list"),
    "recovery_gate_advice": LLMOutputContract(output_kind="recovery_gate_advice", expected_type="dict"),
    "skill_candidates": LLMOutputContract(output_kind="skill_candidates", expected_type="list"),
    "skill_candidate_generation": LLMOutputContract(output_kind="skill_candidate_generation", expected_type="list"),
    "skill_parameters": LLMOutputContract(output_kind="skill_parameters", expected_type="dict"),
    "skill_parameter_drafting": LLMOutputContract(output_kind="skill_parameter_drafting", expected_type="dict"),
    "status_escalation_decision": LLMOutputContract(
        output_kind="status_escalation_decision",
        expected_type="dict",
        required_fields=("should_escalate", "confidence", "reason"),
    ),
    "model_profile_json": LLMOutputContract(output_kind="model_profile_json", expected_type="dict"),
    "creative_task_candidates": LLMOutputContract(output_kind="creative_task_candidates", expected_type="list"),
    "gateway_json": LLMOutputContract(output_kind="gateway_json", expected_type="dict"),
    "ollama_complete_json": LLMOutputContract(output_kind="ollama_complete_json", expected_type="dict"),
    "minimax_complete_json": LLMOutputContract(output_kind="minimax_complete_json", expected_type="dict"),
}


def register_llm_output_contract(contract: LLMOutputContract) -> None:
    _OUTPUT_CONTRACTS[str(contract.output_kind or "generic")] = contract


def llm_output_contract_for(output_kind: str) -> LLMOutputContract:
    kind = str(output_kind or "generic")
    return _OUTPUT_CONTRACTS.get(kind) or LLMOutputContract(output_kind=kind)


def list_llm_output_contracts() -> list[Dict[str, Any]]:
    return [contract.to_trace() for _, contract in sorted(_OUTPUT_CONTRACTS.items())]


def summarize_llm_output_adapter_traces(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = [dict(row or {}) for row in list(traces or []) if isinstance(row, dict)]
    by_kind: dict[str, dict[str, Any]] = {}
    errors: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("output_kind") or "unknown")
        bucket = by_kind.setdefault(
            kind,
            {
                "total": 0,
                "ok": 0,
                "rejected": 0,
                "repair_applied": 0,
                "normalization_applied": 0,
            },
        )
        bucket["total"] += 1
        if bool(row.get("ok")):
            bucket["ok"] += 1
        else:
            bucket["rejected"] += 1
            error = str(row.get("error") or "unknown_error")
            errors[error] = errors.get(error, 0) + 1
        if bool(row.get("repair_applied")):
            bucket["repair_applied"] += 1
        if bool(row.get("normalization_applied")):
            bucket["normalization_applied"] += 1
    total = len(rows)
    ok = sum(int(bucket.get("ok", 0) or 0) for bucket in by_kind.values())
    repair = sum(int(bucket.get("repair_applied", 0) or 0) for bucket in by_kind.values())
    normalized = sum(int(bucket.get("normalization_applied", 0) or 0) for bucket in by_kind.values())
    return {
        "schema_version": LLM_OUTPUT_ADAPTER_VERSION,
        "total": total,
        "ok_count": ok,
        "rejected_count": max(0, total - ok),
        "repair_applied_count": repair,
        "normalization_applied_count": normalized,
        "repair_rate": round(repair / total, 6) if total else 0.0,
        "normalization_rate": round(normalized / total, 6) if total else 0.0,
        "by_output_kind": by_kind,
        "errors": errors,
    }


@dataclass
class LLMOutputAdapterResult:
    output_kind: str
    expected_type: str
    contract_id: str = ""
    repair_strategy: str = ""
    parsed: Any = None
    ok: bool = False
    status: str = "rejected"
    source: str = ""
    error: str = ""
    prefix: str = ""
    raw_excerpt: str = ""
    schema_version: str = LLM_OUTPUT_ADAPTER_VERSION
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def parsed_dict(self) -> Dict[str, Any]:
        return dict(self.parsed) if isinstance(self.parsed, dict) else {}

    def parsed_list(self) -> list[Any]:
        return list(self.parsed) if isinstance(self.parsed, list) else []

    def to_trace(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": str(self.contract_id),
            "output_kind": self.output_kind,
            "expected_type": self.expected_type,
            "ok": bool(self.ok),
            "status": str(self.status),
            "repair_strategy": str(self.repair_strategy),
            "repair_applied": str(self.status).startswith("repaired"),
            "normalization_applied": bool(self.source and (self.source != "full_text" or str(self.status).startswith("repaired"))),
            "source": str(self.source),
            "error": str(self.error),
            "prefix": str(self.prefix),
            "raw_excerpt": str(self.raw_excerpt),
            "attempts": [dict(row) for row in self.attempts[-8:]],
        }


class LLMOutputAdapter:
    """Normalize raw model text into bounded JSON objects/lists before system use."""

    def __init__(
        self,
        *,
        output_kind: str,
        expected_prefixes: Sequence[str] | None = None,
        expected_type: Literal["dict", "list", "any"] | None = None,
    ) -> None:
        self.contract = llm_output_contract_for(output_kind)
        self.output_kind = str(output_kind or self.contract.output_kind or "generic")
        prefixes = self.contract.expected_prefixes if expected_prefixes is None else tuple(expected_prefixes or ())
        self.expected_prefixes = [str(prefix) for prefix in list(prefixes or ()) if str(prefix)]
        self.expected_type = expected_type or self.contract.expected_type

    def normalize(self, raw_output: Any) -> LLMOutputAdapterResult:
        raw = str(raw_output or "")
        result = LLMOutputAdapterResult(
            output_kind=self.output_kind,
            expected_type=self.expected_type,
            contract_id=self.contract.contract_id,
            repair_strategy=self.contract.repair_strategy,
            raw_excerpt=raw[:500],
        )
        candidates = self._candidate_texts(raw)
        if not candidates:
            result.error = "empty_output"
            return result
        last_error = "parse_failed"
        for source, prefix, candidate in candidates:
            parsed, status, error = self._parse_candidate(candidate)
            result.attempts.append(
                {
                    "source": source,
                    "prefix": prefix,
                    "status": status,
                    "error": error,
                    "candidate_excerpt": str(candidate or "")[:240],
                }
            )
            if error:
                last_error = error
                continue
            if not self._matches_expected_type(parsed):
                last_error = f"expected_{self.expected_type}"
                result.attempts[-1]["error"] = last_error
                continue
            result.parsed = parsed
            result.ok = True
            result.status = status
            result.source = source
            result.prefix = prefix
            result.error = ""
            return result
        result.error = last_error
        return result

    def _candidate_texts(self, raw: str) -> list[tuple[str, str, str]]:
        stripped = _strip_markdown_fence(raw)
        candidates: list[tuple[str, str, str]] = []
        for prefix in self.expected_prefixes:
            index = stripped.find(prefix)
            if index >= 0:
                candidates.append(("prefixed_payload", prefix, stripped[index + len(prefix):].strip()))
        candidates.append(("full_text", "", stripped))
        return candidates

    def _parse_candidate(self, candidate: str) -> tuple[Any, str, str]:
        text = _strip_markdown_fence(str(candidate or "").strip())
        if not text:
            return None, "rejected", "empty_candidate"
        parsed, error = self._raw_decode_from_text(text)
        if not error:
            return parsed, "parsed", ""
        fixed = _fix_single_quoted_json_text(text)
        if fixed != text:
            parsed, fixed_error = self._raw_decode_from_text(fixed)
            if not fixed_error:
                return parsed, "repaired_single_quotes", ""
            error = fixed_error
        return None, "rejected", error

    def _raw_decode_from_text(self, text: str) -> tuple[Any, str]:
        starts: list[int] = []
        for index, char in enumerate(text):
            if char in "{[":
                starts.append(index)
        if text and text[0] not in "{[":
            starts = starts[:8]
        elif starts and starts[0] != 0:
            starts.insert(0, 0)
        decoder = json.JSONDecoder()
        last_error = "json_not_found"
        for start in starts:
            try:
                parsed, _end = decoder.raw_decode(text[start:])
                return parsed, ""
            except Exception as exc:
                last_error = type(exc).__name__
        return None, last_error

    def _matches_expected_type(self, parsed: Any) -> bool:
        if self.expected_type == "any":
            return parsed is not None
        if self.expected_type == "dict":
            return isinstance(parsed, dict)
        if self.expected_type == "list":
            return isinstance(parsed, list)
        return False


def normalize_llm_output(
    raw_output: Any,
    *,
    output_kind: str,
    expected_prefixes: Sequence[str] | None = None,
    expected_type: Literal["dict", "list", "any"] | None = None,
) -> LLMOutputAdapterResult:
    return LLMOutputAdapter(
        output_kind=output_kind,
        expected_prefixes=expected_prefixes,
        expected_type=expected_type,
    ).normalize(raw_output)


class JSONAdaptor:
    """
    Adaptive JSON parser that:
    1. Gate: passes through if already valid JSON
    2. Adaptor: converts natural language to structured JSON
    3. Validator: ensures output matches expected schema
    """

    def __init__(self, schema: Optional[Dict[str, type]] = None, strict: bool = False):
        """
        Args:
            schema: Optional dict of {key: expected_type} for validation
            strict: If True, raise on validation failure. If False, attempt repair.
        """
        self.schema = schema or {}
        self.strict = strict

    def run(self, raw_output: str) -> Dict[str, Any]:
        """
        Main entry point: parse raw LLM output into structured JSON.
        
        1. Gate: Try parsing as-is
        2. If fails: Use Adaptor to extract from natural language
        3. Validate against schema
        4. Return validated dict (or {} if validation fails and not strict)
        """
        # Step 1: Gate - try direct JSON parsing
        if self._is_valid_json(raw_output):
            result = self._parse_json(raw_output)
            if self._validate(result):
                return result
            # Fall through to adaptor if validation fails

        # Step 2: Adaptor - parse natural language
        result = self._adapt_from_text(raw_output)
        
        # Step 3: Validate
        if self._validate(result):
            return result
        
        # Step 4: Recovery (if strict=False)
        if not self.strict:
            # Try to return partial result
            return result
        
        raise ValueError(f'Failed to parse LLM output into valid JSON: {raw_output[:200]}')

    def _is_valid_json(self, text: str) -> bool:
        """Gate: Check if text is already valid JSON (or single-quoted)."""
        text = text.strip()
        
        if not text:
            return False
        
        if not (text.startswith('{') or text.startswith('[')):
            return False
        
        # Try standard JSON
        try:
            parsed = json.loads(text)
            return isinstance(parsed, (dict, list))
        except json.JSONDecodeError:
            # Try single-quoted JSON (MiniMax common output)
            fixed = self._fix_single_quoted_json(text)
            try:
                parsed = json.loads(fixed)
                return isinstance(parsed, (dict, list))
            except json.JSONDecodeError:
                return False

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON string, handling markdown code blocks and single quotes."""
        text = text.strip()
        
        # Strip markdown code blocks
        if text.startswith('```'):
            lines = text.split('\n')
            first_triple = text.find('```')
            last_triple = text.rfind('```')
            if last_triple > first_triple + 3:
                text = text[first_triple+3:last_triple].strip()
        
        # Try to find JSON object in text
        start = text.find('{')
        if start >= 0:
            depth = 0
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break
        
        # Handle single-quoted JSON (MiniMax common output)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            fixed = self._fix_single_quoted_json(text)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                raise

    def _adapt_from_text(self, text: str) -> Dict[str, Any]:
        """
        Adaptor: Extract structured key:value pairs from natural language.
        """
        text = text.strip()
        result = {}
        
        def add_pair(key, value_str):
            key = key.strip()
            if key and key not in result:
                try:
                    result[key] = float(value_str) if '.' in value_str else int(float(value_str))
                except ValueError:
                    pass
        
        # Pattern 1: "key: score" / "key = score" / "key score"
        for key, score_str in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=\s]\s*([0-9.]+)', text, re.IGNORECASE):
            add_pair(key, score_str)
        
        # Pattern 2: "key(score)"
        for key, score_str in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([0-9.]+)\s*\)', text, re.IGNORECASE):
            add_pair(key, score_str)
        
        # Pattern 3: "key gets/has/with/score N"
        for key, score_str in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:gets|has|with|score|rating|rank)\s+([0-9.]+)', text, re.IGNORECASE):
            add_pair(key, score_str)
        
        # Pattern 4: "key ... score N"
        for key, score_str in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)[^\n]*?\bscore\b[^\n]*?([0-9.]+)', text, re.IGNORECASE):
            add_pair(key, score_str)
        
        # Pattern 5: "key is best/good, score N"
        for key, score_str in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is\s+)?(?:best|good|better|top)\s*[,:]?\s*score[d]?\s*([0-9.]+)", text, re.IGNORECASE):
            add_pair(key, score_str)
        
        return result


    def _fix_single_quoted_json(self, text: str) -> str:
        """Convert single-quoted JSON to valid double-quoted JSON."""
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            if c == "'":
                prev_char = text[i-1] if i > 0 else ''
                next_char = text[i+1] if i+1 < len(text) else ''
                if prev_char.isalpha() and next_char.isalpha():
                    result.append(c)  # Keep apostrophe in contraction
                else:
                    result.append('"')  # Replace with double quote
            else:
                result.append(c)
            i += 1
        return ''.join(result)
    def _validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate parsed data against schema.
        Returns True if valid (or if schema is empty).
        """
        if not self.schema:
            return True
        
        # Check all required keys are present
        for key, expected_type in self.schema.items():
            if key not in data:
                return False
            if not isinstance(data[key], expected_type):
                return False
        
        return True

    def __repr__(self):
        return f'JSONAdaptor(schema={list(self.schema.keys())}, strict={self.strict})'


class LLMScorerAdaptor(JSONAdaptor):
    """
    Specialized adaptor for LLM function scoring.
    
    Takes raw LLM output and returns {function_name: score} dict.
    """
    
    def __init__(self, expected_fns: Optional[List[str]] = None):
        super().__init__(schema={}, strict=False)
        self.expected_fns = expected_fns or []
    
    def run(self, raw_output: str, visible_fns: List[str]) -> Dict[str, float]:
        """
        Parse LLM output into {fn: score} dict.
        
        Args:
            raw_output: Raw text from LLM
            visible_fns: List of visible function names for context
        
        Returns:
            Dict mapping function name to score (0.0-1.0)
        """
        result = {}
        
        # Use parent adaptation
        parsed = super().run(raw_output)
        
        # Filter to only known function names (case insensitive match)
        for fn in visible_fns:
            # Direct match
            if fn in parsed:
                result[fn] = self._clamp_score(parsed[fn])
            else:
                # Case-insensitive match
                for key, value in parsed.items():
                    if key.lower() == fn.lower():
                        result[fn] = self._clamp_score(value)
                        break
        
        # Also look for partial matches (e.g., "agg" → "aggregate_metrics")
        for fn in visible_fns:
            if fn not in result:
                for key, value in parsed.items():
                    if fn.lower() in key.lower() or key.lower() in fn.lower():
                        result[fn] = self._clamp_score(value)
        
        return result
    
    def _clamp_score(self, score: float) -> float:
        """Clamp score to 0.0-1.0 range."""
        return max(0.0, min(1.0, float(score)))


def test_adaptor():
    """Test the JSONAdaptor with various input formats."""
    
    test_cases = [
        # Already valid JSON
        ('{"aggregate_metrics": 0.9, "summarize_fields": 0.3}', True, 'Valid JSON'),
        
        # Single-quoted JSON (common MiniMax output)
        ("{'aggregate_metrics': 0.9, 'summarize_fields': 0.3}", True, 'Single-quoted JSON'),
        
        # Markdown code block with JSON
        ('```json\n{"key": 0.8}\n```', True, 'Markdown JSON'),
        
        # Natural language with scores
        ("aggregate_metrics: 0.95, summarize_fields: 0.2", True, 'Colon format'),
        
        # Natural language with parentheses
        ("aggregate_metrics (0.9), summarize_fields (0.3)", True, 'Parenthesis format'),
        
        # Natural language explanation
        ("I think aggregate_metrics is the best at 0.95, summarize_fields is about 0.3", True, 'Natural language'),
        
        # Only explanation, no scores
        ("I need to analyze this task...", False, 'No scores'),
        
        # Mixed format
        ('{"aggregate_metrics": 0.9}\n\nThe user wants me to score these functions...', True, 'Mixed'),
        
        # Wrong key types
        ('{"aggregate_metrics": "high", "summarize_fields": 0.3}', False, 'Invalid value types'),
    ]
    
    print('=== JSONAdaptor Test ===')
    adaptor = LLMScorerAdaptor()
    
    for raw, should_succeed, desc in test_cases:
        try:
            result = adaptor.run(raw, ['aggregate_metrics', 'summarize_fields', 'compute_distribution'])
            status = 'OK' if result else 'EMPTY'
            print(f'  [{status}] {desc}: {result}')
        except Exception as e:
            print(f'  [FAIL] {desc}: {e}')
    
    print()
    
    # Test with expected functions
    print('=== LLMScorerAdaptor with expected functions ===')
    scorer = LLMScorerAdaptor(expected_fns=['aggregate_metrics', 'summarize_fields'])
    
    test_outputs = [
        "aggregate_metrics: 0.95, summarize_fields: 0.2",
        "I think aggregate_metrics gets 0.9 and summarize_fields gets 0.4",
        "```json\n{\"aggregate_metrics\": 0.85}\n```",
    ]
    
    for output in test_outputs:
        result = scorer.run(output, ['aggregate_metrics', 'summarize_fields'])
        print(f'  Input: {output[:60]}...')
        print(f'  Output: {result}')
        print()


if __name__ == '__main__':
    test_adaptor()
