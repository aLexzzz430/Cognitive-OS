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
from typing import Dict, Any, Optional, Callable, List


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