"""
modules/llm/minimax_client.py

Minimax API client for LLM interfaces.
Model: MiniMax-M2.7
"""

import requests
import json
import re
import time
import os
from typing import List, Dict, Any, Optional

from modules.llm.json_adaptor import normalize_llm_output


class MinimaxClient:
    """
    Minimax API client that implements the LLM interface expected by all LLM wrappers.

    Usage:
        client = MinimaxClient(api_token='your_token_here')
        response = client.complete('What is 2+2?')
    """

    def __init__(self, api_token: Optional[str] = None, token_file: Optional[str] = None):
        self._token = self._resolve_token(api_token=api_token, token_file=token_file)
        self._url = 'https://api.minimax.io/v1/chat/completions'
        self._model = 'MiniMax-M2.7'
        self._headers = {
            'Authorization': f'Bearer {self._token}',
            'Content-Type': 'application/json',
        }
        self._max_retries = 5
        self._base_delay = 2.0  # seconds
        self._request_count = 0
        self._request_wall_sec = 0.0

    def _resolve_token(self, api_token: Optional[str], token_file: Optional[str]) -> str:
        """
        Resolve API token with priority:
        1) Explicit api_token constructor argument
        2) Explicit token_file constructor argument
        3) Environment variable MINIMAX_API_TOKEN
        4) Token file pointed by environment variable MINIMAX_TOKEN_FILE
        """
        if api_token:
            return api_token.strip()

        resolved_token_file = token_file or os.getenv('MINIMAX_TOKEN_FILE')
        if resolved_token_file:
            with open(resolved_token_file) as f:
                return f.read().strip()

        env_token = os.getenv('MINIMAX_API_TOKEN')
        if env_token:
            return env_token.strip()

        raise ValueError(
            'MiniMax token is required. Provide api_token, token_file, '
            'MINIMAX_API_TOKEN, or MINIMAX_TOKEN_FILE.'
        )

    def complete(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a completion request to MiniMax-M2.7.
        Retries with exponential backoff on 529/503 errors.
        """
        raw_content = self.complete_raw(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
        return self._strip_thinking(raw_content)

    def complete_raw(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a completion request to MiniMax-M2.7 and return raw content, including thinking blocks.
        Retries with exponential backoff on 529/503 errors.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        payload = {
            'model': self._model,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'stream': False,
        }

        last_error = None
        for attempt in range(self._max_retries):
            try:
                started_at = time.perf_counter()
                try:
                    resp = requests.post(self._url, headers=self._headers, json=payload, timeout=30)
                finally:
                    self._request_count += 1
                    self._request_wall_sec += max(0.0, time.perf_counter() - started_at)
                
                if resp.status_code == 200:
                    result = resp.json()
                    return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                elif resp.status_code in (529, 503, 502, 504):
                    # Server overloaded - retry with backoff
                    last_error = f'HTTP {resp.status_code}'
                    if attempt < self._max_retries - 1:
                        delay = self._base_delay * (2 ** attempt)
                        time.sleep(delay)
                        continue
                    else:
                        raise RuntimeError(f'Minimax API error {resp.status_code}: {resp.text}')
                
                else:
                    raise RuntimeError(f'Minimax API error {resp.status_code}: {resp.text}')
            
            except requests.exceptions.Timeout as e:
                last_error = f'Timeout: {e}'
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f'Minimax timeout after {self._max_retries} retries')
            
            except requests.exceptions.ConnectionError as e:
                last_error = f'ConnectionError: {e}'
                if attempt < self._max_retries - 1:
                    delay = self._base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                else:
                    raise RuntimeError(f'Minimax connection error after {self._max_retries} retries')
        
        # Should not reach here, but just in case
        raise RuntimeError(f'Minimax API failed after {self._max_retries} retries: {last_error}')

    def complete_json(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Send a completion request and parse response as JSON.
        Handles single-quoted JSON, markdown code blocks, non-JSON responses.
        """
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        return normalize_llm_output(
            text,
            output_kind="minimax_complete_json",
            expected_type="dict",
        ).parsed_dict()

    def _fix_single_quoted_json(self, text: str) -> str:
        """
        Convert single-quoted JSON to valid double-quoted JSON.
        Handles JavaScript-style output like {'key': 'value'}.
        Preserves apostrophes in contractions like don't.
        """
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

    def _extract_json_fallback(self, text: str) -> Dict[str, Any]:
        """
        Fallback: extract key:value pairs from text.
        """
        result = {}
        matches = re.findall(r'["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?\s*:\s*([0-9.]+|["\'].*?["\'])', text)
        for key, value in matches:
            try:
                if value.startswith('"') or value.startswith("'"):
                    result[key] = value[1:-1]
                else:
                    result[key] = float(value) if '.' in value else int(value)
            except ValueError:
                result[key] = value.strip('"\'')
        return result if result else {}

    def _strip_thinking(self, text: str) -> str:
        """Remove thinking blocks from MiniMax-M2.7 output."""
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.replace('</think>', '')
        return cleaned.strip()

    def __repr__(self):
        return f'MinimaxClient(model={self._model})'

    @property
    def request_count(self) -> int:
        return int(self._request_count)

    @property
    def request_wall_sec(self) -> float:
        return float(self._request_wall_sec)
