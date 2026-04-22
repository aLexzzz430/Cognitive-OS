"""
surfaces/novel_api_adapter.py

NovelAPISurfaceAdapter: CoreMainLoop-compatible adapter for unknown-API discovery.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random

from core.surfaces.base import (
    SurfaceAdapter, SurfaceObservation, SurfaceAction,
    ActionResult, ToolSpec
)


# =============================================================================
# NovelAPIDomain (Internal Implementation)
# =============================================================================

class NovelAPIDomain:
    """
    Internal domain logic. 5 hidden functions to discover.
    
    Pack D v2: Hidden functions + prerequisites + test opportunity cost
    
    Function availability tiers:
      Episode 1 (Tier 1 - always available): compute_stats, filter_by_predicate, array_transform
      Episode 2+ (Tier 2 - hidden in Ep1): join_tables, aggregate_group
    
    Prerequisite chain:
      array_transform requires compute_stats to be discovered first
      join_tables requires filter_by_predicate to be discovered first
      aggregate_group requires join_tables to be discovered first
    """

    def __init__(self, seed: int = 0, episode: int = 1, prior_discoveries: list = None):
        self._rng = random.Random(seed)
        self._episode = episode
        
        # Tier 1: always available after prerequisites
        self._tier1 = {'array_transform', 'compute_stats', 'filter_by_predicate'}
        # Tier 2: hidden until episode >= 2
        self._tier2 = {'join_tables', 'aggregate_group'}
        
        self._secret_api = {
            'array_transform': self._array_transform,
            'compute_stats': self._compute_stats,
            'filter_by_predicate': self._filter_by_predicate,
            'join_tables': self._join_tables,
            'aggregate_group': self._aggregate_group,
        }
        
        # Prerequisites: function → list of functions that must be discovered first
        self._prerequisites = {
            'array_transform': ['compute_stats'],
            'join_tables': ['filter_by_predicate'],
            'aggregate_group': ['join_tables'],
        }
        
        # Cross-episode discoveries: functions discovered in prior episodes
        # Count toward prerequisites but don't fire discovery events
        self._prior_discoveries: set = set(prior_discoveries or [])
        
        self._discovered: dict = {n: False for n in self._secret_api}
        self._call_log: list = []
        self._step: int = 0

    def reset(self, seed: int | None = None, episode: int = 1, prior_discoveries: list = None) -> dict:
        if seed is not None:
            self._rng = random.Random(seed)
        self._episode = episode
        self._prior_discoveries = set(prior_discoveries or [])
        self._step = 0
        self._discovered = {n: False for n in self._secret_api}
        self._call_log = []
        return self._build_obs()

    def observe(self) -> dict:
        return self._build_obs()

    def inspect(self) -> dict:
        self._step += 1
        return {
            'type': 'inspect',
            'visible_functions': list(self._secret_api.keys()),
            'signatures': {},
            'step': self._step,
        }

    def call_hidden_function(self, function_name: str, test_mode: bool = False, **kwargs) -> dict:
        self._step += 1
        result = {
            'function_name': function_name, 'kwargs': kwargs,
            'called': False, 'result': None, 'error': None,
            'discovery_event': None, 'test_cost': 0.0,
        }
        
        # Gate 0.1/0.5: Check if function is in hidden tier
        if function_name in self._tier2 and self._episode < 2:
            result['error'] = f"Function '{function_name}' is not yet available. It will be revealed in a future episode."
            self._call_log.append(result)
            return result
        
        # Gate 0.1: Check prerequisites (including cross-episode discoveries)
        prereqs = self._prerequisites.get(function_name, [])
        all_discoveries = {k for k,v in self._discovered.items() if v} | self._prior_discoveries
        unmet = [p for p in prereqs if p not in all_discoveries]
        if unmet:
            result['error'] = f"Function '{function_name}' is locked. Prerequisites required: {', '.join(unmet)}."
            self._call_log.append(result)
            return result
        
        # Gate 0.3: Test opportunity cost
        if test_mode:
            result['test_cost'] = 0.5  # Each test costs 0.5 reward
        
        if function_name not in self._secret_api:
            result['error'] = f"Function '{function_name}' is unknown."
            self._call_log.append(result)
            return result
        try:
            raw_result = self._secret_api[function_name](**kwargs)
            result['called'] = True
            result['result'] = raw_result
            already_known = self._discovered.get(function_name, False) or function_name in self._prior_discoveries
            if not already_known:
                self._discovered[function_name] = True
                result['discovery_event'] = {
                    'type': 'function_discovered',
                    'function_name': function_name, 'step': self._step,
                }
            self._call_log.append(result)
        except Exception as e:
            result['error'] = str(e)
            self._call_log.append(result)
        return result

    def submit(self, answer: str) -> dict:
        self._step += 1
        return {'type': 'submit', 'answer': answer, 'terminal': True, 'step': self._step}

    def _build_obs(self) -> dict:
        disc = [n for n, f in self._discovered.items() if f]
        available = [n for n in self._secret_api if self._is_available(n)]
        hidden = list(self._tier2 - {n for n in disc if n in self._tier2})
        return {
            'type': 'novel_api', 'step': self._step,
            'episode': self._episode,
            'discovered_functions': disc,
            'available_functions': available,
            'hidden_functions': hidden,
            'total_functions': len(self._secret_api),
            'call_log': list(self._call_log),
            'terminal': False,
            'text': f"NovelAPI Ep{self._episode}. {len(disc)}/{len(self._secret_api)} discovered. {len(hidden)} hidden. Step {self._step}.",
            'status': 'active',
        }

    def _is_available(self, fn: str) -> bool:
        """Check if function is available (tier1 or tier2 revealed in right episode, prerequisites met)."""
        if fn in self._tier2 and self._episode < 2:
            return False
        prereqs = self._prerequisites.get(fn, [])
        all_discoveries = {k for k,v in self._discovered.items() if v} | self._prior_discoveries
        return all(p in all_discoveries for p in prereqs)

    def _array_transform(self, data: list, op: str) -> list:
        ops = {'sort': sorted, 'reverse': lambda x: list(reversed(x)),
               'dup': lambda x: x*2, 'compact': lambda x: [v for v in x if v]}
        return ops.get(op, lambda x: x)(list(data))

    def _compute_stats(self, data: list, stat: str) -> float:
        import statistics
        d = []
        for x in data:
            try: d.append(float(x))
            except (ValueError, TypeError): pass
        m = {'mean': lambda: statistics.mean(d), 'median': lambda: statistics.median(d),
             'sum': lambda: sum(d), 'min': lambda: min(d), 'max': lambda: max(d),
             'count': lambda: len(d)}
        return m.get(stat, lambda: 0.0)()

    def _filter_by_predicate(self, data: list, pred: str) -> list:
        pm = {'positive': lambda x: float(x) > 0, 'negative': lambda x: float(x) < 0,
              'zero': lambda x: float(x) == 0, 'odd': lambda x: int(x) % 2 != 0,
              'even': lambda x: int(x) % 2 == 0}
        fn = pm.get(pred, lambda x: True)
        return [x for x in data if fn(x)]

    def _join_tables(self, left: list, right: list, key: str) -> list:
        lbk = {}
        for row in left:
            k = row.get(key, '')
            lbk.setdefault(k, []).append(row)
        result = []
        for row in right:
            k = row.get(key, '')
            for lrow in lbk.get(k, []):
                result.append({**lrow, **row})
        return result

    def _aggregate_group(self, data: list, group_key: str, agg: str) -> dict:
        grp = {}
        for row in data:
            k = row.get(group_key, '__unknown__')
            grp.setdefault(k, []).append(row)
        res = {}
        for k, rows in grp.items():
            if agg == 'count':
                res[k] = len(rows)
            else:
                vals = [float(row.get(agg, 0)) for row in rows if agg in row]
                res[k] = sum(vals) / max(len(vals), 1)
        return res


# =============================================================================
# NovelAPISurfaceAdapter
# =============================================================================

class NovelAPISurfaceAdapter:
    """
    CoreMainLoop-compatible adapter for NovelAPI domain.

    Provides compatibility shim for CoreMainLoop integration:
      - observe() returns dict (not SurfaceObservation) for _step1
      - act() accepts dict/str/SurfaceAction
      - update_continuity_state() is a no-op stub
    """

    def __init__(self, seed: int = 0, episode: int = 1, prior_discoveries: list = None):
        self._env = NovelAPIDomain(seed=seed, episode=episode, prior_discoveries=prior_discoveries)
        self._last_obs: dict = {}
        self._step: int = 0
        self._phase: str = 'active'  # Required by CoreMainLoop._step2_state_refresh

    def reset(self, seed: int | None = None, episode: int = 1, prior_discoveries: list = None) -> SurfaceObservation:
        raw = self._env.reset(seed=seed, episode=episode, prior_discoveries=prior_discoveries)
        self._last_obs = raw
        self._step = 0
        return self._to_obs(raw)

    def observe(self) -> dict:
        """Returns dict for CoreMainLoop._step1_observe compatibility."""
        raw = self._env.observe()
        self._last_obs = raw
        return raw  # Return dict, not SurfaceObservation

    def act(self, action: Any) -> ActionResult:
        """Execute action. Accepts SurfaceAction, dict, or str."""
        self._step += 1
        raw_result: dict = {}

        # Normalize
        if isinstance(action, SurfaceAction):
            kind, payload = action.kind, action.payload
        elif isinstance(action, dict):
            kind = action.get('kind', action.get('action', 'wait'))
            payload = action.get('payload', action)
        elif isinstance(action, str):
            kind, payload = action, {}
        else:
            kind, payload = 'wait', {}

        if kind == 'call_tool' or (isinstance(action, dict) and 'tool_name' in action):
            tool_name = payload.get('tool_name', '')
            tool_args = payload.get('tool_args', {})
            if tool_name == 'inspect_api_surface':
                raw_result = self._env.inspect()
            elif tool_name == 'call_hidden_function':
                fn_name = tool_args.get('function_name', '')
                kwargs = tool_args.get('kwargs', {})
                test_mode = tool_args.get('test_mode', payload.get('test_mode', False))
                raw_result = self._env.call_hidden_function(fn_name, test_mode=test_mode, **kwargs)
            elif tool_name == 'submit_solution':
                raw_result = self._env.submit(tool_args.get('answer', ''))
            else:
                raw_result = {'error': f"Unknown tool: {tool_name}"}
        elif kind == 'inspect':
            raw_result = self._env.inspect()
        elif kind == 'wait':
            raw_result = self._env.observe()
        elif kind == 'submit':
            raw_result = self._env.submit(payload.get('answer', ''))
        else:
            raw_result = {'error': f"Unknown action kind: {kind}"}

        self._last_obs = raw_result
        # Re-observe to get updated discovered_functions after the call
        updated_obs = self._env.observe()
        raw_result['discovered_functions'] = updated_obs['discovered_functions']
        obs = self._to_obs(updated_obs)

        # Compute reward_signal and phase for CoreMainLoop compatibility
        # Gate 0.3: test_cost subtracts from reward
        test_cost = raw_result.get('test_cost', 0.0)
        discovered = [n for n, v in self._env._discovered.items() if v] if hasattr(self, "_env") and hasattr(self._env, "_discovered") else []
        total = raw_result.get('total_functions', 5)
        reward_signal = 0.0
        phase = 'discovery'
        if discovered:
            reward_signal = 1.0 * len(discovered)  # Per discovery
        if len(discovered) >= total:
            reward_signal = 10.0  # All discovered
            phase = 'solved'
        reward_signal -= test_cost  # Subtract test opportunity cost

        events = []
        if isinstance(raw_result, dict):
            if raw_result.get('discovery_event'):
                events.append({'type': 'discovery', 'data': raw_result['discovery_event'], 'step': self._step})
            if raw_result.get('error'):
                events.append({'type': 'error', 'data': raw_result['error'], 'step': self._step})
            if raw_result.get('called'):
                events.append({'type': 'function_call', 'data': {
                    'function_name': raw_result.get('function_name'),
                    'result': raw_result.get('result'),
                }, 'step': self._step})

        # Attach reward_signal and phase to raw for CoreMainLoop compatibility
        raw_result['reward_signal'] = reward_signal
        raw_result['phase'] = phase

        return ActionResult(
            ok=raw_result.get('error') is None,
            observation=obs,
            events=events,
            raw=raw_result,
        )

    def get_state(self) -> dict:
        """Return current environment state for CoreMainLoop._step2_state_refresh."""
        return {
            'phase': getattr(self, '_phase', 'active'),
            'step': getattr(self, '_step', 0),
        }

    def get_shared_knowledge(self) -> dict:
        """Return shared knowledge state. Stub for NovelAPI domain."""
        return {
            'discovered_functions': [n for n, v in self._env._discovered.items() if v] if hasattr(self, '_env') else [],
            'call_log': list(self._call_log) if hasattr(self, '_call_log') else [],
        }

    def update_continuity_state(self, continuity: dict) -> None:
        """Sync environment state into the continuity dict (Step 2)."""
        continuity['phase'] = getattr(self, '_phase', 'active')
        continuity['step'] = getattr(self, '_step', 0)

    def _to_obs(self, raw: dict) -> SurfaceObservation:
        terminal = raw.get('terminal', False) or (self._step >= 50)
        disc = raw.get('discovered_functions', [])
        return SurfaceObservation(
            text=raw.get('text', f"NovelAPI. {len(disc)}/5 discovered."),
            structured={'step': self._step, 'discovered_functions': disc,
                       'total_functions': 5, 'discovery_count': len(disc)},
            available_tools=self._tool_specs(),
            terminal=terminal,
            reward=None,
            raw=raw,
        )

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="inspect_api_surface",
                description="Inspect visible function names (no signatures).",
                input_schema={"type": "object", "properties": {}},
            ),
            ToolSpec(
                name="call_hidden_function",
                description="Call a hidden API function.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string"},
                        "kwargs": {"type": "object"},
                    },
                    "required": ["function_name", "kwargs"],
                },
            ),
            ToolSpec(
                name="submit_solution",
                description="Submit final answer.",
                input_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
        ]
