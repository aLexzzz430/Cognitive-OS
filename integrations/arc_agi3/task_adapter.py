from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.environment import (
    GenericActionDescriptor,
    GenericActionEnvelope,
    GenericEntity,
    GenericObservation,
    GenericRelation,
    GenericStateDelta,
    GenericTaskSpec,
    GenericTransition,
)
from core.orchestration.action_utils import extract_action_function_name, extract_action_kind
from core.surfaces.base import ActionResult, SurfaceObservation, ToolSpec
from integrations.arc_agi3.action_adapter import ARCActionAdapter, ARCActionChoice
from integrations.arc_agi3.perception_bridge import PerceptionBridge
from integrations.arc_agi3.state_bridge import ArcGameSessionState

try:
    import arc_agi
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
except Exception as exc:  # pragma: no cover - import guard for environments without toolkit
    arc_agi = None
    Arcade = None
    OperationMode = None
    GameAction = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class ARCAGI3SurfaceAdapter:
    """
    Surface adapter that makes ARC-AGI-3 environments look like a CoreMainLoop world.

    Session correctness rules:
    - reset must yield a non-empty guid before an episode is considered valid
    - resolved game_id/guid from the environment are tracked explicitly
    - internal actions never hit env.step(...), but still pay an internal audit cost
    """

    def __init__(
        self,
        game_id: str,
        *,
        operation_mode: str = "ONLINE",
        arc_api_key: Optional[str] = None,
        arc_base_url: Optional[str] = None,
        environments_dir: Optional[str] = None,
        recordings_dir: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
        reset_retry_limit: int = 3,
    ) -> None:
        if Arcade is None or GameAction is None:
            raise RuntimeError(
                "ARC-AGI toolkit is not installed. Install `arc-agi` and `arcengine` first."
            ) from _IMPORT_ERROR

        self.game_id = str(game_id)
        self.seed = int(seed)
        self.logger = logger or logging.getLogger("arc_agi3_adapter")
        self._perception = PerceptionBridge()
        self._action_adapter = ARCActionAdapter()
        self._render_mode = render_mode
        self._episode_index = 1
        self._reset_retry_limit = max(1, int(reset_retry_limit))

        self._requested_game_id = self.game_id
        self._resolved_game_id = self.game_id
        self._current_guid = ""

        op_mode = self._resolve_operation_mode(operation_mode)
        normalized_api_key = str(arc_api_key or "")
        normalized_base_url = str(arc_base_url or "https://three.arcprize.org")
        self._arcade = Arcade(
            operation_mode=op_mode,
            arc_api_key=normalized_api_key,
            arc_base_url=normalized_base_url,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            logger=self.logger,
        )
        make_kwargs: Dict[str, Any] = {}
        if render_mode:
            make_kwargs["render_mode"] = render_mode
        self._env = self._arcade.make(self.game_id, **make_kwargs)

        self._state = ArcGameSessionState(
            requested_game_id=self._requested_game_id,
            resolved_game_id=self._resolved_game_id,
            seed=self.seed,
            episode_index=self._episode_index,
        )
        self._last_obs: Dict[str, Any] = {}

    def reset(self, seed: int | None = None, episode: int | None = None, prior_discoveries: list | None = None) -> SurfaceObservation:
        if seed is not None:
            self.seed = int(seed)
        if episode is not None:
            self._episode_index = int(episode)

        normalized = self._reset_until_session_ready()
        self._state.reset_for_episode(
            guid=self._current_guid,
            resolved_game_id=self._resolved_game_id,
            episode_index=self._episode_index,
        )
        self._state.record_observation(normalized)
        self._last_obs = normalized
        return self._to_surface_observation(normalized)

    def next_episode(self) -> None:
        self._episode_index += 1
        self.reset(episode=self._episode_index)

    def observe(self) -> Dict[str, Any]:
        if self._last_obs:
            return dict(self._last_obs)
        normalized = self._reset_until_session_ready()
        self._state.record_observation(normalized)
        self._last_obs = normalized
        return normalized

    def act(self, action: Any) -> ActionResult:
        normalized_action = dict(action) if isinstance(action, dict) else {"kind": str(action)}

        if not self._current_guid:
            raise RuntimeError(
                "ARC-AGI-3 session is not ready: empty guid. "
                "Reset must succeed with a non-empty guid before actions are executed."
            )

        available_names = self._available_action_names()
        perception = self._last_obs.get("perception", {}) if isinstance(self._last_obs.get("perception", {}), dict) else {}
        try:
            choice = self._action_adapter.to_arc_action(
                normalized_action,
                available_action_names=available_names,
                perception=perception,
            )
        except Exception as exc:
            failure_raw = {
                "success": False,
                "reward": 0.0,
                "terminal": False,
                "done": False,
                "solved": False,
                "state": "ADAPTER_ERROR",
                "failure_reason": str(exc),
                "guid": self._current_guid,
                "resolved_game_id": self._resolved_game_id,
                "requested_game_id": self._requested_game_id,
                "action_input": None,
                "attempted_action": normalized_action,
                "external_cost_paid": 0,
            }
            return ActionResult(
                ok=False,
                observation=self._to_surface_observation(self.observe()),
                raw=failure_raw,
                events=[{"type": "adapter_error", "reason": str(exc)}],
            )

        if not choice.requires_env_step:
            return self._handle_internal_noop(choice, normalized_action)

        if GameAction is None:
            return self._build_step_error_result(
                choice=choice,
                failure_reason="arc_agi3_action_enum_unavailable",
                failure_detail="ARC-AGI-3 action enum is unavailable because the toolkit import guard was triggered.",
            )

        try:
            env_action = getattr(GameAction, choice.action_name)
        except AttributeError:
            return self._build_step_error_result(
                choice=choice,
                failure_reason="arc_agi3_action_enum_resolution_failed",
                failure_detail=f"ARC-AGI-3 action enum has no member {choice.action_name!r}.",
            )
        try:
            raw = self._env.step(env_action, data=choice.data, reasoning=choice.reasoning)
        except Exception as exc:
            failure_reason = self._classify_step_exception_failure_reason(choice.action_name, exc)
            return self._build_step_error_result(
                choice=choice,
                failure_reason=failure_reason,
                failure_detail=str(exc),
            )

        if raw is None:
            failure_reason = (
                "arc_agi3_schema_failure_remote_rejection"
                if choice.action_name == "ACTION6"
                else "remote_step_failed"
            )
            return self._build_step_error_result(
                choice=choice,
                failure_reason=failure_reason,
                failure_detail="environment step returned no observation; remote service likely rejected or failed the action",
            )

        normalized = self._normalize_obs(raw)
        self._sync_session_from_obs(normalized)
        reward = self._infer_reward(normalized)
        self._state.record_action(
            {
                "action_name": choice.action_name,
                "action_id": choice.action_id,
                "data": dict(choice.data or {}),
                "reasoning": dict(choice.reasoning or {}),
            },
            normalized,
            reward,
        )
        self._state.record_observation(normalized)
        self._last_obs = normalized

        result_raw = {
            "success": bool(normalized.get("success", normalized.get("state") == "WIN" or normalized.get("solved", False))),
            "reward": reward,
            "terminal": bool(normalized.get("terminal", False)),
            "done": bool(normalized.get("done", False)),
            "solved": bool(normalized.get("solved", False)),
            "state": normalized.get("state"),
            "guid": normalized.get("guid", ""),
            "resolved_game_id": normalized.get("resolved_game_id", self._resolved_game_id),
            "requested_game_id": self._requested_game_id,
            "novel_api": normalized.get("novel_api", {}),
            "available_actions": normalized.get("available_actions", []),
            "action_input": {
                "id": choice.action_id,
                "name": choice.action_name,
                "data": dict(choice.data or {}),
            },
            "external_cost_paid": int(choice.external_cost),
        }
        events = []
        if result_raw["solved"]:
            events.append({"type": "solved", "game_id": self._resolved_game_id, "turn": self._state.turn_index})
        return ActionResult(
            ok=True,
            observation=self._to_surface_observation(normalized),
            raw=result_raw,
            events=events,
        )

    def _build_step_error_result(
        self,
        *,
        choice: ARCActionChoice,
        failure_reason: str,
        failure_detail: str | None = None,
    ) -> ActionResult:
        failure_raw = {
            "success": False,
            "reward": 0.0,
            "terminal": False,
            "done": False,
            "solved": False,
            "state": "STEP_ERROR",
            "failure_reason": failure_reason,
            "guid": self._current_guid,
            "resolved_game_id": self._resolved_game_id,
            "requested_game_id": self._requested_game_id,
            "action_input": {
                "id": choice.action_id,
                "name": choice.action_name,
                "data": dict(choice.data or {}),
            },
            "external_cost_paid": int(choice.external_cost),
        }
        if failure_detail:
            failure_raw["failure_detail"] = failure_detail

        return ActionResult(
            ok=False,
            observation=self._to_surface_observation(self.observe()),
            raw=failure_raw,
            events=[{"type": "step_error", "reason": failure_reason}],
        )

    def _classify_step_exception_failure_reason(self, action_name: str, exc: Exception) -> str:
        text = str(exc or "").strip().lower()
        if action_name == "ACTION6" and (
            "400" in text
            or "bad request" in text
            or "illegal click" in text
            or "coordinate" in text
            or "remote rejection" in text
        ):
            return "arc_agi3_schema_failure_remote_rejection"
        return str(exc)

    def _handle_internal_noop(
        self,
        choice: ARCActionChoice,
        action: Dict[str, Any],
    ) -> ActionResult:
        obs = self.observe()
        learning_delta = self._build_internal_learning_delta(
            action_kind=str(choice.internal_action_kind or "internal_noop"),
            obs=obs,
        )

        raw_result = {
            "success": True,
            "reward": 0.0,
            "terminal": bool(obs.get("terminal", False)),
            "done": bool(obs.get("done", False)),
            "solved": bool(obs.get("solved", False)),
            "state": obs.get("state"),
            "guid": obs.get("guid", ""),
            "resolved_game_id": obs.get("resolved_game_id", self._resolved_game_id),
            "requested_game_id": self._requested_game_id,
            "novel_api": obs.get("novel_api", {}),
            "available_actions": obs.get("available_actions", []),
            "internal_action": str(choice.internal_action_kind or "internal_noop"),
            "external_cost_paid": int(choice.external_cost),
            "learning_delta": learning_delta,
            "action_input": None,
        }

        self._state.record_internal_action(
            action_kind=str(choice.internal_action_kind or "internal_noop"),
            reward=0.0,
            external_cost=int(choice.external_cost),
            learning_delta=learning_delta,
            response=raw_result,
        )

        return ActionResult(
            ok=True,
            observation=self._to_surface_observation(obs),
            raw=raw_result,
            events=[],
        )

    def _reset_until_session_ready(self) -> Dict[str, Any]:
        last_normalized: Optional[Dict[str, Any]] = None
        for attempt in range(1, self._reset_retry_limit + 1):
            raw = self._env.reset()
            normalized = self._normalize_obs(raw)
            self._sync_session_from_obs(normalized)
            last_normalized = normalized
            if self._current_guid:
                return normalized

            self.logger.warning(
                "ARC-AGI-3 reset attempt %s/%s returned empty guid; retrying. requested_game_id=%s resolved_game_id=%s",
                attempt,
                self._reset_retry_limit,
                self._requested_game_id,
                normalized.get("resolved_game_id", self._resolved_game_id),
            )

        raise RuntimeError(
            "ARC-AGI-3 reset failed to produce a non-empty guid after "
            f"{self._reset_retry_limit} attempts. Last resolved_game_id="
            f"{(last_normalized or {}).get('resolved_game_id', self._resolved_game_id)}"
        )

    def _sync_session_from_obs(self, obs: Dict[str, Any]) -> None:
        guid = str(obs.get("guid", "") or "").strip()
        resolved = str(obs.get("resolved_game_id", "") or "").strip()

        if resolved:
            self._resolved_game_id = resolved
        if guid:
            self._current_guid = guid

        self._state.guid = self._current_guid
        self._state.resolved_game_id = self._resolved_game_id

    def _build_internal_learning_delta(
        self,
        *,
        action_kind: str,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        recent_actions = []
        recent_rewards = []
        for row in self._state.action_history[-3:]:
            action_payload = row.get("action", {}) if isinstance(row, dict) else {}
            if isinstance(action_payload, dict):
                recent_actions.append(str(action_payload.get("action_name", action_payload.get("raw", "")) or ""))
            recent_rewards.append(float(row.get("reward", 0.0) or 0.0))

        repeated_action_streak = 0
        if recent_actions:
            last = recent_actions[-1]
            for item in reversed(recent_actions):
                if item == last:
                    repeated_action_streak += 1
                else:
                    break

        available_names = list(obs.get("available_action_names", []) if isinstance(obs.get("available_action_names", []), list) else [])
        return {
            "internal_action_kind": action_kind,
            "recent_external_actions": recent_actions,
            "recent_reward_sum": round(sum(recent_rewards), 4),
            "repeated_action_streak": repeated_action_streak,
            "available_action_names": available_names,
            "state": str(obs.get("state", "") or ""),
            "observation_changed": False,
            "learning_required_next": True,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "episode_index": self._state.episode_index,
            "turn_index": self._state.turn_index,
            "total_reward": self._state.total_reward,
            "solved": self._state.solved,
            "requested_game_id": self._requested_game_id,
            "resolved_game_id": self._resolved_game_id,
            "guid": self._current_guid,
            "inspect_count": self._state.inspect_count,
            "external_noop_cost": self._state.external_noop_cost,
        }

    def get_shared_knowledge(self) -> Dict[str, Any]:
        return {
            "action_history": list(self._state.action_history),
            "internal_action_history": list(self._state.internal_action_history),
            "observation_count": len(self._state.observation_history),
            "requested_game_id": self._requested_game_id,
            "resolved_game_id": self._resolved_game_id,
            "guid": self._current_guid,
        }

    def update_continuity_state(self, continuity: dict) -> None:
        continuity["phase"] = "solved" if self._state.solved else "active"
        continuity["step"] = self._state.turn_index
        continuity["requested_game_id"] = self._requested_game_id
        continuity["resolved_game_id"] = self._resolved_game_id
        continuity["guid"] = self._current_guid
        continuity["inspect_count"] = self._state.inspect_count
        continuity["external_noop_cost"] = self._state.external_noop_cost

    def scorecard(self) -> Any:
        if hasattr(self._arcade, "get_scorecard"):
            try:
                return self._arcade.get_scorecard()
            except Exception:
                return None
        return None

    def get_generic_task_spec(self) -> GenericTaskSpec:
        available_names = self._available_action_names()
        if not available_names and isinstance(self._last_obs, dict):
            available_names = [
                str(name or "").strip()
                for name in list(self._last_obs.get("available_functions", []) or [])
                if str(name or "").strip()
            ]
        return GenericTaskSpec(
            task_id=str(self._resolved_game_id or self._requested_game_id or self.game_id),
            environment_family="arc_agi3",
            instruction=f"Interact with ARC-AGI-3 game {self._requested_game_id} to make progress and solve the environment.",
            success_criteria=[
                "reach WIN state",
                "increase levels_completed or solve trajectory state",
            ],
            available_action_names=list(available_names),
            metadata={
                "requested_game_id": self._requested_game_id,
                "resolved_game_id": self._resolved_game_id,
                "guid": self._current_guid,
                "episode_index": self._episode_index,
                "seed": self.seed,
            },
        )

    def get_generic_observation(
        self,
        obs: Optional[Dict[str, Any]] = None,
    ) -> GenericObservation:
        normalized = self._coerce_normalized_obs(obs)
        perception = normalized.get("perception", {}) if isinstance(normalized.get("perception", {}), dict) else {}
        entities = self._generic_entities_from_obs(normalized, perception)
        relations = self._generic_relations_from_obs(normalized, perception, entities)
        available_actions = self._generic_action_descriptors(normalized)
        resolved_game_id = str(normalized.get("resolved_game_id", self._resolved_game_id) or self._requested_game_id or self.game_id)
        guid = str(normalized.get("guid", self._current_guid) or "")
        return GenericObservation(
            observation_id=f"{resolved_game_id}:{guid or 'noguid'}:{int(self._state.turn_index if hasattr(self, '_state') else 0)}",
            environment_family="arc_agi3",
            task_id=resolved_game_id,
            text=str(normalized.get("text", "") or ""),
            entities=entities,
            relations=relations,
            available_actions=available_actions,
            state={
                "state": str(normalized.get("state", "") or ""),
                "terminal": bool(normalized.get("terminal", False)),
                "solved": bool(normalized.get("solved", False)),
                "guid": guid,
                "resolved_game_id": resolved_game_id,
                "requested_game_id": self._requested_game_id,
                "levels_completed": normalized.get("levels_completed"),
                "win_levels": normalized.get("win_levels"),
            },
            metadata={
                "grid_shape": dict(perception.get("grid_shape", {}) or {}),
                "background_color": perception.get("background_color"),
                "active_pixel_count": int(perception.get("active_pixel_count", 0) or 0),
                "changed_pixel_count": int(perception.get("changed_pixel_count", 0) or 0),
                "unique_colors": list(perception.get("unique_colors", []) or []),
            },
            raw=dict(normalized),
        )

    def get_generic_action(
        self,
        action: Any,
        *,
        obs: Optional[Dict[str, Any]] = None,
    ) -> GenericActionEnvelope:
        normalized_obs = self._coerce_normalized_obs(obs)
        action_dict = dict(action) if isinstance(action, dict) else {"kind": str(action or "")}
        payload = action_dict.get("payload", {}) if isinstance(action_dict.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        kwargs = dict(tool_args.get("kwargs", {}) or {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
        meta = action_dict.get("_candidate_meta", {}) if isinstance(action_dict.get("_candidate_meta", {}), dict) else {}
        function_name = extract_action_function_name(action_dict, default="wait")
        action_kind = extract_action_kind(action_dict, default="call_tool")
        action_family = self._generic_action_family(function_name, action_kind)
        target_entity_id = str(
            meta.get("anchor_ref", "")
            or meta.get("object_id", "")
            or meta.get("target_anchor_ref", "")
            or ""
        ).strip()
        target_family = str(
            meta.get("target_family", "")
            or meta.get("surface_click_role", "")
            or meta.get("role", "")
            or ""
        ).strip()
        if function_name.upper() == "ACTION6" and target_entity_id == "":
            target_entity_id = self._infer_action6_target_entity_id(kwargs, normalized_obs)
        return GenericActionEnvelope(
            action_name=str(function_name),
            action_family=action_family,
            parameters=kwargs,
            target_entity_id=target_entity_id,
            target_family=target_family,
            native_action=action_dict,
            metadata={
                "kind": action_kind,
                "source": str(action_dict.get("_source", "") or ""),
                "candidate_meta": dict(meta),
            },
        )

    def get_generic_transition(
        self,
        *,
        obs_before: Optional[Dict[str, Any]],
        action: Any,
        result: Any,
    ) -> GenericTransition:
        before = self._coerce_normalized_obs(obs_before)
        after = self._coerce_result_obs(result)
        generic_action = self.get_generic_action(action, obs=before)
        before_perception = before.get("perception", {}) if isinstance(before.get("perception", {}), dict) else {}
        after_perception = after.get("perception", {}) if isinstance(after.get("perception", {}), dict) else {}
        changed_bbox = after_perception.get("changed_bbox", {}) if isinstance(after_perception.get("changed_bbox", {}), dict) else {}
        state_delta = GenericStateDelta(
            entity_deltas=self._generic_entity_deltas(before_perception, after_perception),
            relation_deltas=self._generic_relation_deltas(before_perception, after_perception),
            changed_regions=[dict(changed_bbox)] if changed_bbox else [],
            feedback={
                "changed_pixel_count": int(after_perception.get("changed_pixel_count", 0) or 0),
                "active_pixel_count": int(after_perception.get("active_pixel_count", 0) or 0),
                "schema_failure": bool(
                    isinstance(result, dict)
                    and str(result.get("failure_reason", "") or "").startswith("arc_agi3_schema_failure")
                ),
            },
        )
        reward = 0.0
        terminal = False
        success = False
        raw_result: Dict[str, Any] = {}
        if isinstance(result, ActionResult):
            reward = float(result.raw.get("reward", 0.0) or 0.0)
            terminal = bool(result.raw.get("terminal", False))
            success = bool(result.raw.get("success", False))
            raw_result = dict(result.raw)
        elif isinstance(result, dict):
            reward = float(result.get("reward", 0.0) or 0.0)
            terminal = bool(result.get("terminal", False))
            success = bool(result.get("success", False))
            raw_result = dict(result)
        task_id = str(after.get("resolved_game_id", before.get("resolved_game_id", self._resolved_game_id)) or self._requested_game_id or self.game_id)
        return GenericTransition(
            task_id=task_id,
            environment_family="arc_agi3",
            action=generic_action,
            before_state_signature=self._generic_state_signature(before),
            after_state_signature=self._generic_state_signature(after),
            state_delta=state_delta,
            reward=reward,
            terminal=terminal,
            success=success,
            feedback={
                "guid_before": str(before.get("guid", self._current_guid) or ""),
                "guid_after": str(after.get("guid", self._current_guid) or ""),
                "state_before": str(before.get("state", "") or ""),
                "state_after": str(after.get("state", "") or ""),
            },
            raw=raw_result,
        )

    def _normalize_obs(self, raw: Any) -> Dict[str, Any]:
        obs = self._object_to_dict(raw)
        frames = obs.get("frame", [])
        available_actions = self._extract_available_actions(raw, obs)
        available_names = [self._action_name(a) for a in available_actions]

        function_signatures = {name: {"required": []} for name in available_names if name != "ACTION6"}
        if "ACTION6" in available_names:
            function_signatures["ACTION6"] = {"required": ["x", "y"]}

        perception = self._perception.observe(
            {
                "frame": frames,
                "available_actions": [self._action_id(a) for a in available_actions],
            }
        )

        raw_game_id = str(
            obs.get("game_id")
            or obs.get("id")
            or obs.get("resolved_game_id")
            or self._resolved_game_id
            or self._requested_game_id
        ).strip()
        raw_guid = str(obs.get("guid", "") or "").strip()

        raw_state = str(obs.get("state", "") or "")
        state = self._canonicalize_state(raw_state)
        solved = state == "WIN" or bool(obs.get("solved", False))
        terminal = solved or state in self._TERMINAL_STATES - {"WIN"}
        if not terminal and not available_names:
            scorecard_terminal, scorecard_state = self._infer_terminal_from_scorecard()
            if scorecard_terminal:
                terminal = True
                if scorecard_state and (not state or state == "NOT_FINISHED"):
                    state = scorecard_state
                if scorecard_state == "WIN":
                    solved = True

        normalized = {
            "type": "arc_agi3",
            "requested_game_id": self._requested_game_id,
            "resolved_game_id": raw_game_id,
            "game_id": raw_game_id,
            "guid": raw_guid,
            "frame": frames,
            "state": state,
            "raw_state": raw_state,
            "levels_completed": obs.get("levels_completed"),
            "win_levels": obs.get("win_levels"),
            "available_actions": [self._action_id(a) for a in available_actions],
            "available_action_names": available_names,
            "available_functions": available_names,
            "backend_functions": {name: "available" for name in available_names},
            "function_signatures": function_signatures,
            "novel_api": {
                "visible_functions": available_names,
                "discovered_functions": available_names,
                "available_functions": available_names,
                "frame": frames,
                "state": state,
                "guid": raw_guid,
                "game_id": raw_game_id,
            },
            "world_state": {
                "state": state,
                "task_family": "arc_agi3",
                "active_functions": available_names,
                "levels_completed": obs.get("levels_completed"),
                "win_levels": obs.get("win_levels"),
                "requested_game_id": self._requested_game_id,
                "resolved_game_id": raw_game_id,
                "guid": raw_guid,
            },
            "perception": perception,
            "terminal": terminal,
            "done": terminal,
            "solved": solved,
            "success": solved,
            "text": self._build_text_summary(
                state=state,
                available_names=available_names,
                perception=perception,
                resolved_game_id=raw_game_id,
                guid=raw_guid,
            ),
            "raw_arc_obs": obs,
        }
        return normalized

    def _coerce_normalized_obs(self, obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(obs, dict) and obs:
            return dict(obs)
        if isinstance(self._last_obs, dict) and self._last_obs:
            return dict(self._last_obs)
        return {}

    def _coerce_result_obs(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, ActionResult):
            if isinstance(getattr(result.observation, "raw", None), dict):
                return dict(result.observation.raw)
            if isinstance(result.raw, dict):
                return dict(result.raw)
            return {}
        if isinstance(result, dict):
            if isinstance(result.get("observation", {}), dict):
                return dict(result.get("observation", {}))
            return dict(result)
        return {}

    def _generic_action_descriptors(self, obs: Dict[str, Any]) -> List[GenericActionDescriptor]:
        descriptors: List[GenericActionDescriptor] = []
        for name in list(obs.get("available_functions", []) or []):
            fn_name = str(name or "").strip()
            if not fn_name:
                continue
            descriptors.append(
                GenericActionDescriptor(
                    name=fn_name,
                    action_family=self._generic_action_family(fn_name, "call_tool"),
                    parameter_schema=(
                        {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}
                        if fn_name == "ACTION6"
                        else {"type": "object", "properties": {}}
                    ),
                    enabled=True,
                    source="surface",
                    attributes={"native_name": fn_name},
                )
            )
        return descriptors

    def _generic_entities_from_obs(
        self,
        obs: Dict[str, Any],
        perception: Dict[str, Any],
    ) -> List[GenericEntity]:
        entities: List[GenericEntity] = []
        grid_shape = perception.get("grid_shape", {}) if isinstance(perception.get("grid_shape", {}), dict) else {}
        width = int(grid_shape.get("width", 0) or 0)
        height = int(grid_shape.get("height", 0) or 0)
        entities.append(
            GenericEntity(
                entity_id="grid_root",
                entity_type="grid_world",
                label="ARC grid",
                attributes={
                    "width": width,
                    "height": height,
                    "background_color": perception.get("background_color"),
                    "state": str(obs.get("state", "") or ""),
                },
                bbox_or_region={"x_min": 0, "y_min": 0, "x_max": max(width - 1, 0), "y_max": max(height - 1, 0)},
                provenance={"source": "arc_agi3.perception"},
            )
        )
        changed_bbox = perception.get("changed_bbox", {}) if isinstance(perception.get("changed_bbox", {}), dict) else {}
        if changed_bbox:
            entities.append(
                GenericEntity(
                    entity_id="changed_region",
                    entity_type="state_change_region",
                    label="Changed region",
                    attributes={"changed_pixel_count": int(perception.get("changed_pixel_count", 0) or 0)},
                    bbox_or_region=dict(changed_bbox),
                    provenance={"source": "arc_agi3.perception"},
                )
            )
        for row in list(perception.get("salient_objects", []) or []):
            if not isinstance(row, dict):
                continue
            entity_id = str(row.get("object_id", "") or "").strip()
            if not entity_id:
                continue
            bbox = dict(row.get("bbox", {}) or {}) if isinstance(row.get("bbox", {}), dict) else {}
            centroid = dict(row.get("centroid", {}) or {}) if isinstance(row.get("centroid", {}), dict) else {}
            entities.append(
                GenericEntity(
                    entity_id=entity_id,
                    entity_type="grid_object",
                    label=f"ARC object {entity_id}",
                    attributes={
                        "color": int(row.get("color", 0) or 0),
                        "area": int(row.get("area", 0) or 0),
                        "boundary_contact": bool(row.get("boundary_contact", False)),
                        "goal_like": bool(row.get("goal_like", False)),
                        "changed_overlap": int(row.get("changed_overlap", 0) or 0),
                        "salience_score": float(row.get("salience_score", 0.0) or 0.0),
                        "actionable_score": float(row.get("actionable_score", 0.0) or 0.0),
                        "rarity_score": float(row.get("rarity_score", 0.0) or 0.0),
                        "keepalive_tags": list(row.get("keepalive_tags", []) or []),
                        "centroid": centroid,
                    },
                    bbox_or_region=bbox,
                    provenance={"source": "arc_agi3.perception"},
                )
            )
        return entities

    def _generic_relations_from_obs(
        self,
        obs: Dict[str, Any],
        perception: Dict[str, Any],
        entities: List[GenericEntity],
    ) -> List[GenericRelation]:
        entity_index = {entity.entity_id: entity for entity in entities}
        relations: List[GenericRelation] = []
        salient = [row for row in list(perception.get("salient_objects", []) or []) if isinstance(row, dict)]
        changed_bbox = perception.get("changed_bbox", {}) if isinstance(perception.get("changed_bbox", {}), dict) else {}
        for row in salient:
            entity_id = str(row.get("object_id", "") or "").strip()
            if not entity_id or entity_id not in entity_index:
                continue
            relations.append(
                GenericRelation(
                    relation_id=f"contains:grid_root:{entity_id}",
                    relation_type="contains",
                    source_entity_id="grid_root",
                    target_entity_id=entity_id,
                    attributes={},
                    provenance={"source": "arc_agi3.perception"},
                )
            )
            bbox = dict(row.get("bbox", {}) or {}) if isinstance(row.get("bbox", {}), dict) else {}
            if changed_bbox and self._bbox_overlaps(bbox, changed_bbox):
                relations.append(
                    GenericRelation(
                        relation_id=f"overlaps_changed_region:{entity_id}",
                        relation_type="overlaps_changed_region",
                        source_entity_id=entity_id,
                        target_entity_id="changed_region",
                        attributes={},
                        provenance={"source": "arc_agi3.perception"},
                    )
                )
        for idx, left in enumerate(salient):
            left_id = str(left.get("object_id", "") or "").strip()
            if not left_id:
                continue
            for right in salient[idx + 1:]:
                right_id = str(right.get("object_id", "") or "").strip()
                if not right_id:
                    continue
                left_bbox = dict(left.get("bbox", {}) or {}) if isinstance(left.get("bbox", {}), dict) else {}
                right_bbox = dict(right.get("bbox", {}) or {}) if isinstance(right.get("bbox", {}), dict) else {}
                if int(left.get("color", -1) or -1) == int(right.get("color", -2) or -2):
                    relations.append(
                        GenericRelation(
                            relation_id=f"same_color:{left_id}:{right_id}",
                            relation_type="same_color",
                            source_entity_id=left_id,
                            target_entity_id=right_id,
                            attributes={"color": int(left.get("color", 0) or 0)},
                            provenance={"source": "arc_agi3.perception"},
                        )
                    )
                spatial = self._spatial_relation(left_bbox, right_bbox)
                if spatial:
                    relations.append(
                        GenericRelation(
                            relation_id=f"{spatial}:{left_id}:{right_id}",
                            relation_type=spatial,
                            source_entity_id=left_id,
                            target_entity_id=right_id,
                            attributes={},
                            provenance={"source": "arc_agi3.perception"},
                        )
                    )
        return relations

    @staticmethod
    def _generic_action_family(function_name: str, action_kind: str) -> str:
        fn_name = str(function_name or "").strip().upper()
        kind = str(action_kind or "").strip().lower()
        if fn_name == "ACTION6":
            return "point_interaction"
        if fn_name in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}:
            return "directional_control"
        if fn_name in {"ACTION5", "ACTION7"}:
            return "discrete_control"
        if kind == "wait":
            return "wait"
        if kind == "inspect":
            return "inspection"
        if kind == "probe":
            return "probe"
        return "tool_action"

    def _infer_action6_target_entity_id(self, kwargs: Dict[str, Any], obs: Dict[str, Any]) -> str:
        try:
            x = int(kwargs.get("x"))
            y = int(kwargs.get("y"))
        except Exception:
            return ""
        perception = obs.get("perception", {}) if isinstance(obs.get("perception", {}), dict) else {}
        for row in list(perception.get("salient_objects", []) or []):
            if not isinstance(row, dict):
                continue
            bbox = dict(row.get("bbox", {}) or {}) if isinstance(row.get("bbox", {}), dict) else {}
            if self._point_in_bbox(x, y, bbox):
                return str(row.get("object_id", "") or "")
        return ""

    def _generic_entity_deltas(
        self,
        before_perception: Dict[str, Any],
        after_perception: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        deltas: List[Dict[str, Any]] = []
        before_map = {
            str(row.get("object_id", "") or ""): row
            for row in list(before_perception.get("salient_objects", []) or [])
            if isinstance(row, dict) and str(row.get("object_id", "") or "")
        }
        after_map = {
            str(row.get("object_id", "") or ""): row
            for row in list(after_perception.get("salient_objects", []) or [])
            if isinstance(row, dict) and str(row.get("object_id", "") or "")
        }
        for entity_id, after_row in after_map.items():
            before_row = before_map.get(entity_id, {})
            if not before_row:
                deltas.append({"entity_id": entity_id, "change_type": "added"})
                continue
            change: Dict[str, Any] = {"entity_id": entity_id, "change_type": "updated", "field_changes": {}}
            for field in ("color", "area", "boundary_contact", "goal_like", "changed_overlap"):
                if before_row.get(field) != after_row.get(field):
                    change["field_changes"][field] = {
                        "before": before_row.get(field),
                        "after": after_row.get(field),
                    }
            before_bbox = dict(before_row.get("bbox", {}) or {}) if isinstance(before_row.get("bbox", {}), dict) else {}
            after_bbox = dict(after_row.get("bbox", {}) or {}) if isinstance(after_row.get("bbox", {}), dict) else {}
            if before_bbox != after_bbox:
                change["field_changes"]["bbox"] = {"before": before_bbox, "after": after_bbox}
            if change["field_changes"]:
                deltas.append(change)
        for entity_id in before_map:
            if entity_id not in after_map:
                deltas.append({"entity_id": entity_id, "change_type": "removed"})
        return deltas

    def _generic_relation_deltas(
        self,
        before_perception: Dict[str, Any],
        after_perception: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        def same_color_pairs(perception: Dict[str, Any]) -> set[tuple[str, str]]:
            salient = [row for row in list(perception.get("salient_objects", []) or []) if isinstance(row, dict)]
            pairs: set[tuple[str, str]] = set()
            for idx, left in enumerate(salient):
                left_id = str(left.get("object_id", "") or "").strip()
                if not left_id:
                    continue
                for right in salient[idx + 1:]:
                    right_id = str(right.get("object_id", "") or "").strip()
                    if not right_id:
                        continue
                    if int(left.get("color", -1) or -1) == int(right.get("color", -2) or -2):
                        pairs.add(tuple(sorted((left_id, right_id))))
            return pairs

        before_pairs = same_color_pairs(before_perception)
        after_pairs = same_color_pairs(after_perception)
        deltas: List[Dict[str, Any]] = []
        for pair in sorted(after_pairs - before_pairs):
            deltas.append({"relation_type": "same_color", "change_type": "added", "members": list(pair)})
        for pair in sorted(before_pairs - after_pairs):
            deltas.append({"relation_type": "same_color", "change_type": "removed", "members": list(pair)})
        return deltas

    @staticmethod
    def _generic_state_signature(obs: Dict[str, Any]) -> Dict[str, Any]:
        perception = obs.get("perception", {}) if isinstance(obs.get("perception", {}), dict) else {}
        return {
            "state": str(obs.get("state", "") or ""),
            "guid": str(obs.get("guid", "") or ""),
            "resolved_game_id": str(obs.get("resolved_game_id", "") or ""),
            "changed_pixel_count": int(perception.get("changed_pixel_count", 0) or 0),
            "active_pixel_count": int(perception.get("active_pixel_count", 0) or 0),
            "object_count": len(list(perception.get("salient_objects", []) or [])),
        }

    @staticmethod
    def _bbox_overlaps(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if not left or not right:
            return False
        return not (
            int(left.get("x_max", 0) or 0) < int(right.get("x_min", 0) or 0)
            or int(right.get("x_max", 0) or 0) < int(left.get("x_min", 0) or 0)
            or int(left.get("y_max", 0) or 0) < int(right.get("y_min", 0) or 0)
            or int(right.get("y_max", 0) or 0) < int(left.get("y_min", 0) or 0)
        )

    @staticmethod
    def _point_in_bbox(x: int, y: int, bbox: Dict[str, Any]) -> bool:
        if not bbox:
            return False
        return (
            int(bbox.get("x_min", 0) or 0) <= int(x) <= int(bbox.get("x_max", 0) or 0)
            and int(bbox.get("y_min", 0) or 0) <= int(y) <= int(bbox.get("y_max", 0) or 0)
        )

    @staticmethod
    def _spatial_relation(left_bbox: Dict[str, Any], right_bbox: Dict[str, Any]) -> str:
        if not left_bbox or not right_bbox:
            return ""
        if int(left_bbox.get("x_max", 0) or 0) < int(right_bbox.get("x_min", 0) or 0):
            return "left_of"
        if int(right_bbox.get("x_max", 0) or 0) < int(left_bbox.get("x_min", 0) or 0):
            return "right_of"
        if int(left_bbox.get("y_max", 0) or 0) < int(right_bbox.get("y_min", 0) or 0):
            return "above"
        if int(right_bbox.get("y_max", 0) or 0) < int(left_bbox.get("y_min", 0) or 0):
            return "below"
        return ""

    def _infer_terminal_from_scorecard(self) -> tuple[bool, str]:
        scorecard = self.scorecard()
        if not isinstance(scorecard, dict):
            return False, ""

        environments = scorecard.get("environments", [])
        if not isinstance(environments, list):
            return False, ""

        terminal_state = ""
        for environment in reversed(environments):
            if not isinstance(environment, dict):
                continue
            runs = environment.get("runs", [])
            if isinstance(runs, list):
                for run in reversed(runs):
                    if not isinstance(run, dict):
                        continue
                    state = self._canonicalize_state(run.get("state", ""))
                    completed = bool(run.get("completed", False))
                    if completed or state in self._TERMINAL_STATES:
                        return True, state
                    if not terminal_state and state:
                        terminal_state = state
            state = self._canonicalize_state(environment.get("state", ""))
            completed = bool(environment.get("completed", False))
            if completed or state in self._TERMINAL_STATES:
                return True, state
            if not terminal_state and state:
                terminal_state = state
        return False, terminal_state

    @staticmethod
    def _canonicalize_state(state: Any) -> str:
        raw = str(state or "").strip()
        if not raw:
            return ""
        return raw.split(".")[-1].upper()

    def _to_surface_observation(self, obs: Dict[str, Any]) -> SurfaceObservation:
        tools = [
            ToolSpec(
                name=name,
                description=f"ARC-AGI-3 action {name}",
                input_schema=(
                    {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}
                    if name == "ACTION6"
                    else {"type": "object", "properties": {}}
                ),
                capability_class=(
                    "pointer_interaction"
                    if name == "ACTION6"
                    else ("grid_interaction" if name == "ACTION5" else ("state_reversal" if name == "ACTION7" else "grid_navigation"))
                ),
                side_effect_class=(
                    "environment_interaction"
                    if name in {"ACTION5", "ACTION6"}
                    else "environment_navigation"
                ),
                risk_level="medium" if name in {"ACTION5", "ACTION6"} else "low",
            )
            for name in obs.get("available_functions", [])
        ]
        return SurfaceObservation(
            text=str(obs.get("text", "")),
            structured={
                "requested_game_id": self._requested_game_id,
                "resolved_game_id": obs.get("resolved_game_id", self._resolved_game_id),
                "guid": obs.get("guid", self._current_guid),
                "state": obs.get("state"),
                "available_functions": list(obs.get("available_functions", [])),
                "levels_completed": obs.get("levels_completed"),
                "win_levels": obs.get("win_levels"),
            },
            available_tools=tools,
            terminal=bool(obs.get("terminal", False)),
            reward=float(obs.get("reward", 0.0) or 0.0) if obs.get("reward") is not None else None,
            raw=obs,
        )

    def _available_action_names(self) -> List[str]:
        actions = self._last_obs.get("available_action_names", [])
        return list(actions) if isinstance(actions, list) else []

    def _build_text_summary(
        self,
        *,
        state: str,
        available_names: List[str],
        perception: Dict[str, Any],
        resolved_game_id: str,
        guid: str,
    ) -> str:
        shape = perception.get("grid_shape", {}) if isinstance(perception.get("grid_shape", {}), dict) else {}
        return (
            f"ARC-AGI-3 game {resolved_game_id or self._requested_game_id}. "
            f"State={state or 'UNKNOWN'}. "
            f"Guid={'set' if guid else 'missing'}. "
            f"Grid={shape.get('width', 0)}x{shape.get('height', 0)}. "
            f"Available actions={','.join(available_names) if available_names else 'none'}."
        )

    def _infer_reward(self, normalized: Dict[str, Any]) -> float:
        raw = normalized.get("raw_arc_obs", {}) if isinstance(normalized.get("raw_arc_obs", {}), dict) else {}
        for key in ("reward", "score_delta"):
            value = raw.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        if bool(normalized.get("solved", False)):
            return 1.0
        return 0.0

    @staticmethod
    def _resolve_operation_mode(name: str) -> Any:
        if OperationMode is None:
            return None
        mode = str(name or "ONLINE").strip().upper()
        if hasattr(OperationMode, mode):
            return getattr(OperationMode, mode)
        return getattr(OperationMode, "ONLINE")

    @staticmethod
    def _extract_available_actions(raw_obj: Any, obs: Dict[str, Any]) -> List[Any]:
        if hasattr(raw_obj, "action_space"):
            try:
                actions = list(raw_obj.action_space)
                if actions:
                    return actions
            except Exception:
                pass
        env_actions = obs.get("available_actions", [])
        return list(env_actions) if isinstance(env_actions, list) else []

    @staticmethod
    def _action_name(action: Any) -> str:
        if hasattr(action, "name"):
            return str(action.name)
        try:
            value = int(action)
        except (TypeError, ValueError):
            return str(action)
        return f"ACTION{value}"

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            try:
                return int(action.value)
            except Exception:
                pass
        try:
            return int(action)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _object_to_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return dict(obj)
        if obj is None:
            return {}

        data: Dict[str, Any] = {}
        if hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                if not str(key).startswith("_"):
                    data[key] = value

        # Explicitly materialize common public properties even when they are
        # backed by private attributes like `_frame`.
        for attr_name in (
            "frame",
            "frames",
            "state",
            "guid",
            "game_id",
            "id",
            "resolved_game_id",
            "levels_completed",
            "win_levels",
            "available_actions",
            "action_space",
            "reward",
            "score_delta",
            "solved",
            "success",
        ):
            if attr_name in data:
                continue
            if not hasattr(obj, attr_name):
                continue
            try:
                value = getattr(obj, attr_name)
            except Exception:
                continue
            if callable(value):
                continue
            data[attr_name] = value

        # Property-backed frame fallback from private storage.
        if "frame" not in data:
            for private_name in ("_frame", "_frames"):
                if hasattr(obj, private_name):
                    try:
                        value = getattr(obj, private_name)
                    except Exception:
                        continue
                    if value is not None:
                        data["frame"] = value
                        break

        if data:
            return data
        return {"raw": obj}
    _TERMINAL_STATES = {"WIN", "LOSE", "GAME_OVER", "DONE"}
