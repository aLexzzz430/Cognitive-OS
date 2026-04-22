from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
from integrations.webarena.action_adapter import WebArenaActionAdapter, WebArenaActionChoice
from integrations.webarena.state_bridge import WebArenaSessionState

try:
    from browser_env import ScriptBrowserEnv, create_id_based_action
except Exception as exc:  # pragma: no cover
    ScriptBrowserEnv = None
    create_id_based_action = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class WebArenaSurfaceAdapter:
    """
    Thin WebArena adapter that implements both the current SurfaceAdapter
    contract and the new generic environment protocol.

    Real WebArena dependencies remain optional. Tests can pass a fake env
    exposing Gym-like reset/step semantics.
    """

    def __init__(
        self,
        *,
        env: Any | None = None,
        config_file: str | None = None,
        task_id: str = "",
        instruction: str = "",
        headless: bool = True,
        observation_type: str = "accessibility_tree",
        current_viewport_only: bool = True,
        viewport_size: Optional[Dict[str, int]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if env is None and ScriptBrowserEnv is None:
            raise RuntimeError(
                "WebArena dependencies are not installed. "
                "Install browser_env / WebArena or pass a custom env."
            ) from _IMPORT_ERROR
        self.logger = logger or logging.getLogger("webarena_adapter")
        self.config_file = str(config_file or "")
        self._explicit_task_id = str(task_id or "")
        self._explicit_instruction = str(instruction or "")
        self._viewport_size = dict(viewport_size or {"width": 1280, "height": 720})
        self._env = env or ScriptBrowserEnv(
            headless=headless,
            observation_type=observation_type,
            current_viewport_only=current_viewport_only,
            viewport_size=self._viewport_size,
        )
        self._action_adapter = WebArenaActionAdapter(action_factory=create_id_based_action)
        self._state = WebArenaSessionState(task_id=self._explicit_task_id, task_instruction=self._explicit_instruction)
        self._last_obs: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}

    def reset(self, seed: int | None = None, episode: int | None = None, prior_discoveries: list | None = None) -> SurfaceObservation:
        reset_options: Dict[str, Any] = {}
        if self.config_file:
            reset_options["config_file"] = self.config_file
        if seed is not None:
            reset_options["seed"] = int(seed)
        obs, info = self._call_reset(reset_options)
        normalized = self._normalize_obs_info(obs, info)
        self._last_obs = normalized
        self._last_info = dict(normalized.get("metadata", {}) or {})
        self._state.reset_for_episode(
            task_id=str(normalized.get("metadata", {}).get("task_id", "") or self._explicit_task_id),
            task_instruction=str(normalized.get("metadata", {}).get("instruction", "") or self._explicit_instruction),
            episode_index=int(episode) if episode is not None else None,
        )
        self._state.record_observation(normalized)
        return self._to_surface_observation(normalized)

    def observe(self) -> SurfaceObservation:
        if not self._last_obs:
            return self.reset()
        return self._to_surface_observation(self._last_obs)

    def act(self, action: Any) -> ActionResult:
        action_dict = dict(action) if isinstance(action, dict) else {"kind": str(action or "")}
        available_action_names = [descriptor.name for descriptor in self.get_generic_observation().available_actions]
        try:
            choice = self._action_adapter.to_webarena_action(action_dict, available_action_names=available_action_names)
        except Exception as exc:
            return ActionResult(
                ok=False,
                observation=self.observe(),
                raw={
                    "success": False,
                    "reward": 0.0,
                    "terminal": False,
                    "done": False,
                    "failure_reason": str(exc),
                    "state": "ADAPTER_ERROR",
                    "action_input": action_dict,
                },
                events=[{"type": "adapter_error", "reason": str(exc)}],
            )

        if not choice.requires_env_step:
            raw = {
                "success": True,
                "reward": 0.0,
                "terminal": bool(self._last_obs.get("terminal", False)),
                "done": bool(self._last_obs.get("terminal", False)),
                "state": "INTERNAL_NOOP",
                "internal_action": str(choice.internal_action_kind or "internal_noop"),
            }
            return ActionResult(ok=True, observation=self.observe(), raw=raw, events=[])

        try:
            obs, reward, terminated, truncated, info = self._call_step(choice.native_action)
        except Exception as exc:
            return ActionResult(
                ok=False,
                observation=self.observe(),
                raw={
                    "success": False,
                    "reward": 0.0,
                    "terminal": False,
                    "done": False,
                    "failure_reason": str(exc),
                    "state": "STEP_ERROR",
                    "action_input": action_dict,
                    "native_action": choice.native_action,
                },
                events=[{"type": "step_error", "reason": str(exc)}],
            )

        normalized = self._normalize_obs_info(obs, info)
        normalized["reward"] = float(reward or 0.0)
        normalized["terminal"] = bool(terminated or truncated or normalized.get("terminal", False))
        normalized["done"] = bool(normalized["terminal"])
        self._last_obs = normalized
        self._last_info = dict(normalized.get("metadata", {}) or {})
        raw_result = {
            "success": True,
            "reward": float(reward or 0.0),
            "terminal": bool(normalized.get("terminal", False)),
            "done": bool(normalized.get("done", False)),
            "state": str(normalized.get("state", "") or ""),
            "url": str((normalized.get("page", {}) if isinstance(normalized.get("page", {}), dict) else {}).get("url", "") or ""),
            "native_action": choice.native_action,
            "action_input": action_dict,
        }
        self._state.record_action({"action_name": choice.action_name, "native_action": choice.native_action}, raw_result, float(reward or 0.0))
        self._state.record_observation(normalized)
        return ActionResult(ok=True, observation=self._to_surface_observation(normalized), raw=raw_result, events=[])

    def next_episode(self) -> None:
        self.reset(episode=self._state.episode_index + 1)

    def get_generic_task_spec(self) -> GenericTaskSpec:
        obs = self._last_obs or {}
        meta = obs.get("metadata", {}) if isinstance(obs.get("metadata", {}), dict) else {}
        page = obs.get("page", {}) if isinstance(obs.get("page", {}), dict) else {}
        available = [descriptor.name for descriptor in self.get_generic_observation(obs).available_actions]
        return GenericTaskSpec(
            task_id=str(meta.get("task_id", "") or self._explicit_task_id or self.config_file or "webarena_task"),
            environment_family="webarena",
            instruction=str(meta.get("instruction", "") or self._explicit_instruction),
            success_criteria=list(meta.get("success_criteria", []) or []),
            available_action_names=available,
            metadata={
                "config_file": self.config_file,
                "url": str(page.get("url", "") or ""),
                "title": str(page.get("title", "") or ""),
            },
        )

    def get_generic_observation(self, obs: Optional[Dict[str, Any]] = None) -> GenericObservation:
        normalized = self._coerce_obs(obs)
        page = normalized.get("page", {}) if isinstance(normalized.get("page", {}), dict) else {}
        meta = normalized.get("metadata", {}) if isinstance(normalized.get("metadata", {}), dict) else {}
        interactive = self._interactive_elements(normalized)
        entities = [
            GenericEntity(
                entity_id="page_root",
                entity_type="web_page",
                label=str(page.get("title", "") or "Page"),
                attributes={
                    "url": str(page.get("url", "") or ""),
                    "title": str(page.get("title", "") or ""),
                    "active_tab_id": str(page.get("active_tab_id", "") or ""),
                    "terminal": bool(normalized.get("terminal", False)),
                },
                bbox_or_region={},
                provenance={"source": "webarena.page"},
            )
        ]
        relations: List[GenericRelation] = []
        for row in interactive:
            entity_id = str(row.get("element_id", "") or row.get("id", "") or "").strip()
            if not entity_id:
                continue
            role = str(row.get("role", row.get("tag", "interactive_element")) or "interactive_element")
            bbox = dict(row.get("bbox", {}) or {}) if isinstance(row.get("bbox", {}), dict) else {}
            entities.append(
                GenericEntity(
                    entity_id=entity_id,
                    entity_type="web_element",
                    label=str(row.get("text", row.get("label", row.get("name", entity_id))) or entity_id),
                    attributes={
                        "role": role,
                        "text": str(row.get("text", "") or ""),
                        "enabled": bool(row.get("enabled", True)),
                        "visible": bool(row.get("visible", True)),
                        "value": row.get("value"),
                        "selectable": bool(row.get("selectable", False)),
                        "typable": bool(row.get("typable", False)),
                        "clickable": bool(row.get("clickable", False)),
                    },
                    bbox_or_region=bbox,
                    provenance={"source": "webarena.interactive_element"},
                    extensions={k: v for k, v in row.items() if k not in {"element_id", "id", "text", "label", "name", "role", "tag", "bbox", "enabled", "visible", "value", "selectable", "typable", "clickable"}},
                )
            )
            relations.append(
                GenericRelation(
                    relation_id=f"contains:page_root:{entity_id}",
                    relation_type="contains",
                    source_entity_id="page_root",
                    target_entity_id=entity_id,
                    attributes={},
                    provenance={"source": "webarena.interactive_element"},
                )
            )
            parent_id = str(row.get("parent_id", "") or "").strip()
            if parent_id:
                relations.append(
                    GenericRelation(
                        relation_id=f"parent_of:{parent_id}:{entity_id}",
                        relation_type="parent_of",
                        source_entity_id=parent_id,
                        target_entity_id=entity_id,
                        attributes={},
                        provenance={"source": "webarena.interactive_element"},
                    )
                )
            controls = str(row.get("controls", "") or "").strip()
            if controls:
                relations.append(
                    GenericRelation(
                        relation_id=f"controls:{entity_id}:{controls}",
                        relation_type="controls",
                        source_entity_id=entity_id,
                        target_entity_id=controls,
                        attributes={},
                        provenance={"source": "webarena.interactive_element"},
                    )
                )
        available_actions = self._available_action_descriptors(interactive, normalized)
        task_id = str(meta.get("task_id", "") or self._explicit_task_id or self.config_file or "webarena_task")
        return GenericObservation(
            observation_id=f"{task_id}:{self._state.turn_index}",
            environment_family="webarena",
            task_id=task_id,
            text=str(normalized.get("text", "") or ""),
            entities=entities,
            relations=relations,
            available_actions=available_actions,
            state={
                "url": str(page.get("url", "") or ""),
                "title": str(page.get("title", "") or ""),
                "active_tab_id": str(page.get("active_tab_id", "") or ""),
                "terminal": bool(normalized.get("terminal", False)),
                "task_id": task_id,
            },
            metadata=meta,
            raw=normalized,
        )

    def get_generic_action(self, action: Any, *, obs: Optional[Dict[str, Any]] = None) -> GenericActionEnvelope:
        normalized = self._coerce_obs(obs)
        action_dict = dict(action) if isinstance(action, dict) else {"kind": str(action or "")}
        payload = action_dict.get("payload", {}) if isinstance(action_dict.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        kwargs = dict(tool_args.get("kwargs", {}) or {}) if isinstance(tool_args.get("kwargs", {}), dict) else {}
        function_name = self._action_adapter._canonical_action_name(action_dict)
        target_entity_id = str(kwargs.get("element_id", kwargs.get("id", kwargs.get("element_ref", ""))) or "").strip()
        target_family = self._infer_target_family(target_entity_id, normalized)
        return GenericActionEnvelope(
            action_name=function_name,
            action_family=self._generic_action_family(function_name),
            parameters=kwargs,
            target_entity_id=target_entity_id,
            target_family=target_family,
            native_action=action_dict,
            metadata={
                "kind": extract_action_kind(action_dict, default="call_tool"),
                "source": str(action_dict.get("_source", "") or ""),
            },
        )

    def get_generic_transition(
        self,
        *,
        obs_before: Optional[Dict[str, Any]],
        action: Any,
        result: Any,
    ) -> GenericTransition:
        before = self._coerce_obs(obs_before)
        after = self._result_to_obs(result)
        generic_action = self.get_generic_action(action, obs=before)
        before_index = {row["entity_id"]: row for row in self._entity_rows(before)}
        after_index = {row["entity_id"]: row for row in self._entity_rows(after)}
        entity_deltas: List[Dict[str, Any]] = []
        for entity_id, after_row in after_index.items():
            before_row = before_index.get(entity_id)
            if before_row is None:
                entity_deltas.append({"entity_id": entity_id, "change_type": "added"})
                continue
            field_changes: Dict[str, Any] = {}
            for field in ("text", "value", "enabled", "visible", "role", "url"):
                if before_row.get(field) != after_row.get(field):
                    field_changes[field] = {"before": before_row.get(field), "after": after_row.get(field)}
            if before_row.get("bbox") != after_row.get("bbox"):
                field_changes["bbox"] = {"before": before_row.get("bbox"), "after": after_row.get("bbox")}
            if field_changes:
                entity_deltas.append({"entity_id": entity_id, "change_type": "updated", "field_changes": field_changes})
        for entity_id in before_index:
            if entity_id not in after_index:
                entity_deltas.append({"entity_id": entity_id, "change_type": "removed"})

        before_rel = self._relation_rows(before)
        after_rel = self._relation_rows(after)
        relation_deltas: List[Dict[str, Any]] = []
        for rel in sorted(after_rel - before_rel):
            relation_deltas.append({"change_type": "added", "relation_id": rel})
        for rel in sorted(before_rel - after_rel):
            relation_deltas.append({"change_type": "removed", "relation_id": rel})

        raw = dict(result.raw) if isinstance(result, ActionResult) else dict(result) if isinstance(result, dict) else {}
        task_id = str((after.get("metadata", {}) if isinstance(after.get("metadata", {}), dict) else {}).get("task_id", "") or self._explicit_task_id or self.config_file or "webarena_task")
        return GenericTransition(
            task_id=task_id,
            environment_family="webarena",
            action=generic_action,
            before_state_signature=self._state_signature(before),
            after_state_signature=self._state_signature(after),
            state_delta=GenericStateDelta(
                entity_deltas=entity_deltas,
                relation_deltas=relation_deltas,
                changed_regions=self._changed_regions(before, after),
                feedback={
                    "schema_failure": bool(str(raw.get("failure_reason", "") or "").startswith("webarena_schema_failure")),
                    "url_changed": self._state_signature(before).get("url") != self._state_signature(after).get("url"),
                },
            ),
            reward=float(raw.get("reward", 0.0) or 0.0),
            terminal=bool(raw.get("terminal", False) or raw.get("done", False)),
            success=bool(raw.get("success", False)),
            feedback={
                "state_before": self._state_signature(before),
                "state_after": self._state_signature(after),
            },
            raw=raw,
        )

    def _call_reset(self, options: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raw = self._env.reset(options=options) if options else self._env.reset()
        if isinstance(raw, tuple) and len(raw) == 2:
            return raw[0], dict(raw[1] or {})
        return raw, {}

    def _call_step(self, native_action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        raw = self._env.step(native_action)
        if isinstance(raw, tuple) and len(raw) == 5:
            obs, reward, terminated, truncated, info = raw
            return obs, float(reward or 0.0), bool(terminated), bool(truncated), dict(info or {})
        if isinstance(raw, tuple) and len(raw) == 4:
            obs, reward, terminated, info = raw
            return obs, float(reward or 0.0), bool(terminated), False, dict(info or {})
        raise RuntimeError("Unsupported WebArena step response shape")

    def _normalize_obs_info(self, obs: Any, info: Dict[str, Any]) -> Dict[str, Any]:
        obs_dict = dict(obs) if isinstance(obs, dict) else {"text": str(obs or "")}
        info_dict = dict(info or {})
        page = {
            "url": str(info_dict.get("url", obs_dict.get("url", "")) or ""),
            "title": str(info_dict.get("title", obs_dict.get("title", "")) or ""),
            "active_tab_id": str(info_dict.get("active_tab_id", obs_dict.get("active_tab_id", "")) or ""),
        }
        metadata = {
            "task_id": str(info_dict.get("task_id", obs_dict.get("task_id", "")) or self._explicit_task_id),
            "instruction": str(info_dict.get("instruction", obs_dict.get("instruction", "")) or self._explicit_instruction),
            "success_criteria": list(info_dict.get("success_criteria", obs_dict.get("success_criteria", [])) or []),
        }
        normalized = {
            "text": str(obs_dict.get("text", info_dict.get("text", "")) or ""),
            "page": page,
            "interactive_elements": self._coerce_element_rows(
                info_dict.get("interactive_elements", obs_dict.get("interactive_elements", obs_dict.get("dom_elements", [])))
            ),
            "terminal": bool(obs_dict.get("terminal", info_dict.get("terminal", False))),
            "metadata": metadata,
            "raw_obs": obs_dict,
            "raw_info": info_dict,
            "state": "TERMINAL" if bool(obs_dict.get("terminal", info_dict.get("terminal", False))) else "ACTIVE",
        }
        return normalized

    def _to_surface_observation(self, obs: Dict[str, Any]) -> SurfaceObservation:
        tools = [
            ToolSpec(
                name=descriptor.name,
                description=f"WebArena action {descriptor.name}",
                input_schema=dict(descriptor.parameter_schema),
                capability_class=str(descriptor.action_family or descriptor.name),
                side_effect_class=(
                    "external_submission"
                    if descriptor.name == "submit"
                    else ("external_navigation" if descriptor.action_family == "navigation" else "page_interaction")
                ),
                approval_required=bool(descriptor.name == "submit"),
                risk_level=(
                    "high"
                    if descriptor.name == "submit"
                    else ("medium" if descriptor.action_family in {"navigation", "form_submission", "text_entry"} else "low")
                ),
                risk_notes=(["requires explicit approval before irreversible form submission"] if descriptor.name == "submit" else []),
            )
            for descriptor in self._available_action_descriptors(self._interactive_elements(obs), obs)
        ]
        page = obs.get("page", {}) if isinstance(obs.get("page", {}), dict) else {}
        return SurfaceObservation(
            text=str(obs.get("text", "") or ""),
            structured={
                "task_id": str((obs.get("metadata", {}) if isinstance(obs.get("metadata", {}), dict) else {}).get("task_id", "") or ""),
                "instruction": str((obs.get("metadata", {}) if isinstance(obs.get("metadata", {}), dict) else {}).get("instruction", "") or ""),
                "url": str(page.get("url", "") or ""),
                "title": str(page.get("title", "") or ""),
                "active_tab_id": str(page.get("active_tab_id", "") or ""),
            },
            available_tools=tools,
            terminal=bool(obs.get("terminal", False)),
            reward=float(obs.get("reward", 0.0) or 0.0) if obs.get("reward") is not None else None,
            raw=obs,
        )

    def _interactive_elements(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = obs.get("interactive_elements", []) if isinstance(obs.get("interactive_elements", []), list) else []
        return [dict(row) for row in rows if isinstance(row, dict)]

    @staticmethod
    def _coerce_element_rows(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        return [dict(row) for row in value if isinstance(row, dict)]

    def _available_action_descriptors(self, interactive: List[Dict[str, Any]], obs: Dict[str, Any]) -> List[GenericActionDescriptor]:
        available: Dict[str, GenericActionDescriptor] = {}
        def add(name: str, family: str, schema: Dict[str, Any]) -> None:
            if name not in available:
                available[name] = GenericActionDescriptor(name=name, action_family=family, parameter_schema=schema)

        for row in interactive:
            element_keys = {"type": "object", "properties": {"element_id": {"type": "string"}}, "required": ["element_id"]}
            if bool(row.get("clickable", False)):
                add("click", "pointer_interaction", element_keys)
                if str(row.get("role", "") or "").strip().lower() == "button":
                    add("submit", "form_submission", element_keys)
            if bool(row.get("typable", False)):
                schema = dict(element_keys)
                schema["properties"] = dict(schema["properties"])
                schema["properties"]["text"] = {"type": "string"}
                schema["required"] = ["element_id", "text"]
                add("type", "text_entry", schema)
            if bool(row.get("selectable", False)):
                schema = dict(element_keys)
                schema["properties"] = dict(schema["properties"])
                schema["properties"]["option"] = {"type": "string"}
                schema["required"] = ["element_id", "option"]
                add("select", "selection", schema)
            add("hover", "pointer_interaction", element_keys)
        add("scroll", "viewport_control", {"type": "object", "properties": {"direction": {"type": "string"}, "amount": {"type": "integer"}}})
        add("press", "keyboard_control", {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]})
        add("go_back", "navigation", {"type": "object", "properties": {}})
        add("goto", "navigation", {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})
        add("new_tab", "navigation", {"type": "object", "properties": {"url": {"type": "string"}}})
        add("tab_focus", "navigation", {"type": "object", "properties": {"tab_id": {"type": "string"}}, "required": ["tab_id"]})
        return list(available.values())

    def _infer_target_family(self, target_entity_id: str, obs: Dict[str, Any]) -> str:
        if not target_entity_id:
            return ""
        for row in self._interactive_elements(obs):
            entity_id = str(row.get("element_id", row.get("id", "")) or "").strip()
            if entity_id == target_entity_id:
                return str(row.get("role", row.get("tag", "web_element")) or "web_element")
        return ""

    def _entity_rows(self, obs: Dict[str, Any]) -> List[Dict[str, Any]]:
        page = obs.get("page", {}) if isinstance(obs.get("page", {}), dict) else {}
        rows = [{
            "entity_id": "page_root",
            "text": str(page.get("title", "") or ""),
            "value": None,
            "enabled": True,
            "visible": True,
            "role": "page",
            "bbox": {},
            "url": str(page.get("url", "") or ""),
        }]
        for row in self._interactive_elements(obs):
            rows.append(
                {
                    "entity_id": str(row.get("element_id", row.get("id", "")) or ""),
                    "text": str(row.get("text", row.get("label", row.get("name", ""))) or ""),
                    "value": row.get("value"),
                    "enabled": bool(row.get("enabled", True)),
                    "visible": bool(row.get("visible", True)),
                    "role": str(row.get("role", row.get("tag", "web_element")) or "web_element"),
                    "bbox": dict(row.get("bbox", {}) or {}) if isinstance(row.get("bbox", {}), dict) else {},
                    "url": "",
                }
            )
        return [row for row in rows if row.get("entity_id")]

    def _relation_rows(self, obs: Dict[str, Any]) -> set[str]:
        rows: set[str] = set()
        for row in self._interactive_elements(obs):
            entity_id = str(row.get("element_id", row.get("id", "")) or "").strip()
            if not entity_id:
                continue
            rows.add(f"contains:page_root:{entity_id}")
            parent_id = str(row.get("parent_id", "") or "").strip()
            if parent_id:
                rows.add(f"parent_of:{parent_id}:{entity_id}")
            controls = str(row.get("controls", "") or "").strip()
            if controls:
                rows.add(f"controls:{entity_id}:{controls}")
        return rows

    def _state_signature(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        page = obs.get("page", {}) if isinstance(obs.get("page", {}), dict) else {}
        return {
            "url": str(page.get("url", "") or ""),
            "title": str(page.get("title", "") or ""),
            "active_tab_id": str(page.get("active_tab_id", "") or ""),
            "interactive_count": len(self._interactive_elements(obs)),
            "terminal": bool(obs.get("terminal", False)),
        }

    def _changed_regions(self, before: Dict[str, Any], after: Dict[str, Any]) -> List[Dict[str, Any]]:
        before_boxes = {
            row["entity_id"]: row.get("bbox", {})
            for row in self._entity_rows(before)
            if row["entity_id"] != "page_root" and row.get("bbox")
        }
        after_boxes = {
            row["entity_id"]: row.get("bbox", {})
            for row in self._entity_rows(after)
            if row["entity_id"] != "page_root" and row.get("bbox")
        }
        changed: List[Dict[str, Any]] = []
        for entity_id, bbox in after_boxes.items():
            if before_boxes.get(entity_id) != bbox:
                changed.append(dict(bbox))
        for entity_id, bbox in before_boxes.items():
            if entity_id not in after_boxes:
                changed.append(dict(bbox))
        return changed

    def _result_to_obs(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, ActionResult):
            if isinstance(result.observation.raw, dict):
                return dict(result.observation.raw)
            return {}
        if isinstance(result, dict):
            if isinstance(result.get("observation", {}), dict):
                return dict(result.get("observation", {}))
            return dict(result)
        return {}

    def _coerce_obs(self, obs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(obs, dict) and obs:
            return dict(obs)
        if isinstance(self._last_obs, dict) and self._last_obs:
            return dict(self._last_obs)
        return {
            "text": "",
            "page": {},
            "interactive_elements": [],
            "metadata": {},
            "terminal": False,
            "state": "ACTIVE",
        }

    @staticmethod
    def _generic_action_family(action_name: str) -> str:
        name = str(action_name or "").strip().lower()
        if name in {"click", "hover"}:
            return "pointer_interaction"
        if name == "type":
            return "text_entry"
        if name == "select":
            return "selection"
        if name in {"go_back", "goto", "new_tab", "tab_focus"}:
            return "navigation"
        if name == "scroll":
            return "viewport_control"
        if name == "press":
            return "keyboard_control"
        if name == "submit":
            return "form_submission"
        if name == "wait":
            return "wait"
        return "tool_action"
