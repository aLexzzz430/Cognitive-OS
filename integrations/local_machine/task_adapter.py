from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shlex
from typing import Any, Dict, Iterable, Optional, Sequence

from core.environment.types import (
    GenericActionDescriptor,
    GenericObservation,
    GenericTaskSpec,
)
from core.surfaces.base import ActionResult, SurfaceObservation, ToolSpec
from modules.local_mirror.mirror import (
    DEFAULT_ALLOWED_COMMANDS,
    MirrorScopeError,
    acquire_relevant_files,
    apply_sync_plan,
    build_sync_plan,
    compute_mirror_diff,
    create_empty_mirror,
    materialize_files,
    open_mirror,
    run_mirror_command,
)


LOCAL_MACHINE_ADAPTER_VERSION = "conos.local_machine_adapter/v1"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


class LocalMachineSurfaceAdapter:
    """
    CoreMainLoop adapter for a local-machine mirror.

    The adapter presents the host machine through an empty-first workspace.
    Source files are only materialized by explicit tool calls, command execution
    runs from the mirror workspace, and source writes require an approved sync
    plan.
    """

    def __init__(
        self,
        *,
        instruction: str = "",
        source_root: str | Path = ".",
        mirror_root: str | Path = "runtime/mirrors/local-machine",
        candidate_paths: Optional[Iterable[str | Path]] = None,
        fetch_paths: Optional[Iterable[str | Path]] = None,
        default_command: Optional[Sequence[str] | str] = None,
        allowed_commands: Optional[Iterable[str]] = None,
        reset_mirror: bool = False,
        terminal_after_plan: bool = True,
        expose_apply_tool: bool = False,
        allow_empty_exec: bool = False,
        default_command_timeout_seconds: int = 30,
        task_id: str = "local_machine",
    ) -> None:
        self.instruction = str(instruction or "")
        self.source_root = Path(source_root).resolve()
        self.mirror_root = Path(mirror_root).resolve()
        if self.mirror_root == self.source_root or self.source_root.is_relative_to(self.mirror_root):
            raise MirrorScopeError("mirror_root must not be the source root or a parent of the source root")
        self.candidate_paths = [str(path) for path in list(candidate_paths or [])]
        self.fetch_paths = [str(path) for path in list(fetch_paths or [])]
        self.default_command = self._normalize_command(default_command)
        self.allowed_commands = [str(item) for item in list(allowed_commands or DEFAULT_ALLOWED_COMMANDS)]
        self.reset_mirror_on_first_observe = bool(reset_mirror)
        self.terminal_after_plan = bool(terminal_after_plan)
        self.expose_apply_tool = bool(expose_apply_tool)
        self.allow_empty_exec = bool(allow_empty_exec)
        self.default_command_timeout_seconds = max(1, int(default_command_timeout_seconds or 30))
        self.task_id = str(task_id or "local_machine")

        self._initialized = False
        self._terminal = False
        self._last_plan: Dict[str, Any] = {}
        self._last_action: Dict[str, Any] = {}
        self._command_executed = False
        self._last_command_returncode: int | None = None
        self._command_failed = False
        self._acquire_attempted = False
        self._last_acquire_selected_count = -1
        self._applied = False
        self._episode = 0

    @staticmethod
    def _normalize_command(command: Optional[Sequence[str] | str]) -> list[str]:
        if command is None:
            return []
        if isinstance(command, str):
            return shlex.split(command)
        return [str(part) for part in list(command) if str(part)]

    def reset(self, seed: int | None = None, episode: int | None = None, **_: Any) -> SurfaceObservation:
        self._episode = int(episode or self._episode or 1)
        self._terminal = False
        self._last_action = {}
        self._last_plan = {}
        self._command_executed = False
        self._last_command_returncode = None
        self._command_failed = False
        self._acquire_attempted = False
        self._last_acquire_selected_count = -1
        self._applied = False
        create_empty_mirror(
            self.source_root,
            self.mirror_root,
            reset=bool(self.reset_mirror_on_first_observe),
        )
        self._initialized = True
        return self.observe()

    def next_episode(self) -> None:
        self.reset(episode=self._episode + 1)

    def observe(self) -> SurfaceObservation:
        if not self._initialized:
            return self.reset()
        mirror = open_mirror(self.source_root, self.mirror_root)
        manifest = mirror.to_manifest()
        sync_plan = _load_json(mirror.sync_plan_path)
        if sync_plan:
            self._last_plan = sync_plan
        tools = self._available_tools(manifest, sync_plan)
        raw = self._observation_payload(manifest, sync_plan, tools)
        text = (
            f"Local mirror workspace contains {manifest.get('workspace_file_count', 0)} file(s). "
            f"Source writes require an approved sync plan."
        )
        return SurfaceObservation(
            text=text,
            structured={"local_mirror": raw["local_mirror"]},
            available_tools=tools,
            terminal=bool(self._terminal),
            reward=None,
            raw=raw,
        )

    def act(self, action: Any) -> ActionResult:
        action_dict = self._coerce_action(action)
        function_name, kwargs = self._extract_tool_call(action_dict)
        events: list[Dict[str, Any]] = []
        raw_result: Dict[str, Any]

        try:
            if function_name in {"", "wait"}:
                raw_result = self._raw_success(function_name="wait", reward=0.0, state="WAIT")
            elif function_name == "mirror_acquire":
                mirror = acquire_relevant_files(
                    self.source_root,
                    self.mirror_root,
                    instruction=str(kwargs.get("instruction", "") or self.instruction),
                    candidate_paths=_string_list(kwargs.get("candidate_paths") or kwargs.get("candidates") or self.candidate_paths),
                    limit=int(kwargs.get("limit", 20) or 20),
                )
                selected = [
                    event.get("payload", {}).get("selected_paths", [])
                    for event in mirror.audit_events
                    if event.get("event_type") == "instruction_scoped_acquisition"
                ]
                selected_paths = list(selected[-1] if selected else [])
                self._acquire_attempted = True
                self._last_acquire_selected_count = len(selected_paths)
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.25 if int(mirror.to_manifest().get("workspace_file_count", 0) or 0) else 0.0,
                    state="FILES_ACQUIRED",
                    mirror_manifest=mirror.to_manifest(),
                    selected_paths=selected_paths,
                )
            elif function_name == "mirror_fetch":
                paths = _string_list(kwargs.get("paths") or kwargs.get("relative_paths") or kwargs.get("path") or self.fetch_paths)
                mirror = materialize_files(self.source_root, self.mirror_root, paths)
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.2 if paths else 0.0,
                    state="FILES_FETCHED",
                    mirror_manifest=mirror.to_manifest(),
                    fetched_paths=paths,
                )
            elif function_name == "mirror_exec":
                command = self._normalize_command(kwargs.get("command") or self.default_command)
                allowed = _string_list(kwargs.get("allowed_commands") or self.allowed_commands)
                result = run_mirror_command(
                    self.source_root,
                    self.mirror_root,
                    command,
                    allowed_commands=allowed,
                    timeout_seconds=int(kwargs.get("timeout_seconds", self.default_command_timeout_seconds) or self.default_command_timeout_seconds),
                )
                self._command_executed = True
                self._last_command_returncode = int(result.returncode)
                self._command_failed = result.returncode != 0
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.35 if result.returncode == 0 else 0.0,
                    state="COMMAND_EXECUTED",
                    success=result.returncode == 0,
                    mirror_command=result.to_dict(),
                    sandbox_label="best_effort_local_mirror",
                    not_os_security_sandbox=True,
                )
            elif function_name == "mirror_plan":
                plan = build_sync_plan(self.source_root, self.mirror_root)
                self._last_plan = dict(plan)
                actionable_count = len(list(plan.get("actionable_changes", []) or []))
                waiting_approval = (
                    not bool(self.terminal_after_plan)
                    and actionable_count > 0
                    and not bool(self._command_failed)
                )
                if self.terminal_after_plan or not waiting_approval:
                    self._terminal = True
                state = "WAITING_APPROVAL" if waiting_approval else "SYNC_PLAN_BUILT"
                if self._command_failed:
                    state = "COMMAND_FAILED"
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=0.0 if self._command_failed else (0.75 if actionable_count else 0.1),
                    state=state,
                    sync_plan=plan,
                    approval_status=str(plan.get("approval", {}).get("status", "") or ""),
                    waiting_approval=waiting_approval,
                    approval_request={
                        "type": "local_mirror_sync_plan",
                        "plan_id": str(plan.get("plan_id", "") or ""),
                        "approval_status": str(plan.get("approval", {}).get("status", "") or ""),
                        "actionable_change_count": actionable_count,
                        "source_root": str(self.source_root),
                        "mirror_root": str(self.mirror_root),
                    } if waiting_approval else {},
                )
            elif function_name == "mirror_apply":
                plan_id = str(kwargs.get("plan_id", "") or self._last_plan.get("plan_id", "") or "")
                approved_by = str(kwargs.get("approved_by", "machine") or "machine")
                result = apply_sync_plan(
                    self.source_root,
                    self.mirror_root,
                    plan_id=plan_id,
                    approved_by=approved_by,
                )
                self._applied = True
                self._terminal = True
                raw_result = self._raw_success(
                    function_name=function_name,
                    reward=1.0 if result.get("synced_files") else 0.2,
                    state="SYNC_PLAN_APPLIED",
                    sync_result=result,
                )
            else:
                raise MirrorScopeError(f"unsupported local-machine tool: {function_name}")
        except Exception as exc:
            raw_result = self._raw_failure(function_name=function_name, exc=exc, action=action_dict)
            events.append({"type": "local_machine_error", "function_name": function_name, "reason": str(exc)})

        self._last_action = dict(action_dict)
        observation = self.observe()
        raw_result["terminal"] = bool(observation.terminal)
        raw_result["done"] = bool(observation.terminal)
        return ActionResult(
            ok=bool(raw_result.get("success", False)),
            observation=observation,
            raw=raw_result,
            events=events,
        )

    def decorate_candidate_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tools = self.observe().available_tools
        if not tools:
            return dict(action)
        tool_name = tools[0].name
        return {
            "kind": "call_tool",
            "function_name": tool_name,
            "kwargs": {},
            "payload": {
                "tool_name": "call_hidden_function",
                "tool_args": {"function_name": tool_name, "kwargs": {}},
            },
            "_source": "local_machine_default_tool",
            "_candidate_meta": {"decorated_from_wait": True},
        }

    def get_generic_task_spec(self) -> GenericTaskSpec:
        available = [tool.name for tool in self.observe().available_tools]
        return GenericTaskSpec(
            task_id=self.task_id,
            environment_family="local_machine",
            instruction=self.instruction,
            success_criteria=[
                "files are materialized only after explicit request",
                "source writes occur only through an approved sync plan",
            ],
            available_action_names=available,
            metadata={
                "source_root": str(self.source_root),
                "mirror_root": str(self.mirror_root),
                "adapter_version": LOCAL_MACHINE_ADAPTER_VERSION,
                "default_command_present": bool(self.default_command),
                "allow_empty_exec": bool(self.allow_empty_exec),
                "default_command_timeout_seconds": int(self.default_command_timeout_seconds),
                "terminal_after_plan": bool(self.terminal_after_plan),
            },
        )

    def get_generic_observation(self) -> GenericObservation:
        obs = self.observe()
        mirror_state = dict(obs.raw.get("local_mirror", {}) if isinstance(obs.raw, dict) else {})
        return GenericObservation(
            observation_id=f"{self.task_id}:episode-{self._episode}",
            environment_family="local_machine",
            task_id=self.task_id,
            text=obs.text,
            available_actions=[
                GenericActionDescriptor(
                    name=tool.name,
                    action_family=str(tool.capability_class or "local_machine"),
                    parameter_schema=dict(tool.input_schema or {}),
                    enabled=True,
                    source="local_machine_adapter",
                    attributes={
                        "risk_level": tool.risk_level,
                        "approval_required": tool.approval_required,
                    },
                )
                for tool in obs.available_tools
            ],
            state=mirror_state,
            metadata={
                "source_root": str(self.source_root),
                "mirror_root": str(self.mirror_root),
            },
            raw=dict(obs.raw),
        )

    def _available_tools(self, manifest: Dict[str, Any], sync_plan: Dict[str, Any]) -> list[ToolSpec]:
        if self._terminal and not (self.expose_apply_tool and sync_plan and not self._applied):
            return []

        workspace_count = int(manifest.get("workspace_file_count", 0) or 0)
        if workspace_count <= 0:
            tools: list[ToolSpec] = []
            can_empty_exec = bool(self.default_command and self.allow_empty_exec and not self._command_executed)
            if can_empty_exec:
                tools.append(self._tool_exec())
            if self.instruction or self.candidate_paths:
                skip_repeated_empty_acquire = (
                    self._acquire_attempted
                    and self._last_acquire_selected_count <= 0
                    and bool(self.default_command)
                )
                if not skip_repeated_empty_acquire:
                    tools.append(self._tool_acquire())
            if self.fetch_paths and not tools:
                tools.append(self._tool_fetch())
            if self._command_executed and not sync_plan:
                tools.append(self._tool_plan())
            return tools

        if self.default_command and not self._command_executed:
            return [self._tool_exec()]
        if not sync_plan:
            return [self._tool_plan()]
        if self.expose_apply_tool and not self._applied:
            return [self._tool_apply()]
        return []

    def _observation_payload(
        self,
        manifest: Dict[str, Any],
        sync_plan: Dict[str, Any],
        tools: Sequence[ToolSpec],
    ) -> Dict[str, Any]:
        diff = [entry.to_dict() for entry in compute_mirror_diff(self.source_root, self.mirror_root)]
        local_mirror = {
            "adapter_version": LOCAL_MACHINE_ADAPTER_VERSION,
            "source_root": str(self.source_root),
            "mirror_root": str(self.mirror_root),
            "workspace_root": str(manifest.get("workspace_root", "") or ""),
            "control_root": str(manifest.get("control_root", "") or ""),
            "workspace_initial_state": str(manifest.get("workspace_initial_state", "empty") or "empty"),
            "workspace_file_count": int(manifest.get("workspace_file_count", 0) or 0),
            "materialized_files": list(manifest.get("materialized_files", []) or []),
            "diff_entries": diff,
            "sync_plan": dict(sync_plan),
            "last_action": dict(self._last_action),
            "command_executed": bool(self._command_executed),
            "latest_command_returncode": self._last_command_returncode,
            "command_failed": bool(self._command_failed),
            "default_command_present": bool(self.default_command),
            "allow_empty_exec": bool(self.allow_empty_exec),
            "default_command_timeout_seconds": int(self.default_command_timeout_seconds),
            "terminal_after_plan": bool(self.terminal_after_plan),
            "acquire_attempted": bool(self._acquire_attempted),
            "last_acquire_selected_count": int(self._last_acquire_selected_count),
            "applied": bool(self._applied),
            "audit_events": list(manifest.get("audit_events", []) or []),
        }
        return {
            "success": True,
            "reward": 0.0,
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": "LOCAL_MACHINE_MIRROR",
            "local_mirror": local_mirror,
            "available_functions": [tool.name for tool in tools],
            "function_signatures": {tool.name: dict(tool.input_schema or {}) for tool in tools},
            "sandbox_label": "best_effort_local_mirror",
            "not_os_security_sandbox": True,
        }

    @staticmethod
    def _coerce_action(action: Any) -> Dict[str, Any]:
        if isinstance(action, dict):
            return dict(action)
        if hasattr(action, "payload"):
            return {
                "kind": str(getattr(action, "kind", "") or ""),
                "payload": dict(getattr(action, "payload", {}) or {}),
            }
        return {"kind": str(action or "wait")}

    @staticmethod
    def _extract_tool_call(action: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        if str(action.get("kind", "") or "").strip().lower() == "wait":
            return "wait", {}
        payload = action.get("payload", {}) if isinstance(action.get("payload", {}), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload.get("tool_args", {}), dict) else {}
        function_name = str(
            action.get("function_name")
            or tool_args.get("function_name")
            or payload.get("function_name")
            or ""
        ).strip()
        kwargs = {}
        for candidate in (action.get("kwargs"), tool_args.get("kwargs"), payload.get("kwargs")):
            if isinstance(candidate, dict):
                kwargs.update(candidate)
        return function_name, kwargs

    def _raw_success(self, *, function_name: str, reward: float, state: str, success: bool = True, **extra: Any) -> Dict[str, Any]:
        payload = {
            "success": bool(success),
            "reward": float(reward),
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": state,
            "function_name": str(function_name),
            "local_machine_adapter": LOCAL_MACHINE_ADAPTER_VERSION,
        }
        payload.update(extra)
        return payload

    def _raw_failure(self, *, function_name: str, exc: Exception, action: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": False,
            "reward": 0.0,
            "terminal": bool(self._terminal),
            "done": bool(self._terminal),
            "state": "LOCAL_MACHINE_ERROR",
            "function_name": str(function_name),
            "failure_reason": str(exc),
            "exception_type": type(exc).__name__,
            "action_input": dict(action),
            "local_machine_adapter": LOCAL_MACHINE_ADAPTER_VERSION,
        }

    @staticmethod
    def _zero_arg_schema(description: str) -> Dict[str, Any]:
        return {
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def _tool_acquire(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_acquire",
            description="Materialize source files selected from the current instruction and declared candidates.",
            input_schema=self._zero_arg_schema("Use the adapter instruction and candidate paths."),
            side_effects=["reads declared source files", "writes mirror workspace"],
            risk_notes=["does not write to source root", "selection is limited to supplied candidate paths"],
            capability_class="local_mirror_materialization",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_fetch(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_fetch",
            description="Materialize explicitly declared source files into the mirror workspace.",
            input_schema=self._zero_arg_schema("Use adapter fetch paths."),
            side_effects=["reads declared source files", "writes mirror workspace"],
            risk_notes=["does not write to source root"],
            capability_class="local_mirror_materialization",
            side_effect_class="mirror_workspace_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_exec(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_exec",
            description="Run the configured allowlisted command from inside the mirror workspace.",
            input_schema=self._zero_arg_schema("Use adapter default_command and allowed_commands."),
            side_effects=["executes allowlisted command", "may write mirror workspace"],
            risk_notes=["best-effort local mirror isolation; not an OS security sandbox"],
            capability_class="local_mirror_execution",
            side_effect_class="mirror_workspace_execution",
            approval_required=False,
            risk_level="medium",
        )

    def _tool_plan(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_plan",
            description="Build an auditable sync plan from mirror changes.",
            input_schema=self._zero_arg_schema("No arguments required."),
            side_effects=["writes mirror control sync plan"],
            risk_notes=["does not write to source root"],
            capability_class="local_mirror_review",
            side_effect_class="mirror_control_write",
            approval_required=False,
            risk_level="low",
        )

    def _tool_apply(self) -> ToolSpec:
        return ToolSpec(
            name="mirror_apply",
            description="Apply an approved sync plan back to the source root.",
            input_schema=self._zero_arg_schema("Use the latest plan id with machine approval when eligible."),
            side_effects=["writes approved files to source root"],
            risk_notes=["requires sync plan id", "machine approval is limited to text-like added or modified files"],
            capability_class="local_mirror_sync",
            side_effect_class="source_filesystem_write",
            approval_required=True,
            risk_level="high",
        )

    def to_dict(self) -> Dict[str, Any]:
        spec = self.get_generic_task_spec()
        obs = self.observe()
        return {
            "task_spec": asdict(spec),
            "observation": dict(obs.raw),
        }
