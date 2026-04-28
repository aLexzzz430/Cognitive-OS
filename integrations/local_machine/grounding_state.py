"""Hypothesis lifecycle and action-grounding state updates for local-machine tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence

from core.runtime.hypothesis_lifecycle import (
    HYPOTHESIS_LIFECYCLE_VERSION,
    apply_hypothesis_evidence,
    build_discriminating_test,
    hypothesis_lifecycle_summary,
    mark_competing,
    normalize_hypothesis,
)
from integrations.local_machine.action_grounding import extract_grounding_target_file
from integrations.local_machine.target_binding import bind_target


LOCAL_MACHINE_INVESTIGATION_VERSION = "conos.local_machine.investigation/v1"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


class LocalMachineGroundingStateMixin:
    def _detect_investigation_stall(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        history = [
            dict(row)
            for row in list(state.get("action_history", []) or [])
            if isinstance(row, dict)
        ]
        if len(history) < 4 or self._meaningful_actionable_diff_count() > 0:
            return {}
        recent = history[-4:]
        repeated_actions = {"repo_grep", "read_test_failure", "run_test"}
        repeated_count = sum(1 for row in recent if str(row.get("function_name") or "") in repeated_actions)
        new_file_read = any(str(row.get("function_name") or "") == "file_read" for row in recent)
        if repeated_count < 3 or new_file_read:
            return {}
        binding = dict(state.get("target_binding", {}) or {})
        recommended = "target_binding_or_patch_proposal"
        top_target = str(binding.get("top_target_file") or "")
        if top_target:
            read_paths = {
                str(row.get("path") or "")
                for row in list(state.get("read_files", []) or [])
                if isinstance(row, dict)
            }
            recommended = "propose_patch" if top_target in read_paths else "file_read"
        return {
            "schema_version": LOCAL_MACHINE_INVESTIGATION_VERSION,
            "event_type": "investigation_stalled",
            "recommended_action": recommended,
            "top_target_file": top_target,
            "target_confidence": binding.get("target_confidence", 0.0),
            "recent_actions": [str(row.get("function_name") or "") for row in recent],
            "created_at": _now(),
        }

    def _closed_loop_probe_variant(self) -> str:
        match = re.search(r"\[closed_loop_probe_variant=([A-Za-z0-9_\-]+)\]", self.instruction)
        return str(match.group(1)) if match else "full"

    @staticmethod
    def _latest_failed_validation_run(state: Mapping[str, Any]) -> Dict[str, Any]:
        for row in reversed(list(state.get("validation_runs", []) or [])):
            if isinstance(row, Mapping) and not bool(row.get("success", False)):
                return dict(row)
        return {}

    @staticmethod
    def _candidate_rows_from_binding(binding: Mapping[str, Any]) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for item in list(binding.get("target_file_candidates", []) or []):
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("target_file") or "").strip()
            if not path or path.startswith("tests/") or not path.endswith(".py"):
                continue
            rows.append(
                {
                    "target_file": path,
                    "score": float(item.get("score", 0.0) or 0.0),
                    "reasons": _string_list(item.get("reasons")),
                }
            )
        return rows

    @staticmethod
    def _surface_and_root_targets(binding: Mapping[str, Any], target_file: str) -> tuple[str, str, bool]:
        candidates = LocalMachineGroundingStateMixin._candidate_rows_from_binding(binding)
        top_target = str(target_file or binding.get("top_target_file") or "").strip()
        if not top_target and candidates:
            top_target = str(candidates[0].get("target_file") or "")
        traceback_files = [
            str(path)
            for path in _string_list(binding.get("traceback_files"))
            if str(path).endswith(".py") and not str(path).startswith("tests/")
        ]
        surface = ""
        for path in traceback_files:
            if path and path != top_target:
                surface = path
                break
        if not surface:
            for row in candidates:
                reasons = " ".join(_string_list(row.get("reasons"))).lower()
                path = str(row.get("target_file") or "")
                if path and path != top_target and "traceback" in reasons:
                    surface = path
                    break
        if not surface:
            for row in candidates:
                path = str(row.get("target_file") or "")
                if path and path != top_target:
                    surface = path
                    break
        unresolved_needed = False
        if not surface and top_target:
            surface = top_target
            unresolved_needed = True
        return surface, top_target, unresolved_needed

    def _explicit_hypothesis_id(self, hypothesis_type: str, target_file: str) -> str:
        return f"hyp_{hypothesis_type}_{_json_hash({'task_id': self.task_id, 'target_file': target_file})[:12]}"

    @staticmethod
    def _augment_explicit_hypothesis_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
        hypothesis = dict(row)
        metadata = dict(hypothesis.get("metadata", {}) or {})
        target_file = str(metadata.get("target_file") or hypothesis.get("target_file") or "")
        hypothesis_type = str(metadata.get("hypothesis_type") or hypothesis.get("hypothesis_type") or hypothesis.get("family") or "codebase")
        summary = str(hypothesis.get("summary") or hypothesis.get("claim") or "")
        hypothesis["summary"] = summary
        hypothesis["target_file"] = target_file
        hypothesis["hypothesis_type"] = hypothesis_type
        hypothesis["status"] = str(hypothesis.get("status") or "active")
        hypothesis["posterior"] = float(hypothesis.get("posterior", hypothesis.get("confidence", 0.5)) or 0.5)
        hypothesis["predictions"] = dict(hypothesis.get("predictions", {}) or {})
        hypothesis["predicted_observation_tokens"] = _string_list(
            hypothesis.get("predicted_observation_tokens")
            or metadata.get("predicted_observation_tokens")
            or hypothesis["predictions"].get("predicted_observation_tokens")
        )
        predicted_effects = hypothesis.get("predicted_action_effects") or metadata.get("predicted_action_effects")
        hypothesis["predicted_action_effects"] = dict(predicted_effects or {})
        falsifiers = hypothesis.get("falsifiers", {})
        hypothesis["falsifiers"] = dict(falsifiers or {}) if isinstance(falsifiers, Mapping) else {"observations": _string_list(falsifiers)}
        conflicts = _string_list(hypothesis.get("conflicts_with") or hypothesis.get("competing_with"))
        hypothesis["conflicts_with"] = conflicts
        hypothesis["competing_with"] = conflicts
        hypothesis["evidence_refs"] = _string_list(hypothesis.get("evidence_refs"))
        hypothesis["metadata"] = metadata
        return hypothesis

    def _make_explicit_hypothesis(
        self,
        *,
        hypothesis_type: str,
        target_file: str,
        binding: Mapping[str, Any],
        evidence_refs: Sequence[str],
        confidence: float,
    ) -> Dict[str, Any]:
        failed_test = str(binding.get("latest_failed_test_target") or "")
        failure_symbols = _string_list(binding.get("failure_symbols"))[:6]
        target = str(target_file or "").strip()
        if hypothesis_type == "surface_wrapper":
            summary = f"Surface wrapper {target or '<unknown>'} misroutes or transforms data before the observed failing test."
            observation_tokens = [Path(target).stem, "traceback", "wrapper", "surface"]
            action_effects = {
                "file_read": f"Reading {target} should reveal the wrapper call path or lossy transform.",
                "run_test": "A direct downstream test may continue to fail if the root cause is deeper.",
            }
            falsifiers = {
                "observations": [
                    "A downstream direct test fails without implicating this wrapper.",
                    "A bounded patch to this wrapper does not pass the full verifier suite.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        elif hypothesis_type == "downstream_root_cause":
            summary = f"Downstream implementation {target or '<unknown>'} contains the root cause behind the failing behavior."
            observation_tokens = [Path(target).stem, "downstream", "root", *failure_symbols[:3]]
            action_effects = {
                "file_read": f"Reading {target} should expose logic tied to the failing assertion or symbol.",
                "propose_patch": "A bounded diff on this target should be verifier-gated by targeted and full tests.",
            }
            falsifiers = {
                "observations": [
                    "The target has no relevant symbol, import path, or assertion-linked behavior.",
                    "A verifier-gated bounded patch to this target is rejected.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        else:
            summary = "The current evidence is underspecified; no source patch is safe until more disambiguating evidence is collected."
            observation_tokens = ["ambiguous", "underspecified", *failure_symbols[:3]]
            action_effects = {
                "file_read": "Additional source or test reads should clarify which hypothesis is actionable.",
                "propose_patch": "Patch proposal should refuse if evidence remains insufficient.",
            }
            falsifiers = {
                "observations": [
                    "A target binding reaches sufficient confidence with direct evidence.",
                    "A discriminating test separates competing source-level explanations.",
                ],
                "tests": [failed_test],
            }
            prior = confidence
        hypothesis = normalize_hypothesis(
            hypothesis_id=self._explicit_hypothesis_id(hypothesis_type, target),
            run_id=self.task_id,
            task_family="local_machine",
            family=hypothesis_type,
            claim=summary,
            confidence=prior,
            evidence_refs=evidence_refs,
            predictions={
                "target_file": target,
                "failed_test": failed_test,
                "failure_symbols": failure_symbols,
                "predicted_observation_tokens": observation_tokens,
            },
            falsifiers=falsifiers,
            metadata={
                "source": "local_machine_explicit_hypothesis_lifecycle",
                "created_at_iso": _now(),
                "target_file": target,
                "hypothesis_type": hypothesis_type,
                "binding_reasons": _string_list(binding.get("binding_reasons")),
                "predicted_action_effects": action_effects,
                "predicted_observation_tokens": observation_tokens,
            },
        )
        hypothesis["summary"] = summary
        hypothesis["target_file"] = target
        hypothesis["hypothesis_type"] = hypothesis_type
        hypothesis["predicted_observation_tokens"] = observation_tokens
        hypothesis["predicted_action_effects"] = action_effects
        hypothesis["conflicts_with"] = []
        return self._augment_explicit_hypothesis_fields(hypothesis)

    def _append_hypothesis_lifecycle_event(self, state: Dict[str, Any], event: Mapping[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("schema_version", HYPOTHESIS_LIFECYCLE_VERSION)
        payload.setdefault("run_id", self.task_id)
        payload.setdefault("created_at", _now())
        events = [dict(row) for row in list(state.get("hypothesis_events", []) or []) if isinstance(row, dict)]
        event_key = _json_hash(payload)
        existing_keys = {
            str(row.get("_event_key") or _json_hash(row))
            for row in events[-200:]
            if isinstance(row, Mapping)
        }
        if event_key in existing_keys:
            return
        payload["_event_key"] = event_key
        events.append(payload)
        state["hypothesis_events"] = events[-200:]

    def _sync_hypothesis_conflicts(self, hypotheses: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
        rows = [self._augment_explicit_hypothesis_fields(row) for row in list(hypotheses or []) if isinstance(row, Mapping)]
        ids = [str(row.get("hypothesis_id") or "") for row in rows if str(row.get("hypothesis_id") or "")]
        for left in ids:
            for right in ids:
                if left and right and left != right:
                    rows = mark_competing(rows, left, right)
        return [self._augment_explicit_hypothesis_fields(row) for row in rows]

    def _ensure_bound_discriminating_test(
        self,
        state: Dict[str, Any],
        hypotheses: Sequence[Mapping[str, Any]],
        binding: Mapping[str, Any],
    ) -> None:
        if self._closed_loop_probe_variant() == "no_discriminating_experiment":
            return
        rows = [self._augment_explicit_hypothesis_fields(row) for row in hypotheses if isinstance(row, Mapping)]
        if len(rows) < 2:
            return
        left, right = rows[0], rows[1]
        left_id = str(left.get("hypothesis_id") or "")
        right_id = str(right.get("hypothesis_id") or "")
        if not left_id or not right_id or left_id == right_id:
            return
        tests = [dict(row) for row in list(state.get("discriminating_tests", []) or []) if isinstance(row, dict)]
        for test in tests:
            refs = _string_list(test.get("discriminates_between") or test.get("hypotheses") or [test.get("hypothesis_a"), test.get("hypothesis_b")])
            if {left_id, right_id}.issubset(set(refs)):
                return
        failed_target = str(binding.get("latest_failed_test_target") or ".") or "."
        action = {"action": "run_test", "args": {"target": failed_target, "timeout_seconds": 30}}
        test = build_discriminating_test(
            hypothesis_a=left,
            hypothesis_b=right,
            action=action,
            expected_if_a=f"Evidence continues to implicate {left.get('target_file') or 'the surface path'} and weakens {right_id}.",
            expected_if_b=f"Evidence implicates {right.get('target_file') or 'the downstream path'} and weakens {left_id}.",
            why="The same failing behavior has at least two source-level explanations; the targeted failing test and related reads separate wrapper versus root-cause predictions.",
        )
        test["discriminates_between"] = [left_id, right_id]
        test["expected_outcomes_by_hypothesis"] = {
            left_id: str(test.get("expected_if_a") or ""),
            right_id: str(test.get("expected_if_b") or ""),
        }
        test["expected_information_gain"] = 0.42
        tests.append(test)
        state["discriminating_tests"] = tests[-100:]
        self._append_hypothesis_lifecycle_event(
            state,
            {
                "event_type": "discriminating_test_bound_to_hypotheses",
                "hypotheses": [left_id, right_id],
                "discriminates_between": [left_id, right_id],
                "test_id": test["test_id"],
                "delta": 0.0,
            },
        )

    def _update_explicit_hypothesis_posteriors(
        self,
        state: Dict[str, Any],
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        binding: Mapping[str, Any],
        root_target: str,
        surface_target: str,
        evidence_refs: Sequence[str],
    ) -> list[Dict[str, Any]]:
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        if self._closed_loop_probe_variant() == "no_posterior":
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        success = bool(raw_result.get("success", False))
        state_name = str(raw_result.get("state", "") or "")
        run_ref = str(raw_result.get("run_ref") or state.get("last_run_ref") or "")
        evidence_key = _json_hash(
            {
                "function": function_name,
                "success": success,
                "state": state_name,
                "run_ref": run_ref,
                "patch_sha256": raw_result.get("patch_sha256", ""),
                "root_target": root_target,
                "surface_target": surface_target,
            }
        )
        applied_keys = set(_string_list(state.get("hypothesis_evidence_keys")))
        if evidence_key in applied_keys:
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        should_update = bool(
            (function_name == "run_test" and not success)
            or (function_name in {"propose_patch", "apply_patch"} and success)
            or (function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)))
        )
        if not should_update:
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return hypotheses
        updated_rows: list[Dict[str, Any]] = []
        for hypothesis in hypotheses:
            htype = str(hypothesis.get("hypothesis_type") or "")
            target = str(hypothesis.get("target_file") or "")
            signal = "neutral"
            strength = 0.08
            rationale = "Evidence observed but did not change this hypothesis."
            if function_name == "run_test" and not success:
                if htype == "downstream_root_cause" and target and target == root_target:
                    signal = "support"
                    strength = 0.24
                    rationale = "Failure evidence plus target binding favors the downstream/root-cause explanation."
                elif htype == "surface_wrapper" and root_target and surface_target and root_target != surface_target:
                    signal = "contradict"
                    strength = 0.22
                    rationale = "Failure evidence points past the surface wrapper toward a downstream target."
                elif htype == "unresolved_spec":
                    signal = "contradict" if root_target else "support"
                    strength = 0.1
                    rationale = "Target binding reduced ambiguity." if root_target else "Evidence remains underspecified."
            elif function_name in {"propose_patch", "apply_patch"} and success:
                touched = set(_string_list(raw_result.get("touched_files")))
                if target and target in touched:
                    signal = "support"
                    strength = 0.2
                    rationale = "Verifier-gated patch touched this hypothesis target and succeeded."
                elif htype == "unresolved_spec":
                    signal = "contradict"
                    strength = 0.12
                    rationale = "A verified source patch means evidence was sufficient."
            elif function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)):
                signal = "support" if htype == "unresolved_spec" else "contradict"
                strength = 0.14
                rationale = "Patch proposal refused because evidence remained insufficient."
            if signal == "neutral":
                updated_rows.append(hypothesis)
                continue
            updated, event = apply_hypothesis_evidence(
                hypothesis,
                signal=signal,
                evidence_refs=evidence_refs,
                strength=strength,
                rationale=rationale,
            )
            event["run_id"] = self.task_id
            event["target_file"] = target
            event["binding_top_target_file"] = root_target
            event["source_function"] = str(function_name)
            updated = self._augment_explicit_hypothesis_fields(updated)
            updated_rows.append(updated)
            self._append_hypothesis_lifecycle_event(state, event)
            self._persist_hypothesis_lifecycle(updated, event)
        applied_keys.add(evidence_key)
        state["hypothesis_evidence_keys"] = sorted(applied_keys)[-200:]
        state["hypotheses"] = updated_rows[-100:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(updated_rows)
        return updated_rows

    def _ensure_explicit_hypothesis_lifecycle(
        self,
        state: Dict[str, Any],
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        binding: Mapping[str, Any],
        target_file: str,
    ) -> Dict[str, Any]:
        if not isinstance(binding, Mapping) or not (
            binding.get("target_file_candidates")
            or binding.get("top_target_file")
            or binding.get("traceback_files")
        ):
            hypotheses = [
                self._augment_explicit_hypothesis_fields(row)
                for row in list(state.get("hypotheses", []) or [])
                if isinstance(row, Mapping)
            ]
            state["hypotheses"] = hypotheses[-100:]
            state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
            return state
        failed_run = self._latest_failed_validation_run(state)
        run_ref = str(failed_run.get("run_ref") or raw_result.get("run_ref") or state.get("last_run_ref") or "")
        evidence_refs = [f"failure:{run_ref}"] if run_ref else []
        failed_target = str(binding.get("latest_failed_test_target") or "")
        if failed_target:
            evidence_refs.append(f"test:{failed_target}")
        surface_target, root_target, unresolved_needed = self._surface_and_root_targets(binding, target_file)
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        by_id = {str(row.get("hypothesis_id") or ""): row for row in hypotheses}
        desired: list[Dict[str, Any]] = []
        if surface_target:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="surface_wrapper",
                    target_file=surface_target,
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.46 if root_target and root_target != surface_target else 0.5,
                )
            )
        if root_target and root_target != surface_target:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="downstream_root_cause",
                    target_file=root_target,
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.54,
                )
            )
        if unresolved_needed or len(desired) < 2:
            desired.append(
                self._make_explicit_hypothesis(
                    hypothesis_type="unresolved_spec",
                    target_file="",
                    binding=binding,
                    evidence_refs=evidence_refs,
                    confidence=0.42,
                )
            )
        created_any = False
        for hypothesis in desired:
            hid = str(hypothesis.get("hypothesis_id") or "")
            if hid in by_id:
                current = self._augment_explicit_hypothesis_fields({**hypothesis, **by_id[hid], "metadata": {**dict(hypothesis.get("metadata", {}) or {}), **dict(by_id[hid].get("metadata", {}) or {})}})
                by_id[hid] = current
                continue
            by_id[hid] = hypothesis
            created_any = True
            self._append_hypothesis_lifecycle_event(
                state,
                {
                    "event_type": "hypothesis_created",
                    "hypothesis_id": hid,
                    "hypothesis_type": hypothesis.get("hypothesis_type"),
                    "target_file": hypothesis.get("target_file"),
                    "evidence_refs": list(hypothesis.get("evidence_refs", []) or []),
                    "delta": 0.0,
                },
            )
            self._persist_hypothesis_lifecycle(hypothesis, {"hypothesis_id": hid, "event_type": "hypothesis_created", "evidence_refs": evidence_refs, "delta": 0.0})
        hypotheses = self._sync_hypothesis_conflicts(list(by_id.values()))
        if created_any and len(hypotheses) >= 2:
            ids = [str(row.get("hypothesis_id") or "") for row in hypotheses[:2]]
            self._append_hypothesis_lifecycle_event(
                state,
                {
                    "event_type": "hypothesis_competition_recorded",
                    "hypotheses": ids,
                    "delta": 0.0,
                    "reason": "Automatic lifecycle construction found multiple plausible source-level explanations.",
                },
            )
        state["hypotheses"] = hypotheses[-100:]
        self._ensure_bound_discriminating_test(state, hypotheses, binding)
        hypotheses = self._update_explicit_hypothesis_posteriors(
            state,
            function_name=function_name,
            kwargs=kwargs,
            raw_result=raw_result,
            binding=binding,
            root_target=root_target,
            surface_target=surface_target,
            evidence_refs=evidence_refs,
        )
        state["hypotheses"] = hypotheses[-100:]
        state["hypothesis_lifecycle"] = hypothesis_lifecycle_summary(hypotheses)
        return state

    def _leading_hypothesis_from_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        hypotheses = [
            self._augment_explicit_hypothesis_fields(row)
            for row in list(state.get("hypotheses", []) or [])
            if isinstance(row, Mapping)
        ]
        if not hypotheses:
            return {}
        lifecycle = hypothesis_lifecycle_summary(hypotheses)
        leading_id = str(lifecycle.get("leading_hypothesis_id") or "")
        for row in hypotheses:
            if str(row.get("hypothesis_id") or "") == leading_id:
                return row
        return sorted(hypotheses, key=lambda row: float(row.get("posterior", 0.0) or 0.0), reverse=True)[0]

    def _update_action_grounding_state(
        self,
        *,
        function_name: str,
        kwargs: Mapping[str, Any],
        raw_result: Mapping[str, Any],
        grounding_event: Mapping[str, Any] | None,
        original_function_name: str,
        original_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        state = self._load_investigation_state()
        phase_before = str(state.get("investigation_phase") or "discover")
        action_count = int(state.get("action_count", 0) or 0) + 1
        state["action_count"] = action_count
        action_history = [
            dict(row)
            for row in list(state.get("action_history", []) or [])
            if isinstance(row, dict)
        ]
        action_history.append(
            {
                "tick": action_count,
                "function_name": str(function_name or ""),
                "original_function_name": str(original_function_name or ""),
                "success": bool(raw_result.get("success", False)),
                "state": str(raw_result.get("state", "") or ""),
                "path": str(raw_result.get("path") or kwargs.get("path") or kwargs.get("target_file") or ""),
                "created_at": _now(),
            }
        )
        state["action_history"] = action_history[-100:]
        grounding = dict(state.get("grounding", {}) or {})
        grounding.setdefault("events", [])
        grounding.setdefault("invalid_action_kwargs_events", [])
        grounding.setdefault("repaired_actions", [])
        grounding.setdefault("side_effect_after_verified_completion_events", [])
        if not original_kwargs and str(original_function_name or "") not in {"", "wait"}:
            grounding["empty_kwargs_attempt_count"] = int(grounding.get("empty_kwargs_attempt_count", 0) or 0) + 1
        if grounding_event:
            event = dict(grounding_event)
            events = [dict(row) for row in list(grounding.get("events", []) or []) if isinstance(row, dict)]
            events.append(event)
            grounding["events"] = events[-100:]
            if event.get("event_type") == "invalid_action_kwargs":
                invalid = [
                    dict(row)
                    for row in list(grounding.get("invalid_action_kwargs_events", []) or [])
                    if isinstance(row, dict)
                ]
                invalid.append(event)
                grounding["invalid_action_kwargs_events"] = invalid[-50:]
            elif event.get("event_type") == "local_machine_action_kwargs_repaired":
                repaired = [
                    dict(row)
                    for row in list(grounding.get("repaired_actions", []) or [])
                    if isinstance(row, dict)
                ]
                repaired.append(event)
                grounding["repaired_actions"] = repaired[-50:]
            elif event.get("event_type") == "side_effect_after_verified_completion":
                blocked = [
                    dict(row)
                    for row in list(grounding.get("side_effect_after_verified_completion_events", []) or [])
                    if isinstance(row, dict)
                ]
                blocked.append(event)
                grounding["side_effect_after_verified_completion_events"] = blocked[-50:]

        context = self._action_grounding_context(state_override={**state, "grounding": grounding})
        target_binding = bind_target(context)
        if target_binding:
            state["target_binding"] = target_binding
        target_file = extract_grounding_target_file(context)
        bound_target = str(target_binding.get("top_target_file") or "") if isinstance(target_binding, Mapping) else ""
        try:
            bound_confidence = float(target_binding.get("target_confidence", 0.0) or 0.0) if isinstance(target_binding, Mapping) else 0.0
        except (TypeError, ValueError):
            bound_confidence = 0.0
        if bound_target and (not target_file or bound_confidence >= 0.55):
            target_file = bound_target
        if target_file:
            grounding["target_file"] = target_file
        state = self._ensure_explicit_hypothesis_lifecycle(
            state,
            function_name=function_name,
            kwargs=kwargs,
            raw_result=raw_result,
            binding=dict(target_binding or {}),
            target_file=target_file,
        )

        phase_after = phase_before
        success = bool(raw_result.get("success", False))
        state_name = str(raw_result.get("state", "") or "")
        if phase_before == "complete-ready" and not (function_name == "run_test" and not success):
            phase_after = "complete-ready"
        elif function_name == "repo_tree" and success:
            phase_after = "inspect"
        elif function_name == "run_test" and not success:
            phase_after = "localize"
        elif function_name == "file_read" and success:
            read_path = str(raw_result.get("path") or kwargs.get("path") or "")
            if read_path.endswith(".py") and not (read_path.startswith("tests/") or "/tests/" in read_path):
                current_target = str(grounding.get("target_file") or "")
                if not current_target or read_path == current_target or phase_before in {"localize", "patch"}:
                    grounding["target_file"] = read_path
                target_file = str(grounding.get("target_file") or target_file)
            if read_path and read_path == str(grounding.get("target_file") or "") and target_file:
                phase_after = "patch"
        elif function_name == "apply_patch" and success:
            phase_after = "verify"
            grounding["verification_pending"] = True
            grounding["last_patch"] = {
                "touched_files": list(raw_result.get("touched_files", []) or []),
                "patch_sha256": str(raw_result.get("patch_sha256") or ""),
                "created_at": _now(),
            }
        elif function_name == "propose_patch" and success and bool(raw_result.get("patch_proposal_verified", False)):
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            grounding["last_patch"] = {
                "touched_files": list(raw_result.get("touched_files", []) or []),
                "patch_sha256": str(raw_result.get("patch_sha256") or ""),
                "created_at": _now(),
                "source": "patch_proposal",
            }
            if str(state.get("terminal_state") or "") != "completed_verified":
                state["terminal_state"] = "completed_verified"
                state["completion_reason"] = "bounded patch proposal passed targeted and full verifier tests"
                state["terminal_tick"] = action_count
                state["verified_completion"] = True
        elif function_name == "propose_patch" and bool(raw_result.get("needs_human_review", False)):
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            if str(state.get("terminal_state") or "") != "needs_human_review":
                state["terminal_state"] = "needs_human_review"
                state["completion_reason"] = str(raw_result.get("refusal_reason") or "evidence_insufficient")
                state["terminal_tick"] = action_count
                state["verified_completion"] = False
                state["needs_human_review"] = True
                state["refusal_reason"] = str(raw_result.get("refusal_reason") or "evidence_insufficient")
        elif function_name == "run_test" and success and str(kwargs.get("target") or ".") == ".":
            phase_after = "complete-ready"
            grounding["verification_pending"] = False
            if str(state.get("terminal_state") or "") != "completed_verified":
                state["terminal_state"] = "completed_verified"
                state["completion_reason"] = "full test suite passed after patch verification"
                state["terminal_tick"] = action_count
                state["verified_completion"] = True
        elif function_name == "run_test" and success:
            grounding["last_successful_test_target"] = str(kwargs.get("target") or "")
            if phase_before == "verify":
                phase_after = "verify"
        elif function_name == "mirror_plan" and success and phase_before == "complete-ready":
            phase_after = "complete-ready"
        elif state_name == "INVALID_ACTION_KWARGS":
            phase_after = phase_before

        stall_event = self._detect_investigation_stall({**state, "grounding": grounding})
        if stall_event:
            stalled_events = [
                dict(row)
                for row in list(state.get("stalled_events", []) or [])
                if isinstance(row, dict)
            ]
            last_key = (
                str(stalled_events[-1].get("recommended_action") or ""),
                str(stalled_events[-1].get("top_target_file") or ""),
                str(stalled_events[-1].get("recent_actions") or ""),
            ) if stalled_events else ("", "", "")
            next_key = (
                str(stall_event.get("recommended_action") or ""),
                str(stall_event.get("top_target_file") or ""),
                str(stall_event.get("recent_actions") or ""),
            )
            if next_key != last_key:
                stalled_events.append(stall_event)
                state["stalled_events"] = stalled_events[-50:]
                events = [dict(row) for row in list(grounding.get("events", []) or []) if isinstance(row, dict)]
                events.append(stall_event)
                grounding["events"] = events[-100:]
        state["grounding"] = grounding
        state["investigation_phase"] = phase_after
        self._save_investigation_state(state)
        return {
            "schema_version": LOCAL_MACHINE_INVESTIGATION_VERSION,
            "phase_before": phase_before,
            "phase_after": phase_after,
            "terminal_state": str(state.get("terminal_state") or ""),
            "completion_reason": str(state.get("completion_reason") or ""),
            "terminal_tick": state.get("terminal_tick"),
            "verified_completion": bool(state.get("verified_completion", False)),
            "target_file": str(grounding.get("target_file") or ""),
            "target_binding": dict(state.get("target_binding", {}) or {}),
            "hypothesis_lifecycle": dict(state.get("hypothesis_lifecycle", {}) or {}),
            "stalled_event": dict(stall_event or {}),
            "empty_kwargs_attempt_count": int(grounding.get("empty_kwargs_attempt_count", 0) or 0),
            "repaired_action_count": len(list(grounding.get("repaired_actions", []) or [])),
            "invalid_action_kwargs_count": len(list(grounding.get("invalid_action_kwargs_events", []) or [])),
        }

