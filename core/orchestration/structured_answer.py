"""Structured kwargs synthesis for the distilled local-machine runtime.

The distilled runtime keeps this module focused on one job: turn a selected
tool/action into auditable executable kwargs for local-machine work.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from modules.llm.capabilities import STRUCTURED_OUTPUT_ACTION_KWARGS
from modules.llm.gateway import ensure_llm_gateway


class StructuredAnswerSynthesizer:
    """Populate action kwargs for local-machine tools."""

    _LOCAL_MACHINE_ATOMIC_FUNCTIONS = frozenset(
        {
            "repo_tree",
            "repo_find",
            "repo_grep",
            "file_read",
            "file_outline",
            "file_summary",
            "note_write",
            "hypothesis_add",
            "hypothesis_update",
            "hypothesis_compete",
            "discriminating_test_add",
            "candidate_files_set",
            "candidate_files_update",
            "investigation_status",
            "propose_patch",
            "apply_patch",
            "edit_replace_range",
            "edit_insert_after",
            "create_file",
            "delete_file",
            "run_test",
            "run_lint",
            "run_typecheck",
            "run_build",
            "read_run_output",
            "read_test_failure",
        }
    )
    _LOCAL_MACHINE_FUNCTIONS = _LOCAL_MACHINE_ATOMIC_FUNCTIONS.union(
        {"internet_fetch", "internet_fetch_project", "mirror_exec"}
    )

    def __init__(self) -> None:
        self._llm_draft_cache: "OrderedDict[str, Tuple[Dict[str, Any], Dict[str, Any]]]" = OrderedDict()
        self._llm_draft_cache_limit = 128

    def maybe_populate_action_kwargs(
        self,
        action: Dict[str, Any],
        obs: Dict[str, Any],
        *,
        llm_client: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        payload = action.get("payload", {}) if isinstance(action.get("payload"), dict) else {}
        tool_args = payload.get("tool_args", {}) if isinstance(payload, dict) else {}
        function_name = str(tool_args.get("function_name", "") or "").strip()
        if not function_name or not self._looks_like_structured_answer(function_name, obs):
            return action

        synthesized_kwargs, strategy_name, synthesis_meta = self._synthesize_kwargs(
            function_name,
            obs,
            llm_client=llm_client,
        )
        if not synthesized_kwargs:
            return action

        updated = deepcopy(action)
        updated_payload = updated.setdefault("payload", {})
        updated_tool_args = updated_payload.setdefault("tool_args", {})
        updated_tool_args["kwargs"] = synthesized_kwargs

        meta = updated.setdefault("_candidate_meta", {})
        if isinstance(meta, dict):
            meta["structured_answer_synthesized"] = True
            meta["structured_answer_function"] = function_name
            meta["structured_answer_strategy"] = strategy_name or ""
            meta["structured_answer_kwargs_keys"] = sorted(synthesized_kwargs.keys())
            if isinstance(synthesis_meta, dict):
                meta["structured_answer_fallback_used"] = bool(synthesis_meta.get("fallback_used", False))
                llm_trace = synthesis_meta.get("llm_trace", [])
                if isinstance(llm_trace, list) and llm_trace:
                    meta["structured_answer_llm_trace"] = [
                        dict(row) for row in llm_trace if isinstance(row, dict)
                    ]
                fallback_reason = str(synthesis_meta.get("fallback_reason", "") or "")
                if fallback_reason:
                    meta["structured_answer_fallback_reason"] = fallback_reason
            self._attach_state_abstraction_meta(meta, synthesized_kwargs, obs)
        return updated

    def _looks_like_structured_answer(self, function_name: str, obs: Dict[str, Any]) -> bool:
        if function_name not in self._LOCAL_MACHINE_FUNCTIONS:
            return False
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs, dict) else {}
        if function_name in self._LOCAL_MACHINE_ATOMIC_FUNCTIONS:
            return isinstance(local_mirror, dict)
        if function_name == "mirror_exec":
            return isinstance(local_mirror, dict)
        return isinstance(local_mirror, dict) and bool(local_mirror.get("internet_enabled"))

    def _synthesize_kwargs(
        self,
        function_name: str,
        obs: Dict[str, Any],
        *,
        llm_client: Any = None,
    ) -> Tuple[Dict[str, Any], Optional[str], Dict[str, Any]]:
        prefer_llm_kwargs = self._prefer_llm_kwargs(obs)
        llm_meta: Dict[str, Any] = {}
        if prefer_llm_kwargs and llm_client is not None:
            kwargs, llm_meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
            if kwargs:
                return kwargs, "llm_draft", llm_meta

        if self._deterministic_fallback_enabled(obs):
            fallback_kwargs = self._local_machine_fallback_kwargs(function_name, obs)
            if fallback_kwargs:
                meta: Dict[str, Any] = {"fallback_used": True}
                if isinstance(llm_meta, dict) and llm_meta.get("llm_trace"):
                    meta["llm_candidate_considered"] = bool(llm_meta.get("llm_candidate_considered", False))
                    meta["llm_candidate_selected"] = False
                    meta["llm_trace"] = list(llm_meta.get("llm_trace", []) or [])
                    meta["fallback_reason"] = "llm_draft_empty_or_invalid"
                return fallback_kwargs, "local_machine_fallback", meta

        if llm_client is not None and not prefer_llm_kwargs:
            kwargs, llm_meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
            if kwargs:
                return kwargs, "llm_draft", llm_meta
        return {}, None, {}

    @staticmethod
    def _deterministic_fallback_enabled(obs: Dict[str, Any]) -> bool:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        if not local_mirror:
            return True
        return bool(local_mirror.get("deterministic_fallback_enabled", True))

    @staticmethod
    def _prefer_llm_kwargs(obs: Dict[str, Any]) -> bool:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        return bool(local_mirror.get("prefer_llm_kwargs", False))

    def _local_machine_fallback_kwargs(self, function_name: str, obs: Dict[str, Any]) -> Dict[str, Any]:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        if not local_mirror:
            return {}
        instruction = str(obs.get("instruction") or local_mirror.get("instruction") or "").lower()
        if function_name == "internet_fetch" and any(
            token in instruction for token in ("market", "research", "trend", "competitor", "调研", "市场")
        ):
            return {
                "url": "https://github.com/topics/ai-tools",
                "filename": "ai-tools-market-signal.html",
            }
        if function_name == "mirror_exec" and (
            "generated_product" in instruction
            or ("product" in instruction and "ai" in instruction)
            or "产品" in instruction
        ):
            return {
                "command": ["python3", "-c", self._local_ai_product_builder_script()],
                "purpose": "build",
                "target": "generated_product",
                "timeout_seconds": 90,
            }
        return {}

    @staticmethod
    def _local_ai_product_builder_script() -> str:
        return r'''
from pathlib import Path
import sys
import textwrap

root = Path("generated_product")
pkg = root / "src" / "insightforge_ai"
tests = root / "tests"
docs = root / "docs"
for path in (pkg, tests, docs):
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str((root / "src").resolve()))

(root / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n.venv/\n", encoding="utf-8")
(root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
(root / "pyproject.toml").write_text(textwrap.dedent("""
[project]
name = "insightforge-ai"
version = "0.1.0"
description = "Local-first evidence brief scoring for AI workflows."
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
""").strip() + "\n", encoding="utf-8")
(pkg / "__init__.py").write_text("from .brief import score_brief\n", encoding="utf-8")
(pkg / "brief.py").write_text(textwrap.dedent("""
def score_brief(text: str) -> dict:
    tokens = [part for part in str(text or "").replace("\\n", " ").split(" ") if part]
    evidence_hits = sum(1 for token in tokens if token.lower().startswith(("http", "file:", "evidence")))
    score = min(100, 30 + len(tokens) * 2 + evidence_hits * 15)
    return {"score": score, "token_count": len(tokens), "evidence_hits": evidence_hits}
""").strip() + "\n", encoding="utf-8")
(root / "README.md").write_text(textwrap.dedent("""
# InsightForge AI

InsightForge AI scores short AI-workflow briefs for clarity and evidence density.
It is dependency-light, local-first, and intended as a small product seed rather
than a template clone.
""").strip() + "\n", encoding="utf-8")
(docs / "DESIGN.md").write_text("Market evidence should be linked before roadmap decisions.\n", encoding="utf-8")
(tests / "test_brief.py").write_text(textwrap.dedent("""
from insightforge_ai import score_brief


def test_score_brief_rewards_evidence_links():
    plain = score_brief("summarize a vague idea")
    cited = score_brief("summarize evidence https://example.test/report")
    assert cited["score"] > plain["score"]
""").strip() + "\n", encoding="utf-8")
'''

    def draft_kwargs_with_llm_only(
        self,
        function_name: str,
        obs: Dict[str, Any],
        llm_client: Any,
    ) -> Dict[str, Any]:
        return self._draft_with_llm(function_name, obs, llm_client)

    def _draft_with_llm(self, function_name: str, obs: Dict[str, Any], llm_client: Any) -> Dict[str, Any]:
        kwargs, _meta = self._draft_with_llm_with_trace(function_name, obs, llm_client)
        return kwargs

    def _draft_with_llm_with_trace(
        self,
        function_name: str,
        obs: Dict[str, Any],
        llm_client: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        prompt = self._build_llm_prompt(function_name, obs)
        system_prompt = self._build_llm_system_prompt(function_name, obs)
        cache_key = self._llm_draft_cache_key(function_name, prompt, system_prompt, llm_client)
        cached = self._llm_draft_cache_get(cache_key)
        if cached is not None:
            return cached

        gateway = ensure_llm_gateway(
            llm_client,
            route_name="structured_answer",
            capability_prefix="structured_output",
        )
        if gateway is None:
            return {}, {
                "llm_candidate_considered": False,
                "llm_trace": [{
                    "function_name": function_name,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": "",
                    "parsed_kwargs": {},
                    "error": "llm_gateway_unavailable",
                }],
            }

        max_tokens = 256
        timeout_sec = 8.0
        try:
            response = gateway.request_raw(
                STRUCTURED_OUTPUT_ACTION_KWARGS,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                system_prompt=system_prompt,
                think=False,
                timeout_sec=timeout_sec,
            )
            kwargs = self._parse_llm_kwargs_response(response)
            meta = {
                "llm_candidate_considered": True,
                "llm_candidate_selected": bool(kwargs),
                "llm_trace": [{
                    "function_name": function_name,
                    "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
                    "route_name": "structured_answer",
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": str(response or ""),
                    "parsed_kwargs": dict(kwargs),
                    "error": "",
                }],
            }
            self._llm_draft_cache_put(cache_key, kwargs, meta)
            return kwargs, meta
        except Exception as exc:
            meta = {
                "llm_candidate_considered": True,
                "llm_candidate_selected": False,
                "llm_trace": [{
                    "function_name": function_name,
                    "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
                    "route_name": "structured_answer",
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "response": "",
                    "parsed_kwargs": {},
                    "error": str(exc),
                }],
            }
            self._llm_draft_cache_put(cache_key, {}, meta)
            return {}, meta

    def _llm_draft_cache_key(
        self,
        function_name: str,
        prompt: str,
        system_prompt: str,
        llm_client: Any,
    ) -> str:
        client_fields = {"class": type(llm_client).__name__, "id": id(llm_client)}
        for attr in ("base_url", "model", "model_name", "route_name"):
            value = getattr(llm_client, attr, None)
            if value:
                client_fields[attr] = str(value)
        payload = {
            "function_name": str(function_name or ""),
            "prompt": str(prompt or ""),
            "system_prompt": str(system_prompt or ""),
            "capability": str(STRUCTURED_OUTPUT_ACTION_KWARGS),
            "client": client_fields,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _llm_draft_cache_get(self, cache_key: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        cached = self._llm_draft_cache.get(cache_key)
        if cached is None:
            return None
        self._llm_draft_cache.move_to_end(cache_key)
        kwargs, meta = deepcopy(cached)
        meta["llm_cache_hit"] = True
        llm_trace = meta.get("llm_trace", [])
        if isinstance(llm_trace, list):
            for row in llm_trace:
                if isinstance(row, dict):
                    row["cache_hit"] = True
        return kwargs, meta

    def _llm_draft_cache_put(
        self,
        cache_key: str,
        kwargs: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        self._llm_draft_cache[cache_key] = (deepcopy(kwargs), deepcopy(meta))
        self._llm_draft_cache.move_to_end(cache_key)
        while len(self._llm_draft_cache) > self._llm_draft_cache_limit:
            self._llm_draft_cache.popitem(last=False)

    def _build_llm_prompt(self, function_name: str, obs: Dict[str, Any]) -> str:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        function_signatures = obs.get("function_signatures", {})
        signature = function_signatures.get(function_name, {}) if isinstance(function_signatures, dict) else {}
        available_functions = sorted(str(name) for name in function_signatures.keys()) if isinstance(function_signatures, dict) else []
        visible_context = {
            "instruction": obs.get("instruction") or local_mirror.get("instruction"),
            "query": obs.get("query"),
            "perception": obs.get("perception"),
            "function_name": function_name,
            "function_signature": signature,
            "available_functions": available_functions,
            "local_mirror": self._compact_local_mirror_for_kwargs(local_mirror),
            "generation_contract": {
                "deterministic_fallback_enabled": bool(local_mirror.get("deterministic_fallback_enabled", True)),
                "require_llm_generation": bool(local_mirror.get("require_llm_generation", False)),
                "require_market_evidence_reference": bool(local_mirror.get("require_market_evidence_reference", False)),
                "require_non_template_product": bool(local_mirror.get("require_non_template_product", False)),
            },
        }
        return (
            "Fill executable kwargs for the selected function using the visible task context.\n"
            "Do not explain your reasoning.\n\n"
            f"Context:\n{json.dumps(visible_context, ensure_ascii=False)}\n"
        )

    @staticmethod
    def _compact_local_mirror_for_kwargs(local_mirror: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(local_mirror, dict):
            return {}
        diff_summary = local_mirror.get("diff_summary", {}) if isinstance(local_mirror.get("diff_summary", {}), dict) else {}
        internet_ingress = local_mirror.get("internet_ingress", {}) if isinstance(local_mirror.get("internet_ingress", {}), dict) else {}
        investigation = local_mirror.get("investigation", {}) if isinstance(local_mirror.get("investigation", {}), dict) else {}
        artifacts = [
            {
                "artifact_id": str(row.get("artifact_id", "") or ""),
                "normalized_url": str(row.get("normalized_url", "") or ""),
                "fetch_kind": str(row.get("fetch_kind", "") or ""),
                "local_path": str(row.get("local_path", "") or ""),
                "bytes_written": int(row.get("bytes_written", 0) or 0),
            }
            for row in list(internet_ingress.get("artifacts", []) or [])[:5]
            if isinstance(row, dict)
        ]
        return {
            "instruction": str(local_mirror.get("instruction", "") or ""),
            "source_root": str(local_mirror.get("source_root", "") or ""),
            "mirror_root": str(local_mirror.get("mirror_root", "") or ""),
            "workspace_root": str(local_mirror.get("workspace_root", "") or ""),
            "control_root": str(local_mirror.get("control_root", "") or ""),
            "workspace_file_count": int(local_mirror.get("workspace_file_count", 0) or 0),
            "command_executed": bool(local_mirror.get("command_executed", False)),
            "latest_command_returncode": local_mirror.get("latest_command_returncode"),
            "diff_ref": dict(local_mirror.get("diff_ref", {}) or {}),
            "diff_summary": {
                "entry_count": int(diff_summary.get("entry_count", 0) or 0),
                "status_counts": dict(diff_summary.get("status_counts", {}) or {}),
                "examples": list(diff_summary.get("examples", []) or [])[:8],
                "examples_truncated": bool(diff_summary.get("examples_truncated", False)),
            },
            "investigation": {
                "candidate_files": list(investigation.get("candidate_files", local_mirror.get("candidate_files", [])) or [])[:20],
                "candidate_reason": str(investigation.get("candidate_reason", "") or ""),
                "last_tree": StructuredAnswerSynthesizer._compact_tree_result(investigation.get("last_tree", {}) or {}),
                "last_search": StructuredAnswerSynthesizer._compact_search_result(investigation.get("last_search", {}) or {}),
                "last_read": StructuredAnswerSynthesizer._compact_read_result(investigation.get("last_read", {}) or {}),
                "notes": list(investigation.get("notes", []) or [])[-12:],
                "hypotheses": list(investigation.get("hypotheses", []) or [])[-8:],
                "last_run_ref": str(investigation.get("last_run_ref", "") or ""),
            },
            "external_baselines": [
                {
                    "artifact_id": str(row.get("artifact_id", "") or ""),
                    "workspace_relative_path": str(row.get("workspace_relative_path", "") or ""),
                    "baseline_path": str(row.get("baseline_path", "") or ""),
                }
                for row in list(local_mirror.get("external_baselines", []) or [])[:5]
                if isinstance(row, dict)
            ],
            "internet_artifacts": artifacts,
        }

    @staticmethod
    def _compact_tree_result(last_tree: Any) -> Dict[str, Any]:
        if not isinstance(last_tree, dict):
            return {}
        entries = list(last_tree.get("entries", []) or [])
        slim_entries = []
        for row in entries[:80]:
            if not isinstance(row, dict):
                continue
            slim_entries.append({
                "path": str(row.get("path", "") or ""),
                "kind": str(row.get("kind", "") or ""),
                "depth": int(row.get("depth", 0) or 0),
            })
        return {
            "root": str(last_tree.get("root", "") or ""),
            "depth": int(last_tree.get("depth", 0) or 0),
            "entry_count": int(last_tree.get("entry_count", len(entries)) or 0),
            "entries": slim_entries,
            "entries_truncated": len(entries) > len(slim_entries),
        }

    @staticmethod
    def _compact_search_result(last_search: Any) -> Dict[str, Any]:
        if not isinstance(last_search, dict):
            return {}
        matches = list(last_search.get("matches", last_search.get("results", [])) or [])
        slim_matches = []
        for row in matches[:60]:
            if not isinstance(row, dict):
                continue
            slim_matches.append({
                "path": str(row.get("path", "") or ""),
                "line": row.get("line") if row.get("line") is not None else row.get("line_number"),
                "text": str(row.get("text", row.get("line_text", "")) or "")[:240],
            })
        compacted = {
            "query": str(last_search.get("query", "") or ""),
            "root": str(last_search.get("root", "") or ""),
            "match_count": int(last_search.get("match_count", len(matches)) or 0),
            "matches": slim_matches,
            "matches_truncated": len(matches) > len(slim_matches),
        }
        if "name_pattern" in last_search:
            compacted["name_pattern"] = str(last_search.get("name_pattern", "") or "")
        return compacted

    @staticmethod
    def _compact_read_result(last_read: Any) -> Dict[str, Any]:
        if not isinstance(last_read, dict):
            return {}
        content = str(last_read.get("content", last_read.get("text", "")) or "")
        if not content and isinstance(last_read.get("lines"), list):
            line_texts: List[str] = []
            for row in list(last_read.get("lines", []) or [])[:120]:
                if not isinstance(row, dict):
                    continue
                line_no = row.get("line", row.get("line_number", ""))
                text = str(row.get("text", row.get("content", "")) or "")
                prefix = f"{line_no}: " if line_no not in (None, "") else ""
                line_texts.append(f"{prefix}{text}")
            content = "\n".join(line_texts)
        return {
            "path": str(last_read.get("path", "") or ""),
            "start_line": last_read.get("start_line"),
            "end_line": last_read.get("end_line"),
            "line_count": last_read.get("line_count"),
            "content_excerpt": content[:4000],
            "content_truncated": len(content) > 4000,
        }

    def _build_llm_system_prompt(self, function_name: str, obs: Dict[str, Any]) -> str:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        genuine_product_rules = ""
        if function_name == "mirror_exec" and (
            bool(local_mirror.get("require_llm_generation", False))
            or bool(local_mirror.get("require_market_evidence_reference", False))
            or bool(local_mirror.get("require_non_template_product", False))
        ):
            genuine_product_rules = (
                "This run requires genuine model-generated product work. Do not reproduce built-in examples or templates.\n"
                "If creating a product, choose a fresh product name and implementation from the research context.\n"
                "When market evidence is available, write a design or market note that cites at least one internet artifact id or normalized URL from local_mirror.internet_artifacts.\n"
                "Do not create SignalBrief AI or signalbrief_ai; those are banned template markers for verifier checks.\n"
            )
        return (
            "You are filling structured tool kwargs for a task-solving agent.\n"
            "Return EXACTLY one line in this format:\n"
            f"KWARGS_JSON: {{...}}\n"
            f"The JSON object must be executable kwargs for `{function_name}`.\n"
            f"{genuine_product_rules}"
            "For local-machine codebase work, prefer atomic actions: repo_tree, repo_find, repo_grep, file_read, note_write, candidate_files_set, apply_patch, edit_replace_range, run_test, run_lint, and read_run_output.\n"
            "LLM decides what to inspect or edit; the adapter decides how to execute it. Do not invent shell commands when an atomic action exists.\n"
            "For mirror_exec, use only a short emergency fallback command with purpose, target, and timeout_seconds; long python -c scripts are rejected.\n"
            "For internet_fetch_project, choose a public project URL and use source_type auto, git, or archive.\n"
            "Return no extra text."
        )

    def _parse_llm_kwargs_response(self, response: Any) -> Dict[str, Any]:
        text = str(response or "").strip()
        if not text:
            return {}
        text = self._strip_llm_fences(text)
        for prefix in ("KWARGS_JSON:",):
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(prefix):
                    payload = stripped[len(prefix) :].strip()
                    parsed = self._try_parse_llm_kwargs_json(payload)
                    if parsed:
                        return parsed
        parsed = self._try_parse_llm_kwargs_json(text)
        return parsed if parsed else {}

    def _try_parse_llm_kwargs_json(self, text: str) -> Dict[str, Any]:
        stripped = str(text or "").strip()
        if not stripped:
            return {}
        start = stripped.find("{")
        end = stripped.rfind("}") + 1
        candidate = stripped[start:end] if start >= 0 and end > start else stripped
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _strip_llm_fences(self, text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    def _attach_state_abstraction_meta(
        self,
        meta: Dict[str, Any],
        synthesized_kwargs: Dict[str, Any],
        obs: Dict[str, Any],
    ) -> None:
        local_mirror = obs.get("local_mirror", {}) if isinstance(obs.get("local_mirror", {}), dict) else {}
        diff_summary = local_mirror.get("diff_summary", {}) if isinstance(local_mirror.get("diff_summary", {}), dict) else {}
        investigation = local_mirror.get("investigation", {}) if isinstance(local_mirror.get("investigation", {}), dict) else {}
        meta["structured_answer_state_abstraction"] = {
            "runtime": "local_machine",
            "workspace_file_count": int(local_mirror.get("workspace_file_count", 0) or 0),
            "diff_entry_count": int(diff_summary.get("entry_count", 0) or 0),
            "candidate_file_count": len(list(investigation.get("candidate_files", []) or [])),
            "kwargs_keys": sorted(synthesized_kwargs.keys()),
        }
