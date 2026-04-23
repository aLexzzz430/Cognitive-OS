from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional, Sequence

from core.main_loop import CoreMainLoop
from integrations.webarena.audit import WebArenaAuditWriter, summarize_audit
from integrations.webarena.task_adapter import WebArenaSurfaceAdapter
from modules.llm import build_llm_client


def run_webarena_task(
    *,
    config_file: str | None = None,
    task_id: str = "",
    instruction: str = "",
    agent_id: str = "agi_world_v2",
    run_id: Optional[str] = None,
    max_episodes: int = 1,
    max_ticks_per_episode: int = 64,
    seed: int = 0,
    verbose: bool = False,
    headless: bool = True,
    observation_type: str = "accessibility_tree",
    current_viewport_only: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    llm_client: Any = None,
    llm_mode: str = "integrated",
    env: Any | None = None,
) -> Dict[str, Any]:
    world = WebArenaSurfaceAdapter(
        env=env,
        config_file=config_file,
        task_id=task_id,
        instruction=instruction,
        headless=headless,
        observation_type=observation_type,
        current_viewport_only=current_viewport_only,
        viewport_size={"width": int(viewport_width), "height": int(viewport_height)},
    )
    loop = CoreMainLoop(
        agent_id=agent_id,
        run_id=run_id or f"webarena-{task_id or 'task'}",
        seed=seed,
        max_episodes=max_episodes,
        max_ticks_per_episode=max_ticks_per_episode,
        verbose=verbose,
        world_adapter=world,
        llm_client=llm_client,
        llm_mode=llm_mode,
        world_provider_source="integrations.webarena.runner",
    )
    audit = loop.run()
    final_observation = world.observe()
    task_spec = world.get_generic_task_spec()
    audit["webarena_task_id"] = task_spec.task_id
    audit["webarena_instruction"] = task_spec.instruction
    audit["webarena_task_metadata"] = dict(task_spec.metadata)
    audit["final_surface_structured"] = dict(final_observation.structured or {})
    audit["final_surface_terminal"] = bool(final_observation.terminal)
    audit["final_surface_raw"] = dict(final_observation.raw or {})
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="conos run webarena",
        description="Run AGI_WORLD_V2 against WebArena through the generic environment adapter.",
    )
    parser.add_argument("--config-file", default=None, help="Optional WebArena task config file.")
    parser.add_argument("--task-id", default="", help="Optional task id for logging and task spec.")
    parser.add_argument("--instruction", default="", help="Optional explicit task instruction override.")
    parser.add_argument("--agent-id", default="agi_world_v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-type", default="accessibility_tree")
    parser.add_argument("--current-viewport-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=720)
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="none",
        choices=["none", "minimax", "ollama"],
        help="Optional LLM provider used as an advisory frontend or shadow observer.",
    )
    parser.add_argument("--token-file", default=None, help="Optional token file for the selected LLM provider.")
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional base URL for a local/Ollama-compatible LLM service.",
    )
    parser.add_argument("--llm-model", default=None, help="Optional model name for the selected LLM provider.")
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="integrated",
        choices=["integrated", "shadow", "analyst", "final_candidate"],
        help="Whether the LLM may advise integrated frontends, run in shadow mode, analyze post-action state without execution authority, or only contribute final-stage structured-answer candidates.",
    )
    parser.add_argument("--save-audit", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    llm_client = build_llm_client(
        args.llm_provider,
        token_file=str(args.token_file) if args.token_file is not None else None,
        base_url=args.llm_base_url,
        model=args.llm_model,
    )

    audit = run_webarena_task(
        config_file=args.config_file,
        task_id=args.task_id,
        instruction=args.instruction,
        agent_id=args.agent_id,
        run_id=args.run_id,
        max_episodes=args.max_episodes,
        max_ticks_per_episode=args.max_ticks,
        seed=args.seed,
        verbose=args.verbose,
        headless=args.headless,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        llm_client=llm_client,
        llm_mode=args.llm_mode,
    )

    summary = summarize_audit(audit)
    summary["llm_provider"] = args.llm_provider
    summary["llm_mode"] = args.llm_mode
    summary["token_file"] = str(args.token_file) if args.token_file is not None else None
    summary["llm_base_url"] = args.llm_base_url
    summary["llm_model"] = args.llm_model
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    if args.save_audit:
        task_id = str(audit.get("webarena_task_id", "") or args.task_id or args.config_file or "webarena_task")
        report = WebArenaAuditWriter.build(task_id, audit)
        path = WebArenaAuditWriter.save(args.save_audit, report)
        print(f"saved_audit={path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
