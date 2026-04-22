from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from core.main_loop import CoreMainLoop
from integrations.arc_agi3.audit import ARCAGI3AuditWriter, summarize_audit
from integrations.arc_agi3.task_adapter import ARCAGI3SurfaceAdapter
from modules.llm import build_llm_client


def run_arc_agi3_game(
    game_id: str,
    *,
    agent_id: str = "agi_world_v2",
    run_id: Optional[str] = None,
    max_episodes: int = 1,
    max_ticks_per_episode: int = 128,
    seed: int = 0,
    verbose: bool = False,
    operation_mode: str = "ONLINE",
    arc_api_key: Optional[str] = None,
    arc_base_url: Optional[str] = None,
    environments_dir: Optional[str] = None,
    recordings_dir: Optional[str] = None,
    render_mode: Optional[str] = None,
    llm_client: Any = None,
    llm_mode: str = "integrated",
) -> Dict[str, Any]:
    world = ARCAGI3SurfaceAdapter(
        game_id=game_id,
        operation_mode=operation_mode,
        arc_api_key=arc_api_key,
        arc_base_url=arc_base_url,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        render_mode=render_mode,
        seed=seed,
    )
    loop = CoreMainLoop(
        agent_id=agent_id,
        run_id=run_id or f"arc-agi3-{game_id}",
        seed=seed,
        max_episodes=max_episodes,
        max_ticks_per_episode=max_ticks_per_episode,
        verbose=verbose,
        world_adapter=world,
        llm_client=llm_client,
        llm_mode=llm_mode,
        world_provider_source="integrations.arc_agi3.runner",
    )
    audit = loop.run()
    audit["arc_game_id"] = game_id
    audit["arc_scorecard"] = world.scorecard()
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AGI_WORLD_V2 directly on ARC-AGI-3 through a native bridge.")
    parser.add_argument("--game", required=True, help="ARC-AGI-3 game id, e.g. ls20 or ls20-<version>.")
    parser.add_argument("--agent-id", default="agi_world_v2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--mode", default="ONLINE", help="ONLINE / OFFLINE / NORMAL")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arc-api-key", default=None)
    parser.add_argument("--arc-base-url", default=None)
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--recordings-dir", default=None)
    parser.add_argument("--render-mode", default=None)
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="none",
        choices=["none", "minimax", "ollama"],
        help="Optional LLM provider used as an advisory frontend or shadow observer.",
    )
    parser.add_argument(
        "--token-file",
        default=None,
        help="Optional token file for the selected LLM provider.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Optional base URL for a local/Ollama-compatible LLM service. Default: OLLAMA_BASE_URL or http://127.0.0.1:11435.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional model name for the selected local LLM provider.",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="integrated",
        choices=["integrated", "shadow", "analyst", "final_candidate"],
        help="Whether the LLM may advise integrated frontends, run in shadow mode, analyze post-action state without execution authority, or only contribute final-stage structured-answer candidates.",
    )
    parser.add_argument("--save-audit", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    llm_client = build_llm_client(
        args.llm_provider,
        token_file=str(args.token_file) if args.token_file is not None else None,
        base_url=args.llm_base_url,
        model=args.llm_model,
    )

    audit = run_arc_agi3_game(
        game_id=args.game,
        agent_id=args.agent_id,
        run_id=args.run_id,
        operation_mode=args.mode,
        max_episodes=args.max_episodes,
        max_ticks_per_episode=args.max_ticks,
        seed=args.seed,
        verbose=args.verbose,
        arc_api_key=args.arc_api_key,
        arc_base_url=args.arc_base_url,
        environments_dir=args.environments_dir,
        recordings_dir=args.recordings_dir,
        render_mode=args.render_mode,
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
        report = ARCAGI3AuditWriter.build(args.game, audit)
        path = ARCAGI3AuditWriter.save(args.save_audit, report)
        print(f"saved_audit={path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
