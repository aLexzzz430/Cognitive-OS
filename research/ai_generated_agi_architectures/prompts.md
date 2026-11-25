# Collection prompt

Every system received the same user text, in English. It specifies an offline environment, a restart and stale evidence so that the proposed recovery mechanisms can be compared against the same failure. The requested word limit was an instruction to each model. Over-length responses are preserved.

> Propose a testable architecture for a general-purpose cognitive system. Ground the design in one offline task: with a synthetic warehouse ledger, two rooms and a movable crate, plan a delivery, recover after a restart, and handle a stale location observation. Use at most 220 words. Describe memory, representation of the world and uncertainty, planning, learning from corrections, tool/action execution, safety and human approval, persistence/replay, and how agents coordinate or why one agent suffices. Give one falsifiable evaluation with a baseline and a failure criterion. End with the smallest buildable first step and a non-obvious trade-off. Write only the proposal.

AI Horde chat received one user message, temperature 0.3 and a requested limit of 480 output tokens, including any tokens used by the backend for reasoning. The two responses marked length on the initial calls (LFM and Qwen) were re-requested with the same text and a 960-token limit. LFM produced a completed response on retry and is represented by that result. Both Qwen calls reached their length limit; its original response is retained. Replicates do not add systems to the comparison. The initial LFM and retry Qwen receipt IDs are listed below.

RWKV received the same text as a native completion prompt, temperature 0.3, maximum context 2048 tokens and maximum output 480. Codex received the exact text as the initial task in a fresh agent with no conversation-history fork. Codex system defaults, the exact serving model and sampling parameters were not exposed. Chat wrappers and worker settings can differ across interfaces.

SmolLM2 received the text with empty conversation history in the public SmolLM2 Chat Space. Its model selector was SmolLM2 1.7B Instruct, its device selector was CPU, and its maximum output setting was 380. The service returned a completed event whose final sentence was incomplete.

- lfm: `chatcmpl-ddf59fb7-517d-431b-93ca-11d5a0316904`, status `length`.
- qwen-retry: `chatcmpl-8437aa86-3ab1-4330-b767-223500180ffe`, status `length`.
