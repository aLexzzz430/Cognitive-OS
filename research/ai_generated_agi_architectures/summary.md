# Summary: Common Patterns, Disagreements, and Notable Ideas

Across ten AGI architecture proposals (reconstructed in each model family's
documented philosophy — see `sources.md` for provenance disclosure), this
document synthesizes what the field agrees on, where it genuinely disagrees,
and which ideas deserve attention from Cognitive-OS planning.

## 1. Common patterns (8+/10 agreement)

**The reason-act-observe loop is universal.** Every single proposal centers a
loop of: think, act via tools, observe results, continue. Names differ
(ReAct, agentic loop, interleaved thinking-and-acting, query-decomposition),
but the structure is identical. This is the strongest consensus in the packet:
AGI agency = iterated tool-mediated action, not single-shot generation.

**Tools are the action layer, and function-calling is the interface.**
10/10 proposals use schema-described function/tool calling as the execution
primitive. MCP (Anthropic), Agents SDK handoffs (OpenAI), Qwen-Agent routers,
and Llama Stack all converge on: typed tool schemas + model selects calls.
The interface war is effectively over.

**Verification is a first-class step.** 9/10 explicitly include a verify/check
phase after acting (Anthropic makes it architectural; OpenAI uses verifiers;
Perplexity uses citation checking). Nobody trusts open-loop execution.

**RL on verifiable outcomes is the learning engine.** 8/10 proposals put
reinforcement learning on checkable tasks (math, code, agentic completion) at
the center of improvement — RLVR, GRPO, constitutional RL, agentic-trajectory
RL. Pure supervised imitation is considered saturated.

**Stateless models, stateful harnesses.** 9/10 keep the model itself stateless
per request and persist sessions, context, and tool state in the runtime.
Persistence is a systems problem, not a model problem.

**Safety is layered, never single-mechanism.** All ten combine training-time
alignment with deployment-time controls (classifiers, monitoring, policies).
They differ on *emphasis* (see disagreements), not on layering.

## 2. Genuine disagreements

**D1. Where does memory live? (3 camps)**
- *Context-is-memory* (Google, Kimi, partly OpenAI): make the window big
  enough and memory management becomes a learned skill; delete the RAG stack.
- *Retrieval-is-memory* (Perplexity, DeepSeek-pragmatic): the index/store is
  the memory; the model is the reasoner over it.
- *Files-are-memory* (Anthropic, Meta-local): human-inspectable markdown
  files and on-device stores; memory must be legible and user-owned.
- Implication for Cognitive-OS: the repo already has `modules/memory/` and
  `modules/episodic/` — the decision is whether these become vector stores,
  file-backed working notes, or context-management policies. The packet
  suggests: file-backed working notes (inspectable, matches "evidence-governed"
  ethos) + retrieval for scale.

**D2. Modules vs. scaling (the big philosophical split)**
- *Just scale reasoning* (OpenAI, Kimi, DeepSeek): hand-built cognitive
  modules get outperformed; put the capability in the weights via RL.
- *Engineer the scaffolding* (Anthropic, Google-evolutionary): context
  engineering, evolutionary improvement loops, and careful harness design are
  where the gains are.
- *Efficiency as strategy* (Mistral, DeepSeek): the winner is whoever runs
  cheapest, not whoever reasons deepest.
- Cognitive-OS is a *runtime* project, so it cannot fully take the "just
  train" position — but it should keep modules thin and eval-driven, letting
  the model do maximal work.

**D3. Safety philosophy**
- *Train it in* (Anthropic constitutional, xAI truthfulness): the model's own
  judgment is the strongest layer.
- *Govern it externally* (Google thresholds, Meta Purple Llama tooling):
  staged evals, classifiers, and shared safety infrastructure.
- *Ground it* (Perplexity): citations and evidence requirements structurally
  prevent the most dangerous failure (confident falsehood).
- All three are compatible; Cognitive-OS's `modules/governance/` should
  implement all three layers.

**D4. Open vs. closed as strategy**
- Meta/Mistral/DeepSeek/Qwen/Kimi: openness is distribution, safety
  (many eyes), and the ecosystem flywheel.
- OpenAI/Anthropic/Google: frontier capabilities need staged, controlled
  deployment.
- For Cognitive-OS (local-first, open repo): the open camp's arguments apply
  directly — local-first + open weights + user-owned memory is the coherent
  position.

**D5. World models: implicit vs. explicit**
- 7/10 keep the world model implicit in weights + tool interaction.
- Google (video prediction) and Perplexity (live index + on-demand KGs) argue
  for more explicit predictive representations.
- Cognitive-OS already has `core/world_model/` — the packet suggests keeping
  it as a *tool-observation-grounded* state tracker rather than a learned
  simulator, with explicit prediction only where verifiable (e.g., file-system
  effects of planned actions).

## 3. Notable ideas worth stealing

1. **Prose-level memory (Anthropic):** agent-maintained markdown notes beat
   vector stores for inspectability. Maps directly to Cognitive-OS's
   evidence-governed design — memory entries as *evidence* with provenance.
2. **Evolutionary self-improvement (Google/AlphaEvolve):** LLM-proposed
   mutations to the agent's own code/harness, scored by evals. Maps to
   Cognitive-OS's `evolution/` directory — this packet gives it a concrete
   loop design.
3. **Critic-free RL (DeepSeek/GRPO):** removes the value-model complexity
   from RL training. Relevant if Cognitive-OS ever trains its own policies.
4. **Subagents with blast-radius containment (Anthropic):** isolated contexts
   as a *safety* pattern, not just a capability pattern. Maps to
   `core/orchestration/`.
5. **Citation-grounded outputs (Perplexity):** every consequential claim
   traceable to evidence. This *is* Cognitive-OS's stated ethos
   ("evidence-governed") — the packet validates the positioning and suggests
   making it architectural (every action logged with its evidence).
6. **Efficiency-adjusted evals (Mistral):** score capability per FLOP/euro.
   Cognitive-OS is local-first; it should adopt cost-aware evals natively.
7. **Assumption tracking (xAI):** plan steps list their assumptions for
   challenge. Cheap to implement in `planner/`, high value for reliability.
8. **Long-horizon safety evals (Kimi):** does the agent stay on-task over 50+
   steps? Standard benchmarks don't test this; Cognitive-OS should.

## 4. What's missing from all ten

- **Formal verification of plans** before execution (beyond testing).
- **Economic/multi-agent game theory** for open multi-agent deployments.
- **Graceful degradation** design: what the system does as capabilities or
  tools fail mid-task (partially covered by `modules/recovery/` in
  Cognitive-OS — ahead of the field here).
- **Inter-agent trust protocols** for agents built by different parties.
