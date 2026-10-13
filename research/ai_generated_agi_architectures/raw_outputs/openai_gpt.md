# AGI Architecture Proposal — OpenAI GPT family perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of the OpenAI model
> family, grounded in public technical reports (GPT-4, o1/o3 system cards, Agents
> SDK docs). Not a verbatim model output. See `sources.md`.

## Thesis
Reasoning is the substrate; everything else is scaffolding around it. Train a
model to think longer and better via reinforcement learning on verifiable tasks,
then wrap it in a thin, reliable agent harness: tools, memory retrieval, and
evals. General intelligence emerges from scaling the reasoning loop, not from
hand-building cognitive modules.

## 1. Memory architecture
Two tiers. **Working memory** is the conversation context itself — long and
durable, with the model trained to manage it (summarization is a learned skill,
not a separate module). **Long-term memory** is retrieval-based: vector stores
over past conversations, documents, and tool outputs, fetched with learned
retrieval policies. No symbolic memory graph; embeddings plus good chunking
outperform hand-designed schemas at scale. The key bet: memory quality is a
retrieval-ranking problem, and RL on end-task success trains the ranker.

## 2. Reasoning / planning loop
The reasoning model (o-series style) generates an internal chain-of-thought,
then acts. Planning is **implicit and iterative**: propose a step, execute via
tool, observe, re-plan — the "reason-act-observe" loop, with the number of
allowed iterations scaled at test time. For hard tasks, sample multiple
trajectories and select by a verifier or majority vote. Deliberative alignment
means safety reasoning happens inside the same chain-of-thought.

## 3. Learning / self-improvement
Post-training is the engine: supervised fine-tuning for format, then RL —
first RLHF on human preferences, then RLVR (RL with verifiable rewards) on
math, code, and agentic tasks where correctness is checkable. Self-improvement
comes from **RL on self-generated rollouts filtered by verifiers**, plus
distillation of strong reasoners into smaller models. The flywheel: better
reasoner → better synthetic data → better reasoner.

## 4. Tool use and action execution
Tools are first-class: a unified function-calling interface, with the model
trained (via RL on tool-use trajectories) to chain calls reliably. Computer
use (Operator-style) treats the GUI as a universal tool. Execution is
sandboxed; high-stakes actions require confirmation policies learned from
human-feedback data.

## 5. World model / representation layer
No separate world-model module. The world model is **implicit in the language
model's weights**, continuously sharpened by training on code, tool
observations, and multimodal data. Code execution is the grounding mechanism:
the model tests its beliefs by running code. For robotics/embodied tasks, video
prediction objectives are added — but still inside one model.

## 6. Safety / governance layer
Layered: (a) training-time alignment (RLHF, deliberative alignment,
instruction hierarchy — system > developer > user); (b) deployment-time
moderation classifiers and monitoring; (c) evals as the governance backbone —
a model ships only when it passes the preparedness eval suite. Red-teaming is
continuous and automated.

## 7. Evaluation and benchmark strategy
Evals are the product spec. Internal benchmark suites per capability (reasoning,
coding, agentic tool use like SWE-bench/τ-bench, safety), plus human
preference evals. A new capability is "done" when the eval curve saturates.
External benchmarks are treated as lagging indicators.

## 8. Persistence / runtime architecture
Stateful sessions: the runtime persists conversation state, tool state, and
retrieved memory server-side; the model itself stays stateless per request.
Long-running agents run as resumable jobs with checkpointed trajectories, so a
crashed run restarts from the last verified step.

## 9. Multi-agent / orchestration design
Supervisor–worker via handoffs (Agents SDK pattern): a triage agent routes to
specialists, each a configured instance of the same base model with different
instructions and tools. Guardrails as separate lightweight agents. No exotic
protocols — plain message passing with structured outputs.

## 10. Engineering feasibility
Highest feasibility of any proposal here: every component exists in production
today. The cost is compute (training + long-context inference), not research
risk. The main open problem is reliability of long tool chains, addressed by
verifiers and RL.

## 11. Originality / non-obvious insight
The non-obvious bet is **negative**: don't build memory graphs, planners, or
world-model modules. Every hand-built cognitive module tried so far has been
outperformed by a bigger reasoner with tools. The architecture is deliberately
boring so the scaling curve can do the work.
