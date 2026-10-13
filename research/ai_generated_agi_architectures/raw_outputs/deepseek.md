# AGI Architecture Proposal — DeepSeek perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of DeepSeek, grounded
> in public technical reports (DeepSeek-V3, DeepSeek-R1 papers). Not a verbatim
> model output. See `sources.md`.

## Thesis
Intelligence per FLOP is the metric that matters. Sparse architectures
(fine-grained Mixture-of-Experts), compressed attention (MLA), and critic-free
RL (GRPO) deliver frontier reasoning at a fraction of the cost — and open
weights let everyone build on it. AGI will be **efficient and open**, not just
big and closed.

## 1. Memory architecture
Memory efficiency starts in the architecture: **Multi-head Latent Attention
(MLA)** compresses the KV cache dramatically, so long contexts cost far less.
Beyond that, standard retrieval augmentation over vector stores. The
philosophy: don't add memory modules — make the attention itself cheap enough
that the context window can *be* the memory.

## 2. Reasoning / planning loop
**Long chain-of-thought as the reasoning engine**, trained via RL (not just
SFT): DeepSeek-R1 showed that RL on verifiable rewards (math, code) produces
emergent reasoning behaviors — self-verification, backtracking, reflection —
without distilling them from a teacher. Planning = extended CoT with tool
calls interleaved; test-time scaling (longer CoT) directly improves results.

## 3. Learning / self-improvement
**GRPO (Group Relative Policy Optimization)**: RL without a separate critic
model — advantages computed from grouped rollouts, drastically cheaper than
PPO. Train on verifiable-reward domains, then **distill the reasoning patterns
into smaller dense models** so the capability propagates down the cost curve.
Self-improvement = RL → distill → RL again, each cycle cheaper than the last.

## 4. Tool use and action execution
Tool use trained jointly with reasoning: the CoT learns *when* a tool call
beats more thinking. Function calling is a standard interface; the emphasis is
on training data (agentic trajectories with verifiable outcomes) rather than
fancy protocols.

## 5. World model / representation layer
Implicit in the model, as with others — but with a twist: because training is
cheap, you can afford **specialist world models** (a code-execution-grounded
model, a math model) and route between them, rather than forcing one giant
model to represent everything.

## 6. Safety / governance layer
Standard alignment stack (SFT + RLHF-style preference training), with safety
evals before release. Open weights mean safety must be *in the weights* to a
greater degree, since deployers control the harness. The proposal is honest
that open release trades some control for broad access.

## 7. Evaluation and benchmark strategy
Verifiable-reward benchmarks (math, code, logic) as the training signal;
standard public benchmarks (MMLU-style, HumanEval, agentic) as the scoreboard.
The methodology contribution: **RL training curves on verifiable tasks are
the eval** — if the RL reward keeps climbing, capability is still scaling.

## 8. Persistence / runtime architecture
Stateless model, stateful harness — but optimized for **throughput economics**:
MoE with few active params per token + MLA = cheap long-context inference.
This makes persistent, always-on agents economically viable where dense
frontier models would be prohibitive.

## 9. Multi-agent / orchestration design
**Sparse expert routing as proto-multi-agent**: the MoE already routes tokens
to specialist experts; extend the metaphor — a system of fine-tuned specialist
models (reasoner, coder, planner) orchestrated by a lightweight router, each
cheap to run. No heavyweight agent framework needed.

## 10. Engineering feasibility
The highest feasibility-per-dollar of any proposal: open weights, published
methods, modest hardware requirements (relative). A competent team can
reproduce the core loop today.

## 11. Originality / non-obvious insight
Two insights: (1) **the critic is optional** — GRPO shows RL works without
learned value models, removing a huge complexity/cost center; (2) **reasoning
is distillable** — long-CoT RL behaviors transfer into small models via SFT,
so the frontier's job is to *discover* reasoning strategies that the whole
ecosystem then inherits cheaply.
