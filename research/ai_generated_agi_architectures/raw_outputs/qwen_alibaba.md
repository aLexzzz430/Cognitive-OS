# AGI Architecture Proposal — Qwen/Alibaba perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of the Qwen model family
> (Alibaba), grounded in public releases (Qwen technical reports, Qwen-Agent
> framework, tool-use benchmark results). Not a verbatim model output. See `sources.md`.

## Thesis
The model is an **operating system kernel** and tools are its syscalls. Build
the strongest open tool-calling models, ship a clean agent framework
(Qwen-Agent), and let the ecosystem compose the AGI. Multilinguality and
modality coverage are first-class because the world isn't English-only text.

## 1. Memory architecture
Hybrid: conversation context for working memory plus a **knowledge-base layer**
(documents, structured data) accessed through the same tool interface as
everything else — memory retrieval is just another tool call. This uniformity
keeps the architecture simple: the model only ever does next-token prediction
and tool calls; "memory systems" are tools.

## 2. Reasoning / planning loop
**ReAct-style loop with hierarchical decomposition**: the planner breaks a
goal into subtasks, each handled by a ReAct (reason-act-observe) loop with
tool access. Planning depth adapts to task complexity. Tool-use is trained
aggressively — Qwen models top tool-calling benchmarks (BFCL) because the
training mix is heavy on function-call trajectories.

## 3. Learning / self-improvement
Standard SFT + RLHF/DPO post-training, with heavy emphasis on **agentic
trajectory data**: the models improve by training on more and better tool-use
episodes. Open release creates a flywheel — community fine-tunes flow back as
data and techniques. Self-improvement is partly outsourced to the ecosystem.

## 4. Tool use and action execution
The crown jewel: **best-in-class open function calling**. Tools declared via
schema, nested calls supported, parallel calls supported. Qwen-Agent provides
routers, parsers, and guardrails as reusable components. Execution sandboxes
for code; confirmation policies for side effects.

## 5. World model / representation layer
Multimodal Qwen-VL variants ground language in vision; the representation is
the fused multimodal transformer. For task worlds (APIs, filesystems), the
"world model" is the tool schema documentation plus observed tool outputs —
the agent learns environments by interacting, with schemas as the prior.

## 6. Safety / governance layer
Standard alignment training plus content-safety classifiers. Open-weights
reality acknowledged: governance emphasis on **responsible release practices**
(usage policies, staged releases) since the harness is the deployer's.

## 7. Evaluation and benchmark strategy
Tool-use benchmarks (BFCL, custom agentic suites) as the north star;
multilingual benchmarks (the models are explicitly evaluated across dozens of
languages); multimodal evals for VL variants. If it can't call tools reliably
in 20 languages, it's not done.

## 8. Persistence / runtime architecture
Stateless models behind a **unified serving runtime** (vLLM-style) with
session state in the harness. The Qwen-Agent framework is the reference
runtime: planners, tools, and memory components composed declaratively.

## 9. Multi-agent / orchestration design
**Role-based agent teams**: planner, researcher, coder, critic — each a Qwen
instance with role prompts and tool subsets, coordinated by the framework's
router. The framework (not the model) owns orchestration logic, keeping models
interchangeable.

## 10. Engineering feasibility
Very high: everything is open (weights, framework, benchmarks). A team can
assemble this system from public artifacts this quarter.

## 11. Originality / non-obvious insight
The non-obvious move is **architectural humility**: don't innovate in the
agent loop — innovate in the *model's tool-calling reliability*, then let a
boring, standard loop (ReAct + router) outperform fancier frameworks. Most
agent failures are tool-call failures; fix the model, not the framework.

## 12. Failure modes and adversarial pressures

The kernel-and-syscalls thesis concentrates risk in two places. First,
**tool-schema trust**: if "the world model is the tool schema documentation
plus observed outputs," then a poisoned, stale, or drifted tool schema is a
poisoned world model. Schema provenance, versioning, and change-review become
safety-critical controls, not engineering hygiene. Second, **open-flywheel
monoculture**: when every team assembles from the same public artifacts (same
weights, same framework patterns, same benchmark suites), failures correlate
across deployments — the openness that accelerates progress also synchronizes
vulnerabilities. The mitigation is compositional diversity: keep components
interchangeable (the framework's strength) and deliberately vary the
composition per deployment, so no single failure mode is universal.
