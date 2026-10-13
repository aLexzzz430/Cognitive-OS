# AGI Architecture Proposal — Meta Llama family perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of the Meta Llama
> family, grounded in public materials (Llama 2/3 papers, Llama Stack docs,
> Purple Llama project). Not a verbatim model output. See `sources.md`.

## Thesis
AGI will be **built in the open and run everywhere**. Open weights plus
standardized APIs (Llama Stack) create an ecosystem where intelligence is a
commodity and the value moves to orchestration, safety tooling, and
deployment. The architecture must span datacenter to device.

## 1. Memory architecture
**Local-first memory**: personal memory stores live on the user's device
(on-device vector stores over personal data), synced selectively to cloud.
Working memory is the context window; durable memory is user-owned files and
embeddings. Privacy is architectural: sensitive memory never has to leave the
device.

## 2. Reasoning / planning loop
Standard reason-act-observe loop, with **speculative routing**: a small
on-device model handles easy steps instantly, escalating hard steps to the
large cloud model. Planning happens at the cloud tier; execution at the edge
tier. The loop is cost- and latency-aware by design.

## 3. Learning / self-improvement
Open-weights flywheel: Meta releases strong base models; the community
fine-tunes for every domain; techniques flow back. On-device **personalization
via lightweight adapters** (LoRA) trained on user data locally — the model
improves for *you* without your data leaving your device. Federated learning
as the long-term vision.

## 4. Tool use and action execution
**Llama Stack** standardizes inference, tool calling, safety shields, and
agents behind one API — write once, run on any Llama deployment. Tool use is a
commodity interface; the ecosystem provides the tools.

## 5. World model / representation layer
Multimodal Llama variants (vision) provide grounding; the representation is
the open transformer. The ecosystem bet: specialized community fine-tunes
become the "world models" for specific domains (code, bio, robotics),
composable via routing.

## 6. Safety / governance layer
**Purple Llama**: open safety tooling — Llama Guard (input/output classifiers),
CyberSecEval, Code Shield. Safety as *shared infrastructure* rather than each
deployer's burden. The philosophy: open models need open safety tools, and
responsible release (staged, with evals published) is the governance mechanism.

## 7. Evaluation and benchmark strategy
Standard open benchmarks plus **safety evals as public goods** (the Purple
Llama suites). Evals must run on open weights so anyone can verify claims —
reproducibility is the anti-hype mechanism.

## 8. Persistence / runtime architecture
**Hybrid edge-cloud**: persistent agents live partly on-device (personal
context, quick reactions) and partly in cloud (heavy reasoning, shared
knowledge). Llama Stack is the portability layer — the same agent code runs in
both places. Session state syncs through user-controlled stores.

## 9. Multi-agent / orchestration design
Ecosystem orchestration: different open models (and fine-tunes) as
specialists, coordinated through standard APIs. The orchestrator is itself an
open component. No proprietary protocol lock-in.

## 10. Engineering feasibility
High for the architecture; the constraint is **who trains the frontier open
model** (only Meta-scale actors). But once released, deployment feasibility is
the best in class — from phones to datacenters.

## 11. Originality / non-obvious insight
The strategic insight isn't technical: **openness is the safety strategy**.
Concentrated closed AGI is the risk scenario; ubiquitous open intelligence
with shared safety tooling is the mitigation. Architecturally, the
edge-cloud split with local-first memory is the most privacy-preserving path
to personal AGI.
