# AGI Architecture Profile — OpenAI — GPT / o-series
**Collection method:** `[reconstructed-from-publications]`
**Access date:** 2026-07-21

> Reconstructed from the system's published research per the bounty's allowed method; see `sources.md` for primary sources.

## 1. Core paradigm

Scale a transformer language model, then train reasoning with reinforcement learning on chain-of-thought (the o-series 'deliberative alignment' recipe). No named cognitive architecture; the bet is scale + RL-on-reasoning.

## 2. World model / knowledge

Knowledge is implicit in transformer weights; no explicit world model or KG. o-series emits long internal reasoning traces as a scratchpad.

## 3. Memory systems

Working memory = context window. No native persistent episodic/semantic store (added externally via retrieval/tools).

## 4. Learning mechanisms

Pretraining (self-supervised next-token) + RLHF + RL on verifiable rewards for reasoning (GRPO-style). In-context learning at inference; no continual weight updates in deployment.

## 5. Reasoning & planning

Test-time compute: long RL-trained chain-of-thought 'thinking' before answering; tool use for planning. Search is implicit in the learned reasoning policy rather than explicit tree search.

## 6. Perception / action / tools

Multimodal (text/image/audio in, text out); strong tool/function calling and code execution as the action interface.

## 7. Motivation / alignment

RLHF + 'deliberative alignment' (model reasons about its instructions/values explicitly). Objectives set by trainers; corrigibility via training, not architectural.

## 8. Self-model / metacognition

Limited explicit self-model; some calibrated confidence via sampling; reasoning traces give partial introspection.

## 9. Scalability / path to generality

Generality via scaling laws + RL on ever-harder reasoning tasks; capability grows with parameters/data/test-time compute.

## Sources

- OpenAI o-series / deliberative alignment (2024-25)
- Sam Altman 'Reflections' blog (2025)
