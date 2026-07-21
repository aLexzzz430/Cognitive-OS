# AGI Architecture Profile — DeepSeek — V3 / R1
**Collection method:** `[reconstructed-from-publications]`
**Access date:** 2026-07-21

> Reconstructed from the system's published research per the bounty's allowed method; see `sources.md` for primary sources.

## 1. Core paradigm

Efficient open Mixture-of-Experts transformer; R1 adds RL-trained reasoning (GRPO) without heavy supervised fine-tuning. Paradigm: efficient MoE + RL-driven emergent reasoning.

## 2. World model / knowledge

Parametric knowledge in MoE weights; no explicit world model/KG.

## 3. Memory systems

Context window (long-context variants); no native persistent store.

## 4. Learning mechanisms

Pretraining + GRPO reinforcement learning (R1) that elicits long reasoning; open weights enable community fine-tuning.

## 5. Reasoning & planning

R1's RL-trained long chain-of-thought reasoning with self-verification; test-time reasoning compute.

## 6. Perception / action / tools

Primarily text/code; tool use supported; multimodal limited relative to frontier labs.

## 7. Motivation / alignment

RL with reward signals; alignment via RLHF; open-weight transparency.

## 8. Self-model / metacognition

R1 shows emergent self-reflection/'aha' moments in reasoning traces (partial metacognition).

## 9. Scalability / path to generality

Generality via efficient MoE scaling + RL on reasoning; open ecosystem accelerates capability spread.

## Sources

- DeepSeek-V3 technical report (2024)
- DeepSeek-R1 (2025)
