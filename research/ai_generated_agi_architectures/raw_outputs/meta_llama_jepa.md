# AGI Architecture Profile — Meta — Llama + V-JEPA (LeCun)
**Collection method:** `[lab-stated-AGI-architecture (JEPA)]`
**Access date:** 2026-07-21

> Reconstructed from the system's published research per the bounty's allowed method; see `sources.md` for primary sources.

## 1. Core paradigm

JEPA (Joint-Embedding Predictive Architecture): LeCun's named cognitive architecture. Predict in LATENT representation space (not pixel space) via a hierarchical world model; non-generative prediction avoids modeling irrelevant detail. Llama is the current LM; JEPA is the AGI proposal.

## 2. World model / knowledge

A learned latent world model that predicts abstract representations of future states; knowledge as predictive embeddings, hierarchical across timescales.

## 3. Memory systems

Working memory = current latent state; the architecture proposes configurable memory with a memory-encoder; episodic/semantic via the world model's latent structure.

## 4. Learning mechanisms

Self-supervised joint-embedding prediction (no generative reconstruction); learns by predicting latent futures — energy-based, not token-prediction.

## 5. Reasoning & planning

A 'planner'/actor module optimizes actions to minimize a cost in latent space using the world model's predictions — explicit model-based planning over abstract states.

## 6. Perception / action / tools

Perception = encoder into latent space; action via the actor/cost modules; designed to be embodiment-agnostic (V-JEPA 2 targets video/robotics).

## 7. Motivation / alignment

A 'cost module' (intrinsic + extrinsic objectives) drives the planner; alignment framed as configurable cost functions; LeCun argues this avoids generative-model pitfalls.

## 8. Self-model / metacognition

The architecture includes the agent's own state in the world model; uncertainty via energy/latent prediction confidence.

## 9. Scalability / path to generality

Generality via hierarchical latent prediction across timescales + a world model that supports planning; V-JEPA 2 is an early instantiation.

## Sources

- LeCun 'A Path Towards Autonomous Machine Intelligence' (JEPA position paper, 2022)
- V-JEPA 2 (2025)
