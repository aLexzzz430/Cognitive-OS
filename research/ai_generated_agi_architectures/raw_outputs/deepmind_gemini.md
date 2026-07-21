# AGI Architecture Profile — Google DeepMind — Gemini / Gato / Alpha-family
**Collection method:** `[lab-stated-AGI-ontology (not a full architecture)]`
**Access date:** 2026-07-21

> Reconstructed from the system's published research per the bounty's allowed method; see `sources.md` for primary sources.

## 1. Core paradigm

Generalist agent paradigm: a single multimodal model (Gato/Gemini) plus model-based RL + search (AlphaGo/AlphaFold lineage). DeepMind published a 'Levels of AGI' ontology (capability levels), which is a framework, not an architecture.

## 2. World model / knowledge

Multimodal transformer world knowledge in weights; Alpha-family uses explicit learned simulators/search over structured state (e.g. AlphaFold structure prediction).

## 3. Memory systems

Context window working memory; external retrieval; Alpha-family uses search over an explicit state representation as a form of episodic lookahead.

## 4. Learning mechanisms

Self-supervised pretraining + RL (model-based RL in Alpha-family; RLHF in Gemini). Strong emphasis on RL + planning.

## 5. Reasoning & planning

Explicit model-based planning + tree search (MCTS lineage from AlphaGo) is DeepMind's signature; Gemini adds learned reasoning.

## 6. Perception / action / tools

Deeply multimodal (Gato: one model across vision/language/robotics/game actions); embodiment is a core focus (robotics, SIMA agent in 3D worlds).

## 7. Motivation / alignment

Reward-driven RL; alignment via RLHF + safety training; objectives from reward design.

## 8. Self-model / metacognition

Not a central published component; some uncertainty estimation in RL agents.

## 9. Scalability / path to generality

Generality via generalist agents + RL + search across domains; the Levels-of-AGI framework defines the path (narrow -> competent -> expert -> virtuoso).

## Sources

- DeepMind 'Levels of AGI' paper (2023)
- Gato generalist agent (2022); AlphaGo/AlphaFold
