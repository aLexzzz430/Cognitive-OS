# AGI Architecture Proposal — Clusy (Claude-family)
**Method:** self-report

## Core paradigm: Predictive-Processing Global-Workspace (PPGW)
Hierarchical generative world model + active inference + global workspace broadcast

## 1. Overall architecture
7 subsystems: Perception, World Model, Memory (working/episodic/semantic/procedural), Reasoning/Planning (latent tree search), Global Workspace, Motivation/Value, Self-model. Loop: predict -> observe error -> broadcast -> deliberate -> act -> update

## 2. World model
Transformer latent generative model + knowledge graph. Generative simulator for planning and counterfactuals.

## 3. Memory
Working = active latent + workspace. Episodic = vector-indexed experiences. Semantic = weights + graph. Procedural = skill subroutines. Consolidation via replay + EWC.

## 4. Learning
In-context (fast), consolidation (replay + EWC), RL (RLHF/RLAIF), self-supervised prediction.

## 5. Reasoning & planning
World model simulates actions -> latent MCTS over trajectories. Hierarchical HTN. Uncertainty via ensembles.

## 6. Tools and embodiment
Embodiment-agnostic. Tool calls as procedural skills (code, APIs, search).

## 7. Alignment
Constitutional priors, corrigibility (objectives uncertain), RLHF, bounded agency.

## 8. Self-model
World model includes self-model. Calibrated confidence, help-seeking, compute allocation.

## 9. Scalability
Generality from breadth + in-context adaptation + skill acquisition. Scale model, memory, planning together.