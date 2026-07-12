# AGI Architecture Proposal: Veritas - Causal Truth-Seeking

## Memory Architecture
Working Memory: 64K token buffer with attention-based ranking.
Episodic Memory: Causal-annotated experience replay.
Semantic Memory: Causal knowledge graph with confidence-scored edges.
Procedural Memory: Causal if-then policies with learned effects.

## Reasoning and Planning
Real-time causal inference at 10Hz. Causal tree search in causal space.
Forward and backward causal reasoning.

## Learning
Causal discovery from observation and intervention.
Active experimentation. Online Bayesian updating. Truth-seeking reward.

## Safety
Truth alignment as foundational principle. Causal transparency.
Uncertainty honesty. Online deception monitoring.

## Original Insight
Truth-seeking as primary reward. True beliefs lead to better decisions.
Truth is the universal instrumental goal.
## Technical Details
- Real-time: Sub-100ms causal inference pipeline
- Causal discovery: PC algorithm + NOTEARS for structure learning
- Truth reward: Verified via consistency checks across 10+ random seeds
- Data: Continuous real-time web data with causal annotation
- Infrastructure: 100K H100 cluster with custom networking
