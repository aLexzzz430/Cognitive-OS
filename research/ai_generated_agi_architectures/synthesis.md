# Synthesis: Combined AGI Architecture

| Component | Source |
|-----------|-------|
| Latent-prediction world model | Meta JEPA |
| Model-based RL + tree search | DeepMind Alpha |
| RL-trained deliberation | DeepSeek R1 / OpenAI o |
| Global workspace | Clusy PPGW |
| Episodic/semantic/procedural memory | PPGW + JEPA |
| Active inference | PPGW |
| Hierarchical goals | PPGW + DeepMind |
| Constitutional alignment + corrigibility | PPGW + OpenAI |
| Self-model + metacognition | PPGW + DeepMind |

**Flow:** Senses → JEPA latent encoder → World model → Global workspace → Planner (latent search + RL deliberation) → Skills → Tools → Prediction error → Update model + memory.

Layered alignment: constitutional priors + value uncertainty + corrigibility guide reasoning.