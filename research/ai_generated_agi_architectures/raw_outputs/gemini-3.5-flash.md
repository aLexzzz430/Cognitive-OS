# gemini-3.5-flash

AGI architecture:

- **Core loop:** Perception → world model update → goal selection → reasoning/planning → action → feedback.
- **Memory:**  
  - *Working memory* for current context and task state.  
  - *Episodic memory* to store time-stamped experiences and outcomes.  
  - *Semantic memory* for compressed facts, concepts, and rules.  
  - *Procedural memory* for learned skills and policies.  
  Retrieval is relevance- and confidence-based, with continual consolidation and forgetting.
- **Reasoning:**  
  - Neuro-symbolic planner combining neural pattern recognition with symbolic logic.  
  - Causal/world-model simulation for counterfactuals and long-horizon planning.  
  -