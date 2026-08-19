# kimi-k3

A practical AGI architecture would have 5 layers:

1. **Perception & World Model**: Multimodal encoders feed a predictive latent model that learns causal structure and simulates future states.

2. **Memory**:  
   - **Working memory** for active context and goals.  
   - **Episodic memory** for time-stamped experiences.  
   - **Semantic memory** for compressed knowledge and concepts.  
   - **Procedural memory** for learned skills and policies.  
   Retrieval is hybrid: vector search + symbolic indexing + salience gating.

3. **Reasoning & Planning**: A neural-symbolic core combines chain-of-thought, program synthesis, search, and model-based planning. It uses self-criti