# deepseek-v4

A complete AGI architecture can be organized into five layers:

1. **Perception & Input**: Multimodal encoders for text, vision, audio, and actions convert raw data into shared embeddings.

2. **Memory**:  
   - **Working memory**: short-term context buffer for current tasks.  
   - **Episodic memory**: event store with timestamps, goals, outcomes, and retrieval by similarity.  
   - **Semantic memory**: compressed knowledge graph and learned world model.  
   - **Procedural memory**: reusable skills, policies, and tool-use routines.

3. **Reasoning & Planning**: A hybrid system combining a large foundation model, symbolic planner, causal simulator, and search. It supports de