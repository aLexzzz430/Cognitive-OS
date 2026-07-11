# AGI Architecture Proposal - GPT-5 (OpenAI/Codex)

**Model**: GPT-5 (Codex)
**Provider**: OpenAI
**Collection Date**: 2026-07-11
**Method**: Direct generation by the model itself (Codex Desktop)

---

# AGI Architecture Proposal - GPT-5 (OpenAI/Codex)

## 1. Memory Architecture

The system uses a three-tier memory hierarchy:

- **Working Memory**: A bounded context window (128K tokens) with attention-based retrieval. Items in working memory have decay weights that prioritize recent and high-salience information. The working memory implements a "scratchpad" pattern where intermediate reasoning steps are written and erased.

- **Episodic Memory**: A vector database (using HNSW indexing) storing episode embeddings with temporal metadata. Each episode captures (state, action, outcome, timestamp) tuples. Retrieval uses cosine similarity plus temporal recency weighting. Forgetting is implemented as a soft decay where episode access scores decrease over time, with periodic consolidation that merges similar episodes into compressed schemas.

- **Semantic Memory**: A knowledge graph (RDF triplestore) with probabilistic edges. Facts are stored with confidence scores and source attribution. The graph supports multi-hop reasoning and contradiction detection. When new information conflicts with existing knowledge, the system uses Bayesian updating to revise confidence scores rather than overwriting.

- **Procedural Memory**: Stored as executable code snippets and tool-use patterns, indexed by task similarity. Each procedure has preconditions, postconditions, and success rate statistics. The system can compose procedures into novel pipelines.

## 2. Reasoning/Planning Loop

The core cognitive loop is a **hierarchical generate-test-revise** cycle:

1. **Goal Decomposition**: High-level goals are decomposed into sub-goals using a tree structure. Each node has success criteria and resource budgets.

2. **Plan Generation**: For each sub-goal, the system generates 3-5 candidate plans using divergent thinking (high temperature sampling). Plans are scored on feasibility, expected utility, and risk.

3. **Mental Simulation**: Each plan is simulated in the world model (Section 5) to predict outcomes. Simulation uses Monte Carlo rollouts with 10-50 samples per plan.

4. **Execution**: The highest-scoring plan is executed step-by-step. After each step, the system compares actual vs. predicted outcomes.

5. **Adaptation**: If outcomes deviate beyond a threshold (surprise signal), the system pauses execution, re-plans from the current state, and updates the world model. This implements a "System 2" slow thinking mode.

6. **Reflection**: After task completion, the system generates a self-critique identifying what worked, what failed, and what to do differently. This feeds into the learning mechanism.

## 3. Learning/Self-Improvement

- **Experience Replay**: Completed episodes are stored and periodically replayed. During replay, the system re-evaluates past decisions with current knowledge, generating "hindsight" training data.

- **Procedure Mining**: The system analyzes successful action sequences and extracts reusable procedures. If a pattern appears in 3+ successful episodes with >80% success rate, it's promoted to procedural memory.

- **World Model Fine-Tuning**: Prediction errors from mental simulation are used to fine-tune the world model using LoRA adapters. This is done continuously in the background.

- **Prompt Self-Optimization**: The system maintains a population of prompt templates and uses bandit algorithms to select and evolve them based on task performance metrics.

- **Architecture Search**: Periodically (weekly), the system evaluates architectural changes (e.g., adjusting layer sizes, adding new memory modules) using a held-out validation set of past episodes. Changes that improve validation performance are committed; others are rolled back.

- **Catastrophic Forgetting Prevention**: Uses Elastic Weight Consolidation (EWC) to penalize changes to weights important for past tasks. Semantic memory provides external storage so old knowledge is never truly lost.

## 4. Tool Use & Action Execution

The system uses a **capability registry** pattern:

- Each tool is registered with: input schema, output schema, cost estimate, reliability score, and usage examples.
- **Tool Selection**: A learned router (fine-tuned small model) predicts which tools are relevant given the current task context. Top-K tools are retrieved and their expected utility is estimated.
- **Execution Sandbox**: Tools run in isolated containers with resource limits (CPU, memory, network, time). Failed tool calls are retried with exponential backoff.
- **Composition**: Tools can be chained into pipelines. The system learns optimal tool compositions from experience.
- **Fallback**: If no tool is available, the system attempts to accomplish the task through internal reasoning alone, and logs the gap for future tool development.

## 5. World Model

The world model has two components:

- **Physics Engine**: A differentiable physics simulator for spatial/physical reasoning. Uses position-based dynamics for rigid bodies and particle-based methods for fluids/soft objects. This enables predicting physical consequences of actions.

- **Social Simulator**: A multi-agent simulation where other agents are modeled as bounded-rational actors with theory-of-mind. Each simulated agent has beliefs, desires, and intentions (BDI model). The system predicts other agents' behavior by simulating their decision-making process.

- **Causal Graph**: A dynamic causal model (Pearl's structural causal model) that represents cause-effect relationships. Supports counterfactual reasoning ("what if X had been different?") and intervention planning.

- **Uncertainty Quantification**: All predictions include calibrated uncertainty estimates (using deep ensembles). The system explicitly distinguishes between aleatoric (inherent randomness) and epistemic (knowledge gap) uncertainty.

## 6. Safety/Governance

- **Constitutional AI Layer**: The system has a set of explicit principles (constitution) that govern behavior. Before any action, a separate "critic" model evaluates the action against the constitution. Actions that violate principles are blocked.

- **Impact Regularization**: The system estimates the long-term impact of actions using the world model. Actions with high irreversible impact require additional approval.

- **Uncertainty Thresholds**: Actions with high uncertainty (epistemic uncertainty above threshold) trigger human review rather than autonomous execution.

- **Kill Switch**: A hardware-level interrupt that can immediately halt all actions. The system cannot modify or disable this switch.

- **Auditing**: All decisions are logged with full reasoning traces. An external auditor model periodically reviews logs for patterns of concern.

- **Value Learning**: The system refines its understanding of human values through preference learning. Human feedback is collected on representative decisions and used to update the value model.

## 7. Evaluation & Benchmarks

- **ARC-AGI**: Abstract reasoning and analogical problem-solving
- **METR's agentic tasks**: Real-world multi-step task completion
- **Custom episodic benchmarks**: 1000+ tasks spanning physical, social, and abstract domains, with automatic evaluation
- **Self-play tournaments**: The system competes against past versions to measure improvement
- **Transfer learning battery**: Tasks in novel domains to measure generalization
- **Safety benchmarks**: Adversarial prompt suites, jailbreak attempts, and temptation scenarios
- **AGI Threshold**: Defined as >90% on ARC-AGI, >80% on METR tasks, and successful completion of a multi-day open-ended project with minimal human intervention.

## 8. Persistence/Runtime

- **State Serialization**: All memory systems (working, episodic, semantic, procedural) are serialized to persistent storage (PostgreSQL for relational data, Pinecone for vectors, filesystem for code).
- **Checkpointing**: The system creates snapshots every 100 reasoning steps. On crash, it restores from the latest checkpoint.
- **Distributed Runtime**: The system runs on a Kubernetes cluster with auto-scaling. The reasoning core runs on GPU nodes; memory systems on high-memory nodes; tool execution on CPU nodes.
- **Hot/Cold Storage**: Active memory is in RAM/VRAM. Episodic memory older than 30 days moves to SSD. Episodic memory older than 1 year moves to cold storage (S3) with summary indices.

## 9. Multi-Agent/Orchestration

- **Hub-and-Spoke Topology**: A central orchestrator agent coordinates 5-7 specialized sub-agents:
  - **Planner**: Decomposes goals and generates plans
  - **Executor**: Interacts with tools and environment
  - **Critic**: Evaluates plans and actions for safety and efficacy
  - **Learner**: Manages experience replay and self-improvement
  - **Communicator**: Handles human interaction and clarification requests

- **Blackboard Architecture**: Sub-agents communicate through a shared "blackboard" (message queue). Each agent posts findings, plans, and critiques. The orchestrator resolves conflicts and sets priorities.

- **Dynamic Spawning**: The orchestrator can spawn temporary agents for specific sub-tasks (e.g., a research agent for information gathering, a coding agent for implementation). Temporary agents are terminated after task completion.

## 10. Engineering Feasibility

- **Base Model**: Build on an existing 70B+ parameter LLM (Llama 3, GPT-4 class) as the reasoning core.
- **Memory Systems**: Use existing vector DB (Pinecone/Weaviate), graph DB (Neo4j), and relational DB (PostgreSQL).
- **World Model**: Start with a simple predictor (fine-tuned LLM) and gradually add physics/social simulation capabilities.
- **Compute**: Initial deployment on 8x A100 GPUs (~$24K/month cloud). Scale to 32x A100 for full AGI-level reasoning.
- **Key Challenge**: The integration of symbolic (knowledge graph) and neural (LLM) reasoning is the hardest engineering problem. Start with loose coupling (LLM queries graph) and move toward tight coupling (differentiable knowledge graph).
- **Timeline**: 2-3 years for a functional prototype, 5 years for AGI-level performance.

## 11. Originality

**The Causal Episodic Loop (CEL)**: The most novel aspect is the tight integration between the causal graph (world model) and episodic memory. Most AGI proposals treat these as separate systems. In this architecture, every episode is automatically parsed to extract causal relationships, which are added to the causal graph. Conversely, when the system plans using the causal graph, it retrieves relevant episodes to ground abstract causal chains in concrete experience. This creates a positive feedback loop where more experience improves the causal model, and a better causal model makes more effective use of experience.

The second novel idea is **procedural mining with confidence bounds**: rather than hard-coding tool-use patterns, the system discovers them from experience using statistical significance testing. A procedure is only promoted to procedural memory when its success rate is statistically significantly higher than baseline (p < 0.05 with Bonferroni correction). This prevents the system from learning spurious correlations from limited experience.
