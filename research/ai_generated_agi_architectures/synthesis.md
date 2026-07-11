# Synthesis: Proposed Combined AGI Architecture

## Architecture: Causal Predictive Architecture with Symbolic Guardrails (CPA-SG)

This synthesis extracts the strongest ideas from GLM-5.2, GPT-5/Codex, and GPT-OSS 20B into a single concrete architecture.

### 1. Memory (Hybrid Multi-Tier)

- Working: Transformer KV-cache with utility-scored eviction (GLM-5.2)
- Episodic: HNSW vector DB with Ebbinghaus temporal decay + summary compression (GLM-5.2)
- Semantic: Neo4j property graph with probabilistic edges + Bayesian updating (GPT-5)
- Procedural: AST library with statistical significance promotion p<0.05 Bonferroni (GPT-5)
- Innovation: Causal Episodic Loop - episodes auto-update causal graph, causal graph retrieves episodes for planning (GPT-5)

### 2. Reasoning (Active Inference + Hierarchical MCTS)

- Framework: Active Inference minimizing Expected Free Energy (GLM-5.2)
- Planning: Hierarchical MCTS - transformer coarse plans + MCTS fine expansion (GPT-OSS)
- Evaluation: Monte Carlo simulation 10-50 samples over JEPA world model (GPT-5)
- Adaptation: Surprise signal triggers replanning (GLM-5.2) + post-task reflection (GPT-5)
- Symbolic check: ProbLog consistency before execution (GPT-OSS)

### 3. Learning (Multi-Strategy)

- Continual: Frozen base + LoRA adapters + EWC (GLM-5.2/GPT-5)
- Replay: Prioritized Experience Replay weighted by novelty + error (GLM-5.2/GPT-OSS)
- Meta-learning: MAML for rapid adaptation (GPT-OSS) + bandit prompt optimization (GPT-5)
- Architecture search: NAS with sandbox validation + hot-swap (GLM-5.2)
- Procedure mining: Statistical significance testing (GPT-5)

### 4. Tool Use (Epistemic Bound + Capability Registry)

- Selection: Learned router predicts relevant tools (GPT-5)
- Routing: Epistemic Bound matrix - simulate if confidence > 0.95, else execute (GLM-5.2)
- Sandbox: WebAssembly micro-containers with resource limits (GLM-5.2/GPT-5)
- Composition: Tool chaining with learned optimal compositions (GPT-5)
- Fallback: Internal reasoning when no tool available (GPT-5)

### 5. World Model (JEPA + Causal + Social)

- Core: JEPA - predict in latent space (GLM-5.2)
- Causality: Pearl SCM with do-calculus for counterfactuals (GPT-5)
- Physics: Differentiable physics engine (GPT-5)
- Social: BDI multi-agent simulator with Theory of Mind (GPT-5)
- Uncertainty: Deep ensemble calibration - aleatoric vs epistemic (GPT-5)

### 6. Safety (Multi-Layer Defense)

- Layer 1: External Governor Module with Z3 solver (GLM-5.2)
- Layer 2: Constitutional AI critic model (GPT-5)
- Layer 3: Impact regularization for irreversible actions (GPT-5)
- Layer 4: Uncertainty thresholds trigger human review (GPT-5)
- Layer 5: Hardware kill switch, unmodifiable (GPT-5)
- Enforcement: Cryptographic execution tokens (GLM-5.2)

### 7. Evaluation

ARC-AGI >90%, METR >80%, 1000+ custom benchmarks, self-play, transfer battery, safety benchmarks, multi-day project test (GPT-5)

### 8. Runtime

K8s cluster, GPU/memory/CPU specialization, checkpoint every 100 steps, hot/cold storage, PostgreSQL + Pinecone + filesystem (GPT-5)

### 9. Multi-Agent

Hub-and-spoke: Planner, Executor, Critic, Learner, Communicator; blackboard via message queue; dynamic spawning (GPT-5)

### 10. Feasibility

70B+ LLM base; existing DBs; 8x A100 initial ($24K/mo); 2-3yr prototype, 5yr AGI; key challenge = symbolic-neural integration

### 11. Originality

The Causal Episodic Loop + Epistemic Bound Matrix creates a self-reinforcing cycle: more experience -> better causal model -> more accurate tool simulation -> more efficient learning -> more experience.

## Implementation Phases

1. Phase 1 (Months 1-6): Memory + reasoning loop + safety layer
2. Phase 2 (Months 7-18): JEPA world model + tool use + learning
3. Phase 3 (Months 19-36): Multi-agent + full evaluation + architecture search
4. Phase 4 (Months 37-60): Scale to AGI + safety hardening + deployment
