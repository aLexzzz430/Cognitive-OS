# Summary: Common Patterns, Disagreements & Notable Ideas

## Common Patterns Across All 7 Models

### 1. The Quad-Partite Memory Architecture
**Every single model** proposed a memory system with four distinct stores:
- **Working Memory**: High-speed, volatile buffer (KV-cache, Redis, SSM) for immediate context
- **Episodic Memory**: Vector database with temporal indexing (Milvus, Pinecone, FAISS, Qdrant)
- **Semantic Memory**: Knowledge graph (Neo4j, Dgraph, NebulaGraph) storing facts and relationships
- **Procedural Memory**: Executable skills (LoRA adapters, code templates, RL policies)

All models also proposed some form of **memory consolidation** — periodic offline processing that compresses episodic memories into semantic generalizations and prunes low-utility information. This mirrors the human sleep cycle and was independently arrived at by all 7 models.

### 2. MCTS as the Universal Planning Engine
6 of 7 models (all except Llama-3's "Erebus") explicitly used **Monte Carlo Tree Search** in their reasoning loop. The one holdout (Erebus) used model-based RL planning instead. MCTS is used in conjunction with:
- A **Critic/Value Network** to score candidate plans (GLM-4, Qwen3.5-Plus, Qwen3-Max, DeepSeek-R1)
- A **World Model simulation** to roll out predicted outcomes (all)
- A **confidence threshold** before execution (Qwen3.5-Plus requires >85%)

### 3. Safety as Architectural Constraint, Not Prompt Engineering
All 7 models rejected the idea of safety-through-prompting. Instead, safety is **architecturally enforced**:
- **Separate Guardian/Veto model**: A smaller, hardened model that intercepts every action (GLM-4, Qwen3.5-Plus, Qwen3-Max, DeepSeek-R1)
- **Machine-readable constitutions**: Rules encoded as Datalog, executable constraints, or hard-coded system-level policies (Qwen3.5-Plus, Qwen3-Max, DeepSeek-Chat)
- **Hardware kill switches**: Physical or hardware-enforced circuit breakers (GLM-4, Qwen3.5-Plus, Qwen3-Max, DeepSeek-R1)
- **Immutable audit trails**: All decisions logged to append-only ledgers (Qwen3.5-Plus, Qwen3-Max, DeepSeek-R1)

### 4. Neuro-Symbolic Hybrid World Models
Every model proposed a world model that combines **neural representations** (embeddings, transformers, diffusion models) with **symbolic representations** (knowledge graphs, causal Bayesian networks, formal logic). The purely neural approach is universally considered insufficient for the kind of causal reasoning AGI requires.

### 5. Cloud-Native Runtime Architecture
All 7 models proposed **Kubernetes-based** deployment with:
- StatefulSets for databases
- Containerized microservices
- Object storage (S3/GCS) for backups
- Event logging/message queues (Kafka, RabbitMQ)
- Graceful degradation and fault tolerance

### 6. Mixture-of-Agents with Router/Orchestrator
All models proposed a **hierarchical multi-agent** system with:
- A central orchestrator or Executive agent
- Specialized sub-agents (Coder, Researcher, Critic, etc.)
- A **shared blackboard** communication pattern (Redis, Kafka, custom KV store)
- Consensus or voting mechanisms for conflict resolution

### 7. Dual-Track Learning
All models separated learning into **fast contextual** (RAG, dynamic prompting, memory updates) and **slow parametric** (nightly fine-tuning, LoRA, DPO). This prevents catastrophic forgetting while enabling rapid adaptation.

---

## Key Disagreements Between Models

### 1. The Role of the World Model
This was the **most divergent dimension**:
| Model | World Model Approach |
|---|---|
| **GPT-4o** | Hybrid embeddings + knowledge graph (relatively conservative) |
| **GLM-4 (AETHER)** | Video Diffusion Transformer for outcome simulation + Causal Bayesian Net (most ambitious) |
| **DeepSeek-Chat (Prometheus)** | Causal Transformer trained via contrastive predictive coding (most research-intensive) |
| **DeepSeek-R1 (Sylvan)** | Decomposable NeRFs + multi-simulator (most computationally expensive) |
| **Qwen3-Max (MoRIE)** | Hyperdimensional vectors + Bayesian structural learning (most theoretically novel) |
| **Llama-3 (Erebus)** | Knowledge graph + spatiotemporal DB (most traditional) |
| **Qwen3.5-Plus (CKA)** | Causal State Machine + 3D Gaussian Splats (most practical/specific) |

**Core disagreement**: Should the world model be a neural simulation engine (GLM-4, DeepSeek-Chat) or a symbolic reasoning system with neural backing (GPT-4o, Qwen3.5-Plus)?

### 2. Self-Improvement Autonomy
Models diverged sharply on how much self-modification to allow:
| Model | Self-Improvement Stance |
|---|---|
| **DeepSeek-R1 (Sylvan)** | **Most conservative** — only LoRA/prefix tuning allowed; full model changes require API mediation; formal verification required before any change |
| **DeepSeek-Chat (Prometheus)** | **Cautious** — Oracle module proposes changes but requires sandboxed simulation + human approval |
| **GPT-4o** | **Ambitious** — Neural Architecture Search + evolutionary algorithms for architectural optimization |
| **Qwen3-Max (MoRIE)** | **Most ambitious** — Skill synthesis via genetic programming + automated module replacement |
| **Llama-3 (Erebus)** | **Boldest** — Self-modifying code that can reorganize its own architecture |

**Core disagreement**: Does AGI safety require strict constraints on self-modification (Sylvan, Prometheus) or is self-improvement the very purpose of AGI (MoRIE, Erebus)?

### 3. Evaluation Philosophy
| Model | Evaluation Approach |
|---|---|
| **GPT-4o, Llama-3** | Traditional: static benchmarks + human evaluation |
| **GLM-4, DeepSeek-Chat** | Agent-centric: zero-shot success rate + sample efficiency + long-horizon tasks |
| **Qwen3.5-Plus** | Calibration-focused: Brier Score + Intervention Rate metrics |
| **Qwen3-Max** | Multi-tiered: micro-benchmarks + macro errand completion + continuous generalization monitoring |
| **DeepSeek-R1** | Most comprehensive: capability + alignment + robustness + scalability + interpretability |

**Core disagreement**: Should AGI evaluation focus on task completion (pragmatic) or calibration/alignment (safety-first)?

### 4. The Dual-Process Question
Only **DeepSeek-Chat (Prometheus)** explicitly proposed a genuine **System 1 / System 2** architecture where fast intuitive responses and slow deliberative reasoning coexist as separate processes. Other models (GPT-4o, Qwen3.5-Plus) used MCTS as the *only* planning mechanism, effectively making all decisions deliberative. This is a fundamental architectural disagreement: **is AGI always slow-thinking, or does it need both fast and slow paths?**

### 5. Programming Language Stack
| Model | Primary Stack |
|---|---|
| **DeepSeek-Chat (Prometheus)** | **Rust/C++** for performance-critical + Python for neural — most explicit about systems programming |
| **DeepSeek-R1 (Sylvan)** | Python/JAX/PyTorch + Docker/K8s |
| **All others** | Python + cloud-native (K8s, Docker) |

Only Prometheus argued for Rust/C++ for the memory and planning subsystems, citing the need for performance-critical, zero-overhead execution at AGI scale.

---

## Notable Unique Ideas

### From Each Model

**GPT-4o**
- **Hyperbolic embeddings** for hierarchical concept representation (mathematically elegant)
- **Predictive coding** for world model consistency (neuroscience-inspired)
- **Neural Architecture Search** for autonomous self-optimization (ambitious)

**GLM-4 (Project AETHER)**
- **Active Inference / Free Energy Minimization** as the core cognitive philosophy — the system does not just predict tokens, it minimizes surprise by comparing world model predictions against sensory feedback (most principled theoretical framework)
- **Video Diffusion Transformer** as a planning simulation engine (push the simulation paradigm furthest)
- **Guardian Model in hardware-isolated veto process** (strongest safety isolation)
- **Activation Steering** for monitoring internal neuron firings (most interpretable safety approach)

**Llama-3 (Erebus)**
- **Self-modifying code** — the ability to reorganize its own architecture at runtime (boldest claim, but least specific on implementation)
- **Graph-based neural network for STM** with human-like 7±2 chunk capacity (unique cognitive science grounding)

**Qwen3.5-Plus (CKA)**
- **Event Sourcing** for full cognitive auditability — every thought and action is an immutable event in Kafka (best persistence/audit pattern)
- **Do-Calculus counterfactual reasoning** via Causal State Machine (most rigorous causal reasoning proposal)
- **Brier Score calibration** as a core evaluation metric (most principled about uncertainty quantification)
- **LoRA adapter library** as procedural memory (most practical skill storage mechanism)

**Qwen3-Max (MoRIE)**
- **Hyperdimensional vectors** (Sparsey algorithm) for abstract concept representation enabling analogical reasoning (most novel representation layer)
- **Genetic programming for skill synthesis** — combining existing skills into new ones autonomously (most creative self-improvement)
- **Dynamic recursion depth** — complex tasks spawn child reasoning loops with their own resource budgets (most scalable planning)
- **Chaos engineering** (Gremlin) for fault tolerance testing (most robust runtime)

**DeepSeek-Chat (Project Prometheus)**
- **Differentiable Neural Dictionary (DND)** — content-addressable memory with Hebbian-like updates (most novel memory architecture)
- **Dual-Process Architecture** — genuine System 1 (fast) / System 2 (slow) separation (only model to propose this)
- **Oracle module** for meta-learning with sandboxed simulation testing (most careful self-improvement)
- **Counterfactual dream cycle** for offline learning (most complete learning loop)
- **Rust/C++** for performance-critical subsystems (most systems-engineering-aware)

**DeepSeek-R1 (Sylvan)**
- **Formal verification** (Dafny/Coq integration) in the reasoning loop (strongest safety guarantees)
- **PAC-history** temporal models + **Decomposable NeRFs** for world model rendering (most sophisticated simulation)
- **Control Vectors** (Anthropic-style) for steering model behavior (most controllable alignment)
- **LLM State Fuzzer** for restoring interrupted reasoning chains across reboots (most creative fault tolerance)
- **Blockchain on-chain activity log** for critical actions (most auditable)
- **Auction/Stable Roommate Matching** for agent conflict resolution (most mathematically sophisticated coordination)

---

## Surprising Omissions

1. **No model proposed neuromorphic hardware** — despite the 2025-2030 timeframe, none suggested brain-inspired chips
2. **No model proposed quantum computing** — all solutions are classical
3. **Only 1 model (Prometheus) addressed energy efficiency in detail** — despite AGI's enormous compute demands
4. **No model proposed a concrete number or threshold for "AGI"** — all described benchmarks but none defined the pass/fail criterion
5. **The term "consciousness" appears nowhere** — all models treat AGI as an engineering problem, not a philosophical one
6. **No model discussed economic or societal impact** — purely technical proposals
