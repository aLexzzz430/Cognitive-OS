# Summary: Common Patterns, Disagreements, and Notable Ideas

## Collection Overview

- **Models directly accessed**: GLM-5.2 (Zhipu AI, via Volcengine ARK API), GPT-5/Codex (OpenAI, direct generation), GPT-OSS 20B (OpenAI, via Pollinations.ai)
- **Additional reference**: Anthropic published AGI views (web scrape), arXiv AGI architecture paper (web scrape)
- **Access limitations**: 15 additional AI systems attempted but inaccessible due to SSL failures or missing API keys. See sources.md.
- **Collection date**: 2026-07-11

## Common Patterns

### 1. Multi-Tier Memory Architecture
All three models converge on working + episodic + semantic + procedural memory tiers. GLM-5.2 and GPT-5 both use HNSW vector DBs for episodic memory and Neo4j for semantic memory. GPT-OSS uses FAISS-IVF + Neo4j. Graph databases for semantic memory is universal.

### 2. MCTS-Based Planning
All models use Monte Carlo Tree Search: GLM-5.2 uses MCTS with UCT over world model; GPT-5 uses Monte Carlo rollouts (10-50 samples); GPT-OSS uses hierarchical MCTS with transformer-generated coarse plans. MCTS is the emerging standard for AGI planning.

### 3. LoRA + EWC for Continual Learning
GLM-5.2 and GPT-5 both freeze base models and add LoRA adapters with EWC. GPT-OSS uses MAML + memory replay. The pattern of frozen base + domain adapters is consistent.

### 4. Symbolic-Neural Hybrid
All models combine symbolic reasoning with neural networks: GLM-5.2 uses Z3 solver; GPT-5 uses RDF knowledge graph; GPT-OSS uses ProbLog logic programming with fusion layer.

### 5. Sandboxed Execution
All models emphasize isolated execution: GLM-5.2 uses WebAssembly containers; GPT-5 uses containerized tools with resource limits; GPT-OSS implies API gateway pattern.

## Key Disagreements

### 1. World Model Architecture
- GLM-5.2: JEPA (latent space prediction, LeCun-inspired)
- GPT-5: Dual physics engine + social simulator with Pearl SCM
- GPT-OSS: Embedded in MCTS planner, less specified

GLM-5.2's JEPA is most theoretically grounded; GPT-5's dual approach is most practically detailed.

### 2. Safety Mechanism Design
- GLM-5.2: External Governor Module with Z3 solver + cryptographic tokens
- GPT-5: Constitutional AI + impact regularization + uncertainty thresholds + hardware kill switch
- GPT-OSS: Not fully specified

GLM-5.2's external placement is architecturally cleaner; GPT-5's multi-layer approach provides defense in depth.

### 3. Memory Forgetting Strategy
- GLM-5.2: Ebbinghaus-style decay with summary compression
- GPT-5: Soft decay with EWC weight protection
- GPT-OSS: LRU list + novelty-weighted replay

### 4. Meta-Learning Approach
- GLM-5.2: NAS meta-learner with hot-swap
- GPT-5: Bandit-based prompt optimization + periodic architecture search
- GPT-OSS: MAML + DARTS differentiable architecture search

## Notable Ideas

1. **Causal Episodic Loop (GPT-5)**: Tight integration of causal graph and episodic memory creating a positive feedback loop.
2. **Epistemic Bound Matrix (GLM-5.2)**: Confidence threshold (0.95) for tool simulation vs execution.
3. **Symbolic-Neural Fusion Layer (GPT-OSS)**: Soft-attention weighting between ProbLog and transformer logits.
4. **Procedure Mining with Statistical Significance (GPT-5)**: p<0.05 with Bonferroni correction prevents spurious correlations.
5. **Ebbinghaus Memory Decay (GLM-5.2)**: Psychologically grounded forgetting with summary compression.
6. **Cryptographic Execution Tokens (GLM-5.2)**: Governor Module signs tokens, making safety bypass cryptographically impossible.

## Limitations

This packet includes 3 directly-accessed AI model outputs and 2 scraped reference pages. The 8-model minimum was not met due to API access limitations. GPT-OSS did not fully cover all 11 dimensions. See sources.md for detailed access documentation.
