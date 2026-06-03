# Summary: Common Patterns, Disagreements, and Notable Ideas

## Common Patterns Across All 8 Proposals

### 1. Hierarchical Memory Is Universal
Every single proposal uses a multi-tier memory system (3-5 tiers). The tiers consistently include:
- **Working memory** (active context, attention-based)
- **Episodic memory** (experiences with temporal indexing)
- **Semantic memory** (knowledge graph or structured facts)
- **Parametric memory** (learned neural weights)

All proposals use vector databases (FAISS, HNSW, ScaNN) for retrieval. This is a strong consensus — hierarchical memory with embedding-based retrieval is the accepted baseline.

### 2. Dual-Process Reasoning
Seven of eight proposals explicitly implement a dual-process reasoning system:
- **Fast/intuitive** (System 1): Pattern matching, cached responses, direct retrieval
- **Slow/deliberative** (System 2): Chain-of-thought, search, planning, verification

The Gemini/DeepMind proposal goes further by framing this as MCTS-based search over a latent reasoning space.

### 3. World Model as Central Component
All proposals include an explicit world model, but they disagree on its form:
- **Learned neural network** (GPT-4, Claude, MiMo-2)
- **Knowledge graph + neural hybrid** (MiMo-1, MiMo-3)
- **Code-based formal model** (DeepSeek)
- **Video-learned representation** (Gemini)
- **Compression-based abstraction** (CONTINUUM)

### 4. Tool Use Is Standard
All proposals include structured tool use with:
- Tool registry/catalog
- Capability matching
- Verification of tool outputs
- Composition of simple tools into complex workflows

### 5. Safety Is Mandatory But Under-Specified
All proposals include safety mechanisms, but most are vague on implementation:
- Constitutional AI constraints (GPT-4, Claude)
- Formal verification (MiMo-1, CONTINUUM, NOOSPHERE)
- Runtime monitoring (all)
- RLHF alignment (GPT-4, Claude)

The Claude proposal is most specific about safety, making it a first-class design constraint rather than an add-on.

## Key Disagreements

### 1. Monolithic vs. Modular
- **Monolithic scaling** (GPT-4 perspective): Scale a single transformer, add memory/tools as augmentations
- **Modular composition** (MiMo-1 AXIOM, NOOSPHERE): Multiple specialized modules negotiating via shared bus
- **Middle ground** (others): Core transformer + specialized modules

### 2. Learning Mechanism
- **Meta-learning** (GPT-4, MiMo-1): Outer loop optimization over learning algorithms
- **Predictive processing** (MiMo-2): Unified prediction-error minimization
- **Compression** (CONTINUUM): All learning is lossy compression
- **Co-evolution** (Gemini): Model, search, and value function evolve together

### 3. Search vs. Generation
- **Search-dominant** (Gemini): AGI is search over a learned world model
- **Generation-dominant** (GPT-4, DeepSeek): AGI is scaled generation with augmentation
- **Hybrid** (others): Both search and generation, with meta-cognitive switching

## Notable Non-Obvious Ideas

1. **"Reasoning is compression"** (CONTINUUM): Unifies memory, reasoning, and learning under a single mathematical framework. All cognition is lossy compression over experience traces.

2. **"World model IS memory, planning IS simulation"** (MiMo-2): Eliminates artificial boundaries between subsystems by recognizing they're the same operation at different timescales.

3. **"Cognitive Bus"** (NOOSPHERE): Intelligence emerges from inter-module dynamics, not from any single component. The protocol matters more than the modules.

4. **"AGI is search over a world model"** (Gemini): Frames the entire problem as planning with learned models, where model/search/value co-evolve.

5. **"Universal Memory Address"** (CONTINUUM): Every memory item across all tiers has a single address, enabling cross-tier attention patterns in the transformer.

6. **"Crystallization"** (CONTINUUM): Memories transition from fluid (recent, malleable) to crystallized (stable, verified) — a concrete mechanism for knowledge consolidation.

7. **"Cognitive Operating System"** (NOOSPHERE): The runtime is not a model but an OS where specialized subsystems interact through standardized protocols.
