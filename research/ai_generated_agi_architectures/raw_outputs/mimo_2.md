# A Comprehensive AGI Architecture Specification

## Preamble: Design Philosophy

This architecture is grounded in three principles drawn from neuroscience and cognitive science, translated into engineering terms:

1. **Global Workspace Theory (GWT)**: A shared representational substrate ("global workspace") through which specialized modules broadcast and compete for influence, with a central bottleneck that enables coherent serial reasoning.
2. **Predictive Processing**: The system is fundamentally a prediction engine—perception, reasoning, planning, and action are all unified under a common prediction-error minimization objective over different timescales.
3. **Progressive Specialization**: Knowledge begins as declarative, slow, and explicit, and through experience is progressively compiled into procedural, fast, and implicit representations.

The architecture is organized around a single core insight (elaborated in §11): **the world model IS the memory, and planning IS simulation through that model.** This unification eliminates artificial boundaries between subsystems.

---

## 1. Memory Architecture

### 1.1 Taxonomy

I propose **five memory tiers**, each with distinct storage substrate, access patterns, and timescales:

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                       │
├──────────────┬──────────────┬───────────────────────────────┤
│    Tier      │  Substrate   │  Timescale / Capacity         │
├──────────────┼──────────────┼───────────────────────────────┤
│ Sensory      │ Streaming    │ ~100ms, fixed-size window     │
│ Buffer       │ ring buffer  │                               │
├──────────────┼──────────────┼───────────────────────────────┤
│ Working      │ Attention    │ Seconds-minutes, 7±2 "chunks" │
│ Memory       │ slots in     │ (learned chunking)            │
│              │ global       │                               │
│              │ workspace    │                               │
├──────────────┼──────────────┼───────────────────────────────┤
│ Episodic     │ Experience   │ Hours-Lifetime, TB-scale      │
│ Memory       │ store +      │ indexed by embeddings +       │
│              │ vector DB    │ temporal/spatial/causal coords │
├──────────────┼──────────────┼───────────────────────────────┤
│ Semantic     │ Knowledge    │ Persistent, structured         │
│ Memory       │ graph +      │ graph + parametric store       │
│              │ parametric   │                               │
│              │ weights      │                               │
├──────────────┼──────────────┼───────────────────────────────┤
│ Procedural   │ Compiled     │ Permanent, compressed,         │
│ Memory       │ policies/    │ fast-path execution            │
│              │ skills       │                               │
└──────────────┴──────────────┴───────────────────────────────┘
```

### 1.2 Detailed Mechanisms

**Sensory Buffer**: A fixed-size temporal window over raw input streams. Implemented as a causal transformer with a finite context window (e.g., 4K–128K tokens for language, 30 frames at 30fps for vision). This is the only place where raw data enters the system. No novel engineering here—it's the current state-of-the-art encoder.

**Working Memory**: The core of the architecture. Implemented as a **slot-based attention mechanism** over a fixed number of "workspace vectors" (dimension d = 8192, N = 32 slots).

```python
class WorkingMemory:
    def __init__(self, d_model=8192, n_slots=32):
        self.slots = torch.zeros(n_slots, d_model)  # Workspace state
        self.salience = torch.zeros(n_slots)          # Competition weights
        self.binding_keys = torch.zeros(n_slots, d_model)  # Content-addressable keys
        
    def read(self, query: Tensor) -> Tensor:
        """Content-addressable read via attention."""
        keys = self.binding_keys
        scores = (query @ keys.T) / sqrt(d_model)
        # Salience-gated competition (GWT broadcast)
        scores = scores + self.salience
        weights = softmax(scores, dim=-1)
        return weights @ self.slots
    
    def write(self, key: Tensor, value: Tensor, salience: float):
        """Write with competition for slot allocation."""
        # Find least-salient slot if no match
        match_scores = (key @ self.binding_keys.T) / sqrt(d_model)
        best_match = match_scores.argmax()
        if match_scores[best_match] > WRITE_THRESHOLD:
            # Update existing slot
            self.slots[best_match] = gate(self.slots[best_match], value)
            self.salience[best_match] = max(self.salience[best_match], salience)
        else:
            # Evict least salient
            victim = self.salience.argmin()
            if salience > self.salience[victim]:
                self.slots[victim] = value
                self.binding_keys[victim] = key
                self.salience[victim] = salience
    
    def decay(self, rate=0.01):
        """Temporal decay of salience (forgetting)."""
        self.salience *= (1 - rate)
```

The key engineering insight: working memory slots are **not** passive storage. They are **active representations** that participate in every forward pass of the core reasoning model. The global workspace transformer reads from and writes to these slots each step, creating a persistent-but-mutable context that accumulates structured state across reasoning steps.

**Episodic Memory**: A two-phase system:

- **Encoding**: At the end of each "episode" (a coherent interaction segment), the system produces a compressed episode representation via a learned encoder:
  ```
  E_ep = EpisodicEncoder(
      sensory_trace=S[-W:],        # Last W sensory inputs
      working_memory_trace=M[-T:],  # Last T working memory snapshots  
      action_trace=A[-T:],          # Actions taken
      outcome=Δstate                # Observed state change
  )
  ```
  This produces: (1) A dense embedding vector for similarity retrieval, (2) A structured metadata record (timestamp, location, entities involved, emotional valence, success/failure), (3) A compressed trajectory encoding for reconstruction.

- **Retrieval**: Multi-index retrieval using FAISS or ScaNN at scale, with **four retrieval cues**:
  - **Similarity**: Standard embedding similarity to current context
  - **Recency**: Exponential decay weighting by temporal distance
  - **Temporal contiguity**: If episode A is retrieved, episodes temporally adjacent to A receive boosted retrieval probability (hippocampal chaining)
  - **Causal relevance**: Retrieval episodes that are causal predecessors or successors of