# CONTINUUM: A Concrete AGI Architecture

## Overview

CONTINUUM (Continuous Unified Neural Topology for Executable Unified Modeling) is a modular AGI architecture built on a central thesis: **reasoning is compression, and compression is understanding.** Every cognitive operation—recall, inference, planning, learning—is formally reducible to a lossy compression transform over experience traces. This insight unifies memory, reasoning, and learning into a single mathematical framework rather than treating them as separate subsystems.

---

## 1) Memory Architecture: Crystallization Lattice

### Design

Memory is organized as a **5-tier crystallization lattice** where items mature through tiers based on access frequency, compression ratio, and causal utility:

```
┌─────────────────────────────────────────────────────┐
│  L0: SENSORY BUFFER (Ring buffer, ~500ms)           │
│  ───────────────────────────────────────────────────│
│  L1: WORKING MEMORY (Context window, ~128K tokens)  │
│  ───────────────────────────────────────────────────│
│  L2: EPISODIC CACHE (Vector DB + temporal index)    │
│  ───────────────────────────────────────────────────│
│  L3: SEMANTIC CRYSTAL (Typed hypergraph KG)         │
│  ───────────────────────────────────────────────────│
│  L4: PROCEDURAL CACHE (Compiled skill DSL)          │
│  ───────────────────────────────────────────────────│
│  L5: ARCHIVAL COMPRESS (Learned codec weights)      │
└─────────────────────────────────────────────────────┘
```

### Concrete Implementation

**L0 — Sensory Buffer:** Raw multimodal input is held in a fixed-size ring buffer (configurable, default ~100MB for vision, ~2MB for text/audio). Items are tagged with `timestamp`, `modality_id`, `confidence_score`, and `novelty_score` (computed as 1 - max_similarity to L1 items). Items below a novelty threshold are discarded immediately.

**L1 — Working Memory:** This is the active context window of the primary transformer backbone. However, unlike standard LLMs, the context window is **dynamically partitioned** into segments:

```
struct WorkingMemorySlot {
    content: Tensor<f32>,       // The actual representation
    slot_type: enum {PERCEPT, RETRIEVAL, HYPOTHESIS, GOAL, META},
    attention_mask: BitMatrix,  // Which other slots can attend to this
    ttl: u32,                   // Time-to-live in cognitive cycles
    priority: f32,              // Salience for retention
    source_tier: u8,            // Where this was retrieved from
}
```

A **Memory Manager** module (a small 100M-parameter transformer) runs every cognitive cycle and decides what to evict (lowest priority × age) and what to retrieve from L2/L3 based on current goal state and attention patterns.

**L2 — Episodic Cache:** Every cognitive cycle, the system creates an **episode record**:

```python
@dataclass
class Episode:
    id: UUID
    timestamp: DateTime
    context_snapshot: List[WorkingMemorySlot]  # Compressed repr of L1 state
    action_taken: Optional[Action]
    outcome: Optional[Outcome]
    surprise: float          # KL divergence between predicted and actual
    causal_tags: Set[Entity] # Entities causally involved
    embedding: Vector[4096]  // Learned episodic embedding
    compression_level: int   // 0=verbatim, increasing = more lossy
```

These are stored in a **vector database** (implementing HNSW with learned distance metric) indexed by embedding and a temporal B+ tree. Retrieval uses a composite query:

```
score = α·cosine_sim(query, episode.embedding)
      + β·temporal_recency(episode.timestamp)
      + γ·causal_overlap(current_entities, episode.causal_tags)
      + δ·surprise_weight(episode.surprise)
```

where α, β, γ, δ are learned per-task by the meta-controller.

**L3 — Semantic Crystal (Knowledge Graph):** This is a **typed hypergraph** where:

- **Nodes** represent concepts, entities, propositions
- **Hyperedges** represent n-ary relations (not just binary triples)
- Each node/edge carries: `confidence: f32`, `source_episodes: Set<UUID>`, `last_verified: DateTime`, `contradictions: Set<NodeID>`

```python
class SemanticNode:
    id: NodeID
    type: EntityType  # Concept, Instance, Proposition, Rule, Schema
    embedding: Vector[2048]          # Semantic embedding
    symbolic_repr: SExpr            # Logical form
    confidence: f32                  # Bayesian posterior
    provenance: List[EpisodeID]     # Source episodes
    temporal_validity: (DateTime, Optional[DateTime])  # When this is true
    access_count: u32
    last_modified: DateTime
    compression_of: List[EpisodeID]  # What episodes were compressed into this

class HyperEdge:
    id: EdgeID
    type: RelationType
    nodes: List[NodeID]             # Variable-arity
    weight: f32
    conditions: List[Proposition]   # When this relation holds
    embedding: Vector[1024]         # Relation embedding
```

The KG is stored in a combination of **TigerGraph** (for traversal queries) and a **custom columnar store** (for bulk embedding operations). Key operations:

- **Crystallization:** Episodic memories that co-occur with similar context-action-outcome patterns are compressed into semantic nodes via a learned compression network (see Section 3).
- **Consistency Maintenance:** A background process runs probabilistic inference (mini-bucket elimination) to detect and resolve contradictions. Contradictions are flagged, not silently resolved.
- **Decay:** Nodes with no access and no support from recent episodes decay exponentially, but only to a floor (never fully deleted—this is archival, not garbage collection).

**L4 — Procedural Cache:** Skills are stored as **typed programs** in a domain-specific language (DSL) called SKILL:

```
SKILL navigate_to(target: Entity):
    PRECONDITIONS: [agent.has_location, target.has_location, path_exists(agent.loc, target.loc)]
    POSTCONDITIONS: [distance(agent.loc, target.loc) < THRESHOLD]
    STEPS:
        1. plan_path(agent.loc, target.loc) -> path
        2. FOR waypoint IN path:
              move_to(waypoint)
              IF obstacle_detected():
                  re_plan(remaining_path)
        3. VERIFY distance(agent.loc, target.loc) < THRESHOLD
    FAILURE_MODES: [no_path_exists, obstacle_unmovable, target_moved]
    PARAMETRIC_BINDING: {THRESHOLD: 1.0}
    LEARNED_FROM: [episode_ids...]
    SUCCESS_RATE: 0.94
    INVOCATION_COUNT: 847
```

Skills are compiled from successful episode sequences by the **Skill Compiler** (see Section 3). They can be composed hierarchically: `navigate_to` might call `avoid_obstacle`, which calls `perceive_environment`.

**L5 — Archival Compress:** Old episodes that have already been crystallized into L3/L4 are compressed using a **learned codec**—a variational autoencoder that encodes episode traces into a continuous latent space. This allows "reconstructing" approximate memories from compact codes, analogous to how humans recall gist rather than verbatim.

### Key Technical Property: Unified Addressing

Every memory item across all tiers has a **universal memory address** (UMA):

```
UMA = hash(tier_id | item_id | temporal_coordinate | causal_coordinate)
```

This allows any cognitive module to reference any memory item by address, enabling cross-tier attention patterns. The primary transformer can attend over L2/L3/L4 items by projecting their UMA-indexed representations into the working memory context.

---

## 2) Reasoning/Planning Loop: Global Workspace Cognitive Cycle

### Architecture

Inspired by Global Workspace Theory (Baars, 1988) and implemented concretely:

```
┌──────────────────────────────────────────────────────────────┐
│                    GLOBAL WORKSPACE (L1)                     │
│         Shared broadcast medium - all modules can read       │
├────────────┬────────────┬────────────┬───────────────────────┤
│ PERCEPTION │  REASONING  │  PLANNING  │  METACOGNITION       │
│   MODULE   │   MODULE    │   MODULE   │    MODULE            │
│  (200M)    │   (2B)      │   (1B)     │   (500M)             │
├────────────┴────────────┴────────────┴───────────────────────┤
│              MEMORY MANAGER (100M)                           │
├──────────────────────────────────────────────────────────────┤
│         SKILL EXECUTOR / TOOL INTERFACE (Section 4)          │
└──────────────────────────────────────────────────────────────┘
```

### Cognitive Cycle (CC)

Each cognitive cycle runs at approximately 10Hz (100ms per cycle) and consists of:

```
def cognitive_cycle(state: GlobalWorkspaceState) -> GlobalWorkspaceState:
    # Phase 1: COMPETITION (parallel, ~20ms)
    module_outputs = parallel([
        perception_module.attend_and_propose(state),
        reasoning_module.attend_and_propose(state),
        planning_module.attend_and_propose(state),
        metacognition_module.attend_and_propose(state),
    ])
    
    # Phase 2: BROADCAST SELECTION (~10ms)
    # A learned attention mechanism selects which proposals
    # to broadcast to the workspace
    selected = workspace_arbitrator(state, module_outputs)
    # Workspace arbitrator is a small cross-attention network
    # that scores proposals by: relevance × novelty × urgency
    
    # Phase 3: INTEGRATION (~40ms)
    # All modules process the new broadcast
    new_state = integrate(state, selected)
    
    # Phase 4: ACTION SELECTION (~20ms)
    action = action_selector(new_state)
    
    # Phase 5: METACOGNITIVE REVIEW (~10ms)
    # Did reasoning quality degrade? Should we switch strategy?
    meta_assessment = metacognition_module.review_cycle(new_state, action)
    
    return GlobalWorkspaceState(
        working_memory=new_state.working_memory,
        active_goal=new_state.active_goal,
        cycle_count=state.cycle_count + 1,
        confidence=meta_assessment.confidence,
        strategy=meta_assessment.suggested_strategy,
    )
```

### Reasoning Module (2B parameter transformer)

The reasoning module implements **multi-strategy reasoning**:

```python
class ReasoningModule:
    strategies = {
        'deductive': DeductiveEngine(),      # Formal logic prover (lean4-backed)
        'inductive': PatternMatcher(),       # Statistical generalization
        'abductive': Hypothesis