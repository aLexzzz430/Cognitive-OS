# Raw output — cohere/north-mini-code:free

_recolectado: 2026-10-05 20:35:02_

---

**1. Goal and scope**  
Build a self‑contained, general‑intelligence substrate that continuously grounds its internal representations in a shared external reality, can reason about and plan actions, and can correct its own errors without external supervision. The scope includes perception of multimodal sensors, construction of a predictive world model, episodic/semantic/procedural memory, goal‑directed planning, and meta‑cognitive monitoring.

**2. Core components**  

| Component | One‑line responsibility |
|-----------|--------------------------|
| **Sensor Fusion Engine (SFE)** | Aligns raw camera, lidar, audio, and tactile streams into a common spatio‑temporal coordinate system. |
| **Grounding Map (GM)** | Maintains a persistent, voxel‑based map of the environment with semantic labels, physical properties, and object identities. |
| **World Model (WM)** | A probabilistic, causal graph (e.g., Dynamic Causal Model) that predicts future states given actions. |
| **Episodic Memory Store (EMS)** | Stores time‑stamped experiences as feature vectors plus contextual metadata. |
| **Semantic Memory Index (SMI)** | Hierarchical knowledge base of concepts, ontologies, and learned causal rules. |
| **Procedural Memory Bank (PMB)** | Policies, skills, and motor sequences encoded as parameterised neural networks or differentiable programs. |
| **Attention Allocator (AA)** | Dynamically selects which modules receive which inputs and which internal processes are activated. |
| **Reasoning Engine (RE)** | Executes goal‑oriented inference (e.g., planning, counterfactual simulation) using WM, SMI, and PMB. |
| **Meta‑Cognitive Monitor (MCM)** | Detects inconsistencies, confidence drops, and conflicts between modules; triggers verification cycles. |
| **Verification & Error‑Correction Loop (VECL)** | Runs self‑consistency checks, external reality checks, and discrepancy‑driven corrective updates. |
| **Learning Scheduler (LS)** | Orchestrates reinforcement, supervised, and meta‑learning updates across all neural/representational components. |

**3. Data flow (step‑by‑step)**  

1. **Sensing** – Raw sensor packets are ingested by SFE, which aligns timestamps and calibrates modalities.  
2. **Grounding** – Aligned data are fed to GM; new observations update the voxel grid, assign semantic labels via a pretrained vision‑language model, and store object affordances.  
3. **World Modeling** – GM updates serve as evidence for WM; the causal graph’s node probabilities are re‑computed via a particle filter.  
4. **Memory Ingestion** – The latest WM state and sensorimotor actions are written to EMS (episodic) and PMB (procedural); derived concepts are indexed in SMI.  
5. **Attention Allocation** – AA evaluates current goals, confidence scores (MCM), and task urgency to route data: e.g., high‑uncertainty WM nodes trigger deeper RE analysis, while low‑utility streams are down‑sampled.  
6. **Reasoning** – RE queries WM, SMI, and PMB to generate action policies; it uses Monte‑Carlo Tree Search (MCTS) to evaluate sequences.  
7. **Execution** – Selected policy is dispatched to the **Action Executor** (external robot/control stack) for real‑world interaction.  
8. **Feedback** – New sensor data close the loop, and MCM flags any contradictions (e.g., WM prediction vs. GM observation).  
9. **Correction** – VECL initiates a **Verification Cycle**: (a) internal consistency checks on WM vs. SMI, (b) reality verification against GM, (c) error‑attribution via MCM, and (d) targeted parameter updates.  
10. **Learning** – LS schedules reward signals from goal achievement, intrinsic surprise from prediction errors, and meta‑rewards for self‑correction accuracy, updating SFE, WM, PMB, and SMI via RL, Bayesian updating, or meta‑learning.

**4. Memory model**  

| Memory type | Storage location | Retrieval | Forgetting mechanism |
|-------------|------------------|-----------|----------------------|
| **Episodic** | EMS (key‑value store with time‑sorted B‑tree) | Query by timestamp, spatial proximity, or semantic tag; returns compressed feature vector + metadata. | LRU eviction after 30 days unless “significant” flag (high surprise or reward) is set; permanent retention for flagged episodes. |
| **Semantic** | SMI (graph database with weighted edges) | Keyword, concept, or causal‑path query; traverses relevance links. | Edge weights decay exponentially; low‑importance nodes pruned when graph size exceeds capacity; background “concept drift” pruning based on usage statistics. |
| **Procedural** | PMB (distributed neural network with version control) | Action intent triggers nearest policy via embedding similarity. | Periodic “knowledge distillation” where older policies are merged into newer ones; catastrophic forgetting mitigated by elastic weight consolidation. |
| **Grounding** | GM (persistent voxel map with versioned snapshots) | Spatial query (e.g., “what objects occupy this volume?”) returns labeled mesh. | Incremental coarse‑graining: low‑detail voxels replaced when occlusion > 90 % and re‑generated on revisit. |

**5. Verification and error correction**  

* **Internal consistency** – MCM computes mutual information between WM predictions and SMI causal rules; low mutual information triggers a **Consistency Audit** that re‑weights graph edges.  
* **Reality verification** – VECL runs a **Reality Check**: compare GM object states to WM predictions; discrepancies generate a **Discrepancy Vector** that drives a corrective update to WM and the underlying sensory model.  
* **Confidence calibration** – Each module outputs a calibrated probability; MCM aggregates them into a global trust score. If trust < threshold, VECL initiates a **Meta‑Verification** where an auxiliary “oracle” (e.g., a symbolic model checker) validates the module’s reasoning.  
* **Error attribution** – Using Bayesian model averaging, VECL isolates whether error stems from perception (SFE), grounding (GM), world modeling (WM), or reasoning (RE), and applies targeted remedial updates.

**6. Learning loop**  

1. **Experience** – Actions produce outcomes and sensor data.  
2. **Signal generation** – LS computes (a) extrinsic reward from goal achievement, (b) intrinsic surprise (prediction error), (c) meta‑reward for successful self‑correction.  
3. **Update propagation** –  
   * SFE: supervised fine‑tuning on corrected sensor alignments.  
   * GM: gradient descent on voxel occupancy probabilities.  
   * WM: particle filter weight adjustment + optional neural world model training.  
   * SMI: knowledge graph embedding updates via contrastive learning.  
   * PMB: policy gradient / PPO updates on demonstrated skills.  
4. **Consolidation** – LS applies elastic weight consolidation to prevent interference between concurrent tasks.  
5. **Cycle repeat** – Updated models are re‑integrated into the data flow, and the loop iterates.

**7. Concrete failure modes of this design**  

| Failure mode | Root cause | Observable symptom |
|--------------|------------|--------------------|
| **Model drift** | Gradual misalignment of GM voxel updates vs. WM predictions due to sensor bias. | Systematic prediction errors that accumulate over days; confidence scores rise despite worsening performance. |
| **Hallucinated entities** | Over‑confident SMI ontologies creating phantom objects not present in GM. | Spurious planning actions (e.g., “open non‑existent door”) and repeated reality‑check failures. |
| **Catastrophic forgetting** | Aggressive PMB distillation without proper elastic regularization. | Sudden loss of previously mastered skills after exposure to new tasks. |
| **Verification blind spot** | MCM confidence thresholds set too high, suppressing error detection. | Persistent internal inconsistencies that go unnoticed, leading to cascading reasoning failures. |
| **Resource exhaustion** | GM voxel granularity exceeds memory budget, causing OOM kills. | Sudden halting of perception; subsequent behavior becomes reactive rather than planning‑driven. |

**8. What must be true for this to work (key assumptions)**  

1. **Hardware availability** – Sufficient compute (GPU/TPU clusters) to maintain real‑time particle filtering, voxel updates, and MCTS planning within latency budgets.  
2. **Quality sensor data** – Calibration drift < 0.1 % and reasonable signal‑to‑noise ratio so that GM updates are reliable.  
3. **Ground‑truth bridge** – Access to a symbolic “oracle” (e.g., a formal model checker) capable of evaluating high‑level causal constraints; otherwise self‑verification quality degrades.  
4. **Scalable storage** – Persistent, versioned key‑value and graph stores that support efficient pruning and versioned snapshots without compromising consistency.  
5. **Inter‑module interface stability** – Well‑defined APIs and data schemas; any semantic mismatch will cause hidden errors that MCM may not catch.  
6. **Learning data diversity** – Sufficient coverage of environments and tasks to avoid biased WM/SMI updates; otherwise generalization collapses.  

If these conditions hold, the architecture provides a concrete, modular, and self‑correcting substrate for progressive, general‑intelligence development.
