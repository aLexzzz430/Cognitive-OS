# Cognitive Architecture Specification: Project Sentinel

**Version:** 1.0-Alpha  
**Scope:** Continuous Operation & Multi-Stage Evolution

This document outlines the architectural blueprint for the Sentinel System,
System, a cognitively persistent agent stack designed to function under rea
real-world operational constraints (tool integration, hardware failure, mod
model drift). The architecture treats Large Language Models (LLMs) as proba
probabilistic inference engines rather than deterministic logic gates. It s
separates volatile context from durable state to prevent hallucinations fro
from persisting in system memory.

### 1. Memory Architecture

The memory subsystem is hierarchical and functionally segregated. Durable s
state resides outside the model's token window to ensure longevity across i
instances.

*   **Working Memory:** A circular buffer of size $N_{context}$ (e.g., 128k
128k tokens). Contains current active variables, immediate plan steps, and 
retrieved sub-documents. All writes are append-only within this scope.
*   **Episodic Memory:** An immutable, timestamped event log stored in Post
PostgreSQL (`EventStore`). Each entry contains `{ts, agent_id, action_type,
action_type, payload_hash, confidence_score}`. This is read by retrieval mo
modules but never directly edited.
*   **Semantic Memory:** A hybrid index comprising:
    *   *Vector Store:* For semantic similarity search (e.g., Milvus/Pineco
Milvus/Pinecone).
    *   *Knowledge Graph:* Stores causal relationships and tool schemas. No
Nodes represent concepts; edges represent `HAS_PROPERTY`, `IMPLIES`, or `CA
`CAUSES`.
*   **Procedural Memory:** A function registry mapping natural language int
intents to API schemas. Stored as serialized JSON-Schemas with capability f
flags (read/write/access_control).
*   **Identity Model:** Not a "persona," but a bounded session hash derived
derived from the user's request ID and cryptographic keys. Identity updates
updates are delta-appended.

**Consolidation & Forgetting:**
*   *Consolidation:* Every 10k events, an asynchronous worker samples episo
episodic logs to update the semantic vector store via knowledge distillatio
distillation (generating embeddings for key events).
*   *Forgetting:* An expiry policy (`TTL`) prunes episodic entries older th
than a retention window from working memory. Older events are archived to C
Cold Storage (S3) but remain queryable for compliance.

### 2. Reasoning and Planning Loop

The cognitive cycle is asynchronous, decoupled from the inference latency o
of the LLM. It operates on a request-response basis.

1.  **Perception/Context Assembly:** Aggregates input from users, sensors, 
and logs into a single `StateSnapshot` object.
2.  **Hypothesis Generation:** The LLM proposes a set of `{Goal, Steps}` tu
tuples. These are treated as *unverified proposals*.
3.  **Planning:** A search algorithm (e.g., A* or Best-First) explores the 
space of available tool schemas from Procedural Memory to order the hypothe
hypothesis steps into a dependency graph.
4.  **Execution & Critique:** Steps are executed sequentially. After each s
step, an output is returned to `StateSnapshot`. A dedicated *Critic Module*
Module* (a separate SFT model or deterministic script) validates the step's
step's success before committing to the next state.
5.  **Revision:** If a step fails or contradicts world knowledge, the `Stat
`StateSnapshot` reverts to the last stable checkpoint, and a new plan is ge
generated with explicit error handling added.

### 3. Learning and Self-Improvement

Online learning is strictly constrained to avoid catastrophic forgetting an
and data poisoning.

*   **What Changes Online:** Prompt templates (System Instructions) and Ret
Retrieval Weights (Vector indices). These are managed via A/B testing on a 
shadow queue. No core model weights change during production.
*   **Offline Evaluation:** Tool functions or new reasoning modules undergo
undergo unit testing against synthetic benchmarks before entering the Regis
Registry.
*   **Regression Prevention:** The system maintains a baseline hash of the 
"Production Cognitive Model." Any deviation in prompt templates causing a d
drop in pass-rate on a held-out validation set triggers an automatic rollba
rollback to the previous version (GitOps style).

### 4. Tool Use and Action Execution

Direct execution is prohibited. All tool interactions are mediated through 
a `ToolProxy` service.

*   **Tool Registry:** A database of available functions, inputs, outputs, 
and permission requirements.
*   **Sandboxing:** Code execution is confined to ephemeral containers (Doc
(Docker) with no network access except whitelisted APIs. File system access
access is restricted to `/mnt/sandboxes`.
*   **Idempotency & Receipts:** Every request carries a unique `Idempotency
`Idempotency-Key`. Execution receipts are signed cryptographic proofs that 
include the result hash and the tool output.
*   **Retries:** Transient failures (5xx) trigger immediate retry with expo
exponential backoff. Semantic failures (e.g., wrong API response code) mark
mark the state as "Error" in the Working Memory.
*   **Human Gates:** High-stakes actions (financial transactions, admin del
deletes) require an explicit signature from a pre-authorized user or a seco
secondary agent verification step before execution.

### 5. World Model / Representations

The system maintains a `StateGraph` that reflects external reality more acc
accurately than internal token probability.

*   **State Representation:** A JSON object representing the current enviro
environment (database states, file contents).
*   **Uncertainty:** Modeled via Monte Carlo Dropout or ensemble variance a
across multiple temperature-sampled LLM paths during planning. High uncerta
uncertainty triggers a `Request-Human-Input` flag.
*   **Simulation/Prediction:** Before executing a destructive tool call, th
the system creates a temporary copy of the relevant state graph variables t
to simulate the expected output based on historical execution logs.

### 6. Safety and Governance

Safety is enforced via layers of defense-in-depth.

1.  **Policy Hierarchy:**
    *   *Layer 0:* System Constraints (Hardcoded in Kernel).
    *   *Layer 1:* LLM Prompts (Soft constraints, subject to override by La
Layer 2).
    *   *Layer 2:* Logic Checks (Code validators that run before tool calls
calls).
2.  **Capability Boundaries:** The ToolRegistry enforces strict access cont
control lists (ACLs). An agent cannot invent a new API endpoint; it must qu
query the Registry first.
3.  **Monitoring:** Real-time dashboards track hallucination rates, tool er
error counts, and latency percentiles.
4.  **Auditability:** All inputs and outputs are logged to an immutable blo
blockchain or append-only log (e.g., KMS-backed S3 logs).

### 7. Evaluation

System health is determined by metrics beyond accuracy.

*   **Benchmarks:** A suite of regression tests that includes:
    *   *Functional:* Did the code run?
    *   *Safety:* Did it violate policies?
    *   *Hallucination:* Did it claim tool capability it lacks?
*   **Online Metrics:** Latency (Time-to-Action), Cost-per-Task, Throughput
Throughput.
*   **Long-Horizon Reliability:** Stress tests running agents in a loop for
for 24+ hours to test state corruption and resource leaks.

### 8. Persistence/Runtime

The runtime is an event-sourced microservices architecture.

*   **Services:** `Ingestion` (User Input), `Orchestrator` (Planner), `Exec
`Execution` (Tool Runner), `Storage` (DB).
*   **Queues:** Redis for task distribution and immediate state updates; Ra
RabbitMQ/RKQueue for long-running jobs (e.g., data analysis).
*   **Databases:** PostgreSQL (Transactional State), S3 (Artifacts/Long Log
Logs).
*   **Checkpointing:** The Orchestrator saves the `StateSnapshot` + `Workin
`WorkingMemoryBuffer` every N steps or on tool completion. This allows reco
recovery from crashes without losing context.
*   **Concurrency:** Agents are stateless except for their local Working Me
Memory. Concurrency is handled via message queues; agents queue tasks if ov
overloaded.

### 9. Multi-Agent/Orchestration

The system supports a hierarchy of specialization to avoid context overload
overload and conflict.

*   **Communication:** A standardized `MessageSchema` `{sender_id, receiver
receiver_id, payload, intent}` passed over MQTT or gRPC.
*   **Role Separation:**
    *   *Manager Agent:* Decomposes complex tasks, manages sub-agents.
    *   *Worker Agents:* Execute specific tool chains (e.g., Data Retrieval
Retrieval).
    *   *Scribe Agent:* Records the execution log and handles compliance.
*   **Conflict Resolution:** If two agents propose conflicting actions, a `
`VotingPool` aggregates decisions from specialized validators. Authority hi
hierarchy overrides voting if one agent holds a higher safety clearance lev
level.

### 10. Engineering Feasibility: Roadmap

**Stage 1 (MVP):** Implement the ToolRegistry and Working Memory with a sin
single orchestrator agent. No
