# Architectural Synthesis: CORTEX Cognitive OS

This document proposes a unified, production-grade software architecture named **CORTEX (Cognitive Operating Runtime and Tool Execution engine)**. CORTEX extracts, refines, and combines the strongest concepts from the 8 surveyed AI system proposals into a concrete, implementation-ready design.

```mermaid
graph TB
    subgraph User Interaction
        IS[Input Stream]
    end

    subgraph Orchestration & Planning (Tokio Runtime)
        S1[System 1 Parser<br/>vLLM Reflexive Mode]
        TQ[Task Queue<br/>RabbitMQ Bus]
        PE[Plan Executor]
        S2[System 2 Planner<br/>MCTS + ToT Search]
    end

    subgraph Knowledge & State
        WM[(Causal State Graph<br/>World Model / filesystem DAG)]
        DB[(State Database<br/>PostgreSQL / JSONB)]
    end

    subgraph Security & Execution
        SG{Safety Gate<br/>Llama Guard}
        TS[Tool Sandbox<br/>Firecracker / LXD]
        CL[(Cryptographic Log<br/>system_audit_ledger)]
    end

    IS --> S1
    S1 --> TQ
    TQ --> PE
    PE --> S2
    S2 <--> WM
    PE <--> DB
    PE --> SG
    SG -->|Pass| TS
    SG -->|Fail| CL
    TS --> CL
```

---

## 1. Memory Architecture (Hybrid Context/Index Store)

CORTEX rejects pure RAG and pure large-context storage. It implements a **Hybrid sliding-window context with transactional state indexing**:
*   **Active Execution Context:** Up to 128k tokens containing the recent conversation, system state, execution traces, and active workspace files.
*   **Vector Semantic Store:** ChromaDB with hierarchical document indexing, using query expansion.
*   **Factual & Invariant Ledger:** Relational PostgreSQL schemas representing system settings, tool schemas, and workspace structures.

```sql
CREATE TABLE agent_state (
    session_id UUID PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    context_tokens INT[],
    world_state JSONB NOT NULL
);

CREATE TABLE system_audit_ledger (
    entry_id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id UUID NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    action_payload JSONB NOT NULL,
    previous_hash BYTEA NOT NULL,
    entry_hash BYTEA NOT NULL
);
```

---

## 2. Reasoning and Planning Loop

*   **System 1 (Reflexive Mode):** Direct generation of structured JSON steps for simple, high-confidence operations (confidence > 0.85).
*   **System 2 (Verification/Search Mode):** Monte Carlo Tree Search (MCTS) combined with Tree-of-Thought (ToT) when confidence is low or safety-critical invariants are involved.
*   **Self-Correction:** Any parser or validation errors automatically trigger a correction step, sending the error trace and schema requirements back to the system planner.

---

## 3. Learning & Self-Improvement

CORTEX logs failed tasks to a local dataset. Once every 24 hours, a background thread compiles these traces and executes a local **LoRA fine-tuning** process (using PyTorch and Llama Stack APIs) to adjust reasoning weights and correct repeated failure modes without updating external APIs.

$$\mathcal{L}_{total} = \mathcal{L}_{task\_completion} + \lambda \mathcal{L}_{safety\_alignment}$$

---

## 4. Safe Tool Execution Sandbox

Tools are written as structured Python modules and executed inside ephemeral **LXD containers** or **Firecracker MicroVMs** with strict network egress policies.
*   **Egress Masking Layer:** A mandatory out-of-band proxy parses outgoing data, masking PII and checking against security blacklists before dispatch.

```python
import subprocess
import json

def execute_sandbox_tool(container_id: str, command: list[str]) -> dict:
    # Restrict cpu and memory usage on runtime
    prefix = ["lxc", "exec", container_id, "--", "sudo", "-u", "sandbox"]
    full_cmd = prefix + command
    try:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
        return {
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Execution timed out."
        }
```

---

## 5. World Model & Representation Layer

*   **Causal State Graph:** The environment state is modeled as a Directed Acyclic Graph (DAG). Nodes represent filesystem entities, environment variables, and network configurations. Edges represent dependencies and causal influence.
*   **Action Simulation:** Before committing to a plan, System 2 runs simulations of the proposed actions on a local transition matrix. The actual execution output is compared against the simulation; discrepancies (prediction errors) trigger a revision of the causal graph.

---

## 6. Safety & Governance Layer

*   **Llama Guard Moderation:** Input prompts and output responses are validated using local Llama Guard models to filter out toxic payloads or prompt injections.
*   **Cryptographic Invariant Gate:** To prevent the agent from mutating its history or disabling security checks, all state transitions and outputs are written to the append-only `system_audit_ledger` table. Each entry is cryptographically chained via SHA-256 containing the hash of the previous record, ensuring absolute auditability.

---

## 7. Evaluation & Benchmark Strategy

*   **Needle-in-a-Haystack Probes:** Executed automatically every 24 hours to measure recall consistency across large context windows.
*   **Automated Regression Suites:** Measures task completion rates, query correctness, and execution latencies across 15 standard developer task scenarios.

---

## 8. Persistence & Runtime Architecture

*   **Tokio Async Scheduler:** Written in Rust, leveraging the Tokio async task runner for non-blocking I/O.
*   **State Serialization:** Serialization of active agent frames uses Protocol Buffers (Protobuf) for high performance and low storage overhead.

---

## 9. Multi-Agent & Orchestration Design

*   **Manager-Worker Delegation:** A centralized orchestrator (Manager) decomposes complex instructions, assigning them to specialized sub-agents (e.g., Coder, Security Checker, Sandbox Runner) over a RabbitMQ message bus.
*   **Consensus Mechanism:** Verification tasks require a majority agreement (minimum 2/3) across separate worker instances before state transitions are finalized.

---

## 10. Engineering Feasibility & Originality

*   **High Feasibility:** Leveraging Kubernetes, local SQLite/PostgreSQL, and lightweight LXD/Firecracker sandboxes makes CORTEX highly deployable on local workstations or private cloud nodes.
*   **Originality:** The core innovations include:
    1.  *Cryptographically Signed Invariant Ledgers* preventing history rewriting by the agent itself.
    2.  *PII Regulatory Masking Proxy* embedded directly into the egress execution layers.
    3.  *DAG-Structured Tool Pipelines* allowing sequential outputs to pipe directly to next inputs, bypassing LLM-overhead on deterministic chains.
