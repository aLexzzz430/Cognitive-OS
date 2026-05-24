# Raw Output — Xunfei Spark (Pragmatic Engineer Perspective)

**Model:** astron-code-latest (Xunfei Spark)
**System Prompt:** Pragmatic AI engineer from a production AGI startup
**Date:** 2026-05-24
**Tokens:** 3470

---

# Production AGI Architecture: A Pragmatic Engineering Blueprint

**Context:** We are not building a research toy. We are building a system that must run on consumer hardware or standard cloud clusters, handle millions of requests with sub-second latency, and not hallucinate itself into a lawsuit. The goal is **reliability at scale**, not theoretical optimality.

---

## 1. Memory Architecture
**1. What it is:** A tiered storage system separating transient context (session), episodic logs (history), and semantic knowledge (facts).
**2. How it works:**
*   **Short-term:** Standard KV Cache in the LLM (context window).
*   **Long-term:** A vector database (e.g., Milvus, Pinecone) for semantic retrieval + a graph DB (e.g., Neo4j) for structured relationships.
*   **Semantic:** A compressed, fine-tuned embedding model specific to our domain, stored in a read-only index.
**3. How it connects:** The "Retriever" module acts as a gatekeeper. Before the LLM generates a token, it queries the vector/graph stores with the current query. Relevant chunks are injected into the prompt as "context" or used to update a "memory summary" vector.
**4. Key Design Choices:**
*   **Trade-off:** We do *not* use a massive, monolithic vector store for everything. We use **hierarchical retrieval**: First, a lightweight keyword search filters candidates; then, a dense vector search ranks them. This cuts latency by 40% and cost by 60%.
*   **Failure Mode:** Vector drift. We implement an automated "re-indexing" pipeline weekly to prune stale data.
*   **Cost:** Vector DBs are expensive to scale. We use quantization (4-bit) on embeddings to reduce storage footprint by 75%.

## 2. Reasoning/Planning Loop
**1. What it is:** A deterministic control loop where the agent decomposes tasks, executes them, and verifies results before moving on.
**2. How it works:**
*   **Decomposition:** A "Planner" model breaks a high-level goal into a DAG (Directed Acyclic Graph) of sub-tasks.
*   **Execution:** A "Worker" model executes each node, often calling tools.
*   **Revision:** A "Critic" model checks the output of the node against the success criteria. If it fails, the loop backtracks to the planner.
**3. How it connects:** This is the "brain" of the system. The output of the Planner is a JSON schema that drives the Orchestrator. The Critic's feedback is fed back into the Planner's context window.
**4. Key Design Choices:**
*   **Trade-off:** We avoid "Chain of Thought" (CoT) for every step in production due to latency. Instead, we use **Tree of Thoughts (ToT)** with a strict depth limit (max 3 levels).
*   **Engineering Reality:** We do not let the LLM "think" for 10 minutes. We use a **hard timeout** and a "fallback to heuristic" mode if the loop runs too long.
*   **Failure Mode:** Infinite loops. We enforce a "step counter" and a "max retries" policy that forces a human-in-the-loop escalation.

## 3. Learning / Self-Improvement Mechanism
**1. What it is:** A mechanism to update the system's knowledge or behavior based on execution feedback without retraining the base model from scratch.
**2. How it works:**
*   **Experience Logging:** Every interaction (prompt, tool call, result, human correction) is logged with a unique ID.
*   **Synthetic Data Generation:** A separate "distillation" model reviews successful trajectories and generates new training pairs.
*   **Fine-Tuning:** We perform **PEFT (Parameter-Efficient Fine-Tuning)** (LoRA/QLoRA) on a nightly basis, not online.
**3. How it connects:** The Learning module is decoupled from the Inference engine. The Inference engine runs the base model; the Learning module updates the LoRA adapters, which are swapped in during a zero-downtime deployment.
**4. Key Design Choices:**
*   **Trade-off:** **No Online Learning.** Updating weights in real-time is a security nightmare and causes "catastrophic forgetting." We stick to batch updates.
*   **Safety:** All training data is passed through a "safety filter" before being used to generate new training examples.
*   **Cost:** We only retrain on the top 0.1% of high-value interactions. Training on everything is too expensive.

## 4. Tool Use and Action Execution
**1. What it is:** The capability to interact with external APIs, databases, and code interpreters to perform actions the LLM cannot do natively.
**2. How it works:** The LLM outputs a structured JSON tool call. A "Dispatcher" validates the schema, authenticates the request, executes the code/API call, and returns the result to the LLM.
**3. How it connects:** The Tool Registry is a central config file. The Planner requests tools; the Dispatcher executes them; the result is fed back to the LLM.
**4. Key Design Choices:**
*   **Trade-off:** We do not use "function calling" for everything. We use **Code Interpreter** for complex logic (Python) and **API wrappers** for simple lookups.
*   **Safety:** All tool calls are sandboxed. Network access is restricted via egress firewall rules.
*   **Failure Mode:** API rate limits. We implement a **token-bucket rate limiter** at the Dispatcher level to prevent killing downstream services.
*   **Latency:** We use async execution for tool calls and stream results back to the user immediately.

## 5. World Model / Representation Layer
**1. What it is:** The internal state representation of the environment and the agent's goals.
**2. How it works:** Instead of a complex neural network simulating physics (too expensive), we use a **Symbolic State Machine** combined with the LLM's latent space. The LLM holds the "intent"; the State Machine holds the "truth."
**3. How it connects:** The State Machine is the source of truth for the "Ground Truth" check. The LLM predicts what *should* happen; the State Machine verifies what *did* happen.
**4. Key Design Choices:**
*   **Trade-off:** We reject "Neural World Models" (like DreamerV3) for production. They are too computationally heavy and brittle. We use **Structured JSON State** for the world.
*   **Engineering Reality:** If the LLM hallucinates the state, the State Machine rejects the action. This is the "Guardrail."
*   **Cost:** Minimal. JSON is cheap to store and process.

## 6. Safety / Governance Layer
**1. What it is:** A set of rules and checks that ensure the system adheres to safety, legal, and ethical guidelines.
**2. How it works:** A **multi-stage filter pipeline**:
    *   **Input Filter:** Checks user prompts for PII, jailbreaks, or malicious intent.
    *   **Output Filter:** Checks generated text for toxicity, bias, or PII leakage.
    *   **Action Filter:** Validates tool calls before execution.
**3. How it connects:** This layer sits *outside* the main inference loop. It acts as a proxy. If the filter says "No," the request is blocked immediately, never touching the LLM.
**4. Key Design Choices:**
*   **Trade-off:** We use **Rule-Based + Small Classifier** models for filtering, not the main LLM. This reduces cost by 90% and latency by 50%.
*   **Failure Mode:** False positives blocking valid requests. We maintain a "whitelist" of known good patterns to bypass strict filters.
*   **Cost:** Running the main LLM for safety is wasteful. We use a distilled model (e.g., 7B) for filtering.

## 7. Evaluation and Benchmark Strategy
**1. What it is:** A continuous testing framework to measure performance, safety, and reliability.
**2. How it works:**
*   **Automated Regression:** A suite of 500+ test cases (Golden Set) runs on every deployment.
*   **Human-in-the-Loop (HITL):** A small sample of real traffic is reviewed by humans daily.
*   **Red Teaming:** Automated agents attempt to break the system (jailbreak, prompt injection) continuously.
**3. How it connects:** The evaluation pipeline is part of the CI/CD. If the "Safety Score" drops below 99%, the deployment is automatically rolled back.
**4. Key Design Choices:**
*   **Trade-off:** We do not rely on static benchmarks (MMLU, etc.) for production. They don't reflect real-world utility. We use **Task Success Rate (TSR)** and **Time-to-Completion**.
*   **Engineering Reality:** We track **Latency P99** and **Error Rate** per tool call, not just "accuracy."
*   **Failure Mode:** Overfitting to the test set. We use "Adversarial Test Generation" to constantly evolve the test cases.

## 8. Persistence / Runtime Architecture
**1. What it is:** The infrastructure that ensures state continuity and high availability.
**2. How it works:**
*   **State:** All agent state (memory, plan progress) is stored in a distributed cache (Redis) with a database backup (PostgreSQL).
*   **Runtime:** Microservices architecture. The LLM is a separate service. The Orchestrator is stateless (re-creates state from Redis on restart).
**3. How it connects:** If the Orchestrator crashes, the State Service holds the "save point." The new Orchestrator instance picks up the ID and resumes.
**4. Key Design Choices:**
*   **Trade-off:** We do not store the full context window in Redis. We store the **summary** and the **last N tokens**. This keeps memory usage low.
*   **Cost:** Redis is expensive at scale. We use **TTL (Time To Live)** policies to auto-expire old sessions.
*   **Failure Mode:** Data loss. We implement **event sourcing** for critical state changes, allowing us to replay the last 10 minutes of actions if the state is corrupted.

## 9. Multi-Agent / Orchestration Design
**1. What it is:** A system where multiple specialized agents (e.g., Researcher, Coder, Reviewer) collaborate to solve a problem.
**2. How it works:** A central **Orchestrator** (or "Manager") assigns tasks to specialized agents. Agents communicate via a shared message bus.
**3. How it connects:** The Orchestrator maintains the global state. Agents only see the context relevant to their task.
**4. Key Design Choices:**
*   **Trade-off:** We avoid "swarm intelligence" (too chaotic). We use a **Master-Slave** hierarchy with a fixed set of roles.
*   **Engineering Reality:** We limit the number of agents to **5 max**. More agents = exponential latency and cost.
*   **Failure Mode:** Communication loops. Agents arguing forever. We enforce a "voting mechanism" or a "timeout" to break deadlocks.

## 10. Engineering Feasibility
**1. What it is:** An assessment of what is possible to build *today* with current tech.
**2. How it works:** We prioritize **Open Source Models** (Llama 3, Mistral) over proprietary APIs for cost control and customization. We use **vLLM** for high-throughput serving.
**3. How it connects:** The entire stack is built on standard cloud primitives (AWS/GCP), avoiding proprietary "black box" vendor lock-ins where possible.
**4. Key Design Choices:**
*   **Trade-off:** We accept that our models are less "smart" than GPT-4o, but we gain **predictability, cost control, and data privacy**.
*   **Failure Mode:** Model collapse. We ensure we have human-curated data for fine-tuning, not just synthetic data.
*   **Cost:** We run inference on **Spot Instances** with checkpointing to reduce compute costs by 70%.

## 11. Originality / Non-Obvious Insights
**1. What it is:** The specific "secret sauce" that makes our system better than a standard RAG+LLM setup.
**2. How it works:**
*   **Insight 1:** **"Stateless Intelligence, Stateful Context."** The LLM doesn't need to "remember" everything; it just needs to be told what matters *right now*. We move memory management *out* of the model and *into* the retrieval layer.
*   **Insight 2:** **"Deterministic Verification of Probabilistic Output."** We don't trust the LLM's reasoning. We trust the *verification* of the reasoning. The system is designed to fail fast and correct, not to be perfect on the first try.
**3. How it connects:** This shifts the architecture from "Make the model smarter" to "Make the system more robust."
**4. Key Design Choices:**
*   **Trade-off:** We sacrifice "magic" for "reliability." Users might see the agent "stumble" (retry) rather than getting a hallucinated answer immediately.
*   **Failure Mode:** Over-engineering the verification. We keep the verification rules simple and human-readable.
*   **Cost:** The "non-obvious" part is often **simplicity**. The most robust AGI systems are those that admit they are not AGI and use strict constraints.

---

### Summary of Engineering Philosophy
*   **Do not train from scratch.** Fine-tune or RAG.
*   **Do not trust the LLM.** Verify everything.
*   **Do not optimize for "intelligence."** Optimize for "reliability" and "cost per task."
*   **Do not build for the future.** Build for what ships next Tuesday.

This architecture is designed to be **deployed today**, scaled to millions of users, and audited for safety. It is boring, expensive, and reliable. That is how we build AGI.