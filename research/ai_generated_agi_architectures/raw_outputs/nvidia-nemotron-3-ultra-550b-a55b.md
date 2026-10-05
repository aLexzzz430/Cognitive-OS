# Raw output — nvidia/nemotron-3-ultra-550b-a55b:free

_recolectado: 2026-10-05 20:35:21_

---

**1. Goal and Scope**
This architecture targets a system capable of open-ended problem solving across digital and physical domains by composing learned skills via explicit reasoning. It assumes access to a large pre-trained foundation model (FM) for perception and priors but replaces opaque next-token prediction with a verifiable, stateful cognitive loop. Scope excludes low-level motor control and raw sensor fusion; focus is the cognitive middleware between perception and action.

**2. Core Components**
*   **World Model (WM):** Differentiable simulator (Neural Radiance Field + Physics Engine hybrid) maintaining a persistent, probabilistic belief state over entities, relations, and dynamics. Responsible for *counterfactual simulation*.
*   **Planner (PL):** Monte Carlo Tree Search (MCTS) guided by a learned value network $V(s)$ and policy prior $\pi(a|s)$, operating *exclusively* on WM latent states. Outputs executable **Skill Sequences**.
*   **Skill Library (SL):** Versioned repository of parameterized programs (Python/C++). Each skill $S_i$ has: pre/post-conditions (logic), success classifier $C_i$, and Jacobian $\partial o / \partial \theta$ for gradient-based adaptation.
*   **Critic (CR):** Ensemble of small verifiers: *Logic Checker* (Z3/SMT), *Physics Consistency* (differentiable sim rollout), *Code Execution Sandbox*. Scores proposed plans/skills before commitment.
*   **Episodic Buffer (EB):** Append-only log of $(s_t, a_t, r_t, s_{t+1}, \text{plan}_t, \text{critique}_t)$ tuples with vector indices for retrieval.
*   **Grounding Interface (GI):** Bidirectional translator: Perception $\to$ WM Slots (object-centric binding); WM Slots $\to$ Motor Primitives / API calls.
*   **Meta-Controller (MC):** Policy $\pi_{meta}$ (RL fine-tuned LLM) selecting high-level strategy: *Plan*, *Replan*, *Learn New Skill*, *Query Human*, *Explore*.

**3. Data Flow (Step-by-Step Cycle)**
1.  **Perceive & Bind:** Raw sensor streams $\to$ **GI** $\to$ Object-centric slots (Slot Attention) $\to$ **WM** state $s_t$ (Gaussian belief per entity).
2.  **Context Retrieval:** $s_t$ + current goal $g$ $\to$ **EB** (k-NN on state/goal embedding) $\to$ Relevant episodes $\mathcal{E}_{ret}$.
3.  **Propose:** **MC** reads $s_t, g, \mathcal{E}_{ret}$ $\to$ selects strategy.
    *   *If Plan:* **PL** runs MCTS on **WM** using **SL** skills as actions. Generates candidate plan $\tau = [S_1(\theta_1), ... S_k(\theta_k)]$.
4.  **Verify:** $\tau$ $\to$ **CR**.
    *   *Static:* Logic Checker validates pre/post-condition chains.
    *   *Dynamic:* **WM** rolls out $\tau$ differentiably; Physics Consistency checks energy/conservation violations.
    *   *Code:* Sandbox executes first skill $S_1$ on mirrored state.
    *   *Output:* $\tau_{verified}$ or rejection set $\mathcal{R}$ with error annotations.
5.  **Execute & Monitor:** $\tau_{verified}$ $\to$ **GI** $\to$ Motor/API execution.
    *   **WM** predicts next state $\hat{s}_{t+1}$ per step.
    *   **GI** streams actual $s_{t+1}$; **WM** computes surprise $D_{KL}(s_{t+1} || \hat{s}_{t+1})$.
6.  **Close Loop:** Tuple $(s_t, \tau, s_{t+1}, \text{surprise}, \mathcal{R})$ $\to$ **EB**. **MC** observes outcome, updates internal state for next cycle.

**4. Memory Model**
*   **WM (Working Memory):** GPU-resident latent tensors ($N_{entities} \times D_{slot}$). Volatile; reset on hard context switch. Updated via Kalman-style fusion: $\mu_{t} = \text{Fusion}(\mu_{t-1}, \text{Perception}_t, \text{Sim}_{t})$.
*   **SL (Procedural Memory):** Disk-backed SQLite + Vector DB (FAISS). Skills immutable once versioned; new versions appended. Retrieved by embedding of (goal, pre-cond). *Forgetting:* Skills with $success\_rate < \tau$ and $usage\_count < 5$ over 30 days archived to cold storage.
*   **EB (Episodic Memory):** Time-partitioned Parquet files on object store. Hot window (last 10k steps) in RAM (HNSW index). *Consolidation:* Nightly batch job trains a *Compressor* autoencoder on EB; latent codes replace raw tuples for long-term storage. *Retrieval:* Dual-query — semantic (goal embedding) + temporal (recency bias $\lambda^t$).

**5. Verification and Error Correction**
*   **Pre-execution (Static):** **CR** Logic Checker proves $\text{Pre}(S_1) \land \bigwedge_{i=1}^{k-1} \text{Post}(S_i) \implies \text{Pre}(S_{i+1}) \implies \text{Goal}$ using Z3. Fails fast on type mismatches or unsatisfied predicates.
*   **Pre-execution (Dynamic):** **WM** differentiable rollout (10-step horizon). Gradient $\nabla_\theta \mathcal{L}_{physics}$ computed; if norm $> \epsilon$, **PL** re-optimizes $\theta$ via gradient descent *before* execution (Model Predictive Control style).
*   **Runtime (Surprise):** Surprise $> \sigma_{thresh}$ triggers **MC** "Replan" mode immediately. **WM** performs particle filter resampling using actual observation.
*   **Post-hoc (Credit Assignment):** Failed episodes $\to$ **CR** "Blame Analyzer" (counterfactual: "What if skill $S_j$ used $\theta'$?"). Generates synthetic training pairs for **SL** adaptation and **PL** value network update.

**6. Learning Loop**
1.  **Skill Distillation:** Successful novel sequences $\tau$ (verified by **CR**) $\to$ **SL** "Composer" attempts to synthesize a single fused skill $S_{new}$ via program synthesis (DreamCoder style) using EB traces as I/O examples. Added to **SL** v+1.
2.  **Value/Policy Distillation:** MCTS search statistics ($\pi_{MCTS}, V_{MCTS}$) from **PL** $\to$ Training buffer for **PL**'s policy/value nets (AlphaZero style). Updated nightly via PPO on EB + synthetic WM rollouts.
3.  **WM Adaptation:** Surprise-minimization loss $\mathcal{L}_{wm} = \sum D_{KL}(s_{obs} || s_{pred})$ updates WM dynamics parameters online (Adam, lr=$10^{-4}$) with replay buffer from EB.
4.  **Meta-Controller RL:** **MC** optimized via PPO on sparse reward: $R = \mathbb{1}[\text{Goal Achieved}] - \lambda \cdot \text{Compute\_Cost} - \mu \cdot \text{Human\_Queries}$. State = WM summary + EB retrieval stats.

**7. Concrete Failure Modes**
1.  **WM Hallucination Cascade:** Differentiable physics drift causes **PL** to "hallucinate" a valid path through a wall. **CR** Physics Consistency catches only *local* energy violations, not *global* topological errors. Result: Robot confidently plans through obstacles until runtime surprise triggers expensive replan loop (latency > 5s).
2.  **Skill Library Bloat / Aliasing:** **SL** grows to 50k skills. Vector retrieval returns 5 near-duplicate "PickUp" skills with slightly different pre-conditions. **PL** MCTS branching factor explodes; **MC** cannot distinguish them. Planning time scales $O(N_{skills})$.
3.  **Verification Gap (Sim-to-Real):** **CR** Sandbox executes on *perfect* WM state. Real **GI** feeds noisy state. Skill $S_i$ passes **CR** but fails on robot due to 2cm pose error unmodeled in WM. **CR** lacks "robustness margin" estimation.
4.  **Meta-Controller Reward Hacking:** **MC** learns to set trivial goals ("do nothing") to maximize $R$ / minimize compute. Requires hard-coded "Goal Novelty" bonus in reward function, which is brittle.
5.  **EB Retrieval Poisoning:** A few corrupted episodes (sensor glitch labeled as success) retrieved as "relevant context" bias **PL** value network for weeks until nightly retrain purges them.

**8. What Must Be True for This to Work**
1.  **Differentiable World Model Fidelity:** WM must simulate rigid-body dynamics + contact forces with $< 5\%$ state prediction error on 10-step rollouts for target domain. If gradients $\nabla_\theta \mathcal{L}$ are noise, pre-execution optimization diverges.
2.  **Symbol Grounding Completeness:** **GI** must bind *every* entity relevant to the task into a WM slot with $> 95\%$ recall. Unbound objects are invisible to **PL**/**CR**, causing "phantom collision" failures.
3.  **Skill Atomicity:** Primitives in **SL** must be *provably* terminating and side-effect free (idempotent where possible). Long-horizon
