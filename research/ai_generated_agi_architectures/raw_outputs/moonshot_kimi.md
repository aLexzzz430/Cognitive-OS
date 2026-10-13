# AGI Architecture Proposal — Moonshot AI (Kimi) perspective

> **Attribution note:** Representative reconstruction drafted by the research author
> (AI-assisted) in the documented architectural philosophy of Moonshot AI's Kimi
> family, grounded in public releases (Kimi k1.5 technical report, Kimi K2
> agentic focus, long-context work). Not a verbatim model output. See `sources.md`.

## Thesis
Long context plus agentic tool-use training equals general agency. If the model
can hold a huge working state and has been RL-trained to interleave thinking
with tool calls over long horizons, most "agent architecture" becomes
unnecessary. **Train the agency in; don't scaffold it on.**

## 1. Memory architecture
**The long context is the memory**: 200K–1M token windows hold entire
trajectories, tool outputs, and plans without external stores. For beyond
that, standard retrieval. The research bet (k1.5 lineage): RL training with
long-CoT in long contexts teaches the model to *manage* its own memory —
summarizing, prioritizing, discarding — as a learned skill.

## 2. Reasoning / planning loop
**Interleaved thinking-and-acting**: the model is explicitly trained (K2-style)
on trajectories where reasoning tokens and tool calls alternate over dozens
of steps. Planning emerges from this training: the model learns to set
subgoals, use tools to make progress, and recover from failures — because its
RL rewards came from completing long agentic tasks, not from next-token
prediction.

## 3. Learning / self-improvement
RL on **agentic task completion** as the primary post-training signal:
multi-step tool-use episodes with verifiable outcomes (did the code run? did
the task complete?). Long-context RL (k1.5's contribution) extends this to
very long horizons. Self-improvement = harder agentic tasks + longer horizons
in the RL curriculum.

## 4. Tool use and action execution
Tool use is **native to the training**, not a fine-tuned afterthought: the
base training mix includes massive tool-interaction trajectories. The model
treats tools as extensions of its own action space. Standard sandboxing and
confirmation policies for side effects.

## 5. World model / representation layer
Built through interaction: the long agentic trajectories in training expose
the model to software environments (repos, terminals, browsers), building an
implicit world model of how digital systems behave. The representation is
learned from *doing*, not just reading.

## 6. Safety / governance layer
Standard alignment stack with emphasis on **long-horizon safety**: evals
specifically for multi-step agentic behavior (does the agent stay on-task over
50 steps? does it take unauthorized shortcuts?). The risk surface of agentic
models is different from chatbots, and the safety work targets that
difference.

## 7. Evaluation and benchmark strategy
**Agentic benchmarks as the north star** (SWE-bench-style, multi-step
tool-use suites), long-context benchmarks, and standard reasoning suites. The
key metric: task completion rate over long horizons — the thing RL directly
optimizes.

## 8. Persistence / runtime architecture
Stateless model + harness-managed sessions, with the long context checkpointed
for resume. Open weights (K2) mean the runtime is deployable by anyone; the
architecture is designed so the *training recipe* (long-context agentic RL) is
the moat, not the serving stack.

## 9. Multi-agent / orchestration design
Minimal: a single strong agentic model with a long context often outperforms
multi-agent scaffolding. Where parallelism helps, simple fan-out/fan-in of
subtask agents with shared context. The philosophy: **one great agent beats
five mediocre ones plus a framework**.

## 10. Engineering feasibility
High: open weights, standard serving, no exotic components. The RL training
recipe is the hard part to reproduce, but using the resulting models is easy.

## 11. Originality / non-obvious insight
The sharpest claim in this packet: **agentic RL in long contexts may subsume
agent engineering**. If training directly optimizes 100-step tool-use
trajectories, the hand-built planners, routers, and orchestrators that the
rest of the industry builds become unnecessary — the capability is in the
weights. This is the most aggressive "just train it" position after OpenAI's.
