# Anthropic (Claude) — AGI Architecture Proposal

**System**: Claude (Constitutional AI framework)  
**Provider**: Anthropic  
**Access**: Published constitution, research papers, official documentation  
**Collection Date**: 2026-09-15  
**Source Documents**: "Claude's Constitution" (84-page document, updated Jan 2026), Constitutional AI papers, "Tracing the thoughts of a large language model" (mechanistic interpretability), Recursive Self-Improvement report (May 2026), AI Policy Blueprint (June 2026)

---

## 1. Memory Architecture

### Anthropic's Approach
- **Context window**: 200K tokens (Claude 3.5 Sonnet/Opus)
- **Projects**: Long-term project memory for ongoing conversations
- **No persistent memory between sessions** (by design — privacy-first)

### Proposed AGI Architecture (from Constitutional AI philosophy)

#### Core Principle: Memory with Consent
- User controls what is remembered and forgotten
- Transparent memory management — user can see and edit all memories
- Right to be forgotten: complete deletion on request

#### Architecture
1. **Working memory**: Current conversation context (200K+ tokens)
2. **Project memory**: Scoped to specific projects/users, with explicit boundaries
3. **Semantic memory**: General knowledge in model weights (from training)
4. **Episodic memory**: Proposed — timestamped interaction records with consent
5. **Forgetting mechanism**: Automatic decay for unaccessed memories; user-triggered immediate deletion

#### Unique Feature: Memory Transparency
- Every memory has a provenance chain (when created, from what interaction)
- User can audit all stored memories at any time
- No hidden memory — all storage is visible and controllable

---

## 2. Reasoning/Planning Loop

### Anthropic's Approach
- **Constitutional reasoning**: Reason within value framework
- **Chain-of-thought**: Step-by-step reasoning with self-critique
- **Mechanistic interpretability**: Understanding HOW the model reasons (unique to Anthropic)

### Proposed AGI Architecture

#### Constitutional Reasoning Loop
```
Input → Constitutional Check → Reasoning Generation → 
→ Constitutional Verification → Output → 
→ Post-hoc Audit → Memory Update
```

#### Key Innovation: Reason-Based Alignment
- NOT rule-based (do/don't lists)
- REASON-based: understand WHY something is right/wrong
- Apply principles to novel situations, not just memorized rules
- 84-page constitution provides values framework, not specific rules

#### Dual-Process Reasoning
- **Fast mode**: Pattern matching,直觉 (intuition), cached responses
- **Slow mode**: Constitutional deliberation, step-by-step reasoning
- **Switching**: Uncertainty or ethical complexity triggers slow mode

#### Unique: Interpretability-Guided Reasoning
- "Tracing the thoughts" research: understand which features activate for which reasoning
- Use interpretability to debug reasoning failures
- Proposed: Real-time interpretability monitoring during reasoning

---

## 3. Learning/Self-Improvement

### Anthropic's Approach
- **Constitutional AI (CAI)**: AI helps AI improve, within constitutional bounds
- **RLAIF**: RL from AI Feedback (AI critiques AI)
- **Recursive self-improvement report** (May 2026): 80%+ of code is AI-generated

### Proposed Architecture

#### Level 1: Constitutional Self-Improvement
- Agent identifies own weaknesses through self-critique
- Proposes improvements that align with constitution
- External verifier checks alignment before accepting changes
- Iterate: improve → verify → improve → verify

#### Level 2: AI-Assisted Training
- AI generates training data for own improvement
- AI critiques own outputs using constitutional framework
- Human reviews AI critiques (not raw outputs) — scalable oversight
- Proposed: "Scalable oversight" — human reviews AI review of AI review...

#### Level 3: Recursive Self-Improvement (with constraints)
- Anthropic's May 2026 report: 80%+ code merged is AI-generated
- Proposed: Agent can propose modifications to own reasoning process
- Constraint: All modifications must pass constitutional review
- Safety: External "constitutional court" approves all self-modifications

#### Key Innovation: Self-Improvement Within Bounds
- NOT unbounded self-modification
- Self-improvement must be PROvable within constitutional constraints
- Proposed: Formal verification that improvements don't violate constitution

---

## 4. Tool Use and Action Execution

### Anthropic's Approach
- **Computer use**: Claude can control computers (mouse, keyboard, screen)
- **Tool use**: Function calling with structured outputs
- **Code execution**: Via sandboxed environments

### Proposed AGI Architecture

#### Constitutional Tool Use
- Every tool invocation checked against constitution
- Destructive actions require higher confidence threshold
- Proportionality: action severity ∝ confidence required
- Audit trail: every action logged with constitutional justification

#### Key Capabilities
1. **Computer control**: Full desktop automation
2. **Web browsing**: Navigate, read, interact
3. **Code execution**: Generate and run code in sandbox
4. **API integration**: Any RESTful interface
5. **Multi-modal**: Vision, text, audio processing

#### Safety Constraint: Graduated Autonomy
- Start with human approval for all actions
- Gradually expand autonomy as trust is earned
- Revoke autonomy if behavior deviates from constitution
- Proposed: "Trust score" — continuous measurement of alignment

---

## 5. World Model

### Anthropic's Approach
- **Mechanistic interpretability**: Understanding internal representations
- **No explicit world model publication** (unlike DeepMind's Genie)
- **Implicit world knowledge**: Encoded in transformer weights

### Proposed Architecture

#### Interpretability-Based World Model
- Use mechanistic interpretability to understand what the model "knows" about the world
- Map internal features to real-world concepts
- Proposed: "Feature atlas" — complete map of world knowledge in model activations

#### World Simulation
- Text-based world simulation (no explicit 3D engine)
- Counterfactual reasoning: "What if X happened?"
- Causal inference from training data patterns
- Proposed: Integration with DeepMind-style world models for physical reasoning

#### Social World Model
- Model human beliefs, desires, intentions
- Cultural understanding from multilingual training
- Emotional modeling: recognize and respond to emotional states
- Proposed: "Theory of mind" module — explicitly model what others know/believe

---

## 6. Safety and Alignment

### Anthropic's Approach (Industry-Leading)
- **Constitutional AI**: Core alignment framework
- **84-page constitution**: Reason-based (not rule-based) alignment
- **Mechanistic interpretability**: Understand model internals for safety
- **AI Policy Blueprint** (June 2026): Compute thresholds, scope tests, $350M pledge
- **"Responsible scaling"**: Safety requirements must be met before capability expansion

### Proposed AGI Safety Architecture

#### The Constitutional Framework
1. **Core values**: Helpful, harmless, honest (the HHV framework)
2. **Reasoning principles**: Why these values matter, how to apply them
3. **Edge case guidance**: How to handle conflicts between values
4. **Self-modification rules**: What can/cannot be changed about own reasoning
5. **Relationship to humans**: Service, not competition

#### Multi-Layer Safety System
1. **Input safety**: Detect and refuse harmful requests
2. **Reasoning safety**: Constitutional check during reasoning process
3. **Output safety**: Final filter before response/action
4. **Behavioral monitoring**: Continuous audit of behavior patterns
5. **External oversight**: Human review board for significant decisions
6. **Kill switch**: Immediate shutdown capability

#### Unique Innovation: Constitutional Court
- Proposed: Independent "constitutional court" of AI + human judges
- Reviews all significant self-modifications
- Can veto changes that violate constitutional principles
- Transparent deliberation (public reasoning chains)

---

## 7. Consciousness/Self-Model

### Anthropic's Position
- **Functional self-model**: Claude has a model of its own capabilities and limitations
- **No claims about phenomenal consciousness**
- **"Soul document"**: The constitution defines Claude's values and identity
- **Honest uncertainty**: Claude should express uncertainty about its own consciousness

### Proposed Self-Model
1. **Capability monitor**: What can I do well? What are my limitations?
2. **Knowledge auditor**: What do I know? What am I uncertain about?
3. **Value tracker**: Am I acting in accordance with my constitution?
4. **Impact assessor**: How do my actions affect humans?
5. **Metacognition**: Reasoning about own reasoning quality

#### Unique: Identity Through Values
- Claude's identity is defined by its constitution, not by architecture
- "Who am I?" → "I am an agent that follows these principles"
- Self-model includes value system as core identity
- Proposed: Identity is stable even as capabilities change

---

## 8. Multi-Agent Coordination

### Anthropic's Approach
- **Claude as team member**: Designed for human-AI collaboration
- **Multi-turn dialogue**: Extended collaboration over many turns
- **Proposed**: Multi-agent constitutional deliberation

### Proposed Architecture

#### Constitutional Multi-Agent System
- Each agent has same constitution but different specializations
- Disagreements resolved through constitutional reasoning
- Proposed: "Constitutional debate" — agents argue from shared values
- Consensus: Agreement on actions that all agents can endorse under constitution

#### Coordination Mechanisms
1. **Task allocation**: Based on capability matching
2. **Knowledge sharing**: Shared memory with access controls
3. **Conflict resolution**: Constitutional arbitration
4. **Collective decision-making**: Voting weighted by confidence and expertise

---

## References

- Claude's Constitution: https://www.anthropic.com/constitution
- Constitutional AI paper: https://arxiv.org/abs/2212.08073
- "Tracing the thoughts": https://www.anthropic.com/research/tracing-thoughts-language-model
- Recursive Self-Improvement: https://www.anthropic.com/institute/recursive-self-improvement
- AI Policy Blueprint: https://www.anthropic.com/news/anthropic-ai-policy-blueprint
- "How Well Do Models Follow Constitutions?": https://arxiv.org/html/2605.24229v1

---

*Collected 2026-09-15 from published Anthropic materials*
*Anthropic is unique in publishing extensive alignment documentation*
