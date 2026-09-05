# AGI Architecture Proposal: Perplexity

## Model: Perplexity Pro / Sonar
## Provider: Perplexity AI
## Date: 2026-09-05

---

## Executive Summary

Perplexity's AGI architecture is built around **retrieval-native intelligence** — the thesis that AGI must be grounded in verifiable, real-time knowledge, not static parametric memory. Rather than building a larger foundation model, Perplexity has architected an **AI-native search and reasoning system** that combines multiple best-in-class LLMs with real-time retrieval, source citation, and multi-step research capabilities. This is AGI as an answer engine — intelligence that is always up-to-date, verifiable, and connected to the world's knowledge.

## Core Architecture

### 1. Foundation: Retrieval-Augmented Generation (RAG) as the Core Primitive

Perplexity's architecture treats retrieval not as an add-on but as the central cognitive operation:

- **Multi-source retrieval:** Simultaneous querying of web search, academic indexes, news archives, and structured databases. The retrieval layer is the primary memory system.
- **Real-time web indexing:** Continuous indexing of the public web ensures that answers are grounded in current information.
- **Verification pipeline:** Every claim is cross-referenced against multiple sources. The model is trained to prioritize information corroborated by independent sources.
- **Citation as architecture:** Source attribution is not cosmetic — it's an architectural constraint that the model must produce provenance for every factual claim.

### 2. Multi-Model Orchestration

Rather than one monolithic model, Perplexity orchestrates multiple models:

- **Query understanding model:** A model specialized for parsing user intent and decomposing complex queries into sub-queries.
- **Retrieval reranker:** A dedicated model that scores and ranks retrieved documents by relevance.
- **Generation model:** Multiple backends (GPT-4, Claude, Sonar, Llama) produce the answer, with a selection policy picking the best.
- **Verification model:** A separate verification step checks generated outputs against retrieved sources before delivery.
- **Specialized search models:** Domain-specific models for academic research, code search, news analysis, and general web search.

### 3. Reasoning as Research

Perplexity turns reasoning into an **iterative research process**:

- **Auto-query decomposition:** Complex questions are automatically broken into sub-questions, each requiring its own search+analysis cycle.
- **Multi-step research mode (Pro Search):** The system engages in extended research sessions: ask a question → search → read → refine → search again → synthesize → deliver.
- **Source triangulation:** Information from multiple sources is compared, conflicts are flagged, and the most corroborated answer is preferred.
- **Follow-up chaining:** The system maintains a persistent research context across follow-up questions, building on previous findings.

### 4. Memory Architecture

- **Session-based working memory:** Conversation history maintains context across a research session.
- **Collection system:** Users can save and organize research findings into "collections" — a semantic memory layer.
- **External knowledge retrieval:** No parametric knowledge cutoff — the model always accesses current information.
- **Thread continuation:** Persistent conversation threads with full search history, enabling long-running research projects.

### 5. Interface Architecture

- **Natural language to structured query:** User questions are parsed into structured search queries with operators (site:, date:, filetype:).
- **Multi-format output:** Answers can include text, tables, code blocks, images, and video results.
- **Interactive follow-ups:** Suggested follow-up questions generated based on the current research context.
- **Pro Search depth:** Configurable research depth — from quick answers to deep research with 10+ rounds of iterative analysis.

## Path to AGI

Perplexity's vision:

1. **AGI as the ultimate research assistant** — synthesizing all human knowledge
2. **Always truthful** — every claim verified against sources
3. **Always current** — no knowledge cutoffs, real-time updates
4. **Always accessible** — free and premium tiers democratize access
5. **Collaborative intelligence** — combining human research goals with AI reasoning

## Estimated Key Innovations

- **RAG as primary cognitive architecture** (not an add-on)
- **Multi-model orchestration** with best-of-breed selection
- **Auto-query decomposition** for complex research
- **Citation-native generation** — provenance as architecture constraint
- **Real-time web indexing** as knowledge substrate

## Limitations for AGI

- No independent foundation model — relies on third-party models
- No persistent long-term memory beyond collections
- No autonomous agency — operates in query-response mode
- No world model or causal understanding
- No continuous learning from user interactions
- Heavy dependence on web infrastructure — offline capability limited
- Reasoning depth depends on search quality, not intrinsic capability
- Not designed for autonomous multi-agent coordination