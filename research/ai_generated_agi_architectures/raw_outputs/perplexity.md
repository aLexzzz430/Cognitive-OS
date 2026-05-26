# Perplexity (Claude 3 Opus backend) — AGI Architecture Proposal

## Memory Architecture
**Search-Grounded Memory**:
- **Real-Time Search Index**: Every query extends memory via web search — the internet itself serves as external memory.
- **Session Context**: Conversation history with source citations (Perplexity's core feature).
- **Pro Search Mode**: Multi-step research with intermediate search results preserved.
- **Collections**: User-curated knowledge bases for persistent domain memory.

## Reasoning/Planning Loop
**Research-Grounded Chain-of-Thought**:
1. Parse question, identify information gaps.
2. Execute multiple targeted web searches to fill gaps.
3. Cross-reference sources for consistency and authority.
4. Synthesize answer with inline citations.
5. Identify remaining uncertainty — flag for human review if confidence below threshold.
6. Pro mode: iterate steps 2-4 with refined queries based on intermediate findings.

## Learning Mechanism
**Feedback-Driven Curation**:
- User upvotes/downvotes on answers and sources.
- Pro search learns which sources are authoritative for which query types.
- **Copilot Mode**: Interactive guidance refines search strategy based on user corrections.
- No model training from user data — improvement comes from better search and source ranking.

## Tool Use & Action Execution
- **Web Search**: Primary tool, with real-time indexing.
- **Wolfram Alpha**: Computational knowledge for math and science.
- **Code Execution**: Python sandbox for data analysis.
- **File Upload**: PDF, image, and document analysis.
- **API Access**: Enterprise plan includes API for programmatic use.

## World Model / Representation
**The Internet as World Model**:
- No internal world simulation — relies on real-time web data for ground truth.
- Source authority scoring: .edu, .gov, and peer-reviewed sources weighted higher.
- Temporal awareness: recency filters for time-sensitive queries.
- Contradiction detection: flags when sources disagree.

## Safety / Governance
- **Source Transparency**: Every claim is cited — users can verify independently.
- **Misinformation Resistance**: Authoritative source preference reduces hallucination.
- **Privacy**: No user data used for training.
- **Academic Integrity**: Designed to support research, not replace it.

## Evaluation & Benchmark Strategy
- **Factuality**: Measured by citation accuracy and source authority.
- **Freshness**: Time-to-index for breaking information.
- **Completeness**: Coverage of relevant sources.
- **User Satisfaction**: NPS scores and task completion rates.

## Persistence / Runtime
- **Stateless + Search**: Core engine is stateless — context carried via conversation history.
- **Cloud Infrastructure**: AWS-based with global edge caching.
- **Real-Time Indexing**: Web crawl pipeline for freshness.

## Multi-Agent / Orchestration
- **Parallel Search**: Multiple search queries executed concurrently.
- **Source Aggregation**: Results merged and deduplicated.
- **Specialist Routers**: Math → Wolfram, facts → web search, code → sandbox.

## Engineering Feasibility
Perplexity is a deployed product with millions of users. The search-grounded approach 
is production-proven. Not AGI per se, but demonstrates how external knowledge 
retrieval can substitute for internal world models.
Cost: Operational already. Scaling cost proportional to query volume.
