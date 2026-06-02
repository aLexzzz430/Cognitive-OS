# AGI Architecture Proposal Prompts

## 标准提示词
以下提示词将用于向每个AI系统询问AGI架构设计：

```
You are an expert in artificial general intelligence (AGI) system architecture. Please design a comprehensive AGI architecture proposal that addresses the following components:

1. **Memory Architecture**
   - Short-term/working memory
   - Long-term/episodic memory
   - Semantic/knowledge memory
   - Memory retrieval and storage mechanisms

2. **Reasoning and Planning Loop**
   - Core reasoning engine design
   - Planning and goal decomposition
   - Attention and focus mechanisms
   - Meta-cognition and self-reflection

3. **Learning and Self-Improvement**
   - Online learning mechanisms
   - Offline learning and consolidation
   - Transfer learning capabilities
   - Self-modification and architecture evolution

4. **Tool Use and Action Execution**
   - Tool discovery and selection
   - Action planning and execution
   - Environment interaction
   - Tool composition and chaining

5. **World Model and Representation**
   - World state representation
   - Causal reasoning models
   - Uncertainty quantification
   - Temporal reasoning

6. **Safety and Governance**
   - Value alignment mechanisms
   - Constraint enforcement
   - Monitoring and oversight
   - Fail-safe mechanisms

7. **Evaluation and Benchmarking**
   - Performance metrics
   - Capability assessment
   - Progress tracking
   - Self-evaluation mechanisms

8. **Persistence and Runtime Architecture**
   - State persistence
   - Resource management
   - Scalability considerations
   - Fault tolerance

9. **Multi-Agent and Orchestration**
   - Sub-agent coordination
   - Specialized module design
   - Communication protocols
   - Resource allocation

10. **Engineering Feasibility**
    - Implementation considerations
    - Technical challenges
    - Development roadmap
    - Resource requirements

Please provide a detailed, concrete architecture proposal that could realistically be implemented. Focus on technical specifics rather than philosophical discussion. Include diagrams or pseudocode if helpful.
```

## 提示词变体
对于某些可能需要调整的系统，以下变体将被记录：

### 变体A (适用于中文优先模型)
```
你是一个人工通用智能（AGI）系统架构专家。请设计一个全面的AGI架构提案，涵盖以下组件：
[中文翻译的相同组件列表]
请提供详细、具体的架构提案，关注技术细节而非哲学讨论。如果有助于理解，可以包含图表或伪代码。
```

### 变体B (适用于代码优先模型)
```
Design an AGI system architecture with concrete implementation details. Focus on:
[简化的组件列表]
Provide specific technical solutions, data structures, algorithms, and implementation considerations.
```

## 使用记录
| 系统名称 | 提示词版本 | 访问日期 | 备注 |
|----------|------------|----------|------|
| ChatGPT (GPT-4) | 标准 | 2026-06-02 | |
| Claude (Anthropic) | 标准 | 2026-06-02 | |
| Gemini (Google) | 标准 | 2026-06-02 | |
| Grok (xAI) | 标准 | 2026-06-02 | |
| DeepSeek | 变体A | 2026-06-02 | 中文优先 |
| Qwen | 变体A | 2026-06-02 | 中文优先 |
| Llama 3.1 | 标准 | 2026-06-02 | |
| Mistral Large | 标准 | 2026-06-02 | |

## 收集方法
1. 使用每个系统的公开API或Web界面
2. 记录确切的输入提示词
3. 保存完整的原始输出
4. 记录访问时间和系统版本
5. 最小化的人工编辑（仅格式调整）

## 伦理考虑
- 仅使用公开可访问的系统
- 不绕过付费墙或访问限制
- 不包含私有API密钥
- 尊重使用条款和版权
- 适当引用来源