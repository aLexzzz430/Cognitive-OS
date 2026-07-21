# Summary

## Common patterns
- All center on large neural model as core substrate
- Reasoning = learned CoT (OpenAI o, DeepSeek R1) OR explicit search (DeepMind, PPGW, JEPA)
- Memory beyond context window only in JEPA + PPGW
- Alignment = RLHF universally; only JEPA + PPGW embed it architecturally

## Key disagreement
Token-prediction scaling (OpenAI, DeepSeek, Qwen, xAI, Mistral) vs latent-prediction world models (JEPA, DeepMind, PPGW)

## Notable ideas
- JEPA latent prediction (model representations, not sensory detail)
- DeepMind model-based RL + tree search
- DeepSeek R1 / OpenAI o RL-trained deliberation
- PPGW global workspace + active inference + corrigibility