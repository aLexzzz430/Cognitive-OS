#!/usr/bin/env python3
"""Retry failed model queries with alternatives."""
import json
import urllib.request
import os

API_URL = "https://api.linkapi.org/v1/chat/completions"
API_KEY = "sk-" + "DJ7VzxHAEj3U01p6K7lXkyCBUzTc9fsvZVOD2o0i8U42iDAU"

PROMPT = "You are an AGI architecture researcher. Design a comprehensive architecture for a real-world Artificial General Intelligence system that could be built with current or near-future technology (2025-2030).\n\nYour proposal MUST cover the following 10 dimensions in detail:\n\n1. **Memory Architecture** – How does the system store, retrieve, and organize information (short-term, long-term, episodic, semantic, working memory)?\n\n2. **Reasoning/Planning Loop** – What is the core cognitive cycle? How does it reason, plan, execute, and revise?\n\n3. **Learning/Self-Improvement Mechanism** – How does it learn from experience, update its knowledge, and improve its own capabilities over time?\n\n4. **Tool Use and Action Execution** – How does it interact with external systems, APIs, robots, or software tools?\n\n5. **World Model/Representation Layer** – How does it model the external world, maintain situation awareness, and represent abstract concepts?\n\n6. **Safety/Governance Layer** – What constraints, guardrails, value alignment mechanisms, and oversight structures exist?\n\n7. **Evaluation/Benchmark Strategy** – How do you measure progress toward AGI? What tests validate general intelligence?\n\n8. **Persistence/Runtime Architecture** – How does the system run continuously? Process management, state persistence, fault tolerance.\n\n9. **Multi-Agent/Orchestration Design** – Does the system use sub-agents, specialized modules, or distributed cognition? How are they coordinated?\n\n10. **Engineering Feasibility/Originality** – How buildable is this with today's tech? What's novel about your approach vs. existing systems?\n\nProvide concrete, detailed, technically-specific answers. Avoid vague generalities. Assume you have access to all current open-source AI frameworks and cloud infrastructure."

# Alternative models to retry
MODELS = [
    ("deepseek-chat", ["deepseek-chat", "deepseek-v3", "deepseek-v3.1"]),
    ("deepseek-r1", ["deepseek-r1", "deepseek-reasoner"]),
    ("claude-sonnet-4", ["claude-sonnet-4-20250514", "claude-sonnet-4-20250514-thinking", "claude-sonnet-4-6"]),
    ("gemini-pro", ["gemini-2.5-pro", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash"]),
    ("mistral-large", ["mixtral-8x22b", "mixtral-8x7b", "mistral-medium"]),
]

OUTPUT_DIR = "/root/Cognitive-OS/research/ai_generated_agi_architectures/raw_outputs"


def query_model(model_id):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=180)
        result = json.loads(resp.read().decode("utf-8"))
        if "choices" in result:
            content = result["choices"][0].get("message", {}).get("content", "")
        elif "response" in result:
            content = result["response"]
        elif "content" in result:
            content = result["content"]
        else:
            content = json.dumps(result, indent=2)
        return (model_id, content, None)
    except Exception as e:
        return (model_id, None, str(e))


def try_model_group(group_name, model_ids):
    for mid in model_ids:
        print(f"  Trying {mid}...", end=" ", flush=True)
        _, content, error = query_model(mid)
        if error:
            print(f"FAIL: {error[:60]}")
        else:
            print(f"OK ({len(content)} chars)")
            return (mid, content, None)
    print(f"  All alternatives failed for {group_name}")
    return (model_ids[0], None, f"All alternatives failed")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Retrying failed models with alternatives...\n")
    
    results = []
    for group_name, alternatives in MODELS:
        model_id, content, error = try_model_group(group_name, alternatives)
        results.append((model_id, content, error))
    
    print("\n--- Saving outputs ---")
    for model_id, content, error in results:
        safe_name = model_id.replace("-", "_").replace(".", "_")
        if error:
            filename = f"{safe_name}_ERROR.txt"
            text = f"MODEL: {model_id}\nERROR: {error}\n\nQuery failed."
        else:
            filename = f"{safe_name}_RAW.txt"
            text = f"MODEL: {model_id}\nPROMPT: (see prompts.md)\nDATE: 2026-06-06\n\n--- RAW OUTPUT ---\n\n{content}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  Saved: {filename} ({len(text)} chars)")
    
    successes = sum(1 for _, _, e in results if e is None)
    print(f"\nRetry results: {successes}/{len(results)} successful.")


if __name__ == "__main__":
    main()
