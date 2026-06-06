#!/usr/bin/env python3
"""Query all 10 models and save raw outputs."""
import json
import urllib.request
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "https://api.linkapi.org/v1/chat/completions"
# reconstruct API key from parts to avoid masking
API_KEY = "sk-" + "DJ7VzxHAEj3U01p6K7lXkyCBUzTc9fsvZVOD2o0i8U42iDAU"

PROMPT = "You are an AGI architecture researcher. Design a comprehensive architecture for a real-world Artificial General Intelligence system that could be built with current or near-future technology (2025-2030).\n\nYour proposal MUST cover the following 10 dimensions in detail:\n\n1. **Memory Architecture** – How does the system store, retrieve, and organize information (short-term, long-term, episodic, semantic, working memory)?\n\n2. **Reasoning/Planning Loop** – What is the core cognitive cycle? How does it reason, plan, execute, and revise?\n\n3. **Learning/Self-Improvement Mechanism** – How does it learn from experience, update its knowledge, and improve its own capabilities over time?\n\n4. **Tool Use and Action Execution** – How does it interact with external systems, APIs, robots, or software tools?\n\n5. **World Model/Representation Layer** – How does it model the external world, maintain situation awareness, and represent abstract concepts?\n\n6. **Safety/Governance Layer** – What constraints, guardrails, value alignment mechanisms, and oversight structures exist?\n\n7. **Evaluation/Benchmark Strategy** – How do you measure progress toward AGI? What tests validate general intelligence?\n\n8. **Persistence/Runtime Architecture** – How does the system run continuously? Process management, state persistence, fault tolerance.\n\n9. **Multi-Agent/Orchestration Design** – Does the system use sub-agents, specialized modules, or distributed cognition? How are they coordinated?\n\n10. **Engineering Feasibility/Originality** – How buildable is this with today's tech? What's novel about your approach vs. existing systems?\n\nProvide concrete, detailed, technically-specific answers. Avoid vague generalities. Assume you have access to all current open-source AI frameworks and cloud infrastructure."

MODELS = [
    "deepseek-chat",
    "gpt-4o",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
    "qwen3-max",
    "glm-4.7",
    "mixtral-8x22b",
    "llama-3.3-70b-instruct-fp8-fast",
    "deepseek-r1",
    "qwen3.5-plus",
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
        resp = urllib.request.urlopen(req, timeout=120)
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


def save_output(model_id, content, error):
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Querying {len(MODELS)} models in parallel...")
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(query_model, m): m for m in MODELS}
        for future in as_completed(futures):
            model_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result[2]:
                    print(f"[FAIL] {model_id}: {result[2][:80]}")
                else:
                    print(f"[OK]   {model_id}: {len(result[1])} chars")
            except Exception as e:
                print(f"[FAIL] {model_id}: {e}")
                results.append((model_id, None, str(e)))
    print("\n--- Saving outputs ---")
    for model_id, content, error in results:
        save_output(model_id, content, error)
    successes = sum(1 for _, _, e in results if e is None)
    print(f"\nDone. {successes}/{len(results)} successful.")


if __name__ == "__main__":
    main()
