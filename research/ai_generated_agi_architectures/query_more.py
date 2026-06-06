#!/usr/bin/env python3
"""Query remaining models for AGI architecture research."""
import json, urllib.request, os, sys

API = "https://api.linkapi.org/v1/chat/completions"
DIR = "/root/Cognitive-OS/research/ai_generated_agi_architectures/raw_outputs"
os.makedirs(DIR, exist_ok=True)

PROMPT = "You are an AGI architecture researcher. Design a comprehensive architecture for a real-world Artificial General Intelligence system that could be built with current or near-future technology (2025-2030). ..."

models = [
    ("deepseek-chat", "DeepSeek"),
    ("deepseek-r1", "DeepSeek-R1"),
    ("mistral-large", "Mistral"),
]

for model_id, model_name in models:
    safe = model_id.replace("-", "_").replace(".", "_")
    out = f"{DIR}/{safe}_RAW.txt"
    if os.path.exists(out):
        print(f"  ✅ {model_name} 已有")
        continue
    
    # Get key from env
    KEY = os.environ.get("LLM_KEY", "")
    if not KEY:
        # Try reading from stdin
        KEY = sys.stdin.readline().strip()
    
    if not KEY:
        print(f"  ❌ {model_name}: 无API key")
        continue
    
    print(f"  Querying {model_name}...", end=" ", flush=True)
    try:
        data = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 4096
        }).encode()
        req = urllib.request.Request(API, data=data, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}"
        })
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read())
        content = result["choices"][0]["message"]["content"]
        with open(out, "w") as f: f.write(content)
        print(f"✅ ({len(content)} chars)")
    except Exception as e:
        err = f"{DIR}/{safe}_ERROR.txt"
        with open(err, "w") as f: f.write(str(e)[:200])
        print(f"❌ {str(e)[:60]}")
