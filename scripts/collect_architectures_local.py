#!/usr/bin/env python3
import json, pathlib, urllib.request, datetime, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[1]/'research'/'ai_generated_agi_architectures'
RAW=ROOT/'raw_outputs'
RAW.mkdir(parents=True,exist_ok=True)
MODELS=[
 ('qwen3_8b','qwen3:8b'),
 ('deepseek_r1_8b','deepseek-r1:8b'),
 ('llama31_8b','llama3.1:8b'),
 ('gemma3_4b','gemma3:4b'),
 ('qwen25_coder_7b','qwen2.5-coder:7b'),
 ('mistral_7b','mistral:7b'),
 ('phi4_mini','phi4-mini'),
 ('granite33_2b','granite3.3:2b'),
]
PROMPT="""Design a practical AGI-oriented cognitive architecture that a small engineering team could implement incrementally. Do not describe a vague superintelligence. Give a concrete system architecture.

Cover these dimensions explicitly:
1. memory architecture
2. reasoning and planning loop
3. learning or self-improvement mechanism
4. tool use and action execution
5. world model or representation layer
6. safety and governance layer
7. evaluation and benchmark strategy
8. persistence and runtime architecture
9. multi-agent or orchestration design
10. engineering feasibility and staged implementation
11. one genuinely non-obvious architectural insight

Also include: component interfaces, data flow, failure modes, and the smallest useful prototype. Separate assumptions from claims. Keep the answer implementation-oriented and under 1600 words."""
SYSTEM="You are an independent AI architecture researcher. Think from first principles. Do not imitate consensus merely because the prompt mentions AGI. Be concrete and falsifiable."

def ask(model):
    payload={
      'model':model,'stream':False,'think':False,'keep_alive':0,
      'options':{'temperature':0.45,'num_ctx':8192,'num_predict':1800},
      'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':PROMPT}]
    }
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read().decode())
manifest=[]
for slug,model in MODELS:
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    print('START',model,flush=True)
    try:
      data=ask(model); content=data.get('message',{}).get('content','').strip()
      (RAW/f'{slug}.md').write_text(f'# Raw output: {model}\n\nCollected: {started}\n\n{content}\n')
      manifest.append({'system':slug,'model':model,'collected_at':started,'status':'ok' if content else 'empty','chars':len(content)})
      print('DONE',model,len(content),flush=True)
    except Exception as e:
      manifest.append({'system':slug,'model':model,'collected_at':started,'status':'error','error':repr(e)})
      print('ERR',model,repr(e),flush=True)
(ROOT/'local_manifest.json').write_text(json.dumps(manifest,indent=2))
(ROOT/'shared_prompt.txt').write_text(PROMPT+'\n')
print('COMPLETE',flush=True)
