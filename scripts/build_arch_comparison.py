#!/usr/bin/env python3
import json, pathlib, urllib.request, csv, datetime, re
ROOT=pathlib.Path(__file__).resolve().parents[1]/'research'/'ai_generated_agi_architectures'
RAW=ROOT/'raw_outputs'
OUT=ROOT/'analysis'
OUT.mkdir(exist_ok=True)
files=[
 ('Qwen3 8B','qwen3_8b.md'),
 ('Llama 3.1 8B','llama31_8b.md'),
 ('Gemma 3 4B','gemma3_4b.md'),
 ('Qwen2.5 Coder 7B','qwen25_coder_7b.md'),
 ('Mistral 7B','mistral_7b.md'),
 ('Phi-4 Mini','phi4_mini.md'),
 ('Granite 3.3 2B','granite33_2b.md'),
 ('GPT-5.6 Sol','chatgpt_gpt56_sol.md'),
]
fields=['memory_architecture','reasoning_planning_loop','learning_self_improvement','tool_use_action','world_model','safety_governance','evaluation_benchmarks','persistence_runtime','multi_agent_orchestration','engineering_feasibility','originality_non_obvious']

def ask(text):
    prompt='''Extract the architecture proposal into strict JSON with exactly these string keys:
memory_architecture, reasoning_planning_loop, learning_self_improvement, tool_use_action, world_model, safety_governance, evaluation_benchmarks, persistence_runtime, multi_agent_orchestration, engineering_feasibility, originality_non_obvious.
Each value must be a concise factual summary of what THIS output actually proposes, <=35 words. Do not add ideas not present. Return JSON only.

OUTPUT TO ANALYZE:
'''+text
    payload={'model':'qwen3:8b','stream':False,'think':False,'keep_alive':0,'format':'json',
             'options':{'temperature':0.1,'num_ctx':8192,'num_predict':700},
             'messages':[{'role':'system','content':'You are a precise research data extractor. No invention.'},{'role':'user','content':prompt}]}
    req=urllib.request.Request('http://127.0.0.1:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=600) as r: return json.loads(r.read().decode())
rows=[]
for system,fn in files:
    text=(RAW/fn).read_text()
    print('EXTRACT',system,flush=True)
    try:
        d=ask(text); content=(d.get('message',{}).get('content') or '').strip(); obj=json.loads(content)
        row={'system':system,'file':'raw_outputs/'+fn}
        for f in fields: row[f]=str(obj.get(f,'')).replace('\n',' ').strip()
        rows.append(row)
        (OUT/(fn+'.json')).write_text(json.dumps(row,indent=2,ensure_ascii=False))
        print('DONE',system,flush=True)
    except Exception as e:
        print('ERR',system,repr(e),flush=True)
        rows.append({'system':system,'file':'raw_outputs/'+fn,**{f:'EXTRACTION_ERROR' for f in fields}})
with (ROOT/'comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['system','file']+fields); w.writeheader(); w.writerows(rows)
(ROOT/'analysis.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
print('COMPLETE',len(rows),flush=True)
