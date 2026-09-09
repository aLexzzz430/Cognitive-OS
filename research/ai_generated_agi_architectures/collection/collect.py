"""Collect one real local response; never overwrite an existing record."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=Path, required=True)
parser.add_argument('--slug', required=True)
args = parser.parse_args()
entry = next(m for m in json.loads((ROOT/'models.json').read_text()) if m['slug'] == args.slug)
out = ROOT/'runs'/args.slug
out.mkdir(parents=True, exist_ok=True)
if (out/'response.json').exists():
    raise SystemExit('Existing response preserved; no repeat collection.')
prompt = (ROOT/'prompt.txt').read_text(encoding='utf-8')
messages = [{'role':'user', 'content':prompt}]
torch.set_num_threads(4)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
set_seed(20260909)
started_utc = datetime.now(timezone.utc).isoformat()
trust_code = args.slug == 'spark'
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=trust_code)
rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(rendered, add_special_tokens=False, return_tensors='pt').to('cuda')
print('LOADING', args.slug, 'prompt tokens', inputs.input_ids.shape[-1], flush=True)
model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=trust_code,
    torch_dtype=torch.bfloat16, attn_implementation='eager', device_map='cuda', use_safetensors=True).eval()
config = GenerationConfig(max_new_tokens=1400, do_sample=False, use_cache=True,
    bos_token_id=model.generation_config.bos_token_id, eos_token_id=model.generation_config.eos_token_id,
    pad_token_id=tokenizer.eos_token_id)
request = dict(model=entry, messages=messages, rendered_prompt=rendered,
    input_ids=inputs.input_ids[0].tolist(), generation_config=config.to_dict(),
    seed=20260909, dtype='bfloat16', attention='eager', enable_thinking=False,
    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
(out/'request.json').write_text(json.dumps(request, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
torch.cuda.synchronize()
started = time.perf_counter()
with torch.inference_mode():
    result = model.generate(**inputs, generation_config=config, use_model_defaults=False)
torch.cuda.synchronize()
ids = result[0, inputs.input_ids.shape[-1]:].tolist()
raw = tokenizer.decode(ids, skip_special_tokens=False)
readable = tokenizer.decode(ids, skip_special_tokens=True)
eos = config.eos_token_id
eos = [eos] if isinstance(eos, int) else eos or []
response = dict(output_ids=ids, raw_output=raw, generated_tokens=len(ids),
    termination='eos' if ids and ids[-1] in eos else 'length',
    seconds=time.perf_counter()-started,
    started_utc=started_utc, completed_utc=datetime.now(timezone.utc).isoformat(),
    model=entry, output_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    request_sha256=hashlib.sha256((out/'request.json').read_bytes()).hexdigest(),
    collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    python=platform.python_version(), platform=platform.system(),
    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
    packages={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','huggingface-hub','tokenizers','safetensors']})
(out/'response.json').write_text(json.dumps(response, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
raw_dir = ROOT.parent/'raw_outputs'
raw_dir.mkdir(exist_ok=True)
(raw_dir/f'{args.slug}.txt').write_text(raw, encoding='utf-8')
(out/'readable.txt').write_text(readable, encoding='utf-8')
print('COLLECTED', args.slug, len(ids), response['termination'], flush=True)
