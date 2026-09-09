"""Download pinned public model snapshots without remote code execution."""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--model-root', type=Path, required=True)
parser.add_argument('--slug', required=True)
args = parser.parse_args()
entry = next(m for m in json.loads((ROOT/'models.json').read_text()) if m['slug'] == args.slug)
for attempt in range(1, 4):
    try:
        snapshot_download(entry['model_id'], revision=entry['revision'],
                          local_dir=args.model_root/entry['slug'], max_workers=2,
                          allow_patterns=['*.json', '*.jinja', '*.txt', '*.model', '*.safetensors', 'LICENSE', 'README.md'] + (['*.py'] if args.slug == 'spark' else []),
                          token=False)
        break
    except Exception as exc:
        print('DOWNLOAD_ATTEMPT_FAILED', entry['slug'], attempt, type(exc).__name__, flush=True)
        if attempt == 3:
            raise
        time.sleep(5)
print('SNAPSHOT_READY', entry['slug'], entry['revision'], flush=True)
