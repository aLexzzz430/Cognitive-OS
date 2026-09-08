"""Replay one pinned research request in a new directory, using llama-server.

Downloads one public model (up to about 2.4 GB). Requires the caller to supply
their own llama.cpp b10809 server binary. Never modifies the recorded dataset.
Example: python collection/replay.py smollm2 --server /path/to/llama-server --out /tmp/smollm2-replay
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['smollm2', 'qwen25', 'granite33', 'tinyllama', 'phi3', 'qwen3', 'falcon3', 'danube3'])
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory; must not already exist')
    args = parser.parse_args()
    source = json.loads((ROOT / args.model / 'source.json').read_text())
    request = json.loads((ROOT / args.model / 'request.json').read_text())
    server = args.server.resolve(strict=True)
    out = args.out.resolve()
    if out.exists() or out == ROOT or ROOT in out.parents:
        raise SystemExit('Choose a new output directory outside the recorded collection.')
    out.mkdir(parents=True)
    if shutil.disk_usage(out).free < source['size'] + 750_000_000:
        raise SystemExit('Insufficient disk space for the model and headroom.')
    version = subprocess.run([str(server), '--version'], capture_output=True, text=True, timeout=180, check=True)
    version_text = version.stdout + version.stderr
    (out / 'runtime-version.txt').write_text(version_text)
    if '10809' not in version_text or '5266f24' not in version_text:
        raise SystemExit('This protocol requires llama.cpp b10809, commit 5266f24da.')
    weights = out / source['filename']
    with urllib.request.urlopen(source['url'], timeout=60) as response, weights.open('wb') as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    if weights.stat().st_size != source['size'] or digest(weights) != source['sha256']:
        raise SystemExit('Downloaded weights do not match the recorded model hash/size.')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(server), '-m', str(weights), '--host', '127.0.0.1', '--port', str(port), '-t', '2', '-tb', '2', '-ngl', '0', '-c', '4096', '-np', '1', '--jinja', '--reasoning-budget', '0', '--no-webui']
    (out / 'request.json').write_text(json.dumps(request, indent=2))
    started = datetime.now(timezone.utc).isoformat()
    with (out / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Server exited; inspect server.log')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=3) as r:
                        if json.load(r).get('status') == 'ok':
                            break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Server did not become ready')
                time.sleep(1)
            req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1800) as r:
                response = json.load(r)
            (out / 'response.json').write_text(json.dumps(response, indent=2, ensure_ascii=False))
            text = response['choices'][0]['message']['content']
            (out / 'output.txt').write_text(text)
            (out / 'replay.json').write_text(json.dumps({'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(), 'model_sha256': digest(weights), 'output_sha256': digest(out / 'output.txt'), 'matches_recorded_text': text == (ROOT.parent / 'raw_outputs' / f'{args.model}.txt').read_text(), 'finish_reason': response['choices'][0]['finish_reason'], 'usage': response.get('usage')}, indent=2))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    print(f'Replay saved to {out}. Weights retained there; remove them when no longer needed.')

if __name__ == '__main__':
    main()
