"""Verify collection provenance and packet coverage, not research quality.

Usage: python collection/verify.py [--collection-only]
No network requests, model inference, or credential access.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('smollm2', 'qwen25', 'granite33', 'tinyllama', 'phi3', 'qwen3', 'falcon3', 'danube3')
HOSTED = 'gemini_flash_reference'
DIMENSIONS = ('memory', 'planning', 'learning', 'tools', 'world_representation', 'governance', 'evaluation', 'persistence', 'orchestration', 'feasibility', 'originality')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_model(root, name):
    folder = root / 'collection' / name
    request_file, response_file, run_file = [folder / f'{n}.json' for n in ('request', 'response', 'run')]
    raw = root / 'raw_outputs' / f'{name}.txt'
    request, response, run, source, meta = [json.loads(p.read_text()) for p in (request_file, response_file, run_file, folder / 'source.json', folder / 'model.json')]
    assert request['messages'] == [{'role': 'user', 'content': (root / 'collection/prompt.txt').read_text()}], f'{name}: prompt changed'
    for field, expected in {'temperature': 0.2, 'top_p': 0.9, 'seed': 20260908, 'max_tokens': 2400, 'stream': False}.items():
        assert request[field] == expected, f'{name}: sampling changed: {field}'
    assert request['chat_template_kwargs'] == {'enable_thinking': False}
    choice = response['choices'][0]
    assert isinstance(choice['message']['content'], str) and choice['message']['content'].strip(), f'{name}: no proposal text'
    assert raw.read_text() == choice['message']['content'], f'{name}: raw output differs from response'
    for path, key in [(request_file, 'request_sha256'), (response_file, 'response_sha256'), (raw, 'content_sha256')]:
        assert sha(path) == run[key], f'{name}: {key} mismatch'
    assert run['finish_reason'] == choice['finish_reason'], f'{name}: inconsistent finish reason'
    assert run['usage'] == response['usage'], f'{name}: inconsistent token usage'
    assert run['started_at'] <= run['finished_at'], f'{name}: invalid timestamps'
    assert source['revision'] == meta['sha'] and re.fullmatch('[0-9a-f]{40}', source['revision'])
    assert source['url'] == f'https://huggingface.co/{source["model_repository"]}/resolve/{source["revision"]}/{source["filename"]}'
    entry = next(x for x in meta['siblings'] if x['rfilename'] == source['filename'])
    assert source['sha256'] == entry['lfs']['sha256'] and source['size'] == entry['lfs']['size'], f'{name}: upstream identity mismatch'
    assert re.fullmatch('[0-9a-f]{64}', source['sha256'])
    response_kind = 'prompt_echo' if raw.read_text().strip() == (root / 'collection/prompt.txt').read_text().strip() else 'non_echo_text'
    return {'model': name, 'characters': len(raw.read_text()), 'finish_reason': choice['finish_reason'], 'completion_tokens': response['usage']['completion_tokens'], 'source_sha256': source['sha256'], 'response_kind': response_kind}

def verify_hosted_reference(root):
    folder = root / 'collection/gemini-flash-reference'
    meta = json.loads((folder / 'source.json').read_text())
    assert (folder / 'request.txt').read_bytes() == (root / 'collection/prompt.txt').read_bytes(), 'Hosted reference prompt differs'
    assert meta['backend_model_revision'] is None and meta['sampling_parameters'] is None, 'Do not infer unknown service settings'
    assert [r['label'] for r in meta['responses']] == ['A', 'B']
    for item in meta['responses']:
        raw = folder / f'response-{item["label"]}.txt'
        assert sha(raw) == item['sha256'], 'Hosted reference content hash mismatch'
        assert len(raw.read_text()) == item['characters'], 'Hosted reference character count mismatch'
    combined = root / meta['combined_raw_file']
    expected = ('=== Collector label: Gemini Flash web, candidate A ===\n\n'
                + (folder / 'response-A.txt').read_text()
                + '\n\n=== Collector label: Gemini Flash web, candidate B ===\n\n'
                + (folder / 'response-B.txt').read_text())
    assert combined == root / f'raw_outputs/{HOSTED}.txt'
    assert combined.read_text() == expected and sha(combined) == meta['combined_raw_sha256'], 'Combined hosted view differs from recorded answers'
    with (folder / 'comparison.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    pairs = [(r['candidate'], r['dimension']) for r in rows]
    assert len(pairs) == 22 and set(pairs) == {(c, d) for c in ('A', 'B') for d in DIMENSIONS}
    for row in rows:
        relative, line = row['evidence'].split('#L')
        assert relative == f'response-{row["candidate"]}.txt'
        text = (folder / relative).read_text().splitlines()
        assert 1 <= int(line) <= len(text) and re.match(r'^\d+\. ', text[int(line) - 1]), 'Hosted evidence anchor is not a heading'
    return {'systems': 1, 'candidate_answers': 2, 'comparison_rows': len(rows), 'backend_revision': 'unknown', 'primary_model_count_increment': 0}

def verify(root=ROOT, collection_only=False):
    results = [verify_model(root, name) for name in MODELS]
    hosted_reference = verify_hosted_reference(root)
    assert set(p.stem for p in (root / 'raw_outputs').glob('*.txt')) == set(MODELS) | {HOSTED}, 'Unexpected/missing raw outputs'
    assert [r['model'] for r in results if r['response_kind'] == 'prompt_echo'] == ['danube3'], 'Reassess the documented response classification'
    if not collection_only:
        for name in ('README.md', 'prompts.md', 'summary.md', 'synthesis.md', 'sources.md'):
            assert (root / name).is_file() and (root / name).stat().st_size > 0, f'Missing {name}'
            report = (root / name).read_text()
            assert '**Working draft' not in report and '**Preliminary analysis' not in report, f'{name}: report still marked incomplete'
        with (root / 'comparison.csv').open(newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert set(('model', 'dimension', 'proposal', 'limits', 'evidence', 'cohort')) <= set(reader.fieldnames or []), 'Missing comparison columns'
        expected = {(m, d) for m in (*MODELS, HOSTED) for d in DIMENSIONS}
        actual = [(r['model'], r['dimension']) for r in rows]
        assert set(actual) == expected and len(actual) == len(expected), 'Comparison must cover all 99 system/dimension pairs exactly once'
        for row in rows:
            assert all(row[k].strip() for k in ('proposal', 'limits', 'evidence')), 'Empty comparison evidence'
            assert row['evidence'].startswith(f'raw_outputs/{row["model"]}.txt'), 'Evidence points to a different model'
            assert row['cohort'] == ('exploratory_hosted' if row['model'] == HOSTED else 'primary_local')
            for reference in row['evidence'].split('; '):
                relative, number = reference.split('#L')
                assert relative == f'raw_outputs/{row["model"]}.txt'
                assert 1 <= int(number) <= len((root / relative).read_text().splitlines()), 'Evidence anchor is outside the raw file'
        # Explicitly prevent accidental credential/file bundles in the research packet.
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            assert p.suffix.lower() not in {'.gguf', '.pem', '.key'}, f'Unexpected model/key file: {p}'
            assert not p.name.startswith('.env'), f'Unexpected environment file: {p}'
    return {'local_models_verified': len(results), 'total_systems': len(results) + 1, 'local_prompt_echoes': sum(r['response_kind'] == 'prompt_echo' for r in results), 'comparison_rows_verified': 0 if collection_only else 99, 'results': results, 'exploratory_hosted_reference': hosted_reference, 'scope': 'Verifies recorded artifact consistency, echo classification and structural coverage; non-echo text is not a quality score. Does not authenticate remote provenance independently or judge research quality.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(collection_only=args.collection_only), indent=2))
