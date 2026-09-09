"""Optional local check: reproduce recorded input/output decoding, no generation."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model-root", type=Path, required=True)
parser.add_argument("--spark-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
results = []
for model in json.loads((root / "models.json").read_text()):
    slug = model["slug"]
    folder = root / "runs" / slug
    request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
    model_dir = args.spark_dir if slug == "spark" else args.model_root / slug
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=slug == "spark")
    rendered = tokenizer.apply_chat_template(request["messages"], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    assert rendered == request["rendered_prompt"], slug
    assert tokenizer(rendered, add_special_tokens=False)["input_ids"] == request["input_ids"], slug
    assert tokenizer.decode(response["output_ids"], skip_special_tokens=False) == response["raw_output"], slug
    assert tokenizer.decode(response["output_ids"], skip_special_tokens=True) == (folder / "readable.txt").read_text(encoding="utf-8"), slug
    # HF local snapshot metadata contains the pinned revision and LFS SHA-256.
    files = json.loads((folder / "model-files.json").read_text())["files"]
    verified_weights = 0
    for record in files:
        if not record["file"].endswith(".safetensors"):
            continue
        meta = model_dir / ".cache/huggingface/download" / (record["file"] + ".metadata")
        if meta.exists():
            revision, etag = meta.read_text().splitlines()[:2]
            assert revision == model["revision"], (slug, record["file"])
            if len(etag) == 64:
                assert etag == record["sha256"], (slug, record["file"])
                verified_weights += 1
    results.append(dict(system=slug, template_input_decode_match=True,
                        cached_lfs_hash_matches=verified_weights))
    print(slug, "template and input/output token records match", flush=True)
(root / "tokenizer-verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
