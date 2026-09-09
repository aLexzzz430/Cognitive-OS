"""Record exact local model/tokenizer/code files; omit paths outside the model."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--slug", required=True)
parser.add_argument("--model-dir", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
model = next(m for m in json.loads((root / "models.json").read_text()) if m["slug"] == args.slug)
records = []
for path in sorted(args.model_dir.iterdir()):
    if not path.is_file() or path.suffix not in (".safetensors", ".json", ".py", ".model", ".jinja"):
        continue
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    records.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha))
assert any(r["file"].endswith(".safetensors") for r in records)
out = root / "runs" / args.slug / "model-files.json"
out.write_text(json.dumps(dict(model=model, files=records), indent=2) + "\n", encoding="utf-8")
print(args.slug, len(records), "model files hashed", flush=True)
