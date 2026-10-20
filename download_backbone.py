"""Download an official checkpoint at a recorded immutable revision."""

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("model")
p.add_argument("out", type=Path)
a = p.parse_args()
if not a.model.startswith(("Qwen/", "nvidia/")):
    p.error("Expected an official Qwen or NVIDIA repository")
a.out.mkdir(parents=True, exist_ok=True)
provenance = a.out / "provenance.json"
if provenance.exists():
    info = json.loads(provenance.read_text())
    if info["model"] != a.model:
        p.error("Output belongs to another model")
else:
    info = dict(model=a.model, revision=HfApi().model_info(a.model).sha)
    provenance.write_text(json.dumps(info, indent=2))
print(json.dumps(info), flush=True)
start = time.perf_counter()
snapshot_download(a.model, revision=info["revision"], local_dir=a.out,
                  allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt",
                                  "*.jinja", "*.model", "README.md"], max_workers=8)
assert (a.out / "config.json").exists()
assert list(a.out.glob("*.safetensors"))
(a.out / "download_timing.json").write_text(
    json.dumps(dict(download_seconds=time.perf_counter()-start), indent=2))
print("DOWNLOAD COMPLETE", flush=True)
