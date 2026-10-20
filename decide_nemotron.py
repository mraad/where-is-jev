"""Nemotron vision-language backbone with constrained A/B/C decisions on CUDA."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import rasterio
from patch_decisions import LABELS, QUESTION, chip


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if not 0 <= a.shard_index < a.num_shards or a.batch_size < 1 or a.limit < 0:
        p.error("Invalid shard, batch size, or limit")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a fresh output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        p.error("This experiment requires CUDA")
    manifest = json.loads((a.manifest / "manifest.json").read_text())
    rows = [f["properties"] for f in json.loads((a.manifest / "patches.geojson").read_text())["features"]]
    rows = rows[a.shard_index::a.num_shards]
    if a.limit and a.limit < len(rows):
        rows = [rows[round(i*(len(rows)-1)/(a.limit-1))] for i in range(a.limit)] if a.limit > 1 else rows[:1]
    if not rows:
        p.error("Empty shard")
    config = json.loads((a.model / "config.json").read_text())
    omni = config["model_type"] == "NemotronH_Nano_Omni_Reasoning_V3"
    engine_options = {"moe_backend": "triton"} if omni else {}
    tick = time.perf_counter()
    llm = LLM(model=str(a.model), trust_remote_code=True, dtype="bfloat16",
              tensor_parallel_size=1, max_model_len=4096, max_num_seqs=a.batch_size,
              max_num_batched_tokens=8192, gpu_memory_utilization=.85 if omni else .6,
              enforce_eager=True, enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 1, "video": 0, **({"audio": 0} if omni else {})},
              seed=0, **engine_options)
    tokenizer = llm.get_tokenizer()
    token_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("Candidate letters must be single tokens")
    token_ids = [ids[0] for ids in token_ids]
    sampling = SamplingParams(temperature=0, max_tokens=1, allowed_token_ids=token_ids, logprobs=3)
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": [{"type": "image", "image": ""},
                                     {"type": "text", "text": QUESTION}]}],
        tokenize=False, add_generation_prompt=True, **({"enable_thinking": False} if omni else {}))
    if omni:
        assert prompt.rstrip().endswith("</think>"), "Omni must enter its direct-answer mode"
    load_seconds = time.perf_counter()-tick
    records, engine_seconds = [], 0.
    start = time.perf_counter()
    with rasterio.open(a.image) as src, (a.out / "decisions.jsonl").open("x") as stream:
        if [src.height, src.width] != manifest["shape"]:
            raise ValueError("Source dimensions differ from manifest")
        for offset in range(0, len(rows), a.batch_size):
            batch = rows[offset:offset+a.batch_size]
            requests = [{"prompt": prompt, "multi_modal_data": {"image": chip(src, row, manifest["patch_size"])}} for row in batch]
            tick = time.perf_counter()
            results = llm.generate(requests, sampling, use_tqdm=False)
            engine_seconds += time.perf_counter()-tick
            if len(results) != len(batch):
                raise ValueError("Engine omitted a request")
            for row, result in zip(batch, results):
                generated = result.outputs[0]
                if len(generated.token_ids) != 1 or generated.token_ids[0] not in token_ids:
                    raise ValueError("Decision is not A, B, or C")
                winner = token_ids.index(generated.token_ids[0])
                record = dict(id=row["id"], decision=LABELS[winner])
                records.append(record)
                stream.write(json.dumps(record)+"\n")
            stream.flush()
            print(f"shard={a.shard_index} processed={len(records)}/{len(rows)}", flush=True)
    provenance = json.loads((a.model / "provenance.json").read_text())
    summary = dict(**provenance, shard_index=a.shard_index, num_shards=a.num_shards,
                   batch_size=a.batch_size, limit=a.limit, patches=len(rows), dtype="bfloat16",
                   device="cuda", gpu_name=torch.cuda.get_device_name(), runtime="vLLM eager",
                   reasoning_enabled=False, moe_backend="triton" if omni else None,
                   load_seconds=load_seconds, pipeline_seconds=time.perf_counter()-start,
                   engine_seconds=engine_seconds, prompt=QUESTION,
                   prompt_sha256=hashlib.sha256(QUESTION.encode()).hexdigest(),
                   manifest_sha256=hashlib.sha256((a.manifest / "patches.geojson").read_bytes()).hexdigest(),
                   counts=dict(Counter(r["decision"] for r in records)),
                   scoring="Greedy one-token A/B/C choice constrained on GPU; no task-specific trained head")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt"}, indent=2))


if __name__ == "__main__":
    main()
