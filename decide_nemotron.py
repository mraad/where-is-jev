"""Nemotron vision-language backbone with constrained A/B/C decisions on CUDA."""

import json
import time

import rasterio
from patch_decisions import LABELS, QUESTION, chip, worker_setup, worker_summary


def main():
    p, a, manifest, rows = worker_setup(__doc__)
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        p.error("This experiment requires CUDA")
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
    summary = worker_summary(
        a, rows, records, gpu_name=torch.cuda.get_device_name(), runtime="vLLM eager",
        reasoning_enabled=False, moe_backend="triton" if omni else None,
        load_seconds=load_seconds, pipeline_seconds=time.perf_counter()-start,
        engine_seconds=engine_seconds,
        scoring="Greedy one-token A/B/C choice constrained on GPU; no task-specific trained head")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt"}, indent=2))


if __name__ == "__main__":
    main()
