"""GPU-only, three-candidate scoring of RGB patches with Qwen3-VL."""

import json
import time

import rasterio
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from patch_decisions import LABELS, QUESTION, chip, worker_setup, worker_summary


def main():
    p, a, manifest, rows = worker_setup(__doc__)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error("This experiment requires a BF16-capable CUDA GPU")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tick = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    token_ids = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError("Candidate letters must be single tokens")
    token_ids = [ids[0] for ids in token_ids]
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-tick
    records, gpu_seconds = [], 0.
    start = time.perf_counter()
    with rasterio.open(a.image) as src, (a.out / "decisions.jsonl").open("x") as output:
        for offset in range(0, len(rows), a.batch_size):
            batch = rows[offset:offset+a.batch_size]
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": chip(src, row, manifest["patch_size"])},
                {"type": "text", "text": QUESTION}]}] for row in batch]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", padding=True).to("cuda")
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                weights = logits[:, token_ids].softmax(-1)
                choice_mass = logits.softmax(-1)[:, token_ids].sum(-1)
            end.record()
            torch.cuda.synchronize()
            gpu_seconds += begin.elapsed_time(end) / 1000
            weights, choice_mass = weights.cpu().tolist(), choice_mass.cpu().tolist()
            for row, scores, mass in zip(batch, weights, choice_mass):
                winner = max(range(3), key=lambda i: scores[i])
                record = dict(id=row["id"], decision=LABELS[winner], candidate_weights=scores,
                              candidate_token_mass=mass)
                records.append(record)
                output.write(json.dumps(record)+"\n")
            output.flush()
            print(f"shard={a.shard_index} processed={len(records)}/{len(rows)}", flush=True)
            del inputs, logits
    summary = worker_summary(
        a, rows, records, gpu_name=torch.cuda.get_device_name(),
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(), load_seconds=load_seconds,
        pipeline_seconds=time.perf_counter()-start, gpu_forward_seconds=gpu_seconds,
        scoring="One forward pass; A/B/C next-token logits; relative weights are not calibrated probabilities")
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "prompt"}, indent=2))


if __name__ == "__main__":
    main()
