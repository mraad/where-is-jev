#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p results/gc1_probe
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
pids=()
sizes=(4 8 16 32)
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u detect.py \
        --all-valid --model checkpoints/sam3_hub --prompt "circular storage tank" \
        --limit 64 --batch-size "${sizes[$gpu]}" --dtype bfloat16 \
        --out "results/gc1_probe/batch-${sizes[$gpu]}" \
        > "results/gc1_probe/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
