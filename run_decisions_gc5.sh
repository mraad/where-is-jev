#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
out=results/nemotron50_gpu2
[[ ! -e "$out" ]] || { echo "Output already exists: $out"; exit 2; }
mkdir -p "$out"
/home/ubuntu/.local/bin/uv pip freeze --python .venv-nemotron/bin/python > "$out/environment.txt"
started=$(date +%s)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > "$out/utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
pids=()
for gpu in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" .venv-nemotron/bin/python -u decide_nemotron.py \
        --manifest results/patches50 --image data/20150520_075001.tif \
        --model checkpoints/nemotron_vl_12b --out "$out/shard-$gpu" \
        --shard-index "$gpu" --num-shards 2 --batch-size 16 \
        > "$out/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]] || { touch "$out/FAILED"; exit 1; }
finished=$(date +%s)
printf '{"gpus":2,"wall_seconds":%s,"started_unix":%s,"finished_unix":%s}\n' \
    "$((finished-started))" "$started" "$finished" > "$out/timing.json"
touch "$out/COMPLETE"
cat "$out/timing.json"
