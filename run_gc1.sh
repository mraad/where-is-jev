#!/usr/bin/env bash
# Run inside tmux on gc1; each process sees one physical GPU as cuda:0.
# Usage: bash run_gc1.sh [BATCH_SIZE] [RUN_DIR] [SCREEN_DIR] [PROMPT]
set -euo pipefail
cd -- "$(dirname -- "$0")"
batch_size="${1:-8}"
run_dir="${2:-results/gc1_gpu}"
screen="${3:-results/screen}"
prompt="${4:-circular storage tank}"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "Batch size must be positive"; exit 2; }
[[ ! -e "$run_dir" ]] || { echo "Run directory already exists: $run_dir"; exit 2; }
mkdir -p "$run_dir"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
started=$(date +%s)
trap 'jobs -pr | xargs -r kill' INT TERM
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "$run_dir/gpus.csv"
.venv/bin/python -m pip freeze > "$run_dir/environment.txt" 2>/dev/null || \
  /home/ubuntu/.local/bin/uv pip freeze --python .venv/bin/python > "$run_dir/environment.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 \
  > "$run_dir/utilization.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
pids=()
for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u detect.py \
        --screen "$screen" --all-valid --model checkpoints/sam3_hub \
        --prompt "$prompt" --threshold 0.3 \
        --num-shards 4 --shard-index "$gpu" --batch-size "$batch_size" --dtype bfloat16 \
        --out "$run_dir/shard-$gpu" > "$run_dir/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if [[ "$failed" != 0 ]]; then
    echo "At least one GPU worker failed; inspect gpu-*.log" | tee "$run_dir/FAILED"
    exit 1
fi
.venv/bin/python merge_shards.py --shards "$run_dir" --screen "$screen" \
    --out "$run_dir/merged" | tee "$run_dir/merge.log"
finished=$(date +%s)
printf '{"wall_seconds":%s,"started_unix":%s,"finished_unix":%s}\n' \
    "$((finished-started))" "$started" "$finished" > "$run_dir/timing.json"
touch "$run_dir/COMPLETE"
cat "$run_dir/timing.json"
