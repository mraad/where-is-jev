#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
root=results/qwen50_benchmark
[[ ! -e "$root" ]] || { echo "Output already exists: $root"; exit 2; }
mkdir -p "$root"
/home/ubuntu/.local/bin/uv pip freeze --python .venv/bin/python > "$root/environment.txt"
printf '%s  %s\n' ffdb14f17ceec2e0e3bd2f754e20b6f9a024e3dd8abdb4b95b3015ad0debab69 \
    /home/ubuntu/Tanks/20150520_075001.tif | sha256sum -c -
for count in 1 4; do
    out="$root/gpu-$count"
    mkdir "$out"
    started=$(date +%s)
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > "$out/utilization.csv" &
    monitor=$!
    trap 'kill "$monitor" 2>/dev/null || true' EXIT
    pids=()
    for ((gpu=0; gpu<count; gpu++)); do
        CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python -u decide_qwen.py \
            --manifest results/patches50 --image /home/ubuntu/Tanks/20150520_075001.tif \
            --model checkpoints/qwen3_vl_4b --out "$out/shard-$gpu" \
            --shard-index "$gpu" --num-shards "$count" --batch-size 16 \
            > "$out/gpu-$gpu.log" 2>&1 &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    kill "$monitor" 2>/dev/null || true
    trap - EXIT
    [[ "$failed" == 0 ]] || { touch "$out/FAILED"; exit 1; }
    finished=$(date +%s)
    printf '{"gpus":%s,"wall_seconds":%s,"started_unix":%s,"finished_unix":%s}\n' \
        "$count" "$((finished-started))" "$started" "$finished" > "$out/timing.json"
    touch "$out/COMPLETE"
    cat "$out/timing.json"
done
touch "$root/COMPLETE"
