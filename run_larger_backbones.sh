#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
case "${1:-}" in
    qwen32)
        python=.venv/bin/python
        runner=decide_qwen.py
        model=checkpoints/qwen3_vl_32b
        image=/home/ubuntu/Tanks/20150520_075001.tif
        count=4
        # cuDNN 9.20's convolution plugin requests cuBLAS 12 even with this CUDA 13 wheel.
        blas_dir=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cublas/lib")')
        [[ -f "$blas_dir/libcublasLt.so.12" ]] || { echo "Install nvidia-cublas-cu12"; exit 2; }
        export LD_LIBRARY_PATH="$blas_dir:${LD_LIBRARY_PATH:-}"
        ;;
    omni30)
        python=.venv-nemotron/bin/python
        runner=decide_nemotron.py
        model=checkpoints/nemotron_omni_30b
        image=data/20150520_075001.tif
        count=2
        ;;
    *) echo "Usage: bash run_larger_backbones.sh qwen32|omni30"; exit 2 ;;
esac
out="results/${1}_gpu${count}"
[[ ! -e "$out" ]] || { echo "Output already exists: $out"; exit 2; }
[[ -f "$model/download_timing.json" ]] || { echo "Download is incomplete"; exit 2; }
mkdir -p "$out"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
/home/ubuntu/.local/bin/uv pip freeze --python "$python" > "$out/environment.txt"
printf '%s  %s\n' ffdb14f17ceec2e0e3bd2f754e20b6f9a024e3dd8abdb4b95b3015ad0debab69 \
    "$image" | sha256sum -c -
started=$(date +%s)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > "$out/utilization.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
pids=()
for ((gpu=0; gpu<count; gpu++)); do
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u "$runner" \
        --manifest results/patches50 --image "$image" --model "$model" \
        --out "$out/shard-$gpu" --shard-index "$gpu" --num-shards "$count" \
        --batch-size 8 > "$out/gpu-$gpu.log" 2>&1 &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
kill "$monitor" 2>/dev/null || true
trap - EXIT
[[ "$failed" == 0 ]] || { touch "$out/FAILED"; exit 1; }
finished=$(date +%s)
printf '{"gpus":%s,"batch_size":8,"wall_seconds":%s,"started_unix":%s,"finished_unix":%s}\n' \
    "$count" "$((finished-started))" "$started" "$finished" > "$out/timing.json"
touch "$out/COMPLETE"
cat "$out/timing.json"
