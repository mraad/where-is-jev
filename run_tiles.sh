#!/usr/bin/env bash
# Run inside tmux: each backbone x tile size, one model replica per GPU on every GPU of this host.
# gc1 (4 GPUs): bash run_tiles.sh qwen4b qwen32b; gc5 (2 GPUs, .venv-nemotron): bash run_tiles.sh nemotron12b nemotron30b
set -euo pipefail
cd -- "$(dirname -- "$0")"
image=data/Tanks.tif
root=results/tiles
sizes=(256 384 512)
models=("$@")
(( ${#models[@]} )) || { echo "Usage: bash run_tiles.sh MODEL..."; exit 2; }
gpus=$(nvidia-smi -L | wc -l)
printf '%s  %s\n' 1351163c77327ce2df2a08226f58d6c3e61fb7ede2e99e7d06bf1a36124700d9 "$image" | sha256sum -c -
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
for model in "${models[@]}"; do
    case "$model" in
        qwen4b) python=.venv/bin/python runner=decide_qwen.py ckpt=checkpoints/qwen3_vl_4b batch=16 ;;
        qwen32b) python=.venv/bin/python runner=decide_qwen.py ckpt=checkpoints/qwen3_vl_32b batch=8 ;;
        nemotron12b) python=.venv-nemotron/bin/python runner=decide_nemotron.py ckpt=checkpoints/nemotron_vl_12b batch=16 ;;
        nemotron30b) python=.venv-nemotron/bin/python runner=decide_nemotron.py ckpt=checkpoints/nemotron_omni_30b batch=8 ;;
        *) echo "Unknown model: $model"; exit 2 ;;
    esac
    "$python" -c 'import json, pathlib, sys; d = pathlib.Path(sys.argv[1]); weights = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"].values(); sys.exit(any(not (d / f).is_file() for f in set(weights)))' "$ckpt" \
        || { echo "Checkpoint is incomplete: $ckpt"; exit 2; }
    library_path=${LD_LIBRARY_PATH:-}
    if [[ "$model" == qwen32b ]]; then
        # cuDNN 9.20's convolution plugin requests cuBLAS 12 even with this CUDA 13 wheel.
        blas_dir=$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cublas/lib")')
        [[ -f "$blas_dir/libcublasLt.so.12" ]] || { echo "Install nvidia-cublas-cu12"; exit 2; }
        library_path="$blas_dir:$library_path"
    fi
    for size in "${sizes[@]}"; do
        # Copy one host's manifests to the other so every worker shares the same manifest hash.
        [[ -d "$root/$size/manifest" ]] || \
            "$python" patch_decisions.py prepare "$image" --size "$size" --out "$root/$size/manifest"
        out="$root/$size/$model"
        [[ ! -e "$out" ]] || { echo "Output already exists: $out"; exit 2; }
        mkdir -p "$out"
        /home/ubuntu/.local/bin/uv pip freeze --python "$python" > "$out/environment.txt"
        started=$EPOCHREALTIME
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 1 > "$out/utilization.csv" &
        monitor=$!
        trap 'kill "$monitor" 2>/dev/null || true' EXIT
        pids=()
        for ((gpu=0; gpu<gpus; gpu++)); do
            LD_LIBRARY_PATH="$library_path" CUDA_VISIBLE_DEVICES="$gpu" "$python" -u "$runner" \
                --manifest "$root/$size/manifest" --image "$image" --model "$ckpt" \
                --out "$out/shard-$gpu" --shard-index "$gpu" --num-shards "$gpus" \
                --batch-size "$batch" > "$out/gpu-$gpu.log" 2>&1 &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        kill "$monitor" 2>/dev/null || true
        trap - EXIT
        [[ "$failed" == 0 ]] || { touch "$out/FAILED"; echo "Failed: $out"; exit 1; }
        finished=$EPOCHREALTIME
        printf '{"model":"%s","tile_size":%s,"gpus":%s,"batch_size":%s,"wall_seconds":%.2f,"started_unix":%s,"finished_unix":%s}\n' \
            "$model" "$size" "$gpus" "$batch" "$(awk "BEGIN{print $finished-$started}")" "$started" "$finished" > "$out/timing.json"
        touch "$out/COMPLETE"
        cat "$out/timing.json"
    done
done
echo ALL COMPLETE
