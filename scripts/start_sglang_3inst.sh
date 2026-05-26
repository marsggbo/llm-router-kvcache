#!/usr/bin/env bash
# Start 3 SGLang instances of the same model on consecutive ports (30000-30002).
# Each instance is dedicated to one task type for task-aware routing experiments.
set -euo pipefail

MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
BASE_PORT=${BASE_PORT:-30000}
GPUS=(${GPUS:-"0 1 2"})   # space-separated GPU IDs

NAMES=("coding" "math" "general")

for i in 0 1 2; do
    PORT=$((BASE_PORT + i))
    GPU=${GPUS[$i]}
    NAME=${NAMES[$i]}
    echo "Starting instance '$NAME' on port $PORT (GPU $GPU)..."
    CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$PORT" \
        --host 0.0.0.0 \
        --enable-prefix-caching \
        --log-requests \
        &
done

echo "Waiting for all 3 instances to be ready..."
for PORT in 30000 30001 30002; do
    until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  Port $PORT ready."
done
echo "All 3 instances running."
