#!/usr/bin/env bash
# Start two SGLang instances: weak model (port 30000) and strong model (port 30001).
# Adjust GPU IDs and model paths as needed.
set -euo pipefail

WEAK_MODEL=${WEAK_MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
STRONG_MODEL=${STRONG_MODEL:-"Qwen/Qwen2.5-14B-Instruct"}

echo "Starting weak model ($WEAK_MODEL) on port 30000..."
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$WEAK_MODEL" \
    --port 30000 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --log-requests \
    &

echo "Starting strong model ($STRONG_MODEL) on port 30001..."
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
    --model-path "$STRONG_MODEL" \
    --port 30001 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --log-requests \
    &

echo "Waiting for servers to be ready..."
for port in 30000 30001; do
    until curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo "  Port $port ready."
done

echo "Both servers running. PID file: /tmp/sglang_pids"
echo "$!" > /tmp/sglang_pids
