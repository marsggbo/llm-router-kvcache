#!/usr/bin/env bash
# Start Ollama and pull the two Qwen models for local development on MacBook.
# Model sizes: qwen2.5:0.5b (~400MB), qwen2.5:1.5b (~1GB)
# For more RAM (16GB+):  qwen2.5:3b / qwen2.5:7b-q4
set -euo pipefail

WEAK_MODEL=${WEAK_MODEL:-"qwen2.5:0.5b"}
STRONG_MODEL=${STRONG_MODEL:-"qwen2.5:1.5b"}

if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Install with: brew install ollama"
    exit 1
fi

# Start Ollama server in the background (no-op if already running)
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
else
    echo "Ollama already running."
fi

echo "Pulling $WEAK_MODEL ..."
ollama pull "$WEAK_MODEL"

echo "Pulling $STRONG_MODEL ..."
ollama pull "$STRONG_MODEL"

echo ""
echo "Ready. Test with:"
echo "  curl http://localhost:11434/v1/models"
echo ""
echo "Run benchmark:"
echo "  python -m benchmark.run_benchmark --config configs/mac_local.yaml --dataset mmlu --router routellm"
