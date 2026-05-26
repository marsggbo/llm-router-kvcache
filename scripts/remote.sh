#!/usr/bin/env bash
# Run a command on the GPU server inside the vllm conda environment.
#
# Usage:
#   bash scripts/remote.sh "python -m benchmark.run_benchmark --dataset wildbench --router random"
#
# Environment variables:
#   SERVER   default: xinmatrix@10.9.240.19
#   PROJ_DIR default: /home/xinmatrix/hexin/llm-router-kvcache

SERVER=${SERVER:-"xinmatrix@10.9.240.19"}
PROJ_DIR=${PROJ_DIR:-"/home/xinmatrix/hexin/llm-router-kvcache"}
CMD="$*"

ssh "$SERVER" bash -c "'
    set -e
    source /home/xinmatrix/miniconda3/etc/profile.d/conda.sh
    conda activate vllm
    cd $PROJ_DIR
    $CMD
'" 2>&1
