#!/usr/bin/env bash
# Run full Phase 1 + Phase 2 experiment pipeline.
set -euo pipefail

DATASET=${1:-"mmlu"}
CONFIG=${2:-"configs/default.yaml"}
RESULTS_DIR="results"

echo "===== Phase 1: Baseline benchmarks ====="
for ROUTER in always_weak always_strong random routellm; do
    echo "--- Router: $ROUTER ---"
    python -m benchmark.run_benchmark \
        --config "$CONFIG" \
        --dataset "$DATASET" \
        --router "$ROUTER" \
        --output-dir "$RESULTS_DIR"
done

echo ""
echo "===== Phase 2: Pattern analysis ====="
python -m analysis.collect_cache_stats \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR/analysis" \
    --dataset "$DATASET" \
    --router all

echo ""
echo "===== Plotting ====="
python -m analysis.plot_patterns \
    --analysis-dir "$RESULTS_DIR/analysis" \
    --dataset "$DATASET" \
    --router all

echo ""
echo "Done. Results in $RESULTS_DIR/"
