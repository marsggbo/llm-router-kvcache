#!/usr/bin/env bash
# Start SGLang's built-in cache_aware router as a proxy in front of
# the 3 worker instances (ports 30000-30002).
# The router listens on port 9000 and uses approximate radix-tree
# prefix matching to dispatch requests to the worker with the best
# KV cache affinity.
#
# Prerequisites: 3 SGLang workers already running (start_sglang_3inst.sh)

ROUTER_PORT=${ROUTER_PORT:-9000}
WORKER_URLS="http://localhost:30000 http://localhost:30001 http://localhost:30002"

echo "Starting SGLang cache_aware router on port $ROUTER_PORT ..."
echo "Workers: $WORKER_URLS"

python -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port "$ROUTER_PORT" \
    --worker-urls $WORKER_URLS \
    --policy cache_aware \
    --cache-threshold 0.3 \
    --balance-abs-threshold 32 \
    --balance-rel-threshold 1.5 \
    --eviction-interval-secs 60
