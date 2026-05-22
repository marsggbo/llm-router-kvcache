# llm-router-kvcache

KV cache optimization in LLM routing scenarios.

## Overview

This project investigates how routing decisions affect KV cache usage patterns in multi-model LLM serving,
and designs optimizations that exploit task-aware prefix structure to improve throughput.

**Research questions:**
1. What KV cache usage patterns emerge when a router dispatches requests across model tiers?
2. Does task-aware routing create exploitable prefix clustering?
3. Can routing-informed cache management significantly improve throughput without quality loss?

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

The config and backend are selected automatically by OS, or you can pass `--config` explicitly.

### MacBook (Ollama, Apple Silicon)

```bash
# Install Ollama and pull small Qwen models (~1.5 GB total)
bash scripts/start_ollama.sh

# Run benchmark — auto-uses configs/mac_local.yaml (Qwen2.5-0.5B + 1.5B)
python -m benchmark.run_benchmark --dataset mmlu --router routellm
```

> On Mac, `cached_tokens` is not reported by Ollama.
> Use TTFT-over-time as a proxy for cache warm-up (see `analysis/collect_cache_stats.py`).

### GPU Server (SGLang, CUDA)

```bash
# Start two SGLang instances (weak on :30000, strong on :30001)
bash scripts/start_sglang.sh

# Full experiment: all router types + pattern analysis
bash scripts/run_experiment.sh mmlu
```

Or run steps individually:
```bash
python -m benchmark.run_benchmark --config configs/server_gpu.yaml --dataset mmlu --router routellm
python -m analysis.collect_cache_stats --dataset mmlu --router all
python -m analysis.plot_patterns --dataset mmlu --router all
```

## Project Structure

```
├── configs/default.yaml          # Model URLs, router settings, benchmark params
├── router/
│   ├── base.py                   # BaseRouter + RoutingDecision dataclass
│   ├── simple_routers.py         # Random / AlwaysWeak / AlwaysStrong
│   └── routellm_router.py        # RouteLLM integration
├── benchmark/
│   ├── dataset_loader.py         # MMLU / ShareGPT / WildBench loaders
│   ├── metrics.py                # Per-request and aggregate metrics
│   └── run_benchmark.py          # Async benchmark harness
├── analysis/
│   ├── collect_cache_stats.py    # Phase 2 pattern analysis
│   └── plot_patterns.py          # Visualization
└── scripts/
    ├── start_sglang.sh
    └── run_experiment.sh
```

## Experimental Design

### Phase 1 — Baseline Measurement

| Baseline | Description |
|---|---|
| `always_weak` | All requests → small model (Qwen-7B). Throughput upper bound. |
| `always_strong` | All requests → large model (Qwen-14B). Quality upper bound. |
| `random` | Random routing, no cache awareness. |
| `routellm` | RouteLLM-based routing, default LRU prefix cache. |

### Phase 2 — Pattern Analysis

Collect per-request cache hit rates, prefix overlap, and temporal patterns to answer:
- Does routing scatter prefixes (reducing hit rate)?
- Do task types cluster in ways that enable prefix reuse?
- What does the cache eviction pattern look like over time?

### Phase 3 — Optimization (TBD based on Phase 2 findings)

Candidate optimizations based on observed patterns:
- Task-aware instance affinity
- Routing-informed cache eviction priority
- Prefix pre-warming for high-frequency task types

## Datasets

| Dataset | Task types | KV cache relevance |
|---|---|---|
| MMLU | 57 subject areas | Fixed few-shot prefix per subject → high reuse potential |
| WildBench | coding / math / reasoning / creative | Diverse task types, real-user queries |
| ShareGPT | Multi-turn conversations | Long prefix from conversation history |

## Models

Default: Qwen2.5-7B-Instruct (weak) + Qwen2.5-14B-Instruct (strong).
Configure in `configs/default.yaml`.
