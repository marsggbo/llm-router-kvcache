# llm-router-kvcache — Project Context

## Research Goal

Study KV cache management optimization in LLM routing scenarios.
Core claim: routing decisions determine KV cache structure — jointly optimizing
routing dispatch and cache management improves throughput without quality loss.

## Why This Is Novel

SGLang's `cache_aware` router (default, 2× throughput gain, Nov 2024) solves
cache-aware dispatch for **same-model multiple instances**.

Our problem: **multi-task routing** where same-type requests must be steered to
dedicated instances to create prefix clustering. SGLang's roadmap lists this as
planned but unimplemented (issue #10341).

Fundamental difference from multi-agent KV sharing (DroidSpeak, PrefillShare):
- Multi-agent: pipeline is fixed, optimize KV transfer between predetermined stages
- Routing: dispatch decision is a free variable — routing itself creates or destroys cache opportunities

## Confirmed Design Decisions

### Router
- **Oracle routing** via dataset labels (zero training, no ML model)
- `task_type` field from WildBench / MMLU subject used directly as routing key
- `TaskAwareRouter`: deterministic affinity — same task_type always → same instance
- Baseline: `RandomRouter` — random dispatch across all instances

### Models
- **Qwen2.5-7B-Instruct × 3 instances** (same model, different processes)
- All instances identical → quality controlled, results 100% attributed to cache management
- Ports: 30000 (coding), 30001 (math), 30002 (general)
- Config: `configs/task_aware_server.yaml`

### Datasets
- **WildBench** (`allenai/WildBench`): has `primary_tag` field (coding/math/reasoning/creative)
- **MMLU** (`cais/mmlu`): 57 subjects, grouped into stem/humanities/social/other
- Both already implemented in `benchmark/dataset_loader.py`

### Serving Backend
- **SGLang** (not vLLM): better per-request `cached_tokens` stats, hackable eviction
- `--enable-prefix-caching` flag enables RadixAttention

## Experiment Structure

### Baselines (in priority order)
| ID | Config | Description |
|----|--------|-------------|
| B1 | `random` router | Random dispatch, no cache affinity |
| B2 | `task_aware` router, LRU eviction | Our routing, standard eviction |
| B3 | SGLang built-in `cache_aware` (single model) | Industry SOTA baseline |

### Our Contributions
| ID | Description | Code location |
|----|-------------|---------------|
| C1 | Task-aware instance affinity | `router/task_aware_router.py` |
| C2 | Routing-frequency-weighted eviction | TODO: `kvcache/policies/freq_eviction.py` |
| C1+C2 | Full system | — |

### Key Metrics
- KV cache hit rate (`cached_tokens / prompt_tokens`, from SGLang response)
- TTFT p50/p95/p99
- Throughput (req/s, tok/s)
- Per-task-type breakdown

## Repository Structure

```
router/
  base.py               # BaseRouter + Instance + RoutingDecision dataclasses
  task_aware_router.py  # MAIN: oracle task-type routing + Phase 3 hook
  simple_routers.py     # Random / AlwaysWeak / AlwaysStrong (baselines)
  routellm_router.py    # RouteLLM BERT adapter (not used in main experiments)

benchmark/
  dataset_loader.py     # MMLU + WildBench + ShareGPT loaders
  run_benchmark.py      # Async benchmark harness (aiohttp, streaming)
  metrics.py            # RequestMetrics + BenchmarkSummary

analysis/
  collect_cache_stats.py  # Phase 2 pattern analysis (LCP, temporal, correlation)
  plot_patterns.py        # Visualization

configs/
  task_aware_server.yaml  # MAIN config: 3-instance same-model setup
  server_gpu.yaml         # Legacy 2-tier weak/strong setup
  mac_local.yaml          # Ollama dev config (Qwen2.5-0.5B + 1.5B)

scripts/
  start_sglang_3inst.sh   # Start 3 SGLang instances (ports 30000-30002)
  start_sglang.sh         # Legacy: start weak + strong instances
  start_ollama.sh         # Mac dev: pull and start Ollama models
  run_experiment.sh       # Full pipeline: benchmark + analysis + plots
```

## Experiment Results (MMLU, 300 req, Qwen3-4B × 3, GPU server)

| System | Cache Hit | Throughput | TTFT p50 |
|---|---|---|---|
| random (baseline) | 49.27% | 5.2 req/s / 737 tok/s | 2.84s |
| task_aware Phase 1 | 79.38% | 4.8 req/s ↓ | 3.30s ↑ |
| task_aware Phase 3 (load-balance) | **85.17%** | **5.1 req/s** | **3.10s** |

WildBench showed only +2pp cache improvement (no shared prefix in prompts).
MMLU shows +35.9pp because 5-shot prefix is fixed per subject.

## Current State

- [x] Phase 1 baseline + pattern analysis code
- [x] TaskAwareRouter (affinity + load-balance-aware dispatch)
- [x] CacheManager (polls /get_load for real-time queue state)
- [x] Server experiments run, core results obtained
- [ ] Add task-specific system prompts to WildBench (fix prefix sharing)
- [ ] Phase 2 full pattern analysis
- [ ] Phase 3: freq-weighted eviction (Algorithm 2)
- [ ] Results visualization and comparison plots

## Next Steps on Server

```bash
# 1. Start serving
bash scripts/start_sglang_3inst.sh

# 2. Baseline
python -m benchmark.run_benchmark \
  --config configs/task_aware_server.yaml \
  --dataset wildbench --router random

# 3. Our method (Phase 1)
python -m benchmark.run_benchmark \
  --config configs/task_aware_server.yaml \
  --dataset wildbench --router task_aware

# 4. Pattern analysis
python -m analysis.collect_cache_stats --dataset wildbench --router all

# 5. Plot
python -m analysis.plot_patterns --dataset wildbench --router all
```

## Phase 3 Implementation Plan (freq-weighted eviction)

To be implemented in `kvcache/policies/freq_eviction.py`:

```
eviction_score(prefix P) = time_since_last_use(P) / routing_freq(task_type(P), window=60s)
```

- Router exposes `routing_frequencies()` (already in TaskAwareRouter)
- SGLang eviction hook reads these stats
- High-frequency task prefix → harder to evict → preserved across low-traffic gaps

## Paper Positioning

- Prior work: SGLang cache_aware (same-model multi-instance load balancing)
- Our work: extends to multi-task routing where task type determines cache structure
- Key novelty: routing decision co-optimized with cache management
- Baseline comparison: random dispatch is the natural null hypothesis

## Environment

- Mac dev: Ollama + Qwen2.5-0.5B/1.5B, configs/mac_local.yaml
- GPU server: SGLang + Qwen2.5-7B-Instruct × 3, configs/task_aware_server.yaml
- Python 3.12, dependencies in requirements.txt
