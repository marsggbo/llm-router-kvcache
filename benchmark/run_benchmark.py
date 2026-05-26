"""
Main benchmark script.

Usage:
  # Baseline: routing with no prefix cache
  python -m benchmark.run_benchmark --dataset mmlu --router random --no-prefix-cache

  # Baseline: routing with default SGLang prefix cache
  python -m benchmark.run_benchmark --dataset mmlu --router routellm

  # All router types on MMLU
  python -m benchmark.run_benchmark --dataset mmlu --router all
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import yaml

from router import build_router
from router.base import RoutingDecision
from benchmark.dataset_loader import load_dataset_by_name, Request
from benchmark.metrics import RequestMetrics, compute_summary
from kvcache.cache_manager import CacheManager


async def send_request(
    session: aiohttp.ClientSession,
    router,                   # BaseRouter — routing happens here, not pre-computed
    req: Request,
    req_id: int,
    max_new_tokens: int,
    semaphore: asyncio.Semaphore,
    cache_manager=None,       # Phase 3: optional cache state provider
) -> RequestMetrics:
    # Route at dispatch time with live cache state (Phase 3).
    cache_state = cache_manager.get_state() if cache_manager else None
    decision = router.route(req.prompt, req.task_type, cache_state=cache_state)

    metrics = RequestMetrics(
        request_id=req_id,
        prompt=req.prompt,          # full prompt — required for LCP analysis
        task_type=req.task_type,
        model_tier=decision.instance_name,
        model_name=decision.model_name,
        router_score=decision.score,
        send_time=time.time(),
    )

    # Use non-streaming to get accurate token counts including cached_tokens.
    # SGLang 0.5.x streaming does not populate usage; non-streaming does.
    # total_latency serves as a proxy for end-to-end latency.
    payload = {
        "model": decision.model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": max_new_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with semaphore:
        try:
            t_start = time.perf_counter()

            async with session.post(
                f"{decision.url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()

            metrics.total_latency = time.perf_counter() - t_start
            # Non-streaming: no per-token timing, use total_latency as TTFT proxy
            metrics.ttft = metrics.total_latency

            usage = result.get("usage", {}) or {}
            metrics.prompt_tokens = usage.get("prompt_tokens", 0)
            metrics.completion_tokens = usage.get("completion_tokens", 0)
            metrics.cached_tokens = (
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                or usage.get("cached_tokens", 0)
            )

        except Exception as e:
            metrics.error = str(e)

    return metrics


async def run_benchmark_async(cfg: dict, dataset_name: str, router_type: str,
                               output_dir: Path) -> None:
    cfg["router"]["type"] = router_type
    router = build_router(cfg)

    requests = load_dataset_by_name(dataset_name, cfg)
    num_requests = cfg["benchmark"]["num_requests"]
    requests = requests[:num_requests]

    concurrency = cfg["benchmark"]["concurrency"]
    max_new_tokens = cfg["benchmark"]["max_new_tokens"]
    semaphore = asyncio.Semaphore(concurrency)

    print(f"\n{'='*60}")
    print(f"Router: {router_type} | Dataset: {dataset_name} | N={len(requests)}")
    print(f"{'='*60}")

    results: list[RequestMetrics] = []
    t_wall_start = time.perf_counter()

    # Poll SGLang server info for global cache hit rate (usage per-request not available in 0.5.x)
    server_urls = list({inst.url for inst in router.instances})

    async def get_global_cache_hit_rate(session: aiohttp.ClientSession) -> float:
        rates = []
        for url in server_urls:
            try:
                async with session.get(f"{url}/get_server_info", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        info = await r.json()
                        rate = info.get("cache_hit_rate", info.get("radix_cache_hit_rate", 0))
                        if rate is not None:
                            rates.append(float(rate))
            except Exception:
                pass
        return sum(rates) / len(rates) if rates else 0.0

    async with aiohttp.ClientSession() as session:
        # Start CacheManager for load-balance-aware routing (task_aware router uses this).
        # Other routers (random, always_*) ignore cache_state entirely.
        cache_manager = CacheManager(router.instances, poll_interval=1.0)
        await cache_manager.start(session)

        try:
            tasks = [
                send_request(session, router, req, i, max_new_tokens, semaphore,
                             cache_manager=cache_manager)
                for i, req in enumerate(requests)
            ]

            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                if len(results) % 50 == 0:
                    done = len(results)
                    errors = sum(1 for r in results if r.error)
                    global_hit = await get_global_cache_hit_rate(session)
                    # Also show per-instance queue length
                    state = cache_manager.get_state()
                    loads = " ".join(f"{n}:{s.queue_length}" for n, s in state.items())
                    print(f"  [{done}/{len(requests)}] errors={errors} "
                          f"cache_hit={global_hit:.2%} loads=[{loads}]")
        finally:
            await cache_manager.stop()

    total_time = time.perf_counter() - t_wall_start
    summary = compute_summary(results, router_type, dataset_name, total_time)

    print(f"\nResults:")
    print(f"  Throughput:        {summary.requests_per_sec:.1f} req/s | {summary.tokens_per_sec:.0f} tok/s")
    print(f"  TTFT p50/p95/p99:  {summary.ttft_p50:.3f}s / {summary.ttft_p95:.3f}s / {summary.ttft_p99:.3f}s")
    print(f"  Cache hit rate:    {summary.mean_cache_hit_rate:.2%}")
    print(f"  Routing:           weak={summary.weak_ratio:.1%} strong={summary.strong_ratio:.1%}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dataset_name}_{router_type}"
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2)
    )
    (output_dir / f"{stem}_requests.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for r in results)
    )
    print(f"\nSaved to {output_dir}/{stem}_*.json")


def main():
    parser = argparse.ArgumentParser()
    import platform
    default_cfg = (
        "configs/mac_local.yaml"
        if platform.system() == "Darwin"
        else "configs/server_gpu.yaml"
    )
    parser.add_argument("--config", default=default_cfg)
    parser.add_argument("--dataset", choices=["mmlu", "sharegpt", "wildbench"], required=True)
    parser.add_argument("--router", default="routellm",
                        help="Router type or 'all' to run all variants")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.num_requests:
        cfg["benchmark"]["num_requests"] = args.num_requests
    if args.concurrency:
        cfg["benchmark"]["concurrency"] = args.concurrency

    output_dir = Path(args.output_dir)

    routers_to_run = (
        ["always_weak", "always_strong", "random", "routellm"]
        if args.router == "all"
        else [args.router]
    )

    for router_type in routers_to_run:
        asyncio.run(run_benchmark_async(cfg, args.dataset, router_type, output_dir))


if __name__ == "__main__":
    main()
