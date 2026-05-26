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


async def send_request(
    session: aiohttp.ClientSession,
    router,                   # BaseRouter — routing happens here, not pre-computed
    req: Request,
    req_id: int,
    max_new_tokens: int,
    semaphore: asyncio.Semaphore,
    cache_manager=None,       # Phase 3: optional cache state provider
) -> RequestMetrics:
    # Route at dispatch time so Phase 3 can pass live cache state to the router.
    cache_state = await cache_manager.get_state() if cache_manager else None
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

    payload = {
        "model": decision.model_name,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": max_new_tokens,
        "stream": True,
        # Request usage stats in final streaming chunk (supported by Ollama + SGLang)
        "stream_options": {"include_usage": True},
    }

    async with semaphore:
        try:
            t_start = time.perf_counter()
            first_token_time = None

            async with session.post(
                f"{decision.url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                completion_tokens = 0

                async for line in resp.content:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("choices"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            metrics.ttft = first_token_time - t_start
                        delta = chunk["choices"][0].get("delta", {})
                        if delta.get("content"):
                            completion_tokens += 1

                    if usage := chunk.get("usage"):
                        metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                        metrics.completion_tokens = usage.get("completion_tokens", completion_tokens)
                        # SGLang reports cached tokens; Ollama does not (stays 0).
                        # On Ollama, cache hit rate is not directly observable —
                        # use TTFT improvement over time as a proxy instead.
                        metrics.cached_tokens = (
                            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                            or usage.get("cached_tokens", 0)
                        )

            metrics.total_latency = time.perf_counter() - t_start

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

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, router, req, i, max_new_tokens, semaphore)
            for i, req in enumerate(requests)
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            if len(results) % 50 == 0:
                done = len(results)
                errors = sum(1 for r in results if r.error)
                hit_rates = [r.cache_hit_rate for r in results if r.error is None]
                avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0
                print(f"  [{done}/{len(requests)}] errors={errors} avg_cache_hit={avg_hit:.2%}")

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
