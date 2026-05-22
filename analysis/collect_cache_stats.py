"""
Phase 2: Collect and analyze KV cache usage patterns in the routing scenario.

Run after benchmark to identify patterns from request-level logs.

Usage:
  python -m analysis.collect_cache_stats --results-dir results/ --dataset mmlu
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_requests(results_dir: Path, dataset: str, router: str) -> list[dict]:
    path = results_dir / f"{dataset}_{router}_requests.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run benchmark first.")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze_prefix_overlap(requests: list[dict]) -> dict:
    """
    Measure token-level prefix overlap between requests routed to the same model.
    Uses first 64 tokens as a proxy for shared prefix.
    """
    by_tier = defaultdict(list)
    for r in requests:
        by_tier[r["model_tier"]].append(r["prompt"])

    results = {}
    for tier, prompts in by_tier.items():
        if len(prompts) < 2:
            continue
        # Sample 200 random pairs
        rng = np.random.default_rng(42)
        indices = rng.choice(len(prompts), size=(min(200, len(prompts) // 2), 2), replace=False)
        overlaps = []
        for i, j in indices:
            a, b = prompts[i].split(), prompts[j].split()
            # Longest common prefix length
            lcp = sum(1 for x, y in zip(a, b) if x == y)
            overlap = lcp / max(len(a), len(b), 1)
            overlaps.append(overlap)
        results[tier] = {
            "mean_overlap": float(np.mean(overlaps)),
            "p50_overlap": float(np.percentile(overlaps, 50)),
            "p90_overlap": float(np.percentile(overlaps, 90)),
        }
    return results


def analyze_cache_by_task_type(requests: list[dict]) -> dict:
    """
    Compare KV cache hit rates across different task types.
    Key question: do task-type-aware routing decisions create higher hit rates?
    """
    by_task = defaultdict(list)
    for r in requests:
        if r.get("error"):
            continue
        by_task[r["task_type"]].append(r)

    results = {}
    for task_type, reqs in by_task.items():
        hit_rates = [r["cache_hit_rate"] for r in reqs]
        cached = [r["cached_tokens"] for r in reqs]
        prompt_lens = [r["prompt_tokens"] for r in reqs]
        results[task_type] = {
            "count": len(reqs),
            "mean_cache_hit_rate": float(np.mean(hit_rates)),
            "p50_cache_hit_rate": float(np.percentile(hit_rates, 50)),
            "p90_cache_hit_rate": float(np.percentile(hit_rates, 90)),
            "mean_cached_tokens": float(np.mean(cached)),
            "mean_prompt_tokens": float(np.mean(prompt_lens)),
            "weak_ratio": sum(1 for r in reqs if r["model_tier"] == "weak") / len(reqs),
        }
    return results


def analyze_temporal_pattern(requests: list[dict], window_size: int = 50) -> dict:
    """
    Analyze how cache hit rate evolves over time (arrival order).
    Reveals: cold-start effect, steady-state hit rate, eviction impact.
    """
    sorted_reqs = sorted(requests, key=lambda r: r["send_time"])
    ok = [r for r in sorted_reqs if not r.get("error")]

    windows = []
    for i in range(0, len(ok) - window_size, window_size // 2):
        chunk = ok[i: i + window_size]
        windows.append({
            "window_start": i,
            "mean_cache_hit_rate": float(np.mean([r["cache_hit_rate"] for r in chunk])),
            "mean_cached_tokens": float(np.mean([r["cached_tokens"] for r in chunk])),
        })

    return {"window_size": window_size, "windows": windows}


def analyze_routing_cache_correlation(requests: list[dict]) -> dict:
    """
    Does routing score correlate with prompt length / cache hit rate?
    Hypothesis: complex (high-score) queries have longer prompts → more KV pressure.
    """
    ok = [r for r in requests if not r.get("error")]
    scores = [r["router_score"] for r in ok]
    prompt_lens = [r["prompt_tokens"] for r in ok]
    hit_rates = [r["cache_hit_rate"] for r in ok]

    score_len_corr = float(np.corrcoef(scores, prompt_lens)[0, 1]) if len(ok) > 1 else 0
    score_hit_corr = float(np.corrcoef(scores, hit_rates)[0, 1]) if len(ok) > 1 else 0

    return {
        "router_score_vs_prompt_length_corr": score_len_corr,
        "router_score_vs_cache_hit_rate_corr": score_hit_corr,
        "mean_prompt_len_weak": float(np.mean([r["prompt_tokens"] for r in ok if r["model_tier"] == "weak"])) if any(r["model_tier"] == "weak" for r in ok) else 0,
        "mean_prompt_len_strong": float(np.mean([r["prompt_tokens"] for r in ok if r["model_tier"] == "strong"])) if any(r["model_tier"] == "strong" for r in ok) else 0,
    }


def run_analysis(results_dir: Path, dataset: str, router: str, output_dir: Path):
    print(f"\nAnalyzing: {dataset} / {router}")
    requests = load_requests(results_dir, dataset, router)
    ok = [r for r in requests if not r.get("error")]
    print(f"  Loaded {len(requests)} requests ({len(ok)} successful)")

    report = {
        "dataset": dataset,
        "router": router,
        "total_requests": len(requests),
        "successful": len(ok),
        "prefix_overlap": analyze_prefix_overlap(ok),
        "cache_by_task_type": analyze_cache_by_task_type(ok),
        "temporal_pattern": analyze_temporal_pattern(ok),
        "routing_cache_correlation": analyze_routing_cache_correlation(ok),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{router}_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  Saved analysis → {out_path}")

    # Print key findings
    print("\n  Key findings:")
    for task, stats in report["cache_by_task_type"].items():
        print(f"    {task}: hit_rate={stats['mean_cache_hit_rate']:.2%}  n={stats['count']}")

    corr = report["routing_cache_correlation"]
    print(f"\n  Router score vs prompt length: r={corr['router_score_vs_prompt_length_corr']:.3f}")
    print(f"  Router score vs cache hit:     r={corr['router_score_vs_cache_hit_rate_corr']:.3f}")
    print(f"  Mean prompt len weak/strong:   {corr['mean_prompt_len_weak']:.0f} / {corr['mean_prompt_len_strong']:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--dataset", choices=["mmlu", "sharegpt", "wildbench"], required=True)
    parser.add_argument("--router", default="routellm",
                        help="Router type or 'all'")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    routers = (
        ["always_weak", "always_strong", "random", "routellm"]
        if args.router == "all"
        else [args.router]
    )
    for router in routers:
        run_analysis(results_dir, args.dataset, router, output_dir)


if __name__ == "__main__":
    main()
