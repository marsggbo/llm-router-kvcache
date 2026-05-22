"""
Visualize KV cache patterns discovered in Phase 2.

Usage:
  python -m analysis.plot_patterns --analysis-dir results/analysis --dataset mmlu
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_cache_hit_by_task(analysis: dict, out_path: Path):
    task_stats = analysis["cache_by_task_type"]
    tasks = sorted(task_stats, key=lambda t: task_stats[t]["mean_cache_hit_rate"], reverse=True)
    hit_rates = [task_stats[t]["mean_cache_hit_rate"] for t in tasks]
    counts = [task_stats[t]["count"] for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(tasks, hit_rates, color="steelblue")
    ax.set_xlabel("Mean KV Cache Hit Rate")
    ax.set_title(f"KV Cache Hit Rate by Task Type\n({analysis['router']} router, {analysis['dataset']})")
    ax.set_xlim(0, 1)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={count}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_temporal_pattern(analysis: dict, out_path: Path):
    windows = analysis["temporal_pattern"]["windows"]
    if not windows:
        return
    x = [w["window_start"] for w in windows]
    y = [w["mean_cache_hit_rate"] for w in windows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xlabel("Request Index (window start)")
    ax.set_ylabel("Mean Cache Hit Rate")
    ax.set_title(f"Cache Hit Rate Over Time\n({analysis['router']} router, {analysis['dataset']})")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_router_comparison(analyses: list[dict], out_path: Path):
    """Compare cache hit rates across different router types."""
    routers = [a["router"] for a in analyses]
    hit_rates = [
        np.mean([s["mean_cache_hit_rate"] for s in a["cache_by_task_type"].values()])
        for a in analyses
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(routers, hit_rates, color=colors[:len(routers)])
    ax.set_ylabel("Mean KV Cache Hit Rate")
    ax.set_title(f"Cache Hit Rate by Router Type ({analyses[0]['dataset']})")
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, hit_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.2%}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="results/analysis")
    parser.add_argument("--dataset", choices=["mmlu", "sharegpt", "wildbench"], required=True)
    parser.add_argument("--router", default="all")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = analysis_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    routers_to_plot = (
        ["always_weak", "always_strong", "random", "routellm"]
        if args.router == "all"
        else [args.router]
    )

    analyses = []
    for router in routers_to_plot:
        path = analysis_dir / f"{args.dataset}_{router}_analysis.json"
        if not path.exists():
            print(f"  Skipping {router}: no analysis file found")
            continue
        a = json.loads(path.read_text())
        analyses.append(a)

        plot_cache_hit_by_task(a, out_dir / f"{args.dataset}_{router}_cache_by_task.png")
        plot_temporal_pattern(a, out_dir / f"{args.dataset}_{router}_temporal.png")

    if len(analyses) > 1:
        plot_router_comparison(analyses, out_dir / f"{args.dataset}_router_comparison.png")


if __name__ == "__main__":
    main()
