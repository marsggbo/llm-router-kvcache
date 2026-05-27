"""
Plot final comparison figures for the paper.

Usage:
  python -m analysis.plot_comparison --results-dir results_server/results --output-dir results_server/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROUTERS = ["random", "task_aware"]
ROUTER_LABELS = {
    "random":     "Random dispatch",
    "task_aware": "Task-aware (ours)",
}
COLORS = {
    "random":     "#5B9BD5",
    "task_aware": "#ED7D31",
}


def load_summary(results_dir: Path, dataset: str, router: str) -> dict:
    path = results_dir / f"{dataset}_{router}_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def plot_bar_group(ax, datasets, metric_key, ylabel, title, results_dir, multiplier=1.0):
    x = np.arange(len(datasets))
    width = 0.35
    for i, router in enumerate(ROUTERS):
        values = []
        for ds in datasets:
            s = load_summary(results_dir, ds, router)
            values.append(s.get(metric_key, 0) * multiplier)
        bars = ax.bar(x + i * width - width / 2, values, width,
                      label=ROUTER_LABELS[router], color=COLORS[router],
                      edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(values),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ds.upper() for ds in datasets])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_main(results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["mmlu", "wildbench"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Task-Aware Routing vs Random Dispatch\n(Qwen3-4B × 3 instances, SGLang)",
                 fontsize=12, fontweight="bold")

    plot_bar_group(axes[0], datasets, "mean_cache_hit_rate", "Cache Hit Rate",
                   "KV Cache Hit Rate", results_dir, multiplier=100)
    axes[0].set_ylabel("Cache Hit Rate (%)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plot_bar_group(axes[1], datasets, "requests_per_sec", "Requests / Second",
                   "Throughput (req/s)", results_dir)

    plot_bar_group(axes[2], datasets, "ttft_p50", "Seconds",
                   "Latency (TTFT p50)", results_dir)

    plt.tight_layout()
    out = output_dir / "main_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_cache_hit_breakdown(results_dir: Path, output_dir: Path):
    """Show cache hit rate delta between random and task_aware per dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["mmlu", "wildbench"]
    labels = ["MMLU\n(high prefix ratio)", "WildBench\n(low prefix ratio)"]

    random_hits, aware_hits = [], []
    for ds in datasets:
        r = load_summary(results_dir, ds, "random").get("mean_cache_hit_rate", 0) * 100
        t = load_summary(results_dir, ds, "task_aware").get("mean_cache_hit_rate", 0) * 100
        random_hits.append(r)
        aware_hits.append(t)

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, random_hits, width, label="Random dispatch",
                color=COLORS["random"], edgecolor="white")
    b2 = ax.bar(x + width / 2, aware_hits, width, label="Task-aware (ours)",
                color=COLORS["task_aware"], edgecolor="white")

    # Annotate improvement
    for i, (r, t) in enumerate(zip(random_hits, aware_hits)):
        delta = t - r
        ax.annotate(f"+{delta:.1f}pp",
                    xy=(x[i] + width / 2, t),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9, color="#C00000", fontweight="bold")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("KV Cache Hit Rate (%)")
    ax.set_title("Cache Hit Rate: Task-Aware vs Random Dispatch")
    ax.legend()
    ax.set_ylim(0, max(aware_hits) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "cache_hit_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_table(results_dir: Path):
    print("\n" + "="*70)
    print(f"{'Dataset':<12} {'Router':<15} {'CacheHit%':>10} {'Req/s':>8} {'Tok/s':>8} {'TTFTp50':>9}")
    print("-"*70)
    for ds in ["mmlu", "wildbench"]:
        for router in ROUTERS:
            s = load_summary(results_dir, ds, router)
            if not s:
                continue
            print(f"{ds:<12} {ROUTER_LABELS[router]:<15} "
                  f"{s.get('mean_cache_hit_rate',0)*100:>9.2f}% "
                  f"{s.get('requests_per_sec',0):>8.1f} "
                  f"{s.get('tokens_per_sec',0):>8.0f} "
                  f"{s.get('ttft_p50',0):>8.3f}s")
        print()
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results_server/results")
    parser.add_argument("--output-dir", default="results_server/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print_table(results_dir)
    plot_main(results_dir, output_dir)
    plot_cache_hit_breakdown(results_dir, output_dir)


if __name__ == "__main__":
    main()
