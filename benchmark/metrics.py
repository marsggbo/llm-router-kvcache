"""Per-request and aggregate metrics for benchmark runs."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class RequestMetrics:
    request_id: int
    prompt: str          # full prompt text — needed for prefix overlap analysis
    task_type: str
    model_tier: str      # weak | strong
    model_name: str
    router_score: float

    # Timing (seconds)
    send_time: float = 0.0
    ttft: Optional[float] = None        # time to first token
    total_latency: Optional[float] = None

    # Token counts (from SGLang response meta_info)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0              # tokens served from KV cache

    # Derived
    error: Optional[str] = None

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    @property
    def throughput_tokens(self) -> float:
        if not self.total_latency or self.total_latency == 0:
            return 0.0
        return self.completion_tokens / self.total_latency

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cache_hit_rate"] = self.cache_hit_rate
        d["throughput_tokens"] = self.throughput_tokens
        return d


@dataclass
class BenchmarkSummary:
    router_type: str
    dataset: str
    num_requests: int
    num_errors: int

    # Throughput
    total_time_sec: float = 0.0
    requests_per_sec: float = 0.0
    tokens_per_sec: float = 0.0

    # Latency (seconds)
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0
    latency_p50: float = 0.0
    latency_p99: float = 0.0

    # KV cache
    mean_cache_hit_rate: float = 0.0
    mean_cached_tokens: float = 0.0
    mean_prompt_tokens: float = 0.0

    # Routing distribution
    weak_ratio: float = 0.0
    strong_ratio: float = 0.0

    # Per-task-type breakdown (filled in post-hoc)
    per_task_type: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_summary(results: list[RequestMetrics], router_type: str,
                    dataset: str, total_time: float) -> BenchmarkSummary:
    ok = [r for r in results if r.error is None]
    if not ok:
        return BenchmarkSummary(router_type=router_type, dataset=dataset,
                                num_requests=len(results), num_errors=len(results))

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    latencies = [r.total_latency for r in ok if r.total_latency is not None]
    total_tokens = sum(r.completion_tokens for r in ok)

    weak_count = sum(1 for r in ok if r.model_tier == "weak")

    summary = BenchmarkSummary(
        router_type=router_type,
        dataset=dataset,
        num_requests=len(results),
        num_errors=len(results) - len(ok),
        total_time_sec=total_time,
        requests_per_sec=len(ok) / total_time if total_time > 0 else 0,
        tokens_per_sec=total_tokens / total_time if total_time > 0 else 0,
        ttft_p50=float(np.percentile(ttfts, 50)) if ttfts else 0,
        ttft_p95=float(np.percentile(ttfts, 95)) if ttfts else 0,
        ttft_p99=float(np.percentile(ttfts, 99)) if ttfts else 0,
        latency_p50=float(np.percentile(latencies, 50)) if latencies else 0,
        latency_p99=float(np.percentile(latencies, 99)) if latencies else 0,
        mean_cache_hit_rate=float(np.mean([r.cache_hit_rate for r in ok])),
        mean_cached_tokens=float(np.mean([r.cached_tokens for r in ok])),
        mean_prompt_tokens=float(np.mean([r.prompt_tokens for r in ok])),
        weak_ratio=weak_count / len(ok),
        strong_ratio=1 - weak_count / len(ok),
    )

    # Per task-type breakdown
    task_types = set(r.task_type for r in ok)
    for tt in task_types:
        tt_results = [r for r in ok if r.task_type == tt]
        tt_ttfts = [r.ttft for r in tt_results if r.ttft is not None]
        summary.per_task_type[tt] = {
            "count": len(tt_results),
            "weak_ratio": sum(1 for r in tt_results if r.model_tier == "weak") / len(tt_results),
            "mean_cache_hit_rate": float(np.mean([r.cache_hit_rate for r in tt_results])),
            "ttft_p50": float(np.percentile(tt_ttfts, 50)) if tt_ttfts else 0,
            "mean_prompt_tokens": float(np.mean([r.prompt_tokens for r in tt_results])),
        }

    return summary
