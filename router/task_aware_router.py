"""
Task-Aware Router — routes requests to dedicated instances by task type.

Routing modes (selected by what's passed to route()):

  Phase 1 — Fixed affinity (cache_state=None):
    Same task_type always hashes to the same instance.
    Creates prefix clustering → high KV cache hit rate.
    Problem: can cause load imbalance if task distribution is skewed.

  Phase 3 — Load-balance-aware affinity (cache_state provided):
    Prefers the affinity instance for cache locality.
    Falls back to the least-loaded instance when the preferred one is
    significantly more loaded (load_balance_threshold config).
    Goal: high cache hit rate WITHOUT sacrificing throughput.

    This is the paper's core contribution: jointly optimizing
    cache affinity and load balance in the multi-task routing setting.
"""

import hashlib
from collections import defaultdict

from .base import BaseRouter, Instance, RoutingDecision


class TaskAwareRouter(BaseRouter):
    """
    Routes by task_type label from dataset metadata (oracle routing).

    Config:
      router:
        type: task_aware
        load_balance_threshold: 8   # queue gap before falling back to load-aware
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._load_threshold = cfg["router"].get("load_balance_threshold", 8)
        self._routing_freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def route(self, prompt: str, task_type: str = "unknown",
              cache_state: dict | None = None) -> RoutingDecision:
        candidates = self.compatible_instances(task_type)
        inst = self._select(task_type, candidates, cache_state)
        self._routing_freq[task_type][inst.name] += 1
        return RoutingDecision(
            instance_name=inst.name,
            url=inst.url,
            model_name=inst.model_name,
            score=1.0,
            task_type=task_type,
        )

    def _select(self, task_type: str, candidates: list[Instance],
                cache_state: dict | None) -> Instance:
        # Deterministic affinity target (consistent hash on task_type)
        idx = int(hashlib.md5(task_type.encode()).hexdigest(), 16) % len(candidates)
        preferred = candidates[idx]

        if cache_state is None:
            return preferred  # Phase 1: pure affinity, no load awareness

        # Phase 3: load-balance-aware affinity.
        # All instances run the same model so any can serve any task.
        all_instances = self.instances
        preferred_load = _queue_length(cache_state, preferred.name)
        least_loaded = min(all_instances, key=lambda i: _queue_length(cache_state, i.name))
        least_load = _queue_length(cache_state, least_loaded.name)

        # Use relative ratio threshold to handle SGLang's token-based load metric.
        # Fall back when preferred is more than (1 + threshold/10)x the least-loaded.
        # E.g. threshold=8 → fall back when preferred load > 1.8x least-loaded load.
        # Avoids sensitivity to absolute token counts.
        if least_load > 0:
            ratio = preferred_load / least_load
            if ratio > 1 + self._load_threshold / 10:
                return least_loaded
        elif preferred_load > self._load_threshold * 1000:
            # least_load == 0 but preferred has significant backlog
            return least_loaded
        return preferred

    def routing_frequencies(self) -> dict[str, dict[str, int]]:
        """Expose per-(task_type, instance) routing counts for eviction layer."""
        return dict(self._routing_freq)


def _queue_length(cache_state: dict, instance_name: str) -> int:
    state = cache_state.get(instance_name)
    if state is None:
        return 0
    if hasattr(state, "queue_length"):     # InstanceState dataclass
        return state.queue_length
    return int(state.get("queue_length", 0))  # plain dict fallback
