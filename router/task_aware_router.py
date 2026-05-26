"""
Task-Aware Router — routes requests to dedicated instances by task type.

Phase 1 (baseline):
  Fixed affinity: same task_type always goes to the same instance.
  Creates natural prefix clustering → higher KV cache hit rate.

Phase 3 (optimization):
  When cache_state is provided, selects the instance with highest
  prefix cache affinity among compatible instances.

This router uses dataset labels as routing keys (oracle routing),
isolating the cache management contribution from router quality.
"""

import hashlib
from collections import defaultdict

from .base import BaseRouter, Instance, RoutingDecision


class TaskAwareRouter(BaseRouter):
    """
    Routes by task_type label (from dataset metadata, no ML model needed).

    Config example:
      instances:
        - name: coding
          url: http://localhost:30000
          model: Qwen/Qwen2.5-7B-Instruct
          task_types: [coding]
        - name: math
          url: http://localhost:30001
          model: Qwen/Qwen2.5-7B-Instruct
          task_types: [math]
        - name: general
          url: http://localhost:30002
          model: Qwen/Qwen2.5-7B-Instruct
          task_types: []   # handles everything else
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        # Track request counts per (task_type, instance) for freq-weighted eviction
        self._routing_freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def route(self, prompt: str, task_type: str = "unknown",
              cache_state: dict | None = None) -> RoutingDecision:
        candidates = self.compatible_instances(task_type)

        if cache_state:
            inst = self._cache_affinity_select(prompt, candidates, cache_state)
        else:
            inst = self._affinity_select(task_type, candidates)

        self._routing_freq[task_type][inst.name] += 1

        return RoutingDecision(
            instance_name=inst.name,
            url=inst.url,
            model_name=inst.model_name,
            score=1.0,
            task_type=task_type,
        )

    def _affinity_select(self, task_type: str, candidates: list[Instance]) -> Instance:
        """
        Phase 1: deterministic affinity — hash task_type to a fixed instance.
        Same task_type always lands on the same instance → prefix clustering.
        """
        idx = int(hashlib.md5(task_type.encode()).hexdigest(), 16) % len(candidates)
        return candidates[idx]

    def _cache_affinity_select(self, prompt: str, candidates: list[Instance],
                                cache_state: dict) -> Instance:
        """
        Phase 3: pick the instance with highest estimated prefix overlap.
        cache_state = {instance_name: {"prefix_index": RadixIndex, ...}}
        Falls back to fixed affinity if cache state is unavailable.
        """
        best_inst, best_score = candidates[0], -1.0
        prompt_tokens = prompt.split()

        for inst in candidates:
            state = cache_state.get(inst.name, {})
            prefix_index = state.get("prefix_index")
            if prefix_index is None:
                continue
            hit_tokens = prefix_index.longest_prefix_length(prompt_tokens)
            score = hit_tokens / max(len(prompt_tokens), 1)
            if score > best_score:
                best_score, best_inst = score, inst

        if best_score < 0:
            return self._affinity_select(prompt, candidates)
        return best_inst

    def routing_frequencies(self) -> dict[str, dict[str, int]]:
        """Expose routing frequency stats to the cache eviction layer."""
        return dict(self._routing_freq)
