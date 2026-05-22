from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


ModelTier = Literal["weak", "strong"]


@dataclass
class RoutingDecision:
    tier: ModelTier
    url: str
    model_name: str
    score: float        # router confidence: 0=weak, 1=strong
    task_type: str = "unknown"


class BaseRouter(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.weak_url = cfg["serving"]["weak_url"]
        self.strong_url = cfg["serving"]["strong_url"]
        self.weak_model = cfg["models"]["weak"]
        self.strong_model = cfg["models"]["strong"]

    @abstractmethod
    def route(self, prompt: str, task_type: str = "unknown",
              cache_state: dict | None = None) -> RoutingDecision:
        """
        Return routing decision for the given prompt.

        cache_state: live KV cache state per instance, injected by CacheManager
                     in Phase 3. None in Phase 1/2 (stateless routing).
        """
        ...

    def _make_decision(self, score: float, task_type: str = "unknown") -> RoutingDecision:
        threshold = self.cfg["router"].get("threshold", 0.5)
        if score >= threshold:
            return RoutingDecision(
                tier="strong",
                url=self.strong_url,
                model_name=self.strong_model,
                score=score,
                task_type=task_type,
            )
        return RoutingDecision(
            tier="weak",
            url=self.weak_url,
            model_name=self.weak_model,
            score=score,
            task_type=task_type,
        )
