from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


ModelTier = Literal["weak", "strong", "coding", "math", "general"]


@dataclass
class Instance:
    name: str
    url: str
    model_name: str
    task_types: list[str] = field(default_factory=list)  # empty = handles all


@dataclass
class RoutingDecision:
    instance_name: str
    url: str
    model_name: str
    score: float        # router confidence; semantics depend on router type
    task_type: str = "unknown"


class BaseRouter(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.instances = self._build_instances(cfg)

    def _build_instances(self, cfg: dict) -> list[Instance]:
        """Build instance list from config.

        Supports two config shapes:
          - Legacy (weak/strong binary): models.weak + models.strong
          - Multi-instance:              instances list
        """
        if "instances" in cfg:
            return [
                Instance(
                    name=inst["name"],
                    url=inst["url"],
                    model_name=inst["model"],
                    task_types=inst.get("task_types", []),
                )
                for inst in cfg["instances"]
            ]
        # Legacy two-tier config
        return [
            Instance(
                name="weak",
                url=cfg["serving"]["weak_url"],
                model_name=cfg["models"]["weak"],
                task_types=[],
            ),
            Instance(
                name="strong",
                url=cfg["serving"]["strong_url"],
                model_name=cfg["models"]["strong"],
                task_types=[],
            ),
        ]

    def get_instance(self, name: str) -> Instance:
        for inst in self.instances:
            if inst.name == name:
                return inst
        raise KeyError(f"Instance '{name}' not found")

    def compatible_instances(self, task_type: str) -> list[Instance]:
        """Return instances that can handle this task_type."""
        matches = [i for i in self.instances if task_type in i.task_types]
        if matches:
            return matches
        # Fall back to instances that handle all tasks (empty task_types list)
        fallback = [i for i in self.instances if not i.task_types]
        return fallback or self.instances

    @abstractmethod
    def route(self, prompt: str, task_type: str = "unknown",
              cache_state: dict | None = None) -> RoutingDecision:
        """Return routing decision for the given prompt.

        cache_state: per-instance KV cache stats injected by CacheManager (Phase 3).
        """
        ...

    # ------------------------------------------------------------------ #
    # Legacy helpers — used by simple_routers / routellm_router
    # ------------------------------------------------------------------ #
    @property
    def weak_url(self) -> str:
        return self.get_instance("weak").url

    @property
    def strong_url(self) -> str:
        return self.get_instance("strong").url

    @property
    def weak_model(self) -> str:
        return self.get_instance("weak").model_name

    @property
    def strong_model(self) -> str:
        return self.get_instance("strong").model_name

    def _make_decision(self, score: float, task_type: str = "unknown") -> RoutingDecision:
        threshold = self.cfg["router"].get("threshold", 0.5)
        inst = self.get_instance("strong" if score >= threshold else "weak")
        return RoutingDecision(
            instance_name=inst.name,
            url=inst.url,
            model_name=inst.model_name,
            score=score,
            task_type=task_type,
        )
