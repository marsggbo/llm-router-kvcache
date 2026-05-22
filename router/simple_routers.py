import random
from .base import BaseRouter, RoutingDecision


class RandomRouter(BaseRouter):
    """Routes each request randomly to weak or strong model."""

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        score = random.random()
        return self._make_decision(score, task_type)


class AlwaysWeakRouter(BaseRouter):
    """Always routes to the weak model. Throughput upper bound."""

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        return RoutingDecision(
            tier="weak",
            url=self.weak_url,
            model_name=self.weak_model,
            score=0.0,
            task_type=task_type,
        )


class AlwaysStrongRouter(BaseRouter):
    """Always routes to the strong model. Quality upper bound."""

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        return RoutingDecision(
            tier="strong",
            url=self.strong_url,
            model_name=self.strong_model,
            score=1.0,
            task_type=task_type,
        )
