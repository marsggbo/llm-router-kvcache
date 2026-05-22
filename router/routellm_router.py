from .base import BaseRouter, RoutingDecision


class RouteLLMRouter(BaseRouter):
    """
    Routes requests using RouteLLM's trained router models.

    Requires: pip install routellm[serve]
    Supported backbones: mf (matrix factorization), bert, causal_llm

    RouteLLM outputs a score in [0, 1] representing the probability
    that the strong model is needed. We threshold at cfg.router.threshold.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._controller = None

    def _load(self):
        if self._controller is not None:
            return
        try:
            from routellm.controller import Controller
        except ImportError:
            raise ImportError(
                "RouteLLM not installed. Run: pip install routellm"
            )
        backbone = self.cfg["router"].get("routellm_model", "mf")
        # RouteLLM needs a pair of model names to compute routing scores.
        # We pass the configured model names as strong/weak references.
        self._controller = Controller(
            routers=[backbone],
            strong_model=self.strong_model,
            weak_model=self.weak_model,
        )
        self._backbone = backbone

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        self._load()
        # route_with_thresholds returns the chosen model name;
        # we call the underlying router to get the raw score instead.
        router_obj = self._controller.routers[self._backbone]
        score = router_obj.calculate_strong_win_rate(prompt)
        return self._make_decision(score, task_type)
