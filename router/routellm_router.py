from .base import BaseRouter, RoutingDecision

# Maps backbone alias → (router_type_for_Controller, HuggingFace_checkpoint_path)
# router_type is the key RouteLLM uses internally (mf | bert | causal_llm).
# checkpoint_path is passed as kwargs to the router's __init__.
ROUTELLM_BACKBONES: dict[str, tuple[str, str]] = {
    "bert":             ("bert",        "routellm/bert"),
    "bert_gpt4":        ("bert",        "routellm/bert_gpt4_augmented"),
    "bert_mmlu":        ("bert",        "routellm/bert_mmlu_augmented"),
    "mf":               ("mf",          "routellm/mf"),
    "mf_gpt4":          ("mf",          "routellm/mf_gpt4_augmented"),
    "causal_llm":       ("causal_llm",  "routellm/causal_llm"),
    "causal_llm_gpt4":  ("causal_llm",  "routellm/causal_llm_gpt4_augmented"),
}


def _resolve_device(device_cfg: str) -> str:
    """Resolve 'auto' to the best available device: cuda > mps > cpu."""
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class RouteLLMRouter(BaseRouter):
    """
    Routes requests using RouteLLM's pretrained router models.

    Open-source weights: https://huggingface.co/routellm
    Companion datasets:  https://huggingface.co/datasets/routellm/gpt4_judge_battles

    Recommended backbone: bert_gpt4 (0.3B BERT, low routing latency).
    The router predicts P(strong model needed) and thresholds at cfg.router.threshold.

    Device selection (cfg.router.device):
      "auto"  → cuda (GPU server) | mps (Apple Silicon) | cpu (fallback)
      "cuda"  → force CUDA
      "mps"   → force Apple Silicon GPU
      "cpu"   → force CPU

    Requires: pip install routellm
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._router = None
        self._backbone = cfg["router"].get("routellm_model", "bert_gpt4")
        self._device = _resolve_device(cfg["router"].get("device", "auto"))

    def _load(self):
        if self._router is not None:
            return
        # RouteLLM imports OpenAI client at module level (for similarity_weighted router).
        # Set a dummy key so the import doesn't crash when OPENAI_API_KEY is unset.
        import os
        os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")
        try:
            from routellm.controller import Controller
        except ImportError:
            raise ImportError("Run: pip install routellm")

        entry = ROUTELLM_BACKBONES.get(self._backbone)
        if entry is None:
            raise ValueError(
                f"Unknown backbone '{self._backbone}'. "
                f"Choose from: {list(ROUTELLM_BACKBONES)}"
            )
        router_type, checkpoint_path = entry
        print(f"[RouteLLMRouter] loading {checkpoint_path} ({router_type}) on {self._device}")

        # Controller downloads weights from HuggingFace on first call.
        # strong_model/weak_model are label references from training data —
        # they don't need to match your actual serving models.
        # config keys must match the router_type string used by RouteLLM internally.
        controller = Controller(
            routers=[router_type],
            strong_model="gpt-4-1106-preview",
            weak_model="mixtral-8x7b-instruct-v0.1",
            config={router_type: {"checkpoint_path": checkpoint_path}},
        )
        # Each router exposes calculate_strong_win_rate(prompt) -> float [0, 1]
        self._router = controller.routers[router_type]

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        self._load()
        score = self._router.calculate_strong_win_rate(prompt)
        return self._make_decision(float(score), task_type)
