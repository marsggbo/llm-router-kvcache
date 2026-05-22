from .base import BaseRouter, RoutingDecision

# Pretrained RouteLLM model IDs on HuggingFace.
# bert variants are 0.3B — low routing latency, recommended for serving.
# causal_llm variants are 8B — higher accuracy but adds routing overhead.
ROUTELLM_HF_MODELS = {
    "bert":             "routellm/bert",
    "bert_gpt4":        "routellm/bert_gpt4_augmented",
    "bert_mmlu":        "routellm/bert_mmlu_augmented",
    "mf":               "routellm/mf",
    "mf_gpt4":          "routellm/mf_gpt4_augmented",
    "causal_llm":       "routellm/causal_llm",
    "causal_llm_gpt4":  "routellm/causal_llm_gpt4_augmented",
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
        try:
            from routellm.controller import Controller
        except ImportError:
            raise ImportError("Run: pip install routellm")

        hf_id = ROUTELLM_HF_MODELS.get(self._backbone)
        if hf_id is None:
            raise ValueError(
                f"Unknown backbone '{self._backbone}'. "
                f"Choose from: {list(ROUTELLM_HF_MODELS)}"
            )

        print(f"[RouteLLMRouter] loading {hf_id} on {self._device}")
        # Controller downloads weights from HuggingFace on first call.
        # strong_model/weak_model are label references from training data —
        # they don't need to match your actual serving models.
        controller = Controller(
            routers=[self._backbone.split("_")[0]],   # mf | bert | causal_llm
            strong_model="gpt-4-1106-preview",
            weak_model="mixtral-8x7b-instruct-v0.1",
            config={"model": hf_id, "device": self._device},
        )
        # Each router exposes calculate_strong_win_rate(prompt) -> float [0, 1]
        self._router = controller.routers[self._backbone.split("_")[0]]

    def route(self, prompt: str, task_type: str = "unknown") -> RoutingDecision:
        self._load()
        score = self._router.calculate_strong_win_rate(prompt)
        return self._make_decision(float(score), task_type)
