from .base import BaseRouter
from .routellm_router import RouteLLMRouter
from .simple_routers import RandomRouter, AlwaysWeakRouter, AlwaysStrongRouter
from .task_aware_router import TaskAwareRouter

ROUTER_REGISTRY = {
    "random": RandomRouter,
    "always_weak": AlwaysWeakRouter,
    "always_strong": AlwaysStrongRouter,
    "routellm": RouteLLMRouter,
    "task_aware": TaskAwareRouter,
}


def build_router(cfg: dict) -> BaseRouter:
    router_type = cfg["router"]["type"]
    if router_type not in ROUTER_REGISTRY:
        raise ValueError(f"Unknown router type: {router_type}. Choose from {list(ROUTER_REGISTRY)}")
    return ROUTER_REGISTRY[router_type](cfg)
