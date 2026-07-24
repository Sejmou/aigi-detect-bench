from .base import REGISTRY, Detector, MeanEnsemble, register  # noqa: F401


def build(name: str, **kwargs) -> Detector:
    """Instantiate a registered detector by name (heavy imports are lazy)."""
    if name not in REGISTRY:
        # Trigger lazy registration of built-ins.
        if name == "clip_linear":
            from . import clip_linear  # noqa: F401
        elif name == "npr":
            from . import npr  # noqa: F401
        elif name == "torch_module":
            from . import external  # noqa: F401
    if name not in REGISTRY:
        raise KeyError(f"Unknown detector '{name}'. Registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
