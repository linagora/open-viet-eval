"""Model loaders for Vietnamese LLM benchmarking."""

from .qwen35 import Qwen35Model
from .qwen3 import Qwen3Model
from .unicorn import UnicornModel

MODEL_REGISTRY = {
    "qwen35": Qwen35Model,
    "qwen3": Qwen3Model,
    "unicorn": UnicornModel,
}


def get_model(name: str):
    """Look up a model class by short name."""
    cls = MODEL_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    return cls()
