"""Abstract base class for all benchmark models."""

import abc
import gc
import torch


class BaseModel(abc.ABC):
    """Base class that every model adapter must implement."""

    model_id: str = ""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def load(self):
        """Load model, tokenizer (and processor if needed) onto GPU."""

    def unload(self):
        """Free GPU memory occupied by this model."""
        vram_before = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        for attr in ("model", "tokenizer", "processor"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        vram_after = (
            torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        )
        print(
            f"[{self.model_id}] Unloaded. VRAM freed: {vram_before - vram_after:.2f} GB"
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def generate(
        self,
        prompt_or_messages,
        max_new_tokens: int = 512,
        debug: bool = False,
    ) -> str:
        """Run inference and return the cleaned output string."""
