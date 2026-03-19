"""Unicorn-VL-R3 model adapter.

Unicorn uses raw <|im_start|> prompts (no apply_chat_template) and
requires `dtype` instead of `torch_dtype`. Each dataset benchmark has
its own one-shot prompt variant, so this class exposes both:
  - generate(messages)  for chat-message dicts  (builds raw prompt internally)
  - generate_raw(prompt) for pre-built prompt strings
"""

import re
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .base import BaseModel


class UnicornModel(BaseModel):
    """unicorn-team/Unicorn-VL-R3 — vision-language model used as text-only."""

    model_id = "unicorn-team/Unicorn-VL-R3"

    def load(self):
        print(f"Loading {self.model_id} ...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,   # Unicorn uses `dtype`, NOT `torch_dtype`
            device_map="auto",
        )
        self.model.eval()
        print(f"[{self.model_id}] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate from a raw prompt string (used by dataset-specific prompts)
    @torch.inference_mode()
    def generate_raw(self, prompt: str, max_new_tokens: int = 512, debug: bool = False) -> str:
        """Generate from a pre-built raw prompt string."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = gen_ids[0][inputs.input_ids.shape[-1]:]
        raw_out = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        out = re.sub(r"<think>.*?</think>", "", raw_out, flags=re.DOTALL).strip()

        if debug:
            sep = "=" * 65
            print(sep, f"\n[PROMPT → {self.model_id}]\n", sep)
            print(prompt[:2000], "..." if len(prompt) > 2000 else "")
            print(sep, "\n[RAW OUTPUT]\n", sep)
            print(repr(raw_out[:600]))
            print(sep, "\n[CLEANED]\n", sep)
            print(repr(out))
            print(sep)

        return out

    # Generate from chat messages (simple wrapper for VMLU etc.)
    @torch.inference_mode()
    def generate(self, messages, max_new_tokens: int = 512, debug: bool = False) -> str:
        """Generate from chat messages (str or list).

        If ``messages`` is already a string, it is used as the user content
        directly inside a basic <|im_start|> template.  If it is a list of
        dicts, the first dict's content is used.
        """
        if isinstance(messages, list):
            # Extract content from role/content dicts
            content = "\n\n".join(m["content"] for m in messages)
        else:
            content = str(messages)

        prompt = (
            "<|im_start|>system\n"
            "Answer all questions in Vietnamese\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{content}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        return self.generate_raw(prompt, max_new_tokens=max_new_tokens, debug=debug)
