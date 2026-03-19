"""Qwen3-8B model adapter."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseModel
from utils import strip_thinking


class Qwen3Model(BaseModel):
    """Qwen/Qwen3-8B — text-only causal LM."""

    model_id = "Qwen/Qwen3-8B"

    def load(self):
        print(f"Loading {self.model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[{self.model_id}] Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    @torch.inference_mode()
    def generate(self, messages: list, max_new_tokens: int = 512, debug: bool = False) -> str:
        """Generate from a list of chat messages (dicts with role/content)."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
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
        out = strip_thinking(raw_out)

        if debug:
            sep = "=" * 65
            print(sep, f"\n[PROMPT → {self.model_id}]\n", sep)
            print(prompt)
            print(sep, "\n[RAW OUTPUT]\n", sep)
            print(repr(raw_out[:600]))
            print(sep, "\n[CLEANED]\n", sep)
            print(repr(out))
            print(sep)

        return out
