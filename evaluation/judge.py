"""SeaLLMs judge model for scoring model predictions.

Supports two evaluation modes:
  - Summarisation (ViMs): faithfulness, coverage, coherence
  - Instruction-following (VTSNLP): accuracy, completeness, coherence, relevance
"""

import gc
import json as _json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


JUDGE_MODEL_ID = "SeaLLMs/SeaLLMs-v3-7B-Chat"

# ── System prompts ───────────────────────────────────────────────────────────

JUDGE_SYSTEM_SUMMARISATION = (
    "Bạn là chuyên gia đánh giá chất lượng tóm tắt tin tức tiếng Việt.\n"
    "Đánh giá bản tóm tắt của mô hình theo 3 tiêu chí:\n"
    "1. Faithfulness: tóm tắt có bịa đặt thông tin không có trong bài gốc không?\n"
    "2. Coverage: tóm tắt có bao quát các ý chính từ tất cả các bài báo không?\n"
    "3. Coherence: tóm tắt có rõ ràng, trôi chảy, dễ đọc không?\n\n"
    "Trả lời ĐÚNG định dạng JSON, KHÔNG thêm bất kỳ văn bản nào khác:\n"
    '{"score": <1-10>, "faithfulness": <1-10>, "coverage": <1-10>, '
    '"coherence": <1-10>, "rationale": "<tối đa 2 câu tiếng Việt>"}'
)

JUDGE_SYSTEM_INSTRUCT = (
    "Bạn là một giám khảo AI chuyên đánh giá chất lượng câu trả lời bằng tiếng Việt.\n"
    "Đánh giá theo: độ chính xác, tính đầy đủ, tính mạch lạc, độ phù hợp.\n"
    "Trả lời CHÍNH XÁC theo định dạng JSON sau, không thêm bất kỳ văn bản nào khác:\n"
    '{"score": <số nguyên từ 1 đến 10>, "rationale": "<lý do ngắn gọn bằng tiếng Việt, tối đa 2 câu>"}'
)


class JudgeModel:
    """Wrapper around the SeaLLMs judge model."""

    def __init__(self, model_id: str = JUDGE_MODEL_ID):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        print(f"[Judge] Loading {self.model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True, padding_side="left"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[Judge] Ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def unload(self):
        vram_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        del self.model, self.tokenizer
        self.model = self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        vram_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(f"[Judge] Unloaded. VRAM freed: {vram_before - vram_after:.2f} GB")

    # Prompt builders
    @staticmethod
    def build_summarisation_messages(documents: str, reference: str, prediction: str) -> list:
        """Build judge messages for summarisation evaluation (ViMs)."""
        user = (
            f"### Bài báo gốc (trích đoạn 2000 ký tự):\n{documents[:2000]}\n\n"
            f"### Tóm tắt tham chiếu:\n{reference[:500]}\n\n"
            f"### Tóm tắt của mô hình:\n{prediction[:500]}\n\n"
            "Đánh giá và trả về JSON."
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_SUMMARISATION},
            {"role": "user",   "content": user},
        ]

    @staticmethod
    def build_instruct_messages(instruct: str, inp: str, reference: str, prediction: str) -> list:
        """Build judge messages for instruction-following evaluation (VTSNLP)."""
        content = (
            f"### Hướng dẫn:\n{str(instruct).strip()}\n\n"
            f"### Câu hỏi:\n{str(inp).strip()[:800]}\n\n"
            f"### Tham chiếu:\n{str(reference).strip()[:600]}\n\n"
            f"### Mô hình:\n{str(prediction).strip()[:600]}\n\n"
            "Đánh giá và trả JSON."
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_INSTRUCT},
            {"role": "user",   "content": content},
        ]

    # Scoring
    @torch.inference_mode()
    def score(self, messages: list, debug: bool = False) -> dict:
        """Score a single prediction via the judge model. Returns parsed dict."""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            gen_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        )
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if debug:
            sep = "=" * 65
            print(sep, "\n[JUDGE PROMPT (first 1500 chars)]\n", sep)
            print(prompt[:1500], "...")
            print(sep, "\n[JUDGE RAW OUTPUT]\n", sep)
            print(repr(raw))
            print(sep)

        result = parse_judge_output(raw)
        result["raw_output"] = raw
        return result


def parse_judge_output(raw: str) -> dict:
    """Parse JSON from judge output with multiple fallback strategies."""
    keys_full  = ("score", "faithfulness", "coverage", "coherence", "rationale")
    keys_basic = ("score", "rationale")

    raw = raw.strip()

    # Strategy 1: direct JSON parse
    try:
        obj = _json.loads(raw)
        # Determine which keys to use based on what's present
        if "faithfulness" in obj:
            return {k: obj.get(k, 5 if k != "rationale" else "") for k in keys_full}
        return {"score": int(obj.get("score", 5)), "rationale": obj.get("rationale", "")}
    except Exception:
        pass

    # Strategy 2: find JSON object in output
    m = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if "faithfulness" in obj:
                return {k: obj.get(k, 5 if k != "rationale" else "") for k in keys_full}
            return {"score": int(obj.get("score", 5)), "rationale": obj.get("rationale", "")}
        except Exception:
            pass

    # Strategy 3: regex fallback
    sm = re.search(r'"score"\s*:\s*(\d+)', raw)
    sc = min(max(int(sm.group(1)), 1), 10) if sm else 5
    return {"score": sc, "faithfulness": 5, "coverage": 5, "coherence": 5, "rationale": "(parse fallback)"}
