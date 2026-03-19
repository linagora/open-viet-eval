"""ViQuAD / Vietnamese SQuAD — Extractive Reading Comprehension.

Loads from HuggingFace: taidng/UIT-ViQuAD2.0 (validation split).
Supports unanswerable questions.
"""

import random
from datasets import load_dataset


# ── One-shot example for Unicorn's SQuAD prompt ─────────────────────────────
ONE_SHOT_CONTEXT = (
    "Vào năm 1945, Hồ Chí Minh đọc bản Tuyên ngôn Độc lập tại Quảng trường Ba Đình, "
    "khai sinh ra nước Việt Nam Dân chủ Cộng hòa. Sự kiện này đánh dấu sự kết thúc "
    "của chế độ thực dân Pháp tại Việt Nam."
)
ONE_SHOT_QUESTION = "Hồ Chí Minh đọc Tuyên ngôn Độc lập ở đâu?"
ONE_SHOT_ANSWER   = "Quảng trường Ba Đình"


def load_viquad(
    total: int = 1000,
    unanswerable_ratio: float = 0.1,
    seed: int = 42,
    max_samples: int | None = None,
):
    """Load and sample from UIT-ViQuAD2.0.

    Returns
    -------
    list[dict]  — keys: id, context, question, gold_answers
    """
    random.seed(seed)
    ds = load_dataset("taidng/UIT-ViQuAD2.0", split="validation")
    print(f"[ViQuAD] Loaded {len(ds)} raw samples.")

    answerable, unanswerable = [], []
    for item in ds:
        answers = item.get("answers", {}).get("text", [])
        (answerable if answers else unanswerable).append(item)

    unans_target = int(total * unanswerable_ratio)
    ans_target   = total - unans_target

    print(f"[ViQuAD] Answerable pool: {len(answerable)}, Unanswerable pool: {len(unanswerable)}")

    selected = (
        random.sample(answerable, min(ans_target, len(answerable)))
        + random.sample(unanswerable, min(unans_target, len(unanswerable)))
    )
    random.shuffle(selected)

    samples = []
    for i, item in enumerate(selected):
        samples.append({
            "id":           item.get("id", f"sample_{i}"),
            "context":      item["context"],
            "question":     item["question"],
            "gold_answers": item.get("answers", {}).get("text", []),
        })

    if max_samples:
        samples = samples[:max_samples]
    print(f"[ViQuAD] Evaluation set: {len(samples)} samples.")
    return samples


def build_squad_messages(context: str, question: str) -> list:
    """Build chat messages for a single ViQuAD question (for Qwen models)."""
    content = (
        "Dựa vào đoạn văn bản dưới đây, hãy trả lời câu hỏi.\n"
        "- Nếu câu trả lời tồn tại trong đoạn văn, hãy trích xuất một đoạn ngắn "
        "trực tiếp từ văn bản.\n"
        "- Nếu không có câu trả lời trong đoạn văn, hãy trả lời: "
        "'không có câu trả lời'.\n\n"
        f"Đoạn văn:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời:"
    )
    return [{"role": "user", "content": content}]


def build_squad_prompt_unicorn(context: str, question: str) -> str:
    """Build a raw <|im_start|> prompt for Unicorn on ViQuAD, with one-shot example."""
    content = (
        "Dựa vào đoạn văn bản dưới đây, hãy trả lời câu hỏi ngắn gọn nhất có thể.\n"
        "- Nếu câu trả lời tồn tại trong đoạn văn, hãy trích xuất một đoạn ngắn "
        "trực tiếp từ văn bản.\n"
        "- Nếu không có câu trả lời trong đoạn văn, hãy trả lời: "
        "'không có câu trả lời'.\n\n"
        f"Đoạn văn:\n{context}\n\n"
        f"Câu hỏi: {question}\n\n"
    )
    prompt = (
        "<|im_start|>system\n"
        "Bạn là trợ lý trả lời câu hỏi. Trả lời bằng Tiếng Việt, ngắn gọn, "
        "chỉ trích xuất cụm từ từ đoạn văn.<|im_end|>\n"
        # ── one-shot example ──
        "<|im_start|>user\n"
        "Dựa vào đoạn văn bản dưới đây, hãy trả lời câu hỏi ngắn gọn nhất có thể.\n"
        "- Nếu câu trả lời tồn tại trong đoạn văn, hãy trích xuất một đoạn ngắn "
        "trực tiếp từ văn bản.\n"
        "- Nếu không có câu trả lời trong đoạn văn, hãy trả lời: "
        "'không có câu trả lời'.\n\n"
        f"Đoạn văn:\n{ONE_SHOT_CONTEXT}\n\n"
        f"Câu hỏi: {ONE_SHOT_QUESTION}\n\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"{ONE_SHOT_ANSWER}<|im_end|>\n"
        # ── actual query ──
        "<|im_start|>user\n"
        f"{content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return prompt
