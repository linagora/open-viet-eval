"""VMLU — Vietnamese Multiple-choice Language Understanding.

Loads from HuggingFace: tridm/VMLU (validation split).
"""

import re
from datasets import load_dataset


# ── Subject-ID → broad category mapping ──────────────────────────────────────
CATEGORY_MAP = {
    **{str(i).zfill(2): "STEM"           for i in range(1,  22)},
    **{str(i).zfill(2): "Social Science" for i in range(22, 32)},
    **{str(i).zfill(2): "Humanities"     for i in range(32, 50)},
    **{str(i).zfill(2): "Other"          for i in range(50, 59)},
}


def load_vmlu(max_samples: int | None = None):
    """Load the VMLU validation split.

    Returns
    -------
    list[dict]  — each dict has keys:  id, question, choices, answer
    """
    ds = load_dataset("tridm/VMLU", split="validation")
    print(f"[VMLU] Loaded {len(ds)} samples from HuggingFace.")

    samples = []
    for item in ds:
        samples.append({
            "id":       item["id"],
            "question": item["question"],
            "choices":  item["choices"],
            "answer":   item.get("answer"),
        })
    if max_samples:
        samples = samples[:max_samples]
    return samples


def build_vmlu_messages(question: str, choices: list) -> list:
    """Build chat messages for a single VMLU question."""
    choices_text = "\n".join(choices)
    content = (
        "Hãy trả lời câu hỏi trắc nghiệm sau bằng cách chọn đúng một đáp án "
        "(A, B, C, hoặc D).\n"
        "Chỉ trả lời bằng MỘT chữ cái duy nhất, không giải thích thêm.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"{choices_text}\n\n"
        "Đáp án:"
    )
    return [{"role": "user", "content": content}]


def build_vmlu_prompt_unicorn(question: str, choices: list) -> str:
    """Build a raw <|im_start|> prompt for Unicorn on VMLU."""
    choices_text = "\n".join(choices)
    content = (
        "Hãy trả lời câu hỏi trắc nghiệm sau bằng cách chọn đúng một đáp án "
        "(A, B, C, hoặc D). Trả lời bằng Tiếng Việt\n"
        "Chỉ trả lời bằng MỘT chữ cái duy nhất, không giải thích thêm.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"{choices_text}\n\n"
    )
    return content


def extract_mcq_answer(text: str) -> str:
    """Extract the first A/B/C/D token from raw model output."""
    text = text.strip()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text.upper())
    if m:
        return m.group(1)
    return "A"  # last-resort fallback
