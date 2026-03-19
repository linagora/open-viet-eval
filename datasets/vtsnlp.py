"""VTSNLP — Vietnamese Instruction-following / General NLP Tasks.

Loads from a CSV file with columns: category, instruct, input, output, max_len.
"""

import random
from collections import Counter

import pandas as pd


# ── One-shot example for Unicorn's instruct prompt ───────────────────────────
_UNICORN_ONESHOT_CONTEXT = (
    "Nguyễn Trãi (1380 – 1442), hiệu là Ức Trai, là một nhà chính trị, nhà văn, người đã tham gia "
    "tích cực Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo. Ông trở thành khai quốc công thần của triều đại "
    "nhà Lê sơ. Thân thế Nguyễn Trãi quê ở làng Chi Ngại, nay thuộc Chí Linh, Hải Dương. Cha của "
    "cậu là Nguyễn Phi Khanh, một học giả nổi tiếng. Từ nhỏ, Ức Trai đã nổi tiếng thông minh. Sau "
    "khi thi đỗ thái học sinh, vị quan trẻ tuổi này ra làm quan dưới triều Hồ. Sự nghiệp Khi quân Minh "
    "xâm lược nước ta, cha ông bị bắt giải sang Trung Quốc. Nguyễn Trãi nghe lời cha quay về tìm cách "
    "rửa nhục cho đất nước. Về sau, nhà chiến lược đại tài này đã tìm đến Lỗi Giang để theo Lê Lợi. "
    "Dưới trướng của Bình Định Vương, ông đã dâng \"Bình Ngô sách\", hiến kế đánh đuổi quân Minh. Sau "
    "khi kháng chiến thành công, Nguyễn Trãi viết Bình Ngô đại cáo. Gia đình Một trong những người vợ "
    "của ông là Nguyễn Thị Lộ. Bà là một người phụ nữ tài sắc, được vua Lê Thái Tông phong làm Lễ nghi "
    "học sĩ. Tuy nhiên, Lệ Chi Viên đã trở thành vụ án oan khốc liệt khiến cả gia đình nhà văn vĩ đại "
    "này bị chu di tam tộc. Năm 1980, Nguyễn Trãi được UNESCO công nhận là Danh nhân văn hóa thế giới."
)

_UNICORN_ONESHOT_Q = "Xác định các danh từ, đại từ thay thế để có thể nhận ra chúng đề cập đến cùng một đối tượng"

_UNICORN_ONESHOT_A = (
    "- Nguyễn Trãi: Ức Trai, nhà chính trị, nhà văn, người, ông, khai quốc công thần, cậu, vị quan "
    "trẻ tuổi này, nhà chiến lược đại tài này, nhà văn vĩ đại này, Danh nhân văn hóa thế giới\n"
    "- Nguyễn Phi Khanh: Cha của cậu, học giả nổi tiếng, cha ông, cha\n"
    "- Lê Lợi: Bình Định Vương\n"
    "- Nguyễn Thị Lộ: vợ của ông, Bà, người phụ nữ tài sắc, Lễ nghi học sĩ\n"
    "- Lê Thái Tông: vua Lê Thái Tông"
)


def load_vtsnlp(
    csv_path: str = "/kaggle/input/datasets/qnfuioyhgvqpwo/sample-vtsnlp-instruct-dataset/sampled_instruct_general_dataset.csv",
    n_per_cat: int = 10,
    seed: int = 42,
    max_samples: int | None = None,
) -> list:
    """Load and stratified-sample the VTSNLP instruct dataset.

    Returns
    -------
    list[dict]  — keys: category, instruct, input, output, max_len
    """
    random.seed(seed)
    df_raw = pd.read_csv(csv_path)
    print(f"[VTSNLP] Loaded {len(df_raw)} rows from CSV.")

    samples = (
        df_raw.groupby("category", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), n_per_cat), random_state=seed))
              .reset_index(drop=True)
              .to_dict("records")
    )

    cat_counts = Counter(s["category"] for s in samples)
    print(f"[VTSNLP] Total samples: {len(samples)}")
    print("[VTSNLP] Category distribution:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:<25}: {cnt}")

    if max_samples:
        samples = samples[:max_samples]
    return samples


def build_instruct_messages(instruct_text: str, input_text: str) -> list:
    """Build chat messages for a VTSNLP instruct sample (for Qwen models)."""
    messages = []
    if instruct_text and str(instruct_text).strip():
        messages.append({"role": "system", "content": str(instruct_text).strip()})
    messages.append({"role": "user", "content": str(input_text).strip()})
    return messages


def build_instruct_prompt_unicorn(instruct_text: str, input_text: str) -> str:
    """Build a raw <|im_start|> prompt for Unicorn on VTSNLP with one-shot."""
    instruct_text = str(instruct_text).strip()
    input_text    = str(input_text).strip()

    if instruct_text:
        user_content = f"{instruct_text}\n\n{input_text}"
    else:
        user_content = input_text

    return (
        "<|im_start|>system\n"
        "Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt, ngắn gọn và chính xác.<|im_end|>\n"
        # ── one-shot turn ──
        "<|im_start|>user\n"
        "Đây là một ví dụ về yêu cầu và kết quả của tác vụ mà tôi muốn bạn thực hiện \n"
        f"Đoạn văn mẫu:\n{_UNICORN_ONESHOT_CONTEXT}\n\n"
        f"Câu hỏi mẫu: {_UNICORN_ONESHOT_Q}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"Câu trả lời mẫu: {_UNICORN_ONESHOT_A}<|im_end|>\n"
        # ── actual query ──
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
