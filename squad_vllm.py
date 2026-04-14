import ast
import re
import csv
import string
import asyncio
import argparse
import logging
import os
import random
random.seed(42)
from collections import Counter

import pandas as pd
from tqdm.asyncio import tqdm
from openai import AsyncOpenAI

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Vietnamese stopwords
VI_STOPWORDS = {"trong", "ở", "vào", "tại", "là", "của", "được"}

# Thinking-block suppression
_THINKING_PATTERNS = [
    re.compile(r"<think>.*?</think>",     re.DOTALL),
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
]

def strip_thinking_blocks(text: str) -> str:
    """Remove model reasoning blocks (<think>…</think>, <thought>…</thought>)."""
    if not text:
        return ""
    for pattern in _THINKING_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

# Prompt builder
def build_messages(context: str, question: str) -> list[dict]:
    """
    Build a few-shot chat message list for extractive QA.
    The model must copy a short span verbatim from the context.
    """
    system_msg = (
        "Bạn là một hệ thống Trích xuất thông tin (Extractive QA) tự động. "
        "Quy tắc tối thượng: BẮT BUỘC phải copy chính xác một cụm từ ngắn có sẵn trong 'Đoạn văn' để làm câu trả lời. "
        "TUYỆT ĐỐI KHÔNG trả lời thành một câu hoàn chỉnh. TUYỆT ĐỐI KHÔNG thêm từ ngữ của bạn vào. "
        "Nếu không có thông tin trong đoạn văn, chỉ in ra đúng chuỗi: 'không có câu trả lời'."
    )

    example_context = (
        "Thành phố Hồ Chí Minh (thường được gọi là Sài Gòn) là một thành phố ở miền Nam Việt Nam. "
        "Năm 2021, dân số thành phố đạt khoảng 9 triệu người, là trung tâm kinh tế lớn nhất cả nước."
    )

    return [
        {"role": "system", "content": system_msg},
        # Few-shot: answerable
        {
            "role": "user",
            "content": (
                f"Đoạn văn:\n{example_context}\n\n"
                "Câu hỏi: Sài Gòn nằm ở miền nào của Việt Nam?\n\n"
                "Đáp án trích xuất:"
            ),
        },
        {"role": "assistant", "content": "miền Nam"},
        # Few-shot: unanswerable
        {
            "role": "user",
            "content": (
                f"Đoạn văn:\n{example_context}\n\n"
                "Câu hỏi: Ai là thị trưởng của thành phố?\n\n"
                "Đáp án trích xuất:"
            ),
        },
        {"role": "assistant", "content": "không có câu trả lời"},
        # Actual query
        {
            "role": "user",
            "content": (
                f"Đoạn văn:\n{context}\n\n"
                f"Câu hỏi: {question}\n\n"
                "Đáp án trích xuất:"
            ),
        },
    ]


# Text normalisation & scoring
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in VI_STOPWORDS]
    return " ".join(tokens)

def is_no_answer(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in [
            "không có câu trả lời",
            "không có thông tin",
            "không được đề cập",
            "không có",
        ]
    )

def compute_exact_match(prediction: str, gold_answers: list[str]) -> int:
    if not gold_answers:
        return int(is_no_answer(prediction))
    pred_norm = normalize_text(prediction)
    return int(any(pred_norm == normalize_text(g) for g in gold_answers))

def compute_f1(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 1.0 if is_no_answer(prediction) else 0.0

    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))

    return best_f1

# Data parsing
def extract_gold_answers(ans_str) -> list[str]:
    """
    Safely parse the gold answers from a stringified pandas/numpy dict that
    looks like: {'text': array(['answer'], dtype=object), 'answer_start': ...}
    """
    if pd.isna(ans_str):
        return []

    # Handle array([ ... ], dtype=object) format
    match = re.search(
        r"'text':\s*array\(\[(.*?)\](?:,\s*dtype=object)?\)",
        str(ans_str),
        flags=re.DOTALL,
    )
    if match:
        list_str = "[" + match.group(1) + "]"
        try:
            return ast.literal_eval(list_str)
        except (SyntaxError, ValueError):
            pass

    # Fallback: plain dict format
    try:
        parsed = ast.literal_eval(ans_str)
        if isinstance(parsed, dict) and "text" in parsed:
            return parsed["text"]
    except (SyntaxError, ValueError):
        pass

    return []

# Async worker
async def process_sample(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    sample: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    suppress_thinking: bool = False,
) -> dict:
    """Process a single SQuAD sample and return scored results."""
    sample_id = sample["id"]

    is_impossible = str(sample.get("is_impossible", False)).strip().lower() == "true"
    gold_answers = [] if is_impossible else extract_gold_answers(sample.get("answers", ""))

    messages = build_messages(sample["context"], sample["question"])

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            prediction = response.choices[0].message.content.strip()
            if suppress_thinking:
                prediction = strip_thinking_blocks(prediction)
        except Exception as exc:
            log.error("[FAIL] id=%s – %s", sample_id, exc)
            prediction = "ERROR"

    em = compute_exact_match(prediction, gold_answers)
    f1 = compute_f1(prediction, gold_answers)

    return {
        "id": sample_id,
        "uit_id": sample.get("uit_id", ""),
        "question": sample["question"],
        "gold_answers": " | ".join(gold_answers) if gold_answers else "UNANSWERABLE",
        "prediction": prediction,
        "exact_match": em,
        "f1": round(f1, 4),
    }

# Main
async def main(args: argparse.Namespace) -> None:
    log.info("Loading dataset from %s …", args.input_file)

    if not os.path.exists(args.input_file):
        log.error("Input file not found: %s", args.input_file)
        return

    try:
        df_input = pd.read_csv(args.input_file)
        dataset = df_input.to_dict(orient="records")
        log.info("Loaded %d samples.", len(dataset))
    except Exception as exc:
        log.error("Failed to load CSV: %s", exc)
        return

    # Sampling: keep a random subset of the dataset.
    if 0.0 < args.sample_rate < 1.0:
        k = max(1, int(len(dataset) * args.sample_rate))
        dataset = random.sample(dataset, k)
        log.info("Sampling %.1f%% → %d samples.", args.sample_rate * 100, len(dataset))

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)

    log.info(
        "Evaluating model=%s  concurrency=%d  max_tokens=%d  temperature=%.1f  suppress_thinking=%s",
        args.model, args.concurrency, args.max_tokens, args.temperature, args.suppress_thinking,
    )

    tasks = [
        process_sample(semaphore, client, sample, args.model, args.max_tokens, args.temperature, args.suppress_thinking)
        for sample in dataset
    ]
    results = await tqdm.gather(*tasks, desc="Evaluating ViQuAD")

    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results["prediction"] != "ERROR"]

    if not df_valid.empty:
        mean_em = df_valid["exact_match"].mean() * 100
        mean_f1 = df_valid["f1"].mean() * 100
        print(f"\n{'='*55}")
        print(f"  Vietnamese SQuAD Results")
        print(f"  Model           : {args.model}")
        print(f"  Total Evaluated : {len(df_valid)}")
        print(f"  Errors skipped  : {len(df_results) - len(df_valid)}")
        print(f"  Exact Match     : {mean_em:.2f}%")
        print(f"  F1 Score        : {mean_f1:.2f}%")
        print(f"{'='*55}\n")
    else:
        log.warning("No valid predictions — all requests errored.")

    df_results.to_csv(args.output_file, index=False, encoding="utf-8")
    log.info("Results saved to %s", args.output_file)
    print(df_results[["id", "gold_answers", "prediction"]].head(5).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on ViQuAD (Vietnamese SQuAD) via vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file",   default="data/validation.csv",           help="Path to the input CSV file.")
    parser.add_argument("--output_file",  default="viquad_async_results.csv",      help="Path to save results CSV.")
    parser.add_argument("--base_url",     default="http://localhost:8000/v1",      help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model",        default="Qwen/Qwen3-0.6B",               help="Model name as registered in vLLM.")
    parser.add_argument("--concurrency",  type=int,   default=16,                  help="Max concurrent API requests.")
    parser.add_argument("--max_tokens",   type=int,   default=1024,                 help="Max tokens for the model response.")
    parser.add_argument("--temperature",  type=float, default=0.0,                 help="Sampling temperature.")
    parser.add_argument(
        "--suppress_thinking",
        action="store_true",
        default=True,
        help="Strip <think>…</think> / <thought>…</thought> blocks from model output before scoring.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="Fraction of the dataset to evaluate, e.g. 0.1 for 10%%. Must be in (0, 1]. Default: 1.0 (all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))