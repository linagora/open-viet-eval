"""
Evaluate an LLM on the Vietnamese Multiple-choice Language Understanding (VMLU)
benchmark. The model is asked to pick the single correct option (A/B/C/D) for
each multiple-choice question.

Supports resuming an interrupted run: already-processed question IDs are read
from the output CSV and skipped automatically.

Usage:
    python vlmu_vllm.py \
        --input_file /path/to/VMLU/test.jsonl \
        --output_file vmlu_results.csv \
        --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --concurrency 16 \
        --max_tokens 5 \
        --temperature 0.0 \
        --save_every 50 \
        --suppress_thinking \
        --sample_rate 0.1
"""

import json
import csv
import re
import asyncio
import argparse
import logging
import os
import random

from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
def build_messages(question_text: str, choices: list[str]) -> list[dict]:
    """
    Build a chat message list for a multiple-choice question.
    The model is instructed to reply with exactly one letter: A, B, C, or D.
    """
    system_msg = (
        "Bạn là một hệ thống giải đề thi tự động. "
        "Bạn CHỈ ĐƯỢC PHÉP trả lời bằng MỘT chữ cái duy nhất (A, B, C, hoặc D) "
        "tương ứng với đáp án đúng. "
        "Tuyệt đối không giải thích, không viết thêm bất kỳ từ nào khác."
    )
    one_shot_example = (
        "Ví dụ:\n"
        "Câu hỏi: Việc phát triển nông-lâm-thủy sản tạo cơ sở nguyên liệu cho ngành phát triển công nghiệp nào?\n"
        "A. Công nghiệp năng lượng\n"
        "B. Công nghiệp chế biến lương thực thực phẩm\n"
        "C. Công nghiệp hóa chất\n"
        "D. Công nghiệp sản xuất vật liệu xây dựng\n"
        "Đáp án: B"
    )
    user_msg = (
        f"{one_shot_example}\n\n"
        f"Câu hỏi:\n{question_text}\n\n"
        f"Các lựa chọn:\n{chr(10).join(choices)}\n\n"
        "Đáp án đúng là:"
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

# Async worker
async def process_sample(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    sample: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    suppress_thinking: bool = False,
) -> tuple[str, str]:
    """
    Process a single VMLU question and return (question_id, predicted_answer).
    The predicted answer is a single letter (A–D) or 'N/A'/'ERROR'.
    """
    question_id   = sample["id"]
    question_text = sample["question"]
    choices       = sample["choices"]

    messages = build_messages(question_text, choices)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = response.choices[0].message.content.strip()
            if suppress_thinking:
                raw = strip_thinking_blocks(raw)
            match = re.search(r"[ABCD]", raw.upper())
            return question_id, (match.group(0) if match else "N/A")
        except Exception as exc:
            log.error("[FAIL] id=%s – %s", question_id, exc)
            return question_id, "ERROR"

# Main
async def main(args: argparse.Namespace) -> None:

    # Resume: collect already-processed IDs from an existing output file.
    processed_ids: set[str] = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    processed_ids.add(row[0])
        log.info("Resuming: %d questions already processed.", len(processed_ids))

    # Load remaining questions.
    pending: list[dict] = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample["id"] not in processed_ids:
                pending.append(sample)

    if not pending:
        log.info("All questions have already been processed. Exiting.")
        return

    # Sampling: keep a random subset of the pending questions.
    if 0.0 < args.sample_rate < 1.0:
        k = max(1, int(len(pending) * args.sample_rate))
        pending = random.sample(pending, k)
        log.info("Sampling %.1f%% → %d questions.", args.sample_rate * 100, len(pending))

    log.info(
        "Evaluating %d questions  model=%s  concurrency=%d  max_tokens=%d  suppress_thinking=%s",
        len(pending), args.model, args.concurrency, args.max_tokens, args.suppress_thinking,
    )

    client    = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Open CSV (append if resuming, write-new otherwise) and batch-process.
    file_mode = "a" if processed_ids else "w"
    with open(args.output_file, file_mode, encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        if not processed_ids:
            writer.writerow(["id", "answer"])

        total = len(pending)
        for batch_start in range(0, total, args.save_every):
            batch = pending[batch_start : batch_start + args.save_every]
            tasks = [
                process_sample(semaphore, client, sample, args.model, args.max_tokens, args.temperature, args.suppress_thinking)
                for sample in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for q_id, answer in batch_results:
                writer.writerow([q_id, answer])
            out_f.flush()

            done = min(batch_start + len(batch), total)
            log.info("Saved %d / %d questions …", done, total)

    log.info("Evaluation complete. Results saved to %s", args.output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on the VMLU benchmark via vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file",   default="data/test.jsonl", help="Path to the VMLU test JSONL file.")
    parser.add_argument("--output_file",  default="qwen3.5-0.8B_yes_think.csv",                          help="Path to save results CSV.")
    parser.add_argument("--base_url",     default="http://localhost:8000/v1",                       help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model",        default="Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ",                                help="Model name as registered in vLLM.")
    parser.add_argument("--concurrency",  type=int,   default=16,                                  help="Max concurrent API requests.")
    parser.add_argument("--max_tokens",   type=int,   default=512,                                   help="Max tokens for the model response.")
    parser.add_argument("--temperature",  type=float, default=0.0,                                 help="Sampling temperature.")
    parser.add_argument("--save_every",   type=int,   default=50,                                  help="Flush results to disk after every N questions.")
    parser.add_argument(
        "--suppress_thinking",
        action="store_true",
        default=True,
        help="Strip <think>…</think> / <thought>…</thought> blocks before extracting the answer letter.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Fraction of questions to evaluate, e.g. 0.1 for 10%%. Must be in (0, 1]. Default: 1.0 (all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))