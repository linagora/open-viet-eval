"""
Each row contains an instruction and an optional input. The model response
is written to a new 'llm_output' column. A `--instruct_mode` flag controls
whether the model's internal reasoning (<think>…</think>) blocks are
preserved or stripped from the output.

Usage:
    python vtsnlp_vllm.py \
        --input_file data/filtered_data.parquet \
        --output_file processed_dataset.csv \
        --base_url http://localhost:8000/v1 \
        --model Qwen/Qwen3-0.6B \
        --concurrency 16 \
        --max_tokens 2048 \
        --temperature 0.7 \
        --save_every 100 \
        --instruct_mode \
        --sample_rate 0.5 \
        --limit 0
"""

import asyncio
import argparse
import logging
import re
import random
random.seed(42)
import time

import aiohttp
import pandas as pd


# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Text post-processing
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
def build_messages(instruct_text: str, input_text: str | None) -> list[dict]:
    """
    Build the chat messages for a single dataset row.
    Combines instruction + input when an input is present.
    """
    user_prompt = str(instruct_text)
    if input_text and str(input_text).strip():
        user_prompt += f"\n\n{input_text}"
    return [
        {"role": "system", "content": "Bạn là trợ lý đắc lực, hoàn thành các tác vụ bằng câu trả lời ngắn gọn nhất có thể và sử dụng tiếng Việt"},
        {"role": "user",   "content": user_prompt},
    ]

# Async worker
async def process_sample(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_url: str,
    model: str,
    row_index: int,
    instruct_text: str,
    input_text: str | None,
    max_tokens: int,
    temperature: float,
    instruct_mode: bool,
) -> tuple[int, str]:
    """
    Send a single async request to the vLLM server.
    Returns (row_index, generated_content).
    """
    payload = {
        "model": model,
        "messages": build_messages(instruct_text, input_text),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    async with semaphore:
        try:
            async with session.post(api_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    if not instruct_mode:
                        content = strip_thinking_blocks(content)
                    return row_index, content
                else:
                    error_text = await response.text()
                    log.error("Row %d: HTTP %d – %s", row_index, response.status, error_text)
                    return row_index, f"ERROR: HTTP {response.status}"
        except Exception as exc:
            log.error("Row %d: request failed – %s", row_index, exc)
            return row_index, f"ERROR: {exc}"

# Main
async def main(args: argparse.Namespace) -> None:
    log.info("Loading dataset: %s", args.input_file)
    df = pd.read_parquet(args.input_file)

    # Sampling: keep a random subset of rows.
    if 0.0 < args.sample_rate < 1.0:
        k = max(1, int(len(df) * args.sample_rate))
        df = df.sample(n=k, random_state=42).reset_index(drop=True)
        log.info("Sampling %.1f%% → %d rows.", args.sample_rate * 100, len(df))

    log.info(
        "Processing %d rows  model=%s  concurrency=%d  instruct_mode=%s  suppress_thinking=%s",
        len(df), args.model, args.concurrency, args.instruct_mode, not args.instruct_mode,
    )

    df["llm_output"] = ""
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout   = aiohttp.ClientTimeout(total=300)
    api_url   = args.base_url.rstrip("/") + "/chat/completions"

    start_time = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            process_sample(
                session, semaphore, api_url, args.model,
                index,
                row.get("instruct", ""),
                row.get("input", None),
                args.max_tokens, args.temperature, args.instruct_mode,
            )
            for index, row in df.iterrows()
        ]
        log.info("Sending %d requests to vLLM …", len(tasks))
        completed = 0
        total = len(tasks)
        for finished in asyncio.as_completed(tasks):
            row_index, content = await finished
            df.at[row_index, "llm_output"] = content
            completed += 1

            if args.save_every > 0 and completed % args.save_every == 0:
                df.to_csv(args.output_file, index=False)
                log.info(
                    "Checkpoint saved (%d/%d rows) to %s",
                    completed,
                    total,
                    args.output_file,
                )

    df.to_csv(args.output_file, index=False)

    elapsed = time.time() - start_time
    log.info("Done in %.2f seconds. Results saved to %s", elapsed, args.output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM outputs for a Vietnamese instruction-following dataset via vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_file",    default="data/filtered_data.parquet", help="Path to the input Parquet file.")
    parser.add_argument("--output_file",   default="Qwen3.5-0.8B-W4A16-AutoRound-AWQ_dataset.csv",               help="Path to save the output CSV file.")
    parser.add_argument("--base_url",      default="http://localhost:8000/v1",             help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--model",         default="Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ",                     help="Model name as registered in vLLM.")
    parser.add_argument("--concurrency",   type=int,   default=16,                        help="Max concurrent API requests.")
    parser.add_argument("--max_tokens",    type=int,   default=1024,                      help="Max tokens for each model response.")
    parser.add_argument("--temperature",   type=float, default=0.7,                       help="Sampling temperature.")
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save partial outputs every N completed rows. Set 0 to disable periodic saves.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.01,
        help="Fraction of rows to process, e.g. 0.5 for 50%%. Must be in (0, 1]. Applied after --limit. Default: 1.0 (all).",
    )
    parser.add_argument(
        "--instruct_mode",
        action="store_true",
        default=True,
        help=(
            "When set, keep the model's reasoning blocks (<think>…</think>) in the output. "
            "When omitted (default), those blocks are stripped."
        ),
    )
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main(parse_args()))