import os
import json
import argparse
import logging
import asyncio
import random
random.seed(42)
import re
from pathlib import Path

from openai import AsyncOpenAI

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Thinking-block suppression
_THINKING_PATTERNS = [
    re.compile(r"<think>.*?</think>",     re.DOTALL),
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
]
SERVER_TOKEN = 1024
def strip_thinking_blocks(text: str) -> str:
    """Remove model reasoning blocks (<think>…</think>, <thought>…</thought>)."""
    if not text:
        return ""
    for pattern in _THINKING_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()

# S3 important-sentence loading
def load_s3_important_sentences(s3_path: Path) -> list[str]:
    """
    Parse a .s3.txt file and return only the sentences marked as important
    (lines where the first tab-delimited field is "1").

    File format (one sentence per line):
        <label>\\t<sentence text>
    where label == "1" means the sentence is important.
    """
    sentences: list[str] = []
    try:
        text = s3_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        log.warning("Could not read %s: %s", s3_path, exc)
        return sentences

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip() == "1":
            sentences.append(parts[1].strip())

    return sentences


def read_cluster_important_sentences(cluster_dir: Path, s3_root: Path) -> list[str]:
    """
    Locate the s3 important-sentence file for this cluster and return the
    selected sentences.  The expected path is:

        <s3_root>/<cluster_name>/0.s3.txt

    Returns an empty list if the file is missing or contains no important
    sentences.
    """
    cluster_name = cluster_dir.name
    s3_file = s3_root / cluster_name / "0.s3.txt"

    if not s3_file.exists():
        log.warning("[%s] S3 file not found: %s", cluster_name, s3_file)
        return []

    sentences = load_s3_important_sentences(s3_file)
    if not sentences:
        log.warning("[%s] No important sentences found in %s", cluster_name, s3_file)
    return sentences

# Prompt builder
def build_prompt(sentences: list[str], max_tokens: int) -> str:
    """
    Build a prompt from pre-extracted important sentences, mirroring the
    approach used in vims-qwen.py.
    """
    doc_parts = [f"[Câu {i}] {sent}" for i, sent in enumerate(sentences, 1)]
    documents = "\n".join(doc_parts)

    # Truncate to stay within the context window
    if len(documents) > (SERVER_TOKEN - max_tokens) * 2:  
        clipped = documents[: (SERVER_TOKEN - max_tokens) * 2]
        last_period = clipped.rfind(".")
        if last_period != -1:
            clipped = clipped[: last_period + 1]
        documents = clipped.rstrip() + "\n..."

    user = (
        f"Các câu quan trọng:\n\n{documents}\n\n"
        "Hãy viết ngay bản tóm tắt tổng hợp (4–6 câu, tiếng Việt) nêu bật "
        "những thông tin quan trọng nhất từ các câu trên. "
        "Bắt đầu tóm tắt ngay, không giải thích, không dùng tiếng Anh."
    )
    return user

# Async API call
async def call_model(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    suppress_thinking: bool = False,
) -> str:
    """Send an async chat-completion request to the vLLM OpenAI-compatible endpoint."""
    system_msg = (
        "Bạn PHẢI trả lời HOÀN TOÀN bằng tiếng Việt. "
        "Bạn là trợ lý AI chuyên tóm tắt tin tức tiếng Việt. "
        "Dưới đây là các câu quan trọng đã được chọn lọc từ nhiều bài báo "
        "cùng chủ đề. Hãy viết NGAY một bản tóm tắt tổng hợp ngắn gọn, "
        "trung thực, đầy đủ bằng tiếng Việt dựa trên các câu này. "
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = response.choices[0].message.content.strip()
    if suppress_thinking:
        result = strip_thinking_blocks(result)
    return result

# Async worker
async def process_cluster(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    cluster_dir: Path,
    s3_root: Path,
    model: str,
    output_dir: Path,
    max_tokens: int,
    temperature: float,
    suppress_thinking: bool = False,
) -> dict:
    """
    Load important sentences for a cluster, call the model, and save the
    result.  Skips clusters that already have a saved summary.
    """
    cluster_name = cluster_dir.name
    out_file = output_dir / f"{cluster_name}_summary.json"

    # Resume support: skip clusters that already have a saved summary.
    if out_file.exists():
        log.info("[SKIP] %s – summary already exists.", cluster_name)
        with open(out_file, encoding="utf-8") as f:
            return json.load(f)

    sentences = read_cluster_important_sentences(cluster_dir, s3_root)
    if not sentences:
        log.warning("[EMPTY] %s – no important sentences found.", cluster_name)
        return {
            "cluster": cluster_name,
            "summary": "",
            "num_sentences": 0,
        }

    prompt = build_prompt(sentences, max_tokens)

    async with semaphore:
        try:
            summary = await call_model(
                client, prompt, model, max_tokens, temperature, suppress_thinking
            )
            log.info("[OK] %s (%d sentences)", cluster_name, len(sentences))
        except Exception as exc:
            log.error("[FAIL] %s – %s", cluster_name, exc)
            summary = f"ERROR: {exc}"

    result = {
        "cluster": cluster_name,
        "num_sentences": len(sentences),
        "summary": summary,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

# Main
async def main(args: argparse.Namespace) -> None:
    input_dir  = Path(args.input_dir)
    s3_root    = Path(args.s3_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect cluster directories
    if args.clusters:
        cluster_dirs = [input_dir / c for c in args.clusters]
    else:
        cluster_dirs = sorted(
            d for d in input_dir.iterdir()
            if d.is_dir() and d.name.startswith("Cluster_")
        )

    log.info("Found %d clusters to process.", len(cluster_dirs))

    # Sampling: keep a random subset of clusters.
    if 0.0 < args.sample_rate < 1.0:
        k = max(1, int(len(cluster_dirs) * args.sample_rate))
        cluster_dirs = random.sample(cluster_dirs, k)
        log.info("Sampling %.1f%% → %d clusters.", args.sample_rate * 100, len(cluster_dirs))

    log.info("S3 sentences root → %s", s3_root.resolve())
    log.info("Output            → %s", output_dir.resolve())

    client    = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = [
        process_cluster(
            semaphore, client, cdir, s3_root,
            args.model, output_dir,
            args.max_tokens, args.temperature,
            args.suppress_thinking,
        )
        for cdir in cluster_dirs
    ]
    all_results = await asyncio.gather(*tasks)

    # Write combined output
    all_results = sorted(all_results, key=lambda r: r.get("cluster", ""))
    combined_path = output_dir / "all_summaries.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    log.info("Done! Combined summaries saved to %s", combined_path.resolve())

    # Console preview
    print(f"\n{'='*70}")
    print("PREVIEW – first 3 summaries")
    print(f"{'='*70}")
    for r in all_results[:3]:
        print(f"\n[{r.get('cluster')}]  ({r.get('num_sentences')} sentences)")
        print(r.get("summary", ""))
        print("-" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ViMs news clusters via vLLM using pre-extracted important sentences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        default="data/ViMs/S3_summary",
        help="Root directory containing Cluster_001 … Cluster_N.",
    )
    parser.add_argument(
        "--s3_dir",
        default="data/ViMs/S3_summary",
        help=(
            "Root directory for pre-extracted important-sentence files. "
            "Expected layout: <s3_dir>/<cluster_name>/0.s3.txt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./summaries",
        help="Directory for per-cluster JSON summaries.",
    )
    parser.add_argument(
        "--base_url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--model",
        default="Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ",
        help="Model name as registered in vLLM.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max tokens for each summary.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Max concurrent API requests.",
    )
    parser.add_argument(
        "--clusters",
        nargs="*",
        help="Restrict to specific clusters, e.g. --clusters Cluster_001 Cluster_002.",
    )
    parser.add_argument(
        "--suppress_thinking",
        action="store_true",
        default=True,
        help="Strip <think>…</think> / <thought>…</thought> blocks from model output.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="Fraction of clusters to process, e.g. 0.2 for 20%%. Must be in (0, 1]. Default: 0.1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))