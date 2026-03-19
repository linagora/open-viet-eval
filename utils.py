"""Shared utilities: text cleaning, file I/O, GPU memory management."""

import gc
import re
from pathlib import Path

import torch


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks and skip to Vietnamese content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lines = text.strip().splitlines()
    viet_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        has_viet = bool(re.search(
            r'[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ'
            r'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ]',
            stripped,
        ))
        if has_viet:
            viet_start = i
            break
    if viet_start is not None:
        text = "\n".join(lines[viet_start:])
    return text.strip()


def read_txt(path: Path) -> str:
    """Read a text file trying multiple encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc).strip()
        except Exception:
            continue
    return ""


def free_gpu_memory(*objects):
    """Delete objects, run GC, and clear CUDA cache."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_block(title: str, text: str):
    """Pretty-print a labelled text block."""
    hr = "═" * 65
    print(f"\n{hr}")
    print(f"  {title}")
    print(hr)
    print(text)
