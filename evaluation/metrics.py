"""Automatic evaluation metrics: ROUGE, Exact Match, F1."""

import string
from collections import Counter

from rouge_score import rouge_scorer as _rs
# ── ROUGE ────────────────────────────────────────────────────────────────────

_scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def rouge_single(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1/2/L F-measure for a single (prediction, reference) pair."""
    if not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = _scorer.score(reference, prediction)
    return {m: s[m].fmeasure for m in ["rouge1", "rouge2", "rougeL"]}

# ── Vietnamese text normalisation ────────────────────────────────────────────

VI_STOPWORDS = {"trong", "ở", "vào", "tại", "là", "của", "được"}

def normalize_text(text: str) -> str:
    """Aggressive normalisation for Vietnamese QA: lowercase, strip punctuation, remove fillers."""
    if text is None:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in VI_STOPWORDS]
    return " ".join(tokens)

def is_no_answer(text: str) -> bool:
    """Detect model predictions meaning 'no answer'."""
    text = normalize_text(text)
    return any(
        phrase in text
        for phrase in [
            "không có câu trả lời",
            "không có thông tin",
            "không được đề cập",
            "không có",
        ]
    )

# ── Exact Match & F1 ────────────────────────────────────────────────────────

def exact_match_score(prediction: str, gold_answers: list) -> int:
    """Exact match, supports unanswerable questions (empty gold_answers)."""
    if len(gold_answers) == 0:
        return int(is_no_answer(prediction))
    pred_norm = normalize_text(prediction)
    return int(any(pred_norm == normalize_text(g) for g in gold_answers))

def f1_score(prediction: str, gold_answers: list) -> float:
    """Max token-level F1 across all gold answers. Handles unanswerable questions."""
    if len(gold_answers) == 0:
        return 1.0 if is_no_answer(prediction) else 0.0

    pred_tokens = normalize_text(prediction).split()
    if len(pred_tokens) == 0:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall    = n_common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
