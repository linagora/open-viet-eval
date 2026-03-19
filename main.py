"""
Vietnamese LLM Benchmark — CLI Entry Point
===========================================

Run benchmarks for Qwen3.5-9B, Qwen3-8B, or Unicorn-VL-R3
on VMLU, ViQuAD, ViMs, or VTSNLP datasets.

Usage
-----
    python main.py --model qwen35 --dataset vmlu
    python main.py --model unicorn --dataset vims --max-samples 10 --skip-judge
    python main.py --model qwen3 --dataset vtsnlp --debug
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# Ensure project root is on sys.path so local packages resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import get_model
from models.unicorn import UnicornModel
from evaluation.judge import JudgeModel


# ═════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Vietnamese LLM Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",   required=True, choices=["qwen35", "qwen3", "unicorn"],
                   help="Model to benchmark")
    p.add_argument("--dataset", required=True, choices=["vmlu", "viquad", "vims", "vtsnlp"],
                   help="Dataset to evaluate on")
    p.add_argument("--dataset-path", default=None,
                   help="Override default dataset path (for ViMs: folder root, for VTSNLP: CSV path)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit the number of samples (for quick testing)")
    p.add_argument("--output-dir", default="./results",
                   help="Directory to save result CSVs (default: ./results)")
    p.add_argument("--skip-judge", action="store_true",
                   help="Skip SeaLLMs judge evaluation")
    p.add_argument("--debug", action="store_true",
                   help="Print detailed debug output for first 3 samples")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Dataset-specific benchmark runners
# ═════════════════════════════════════════════════════════════════════════════

def run_vmlu(model, args):
    """Benchmark on VMLU (multiple-choice QA)."""
    from datasets.vmlu import load_vmlu, build_vmlu_messages, build_vmlu_prompt_unicorn, extract_mcq_answer, CATEGORY_MAP

    samples = load_vmlu(max_samples=args.max_samples)
    is_unicorn = isinstance(model, UnicornModel)

    results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[{model.model_id}] VMLU")):
        if is_unicorn:
            prompt_content = build_vmlu_prompt_unicorn(sample["question"], sample["choices"])
            raw = model.generate(prompt_content, max_new_tokens=8, debug=(args.debug and i < 3))
        else:
            msgs = build_vmlu_messages(sample["question"], sample["choices"])
            raw = model.generate(msgs, max_new_tokens=8, debug=(args.debug and i < 3))

        pred = extract_mcq_answer(raw)
        results.append({
            "id":           sample["id"],
            "answer":       pred,
            "ground_truth": sample.get("answer"),
            "raw_output":   raw,
        })

    df = pd.DataFrame(results)

    # ── Accuracy report ──
    if df["ground_truth"].notna().any():
        df["subject_id"] = df["id"].str.split("-").str[0]
        df["category"]   = df["subject_id"].map(CATEGORY_MAP)
        df["correct"]    = df["answer"] == df["ground_truth"]
        overall = df["correct"].mean() * 100

        print(f"\n{'='*55}")
        print(f"  VMLU Overall Accuracy: {overall:.2f}%")
        print(f"{'='*55}")
        print("\nAccuracy by Category:")
        print(df.groupby("category")["correct"].mean().mul(100).round(2).to_string())
    else:
        print("No ground-truth labels — accuracy not computed.")

    return df


def run_viquad(model, args):
    """Benchmark on ViQuAD / Vietnamese SQuAD (extractive QA)."""
    from datasets.viquad import load_viquad, build_squad_messages, build_squad_prompt_unicorn
    from evaluation.metrics import exact_match_score, f1_score

    samples = load_viquad(max_samples=args.max_samples)
    is_unicorn = isinstance(model, UnicornModel)

    results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[{model.model_id}] ViQuAD")):
        if is_unicorn:
            prompt = build_squad_prompt_unicorn(sample["context"], sample["question"])
            raw = model.generate_raw(prompt, max_new_tokens=64, debug=(args.debug and i < 3))
        else:
            msgs = build_squad_messages(sample["context"], sample["question"])
            raw = model.generate(msgs, max_new_tokens=64, debug=(args.debug and i < 3))

        prediction = raw.strip()
        em = exact_match_score(prediction, sample["gold_answers"])
        f1 = f1_score(prediction, sample["gold_answers"])

        results.append({
            "id":           sample["id"],
            "question":     sample["question"],
            "gold_answers": " | ".join(sample["gold_answers"]),
            "prediction":   prediction,
            "exact_match":  em,
            "f1":           round(f1, 4),
        })

    df = pd.DataFrame(results)
    mean_em = df["exact_match"].mean() * 100
    mean_f1 = df["f1"].mean() * 100

    print(f"\n{'='*55}")
    print(f"  ViQuAD Results — {model.model_id}")
    print(f"  Exact Match : {mean_em:.2f}%")
    print(f"  F1 Score    : {mean_f1:.2f}%")
    print(f"{'='*55}")

    return df


def run_vims(model, args):
    """Benchmark on ViMs (multi-document summarisation)."""
    from datasets.vims import load_vims, build_summarization_messages, build_summarization_prompt_unicorn
    from evaluation.metrics import rouge_single

    dataset_path = args.dataset_path or "/kaggle/input/datasets/vtrnanh/sust-feature-data-new/ViMs"
    df = load_vims(dataset_root=dataset_path, max_samples=args.max_samples)
    is_unicorn = isinstance(model, UnicornModel)

    predictions = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"[{model.model_id}] ViMs")):
        if is_unicorn:
            prompt = build_summarization_prompt_unicorn(row["documents"])
            pred = model.generate_raw(prompt, max_new_tokens=512, debug=(args.debug and i < 3))
        else:
            msgs = build_summarization_messages(row["documents"])
            pred = model.generate(msgs, max_new_tokens=512, debug=(args.debug and i < 3))
        predictions.append(pred)

    df["prediction"] = predictions

    # ── ROUGE ──
    r1, r2, rL = [], [], []
    for _, row in df.iterrows():
        sc = rouge_single(row["prediction"], row["ref_summary"])
        r1.append(sc["rouge1"])
        r2.append(sc["rouge2"])
        rL.append(sc["rougeL"])

    df["rouge1"], df["rouge2"], df["rougeL"] = r1, r2, rL

    print(f"\n{'='*55}")
    print(f"  ROUGE — {model.model_id}")
    print(f"  ROUGE-1 : {sum(r1)/len(r1)*100:.2f}%")
    print(f"  ROUGE-2 : {sum(r2)/len(r2)*100:.2f}%")
    print(f"  ROUGE-L : {sum(rL)/len(rL)*100:.2f}%")
    print(f"{'='*55}")

    return df


def run_vtsnlp(model, args):
    """Benchmark on VTSNLP (instruction-following)."""
    from datasets.vtsnlp import load_vtsnlp, build_instruct_messages, build_instruct_prompt_unicorn

    csv_path = args.dataset_path or "/kaggle/input/datasets/qnfuioyhgvqpwo/sample-vtsnlp-instruct-dataset/sampled_instruct_general_dataset.csv"
    samples = load_vtsnlp(csv_path=csv_path, max_samples=args.max_samples)
    is_unicorn = isinstance(model, UnicornModel)

    results = []
    for i, sample in enumerate(tqdm(samples, desc=f"[{model.model_id}] VTSNLP")):
        max_len = int(sample.get("max_len", 512))
        if is_unicorn:
            prompt = build_instruct_prompt_unicorn(sample["instruct"], sample["input"])
            pred = model.generate_raw(prompt, max_new_tokens=max_len, debug=(args.debug and i < 3))
        else:
            msgs = build_instruct_messages(sample["instruct"], sample["input"])
            pred = model.generate(msgs, max_new_tokens=max_len, debug=(args.debug and i < 3))

        results.append({
            "idx":        i,
            "category":   sample["category"],
            "instruct":   sample["instruct"],
            "input":      sample["input"],
            "reference":  sample["output"],
            "prediction": pred,
            "model":      model.model_id,
        })

    df = pd.DataFrame(results)
    print(f"[{model.model_id}] Generated {len(df)} VTSNLP predictions.")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Judge scoring
# ═════════════════════════════════════════════════════════════════════════════

def run_judge_scoring(df: pd.DataFrame, dataset: str, args):
    """Load the judge model, score all predictions, and return updated DataFrame."""
    judge = JudgeModel()
    judge.load()

    scores, rationales, raws = [], [], []
    extra_cols = {}

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="[Judge] Scoring")):
        if dataset == "vims":
            ref = row["ref_summary"] if str(row["ref_summary"]).strip() else "(không có tóm tắt tham chiếu)"
            msgs = judge.build_summarisation_messages(row["documents"], ref, row["prediction"])
        elif dataset == "vtsnlp":
            msgs = judge.build_instruct_messages(
                row["instruct"], row["input"], row["reference"], row["prediction"]
            )
        else:
            # VMLU/ViQuAD: judge not applicable
            break

        result = judge.score(msgs, debug=(args.debug and i < 3))
        scores.append(result.get("score", 5))
        rationales.append(result.get("rationale", ""))
        raws.append(result.get("raw_output", ""))

        # Collect summarisation-specific scores
        if dataset == "vims":
            extra_cols.setdefault("faithfulness", []).append(result.get("faithfulness", 5))
            extra_cols.setdefault("coverage", []).append(result.get("coverage", 5))
            extra_cols.setdefault("coherence", []).append(result.get("coherence", 5))

    judge.unload()

    if scores:
        df = df.copy()
        df["judge_score"]     = scores
        df["judge_rationale"] = rationales
        df["judge_raw"]       = raws
        for col, vals in extra_cols.items():
            df[f"judge_{col}"] = vals

        print(f"\n  Judge Score (mean): {sum(scores)/len(scores):.2f}/10")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "vmlu":   run_vmlu,
    "viquad": run_viquad,
    "vims":   run_vims,
    "vtsnlp": run_vtsnlp,
}

JUDGE_DATASETS = {"vims", "vtsnlp"}  # datasets that support judge scoring


def main():
    args = parse_args()

    # ── Load model ──
    model = get_model(args.model)
    model.load()

    # ── Run benchmark ──
    runner = RUNNERS[args.dataset]
    df = runner(model, args)

    # ── Unload model ──
    model.unload()

    # ── Judge scoring (optional) ──
    if not args.skip_judge and args.dataset in JUDGE_DATASETS:
        df = run_judge_scoring(df, args.dataset, args)

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model
    out_path = Path(args.output_dir) / f"{args.dataset}_{model_short}_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("#" * 65)


if __name__ == "__main__":
    main()
