# %% [markdown]
# # Qwen3 Evaluation on VMLU + Vietnamese SQuAD
# 
# Evaluates **Qwen/Qwen3-8B** (text-only, no vision encoder) on two Vietnamese benchmarks:
# 
# | Dataset | Task | Metric |
# |---|---|---|
# | **VMLU Vi-MQA** | Multiple-choice QA — 58 subjects | Accuracy |
# | **Vietnamese SQuAD** | Extractive reading comprehension | Exact Match & F1 |
# 
# **Key settings:**
# - Model loaded in `bfloat16` with `device_map="auto"` 
# - `enable_thinking=False` — thinking chain suppressed, model outputs answers directly
# - Debug cells print the full prompt, raw model output, and extracted answer for the first 3 samples of each dataset

# %%
%%capture
!pip uninstall -y transformers
!pip install -q "transformers>=4.48.0"
!pip install -q accelerate datasets pillow
!pip install -q gguf>=0.10.0

# %%
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from accelerate import infer_auto_device_map, init_empty_weights

MODEL_ID = "Qwen/Qwen3.5-9B"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer

print("Loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",      # allow multi-GPU sharding
    trust_remote_code=True,
)

model.eval()

print("Loaded on devices:", model.hf_device_map)

# %%
import re

@torch.inference_mode()
def generate_response(
    messages: list,
    max_new_tokens: int = 64,
    debug: bool = False,
) -> tuple:
    """
    Run inference with thinking disabled.

    Returns
    -------
    prompt_text : the full rendered prompt string
    output_text : raw model output after the prompt
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # suppresses <think>...</think> blocks
    )

    inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = generated_ids[0][inputs.input_ids.shape[-1]:]
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    if debug:
        border = "=" * 65
        print(border)
        print("[FULL PROMPT SENT TO MODEL]")
        print(border)
        print(prompt_text)
        print(border)
        print("[RAW MODEL OUTPUT (before post-processing)]")
        print(repr(output_text))
        print(border)

    return prompt_text, output_text

print("generate_response() ready.")

# %%
from datasets import load_dataset

print("Loading VMLU from HuggingFace ...")
vmlu_dataset = load_dataset("tridm/VMLU", split="validation")
print(f"Loaded {len(vmlu_dataset)} VMLU samples.")
print("\nRaw first sample:")
print(vmlu_dataset[0])

# %%
def build_vmlu_messages(question: str, choices: list) -> list:
    choices_text = "\n".join(choices)
    content = (
        "Hãy trả lời câu hỏi trắc nghiệm sau bằng cách chọn đúng một đáp án (A, B, C, hoặc D).\n"
        "Chỉ trả lời bằng MỘT chữ cái duy nhất, không giải thích thêm.\n\n"
        f"Câu hỏi: {question}\n\n"
        f"{choices_text}\n\n"
        "Đáp án:"
    )
    return [{"role": "user", "content": content}]

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


print("VMLU helpers ready.")

# %%
tokenizer = processor.tokenizer
"""
print("  VMLU DEBUG PREVIEW (first 3 samples)")

for idx, sample in enumerate(vmlu_dataset.select(range(3))):
    print(f"\n>>> Sample {idx+1} / ID: {sample['id']}")
    print(f"    Question : {sample['question']}")
    print(f"    Choices  : {sample['choices']}")
    print(f"    Gold ans : {sample.get('answer', 'N/A')}")

    messages = build_vmlu_messages(sample["question"], sample["choices"])
    _, raw_output = generate_response(messages, max_new_tokens=8, debug=True)

    predicted = extract_mcq_answer(raw_output)
    print(f"    Extracted answer  : {predicted}")
    correct = (predicted == sample.get("answer", ""))
    print(f"    Correct?          : {'YES' if correct else 'NO'}")
    print("-" * 70)
"""

# %%
import pandas as pd
from tqdm.auto import tqdm

vmlu_results = []

for sample in tqdm(vmlu_dataset, desc="VMLU evaluation"):
    msgs = build_vmlu_messages(sample["question"], sample["choices"])
    _, raw = generate_response(msgs, max_new_tokens=8, debug=False)
    pred = extract_mcq_answer(raw)
    vmlu_results.append({
        "id":           sample["id"],
        "answer":       pred,
        "ground_truth": sample.get("answer", None),
        "raw_output":   raw,
    })

df_vmlu = pd.DataFrame(vmlu_results)
print(f"Collected {len(df_vmlu)} VMLU predictions.")

# %%
CATEGORY_MAP = {
    **{str(i).zfill(2): "STEM"           for i in range(1,  22)},
    **{str(i).zfill(2): "Social Science" for i in range(22, 32)},
    **{str(i).zfill(2): "Humanities"     for i in range(32, 50)},
    **{str(i).zfill(2): "Other"          for i in range(50, 59)},
}

if df_vmlu["ground_truth"].notna().any():
    df_vmlu["subject_id"] = df_vmlu["id"].str.split("-").str[0]
    df_vmlu["category"]   = df_vmlu["subject_id"].map(CATEGORY_MAP)
    df_vmlu["correct"]    = df_vmlu["answer"] == df_vmlu["ground_truth"]

    overall_acc = df_vmlu["correct"].mean() * 100
    print(f"\n{'='*55}")
    print(f"  VMLU Overall Accuracy : {overall_acc:.2f}%")
    print(f"{'='*55}")

    print("\nAccuracy by Category:")
    print(df_vmlu.groupby("category")["correct"].mean().mul(100).round(2).to_string())

    print("\nAccuracy by Subject ID:")
    print(df_vmlu.groupby("subject_id")["correct"].mean().mul(100).round(2).to_string())
else:
    print("No ground-truth labels in this split — accuracy not computed.")

# Save submission in VMLU-required format: id,answer
df_vmlu[["id", "answer"]].to_csv("vmlu_submission.csv", index=False, encoding="utf-8")
print("\nSaved: vmlu_submission.csv")
print(df_vmlu[["id", "answer"]].head(10).to_string(index=False))

# %%
from datasets import load_dataset
import random
random.seed(42)

print("Loading UIT-ViQuAD2.0 from HuggingFace...")

dataset = load_dataset("taidng/UIT-ViQuAD2.0", split="validation")

print(f"Loaded {len(dataset)} raw samples.")


def build_eval_set(dataset, total=1000, unanswerable_ratio=0.1):
    """Create evaluation set with controlled unanswerable ratio."""
    
    unans_target = int(total * unanswerable_ratio)
    ans_target = total - unans_target

    answerable = []
    unanswerable = []

    for item in dataset:
        answers = item.get("answers", {}).get("text", [])

        if answers:
            answerable.append(item)
        else:
            unanswerable.append(item)

    print(f"Answerable pool: {len(answerable)}")
    print(f"Unanswerable pool: {len(unanswerable)}")

    selected = (
        random.sample(answerable, ans_target) +
        random.sample(unanswerable, unans_target)
    )

    random.shuffle(selected)

    samples = []

    for i, item in enumerate(selected):
        samples.append({
            "id": item.get("id", f"sample_{i}"),
            "context": item["context"],
            "question": item["question"],
            "gold_answers": item.get("answers", {}).get("text", [])
        })

    return samples


squad_samples = build_eval_set(dataset, total=1000, unanswerable_ratio=0.1)

print(f"\nFinal evaluation set size: {len(squad_samples)}")

s0 = squad_samples[0]
print("\nFirst sample:")
print(f"ID          : {s0['id']}")
print(f"Context     : {s0['context'][:200]}...")
print(f"Question    : {s0['question']}")
print(f"Gold answers: {s0['gold_answers']}")

# %%
import string
import re
from collections import Counter


def build_squad_messages(context: str, question: str) -> list:
    """
    Prompt format adapted for UIT-ViQuAD2.0 which contains unanswerable questions.
    """
    
    ctx = context

    content = (
        "Dựa vào đoạn văn bản dưới đây, hãy trả lời câu hỏi.\n"
        "- Nếu câu trả lời tồn tại trong đoạn văn, hãy trích xuất một đoạn ngắn trực tiếp từ văn bản.\n"
        "- Nếu không có câu trả lời trong đoạn văn, hãy trả lời: 'không có câu trả lời'.\n\n"
        f"Đoạn văn:\n{ctx}\n\n"
        f"Câu hỏi: {question}\n\n"
        "Trả lời:"
    )

    return [{"role": "user", "content": content}]


# Common Vietnamese filler words that often cause EM mismatch
VI_STOPWORDS = {
    "trong", "ở", "vào", "tại", "là", "của", "được"
}


def normalize_text(text: str) -> str:
    """
    Aggressive normalization for Vietnamese QA:
    - lowercase
    - remove punctuation
    - remove filler stopwords
    - normalize whitespace
    """

    if text is None:
        return ""

    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in VI_STOPWORDS]

    return " ".join(tokens)


def is_no_answer(text: str) -> bool:
    """
    Detect model predictions meaning 'no answer'.
    """
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


def exact_match_score(prediction: str, gold_answers: list) -> int:
    """
    Exact match adapted for datasets with unanswerable questions.
    """

    # Case 1: unanswerable question
    if len(gold_answers) == 0:
        return int(is_no_answer(prediction))

    pred_norm = normalize_text(prediction)

    return int(
        any(pred_norm == normalize_text(g) for g in gold_answers)
    )


def f1_score(prediction: str, gold_answers: list) -> float:
    """
    Max token-level F1 across all gold answers.
    Handles unanswerable questions.
    """

    # Case 1: unanswerable question
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
        recall = n_common / len(gold_tokens)

        best = max(best, 2 * precision * recall / (precision + recall))

    return best


print("ViQuAD2.0 helpers ready (EM, F1, unanswerable support).")

# %%
print("  Vietnamese SQuAD DEBUG PREVIEW (first 3 samples)")

for idx, sample in enumerate(squad_samples[:3]):
    print(f"\n>>> Sample {idx+1} / ID: {sample['id']}")
    print(f"    Context (first 200 chars): {sample['context'][:200]}...")
    print(f"    Question    : {sample['question']}")
    print(f"    Gold answers: {sample['gold_answers']}")

    messages = build_squad_messages(sample["context"], sample["question"])
    _, raw_output = generate_response(messages, max_new_tokens=64, debug=True)

    em = exact_match_score(raw_output, sample["gold_answers"])
    f1 = f1_score(raw_output, sample["gold_answers"])
    print(f"    Prediction  : {raw_output.strip()!r}")
    print(f"EM={em}  F1={f1:.3f}")
    print("-" * 70)

# %% [markdown]
# 

# %%
from tqdm import tqdm
import pandas as pd
squad_results = []

for sample in tqdm(squad_samples, desc="Vi-SQuAD evaluation",dynamic_ncols=True,leave=False):
    msgs = build_squad_messages(sample["context"], sample["question"])
    _, raw = generate_response(msgs, max_new_tokens=64, debug=False)
    prediction = raw.strip()

    em = exact_match_score(prediction, sample["gold_answers"])
    f1 = f1_score(prediction, sample["gold_answers"])

    squad_results.append({
        "id":           sample["id"],
        "question":     sample["question"],
        "gold_answers": " | ".join(sample["gold_answers"]),
        "prediction":   prediction,
        "exact_match":  em,
        "f1":           round(f1, 4),
    })

df_squad = pd.DataFrame(squad_results)
print(f"Collected {len(df_squad)} SQuAD predictions.")

# %%
mean_em = df_squad["exact_match"].mean() * 100
mean_f1 = df_squad["f1"].mean() * 100

print(f"\n{'='*55}")
print(f"  Vietnamese SQuAD Results")
print(f"  Exact Match : {mean_em:.2f}%")
print(f"  F1 Score    : {mean_f1:.2f}%")
print(f"{'='*55}")

df_squad.to_csv("squad_predictions.csv", index=False, encoding="utf-8")
print("\nSaved: squad_predictions.csv")
print(df_squad[["id", "prediction", "exact_match", "f1"]].head(10).to_string(index=False))

# %% [markdown]
# ---
# # Final Summary

# %%
print("  EVALUATION SUMMARY — Qwen3-8B on Vietnamese Benchmarks")
# VMLU
print("\n[1] VMLU Vi-MQA — Multiple-Choice Accuracy")
if df_vmlu["ground_truth"].notna().any():
    print(f"    Overall : {df_vmlu['correct'].mean()*100:.2f}%")
    cat_acc = df_vmlu.groupby("category")["correct"].mean() * 100
    for cat, acc in cat_acc.items():
        print(f"    {cat:<20}: {acc:.2f}%")
else:
    print("    Labels not available — see vmlu_submission.csv")

# SQuAD
print("\n[2] Vietnamese SQuAD — Reading Comprehension")
print(f"    Exact Match : {mean_em:.2f}%")
print(f"    F1 Score    : {mean_f1:.2f}%")

print("\nOutput files:")
print("    vmlu_submission.csv   — VMLU predictions (id, answer) — ready to submit")
print("    squad_predictions.csv — SQuAD predictions with EM and F1")
print("#" * 65)


