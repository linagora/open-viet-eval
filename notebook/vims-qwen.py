# %% [markdown]
# # Qwen3.5-9B & Qwen3-8B — ViMs Vietnamese Multi-Document Summarization
# #
# **Dataset:** ViMs — 300 news clusters, ~1 945 articles
# #
# **Actual folder structure:**
# ```
# ViMs/
#   original/
#     Cluster_001/
#       original/
#         1.txt   ← article (Title:/Source:/Content: headers)
#         2.txt
#         ...
#   summary/
#     Cluster_001/
#       0.gold.txt  ← reference summary 1
#       1.gold.txt  ← reference summary 2
# ```
# #
# **Pipeline:**
# 1. Parse the ViMs folder structure into a clean per-cluster DataFrame
# 2. Load Qwen3.5-9B — generate summaries for every cluster
# 3. Free GPU → Load Qwen3-8B — generate summaries for every cluster
# 4. Compute ROUGE-1 / ROUGE-2 / ROUGE-L vs human reference for BOTH models
# 5. Free GPU → load SeaLLMs-v3-7B-Chat as LLM judge
# 6. Judge scores faithfulness, coverage, coherence (1–10 each) for BOTH models
# 7. Side-by-side final report comparing both models

# %%
%%capture
!pip install -q 'transformers @ git+https://github.com/huggingface/transformers.git@main'
!pip install -q accelerate gguf rouge-score

# %%
import os, re, gc
from pathlib import Path
from collections import defaultdict
import torch
import pandas as pd

DATASET_ROOT = Path("/kaggle/input/datasets/vtrnanh/sust-feature-data-new/ViMs")
all_files = sorted(DATASET_ROOT.rglob("*"))
print(f"Total paths: {len(all_files)}")

# %% [markdown]
# ## 1. Dataset Parsing Helpers

# %%
def find_dir(root: Path, name: str):
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None

ORIGINAL_DIR = find_dir(DATASET_ROOT, "original")
SUMMARY_DIR  = find_dir(DATASET_ROOT, "summary")
S3_DIR       = find_dir(DATASET_ROOT, "s3_summary")

print(f"Dataset root : {DATASET_ROOT}")
print(f"Original dir : {ORIGINAL_DIR}")
print(f"Summary dir  : {SUMMARY_DIR}")
print(f"S3 dir       : {S3_DIR}")

if ORIGINAL_DIR is None:
    raise RuntimeError("Could not find 'original' directory. Check DATASET_ROOT path above.")

clusters = sorted([d for d in ORIGINAL_DIR.iterdir() if d.is_dir()])
print(f"\nClusters found: {len(clusters)}")

sample_art_dir = clusters[0] / "original"
if not sample_art_dir.exists():
    sample_art_dir = clusters[0]
print("Files in Cluster_001/original/:",
      [f.name for f in sorted(sample_art_dir.iterdir()) if not f.name.startswith('.')])

# %%
def read_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc).strip()
        except Exception:
            continue
    return ""


def load_s3_important_sentences(s3_path: Path) -> list:
    text = read_txt(s3_path)
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip() == "1":
            sentences.append(parts[1].strip())
    return sentences


def load_cluster(cluster_dir: Path, summary_dir, s3_dir) -> dict:
    cid = cluster_dir.name
    if s3_dir is None:
        return None
    s3_cluster = s3_dir / cid
    s3_file = s3_cluster / "0.s3.txt"
    if not s3_file.exists():
        return None
    important_sents = load_s3_important_sentences(s3_file)
    if not important_sents:
        return None
    doc_parts = [f"[Câu {i}] {sent}" for i, sent in enumerate(important_sents, 1)]
    ref = ""
    if summary_dir:
        sd = summary_dir / cid
        gold_file = sd / "0.gold.txt"
        if gold_file.exists():
            ref = read_txt(gold_file)
    return {
        "cluster_id":  cid,
        "n_sentences": len(important_sents),
        "documents":   "\n".join(doc_parts),
        "ref_summary": ref,
    }


records, errors = [], []
for cd in clusters[:100]:
    try:
        rec = load_cluster(cd, SUMMARY_DIR, S3_DIR)
        if rec:
            records.append(rec)
    except Exception as e:
        errors.append((cd.name, str(e)))

df = pd.DataFrame(records)
print(f"Parsed {len(df)} clusters  ({len(errors)} errors)")
if errors:
    print("Errors:", errors[:5])
print(f"\nImportant sentences/cluster: min={df['n_sentences'].min()} "
      f"max={df['n_sentences'].max()} mean={df['n_sentences'].mean():.1f}")
print(f"Clusters with ref_summary: {(df['ref_summary']!='').sum()}")
df = df[:200]
df.head(3)

# %% [markdown]
# ## 2. Shared Inference Helpers

# %%
MAX_DOC_CHARS = 10_000

def build_summarization_messages(documents: str) -> list:
    truncated = documents[:MAX_DOC_CHARS]
    if len(documents) > MAX_DOC_CHARS:
        truncated += "\n[... (nội dung bị cắt bớt)]"
    system = (
        "Bạn PHẢI trả lời HOÀN TOÀN bằng tiếng Việt. "
        "Bạn là trợ lý AI chuyên tóm tắt tin tức tiếng Việt. "
        "Dưới đây là các câu quan trọng đã được chọn lọc từ nhiều bài báo "
        "cùng chủ đề. Hãy viết NGAY một bản tóm tắt tổng hợp ngắn gọn, "
        "trung thực, đầy đủ bằng tiếng Việt dựa trên các câu này. "
        "Không giải thích, không phân tích, chỉ viết bản tóm tắt."
    )
    user = (
        f"Các câu quan trọng:\n\n{truncated}\n\n"
        "Hãy viết ngay bản tóm tắt tổng hợp (4–6 câu, tiếng Việt) nêu bật "
        "những thông tin quan trọng nhất từ các câu trên. "
        "Bắt đầu tóm tắt ngay, không giải thích, không dùng tiếng Anh."
    )
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    lines = text.strip().splitlines()
    viet_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        has_viet_chars = bool(re.search(
            r'[àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ'
            r'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ]',
            stripped
        ))
        if has_viet_chars:
            viet_start = i
            break
    if viet_start is not None:
        text = "\n".join(lines[viet_start:])
    return text.strip()

print("Shared helpers ready.")

# %% [markdown]
# ## 3. Model A — Qwen3.5-9B

# %%
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_A_ID = "Qwen/Qwen3.5-9B"
print(f"Loading Model A: {MODEL_A_ID} ...")

processor_a = AutoProcessor.from_pretrained(MODEL_A_ID, trust_remote_code=True)
tokenizer_a = processor_a.tokenizer

model_a = AutoModelForImageTextToText.from_pretrained(
    MODEL_A_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model_a.eval()
print("Model A loaded on devices:", model_a.hf_device_map)


@torch.inference_mode()
def generate_response_a(messages: list, max_new_tokens: int = 512, debug: bool = False) -> tuple:
    prompt = tokenizer_a.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs  = tokenizer_a([prompt], return_tensors="pt").to(model_a.device)
    gen_ids = model_a.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer_a.eos_token_id,
    )
    new_ids = gen_ids[0][inputs.input_ids.shape[-1]:]
    raw_out = tokenizer_a.decode(new_ids, skip_special_tokens=True)
    out = _strip_thinking(raw_out)
    if debug:
        sep = "=" * 65
        print(sep)
        print("[FULL PROMPT SENT TO MODEL A]")
        print(sep)
        print(prompt)
        print(sep, "\n[RAW MODEL A OUTPUT]\n", sep)
        print(repr(raw_out[:600]))
        print(sep, "\n[CLEANED OUTPUT]\n", sep)
        print(repr(out))
        print(sep)
    return prompt, out

# %% [markdown]
# ### 3a. Debug Preview — Model A (First 3 Clusters)

# %%
print("#" * 70)
print("  MODEL A (Qwen3.5-9B) — DEBUG PREVIEW (first 3 clusters)")
print("#" * 70)

for i, row in df.head(3).iterrows():
    msgs = build_summarization_messages(row["documents"])
    _, prediction = generate_response_a(msgs, max_new_tokens=512, debug=True)
    print(f"\n>>> Cluster {i+1}: {row['cluster_id']}")
    print(f"    ref_summary : {row['ref_summary'][:200]}...")
    print(f"    prediction  : {prediction[:200]}...")
    print("-" * 70)

# %% [markdown]
# ### 3b. Generate Summaries for All Clusters — Model A

# %%
from tqdm.auto import tqdm

predictions_a = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="[Model A] Generating summaries"):
    msgs = build_summarization_messages(row["documents"])
    _, pred = generate_response_a(msgs, max_new_tokens=512, debug=False)
    predictions_a.append(pred)

df["prediction_a"] = predictions_a
print(f"[Model A] Generated {len(df)} summaries.")

# %% [markdown]
# ## 4. Free Model A → Load Model B — Qwen3-8B

# %%
del model_a
del tokenizer_a
del processor_a

import gc
gc.collect()

torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"Current VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Ready to load Model B.")

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_B_ID = "Qwen/Qwen3-8B"
print(f"Loading Model B: {MODEL_B_ID} ...")

tokenizer_b = AutoTokenizer.from_pretrained(MODEL_B_ID, trust_remote_code=True)
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_B_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model_b.eval()
print(f"Model B loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


@torch.inference_mode()
def generate_response_b(messages: list, max_new_tokens: int = 512, debug: bool = False) -> tuple:
    prompt = tokenizer_b.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs  = tokenizer_b([prompt], return_tensors="pt").to(model_b.device)
    gen_ids = model_b.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer_b.eos_token_id,
    )
    new_ids = gen_ids[0][inputs.input_ids.shape[-1]:]
    raw_out = tokenizer_b.decode(new_ids, skip_special_tokens=True)
    out = _strip_thinking(raw_out)
    if debug:
        sep = "=" * 65
        print(sep)
        print("[FULL PROMPT SENT TO MODEL B]")
        print(sep)
        print(prompt)
        print(sep, "\n[RAW MODEL B OUTPUT]\n", sep)
        print(repr(raw_out[:600]))
        print(sep, "\n[CLEANED OUTPUT]\n", sep)
        print(repr(out))
        print(sep)
    return prompt, out

# %% [markdown]
# ### 4a. Debug Preview — Model B (First 3 Clusters)

# %%
print("#" * 70)
print("  MODEL B (Qwen3-8B) — DEBUG PREVIEW (first 3 clusters)")
print("#" * 70)

for i, row in df.head(3).iterrows():
    msgs = build_summarization_messages(row["documents"])
    _, prediction = generate_response_b(msgs, max_new_tokens=512, debug=True)
    print(f"\n>>> Cluster {i+1}: {row['cluster_id']}")
    print(f"    ref_summary : {row['ref_summary'][:200]}...")
    print(f"    prediction  : {prediction[:200]}...")
    print("-" * 70)

# %% [markdown]
# ### 4b. Generate Summaries for All Clusters — Model B

# %%
predictions_b = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="[Model B] Generating summaries"):
    msgs = build_summarization_messages(row["documents"])
    _, pred = generate_response_b(msgs, max_new_tokens=512, debug=False)
    predictions_b.append(pred)

df["prediction_b"] = predictions_b
print(f"[Model B] Generated {len(df)} summaries.")

# %% [markdown]
# ## 5. ROUGE Evaluation — Both Models

# %%
from rouge_score import rouge_scorer as _rs

_scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def rouge_single(prediction: str, reference: str) -> dict:
    if not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = _scorer.score(reference, prediction)
    return {m: s[m].fmeasure for m in ["rouge1", "rouge2", "rougeL"]}


for model_label, pred_col in [("a", "prediction_a"), ("b", "prediction_b")]:
    r1_list, r2_list, rL_list = [], [], []
    for _, row in df.iterrows():
        sc = rouge_single(row[pred_col], row["ref_summary"])
        r1_list.append(sc["rouge1"])
        r2_list.append(sc["rouge2"])
        rL_list.append(sc["rougeL"])
    df[f"rouge1_{model_label}"] = r1_list
    df[f"rouge2_{model_label}"] = r2_list
    df[f"rougeL_{model_label}"] = rL_list
    print(f"\n{'='*55}")
    print(f"  ROUGE — {'Qwen3.5-9B' if model_label=='a' else 'Qwen3-8B'} (vs annotator 0 reference)")
    print(f"{'='*55}")
    print(f"  ROUGE-1 : {sum(r1_list)/len(r1_list)*100:.2f}%")
    print(f"  ROUGE-2 : {sum(r2_list)/len(r2_list)*100:.2f}%")
    print(f"  ROUGE-L : {sum(rL_list)/len(rL_list)*100:.2f}%")
    print(f"{'='*55}")

# %% [markdown]
# ## 6. Free Model B → Load SeaLLMs-v3-7B-Chat as Judge

# %%
del model_b
del tokenizer_b

import gc
gc.collect()

torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"Current VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Ready to load SeaLLMs judge.")

# %%
JUDGE_MODEL = "SeaLLMs/SeaLLMs-v3-7B-Chat"

print(f"Loading judge: {JUDGE_MODEL} in bfloat16 ...")
judge_tokenizer = AutoTokenizer.from_pretrained(
    JUDGE_MODEL, trust_remote_code=True, padding_side="left",
)
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
judge_model.eval()
print(f"Judge ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# ## 7. Judge Helpers

# %%
import json as _json

JUDGE_SYSTEM = (
    "Bạn là chuyên gia đánh giá chất lượng tóm tắt tin tức tiếng Việt.\n"
    "Đánh giá bản tóm tắt của mô hình theo 3 tiêu chí:\n"
    "1. Faithfulness: tóm tắt có bịa đặt thông tin không có trong bài gốc không?\n"
    "2. Coverage: tóm tắt có bao quát các ý chính từ tất cả các bài báo không?\n"
    "3. Coherence: tóm tắt có rõ ràng, trôi chảy, dễ đọc không?\n\n"
    "Trả lời ĐÚNG định dạng JSON, KHÔNG thêm bất kỳ văn bản nào khác:\n"
    '{"score": <1-10>, "faithfulness": <1-10>, "coverage": <1-10>, '
    '"coherence": <1-10>, "rationale": "<tối đa 2 câu tiếng Việt>"}'
)


def build_judge_messages(documents: str, reference: str, prediction: str) -> list:
    user = (
        f"### Bài báo gốc (trích đoạn 2000 ký tự):\n{documents[:2000]}\n\n"
        f"### Tóm tắt tham chiếu:\n{reference[:500]}\n\n"
        f"### Tóm tắt của mô hình:\n{prediction[:500]}\n\n"
        "Đánh giá và trả về JSON."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user}]


def parse_judge_output(raw: str) -> dict:
    keys = ("score", "faithfulness", "coverage", "coherence", "rationale")
    raw  = raw.strip()
    try:
        obj = _json.loads(raw)
        return {k: obj.get(k, 5) for k in keys}
    except Exception:
        pass
    m = re.search(r'\{[^{}]*"score"\s*:\s*\d+[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            return {k: obj.get(k, 5) for k in keys}
        except Exception:
            pass
    sm = re.search(r'"score"\s*:\s*(\d+)', raw)
    sc = min(max(int(sm.group(1)), 1), 10) if sm else 5
    return {"score": sc, "faithfulness": 5, "coverage": 5,
            "coherence": 5, "rationale": "(parse fallback)"}


@torch.inference_mode()
def judge_score(documents: str, reference: str, prediction: str, debug: bool = False) -> dict:
    ref  = reference if reference.strip() else "(không có tóm tắt tham chiếu)"
    msgs = build_judge_messages(documents, ref, prediction)

    # SeaLLMs uses standard chat template; try with enable_thinking=False first,
    # fall back gracefully if the model doesn't support that kwarg.
    try:
        prompt = judge_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        prompt = judge_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )

    inputs  = judge_tokenizer([prompt], return_tensors="pt").to(judge_model.device)
    gen_ids = judge_model.generate(
        **inputs, max_new_tokens=200,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=judge_tokenizer.eos_token_id,
    )
    raw = judge_tokenizer.decode(
        gen_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
    )
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if debug:
        sep = "=" * 65
        print(sep, "\n[JUDGE PROMPT (first 1500 chars)]\n", sep)
        print(prompt[:1500], "...")
        print(sep, "\n[JUDGE RAW OUTPUT]\n", sep)
        print(repr(raw))
        print(sep)

    result = parse_judge_output(raw)
    result["raw_output"] = raw
    return result


print("Judge helpers ready.")

# %% [markdown]
# ## 8. Debug Preview — First 3 Judge Evaluations (Both Models)

# %%
for model_label, pred_col, model_name in [
    ("a", "prediction_a", "Qwen3.5-9B"),
    ("b", "prediction_b", "Qwen3-8B"),
]:
    print("#" * 70)
    print(f"  SeaLLMs JUDGE — {model_name} — DEBUG PREVIEW (first 3 clusters)")
    print("#" * 70)
    for i, row in df.head(3).iterrows():
        ref    = row["ref_summary"]
        result = judge_score(row["documents"], ref, row[pred_col], debug=True)
        print(f"\n>>> Cluster {i+1}: {row['cluster_id']}  [{model_name}]")
        print(f"    Reference  : {ref[:150]}...")
        print(f"    Prediction : {row[pred_col][:150]}...")
        print(f"    Score      : {result['score']}/10  "
              f"(faith={result['faithfulness']}, cov={result['coverage']}, coh={result['coherence']})")
        print(f"    Rationale  : {result['rationale']}")
        print("-" * 70)

# %% [markdown]
# ## 9. Full Judge Evaluation — Both Models

# %%
for model_label, pred_col, model_name in [
    ("a", "prediction_a", "Qwen3.5-9B"),
    ("b", "prediction_b", "Qwen3-8B"),
]:
    scores, faiths, covs, cohs, rationales, raws = [], [], [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[Judge] Scoring {model_name}"):
        ref = row["ref_summary"]
        r   = judge_score(row["documents"], ref, row[pred_col], debug=False)
        scores.append(r["score"])
        faiths.append(r["faithfulness"])
        covs.append(r["coverage"])
        cohs.append(r["coherence"])
        rationales.append(r["rationale"])
        raws.append(r["raw_output"])

    df[f"judge_score_{model_label}"]        = scores
    df[f"judge_faithfulness_{model_label}"] = faiths
    df[f"judge_coverage_{model_label}"]     = covs
    df[f"judge_coherence_{model_label}"]    = cohs
    df[f"judge_rationale_{model_label}"]    = rationales
    df[f"judge_raw_{model_label}"]          = raws
    print(f"[{model_name}] Judging complete. {len(df)} clusters scored.")

# %% [markdown]
# ## 10. Final Report — Side-by-Side Comparison

# %%
def model_stats(label: str, name: str):
    return {
        "model":         name,
        "rouge1":        df[f"rouge1_{label}"].mean() * 100,
        "rouge2":        df[f"rouge2_{label}"].mean() * 100,
        "rougeL":        df[f"rougeL_{label}"].mean() * 100,
        "judge_score":   df[f"judge_score_{label}"].mean(),
        "faithfulness":  df[f"judge_faithfulness_{label}"].mean(),
        "coverage":      df[f"judge_coverage_{label}"].mean(),
        "coherence":     df[f"judge_coherence_{label}"].mean(),
    }

stats = [
    model_stats("a", "Qwen3.5-9B"),
    model_stats("b", "Qwen3-8B"),
]

print("\n" + "#" * 75)
print("  EVALUATION SUMMARY — SIDE BY SIDE")
print("  Judge   : SeaLLMs-v3-7B-Chat")
print("  Dataset : ViMs — Vietnamese Multi-Document Summarization")
print(f"  Clusters: {len(df)}")
print("#" * 75)

header = f"  {'Metric':<22} {'Qwen3.5-9B':>14} {'Qwen3-8B':>14}  {'Winner':>10}"
print(f"\n{header}")
print("  " + "-" * 65)

metrics = [
    ("ROUGE-1 (%)",      "rouge1"),
    ("ROUGE-2 (%)",      "rouge2"),
    ("ROUGE-L (%)",      "rougeL"),
    ("Judge Score",      "judge_score"),
    ("Faithfulness",     "faithfulness"),
    ("Coverage",         "coverage"),
    ("Coherence",        "coherence"),
]

for label, key in metrics:
    va, vb = stats[0][key], stats[1][key]
    winner = "Qwen3.5-9B" if va > vb else ("Qwen3-8B" if vb > va else "Tie")
    fmt = ".2f"
    print(f"  {label:<22} {va:>14{fmt}} {vb:>14{fmt}}  {winner:>10}")

print("\n" + "#" * 75)

# Per-model score distributions
for label, name in [("a", "Qwen3.5-9B"), ("b", "Qwen3-8B")]:
    print(f"\n  Score distribution — {name}:")
    for sc, cnt in df[f"judge_score_{label}"].value_counts().sort_index().items():
        print(f"    {sc:2d}/10 : {'█' * int(cnt)} ({cnt})")

# 5 worst clusters per model
for label, name in [("a", "Qwen3.5-9B"), ("b", "Qwen3-8B")]:
    print(f"\n  5 worst clusters (by judge score) — {name}:")
    for _, r in df.nsmallest(5, f"judge_score_{label}").iterrows():
        print(f"    {r['cluster_id']}  "
              f"score={r[f'judge_score_{label}']}  "
              f"rouge1={r[f'rouge1_{label}']*100:.1f}%")
        print(f"      rationale : {r[f'judge_rationale_{label}']}")
        print(f"      prediction: {str(r[f'prediction_{label}'])[:120]}...")

# %% [markdown]
# ## 11. Save Results

# %%
out_cols = [
    "cluster_id", "n_sentences", "ref_summary",
    # Predictions
    "prediction_a", "prediction_b",
    # ROUGE
    "rouge1_a", "rouge2_a", "rougeL_a",
    "rouge1_b", "rouge2_b", "rougeL_b",
    # Judge scores
    "judge_score_a", "judge_faithfulness_a", "judge_coverage_a",
    "judge_coherence_a", "judge_rationale_a",
    "judge_score_b", "judge_faithfulness_b", "judge_coverage_b",
    "judge_coherence_b", "judge_rationale_b",
]
df[out_cols].to_csv("vims_dual_model_results.csv", index=False, encoding="utf-8")
print("Saved: vims_dual_model_results.csv")
print("#" * 75)


