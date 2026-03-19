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
df = df[:100]

# %% [markdown]
# ## 2. Shared Inference Helpers

# %%
from tqdm.auto import tqdm

MAX_DOC_CHARS = 10_000

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
# ## 3. Load Unicorn

# %%
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

UNICORN_MODEL_ID = "unicorn-team/Unicorn-VL-R3"

# ── One-shot example for prompt grounding ───────────────────────────────────
_ONESHOT_CONTEXT = (
    "Các câu quan trọng: "
    "[Câu 1] Ngày 20/11/2023, Tòa án nhân dân cấp cao đã mở phiên tòa phúc thẩm xét xử vụ án hình sự. "
    "[Câu 2] Bị cáo Trần Văn B bị truy tố về hành vi lạm dụng tín nhiệm chiếm đoạt tài sản. "
    "[Câu 3] Theo hồ sơ vụ án, B đã lợi dụng chức vụ quản lý để bỏ ngoài sổ sách nhiều khoản thu của doanh nghiệp. "
    "[Câu 4] Cơ quan chức năng xác định tổng số tiền công ty bị thất thoát do hành vi này là hơn 12 tỷ đồng. "
    "[Câu 5] Bị cáo khai nhận tại cơ quan điều tra rằng đã dùng số tiền này để đầu tư chứng khoán nhưng thua lỗ nặng. "
    "[Câu 6] Đại diện hợp pháp của công ty đã nộp đơn yêu cầu bị cáo phải bồi thường toàn bộ thiệt hại đã gây ra. "
    "[Câu 7] Tại phiên phúc thẩm, bị cáo đã nộp lại một phần tiền khắc phục hậu quả nhằm xin giảm nhẹ hình phạt. "
    "[Câu 8] Tuy nhiên, Hội đồng xét xử nhận định hành vi của bị cáo là đặc biệt nghiêm trọng, có tổ chức. "
    "[Câu 9] Kết thúc phiên làm việc, Tòa quyết định y án sơ thẩm, tuyên phạt Trần Văn B 12 năm tù giam."
)

_ONESHOT_A = (
    "Ngày 20/11/2023, Tòa án nhân dân cấp cao mở phiên phúc thẩm xét xử Trần Văn B về tội "
    "lạm dụng tín nhiệm chiếm đoạt tài sản.\n"
    "Lợi dụng chức vụ quản lý, bị cáo đã để ngoài sổ sách các khoản thu, chiếm đoạt hơn "
    "12 tỷ đồng của doanh nghiệp để đầu tư chứng khoán.\n"
    "Phía công ty đã yêu cầu bị cáo phải bồi thường toàn bộ số tiền thiệt hại nói trên.\n"
    "Dù bị cáo đã nộp một phần tiền khắc phục hậu quả, Hội đồng xét xử vẫn đánh giá hành vi "
    "đặc biệt nghiêm trọng nên quyết định y án sơ thẩm, phạt B 12 năm tù giam."
)

def build_unicorn_summarization_prompt(documents: str) -> str:
    """
    Build a raw <|im_start|> prompt for the Unicorn model.
    Mirrors build_summarization_messages() but uses Unicorn's native chat format
    instead of apply_chat_template, with a one-shot example for style grounding.
    """
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
    user_content = (
        f"Các câu quan trọng:\n\n{truncated}\n\n"
        "Hãy viết ngay bản tóm tắt tổng hợp (4–6 câu, tiếng Việt) nêu bật "
        "những thông tin quan trọng nhất từ các câu trên. "
        "Bắt đầu tóm tắt ngay, không giải thích, không dùng tiếng Anh."
    )

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        # ── one-shot turn ────────────────────────────────────────────────────
        "<|im_start|>user\n"
        f"Các câu quan trọng mẫu:\n\n{_ONESHOT_CONTEXT}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"Mẫu tóm tắt: {_ONESHOT_A}<|im_end|>\n"
        # ── actual query ─────────────────────────────────────────────────────
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


print(f"Loading Unicorn: {UNICORN_MODEL_ID} ...")
processor = AutoProcessor.from_pretrained(UNICORN_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(UNICORN_MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    UNICORN_MODEL_ID,
    dtype=torch.bfloat16,   # Unicorn uses `dtype`, NOT `torch_dtype`
    device_map="auto",
)
model.eval()
print(f"Unicorn loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


@torch.inference_mode()
def generate_unicorn_response(prompt: str, max_new_tokens: int = 512) -> str:
    """Generate and return cleaned output from the loaded Unicorn model."""
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = gen_ids[0][inputs.input_ids.shape[-1]:]
    raw_out = tokenizer.decode(new_ids, skip_special_tokens=True)
    return _strip_thinking(raw_out)

# %% [markdown]
# ## 4. Sanity Check

# %%
SANITY_IDX = 0   # change to inspect a different cluster

_HR = "═" * 65

def _print_block(title: str, text: str):
    print(f"\n{_HR}")
    print(f"  {title}")
    print(_HR)
    print(text)

_row            = df.iloc[SANITY_IDX]
_sanity_prompt  = build_unicorn_summarization_prompt(_row["documents"])
_sanity_ref     = str(_row["ref_summary"]).strip()
_sanity_answer  = generate_unicorn_response(_sanity_prompt)

_print_block("FULL PROMPT", _sanity_prompt)
_print_block("EXPECTED ANSWER (ref_summary)", _sanity_ref)
_print_block("MODEL ANSWER", _sanity_answer)
print(f"\n{_HR}\n")

# %% [markdown]
# ## 5. Generate Summaries — Unicorn

# %%
predictions = []
for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[Unicorn] Generating summaries"):
    prompt = build_unicorn_summarization_prompt(row["documents"])
    pred   = generate_unicorn_response(prompt, max_new_tokens=512)
    predictions.append(pred)

df["prediction"] = predictions
print(f"[Unicorn] Generated {len(df)} summaries.")

# %% [markdown]
# ## 6. Free Unicorn → Load Judge

# %%
_v0 = torch.cuda.memory_allocated()/1e9
del model, tokenizer, processor
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"Unicorn unloaded. VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")
print("Ready to load SeaLLMs judge.")

# %%
from transformers import AutoModelForCausalLM

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
# ## 8. Full Judge Evaluation — Unicorn

# %%
scores, faiths, covs, cohs, rationales, raws = [], [], [], [], [], []
for _, row in tqdm(df.iterrows(), total=len(df), desc="[Judge] Scoring Unicorn"):
    r = judge_score(row["documents"], row["ref_summary"], row["prediction"], debug=False)
    scores.append(r["score"])
    faiths.append(r["faithfulness"])
    covs.append(r["coverage"])
    cohs.append(r["coherence"])
    rationales.append(r["rationale"])
    raws.append(r["raw_output"])

df["judge_score"]       = scores
df["judge_faithfulness"] = faiths
df["judge_coverage"]    = covs
df["judge_coherence"]   = cohs
df["judge_rationale"]   = rationales
df["judge_raw"]         = raws
print(f"[Unicorn] Judging complete. {len(df)} clusters scored.")

# %% [markdown]
# ## 9. Unload Judge

# %%
_v0 = torch.cuda.memory_allocated()/1e9
del judge_model, judge_tokenizer
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"Judge unloaded. VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# ## 10. ROUGE Evaluation

# %%
from rouge_score import rouge_scorer as _rs

_scorer = _rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

def rouge_single(prediction: str, reference: str) -> dict:
    if not reference.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = _scorer.score(reference, prediction)
    return {m: s[m].fmeasure for m in ["rouge1", "rouge2", "rougeL"]}


r1_list, r2_list, rL_list = [], [], []
for _, row in df.iterrows():
    sc = rouge_single(row["prediction"], row["ref_summary"])
    r1_list.append(sc["rouge1"])
    r2_list.append(sc["rouge2"])
    rL_list.append(sc["rougeL"])

df["rouge1"] = r1_list
df["rouge2"] = r2_list
df["rougeL"] = rL_list

print(f"\n{'='*55}")
print(f"  ROUGE — {UNICORN_MODEL_ID}")
print(f"{'='*55}")
print(f"  ROUGE-1 : {sum(r1_list)/len(r1_list)*100:.2f}%")
print(f"  ROUGE-2 : {sum(r2_list)/len(r2_list)*100:.2f}%")
print(f"  ROUGE-L : {sum(rL_list)/len(rL_list)*100:.2f}%")
print(f"{'='*55}")

# %% [markdown]
# ## 11. Final Report

# %%
print("\n" + "#" * 75)
print("  EVALUATION SUMMARY")
print(f"  Model   : {UNICORN_MODEL_ID}")
print("  Judge   : SeaLLMs-v3-7B-Chat")
print("  Dataset : ViMs — Vietnamese Multi-Document Summarization")
print(f"  Clusters: {len(df)}")
print("#" * 75)

metrics = [
    ("ROUGE-1 (%)",  df["rouge1"].mean() * 100),
    ("ROUGE-2 (%)",  df["rouge2"].mean() * 100),
    ("ROUGE-L (%)",  df["rougeL"].mean() * 100),
    ("Judge Score",  df["judge_score"].mean()),
    ("Faithfulness", df["judge_faithfulness"].mean()),
    ("Coverage",     df["judge_coverage"].mean()),
    ("Coherence",    df["judge_coherence"].mean()),
]

print(f"\n  {'Metric':<22} {'Value':>10}")
print("  " + "-" * 35)
for label, val in metrics:
    print(f"  {label:<22} {val:>10.2f}")

print("\n  Score distribution:")
for sc, cnt in df["judge_score"].value_counts().sort_index().items():
    print(f"    {sc:2d}/10 : {'█' * int(cnt)} ({cnt})")

print(f"\n  5 worst clusters (by judge score):")
for _, r in df.nsmallest(5, "judge_score").iterrows():
    print(f"    {r['cluster_id']}  score={r['judge_score']}  rouge1={r['rouge1']*100:.1f}%")
    print(f"      rationale : {r['judge_rationale']}")
    print(f"      prediction: {str(r['prediction'])[:120]}...")

print("\n" + "#" * 75)

# %% [markdown]
# ## 12. Save Results

# %%
out_cols = [
    "cluster_id", "n_sentences", "ref_summary",
    "prediction",
    "rouge1", "rouge2", "rougeL",
    "judge_score", "judge_faithfulness", "judge_coverage",
    "judge_coherence", "judge_rationale",
]
df[out_cols].to_csv("vims_unicorn_results.csv", index=False, encoding="utf-8")
print("Saved: vims_unicorn_results.csv")
print("#" * 75)


