# %% [markdown]
# # Setup and Installation

# %%
!pip uninstall -y transformers
!pip install -q "transformers>=4.48.0"
!pip install -q accelerate datasets pillow
!pip install -q gguf>=0.10.0

# %% [markdown]
# # 1. Dataset Loading 

# %%
import pandas as pd
import random
from collections import Counter

CSV_PATH  = "/kaggle/input/datasets/qnfuioyhgvqpwo/sample-vtsnlp-instruct-dataset/sampled_instruct_general_dataset.csv" 
SEED      = 42
N_PER_CAT = 10

random.seed(SEED)

df_raw = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_raw)} rows from CSV.")
print("Columns:", list(df_raw.columns))

# Sample N_PER_CAT per category
instruct_samples = (
    df_raw.groupby("category", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), N_PER_CAT), random_state=SEED))
          .reset_index(drop=True)
          .to_dict("records")
)

cat_counts = Counter(s["category"] for s in instruct_samples)
print(f"\nTotal samples: {len(instruct_samples)}")
print("Category distribution:")
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    print(f"  {cat:<25}: {cnt}")

# %% [markdown]
# # 2. Shared Helpers

# %%
import re
import gc
import json as _json
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_instruct_messages(instruct_text: str, input_text: str) -> list:
    messages = []
    if instruct_text and str(instruct_text).strip():
        messages.append({"role": "system", "content": str(instruct_text).strip()})
    messages.append({"role": "user", "content": str(input_text).strip()})
    return messages

# %% [markdown]
# # 3. Generation Loop

# %%
def load_candidate_model(model_id: str):
    print(f"\n{'='*65}\n  Loading: {model_id}\n{'='*65}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    mdl.eval()
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return mdl, tok


@torch.inference_mode()
def generate_response(model, tokenizer, messages, max_new_tokens=512):
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    gen_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, temperature=None, top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return output


def run_generation(model_id: str, samples: list) -> tuple:
    model, tokenizer = load_candidate_model(model_id)

    results = []
    for i, sample in enumerate(tqdm(samples, desc=f"Generating [{model_id}]")):
        # Respect per-sample max_len if present
        max_len = int(sample.get("max_len", 512))
        msgs = build_instruct_messages(sample["instruct"], sample["input"])
        prediction = generate_response(model, tokenizer, msgs, max_new_tokens=max_len)
        results.append({
            "idx":        i,
            "category":   sample["category"],
            "instruct":   sample["instruct"],
            "input":      sample["input"],
            "reference":  sample["output"],
            "prediction": prediction,
            "model":      model_id,
        })

    df = pd.DataFrame(results)
    print(f"Generated {len(df)} predictions for {model_id}.")
    return df, model, tokenizer

# %% [markdown]
# # 4. Evaluate Qwen3.5-9B

# %%
df_qwen35_9b, model, tokenizer = run_generation("Qwen/Qwen3.5-9B", instruct_samples)

_v0 = torch.cuda.memory_allocated()/1e9
del model, tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print(f"VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# # 5. Evaluate Qwen3-8B

# %%
df_qwen3_8b, model, tokenizer = run_generation("Qwen/Qwen3-8B", instruct_samples)

_v0 = torch.cuda.memory_allocated()/1e9
del model, tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print(f"VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# # 6. Load Judge — SeaLLMs-v3-7B-Chat

# %%
JUDGE_MODEL = "SeaLLMs/SeaLLMs-v3-7B-Chat"
judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True, padding_side="left")
judge_model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
judge_model.eval()
print(f"Judge ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# # 7. Judge Helpers

# %%
JUDGE_SYSTEM_PROMPT = """Bạn là một giám khảo AI chuyên đánh giá chất lượng câu trả lời bằng tiếng Việt.
Đánh giá theo: độ chính xác, tính đầy đủ, tính mạch lạc, độ phù hợp.
Trả lời CHÍNH XÁC theo định dạng JSON sau, không thêm bất kỳ văn bản nào khác:
{"score": <số nguyên từ 1 đến 10>, "rationale": "<lý do ngắn gọn bằng tiếng Việt, tối đa 2 câu>"}"""


def build_judge_messages(instruct, inp, reference, prediction):
    content = (
        f"### Hướng dẫn:\n{str(instruct).strip()}\n\n"
        f"### Câu hỏi:\n{str(inp).strip()[:800]}\n\n"
        f"### Tham chiếu:\n{str(reference).strip()[:600]}\n\n"
        f"### Mô hình:\n{str(prediction).strip()[:600]}\n\n"
        "Đánh giá và trả JSON."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": content}]


def parse_judge_output(raw: str) -> dict:
    try:
        obj = _json.loads(raw.strip())
        return {"score": int(obj["score"]), "rationale": obj.get("rationale", "")}
    except Exception:
        pass
    m = re.search(r'\{[^{}]*"score"\s*:\s*(\d+)[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            return {"score": int(obj["score"]), "rationale": obj.get("rationale", "")}
        except Exception:
            pass
    m2 = re.search(r'"score"\s*:\s*(\d+)', raw)
    return {"score": min(max(int(m2.group(1)) if m2 else 5, 1), 10), "rationale": "(parse fallback)"}


@torch.inference_mode()
def judge_score(instruct, inp, reference, prediction) -> dict:
    messages = build_judge_messages(instruct, inp, reference, prediction)
    try:
        prompt = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{JUDGE_SYSTEM_PROMPT}\n\nUser: {messages[1]['content']}\nAssistant:"

    inputs = judge_tokenizer([prompt], return_tensors="pt").to(judge_model.device)
    gen_ids = judge_model.generate(
        **inputs, max_new_tokens=256, do_sample=False,
        temperature=None, top_p=None, pad_token_id=judge_tokenizer.eos_token_id,
    )
    raw = judge_tokenizer.decode(gen_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    parsed = parse_judge_output(raw)
    parsed["raw_output"] = raw
    return parsed


def run_judge(df: pd.DataFrame, label: str) -> pd.DataFrame:
    scores, rationales, raws = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Judging [{label}]"):
        result = judge_score(row["instruct"], row["input"], row["reference"], row["prediction"])
        scores.append(result["score"])
        rationales.append(result["rationale"])
        raws.append(result["raw_output"])
    df = df.copy()
    df["judge_score"]     = scores
    df["judge_rationale"] = rationales
    df["judge_raw"]       = raws
    print(f"Judged {len(df)} samples for {label}.")
    return df

# %% [markdown]
# # 8. Judge Both Models

# %%
df_qwen35_9b = run_judge(df_qwen35_9b, "Qwen/Qwen3.5-9B")
df_qwen3_8b  = run_judge(df_qwen3_8b,  "Qwen/Qwen3-8B")

# %% [markdown]
# # 9. Unload Judge

# %%
_v0 = torch.cuda.memory_allocated()/1e9
del judge_model, judge_tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print(f"Judge unloaded. VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# # 10. Results

# %%
MODEL_A_ID = "Qwen/Qwen3.5-9B"
MODEL_B_ID = "Qwen/Qwen3-8B"

def model_summary(df, model_id):
    s = df["judge_score"]
    print(f"\n── {model_id} ──")
    print(f"  Mean±Std : {s.mean():.2f} ± {s.std():.2f}  |  Min/Max: {s.min()}/{s.max()}")
    # Score distribution (compact bar)
    dist = s.value_counts().sort_index()
    bars = "  " + "  ".join(f"{k}▕{'█'*v}▏{v}" for k, v in dist.items())
    print(f"  Dist     : {bars}")
    # Per-category mean (table)
    cat = df.groupby("category")["judge_score"].mean().sort_values(ascending=False).round(2)
    print("  By cat   :", dict(cat))

model_summary(df_qwen35_9b, MODEL_A_ID)
model_summary(df_qwen3_8b,  MODEL_B_ID)

# %% [markdown]
# ## Head-to-head

# %%
score_a = df_qwen35_9b["judge_score"].mean()
score_b = df_qwen3_8b["judge_score"].mean()
delta   = score_a - score_b
winner  = MODEL_A_ID if delta > 0 else MODEL_B_ID if delta < 0 else "TIE"

wins_a = (df_qwen35_9b["judge_score"].values > df_qwen3_8b["judge_score"].values).sum()
wins_b = (df_qwen35_9b["judge_score"].values < df_qwen3_8b["judge_score"].values).sum()
ties   = (df_qwen35_9b["judge_score"].values == df_qwen3_8b["judge_score"].values).sum()

print(f"\n{'─'*55}")
print(f"  {MODEL_A_ID:<28} {score_a:.2f}/10")
print(f"  {MODEL_B_ID:<28} {score_b:.2f}/10")
print(f"  Delta (A-B): {delta:+.2f}   Winner: {winner}")
print(f"  Wins A/B/Tie: {wins_a} / {wins_b} / {ties}")

# Per-category comparison table
cat_a = df_qwen35_9b.groupby("category")["judge_score"].mean().rename("A")
cat_b = df_qwen3_8b.groupby("category")["judge_score"].mean().rename("B")
cmp   = pd.concat([cat_a, cat_b], axis=1).assign(**{"A-B": lambda d: (d["A"]-d["B"]).round(2)})
cmp[["A","B"]] = cmp[["A","B"]].round(2)
print(f"\n{cmp.sort_values('A-B', ascending=False).to_string()}")

# %% [markdown]
# # 11. Save Results

# %%
df_qwen35_9b.to_csv("results_qwen35_9b.csv", index=False, encoding="utf-8")
df_qwen3_8b.to_csv("results_qwen3_8b.csv",  index=False, encoding="utf-8")
pd.concat([df_qwen35_9b, df_qwen3_8b], ignore_index=True).to_csv("results_combined.csv", index=False, encoding="utf-8")
print("Saved: results_qwen35_9b.csv, results_qwen3_8b.csv, results_combined.csv")


