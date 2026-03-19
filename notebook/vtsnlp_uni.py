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
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor

# One-shot example reused across all Unicorn prompts 
_UNICORN_ONESHOT_CONTEXT = (
    "Nguyễn Trãi (1380 – 1442), hiệu là Ức Trai, là một nhà chính trị, nhà văn, người đã tham gia "
    "tích cực Khởi nghĩa Lam Sơn do Lê Lợi lãnh đạo. Ông trở thành khai quốc công thần của triều đại "
    "nhà Lê sơ. Thân thế Nguyễn Trãi quê ở làng Chi Ngại, nay thuộc Chí Linh, Hải Dương. Cha của "
    "cậu là Nguyễn Phi Khanh, một học giả nổi tiếng. Từ nhỏ, Ức Trai đã nổi tiếng thông minh. Sau "
    "khi thi đỗ thái học sinh, vị quan trẻ tuổi này ra làm quan dưới triều Hồ. Sự nghiệp Khi quân Minh "
    "xâm lược nước ta, cha ông bị bắt giải sang Trung Quốc. Nguyễn Trãi nghe lời cha quay về tìm cách "
    "rửa nhục cho đất nước. Về sau, nhà chiến lược đại tài này đã tìm đến Lỗi Giang để theo Lê Lợi. "
    "Dưới trướng của Bình Định Vương, ông đã dâng \"Bình Ngô sách\", hiến kế đánh đuổi quân Minh. Sau "
    "khi kháng chiến thành công, Nguyễn Trãi viết Bình Ngô đại cáo. Gia đình Một trong những người vợ "
    "của ông là Nguyễn Thị Lộ. Bà là một người phụ nữ tài sắc, được vua Lê Thái Tông phong làm Lễ nghi "
    "học sĩ. Tuy nhiên, Lệ Chi Viên đã trở thành vụ án oan khốc liệt khiến cả gia đình nhà văn vĩ đại "
    "này bị chu di tam tộc. Năm 1980, Nguyễn Trãi được UNESCO công nhận là Danh nhân văn hóa thế giới."
)

_UNICORN_ONESHOT_Q = "Xác định các danh từ, đại từ thay thế để có thể nhận ra chúng đề cập đến cùng một đối tượng"

_UNICORN_ONESHOT_A = (
    "- Nguyễn Trãi: Ức Trai, nhà chính trị, nhà văn, người, ông, khai quốc công thần, cậu, vị quan "
    "trẻ tuổi này, nhà chiến lược đại tài này, nhà văn vĩ đại này, Danh nhân văn hóa thế giới\n"
    "- Nguyễn Phi Khanh: Cha của cậu, học giả nổi tiếng, cha ông, cha\n"
    "- Lê Lợi: Bình Định Vương\n"
    "- Nguyễn Thị Lộ: vợ của ông, Bà, người phụ nữ tài sắc, Lễ nghi học sĩ\n"
    "- Lê Thái Tông: vua Lê Thái Tông"
)


def build_unicorn_instruct_prompt(instruct_text: str, input_text: str) -> str:
    """
    Build a raw prompt string in Unicorn's <|im_start|> chat format.

    The instruct field becomes part of the user turn (mirrors how build_instruct_messages
    prepends it as a system message), with a one-shot example for output-style grounding.
    """
    instruct_text = str(instruct_text).strip()
    input_text    = str(input_text).strip()

    # Compose the actual user content, prepending the instruction when available
    if instruct_text:
        user_content = f"{instruct_text}\n\n{input_text}"
    else:
        user_content = input_text

    prompt = (
        "<|im_start|>system\n"
        "Bạn là trợ lý AI hữu ích, trả lời bằng tiếng Việt, ngắn gọn và chính xác.<|im_end|>\n"
        # ── one-shot turn ────────────────────────────────────────────────────
        "<|im_start|>user\n"
        "Đây là một ví dụ về yêu cầu và kết quả của tác vụ mà tôi muốn bạn thực hiện \n"
        f"Đoạn văn mẫu:\n{_UNICORN_ONESHOT_CONTEXT}\n\n"
        f"Câu hỏi mẫu: {_UNICORN_ONESHOT_Q}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"Câu trả lời mẫu: {_UNICORN_ONESHOT_A}<|im_end|>\n"
        # ── actual query ─────────────────────────────────────────────────────
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    return prompt

# %% [markdown]
# # 3b. Generation Loop — Unicorn (ImageTextToText)

# %%
UNICORN_MODEL_ID = "unicorn-team/Unicorn-VL-R3"

print(f"\n{'='*65}\n  Loading Unicorn: {UNICORN_MODEL_ID}\n{'='*65}")
processor = AutoProcessor.from_pretrained(UNICORN_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(UNICORN_MODEL_ID, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    UNICORN_MODEL_ID,
    dtype=torch.bfloat16,   # Unicorn uses `dtype`, NOT `torch_dtype`
    device_map="auto",
)
model.eval()
print(f"Unicorn ready. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

@torch.inference_mode()
def generate_unicorn_response(prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the loaded Unicorn model for a pre-built prompt string."""
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
    output  = tokenizer.decode(new_ids, skip_special_tokens=True)
    output  = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    return output

# %%
SANITY_IDX = 0   # change to probe a different sample

_HR = "═" * 65

def _print_block(title: str, text: str):
    print(f"\n{_HR}")
    print(f"  {title}")
    print(_HR)
    print(text)

_sample          = instruct_samples[SANITY_IDX]
_sanity_prompt   = build_unicorn_instruct_prompt(_sample["instruct"], _sample["input"])
_sanity_expected = str(_sample["output"]).strip()
_sanity_answer   = generate_unicorn_response(_sanity_prompt)

_print_block("FULL PROMPT", _sanity_prompt)
_print_block("EXPECTED ANSWER", _sanity_expected)
_print_block("MODEL ANSWER", _sanity_answer)
print(f"\n{_HR}\n")

# %%
results = []
for i, sample in enumerate(tqdm(instruct_samples, desc=f"Generating [{UNICORN_MODEL_ID}]")):
    max_len    = int(sample.get("max_len", 512))
    prompt     = build_unicorn_instruct_prompt(sample["instruct"], sample["input"])
    prediction = generate_unicorn_response(prompt, max_new_tokens=max_len)
    results.append({
        "idx":        i,
        "category":   sample["category"],
        "instruct":   sample["instruct"],
        "input":      sample["input"],
        "reference":  sample["output"],
        "prediction": prediction,
        "model":      UNICORN_MODEL_ID,
    })

df_unicorn = pd.DataFrame(results)
print(f"Generated {len(df_unicorn)} predictions.")

_v0 = torch.cuda.memory_allocated()/1e9
del model, tokenizer, processor; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print(f"Unicorn unloaded. VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

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
# # 8. Judge All Three Models

# %%
df_unicorn   = run_judge(df_unicorn,   UNICORN_MODEL_ID)

# %% [markdown]
# # 9. Unload Judge

# %%
_v0 = torch.cuda.memory_allocated()/1e9
del judge_model, judge_tokenizer; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
print(f"Judge unloaded. VRAM freed: {_v0 - torch.cuda.memory_allocated()/1e9:.2f} GB")

# %% [markdown]
# # 10. Results

# %%
s = df_unicorn["judge_score"]
print(f"\n── {UNICORN_MODEL_ID} ──")
print(f"  Mean±Std : {s.mean():.2f} ± {s.std():.2f}  |  Min/Max: {s.min()}/{s.max()}")
dist = s.value_counts().sort_index()
bars = "  " + "  ".join(f"{k}▕{'█'*v}▏{v}" for k, v in dist.items())
print(f"  Dist     : {bars}")
cat = df_unicorn.groupby("category")["judge_score"].mean().sort_values(ascending=False).round(2)
print("  By cat   :", dict(cat))

df_unicorn.to_csv("results_unicorn.csv", index=False, encoding="utf-8")
print("Saved: results_unicorn.csv")


