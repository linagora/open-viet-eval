# Vietnamese LLM Benchmark

Benchmark suite for evaluating large language models on Vietnamese NLP tasks.

## Models

| Model | Architecture | HuggingFace ID |
|---|---|---|
| **Qwen3.5-9B** | ImageTextToText (text-only) | `Qwen/Qwen3.5-9B` |
| **Qwen3-8B** | CausalLM | `Qwen/Qwen3-8B` |
| **Unicorn-VL-R3** | ImageTextToText (text-only) | `unicorn-team/Unicorn-VL-R3` |

## Datasets

| Dataset | Task | Metrics | Source |
|---|---|---|---|
| **VMLU** | Multiple-choice QA (58 subjects) | Accuracy | HuggingFace `tridm/VMLU` |
| **ViQuAD 2.0** | Extractive reading comprehension | Exact Match, F1 | HuggingFace `taidng/UIT-ViQuAD2.0` |
| **ViMs** | Multi-document summarisation | ROUGE-1/2/L + LLM Judge | Kaggle `vtrnanh/sust-feature-data-new` |
| **VTSNLP** | Instruction-following (general NLP) | LLM Judge | HuggingFace `VTSNLP/instruct_general_dataset` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a benchmark
python main.py --model qwen35 --dataset vmlu
python main.py --model unicorn --dataset vims --dataset-path /path/to/ViMs
python main.py --model qwen3 --dataset vtsnlp --skip-judge
```

## CLI Options

```
python main.py --model {qwen35,qwen3,unicorn}
               --dataset {vmlu,viquad,vims,vtsnlp}
               [--dataset-path PATH]     # Override default dataset location
               [--max-samples N]         # Limit samples (for quick testing)
               [--output-dir DIR]        # Where to save CSVs (default: ./results)
               [--skip-judge]            # Skip SeaLLMs judge scoring
               [--debug]                 # Print full prompts/outputs for first 3 samples
```

## Project Structure

```
vbenchmark/
├── main.py                  # CLI entry point
├── requirements.txt
├── utils.py                 # Text cleaning, file I/O, GPU memory helpers
├── models/
│   ├── base.py              # Abstract BaseModel interface
│   ├── qwen35.py            # Qwen3.5-9B adapter
│   ├── qwen3.py             # Qwen3-8B adapter
│   └── unicorn.py           # Unicorn-VL-R3 adapter (with one-shot prompting)
├── datasets/
│   ├── vmlu.py              # VMLU loader + prompt builder
│   ├── viquad.py            # ViQuAD loader + prompt builder
│   ├── vims.py              # ViMs folder parser + prompt builder
│   └── vtsnlp.py            # VTSNLP CSV loader + prompt builder
├── evaluation/
│   ├── metrics.py           # ROUGE, Exact Match, F1
│   └── judge.py             # SeaLLMs-v3-7B-Chat judge model
└── notebook/                # Original Jupyter notebooks (reference)
```

## Pipeline

Each benchmark run follows this pipeline:

1. **Load dataset** — from HuggingFace or local files
2. **Load model** → generate predictions → **unload model**
3. **Compute automatic metrics** — ROUGE (summarisation), EM/F1 (QA), or accuracy (MCQ)
4. *(Optional)* **Load judge** → score predictions → **unload judge**
5. **Save results** — CSV to `--output-dir`

Models are loaded and unloaded sequentially to fit within a single GPU's memory (e.g. Kaggle T4).

## LLM Judge

Datasets that support judge evaluation (`vims`, `vtsnlp`) use **SeaLLMs-v3-7B-Chat** to score predictions on a 1–10 scale:

- **ViMs** — evaluates faithfulness, coverage, and coherence
- **VTSNLP** — evaluates accuracy, completeness, coherence, and relevance

## Examples

```bash
# Quick test: 5 VMLU samples with debug output, no judge
python main.py --model qwen35 --dataset vmlu --max-samples 5 --debug --skip-judge

# Full ViMs benchmark with Unicorn
python main.py --model unicorn --dataset vims --dataset-path /data/ViMs

# Compare models on ViQuAD
python main.py --model qwen35 --dataset viquad --output-dir ./results
python main.py --model qwen3  --dataset viquad --output-dir ./results
python main.py --model unicorn --dataset viquad --output-dir ./results
```
