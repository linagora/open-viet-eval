# Vietnamese LLM Finetuning datasets
The Vietnamese LLM is rapidly maturing, while early efforts focused on building smaller models from scratch (e.g., PhoGPT), the current dominant and most cost-effective strategy is Continual Pre-Training (CPT) and Instruction Fine-Tuning of highly capable global open-source models (like Llama 3, Gemma, and Qwen) using localized Vietnamese datasets. Instead of training from scratch, teams are achieving State-of-the-Art results by adapting top-tier base models. A prime example is Vistral-7B-Chat (based on Mistral), which has been heavily fine-tuned to understand Vietnamese nuances.

## Finetuning Datasets categorized by LLM Task
To effectively train or fine-tune an LLM for Vietnamese, datasets are generally divided into three stages/tasks: Continual Pre-training (teaching the model the language), Instruction Tuning (teaching the model to follow commands), and Domain-Specific Fine-tuning (teaching the model specialized knowledge).

### Task 1: Continual Pre-Training (CPT) / Language Adaptation
**What it achieves:** CPT takes a base model (which primarily "thinks" in English) and force-feeds it massive amounts of raw, unstructured Vietnamese text. This teaches the model the grammar, syntax, vocabulary, idioms, and cultural context of Vietnam.
**Why it is important:** Without CPT, an English model trying to speak Vietnamese relies on fragmented, multi-lingual tokens. This leads to broken grammar, unnatural phrasing ("translationese"), and high hallucination rates. CPT builds a native-level foundation. However, it is not strictly necessary if the base model is already multilingual

* **[Vietnamese mC4 Subset (336GB)](https://huggingface.co/datasets/allenai/c4/viewer/vi):** A massive, filtered crawl of the Vietnamese internet. Used as the backbone for base language understanding.
* **[Vietnamese OSCAR-2301 Subset (88GB)](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301):** A highly cleaned corpus extracted from the Common Crawl, crucial for high-quality language modeling.
* **["binhvq" News Corpus (40GB)](https://github.com/binhvq/news-corpus):** A highly valuable, structured dataset comprising Vietnamese news articles. Excellent for teaching models formal Vietnamese phrasing and current events up to 2021/2023.
* **[SEACrowd / cc100](https://huggingface.co/datasets/SEACrowd/cc100):** A multilingual dataset that includes a high-quality Vietnamese split.
* **Vietnamese Wikipedia:** The standard dataset for factual, encyclopedic knowledge in Vietnamese.
* **Vietnamese Books Corpus:** Publicly available books spanning various genres, can be used to improve long-context understanding and literary phrasing.

### Task 2: Instruction Tuning & Conversational Alignment 
**What it achieves:** Instruction Tuning transforms a "text-completion" model into a helpful conversational agent. By feeding it formatted Prompt-Response pairs, the model learns how to answer questions, format lists, summarize text, and follow specific constraints.
**Why it is important:** A model that has only finished Task 1 might respond to a question by generating more questions. SFT teaches the model the expected behavior of an AI assistant.

* **[BKAI Foundation vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca):** A foundational Vietnamese instruction-tuning dataset. It provides clean, diverse, translated, and localized prompt-response pairs to teach models basic instruction-following behavior.

### Task 3: Domain-Specific Fine-Tuning
**What it achieves:** This task injects highly specialized, technical knowledge into the model that isn't present in general web data. 
**Why it is important:** General LLMs often fail at niche tasks (like drafting a legal contract under Vietnamese law or understanding local medical terminology). This step is crucial for enterprise deployments.

* **Vietnamese Legal Corpus:** `[thuvienphapluat.vn](https://thuvienphapluat.vn/)` and `[lawnet.vn](https://lawnet.vn/)` provide high quality legal texts.

