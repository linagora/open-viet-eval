"""Dataset loaders for Vietnamese LLM benchmarking."""

from .vmlu import load_vmlu, build_vmlu_messages, extract_mcq_answer, CATEGORY_MAP
from .viquad import load_viquad, build_squad_messages
from .vims import load_vims, build_summarization_messages
from .vtsnlp import load_vtsnlp, build_instruct_messages

DATASET_REGISTRY = {
    "vmlu":   "vmlu",
    "viquad": "viquad",
    "vims":   "vims",
    "vtsnlp": "vtsnlp",
}
