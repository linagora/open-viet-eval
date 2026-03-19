"""ViMs — Vietnamese Multi-Document Summarization.

Loads from a folder with structure:
    ViMs/
      original/ClusterXXX/original/*.txt
      summary/ClusterXXX/*.gold.txt
      s3_summary/ClusterXXX/0.s3.txt
"""

from pathlib import Path

import pandas as pd

from utils import read_txt


MAX_DOC_CHARS = 10_000


# ── One-shot example for Unicorn summarization prompt ────────────────────────
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

_ONESHOT_ANSWER = (
    "Ngày 20/11/2023, Tòa án nhân dân cấp cao mở phiên phúc thẩm xét xử Trần Văn B về tội "
    "lạm dụng tín nhiệm chiếm đoạt tài sản.\n"
    "Lợi dụng chức vụ quản lý, bị cáo đã để ngoài sổ sách các khoản thu, chiếm đoạt hơn "
    "12 tỷ đồng của doanh nghiệp để đầu tư chứng khoán.\n"
    "Phía công ty đã yêu cầu bị cáo phải bồi thường toàn bộ số tiền thiệt hại nói trên.\n"
    "Dù bị cáo đã nộp một phần tiền khắc phục hậu quả, Hội đồng xét xử vẫn đánh giá hành vi "
    "đặc biệt nghiêm trọng nên quyết định y án sơ thẩm, phạt B 12 năm tù giam."
)


# ── Folder-structure helpers ─────────────────────────────────────────────────

def _find_dir(root: Path, name: str):
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() == name.lower():
            return p
    return None


def _load_s3_important_sentences(s3_path: Path) -> list:
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


def _load_cluster(cluster_dir: Path, summary_dir, s3_dir) -> dict | None:
    cid = cluster_dir.name
    if s3_dir is None:
        return None
    s3_cluster = s3_dir / cid
    s3_file = s3_cluster / "0.s3.txt"
    if not s3_file.exists():
        return None
    important_sents = _load_s3_important_sentences(s3_file)
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


def load_vims(
    dataset_root: str = "/kaggle/input/datasets/vtrnanh/sust-feature-data-new/ViMs",
    max_clusters: int = 100,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Parse ViMs folder structure into a DataFrame.

    Returns DataFrame with columns: cluster_id, n_sentences, documents, ref_summary
    """
    root = Path(dataset_root)
    original_dir = _find_dir(root, "original")
    summary_dir  = _find_dir(root, "summary")
    s3_dir       = _find_dir(root, "s3_summary")

    print(f"[ViMs] Dataset root : {root}")
    print(f"[ViMs] Original dir : {original_dir}")
    print(f"[ViMs] Summary dir  : {summary_dir}")
    print(f"[ViMs] S3 dir       : {s3_dir}")

    if original_dir is None:
        raise RuntimeError("Could not find 'original' directory. Check dataset_root path.")

    clusters = sorted([d for d in original_dir.iterdir() if d.is_dir()])
    print(f"[ViMs] Clusters found: {len(clusters)}")

    records, errors = [], []
    for cd in clusters[:max_clusters]:
        try:
            rec = _load_cluster(cd, summary_dir, s3_dir)
            if rec:
                records.append(rec)
        except Exception as e:
            errors.append((cd.name, str(e)))

    df = pd.DataFrame(records)
    print(f"[ViMs] Parsed {len(df)} clusters ({len(errors)} errors)")
    if errors:
        print("[ViMs] Errors:", errors[:5])

    if max_samples:
        df = df[:max_samples]
    return df


def build_summarization_messages(documents: str) -> list:
    """Build chat messages for ViMs summarization (for Qwen models)."""
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
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_summarization_prompt_unicorn(documents: str) -> str:
    """Build a raw <|im_start|> prompt for Unicorn summarization with one-shot."""
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
        # ── one-shot turn ──
        "<|im_start|>user\n"
        f"Các câu quan trọng mẫu:\n\n{_ONESHOT_CONTEXT}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
        f"Mẫu tóm tắt: {_ONESHOT_ANSWER}<|im_end|>\n"
        # ── actual query ──
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
