# Open-Viet-Eval full benchmark result

## Dataset:  
**VMLU**: https://huggingface.co/datasets/tridm/VMLU

- Multiple choice question answering benchmark designed to evaluate general knowledge and reasoning capabilities. It includes questions that span difficulty levels from basic understanding to advanced professional expertise.  
- Total: 9833 questions  
- Example: “Elementary Mathematics (in STEM)  
  Tính chất nào sau đây không phải là tính chất của thủy tinh chất lượng cao:  
  - A. Rất trong  
  - B. Bền, khó vỡ  
  - C. Chịu được nóng, lạnh  
  - D. Dễ cháy

  Answer: D”

**UIT-ViSquAD2.0:** [https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)

- QA pairs based on 174 Vietnamese Wikipedia articles. With both answerable and unanswerable questions.  
- Total: 3814 questions  
- Example: “	Hiện nay, hầu như tất cả các chương trình máy tính trong thực tế đều được viết bằng các ngôn ngữ bậc cao hay (đôi khi) hợp ngữ, và sau đó được dịch thành mã máy thực thi bằng các công cụ phụ trợ như trình biên dịch, trình hợp dịch hay trình liên kết. Ngoài ra, các chương trình được viết bằng ngôn ngữ thông dịch thì được dịch sang mã máy nhờ trình thông dịch tương ứng (có thể xem như là trình thực thi hay trình xử lý). Các trình thông dịch này thường bao gồm các mã máy thực thi trực tiếp (sinh ra từ mã nguồn hợp ngữ hay các ngôn ngữ bậc cao).  
  Question: Thành phần nào thường có trong các trình thông dịch?  
  Answer: { "text": \[ "các mã máy thực thi trực tiếp"\], "answer\_start": \[462\]}”

**Vietnamese Multiple Document Summarization Dataset (ViM):** [https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new](https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new)

- 300 news clusters for Vietnamese abstractive multi-document summarization   
- The model write a summarization based on important sentences from news article related to a topic  
- Example: Các câu quan trọng:  
  \[Câu 1\] Bộ trưởng Hàng không Ai Cập Sherif Fathy nghiêng về khả năng MS804 bị khủng bố nhiều hơn là trục trặc kỹ thuật.  
  \[Câu 2\] Các chuyên gia hàng không lo ngại một hành động cố ý đã khiến máy bay biến mất khỏi màn hình radar vào sáng ngày 19.5, theo tờ South China Morning Post (Hồng Kông) ngày 20.5.  
  \[Câu 3\] Ezeddin Samar cũng chính là nữ tiếp viên có mặt trên chuyến bay mang số hiệu MS804 mất tích hôm qua 19/5.  
  \[Câu 4\] Họ nghi ngờ một quả bom khiến máy bay nổ tung, sau đó rơi xuống biển.  
  \[Câu 5\] Tổng thư ký Tổ chức Hiệp ước Bắc Đại Tây Dương (NATO) Jens Stoltenberg ngày 19/5 cho biết, nếu Ai Cập đề nghị, liên minh này sẽ hỗ trợ công tác tìm kiếm chiếc máy bay mang số hiệu MS 804 của hãng hàng không Ai Cập chở 66 người mất tích trước đó cùng ngày.  
  ….  
  Generate a summarization based on the provided information


**Vietnamese Instruct General Dataset:** [https://huggingface.co/datasets/VTSNLP/instruct\_general\_dataset](https://huggingface.co/datasets/VTSNLP/instruct_general_dataset)

- Contains 1341 samples with diverse tasks such as summarizing, questioning, translating, inferring, and content generation. It was built from aggregated and translated data from multiple sources, combined with data generation and quality checking using Gemini.    
- Example:   
  Instruct: Hãy trình bày từng bước để trả lời câu hỏi dưới đây một cách chi tiết.  
    
  Input: (CNN) – Một trong ba thiếu niên bị buộc tội đốt một cậu bé khác ở Florida vào năm 2009 đã bị kết tội nhẹ hơn về hành vi bạo hành trầm trọng hơn, các quan chức cho biết. Theo Kim Fontana tại Văn phòng Luật sư Tiểu bang ở Quận Broward, Matthew Bent, 17 tuổi, bị buộc tội với tư cách là người trưởng thành, sẽ bị kết án vào ngày 23 tháng 7\. Bent phải đối mặt với mức án từ 4 năm 5 tháng đến 15 năm. Bent ban đầu bị buộc tội cố ý giết người cấp độ hai trong vụ tấn công ở bãi biển Deerfield. Bồi thẩm đoàn đã trả lại phán quyết hôm thứ ba. Ba thiếu niên – Bent, Denver Jarvis và Jesus Mendez – bị cáo buộc đổ rượu vào người Michael Brewer và châm lửa đốt anh ta sau một cuộc tranh chấp về số tiền 40 đô la. @highlight Matthew Bent bị kết tội hành hung nghiêm trọng @highlight Anh ấy và hai thiếu niên khác bị buộc tội đốt cháy Michael Brewer vào năm 2009 @highlight Brewer bị bỏng hơn 65% cơ thể và phải nhập viện hơn hai tháng. Các thám tử cho biết các nhân chứng đã nói với họ rằng @placeholder lúc đó 15 tuổi đã đổ rượu lên người Brewer và Mendez lúc đó 16 tuổi đã dùng bật lửa để châm lửa. Trong câu hỏi trên, "@placeholder" là viết tắt của  
    
  Answer: Bối cảnh đề cập rằng Denver Jarvis mới 15 tuổi khi vụ việc xảy ra. Ba thiếu niên \-- Bent, @placeholder và Jesus Mendez-- bị cáo buộc đổ rượu vào người Michael Brewer và châm lửa đốt anh ta sau một cuộc tranh chấp về số tiền 40 đô la. Do đó, dựa trên mô tả này, có thể kết luận rằng @placeholder là viết tắt của Denver Jarvis trong văn bản. Tóm lại, câu trả lời là: Denver Jarvis.\<|end\_of\_text|\>

## Model Overview

The models evaluated in this report are based on the Qwen3 family, a series of open-weight large language models designed for strong performance across reasoning, multilingual understanding, and general NLP tasks. Qwen3 models are trained on approximately 36 trillion tokens spanning 119 languages including Vietnamese, enabling effective cross-lingual understanding and generation. The models are trained through a multi-stage pipeline that includes large-scale pretraining, reasoning-focused fine-tuning with reinforcement learning, and general alignment for instruction following, covering capabilities such as reasoning, coding, multilingual QA, summarization, and agent-style interactions.

Unicorn-R3 is a fine-tuned variant of Qwen 3 8B, specifically adapted for Vietnamese through a combination of synthetic data generation and filtered public datasets. Its training emphasizes reasoning and academic-style tasks via chain-of-thought augmentation, aiming to improve logical inference and domain-specific understanding in Vietnamese contexts.

## Results Analysis
| Dataset |  | [Qwen 3.5 9B AWQ](https://huggingface.co/QuantTrio/Qwen3.5-9B-AWQ) | [Qwen 3 8B AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ)  | [Unicorn-R3](https://huggingface.co/unicorn-team/Unicorn-R3) |
| :---- | :---- | :---: | :---: | :---: |
| VMLU | Accuracy: | 65.25% | 60.31% | 61.92% |
| ViSquAD2.0 | F1: | 76.01% | 66.04% | 64.59% |
|  | EM: | 63.24% | 51.39% | 49.82% |
| ViM | ROUGE-L: | 35.98 | 32.79 | 33.05 |
|  | LLM-as-judge: | 4.60 | 4.39 | 4.41 |
| VTSNLP | LLM-as-judge: | 3.17 | 3.39 | 3.32 |

LLM-as-judge result use Qwen 3.5 9B judge model  

Qwen 3.5 9B (AWQ quantized) demonstrates the strongest performance. It achieves the highest score on VMLU (65.25%), indicating superior general knowledge and reasoning ability, and significantly outperforms other models on UIT-ViSquAD2.0 with 76.01 F1 / 63.24 EM, highlighting its strength in reading comprehension and precise answer extraction. The model also leads in summarization (ViM) suggesting better content selection and fluency. However, its performance on the VTSNLP instruction-following dataset is slightly lower than smaller models, implying that while it excels in strucmese fine-tuning can yield improvements in reasoning and summarization, these gains are task-dependent and relatively modest. Mtured and knowledge-intensive tasks, it may not have a clear advantage in diverse, instruction-heavy scenarios. 

Qwen 3 8B (AWQ quantized) shows consistently lower performance across most benchmarks. On UIT-ViSquAD2.0, its performance drops notably (66.04 F1 / 51.39 EM), suggesting limitations in precise comprehension and span extraction. Similarly, in summarization (ViM), it produces lower-quality outputs compared to Qwen 3.5 9B. Interestingly, it achieves the highest score on VTSNLP (6.78), which may indicate relatively better alignment with general instruction-following tasks compared to the other two models. 

Unicorn-R3 (Qwen 3 8B-based, fine-tuned) shows targeted improvements from Vietnamese-specific training but with mixed results. On VMLU (61.92%), it outperforms the base Qwen 3 8B, demonstrating that synthetic reasoning data and chain-of-thought-style training can enhance structured reasoning ability. It also slightly improves over the baseline in summarization (ViM), suggesting gains in abstraction and content synthesis. However, Unicorn-R3 underperforms on UIT-ViSquAD2.0 (64.59 F1 / 49.82 EM), even falling below the base model, which indicates that its training is less effective for extractive QA tasks. Its performance on VTSNLP (6.64) is comparable but not superior, showing limited generalization to diverse instruction tasks. Overall, Unicorn-R3 highlights that synthetic Vietnamese fine-tuning can improve reasoning-focused benchmarks, but its benefits do not consistently transfer across all task types.

Previously only the Unicorn model is provided with one-shot example for the Squad to ensure accurate output format. In the full test, we provide one-shot example for all model when benchmarking on the VMLU and Squad dataset, which significantly increase exact match
accross all models.  

In conclusion, the results reveal a clear hierarchy: Qwen 3.5 9B > Unicorn-R3 ≈ Qwen 3 8B, with model scale being the most influential factor. While Unicorn-R3 demonstrates that targeted Vietnaoreover, fine-tuning may even degrade performance in tasks requiring high precision, such as extractive QA. Despite AWQ quantization, larger models maintain a significant advantage, suggesting that capacity and generalization outweigh lightweight adaptation strategies in this setting.

For a retrieval-augmented generation (RAG) use case, Qwen 3.5 9B (AWQ quantized) is the most suitable model among the three. Its strong performance on UIT-ViSquAD2.0 (highest F1 and EM) indicates superior ability to accurately extract and ground answers from retrieved documents, which is a critical requirement for RAG systems. Additionally, its leading results on ViM summarization demonstrate better capability in synthesizing information across multiple sources, while its top performance on VMLU suggests robust reasoning when combining evidence from different contexts.

