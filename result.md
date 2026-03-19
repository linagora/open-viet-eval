Dataset:  
**VMLU**: https://huggingface.co/datasets/tridm/VMLU

- Multiple choice question answering benchmark designed to evaluate general knowledge and reasoning capabilities. It includes questions that span difficulty levels from basic understanding to advanced professional expertise.  
- Total: 744 questions  
- Example: “Elementary Mathematics (in STEM)  
  Tính chất nào sau đây không phải là tính chất của thủy tinh chất lượng cao:  
  - A. Rất trong  
  - B. Bền, khó vỡ  
  - C. Chịu được nóng, lạnh  
  - D. Dễ cháy

  Answer: D”

**UIT-ViSquAD2.0:** [https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)

- QA pairs based on 174 Vietnamese Wikipedia articles. With both answerable and unanswerable questions.  
- Total: 1000 questions, 10% unanswerable  
- Example: “	Hiện nay, hầu như tất cả các chương trình máy tính trong thực tế đều được viết bằng các ngôn ngữ bậc cao hay (đôi khi) hợp ngữ, và sau đó được dịch thành mã máy thực thi bằng các công cụ phụ trợ như trình biên dịch, trình hợp dịch hay trình liên kết. Ngoài ra, các chương trình được viết bằng ngôn ngữ thông dịch thì được dịch sang mã máy nhờ trình thông dịch tương ứng (có thể xem như là trình thực thi hay trình xử lý). Các trình thông dịch này thường bao gồm các mã máy thực thi trực tiếp (sinh ra từ mã nguồn hợp ngữ hay các ngôn ngữ bậc cao).  
  Question: Thành phần nào thường có trong các trình thông dịch?  
  Answer: { "text": \[ "các mã máy thực thi trực tiếp"\], "answer\_start": \[462\]}”

**Vietnamese Multiple Document Summarization Dataset (ViM):** [https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new](https://www.kaggle.com/datasets/vtrnanh/sust-feature-data-new)

- 100 news clusters for Vietnamese abstractive multi-document summarization   
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

- Contains 4.5 million samples with diverse tasks such as summarizing, questioning, translating, inferring, and content generation. It was built from aggregated and translated data from multiple sources, combined with data generation and quality checking using Gemini.  
- The benchmark sample random 10 questions from each of the 11 category  
- Example:   
  Instruct: Hãy trình bày từng bước để trả lời câu hỏi dưới đây một cách chi tiết.  
    
  Input: (CNN) – Một trong ba thiếu niên bị buộc tội đốt một cậu bé khác ở Florida vào năm 2009 đã bị kết tội nhẹ hơn về hành vi bạo hành trầm trọng hơn, các quan chức cho biết. Theo Kim Fontana tại Văn phòng Luật sư Tiểu bang ở Quận Broward, Matthew Bent, 17 tuổi, bị buộc tội với tư cách là người trưởng thành, sẽ bị kết án vào ngày 23 tháng 7\. Bent phải đối mặt với mức án từ 4 năm 5 tháng đến 15 năm. Bent ban đầu bị buộc tội cố ý giết người cấp độ hai trong vụ tấn công ở bãi biển Deerfield. Bồi thẩm đoàn đã trả lại phán quyết hôm thứ ba. Ba thiếu niên – Bent, Denver Jarvis và Jesus Mendez – bị cáo buộc đổ rượu vào người Michael Brewer và châm lửa đốt anh ta sau một cuộc tranh chấp về số tiền 40 đô la. @highlight Matthew Bent bị kết tội hành hung nghiêm trọng @highlight Anh ấy và hai thiếu niên khác bị buộc tội đốt cháy Michael Brewer vào năm 2009 @highlight Brewer bị bỏng hơn 65% cơ thể và phải nhập viện hơn hai tháng. Các thám tử cho biết các nhân chứng đã nói với họ rằng @placeholder lúc đó 15 tuổi đã đổ rượu lên người Brewer và Mendez lúc đó 16 tuổi đã dùng bật lửa để châm lửa. Trong câu hỏi trên, "@placeholder" là viết tắt của  
    
  Answer: Bối cảnh đề cập rằng Denver Jarvis mới 15 tuổi khi vụ việc xảy ra. Ba thiếu niên \-- Bent, @placeholder và Jesus Mendez-- bị cáo buộc đổ rượu vào người Michael Brewer và châm lửa đốt anh ta sau một cuộc tranh chấp về số tiền 40 đô la. Do đó, dựa trên mô tả này, có thể kết luận rằng @placeholder là viết tắt của Denver Jarvis trong văn bản. Tóm lại, câu trả lời là: Denver Jarvis.\<|end\_of\_text|\>

| Dataset |  | Qwen 3.5 9B | Qwen 3 8B | Unicorn-VL-R3 |
| :---- | :---- | :---: | :---: | :---: |
| VMLU |Accuracy:  | 75.91% | 66.67% | 67.07% |
| ViSquAD2.0 | F1: | 75.42% | 50.75% | 67.73% |
|  | EM: | 48.90% | 6.90% | 39.20% |
| ViM | ROUGE-L: | 46.69 | 45.73 | 50.73 |
|  | LLM-as-judge: | 7.48 | 7.62 | 7.59 |
| VTSNLP | LLM-as-judge: | 7.57 | 7.52 | 7.53 |

 LLM-as-judge result use SeaLLMs-v3-7B-Chat as judge model  
Unicorn-VL-R3 is a Qwen3-VL-8B-Thinking model fintuned on synthetic Vietnamese data. The training dataset combines custom-generated synthetic data and rigorously filtered public data, resulting in nearly 10,000 high-quality Vietnamese samples.

**1\. Synthetic Data Generation (\~4,800 samples)** 

* **Synthetic Syllabus Mapping:** The team constructed a comprehensive educational framework, categorizing the data by education level (Primary, Middle, and High School), grade, subject, and specific topics.  
* **Targeted Question Generation:** Using Gemini 2.5 Pro, they generated questions that mapped directly to this syllabus and aligned with the 12 specific tasks defined by the VMLU.  
* **Response Generation:** The final answers to these generated questions were produced using Gemini 3 Pro and Gemini 2.5 Pro.  
* **Simulated Chain of Thought (CoT):** In the final step, Gemini 2.5 Pro/Flash was used to retroactively generate the reasoning steps bridging the question and the answer. This synthetic thinking process was styled to mimic the Qwen model's reasoning behavior. This technique optimizes the model's logical problem-solving capabilities for academic contexts and aligns it closer to reality.

**2\. Public Data Integration (\~5,000 samples)**

* **Dolci-SFT Filtering:** The team extracted data from the massive 2-million-sample Dolci-SFT open-source dataset. By filtering specifically for the Vietnamese language and relevant target tasks, they isolated approximately 5,000 high-quality samples.

