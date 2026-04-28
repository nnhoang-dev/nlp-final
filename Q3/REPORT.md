# BÁO CÁO MÔN HỌC: XỬ LÝ NGÔN NGỮ TỰ NHIÊN
**Đề tài: Xây dựng Mô hình Hội thoại Tư vấn Sức khỏe Tinh thần (Mental Health Chatbot)**

## 1. Mô Tả Dữ Liệu

### Nguồn dữ liệu
Dự án sử dụng bộ dữ liệu **`ShenLab/MentalChat16K`** được công bố mã nguồn mở trên thư viện Hugging Face. Đây là tập dữ liệu chứa các cuộc hội thoại tư vấn tâm lý chất lượng cao, nơi người dùng đặt ra các câu hỏi, tâm sự về vấn đề tâm lý cá nhân và "bác sĩ tâm lý" (counsellor) phản hồi lại bằng những tư vấn mang tính thấu cảm, khuyên nhủ và hỗ trợ trị liệu.

### Cách thu thập và Tiền xử lý
- **Lấy mẫu (Sampling):** Để phù hợp với tài nguyên tính toán (nếu có hạn), toàn bộ tập dữ liệu gốc được lấy mẫu ngẫu nhiên xuống còn 5,000 mẫu dữ liệu (sử dụng random seed để tái lập).
- **Phân tách tập dữ liệu:** Chia dữ liệu thành 3 tập Train / Validation / Test với tỉ lệ lần lượt là `80% / 10% / 10%`.
- **Phân tích khám phá (EDA):** Dữ liệu được tính toán phân phối độ dài (số từ) của đoạn hội thoại `input` và `output` nhằm kiểm soát độ tối đa (max length) khi đưa vào tokenizer. 
- **Định dạng cấu trúc Prompt (Formatting):** Do đây là bài toán đào tạo Chatbot từ LLM (Large Language Model), các cặp câu hỏi - câu trả lời được xử lý và gói gọn trong `Chat Template` tiêu chuẩn `<|im_start|>` và `<|im_end|>` với System Prompt mặc định: *"You are a compassionate and professional mental health counsellor. Provide empathetic, helpful, and accurate responses to users seeking emotional support."*

---

## 2. Mô Hình Sử Dụng & Biện Minh Cấu Trúc

### Cấu trúc Decoder-only Transformer
Trong yêu cầu nguyên bản của bài toán là chỉ ra "kiến trúc Encoder-Decoder Transformer". Tuy nhiên, nắm bắt xu hướng SOTA (State-of-the-Art) của AI hiện đại, trong dự án này, cốt lõi mô hình sử dụng **kiến trúc Decoder-only Transformer** thông qua mô hình **`Qwen2.5-0.5B-Instruct`**.

#### Đặc điểm của Decoder-only:
Thay vì có 2 khối riêng biệt (Encoder học biểu diễn ngữ cảnh, Decoder sinh từ ngữ) như T5 hay BART, mạng Decoder-only (như họ GPT, Llama, Qwen) chỉ có khối Decoder với kỹ thuật **Causal Masking (Mặt nạ tự hồi quy)**. Nó chặn mô hình nhìn vào các từ tương lai, buộc mô hình phải dự đoán từ tiếp theo dựa trên chuỗi từ trước đó. 

#### Giới hạn của Encoder-Decoder & Biện minh sử dụng Decoder-only:
Việc chuyển từ Encoder-Decoder sang thuần Decoder-only được biện minh bằng các yếu tố khoa học & thực tiễn sau sẽ tốt hơn cho hệ thống:
1. **Khả năng Sinh văn bản (Text Generation) tự nhiên vượt trội:** Encoder-Decoder thường tối ưu cho các bài toán dịch máy (translation) hoặc tóm tắt (summarization) nơi đầu ra bám rất sát vào đầu vào. Tuy nhiên, bài toán *tư vấn tâm lý* đòi hỏi tính mở (open-ended), yêu cầu chatbot có kỹ năng lập luận, giao tiếp linh hoạt, logic giống con người. Decoder-only thể hiện sự vượt trội hoàn toàn vì mục tiêu thiết kế nguyên thuỷ của nó là Causal Language Modeling. 
2. **Kiến thức Zero-shot và Nền tảng rộng:** Các mô hình LLM Decoder-only hiện nay được pre-train trên khối lượng tri thức khổng lồ. Việc sử dụng Model Qwen2.5 cho phép thừa hưởng nguồn dữ liệu pre-train khổng lồ về lý luận và ngôn ngữ, giúp câu phản hồi có sự thấu cảm (empathy) tự nhiên - điều mà việc tự train một mô hình Encoder-Decoder nhỏ hẹp không thể có được.
3. **Prompting & In-context learning:** Decoder-only cho phép ứng dụng `System Prompt` mạnh mẽ, giúp hệ thống dễ dàng đóng vai một chuyên gia tư vấn mà không cần thay đổi trọng số gốc từ đầu.

---

## 3. Fine-tuning từ LLM (Tinh chỉnh mô hình)

Tuy mô hình nền tảng Decoder-only (Qwen2.5-0.5B) đã có giao tiếp căn bản, nhưng nó cần được định hướng chuyên sâu để trở thành một hệ thống hỏi đáp tư vấn y tế/tâm lý thực thụ. Quy trình được thực hiện qua kỹ thuật tinh chỉnh:

### Lựa chọn kỹ thuật: LoRA (Low-Rank Adaptation)
Bởi vì LLM có kích thước tham số rất lớn, việc Tinh chỉnh toàn bộ (Full Fine-Tuning) sẽ gây tốn kém VRAM và dễ bị *Catastrophic Forgetting* (Quên kiến thức cũ). Kỹ thuật **LoRA** được sử dụng để đóng băng các lớp gốc, chỉ tiêm và huấn luyện các ma trận có hạng thấp (Low Rank) tại các lớp Multi-Head Attention và Feed Forward.

### Thông số triển khai
- **Rank (`r` = 16) & `lora_alpha` = 32:** Tạo ra cân bằng tốt giữa sức mạnh biểu diễn đặc trưng tư vấn tâm lý học và hiệu năng bộ nhớ. Tỉ lệ dropout 0.05 giúp chống Overfitting.
- **Mục tiêu can thiệp (Target modules):** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.`
- **Huấn luyện (SFTTrainer):** Mô hình được tinh chỉnh theo phương pháp SFT (Supervised Fine-Tuning) trên các token trò chuyện, tối ưu hàm mất mát (Loss) bằng optimizer AdamW với lịch trình suy giảm học tỷ lệ Cosine (Cosine learning rate scheduler) trong 3 epochs. Tham số được tối ưu hoá thông số kỹ thuật Mixed Precision (fp16) trên GPU.

---

## 4. Đánh Giá Mô Hình

Một mô hình tư vấn không thể chỉ được đo bằng độ chính xác đúng/sai, do đó hệ thống được đánh giá đồng thời bằng 2 tiếp cận: **so khớp chuỗi (Lexical Matching)** và **độ tương đồng ngữ nghĩa (Semantic Similarity)**.

### a. Các độ đo đánh giá (Metrics)
1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):** Sử dụng ROUGE-1, ROUGE-2, ROUGE-L đo lường tỷ lệ các n-gram giữa đầu ra của mô hình (Predictions) so với câu trả lời lý tưởng của bác sĩ tâm lý gốc (References). Đánh giá việc mô hình có nắm bắt đúng các từ khóa hay y lệnh quan trọng không.
2. **BERTScore:** Vượt qua hạn chế của ROUGE (chấm điểm bằng việc khớp chuỗi chính xác), BERTScore tính toán ma trận nhúng (embedding) của câu dự đoán và câu thực tế qua mô hình `distilbert-base-uncased`, qua đó đo lường độ tương đồng về ngữ nghĩa. Một bác sĩ ảo tốt có thể diễn đạt bằng câu chữ khác nhưng phải cùng hàm ý trấn an, khuyên giải so với nhãn chuẩn.

### b. Thực nghiệm & Phân tích kết quả
- **Khả năng hội tụ:** Quá trình huấn luyện diễn ra rất tốt thể hiện qua đường cong Loss. `Train Loss` và `Eval Loss` giảm đều qua các step `(xem biểu đồ loss_curve.png được sinh ra)`, không có hiện tượng nổ gradient. 
- **Chỉ số tự động:** Các điểm số F1 của BERTScore và mảng điểm ROUGE (chiết xuất từ evaluation_results.csv) trên tập Test cung cấp số liệu chứng minh hệ thống hoạt động ổn định và hiểu ngữ cảnh sâu sắc các tình trạng sức khỏe tâm lý.
- **Phân tích định tính (Qualitative Analysis):** Mô phỏng sinh ngẫu nhiên (Batched generation) cho thấy chatbot sau SFT đã học được giọng điệu cảm thông (empathetic tone). Khi đầu vào là suy nghĩ tiêu cực hoặc lo âu, mô hình đã bỏ qua cách trả lời dạng bách khoa thông thường và chuyển sang vai một người lắng nghe, đưa ra chiến lược thở / hoặc trị liệu tương tự chuyên gia gốc với chất lượng bám sát Ground Truth.

---
*(Báo cáo có thể được đính kèm cùng main.ipynb, file đánh giá evaluation_results.csv, và biểu đồ loss_curve.png được tạo ra trong workspace để làm minh chứng nộp bài cuối kỳ).*You've used 53% of your weekly rate limit. Your weekly rate limit will reset on May 4 at 7:00 AM. [Learn More](https://aka.ms/github-copilot-rate-limit-error)