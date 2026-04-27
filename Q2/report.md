# BÁO CÁO CUỐI KỲ MÔN NLP
## Chủ đề: Doc2Vec với Deep Learning  
### So sánh hai hướng tiếp cận: Pretrained vs Train từ đầu

---

## 1. MỤC TIÊU BÀI TOÁN

Mục tiêu của bài là biểu diễn văn bản ở mức **document embedding** và đánh giá hiệu quả phân loại văn bản với 2 hướng:

- **Q2A (Pretrained):** dùng Sentence Transformers (`all-MiniLM-L6-v2`) để sinh embedding.
- **Q2B (From scratch):** tự huấn luyện Doc2Vec bằng Gensim với hai biến thể:
  - PV-DM (Distributed Memory)
  - PV-DBOW (Distributed Bag of Words)

Bài toán đánh giá: phân loại văn bản theo chủ đề trên tập 20 Newsgroups.

---

## 2. DỮ LIỆU VÀ THIẾT LẬP THÍ NGHIỆM

### 2.1 Dataset
Sử dụng **20 Newsgroups** với 4 lớp:
- `sci.med`
- `sci.space`
- `rec.sport.baseball`
- `talk.politics.misc`

Quy mô:
- Train: **2249** văn bản
- Test: **1497** văn bản

Tiền xử lý:
- Loại bỏ headers / footers / quotes
- Chuẩn hóa văn bản (lowercase, tokenization)
- Giới hạn độ dài văn bản đầu vào (trong Q2A)

### 2.2 Bộ phân loại downstream
Sau khi có document embeddings, dùng **Logistic Regression** để phân loại và đo **Accuracy**.

---

## 3. PHƯƠNG PHÁP

## 3.1 Q2A – Dùng pretrained model
- Model: `all-MiniLM-L6-v2` (Sentence Transformers)
- Kích thước embedding: **384 chiều**
- Số tham số: khoảng **22M**
- Cơ chế: tận dụng tri thức ngữ nghĩa học được từ corpora lớn trước đó.

## 3.2 Q2B – Train từ đầu với Doc2Vec
Hai mô hình được huấn luyện:
- **PV-DM:** giữ ngữ cảnh/ thứ tự từ tốt hơn
- **PV-DBOW:** đơn giản hơn, thường học nhanh hơn

Cấu hình chính (đã dùng trong notebook):
- `vector_size`: 100–150 (tùy cell thực nghiệm)
- `window`: 5
- `min_count`: 2
- `epochs`: 20–40

---

## 4. KẾT QUẢ THỰC NGHIỆM

| Phương pháp | Accuracy |
|------------|----------|
| **Sentence Transformers (Q2A)** | **87.71%** |
| Gensim PV-DBOW (Q2B) | ~75–76% |
| Gensim PV-DM (Q2B) | ~66–68% |

Nhận xét nhanh:
- Mô hình pretrained vượt trội rõ rệt.
- PV-DBOW tốt hơn PV-DM trên bộ dữ liệu này.

---

## 5. PHÂN TÍCH

### 5.1 Ưu điểm của hướng pretrained (Q2A)
- **Độ chính xác cao** trên dữ liệu vừa/nhỏ.
- **Triển khai nhanh**, gần như không cần huấn luyện embedding từ đầu.
- Embedding có tính ngữ nghĩa tốt, ổn định giữa nhiều tác vụ tiếng Anh.

### 5.2 Hạn chế của hướng pretrained
- Phụ thuộc vào domain đã pretrain; domain đặc thù có thể cần fine-tune.
- Chi phí suy luận có thể cao hơn phương pháp cổ điển.
- Tùy biến kiến trúc khó hơn so với tự train.

### 5.3 Ưu điểm của hướng train từ đầu (Q2B)
- **Chủ động kiểm soát** quá trình học biểu diễn.
- Phù hợp khi có dữ liệu domain riêng lớn.
- Mô hình nhẹ, dễ tích hợp trong pipeline truyền thống.

### 5.4 Hạn chế của hướng train từ đầu
- Chất lượng thường thấp hơn pretrained nếu dữ liệu không đủ lớn.
- Nhạy cảm với siêu tham số (vector size, epochs, min_count...).
- Tốn thời gian tinh chỉnh để đạt hiệu quả tối ưu.

---

## 6. SO SÁNH TỔNG HỢP

| Tiêu chí | Q2A: Pretrained | Q2B: Train từ đầu |
|---------|------------------|-------------------|
| Chất lượng | Rất tốt | Trung bình–khá |
| Tốc độ triển khai | Nhanh | Chậm hơn (cần train) |
| Tùy biến domain | Trung bình | Cao |
| Phù hợp dataset nhỏ | Tốt | Kém hơn |
| Phù hợp domain đặc thù lớn | Có thể cần fine-tune | Tốt nếu dữ liệu đủ lớn |

---

## 7. KẾT LUẬN

- Cách tiếp cận **pretrained (Sentence Transformers)** cho kết quả tốt nhất trong thí nghiệm này (**87.71%**).
- **Doc2Vec train từ đầu** vẫn là baseline hữu ích, đặc biệt khi cần kiểm soát mô hình hoặc triển khai nhẹ.
- Lựa chọn phương pháp nên dựa trên:
  1) quy mô dữ liệu,  
  2) mức độ đặc thù của domain,  
  3) tài nguyên tính toán và thời gian triển khai.

---

## 8. HƯỚNG PHÁT TRIỂN

- Fine-tune Sentence Transformers trực tiếp trên dữ liệu bài toán.
- Mở rộng thêm số lớp trong 20 Newsgroups để kiểm tra tính tổng quát.
- Thử thêm các mô hình embedding hiện đại (e.g., MPNet, E5) để đối chiếu.
- Tối ưu Doc2Vec bằng grid search siêu tham số và tiền xử lý tốt hơn.