# Q2B: Doc2Vec - Train từ đầu (From Scratch)

## 1. Mục tiêu
Tự xây dựng mô hình Doc2Vec không dùng pretrained và đánh giá khả năng phân loại văn bản.

## 2. Phương pháp

### 2.1 Kiến trúc sử dụng
- **PV-DM (Distributed Memory)**: dùng ngữ cảnh và thứ tự từ tốt hơn.
- **PV-DBOW (Distributed Bag of Words)**: học nhanh, thường hiệu quả khi dữ liệu vừa.

### 2.2 Cấu hình huấn luyện
- `vector_size`: 100–150
- `window`: 5
- `min_count`: 2
- `epochs`: 20–40

### 2.3 Quy trình
1. Tokenize + lowercase văn bản.
2. Tạo `TaggedDocument`.
3. Train riêng mô hình PV-DM và PV-DBOW.
4. Infer vector cho tập test.
5. Dùng Logistic Regression để đánh giá phân loại.

## 3. Kết quả thực nghiệm

| Model | Accuracy |
|-------|----------|
| PV-DBOW | ~75–76% |
| PV-DM | ~66–68% |

## 4. So sánh với Q2A (Pretrained)

| Phương pháp | Accuracy |
|------------|----------|
| Sentence Transformers (Q2A) | **87.71%** |
| Doc2Vec train từ đầu (Q2B, tốt nhất PV-DBOW) | ~75–76% |

Chênh lệch thực nghiệm: pretrained cao hơn khoảng **12%**.

## 5. Phân tích

### Ưu điểm
- Tự chủ pipeline, dễ tùy biến theo domain.
- Mô hình nhẹ, không phụ thuộc trọng số pretrained.
- Phù hợp làm baseline và nghiên cứu nội bộ.

### Hạn chế
- Độ chính xác thấp hơn pretrained trong bài toán này.
- Cần tuning nhiều để cải thiện chất lượng.
- Hiệu quả phụ thuộc mạnh vào kích thước/chất lượng dữ liệu.

## 6. Kết luận
Doc2Vec train từ đầu vẫn hữu ích, nhưng với dữ liệu hiện tại, kết quả chưa vượt được hướng pretrained.