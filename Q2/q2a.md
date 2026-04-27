# Q2A: Doc2Vec với Pretrained Model (Sentence Transformers)

## 1. Mục tiêu
Sử dụng mô hình pretrained để tạo document embeddings và đánh giá hiệu quả phân loại văn bản trên 20 Newsgroups.

## 2. Phương pháp

### 2.1 Mô hình sử dụng
- **Sentence Transformers**: `all-MiniLM-L6-v2`
- Embedding: **384 chiều**
- Đặc trưng: đã pretrain trên dữ liệu lớn, học tốt quan hệ ngữ nghĩa.

### 2.2 Quy trình thực nghiệm
1. Load dữ liệu 20 Newsgroups (4 categories).
2. Làm sạch dữ liệu cơ bản và cắt độ dài văn bản.
3. Encode văn bản train/test thành vectors.
4. Huấn luyện Logistic Regression trên vectors train.
5. Đánh giá accuracy trên tập test.

## 3. Kết quả thực nghiệm

| Model | Accuracy |
|-------|----------|
| **Sentence Transformers (all-MiniLM-L6-v2)** | **87.71%** |

## 4. Phân tích

### Ưu điểm
- Không cần train embedding từ đầu.
- Hiệu năng cao ngay cả với dữ liệu không lớn.
- Triển khai nhanh, dễ tái sử dụng.

### Hạn chế
- Phụ thuộc vào pretrained corpus (domain mismatch có thể xảy ra).
- Chi phí tính toán cao hơn một số mô hình truyền thống.
- Khó can thiệp sâu vào cơ chế biểu diễn nội bộ.

## 5. Kết luận
Cách tiếp cận pretrained là lựa chọn tốt nhất trong bài này, đạt accuracy cao và thời gian triển khai ngắn.