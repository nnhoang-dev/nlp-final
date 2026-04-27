### 2. Doc2Vec với Deep Learning — Bản thực hiện

#### a) Có sử dụng pretrained models

- Đã áp dụng pretrained model: **Sentence Transformers (`all-MiniLM-L6-v2`)**.
- Đã phân tích:
  - **Ưu điểm**: chất lượng cao, triển khai nhanh, không cần train embedding từ đầu.
  - **Hạn chế**: phụ thuộc domain pretrain, khó tùy biến sâu, chi phí suy luận cao hơn baseline.
  - **Kết quả thực nghiệm**: Accuracy đạt **87.71%**.

#### b) Không sử dụng pretrained models

- Đã tự build & train model với **Gensim Doc2Vec**:
  - PV-DM
  - PV-DBOW
- Kết quả:
  - PV-DBOW: ~75–76%
  - PV-DM: ~66–68%
- Đã so sánh với pretrained:
  - Pretrained tốt hơn khoảng **12%** so với mô hình train từ đầu tốt nhất.
