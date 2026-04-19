# Q2B: Doc2Vec - Train từ đầu

## Giới thiệu

Notebook này xây dựng và huấn luyện mô hình Doc2Vec **hoàn toàn từ đầu** (from scratch) không dùng pretrained models.

## Phương pháp

### Doc2Vec (PV-DM và PV-DBOW)
Doc2Vec mở rộng từ Word2Vec để tạo vector cho documents:

- **PV-DM (Distributed Memory)**: Dùng document vector + word vectors để dự đoán từ tiếp theo
- **PV-DBOW (Distributed Bag of Words)**: Chỉ dùng document vector để dự đoán từ

### Tham số huấn luyện
- vector_size = 100
- window = 5
- min_count = 2
- epochs = 20

## Dataset

- **20 Newsgroups** với 4 categories:
  - sci.med, sci.space, rec.sport.baseball, talk.politics.misc
- Train: 2249 samples, Test: 1497 samples
- Preprocessing: Truncate 3000 chars

## Kết quả

| Model | Accuracy |
|-------|----------|
| PV-DBOW | ~76% |
| PV-DM | ~68% |

## So sánh với Pretrained (Q2A)

| Phương pháp | Accuracy |
|------------|----------|
| Sentence Transformers (pretrained) | 87.71% |
| **Doc2Vec (train from scratch)** | ~76% |

**Cải thiện**: +12% khi dùng pretrained

## Ưu điểm

- Có thể tùy biến cho domain cụ thể
- Training trên dữ liệu của chính mình
- Không phụ thuộc vào pretrained data

## Hạn chế

- Cần thời gian train
- Chất lượng thấp hơn pretrained
- Cần dataset đủ lớn