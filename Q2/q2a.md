# Q2A: Doc2Vec với Pretrained Model

## Giới thiệu

Notebook này sử dụng **Sentence Transformers** (pretrained model) để tạo document embeddings và so sánh với Gensim Doc2Vec truyền thống.

## Phương pháp

### Sentence Transformers
- **Model**: all-MiniLM-L6-v2
- **Embedding**: 384 chiều
- **Parameters**: ~22 triệu
- **Đặc điểm**: Pre-trained trên 1 triệu+ cặp câu từ SNLI, MultiNLI, Wikipedia

### Gensim Doc2Vec (để so sánh)
- PV-DM (Distributed Memory): 100D
- PV-DBOW (Distributed Bag of Words): 100D

## Dataset

- **20 Newsgroups** với 4 categories:
  - sci.med (Y học)
  - sci.space (Khoa học vũ trụ)
  - rec.sport.baseball (Bóng chày)
  - talk.politics.misc (Chính trị)
- Train: 2249 samples, Test: 1497 samples
- Preprocessing: Truncate 3000 chars

## Kết quả

| Model | Accuracy |
|-------|----------|
| **Sentence Transformers** | **87.71%** |
| Gensim PV-DBOW | 75.22% |
| Gensim PV-DM | 66.33% |

## Ưu điểm

- **Không cần train**: Dùng pretrained weights
- **Chất lượng cao**: Hiểu ngữ nghĩa sâu
- **Nhanh**: Sử dụng được ngay

## Hạn chế

- Chủ yếu Tiếng Anh
- Model lớn (22M params)
- Khó tùy biến cho domain đặc biệt