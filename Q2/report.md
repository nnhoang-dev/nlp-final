# BÁO CÁO TỔNG HỢP
## Doc2Vec: So sánh Pretrained vs Train từ đầu

---

## 1. MỤC TIÊU

So sánh hai phương pháp tạo document embeddings:
- **Q2A**: Dùng pretrained models (Sentence Transformers)
- **Q2B**: Train từ đầu (Gensim Doc2Vec)

---

## 2. PHƯƠNG PHÁP

### Q2A - Pretrained Model
- **Model**: all-MiniLM-L6-v2
- **Embedding**: 384D
- **Parameters**: 22 triệu
- **Đặc điểm**: Pre-trained trên 1M+ cặp câu

### Q2B - Train từ đầu
- **PV-DM**: Distributed Memory (100D)
- **PV-DBOW**: Distributed Bag of Words (100D)
- **Epochs**: 20

---

## 3. DATASET

- **20 Newsgroups** - 4 categories:
  - sci.med, sci.space, rec.sport.baseball, talk.politics.misc
- Train: 2249 | Test: 1497

---

## 4. KẾT QUẢ

| Phương pháp | Accuracy |
|------------|----------|
| **Sentence Transformers (Q2A)** | **87.71%** |
| Gensim PV-DBOW (Q2B) | 75-76% |
| Gensim PV-DM (Q2B) | 66-68% |

**So sánh**: Pretrained tốt hơn ~12-20% so với train từ đầu

---

## 5. PHÂN TÍCH

### Khi nào dùng Pretrained (Q2A)?

| Tình huống | Nên dùng |
|-----------|----------|
| Dataset nhỏ (<10k) | ✅ Sentence Transformers |
| Generic domain | ✅ |
| Cần chất lượng cao | ✅ |
| Tiếng Anh | ✅ |

### Khi nào Train từ đầu (Q2B)?

| Tình huống | Nên dùng |
|-----------|----------|
| Dataset lớn, domain cụ thể | ✅ Gensim |
| Cần tùy biến | ✅ |
| Resource limited | ✅ |

---

## 6. KẾT LUẬN

✅ **Sentence Transformers** với pretrained model đạt kết quả cao hơn (87.71%)

✅ **Gensim Doc2Vec** train từ đầu vẫn có thể sử dụng được (~76%)

✅ Lựa chọn phụ thuộc vào:
- Kích thước dataset
- Domain (generic vs specific)
- Resource (GPU availability)

---

## 7. FILES TRONG PROJECT

```
Q2A/
├── q2a.ipynb         # Pretrained model (Q2A)
├── q2a.md            # Mô tả Q2A
├── q2b.ipynb         # Train from scratch (Q2B)
├── q2b.md            # Mô tả Q2B  
├── REPORT.md         # Báo cáo tổng hợp
├── run_doc2vec.py    # Code Q2A
└── requirements.txt # Dependencies
```