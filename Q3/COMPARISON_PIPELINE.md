# Model Comparison Pipeline

Pipeline so sánh model trước và sau khi fine-tuning với Qwen2.5-0.5B-Instruct.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  COMPARISON PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Dataset   │───▶│  Load Base  │───▶│  Evaluate   │  │
│  │  (10 rows) │    │   Model     │    │   Before    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                │         │
│                                                ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Results  │◀───│   Compare   │◀───│  Evaluate   │  │
│  │   Report   │    │   Metrics   │    │    After    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files

- `compare_models.py` - Script so sánh model trước và sau khi train
- `comparison_results.json` - Kết quả so sánh (tự động tạo)

## Usage

```bash
python compare_models.py
```

## Metrics

| Metric | Description |
|-------|-------------|
| Exact Match | So khớp chính xác câu trả lời |
| Precision | Tỷ lệ từ đúng trong câu trả lời |
| Recall | Tỷ lệ từ trong câu trả lời khớp với ground truth |
| F1 Score | Trung bình điều hòa của Precision và Recall |

## Output

Script tạo file `comparison_results.json` chứa:
- `timestamp`: Thời điểm chạy
- `test_size`: Số lượng mẫu test
- `before_finetuning`: Metrics và samples trước khi train
- `after_finetuning`: Metrics và samples sau khi train
- `improvement`: Chênh lệch giữa before và after

## Example Output

```
============================================================
COMPARISON SUMMARY
============================================================

Metric Comparison:
Metric            Before       After        Change     
---------------------------------------------------
exact_match       0.1000      0.4000       +0.3000    
precision        0.1500      0.4500       +0.3000    
recall           0.1200      0.3800       +0.2600    
f1               0.1300      0.4000       +0.2700    
```

## Workflow

1. **Load Dataset**: Đọc 10 sample từ dataset.csv
2. **Load Base Model**: Load Qwen2.5-0.5B-Instruct gốc
3. **Evaluate Before**: Đánh giá model gốc trên test set
4. **Load Trained Model**: Load model đã fine-tune từ qwen_output/
5. **Evaluate After**: Đánh giá model đã train
6. **Compare**: Tính toán và hiển thị chênh lệch metrics
7. **Save Results**: Lưu kết quả ra comparison_results.json