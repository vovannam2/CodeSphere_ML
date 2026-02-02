# 🤖 CodeSphere ML - Recommendation System

## 📚 TÀI LIỆU HƯỚNG DẪN

### 1. [README_TRAINING.md](./README_TRAINING.md)
**Giải thích chi tiết:** Training data và mô hình ML từng bước

### 2. [CONCEPTS_EXPLAINED.md](./CONCEPTS_EXPLAINED.md)
**Giải thích đơn giản:** CSV, Matrix và NMF là gì

### 3. [OPENAI_PROMPT_EXPLANATION.md](./OPENAI_PROMPT_EXPLANATION.md)
**Giải thích chi tiết:** Cách gọi OpenAI API với prompt

### 4. [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)
**Cấu trúc project:** File nào chạy, file nào đã xóa

---

## 🚀 QUICK START

### 1. Train Model (Lần đầu):
```bash
python src/training/auto_retrain.py
```

### 2. Chạy API:
```bash
python src/api/recommendation_api_hybrid.py
```

### 3. Test API:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/recommendations/1?limit=10&use_openai=true
```

---

## 📁 CẤU TRÚC FILE

```
CodeSphere_ML/
├── src/
│   ├── data_collection/
│   │   └── export_data.py              ✅ Export data từ MySQL
│   ├── preprocessing/
│   │   └── preprocess.py                ✅ Tạo User-Problem Matrix
│   ├── training/
│   │   ├── train_recommendation_model_simple.py  ✅ Train NMF
│   │   └── auto_retrain.py              ✅ Auto retrain
│   ├── api/
│   │   └── recommendation_api_hybrid.py ✅ API chính
│   └── utils/
│       └── openai_client.py             ✅ OpenAI client
├── data/
│   ├── raw/                             📄 CSV files
│   ├── processed/                       📄 Matrix
│   └── models/                          📄 Model .pkl
└── README.md                            📖 File này
```

---

## 🔄 FLOW TỔNG QUAN

### Training:
```
Database → CSV → Matrix → NMF Model
```

### Prediction:
```
User ID → API → Load Model → Predict → (Optional) OpenAI → JSON
```

---

## 📖 KHÁI NIỆM QUAN TRỌNG

### CSV (Comma-Separated Values)
File text lưu data dạng bảng, dùng dấu phẩy phân cách.

### Matrix (Ma trận)
Bảng số 2 chiều: Rows (users) × Columns (problems).

### NMF (Non-negative Matrix Factorization)
Thuật toán ML phân tích matrix thành embeddings để predict ratings.

---

## 🔗 KẾT NỐI JAVA BACKEND

Java backend gọi ML API tại: `http://localhost:8000`

**Endpoint:**
- `GET /health` - Health check
- `GET /recommendations/{user_id}?limit=10&use_openai=true` - Get recommendations

---

## ❓ FAQ

**Q: File nào đang được dùng?**
A: Xem [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)

**Q: CSV, Matrix, NMF là gì?**
A: Xem [CONCEPTS_EXPLAINED.md](./CONCEPTS_EXPLAINED.md)

**Q: Cách train model?**
A: Xem [README_TRAINING.md](./README_TRAINING.md)

**Q: Cách OpenAI refine recommendations?**
A: Xem [OPENAI_PROMPT_EXPLANATION.md](./OPENAI_PROMPT_EXPLANATION.md)

---

## 📝 TÓM TẮT

1. **Export** data từ database → CSV
2. **Preprocess** CSV → Matrix
3. **Train** Matrix → NMF Model
4. **Predict** User ID → Recommendations
5. **(Optional)** OpenAI refine → Explanations

**File chính:**
- `recommendation_api_hybrid.py` - API chính
- `train_recommendation_model_simple.py` - Train model
- `auto_retrain.py` - Auto retrain
