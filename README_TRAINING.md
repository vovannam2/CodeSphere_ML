# 📚 HƯỚNG DẪN CHI TIẾT: TRAINING MODEL VÀ GỌI API

## 🎯 TỔNG QUAN

Project ML có 4 bước chính:
1. **Export Data** - Lấy dữ liệu từ database MySQL → CSV
2. **Preprocessing** - Chuyển CSV → User-Problem Matrix
3. **Train Model** - Train NMF model từ Matrix
4. **Predict** - Dùng model để recommend cho users

---

## 📁 CẤU TRÚC FILE

### ✅ FILE ĐANG DÙNG:

```
CodeSphere_ML/
├── src/
│   ├── data_collection/
│   │   └── export_data.py              ✅ Export data từ MySQL
│   ├── preprocessing/
│   │   └── preprocess.py                ✅ Tạo User-Problem Matrix
│   ├── training/
│   │   ├── train_recommendation_model_simple.py  ✅ Train NMF model
│   │   └── auto_retrain.py              ✅ Auto retrain (scheduled)
│   ├── api/
│   │   └── recommendation_api_hybrid.py ✅ API chính (NMF + OpenAI)
│   └── utils/
│       └── openai_client.py             ✅ OpenAI client
└── data/
    ├── raw/                             📄 CSV files từ database
    ├── processed/                       📄 User-Problem Matrix
    └── models/                          📄 Trained model (.pkl)
```

---

## 📖 GIẢI THÍCH KHÁI NIỆM

### 1. CSV LÀ GÌ?

**CSV (Comma-Separated Values)** là file text đơn giản để lưu dữ liệu dạng bảng.

#### Ví dụ: `user_problem_interactions.csv`
```csv
user_id,problem_id,interaction_score,interaction_type
1,100,5,submission
1,101,2,submission
2,100,5,submission
2,102,2,submission
```

**Giải thích:**
- Dòng đầu: Header (tên cột)
- Các dòng sau: Data (mỗi dòng = 1 record)
- Dấu phẩy `,` phân cách các cột
- Dễ đọc, dễ xử lý bằng Python (pandas)

**Tại sao dùng CSV?**
- ✅ Dễ export từ database
- ✅ Dễ đọc bằng pandas
- ✅ Nhẹ, không cần database connection khi training

---

### 2. MATRIX LÀ GÌ?

**Matrix (Ma trận)** là bảng số 2 chiều: **Rows × Columns**

#### Ví dụ: User-Problem Matrix

```
        Problem 100  Problem 101  Problem 102  Problem 103
User 1      5           2           0           3
User 2      5           0           2           0
```

**Giải thích:**
- **Rows (dòng)** = Users (1, 2)
- **Columns (cột)** = Problems (100, 101, 102, 103)
- **Values (giá trị)** = Interaction score (0-5)
  - `5` = Solved
  - `2` = Attempted
  - `3` = Bookmarked
  - `0` = Không có interaction

**Tại sao cần Matrix?**
- ✅ ML model cần input dạng số (không phải text)
- ✅ Dễ tính toán (matrix multiplication)
- ✅ Biểu diễn được mối quan hệ user-problem

**Ví dụ trong code:**
```python
matrix = pd.DataFrame([
    [5, 2, 0, 3],  # User 1
    [5, 0, 2, 0],  # User 2
], 
index=[1, 2],           # user_ids
columns=[100, 101, 102, 103]  # problem_ids
)

# Shape: (2 users × 4 problems)
```

---

### 3. NMF LÀ GÌ?

**NMF (Non-negative Matrix Factorization)** là thuật toán ML để phân tích matrix.

#### Công thức: `R ≈ W × H`

- **R**: User-Problem Matrix (users × problems)
- **W**: User embeddings (users × components)
- **H**: Problem embeddings (components × problems)

#### Ví dụ cụ thể:

**Input Matrix R:**
```
        Problem 100  Problem 101  Problem 102
User 1      5           2           0
User 2      5           0           2
```

**NMF phân tích thành:**

**W (User Embeddings):**
```
        Component 1  Component 2  ...  Component 50
User 1     0.12         0.34      ...      0.78
User 2     0.23         0.45      ...      0.89
```

**H (Problem Embeddings):**
```
                Problem 100  Problem 101  Problem 102
Component 1        0.11         0.22         0.33
Component 2        0.12         0.23         0.34
...
Component 50       0.19         0.29         0.39
```

**Tại sao cần NMF?**
- ✅ Tìm patterns ẩn trong data
- ✅ Dự đoán rating cho user-problem chưa có interaction
- ✅ Giảm chiều dữ liệu (từ 1000 problems → 50 components)

**Cách hoạt động:**
1. NMF học từ data: "User nào giống user nào?"
2. Tạo embeddings: Mỗi user/problem = vector 50 số
3. Predict: `rating = user_embedding · problem_embedding`

---

## 🔄 QUY TRÌNH CHI TIẾT TỪNG BƯỚC

### BƯỚC 1: EXPORT DATA (`export_data.py`)

#### Input: MySQL Database
```
Table: submissions
+---------+------------+-------------+
| user_id | problem_id | is_accepted |
+---------+------------+-------------+
|    1    |    100     |      1      |  ← Solved
|    1    |    101     |      0      |  ← Attempted
|    2    |    100     |      1      |  ← Solved
+---------+------------+-------------+
```

#### Process:
```python
# 1. Query submissions
SELECT user_id, problem_id, 
       CASE WHEN is_accepted=1 THEN 5 ELSE 2 END as score
FROM submissions

# 2. Query bookmarks
SELECT user_id, problem_id, 3 as score
FROM problem_bookmarks

# 3. Combine và keep best score
# User 1 + Problem 100: Solved(5) > Attempted(2) → Giữ 5
```

#### Output: `data/raw/user_problem_interactions.csv`
```csv
user_id,problem_id,interaction_score
1,100,5
1,101,2
2,100,5
```

**Kết quả:**
- ✅ File CSV với tất cả interactions
- ✅ Score: 5 (solved), 2 (attempted), 3 (bookmarked)

---

### BƯỚC 2: PREPROCESSING (`preprocess.py`)

#### Input: `data/raw/user_problem_interactions.csv`
```csv
user_id,problem_id,interaction_score
1,100,5
1,101,2
2,100,5
```

#### Process:
```python
# 1. Load CSV
interactions_df = pd.read_csv('user_problem_interactions.csv')

# 2. Group by user_id + problem_id, keep max score
matrix_data = interactions_df.groupby(['user_id', 'problem_id'])['interaction_score'].max()

# 3. Pivot: user_id × problem_id
matrix = matrix_data.pivot_table(
    index='user_id',
    columns='problem_id',
    values='interaction_score',
    fill_value=0  # No interaction = 0
)
```

#### Output: `data/processed/user_problem_matrix.csv`
```csv
user_id,100,101
1,5,2
2,5,0
```

**Giải thích Matrix:**
```
        Problem 100  Problem 101
User 1      5           2
User 2      5           0
```

**Kết quả:**
- ✅ User-Problem Matrix (users × problems)
- ✅ Values: 0-5 (interaction scores)
- ✅ Shape: (số users × số problems)

---

### BƯỚC 3: TRAIN MODEL (`train_recommendation_model_simple.py`)

#### Input: `data/processed/user_problem_matrix.csv`
```csv
user_id,100,101
1,5,2
2,5,0
```

#### Process:

**3.1. Load Matrix:**
```python
matrix = pd.read_csv('user_problem_matrix.csv', index_col=0)
# matrix = [[5, 2], [5, 0]]
```

**3.2. Convert to NumPy và Scale:**
```python
R = matrix.values.astype(np.float32)
# R = [[5.0, 2.0], [5.0, 0.0]]

# Scale về [0, 1] vì NMF yêu cầu non-negative
scaler = MinMaxScaler()
R_scaled = scaler.fit_transform(R)
# R_scaled = [[1.0, 0.4], [1.0, 0.0]]
```

**3.3. Train NMF:**
```python
model = NMF(n_components=50, max_iter=200, random_state=42)
W = model.fit_transform(R_scaled)  # User embeddings (users × 50)
H = model.components_              # Problem embeddings (50 × problems)
```

**Giải thích Training:**
- NMF học từ R_scaled để tìm W và H sao cho `R_scaled ≈ W × H`
- Sau 200 iterations, W và H được tối ưu
- W: Mỗi user = vector 50 số (embeddings)
- H: Mỗi problem = vector 50 số (embeddings)

**3.4. Tính Reconstruction Error:**
```python
R_reconstructed = np.dot(W, H)  # W × H
error = np.mean((R_scaled - R_reconstructed) ** 2)
# error ≈ 0.001 (càng nhỏ càng tốt)
```

#### Output: `data/models/recommendation_model.pkl`
```python
model_data = {
    'user_embeddings': W,      # (users × 50)
    'problem_embeddings': H,   # (50 × problems)
    'user_ids': [1, 2],
    'problem_ids': [100, 101],
    'scaler': scaler,
    'model': model
}
```

**Kết quả:**
- ✅ Model đã train xong
- ✅ Có thể predict rating cho bất kỳ user-problem nào

---

### BƯỚC 4: PREDICT (`recommendation_api_hybrid.py`)

#### Input: User ID (từ Java API)
```
GET /recommendations/1?limit=10&use_openai=true
```

#### Process:

**4.1. Load Model:**
```python
with open('recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
# model_data = {user_embeddings: W, problem_embeddings: H, ...}
```

**4.2. Tìm User Index:**
```python
user_ids = model_data['user_ids']  # [1, 2]
user_idx = user_ids.index(user_id)  # user_id=1 → user_idx=0
```

**4.3. Predict Rating cho tất cả Problems:**
```python
for problem_idx, problem_id in enumerate(problem_ids):
    # Predict rating
    W = model_data['user_embeddings']      # (2 × 50)
    H = model_data['problem_embeddings']   # (50 × 5)
    
    # Dot product: user_embedding · problem_embedding
    rating = np.dot(W[user_idx], H[:, problem_idx])
    # rating = 0.95 (trong [0, 1])
    
    # Scale về [1, 5]
    rating_scaled = rating * 4 + 1
    # rating_scaled = 4.8
```

**4.4. Sort và Filter:**
```python
# Sort theo rating (cao → thấp)
predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)

# Filter bài đã làm
seen_problems = {100, 101}  # User 1 đã làm
filtered = [p for p in predictions if p['problem_id'] not in seen_problems]

# Lấy top 20
top_20 = filtered[:20]
```

**4.5. (Optional) Gọi OpenAI để Refine:**
```python
if use_openai:
    # Gọi OpenAI với prompt
    refined = openai_client.refine_recommendations(
        top_20,      # Top 20 từ NMF
        user_history,
        user_id
    )
    # OpenAI trả về top 10 + explanation
```

#### Output: JSON Response
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "problem_id": 102,
      "predicted_rating": 4.8,
      "title": "Two Sum",
      "explanation": "This problem matches your skill level..."
    }
  ],
  "total": 10,
  "source": "hybrid"
}
```

**Kết quả:**
- ✅ List recommendations với predicted ratings
- ✅ Có explanation nếu dùng OpenAI

---

## 🚀 CÁCH CHẠY PROJECT

### 1. Train Model (Lần đầu hoặc retrain):

```bash
# Bước 1: Export data
python src/data_collection/export_data.py

# Bước 2: Preprocessing
python src/preprocessing/preprocess.py

# Bước 3: Train model
python src/training/train_recommendation_model_simple.py
```

**Hoặc dùng auto retrain:**
```bash
python src/training/auto_retrain.py
```

### 2. Chạy API:

```bash
python src/api/recommendation_api_hybrid.py
```

API sẽ chạy tại: `http://localhost:8000`

### 3. Test API:

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl http://localhost:8000/recommendations/1?limit=10&use_openai=true
```

---

## 🔗 KẾT NỐI VỚI JAVA BACKEND

Java backend gọi ML API qua `RestTemplate`:

```java
String url = "http://localhost:8000/recommendations/" + userId + "?limit=10&use_openai=true";
ResponseEntity<Map<String, Object>> response = restTemplate.exchange(
    url, HttpMethod.GET, null,
    new ParameterizedTypeReference<Map<String, Object>>() {}
);
```

---

## 📊 VÍ DỤ MINH HỌA ĐẦY ĐỦ

### Scenario: User 1 muốn recommendations

**1. Database có:**
- User 1 đã solved: Problem 100, 101
- User 1 đã attempted: Problem 102

**2. Export → CSV:**
```csv
user_id,problem_id,interaction_score
1,100,5
1,101,5
1,102,2
```

**3. Preprocessing → Matrix:**
```
        Problem 100  Problem 101  Problem 102  Problem 103
User 1      5           5           2           0
```

**4. Train NMF:**
- W[0] = [0.12, 0.34, ..., 0.78] (User 1 embedding)
- H[:, 0] = [0.11, 0.22, ..., 0.19] (Problem 100 embedding)
- H[:, 3] = [0.15, 0.25, ..., 0.25] (Problem 103 embedding)

**5. Predict:**
- User 1 + Problem 103:
  - rating = dot(W[0], H[:, 3]) = 0.85
  - rating_scaled = 0.85 * 4 + 1 = 4.4

**6. Filter & Sort:**
- Bỏ Problem 100, 101, 102 (đã làm)
- Giữ Problem 103 với rating 4.4

**7. Response:**
```json
{
  "recommendations": [
    {"problem_id": 103, "predicted_rating": 4.4}
  ]
}
```

---

## ❓ FAQ

**Q: Tại sao cần scale về [0, 1]?**
A: NMF yêu cầu non-negative values. Scale giúp đảm bảo tất cả values ≥ 0.

**Q: Tại sao dùng 50 components?**
A: Components = số chiều của embeddings. 50 là balance giữa accuracy và performance.

**Q: Khi nào cần retrain?**
A: Khi có nhiều data mới (users/problems mới). Có thể schedule auto retrain hàng ngày.

**Q: OpenAI làm gì?**
A: Refine top 20 từ NMF → top 10 + explanation chi tiết tại sao recommend bài đó.

---

## 📝 TÓM TẮT

1. **CSV**: File text lưu data dạng bảng
2. **Matrix**: Bảng số 2 chiều (users × problems)
3. **NMF**: Thuật toán phân tích matrix thành embeddings
4. **Training**: Học từ matrix để tạo embeddings
5. **Predict**: Dùng embeddings để dự đoán rating

**Flow:**
```
Database → CSV → Matrix → NMF Model → Predictions → API Response
```

