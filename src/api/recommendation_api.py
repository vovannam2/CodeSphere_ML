"""
FastAPI service để phục vụ recommendations
Java backend sẽ gọi API này để lấy recommendations cho users
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import pandas as pd
from typing import List, Optional
import uvicorn

# Khởi tạo FastAPI app
app = FastAPI(
    title="CodeSphere Recommendation API",
    description="API để lấy recommendations cho users",
    version="1.0.0"
)

# CORS middleware để cho phép Java backend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables để lưu model và data
model = None
problem_ids = None
user_ids = None

class RecommendationResponse(BaseModel):
    """Response model cho recommendations"""
    problem_id: int
    predicted_rating: float
    title: Optional[str] = None

class RecommendationListResponse(BaseModel):
    """Response model cho list recommendations"""
    user_id: int
    recommendations: List[RecommendationResponse]
    total: int

def load_model():
    """
    Load model đã train từ file
    """
    global model, problem_ids, user_ids
    
    model_file = os.path.join(os.path.dirname(__file__), '../../data/models/recommendation_model.pkl')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy model file: {model_file}")
    
    print(f"Đang load model từ {model_file}...")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    print("Model đã được load thành công!")
    
    # Load problem IDs từ matrix
    matrix_file = os.path.join(os.path.dirname(__file__), '../../data/processed/user_problem_matrix.csv')
    if os.path.exists(matrix_file):
        matrix = pd.read_csv(matrix_file, index_col=0)
        problem_ids = [int(col) for col in matrix.columns]
        user_ids = [int(idx) for idx in matrix.index]
        print(f"Đã load {len(problem_ids)} problems và {len(user_ids)} users")
    else:
        print("Cảnh báo: Không tìm thấy matrix file, không thể validate user/problem IDs")

@app.on_event("startup")
async def startup_event():
    """
    Load model khi API khởi động
    """
    try:
        load_model()
    except Exception as e:
        print(f"LỖI khi load model: {e}")
        print("API vẫn khởi động nhưng sẽ không thể phục vụ recommendations")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/recommendations/{user_id}", response_model=RecommendationListResponse)
async def get_recommendations(user_id: int, limit: int = 10):
    """
    Lấy recommendations cho một user
    
    Parameters:
    - user_id: ID của user
    - limit: Số lượng recommendations muốn lấy (mặc định 10)
    
    Returns:
    - List các problems được recommend kèm predicted rating
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load. Vui lòng train model trước.")
    
    if user_ids and user_id not in user_ids:
        # User mới (cold start) - có thể trả về popular problems hoặc empty
        print(f"User {user_id} là user mới (cold start)")
        return RecommendationListResponse(
            user_id=user_id,
            recommendations=[],
            total=0
        )
    
    try:
        # Lấy tất cả problems
        if problem_ids is None:
            raise HTTPException(status_code=500, detail="Không thể lấy danh sách problems")
        
        # Predict rating cho tất cả problems user chưa tương tác
        predictions = []
        
        for problem_id in problem_ids:
            # Predict rating
            prediction = model.predict(user_id, problem_id)
            
            # Chỉ lấy predictions có rating > 0 (có khả năng user thích)
            if prediction.est > 0:
                predictions.append({
                    'problem_id': problem_id,
                    'predicted_rating': float(prediction.est)
                })
        
        # Sắp xếp theo predicted rating (cao -> thấp)
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Lấy top N recommendations
        top_predictions = predictions[:limit]
        
        # Tạo response
        recommendations = [
            RecommendationResponse(
                problem_id=p['problem_id'],
                predicted_rating=p['predicted_rating']
            )
            for p in top_predictions
        ]
        
        return RecommendationListResponse(
            user_id=user_id,
            recommendations=recommendations,
            total=len(recommendations)
        )
        
    except Exception as e:
        print(f"Lỗi khi predict: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo recommendations: {str(e)}")

@app.post("/train")
async def trigger_training():
    """
    Endpoint để trigger training model (chạy script training)
    Lưu ý: Trong production nên chạy training ở background job, không phải qua API
    """
    return {
        "message": "Training nên được chạy bằng script riêng, không qua API",
        "instruction": "Chạy: python src/training/train_recommendation_model.py"
    }

if __name__ == "__main__":
    # Chạy API server
    # Port mặc định: 8000
    uvicorn.run(
        "recommendation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload khi code thay đổi (chỉ dùng cho development)
    )

