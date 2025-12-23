"""
FastAPI service để phục vụ recommendations (sử dụng NMF model)
Java backend sẽ gọi API này để lấy recommendations cho users
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np
from typing import List, Optional
import uvicorn

# Khởi tạo FastAPI app
app = FastAPI(
    title="CodeSphere Recommendation API",
    description="API để lấy recommendations cho users (NMF model)",
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
model_data = None

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
    global model_data
    
    model_file = os.path.join(os.path.dirname(__file__), '../../data/models/recommendation_model.pkl')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy model file: {model_file}")
    
    print(f"Đang load model từ {model_file}...")
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    print("Model đã được load thành công!")
    print(f"Đã load {len(model_data['user_ids'])} users và {len(model_data['problem_ids'])} problems")

def predict_rating_nmf(model_data, user_idx, problem_idx):
    """
    Dự đoán rating cho một user-problem pair sử dụng NMF
    """
    W = model_data['user_embeddings']
    H = model_data['problem_embeddings']
    
    # Rating = dot product của user embedding và problem embedding
    rating = np.dot(W[user_idx], H[:, problem_idx])
    
    # Scale lại về range gốc (1-5)
    rating_scaled = rating * 4 + 1  # Scale từ [0,1] về [1,5]
    
    return float(rating_scaled)

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
        "model_loaded": model_data is not None
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
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model chưa được load. Vui lòng train model trước.")
    
    try:
        # Tìm user index
        user_ids = model_data['user_ids']
        if user_id not in user_ids:
            # User mới (cold start)
            print(f"User {user_id} là user mới (cold start)")
            return RecommendationListResponse(
                user_id=user_id,
                recommendations=[],
                total=0
            )
        
        user_idx = user_ids.index(user_id)
        problem_ids = model_data['problem_ids']
        
        # Predict rating cho tất cả problems
        predictions = []
        
        for problem_idx, problem_id in enumerate(problem_ids):
            # Predict rating
            rating = predict_rating_nmf(model_data, user_idx, problem_idx)
            
            # Chỉ lấy predictions có rating > 0
            if rating > 0:
                predictions.append({
                    'problem_id': int(problem_id),
                    'predicted_rating': float(rating)
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
    Endpoint để trigger training model
    """
    return {
        "message": "Training nên được chạy bằng script riêng",
        "instruction": "Chạy: python src/training/train_recommendation_model_simple.py"
    }

if __name__ == "__main__":
    # Chạy API server
    uvicorn.run(
        "recommendation_api_nmf:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

