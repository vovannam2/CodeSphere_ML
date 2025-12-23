"""
FastAPI service với Hybrid Approach: Local Model (NMF) + OpenAI
Java backend sẽ gọi API này để lấy recommendations cho users
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Set
import uvicorn
import sys
from pathlib import Path

# Thêm path để import utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.openai_client import OpenAIClient

# Khởi tạo FastAPI app
app = FastAPI(
    title="CodeSphere Recommendation API (Hybrid)",
    description="API để lấy recommendations cho users (Local Model + OpenAI)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base dir of ML project (CodeSphere_ML)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Global variables
model_data = None
openai_client = None
problem_titles: Dict[int, str] = {}
user_interactions: Dict[int, Set[int]] = {}

class RecommendationResponse(BaseModel):
    """Response model cho recommendations"""
    problem_id: int
    predicted_rating: float
    title: Optional[str] = None
    explanation: Optional[str] = None  # Explanation từ OpenAI

class RecommendationListResponse(BaseModel):
    """Response model cho list recommendations"""
    user_id: int
    recommendations: List[RecommendationResponse]
    total: int
    source: str  # "local", "hybrid", "openai_only"


def load_problem_metadata():
    """Load problem metadata (titles) từ CSV để map problem_id -> title."""
    global problem_titles
    
    metadata_file = RAW_DIR / "problem_metadata.csv"
    if not metadata_file.exists():
        print(f"[WARNING] problem_metadata.csv not found at {metadata_file}. Titles will be null.")
        problem_titles = {}
        return
    
    try:
        df = pd.read_csv(metadata_file)
        if "problem_id" not in df.columns or "title" not in df.columns:
            print(f"[WARNING] problem_metadata.csv missing required columns. Got columns: {df.columns.tolist()}")
            problem_titles = {}
            return
        
        problem_titles = {
            int(row["problem_id"]): str(row["title"])
            for _, row in df.iterrows()
        }
        print(f"[INFO] Loaded {len(problem_titles)} problem titles from {metadata_file}")
    except Exception as e:
        print(f"[ERROR] Failed to load problem_metadata.csv: {e}")
        problem_titles = {}


def load_user_interactions():
    """
    Load user-problem interactions để biết user đã tương tác bài nào
    → tránh recommend lại các bài user đã làm.
    """
    global user_interactions
    
    interactions_file = RAW_DIR / "user_problem_interactions.csv"
    if not interactions_file.exists():
        print(f"[WARNING] user_problem_interactions.csv not found at {interactions_file}. Will not filter solved problems.")
        user_interactions = {}
        return
    
    try:
        df = pd.read_csv(interactions_file)
        if "user_id" not in df.columns or "problem_id" not in df.columns:
            print(f"[WARNING] user_problem_interactions.csv missing required columns. Got columns: {df.columns.tolist()}")
            user_interactions = {}
            return
        
        user_interactions = {}
        for _, row in df[["user_id", "problem_id"]].dropna().iterrows():
            try:
                uid = int(row["user_id"])
                pid = int(row["problem_id"])
            except Exception:
                continue
            if uid not in user_interactions:
                user_interactions[uid] = set()
            user_interactions[uid].add(pid)
        
        print(f"[INFO] Loaded interactions for {len(user_interactions)} users from {interactions_file}")
    except Exception as e:
        print(f"[ERROR] Failed to load user_problem_interactions.csv: {e}")
        user_interactions = {}

def load_model():
    """Load model đã train từ file"""
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
    """Dự đoán rating cho một user-problem pair sử dụng NMF"""
    W = model_data['user_embeddings']
    H = model_data['problem_embeddings']
    
    rating = np.dot(W[user_idx], H[:, problem_idx])
    rating_scaled = rating * 4 + 1  # Scale từ [0,1] về [1,5]
    
    return float(rating_scaled)

@app.on_event("startup")
async def startup_event():
    """Load model và init OpenAI client khi API khởi động"""
    global model_data, openai_client
    
    try:
        load_model()
    except Exception as e:
        print(f"LỖI khi load model: {e}")
        print("API vẫn khởi động nhưng sẽ không thể phục vụ recommendations")
    
    # Load problem metadata & interactions
    try:
        load_problem_metadata()
    except Exception as e:
        print(f"[ERROR] Failed to load problem metadata at startup: {e}")
    
    try:
        load_user_interactions()
    except Exception as e:
        print(f"[ERROR] Failed to load user interactions at startup: {e}")
    
    # Init OpenAI client
    openai_client = OpenAIClient()
    if openai_client.api_key:
        print("OpenAI client đã được khởi tạo")
    else:
        print("Cảnh báo: OpenAI API key chưa được cấu hình, sẽ chỉ dùng Local Model")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "openai_configured": openai_client.api_key != "" if openai_client else False
    }

@app.get("/recommendations/{user_id}", response_model=RecommendationListResponse)
async def get_recommendations(
    user_id: int, 
    limit: int = Query(10, ge=1, le=20),
    use_openai: bool = Query(True, description="Có dùng OpenAI để refine và thêm explanation không")
):
    """
    Lấy recommendations cho một user
    
    Parameters:
    - user_id: ID của user
    - limit: Số lượng recommendations muốn lấy (mặc định 10)
    - use_openai: Có dùng OpenAI không (mặc định False - chỉ dùng Local Model)
    
    Returns:
    - List các problems được recommend kèm predicted rating và explanation (nếu dùng OpenAI)
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model chưa được load. Vui lòng train model trước.")
    
    try:
        # Tìm user index
        user_ids = model_data['user_ids']
        if user_id not in user_ids:
            # User mới (cold start)
            print(f"User {user_id} là user mới (cold start)")
            
            # Nếu có OpenAI và user yêu cầu, có thể dùng OpenAI cho cold start
            # TODO: Implement OpenAI cold start recommendation
            if use_openai and openai_client and openai_client.api_key:
                pass
            
            return RecommendationListResponse(
                user_id=user_id,
                recommendations=[],
                total=0,
                source="local"
            )
        
        user_idx = user_ids.index(user_id)
        problem_ids = model_data['problem_ids']
        
        # Các bài user đã tương tác (solved/attempted/bookmarked)
        seen_problems = user_interactions.get(user_id, set())
        
        # Bước 1: Local Model - Predict rating cho tất cả problems
        predictions = []
        
        for problem_idx, problem_id in enumerate(problem_ids):
            pid = int(problem_id)
            
            # Bỏ qua các bài user đã tương tác (đã làm / bookmark)
            if pid in seen_problems:
                continue
            
            rating = predict_rating_nmf(model_data, user_idx, problem_idx)
            
            if rating > 0:
                predictions.append({
                    'problem_id': pid,
                    'predicted_rating': float(rating)
                })
        
        # Sắp xếp theo predicted rating (cao -> thấp)
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        # Lấy top 20 từ Local Model (để OpenAI refine)
        top_20 = predictions[:20]
        
        # Bước 2: Nếu dùng OpenAI, refine top 20 → top 10 + explanation
        if use_openai and openai_client and openai_client.api_key:
            try:
                # Lấy user history (có thể từ database hoặc cache)
                user_history = {
                    'solved': [],  # TODO: Lấy từ database
                    'attempted': []  # TODO: Lấy từ database
                }
                
                # Gọi OpenAI để refine
                refined = openai_client.refine_recommendations(
                    top_20,
                    user_history,
                    user_id
                )
                
                if refined and 'recommendations' in refined:
                    # Map OpenAI results với top_20
                    refined_problems = {}
                    for rec in refined['recommendations'][:limit]:
                        problem_id = rec.get('problem_id')
                        explanation = rec.get('reason', '')
                        refined_problems[problem_id] = explanation
                    
                    # Tạo recommendations với explanation
                    recommendations = []
                    for rec in refined['recommendations'][:limit]:
                        problem_id = int(rec.get('problem_id'))
                        # Tìm predicted_rating từ top_20
                        original = next((p for p in top_20 if p['problem_id'] == problem_id), None)
                        if original:
                            recommendations.append(RecommendationResponse(
                                problem_id=problem_id,
                                predicted_rating=original['predicted_rating'],
                                title=problem_titles.get(problem_id),
                                explanation=rec.get('reason', '')
                            ))
                    
                    return RecommendationListResponse(
                        user_id=user_id,
                        recommendations=recommendations,
                        total=len(recommendations),
                        source="hybrid"
                    )
            except Exception as e:
                print(f"Lỗi khi gọi OpenAI, fallback về Local Model: {e}")
                # Fallback về Local Model nếu OpenAI fail
        
        # Bước 3: Chỉ dùng Local Model (không có OpenAI hoặc OpenAI fail)
        top_predictions = top_20[:limit]
        
        recommendations = [
            RecommendationResponse(
                problem_id=p['problem_id'],
                predicted_rating=p['predicted_rating'],
                title=problem_titles.get(p['problem_id'])
            )
            for p in top_predictions
        ]
        
        return RecommendationListResponse(
            user_id=user_id,
            recommendations=recommendations,
            total=len(recommendations),
            source="local"
        )
        
    except Exception as e:
        print(f"Lỗi khi predict: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo recommendations: {str(e)}")

@app.post("/train")
async def trigger_training():
    """Endpoint để trigger training model"""
    return {
        "message": "Training nên được chạy bằng script riêng",
        "instruction": "Chạy: python src/training/train_recommendation_model_simple.py"
    }

if __name__ == "__main__":
    uvicorn.run(
        "recommendation_api_hybrid:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

