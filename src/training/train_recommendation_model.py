"""
Script để train Recommendation Model sử dụng Collaborative Filtering
Sử dụng thư viện Surprise với thuật toán SVD (Singular Value Decomposition)
"""

import pandas as pd
import pickle
import os
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import numpy as np
from pathlib import Path

# Project root: CodeSphere_ML
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_MODELS_DIR = BASE_DIR / "data" / "models"

def load_processed_data(data_dir: str | os.PathLike = DEFAULT_PROCESSED_DIR):
    """
    Load dữ liệu đã xử lý (User-Problem Matrix)
    """
    print("Đang load dữ liệu đã xử lý...")
    
    data_dir = Path(data_dir)
    matrix_file = data_dir / 'user_problem_matrix.csv'
    
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Không tìm thấy file: {matrix_file}")
    
    # Load matrix
    matrix = pd.read_csv(matrix_file, index_col=0)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Số users: {matrix.shape[0]}")
    print(f"Số problems: {matrix.shape[1]}")
    
    return matrix

def prepare_surprise_dataset(matrix):
    """
    Chuyển đổi matrix thành format của Surprise library
    Surprise cần format: user_id, problem_id, rating
    """
    print("Đang chuẩn bị dữ liệu cho Surprise...")
    
    # Chuyển matrix thành long format
    data = []
    for user_id in matrix.index:
        for problem_id in matrix.columns:
            rating = matrix.loc[user_id, problem_id]
            if rating > 0:  # Chỉ lấy interactions (rating > 0)
                data.append({
                    'user_id': int(user_id),
                    'problem_id': int(problem_id),
                    'rating': float(rating)
                })
    
    df = pd.DataFrame(data)
    
    print(f"Tổng số interactions: {len(df)}")
    print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
    
    # Tạo Reader với scale 1-5 (vì rating là 1, 2, 3, 5)
    reader = Reader(rating_scale=(1, 5))
    
    # Load vào Dataset
    dataset = Dataset.load_from_df(df[['user_id', 'problem_id', 'rating']], reader)
    
    return dataset, df

def train_model(dataset, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
    """
    Train SVD model với các hyperparameters
    
    Parameters:
    - n_factors: Số latent factors (số chiều của vector embedding)
    - n_epochs: Số lần lặp training
    - lr_all: Learning rate
    - reg_all: Regularization parameter
    """
    print("=" * 50)
    print("BẮT ĐẦU TRAIN MODEL")
    print("=" * 50)
    
    # Khởi tạo model SVD
    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        verbose=True
    )
    
    # Cross-validation để đánh giá model
    print("\nĐang chạy cross-validation...")
    cv_results = cross_validate(
        model,
        dataset,
        measures=['RMSE', 'MAE'],
        cv=5,  # 5-fold cross-validation
        verbose=True
    )
    
    print("\nKết quả Cross-Validation:")
    print(f"RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
    print(f"MAE: {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")
    
    # Train trên toàn bộ dataset
    print("\nĐang train model trên toàn bộ dataset...")
    trainset = dataset.build_full_trainset()
    model.fit(trainset)
    
    print("Train hoàn tất!")
    
    return model, cv_results

def evaluate_model(model, dataset):
    """
    Đánh giá model trên test set
    """
    print("\nĐang đánh giá model...")
    
    # Chia train/test
    trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Train lại trên trainset
    model.fit(trainset)
    
    # Predict trên testset
    predictions = model.test(testset)
    
    # Tính RMSE và MAE
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    print(f"RMSE trên test set: {rmse:.4f}")
    print(f"MAE trên test set: {mae:.4f}")
    
    return rmse, mae

def save_model(model, output_dir: str | os.PathLike = DEFAULT_MODELS_DIR):
    """
    Lưu model đã train
    """
    print("\nĐang lưu model...")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model_file = output_dir / 'recommendation_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Đã lưu model vào {model_file}")
    
    return model_file

def test_predictions(model, dataset, n_test=5):
    """
    Test một vài predictions để xem model hoạt động như thế nào
    """
    print(f"\nĐang test {n_test} predictions mẫu...")
    
    trainset = dataset.build_full_trainset()
    
    # Lấy một vài users và problems để test
    test_users = list(trainset.all_users())[:n_test]
    
    for uid in test_users:
        # Lấy problems user này đã tương tác
        user_items = [iid for (iid, _) in trainset.ur[uid]]
        
        # Tìm problem user chưa tương tác
        all_items = set(trainset.all_items())
        unrated_items = all_items - set(user_items)
        
        if len(unrated_items) > 0:
            # Predict cho 1 problem chưa rated
            test_item = list(unrated_items)[0]
            prediction = model.predict(uid, test_item)
            
            print(f"User {uid} - Problem {test_item}: Predicted rating = {prediction.est:.2f}")

def main():
    """
    Hàm chính để train model
    """
    print("=" * 50)
    print("RECOMMENDATION MODEL TRAINING")
    print("=" * 50)
    
    try:
        # Load dữ liệu
        matrix = load_processed_data()
        
        # Chuẩn bị dataset cho Surprise
        dataset, df = prepare_surprise_dataset(matrix)
        
        # Train model
        # Có thể điều chỉnh hyperparameters ở đây
        model, cv_results = train_model(
            dataset,
            n_factors=50,      # Số latent factors
            n_epochs=20,       # Số epochs
            lr_all=0.005,     # Learning rate
            reg_all=0.02      # Regularization
        )
        
        # Đánh giá model
        rmse, mae = evaluate_model(model, dataset)
        
        # Test predictions
        test_predictions(model, dataset)
        
        # Lưu model
        model_file = save_model(model)
        
        print("\n" + "=" * 50)
        print("TRAIN MODEL HOÀN TẤT!")
        print("=" * 50)
        print(f"\nModel đã được lưu tại: {model_file}")
        print(f"\nKết quả đánh giá:")
        print(f"- RMSE: {rmse:.4f}")
        print(f"- MAE: {mae:.4f}")
        
    except Exception as e:
        print(f"\nLỖI: {e}")
        raise

if __name__ == "__main__":
    main()

