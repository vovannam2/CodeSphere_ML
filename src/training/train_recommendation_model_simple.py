"""
Train recommendation model using Collaborative Filtering (NMF).
Uses scikit-learn NMF (Non-negative Matrix Factorization), no C++ build needed.
"""

import pandas as pd
import pickle
import os
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Project root: CodeSphere_ML
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DEFAULT_MODELS_DIR = BASE_DIR / "data" / "models"

def load_processed_data(data_dir: str | os.PathLike = DEFAULT_PROCESSED_DIR):
    """
    Load processed data (User-Problem Matrix).
    """
    print("Loading processed data (User-Problem Matrix)...")
    
    data_dir = Path(data_dir)
    matrix_file = data_dir / 'user_problem_matrix.csv'
    
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Missing file: {matrix_file}")
    
    # Load matrix
    matrix = pd.read_csv(matrix_file, index_col=0)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Users: {matrix.shape[0]}")
    print(f"Problems: {matrix.shape[1]}")
    
    return matrix

def train_nmf_model(matrix, n_components=50, max_iter=200):
    """
    Train NMF model with User-Problem Matrix.
    
    NMF (Non-negative Matrix Factorization) is similar to SVD but:
    - Only non-negative values
    - Suitable for rating data (1-5)
    - No C++ build required
    """
    print("=" * 50)
    print("START TRAINING MODEL WITH NMF")
    print("=" * 50)
    
    # Convert matrix to numpy array
    R = matrix.values.astype(np.float32)
    
    # Scale to 0-1 range for NMF (requires >= 0)
    scaler = MinMaxScaler()
    R_scaled = scaler.fit_transform(R)
    
    print(f"\nTraining NMF model with {n_components} components...")
    
    # Train NMF
    # Note: For scikit-learn==1.3.2, NMF does not accept 'alpha' kwarg.
    # Use default regularization (alpha_W/alpha_H) or tune later if needed.
    model = NMF(
        n_components=n_components,
        max_iter=max_iter,
        random_state=42,
    )
    
    # W: user embeddings (users x components)
    # H: problem embeddings (components x problems)
    W = model.fit_transform(R_scaled)
    H = model.components_
    
    print("Training finished!")
    print(f"User embeddings shape: {W.shape}")
    print(f"Problem embeddings shape: {H.shape}")
    
    # Reconstruction error
    R_reconstructed = np.dot(W, H)
    error = np.mean((R_scaled - R_reconstructed) ** 2)
    print(f"Reconstruction error (MSE): {error:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'user_embeddings': W,
        'problem_embeddings': H,
        'user_ids': matrix.index.tolist(),
        'problem_ids': matrix.columns.tolist()
    }

def predict_rating(model_data, user_idx, problem_idx):
    """Predict rating for a user-problem pair."""
    W = model_data['user_embeddings']
    H = model_data['problem_embeddings']
    
    # Rating = dot product of user and problem embeddings
    rating = np.dot(W[user_idx], H[:, problem_idx])
    
    # Scale back to [1,5] from [0,1]
    rating_scaled = rating * 4 + 1  # [0,1] -> [1,5]
    
    return float(rating_scaled)

def save_model(model_data, output_dir: str | os.PathLike = DEFAULT_MODELS_DIR):
    """
    Save trained model data (NMF + embeddings).
    """
    print("\nSaving model...")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    model_file = output_dir / 'recommendation_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {model_file}")
    
    return model_file

def test_predictions(model_data, n_test=5):
    """
    Test a few predictions to inspect model behavior.
    """
    print(f"\nTesting {n_test} sample predictions...")
    
    user_ids = model_data['user_ids']
    problem_ids = model_data['problem_ids']
    
    for i in range(min(n_test, len(user_ids))):
        user_idx = i
        # pick first problem (for demo; in real use, filter unseen problems)
        if len(problem_ids) > 0:
            problem_idx = 0
            rating = predict_rating(model_data, user_idx, problem_idx)
            print(f"User {user_ids[user_idx]} - Problem {problem_ids[problem_idx]}: Predicted rating = {rating:.2f}")

def main():
    """
    Main entry to train model.
    """
    print("=" * 50)
    print("RECOMMENDATION MODEL TRAINING (NMF)")
    print("=" * 50)
    
    try:
        # Load processed data
        matrix = load_processed_data()
        
        # Train model (hyperparameters adjustable here)
        model_data = train_nmf_model(
            matrix,
            n_components=50,  # latent factors
            max_iter=200       # iterations
        )
        
        # Test predictions
        test_predictions(model_data)
        
        # Save model
        model_file = save_model(model_data)
        
        print("\n" + "=" * 50)
        print("TRAIN MODEL DONE!")
        print("=" * 50)
        print(f"\nModel saved at: {model_file}")
        print(f"\nModel uses NMF (Non-negative Matrix Factorization)")
        print(f"- Components: {model_data['user_embeddings'].shape[1]}")
        print(f"- Users: {len(model_data['user_ids'])}")
        print(f"- Problems: {len(model_data['problem_ids'])}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()

