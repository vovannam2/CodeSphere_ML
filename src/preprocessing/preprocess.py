"""
Preprocessing script to prepare data for training recommendation model.
Creates User-Problem Matrix from interactions.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Project root: CodeSphere_ML
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_data(data_dir: str | os.PathLike = DEFAULT_RAW_DIR):
    """
    Load data exported from database.
    """
    print("Loading data...")
    
    data_dir = Path(data_dir)
    interactions_file = data_dir / 'user_problem_interactions.csv'
    problems_file = data_dir / 'problem_metadata.csv'
    
    if not os.path.exists(interactions_file):
        raise FileNotFoundError(f"Missing file: {interactions_file}")
    if not os.path.exists(problems_file):
        raise FileNotFoundError(f"Missing file: {problems_file}")
    
    interactions_df = pd.read_csv(interactions_file)
    problems_df = pd.read_csv(problems_file)
    
    print(f"Loaded {len(interactions_df)} interactions")
    print(f"Loaded {len(problems_df)} problems")
    
    return interactions_df, problems_df

def create_user_problem_matrix(interactions_df):
    """
    Create User-Problem Matrix from interactions
    - Rows: users
    - Columns: problems
    - Values: max interaction score (solved=5, attempted=2, bookmarked=3)
    """
    print("Creating User-Problem Matrix...")
    
    # group by user_id and problem_id, keep max score
    matrix_data = interactions_df.groupby(['user_id', 'problem_id'])['interaction_score'].max().reset_index()
    
    # pivot: user_id x problem_id
    matrix = matrix_data.pivot_table(
        index='user_id',
        columns='problem_id',
        values='interaction_score',
        fill_value=0  # no interaction = 0
    )
    
    print(f"Matrix shape: {matrix.shape} (users x problems)")
    print(f"Users: {matrix.shape[0]}")
    print(f"Problems: {matrix.shape[1]}")
    print(f"Non-zero interactions: {(matrix > 0).sum().sum()}")
    
    return matrix

def handle_cold_start(matrix, problems_df):
    """
    Handle cold start problems (no interactions yet) by building feature vectors.
    """
    print("Handling cold start problems...")
    
    # problems with interactions
    problems_in_matrix = set(matrix.columns)
    
    # all problems
    all_problems = set(problems_df['problem_id'].astype(int))
    
    # problems without interactions (cold start)
    cold_start_problems = all_problems - problems_in_matrix
    
    print(f"Cold start problems: {len(cold_start_problems)}")
    
    # build feature vectors for cold start problems (for future content-based)
    problem_features = {}
    
    for _, row in problems_df.iterrows():
        problem_id = int(row['problem_id'])
        if problem_id in cold_start_problems:
            # build feature vector from metadata
            features = {
                'level': row['level'],
                'time_limit': row['time_limit_ms'],
                'memory_limit': row['memory_limit_mb'],
                'category_ids': str(row['category_ids']) if pd.notna(row['category_ids']) else '',
                'tag_ids': str(row['tag_ids']) if pd.notna(row['tag_ids']) else ''
            }
            problem_features[problem_id] = features
    
    return problem_features

def save_processed_data(matrix, problem_features, output_dir: str | os.PathLike = DEFAULT_PROCESSED_DIR):
    """
    Save processed data (matrix, features, stats).
    """
    print("Saving processed data...")
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # save matrix CSV
    matrix_file = output_dir / 'user_problem_matrix.csv'
    matrix.to_csv(matrix_file)
    print(f"Saved matrix to {matrix_file}")
    
    # save problem features
    if problem_features:
        features_df = pd.DataFrame.from_dict(problem_features, orient='index')
        features_df.index.name = 'problem_id'
        features_file = output_dir / 'problem_features.csv'
        features_df.to_csv(features_file)
        print(f"Saved problem features to {features_file}")
    
    # save stats
    stats = {
        'num_users': matrix.shape[0],
        'num_problems': matrix.shape[1],
        'num_interactions': int((matrix > 0).sum().sum()),
        'sparsity': float((matrix == 0).sum().sum() / (matrix.shape[0] * matrix.shape[1]))
    }
    
    stats_file = output_dir / 'matrix_stats.txt'
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("USER-PROBLEM MATRIX STATS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Users: {stats['num_users']}\n")
        f.write(f"Problems: {stats['num_problems']}\n")
        f.write(f"Interactions: {stats['num_interactions']}\n")
        f.write(f"Sparsity: {stats['sparsity']:.2%}\n")
    
    print(f"Saved stats to {stats_file}")
    print(f"\nStats:")
    print(f"- Users: {stats['num_users']}")
    print(f"- Problems: {stats['num_problems']}")
    print(f"- Interactions: {stats['num_interactions']}")
    print(f"- Sparsity: {stats['sparsity']:.2%}")

def main():
    """
    Main entry for preprocessing.
    """
    print("=" * 50)
    print("START PREPROCESSING DATA")
    print("=" * 50)
    
    try:
        # Load data
        interactions_df, problems_df = load_data()
        
        # Create User-Problem Matrix
        matrix = create_user_problem_matrix(interactions_df)
        
        # Handle cold start
        problem_features = handle_cold_start(matrix, problems_df)
        
        # Save processed data
        save_processed_data(matrix, problem_features)
        
        print("\n" + "=" * 50)
        print("PREPROCESSING DONE!")
        print("=" * 50)
        
        return matrix, problem_features
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()

