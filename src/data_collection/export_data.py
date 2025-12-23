"""
Script to export data from MySQL.
Exports user-problem interactions and problem metadata to train model.
"""

import mysql.connector
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# Load env vars from .env
load_dotenv()

# Project root: CodeSphere_ML
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"

def get_db_connection():
    """Connect to MySQL using env variables or defaults."""
    try:
        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'codesphere_db')
        )
        return conn
    except Exception as e:
        print(f"DB connection error: {e}")
        raise

def export_user_problem_interactions(output_dir: str | os.PathLike = DEFAULT_RAW_DIR):
    """
    Export user-problem interactions: submissions (solved/attempted), bookmarks.
    """
    print("Exporting user-problem interactions...")
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # submissions: solved = 5, attempted = 2
    query_submissions = """
        SELECT 
            s.user_id,
            s.problem_id,
            CASE 
                WHEN s.is_accepted = 1 THEN 5  -- Solved: 5 diem
                ELSE 2                          -- Attempted: 2 diem
            END as interaction_score,
            'submission' as interaction_type,
            s.created_at
        FROM submissions s
        WHERE s.is_deleted = 0
        ORDER BY s.user_id, s.problem_id, s.created_at DESC
    """
    
    cursor.execute(query_submissions)
    submissions = cursor.fetchall()
    
    # bookmarks
    query_bookmarks = """
        SELECT 
            pb.user_id,
            pb.problem_id,
            3 as interaction_score,  -- Bookmarked: 3 diem
            'bookmark' as interaction_type,
            pb.created_at
        FROM problem_bookmarks pb
        WHERE pb.is_deleted = 0
    """
    
    cursor.execute(query_bookmarks)
    bookmarks = cursor.fetchall()
    
    # combine
    all_interactions = []
    
    # keep best submission per user-problem
    user_problem_best = {}
    for sub in submissions:
        key = (sub['user_id'], sub['problem_id'])
        if key not in user_problem_best:
            user_problem_best[key] = sub
        elif sub['interaction_score'] > user_problem_best[key]['interaction_score']:
            # prefer solved (5)
            user_problem_best[key] = sub
    
    # add best submissions
    for sub in user_problem_best.values():
        all_interactions.append(sub)
    
    # add bookmarks if missing
    for bookmark in bookmarks:
        key = (bookmark['user_id'], bookmark['problem_id'])
        if key not in user_problem_best:
            all_interactions.append(bookmark)
    
    # to dataframe
    df = pd.DataFrame(all_interactions)
    
    # save csv
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / 'user_problem_interactions.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Da export {len(df)} interactions vao {output_file}")
    print(f"So users: {df['user_id'].nunique()}")
    print(f"So problems: {df['problem_id'].nunique()}")
    
    cursor.close()
    conn.close()
    
    return df

def export_problem_metadata(output_dir: str | os.PathLike = DEFAULT_RAW_DIR):
    """Export problem metadata (categories, tags, level)."""
    print("Exporting problem metadata...")
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # query problem metadata
    query_problems = """
        SELECT 
            p.id as problem_id,
            p.title,
            p.level,
            p.time_limit_ms,
            p.memory_limit_mb,
            GROUP_CONCAT(DISTINCT c.id ORDER BY c.id) as category_ids,
            GROUP_CONCAT(DISTINCT t.id ORDER BY t.id) as tag_ids
        FROM problems p
        LEFT JOIN problem_categories pc ON p.id = pc.problem_id
        LEFT JOIN categories c ON pc.category_id = c.id
        LEFT JOIN problem_tags pt ON p.id = pt.problem_id
        LEFT JOIN tags t ON pt.tag_id = t.id
        WHERE p.status = 1
        GROUP BY p.id, p.title, p.level, p.time_limit_ms, p.memory_limit_mb
    """
    
    cursor.execute(query_problems)
    problems = cursor.fetchall()
    
    # to dataframe
    df = pd.DataFrame(problems)
    
    # save csv
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / 'problem_metadata.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Exported {len(df)} problems to {output_file}")
    
    cursor.close()
    conn.close()
    
    return df

def export_user_statistics(output_dir: str | os.PathLike = DEFAULT_RAW_DIR):
    """Export user statistics (solved/attempted counts)."""
    print("Exporting user statistics...")
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    query_users = """
        SELECT 
            u.id as user_id,
            COUNT(DISTINCT CASE WHEN s.is_accepted = 1 THEN s.problem_id END) as solved_count,
            COUNT(DISTINCT s.problem_id) as attempted_count,
            COUNT(DISTINCT CASE WHEN s.is_accepted = 0 THEN s.problem_id END) as failed_count
        FROM users u
        LEFT JOIN submissions s ON u.id = s.user_id AND s.is_deleted = 0
        GROUP BY u.id
        HAVING attempted_count > 0  -- users with submissions
    """
    
    cursor.execute(query_users)
    users = cursor.fetchall()
    
    df = pd.DataFrame(users)
    
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / 'user_statistics.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"Exported {len(df)} users to {output_file}")
    
    cursor.close()
    conn.close()
    
    return df

def main():
    """Main entry: export all data."""
    print("=" * 50)
    print("START EXPORT DATA FROM DATABASE")
    print("=" * 50)
    
    try:
        # Export all
        interactions_df = export_user_problem_interactions()
        problems_df = export_problem_metadata()
        users_df = export_user_statistics()
        
        print("\n" + "=" * 50)
        print("EXPORT DONE!")
        print("=" * 50)
        print(f"\nSummary:")
        print(f"- User-Problem Interactions: {len(interactions_df)} records")
        print(f"- Problems: {len(problems_df)} records")
        print(f"- Users: {len(users_df)} records")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        raise

if __name__ == "__main__":
    main()

